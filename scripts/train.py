#!/usr/bin/env python3
"""Unified training entry point for the Bayesian Temporal Pose Network.

Supports four training stages:
    foundation  -- Kinematic foundation model (multi-scale transformer)
    ssl         -- Visual SSL pre-training (MVR, VCL, VTOP, VKA tasks)
    supervised  -- Full BTPN 3-phase supervised training
    detection   -- YOLO segmentation / pose training wrapper

Usage::

    python scripts/train.py --stage foundation --config configs/kinematic_foundation.yaml
    python scripts/train.py --stage ssl --config configs/btpn.yaml
    python scripts/train.py --stage supervised --config configs/btpn.yaml
    python scripts/train.py --stage detection --task segmentation --data-yaml data/yolo/data.yaml

Author: BTPN Publication Repository
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from btpn.config import BTPNConfig
from btpn.model import KinematicFoundationModel, BTPN
from btpn.losses import KinematicLoss, BTPNLoss
from btpn.dataset import (
    NormalizationStats,
    create_dataloaders,
)
from btpn.detection import DetectionPipeline, DetectionConfig
from btpn.utils import (
    CosineWarmupScheduler,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    set_seed,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SSL Task Heads (inline, used only in Stage 1)
# =============================================================================


class MVRHead(nn.Module):
    """Masked Visual Reconstruction head.

    Masks a random fraction of visual frames and reconstructs the
    original features from context.

    Args:
        d_model: Transformer model dimension.
        visual_dim: Raw visual feature dimension.
    """

    def __init__(self, d_model: int, visual_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, visual_dim),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def apply_mask(
        self,
        visual_feats: torch.Tensor,
        mask_ratio: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random masking to visual feature sequence.

        Args:
            visual_feats: (B, T, d_model) projected visual features.
            mask_ratio: Fraction of frames to mask.

        Returns:
            Tuple of (masked_feats, mask, original_feats).
        """
        B, T, D = visual_feats.shape
        original_feats = visual_feats.clone().detach()

        n_mask = max(1, int(T * mask_ratio))
        mask = torch.zeros(B, T, dtype=torch.bool, device=visual_feats.device)
        for b in range(B):
            indices = torch.randperm(T, device=visual_feats.device)[:n_mask]
            mask[b, indices] = True

        mask_expanded = mask.unsqueeze(-1).expand_as(visual_feats)
        mask_token = self.mask_token.expand(B, T, -1)
        masked_feats = visual_feats.clone()
        masked_feats[mask_expanded] = mask_token[mask_expanded]

        return masked_feats, mask, original_feats

    def forward(
        self,
        encoder_output: torch.Tensor,
        mask: torch.Tensor,
        original_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss for masked positions.

        Args:
            encoder_output: (B, T, d_model) encoder output.
            mask: (B, T) boolean mask.
            original_feats: (B, T, d_model) original features.

        Returns:
            Scalar MSE reconstruction loss.
        """
        reconstructed = self.proj(encoder_output)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=encoder_output.device)

        recon_masked = reconstructed[mask]
        target_masked = original_feats[mask]

        min_dim = min(recon_masked.shape[-1], target_masked.shape[-1])
        recon_masked = recon_masked[..., :min_dim]
        target_masked = target_masked[..., :min_dim]

        return F.mse_loss(recon_masked, target_masked)


class VCLHead(nn.Module):
    """Visual Contrastive Learning head (InfoNCE).

    Args:
        d_model: Transformer model dimension.
        proj_dim: Projection head output dimension.
    """

    def __init__(self, d_model: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(
        self,
        cls1: torch.Tensor,
        cls2: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss between two views.

        Args:
            cls1: (B, d_model) CLS token from view 1.
            cls2: (B, d_model) CLS token from view 2.
            temperature: Temperature scaling.

        Returns:
            Scalar InfoNCE loss.
        """
        z1 = F.normalize(self.projector(cls1), dim=-1)
        z2 = F.normalize(self.projector(cls2), dim=-1)

        B = z1.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=z1.device)

        sim_12 = torch.mm(z1, z2.t()) / temperature
        sim_21 = torch.mm(z2, z1.t()) / temperature
        labels = torch.arange(B, device=z1.device)

        return (
            F.cross_entropy(sim_12, labels)
            + F.cross_entropy(sim_21, labels)
        ) / 2.0


class VTOPHead(nn.Module):
    """Visual Temporal Order Prediction head.

    Args:
        d_model: Transformer model dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)

    def forward(
        self,
        cls_token: torch.Tensor,
        is_shuffled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal order classification loss.

        Args:
            cls_token: (B, d_model) CLS token.
            is_shuffled: (B,) binary labels.

        Returns:
            Scalar cross-entropy loss.
        """
        return F.cross_entropy(
            self.classifier(cls_token), is_shuffled.long()
        )


class VKAHead(nn.Module):
    """Visual-Kinematic Alignment head.

    Args:
        d_model: Visual model dimension.
        kin_dim: Kinematic embedding dimension.
        proj_dim: Shared projection dimension.
    """

    def __init__(self, d_model: int, kin_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.kin_proj = nn.Sequential(
            nn.Linear(kin_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(
        self,
        visual_cls: torch.Tensor,
        kin_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine alignment loss.

        Args:
            visual_cls: (B, d_model) visual CLS token.
            kin_embedding: (B, kin_dim) kinematic embedding.

        Returns:
            Scalar loss (1 - mean cosine similarity).
        """
        v = F.normalize(self.visual_proj(visual_cls), dim=-1)
        k = F.normalize(self.kin_proj(kin_embedding.detach()), dim=-1)
        return 1.0 - (v * k).sum(dim=-1).mean()


# =============================================================================
# SSL Augmentation Utilities
# =============================================================================


def _temporal_crop(
    feats: torch.Tensor,
    min_ratio: float = 0.6,
    max_ratio: float = 0.9,
) -> torch.Tensor:
    """Create a random temporal crop, zero-padded to original length.

    Args:
        feats: (B, T, D) feature sequence.
        min_ratio: Minimum crop ratio.
        max_ratio: Maximum crop ratio.

    Returns:
        Cropped and padded features (B, T, D).
    """
    B, T, D = feats.shape
    crop_len = max(1, min(int(T * np.random.uniform(min_ratio, max_ratio)), T))
    start = np.random.randint(0, T - crop_len + 1)
    cropped = torch.zeros_like(feats)
    cropped[:, :crop_len, :] = feats[:, start : start + crop_len, :]
    return cropped


def _shuffle_temporal(
    feats: torch.Tensor,
    prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly shuffle temporal order of frames.

    Args:
        feats: (B, T, D) feature sequence.
        prob: Per-sample shuffle probability.

    Returns:
        Tuple of (shuffled_feats, is_shuffled labels).
    """
    B, T, D = feats.shape
    shuffled = feats.clone()
    labels = torch.zeros(B, device=feats.device)
    for b in range(B):
        if np.random.random() < prob:
            perm = torch.randperm(T, device=feats.device)
            shuffled[b] = feats[b, perm]
            labels[b] = 1.0
    return shuffled, labels


# =============================================================================
# SSL Visual Temporal Encoder
# =============================================================================


class VisualTemporalEncoderSSL(nn.Module):
    """Visual temporal encoder returning both CLS token and full sequence.

    Used during Stage 1 SSL pre-training to provide sequence outputs
    for MVR reconstruction and CLS tokens for contrastive/alignment tasks.

    Args:
        d_model: Model dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode visual features with bidirectional attention.

        Args:
            x: (B, T, d_model) visual feature sequence.

        Returns:
            Tuple of (cls_token (B, d_model), seq_output (B, T, d_model)).
        """
        B, T, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_encoding[:, : T + 1, :]
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0, :], x[:, 1:, :]


# =============================================================================
# Stage 0: Kinematic Foundation Training
# =============================================================================


def train_foundation(config: BTPNConfig, args: argparse.Namespace) -> None:
    """Train the kinematic foundation model.

    Single-phase end-to-end training with cosine warmup LR schedule,
    early stopping, and periodic checkpoint saving. Uses multi-scale
    causal windowing with KinematicDataset and KinematicLoss.

    Args:
        config: BTPN configuration.
        args: Command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    logger.info("=" * 70)
    logger.info("BTPN -- Stage 0: Kinematic Foundation Model Training")
    logger.info("=" * 70)
    logger.info("Device: %s", device)
    logger.info(
        "Epochs: %d, Batch size: %d, LR: %.2e",
        config.epochs, config.batch_size, config.lr,
    )
    logger.info("Window scales: %s", config.window_scales)
    logger.info(
        "Covariance: %s (Cholesky warmup: %d epochs)",
        config.covariance_type, config.cholesky_warmup_epochs,
    )

    # -- Data --
    paths_config = _load_paths_config(args)
    train_loader, val_loader, norm_stats = create_dataloaders(
        config.__dict__, paths_config,
    )
    norm_stats.save(ckpt_dir / "norm_stats.npz")

    # Position scale factor for denormalizing MAE to mm
    pos_std = float(np.mean(norm_stats.std[[0, 1, 2, 8, 9, 10]]))
    logger.info("Position scale (mean std): %.2f mm/unit", pos_std)

    # -- Model --
    model = KinematicFoundationModel(config).to(device)
    n_params = model.get_num_parameters()
    logger.info("Model parameters: %s", f"{n_params:,}")

    breakdown = model.get_parameter_breakdown()
    for name, count in breakdown.items():
        if name != "total" and count > 0:
            logger.info("  %s: %s", name, f"{count:,}")

    # -- Optimizer --
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        min_lr=config.cosine_min_lr,
    )
    scaler = GradScaler("cuda", enabled=config.use_amp)

    # -- Loss --
    criterion = KinematicLoss(
        beta=config.beta_nll,
        lambda_position=config.lambda_position,
        lambda_quaternion=config.lambda_quaternion,
        lambda_angle=config.lambda_angle,
        lambda_jaw_state=config.lambda_jaw_state,
        lambda_calibration=config.lambda_calibration,
        min_sigma=config.min_sigma,
        max_sigma=config.max_sigma,
        min_kappa=config.min_kappa,
    )
    criterion.set_quat_norm_stats(
        torch.tensor(norm_stats.mean, dtype=torch.float32),
        torch.tensor(norm_stats.std, dtype=torch.float32),
    )

    # -- Resume --
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        meta = load_checkpoint(
            args.resume, model, optimizer, scaler=scaler,
        )
        start_epoch = meta["epoch"] + 1
        best_val_loss = meta["best_loss"]
        logger.info(
            "Resumed at epoch %d, best_val_loss=%.4f",
            start_epoch, best_val_loss,
        )

    # -- Save config --
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in config.__dict__.items()
                if not k.startswith("_")
            },
            f, indent=2, default=str,
        )

    # -- Training Loop --
    stopper = EarlyStopping(patience=config.early_stopping_patience)
    history: list[dict[str, Any]] = []

    logger.info("Starting foundation model training...")
    logger.info("-" * 70)

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        lr = scheduler.step(epoch)
        use_cholesky = config.get_cholesky_enabled(epoch)
        cal_weight = config.get_calibration_weight(epoch)

        # -- Train --
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.epochs} [train]",
            leave=False,
        )
        for batch in pbar:
            scales = [s.to(device) for s in batch["scales"]]
            target = batch["target"].to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=config.use_amp):
                outputs = model(
                    multi_scale_inputs=scales,
                    force_diagonal=not use_cholesky,
                )
                loss, _ = criterion(
                    pred=outputs, target=target,
                    use_cholesky=use_cholesky,
                    calibration_weight=cal_weight,
                )

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(n_batches, 1)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_pos_mae = 0.0
        val_rot_err = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch} [val]", leave=False,
            ):
                scales = [s.to(device) for s in batch["scales"]]
                target = batch["target"].to(device)

                with autocast("cuda", enabled=config.use_amp):
                    outputs = model(
                        multi_scale_inputs=scales,
                        force_diagonal=not use_cholesky,
                    )
                    loss, _ = criterion(
                        pred=outputs, target=target,
                        use_cholesky=use_cholesky,
                    )

                if torch.isfinite(loss):
                    val_loss += loss.item()
                    n_val += 1

                    metrics = criterion.compute_metrics(outputs, target)
                    val_pos_mae += metrics.get("position_mae_mm", 0)
                    val_rot_err += metrics.get("rotation_error_deg", 0)

        val_loss /= max(n_val, 1)
        val_pos_mae_mm = val_pos_mae / max(n_val, 1) * pos_std
        val_rot_err_deg = val_rot_err / max(n_val, 1)
        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %3d/%d | Train: %.4f | Val: %.4f | "
            "Pos MAE: %.2fmm | Rot: %.2fdeg | LR: %.2e | %.1fs",
            epoch, config.epochs, train_loss, val_loss,
            val_pos_mae_mm, val_rot_err_deg, lr, epoch_time,
        )

        if use_cholesky:
            logger.info("  [Cholesky ENABLED] Cal weight: %.3f", cal_weight)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_pos_mae_mm": val_pos_mae_mm,
            "val_rot_err_deg": val_rot_err_deg,
            "lr": lr, "time": epoch_time,
        })

        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                ckpt_dir / "best_model.pt",
                best_loss=best_val_loss, scaler=scaler,
                norm_mean=norm_stats.mean, norm_std=norm_stats.std,
            )
            logger.info("  ** New best model (val_loss=%.4f)", best_val_loss)

        # Periodic checkpoint
        if (epoch + 1) % config.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                best_loss=best_val_loss, scaler=scaler,
                norm_mean=norm_stats.mean, norm_std=norm_stats.std,
            )

        # Latest checkpoint (for resume)
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            ckpt_dir / "latest.pt",
            best_loss=best_val_loss, scaler=scaler,
            norm_mean=norm_stats.mean, norm_std=norm_stats.std,
        )

        # Early stopping
        if stopper.step(val_loss, epoch):
            logger.info(
                "Early stopping at epoch %d (patience=%d)",
                epoch, config.early_stopping_patience,
            )
            break

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info(
        "Foundation training complete. Best val loss: %.4f", best_val_loss,
    )
    logger.info("Checkpoints: %s", ckpt_dir)
    logger.info("=" * 70)


# =============================================================================
# Stage 1: Visual SSL Pre-training
# =============================================================================


def train_ssl(config: BTPNConfig, args: argparse.Namespace) -> None:
    """Train visual encoder with self-supervised tasks (Stage 1).

    Four auxiliary tasks with weighted combination:
        MVR (weight 1.0) -- Masked Visual Reconstruction
        VCL (weight 0.5) -- Visual Contrastive Learning (InfoNCE)
        VTOP (weight 0.2) -- Visual Temporal Order Prediction
        VKA (weight 0.3) -- Visual-Kinematic Alignment

    Only the visual encoder and SSL task heads are trainable.
    The kinematic prior remains frozen throughout.

    Best model is selected by validation MVR loss (primary metric).

    Args:
        config: BTPN configuration.
        args: Command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / "stage1"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    max_epochs = config.stage1_max_epochs
    patience = config.stage1_patience

    logger.info("=" * 70)
    logger.info("BTPN -- Stage 1: Visual SSL Pre-training")
    logger.info("=" * 70)
    logger.info("Device: %s", device)
    logger.info(
        "Max epochs: %d (patience=%d)", max_epochs, patience,
    )
    logger.info("Batch size: %d, LR: %.2e", config.batch_size, config.stage1_lr)
    logger.info(
        "Loss weights: MVR=%.1f, VCL=%.1f, VTOP=%.1f, VKA=%.1f",
        config.lambda_mvr, config.lambda_vcl,
        config.lambda_vtop, config.lambda_vka,
    )
    logger.info("MVR mask ratio: %.2f", config.mvr_mask_ratio)

    # -- Data --
    paths_config = _load_paths_config(args)
    train_loader, val_loader, norm_stats = create_dataloaders(
        config.__dict__, paths_config,
    )
    norm_stats.save(ckpt_dir / "norm_stats.npz")

    # -- Kinematic Prior (frozen) --
    logger.info("Loading kinematic prior...")
    kin_model = KinematicFoundationModel(config).to(device)
    if config.kinematic_checkpoint and Path(config.kinematic_checkpoint).exists():
        ckpt = torch.load(
            config.kinematic_checkpoint, map_location=device,
            weights_only=False,
        )
        kin_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "  Loaded kinematic prior (epoch %s)", ckpt.get("epoch", "?"),
        )
    else:
        logger.warning(
            "  Kinematic checkpoint not found at %s -- using random init",
            config.kinematic_checkpoint,
        )
    kin_model.eval()
    for p in kin_model.parameters():
        p.requires_grad = False

    # -- SSL Components --
    d_model = config.visual_d_model
    kin_dim = config.d_model

    visual_encoder = VisualTemporalEncoderSSL(
        d_model=d_model,
        n_layers=config.visual_n_layers,
        n_heads=config.visual_n_heads,
        dropout=config.visual_dropout,
    ).to(device)

    mvr_head = MVRHead(d_model, d_model).to(device)
    vcl_head = VCLHead(d_model, proj_dim=128).to(device)
    vtop_head = VTOPHead(d_model).to(device)
    vka_head = VKAHead(d_model, kin_dim, proj_dim=128).to(device)

    all_params = (
        list(visual_encoder.parameters())
        + list(mvr_head.parameters())
        + list(vcl_head.parameters())
        + list(vtop_head.parameters())
        + list(vka_head.parameters())
    )
    n_params = sum(p.numel() for p in all_params)
    logger.info("SSL trainable parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        all_params, lr=config.stage1_lr, weight_decay=config.weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=5, total_epochs=max_epochs,
    )
    scaler = GradScaler("cuda", enabled=config.use_amp)

    # -- Save config --
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in config.__dict__.items()
                if not k.startswith("_")
            },
            f, indent=2, default=str,
        )

    # -- Training Loop --
    stopper = EarlyStopping(patience=patience)
    best_val_mvr = float("inf")
    history: list[dict[str, Any]] = []

    logger.info(
        "Starting Stage 1 SSL training (max %d epochs, patience %d)...",
        max_epochs, patience,
    )
    logger.info("-" * 70)

    def _ssl_forward(
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run all four SSL tasks on a batch and return losses.

        Args:
            batch: Data batch from the loader.

        Returns:
            Dict of per-task scalar losses.
        """
        kin_windows = [w.to(device) for w in batch["kinematic_windows"]]

        # Extract visual features from batch if available
        if "visual_feats" in batch:
            visual_feats = batch["visual_feats"].to(device)
        else:
            # Fallback: zero features (in production, VisualTemporalDataset
            # provides full visual pipeline features)
            finest = kin_windows[0]
            B, T, _ = finest.shape
            visual_feats = torch.zeros(B, T, d_model, device=device)

        losses: dict[str, torch.Tensor] = {}

        # Task 1: MVR (Masked Visual Reconstruction)
        masked_feats, mask, orig = mvr_head.apply_mask(
            visual_feats, config.mvr_mask_ratio,
        )
        cls_masked, seq_masked = visual_encoder(masked_feats)
        losses["mvr"] = mvr_head(seq_masked, mask, orig)

        # Task 2: VCL (Visual Contrastive Learning)
        view1 = _temporal_crop(visual_feats.detach())
        view1 = view1 + torch.randn_like(view1) * 0.05
        cls_v1, _ = visual_encoder(view1)
        view2 = _temporal_crop(visual_feats.detach())
        view2 = view2 + torch.randn_like(view2) * 0.05
        cls_v2, _ = visual_encoder(view2)
        losses["vcl"] = vcl_head(cls_v1, cls_v2)

        # Task 3: VTOP (Visual Temporal Order Prediction)
        shuffled, is_shuffled = _shuffle_temporal(visual_feats.detach())
        cls_vtop, _ = visual_encoder(shuffled)
        losses["vtop"] = vtop_head(cls_vtop, is_shuffled)

        # Task 4: VKA (Visual-Kinematic Alignment)
        cls_clean, _ = visual_encoder(visual_feats)
        with torch.no_grad():
            kin_out = kin_model(
                multi_scale_inputs=kin_windows, force_diagonal=True,
            )
            # Use scale representation if available, else CLS-sized zeros
            kin_embedding = kin_out.get(
                "scale_representation", cls_clean.detach(),
            )
        losses["vka"] = vka_head(cls_clean, kin_embedding)

        return losses

    for epoch in range(max_epochs):
        epoch_start = time.time()
        lr = scheduler.step(epoch)

        # -- Train --
        visual_encoder.train()
        mvr_head.train()
        vcl_head.train()
        vtop_head.train()
        vka_head.train()

        loss_accum: dict[str, float] = {
            "total": 0.0, "mvr": 0.0, "vcl": 0.0, "vtop": 0.0, "vka": 0.0,
        }
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{max_epochs} [SSL train]",
            leave=False,
        )
        for batch in pbar:
            optimizer.zero_grad()
            with autocast("cuda", enabled=config.use_amp):
                ssl_losses = _ssl_forward(batch)
                total_loss = (
                    config.lambda_mvr * ssl_losses["mvr"]
                    + config.lambda_vcl * ssl_losses["vcl"]
                    + config.lambda_vtop * ssl_losses["vtop"]
                    + config.lambda_vka * ssl_losses["vka"]
                )

            if not torch.isfinite(total_loss):
                optimizer.zero_grad()
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_accum["total"] += total_loss.item()
            for k in ("mvr", "vcl", "vtop", "vka"):
                loss_accum[k] += ssl_losses[k].item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        train_avg = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}

        # -- Validate --
        visual_encoder.eval()
        mvr_head.eval()
        vcl_head.eval()
        vtop_head.eval()
        vka_head.eval()

        val_accum: dict[str, float] = {
            "total": 0.0, "mvr": 0.0, "vcl": 0.0, "vtop": 0.0, "vka": 0.0,
        }
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                with autocast("cuda", enabled=config.use_amp):
                    ssl_losses = _ssl_forward(batch)
                    total_loss = (
                        config.lambda_mvr * ssl_losses["mvr"]
                        + config.lambda_vcl * ssl_losses["vcl"]
                        + config.lambda_vtop * ssl_losses["vtop"]
                        + config.lambda_vka * ssl_losses["vka"]
                    )

                if torch.isfinite(total_loss):
                    val_accum["total"] += total_loss.item()
                    for k in ("mvr", "vcl", "vtop", "vka"):
                        val_accum[k] += ssl_losses[k].item()
                    n_val += 1

        val_avg = {k: v / max(n_val, 1) for k, v in val_accum.items()}
        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %3d/%d | "
            "Train: %.4f (MVR=%.4f VCL=%.4f VTOP=%.4f VKA=%.4f) | "
            "Val MVR: %.4f | LR: %.2e | %.1fs",
            epoch, max_epochs,
            train_avg["total"], train_avg["mvr"], train_avg["vcl"],
            train_avg["vtop"], train_avg["vka"],
            val_avg["mvr"], lr, epoch_time,
        )

        history.append({
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_mvr": train_avg["mvr"],
            "train_vcl": train_avg["vcl"],
            "train_vtop": train_avg["vtop"],
            "train_vka": train_avg["vka"],
            "val_total": val_avg["total"],
            "val_mvr": val_avg["mvr"],
            "val_vcl": val_avg["vcl"],
            "val_vtop": val_avg["vtop"],
            "val_vka": val_avg["vka"],
            "lr": lr,
            "time": epoch_time,
        })

        # Best model (based on validation MVR loss)
        if val_avg["mvr"] < best_val_mvr:
            best_val_mvr = val_avg["mvr"]
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_mvr": best_val_mvr,
                    "visual_encoder_state_dict": visual_encoder.state_dict(),
                    "mvr_head_state_dict": mvr_head.state_dict(),
                    "vcl_head_state_dict": vcl_head.state_dict(),
                    "vtop_head_state_dict": vtop_head.state_dict(),
                    "vka_head_state_dict": vka_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "stage": "stage1_ssl",
                },
                ckpt_dir / "best_model.pt",
            )
            logger.info("  ** New best val MVR: %.6f", best_val_mvr)

        # Latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_mvr": best_val_mvr,
                    "visual_encoder_state_dict": visual_encoder.state_dict(),
                    "mvr_head_state_dict": mvr_head.state_dict(),
                    "vcl_head_state_dict": vcl_head.state_dict(),
                    "vtop_head_state_dict": vtop_head.state_dict(),
                    "vka_head_state_dict": vka_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "stage": "stage1_ssl",
                },
                ckpt_dir / "latest.pt",
            )

        # Save history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        # Early stopping
        if stopper.step(val_avg["mvr"], epoch):
            logger.info(
                "Early stopping at epoch %d (patience=%d, best MVR=%.6f)",
                epoch, patience, best_val_mvr,
            )
            break

    logger.info("=" * 70)
    logger.info("Stage 1 complete. Best val MVR: %.6f", best_val_mvr)
    logger.info("Checkpoint: %s", ckpt_dir / "best_model.pt")
    logger.info("=" * 70)


# =============================================================================
# Stage 2: Supervised 3-Phase Training
# =============================================================================


def _create_param_groups(
    model: BTPN,
    config: BTPNConfig,
    phase: str,
) -> list[dict[str, Any]]:
    """Create parameter groups with phase-aware learning rates.

    During warmup, only fusion and head parameters are trainable.
    During full and finetune phases, visual components are unfrozen
    with scaled LR (0.5x) to preserve Stage 1 features.

    Args:
        model: Full BTPN model.
        config: Configuration.
        phase: Training phase ("warmup", "full", "finetune").

    Returns:
        List of parameter group dicts for the optimizer.
    """
    visual_proj_params: list[nn.Parameter] = []
    encoder_params: list[nn.Parameter] = []
    fusion_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []

    visual_prefixes = (
        "seg_neck_proj", "seg_backbone_proj", "seg_fusion",
        "depth_proj", "scene_fusion", "visual_proj",
        "pose_kp_proj", "pose_enc_proj", "pose_geo_proj", "pose_fusion",
    )
    encoder_prefixes = ("clinical_encoder",)
    fusion_prefixes = ("kin_embed_proj", "kv_fusion")
    head_prefixes = ("confidence_gate", "residual_head")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in visual_prefixes):
            visual_proj_params.append(param)
        elif any(name.startswith(p) for p in encoder_prefixes):
            encoder_params.append(param)
        elif any(name.startswith(p) for p in fusion_prefixes):
            fusion_params.append(param)
        elif any(name.startswith(p) for p in head_prefixes):
            head_params.append(param)

    lr = config.stage2_lr
    if phase == "finetune":
        lr *= config.stage2_finetune_lr_scale

    if phase == "warmup":
        groups = [
            {"params": fusion_params, "lr": lr, "name": "fusion"},
            {"params": head_params, "lr": lr, "name": "heads"},
        ]
    else:
        groups = [
            {"params": visual_proj_params, "lr": lr * 0.5, "name": "visual_proj"},
            {"params": encoder_params, "lr": lr * 0.5, "name": "encoder"},
            {"params": fusion_params, "lr": lr, "name": "fusion"},
            {"params": head_params, "lr": lr, "name": "heads"},
        ]

    # Filter out empty groups
    groups = [g for g in groups if len(g["params"]) > 0]
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        logger.info("  %s: %s params, lr=%.2e", g["name"], f"{n:,}", g["lr"])

    return groups


def train_supervised(config: BTPNConfig, args: argparse.Namespace) -> None:
    """Train the full BTPN with 3-phase supervised learning (Stage 2).

    Phase progression:
        warmup (0 to stage2_warmup_epochs):
            Only fusion + gate + residual heads trainable.
        full (stage2_warmup_epochs to stage2_full_end_epoch):
            All visual parameters unfrozen with cosine LR decay.
        finetune (after stage2_full_end_epoch):
            Reduced LR, stricter confidence threshold, pivot active.

    On each phase transition the optimizer is rebuilt with new parameter
    groups and the patience counter is reset.

    Args:
        config: BTPN configuration.
        args: Command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / "stage2"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    logger.info("=" * 70)
    logger.info("BTPN -- Stage 2: Supervised Pose Prediction")
    logger.info("=" * 70)
    logger.info("Device: %s", device)
    logger.info(
        "Max epochs: %d, Batch size: %d, LR: %.2e",
        config.stage2_max_epochs, config.batch_size, config.stage2_lr,
    )
    logger.info(
        "Phases: warmup(0-%d), full(%d-%d), finetune(%d-%d)",
        config.stage2_warmup_epochs, config.stage2_warmup_epochs,
        config.stage2_full_end_epoch, config.stage2_full_end_epoch,
        config.stage2_max_epochs,
    )
    logger.info(
        "Gate ceilings: pos=%.2f, rot=%.2f, angle=%.2f",
        config.max_gate_position, config.max_gate_rotation,
        config.max_gate_angle,
    )
    logger.info("Relative tracking: %s", config.use_relative_tracking)
    logger.info("Pivot estimation: %s", config.use_pivot_estimation)

    # -- Data --
    paths_config = _load_paths_config(args)
    train_loader, val_loader, norm_stats = create_dataloaders(
        {**config.__dict__, "use_visual_features": True}, paths_config,
    )
    norm_stats.save(ckpt_dir / "norm_stats.npz")

    pos_std = float(np.mean(norm_stats.std[[0, 1, 2, 8, 9, 10]]))
    logger.info("Position scale: %.2f mm/unit", pos_std)

    # -- Kinematic Prior --
    logger.info("Loading kinematic prior...")
    kin_model = KinematicFoundationModel(config).to(device)
    if config.kinematic_checkpoint and Path(config.kinematic_checkpoint).exists():
        ckpt = torch.load(
            config.kinematic_checkpoint, map_location=device,
            weights_only=False,
        )
        kin_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "  Loaded kinematic prior (epoch %s)", ckpt.get("epoch", "?"),
        )
    kin_model.eval()

    # -- Model --
    model = BTPN(config, kinematic_model=kin_model).to(device)

    # Load Stage 1 weights if available
    stage1_ckpt = getattr(args, "stage1_checkpoint", None)
    if stage1_ckpt is None:
        default_s1 = (
            Path(args.output_dir) / "stage1" / "checkpoints" / "best_model.pt"
        )
        if default_s1.exists():
            stage1_ckpt = str(default_s1)

    if stage1_ckpt and Path(stage1_ckpt).exists():
        logger.info("Loading Stage 1 weights from: %s", stage1_ckpt)
        model.load_stage1_weights(stage1_ckpt)
    else:
        logger.warning(
            "  No Stage 1 checkpoint found. Training visual encoder from scratch.",
        )
        stage1_ckpt = None

    breakdown = model.get_parameter_breakdown()
    for k, v in breakdown.items():
        if v > 0:
            logger.info("  %s: %s", k, f"{v:,}")

    # -- Loss --
    criterion = BTPNLoss(
        beta=config.beta_nll,
        lambda_position=config.lambda_position,
        lambda_quaternion=config.lambda_quaternion,
        lambda_angle=config.lambda_angle,
        lambda_jaw_state=config.lambda_jaw_state,
        lambda_residual_reg=config.lambda_residual_reg,
        lambda_displacement=config.lambda_displacement,
        lambda_gate_entropy=config.gate_entropy_weight,
        lambda_pivot=config.lambda_pivot,
        lambda_smoothness=config.lambda_smoothness,
        min_sigma=config.min_sigma,
        max_sigma=config.max_sigma,
        min_kappa=config.min_kappa,
    )
    criterion.set_quat_norm_stats(
        torch.tensor(norm_stats.mean, dtype=torch.float32),
        torch.tensor(norm_stats.std, dtype=torch.float32),
    )

    # -- Initial phase --
    phase = config.get_stage2_phase(0)
    model.set_training_phase(phase)
    logger.info("Initial phase: %s", phase)

    param_groups = _create_param_groups(model, config, phase)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config.weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=5, total_epochs=config.stage2_max_epochs,
    )
    scaler = GradScaler("cuda", enabled=config.use_amp)

    # -- Save config --
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in config.__dict__.items()
                if not k.startswith("_")
            },
            f, indent=2, default=str,
        )

    # -- Resume --
    start_epoch = 0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    if args.resume:
        ckpt_data = torch.load(
            args.resume, map_location=device, weights_only=False,
        )
        model.load_state_dict(ckpt_data["model_state_dict"])
        start_epoch = ckpt_data.get("epoch", 0) + 1
        best_val_loss = ckpt_data.get("best_val_loss", float("inf"))
        logger.info(
            "Resumed at epoch %d, best_val_loss=%.4f",
            start_epoch, best_val_loss,
        )

        # Restore correct phase for the resumed epoch
        phase = config.get_stage2_phase(start_epoch)
        model.set_training_phase(phase)

        # Load history if available
        history_path = output_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            logger.info("  Restored %d history records", len(history))

    # -- Training Loop --
    stopper = EarlyStopping(patience=config.stage2_patience)

    logger.info("Starting Stage 2 supervised training...")
    logger.info("-" * 70)

    for epoch in range(start_epoch, config.stage2_max_epochs):
        epoch_start = time.time()

        # Phase transition
        new_phase = config.get_stage2_phase(epoch)
        if new_phase != phase:
            phase = new_phase
            logger.info(
                "\n%s Phase transition: %s %s",
                "=" * 30, phase, "=" * 30,
            )
            model.set_training_phase(phase)
            param_groups = _create_param_groups(model, config, phase)
            optimizer = torch.optim.AdamW(
                param_groups, weight_decay=config.weight_decay,
            )
            scheduler = CosineWarmupScheduler(
                optimizer, warmup_epochs=5,
                total_epochs=config.stage2_max_epochs - epoch,
            )
            # Reset early stopping on phase transition
            stopper = EarlyStopping(patience=config.stage2_patience)

        lr = scheduler.step(epoch)

        # Phase-aware parameters
        conf_threshold = config.get_conf_threshold(epoch)
        residual_reg = config.get_residual_reg_weight(epoch)
        pivot_w = config.get_pivot_warmup_weight(epoch)

        # -- Train --
        model.train()
        model.kinematic_prior.eval()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.stage2_max_epochs} [{phase}]",
            leave=False,
        )
        for batch in pbar:
            kin_windows = [w.to(device) for w in batch["kinematic_windows"]]
            seg_neck = [w.to(device) for w in batch["seg_neck_windows"]]
            seg_backbone = [
                w.to(device) for w in batch["seg_backbone_windows"]
            ]
            depth = [w.to(device) for w in batch["depth_windows"]]
            det_conf = [w.to(device) for w in batch["detection_conf"]]
            target = batch["target"].to(device)
            current_pos = batch.get("current_position")
            if current_pos is not None:
                current_pos = current_pos.to(device)

            pose_kp = pose_bb = pose_geo = pose_conf = None
            if "pose_kp_windows" in batch:
                pose_kp = [
                    w.to(device) for w in batch["pose_kp_windows"]
                ]
                pose_bb = [
                    w.to(device) for w in batch["pose_backbone_windows"]
                ]
                pose_geo = [
                    w.to(device) for w in batch["pose_geometric_windows"]
                ]
                pose_conf = [w.to(device) for w in batch["pose_conf"]]

            optimizer.zero_grad()
            with autocast("cuda", enabled=config.use_amp):
                outputs = model(
                    kinematic_windows=kin_windows,
                    seg_neck_windows=seg_neck,
                    seg_backbone_windows=seg_backbone,
                    depth_windows=depth,
                    detection_conf=det_conf,
                    pose_kp_windows=pose_kp,
                    pose_backbone_windows=pose_bb,
                    pose_geometric_windows=pose_geo,
                    pose_conf=pose_conf,
                    current_position=current_pos,
                )

                if "target_displacement" in batch:
                    outputs["target_displacement"] = (
                        batch["target_displacement"].to(device)
                    )

                # Compute pivot loss if estimator is initialized
                pivot_loss_val = None
                if (
                    hasattr(model, "pivot_estimator")
                    and hasattr(model.pivot_estimator, "is_initialized")
                    and model.pivot_estimator.is_initialized
                    and pivot_w > 0
                ):
                    pivot_loss_val = (
                        model.pivot_estimator.pivot_consistency_loss(
                            outputs["mu_position"],
                            outputs["mu_quaternion"],
                        )
                    )

                loss, _ = criterion(
                    pred=outputs,
                    target=target,
                    detection_conf=det_conf,
                    conf_threshold=conf_threshold,
                    lambda_residual_reg=residual_reg,
                    pivot_loss=pivot_loss_val,
                    pivot_weight=pivot_w,
                )

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.2f}")

        train_loss /= max(n_batches, 1)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_pos_mae = 0.0
        val_rot_err = 0.0
        val_gate_accum: dict[str, list[float]] = {}
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch} [val]",
                leave=False,
            ):
                kin_windows = [
                    w.to(device) for w in batch["kinematic_windows"]
                ]
                seg_neck = [
                    w.to(device) for w in batch["seg_neck_windows"]
                ]
                seg_backbone = [
                    w.to(device) for w in batch["seg_backbone_windows"]
                ]
                depth = [w.to(device) for w in batch["depth_windows"]]
                det_conf = [w.to(device) for w in batch["detection_conf"]]
                target = batch["target"].to(device)
                current_pos = batch.get("current_position")
                if current_pos is not None:
                    current_pos = current_pos.to(device)

                pose_kp = pose_bb = pose_geo = pose_conf = None
                if "pose_kp_windows" in batch:
                    pose_kp = [
                        w.to(device) for w in batch["pose_kp_windows"]
                    ]
                    pose_bb = [
                        w.to(device)
                        for w in batch["pose_backbone_windows"]
                    ]
                    pose_geo = [
                        w.to(device)
                        for w in batch["pose_geometric_windows"]
                    ]
                    pose_conf = [w.to(device) for w in batch["pose_conf"]]

                outputs = model(
                    kinematic_windows=kin_windows,
                    seg_neck_windows=seg_neck,
                    seg_backbone_windows=seg_backbone,
                    depth_windows=depth,
                    detection_conf=det_conf,
                    pose_kp_windows=pose_kp,
                    pose_backbone_windows=pose_bb,
                    pose_geometric_windows=pose_geo,
                    pose_conf=pose_conf,
                    current_position=current_pos,
                )

                if "target_displacement" in batch:
                    outputs["target_displacement"] = (
                        batch["target_displacement"].to(device)
                    )

                loss, _ = criterion(
                    pred=outputs, target=target, detection_conf=det_conf,
                    conf_threshold=conf_threshold,
                )
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    n_val += 1

                # Evaluation metrics
                metrics = criterion.compute_metrics(outputs, target)
                val_pos_mae += metrics.get("position_mae_mm", 0)
                val_rot_err += metrics.get("rotation_error_deg", 0)

                # Per-channel gate values
                if "gates" in outputs and outputs["gates"]:
                    for gname, gval in outputs["gates"].items():
                        if gname not in val_gate_accum:
                            val_gate_accum[gname] = []
                        val_gate_accum[gname].append(gval.mean().item())

        val_loss /= max(n_val, 1)
        val_pos_mm = val_pos_mae / max(n_val, 1) * pos_std
        val_rot_deg = val_rot_err / max(n_val, 1)
        epoch_time = time.time() - epoch_start

        # Gate summaries
        pg = float(np.mean(val_gate_accum.get("pos_gate", [0])))
        rg = float(np.mean(val_gate_accum.get("rot_gate", [0])))
        ag = float(np.mean(val_gate_accum.get("angle_gate", [0])))

        logger.info(
            "Epoch %3d/%d [%7s] | Train: %.2f | Val: %.2f | "
            "Pos: %.2fmm | Rot: %.1fdeg | "
            "PG: %.2f RG: %.2f AG: %.2f | LR: %.2e | %.1fs",
            epoch, config.stage2_max_epochs, phase,
            train_loss, val_loss, val_pos_mm, val_rot_deg,
            pg, rg, ag, lr, epoch_time,
        )

        if pivot_w > 0:
            logger.info("  [Pivot] warmup weight: %.2f", pivot_w)

        # -- History --
        record: dict[str, Any] = {
            "epoch": epoch, "phase": phase,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_pos_mae_mm": val_pos_mm, "val_rot_err_deg": val_rot_deg,
            "gate_pos_mean": pg, "gate_rot_mean": rg, "gate_angle_mean": ag,
            "lr": lr, "time": epoch_time,
        }
        history.append(record)

        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                ckpt_dir / "best_model.pt",
                best_loss=best_val_loss, scaler=scaler,
                norm_mean=norm_stats.mean, norm_std=norm_stats.std,
                phase=phase, stage1_checkpoint=stage1_ckpt,
            )
            logger.info("  ** New best val loss: %.4f", best_val_loss)

        # Latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                ckpt_dir / "latest.pt",
                best_loss=best_val_loss, scaler=scaler,
                norm_mean=norm_stats.mean, norm_std=norm_stats.std,
                phase=phase,
            )

        # Save history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        # Early stopping (not during warmup)
        if phase != "warmup" and stopper.step(val_loss, epoch):
            logger.info(
                "Early stopping at epoch %d (patience=%d)",
                epoch, config.stage2_patience,
            )
            break

    logger.info("=" * 70)
    logger.info("Stage 2 complete. Best val loss: %.4f", best_val_loss)
    logger.info("Checkpoint: %s", ckpt_dir / "best_model.pt")
    logger.info("=" * 70)


# =============================================================================
# Stage: Detection Training
# =============================================================================


def train_detection(config: BTPNConfig, args: argparse.Namespace) -> None:
    """Train YOLO detection models.

    Supports segmentation and keypoint (pose) model training via
    the DetectionPipeline wrapper around ultralytics YOLO.

    Args:
        config: BTPN configuration (unused for detection but kept
            for uniform dispatch signature).
        args: Command-line arguments. Must include --task and --data-yaml.
    """
    det_config = DetectionConfig()

    # Apply output dir if specified
    if args.output_dir:
        det_config.project_dir = str(Path(args.output_dir) / "detection")

    pipeline = DetectionPipeline()
    task = getattr(args, "task", "segmentation")
    data_yaml = getattr(args, "data_yaml", None)

    if task == "segmentation":
        if not data_yaml:
            raise ValueError(
                "--data-yaml is required for segmentation training"
            )
        logger.info("Training YOLO segmentation model...")
        logger.info("  Dataset: %s", data_yaml)
        logger.info(
            "  Epochs: %d, Patience: %d, Batch: %d",
            det_config.epochs, det_config.patience, det_config.batch_size,
        )
        weights = pipeline.train_segmentation(
            data_yaml, det_config, name="yolo_seg",
        )
        logger.info("Best weights: %s", weights)

    elif task in ("keypoints", "pose"):
        if not data_yaml:
            raise ValueError(
                "--data-yaml is required for keypoint training"
            )
        logger.info("Training YOLO keypoint (pose) model...")
        logger.info("  Dataset: %s", data_yaml)
        logger.info(
            "  Epochs: %d, Patience: %d, Batch: %d",
            det_config.pose_epochs, det_config.pose_patience,
            det_config.batch_size,
        )
        weights = pipeline.train_keypoints(
            data_yaml, det_config, name="yolo_pose",
        )
        logger.info("Best weights: %s", weights)

    else:
        raise ValueError(
            f"Unknown detection task: {task}. "
            "Choose from: segmentation, keypoints"
        )


# =============================================================================
# Utilities
# =============================================================================


def _load_paths_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load paths configuration from YAML or construct from args.

    Args:
        args: Command-line arguments with optional --paths field.

    Returns:
        Paths configuration dictionary.
    """
    paths_yaml = getattr(args, "paths", None)
    if paths_yaml and Path(paths_yaml).exists():
        import yaml

        with open(paths_yaml) as f:
            return yaml.safe_load(f)

    # Construct default paths from --data-root
    data_root = getattr(args, "data_root", "data")
    return {
        "data_root": data_root,
        "dataset_a": "7DOF2024",
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Parse arguments and dispatch to the appropriate training function."""
    parser = argparse.ArgumentParser(
        description="BTPN Unified Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --stage foundation --config configs/kinematic_foundation.yaml
  python scripts/train.py --stage ssl --config configs/btpn.yaml
  python scripts/train.py --stage supervised --config configs/btpn.yaml
  python scripts/train.py --stage detection --task segmentation --data-yaml data/yolo/seg/data.yaml
  python scripts/train.py --stage detection --task keypoints --data-yaml data/yolo/pose/data.yaml
""",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["foundation", "ssl", "supervised", "detection"],
        help="Training stage to run.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file path.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/btpn",
        help="Output directory.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config).",
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root directory for datasets.",
    )
    parser.add_argument(
        "--paths", type=str, default=None,
        help="Paths YAML config file.",
    )

    # Stage-specific arguments
    parser.add_argument(
        "--stage1-checkpoint", type=str, default=None,
        help="Stage 1 checkpoint for supervised training.",
    )
    parser.add_argument(
        "--task", type=str, default="segmentation",
        choices=["segmentation", "keypoints", "pose"],
        help="Detection task (for --stage detection).",
    )
    parser.add_argument(
        "--data-yaml", type=str, default=None,
        help="YOLO dataset YAML (for detection training).",
    )

    # Override arguments
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    if args.config and Path(args.config).exists():
        config = BTPNConfig.from_yaml(args.config)
        logger.info("Loaded config from: %s", args.config)
    else:
        config = BTPNConfig()
        if args.config:
            logger.warning(
                "Config file not found: %s -- using defaults", args.config,
            )

    # Apply CLI overrides
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.epochs = args.epochs
        config.stage1_max_epochs = args.epochs
        config.stage2_max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
        config.stage1_lr = args.lr
        config.stage2_lr = args.lr

    # Dispatch
    dispatch: dict[str, Any] = {
        "foundation": train_foundation,
        "ssl": train_ssl,
        "supervised": train_supervised,
        "detection": train_detection,
    }
    dispatch[args.stage](config, args)


if __name__ == "__main__":
    main()
