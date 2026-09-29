"""Loss functions for BTPN pose prediction with uncertainty quantification.

This module provides the complete loss function hierarchy for training both
the kinematic foundation model and the full visual-temporal BTPN:

Standalone Functions:
    - ``geodesic_distance``  -- SO(3) geodesic between unit quaternions.
    - ``beta_nll_loss``      -- Beta-NLL diagonal Gaussian (Seitzer et al., 2022).
    - ``beta_vmf_loss``      -- Beta-VMF on S^3 for rotation uncertainty.
    - ``jaw_state_loss``     -- Binary CE for open/closed jaw classification.
    - ``calibrate_jaw_angle``-- Voltage to 0-100 % opening via percentile method.
    - ``differentiable_ece`` -- Differentiable Expected Calibration Error.

Loss Modules:
    - ``KinematicLoss``  -- Foundation model training.  Position Beta-NLL +
      rotation Beta-VMF + jaw angle + ECE + jaw-state BCE.
    - ``BTPNLoss``       -- Full BTPN training.  Wraps ``KinematicLoss`` and
      adds gate entropy, displacement consistency, pivot residual, smoothness,
      and residual regularisation terms.

Both modules expose ``set_quat_norm_stats()`` to undo z-score normalisation on
quaternion targets before computing geodesic metrics -- a critical correctness
requirement (without it, geodesic error is ~90 deg instead of ~12 deg).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Quaternion Utilities
# ============================================================================


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise quaternions to unit length.

    Args:
        q: Quaternions (..., 4) in ``[w, x, y, z]`` order.
        eps: Clamping floor for the norm denominator.

    Returns:
        Unit quaternions of the same shape.
    """
    norm = torch.norm(q, dim=-1, keepdim=True).clamp(min=eps)
    return q / norm


def quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``[w, x, y, z]`` to 3x3 rotation matrices.

    Args:
        q: Quaternions (..., 4).

    Returns:
        Rotation matrices (..., 3, 3).
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


# ============================================================================
# Geodesic Distance
# ============================================================================


def geodesic_distance(
    q1: torch.Tensor,
    q2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Geodesic (angular) distance between two quaternions on SO(3).

    ``d(q1, q2) = 2 * arccos(|q1 . q2|)``

    The absolute dot product handles the sign ambiguity of quaternions
    (``q`` and ``-q`` represent the same rotation).

    Args:
        q1: First quaternion (..., 4) in ``[w, x, y, z]`` order.
        q2: Second quaternion (..., 4).
        eps: Numerical-stability margin for ``arccos``.

    Returns:
        Geodesic distance in radians (...,).
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = torch.abs(torch.sum(q1 * q2, dim=-1))
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    return 2.0 * torch.acos(dot)


def geodesic_loss(
    pred_quat: torch.Tensor,
    target_quat: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean geodesic loss for quaternion predictions.

    Args:
        pred_quat: Predicted quaternions (..., 4).
        target_quat: Target quaternions (..., 4).
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.

    Returns:
        Geodesic loss (scalar or per-element).
    """
    distance = geodesic_distance(pred_quat, target_quat)
    if reduction == "mean":
        return distance.mean()
    if reduction == "sum":
        return distance.sum()
    return distance


# ============================================================================
# Beta-NLL Loss (Diagonal Gaussian)
# ============================================================================


def beta_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
    reduction: str = "mean",
    min_sigma: float = 1e-4,
    max_sigma: float = 10.0,
) -> torch.Tensor:
    """Beta-NLL loss for diagonal Gaussian (Seitzer et al., 2022).

    Standard NLL permits sigma-collapse (sigma -> 0). Beta-NLL re-weights
    both the squared-error and log-variance terms so that neither dominates:

    .. math::

        L_{\\beta} = \\sigma^{2\\beta} \\frac{(y - \\mu)^2}{\\sigma^2}
                   + \\sigma^{2(1 - \\beta)} \\log \\sigma^2

    With ``beta = 0.5`` both terms are weighted by ``sigma``, giving balanced
    gradients.

    Args:
        mu: Predicted mean (..., D).
        sigma: Predicted standard deviation (..., D).
        target: Target values (..., D).
        beta: Beta parameter (0.5 = balanced).
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
        min_sigma: Lower sigma clamp.
        max_sigma: Upper sigma clamp.

    Returns:
        Beta-NLL loss value.
    """
    sigma = torch.clamp(sigma, min=min_sigma, max=max_sigma)
    variance = sigma ** 2
    squared_error = (target - mu) ** 2

    # Detach variance for the beta-weighting (Seitzer et al.)
    error_term = (variance.detach() ** beta) * squared_error / variance
    log_term = (variance.detach() ** (1 - beta)) * torch.log(variance)

    loss = 0.5 * (error_term + log_term)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ============================================================================
# VMF Normalisation on S^3
# ============================================================================


def _vmf_log_normalizer_s3(kappa: torch.Tensor) -> torch.Tensor:
    """Log normalisation constant log C_4(kappa) for the von Mises-Fisher on S^3.

    For p = 4 dimensions, C_p(kappa) = kappa^(p/2 - 1) / ((2 pi)^(p/2) I_(p/2 - 1)(kappa)),
    so

    .. math::

        \\log C_4(\\kappa) = \\log \\kappa - \\log(4\\pi^2) - \\log I_1(\\kappa)

    where I_1 is the modified Bessel function of the first kind. log I_1 is
    computed stably as ``log(i1e(kappa)) + kappa`` (i1e is the exponentially
    scaled Bessel function), which is accurate for small and large kappa alike.
    As kappa -> 0, C_4 -> 1 / (2 pi^2), the uniform density on S^3.

    Args:
        kappa: Concentration parameter (...,), kappa > 0.

    Returns:
        Log normalisation constant (...,).
    """
    log_i1 = torch.log(torch.special.i1e(kappa)) + kappa
    return torch.log(kappa) - math.log(4.0 * math.pi ** 2) - log_i1


# ============================================================================
# Beta-VMF Loss (Rotation Uncertainty)
# ============================================================================


def beta_vmf_loss(
    mu_quat: torch.Tensor,
    kappa: torch.Tensor,
    target_quat: torch.Tensor,
    beta: float = 0.5,
    reduction: str = "mean",
    min_kappa: float = 1.0,
    max_kappa: float = 500.0,
) -> torch.Tensor:
    """Beta-weighted von Mises-Fisher NLL for quaternion rotations on S^3.

    Quaternions q and -q encode the same rotation, so the likelihood is the
    antipodally symmetric vMF

    .. math::

        p(q \\mid \\mu, \\kappa) = C_4(\\kappa) \\cosh(\\kappa \\, \\mu \\cdot q),

    which is exactly normalised (the odd part of exp(kappa mu.q) integrates to
    zero over S^3). The negative log-likelihood is

    .. math::

        \\mathrm{NLL} = -\\log\\cosh(\\kappa |\\mu \\cdot q|) - \\log C_4(\\kappa).

    Following beta-NLL (Seitzer et al., 2022), each sample is weighted by the
    stop-gradient variance raised to ``beta``. Rotational variance scales as
    1 / kappa, so the weight is ``stopgrad(kappa) ** -beta``. ``beta = 0``
    recovers the plain NLL; the weight never changes where the optimum lies
    for a given sample, only how much each sample counts.

    Args:
        mu_quat: Predicted mean quaternion (..., 4).
        kappa: Predicted concentration (..., 1) or (...,).
        target_quat: Target quaternion (..., 4).
        beta: Beta-NLL exponent in [0, 1].
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
        min_kappa: Lower kappa clamp.
        max_kappa: Upper kappa clamp.

    Returns:
        Beta-weighted vMF NLL.
    """
    mu_quat = normalize_quaternion(mu_quat)
    target_quat = normalize_quaternion(target_quat)

    kappa = kappa.squeeze(-1) if kappa.dim() > mu_quat.dim() - 1 else kappa
    kappa = torch.clamp(kappa, min=min_kappa, max=max_kappa)

    dot = torch.abs(torch.sum(mu_quat * target_quat, dim=-1))
    kd = kappa * dot
    log_cosh = kd + torch.log1p(torch.exp(-2.0 * kd)) - math.log(2.0)
    nll = -log_cosh - _vmf_log_normalizer_s3(kappa)

    loss = kappa.detach() ** (-beta) * nll

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ============================================================================
# Jaw-State Loss & Calibration
# ============================================================================


def jaw_state_loss(
    logits: torch.Tensor,
    target_pct: torch.Tensor,
    threshold: float = 50.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross-entropy loss for jaw open/closed prediction.

    Args:
        logits: Predicted logits (B, 2) for both tools.
        target_pct: Calibrated jaw opening percentage (B, 2).
        threshold: Percentage threshold for open/closed (default 50 %).
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.

    Returns:
        BCE loss.
    """
    target_labels = (target_pct >= threshold).float()
    return F.binary_cross_entropy_with_logits(logits, target_labels, reduction=reduction)


def calibrate_jaw_angle(
    raw_voltage: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
) -> torch.Tensor:
    """Calibrate raw voltage to 0-100 % opening via percentile method.

    Uses the 10-90 percentile bounds computed per trial:

    ``calibrated = (voltage - P_lower) / (P_upper - P_lower) * 100``

    Args:
        raw_voltage: Raw voltage values (...).
        lower_bound: Lower percentile value (per trial).
        upper_bound: Upper percentile value (per trial).

    Returns:
        Calibrated percentage clipped to [0, 100].
    """
    eps = 1e-8
    range_val = upper_bound - lower_bound + eps
    calibrated = (raw_voltage - lower_bound) / range_val * 100.0
    return torch.clamp(calibrated, 0.0, 100.0)


# ============================================================================
# Differentiable Expected Calibration Error
# ============================================================================


def _normal_quantile(p: float) -> float:
    """Approximate inverse normal CDF (Abramowitz & Stegun).

    Args:
        p: Probability in (0, 1).

    Returns:
        z such that ``P(Z < z) = p``.
    """
    t = math.sqrt(-2.0 * math.log(p if p <= 0.5 else 1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t ** 3)
    return -z if p <= 0.5 else z


def differentiable_ece(
    predicted_sigma: torch.Tensor,
    errors: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """Differentiable Expected Calibration Error for regression.

    For each confidence quantile alpha in {0.1, ..., 0.9}, checks whether
    the fraction of absolute errors falling within ``z_alpha * sigma``
    matches the expected alpha.  A sigmoid soft-indicator provides gradients.

    Args:
        predicted_sigma: Predicted standard deviation (...,).
        errors: Absolute prediction errors ``|y - mu|`` (...,).
        n_bins: Number of confidence bins.

    Returns:
        Differentiable ECE (scalar).
    """
    sigma_flat = predicted_sigma.reshape(-1)
    errors_flat = errors.reshape(-1)

    if sigma_flat.shape[0] == 0:
        return torch.tensor(0.0, device=predicted_sigma.device)

    ece = torch.tensor(0.0, device=predicted_sigma.device)
    temperature = 0.1

    for i in range(1, n_bins):
        alpha = i / n_bins
        z = _normal_quantile((1 + alpha) / 2)
        threshold = z * sigma_flat

        # Soft indicator with sigmoid (force float32 for AMP safety)
        sigmoid_arg = (threshold - errors_flat).float() / (
            temperature * sigma_flat.float() + 1e-8
        )
        sigmoid_arg = sigmoid_arg.clamp(-20.0, 20.0)
        observed = torch.sigmoid(sigmoid_arg).mean()
        ece = ece + (observed - alpha) ** 2

    return ece / n_bins


def differentiable_ece_vmf(
    kappa: torch.Tensor,
    geodesic_errors: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """Differentiable ECE for vMF rotation uncertainty.

    Uses the vMF inverse CDF approximation on geodesic errors:

    ``theta_alpha = arccos(1 - log(1 / (1 - alpha)) / kappa)``

    Args:
        kappa: VMF concentration (...,).
        geodesic_errors: Geodesic angular errors in radians (...,).
        n_bins: Number of confidence bins.

    Returns:
        Differentiable ECE (scalar).
    """
    kappa_flat = kappa.reshape(-1).clamp(min=1.0)
    errors_flat = geodesic_errors.reshape(-1)

    if kappa_flat.shape[0] == 0:
        return torch.tensor(0.0, device=kappa.device)

    ece = torch.tensor(0.0, device=kappa.device)
    temperature = 0.1

    for i in range(1, n_bins):
        alpha = i / n_bins
        log_inv = math.log(1.0 / (1.0 - alpha))
        cos_threshold = (1.0 - log_inv / kappa_flat).clamp(-1.0, 1.0)
        theta_alpha = torch.acos(cos_threshold)

        inv_kappa = 1.0 / kappa_flat
        sigmoid_arg = (theta_alpha - errors_flat).float() / (
            temperature * inv_kappa.float() + 1e-8
        )
        sigmoid_arg = sigmoid_arg.clamp(-20.0, 20.0)
        observed = torch.sigmoid(sigmoid_arg).mean()
        ece = ece + (observed - alpha) ** 2

    return ece / n_bins


# ============================================================================
# Trocar (Entry Point) Constraint
# ============================================================================


def compute_trocar_residual(
    mu_position: torch.Tensor,
    mu_quaternion: torch.Tensor,
    trocar_pos: torch.Tensor,
    shaft_axis: torch.Tensor,
) -> torch.Tensor:
    """Perpendicular distance from the trocar to the predicted tool shaft line.

    For a tool with tip position *P*, world-frame shaft direction *d*, and
    trocar position *T*:

    ``residual = ||(T - P) - ((T - P) . d) d||``

    A physically valid pose has the shaft passing through the trocar, so
    the residual should be approximately 0 mm.

    Args:
        mu_position: Tool tip positions (B, 2, 3) in mm.
        mu_quaternion: Tool quaternions (B, 2, 4) ``[w, x, y, z]``.
        trocar_pos: Trocar positions (B, 2, 3) in mm.
        shaft_axis: Local shaft axis (B, 2, 3) unit vector.

    Returns:
        Trocar residuals (B, 2) in mm.
    """
    R = quat_to_rotation_matrix(mu_quaternion)  # (B, 2, 3, 3)
    shaft_world = torch.matmul(R, shaft_axis.unsqueeze(-1)).squeeze(-1)
    shaft_world = F.normalize(shaft_world, dim=-1)

    v = trocar_pos - mu_position
    dot = (v * shaft_world).sum(dim=-1, keepdim=True)
    perp = v - dot * shaft_world

    return torch.norm(perp, dim=-1)  # (B, 2)


# ============================================================================
# Gate Entropy Regularisation
# ============================================================================


def _gate_entropy_loss(gates: dict[str, torch.Tensor]) -> torch.Tensor:
    """Penalise gate saturation by encouraging binary entropy.

    For each per-channel gate value ``p in [0, 1]`` computes
    ``H(p) = -p log p - (1-p) log(1-p)`` and returns negative mean entropy
    so that minimising the loss maximises entropy.

    Args:
        gates: Dict mapping channel name to gate tensor (B, 1) in [0, 1].

    Returns:
        Negative mean entropy (scalar).
    """
    if not gates:
        return torch.tensor(0.0)

    device = next(iter(gates.values())).device
    loss = torch.tensor(0.0, device=device)
    for _name, gate in gates.items():
        p = gate.clamp(1e-6, 1 - 1e-6)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        loss = loss - entropy.mean()
    return loss / len(gates)


# ============================================================================
# Displacement Consistency Loss
# ============================================================================


def _displacement_loss(
    pred_displacement: torch.Tensor,
    target_displacement: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss on frame-to-frame displacement predictions.

    Args:
        pred_displacement: Predicted motion (B, 2, 3).
        target_displacement: Ground-truth motion (B, 2, 3).
        mask: Optional boolean mask (B,) for valid samples.

    Returns:
        Displacement MSE (scalar).
    """
    mse = ((pred_displacement - target_displacement) ** 2).mean(dim=(1, 2))
    if mask is not None and mask.any():
        mask_f = mask.float()
        return (mse * mask_f).sum() / mask_f.sum().clamp(min=1)
    return mse.mean()


# ============================================================================
# Temporal Smoothness Loss
# ============================================================================


def _temporal_smoothness_loss(
    predictions: torch.Tensor,
    prev_predictions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalise jerky position-delta predictions.

    Args:
        predictions: Position deltas (B, 2, 3).
        prev_predictions: Previous batch last deltas (2, 3), optional.

    Returns:
        Smoothness loss (scalar).
    """
    delta_norms = predictions.norm(dim=-1)
    loss = delta_norms.mean()
    if prev_predictions is not None:
        jump = (predictions[0] - prev_predictions).norm(dim=-1).mean()
        loss = loss + 0.1 * jump
    return loss


# ============================================================================
# KinematicLoss (Foundation Model)
# ============================================================================


class KinematicLoss(nn.Module):
    """Combined loss for the kinematic foundation model.

    Integrates Beta-NLL for position, Beta-VMF for rotation, Beta-NLL for
    jaw angle, BCE for jaw state, differentiable ECE for calibration, and
    an optional auxiliary reconstruction term.

    CRITICAL: call ``set_quat_norm_stats()`` before training to enable
    quaternion denormalisation.  Without it, z-scored targets produce
    ~90 deg geodesic error instead of ~12 deg.

    Attributes:
        beta: Beta-NLL / Beta-VMF parameter.
        lambda_position: Position loss weight.
        lambda_quaternion: Rotation loss weight.
        lambda_angle: Jaw angle loss weight.
        lambda_jaw_state: Jaw state BCE weight.
        lambda_calibration: Calibration ECE weight.
        lambda_reconstruction: Auxiliary reconstruction weight.
        quat_denorm_enabled: Whether quaternion denormalisation is active.
    """

    def __init__(
        self,
        beta: float = 0.5,
        lambda_position: float = 1.0,
        lambda_quaternion: float = 2.0,
        lambda_angle: float = 1.0,
        lambda_jaw_state: float = 0.5,
        lambda_calibration: float = 0.1,
        lambda_reconstruction: float = 0.1,
        min_sigma: float = 1e-4,
        max_sigma: float = 10.0,
        min_kappa: float = 1.0,
        jaw_state_threshold: float = 50.0,
        n_calibration_bins: int = 10,
        lambda_geodesic: float = 0.0,
        lambda_calibration_rotation: float = 0.0,
        lambda_calibration_rotation_vmf: float = 0.0,
        lambda_calibration_angle: float = 0.0,
    ):
        """Initialise kinematic loss.

        Args:
            beta: Beta parameter for Beta-NLL (0.5 = balanced).
            lambda_position: Weight for position loss.
            lambda_quaternion: Weight for rotation loss.
            lambda_angle: Weight for jaw angle loss.
            lambda_jaw_state: Weight for jaw state BCE.
            lambda_calibration: Weight for position calibration ECE.
            lambda_reconstruction: Weight for auxiliary reconstruction.
            min_sigma: Minimum predicted sigma.
            max_sigma: Maximum predicted sigma.
            min_kappa: Minimum VMF concentration.
            jaw_state_threshold: Threshold for open/closed (50 %).
            n_calibration_bins: Number of ECE bins.
            lambda_geodesic: Weight for complementary geodesic loss.
            lambda_calibration_rotation: Weight for rotation ECE (Fisher sigma).
            lambda_calibration_rotation_vmf: Weight for rotation ECE (vMF native).
            lambda_calibration_angle: Weight for jaw angle ECE.
        """
        super().__init__()
        self.beta = beta
        self.lambda_position = lambda_position
        self.lambda_quaternion = lambda_quaternion
        self.lambda_angle = lambda_angle
        self.lambda_jaw_state = lambda_jaw_state
        self.lambda_calibration = lambda_calibration
        self.lambda_reconstruction = lambda_reconstruction
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_kappa = min_kappa
        self.jaw_state_threshold = jaw_state_threshold
        self.n_calibration_bins = n_calibration_bins
        self.lambda_geodesic = lambda_geodesic
        self.lambda_calibration_rotation = lambda_calibration_rotation
        self.lambda_calibration_rotation_vmf = lambda_calibration_rotation_vmf
        self.lambda_calibration_angle = lambda_calibration_angle

        # Quaternion denormalisation buffers
        self.register_buffer("quat_mean", torch.zeros(2, 4))
        self.register_buffer("quat_std", torch.ones(2, 4))
        self.quat_denorm_enabled = False

    # ------------------------------------------------------------------
    # Quaternion denormalisation (CRITICAL for correct geodesic error)
    # ------------------------------------------------------------------

    def set_quat_norm_stats(
        self, mean_30d: torch.Tensor, std_30d: torch.Tensor
    ) -> None:
        """Set quaternion denormalisation stats from 30D normalisation vectors.

        The kinematic data is z-scored over all 30 features. Quaternion
        channels (indices 3:7 and 11:15) must be denormalised before
        computing geodesic distance or VMF likelihood.  Without this step,
        targets remain in z-scored space and geodesic errors are ~90 deg.

        Args:
            mean_30d: (30,) mean from z-score normalisation.
            std_30d: (30,) std from z-score normalisation.
        """
        self.quat_mean.copy_(torch.stack([mean_30d[3:7], mean_30d[11:15]]))
        self.quat_std.copy_(torch.stack([std_30d[3:7], std_30d[11:15]]))
        self.quat_denorm_enabled = True

    def _denorm_target_quat(self, target_quat: torch.Tensor) -> torch.Tensor:
        """Denormalise z-scored quaternion targets to physical unit quaternions.

        Args:
            target_quat: (B, 2, 4) z-scored quaternions.

        Returns:
            (B, 2, 4) unit quaternions in physical space.
        """
        if not self.quat_denorm_enabled:
            return target_quat
        mean = self.quat_mean.unsqueeze(0).to(target_quat.device)
        std = self.quat_std.unsqueeze(0).to(target_quat.device)
        q = target_quat * std + mean
        return F.normalize(q, dim=-1)

    # ------------------------------------------------------------------
    # Target extraction helpers
    # ------------------------------------------------------------------

    def _extract_targets(
        self, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and denormalise per-component targets.

        Target layout (16D):
            Tool 1: pos[0:3], quat[3:7], angle[7]
            Tool 2: pos[8:11], quat[11:15], angle[15]

        Args:
            target: (B, 16) target poses.

        Returns:
            target_pos: (B, 2, 3).
            target_quat: (B, 2, 4) -- denormalised unit quaternions.
            target_angle: (B, 2, 1).
        """
        target_pos = torch.stack([target[:, 0:3], target[:, 8:11]], dim=1)
        target_quat = torch.stack([target[:, 3:7], target[:, 11:15]], dim=1)
        target_quat = self._denorm_target_quat(target_quat)
        target_angle = torch.stack([target[:, 7:8], target[:, 15:16]], dim=1)
        return target_pos, target_quat, target_angle

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        use_cholesky: bool = False,
        calibration_weight: float = 0.0,
        jaw_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined kinematic foundation loss.

        Args:
            pred: Model predictions containing:
                - ``mu_position``: (B, 2, 3) position means.
                - ``sigma_position``: (B, 2, 3) position sigmas.
                - ``mu_quaternion``: (B, 2, 4) quaternion means.
                - ``kappa_quaternion``: (B, 2, 1) VMF concentrations.
                - ``mu_angle``: (B, 2, 1) jaw angle means.
                - ``sigma_angle``: (B, 2, 1) jaw angle sigmas.
                - ``jaw_state_logits``: (B, 2) jaw state logits (optional).
                - ``reconstruction``, ``reconstruction_target`` (optional).
            target: (B, 16) target poses.
            use_cholesky: Reserved for future Cholesky covariance mode.
            calibration_weight: Current ECE weight (from scheduler).
            jaw_mask: (B,) bool -- False for 6-DoF samples without jaw data.

        Returns:
            total_loss: Scalar.
            loss_dict: Per-component losses for logging.
        """
        loss_dict: dict[str, torch.Tensor] = {}
        device = target.device
        total_loss = torch.tensor(0.0, device=device)

        target_pos, target_quat, target_angle = self._extract_targets(target)

        # ---- Position Beta-NLL ----
        if "mu_position" in pred and "sigma_position" in pred:
            pos_loss = beta_nll_loss(
                mu=pred["mu_position"],
                sigma=pred["sigma_position"],
                target=target_pos,
                beta=self.beta,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
            )
            loss_dict["position"] = pos_loss
            total_loss = total_loss + self.lambda_position * pos_loss

        # ---- Rotation Beta-VMF ----
        if "mu_quaternion" in pred and "kappa_quaternion" in pred:
            quat_loss = beta_vmf_loss(
                mu_quat=pred["mu_quaternion"],
                kappa=pred["kappa_quaternion"],
                target_quat=target_quat,
                beta=self.beta,
                min_kappa=self.min_kappa,
            )
            loss_dict["quaternion"] = quat_loss
            total_loss = total_loss + self.lambda_quaternion * quat_loss

        # ---- Geodesic rotation loss (complementary) ----
        if self.lambda_geodesic > 0 and "mu_quaternion" in pred:
            geo_loss = geodesic_loss(
                pred["mu_quaternion"].reshape(-1, 4),
                target_quat.reshape(-1, 4),
            )
            loss_dict["geodesic"] = geo_loss
            total_loss = total_loss + self.lambda_geodesic * geo_loss

        # ---- Jaw angle Beta-NLL ----
        if "mu_angle" in pred and "sigma_angle" in pred:
            if jaw_mask is not None:
                angle_per = beta_nll_loss(
                    mu=pred["mu_angle"],
                    sigma=pred["sigma_angle"],
                    target=target_angle,
                    beta=self.beta,
                    min_sigma=self.min_sigma,
                    max_sigma=self.max_sigma,
                    reduction="none",
                )
                masked = angle_per * jaw_mask.unsqueeze(-1).unsqueeze(-1).float()
                n_valid = jaw_mask.sum() * 2
                angle_loss = masked.sum() / n_valid.clamp(min=1)
            else:
                angle_loss = beta_nll_loss(
                    mu=pred["mu_angle"],
                    sigma=pred["sigma_angle"],
                    target=target_angle,
                    beta=self.beta,
                    min_sigma=self.min_sigma,
                    max_sigma=self.max_sigma,
                )
            loss_dict["angle"] = angle_loss
            total_loss = total_loss + self.lambda_angle * angle_loss

        # ---- Jaw state BCE ----
        if "jaw_state_logits" in pred:
            target_jaw_pct = torch.stack(
                [target[:, 7:8], target[:, 15:16]], dim=1
            ).squeeze(-1)
            jaw_loss = jaw_state_loss(
                pred["jaw_state_logits"],
                target_jaw_pct,
                threshold=self.jaw_state_threshold,
            )
            loss_dict["jaw_state"] = jaw_loss
            total_loss = total_loss + self.lambda_jaw_state * jaw_loss

        # ---- Position calibration ECE ----
        if calibration_weight > 0 and "sigma_position" in pred:
            pos_errors = torch.abs(pred["mu_position"] - target_pos)
            cal_loss = differentiable_ece(
                pred["sigma_position"], pos_errors, self.n_calibration_bins
            )
            loss_dict["calibration"] = cal_loss
            total_loss = total_loss + calibration_weight * cal_loss

        # ---- Rotation calibration ECE (Fisher sigma = 1 / sqrt(kappa)) ----
        if (
            self.lambda_calibration_rotation > 0
            and "kappa_quaternion" in pred
            and "mu_quaternion" in pred
        ):
            kappa = pred["kappa_quaternion"].squeeze(-1).clamp(min=self.min_kappa)
            rot_sigma = 1.0 / torch.sqrt(kappa)
            geo_errors = geodesic_distance(
                pred["mu_quaternion"].reshape(-1, 4),
                target_quat.reshape(-1, 4),
            ).reshape(kappa.shape)
            cal_rot = differentiable_ece(rot_sigma, geo_errors, self.n_calibration_bins)
            loss_dict["calibration_rotation_fisher"] = cal_rot
            total_loss = total_loss + self.lambda_calibration_rotation * cal_rot

        # ---- Rotation calibration ECE (vMF native) ----
        if (
            self.lambda_calibration_rotation_vmf > 0
            and "kappa_quaternion" in pred
            and "mu_quaternion" in pred
        ):
            kappa = pred["kappa_quaternion"].squeeze(-1).clamp(min=self.min_kappa)
            geo_errors = geodesic_distance(
                pred["mu_quaternion"].reshape(-1, 4),
                target_quat.reshape(-1, 4),
            ).reshape(kappa.shape)
            cal_rot_vmf = differentiable_ece_vmf(
                kappa, geo_errors, self.n_calibration_bins
            )
            loss_dict["calibration_rotation_vmf"] = cal_rot_vmf
            total_loss = total_loss + self.lambda_calibration_rotation_vmf * cal_rot_vmf

        # ---- Jaw angle calibration ECE ----
        if (
            self.lambda_calibration_angle > 0
            and "sigma_angle" in pred
            and "mu_angle" in pred
        ):
            angle_errors = torch.abs(pred["mu_angle"] - target_angle)
            sigma_angle = pred["sigma_angle"].clamp(
                min=self.min_sigma, max=self.max_sigma
            )
            cal_jaw = differentiable_ece(
                sigma_angle, angle_errors, self.n_calibration_bins
            )
            loss_dict["calibration_angle"] = cal_jaw
            total_loss = total_loss + self.lambda_calibration_angle * cal_jaw

        # ---- Auxiliary reconstruction ----
        if "reconstruction" in pred and "reconstruction_target" in pred:
            recon_loss = F.mse_loss(
                pred["reconstruction"], pred["reconstruction_target"]
            )
            loss_dict["reconstruction"] = recon_loss
            total_loss = total_loss + self.lambda_reconstruction * recon_loss

        loss_dict["total"] = total_loss
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        pred: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute non-differentiable evaluation metrics.

        Args:
            pred: Model predictions.
            target: (B, 16) target poses.

        Returns:
            Dictionary of metric name to float value.
        """
        metrics: dict[str, float] = {}
        target_pos, target_quat, target_angle = self._extract_targets(target)

        if "mu_position" in pred:
            pos_mae = torch.abs(pred["mu_position"] - target_pos).mean()
            metrics["position_mae_mm"] = pos_mae.item()
            metrics["tool1_pos_mae"] = (
                torch.abs(pred["mu_position"][:, 0] - target_pos[:, 0]).mean().item()
            )
            metrics["tool2_pos_mae"] = (
                torch.abs(pred["mu_position"][:, 1] - target_pos[:, 1]).mean().item()
            )

        if "mu_quaternion" in pred:
            geo = geodesic_distance(
                pred["mu_quaternion"].reshape(-1, 4),
                target_quat.reshape(-1, 4),
            )
            metrics["rotation_error_deg"] = torch.rad2deg(geo).mean().item()

        if "mu_angle" in pred:
            metrics["angle_mae"] = (
                torch.abs(pred["mu_angle"] - target_angle).mean().item()
            )

        if "jaw_state_logits" in pred:
            target_jaw_pct = torch.stack(
                [target[:, 7:8], target[:, 15:16]], dim=1
            ).squeeze(-1)
            target_state = (target_jaw_pct >= self.jaw_state_threshold).float()
            pred_state = (pred["jaw_state_logits"] > 0).float()
            metrics["jaw_state_accuracy"] = (
                (pred_state == target_state).float().mean().item()
            )

        if "sigma_position" in pred:
            metrics["sigma_position_mean"] = pred["sigma_position"].mean().item()
        if "kappa_quaternion" in pred:
            metrics["kappa_quaternion_mean"] = pred["kappa_quaternion"].mean().item()

        return metrics


# ============================================================================
# BTPNLoss (Full Visual-Temporal Model)
# ============================================================================


class BTPNLoss(nn.Module):
    """Combined loss for the full visual-temporal BTPN.

    Extends the kinematic ``KinematicLoss`` with visual-temporal terms:

    - **Gate entropy** -- prevents per-channel gates from saturating.
    - **Displacement consistency** -- encourages accurate inter-frame motion.
    - **Pivot constraint** -- trocar (entry point) consistency.
    - **Temporal smoothness** -- penalises jerky position deltas.
    - **Residual regularisation** -- keeps visual corrections small.

    Loss formulation::

        L = KinematicLoss(pred, target)
          + lambda_gate * gate_entropy
          + lambda_disp * displacement_loss
          + lambda_reg  * residual_reg
          + lambda_pivot * pivot_consistency   (after warmup)
          + lambda_smooth * temporal_smoothness

    Attributes:
        kinematic_loss: Inner ``KinematicLoss`` for the core terms.
        quat_denorm_enabled: Forwarded from inner loss.
    """

    def __init__(
        self,
        *,
        # Kinematic loss parameters
        beta: float = 0.5,
        lambda_position: float = 1.0,
        lambda_quaternion: float = 3.0,
        lambda_angle: float = 1.0,
        lambda_jaw_state: float = 0.5,
        min_sigma: float = 1e-3,
        max_sigma: float = 10.0,
        min_kappa: float = 1.0,
        # Visual-temporal parameters
        lambda_residual_reg: float = 0.005,
        lambda_displacement: float = 0.5,
        lambda_gate_entropy: float = 0.01,
        lambda_pivot: float = 0.1,
        lambda_smoothness: float = 0.01,
    ):
        """Initialise full BTPN loss.

        Args:
            beta: Beta-NLL parameter.
            lambda_position: Position BetaNLL weight.
            lambda_quaternion: Rotation BetaVMF weight.
            lambda_angle: Jaw angle BetaNLL weight.
            lambda_jaw_state: Jaw state BCE weight.
            min_sigma: Minimum sigma clamp.
            max_sigma: Maximum sigma clamp.
            min_kappa: Minimum VMF concentration.
            lambda_residual_reg: Residual regularisation weight.
            lambda_displacement: Displacement consistency weight.
            lambda_gate_entropy: Gate anti-saturation weight.
            lambda_pivot: Pivot consistency weight (ramped after warmup).
            lambda_smoothness: Temporal smoothness weight.
        """
        super().__init__()
        self.lambda_residual_reg = lambda_residual_reg
        self.lambda_displacement = lambda_displacement
        self.lambda_gate_entropy = lambda_gate_entropy
        self.lambda_pivot = lambda_pivot
        self.lambda_smoothness = lambda_smoothness

        # Core kinematic loss (shared buffers for quat denorm)
        self.kinematic_loss = KinematicLoss(
            beta=beta,
            lambda_position=lambda_position,
            lambda_quaternion=lambda_quaternion,
            lambda_angle=lambda_angle,
            lambda_jaw_state=lambda_jaw_state,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            min_kappa=min_kappa,
        )

        # Expose beta for external use
        self.beta = beta
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_kappa = min_kappa

        # Forward quat denorm buffers from inner loss
        self.register_buffer("quat_mean", torch.zeros(2, 4))
        self.register_buffer("quat_std", torch.ones(2, 4))
        self.quat_denorm_enabled = False

    # ------------------------------------------------------------------
    # Quaternion denormalisation (delegates to inner loss)
    # ------------------------------------------------------------------

    def set_quat_norm_stats(
        self, mean_30d: torch.Tensor, std_30d: torch.Tensor
    ) -> None:
        """Set quaternion denormalisation stats for both inner and outer loss.

        Args:
            mean_30d: (30,) mean from z-score normalisation.
            std_30d: (30,) std from z-score normalisation.
        """
        self.kinematic_loss.set_quat_norm_stats(mean_30d, std_30d)
        self.quat_mean.copy_(self.kinematic_loss.quat_mean)
        self.quat_std.copy_(self.kinematic_loss.quat_std)
        self.quat_denorm_enabled = True

    def _denorm_target_quat(self, target_quat: torch.Tensor) -> torch.Tensor:
        """Delegate to kinematic loss denormalisation."""
        return self.kinematic_loss._denorm_target_quat(target_quat)

    # ------------------------------------------------------------------
    # Positive-only mask from detection confidence
    # ------------------------------------------------------------------

    def compute_positive_mask(
        self,
        detection_conf: list[torch.Tensor] | None,
        conf_threshold: float = 0.3,
    ) -> torch.Tensor | None:
        """Compute mask from detection confidence (finest scale, last 10 frames).

        Args:
            detection_conf: Per-scale detection confidence tensors (B, T, 2).
            conf_threshold: Minimum average confidence threshold.

        Returns:
            Boolean mask (B,) or None if no confidence data.
        """
        if detection_conf is None or len(detection_conf) == 0:
            return None
        finest = detection_conf[0]
        last_n = min(10, finest.shape[1])
        avg_conf = finest[:, -last_n:, :].mean(dim=(1, 2))
        return avg_conf > conf_threshold

    # ------------------------------------------------------------------
    # Residual regularisation
    # ------------------------------------------------------------------

    def residual_regularization(
        self, pred: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """L2 regularisation on delta (residual correction) channels.

        Penalises large position, quaternion, and angle deltas to keep
        visual corrections small and stable.

        Args:
            pred: Model output dict with optional ``delta_position``,
                ``delta_quaternion``, ``delta_angle`` keys.

        Returns:
            Regularisation loss (scalar).
        """
        device = next(iter(pred.values())).device
        reg = torch.tensor(0.0, device=device)
        n_terms = 0

        if "delta_position" in pred:
            reg = reg + pred["delta_position"].norm(dim=-1).mean()
            n_terms += 1
        if "delta_quaternion" in pred:
            reg = reg + pred["delta_quaternion"].norm(dim=-1).mean()
            n_terms += 1
        if "delta_angle" in pred:
            reg = reg + pred["delta_angle"].abs().mean()
            n_terms += 1

        return reg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        detection_conf: list[torch.Tensor] | None = None,
        conf_threshold: float = 0.3,
        lambda_residual_reg: float | None = None,
        pivot_loss: torch.Tensor | None = None,
        pivot_weight: float | None = None,
        prev_delta_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined visual-temporal BTPN loss.

        Args:
            pred: Model output dict with keys:
                - ``mu_position``: (B, 2, 3) predicted positions.
                - ``sigma_position``: (B, 2, 3) position uncertainty.
                - ``mu_quaternion``: (B, 2, 4) predicted quaternions.
                - ``kappa_quaternion``: (B, 2, 1) VMF concentration.
                - ``mu_angle``: (B, 2, 1) predicted jaw angle.
                - ``sigma_angle``: (B, 2, 1) angle uncertainty.
                - ``jaw_state_logits``: (B, 2) jaw state logits (optional).
                - ``delta_position``: (B, 2, 3) residual corrections (optional).
                - ``delta_quaternion``: (B, 2, 4) (optional).
                - ``delta_angle``: (B, 2, 1) (optional).
                - ``gates``: dict of per-channel gate tensors (optional).
                - ``visual_displacement``: (B, 2, 3) predicted displacement.
                - ``target_displacement``: (B, 2, 3) actual displacement.
            target: (B, 16) target poses [Tool1(8) + Tool2(8)].
            detection_conf: Per-scale detection confidence for masking.
            conf_threshold: Confidence threshold for positive mask.
            lambda_residual_reg: Override for residual reg weight.
            pivot_loss: Pre-computed pivot consistency loss (scalar).
            pivot_weight: Override pivot loss weight.
            prev_delta_position: Previous batch last deltas (2, 3).

        Returns:
            total_loss: Scalar.
            loss_dict: Per-component losses.
        """
        loss_dict: dict[str, torch.Tensor] = {}
        device = target.device
        total_loss = torch.tensor(0.0, device=device)

        # Compute positive-only mask from detection confidence
        mask = self.compute_positive_mask(detection_conf, conf_threshold)
        if mask is not None and mask.sum() == 0:
            loss_dict["total"] = total_loss
            loss_dict["n_positive"] = torch.tensor(0.0, device=device)
            return total_loss, loss_dict

        # ---- Extract and denormalise targets ----
        target_pos = torch.stack([target[:, 0:3], target[:, 8:11]], dim=1)
        target_quat = torch.stack([target[:, 3:7], target[:, 11:15]], dim=1)
        target_quat = self._denorm_target_quat(target_quat)
        target_angle = torch.stack([target[:, 7:8], target[:, 15:16]], dim=1)

        # ---- Position Beta-NLL (masked) ----
        if "mu_position" in pred and "sigma_position" in pred:
            pos_loss = beta_nll_loss(
                mu=pred["mu_position"],
                sigma=pred["sigma_position"],
                target=target_pos,
                beta=self.beta,
                reduction="none",
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
            )
            pos_per = pos_loss.mean(dim=(1, 2))
            if mask is not None:
                mask_f = mask.float()
                pos_masked = (pos_per * mask_f).sum() / mask_f.sum().clamp(min=1)
            else:
                pos_masked = pos_per.mean()
            loss_dict["position"] = pos_masked
            total_loss = total_loss + self.kinematic_loss.lambda_position * pos_masked

        # ---- Rotation Beta-VMF (masked) ----
        if "mu_quaternion" in pred and "kappa_quaternion" in pred:
            quat_loss = beta_vmf_loss(
                mu_quat=pred["mu_quaternion"],
                kappa=pred["kappa_quaternion"],
                target_quat=target_quat,
                beta=self.beta,
                reduction="none",
                min_kappa=self.min_kappa,
            )
            quat_per = quat_loss.mean(dim=-1)
            if mask is not None:
                mask_f = mask.float()
                quat_masked = (quat_per * mask_f).sum() / mask_f.sum().clamp(min=1)
            else:
                quat_masked = quat_per.mean()
            loss_dict["quaternion"] = quat_masked
            total_loss = total_loss + self.kinematic_loss.lambda_quaternion * quat_masked

        # ---- Jaw angle Beta-NLL (masked) ----
        if "mu_angle" in pred and "sigma_angle" in pred:
            angle_loss = beta_nll_loss(
                mu=pred["mu_angle"],
                sigma=pred["sigma_angle"],
                target=target_angle,
                beta=self.beta,
                reduction="none",
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
            )
            angle_per = angle_loss.mean(dim=(1, 2))
            if mask is not None:
                mask_f = mask.float()
                angle_masked = (angle_per * mask_f).sum() / mask_f.sum().clamp(min=1)
            else:
                angle_masked = angle_per.mean()
            loss_dict["angle"] = angle_masked
            total_loss = total_loss + self.kinematic_loss.lambda_angle * angle_masked

        # ---- Jaw state BCE (masked) ----
        if "jaw_state_logits" in pred:
            target_jaw_pct = torch.stack(
                [target[:, 7:8], target[:, 15:16]], dim=1
            ).squeeze(-1)
            if mask is not None:
                jaw_per = jaw_state_loss(
                    pred["jaw_state_logits"],
                    target_jaw_pct,
                    threshold=50.0,
                    reduction="none",
                )
                jaw_sample = jaw_per.mean(dim=-1)
                jaw_masked = (jaw_sample * mask.float()).sum() / mask.float().sum().clamp(min=1)
            else:
                jaw_masked = jaw_state_loss(
                    pred["jaw_state_logits"],
                    target_jaw_pct,
                    threshold=50.0,
                )
            loss_dict["jaw_state"] = jaw_masked
            total_loss = total_loss + self.kinematic_loss.lambda_jaw_state * jaw_masked

        # ---- Gate entropy regularisation ----
        if "gates" in pred and pred["gates"]:
            g_entropy = _gate_entropy_loss(pred["gates"])
            loss_dict["gate_entropy"] = g_entropy
            total_loss = total_loss + self.lambda_gate_entropy * g_entropy

        # ---- Displacement consistency ----
        if "visual_displacement" in pred and "target_displacement" in pred:
            disp_loss = _displacement_loss(
                pred["visual_displacement"],
                pred["target_displacement"],
                mask=mask,
            )
            loss_dict["displacement"] = disp_loss
            total_loss = total_loss + self.lambda_displacement * disp_loss

        # ---- Residual regularisation ----
        has_deltas = any(
            k in pred for k in ["delta_position", "delta_quaternion", "delta_angle"]
        )
        if has_deltas:
            reg_w = (
                lambda_residual_reg
                if lambda_residual_reg is not None
                else self.lambda_residual_reg
            )
            reg_loss = self.residual_regularization(pred)
            loss_dict["residual_reg"] = reg_loss
            total_loss = total_loss + reg_w * reg_loss

        # ---- Pivot consistency (externally computed) ----
        if pivot_loss is not None:
            pw = pivot_weight if pivot_weight is not None else self.lambda_pivot
            loss_dict["pivot"] = pivot_loss
            total_loss = total_loss + pw * pivot_loss

        # ---- Temporal smoothness ----
        if "delta_position" in pred and self.lambda_smoothness > 0:
            smooth_loss = _temporal_smoothness_loss(
                pred["delta_position"],
                prev_delta_position,
            )
            loss_dict["smoothness"] = smooth_loss
            total_loss = total_loss + self.lambda_smoothness * smooth_loss

        # ---- Logging info ----
        if mask is not None:
            loss_dict["n_positive"] = mask.float().sum()
            loss_dict["positive_rate"] = mask.float().mean()
        else:
            loss_dict["n_positive"] = torch.tensor(float(target.shape[0]), device=device)
            loss_dict["positive_rate"] = torch.tensor(1.0, device=device)

        loss_dict["total"] = total_loss
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        pred: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute evaluation metrics for the full BTPN.

        Extends ``KinematicLoss.compute_metrics`` with per-channel gate values,
        displacement error, and residual-norm diagnostics.

        Args:
            pred: Model output dict.
            target: (B, 16) target poses.

        Returns:
            Dictionary of metric name to float value.
        """
        metrics: dict[str, float] = {}

        target_pos = torch.stack([target[:, 0:3], target[:, 8:11]], dim=1)
        target_quat = torch.stack([target[:, 3:7], target[:, 11:15]], dim=1)
        target_quat = self._denorm_target_quat(target_quat)
        target_angle = torch.stack([target[:, 7:8], target[:, 15:16]], dim=1)

        # Position MAE
        if "mu_position" in pred:
            pos_mae = torch.abs(pred["mu_position"] - target_pos).mean()
            metrics["position_mae_mm"] = pos_mae.item()
            metrics["tool1_pos_mae"] = (
                torch.abs(pred["mu_position"][:, 0] - target_pos[:, 0]).mean().item()
            )
            metrics["tool2_pos_mae"] = (
                torch.abs(pred["mu_position"][:, 1] - target_pos[:, 1]).mean().item()
            )

        # Rotation error
        if "mu_quaternion" in pred:
            geo = geodesic_distance(
                pred["mu_quaternion"].reshape(-1, 4),
                target_quat.reshape(-1, 4),
            )
            metrics["rotation_error_deg"] = torch.rad2deg(geo).mean().item()

        # Jaw angle RMSE
        if "mu_angle" in pred:
            angle_mse = ((pred["mu_angle"] - target_angle) ** 2).mean()
            metrics["jaw_angle_rmse"] = angle_mse.sqrt().item()

        # Jaw state accuracy
        if "jaw_state_logits" in pred:
            target_jaw_pct = torch.stack(
                [target[:, 7:8], target[:, 15:16]], dim=1
            ).squeeze(-1)
            target_state = (target_jaw_pct >= 50.0).float()
            pred_state = (pred["jaw_state_logits"] > 0).float()
            metrics["jaw_state_accuracy"] = (
                (pred_state == target_state).float().mean().item()
            )

        # Residual norms
        if "delta_position" in pred:
            metrics["delta_position_norm"] = (
                pred["delta_position"].norm(dim=-1).mean().item()
            )
        if "delta_quaternion" in pred:
            metrics["delta_quaternion_norm"] = (
                pred["delta_quaternion"].norm(dim=-1).mean().item()
            )
        if "delta_angle" in pred:
            metrics["delta_angle_norm"] = pred["delta_angle"].abs().mean().item()

        # Uncertainty statistics
        if "sigma_position" in pred:
            metrics["sigma_position_mean"] = pred["sigma_position"].mean().item()
        if "kappa_quaternion" in pred:
            metrics["kappa_quaternion_mean"] = pred["kappa_quaternion"].mean().item()

        # Per-channel gate values
        if "gates" in pred and pred["gates"]:
            for name, gate_val in pred["gates"].items():
                metrics[f"gate_{name}_mean"] = gate_val.mean().item()
                metrics[f"gate_{name}_std"] = gate_val.std().item()

        # Displacement error
        if "visual_displacement" in pred and "target_displacement" in pred:
            disp_err = torch.abs(
                pred["visual_displacement"] - pred["target_displacement"]
            ).mean()
            metrics["displacement_mae"] = disp_err.item()

        # Pivot residual
        if "pivot_residual" in pred:
            metrics["pivot_residual_mm"] = pred["pivot_residual"].mean().item()

        return metrics
