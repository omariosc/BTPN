"""Data loading and dataset classes for BTPN training.

This module provides everything needed to load surgical peg-transfer kinematic
data and pre-computed visual features for training both the kinematic
foundation model and the full visual-temporal BTPN.

Data Loaders:
    - ``load_trial``       -- Load a single trial (7-DoF ``label.json`` or
      6-DoF ``Test N/`` directory).
    - ``discover_trials``  -- Discover all trial directories for a dataset.

Normalisation:
    - ``NormalizationStats`` -- Dataclass for z-score mean/std with
      ``save`` / ``load`` persistence.

Datasets:
    - ``KinematicDataset``       -- Multi-scale causal windowing for the
      kinematic foundation model.  Includes augmentation (noise, speed
      perturbation, tool swap).
    - ``TrialVisualCache``       -- Loads consolidated ``_visual_cache.npz``
      per trial (seg + depth + pose features).
    - ``VisualTemporalDataset``  -- Dense sampling with visual features for
      full BTPN training.

Factory:
    - ``create_dataloaders``     -- Convenience function to build train/val
      ``DataLoader`` pairs from a config dict and paths.

Supported dataset layouts
-------------------------

7-DoF (30D):  ``7DOF2024/Attempt*/Trial*/label.json``  and
              ``7DOF2024/Trial*/label.json`` (trials 33-60).
              Feature vector: ``[Tool1(8), Tool2(8), Camera(7), World(7)]``
              where Tool = pos(3) + quat(4) + angle(1).

BAPES (30D):  Same format under ``BAPES2024/{Industry,MIS Course}/Trial*/``.

6-DoF (28D padded to 30D):  ``6DOF2023/Test N/`` directories containing
              per-frame ``.txt`` files.  Padded and resampled from 26 fps to
              13 fps.  Feature vector: ``[Ref(7), Fenestrated(7), Curved(7),
              Camera(7)]`` mapped to 7-DoF layout.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Constants
# ============================================================================

FRAME_RATES: dict[str, float] = {
    "6DOF2023": 26.0,
    "7DOF2024": 13.0,
    "BAPES2024": 13.0,
}

TARGET_FPS: float = 13.0

#: Default multi-scale window sizes (frames at 13 fps):
#: 10 (~0.77 s micro-gesture), 50 (~3.85 s action), 100 (~7.69 s cycle).
DEFAULT_SCALES: list[int] = [10, 50, 100]

# Visual feature dimensions
KINEMATIC_DIM: int = 30
SEG_NECK_DIM: int = 256
SEG_BACKBONE_DIM: int = 512
DEPTH_DIM: int = 384   # DINOv2-S CLS token
POSE_BACKBONE_DIM: int = 512
POSE_NUM_KEYPOINTS: int = 8
POSE_GEOMETRIC_DIM: int = 6  # shaft_width(2) + midline_angle(2) + jaw_opening(2)

# Excluded trials
EXCLUDED_BAPES_TRIALS: set[str] = {"Trial18"}

# Fixed validation trials for 7DOF2024 (trials 40-60 inclusive)
VAL_TRIAL_NUMBERS_7DOF2024: set[int] = set(range(40, 61))

# Consolidated visual cache filename
VISUAL_CACHE_FILENAME: str = "_visual_cache.npz"


# ============================================================================
# JSON Utilities
# ============================================================================


def _fix_json(content: str) -> str:
    """Remove trailing commas in JSON that would cause parse errors.

    Args:
        content: Raw JSON string.

    Returns:
        Cleaned JSON string.
    """
    return re.sub(r",\s*([}\]])", r"\1", content)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, tolerating trailing commas.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON is fundamentally malformed.
    """
    with open(path, "r") as f:
        content = f.read()
    return json.loads(_fix_json(content))


# ============================================================================
# NormalizationStats
# ============================================================================


@dataclass
class NormalizationStats:
    """Per-feature z-score normalisation statistics.

    Attributes:
        mean: Feature means (D,).
        std: Feature stds (D,), clamped to a minimum of 1e-8.
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        self.std = np.clip(self.std, a_min=1e-8, a_max=None)

    @classmethod
    def from_trials(cls, trial_data_list: list[np.ndarray]) -> NormalizationStats:
        """Compute statistics from a list of trial data arrays.

        Args:
            trial_data_list: List of (N_i, D) arrays.

        Returns:
            ``NormalizationStats`` instance.
        """
        all_data = np.concatenate(trial_data_list, axis=0)
        return cls(mean=all_data.mean(axis=0), std=all_data.std(axis=0))

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation.

        Args:
            data: Array of shape (..., D).

        Returns:
            Normalised array.
        """
        return (data - self.mean) / self.std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse z-score normalisation.

        Args:
            data: Normalised array of shape (..., D).

        Returns:
            Original-scale array.
        """
        return data * self.std + self.mean

    def save(self, path: Path) -> None:
        """Persist statistics to an ``.npz`` file.

        Args:
            path: Destination file path.
        """
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> NormalizationStats:
        """Load statistics from an ``.npz`` file.

        Args:
            path: Source file path.

        Returns:
            ``NormalizationStats`` instance.
        """
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"])


# ============================================================================
# Trial Loading — 7-DoF / BAPES
# ============================================================================


def _load_7dof_trial(trial_dir: Path) -> dict[str, Any] | None:
    """Load a single 7-DoF or BAPES trial from ``label.json``.

    The annotation file contains per-frame entries with Tool 1, Tool 2,
    Camera, and World sensor readings.  Frames containing NaN values in any
    tool position or rotation channel are dropped.

    Args:
        trial_dir: Path to trial directory containing ``label.json``.

    Returns:
        Dict with ``"data"`` (N, 30) float32 array and ``"metadata"`` dict,
        or ``None`` if the trial is invalid / empty.
    """
    label_path = trial_dir / "label.json"
    if not label_path.exists():
        return None

    try:
        data = _load_json(label_path)
    except Exception:
        return None

    annotations = data.get("annotations", [])
    if not annotations:
        return None

    frames: list[list[float]] = []
    for ann in annotations:
        feat: list[float] = []
        for obj_name in ["Tool 1", "Tool 2"]:
            obj = ann.get(obj_name, {})
            feat.extend(obj.get("Position", [0, 0, 0]))
            feat.extend(obj.get("Rotation", [1, 0, 0, 0]))
            feat.append(obj.get("Angle", 0))
        for obj_name in ["Camera", "World"]:
            obj = ann.get(obj_name, {})
            feat.extend(obj.get("Position", [0, 0, 0]))
            feat.extend(obj.get("Rotation", [1, 0, 0, 0]))

        if not any(np.isnan(feat)):
            frames.append(feat)

    if not frames:
        return None

    arr = np.array(frames, dtype=np.float32)
    dataset = "7DOF2024" if "7DOF" in str(trial_dir) else "BAPES2024"

    info = data.get("info", {})
    total_proc = info.get("Total_Procedures", 0)
    if total_proc < 20:
        skill = "novice"
    elif total_proc < 100:
        skill = "intermediate"
    else:
        skill = "expert"

    metadata = {
        "trial_name": trial_dir.name,
        "dataset": dataset,
        "n_frames": len(frames),
        "frame_rate": FRAME_RATES[dataset],
        "participant_number": info.get("Participant_Number", -1),
        "total_procedures": total_proc,
        "skill_category": skill,
    }
    return {"data": arr, "metadata": metadata}


# ============================================================================
# Trial Loading — 6-DoF
# ============================================================================


def _resample(
    data: np.ndarray, src_fps: float, tgt_fps: float
) -> np.ndarray:
    """Resample kinematic data from source to target frame rate.

    Uses linear interpolation per feature channel.

    Args:
        data: (N, D) array at source frame rate.
        src_fps: Source frame rate.
        tgt_fps: Target frame rate.

    Returns:
        Resampled array at target frame rate.
    """
    if abs(src_fps - tgt_fps) < 0.1:
        return data

    n_frames, n_feat = data.shape
    duration = n_frames / src_fps
    n_target = max(2, int(duration * tgt_fps))

    src_t = np.linspace(0, duration, n_frames)
    tgt_t = np.linspace(0, duration, n_target)

    out = np.zeros((n_target, n_feat), dtype=data.dtype)
    for i in range(n_feat):
        out[:, i] = np.interp(tgt_t, src_t, data[:, i])
    return out


def _pad_6dof_to_30dim(data: np.ndarray) -> np.ndarray:
    """Pad 28D 6-DoF data to 30D 7-DoF format.

    Mapping:
        - Tool 1 <-- Fenestrated (pos + quat) + jaw_angle = 0
        - Tool 2 <-- Curved (pos + quat) + jaw_angle = 0
        - Camera  <-- Camera (pos + quat)
        - World   <-- Reference (pos + quat)

    Args:
        data: (N, 28) array.

    Returns:
        (N, 30) array.
    """
    n = data.shape[0]
    ref = data[:, 0:7]
    tool1 = data[:, 7:14]
    tool2 = data[:, 14:21]
    cam = data[:, 21:28]
    zeros = np.zeros((n, 1), dtype=data.dtype)

    return np.concatenate(
        [
            tool1, zeros,  # Tool1: pos(3) + quat(4) + angle(1) = 0
            tool2, zeros,  # Tool2: pos(3) + quat(4) + angle(1) = 0
            cam,           # Camera: pos(3) + quat(4)
            ref,           # World (was Ref): pos(3) + quat(4)
        ],
        axis=1,
    )


def _load_6dof_frame(txt_path: Path) -> dict[str, Any] | None:
    """Load a single frame from a 6-DoF ``.txt`` file.

    Each file contains a timestamp line followed by one line per sensor:
    ``ObjectName<TAB>X<TAB>Y<TAB>Z<TAB>qw<TAB>qx<TAB>qy<TAB>qz``

    Args:
        txt_path: Path to the frame file.

    Returns:
        Dict with ``"time_ms"`` and per-object ``"position"``/``"rotation"``
        arrays, or ``None`` on failure.
    """
    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()

        result: dict[str, Any] = {}
        first_line = lines[0].strip().split("\t")
        result["time_ms"] = float(first_line[1]) if len(first_line) > 1 else 0.0

        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split("\t")
                name = parts[0]
                values = [float(v) for v in parts[1:] if v]
                if len(values) == 7:
                    result[name] = {
                        "position": np.array(values[:3]),
                        "rotation": np.array(values[3:]),
                    }
        return result
    except Exception:
        return None


def _load_6dof_trial_raw(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all frames from a 6-DoF trial directory.

    Args:
        trial_dir: Path to ``Test N/`` directory.

    Returns:
        features: (N, 28) array of kinematic features.
        times: (N,) array of timestamps in seconds.
    """
    txt_files = sorted(
        trial_dir.glob("*.txt"),
        key=lambda x: int(x.stem) if x.stem.isdigit() else 0,
    )

    features_list: list[list[float]] = []
    times_list: list[float] = []

    for txt_path in txt_files:
        data = _load_6dof_frame(txt_path)
        if data is None:
            continue

        feature: list[float] = []
        for obj_name in ["Reference", "Fenestrated", "Curved", "Camera"]:
            if obj_name in data:
                feature.extend(data[obj_name]["position"])
                feature.extend(data[obj_name]["rotation"])
            else:
                feature.extend([0.0] * 7)

        if len(feature) == 28:
            features_list.append(feature)
            times_list.append(data["time_ms"] / 1000.0)

    if not features_list:
        return np.array([]), np.array([])

    return np.array(features_list), np.array(times_list)


def _load_6dof_trial(trial_dir: Path) -> dict[str, Any] | None:
    """Load a 6-DoF trial, pad to 30D, and resample to 13 fps.

    Prefers ``transformed_label.json`` (pre-processed) over per-frame
    ``.txt`` loading.  NaN values are linearly interpolated.

    Args:
        trial_dir: Path to ``Test N/`` directory.

    Returns:
        Dict with ``"data"`` (N, 30) and ``"metadata"``, or ``None``.
    """
    # Try pre-processed JSON first
    label_path = trial_dir / "transformed_label.json"
    if not label_path.exists():
        label_path = trial_dir / "label.json"

    if label_path.exists():
        try:
            data = _load_json(label_path)
            annotations = data.get("annotations", [])
            if annotations:
                frames: list[list[float]] = []
                for ann in annotations:
                    feat: list[float] = []
                    for obj_primary, obj_fallback in [
                        ("Reference", "Ref"),
                        ("Tool 1", "Tool1"),
                        ("Tool 2", "Tool2"),
                        ("Camera", "Cam"),
                    ]:
                        obj = ann.get(obj_primary) or ann.get(obj_fallback, {})
                        feat.extend(obj.get("Position", [0, 0, 0]))
                        feat.extend(obj.get("Rotation", [1, 0, 0, 0]))
                    frames.append(feat)

                arr_28 = np.array(frames, dtype=np.float32)

                # Interpolate NaN values
                if np.isnan(arr_28).any():
                    for i in range(arr_28.shape[1]):
                        mask = np.isnan(arr_28[:, i])
                        if mask.any() and (~mask).sum() > 0:
                            arr_28[mask, i] = np.interp(
                                np.where(mask)[0],
                                np.where(~mask)[0],
                                arr_28[~mask, i],
                            )

                arr_30 = _pad_6dof_to_30dim(arr_28)
                arr_30 = _resample(arr_30, 26.0, TARGET_FPS)

                metadata = {
                    "trial_name": trial_dir.name,
                    "dataset": "6DOF2023",
                    "n_frames": arr_30.shape[0],
                    "frame_rate": TARGET_FPS,
                    "participant_number": -1,
                    "total_procedures": 0,
                    "skill_category": "unknown",
                }
                return {"data": arr_30, "metadata": metadata}
        except Exception:
            pass

    # Fallback: load per-frame .txt files
    features_28, _times = _load_6dof_trial_raw(trial_dir)
    if len(features_28) == 0:
        return None

    arr_28 = features_28.astype(np.float32)

    # Interpolate NaN
    if np.isnan(arr_28).any():
        for i in range(arr_28.shape[1]):
            nan_mask = np.isnan(arr_28[:, i])
            if nan_mask.any() and (~nan_mask).sum() > 0:
                arr_28[nan_mask, i] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(~nan_mask)[0],
                    arr_28[~nan_mask, i],
                )

    arr_30 = _pad_6dof_to_30dim(arr_28)
    arr_30 = _resample(arr_30, 26.0, TARGET_FPS)

    metadata = {
        "trial_name": trial_dir.name,
        "dataset": "6DOF2023",
        "n_frames": arr_30.shape[0],
        "frame_rate": TARGET_FPS,
        "participant_number": -1,
        "total_procedures": 0,
        "skill_category": "unknown",
    }
    return {"data": arr_30, "metadata": metadata}


# ============================================================================
# Unified Trial Loader
# ============================================================================


def load_trial(path: Path) -> dict[str, Any] | None:
    """Load a single trial from disk, auto-detecting the format.

    7-DoF / BAPES trials are detected by the presence of ``label.json``.
    6-DoF trials are detected by the ``Test N`` naming convention or, as a
    fallback, by the presence of numbered per-frame ``.txt`` files (so the
    bundled ``data/sample_c`` sample, which is not named ``Test N``, also
    loads).

    Args:
        path: Path to trial directory.

    Returns:
        Dict with ``"data"`` (N, 30) float32 array and ``"metadata"`` dict,
        or ``None`` if the trial could not be loaded.
    """
    if (path / "label.json").exists():
        return _load_7dof_trial(path)
    if path.name.startswith("Test ") or any(path.glob("*.txt")):
        return _load_6dof_trial(path)
    return None


# ============================================================================
# Trial Discovery
# ============================================================================


def _extract_trial_number(trial_dir: Path) -> int:
    """Extract numeric trial ID from a trial directory name.

    Handles ``"Trial1"``, ``"Trial 14"``, ``"Trial1 - Day 1 with Latency"``,
    ``"Trial 14 (Part 1)"``, etc.

    Args:
        trial_dir: Path to trial directory.

    Returns:
        Integer trial number, or -1 if not parseable.
    """
    match = re.search(r"Trial\s*(\d+)", trial_dir.name)
    return int(match.group(1)) if match else -1


def discover_trials(
    data_root: Path,
    dataset_name: str,
) -> list[Path]:
    """Discover all trial directories for a dataset.

    Supports the three dataset layouts:

    - **7DOF2024**: ``Attempt*/Trial*/`` (trials 1-32) and
      top-level ``Trial*/`` (trials 33-60).
    - **BAPES2024**: ``{Industry,MIS Course}/Trial*/``.
      Excludes trials in ``EXCLUDED_BAPES_TRIALS``.
    - **6DOF2023**: ``Test N/`` directories.

    Args:
        data_root: Root data directory (e.g. ``D:/Data/AI-ELT``).
        dataset_name: One of ``"7DOF2024"``, ``"BAPES2024"``, ``"6DOF2023"``.

    Returns:
        Sorted list of trial directory paths.

    Raises:
        ValueError: If ``dataset_name`` is not recognised.
    """
    if dataset_name == "7DOF2024":
        return _discover_7dof(data_root / "7DOF2024")
    if dataset_name == "BAPES2024":
        return _discover_bapes(data_root / "BAPES2024")
    if dataset_name == "6DOF2023":
        return _discover_6dof(data_root / "6DOF2023")
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _discover_7dof(data_dir: Path) -> list[Path]:
    """Find all 7DOF2024 trial directories.

    Trials 1-32 live under ``Attempt*/Trial*/`` and trials 33-60 live
    directly under the dataset root.  Deduplication prevents double-counting.

    Args:
        data_dir: Root ``7DOF2024/`` directory.

    Returns:
        Sorted list of unique trial directory paths.
    """
    trial_dirs: list[Path] = []

    # Attempt subdirectories
    for attempt_dir in sorted(data_dir.glob("Attempt*")):
        for td in sorted(attempt_dir.glob("Trial*")):
            if (td / "label.json").exists():
                trial_dirs.append(td)

    # Standalone trials
    for td in sorted(data_dir.glob("Trial*")):
        if (td / "label.json").exists():
            trial_dirs.append(td)

    # Deduplicate by trial name prefix
    seen: set[str] = set()
    unique: list[Path] = []
    for td in trial_dirs:
        name = td.name.split(" - ")[0].split(" (")[0]
        if name not in seen:
            seen.add(name)
            unique.append(td)

    return unique


def _discover_bapes(data_dir: Path) -> list[Path]:
    """Find all BAPES2024 trial directories.

    Excludes ``EXCLUDED_BAPES_TRIALS`` (e.g. Trial 18 with all-NaN data).

    Args:
        data_dir: Root ``BAPES2024/`` directory.

    Returns:
        Sorted list of trial directory paths.
    """
    trial_dirs: list[Path] = []
    for subdir_name in ["Industry", "MIS Course"]:
        subdir_path = data_dir / subdir_name
        if subdir_path.exists():
            trial_dirs.extend(
                d
                for d in sorted(subdir_path.glob("Trial*"))
                if d.is_dir()
                and (d / "label.json").exists()
                and d.name not in EXCLUDED_BAPES_TRIALS
            )
    return trial_dirs


def _discover_6dof(data_dir: Path) -> list[Path]:
    """Find all 6DOF2023 trial directories.

    Args:
        data_dir: Root ``6DOF2023/`` directory.

    Returns:
        Sorted list of trial directory paths.
    """
    return [
        d
        for d in sorted(data_dir.glob("Test *"))
        if d.is_dir()
        and not d.name.endswith((" png", " txt"))
        and ((d / "transformed_label.json").exists() or (d / "label.json").exists())
    ]


# ============================================================================
# Trial-Level Split
# ============================================================================


def split_7dof_fixed(
    trial_dirs: list[Path],
    n_val: int | None = None,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split 7DOF2024 trials, holding out trials 40-60 for validation.

    By default every available trial numbered 40-60 is held out; these are
    the trials the Dataset-A metrics are reported on.

    Args:
        trial_dirs: All discovered 7DOF2024 trial directories.
        n_val: Number of validation trials to sample from the pool. ``None``
            (default) holds out the whole pool.
        seed: Random seed, used only when ``n_val`` subsamples the pool.

    Returns:
        ``(train_dirs, val_dirs)`` tuple.
    """
    import random

    pool_dirs: list[Path] = []
    non_pool_dirs: list[Path] = []
    for td in trial_dirs:
        num = _extract_trial_number(td)
        if num in VAL_TRIAL_NUMBERS_7DOF2024:
            pool_dirs.append(td)
        else:
            non_pool_dirs.append(td)

    if n_val is None:
        n_val = len(pool_dirs)
    n_val = min(n_val, len(pool_dirs))

    rng = random.Random(seed)
    shuffled_pool = list(pool_dirs)
    rng.shuffle(shuffled_pool)

    val_dirs = shuffled_pool[:n_val]
    train_dirs = non_pool_dirs + shuffled_pool[n_val:]
    return train_dirs, val_dirs


# ============================================================================
# Collate Functions
# ============================================================================


def collate_multiscale(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate for ``KinematicDataset`` multi-scale batches.

    Stacks per-scale kinematic windows, targets, and optional trocar data.

    Args:
        batch: List of sample dicts from ``KinematicDataset``.

    Returns:
        Batched dictionary with:
            - ``"scales"``: list of (B, T_i, 30) tensors.
            - ``"target"``: (B, 16) tensor.
            - ``"trial_idx"`` / ``"frame_idx"``: (B,) tensors.
            - ``"trocar_pos"`` / ``"shaft_axis"`` / ``"trocar_mask"``
              (optional).
    """
    n_scales = len(batch[0]["scales"])

    scales = [
        torch.stack([sample["scales"][i] for sample in batch])
        for i in range(n_scales)
    ]
    targets = torch.stack([sample["target"] for sample in batch])
    trial_indices = torch.tensor([sample["trial_idx"] for sample in batch])
    frame_indices = torch.tensor([sample["frame_idx"] for sample in batch])

    result: dict[str, Any] = {
        "scales": scales,
        "target": targets,
        "trial_idx": trial_indices,
        "frame_idx": frame_indices,
    }

    # Trocar information (if any sample has it)
    has_trocar_list = [sample.get("has_trocar", False) for sample in batch]
    if any(has_trocar_list):
        B = len(batch)
        trocar_pos = torch.zeros(B, 2, 3, dtype=torch.float32)
        shaft_axis = torch.zeros(B, 2, 3, dtype=torch.float32)
        trocar_mask = torch.zeros(B, dtype=torch.bool)
        for i, sample in enumerate(batch):
            if sample.get("has_trocar", False):
                trocar_pos[i] = sample["trocar_pos"]
                shaft_axis[i] = sample["shaft_axis"]
                trocar_mask[i] = True
        result["trocar_pos"] = trocar_pos
        result["shaft_axis"] = shaft_axis
        result["trocar_mask"] = trocar_mask

    return result


def collate_visual_temporal(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Custom collate for ``VisualTemporalDataset`` batches.

    Handles per-scale kinematic and visual windows, displacement targets,
    and optional pose features.

    Args:
        batch: List of sample dicts from ``VisualTemporalDataset``.

    Returns:
        Batched dictionary.
    """
    n_kin_scales = len(batch[0]["kinematic_windows"])
    n_vis_scales = len(batch[0]["seg_neck_windows"])
    has_pose = "pose_kp_windows" in batch[0]

    result: dict[str, Any] = {
        "kinematic_windows": [],
        "seg_neck_windows": [],
        "seg_backbone_windows": [],
        "depth_windows": [],
        "visual_valid_mask": [],
        "detection_conf": [],
        "trial_idx": torch.tensor([s["trial_idx"] for s in batch]),
        "target_position": torch.stack([s["target_position"] for s in batch]),
        "target_quaternion": torch.stack([s["target_quaternion"] for s in batch]),
        "target_angle": torch.stack([s["target_angle"] for s in batch]),
        "target": torch.stack([s["target"] for s in batch]),
        "current_position": torch.stack([s["current_position"] for s in batch]),
        "target_displacement": torch.stack(
            [s["target_displacement"] for s in batch]
        ),
    }

    if has_pose:
        result["pose_kp_windows"] = []
        result["pose_backbone_windows"] = []
        result["pose_geometric_windows"] = []
        result["pose_conf"] = []

    for i in range(n_kin_scales):
        result["kinematic_windows"].append(
            torch.stack([s["kinematic_windows"][i] for s in batch])
        )

    for i in range(n_vis_scales):
        result["seg_neck_windows"].append(
            torch.stack([s["seg_neck_windows"][i] for s in batch])
        )
        result["seg_backbone_windows"].append(
            torch.stack([s["seg_backbone_windows"][i] for s in batch])
        )
        result["depth_windows"].append(
            torch.stack([s["depth_windows"][i] for s in batch])
        )
        result["visual_valid_mask"].append(
            torch.stack([s["visual_valid_mask"][i] for s in batch])
        )
        result["detection_conf"].append(
            torch.stack([s["detection_conf"][i] for s in batch])
        )
        if has_pose:
            result["pose_kp_windows"].append(
                torch.stack([s["pose_kp_windows"][i] for s in batch])
            )
            result["pose_backbone_windows"].append(
                torch.stack([s["pose_backbone_windows"][i] for s in batch])
            )
            result["pose_geometric_windows"].append(
                torch.stack([s["pose_geometric_windows"][i] for s in batch])
            )
            result["pose_conf"].append(
                torch.stack([s["pose_conf"][i] for s in batch])
            )

    return result


# ============================================================================
# KinematicDataset (Foundation Model)
# ============================================================================


class KinematicDataset(Dataset):
    """Multi-scale dataset for kinematic foundation model training.

    For each sample at target frame ``t``:
        - ``scale_0``:  frames ``[t-9, ..., t]``     (10 frames, ~0.77 s)
        - ``scale_1``:  frames ``[t-49, ..., t]``    (50 frames, ~3.85 s)
        - ``scale_2``:  frames ``[t-99, ..., t]``   (100 frames, ~7.69 s)
        - ``target``:   frame ``t+1`` pose (16D)

    Windows shorter than the requested scale are zero-padded on the left.

    Augmentations (when ``augment=True``):
        - Additive Gaussian noise on kinematic features.
        - Speed perturbation is handled externally by the training script.
        - Tool swap (Tool 1 <-> Tool 2) is handled externally.

    Attributes:
        trials: List of ``(normalised_data, metadata)`` tuples.
        window_scales: List of integer window sizes.
        indices: List of ``(trial_idx, frame_idx)`` pairs.
        norm_stats: Normalisation statistics used (or ``None``).
    """

    def __init__(
        self,
        trial_data_list: list[np.ndarray],
        trial_metadata_list: list[dict[str, Any]],
        window_scales: list[int] | None = None,
        normalize: bool = True,
        norm_stats: NormalizationStats | None = None,
        augment: bool = False,
        noise_std: float = 0.01,
        stride: int = 1,
    ):
        """Initialise multi-scale kinematic dataset.

        Args:
            trial_data_list: List of (N_i, 30) arrays per trial.
            trial_metadata_list: List of metadata dicts per trial.
            window_scales: Window sizes (default ``[10, 50, 100]``).
            normalize: Apply z-score normalisation.
            norm_stats: Pre-computed stats; computed from data if ``None``.
            augment: Apply training augmentations.
            noise_std: Gaussian noise std for augmentation.
            stride: Frame stride for sampling (default 1).
        """
        super().__init__()
        self.window_scales = window_scales or DEFAULT_SCALES
        self.max_window = max(self.window_scales)
        self.augment = augment
        self.noise_std = noise_std

        # Normalisation
        if normalize:
            if norm_stats is None:
                norm_stats = NormalizationStats.from_trials(trial_data_list)
            self.norm_stats = norm_stats
            self.trials = [
                (norm_stats.normalize(data), meta)
                for data, meta in zip(trial_data_list, trial_metadata_list)
            ]
        else:
            self.norm_stats = None
            self.trials = list(zip(trial_data_list, trial_metadata_list))

        # Build index: (trial_idx, frame_idx) for valid target frames
        self.indices: list[tuple[int, int]] = []
        for trial_idx, (data, _) in enumerate(self.trials):
            n_frames = data.shape[0]
            for frame_idx in range(0, n_frames - 1, stride):
                self.indices.append((trial_idx, frame_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get multi-scale windows and next-frame target.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - ``"scales"``: list of (T_i, 30) tensors per window scale.
                - ``"target"``: (16,) target pose for frame ``t+1``.
                - ``"trial_idx"``: integer trial index.
                - ``"frame_idx"``: integer frame index.
        """
        trial_idx, frame_idx = self.indices[idx]
        data, _meta = self.trials[trial_idx]

        scales: list[torch.Tensor] = []
        for window_size in self.window_scales:
            start = frame_idx - window_size + 1
            end = frame_idx + 1

            if start >= 0:
                window = data[start:end].copy()
            else:
                available = data[:end].copy()
                pad_size = window_size - available.shape[0]
                padding = np.zeros((pad_size, data.shape[1]), dtype=data.dtype)
                window = np.concatenate([padding, available], axis=0)

            if self.augment and self.noise_std > 0:
                window = window + np.random.randn(*window.shape).astype(
                    np.float32
                ) * self.noise_std

            scales.append(torch.from_numpy(window))

        # Target: next frame tool poses (16D)
        target_frame = data[frame_idx + 1]
        target = np.concatenate(
            [target_frame[0:8], target_frame[8:16]]
        )

        return {
            "scales": scales,
            "target": torch.from_numpy(target),
            "trial_idx": trial_idx,
            "frame_idx": frame_idx,
        }


# ============================================================================
# TrialVisualCache
# ============================================================================


@dataclass
class TrialVisualCache:
    """Pre-loaded visual features for a single trial.

    All arrays are indexed by file frame index.  Missing frames are
    zero-filled.  The consolidated ``_visual_cache.npz`` file contains
    all feature types concatenated for fast I/O (~100 ms per trial).

    Attributes:
        n_frames: Number of allocated frames.
        seg_neck: (N, 2, 256) per-tool FPN neck features.
        seg_backbone: (N, 512) global backbone features.
        seg_conf: (N, 2) per-tool detection confidence.
        seg_valid: (N,) bool whether segmentation detected tools.
        depth: (N, 384) global depth embeddings.
        pose_kp: (N, 2, 8, 3) per-tool keypoints (optional).
        pose_backbone: (N, 2, 512) per-tool backbone (optional).
        pose_geometric: (N, 6) geometric features (optional).
        pose_conf: (N, 2) per-tool pose confidence (optional).
    """

    n_frames: int
    seg_neck: np.ndarray
    seg_backbone: np.ndarray
    seg_conf: np.ndarray
    seg_valid: np.ndarray
    depth: np.ndarray
    pose_kp: np.ndarray | None = None
    pose_backbone: np.ndarray | None = None
    pose_geometric: np.ndarray | None = None
    pose_conf: np.ndarray | None = None


def load_visual_cache(
    trial_path: Path,
    n_file_frames: int,
    *,
    seg_dir: str = "YOLO_FEATURES",
    depth_dir: str = "DEPTH/embeddings",
    pose_dir: str = "POSE_FEATURES",
    use_pose: bool = True,
) -> TrialVisualCache:
    """Load consolidated visual features for a trial.

    Reads from ``_visual_cache.npz`` if available.  Falls back to
    allocating zero arrays (kinematic-only mode).

    Args:
        trial_path: Path to trial directory.
        n_file_frames: Number of file frames to allocate.
        seg_dir: Subdirectory for YOLO segmentation features.
        depth_dir: Subdirectory for depth embeddings.
        pose_dir: Subdirectory for pose features.
        use_pose: Whether to load pose feature arrays.

    Returns:
        ``TrialVisualCache`` with contiguous arrays.
    """
    cache_path = trial_path / VISUAL_CACHE_FILENAME
    N = n_file_frames

    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            cached_n = data["seg_neck"].shape[0]
            if cached_n < N:
                # Cache is stale
                cache_path.unlink()
                raise FileNotFoundError("Cache stale")

            pose_kp = pose_backbone = pose_geometric = pose_conf = None
            if use_pose and "pose_kp" in data:
                pose_kp = data["pose_kp"][:N].astype(np.float32)
                pose_backbone = data["pose_backbone"][:N].astype(np.float32)
                pose_geometric = data["pose_geometric"][:N].astype(np.float32)
                pose_conf = data["pose_conf"][:N].astype(np.float32)

            return TrialVisualCache(
                n_frames=N,
                seg_neck=data["seg_neck"][:N].astype(np.float32),
                seg_backbone=data["seg_backbone"][:N].astype(np.float32),
                seg_conf=data["seg_conf"][:N].astype(np.float32),
                seg_valid=data["seg_valid"][:N].astype(bool),
                depth=data["depth"][:N].astype(np.float32),
                pose_kp=pose_kp,
                pose_backbone=pose_backbone,
                pose_geometric=pose_geometric,
                pose_conf=pose_conf,
            )
        except Exception:
            pass

    # Fallback: zero arrays (kinematic-only mode)
    return TrialVisualCache(
        n_frames=N,
        seg_neck=np.zeros((N, 2, SEG_NECK_DIM), dtype=np.float32),
        seg_backbone=np.zeros((N, SEG_BACKBONE_DIM), dtype=np.float32),
        seg_conf=np.zeros((N, 2), dtype=np.float32),
        seg_valid=np.zeros(N, dtype=bool),
        depth=np.zeros((N, DEPTH_DIM), dtype=np.float32),
        pose_kp=np.zeros((N, 2, POSE_NUM_KEYPOINTS, 3), dtype=np.float32) if use_pose else None,
        pose_backbone=np.zeros((N, 2, POSE_BACKBONE_DIM), dtype=np.float32) if use_pose else None,
        pose_geometric=np.zeros((N, POSE_GEOMETRIC_DIM), dtype=np.float32) if use_pose else None,
        pose_conf=np.zeros((N, 2), dtype=np.float32) if use_pose else None,
    )


# ============================================================================
# VisualTemporalDataset (Full BTPN)
# ============================================================================


class VisualTemporalDataset(Dataset):
    """Dense-sampling dataset with visual features for full BTPN training.

    Combines multi-scale kinematic windows with multi-scale visual feature
    windows (segmentation, depth, pose) and provides displacement targets
    for relative tracking.

    Dense sampling at ``interval=2`` produces ~47K training samples from
    the 7DOF2024 dataset.

    Args:
        trials: List of trial info dicts with keys ``"name"``, ``"path"``,
            ``"data"`` (N, 30), ``"metadata"``, and optionally
            ``"valid_frame_indices"``.
        scales: Kinematic window sizes (default ``[10, 50, 100]``).
        visual_scales: Visual window sizes (default ``[8, 40, 100, 200]``).
        norm_stats: Kinematic normalisation statistics.
        sample_interval: Frame interval for dense sampling (default 2).
        seg_features_dir: Subdirectory for YOLO features.
        depth_embeddings_dir: Subdirectory for depth embeddings.
        pose_features_dir: Subdirectory for pose features.
        use_pose_features: Enable pose keypoint modality.
        augment: Apply data augmentation.
        noise_std: Gaussian noise std for kinematic augmentation.
        tool_swap_prob: Probability of tool swap augmentation.
    """

    def __init__(
        self,
        trials: list[dict[str, Any]],
        scales: list[int] | None = None,
        visual_scales: list[int] | None = None,
        norm_stats: NormalizationStats | None = None,
        sample_interval: int = 2,
        seg_features_dir: str = "YOLO_FEATURES",
        depth_embeddings_dir: str = "DEPTH/embeddings",
        pose_features_dir: str = "POSE_FEATURES",
        use_pose_features: bool = True,
        augment: bool = False,
        noise_std: float = 0.01,
        tool_swap_prob: float = 0.5,
    ):
        super().__init__()
        self.scales = scales or DEFAULT_SCALES
        self.visual_scales = visual_scales or [8, 40, 100, 200]
        self.max_scale = max(max(self.scales), max(self.visual_scales))
        self.norm_stats = norm_stats
        self.use_pose_features = use_pose_features
        self.augment = augment
        self.noise_std = noise_std
        self.tool_swap_prob = tool_swap_prob

        # Pre-load visual caches
        print(f"  Pre-loading visual features for {len(trials)} trials...")
        t0 = time.time()
        self._cache: dict[str, TrialVisualCache] = {}
        total_frames = 0

        for i, trial in enumerate(trials):
            trial_path = trial["path"]
            path_key = str(trial_path)
            valid_indices = trial.get(
                "valid_frame_indices", list(range(len(trial["data"])))
            )
            n_file_frames = (
                max(valid_indices) + 1 if valid_indices else len(trial["data"])
            )
            cache = load_visual_cache(
                trial_path=trial_path,
                n_file_frames=n_file_frames,
                seg_dir=seg_features_dir,
                depth_dir=depth_embeddings_dir,
                pose_dir=pose_features_dir,
                use_pose=use_pose_features,
            )
            self._cache[path_key] = cache
            total_frames += cache.n_frames

            if (i + 1) % 10 == 0 or i == len(trials) - 1:
                elapsed = time.time() - t0
                print(
                    f"    [{i + 1}/{len(trials)}] "
                    f"{total_frames:,} frames cached ({elapsed:.1f}s)"
                )

        elapsed = time.time() - t0
        print(f"  Visual cache: {total_frames:,} frames in {elapsed:.1f}s")

        # Normalise kinematic data
        self._trials_normalized: list[dict[str, Any]] = []
        for trial in trials:
            trial_copy = dict(trial)
            raw_data = trial["data"]
            trial_copy["data_norm"] = (
                norm_stats.normalize(raw_data) if norm_stats is not None else raw_data
            )
            trial_copy["data_raw"] = raw_data
            self._trials_normalized.append(trial_copy)

        # Build dense sample list
        self.samples: list[tuple[int, dict[str, Any], int, int]] = []
        for trial_idx, trial in enumerate(self._trials_normalized):
            data = trial["data_norm"]
            n_frames = len(data)
            valid_indices = trial.get(
                "valid_frame_indices", list(range(n_frames))
            )
            for kin_idx in range(self.max_scale, n_frames - 1, sample_interval):
                file_idx = (
                    valid_indices[kin_idx]
                    if kin_idx < len(valid_indices)
                    else kin_idx
                )
                self.samples.append((trial_idx, trial, kin_idx, file_idx))

        print(
            f"  Samples: {len(self.samples):,} "
            f"(interval={sample_interval}, max_scale={self.max_scale})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get multi-scale kinematic + visual windows with displacement targets.

        Returns:
            Dictionary with:
                - ``kinematic_windows``: list of (T, 30) tensors per kin scale.
                - ``seg_neck_windows`` / ``seg_backbone_windows`` /
                  ``depth_windows``: list of visual tensors per visual scale.
                - ``visual_valid_mask`` / ``detection_conf``: per visual scale.
                - ``target``: (16,) normalised target for frame ``t+1``.
                - ``target_position``: (2, 3).
                - ``target_quaternion``: (2, 4).
                - ``target_angle``: (2, 1).
                - ``current_position``: (2, 3) at frame ``t``.
                - ``target_displacement``: (2, 3) = target_pos - current_pos.
                - Optionally ``pose_kp_windows``, ``pose_backbone_windows``,
                  ``pose_geometric_windows``, ``pose_conf``.
        """
        trial_idx, trial, center_idx, center_file_idx = self.samples[idx]
        data_norm = trial["data_norm"]
        path_key = str(trial["path"])
        cache = self._cache[path_key]

        # --- Kinematic windows ---
        kinematic_windows: list[np.ndarray] = []
        for scale in self.scales:
            start = max(0, center_idx - scale)
            end = center_idx
            window = data_norm[start:end].copy()
            if len(window) < scale:
                pad = np.zeros(
                    (scale - len(window), KINEMATIC_DIM), dtype=np.float32
                )
                window = np.concatenate([pad, window], axis=0)
            kinematic_windows.append(window)

        if self.augment and self.noise_std > 0:
            for i, w in enumerate(kinematic_windows):
                kinematic_windows[i] = w + np.random.randn(*w.shape).astype(
                    np.float32
                ) * self.noise_std

        # --- Visual windows (at visual_scales, not kinematic scales) ---
        seg_neck_windows: list[np.ndarray] = []
        seg_backbone_windows: list[np.ndarray] = []
        depth_windows: list[np.ndarray] = []
        detection_conf_windows: list[np.ndarray] = []
        visual_valid_windows: list[np.ndarray] = []
        pose_kp_windows: list[np.ndarray] = []
        pose_backbone_windows: list[np.ndarray] = []
        pose_geometric_windows: list[np.ndarray] = []
        pose_conf_windows: list[np.ndarray] = []

        for scale in self.visual_scales:
            start_fi = max(0, center_file_idx - scale)
            end_fi = center_file_idx
            actual_len = end_fi - start_fi
            n_pad = scale - actual_len

            neck = cache.seg_neck[start_fi:end_fi]
            bb = cache.seg_backbone[start_fi:end_fi]
            dep = cache.depth[start_fi:end_fi]
            valid = cache.seg_valid[start_fi:end_fi]
            conf = cache.seg_conf[start_fi:end_fi]

            if n_pad > 0:
                neck = np.concatenate(
                    [np.zeros((n_pad, 2, SEG_NECK_DIM), dtype=np.float32), neck]
                )
                bb = np.concatenate(
                    [np.zeros((n_pad, SEG_BACKBONE_DIM), dtype=np.float32), bb]
                )
                dep = np.concatenate(
                    [np.zeros((n_pad, DEPTH_DIM), dtype=np.float32), dep]
                )
                valid = np.concatenate([np.zeros(n_pad, dtype=bool), valid])
                conf = np.concatenate(
                    [np.zeros((n_pad, 2), dtype=np.float32), conf]
                )

            seg_neck_windows.append(neck)
            seg_backbone_windows.append(bb)
            depth_windows.append(dep)
            visual_valid_windows.append(valid)
            detection_conf_windows.append(conf)

            if self.use_pose_features and cache.pose_kp is not None:
                pkp = cache.pose_kp[start_fi:end_fi]
                pbb = cache.pose_backbone[start_fi:end_fi]
                pgeo = cache.pose_geometric[start_fi:end_fi]
                pcf = cache.pose_conf[start_fi:end_fi]

                if n_pad > 0:
                    pkp = np.concatenate([
                        np.zeros((n_pad, 2, POSE_NUM_KEYPOINTS, 3), dtype=np.float32),
                        pkp,
                    ])
                    pbb = np.concatenate([
                        np.zeros((n_pad, 2, POSE_BACKBONE_DIM), dtype=np.float32),
                        pbb,
                    ])
                    pgeo = np.concatenate([
                        np.zeros((n_pad, POSE_GEOMETRIC_DIM), dtype=np.float32),
                        pgeo,
                    ])
                    pcf = np.concatenate([
                        np.zeros((n_pad, 2), dtype=np.float32), pcf
                    ])

                pose_kp_windows.append(pkp)
                pose_backbone_windows.append(pbb)
                pose_geometric_windows.append(pgeo)
                pose_conf_windows.append(pcf)

        # --- Current position at frame t ---
        current_frame = data_norm[center_idx]
        current_pos = np.stack(
            [current_frame[0:3], current_frame[8:11]]
        )  # (2, 3)

        # --- Target: frame t+1 ---
        target_frame = data_norm[center_idx + 1]
        target_pos = np.stack([target_frame[0:3], target_frame[8:11]])
        target_quat = np.stack([target_frame[3:7], target_frame[11:15]])
        target_angle = np.stack([target_frame[7:8], target_frame[15:16]])
        target_16 = np.concatenate([target_frame[0:8], target_frame[8:16]])

        # --- Displacement ---
        target_displacement = target_pos - current_pos

        # --- Tool swap augmentation ---
        if self.augment and np.random.random() < self.tool_swap_prob:
            current_pos = current_pos[[1, 0]]
            target_pos = target_pos[[1, 0]]
            target_quat = target_quat[[1, 0]]
            target_angle = target_angle[[1, 0]]
            target_displacement = target_displacement[[1, 0]]
            target_16 = np.concatenate([target_16[8:16], target_16[0:8]])

            for i, w in enumerate(kinematic_windows):
                swapped = w.copy()
                swapped[:, 0:8] = w[:, 8:16]
                swapped[:, 8:16] = w[:, 0:8]
                kinematic_windows[i] = swapped

            for i in range(len(seg_neck_windows)):
                seg_neck_windows[i] = seg_neck_windows[i][:, [1, 0], :]
                detection_conf_windows[i] = detection_conf_windows[i][:, [1, 0]]

            if self.use_pose_features and len(pose_kp_windows) > 0:
                for i in range(len(pose_kp_windows)):
                    pose_kp_windows[i] = pose_kp_windows[i][:, [1, 0], :]
                    pose_backbone_windows[i] = pose_backbone_windows[i][:, [1, 0], :]
                    pose_conf_windows[i] = pose_conf_windows[i][:, [1, 0]]
                    geo = pose_geometric_windows[i].copy()
                    geo[:, [0, 1]] = geo[:, [1, 0]]
                    geo[:, [2, 3]] = geo[:, [3, 2]]
                    geo[:, [4, 5]] = geo[:, [5, 4]]
                    pose_geometric_windows[i] = geo

        # Build result
        result: dict[str, Any] = {
            "trial_idx": trial_idx,
            "kinematic_windows": [
                torch.from_numpy(w).float() for w in kinematic_windows
            ],
            "seg_neck_windows": [
                torch.from_numpy(w).float() for w in seg_neck_windows
            ],
            "seg_backbone_windows": [
                torch.from_numpy(w).float() for w in seg_backbone_windows
            ],
            "depth_windows": [
                torch.from_numpy(w).float() for w in depth_windows
            ],
            "visual_valid_mask": [
                torch.from_numpy(w) for w in visual_valid_windows
            ],
            "detection_conf": [
                torch.from_numpy(w).float() for w in detection_conf_windows
            ],
            "target_position": torch.from_numpy(target_pos).float(),
            "target_quaternion": torch.from_numpy(target_quat).float(),
            "target_angle": torch.from_numpy(target_angle).float(),
            "target": torch.from_numpy(target_16).float(),
            "current_position": torch.from_numpy(current_pos).float(),
            "target_displacement": torch.from_numpy(target_displacement).float(),
        }

        if self.use_pose_features and len(pose_kp_windows) > 0:
            result["pose_kp_windows"] = [
                torch.from_numpy(w).float() for w in pose_kp_windows
            ]
            result["pose_backbone_windows"] = [
                torch.from_numpy(w).float() for w in pose_backbone_windows
            ]
            result["pose_geometric_windows"] = [
                torch.from_numpy(w).float() for w in pose_geometric_windows
            ]
            result["pose_conf"] = [
                torch.from_numpy(w).float() for w in pose_conf_windows
            ]

        return result


# ============================================================================
# Factory: create_dataloaders
# ============================================================================


def create_dataloaders(
    config: dict[str, Any],
    paths_config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, NormalizationStats]:
    """Create train and validation DataLoaders from YAML-loaded configs.

    Supports both **kinematic-only** mode (``KinematicDataset``) and
    **visual-temporal** mode (``VisualTemporalDataset``) depending on
    whether ``use_visual_features`` is set in the config.

    Args:
        config: Model/training configuration dict (from ``btpn.yaml`` or
            ``kinematic_foundation.yaml``).
        paths_config: Paths configuration dict (from ``paths.yaml``).

    Returns:
        ``(train_loader, val_loader, norm_stats)`` tuple.
    """
    data_root = Path(paths_config["data_root"])
    dataset_a_name = paths_config.get("dataset_a", "7DOF2024")

    # Discover and split trials
    print(f"Loading {dataset_a_name} dataset...")
    trial_dirs = discover_trials(data_root, dataset_a_name)
    print(f"  Found {len(trial_dirs)} trials")

    train_dirs, val_dirs = split_7dof_fixed(trial_dirs, seed=config.get("seed", 42))
    val_numbers = sorted(_extract_trial_number(td) for td in val_dirs)
    print(
        f"  Train: {len(train_dirs)} trials, "
        f"Val: {len(val_dirs)} trials (numbers {val_numbers})"
    )

    # Load trials
    train_data: list[np.ndarray] = []
    train_meta: list[dict[str, Any]] = []
    for td in train_dirs:
        result = load_trial(td)
        if result is not None:
            train_data.append(result["data"])
            train_meta.append(result["metadata"])

    val_data: list[np.ndarray] = []
    val_meta: list[dict[str, Any]] = []
    for td in val_dirs:
        result = load_trial(td)
        if result is not None:
            val_data.append(result["data"])
            val_meta.append(result["metadata"])

    print(f"  Loaded: {len(train_data)} train, {len(val_data)} val trials")

    # Compute normalisation from training set only
    norm_stats = NormalizationStats.from_trials(train_data)

    window_scales = config.get("window_scales", DEFAULT_SCALES)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", True)

    # Choose dataset class based on config
    use_visual = config.get("use_visual_features", False)

    if use_visual:
        # Build trial info dicts for VisualTemporalDataset
        train_trials_info = _build_trial_info(
            train_dirs, train_data, train_meta
        )
        val_trials_info = _build_trial_info(val_dirs, val_data, val_meta)

        visual_scales = config.get("visual_window_scales", [8, 40, 100, 200])
        sample_interval = config.get("sample_interval", 2)

        train_dataset = VisualTemporalDataset(
            trials=train_trials_info,
            scales=window_scales,
            visual_scales=visual_scales,
            norm_stats=norm_stats,
            sample_interval=sample_interval,
            seg_features_dir=paths_config.get("seg_features_dir", "YOLO_FEATURES"),
            depth_embeddings_dir=paths_config.get(
                "depth_embeddings_dir", "DEPTH/embeddings"
            ),
            pose_features_dir=paths_config.get("pose_features_dir", "POSE_FEATURES"),
            use_pose_features=config.get("use_pose_features", True),
            augment=config.get("augment_train", True),
            noise_std=config.get("noise_std", 0.01),
            tool_swap_prob=config.get("tool_swap_prob", 0.5),
        )
        val_dataset = VisualTemporalDataset(
            trials=val_trials_info,
            scales=window_scales,
            visual_scales=visual_scales,
            norm_stats=norm_stats,
            sample_interval=sample_interval,
            seg_features_dir=paths_config.get("seg_features_dir", "YOLO_FEATURES"),
            depth_embeddings_dir=paths_config.get(
                "depth_embeddings_dir", "DEPTH/embeddings"
            ),
            pose_features_dir=paths_config.get("pose_features_dir", "POSE_FEATURES"),
            use_pose_features=config.get("use_pose_features", True),
            augment=False,
        )

        collate_fn = collate_visual_temporal
    else:
        train_dataset = KinematicDataset(
            trial_data_list=train_data,
            trial_metadata_list=train_meta,
            window_scales=window_scales,
            normalize=True,
            norm_stats=norm_stats,
            augment=config.get("augment_train", True),
            noise_std=config.get("noise_std", 0.01),
            stride=config.get("stride", 1),
        )
        val_dataset = KinematicDataset(
            trial_data_list=val_data,
            trial_metadata_list=val_meta,
            window_scales=window_scales,
            normalize=True,
            norm_stats=norm_stats,
            augment=False,
        )
        collate_fn = collate_multiscale

    print(
        f"  Dataset samples: {len(train_dataset)} train, {len(val_dataset)} val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(
        f"  Batches: {len(train_loader)} train, {len(val_loader)} val"
    )

    return train_loader, val_loader, norm_stats


def _build_trial_info(
    trial_dirs: list[Path],
    data_list: list[np.ndarray],
    meta_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build trial info dicts for ``VisualTemporalDataset``.

    Args:
        trial_dirs: Paths to trial directories.
        data_list: Kinematic data arrays.
        meta_list: Metadata dicts.

    Returns:
        List of trial info dicts with ``"name"``, ``"path"``, ``"data"``,
        ``"metadata"``, and ``"valid_frame_indices"`` keys.
    """
    trials_info: list[dict[str, Any]] = []
    for td, data, meta in zip(trial_dirs, data_list, meta_list):
        n_frames = data.shape[0]
        valid_mask = ~np.isnan(data).any(axis=1)
        valid_indices = np.where(valid_mask)[0].tolist() if not valid_mask.all() else list(range(n_frames))

        trials_info.append({
            "name": td.name,
            "path": td,
            "data": data,
            "metadata": meta,
            "valid_frame_indices": valid_indices,
        })
    return trials_info
