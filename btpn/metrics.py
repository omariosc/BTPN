"""Evaluation metrics for the Bayesian Temporal Pose Network (BTPN).

Pure NumPy/SciPy metric functions for evaluating surgical tool pose
predictions, with no PyTorch dependency. Covers:

- Position error metrics (MAE, RMSE, percentiles)
- Rotation error metrics (geodesic, Euler decomposition)
- Jaw angle metrics
- Uncertainty calibration (ECE, AUSE, coverage)
- Formatted result tables (plain text and LaTeX)

All quaternions are expected in (w, x, y, z) convention.
Positions are in millimetres.

Paper reference: Section 5 (Experiments and Evaluation).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# Quaternion Utilities
# =============================================================================


def _sanitize_quaternions(quats: np.ndarray) -> np.ndarray:
    """Replace zero-norm, near-zero-norm, or NaN quaternions with identity.

    Args:
        quats: (..., 4) quaternions in (w, x, y, z) order.

    Returns:
        Copy of *quats* with invalid entries replaced by [1, 0, 0, 0].
    """
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    bad = (norms.squeeze(-1) < 1e-7) | np.any(~np.isfinite(quats), axis=-1)
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        quats = quats.copy()
        quats[bad] = np.array([1.0, 0.0, 0.0, 0.0])
    return quats


def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    """Project quaternions onto the unit hypersphere.

    Args:
        quats: (..., 4) quaternions (need not be unit norm).

    Returns:
        Unit-norm quaternions with the same leading shape.
    """
    quats = _sanitize_quaternions(quats)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats / np.clip(norms, 1e-8, None)


# =============================================================================
# Position Metrics
# =============================================================================


def compute_position_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute position error statistics in millimetres.

    Computes Euclidean distance between predicted and target positions,
    then reports MAE, RMSE, standard deviation, and percentiles
    (P50, P75, P90, P95, P99).

    Args:
        pred: (N, 3) predicted positions in mm.
        target: (N, 3) ground-truth positions in mm.

    Returns:
        Dictionary with keys:
            mae_mm, rmse_mm, std_mm, p50_mm, p75_mm, p90_mm, p95_mm, p99_mm,
            per_axis (sub-dict with X/Y/Z MAE and RMSE).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

    # Euclidean error per sample
    err = np.linalg.norm(pred - target, axis=-1)  # (N,)
    axis_err = np.abs(pred - target)  # (N, 3)

    result: dict[str, Any] = {
        "mae_mm": float(err.mean()),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "std_mm": float(err.std()),
        "p50_mm": float(np.percentile(err, 50)),
        "p75_mm": float(np.percentile(err, 75)),
        "p90_mm": float(np.percentile(err, 90)),
        "p95_mm": float(np.percentile(err, 95)),
        "p99_mm": float(np.percentile(err, 99)),
    }

    # Per-axis breakdown
    axis_names = ["X", "Y", "Z"]
    per_axis: dict[str, dict[str, float]] = {}
    for i, name in enumerate(axis_names):
        ax = axis_err[:, i]
        per_axis[name] = {
            "mae_mm": float(ax.mean()),
            "rmse_mm": float(np.sqrt(np.mean(ax ** 2))),
        }
    result["per_axis"] = per_axis

    return result


# =============================================================================
# Rotation Metrics
# =============================================================================


def compute_geodesic_error(
    pred_quat: np.ndarray,
    target_quat: np.ndarray,
) -> dict[str, float]:
    """Compute geodesic rotation error between quaternion predictions.

    Uses the double-cover-safe formula:
        error = 2 * arccos(|q_pred . q_target|)

    Both inputs are sanitized and re-normalized to unit norm before
    the dot product.

    Args:
        pred_quat: (N, 4) predicted quaternions in (w, x, y, z) order.
        target_quat: (N, 4) ground-truth quaternions in (w, x, y, z) order.

    Returns:
        Dictionary with keys:
            mean_deg, rmse_deg, std_deg, median_deg,
            p50_deg, p75_deg, p90_deg, p95_deg, p99_deg.
    """
    q1 = _normalize_quaternions(pred_quat)
    q2 = _normalize_quaternions(target_quat)

    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    geo_deg = 2.0 * np.arccos(dot) * 180.0 / np.pi  # (N,)

    return {
        "mean_deg": float(geo_deg.mean()),
        "rmse_deg": float(np.sqrt(np.mean(geo_deg ** 2))),
        "std_deg": float(geo_deg.std()),
        "median_deg": float(np.median(geo_deg)),
        "p50_deg": float(np.percentile(geo_deg, 50)),
        "p75_deg": float(np.percentile(geo_deg, 75)),
        "p90_deg": float(np.percentile(geo_deg, 90)),
        "p95_deg": float(np.percentile(geo_deg, 95)),
        "p99_deg": float(np.percentile(geo_deg, 99)),
    }


def compute_euler_errors(
    pred_quat: np.ndarray,
    target_quat: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-axis angular errors via Euler angle decomposition.

    Converts quaternion predictions and ground truth to Euler angles
    (XYZ intrinsic convention) and computes the absolute angular
    difference per axis, wrapping to [-180, 180] before taking the
    absolute value.

    Args:
        pred_quat: (N, 4) predicted quaternions in (w, x, y, z) order.
        target_quat: (N, 4) ground-truth quaternions in (w, x, y, z) order.

    Returns:
        Dictionary keyed by {"roll", "pitch", "yaw"}, each containing
        mae_deg, rmse_deg, std_deg.
    """
    pred_quat = _sanitize_quaternions(pred_quat)
    target_quat = _sanitize_quaternions(target_quat)

    # scipy expects (x, y, z, w) order
    pred_euler = Rotation.from_quat(
        pred_quat[:, [1, 2, 3, 0]]
    ).as_euler("xyz", degrees=True)
    gt_euler = Rotation.from_quat(
        target_quat[:, [1, 2, 3, 0]]
    ).as_euler("xyz", degrees=True)

    diff = pred_euler - gt_euler
    # Wrap to [-180, 180]
    diff = (diff + 180.0) % 360.0 - 180.0
    abs_diff = np.abs(diff)  # (N, 3)

    axis_names = ["roll", "pitch", "yaw"]
    result: dict[str, dict[str, float]] = {}
    for i, name in enumerate(axis_names):
        ax = abs_diff[:, i]
        result[name] = {
            "mae_deg": float(ax.mean()),
            "rmse_deg": float(np.sqrt(np.mean(ax ** 2))),
            "std_deg": float(ax.std()),
        }

    return result


# =============================================================================
# Jaw Angle Metrics
# =============================================================================


def compute_jaw_metrics(
    pred_angle: np.ndarray,
    target_angle: np.ndarray,
) -> dict[str, float]:
    """Compute jaw/gripper angle prediction errors.

    Args:
        pred_angle: (N,) or (N, 1) predicted jaw angles.
        target_angle: (N,) or (N, 1) ground-truth jaw angles.

    Returns:
        Dictionary with mae and rmse.
    """
    pred_angle = pred_angle.ravel()
    target_angle = target_angle.ravel()

    err = np.abs(pred_angle - target_angle)

    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


# =============================================================================
# Uncertainty Calibration
# =============================================================================


def compute_ece(
    errors: np.ndarray,
    sigmas: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error for Gaussian uncertainty.

    Bins predictions by predicted sigma quantile. For each bin, checks
    whether the fraction of errors within the predicted confidence
    interval matches the expected coverage.

    The ECE value is based on the 1-sigma (68.3%) coverage level:
    for a well-calibrated Gaussian, 68.3% of errors should fall within
    1 sigma. The ECE measures the average absolute deviation from this
    expected coverage across quantile bins.

    Args:
        errors: (N,) absolute prediction errors.
        sigmas: (N,) predicted standard deviations (uncertainty).
        n_bins: Number of quantile bins for sigma stratification.

    Returns:
        Dictionary with:
            ece: Expected Calibration Error (scalar, lower is better).
            n_bins: Number of bins used.
            n_total: Number of samples binned.
            bins: List of per-bin calibration data.
            global_coverages: Dict mapping coverage level strings
                to observed coverage fractions.
    """
    errors = errors.ravel()
    sigmas = sigmas.ravel()

    if len(errors) != len(sigmas):
        raise ValueError(
            f"Length mismatch: errors ({len(errors)}) vs sigmas ({len(sigmas)})"
        )

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(sigmas, quantiles)

    coverage_levels = [0.5, 0.683, 0.9, 0.95, 0.99]
    z_scores = [0.6745, 1.0, 1.6449, 1.96, 2.576]

    bin_data: list[dict[str, Any]] = []
    total_ece = 0.0
    total_count = 0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1] if i < n_bins - 1 else np.inf
        if i == 0:
            mask = (sigmas >= lo) & (sigmas <= hi)
        else:
            mask = (sigmas > lo) & (sigmas <= hi)

        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue

        bin_errors = errors[mask]
        bin_sigmas = sigmas[mask]

        coverages: dict[str, float] = {}
        for level, z in zip(coverage_levels, z_scores):
            within = float((bin_errors <= z * bin_sigmas).mean())
            coverages[f"{level:.3f}"] = within

        # Use 68.3% (1-sigma) coverage for ECE computation
        expected_68 = 0.683
        observed_68 = coverages["0.683"]
        ece_contribution = abs(observed_68 - expected_68) * n_in_bin
        total_ece += ece_contribution
        total_count += n_in_bin

        bin_data.append({
            "bin_idx": i,
            "sigma_range": [float(lo), float(hi if hi != np.inf else bin_edges[-1])],
            "n_samples": n_in_bin,
            "mean_sigma": float(bin_sigmas.mean()),
            "mean_error": float(bin_errors.mean()),
            "coverages": coverages,
        })

    ece_value = total_ece / max(total_count, 1)

    # Global coverages
    global_coverages: dict[str, float] = {}
    for level, z in zip(coverage_levels, z_scores):
        within = float((errors <= z * sigmas).mean())
        global_coverages[f"{level:.3f}"] = within

    return {
        "ece": float(ece_value),
        "n_bins": n_bins,
        "n_total": total_count,
        "bins": bin_data,
        "global_coverages": global_coverages,
    }


def vmf_rotation_angle_cdf(
    theta: np.ndarray, kappa: np.ndarray, n_nodes: int = 96
) -> np.ndarray:
    """P(rotation angle <= theta) for q ~ vMF(mu, kappa) on S^3.

    The rotation angle between q and mu is theta = 2 * arccos(|mu . q|), in
    [0, pi]. With s = |mu . q|, the vMF implies the density

        g(s) = 8 pi C_4(kappa) sqrt(1 - s^2) cosh(kappa s),   s in [0, 1],

    with C_4(kappa) = kappa / (4 pi^2 I_1(kappa)). So

        P(theta' <= theta) = integral of g(s) from cos(theta / 2) to 1.

    The integral is evaluated by Gauss-Legendre quadrature after the
    substitution s = 1 - v^2, which removes the square-root endpoint
    singularity; all terms are combined in log space so large kappa is stable.

    Args:
        theta: Rotation angles in radians (...,).
        kappa: vMF concentrations (...,), same shape as ``theta``.
        n_nodes: Quadrature nodes.

    Returns:
        CDF values in [0, 1] (...,).
    """
    from scipy.special import i1e

    theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, np.pi)
    kappa = np.asarray(kappa, dtype=np.float64)
    theta, kappa = np.broadcast_arrays(theta, kappa)

    log_c4 = np.log(kappa) - np.log(4.0 * np.pi**2) - (np.log(i1e(kappa)) + kappa)
    v_max = np.sqrt(1.0 - np.cos(theta / 2.0))  # s = 1 - v^2 runs from 1 down to cos(theta/2)

    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    v = 0.5 * v_max[..., None] * (nodes + 1.0)          # (..., n_nodes) in [0, v_max]
    w = 0.5 * v_max[..., None] * weights
    s = 1.0 - v**2
    k = kappa[..., None]
    ks = k * s
    log_cosh = ks + np.log1p(np.exp(-2.0 * ks)) - np.log(2.0)
    # ds = 2 v dv; sqrt(1 - s^2) = v * sqrt(2 - v^2)
    log_integrand = (
        np.log(8.0 * np.pi) + log_c4[..., None] + log_cosh
        + np.log(2.0) + 2.0 * np.log(np.maximum(v, 1e-300)) + 0.5 * np.log(2.0 - v**2)
    )
    cdf = np.sum(w * np.exp(log_integrand), axis=-1)
    return np.clip(cdf, 0.0, 1.0)


def compute_rotation_ece_vmf(
    angle_errors: np.ndarray,
    kappas: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Expected Calibration Error for vMF rotation uncertainty.

    Mirrors :func:`compute_ece` (quantile bins of the predicted uncertainty,
    68.3 % coverage as the headline level) but uses the exact rotation-angle
    distribution implied by each kappa via :func:`vmf_rotation_angle_cdf`.
    An error is "within the level-alpha region" when its predicted CDF value
    (probability integral transform) is at most alpha.

    Args:
        angle_errors: (N,) geodesic rotation errors in radians.
        kappas: (N,) predicted vMF concentrations.
        n_bins: Number of quantile bins over kappa.

    Returns:
        Dict with ``ece`` (68.3 % level), ``global_coverages`` for levels
        0.5 / 0.683 / 0.9 / 0.95 / 0.99, ``n_bins`` and ``n_total``.
    """
    errors = np.asarray(angle_errors, dtype=np.float64).ravel()
    kappas = np.asarray(kappas, dtype=np.float64).ravel()
    if errors.shape != kappas.shape:
        raise ValueError(f"Length mismatch: errors {errors.shape} vs kappas {kappas.shape}")

    pit = vmf_rotation_angle_cdf(errors, kappas)
    levels = [0.5, 0.683, 0.9, 0.95, 0.99]

    edges = np.quantile(kappas, np.linspace(0.0, 1.0, n_bins + 1))
    total, count = 0.0, 0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (kappas >= lo) & (kappas <= hi) if i == 0 else (kappas > lo) & (kappas <= hi)
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        total += abs(float((pit[mask] <= 0.683).mean()) - 0.683) * n_in
        count += n_in

    return {
        "ece": float(total / max(count, 1)),
        "global_coverages": {f"{a:.3f}": float((pit <= a).mean()) for a in levels},
        "n_bins": n_bins,
        "n_total": count,
    }


def compute_ause(
    errors: np.ndarray,
    sigmas: np.ndarray,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Compute Area Under the Sparsification Error curve (AUSE).

    Sorts predictions by uncertainty (descending). Incrementally
    removes the highest-uncertainty predictions and measures the
    remaining error. A well-calibrated model should see error
    decrease as uncertain samples are removed.

    The normalized AUSE compares the model's sparsification curve
    against a random baseline and an oracle (sorted by actual error):
        AUSE_norm = (AUSE_model - AUSE_oracle) / (AUSE_random - AUSE_oracle)

    A value of 0 indicates oracle-level calibration; 1 indicates
    no better than random ordering.

    Args:
        errors: (N,) absolute prediction errors.
        sigmas: (N,) predicted uncertainty values.
        n_steps: Number of sparsification steps (fractions to evaluate).

    Returns:
        Dictionary with:
            ause: Raw area under the sparsification curve.
            ause_oracle: Oracle sparsification area (lower bound).
            ause_random: Random sparsification area (upper bound).
            ause_normalized: Normalized AUSE in [0, 1].
            fractions: Removal fractions evaluated.
            sigma_curve: Mean error at each fraction (model ordering).
            oracle_curve: Mean error at each fraction (oracle ordering).
            random_curve: Mean error at each fraction (no removal effect).
    """
    errors = errors.ravel()
    sigmas = sigmas.ravel()

    n = len(errors)
    fractions = np.linspace(0, 1, n_steps + 1)[:-1]  # [0.0, 0.05, ..., 0.95]

    # Sort by sigma descending (remove most uncertain first)
    sigma_order = np.argsort(-sigmas)
    # Oracle: sort by actual error descending
    error_order = np.argsort(-errors)

    random_curve: list[float] = []
    sigma_curve: list[float] = []
    oracle_curve: list[float] = []

    for frac in fractions:
        n_remove = int(frac * n)
        n_keep = n - n_remove
        if n_keep == 0:
            break

        keep_sigma = sigma_order[n_remove:]
        sigma_curve.append(float(errors[keep_sigma].mean()))

        keep_oracle = error_order[n_remove:]
        oracle_curve.append(float(errors[keep_oracle].mean()))

        random_curve.append(float(errors.mean()))

    x = fractions[:len(sigma_curve)]
    ause_sigma = float(np.trapezoid(sigma_curve, x)) if len(x) > 1 else 0.0
    ause_oracle = float(np.trapezoid(oracle_curve, x)) if len(x) > 1 else 0.0
    ause_random = float(np.trapezoid(random_curve, x)) if len(x) > 1 else 0.0

    ause_normalized = (
        (ause_sigma - ause_oracle) / max(ause_random - ause_oracle, 1e-8)
    )

    return {
        "ause": ause_sigma,
        "ause_oracle": ause_oracle,
        "ause_random": ause_random,
        "ause_normalized": float(ause_normalized),
        "fractions": x.tolist(),
        "sigma_curve": sigma_curve,
        "oracle_curve": oracle_curve,
        "random_curve": random_curve,
    }


def compute_coverage(
    errors: np.ndarray,
    sigmas: np.ndarray,
    levels: list[float] | None = None,
) -> dict[str, float]:
    """Compute prediction interval coverage at specified confidence levels.

    For each confidence level, computes the fraction of errors that
    fall within the Gaussian prediction interval at that level.

    Args:
        errors: (N,) absolute prediction errors.
        sigmas: (N,) predicted standard deviations.
        levels: Confidence levels to evaluate. Defaults to
            [0.5, 0.683, 0.9, 0.95, 0.99].

    Returns:
        Dictionary mapping "{level}" to observed coverage fraction.
        For example: {"0.683": 0.71, "0.950": 0.93}.
    """
    if levels is None:
        levels = [0.5, 0.683, 0.9, 0.95, 0.99]

    errors = errors.ravel()
    sigmas = sigmas.ravel()

    # z-scores corresponding to each confidence level
    from scipy.stats import norm as normal_dist

    result: dict[str, float] = {}
    for level in levels:
        z = normal_dist.ppf(0.5 + level / 2.0)
        within = float((errors <= z * sigmas).mean())
        result[f"{level:.3f}"] = within

    return result


# =============================================================================
# Result Formatting -- Plain Text
# =============================================================================


def format_results_table(
    results: dict[str, dict[str, Any]],
    title: str = "Evaluation Results",
) -> str:
    """Format a multi-model results comparison as a plain text table.

    Expects *results* to be a dictionary mapping model names to metric
    dictionaries with keys "position", "rotation", and optionally
    "jaw_angle" and "uncertainty".

    Args:
        results: Mapping from model name to metric dict. Each metric
            dict should contain "position" (with mae_mm, rmse_mm, p50_mm,
            p90_mm, p99_mm) and "rotation" (with mean_deg, rmse_deg,
            median_deg, p90_deg).
        title: Title string for the table header.

    Returns:
        Formatted multi-line string.
    """
    model_names = list(results.keys())
    col_width = max(12, max((len(n) for n in model_names), default=8) + 2)

    # Build header
    header = f"{'Metric':<26}"
    for name in model_names:
        header += f" {name:>{col_width}}"
    sep = "-" * len(header)

    lines: list[str] = [title, "=" * len(title), "", header, sep]

    # Metric rows
    metric_specs: list[tuple[str, str, str]] = [
        ("Position MAE (mm)", "position", "mae_mm"),
        ("Position RMSE (mm)", "position", "rmse_mm"),
        ("Position P50 (mm)", "position", "p50_mm"),
        ("Position P90 (mm)", "position", "p90_mm"),
        ("Position P99 (mm)", "position", "p99_mm"),
        ("Rotation Mean (deg)", "rotation", "mean_deg"),
        ("Rotation RMSE (deg)", "rotation", "rmse_deg"),
        ("Rotation Median (deg)", "rotation", "median_deg"),
        ("Rotation P90 (deg)", "rotation", "p90_deg"),
    ]

    for label, section, key in metric_specs:
        row = f"{label:<26}"
        for name in model_names:
            val = results[name].get(section, {}).get(key)
            if val is not None:
                row += f" {val:>{col_width}.3f}"
            else:
                row += f" {'--':>{col_width}}"
        lines.append(row)

    # Optional: jaw angle
    has_jaw = any("jaw_angle" in results[n] for n in model_names)
    if has_jaw:
        row = f"{'Jaw Angle RMSE':<26}"
        for name in model_names:
            val = results[name].get("jaw_angle", {}).get("rmse")
            if val is not None:
                row += f" {val:>{col_width}.4f}"
            else:
                row += f" {'--':>{col_width}}"
        lines.append(row)

    # Optional: uncertainty
    has_ece = any("uncertainty" in results[n] for n in model_names)
    if has_ece:
        lines.append(sep)
        for label, key in [
            ("ECE (position)", "ece"),
            ("AUSE (normalized)", "ause_normalized"),
        ]:
            row = f"{label:<26}"
            for name in model_names:
                val = results[name].get("uncertainty", {}).get(key)
                if val is not None:
                    row += f" {val:>{col_width}.4f}"
                else:
                    row += f" {'--':>{col_width}}"
            lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


# =============================================================================
# Result Formatting -- LaTeX
# =============================================================================


def format_latex_table(
    results: dict[str, dict[str, Any]],
    caption: str = "Pose prediction evaluation results.",
    label: str = "tab:results",
    bold_best: bool = True,
) -> str:
    """Format evaluation results as a LaTeX tabular environment.

    Generates a complete table with position (per-axis and overall RMSE),
    rotation (per-axis Euler and geodesic RMSE), jaw angle RMSE, and ECE
    columns.

    Args:
        results: Mapping from model name to metric dict. Each metric dict
            should contain "position" (with per_axis and rmse_mm),
            "rotation" / "euler" (with roll/pitch/yaw rmse_deg and
            geodesic rmse_deg), optionally "jaw_angle" (rmse), and
            optionally "uncertainty" (ece).
        caption: LaTeX caption text.
        label: LaTeX label for cross-referencing.
        bold_best: If True, bold the best (lowest) value in each column.

    Returns:
        Complete LaTeX table string ready for inclusion in a document.
    """
    model_names = list(results.keys())

    # Collect columns: (col_header, extract_fn)
    columns: list[tuple[str, str]] = [
        ("$x$", "pos_rmse_x"),
        ("$y$", "pos_rmse_y"),
        ("$z$", "pos_rmse_z"),
        ("All", "pos_rmse_all"),
        ("Roll", "rot_rmse_roll"),
        ("Pitch", "rot_rmse_pitch"),
        ("Yaw", "rot_rmse_yaw"),
        ("Geo.", "rot_geodesic"),
        ("Jaw", "jaw_rmse"),
        ("ECE", "ece"),
    ]

    def _extract(model_name: str, col_key: str) -> float | None:
        """Extract a metric value by column key."""
        m = results[model_name]
        if col_key == "pos_rmse_x":
            return m.get("position", {}).get("per_axis", {}).get("X", {}).get("rmse_mm")
        if col_key == "pos_rmse_y":
            return m.get("position", {}).get("per_axis", {}).get("Y", {}).get("rmse_mm")
        if col_key == "pos_rmse_z":
            return m.get("position", {}).get("per_axis", {}).get("Z", {}).get("rmse_mm")
        if col_key == "pos_rmse_all":
            return m.get("position", {}).get("rmse_mm")
        if col_key == "rot_rmse_roll":
            return m.get("euler", {}).get("roll", {}).get("rmse_deg")
        if col_key == "rot_rmse_pitch":
            return m.get("euler", {}).get("pitch", {}).get("rmse_deg")
        if col_key == "rot_rmse_yaw":
            return m.get("euler", {}).get("yaw", {}).get("rmse_deg")
        if col_key == "rot_geodesic":
            return m.get("rotation", {}).get("rmse_deg")
        if col_key == "jaw_rmse":
            return m.get("jaw_angle", {}).get("rmse")
        if col_key == "ece":
            return m.get("uncertainty", {}).get("ece")
        return None

    # Find best (lowest) per column
    best_vals: dict[str, float] = {}
    if bold_best:
        for _, col_key in columns:
            vals = [
                _extract(name, col_key)
                for name in model_names
                if _extract(name, col_key) is not None
            ]
            if vals:
                best_vals[col_key] = min(vals)

    def _fmt(val: float | None, col_key: str, precision: int = 1) -> str:
        """Format a single value, optionally bolding the best."""
        if val is None:
            return "--"
        if col_key in ("jaw_rmse", "ece"):
            s = f"{val:.3f}"
        else:
            s = f"{val:.{precision}f}"
        if bold_best and col_key in best_vals and abs(val - best_vals[col_key]) < 1e-6:
            return f"\\textbf{{{s}}}"
        return s

    # Build header
    n_cols = 1 + len(columns)  # model name + metric columns
    col_spec = "l" + "r" * len(columns)

    lines: list[str] = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Column headers (two groups: position and rotation)
    pos_cols = [h for h, k in columns if k.startswith("pos")]
    rot_cols = [h for h, k in columns if k.startswith("rot")]
    other_cols = [h for h, k in columns if not k.startswith("pos") and not k.startswith("rot")]

    header = "Method"
    for col_header, _ in columns:
        header += f" & {col_header}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for name in model_names:
        row = name
        for _, col_key in columns:
            val = _extract(name, col_key)
            row += " & " + _fmt(val, col_key)
        row += " \\\\"
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def format_latex_row(
    label: str,
    metrics: dict[str, float],
    bold: bool = False,
    jaw_na: bool = False,
    include_ece: bool = True,
) -> str:
    """Format a single LaTeX table row from a flat metrics dictionary.

    Expects a metrics dict with keys: pos_rmse_x_mm, pos_rmse_y_mm,
    pos_rmse_z_mm, pos_rmse_all_mm, rot_rmse_roll_deg, rot_rmse_pitch_deg,
    rot_rmse_yaw_deg, rot_geodesic_deg, jaw_rmse_rad, ece.

    This matches the per-row metric format produced by collecting
    predictions and computing paper-style metrics externally.

    Args:
        label: Row label (e.g., "Kinematic only" or "\\textbf{Full BTPN}").
        metrics: Flat dictionary of metric values.
        bold: If True, wrap all numeric cells in \\textbf{}.
        jaw_na: If True, show "N/A" for jaw RMSE (e.g. for 6DOF dataset).
        include_ece: If True, include the ECE column.

    Returns:
        LaTeX table row string ending with "\\\\".
    """
    def fmt(val: float, prec: int = 1) -> str:
        s = f"{val:.{prec}f}"
        return f"\\textbf{{{s}}}" if bold else s

    jaw = "N/A" if jaw_na else fmt(metrics.get("jaw_rmse_rad", 0.0), 3)
    if bold and not jaw_na:
        jaw = f"\\textbf{{{metrics.get('jaw_rmse_rad', 0.0):.3f}}}"

    parts = [
        label,
        fmt(metrics.get("pos_rmse_x_mm", 0.0)),
        fmt(metrics.get("pos_rmse_y_mm", 0.0)),
        fmt(metrics.get("pos_rmse_z_mm", 0.0)),
        fmt(metrics.get("pos_rmse_all_mm", 0.0)),
        fmt(metrics.get("rot_rmse_roll_deg", 0.0)),
        fmt(metrics.get("rot_rmse_pitch_deg", 0.0)),
        fmt(metrics.get("rot_rmse_yaw_deg", 0.0)),
        fmt(metrics.get("rot_geodesic_deg", 0.0)),
        jaw,
    ]
    if include_ece:
        parts.append(fmt(metrics.get("ece", 0.0), 3))

    return " & ".join(parts) + " \\\\"
