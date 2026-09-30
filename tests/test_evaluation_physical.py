"""Check Dataset A metrics against the saved retrain predictions."""

from pathlib import Path

import numpy as np
import pytest

from btpn.dataset import NormalizationStats
from scripts.evaluate import compute_per_tool_metrics, evaluate_from_npz


PREDICTIONS = Path(__file__).resolve().parents[1] / "results/evaluation_data_vmf_fix.npz"


def test_retrain_rotation_metrics_use_physical_quaternions() -> None:
    with np.load(PREDICTIONS, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    stats = NormalizationStats(mean=arrays["mean"], std=arrays["std"])
    metrics = compute_per_tool_metrics(arrays, stats)

    assert metrics["rotation"]["rmse_deg"] == pytest.approx(8.852, abs=0.01)
    assert metrics["uncertainty"]["rot_ece_fisher"] == pytest.approx(0.255, abs=0.001)
    assert metrics["uncertainty"]["rot_ece_vmf"] == pytest.approx(0.290, abs=0.001)
    assert metrics["position"]["rmse_mm"] == pytest.approx(9.030, abs=0.01)


def test_retrain_table_is_reproduced_from_saved_predictions(tmp_path: Path) -> None:
    result = evaluate_from_npz(PREDICTIONS, tmp_path / "unused.npz", tmp_path)

    assert result["geo"] == pytest.approx(8.852, abs=0.01)
    assert result["jaw_pct"] == pytest.approx(14.720, abs=0.01)
    assert result["rot_ece_fisher"] == pytest.approx(0.255, abs=0.001)
