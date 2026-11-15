"""Do not hide missing outputs by evaluating only the successfully traced samples."""
import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks.wsp_pilot import signal_metrics


def test_completely_failed_prediction_remains_in_snr():
    truth = np.array([[1, -1, 1, -1]], dtype=float)
    mask = np.ones_like(truth, bool)
    failed = signal_metrics(truth, np.full_like(truth, np.nan), mask)
    assert failed["snr_db"] == pytest.approx(0)
    assert failed["rmse_mv"] == pytest.approx(1)
    assert failed["coverage"] == 0
    assert failed["missing_samples"] == 4


def test_partial_prediction_is_penalized_without_dropping_missing_samples():
    truth = np.array([[1, 1, 1, 1]], dtype=float)
    mask = np.ones_like(truth, bool)
    partial = signal_metrics(truth, np.array([[1, 1, np.nan, np.nan]]), mask)
    assert partial["snr_db"] == pytest.approx(10 * np.log10(2))
    assert partial["coverage"] == 0.5
    assert partial["missing_samples"] == 2
