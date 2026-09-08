from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.training.harvard_gdp_trainer import (
    _platt_parameters,
    _stratified_calibration_indices,
)


def test_harvard_calibration_split_preserves_both_classes() -> None:
    labels = np.asarray([0] * 20 + [1] * 5)
    development, calibration = _stratified_calibration_indices(labels, 0.2, 7)
    assert set(labels[development]) == {0, 1}
    assert set(labels[calibration]) == {0, 1}
    assert not set(development) & set(calibration)


def test_platt_calibration_can_correct_base_rate_shift() -> None:
    logits = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    labels = np.asarray([0, 0, 0, 0, 1])
    slope, bias = _platt_parameters(logits, labels)
    assert slope > 0
    assert bias < 0
