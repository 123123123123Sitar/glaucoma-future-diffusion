"""Image similarity metrics."""

from __future__ import annotations

import math

import numpy as np


def psnr(reference: np.ndarray, prediction: np.ndarray, max_value: float = 255.0) -> float:
    mse = float(np.mean((reference.astype(float) - prediction.astype(float)) ** 2))
    if mse == 0:
        return math.inf
    return 20 * math.log10(max_value / math.sqrt(mse))


def simple_ssim(reference: np.ndarray, prediction: np.ndarray, max_value: float = 255.0) -> float:
    x = reference.astype(float).ravel()
    y = prediction.astype(float).ravel()
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))
