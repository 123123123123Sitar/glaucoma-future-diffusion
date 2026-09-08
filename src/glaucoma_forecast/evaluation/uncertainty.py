"""Uncertainty summarization."""

from __future__ import annotations

import numpy as np


def summarize_samples(samples: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(samples)
    if arr.ndim < 1 or arr.shape[0] < 1:
        raise ValueError("Expected at least one sample")
    return {
        "mean": arr.mean(axis=0),
        "median": np.median(arr, axis=0),
        "p05": np.percentile(arr, 5, axis=0),
        "p95": np.percentile(arr, 95, axis=0),
        "std": arr.std(axis=0),
    }
