"""Anatomy metric placeholders with executable VCDR helpers."""

from __future__ import annotations

import numpy as np


def vertical_extent(mask: np.ndarray) -> int:
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return 0
    return int(ys.max() - ys.min() + 1)


def vcdr_from_masks(cup_mask: np.ndarray, disc_mask: np.ndarray) -> float:
    disc = vertical_extent(disc_mask)
    if disc == 0:
        raise ValueError("Cannot compute VCDR with empty disc mask")
    return vertical_extent(cup_mask) / disc
