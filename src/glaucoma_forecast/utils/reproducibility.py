"""Determinism and backend device helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def select_device(requested: str = "auto") -> str:
    """Select CUDA, MPS, or CPU without assuming PyTorch is installed."""

    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def mixed_precision_enabled(device: str, requested: Any = "auto") -> bool:
    if requested == "auto":
        return device == "cuda"
    return bool(requested)
