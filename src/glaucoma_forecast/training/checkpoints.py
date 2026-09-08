"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

from glaucoma_forecast.models._torch import require_torch


def save_checkpoint(path: str | Path, model, optimizer=None, step: int = 0, extra: dict | None = None) -> None:
    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, map_location: str = "cpu") -> int:
    torch = require_torch()
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))
