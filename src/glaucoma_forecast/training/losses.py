"""Loss builders."""

from __future__ import annotations

from glaucoma_forecast.models._torch import require_torch


def diffusion_loss(prediction, target, objective: str = "epsilon"):
    torch = require_torch()
    if objective not in {"epsilon", "velocity"}:
        raise ValueError(f"Unsupported diffusion objective: {objective}")
    return torch.nn.functional.mse_loss(prediction, target)
