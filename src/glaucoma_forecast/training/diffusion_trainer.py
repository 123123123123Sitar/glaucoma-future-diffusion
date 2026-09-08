"""Diffusion trainer smoke path."""

from __future__ import annotations

from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.utils.reproducibility import select_device


def run_one_batch_smoke(device: str = "auto") -> dict[str, float | str]:
    torch = require_torch()
    selected = select_device(device)
    module = torch.nn.Conv2d(3, 3, 3, padding=1).to(selected)
    opt = torch.optim.AdamW(module.parameters(), lr=1e-4)
    x = torch.rand(1, 3, 32, 32, device=selected)
    y = module(x).mean()
    y.backward()
    opt.step()
    return {"device": selected, "loss": float(y.detach().cpu())}
