from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_residual_diffusion_accepts_progression_condition() -> None:
    import torch

    from glaucoma_forecast.models.residual_diffusion import build_residual_diffusion

    model = build_residual_diffusion(base_channels=8, clinical_dim=1)
    residual = torch.rand(2, 3, 32, 32)
    baseline = torch.rand(2, 3, 32, 32)
    vessel = torch.rand(2, 1, 32, 32)
    output = model(
        residual,
        baseline,
        vessel,
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.25, 0.25]),
        torch.tensor([[0.2], [0.8]]),
    )
    assert output.shape == residual.shape
