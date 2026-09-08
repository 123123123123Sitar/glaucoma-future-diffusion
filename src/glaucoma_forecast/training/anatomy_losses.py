"""Differentiable optic-disc sector losses for longitudinal training."""

from __future__ import annotations

from glaucoma_forecast.models._torch import require_torch


ISNT_SECTORS = ("inferior", "superior", "nasal", "temporal")


def disc_sector_masks(
    height: int,
    width: int,
    centers,
    lateralities: list[str],
    inner_radius: float = 0.06,
    outer_radius: float = 0.30,
):
    """Return I/S/N/T annular attention masks around each optic disc."""

    torch = require_torch()
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape [batch, 2]")
    if len(lateralities) != centers.shape[0]:
        raise ValueError("lateralities must match the batch size")
    if not 0 <= inner_radius < outer_radius:
        raise ValueError("Require 0 <= inner_radius < outer_radius")
    y_axis = torch.linspace(0.0, 1.0, height, device=centers.device)
    x_axis = torch.linspace(0.0, 1.0, width, device=centers.device)
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
    dx = xx[None] - centers[:, 0, None, None]
    dy = yy[None] - centers[:, 1, None, None]
    scale = float(min(height, width))
    radius = torch.sqrt((dx * width / scale) ** 2 + (dy * height / scale) ** 2)
    annulus = (radius >= inner_radius) & (radius <= outer_radius)
    vertical = dy.abs() >= dx.abs()
    inferior = annulus & vertical & (dy >= 0)
    superior = annulus & vertical & (dy < 0)
    nasal_masks = []
    temporal_masks = []
    horizontal = annulus & ~vertical
    for batch_index, laterality in enumerate(lateralities):
        if laterality not in {"OD", "OS"}:
            raise ValueError(f"Unsupported laterality: {laterality}")
        temporal_positive_x = laterality == "OD"
        temporal = horizontal[batch_index] & (
            dx[batch_index] >= 0 if temporal_positive_x else dx[batch_index] < 0
        )
        nasal = horizontal[batch_index] & ~temporal
        temporal_masks.append(temporal)
        nasal_masks.append(nasal)
    return torch.stack(
        [
            inferior,
            superior,
            torch.stack(nasal_masks),
            torch.stack(temporal_masks),
        ],
        dim=1,
    ).float()


def rnfl_decline_weights(baseline, future, available, emphasis: float = 3.0):
    """Convert observed quadrant RNFL loss into normalized sector weights."""

    torch = require_torch()
    if baseline.shape != future.shape or baseline.ndim != 2 or baseline.shape[1] != 4:
        raise ValueError("RNFL tensors must both have shape [batch, 4]")
    finite = torch.isfinite(baseline).all(dim=1) & torch.isfinite(future).all(dim=1)
    valid = available.bool() & finite
    baseline_safe = torch.where(torch.isfinite(baseline), baseline, torch.ones_like(baseline))
    future_safe = torch.where(torch.isfinite(future), future, baseline_safe)
    proportional_decline = ((baseline_safe - future_safe) / baseline_safe.clamp_min(1.0)).clamp_min(0.0)
    maximum = proportional_decline.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    normalized = proportional_decline / maximum
    weights = 1.0 + float(emphasis) * normalized
    return torch.where(valid[:, None], weights, torch.ones_like(weights)), valid


def sector_weighted_residual_loss(prediction, target, masks, weights, valid):
    """Measure residual reconstruction error by anatomy sector."""

    if not valid.any():
        return prediction.sum() * 0.0
    pixel_error = (prediction - target).abs().mean(dim=1, keepdim=True)
    denominators = masks.sum(dim=(2, 3)).clamp_min(1.0)
    sector_error = (pixel_error * masks).sum(dim=(2, 3)) / denominators
    weighted = (sector_error * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return weighted[valid].mean()
