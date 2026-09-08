"""Auxiliary losses from a frozen expert-contour disc/cup segmenter."""

from __future__ import annotations

from glaucoma_forecast.models._torch import require_torch


def tight_disc_crop(images, centers, source_view: str, output_size: int = 256):
    """Extract the 30%-field crop used to train the PAPILA segmenter."""

    torch = require_torch()
    if source_view == "optic_disc":
        crop_fraction = 0.60
        crop_centers = images.new_full((images.shape[0], 2), 0.5)
    elif source_view == "full_field":
        crop_fraction = 0.30
        crop_centers = centers
    else:
        raise ValueError("source_view must be full_field or optic_disc")
    crops = []
    height, width = images.shape[-2:]
    side = max(4, int(round(min(height, width) * crop_fraction)))
    for image, center in zip(images, crop_centers):
        center_x = int(round(float(center[0].detach().cpu()) * width))
        center_y = int(round(float(center[1].detach().cpu()) * height))
        left = min(max(0, center_x - side // 2), width - side)
        top = min(max(0, center_y - side // 2), height - side)
        crops.append(image[:, top : top + side, left : left + side])
    return torch.nn.functional.interpolate(
        torch.stack(crops),
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )


def frozen_disc_cup_loss(segmenter, prediction, target, centers, source_view: str):
    """Match target cup/rim structure only when target masks are plausible."""

    torch = require_torch()
    prediction_crop = tight_disc_crop(prediction, centers, source_view)
    target_crop = tight_disc_crop(target, centers, source_view)
    prediction_probability = torch.sigmoid(segmenter(prediction_crop))
    with torch.no_grad():
        target_probability = torch.sigmoid(segmenter(target_crop))
        disc_area = target_probability[:, 0].mean(dim=(1, 2))
        cup_area = target_probability[:, 1].mean(dim=(1, 2))
        plausible = (
            (disc_area >= 0.08)
            & (disc_area <= 0.75)
            & (cup_area >= 0.01)
            & (cup_area < disc_area)
        )
    if not plausible.any():
        return prediction.sum() * 0.0, prediction.sum() * 0.0, plausible
    probability_loss = torch.nn.functional.smooth_l1_loss(
        prediction_probability[plausible], target_probability[plausible]
    )
    prediction_ratio = prediction_probability[:, 1].mean(dim=(1, 2)) / prediction_probability[:, 0].mean(dim=(1, 2)).clamp_min(1e-4)
    target_ratio = target_probability[:, 1].mean(dim=(1, 2)) / target_probability[:, 0].mean(dim=(1, 2)).clamp_min(1e-4)
    ratio_loss = torch.nn.functional.smooth_l1_loss(
        prediction_ratio[plausible], target_ratio[plausible]
    )
    return probability_loss, ratio_loss, plausible


def _soft_dice_loss(probability, target):
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))


def _sector_rim_areas(masks):
    """Return differentiable inferior/superior/nasal/temporal rim areas."""

    torch = require_torch()
    height, width = masks.shape[-2:]
    y = torch.arange(height, device=masks.device)[None, :, None]
    x = torch.arange(width, device=masks.device)[None, None, :]
    vertical = (y - (height - 1) / 2).abs() >= (x - (width - 1) / 2).abs()
    inferior = vertical & (y >= (height - 1) / 2)
    superior = vertical & ~inferior
    horizontal = ~vertical
    nasal = horizontal & (x < (width - 1) / 2)
    temporal = horizontal & ~nasal
    rim = (masks[:, 0] - masks[:, 1]).clamp_min(0.0)
    values = []
    for sector in (inferior, superior, nasal, temporal):
        sector = sector.expand(rim.shape[0], -1, -1)
        values.append((rim * sector).sum(dim=(1, 2)) / sector.sum().clamp_min(1))
    return torch.stack(values, dim=1)


def annotated_disc_cup_loss(
    segmenter,
    prediction,
    centers,
    source_view: str,
    baseline_masks,
    future_masks,
    annotation_available,
    stable_labels,
):
    """Match human contours while preserving the outer disc and stable controls."""

    torch = require_torch()
    if not annotation_available.any():
        zero = prediction.sum() * 0.0
        return zero, zero, zero, zero
    prediction_crop = tight_disc_crop(prediction, centers, source_view)
    probability = torch.sigmoid(segmenter(prediction_crop))
    target_size = probability.shape[-2:]
    baseline_masks = torch.nn.functional.interpolate(
        baseline_masks, size=target_size, mode="nearest"
    )
    future_masks = torch.nn.functional.interpolate(
        future_masks, size=target_size, mode="nearest"
    )
    available = annotation_available.bool()
    anatomy_loss = _soft_dice_loss(probability[available], future_masks[available])
    anatomy_loss = (anatomy_loss[:, 0] + 2.0 * anatomy_loss[:, 1]).mean() / 3.0
    prediction_ratio = probability[:, 1].mean(dim=(1, 2)) / probability[:, 0].mean(
        dim=(1, 2)
    ).clamp_min(1e-4)
    target_ratio = future_masks[:, 1].mean(dim=(1, 2)) / future_masks[:, 0].mean(
        dim=(1, 2)
    ).clamp_min(1e-4)
    ratio_loss = torch.nn.functional.smooth_l1_loss(
        prediction_ratio[available], target_ratio[available]
    )
    outer_disc_loss = torch.nn.functional.smooth_l1_loss(
        probability[available, 0], baseline_masks[available, 0]
    )
    predicted_sectors = _sector_rim_areas(probability)
    target_sectors = _sector_rim_areas(future_masks)
    sector_loss = torch.nn.functional.smooth_l1_loss(
        predicted_sectors[available], target_sectors[available]
    )
    stable = available & stable_labels.bool()
    if stable.any():
        stability_loss = torch.nn.functional.smooth_l1_loss(
            probability[stable], baseline_masks[stable]
        )
    else:
        stability_loss = prediction.sum() * 0.0
    return anatomy_loss + sector_loss, ratio_loss, outer_disc_loss, stability_loss
