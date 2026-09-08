#!/usr/bin/env python3
"""Audit one generated sequence for measured cup change and identity preservation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.preprocessing import optic_disc_crop
from glaucoma_forecast.evaluation.anatomy_metrics import vcdr_from_masks
from glaucoma_forecast.evaluation.anatomy_release_gate import evaluate_anatomy_release
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter
from glaucoma_forecast.utils.reproducibility import select_device


def _tensor(image: Image.Image, device: str):
    torch = require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0).to(device)


def _segment(model, image: Image.Image, device: str):
    torch = require_torch()
    with torch.no_grad():
        probability = torch.sigmoid(model(_tensor(image, device)))[0].cpu().numpy()
    disc = probability[0] >= 0.5
    cup = (probability[1] >= 0.5) & disc
    return disc, cup


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(left.sum() + right.sum())
    return float(2 * np.logical_and(left, right).sum() / denominator) if denominator else 0.0


def _edge_correlation(left: Image.Image, right: Image.Image) -> float:
    values = []
    for image in (left, right):
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        gy, gx = np.gradient(gray)
        values.append(np.sqrt(gx * gx + gy * gy).ravel())
    if values[0].std() < 1e-6 or values[1].std() < 1e-6:
        return 0.0
    return float(np.corrcoef(values[0], values[1])[0, 1])


def _mask_vcdr(path: str) -> float:
    with Image.open(path) as source:
        return np.asarray(source.convert("L")) > 127


def _overlay(image: Image.Image, disc: np.ndarray, cup: np.ndarray) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    for mask, color in ((disc, (71, 215, 255)), (cup, (255, 209, 102))):
        edge = mask ^ (
            np.roll(mask, 1, 0)
            & np.roll(mask, -1, 0)
            & np.roll(mask, 1, 1)
            & np.roll(mask, -1, 1)
        )
        for y, x in np.argwhere(edge):
            draw.point((int(x), int(y)), fill=color)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--segmenter-checkpoint", required=True)
    parser.add_argument("--segmenter-report", required=True)
    parser.add_argument("--baseline-disc-mask", required=True)
    parser.add_argument("--baseline-cup-mask", required=True)
    parser.add_argument("--future-disc-mask", required=True)
    parser.add_argument("--future-cup-mask", required=True)
    parser.add_argument("--repeat-annotation-error", type=float, required=True)
    parser.add_argument("--stable-controls-pass", action="store_true")
    parser.add_argument("--human-accepted", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch = require_torch()
    device = select_device(args.device)
    report = json.loads(Path(args.report).read_text())
    checkpoint = torch.load(args.segmenter_checkpoint, map_location=device, weights_only=False)
    model = build_disc_cup_segmenter(int(checkpoint.get("config", {}).get("base_channels", 24))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    center = tuple(float(value) for value in report["disc_center_normalized"])
    baseline_path = report["input_artifacts"]["full_field"]
    image_size = Image.open(baseline_path).size[0]
    pixel_center = (round(center[0] * image_size), round(center[1] * image_size))
    paths = [baseline_path, *report["representative_images"]]
    crops = []
    masks = []
    for path in paths:
        with Image.open(path) as source:
            crop = optic_disc_crop(source.convert("RGB"), pixel_center, 0.30, 256)
        crops.append(crop)
        masks.append(_segment(model, crop, device))
    vcdr = [vcdr_from_masks(cup, disc) for disc, cup in masks]
    observed_baseline = vcdr_from_masks(
        _mask_vcdr(args.baseline_cup_mask), _mask_vcdr(args.baseline_disc_mask)
    )
    observed_future = vcdr_from_masks(
        _mask_vcdr(args.future_cup_mask), _mask_vcdr(args.future_disc_mask)
    )
    segmenter_report = json.loads(Path(args.segmenter_report).read_text())
    last = segmenter_report.get("history", [{}])[-1]
    disc_dice = last.get("grape_validation_disc_dice") or last.get("validation_disc_dice", 0.0)
    cup_dice = last.get("grape_validation_cup_dice") or last.get("validation_cup_dice", 0.0)
    annotation_error = float(args.repeat_annotation_error)
    final_disc_dice = _dice(masks[0][0], masks[-1][0])
    vessel_correlation = _edge_correlation(crops[0], crops[-1])
    outside_disc = ~masks[0][0]
    baseline_gray = np.asarray(crops[0].convert("L"), dtype=np.float32) / 255.0
    final_gray = np.asarray(crops[-1].convert("L"), dtype=np.float32) / 255.0
    outside_change = float(np.abs(final_gray - baseline_gray)[outside_disc].mean())
    audit = {
        "segmenter_disc_dice": float(disc_dice),
        "segmenter_cup_dice": float(cup_dice),
        "repeat_annotation_error": annotation_error,
        "observed_vcdr_change": observed_future - observed_baseline,
        "generated_vcdr_change_5y": vcdr[-1] - vcdr[0],
        "generated_vcdr_by_year": dict(zip([0.0, *report["requested_years"]], vcdr)),
        "monotonic_within_error": bool(np.all(np.diff(vcdr) >= -annotation_error)),
        "outer_disc_dice_baseline_to_5y": final_disc_dice,
        "outer_disc_preserved": final_disc_dice >= 0.95,
        "vessel_edge_correlation": vessel_correlation,
        "vessel_topology_preserved": vessel_correlation >= 0.90,
        "outside_disc_mean_absolute_change": outside_change,
        "photometric_artifact_rejected": outside_change <= 0.03,
        "stable_controls_pass": args.stable_controls_pass,
        "human_accepted": args.human_accepted,
    }
    audit["release_gate"] = evaluate_anatomy_release(audit)
    output = Path(args.report).parent / "anatomy_audit"
    output.mkdir(exist_ok=True)
    for index, (crop, (disc, cup)) in enumerate(zip(crops, masks)):
        _overlay(crop, disc, cup).save(output / f"contours_{index:02d}.png")
    (output / "report.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
