#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from build_sigf_locked_review_pack import papila_style_contour_mask, tensor
from glaucoma_forecast.data.highres import (
    estimate_rnfl_contrast_map,
    estimate_vessel_map,
    preprocess_high_resolution,
)
from glaucoma_forecast.data.preprocessing import match_color_statistics, mirror_left_eye
from glaucoma_forecast.evaluation.anatomy_metrics import vcdr_from_masks
from glaucoma_forecast.evaluation.image_metrics import simple_ssim
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter
from glaucoma_forecast.utils.reproducibility import select_device


def automatic_cup(segmenter, image: Image.Image, reference: Image.Image, device: str):
    torch = require_torch()
    crop = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    normalized = match_color_statistics(crop, reference.resize(crop.size))
    with torch.no_grad():
        original = torch.sigmoid(segmenter(tensor(crop, device)))[0]
        matched = torch.sigmoid(segmenter(tensor(normalized, device)))[0]
        probability = ((original + matched) / 2.0).cpu().numpy()
    disc = papila_style_contour_mask(probability[0] >= 0.5)
    cup = papila_style_contour_mask((probability[1] >= 0.5) & disc) & disc
    vcdr = vcdr_from_masks(cup, disc) if cup.any() and disc.any() else float("nan")
    return cup, vcdr


def dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(left.sum() + right.sum())
    return 1.0 if denominator == 0 else 2.0 * float((left & right).sum()) / denominator


def correlation(left: Image.Image, right: Image.Image) -> float:
    left_values = np.asarray(left, dtype=np.float32).ravel()
    right_values = np.asarray(right, dtype=np.float32).ravel()
    if float(left_values.std()) < 1e-6 or float(right_values.std()) < 1e-6:
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def split_summary(pairs, split: str) -> dict[str, object]:
    frame = pairs[pairs["split"].eq(split)].copy()
    transitions = (
        frame["baseline_glaucoma_label"].astype(int).astype(str)
        + "->"
        + frame["future_glaucoma_label"].astype(int).astype(str)
    )
    return {
        "pairs": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "eyes": int(frame["eye_id"].nunique()),
        "mean_horizon_years": float(frame["horizon_years"].mean()),
        "median_horizon_years": float(frame["horizon_years"].median()),
        "conversion_pairs": int((transitions == "0->1").sum()),
        "conversion_percent": float(100.0 * (transitions == "0->1").mean()),
        "glaucoma_present_future_percent": float(
            100.0 * frame["future_glaucoma_label"].astype(int).mean()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute two-eye manuscript metrics.")
    parser.add_argument(
        "--review-dir",
        default="outputs/review/sigf_multitask_papila_contours_20260816",
    )
    parser.add_argument("--case-ids", nargs="+", default=["C02", "C04"])
    parser.add_argument(
        "--segmenter",
        default="outputs/training/papila_disc_cup_segmenter_v3_weighted/best.pt",
    )
    parser.add_argument("--pairs", default="data/sigf/pairs_5y_leakage_safe.csv")
    parser.add_argument(
        "--output-dir", default="outputs/manuscripts/amia_hssp_sigf_20260816"
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    pd = __import__("pandas")
    torch = require_torch()
    device = select_device(args.device)
    state = torch.load(args.segmenter, map_location=device, weights_only=False)
    segmenter = build_disc_cup_segmenter(
        int(state.get("config", {}).get("base_channels", 24))
    ).to(device)
    segmenter.load_state_dict(state["model"])
    segmenter.eval()

    review = Path(args.review_dir)
    metadata = {
        row["case_id"]: row
        for row in json.loads((review / "case_metadata.json").read_text())
    }
    rows = []
    for case_id in args.case_ids:
        case = metadata[case_id]
        laterality = str(case["eye_id"]).rsplit("_", 1)[-1]
        baseline = preprocess_high_resolution(
            case["baseline_image_path"], 256, crop_fraction=0.36
        ).optic_disc
        baseline, _ = mirror_left_eye(baseline, laterality)
        for target_year, followup in zip((1, 3, 5), case["actual_followups"]):
            actual = preprocess_high_resolution(
                followup["image_path"], 256, crop_fraction=0.36
            ).optic_disc
            actual, _ = mirror_left_eye(actual, laterality)
            generated = Image.open(
                review / case_id / f"generated_year_{target_year}.png"
            ).convert("RGB")
            actual_cup, actual_vcdr = automatic_cup(
                segmenter, actual, baseline, device
            )
            generated_cup, generated_vcdr = automatic_cup(
                segmenter, generated, baseline, device
            )
            rows.append(
                {
                    "case_id": case_id,
                    "target_year": target_year,
                    "observed_followup_year": float(followup["years"]),
                    "cup_dice": dice(actual_cup, generated_cup),
                    "real_vcdr": actual_vcdr,
                    "generated_vcdr": generated_vcdr,
                    "vcdr_absolute_error": abs(actual_vcdr - generated_vcdr),
                    "vessel_map_correlation": correlation(
                        estimate_vessel_map(actual), estimate_vessel_map(generated)
                    ),
                    "rnfl_contrast_correlation": correlation(
                        estimate_rnfl_contrast_map(actual),
                        estimate_rnfl_contrast_map(generated),
                    ),
                    "image_ssim": simple_ssim(
                        np.asarray(actual, dtype=np.float32),
                        np.asarray(generated, dtype=np.float32),
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    by_year = (
        frame.groupby("target_year")
        .agg(
            eyes=("case_id", "count"),
            mean_cup_dice=("cup_dice", "mean"),
            mean_real_vcdr=("real_vcdr", "mean"),
            mean_generated_vcdr=("generated_vcdr", "mean"),
            mean_vcdr_absolute_error=("vcdr_absolute_error", "mean"),
            mean_vessel_map_correlation=("vessel_map_correlation", "mean"),
            mean_rnfl_contrast_correlation=("rnfl_contrast_correlation", "mean"),
            mean_image_ssim=("image_ssim", "mean"),
        )
        .reset_index()
    )
    pairs = pd.read_csv(args.pairs)
    report = {
        "metric_scope": (
            "Two displayed held-out eyes; PAPILA-style automatic contours, not expert SIGF annotations"
        ),
        "training_split": split_summary(pairs, "train"),
        "test_split": split_summary(pairs, "test"),
        "by_year": by_year.to_dict("records"),
        "overall": {
            "mean_cup_dice": float(frame["cup_dice"].mean()),
            "mean_vcdr_absolute_error": float(frame["vcdr_absolute_error"].mean()),
            "mean_vessel_map_correlation": float(
                frame["vessel_map_correlation"].mean()
            ),
            "mean_rnfl_contrast_correlation": float(
                frame["rnfl_contrast_correlation"].mean()
            ),
            "mean_image_ssim": float(frame["image_ssim"].mean()),
        },
        "definitions": {
            "cup_dice": "Overlap of real and generated automatic cup masks; 1.0 is perfect overlap",
            "vcdr": "Vertical cup height divided by vertical disc height",
            "vessel_map_correlation": "Similarity of automatic vessel-contrast maps; 1.0 is perfect linear agreement",
            "rnfl_contrast_correlation": "Similarity of broad red-free RNFL-contrast maps; 1.0 is perfect linear agreement",
            "image_ssim": "Structural image similarity; 1.0 is identical",
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "two_eye_per_image_metrics.csv", index=False)
    by_year.to_csv(output / "two_eye_by_year_metrics.csv", index=False)
    (output / "metrics_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
