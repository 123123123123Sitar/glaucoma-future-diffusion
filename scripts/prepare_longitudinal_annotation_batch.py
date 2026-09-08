#!/usr/bin/env python3
"""Prepare same-eye long-horizon optic-disc pairs for contour annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.highres import preprocess_high_resolution
from glaucoma_forecast.data.longitudinal_annotations import longest_longitudinal_pairs
from glaucoma_forecast.data.preprocessing import match_color_statistics, optic_disc_crop
from glaucoma_forecast.data.registration import apply_translation, estimate_translation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-manifest", default="data/derived/balanced_stability_pairs_v2_isnt.csv"
    )
    parser.add_argument("--output-dir", default="outputs/annotations/grape_longitudinal_v1")
    parser.add_argument("--minimum-years", type=float, default=3.0)
    parser.add_argument("--crop-fraction", type=float, default=0.30)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--repeat-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output = Path(args.output_dir)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    pairs = longest_longitudinal_pairs(
        pd.read_csv(args.pair_manifest), args.minimum_years
    )
    records = []
    for index, row in pairs.iterrows():
        baseline = preprocess_high_resolution(str(row["baseline_image_path"]), args.image_size)
        future = preprocess_high_resolution(str(row["future_image_path"]), args.image_size)
        shift_x, shift_y, confidence = estimate_translation(
            baseline.full_field,
            future.full_field,
            max_shift=max(4, args.image_size // 12),
        )
        # Review unaltered native-resolution crops. Registration and color
        # matching belong to model training, not to the human visual source.
        baseline_crop = baseline.optic_disc
        future_crop = future.optic_disc
        annotation_id = f"{row['eye_id']}_{float(row['horizon_years']):.3f}y"
        baseline_asset = assets / f"{annotation_id}_baseline.jpg"
        future_asset = assets / f"{annotation_id}_future.jpg"
        baseline_crop.save(baseline_asset, quality=95)
        future_crop.save(future_asset, quality=95)
        records.append(
            {
                "annotation_id": annotation_id,
                "patient_id": str(row["patient_id"]),
                "eye_id": str(row["eye_id"]),
                "laterality": str(row["laterality"]),
                "baseline_visit_id": str(row.get("baseline_visit_id", "")),
                "future_visit_id": str(row.get("future_visit_id", "")),
                "baseline_image_path": str(row["baseline_image_path"]),
                "future_image_path": str(row["future_image_path"]),
                "baseline_asset": baseline_asset.relative_to(output).as_posix(),
                "future_asset": future_asset.relative_to(output).as_posix(),
                "horizon_years": float(row["horizon_years"]),
                "split": str(row["split"]),
                "registration_confidence": float(confidence),
                "repeat_group": "",
            }
        )

    repeat_count = max(1, round(len(records) * args.repeat_fraction))
    repeats = pairs.sample(n=repeat_count, random_state=args.seed).index.tolist()
    for repeat_number, source_index in enumerate(repeats, start=1):
        duplicate = dict(records[source_index])
        duplicate["annotation_id"] = f"{duplicate['annotation_id']}_repeat{repeat_number}"
        duplicate["repeat_group"] = records[source_index]["annotation_id"]
        records.append(duplicate)

    payload = {
        "schema_version": "1.0",
        "research_only": True,
        "instructions": (
            "Trace the outer disc and pale cup on both images. Label observed structural "
            "change; do not use brightness or framing differences as progression."
        ),
        "project_root": str(root),
        "records": records,
    }
    (output / "batch.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(records), "unique": len(pairs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
