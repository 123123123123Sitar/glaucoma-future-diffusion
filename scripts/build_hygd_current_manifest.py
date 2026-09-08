#!/usr/bin/env python3
"""Build a cross-sectional current-glaucoma manifest from verified HYGD data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    labels = pd.read_csv(root / "Labels.csv")
    labels.columns = [str(column).strip() for column in labels.columns]
    required = {"Image Name", "Patient", "Label", "Quality Score"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"HYGD labels missing columns: {missing}")
    label_map = {"GON-": 0, "GON+": 1}
    mapped = labels["Label"].map(label_map)
    if mapped.isna().any():
        raise ValueError(f"Unexpected HYGD labels: {sorted(labels['Label'].unique())}")
    frame = pd.DataFrame(
        {
            "patient_id": "HYGD_" + labels["Patient"].astype(str),
            # Laterality is unavailable. Each image receives a unique eye key
            # while patient_id still enforces leakage-safe grouping.
            "eye_id": "HYGD_image_" + labels["Image Name"].map(lambda name: Path(name).stem),
            "image_path": labels["Image Name"].map(lambda name: str(root / "Images" / name)),
            "event_observed": 0,
            "event_or_censor_time_years": 0.0,
            "incidence_eligible": 0,
            "current_glaucoma_label": mapped.astype(int),
            "quality_score": pd.to_numeric(labels["Quality Score"], errors="coerce"),
            "source_dataset": "HYGD_1.1.0",
            "camera_device": "TOPCON_DRI_OCT_Triton_45deg",
        }
    )
    missing_images = [path for path in frame["image_path"] if not Path(path).exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing HYGD images: {missing_images[:5]}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(
        {
            "output": str(output),
            "images": len(frame),
            "patients": int(frame["patient_id"].nunique()),
            "positive": int(frame["current_glaucoma_label"].sum()),
            "negative": int((frame["current_glaucoma_label"] == 0).sum()),
            "incidence_eligible": 0,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
