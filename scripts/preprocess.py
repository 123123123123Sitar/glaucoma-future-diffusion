#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.preprocessing import preprocess_image, save_registration_overlay
from glaucoma_forecast.data.schema import read_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess fundus images from a common manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/preprocessed")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    df = read_manifest(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_by_eye = {}
    for _, row in df.iterrows():
        result = preprocess_image(row["image_path"], args.image_size, row.get("laterality"))
        rel = f"{row['eye_id']}_{row['visit_id']}.png".replace("/", "_")
        output_path = out / "images" / rel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(output_path)
        if row["eye_id"] not in baseline_by_eye:
            baseline_by_eye[row["eye_id"]] = result.image
        overlay_path = out / "overlays" / rel
        save_registration_overlay(baseline_by_eye[row["eye_id"]], result.image, overlay_path)
        rows.append(
            {
                "patient_id": row["patient_id"],
                "eye_id": row["eye_id"],
                "visit_id": row["visit_id"],
                "preprocessed_path": str(output_path),
                "overlay_path": str(overlay_path),
                "fov_box": result.fov_box,
                "mirrored": result.mirrored,
            }
        )
    pd.DataFrame(rows).to_csv(out / "preprocessed_manifest.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"processed": len(rows)}, indent=2), encoding="utf-8")
    print(f"Preprocessed {len(rows)} images into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
