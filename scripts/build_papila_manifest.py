#!/usr/bin/env python3
"""Build a labeled image manifest for the official PAPILA release."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DIAGNOSIS_NAMES = {0: "healthy", 1: "glaucoma", 2: "suspect"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for side in ("od", "os"):
        table = pd.read_excel(args.root / "ClinicalData" / f"patient_data_{side}.xlsx")
        table["diagnosis_code"] = pd.to_numeric(table["Diagnosis"], errors="coerce")
        table = table[table["diagnosis_code"].notna()].copy()

        for _, row in table.iterrows():
            patient_id = str(row["Unnamed: 0"]).replace("#", "")
            diagnosis_code = int(row["diagnosis_code"])
            image_id = f"RET{patient_id}{side.upper()}"
            image_path = args.root / "FundusImages" / f"{image_id}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing PAPILA image: {image_path}")
            records.append(
                {
                    "source_dataset": "PAPILA_v1.1",
                    "patient_id": patient_id,
                    "eye_id": image_id,
                    "laterality": side.upper(),
                    "image_path": str(image_path),
                    "diagnosis_code": diagnosis_code,
                    "diagnosis": DIAGNOSIS_NAMES[diagnosis_code],
                    "longitudinal": False,
                }
            )

    manifest = pd.DataFrame(records).sort_values(["patient_id", "laterality"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(manifest["diagnosis"].value_counts().to_string())
    print(f"\nWrote {len(manifest)} images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
