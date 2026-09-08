#!/usr/bin/env python3
"""Build long-horizon GRAPE pairs using the official ophthalmologist polygons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.grape_expert_anatomy import anatomy_measurements, load_labelme_masks


def _split_map(source: Path) -> dict[str, str]:
    if not source.exists():
        return {}
    frame = pd.read_csv(source)
    patient_ids = frame["patient_id"].astype(str).str.removeprefix("GRAPE_")
    return dict(zip(patient_ids, frame["split"].astype(str)))


def _new_splits(patient_ids: list[str], seed: int) -> dict[str, str]:
    patients = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    test_count = max(1, round(len(patients) * 0.15))
    validation_count = max(1, round(len(patients) * 0.15))
    return {
        patient: (
            "test" if index < test_count
            else "validation" if index < test_count + validation_count
            else "train"
        )
        for index, patient in enumerate(patients)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grape-manifest", default="data/grape/manifest.csv")
    parser.add_argument("--json-dir", default="data/external/grape_official/json")
    parser.add_argument("--roi-dir", default="data/external/grape_official/roi")
    parser.add_argument("--source-splits", default="outputs/annotations/grape_pseudo_anatomy_native_v3/manifest.csv")
    parser.add_argument("--output-dir", default="outputs/annotations/grape_expert_anatomy_v1")
    parser.add_argument("--minimum-years", type=float, default=3.0)
    parser.add_argument("--progressor-delta", type=float, default=0.05)
    parser.add_argument("--stable-delta", type=float, default=0.02)
    parser.add_argument("--mask-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--held-out-demo-patient", default="12")
    args = parser.parse_args()

    visits = pd.read_csv(args.grape_manifest)
    visits = visits.sort_values(["eye_id", "time_from_baseline_years", "visit_index"])
    candidates = []
    for eye_id, group in visits.groupby("eye_id"):
        baseline = group.iloc[0]
        future = group.iloc[-1]
        horizon = float(future["time_from_baseline_years"] - baseline["time_from_baseline_years"])
        if horizon < args.minimum_years:
            continue
        candidates.append((str(eye_id), baseline, future, horizon))

    output = Path(args.output_dir)
    masks_dir = output / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    split_by_patient = _split_map(Path(args.source_splits))
    if not split_by_patient:
        split_by_patient = _new_splits([str(row[1]["patient_id"]) for row in candidates], args.seed)

    rows = []
    rejected = []
    for eye_id, baseline, future, horizon in candidates:
        patient_id = str(baseline["patient_id"])
        baseline_stem = Path(str(baseline["image_path"])).stem
        future_stem = Path(str(future["image_path"])).stem
        baseline_json = Path(args.json_dir) / f"{baseline_stem}.json"
        future_json = Path(args.json_dir) / f"{future_stem}.json"
        baseline_roi = Path(args.roi_dir) / f"{baseline_stem}.jpg"
        future_roi = Path(args.roi_dir) / f"{future_stem}.jpg"
        if not all(path.exists() for path in (baseline_json, future_json, baseline_roi, future_roi)):
            rejected.append({"eye_id": eye_id, "reason": "missing_official_visit_asset"})
            continue
        try:
            baseline_masks = load_labelme_masks(baseline_json, args.mask_size)
            future_masks = load_labelme_masks(future_json, args.mask_size)
        except ValueError as error:
            rejected.append({"eye_id": eye_id, "reason": str(error)})
            continue
        baseline_metrics = anatomy_measurements(baseline_masks)
        future_metrics = anatomy_measurements(future_masks)
        delta_vcdr = future_metrics["vcdr"] - baseline_metrics["vcdr"]
        if delta_vcdr >= args.progressor_delta:
            label = "progressor"
        elif abs(delta_vcdr) <= args.stable_delta:
            label = "stable"
        else:
            label = "uncertain"
        annotation_id = f"GRAPE_{eye_id}_{horizon:.3f}y"
        mask_paths = {}
        for visit_name, masks in (("baseline", baseline_masks), ("future", future_masks)):
            for structure, mask in masks.items():
                path = (masks_dir / f"{annotation_id}_{visit_name}_{structure}.png").resolve()
                Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                mask_paths[f"{visit_name}_{structure}_mask_path"] = str(path)
        row = {
            "annotation_id": annotation_id,
            "patient_id": patient_id,
            "eye_id": eye_id,
            "laterality": str(baseline["laterality"]),
            "baseline_visit_id": str(baseline["visit_id"]),
            "future_visit_id": str(future["visit_id"]),
            "baseline_image_path": str(baseline_roi),
            "future_image_path": str(future_roi),
            "baseline_full_field_path": str(baseline["image_path"]),
            "future_full_field_path": str(future["image_path"]),
            "horizon_years": horizon,
            "split": split_by_patient.get(patient_id, "train"),
            **mask_paths,
            "progression_label": label,
            "quality_acceptable": True,
            "baseline_vcdr": baseline_metrics["vcdr"],
            "future_vcdr": future_metrics["vcdr"],
            "delta_vcdr": delta_vcdr,
            "label_source": "official_grape_ophthalmologist_polygon",
        }
        for sector in ("inferior", "superior", "nasal", "temporal"):
            row[f"baseline_rim_{sector}"] = baseline_metrics[f"rim_{sector}"]
            row[f"future_rim_{sector}"] = future_metrics[f"rim_{sector}"]
            row[f"delta_rim_{sector}"] = (
                future_metrics[f"rim_{sector}"] - baseline_metrics[f"rim_{sector}"]
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No official expert anatomy pairs were produced")
    if args.held_out_demo_patient:
        frame.loc[
            frame["patient_id"].astype(str) == str(args.held_out_demo_patient), "split"
        ] = "test"
    patient_splits = frame.groupby("patient_id")["split"].nunique()
    if int(patient_splits.max()) > 1:
        raise ValueError("Patient leakage across expert anatomy splits")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    frame.to_csv(manifest, index=False)
    eligible_demo = frame[
        (frame["split"] == "test")
        & (frame["progression_label"] == "progressor")
        & frame["delta_vcdr"].between(0.05, 0.20)
    ].copy()
    def sharpness(path: str) -> float:
        with Image.open(path) as source:
            edge = np.asarray(source.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        return float(edge[10:-10, 10:-10].var())

    eligible_demo["baseline_sharpness"] = eligible_demo["baseline_image_path"].map(sharpness)
    eligible_demo["demo_score"] = (
        eligible_demo["baseline_sharpness"]
        - 50.0 * (eligible_demo["delta_vcdr"] - 0.10).abs()
    )
    demo = None if eligible_demo.empty else eligible_demo.sort_values("demo_score", ascending=False).iloc[0]
    report = {
        "manifest": str(manifest),
        "pairs": int(len(frame)),
        "label_counts": frame["progression_label"].value_counts().to_dict(),
        "split_counts": frame["split"].value_counts().to_dict(),
        "rejected": rejected,
        "recommended_demo_annotation_id": None if demo is None else str(demo["annotation_id"]),
        "recommended_demo_delta_vcdr": None if demo is None else float(demo["delta_vcdr"]),
        "recommended_demo_baseline_sharpness": None if demo is None else float(demo["baseline_sharpness"]),
        "label_source": "official_grape_ophthalmologist_polygon",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
