#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image


DATE_PATTERN = re.compile(r"_(\d{4})_(\d{2})_(\d{2})_")
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def acquisition_date(path: Path) -> datetime:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse acquisition date: {path}")
    return datetime(*(int(value) for value in match.groups()))


def archive_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare official SIGF sequences for leakage-safe residual diffusion training."
    )
    parser.add_argument("--root", default="data/sigf/raw/SIGF-database")
    parser.add_argument("--manifest", default="data/sigf/manifest.csv")
    parser.add_argument("--pairs", default="data/sigf/pairs_5y_leakage_safe.csv")
    parser.add_argument("--summary", default="data/sigf/preparation_summary.json")
    parser.add_argument("--archive")
    parser.add_argument("--minimum-horizon", type=float, default=0.25)
    parser.add_argument("--maximum-horizon", type=float, default=5.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    eye_records: list[dict[str, object]] = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for official_split in SPLIT_PRIORITY:
        image_root = root / official_split / "image"
        label_root = root / official_split / "label"
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(f"Incomplete SIGF split: {official_split}")
        for eye_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            eye_id = eye_dir.name
            if "_" not in eye_id:
                raise ValueError(f"Unexpected eye identifier: {eye_id}")
            patient_id, laterality = eye_id.rsplit("_", 1)
            images = sorted(eye_dir.glob("*.[Jj][Pp][Gg]"), key=lambda path: (acquisition_date(path), path.name))
            label_path = label_root / f"{eye_id}.txt"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            labels = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
            if len(images) != len(labels):
                raise ValueError(
                    f"Image/label mismatch for {eye_id}: {len(images)} images, {len(labels)} labels"
                )
            if any(label not in {"0", "1"} for label in labels):
                raise ValueError(f"Non-binary label in {label_path}")
            patient_splits[patient_id].add(official_split)
            eye_records.append(
                {
                    "patient_id": patient_id,
                    "eye_id": eye_id,
                    "laterality": laterality,
                    "official_split": official_split,
                    "images": images,
                    "labels": labels,
                }
            )

    assigned_split = {
        patient_id: max(splits, key=SPLIT_PRIORITY.__getitem__)
        for patient_id, splits in patient_splits.items()
    }
    overlapping = {
        patient_id: sorted(splits, key=SPLIT_PRIORITY.__getitem__)
        for patient_id, splits in patient_splits.items()
        if len(splits) > 1
    }

    manifest_rows: list[dict[str, object]] = []
    for record in eye_records:
        images = record["images"]
        labels = record["labels"]
        baseline_date = acquisition_date(images[0])
        for visit_index, (image_path, label) in enumerate(zip(images, labels)):
            date = acquisition_date(image_path)
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as error:
                raise ValueError(f"Unreadable SIGF image: {image_path}") from error
            manifest_rows.append(
                {
                    "patient_id": record["patient_id"],
                    "eye_id": record["eye_id"],
                    "laterality": record["laterality"],
                    "visit_id": image_path.stem,
                    "visit_index": visit_index,
                    "acquisition_date": date.date().isoformat(),
                    "time_from_baseline_years": (date - baseline_date).days / 365.25,
                    "image_path": str(image_path.resolve()),
                    "glaucoma_label": int(label),
                    "progression_label": pd.NA,
                    "camera_device": pd.NA,
                    "image_width": width,
                    "image_height": height,
                    "quality_score": pd.NA,
                    "iop": pd.NA,
                    "rnfl_mean": pd.NA,
                    "rnfl_superior": pd.NA,
                    "rnfl_inferior": pd.NA,
                    "rnfl_nasal": pd.NA,
                    "rnfl_temporal": pd.NA,
                    "visual_field_md": pd.NA,
                    "source_dataset": "SIGF",
                    "official_split": record["official_split"],
                    "split": assigned_split[str(record["patient_id"])],
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["split", "patient_id", "eye_id", "visit_index"], kind="stable"
    )
    pair_rows: list[dict[str, object]] = []
    for eye_id, visits in manifest.groupby("eye_id", sort=True):
        visits = visits.sort_values("visit_index", kind="stable").to_dict("records")
        for baseline_index, baseline in enumerate(visits[:-1]):
            for future in visits[baseline_index + 1 :]:
                horizon = float(future["time_from_baseline_years"]) - float(
                    baseline["time_from_baseline_years"]
                )
                if not args.minimum_horizon <= horizon <= args.maximum_horizon:
                    continue
                pair_rows.append(
                    {
                        "patient_id": baseline["patient_id"],
                        "eye_id": eye_id,
                        "laterality": baseline["laterality"],
                        "baseline_visit_id": baseline["visit_id"],
                        "future_visit_id": future["visit_id"],
                        "baseline_image_path": baseline["image_path"],
                        "future_image_path": future["image_path"],
                        "horizon_years": horizon,
                        "baseline_glaucoma_label": baseline["glaucoma_label"],
                        "future_glaucoma_label": future["glaucoma_label"],
                        "official_split": baseline["official_split"],
                        "split": baseline["split"],
                    }
                )
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["split", "patient_id", "eye_id", "baseline_visit_id", "future_visit_id"],
        kind="stable",
    )
    if pairs.empty:
        raise ValueError("No valid longitudinal pairs were produced")
    if int(pairs.groupby("patient_id")["split"].nunique().max()) != 1:
        raise ValueError("Patient leakage remains in pair split")

    manifest_path = Path(args.manifest)
    pairs_path = Path(args.pairs)
    summary_path = Path(args.summary)
    for path in (manifest_path, pairs_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    pairs.to_csv(pairs_path, index=False)

    transitions = (
        pairs.assign(
            transition=pairs["baseline_glaucoma_label"].astype(str)
            + "->"
            + pairs["future_glaucoma_label"].astype(str)
        )["transition"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary = {
        "root": str(root),
        "archive": str(Path(args.archive).resolve()) if args.archive else None,
        "archive_sha256": archive_sha256(Path(args.archive) if args.archive else None),
        "images": int(len(manifest)),
        "eyes": int(manifest["eye_id"].nunique()),
        "patients": int(manifest["patient_id"].nunique()),
        "official_image_counts": manifest["official_split"].value_counts().sort_index().to_dict(),
        "leakage_safe_image_counts": manifest["split"].value_counts().sort_index().to_dict(),
        "overlapping_official_patients": overlapping,
        "assigned_overlap_splits": {patient: assigned_split[patient] for patient in overlapping},
        "pairs": int(len(pairs)),
        "pair_counts": pairs["split"].value_counts().sort_index().to_dict(),
        "label_transitions": transitions,
        "minimum_horizon_years": float(pairs["horizon_years"].min()),
        "maximum_horizon_years": float(pairs["horizon_years"].max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
