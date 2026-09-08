#!/usr/bin/env python3
"""Combine frozen GRAPE pairs with balanced PAPILA healthy stability controls.

PAPILA is cross-sectional. Its healthy images are therefore repeated as
explicit pseudo-longitudinal zero-change targets at supported horizons. These
rows teach abstention/stability; they are not evidence of normal progression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HORIZONS = (1.0, 3.0, 4.83)


def _grape_pairs(directory: Path, manifest_path: Path) -> pd.DataFrame:
    names = {
        "train": "train_pairs.csv",
        "validation": "validation_pairs.csv",
        "test": "locked_test_pairs.csv",
    }
    manifest = pd.read_csv(manifest_path)
    sectors = ("inferior", "superior", "nasal", "temporal")
    anatomy_columns = [f"rnfl_{sector}" for sector in sectors]
    visits = manifest[["eye_id", "visit_id", *anatomy_columns]].copy()
    visits["eye_id"] = visits["eye_id"].astype(str)
    visits["visit_id"] = visits["visit_id"].astype(str)
    frames = []
    for split, filename in names.items():
        frame = pd.read_csv(directory / filename)
        frame["eye_id"] = frame["eye_id"].astype(str)
        for prefix, visit_column in (
            ("baseline", "baseline_visit_id"),
            ("future", "future_visit_id"),
        ):
            renamed = visits.rename(
                columns={
                    "visit_id": visit_column,
                    **{
                        column: f"{prefix}_{column}"
                        for column in anatomy_columns
                    },
                }
            )
            frame = frame.merge(
                renamed,
                on=["eye_id", visit_column],
                how="left",
                validate="many_to_one",
            )
        frame["split"] = split
        frame["cohort"] = "grape_glaucoma_longitudinal"
        frame["target_type"] = "observed_longitudinal_change"
        frame["diagnosis"] = "glaucoma"
        frame["patient_id"] = "GRAPE_" + frame["patient_id"].astype(str)
        frame["eye_id"] = "GRAPE_" + frame["eye_id"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _select_healthy_eyes(
    healthy: pd.DataFrame, split_targets: dict[str, int], seed: int
) -> dict[str, pd.DataFrame]:
    patients = healthy["patient_id"].astype(str).drop_duplicates().sample(
        frac=1.0, random_state=seed
    )
    unused = healthy.copy()
    selected: dict[str, pd.DataFrame] = {}
    used_patients: set[str] = set()
    for split, pair_target in split_targets.items():
        needed_eyes = (pair_target + len(HORIZONS) - 1) // len(HORIZONS)
        rows = []
        for patient in patients:
            patient = str(patient)
            if patient in used_patients:
                continue
            group = unused[unused["patient_id"].astype(str) == patient]
            if group.empty:
                continue
            used_patients.add(patient)
            remaining = needed_eyes - sum(len(part) for part in rows)
            rows.append(group.head(remaining))
            if sum(len(part) for part in rows) >= needed_eyes:
                break
        chosen = pd.concat(rows, ignore_index=True)
        if len(chosen) < needed_eyes:
            raise RuntimeError(f"Not enough unique healthy PAPILA eyes for {split}")
        selected[split] = chosen.head(needed_eyes)
    return selected


def _control_pairs(
    selected: dict[str, pd.DataFrame], split_targets: dict[str, int]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, eyes in selected.items():
        candidates: list[dict[str, object]] = []
        for eye in eyes.to_dict("records"):
            for horizon in HORIZONS:
                patient = f"PAPILA_{eye['patient_id']}"
                eye_id = f"PAPILA_{eye['eye_id']}"
                candidates.append(
                    {
                        "patient_id": patient,
                        "eye_id": eye_id,
                        "laterality": eye["laterality"],
                        "baseline_visit_id": "STABILITY_BASELINE",
                        "future_visit_id": f"PSEUDO_STABLE_{horizon:g}Y",
                        "baseline_image_path": eye["image_path"],
                        "future_image_path": eye["image_path"],
                        "horizon_years": horizon,
                        "split": split,
                        "cohort": "papila_healthy_stability_control",
                        "target_type": "pseudo_longitudinal_zero_change",
                        "diagnosis": "healthy",
                    }
                )
        rows.extend(candidates[: split_targets[split]])
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grape-split-dir",
        default="outputs/training/grape_residual_diffusion_v3_512_mps",
    )
    parser.add_argument(
        "--papila-manifest", default="data/external/papila/manifest.csv"
    )
    parser.add_argument("--grape-manifest", default="data/grape/manifest.csv")
    parser.add_argument(
        "--output", default="data/derived/balanced_stability_pairs_v1.csv"
    )
    parser.add_argument(
        "--report", default="data/derived/balanced_stability_pairs_v1_report.json"
    )
    parser.add_argument(
        "--conditioning-output",
        default="data/derived/balanced_stability_conditioning_v1.csv",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    grape = _grape_pairs(Path(args.grape_split_dir), Path(args.grape_manifest))
    split_targets = grape["split"].value_counts().to_dict()
    papila = pd.read_csv(args.papila_manifest)
    healthy = papila[papila["diagnosis"] == "healthy"].copy()
    selected = _select_healthy_eyes(healthy, split_targets, args.seed)
    controls = _control_pairs(selected, split_targets)
    combined = pd.concat([grape, controls], ignore_index=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    conditioning = (
        combined[["eye_id", "diagnosis"]]
        .drop_duplicates()
        .assign(
            diagnosis_condition=lambda frame: (
                frame["diagnosis"] == "glaucoma"
            ).astype(float)
        )[["eye_id", "diagnosis_condition"]]
    )
    conditioning.to_csv(args.conditioning_output, index=False)
    report = {
        "schema_version": "1.0",
        "rows": len(combined),
        "class_counts": combined["diagnosis"].value_counts().to_dict(),
        "cohort_counts": combined["cohort"].value_counts().to_dict(),
        "split_by_class": combined.groupby(["split", "diagnosis"]).size().unstack(
            fill_value=0
        ).to_dict(orient="index"),
        "patient_leakage": bool(
            (combined.groupby("patient_id")["split"].nunique() > 1).any()
        ),
        "conditioning_manifest": args.conditioning_output,
        "warning": (
            "PAPILA healthy rows are pseudo-longitudinal zero-change stability "
            "controls, not observed future photographs."
        ),
        "sources": {
            "glaucoma": "GRAPE longitudinal pairs with the original frozen split",
            "healthy": "PAPILA v1.1 cross-sectional healthy images",
        },
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
