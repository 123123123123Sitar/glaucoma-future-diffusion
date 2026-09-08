from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.data.grape_progression import (
    build_grape_progression_cohort,
    patient_stratified_folds,
)
from glaucoma_forecast.evaluation.progression_metrics import (
    average_precision,
    binary_classification_metrics,
)


def _frames():
    manifest = pd.DataFrame(
        [
            {
                "patient_id": patient,
                "eye_id": f"{patient}_OD",
                "laterality": "OD",
                "visit_index": visit,
                "time_from_baseline_years": float(visit),
                "image_path": f"{patient}_{visit}.jpg",
                "camera_device": "camera",
            }
            for patient in ["p1", "p2", "p3", "p4", "p5"]
            for visit in [0, 2]
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "patient_id": patient,
                "eye_id": f"{patient}_OD",
                "progression_plr2": int(patient in {"p1", "p2"}),
                "progression_plr3": 0,
                "progression_md": int(patient == "p1"),
            }
            for patient in ["p1", "p2", "p3", "p4", "p5"]
        ]
    )
    return manifest, metadata


def test_progression_cohort_uses_only_earliest_image() -> None:
    manifest, metadata = _frames()
    cohort = build_grape_progression_cohort(manifest, metadata)
    assert len(cohort) == 5
    assert set(cohort["image_path"].str.endswith("_0.jpg")) == {True}
    assert set(cohort["observed_cfp_followup_years"]) == {2.0}


def test_patient_folds_never_split_a_patient() -> None:
    manifest, metadata = _frames()
    cohort = build_grape_progression_cohort(manifest, metadata)
    cohort = pd.concat([cohort, cohort.assign(eye_id=cohort.eye_id + "_OS")])
    folds = patient_stratified_folds(cohort, folds=5)
    check = cohort.assign(fold=folds).groupby("patient_id")["fold"].nunique()
    assert int(check.max()) == 1


def test_progression_metrics_are_well_formed() -> None:
    labels = [0, 1, 0, 1]
    probability = [0.1, 0.9, 0.3, 0.8]
    assert average_precision(labels, probability) == 1.0
    metrics = binary_classification_metrics(labels, probability, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
