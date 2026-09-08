from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.data.schema import normalize_manifest, validate_manifest
from glaucoma_forecast.data.splitting import assert_patient_disjoint_splits, create_patient_splits


def synthetic_manifest() -> pd.DataFrame:
    rows = []
    for patient in range(6):
        for eye in ["R", "L"]:
            eye_id = f"p{patient}_{eye}"
            for visit in range(3):
                rows.append(
                    {
                        "patient_id": f"p{patient}",
                        "eye_id": eye_id,
                        "laterality": eye,
                        "visit_id": f"v{visit}",
                        "visit_index": visit,
                        "time_from_baseline_years": visit * 0.9,
                        "image_path": f"/tmp/{eye_id}_{visit}.png",
                        "glaucoma_label": 1 if patient == 0 and visit == 2 else 0,
                    }
                )
    return normalize_manifest(pd.DataFrame(rows), source_dataset="synthetic")


class SchemaSplittingTests(unittest.TestCase):
    def test_manifest_validation(self) -> None:
        result = validate_manifest(synthetic_manifest())
        self.assertEqual(result.patients, 6)
        self.assertEqual(result.eyes, 12)

    def test_patient_level_splits_no_leakage(self) -> None:
        frames = create_patient_splits(synthetic_manifest(), seed=1)
        assert_patient_disjoint_splits(frames)
        assigned = sum(frame["patient_id"].nunique() for frame in frames.values())
        self.assertEqual(assigned, 6)

    def test_detects_patient_leakage(self) -> None:
        df = synthetic_manifest()
        patient = df[df["patient_id"] == "p0"]
        with self.assertRaises(ValueError):
            assert_patient_disjoint_splits({"train": patient, "test": patient})

    def test_split_manifests_keep_fellow_eyes_together(self) -> None:
        frames = create_patient_splits(synthetic_manifest(), seed=4)
        patient_to_split = {}
        for split, frame in frames.items():
            for patient in frame["patient_id"].unique():
                patient_to_split.setdefault(patient, split)
                self.assertEqual(patient_to_split[patient], split)


if __name__ == "__main__":
    unittest.main()
