from __future__ import annotations

import json

import pandas as pd
import pytest
from PIL import Image

from glaucoma_forecast.data.longitudinal_annotations import (
    export_annotation_masks,
    longest_longitudinal_pairs,
    validate_annotation_manifest,
)


def test_longest_pair_selection_preserves_one_pair_per_eye() -> None:
    frame = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "eye_id": "e1",
                "laterality": "OD",
                "baseline_image_path": "a.jpg",
                "future_image_path": "b.jpg",
                "horizon_years": 3.1,
                "split": "train",
                "cohort": "grape_glaucoma_longitudinal",
            },
            {
                "patient_id": "p1",
                "eye_id": "e1",
                "laterality": "OD",
                "baseline_image_path": "a.jpg",
                "future_image_path": "c.jpg",
                "horizon_years": 4.2,
                "split": "train",
                "cohort": "grape_glaucoma_longitudinal",
            },
        ]
    )
    selected = longest_longitudinal_pairs(frame)
    assert len(selected) == 1
    assert selected.iloc[0]["horizon_years"] == pytest.approx(4.2)


def test_annotation_export_requires_cup_inside_disc(tmp_path) -> None:
    square = [
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.1},
        {"x": 0.9, "y": 0.9},
        {"x": 0.1, "y": 0.9},
    ]
    cup = [
        {"x": 0.4, "y": 0.4},
        {"x": 0.6, "y": 0.4},
        {"x": 0.6, "y": 0.6},
        {"x": 0.4, "y": 0.6},
    ]
    payload = {
        "annotation_id": "example",
        "baseline_disc_points": square,
        "baseline_cup_points": cup,
        "future_disc_points": square,
        "future_cup_points": cup,
    }
    paths = export_annotation_masks(payload, tmp_path, size=32)
    assert set(paths) == {
        "baseline_disc_mask_path",
        "baseline_cup_mask_path",
        "future_disc_mask_path",
        "future_cup_mask_path",
    }
    assert Image.open(paths["future_cup_mask_path"]).size == (32, 32)


def test_annotation_manifest_rejects_patient_leakage() -> None:
    base = {
        "annotation_id": "a",
        "patient_id": "p1",
        "eye_id": "e1",
        "laterality": "OD",
        "baseline_image_path": "a",
        "future_image_path": "b",
        "horizon_years": 4.0,
        "baseline_disc_mask_path": "d",
        "baseline_cup_mask_path": "c",
        "future_disc_mask_path": "fd",
        "future_cup_mask_path": "fc",
        "progression_label": "progressor",
        "quality_acceptable": True,
    }
    frame = pd.DataFrame([{**base, "split": "train"}, {**base, "annotation_id": "b", "split": "test"}])
    with pytest.raises(ValueError, match="more than one"):
        validate_annotation_manifest(frame)
