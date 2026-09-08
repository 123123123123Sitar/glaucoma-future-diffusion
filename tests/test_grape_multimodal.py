from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.data.grape_multimodal import (
    CLINICAL_COLUMNS,
    VF_COLUMNS,
    build_three_year_progression_cohort,
    clinical_vector,
)


def _baseline() -> pd.DataFrame:
    rows = []
    for patient, label in [("p1", 1), ("p2", 0), ("p3", 0)]:
        row = {
            "patient_id": patient,
            "eye_id": f"{patient}_OD",
            "baseline_cfp": f"{patient}.jpg",
            "progression_plr2": label,
        }
        row.update({column: 20.0 for column in CLINICAL_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def _visits() -> pd.DataFrame:
    rows = []
    for patient, final_year in [("p1", 3.0), ("p2", 2.4), ("p3", 3.6)]:
        for year in [0.0, final_year / 2, final_year]:
            row = {
                "patient_id": patient,
                "eye_id": f"{patient}_OD",
                "interval_years": year,
            }
            row.update({column: 20.0 for column in VF_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_three_year_window_excludes_short_and_long_followup() -> None:
    cohort = build_three_year_progression_cohort(
        _baseline(), _visits(), image_dir="/images"
    )
    assert cohort["patient_id"].tolist() == ["p1"]
    assert cohort["forecast_horizon_years"].tolist() == [3.0]
    assert cohort["primary_progression_label"].tolist() == [1]


def test_clinical_vector_has_explicit_missingness() -> None:
    row = {column: 1.0 for column in CLINICAL_COLUMNS}
    row["baseline_iop"] = np.nan
    values, mask = clinical_vector(row)
    index = list(CLINICAL_COLUMNS).index("baseline_iop")
    assert values[index] == 0.0
    assert mask[index] == 0.0
    assert values.shape == mask.shape == (len(CLINICAL_COLUMNS),)
