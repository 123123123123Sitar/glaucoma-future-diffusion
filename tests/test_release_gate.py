from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.evaluation.release_gate import evaluate_progression_candidate


def test_release_gate_requires_all_scientific_checks() -> None:
    report = {
        "aggregate_oof": {
            "auroc": 0.80,
            "sensitivity": 0.90,
            "specificity": 0.60,
        },
        "patient_bootstrap_95ci": {"auroc": [0.65, 0.90]},
    }
    assert evaluate_progression_candidate(report)["passed"]
    report["aggregate_oof"]["specificity"] = 0.40
    assert not evaluate_progression_candidate(report)["passed"]
