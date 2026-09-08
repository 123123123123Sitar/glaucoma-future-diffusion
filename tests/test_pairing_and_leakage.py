from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.data.pairing import (
    assert_no_target_label_leakage,
    make_forecast_examples,
    one_year_pairs,
)
from glaucoma_forecast.data.schema import normalize_manifest


class PairingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = normalize_manifest(
            pd.DataFrame(
                [
                    {"patient_id": "p1", "eye_id": "e1", "visit_id": "v0", "time_from_baseline_years": 0.0, "image_path": "a.png", "glaucoma_label": 0},
                    {"patient_id": "p1", "eye_id": "e1", "visit_id": "v1", "time_from_baseline_years": 0.8, "image_path": "b.png", "glaucoma_label": 0},
                    {"patient_id": "p1", "eye_id": "e1", "visit_id": "v2", "time_from_baseline_years": 2.1, "image_path": "c.png", "glaucoma_label": 1},
                ]
            )
        )

    def test_forecast_examples_use_only_past_context(self) -> None:
        examples = make_forecast_examples(self.manifest, max_context_visits=2)
        self.assertEqual(len(examples), 2)
        for example in examples:
            self.assertTrue(all(t < example.target_time for t in example.context_times))
            self.assertGreater(example.forecast_horizon, 0)

    def test_one_year_pairs_preserve_actual_interval(self) -> None:
        pairs = one_year_pairs(self.manifest, tolerance=(0.75, 1.25))
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(float(pairs.iloc[0]["actual_interval_years"]), 0.8)

    def test_forecasting_mode_blocks_target_label_leakage(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_target_label_leakage("forecasting", {"target_glaucoma_label"})
        assert_no_target_label_leakage("controlled_synthesis", {"target_glaucoma_label"})


if __name__ == "__main__":
    unittest.main()
