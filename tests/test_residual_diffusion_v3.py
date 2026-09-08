from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.training.residual_diffusion_trainer import (
    make_all_future_pairs,
    patient_split,
)


class ResidualPairingTests(unittest.TestCase):
    def test_all_real_future_pairs_and_patient_split(self) -> None:
        rows = []
        for patient in range(8):
            for visit, time in enumerate((0.0, 1.0, 3.0), start=1):
                rows.append(
                    {
                        "patient_id": str(patient),
                        "eye_id": f"{patient}_OD",
                        "laterality": "OD",
                        "visit_id": f"V{visit}",
                        "visit_index": visit - 1,
                        "time_from_baseline_years": time,
                        "image_path": f"{patient}_{visit}.jpg",
                    }
                )
        pairs = make_all_future_pairs(pd.DataFrame(rows))
        self.assertEqual(len(pairs), 8 * 3)
        self.assertEqual(set(pairs["horizon_years"]), {1.0, 2.0, 3.0})
        self.assertTrue(
            {
                "baseline_rnfl_inferior", "future_rnfl_inferior",
                "baseline_rnfl_temporal", "future_rnfl_temporal",
            }.issubset(pairs.columns)
        )
        train, validation, test = patient_split(pairs, 17, 1, 1)
        partitions = [
            set(frame["patient_id"].unique())
            for frame in (train, validation, test)
        ]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(torch_available(), "PyTorch not installed")
class ResidualDiffusionTorchTests(unittest.TestCase):
    def test_forward_and_direct_multiyear_sampling(self) -> None:
        import torch

        from glaucoma_forecast.models.residual_diffusion import (
            build_residual_diffusion,
            sample_residual_trajectory,
        )

        model = build_residual_diffusion(base_channels=8)
        low = torch.rand(1, 3, 32, 32)
        vessel = torch.rand(1, 1, 32, 32)
        output = model(
            torch.randn_like(low),
            low,
            vessel,
            torch.tensor([0.25]),
            torch.tensor([0.5]),
        )
        self.assertEqual(tuple(output.shape), tuple(low.shape))
        trajectories = sample_residual_trajectory(
            model,
            torch.rand(1, 3, 64, 64),
            low,
            vessel,
            years=[1.0, 2.0],
            max_supported_horizon=4.5,
            max_change=0.18,
            trajectories=2,
            diffusion_steps=2,
        )
        self.assertEqual(tuple(trajectories.shape), (1, 2, 2, 3, 64, 64))

    def test_structural_feature_channels(self) -> None:
        import torch

        from glaucoma_forecast.models.residual_diffusion import build_residual_diffusion

        model = build_residual_diffusion(base_channels=8, structural_channels=3)
        baseline = torch.rand(2, 3, 32, 32)
        output = model(
            torch.randn_like(baseline),
            baseline,
            torch.rand(2, 3, 32, 32),
            torch.tensor([0.2, 0.8]),
            torch.tensor([0.5, 0.5]),
        )
        self.assertEqual(tuple(output.shape), tuple(baseline.shape))

    def test_progression_head_uses_single_baseline_and_horizon(self) -> None:
        import torch

        from glaucoma_forecast.models.residual_diffusion import build_residual_diffusion

        model = build_residual_diffusion(
            base_channels=8,
            structural_channels=3,
            progression_head=True,
        )
        baseline = torch.rand(2, 3, 32, 32)
        structural = torch.rand(2, 3, 32, 32)
        horizon = torch.tensor([0.2, 0.8])
        logits = model.predict_progression(baseline, structural, horizon)
        self.assertEqual(tuple(logits.shape), (2,))
        output = model(
            torch.randn_like(baseline),
            baseline,
            structural,
            horizon,
            torch.tensor([0.5, 0.5]),
        )
        self.assertEqual(tuple(output.shape), tuple(baseline.shape))


if __name__ == "__main__":
    unittest.main()
