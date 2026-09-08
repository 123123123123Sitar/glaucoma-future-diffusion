from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.data.highres import preprocess_high_resolution
from glaucoma_forecast.inference.risk_predictor import predict_single_fundus_risk
from glaucoma_forecast.training.risk_trainer import validate_risk_manifest


def synthetic_fundus(path: Path, size: int = 768) -> None:
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 30, size - 30, size - 30), fill=(135, 58, 35))
    draw.ellipse(
        (int(size * 0.66), int(size * 0.42), int(size * 0.78), int(size * 0.58)),
        fill=(245, 190, 95),
    )
    for offset in range(-100, 101, 25):
        draw.line(
            (int(size * 0.72), size // 2, size // 2, size // 2 + offset),
            fill=(60, 22, 20),
            width=4,
        )
    image.save(path)


class HighResolutionPipelineTests(unittest.TestCase):
    def test_preprocess_produces_true_512_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fundus.png"
            synthetic_fundus(path)
            result = preprocess_high_resolution(path)
            self.assertEqual(result.full_field.size, (512, 512))
            self.assertEqual(result.optic_disc.size, (512, 512))
            self.assertEqual(result.vessel_map.size, (512, 512))
            self.assertEqual(result.quality.width, 768)

    def test_inference_without_checkpoint_abstains_from_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fundus.png"
            out = Path(tmp) / "prediction"
            synthetic_fundus(path)
            report = predict_single_fundus_risk(str(path), str(out), risk_checkpoints=[])
            self.assertEqual(report["model_status"], "risk_checkpoint_required")
            self.assertIsNone(report["risk_10y"])
            self.assertFalse(report["annual_scenario_images_512px"])
            on_disk = json.loads((out / "report.json").read_text())
            self.assertEqual(on_disk["schema_version"], "2.0")

    def test_manifest_requires_survival_outcomes(self) -> None:
        import pandas as pd

        valid = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "eye_id": ["p1_od"],
                "image_path": ["image.png"],
                "event_observed": [0],
                "event_or_censor_time_years": [10.0],
            }
        )
        validate_risk_manifest(valid)
        with self.assertRaises(ValueError):
            validate_risk_manifest(valid.drop(columns=["event_observed"]))


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(torch_available(), "PyTorch not installed")
class V2TorchTests(unittest.TestCase):
    def test_survival_loss_and_monotonic_risk(self) -> None:
        import torch

        from glaucoma_forecast.training.survival import (
            cumulative_risk_from_logits,
            discrete_time_survival_loss,
        )

        logits = torch.zeros(3, 10, requires_grad=True)
        loss = discrete_time_survival_loss(
            logits, torch.tensor([2.2, 5.0, 10.0]), torch.tensor([1.0, 0.0, 0.0])
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        risk = cumulative_risk_from_logits(logits.detach())
        self.assertTrue(torch.all(risk[:, 1:] >= risk[:, :-1]))

    def test_compact_dual_view_forward(self) -> None:
        import torch

        from glaucoma_forecast.models.risk_model import build_dual_view_risk_model

        model = build_dual_view_risk_model(years=10, feature_dim=32, backbone="compact")
        image = torch.rand(1, 3, 64, 64)
        output = model(image, image)
        self.assertEqual(tuple(output["hazard_logits"].shape), (1, 10))
        self.assertEqual(tuple(output["disc_cup_mask_logits"].shape), (1, 2, 64, 64))

    def test_scenario_forward_small(self) -> None:
        import torch

        from glaucoma_forecast.models.scenario_diffusion import build_scenario_diffusion

        model = build_scenario_diffusion(latent_channels=2, base_channels=8)
        image = torch.rand(1, 3, 64, 64)
        latent = model.encode(image)
        output = model(
            latent,
            latent,
            latent,
            torch.tensor([0.1]),
            torch.tensor([0.2]),
            torch.tensor([0.5]),
        )
        self.assertEqual(tuple(output.shape), tuple(latent.shape))
        self.assertEqual(tuple(model.decode(latent).shape), tuple(image.shape))


if __name__ == "__main__":
    unittest.main()
