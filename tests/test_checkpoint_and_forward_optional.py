from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(torch_available(), "PyTorch not installed")
class TorchOptionalTests(unittest.TestCase):
    def test_change_head_starts_at_persistence(self) -> None:
        import torch

        from glaucoma_forecast.training.grape_glaucoma_trainer import build_compact_denoiser

        model = build_compact_denoiser(base_channels=4)
        context = torch.rand(2, 3, 16, 16)
        horizon = torch.tensor([0.2, 0.6])
        prediction = model.predict_future(context, horizon)
        self.assertTrue(torch.equal(prediction, context))

    def test_forward_and_checkpoint(self) -> None:
        import torch

        from glaucoma_forecast.models.latent_autoencoder import build_autoencoder
        from glaucoma_forecast.training.checkpoints import load_checkpoint, save_checkpoint

        model = build_autoencoder(base_channels=4, latent_channels=2)
        x = torch.rand(1, 3, 16, 16)
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, opt, step=3)
            self.assertEqual(load_checkpoint(path, model, opt), 3)


if __name__ == "__main__":
    unittest.main()
