import pytest

torch = pytest.importorskip("torch")

from glaucoma_forecast.training.disc_cup_losses import tight_disc_crop


def test_tight_disc_crop_from_full_and_detail_views():
    image = torch.rand(2, 3, 100, 100)
    centers = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
    full = tight_disc_crop(image, centers, "full_field", 64)
    detail = tight_disc_crop(image, centers, "optic_disc", 64)
    assert full.shape == (2, 3, 64, 64)
    assert detail.shape == (2, 3, 64, 64)
