import pytest

torch = pytest.importorskip("torch")

from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter


def test_disc_cup_segmenter_preserves_spatial_shape():
    model = build_disc_cup_segmenter(base_channels=4)
    output = model(torch.rand(2, 3, 64, 64))
    assert output.shape == (2, 2, 64, 64)
