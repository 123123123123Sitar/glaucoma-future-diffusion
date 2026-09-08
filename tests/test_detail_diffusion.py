import pytest

torch = pytest.importorskip("torch")

from glaucoma_forecast.models.detail_diffusion import (
    compose_detail_change,
    disc_bounds,
    extract_disc_tensor,
)


def test_disc_bounds_remain_inside_image():
    assert disc_bounds(512, 512, (0.99, 0.02), 0.5) == (256, 0, 512, 256)


def test_extract_disc_tensor_shape():
    image = torch.zeros(2, 3, 512, 512)
    crop = extract_disc_tensor(image, (0.4, 0.5), output_size=256)
    assert crop.shape == (2, 3, 256, 256)


def test_composition_preserves_pixels_outside_disc():
    baseline = torch.full((1, 3, 512, 512), 0.4)
    global_frame = baseline.clone()
    detail = torch.full((1, 3, 512, 512), 0.5)
    combined = compose_detail_change(global_frame, baseline, detail, (0.5, 0.5))
    assert torch.equal(combined[..., :100, :100], global_frame[..., :100, :100])
    assert combined[..., 256, 256].mean() > global_frame[..., 256, 256].mean()
