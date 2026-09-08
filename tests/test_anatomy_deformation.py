import pytest


torch = pytest.importorskip("torch")

from glaucoma_forecast.models.anatomy_deformation import (  # noqa: E402
    build_anatomy_deformation_model,
    displacement_jacobian,
    integrate_stationary_velocity,
    radial_cup_expansion_velocity,
    warp_tensor,
)


def test_zero_velocity_is_identity():
    image = torch.rand(1, 3, 32, 32)
    velocity = torch.zeros(1, 2, 32, 32)
    displacement = integrate_stationary_velocity(velocity)
    assert torch.allclose(warp_tensor(image, displacement), image, atol=1e-5)
    assert torch.all(displacement_jacobian(displacement) > 0)


def test_model_preserves_shape_and_bounds():
    model = build_anatomy_deformation_model(base_channels=8, max_displacement=4)
    image = torch.rand(2, 3, 32, 32)
    disc = torch.zeros(2, 1, 32, 32)
    cup = torch.zeros_like(disc)
    disc[:, :, 6:26, 6:26] = 1
    cup[:, :, 11:21, 11:21] = 1
    output = model(image, disc, cup, torch.ones(2), torch.ones(2) * 0.1)
    assert output["generated"].shape == image.shape
    assert output["generated"].min() >= 0
    assert output["generated"].max() <= 1


def test_radial_prior_enlarges_vertical_cup_extent():
    disc = torch.zeros(1, 1, 64, 64)
    cup = torch.zeros_like(disc)
    disc[:, :, 8:56, 8:56] = 1
    cup[:, :, 22:42, 22:42] = 1
    velocity = radial_cup_expansion_velocity(disc, cup, torch.tensor([0.10]))
    warped = warp_tensor(cup, integrate_stationary_velocity(velocity)) >= 0.5
    original_height = int(torch.where(cup[0, 0] > 0)[0].max() - torch.where(cup[0, 0] > 0)[0].min() + 1)
    warped_height = int(torch.where(warped[0, 0])[0].max() - torch.where(warped[0, 0])[0].min() + 1)
    assert warped_height > original_height
