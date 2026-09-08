import pytest

torch = pytest.importorskip("torch")

from glaucoma_forecast.training.anatomy_losses import (
    disc_sector_masks,
    rnfl_decline_weights,
    sector_weighted_residual_loss,
)


def test_sector_masks_respect_eye_specific_temporal_direction():
    centers = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    masks = disc_sector_masks(32, 32, centers, ["OD", "OS"], 0.0, 0.45)
    assert masks[0, 3, :, 20:].sum() > masks[0, 3, :, :12].sum()
    assert masks[1, 3, :, :12].sum() > masks[1, 3, :, 20:].sum()


def test_rnfl_decline_emphasizes_largest_observed_sector_loss():
    baseline = torch.tensor([[120.0, 110.0, 80.0, 70.0]])
    future = torch.tensor([[90.0, 105.0, 80.0, 70.0]])
    weights, valid = rnfl_decline_weights(baseline, future, torch.tensor([True]), emphasis=3.0)
    assert valid.tolist() == [True]
    assert weights[0, 0].item() == pytest.approx(4.0)
    assert weights[0, 0] > weights[0, 1]


def test_anatomy_loss_is_zero_without_observed_rnfl():
    prediction = torch.ones(1, 3, 16, 16, requires_grad=True)
    target = torch.zeros_like(prediction)
    masks = disc_sector_masks(16, 16, torch.tensor([[0.5, 0.5]]), ["OD"], 0.0, 0.45)
    loss = sector_weighted_residual_loss(
        prediction, target, masks, torch.ones(1, 4), torch.tensor([False])
    )
    assert loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None
