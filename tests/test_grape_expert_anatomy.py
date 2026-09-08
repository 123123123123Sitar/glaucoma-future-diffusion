import json

import numpy as np

from glaucoma_forecast.data.grape_expert_anatomy import (
    anatomy_measurements,
    load_labelme_masks,
    sector_rim_fractions,
)


def test_load_labelme_masks_and_measurements(tmp_path):
    payload = {
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [
            {"label": "OD", "shape_type": "polygon", "points": [[10, 10], [90, 10], [90, 90], [10, 90]]},
            {"label": "OC", "shape_type": "polygon", "points": [[35, 30], [65, 30], [65, 70], [35, 70]]},
        ],
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(payload))
    masks = load_labelme_masks(path, 50)
    assert masks["disc"].shape == (50, 50)
    assert not np.any(masks["cup"] & ~masks["disc"])
    metrics = anatomy_measurements(masks)
    assert 0.4 < metrics["vcdr"] < 0.6


def test_sector_rim_fractions_uses_isnt_order():
    disc = np.ones((20, 20), dtype=bool)
    cup = np.zeros_like(disc)
    sectors = sector_rim_fractions(disc, cup)
    assert set(sectors) == {"inferior", "superior", "nasal", "temporal"}
    assert all(value == 1.0 for value in sectors.values())
