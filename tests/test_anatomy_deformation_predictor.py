import pandas as pd

from glaucoma_forecast.inference.anatomy_deformation_predictor import automatic_progression_strength


def test_automatic_strength_ignores_test_rows():
    frame = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "progression_label": ["progressor", "progressor", "progressor"],
            "delta_vcdr": [0.06, 0.10, 0.40],
        }
    )
    assert 0.06 <= automatic_progression_strength(frame) <= 0.10
