from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.data.grape_multimodal import CLINICAL_COLUMNS
from glaucoma_forecast.training.multimodal_progression_trainer import (
    _fit_scaler,
    encode_clinical,
)


def test_scaler_uses_available_training_values_only() -> None:
    frame = pd.DataFrame(
        [
            {column: float(index + 1) for index, column in enumerate(CLINICAL_COLUMNS)},
            {column: float(index + 3) for index, column in enumerate(CLINICAL_COLUMNS)},
        ]
    )
    frame.loc[1, "baseline_iop"] = np.nan
    mean, scale = _fit_scaler(frame)
    encoded = encode_clinical(frame.iloc[1], mean, scale)
    width = len(CLINICAL_COLUMNS)
    iop_index = list(CLINICAL_COLUMNS).index("baseline_iop")
    assert encoded.shape == (width * 2 + 3,)
    assert encoded[iop_index] == 0.0
    assert encoded[width + iop_index] == 0.0


def test_all_model_variants_forward() -> None:
    import torch

    from glaucoma_forecast.models.multimodal_progression import (
        build_multimodal_progression_model,
    )

    image = torch.rand(2, 3, 64, 64)
    clinical = torch.rand(2, 17)
    for variant in ["clinical_only", "image_only", "multimodal"]:
        model = build_multimodal_progression_model(
            clinical_dim=17,
            feature_dim=16,
            backbone="compact",
            variant=variant,
        )
        output = model(image, image, clinical)
        assert tuple(output["progression_logit"].shape) == (2,)
