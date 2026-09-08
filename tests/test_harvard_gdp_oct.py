from __future__ import annotations

import numpy as np
import pandas as pd

from glaucoma_forecast.training.harvard_gdp_oct_trainer import (
    _load_rnflt,
    _normalise_rnflt,
)


def test_rnflt_loader_and_mask_normalisation(tmp_path):
    value = np.full((225, 225), 80.0, dtype=np.float32)
    value[:10, :10] = -2
    np.savez(tmp_path / "data_0001.npz", rnflt=value)
    frame = pd.DataFrame({"filename": ["data_0001"]})
    loaded = _load_rnflt(frame, str(tmp_path))
    transformed, test, mean, scale = _normalise_rnflt(
        loaded, loaded.copy(), np.array([0])
    )
    assert loaded.shape == (1, 224, 224)
    assert transformed.shape == (1, 2, 224, 224)
    assert transformed[0, 1, 0, 0] == 0
    assert transformed[0, 1, 20, 20] == 1
    assert np.isfinite(transformed).all()
    assert np.isfinite(test).all()
    assert mean == 80.0
    assert scale > 0
