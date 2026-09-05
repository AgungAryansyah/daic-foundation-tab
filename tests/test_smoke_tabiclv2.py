from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab.models.tabiclv2 import TabICLv2Model


@pytest.mark.real_model
def test_real_tabiclv2_classifier_smoke() -> None:
    if os.environ.get("DAIC_RUN_REAL_MODEL") != "1":
        pytest.skip("Set DAIC_RUN_REAL_MODEL=1 to run the checkpoint smoke test")
    model = TabICLv2Model(
        {
            "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
            "parameters": {"n_estimators": 1, "batch_size": 1, "device": "auto", "kv_cache": False},
        }
    )
    features = pd.DataFrame(np.arange(16, dtype=float).reshape(4, 4))
    target = pd.Series([0, 1, 0, 1])

    model.fit(features, target)

    assert model.predict(features).shape == (4,)
    assert model.predict_proba(features).shape == (4, 2)
