from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab.models.tabiclv2_ft_regressor import TabICLv2FineTunedRegressor


@pytest.mark.real_model
def test_real_tabiclv2_regression_fine_tuning_smoke(tmp_path) -> None:
    if os.environ.get("DAIC_RUN_REAL_FINETUNE") != "1":
        pytest.skip("Set DAIC_RUN_REAL_FINETUNE=1 to run the regression checkpoint smoke test")
    model = TabICLv2FineTunedRegressor(
        {
            "checkpoint_version": "tabicl-regressor-v2-20260212.ckpt",
            "parameters": {
                "epochs": 1,
                "learning_rate": 0.00001,
                "n_estimators_finetune": 1,
                "n_estimators_validation": 1,
                "n_estimators_inference": 1,
                "early_stopping": True,
                "patience": 1,
                "eval_metric": "mae",
                "amp": True,
                "device": "cuda:0",
                "random_state": 42,
                "save_interval": 0,
            },
        }
    )
    features = pd.DataFrame(np.arange(48, dtype=float).reshape(12, 4))
    target = pd.Series([1.0, 4.0, 8.0, 12.0, 16.0, 20.0] * 2)

    model.fit(
        features.iloc[:10],
        target.iloc[:10],
        validation_features=features.iloc[10:],
        validation_target=target.iloc[10:],
        checkpoint_directory=tmp_path / "checkpoints",
    )

    prediction = np.asarray(model.predict(features), dtype=float)
    assert prediction.shape == (12,)
    assert np.isfinite(prediction).all()
    assert (tmp_path / "checkpoints" / "best.ckpt").is_file()
