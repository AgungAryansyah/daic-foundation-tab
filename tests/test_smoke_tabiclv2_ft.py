from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab.models.tabiclv2_ft import TabICLv2FineTunedModel


@pytest.mark.real_model
def test_real_tabiclv2_fine_tuning_smoke(tmp_path) -> None:
    if os.environ.get("DAIC_RUN_REAL_FINETUNE") != "1":
        pytest.skip("Set DAIC_RUN_REAL_FINETUNE=1 to run the fine-tuning checkpoint smoke test")
    model = TabICLv2FineTunedModel(
        {
            "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
            "parameters": {
                "epochs": 1,
                "learning_rate": 0.00001,
                "n_estimators_finetune": 1,
                "n_estimators_validation": 1,
                "n_estimators_inference": 1,
                "early_stopping": True,
                "patience": 1,
                "eval_metric": "roc_auc",
                "amp": True,
                "device": "cuda:0",
                "random_state": 42,
                "save_interval": 0,
            },
        }
    )
    features = pd.DataFrame(np.arange(32, dtype=float).reshape(8, 4))
    target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    model.fit(
        features.iloc[:6],
        target.iloc[:6],
        validation_features=features.iloc[6:],
        validation_target=target.iloc[6:],
        checkpoint_directory=tmp_path / "checkpoints",
    )

    assert model.predict(features).shape == (8,)
    assert model.predict_proba(features).shape == (8, 2)
    assert (tmp_path / "checkpoints" / "best.ckpt").is_file()
