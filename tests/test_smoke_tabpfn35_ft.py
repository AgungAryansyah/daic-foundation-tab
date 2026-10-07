from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab.config import load_config
from daic_foundation_tab.models.registry import create_model


@pytest.mark.parametrize("regression", [False, True])
def test_native_tabpfn_constructor_accepts_preset_parameters(regression):
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor

    suffix = "regression_" if regression else ""
    config = load_config(f"configs/experiments/tabpfn35_ft_{suffix}audio_visual_complete_cohort.yaml")
    fine_tuner = FinetunedTabPFNRegressor if regression else FinetunedTabPFNClassifier
    estimator = fine_tuner(**config["model"]["parameters"], model_version=ModelVersion.V3_5)
    assert estimator.model_version is ModelVersion.V3_5
    assert estimator.eval_metric == ("mse" if regression else "roc_auc")


@pytest.mark.real_model
@pytest.mark.parametrize("regression", [False, True])
def test_real_tabpfn35_fine_tuning_and_checkpoint_reload(tmp_path, regression):
    if os.environ.get("DAIC_RUN_REAL_TABPFN_FINETUNE") != "1":
        pytest.skip("Set DAIC_RUN_REAL_TABPFN_FINETUNE=1 on the remote GPU to run")
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    assert torch.cuda.is_available(), "The opt-in check requires CUDA"
    source = Path(os.environ["DAIC_TABPFN35_CHECKPOINT"]).expanduser().resolve()
    assert source.is_file(), "Supply the locally authorized TabPFN-3.5 checkpoint"
    suffix = "regression_" if regression else ""
    config = load_config(f"configs/experiments/tabpfn35_ft_{suffix}audio_visual_complete_cohort_smoke.yaml")
    parameters = config["model"]["parameters"]
    parameters.update(
        n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=1,
        **{"extra_regressor_kwargs" if regression else "extra_classifier_kwargs": {"model_path": str(source)}},
    )
    model = create_model(config["model"])
    features = pd.DataFrame(np.random.default_rng(42).normal(size=(20, 4)))
    target = pd.Series([2.0, 12.0] * 10 if regression else [0, 1] * 10)
    epochs = []
    model.fit(
        features.iloc[:16], target.iloc[:16],
        validation_features=features.iloc[16:], validation_target=target.iloc[16:],
        checkpoint_directory=tmp_path / "checkpoints", epoch_callback=epochs.append,
    )
    assert len(epochs) == 1
    prediction = np.asarray(model.predict(features.iloc[16:]))
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
    if not regression:
        assert np.isfinite(model.predict_proba(features.iloc[16:])).all()
    checkpoint = Path(model.finetune_metadata()["checkpoint_path"])
    assert checkpoint.suffix == ".pth" and checkpoint.is_file()
    estimator_class = TabPFNRegressor if regression else TabPFNClassifier
    restored = estimator_class(model_path=str(checkpoint), device="cuda:0", n_estimators=1, random_state=42)
    restored.fit(features.iloc[:16], target.iloc[:16])
    reloaded_prediction = np.asarray(restored.predict(features.iloc[16:]))
    assert reloaded_prediction.shape == (4,)
    assert np.isfinite(reloaded_prediction).all()
