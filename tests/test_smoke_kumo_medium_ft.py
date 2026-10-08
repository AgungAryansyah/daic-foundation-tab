from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab.config import load_config
from daic_foundation_tab.models.registry import create_model


@pytest.mark.real_model
@pytest.mark.parametrize("regression", [False, True])
def test_real_medium_updates_weights_and_reloads_selected_checkpoint(
    monkeypatch, tmp_path, regression
):
    if os.environ.get("DAIC_RUN_REAL_KUMO_FINETUNE") != "1":
        pytest.skip("Set DAIC_RUN_REAL_KUMO_FINETUNE=1 to download Medium weights and check CUDA")
    import sdm
    import torch

    assert torch.cuda.is_available(), "The opt-in check requires CUDA"
    suffix = "regression_" if regression else ""
    config = load_config(
        f"configs/experiments/kumo_medium_ft_{suffix}edaic_audio_visual_smoke.yaml"
    )
    model = create_model(config["model"])
    load = model._load_model
    initial = {}

    def capture_source():
        native, source = load()
        initial.update(
            {key: value.detach().cpu().clone() for key, value in native.named_parameters()}
        )
        return native, source

    monkeypatch.setattr(model, "_load_model", capture_source)
    features = pd.DataFrame(np.random.default_rng(42).normal(size=(20, 4)), columns=list("abcd"))
    target = pd.Series([2.0, 12.0] * 10 if regression else [0, 1] * 10)
    epochs = []

    def epoch_callback(metrics):
        assert any(
            not torch.equal(value.detach().cpu(), initial[key])
            for key, value in model._model.named_parameters()
        )
        epochs.append(metrics)

    model.fit(
        features.iloc[:16],
        target.iloc[:16],
        validation_features=features.iloc[16:],
        validation_target=target.iloc[16:],
        checkpoint_directory=tmp_path / "checkpoints",
        epoch_callback=epoch_callback,
    )
    assert len(epochs) == 1 and model.finetune_metadata()["optimizer_steps"] == 1
    prediction = model.predict(features.iloc[16:])
    assert prediction.shape == (4,) and np.isfinite(prediction).all()
    checkpoint = Path(model.finetune_metadata()["checkpoint_path"])
    restored = sdm.models.KumoTabular(
        task=model.task, size="medium", pretrained=False, device="cuda:0"
    )
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cuda:0", weights_only=True), strict=True
    )
    restored.eval()
    x, y = model._tables(features.iloc[:16], target.iloc[:16])
    query = model._tables(features.iloc[16:])
    with torch.amp.autocast("cuda", dtype=torch.float16):
        restored.fit(x, y, num_estimators=8, estimator_batch_size=1, generator=model._generator())
        out = restored.predict(query)
    reloaded = model._prediction(out)
    if regression:
        np.testing.assert_allclose(reloaded, prediction, rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_allclose(
            reloaded, model.predict_proba(features.iloc[16:]), rtol=1e-5, atol=1e-5
        )
