import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from daic_foundation_tab.tabular_study.bundle import prepare_bundle, read_bundle
from daic_foundation_tab.tabular_study.condition import run_condition
from daic_foundation_tab.tabular_study.results import write_result
from tests.test_tabular_study_bundle import study_input
from tests.test_tabular_study_configuration import ROOT


@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("DAIC_RUN_TABULAR_CV_SMOKE") != "1", reason="GPU smoke test is opt-in"
)
@pytest.mark.parametrize("family", ["tabpfn35", "kumo_medium", "kumo_large"])
@pytest.mark.parametrize("mode", ["frozen_icl", "fine_tuning"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_real_gpu_fold_condition(tmp_path, monkeypatch, family, mode, task):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    config_path = study_input(tmp_path)
    profiles = tmp_path / "profiles"
    shutil.copytree(ROOT / "configs/models", profiles)
    for name in (
        "tabpfn35_ft.yaml",
        "kumo_medium_ft.yaml",
        "kumo_large_ft.yaml",
        "kumo_large_ft_regressor.yaml",
    ):
        path = profiles / name
        profile = yaml.safe_load(path.read_text())
        profile["model"].setdefault("parameters", {})["epochs"] = 1
        path.write_text(yaml.safe_dump(profile))
    config = yaml.safe_load(config_path.read_text())
    config["tabular_study"]["tabular_profiles"] = str(profiles)
    if family == "kumo_large":
        config["tabular_study"]["model_families"] = ["kumo_large"]
    config_path.write_text(yaml.safe_dump(config))
    bundle = prepare_bundle(config_path, tmp_path / "bundle")
    context = read_bundle(bundle)
    condition = next(
        item
        for item in context[0]["conditions"]["cv"]
        if item["family"] == family
        and item["mode"] == mode
        and item["task"] == task
        and item["modality"] == "audio_visual"
        and item["fold"] == 1
    )
    load_state_paths = []
    if family == "kumo_large" and mode == "fine_tuning":
        import sdm

        original_load_state_dict = sdm.models.KumoTabular.load_state_dict
        loaded_states = {}
        original_torch_load = torch.load

        def capture_torch_load(path, *args, **kwargs):
            state = original_torch_load(path, *args, **kwargs)
            loaded_states[id(state)] = Path(path).resolve()
            return state

        def capture_load_state_dict(model, state, *args, **kwargs):
            load_state_paths.append(loaded_states.get(id(state)))
            return original_load_state_dict(model, state, *args, **kwargs)

        monkeypatch.setattr(torch, "load", capture_torch_load)
        monkeypatch.setattr(sdm.models.KumoTabular, "load_state_dict", capture_load_state_dict)
    output = run_condition(
        bundle,
        condition,
        {"output_root": str(tmp_path / "runs"), "wandb": {"mode": "disabled"}},
        "smoke",
    )
    result = write_result(context, condition, output)
    predictions = pd.read_csv(output / "predictions_cv.csv")
    column = "y_pred" if task == "classification" else "phq8_pred"
    assert np.isfinite(predictions[column]).all()
    assert (result["checkpoint"] is not None) == (mode == "fine_tuning")
    if family == "kumo_large":
        detail = json.loads((output / "adaptation_metadata.json").read_text())
        record = json.loads((output / "study_run.json").read_text())
        saved_config = yaml.safe_load((output / "config_resolved.yaml").read_text())
        source_hash = detail["source_checkpoint_sha256"]
        expected_source_hash = (
            "cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613"
            if task == "classification"
            else "745e0e5d97e4256d8a24cb61c2eecbd1128da6c83938c196b7589ea6f62e7421"
        )
        assert saved_config["model"]["size"] == "large"
        assert (
            saved_config["model"]["checkpoint_revision"]
            == "3c3e10bbdb590ace29e7026847f92db3c603096d"
        )
        assert saved_config["model"]["checkpoint_version"] == (
            "large/classifier.pt" if task == "classification" else "large/regressor.pt"
        )
        assert source_hash == expected_source_hash
        assert record["source_checkpoint_sha256"] == source_hash
        if task == "classification":
            probabilities = predictions["prob_depressed"].to_numpy(dtype=float)
            assert np.isfinite(probabilities).all()
            assert np.all((probabilities >= 0) & (probabilities <= 1))
        if mode == "frozen_icl":
            assert detail["weights_frozen"] is True
            assert detail["gradients_enabled"] is False
            assert detail["optimizer_steps"] == detail["epochs_completed"] == 0
            assert detail["checkpoint_sha256"] == source_hash
            assert len(detail["weights_before_inference_sha256"]) == 64
            assert result["checkpoint"] is None
        else:
            assert saved_config["model"]["parameters"]["epochs"] == 1
            assert detail["optimizer_steps"] >= 1
            assert detail["epochs_completed"] == 1
            assert result["checkpoint"]["sha256"] == detail["checkpoint_sha256"]
            selected_checkpoint = Path(detail["checkpoint_path"]).resolve()
            assert selected_checkpoint.is_file()
            assert load_state_paths == [selected_checkpoint]
