import os
import shutil

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
@pytest.mark.skipif(os.environ.get("DAIC_RUN_TABULAR_CV_SMOKE") != "1", reason="GPU smoke test is opt-in")
@pytest.mark.parametrize("family", ["tabpfn35", "kumo_medium"])
@pytest.mark.parametrize("mode", ["frozen_icl", "fine_tuning"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_real_gpu_fold_condition(tmp_path, family, mode, task):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    config_path = study_input(tmp_path)
    profiles = tmp_path / "profiles"
    shutil.copytree(ROOT / "configs/models", profiles)
    for name in ("tabpfn35_ft.yaml", "kumo_medium_ft.yaml"):
        path = profiles / name
        profile = yaml.safe_load(path.read_text())
        profile["model"]["parameters"]["epochs"] = 1
        path.write_text(yaml.safe_dump(profile))
    config = yaml.safe_load(config_path.read_text())
    config["tabular_study"]["tabular_profiles"] = str(profiles)
    config_path.write_text(yaml.safe_dump(config))
    bundle = prepare_bundle(config_path, tmp_path / "bundle")
    context = read_bundle(bundle)
    condition = next(item for item in context[0]["conditions"]["cv"]
                     if item["family"] == family and item["mode"] == mode
                     and item["task"] == task and item["modality"] == "audio_visual"
                     and item["fold"] == 1)
    output = run_condition(bundle, condition, {"output_root": str(tmp_path / "runs"),
                           "wandb": {"mode": "disabled"}}, "smoke")
    result = write_result(context, condition, output)
    predictions = pd.read_csv(output / "predictions_cv.csv")
    column = "y_pred" if task == "classification" else "phq8_pred"
    assert np.isfinite(predictions[column]).all()
    assert (result["checkpoint"] is not None) == (mode == "fine_tuning")
