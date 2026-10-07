from __future__ import annotations

from pathlib import Path

import pytest

from daic_foundation_tab.config import ConfigError, load_config, validate_config

_ROOT = Path(__file__).parents[1]
_PRESETS = sorted((_ROOT / "configs/experiments").glob("tabiclv2_ft*.yaml"))


@pytest.mark.parametrize("baseline", _PRESETS, ids=lambda path: path.stem)
def test_tabpfn_presets_mirror_data_and_evaluation(baseline):
    tabicl = load_config(baseline)
    tabpfn = load_config(baseline.with_name(baseline.name.replace("tabiclv2_ft", "tabpfn35_ft")))
    for section in ("project", "runtime", "data", "modalities", "aggregation", "evaluation", "bootstrap"):
        assert tabpfn[section] == tabicl[section], section
    assert tabpfn["experiment"]["task"] == tabicl["experiment"]["task"]
    assert tabpfn["experiment"]["feature_set"] == tabicl["experiment"]["feature_set"]
    assert tabpfn["experiment"]["name"] == tabicl["experiment"]["name"].replace(
        "tabiclv2_ft", "tabpfn35_ft"
    )
    assert tabpfn["model"]["model_version"] == "v3.5"
    assert tabpfn["model"]["checkpoint_version"] == "tabpfn-v3.5-20260909.safetensors"
    parameters = tabpfn["model"]["parameters"]
    assert parameters["epochs"] == tabicl["model"]["parameters"]["epochs"]
    assert parameters["learning_rate"] == 1e-5
    assert parameters["n_estimators_final_inference"] == 8
    assert parameters["save_checkpoint_interval"] is None
    assert parameters["finetune_ctx_query_split_ratio"] == 0.2
    assert parameters["n_finetune_ctx_plus_query_samples"] == 10000
    assert parameters["early_stopping_patience"] == 10
    regression = tabpfn["experiment"]["task"] == "regression"
    assert parameters["eval_metric"] == ("mse" if regression else "roc_auc")
    for tabicl_only in ("amp", "patience", "save_interval", "warmup_proportion", "n_estimators_inference"):
        assert tabicl_only not in parameters


def test_tabpfn_matrix_has_twelve_full_and_four_smoke_presets():
    paths = list((_ROOT / "configs/experiments").glob("tabpfn35_ft*.yaml"))
    assert len(paths) == 16
    assert sum(path.stem.endswith("_smoke") for path in paths) == 4


@pytest.mark.parametrize(
    "field,value,error",
    [("model_version", "v3", "v3.5"), ("name", "tabpfn35_ft_regressor", "classification requires")],
)
def test_tabpfn_rejects_wrong_version_and_task(field, value, error):
    config = load_config(_ROOT / "configs/experiments/tabpfn35_ft_audio_complete_cohort.yaml")
    config["model"][field] = value
    with pytest.raises(ConfigError, match=error):
        validate_config(config)


def test_tabpfn_regression_requires_mse_and_checkpoint_selection():
    config = load_config(_ROOT / "configs/experiments/tabpfn35_ft_regression_edaic_audio.yaml")
    config["model"]["parameters"]["eval_metric"] = "mae"
    with pytest.raises(ConfigError, match="mse"):
        validate_config(config)
    config["model"]["parameters"]["eval_metric"] = "mse"
    config["model"]["parameters"]["early_stopping"] = False
    with pytest.raises(ConfigError, match="early_stopping"):
        validate_config(config)
