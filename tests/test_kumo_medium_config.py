from __future__ import annotations

from pathlib import Path

import pytest

from daic_foundation_tab.config import ConfigError, load_config, validate_config

PRESETS = sorted(Path("configs/experiments").glob("kumo_medium_ft_*.yaml"))


def test_kumo_matrix_has_six_full_and_two_smoke_presets():
    assert len(PRESETS) == 8
    assert sum(path.stem.endswith("_smoke") for path in PRESETS) == 2


@pytest.mark.parametrize("preset", PRESETS, ids=lambda path: path.stem)
def test_kumo_presets_mirror_participants_features_and_evaluation(preset):
    kumo = load_config(preset)
    tabicl = load_config(preset.with_name(preset.name.replace("kumo_medium_ft", "tabiclv2_ft")))
    for section in (
        "project",
        "runtime",
        "data",
        "modalities",
        "aggregation",
        "evaluation",
        "bootstrap",
    ):
        assert kumo[section] == tabicl[section], section
    assert kumo["experiment"] == {
        **tabicl["experiment"],
        "name": tabicl["experiment"]["name"].replace("tabiclv2_ft", "kumo_medium_ft"),
    }
    parameters = kumo["model"]["parameters"]
    assert parameters["epochs"] == tabicl["model"]["parameters"]["epochs"]
    for key in (
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "warmup_proportion",
        "n_estimators_finetune",
        "n_estimators_validation",
        "n_estimators_inference",
        "max_data_size",
        "finetune_ctx_query_ratio",
        "patience",
        "min_delta",
        "random_state",
        "amp",
    ):
        assert parameters[key] == tabicl["model"]["parameters"][key]
    assert parameters["eval_metric"] == (
        "mse" if kumo["experiment"]["task"] == "regression" else "roc_auc"
    )
    assert kumo["data"]["root"] == "data/edaic"
    assert kumo["model"]["size"] == "medium"
    assert "use_lr_scheduler" not in parameters


@pytest.mark.parametrize(
    "key,value",
    [
        ("size", "large"),
        ("model_version", "latest"),
        ("checkpoint_revision", "main"),
        ("package_revision", "main"),
        ("checkpoint_version", "medium/regressor.pt"),
    ],
)
def test_kumo_rejects_wrong_model_identity(key, value):
    config = load_config(PRESETS[0])
    config["model"][key] = value
    with pytest.raises(ConfigError, match=key):
        validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("device", "cpu"),
        ("epochs", 0),
        ("patience", True),
        ("amp", False),
        ("learning_rate", float("nan")),
        ("warmup_proportion", 1),
        ("finetune_ctx_query_ratio", 0),
        ("eval_metric", "mae"),
        ("early_stopping_patience", 10),
    ],
)
def test_kumo_rejects_invalid_or_inherited_parameters(key, value):
    config = load_config(PRESETS[0])
    config["model"]["parameters"][key] = value
    with pytest.raises(ConfigError):
        validate_config(config)
