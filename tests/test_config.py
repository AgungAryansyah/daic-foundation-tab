from __future__ import annotations

from pathlib import Path

import pytest

from daic_foundation_tab.config import ConfigError, load_config, validate_config


def test_composes_experiment_and_model_presets() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml")

    assert config["experiment"]["feature_set"] == "audio_visual"
    assert config["model"]["name"] == "tabiclv2_ft"
    assert config["model"]["parameters"]["epochs"] == 50


def test_rejects_retired_icl_model_name() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["model"]["name"] = "tabiclv2"

    with pytest.raises(ConfigError, match="retired"):
        validate_config(config)


def test_all_experiments_enable_test_predictions() -> None:
    project_root = Path(__file__).parents[1]
    for path in (project_root / "configs/experiments").glob("*.yaml"):
        config = load_config(path)
        assert config["evaluation"]["test_predictions"] is True, path.name


def test_rejects_cpu_runtime() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["runtime"]["device"] = "cpu"

    with pytest.raises(ConfigError, match="runtime.device"):
        validate_config(config)


def test_rejects_automatic_fine_tuning_device() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["model"]["parameters"]["device"] = "auto"

    with pytest.raises(ConfigError, match="must match runtime.device"):
        validate_config(config)


def test_rejects_invalid_decision_threshold() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["evaluation"]["threshold"] = 1.1

    with pytest.raises(ConfigError, match="evaluation.threshold"):
        validate_config(config)


def test_regression_config_requires_regressor_and_mae() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["experiment"]["task"] = "regression"

    with pytest.raises(ConfigError, match="tabiclv2_ft_regressor"):
        validate_config(config)

    config["model"]["name"] = "tabiclv2_ft_regressor"
    with pytest.raises(ConfigError, match="eval_metric"):
        validate_config(config)

    config["model"]["parameters"]["eval_metric"] = "mae"
    config["evaluation"].pop("threshold")
    validate_config(config)

    config["experiment"]["task"] = "classification"
    with pytest.raises(ConfigError, match="tabiclv2_ft"):
        validate_config(config)


@pytest.mark.parametrize("dataset", ["", "edaic_"])
@pytest.mark.parametrize("modality", ["audio", "visual", "audio_visual"])
def test_regression_presets_cover_each_dataset_and_modality(dataset, modality) -> None:
    root = Path(__file__).parents[1] / "configs/experiments"
    suffix = "" if dataset else "_complete_cohort"
    config = load_config(
        root / f"tabiclv2_ft_regression_{dataset}{modality}{suffix}.yaml"
    )

    assert config["experiment"]["task"] == "regression"
    assert config["experiment"]["feature_set"] == modality
    assert config["model"]["name"] == "tabiclv2_ft_regressor"
    assert config["model"]["checkpoint_version"] == "tabicl-regressor-v2-20260212.ckpt"
    assert config["model"]["parameters"]["eval_metric"] == "mae"
    assert config["evaluation"]["test_predictions"] is True


@pytest.mark.parametrize("dataset", ["", "edaic_"])
def test_regression_smoke_presets_skip_repeated_evaluation(dataset) -> None:
    root = Path(__file__).parents[1] / "configs/experiments"
    suffix = "" if dataset else "_complete_cohort"
    config = load_config(
        root / f"tabiclv2_ft_regression_{dataset}audio_visual{suffix}_smoke.yaml"
    )

    assert config["model"]["parameters"]["epochs"] == 1
    assert config["evaluation"]["repeated_holdout"]["enabled"] is False
    assert config["bootstrap"]["enabled"] is False


def test_rejects_invalid_wandb_mode() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["tracking"]["wandb"]["mode"] = "shared"

    with pytest.raises(ConfigError, match="online, offline, or disabled"):
        validate_config(config)


def test_rejects_malformed_wandb_configuration() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["tracking"]["wandb"] = []

    with pytest.raises(ConfigError, match="must be a mapping"):
        validate_config(config)
