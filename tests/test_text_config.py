from copy import deepcopy
from pathlib import Path

import pytest

from daic_foundation_tab.config import ConfigError, load_config, validate_config
from daic_foundation_tab.models.capabilities import model_capabilities
from daic_foundation_tab.models.registry import ModelRegistryError, create_text_model


def text_config():
    return {
        "input": {"kind": "transcript"},
        "project": {"seed": 42, "cache_root": "cache", "output_root": "outputs"},
        "runtime": {"require_cuda": True, "device": "cuda:0"},
        "data": {"dataset_id": "synthetic", "transcripts": {"pattern": "{participant_id}.tsv"}},
        "experiment": {"task": "classification", "mode": "frozen_icl"},
        "model": {
            "name": "qwen35_2b",
            "model_id": "Qwen/Qwen3.5-2B",
            "parameters": {"max_length": 4096},
        },
        "evaluation": {"threshold": 0.5, "fine_tuning": {"validation_fraction": 0.2}},
        "tracking": {"wandb": {"enabled": False}},
    }


def test_text_config_does_not_require_aggregation():
    config = text_config()
    validate_config(config)
    config["experiment"].update(task="regression", mode="fine_tuning")
    config["evaluation"]["threshold"] = None
    validate_config(config)
    assert model_capabilities("kumo_medium_ft_regressor").input_kind == "tabular"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("input", "kind", "audio"),
        ("model", "name", "tabiclv2_ft"),
        ("model", "name", None),
        ("model", "model_id", "other"),
        ("runtime", "device", "cpu"),
    ],
)
def test_rejects_incompatible_configuration(section, key, value):
    config = deepcopy(text_config())
    config[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("checkpoint_steps", 0),
        ("max_length", True),
        ("demonstrations", -1),
        ("learning_rate", float("nan")),
        ("adapter_method", "qlora"),
        ("adapter_method", "dora"),
    ],
)
def test_rejects_invalid_execution_settings(key, value):
    config = text_config()
    config["model"]["parameters"][key] = value
    with pytest.raises(ConfigError):
        validate_config(config)


def test_dora_is_valid_in_fine_tuning_mode():
    config = text_config()
    config["experiment"]["mode"] = "fine_tuning"
    config["model"]["parameters"]["adapter_method"] = "dora"
    validate_config(config)


def test_all_text_examples_validate_without_loading_models():
    root = Path(__file__).parents[1] / "configs/examples/qwen35_2b"
    paths = list(root.glob("*.yaml"))
    assert len(paths) == 5
    for path in paths:
        config = load_config(path)
        assert config["input"]["kind"] == "transcript"


def test_registered_factory_does_not_load_weights():
    adapter = create_text_model(text_config())
    assert adapter.model is None
    config = text_config()
    config["model"]["name"] = "tabiclv2_ft"
    with pytest.raises(ModelRegistryError):
        create_text_model(config)


def test_text_config_rejects_dialogue_input():
    config = text_config()
    config["data"]["transcripts"]["speaker_policy"] = "dialogue"
    with pytest.raises(ConfigError, match="participant_only"):
        validate_config(config)


def test_native_edaic_format_requires_explicit_dataset_identity():
    config = text_config()
    config["data"]["transcripts"]["format"] = "edaic"
    with pytest.raises(ConfigError, match="dataset_id=edaic"):
        validate_config(config)
    config["data"]["dataset_id"] = "edaic"
    validate_config(config)
    config["data"]["transcripts"]["format"] = "automatic"
    with pytest.raises(ConfigError, match="format"):
        validate_config(config)


def test_removed_training_subsampling_option_is_rejected():
    config = text_config()
    config["study"] = {"label_budget": 4}
    with pytest.raises(ConfigError, match="was removed"):
        validate_config(config)


@pytest.mark.parametrize("family,model_id", [
    ("qwen35_2b", "Qwen/Qwen3.5-2B"),
    ("qwen35_0_8b", "Qwen/Qwen3.5-0.8B"),
])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mode", ["frozen_icl", "fine_tuning"])
def test_qwen_model_sizes_validate_and_reject_mismatched_checkpoints(family, model_id, task, mode):
    config = text_config()
    config["model"].update(name=family, model_id=model_id)
    config["experiment"].update(task=task, mode=mode)
    config["evaluation"]["threshold"] = .5 if task == "classification" else None
    validate_config(config)
    adapter = create_text_model(config)
    assert adapter.config["model"]["model_id"] == model_id and adapter.model is None
    config["model"]["model_id"] = "Qwen/Qwen3.5-2B" if family == "qwen35_0_8b" else "Qwen/Qwen3.5-0.8B"
    with pytest.raises(ConfigError, match="Use Qwen"):
        validate_config(config)
