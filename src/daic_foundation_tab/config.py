from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, *, _validate: bool = True) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")

    parents = raw.get("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ConfigError("'extends' must be a path or a list of paths")

    resolved: dict[str, Any] = {}
    for parent in parents:
        if not isinstance(parent, str):
            raise ConfigError("Every 'extends' entry must be a path string")
        resolved = deep_merge(resolved, load_config(config_path.parent / parent, _validate=False))

    current = dict(raw)
    current.pop("extends", None)
    resolved = deep_merge(resolved, current)
    resolved["_config_path"] = str(config_path)
    if _validate:
        validate_config(resolved)
    return resolved


def validate_config(config: Mapping[str, Any]) -> None:
    input_kind = config.get("input", {}).get("kind", "tabular")
    if input_kind == "transcript":
        from daic_foundation_tab.text_config import validate_text_config

        validate_text_config(config)
        return
    if input_kind != "tabular":
        raise ConfigError("input.kind must be tabular or transcript")
    for section in (
        "project",
        "runtime",
        "data",
        "aggregation",
        "experiment",
        "model",
        "evaluation",
        "tracking",
    ):
        if section not in config:
            raise ConfigError(f"Missing required configuration section: {section}")

    task = config["experiment"].get("task")
    if task not in {"classification", "regression"}:
        raise ConfigError("experiment.task must be classification or regression")
    mode = config["experiment"].get("mode", "fine_tuning")
    if mode not in {"fine_tuning", "frozen_icl"}:
        raise ConfigError("experiment.mode must be fine_tuning or frozen_icl")
    if config["experiment"].get("feature_set") not in {"audio", "visual", "audio_visual"}:
        raise ConfigError("Phase 1 feature_set must be audio, visual, or audio_visual")
    model_name = config["model"].get("name")
    if model_name == "tabiclv2":
        raise ConfigError("Use tabiclv2_ft for fine-tuning or tabiclv2_icl for frozen inference")
    suffix = "_regressor" if task == "regression" else ""
    families = ("tabiclv2", "tabpfn35", "kumo_medium")
    method = "icl" if mode == "frozen_icl" else "ft"
    expected_models = tuple(f"{family}_{method}{suffix}" for family in families)
    if model_name not in expected_models:
        raise ConfigError(f"{task} requires model.name to be {' or '.join(expected_models)}")
    is_tabpfn = model_name.startswith("tabpfn35")
    is_kumo = model_name.startswith("kumo_medium")
    is_icl = mode == "frozen_icl"
    if is_icl:
        expected_checkpoint = {
            "tabiclv2": "tabicl-regressor-v2-20260212.ckpt"
            if task == "regression"
            else "tabicl-classifier-v2-20260212.ckpt",
            "tabpfn35": "tabpfn-v3.5-20260909.safetensors",
            "kumo_medium": "medium/regressor.pt"
            if task == "regression"
            else "medium/classifier.pt",
        }[model_name.removesuffix("_regressor").removesuffix("_icl")]
        if config["model"].get("checkpoint_version") != expected_checkpoint:
            raise ConfigError(f"Frozen ICL requires checkpoint {expected_checkpoint!r}")
        if is_kumo:
            expected_kumo = {
                "size": "medium",
                "model_version": "v1.0.1",
                "package_revision": "842c408fe2a8711bdf2e7cff4bfbe54266d6b940",
                "checkpoint_revision": "3c3e10bbdb590ace29e7026847f92db3c603096d",
            }
            if any(config["model"].get(key) != value for key, value in expected_kumo.items()):
                raise ConfigError("Frozen Kumo ICL requires the pinned Medium model and checkpoint")
    if is_tabpfn and config["model"].get("model_version") != "v3.5":
        raise ConfigError("TabPFN-3.5 requires model.model_version to be v3.5")
    runtime = config["runtime"]
    if not isinstance(runtime, Mapping):
        raise ConfigError("runtime must be a mapping")
    if runtime.get("require_cuda") is not True:
        raise ConfigError("GPU-only runs require runtime.require_cuda to be true")
    if runtime.get("device") != "cuda:0":
        raise ConfigError("GPU-only runs require runtime.device to be 'cuda:0'")
    parameters = config["model"].get("parameters")
    if not isinstance(parameters, Mapping):
        raise ConfigError("model.parameters must be a mapping")
    if is_kumo and not is_icl:
        from daic_foundation_tab.models.kumo_medium_ft import validate_kumo_config

        try:
            validate_kumo_config(config["model"])
        except ValueError as error:
            raise ConfigError(str(error)) from error
    if parameters.get("device") != runtime["device"]:
        raise ConfigError("model.parameters.device must match runtime.device")
    if is_icl:
        if set(parameters) != {"n_estimators", "device", "random_state"}:
            raise ConfigError("Frozen ICL model.parameters must contain n_estimators, device, and random_state only")
        estimators = parameters.get("n_estimators")
        if isinstance(estimators, bool) or not isinstance(estimators, int) or estimators < 1:
            raise ConfigError("Frozen ICL n_estimators must be a positive integer")
        random_state = parameters.get("random_state")
        if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
            raise ConfigError("Frozen ICL random_state must be a nonnegative integer")
    if not is_tabpfn and not is_icl and parameters.get("amp") is not True:
        raise ConfigError("GPU-only runs require model.parameters.amp to be true")
    if is_tabpfn and not is_icl and parameters.get("early_stopping", True) is not True:
        raise ConfigError("TabPFN experiments require early_stopping for checkpoint selection")
    if task == "classification":
        threshold = config["evaluation"].get("threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1
        ):
            raise ConfigError("evaluation.threshold must be between zero and one")
    elif not is_icl and parameters.get("eval_metric") != ("mse" if is_tabpfn or is_kumo else "mae"):
        metric = "mse" if is_tabpfn or is_kumo else "mae"
        raise ConfigError(f"Regression fine-tuning requires model.parameters.eval_metric to be {metric}")
    elif config["evaluation"].get("threshold") is not None:
        raise ConfigError("Regression does not use evaluation.threshold; set it to null")
    fine_tuning = config["evaluation"].get("fine_tuning")
    if not isinstance(fine_tuning, Mapping):
        raise ConfigError("Fine-tuning requires evaluation.fine_tuning settings")
    validation_fraction = fine_tuning.get("validation_fraction")
    if not isinstance(validation_fraction, (int, float)) or not 0 < validation_fraction < 1:
        raise ConfigError("evaluation.fine_tuning.validation_fraction must be between zero and one")

    tracking = config["tracking"]
    if not isinstance(tracking, Mapping):
        raise ConfigError("tracking must be a mapping")
    wandb = tracking.get("wandb")
    if not isinstance(wandb, Mapping):
        raise ConfigError("tracking.wandb must be a mapping")
    if not isinstance(wandb.get("enabled"), bool):
        raise ConfigError("tracking.wandb.enabled must be a boolean")
    if not isinstance(wandb.get("project"), str) or not wandb["project"].strip():
        raise ConfigError("tracking.wandb.project must be a non-empty string")
    if wandb.get("entity") is not None and (
        not isinstance(wandb["entity"], str) or not wandb["entity"].strip()
    ):
        raise ConfigError("tracking.wandb.entity must be null or a non-empty string")
    if wandb.get("mode") not in {"online", "offline", "disabled"}:
        raise ConfigError("tracking.wandb.mode must be online, offline, or disabled")
    tags = wandb.get("tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        raise ConfigError("tracking.wandb.tags must be a list of non-empty strings")


def write_config(config: Mapping[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(serializable, stream, sort_keys=False)
