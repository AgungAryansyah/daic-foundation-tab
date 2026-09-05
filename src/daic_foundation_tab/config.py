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
    for section in (
        "project",
        "data",
        "aggregation",
        "experiment",
        "model",
        "evaluation",
        "tracking",
    ):
        if section not in config:
            raise ConfigError(f"Missing required configuration section: {section}")

    if config["experiment"].get("task") != "classification":
        raise ConfigError("Phase 1 supports classification only")
    if config["experiment"].get("feature_set") not in {"audio", "visual", "audio_visual"}:
        raise ConfigError("Phase 1 feature_set must be audio, visual, or audio_visual")
    if config["model"].get("name") != "tabiclv2_ft":
        raise ConfigError("Only tabiclv2_ft is supported; the standalone tabiclv2 ICL path was retired")
    if config["evaluation"].get("test_predictions", False):
        raise ConfigError("TabICLv2-FT test predictions require a separate frozen finalization workflow")
    fine_tuning = config["evaluation"].get("fine_tuning")
    if not isinstance(fine_tuning, Mapping):
        raise ConfigError("TabICLv2-FT requires evaluation.fine_tuning settings")
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
