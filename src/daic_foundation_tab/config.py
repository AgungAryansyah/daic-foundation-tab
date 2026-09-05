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
    for section in ("project", "data", "aggregation", "experiment", "model", "evaluation"):
        if section not in config:
            raise ConfigError(f"Missing required configuration section: {section}")

    if config["experiment"].get("task") != "classification":
        raise ConfigError("Phase 1 supports classification only")
    if config["experiment"].get("feature_set") not in {"audio", "visual", "audio_visual"}:
        raise ConfigError("Phase 1 feature_set must be audio, visual, or audio_visual")
    if config["model"].get("name") != "tabiclv2":
        raise ConfigError("Phase 1 supports the tabiclv2 model only")


def write_config(config: Mapping[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(serializable, stream, sort_keys=False)
