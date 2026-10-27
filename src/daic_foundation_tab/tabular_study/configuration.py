from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path

from daic_foundation_tab.config import ConfigError, deep_merge, load_config, validate_config

OPTIONS = {
    "model_families": {"tabpfn35", "kumo_medium"},
    "modalities": {"audio", "visual", "audio_visual"},
    "tasks": {"classification", "regression"},
    "methods": {"frozen_icl", "fine_tuning"},
}


def load_study_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = load_config(path, _validate=False)
    settings = config.get("tabular_study")
    if (
        not isinstance(settings, dict)
        or not isinstance(settings.get("name"), str)
        or not settings["name"].strip()
    ):
        raise ConfigError("tabular_study.name must be a nonempty string")
    if not config.get("data", {}).get("dataset_id"):
        raise ConfigError("data.dataset_id is required")
    if "label_budget" in config.get("study", {}):
        raise ConfigError("Label budgets are not part of this study")
    for key, allowed in OPTIONS.items():
        values = settings.get(key)
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
            or not set(values) <= allowed
        ):
            raise ConfigError(f"tabular_study.{key} must contain unique supported values")
    seeds = settings.get("model_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ConfigError("tabular_study.model_seeds must contain unique nonnegative integers")
    for key, minimum in (
        ("folds", 2),
        ("fold_seed", 0),
        ("selection_seed_offset", 0),
        ("bootstrap_iterations", 1),
        ("bootstrap_seed", 0),
    ):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigError(f"tabular_study.{key} must be an integer >= {minimum}")
    fraction = settings.get("selection_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ConfigError("tabular_study.selection_fraction must be between zero and one")
    profile_dir = (path.parent / settings["tabular_profiles"]).resolve()
    profiles = {}
    for family, task, mode in product(
        settings["model_families"], settings["tasks"], settings["methods"]
    ):
        method = "icl" if mode == "frozen_icl" else "ft"
        profile = load_config(profile_dir / f"{family}_{method}.yaml", _validate=False)
        if task == "regression":
            profile = deep_merge(
                profile,
                load_config(profile_dir / f"{family}_{method}_regressor.yaml", _validate=False),
            )
        model = profile["model"]
        profiles[f"{family}:{task}:{mode}"] = model
        candidate = deepcopy(config)
        candidate["model"] = deepcopy(model)
        candidate["experiment"].update(task=task, mode=mode, feature_set="audio_visual")
        candidate["evaluation"]["threshold"] = 0.5 if task == "classification" else None
        validate_config(candidate)
    config.pop("_config_path", None)
    config["tabular_study"] = {
        key: deepcopy(value) for key, value in settings.items() if key != "tabular_profiles"
    }
    config["profiles"] = profiles
    return config


def condition_specs(config: dict, stage: str = "cv") -> list[dict]:
    if stage not in {"cv", "final"}:
        raise ConfigError("Stage must be cv or final")
    settings = config["tabular_study"]
    folds = range(1, settings["folds"] + 1) if stage == "cv" else [None]
    conditions = []
    for family, modality, task, mode, seed, fold in product(
        settings["model_families"],
        settings["modalities"],
        settings["tasks"],
        settings["methods"],
        settings["model_seeds"],
        folds,
    ):
        conditions.append(
            {
                "stage": stage,
                "family": family,
                "modality": modality,
                "task": task,
                "mode": mode,
                "model_seed": seed,
                "fold": fold,
            }
        )
    return conditions
