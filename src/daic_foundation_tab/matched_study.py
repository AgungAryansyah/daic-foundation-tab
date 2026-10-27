from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from daic_foundation_tab.config import deep_merge, load_config, validate_config, write_config
from daic_foundation_tab.data.labels import load_official_labels
from daic_foundation_tab.data.participants import fingerprint, prepare_manifest
from daic_foundation_tab.data.transcripts import load_transcripts
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase

FAMILIES = ("tabpfn35", "tabiclv2", "kumo_medium", "qwen35_2b")
TASKS = ("classification", "regression")
MODES = ("frozen_icl", "fine_tuning")


def condition_configs(path: str | Path) -> list[tuple[str, dict]]:
    path = Path(path).resolve()
    base = load_config(path, _validate=False)
    settings = base.pop("matched_study")
    if "label_budget" in base.get("study", {}):
        raise ValueError("Label budgets are not part of this study")
    if not base["data"].get("dataset_id"):
        raise ValueError("Matched studies require data.dataset_id")
    profiles = (path.parent / settings["tabular_profiles"]).resolve()
    text_profile = load_config(path.parent / settings["text_profile"])
    conditions = []
    for task in TASKS:
        for family in FAMILIES:
            for mode in MODES:
                name = f"{task}_{family}_{mode}"
                config = deepcopy(base)
                config.pop("_config_path", None)
                config["experiment"].update(
                    name=name, task=task, mode=mode, feature_set="audio_visual"
                )
                config["evaluation"]["threshold"] = 0.5 if task == "classification" else None
                config["evaluation"]["test_predictions"] = True
                config["evaluation"]["repeated_holdout"]["enabled"] = False
                config["bootstrap"]["enabled"] = False
                config["study"] = {
                    "campaign_id": settings["name"],
                    "required_inputs": ["tabular", "transcript"],
                    "feature_set": "audio_visual",
                }
                is_text = family == "qwen35_2b"
                config["input"] = {"kind": "transcript" if is_text else "tabular"}
                if is_text:
                    config["model"] = deepcopy(text_profile["model"])
                    config["model"]["parameters"].update(
                        max_length=settings["text_max_length"],
                        demonstrations=settings["demonstrations"] if mode == "frozen_icl" else 0,
                        epochs=settings["text_epochs"],
                    )
                else:
                    method = "icl" if mode == "frozen_icl" else "ft"
                    suffix = "_regressor" if task == "regression" else ""
                    profile = load_config(profiles / f"{family}_{method}.yaml", _validate=False)
                    if suffix:
                        profile = deep_merge(profile, load_config(
                            profiles / f"{family}_{method}{suffix}.yaml", _validate=False
                        ))
                    config = deep_merge(config, {"model": profile["model"]})
                    config["model"]["parameters"]["random_state"] = config["project"]["seed"]
                config["data"]["root"] = str(Path(config["data"]["root"]).expanduser().resolve())
                for key in ("cache_root", "output_root"):
                    config["project"][key] = str(
                        Path(config["project"][key]).expanduser().resolve()
                    )
                validate_config(config)
                conditions.append((name, config))
    return conditions


def _save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare_study(config_path: str | Path, destination: str | Path) -> Path:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    logger = configure_run_logger(destination / "prepare.log", "INFO")
    conditions = condition_configs(config_path)
    identity = fingerprint(conditions)
    plan_path = destination / "campaign.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["config_identity"] != identity:
            raise ValueError("Study settings changed. Use a new study directory")
        verify_prepared_study(destination)
        return destination
    if any(destination.glob("*.yaml")) or any(destination.glob("participants_*.json")):
        raise ValueError("Study preparation is incomplete. Use a new study directory")
    manifests = {}
    for task in TASKS:
        config = next(config for _, config in conditions if config["experiment"]["task"] == task)
        with phase(logger, f"preparing {task} participants") as started:
            manifest = prepare_manifest(config, lambda n, total, task=task: log_progress(
                logger, f"preparing {task} participants", n, total, started
            ))
        maximum = float(config["data"]["max_exclusion_fraction"])
        labels = load_official_labels(config["data"]).table
        for split in ("train", "dev", "test"):
            eligible = sum(row["split"] == split for row in manifest["participants"])
            total = int((labels["split"] == split).sum())
            if (total - eligible) / total > maximum:
                raise ValueError(f"Shared {split} cohort exceeds max_exclusion_fraction")
        manifest_path = destination / f"participants_{task}.json"
        _save_json(manifest_path, manifest)
        manifests[task] = manifest
        text_config = next(
            config for _, config in conditions
            if config["experiment"]["task"] == task and config["input"]["kind"] == "transcript"
        )
        text_config = deepcopy(text_config)
        text_config["experiment"]["partition_manifest"] = str(manifest_path)
        with phase(logger, f"validating {task} transcripts") as started:
            load_transcripts(text_config, lambda n, total, task=task: log_progress(
                logger, f"validating {task} transcripts", n, total, started
            ))
    entries = []
    for name, config in conditions:
        task = config["experiment"]["task"]
        manifest = manifests[task]
        config["experiment"]["partition_manifest"] = str(destination / f"participants_{task}.json")
        config["project"]["output_root"] = str(destination / "runs" / name)
        config["study"].update(
            manifest_id=manifest["manifest_id"],
            cohort_id=fingerprint([
                (str(row["participant_id"]), row["split"], row["role"])
                for row in manifest["participants"]
            ]),
        )
        config["tracking"]["wandb"]["group"] = (
            f"{config['study']['campaign_id']}-{task}-seed{config['project']['seed']}-"
            f"{manifest['manifest_id'][:12]}"
        )
        config["tracking"]["wandb"].setdefault("tags", []).append("matched-adaptation")
        path = destination / f"{name}.yaml"
        write_config(config, path)
        entries.append({"name": name, "config": path.name, "identity": fingerprint(config)})
    _save_json(plan_path, {
        "version": 1, "config_identity": identity, "conditions": entries,
        "manifests": {task: value["manifest_id"] for task, value in manifests.items()},
    })
    logger.info("Prepared %d/%d conditions (100%%). No model training started", len(entries), len(entries))
    return destination


def verify_prepared_study(destination: str | Path) -> list[tuple[dict, dict]]:
    from daic_foundation_tab.data.participants import load_manifest

    destination = Path(destination).resolve()
    plan = json.loads((destination / "campaign.json").read_text())
    conditions = []
    if len(plan["conditions"]) != len(TASKS) * len(FAMILIES) * len(MODES):
        raise ValueError("Study must contain all 16 conditions")
    for entry in plan["conditions"]:
        config = load_config(destination / entry["config"])
        payload = {key: value for key, value in config.items() if not key.startswith("_")}
        if fingerprint(payload) != entry["identity"]:
            raise ValueError(f"Prepared condition changed: {entry['name']}")
        manifest = load_manifest(config["experiment"]["partition_manifest"], config)
        if manifest["manifest_id"] != plan["manifests"][config["experiment"]["task"]]:
            raise ValueError("Prepared participant manifest changed")
        conditions.append((entry, config))
    return conditions
