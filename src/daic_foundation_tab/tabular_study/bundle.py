from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import pandas as pd

from daic_foundation_tab.data.dataset import build_or_load_dataset
from daic_foundation_tab.data.labels import load_official_labels
from daic_foundation_tab.data.participants import fingerprint, prepare_manifest
from daic_foundation_tab.tabular_study.configuration import condition_specs, load_study_config
from daic_foundation_tab.tabular_study.folds import make_folds
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase

BUNDLE_FILES = ("features.parquet", "features.csv", "participants.json", "folds.json")


def _json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def software_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src").rglob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    return fingerprint([(str(path.relative_to(root)), file_hash(path)) for path in paths])


def _logical_config(config: dict) -> dict:
    result = deepcopy(config)
    result["data"].pop("root", None)
    result["data"].pop("test_ground_truth", None)
    for key in ("cache_root", "output_root"):
        result["project"].pop(key, None)
    return result


def _study_id(config: dict, files: dict, software: str) -> str:
    core = deepcopy(config)
    core["tabular_study"].pop("model_seeds")
    return fingerprint({"configuration": core, "files": files, "software": software})


def _study_record(config: dict, directory: Path, software: str) -> dict:
    files = {name: file_hash(directory / name) for name in BUNDLE_FILES}
    study_id = _study_id(config, files, software)
    conditions = {
        stage: [
            {**condition, "condition_id": fingerprint({"study_id": study_id, **condition})}
            for condition in condition_specs(config, stage)
        ]
        for stage in ("cv", "final")
    }
    return {
        "version": 1,
        "study_id": study_id,
        "software_identity": software,
        "configuration": config,
        "files": files,
        "conditions": conditions,
        "schedule_id": fingerprint(conditions),
    }


def verify_bundle(directory: str | Path, *, verify_software: bool = True) -> dict:
    directory = Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    if study.get("version") != 1 or set(study["files"]) != set(BUNDLE_FILES):
        raise ValueError("Unsupported or incomplete tabular study bundle")
    if any(file_hash(directory / name) != digest for name, digest in study["files"].items()):
        raise ValueError("Prepared feature bundle content changed")
    expected = _study_record(study["configuration"], directory, study["software_identity"])
    if study != expected:
        raise ValueError("Study configuration or condition identity changed")
    if verify_software and study["software_identity"] != software_identity():
        raise ValueError("Executed source or dependency lock differs from the prepared study")
    return study


def prepare_bundle(
    config_path: str | Path, destination: str | Path, *, reuse: str | Path | None = None
) -> Path:
    config = load_study_config(config_path)
    logical = _logical_config(config)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Use an empty destination for study preparation")
    logger = configure_run_logger(
        destination / "prepare.log", config.get("logging", {}).get("level", "INFO")
    )
    software = software_identity()
    if reuse:
        source = Path(reuse).resolve()
        old = verify_bundle(source)
        before, after = deepcopy(old["configuration"]), deepcopy(logical)
        before["tabular_study"].pop("model_seeds")
        after["tabular_study"].pop("model_seeds")
        if before != after:
            raise ValueError("Only model seeds may change when reusing a prepared bundle")
        with phase(logger, "copying prepared features") as started:
            for completed, name in enumerate(BUNDLE_FILES, 1):
                shutil.copy2(source / name, destination / name)
                log_progress(
                    logger, "copying prepared features", completed, len(BUNDLE_FILES), started
                )
    else:
        working = deepcopy(config)
        working.pop("tabular_study", None)
        working.pop("profiles", None)
        working["input"] = {"kind": "tabular"}
        working["study"] = {"required_inputs": ["tabular"], "feature_set": "audio_visual"}
        working["experiment"].update(
            task=config["tabular_study"]["tasks"][0], feature_set="audio_visual"
        )
        working["evaluation"]["fine_tuning"].update(
            validation_fraction=config["tabular_study"]["selection_fraction"],
            validation_seed=config["tabular_study"]["fold_seed"],
        )
        with phase(logger, "preparing shared participants") as started:
            manifest = prepare_manifest(
                working,
                lambda n, total: log_progress(logger, "checking input files", n, total, started),
            )
        labels = load_official_labels(working["data"]).table
        for split in ("train", "dev", "test"):
            total = int((labels["split"] == split).sum())
            eligible = sum(row["split"] == split for row in manifest["participants"])
            if total == 0 or (total - eligible) / total > float(
                working["data"].get("max_exclusion_fraction", 0.05)
            ):
                raise ValueError(f"Shared {split} cohort exceeds max_exclusion_fraction")
        _json(destination / "participants.json", manifest)
        working["experiment"]["partition_manifest"] = str(destination / "participants.json")
        with tempfile.TemporaryDirectory(prefix="daic-study-cache-") as cache:
            working["project"]["cache_root"] = cache
            with phase(logger, "preparing numeric features") as started:
                dataset = build_or_load_dataset(
                    working,
                    lambda n, total: log_progress(logger, "preparing features", n, total, started),
                )
        table = dataset.table.copy()
        if (
            table.loc[table["split"] == "test", ["target_binary", "target_phq8"]]
            .notna()
            .any()
            .any()
        ):
            raise ValueError("Prepared bundle must not contain test targets")
        table.to_parquet(destination / "features.parquet", index=False)
        dataset.manifest.to_csv(destination / "features.csv", index=False)
        with phase(logger, "preparing participant folds"):
            folds = make_folds(table, config["tabular_study"])
        _json(destination / "folds.json", folds)
    study = _study_record(logical, destination, software)
    _json(destination / "study.json", study)
    logger.info(
        "Prepared %d CV conditions in %d groups; no training started",
        len(study["conditions"]["cv"]),
        len(config["tabular_study"]["model_families"]) * len(config["tabular_study"]["modalities"]),
    )
    return destination


def read_bundle(directory: str | Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    directory = Path(directory).resolve()
    study = verify_bundle(directory)
    return (
        study,
        pd.read_parquet(directory / "features.parquet"),
        pd.read_csv(directory / "features.csv"),
        json.loads((directory / "folds.json").read_text(encoding="utf-8")),
    )
