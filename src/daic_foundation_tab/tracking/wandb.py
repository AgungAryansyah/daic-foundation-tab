from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .artifacts import RunArtifacts

_COHORT_FIELDS = (
    "participants",
    "depressed",
    "non_depressed",
    "positive_rate",
    "feature_count",
    "missing_cell_percentage",
    "complete_cases",
)
_EXCLUSION_FIELDS = ("participants", "excluded", "exclusion_fraction")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return "<redacted-path>"
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def _redact_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact_paths(item)
            for key, item in value.items()
            if not any(token in str(key).lower() for token in ("path", "root", "directory", "file"))
        }
    if isinstance(value, (list, tuple)):
        return [_redact_paths(item) for item in value]
    if isinstance(value, Path):
        return "<redacted-path>"
    if isinstance(value, str) and value.startswith(("/", "~")):
        return "<redacted-path>"
    return _json_value(value)


def _study_id(config: Mapping[str, Any]) -> str:
    data = config["data"]
    source_groups = data.get("source_groups", {})
    payload = {
        "task": config["experiment"]["task"],
        "data": {
            "missing_modality_policy": data.get("missing_modality_policy"),
            "max_exclusion_fraction": data.get("max_exclusion_fraction"),
            "source_groups": {
                name: {
                    "modality": group.get("modality"),
                    "parser": group.get("parser"),
                }
                for name, group in sorted(source_groups.items())
            },
        },
        "modalities": config["modalities"],
        "aggregation": config["aggregation"],
    }
    encoded = json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"))
    return f"study-{hashlib.sha256(encoded.encode()).hexdigest()[:16]}"


def sanitized_wandb_config(
    config: Mapping[str, Any], dataset_cache_key: str, study_id: str
) -> dict[str, Any]:
    data = config["data"]
    source_groups = data.get("source_groups", {})
    return {
        "project": {"seed": config["project"]["seed"]},
        "experiment": {
            "name": config["experiment"].get("name"),
            "task": config["experiment"]["task"],
            "feature_set": config["experiment"]["feature_set"],
        },
        "model": _redact_paths(config["model"]),
        "aggregation": _json_value(config["aggregation"]),
        "evaluation": _json_value(config["evaluation"]),
        "bootstrap": _json_value(config["bootstrap"]),
        "data": {
            "cache_key": dataset_cache_key,
            "study_id": study_id,
            "missing_modality_policy": data.get("missing_modality_policy"),
            "max_exclusion_fraction": data.get("max_exclusion_fraction"),
            "source_groups": {
                name: {
                    "modality": group.get("modality"),
                    "parser": group.get("parser"),
                }
                for name, group in sorted(source_groups.items())
            },
            "modalities": _json_value(config["modalities"]),
        },
    }


def _cohort_card(
    validation: Mapping[str, Any], dataset_cache_key: str, study_id: str, test_predictions: bool
) -> dict[str, Any]:
    statistics = validation["split_statistics"]
    exclusions = validation.get("exclusions", {})
    return {
        "dataset_cache_key": dataset_cache_key,
        "study_id": study_id,
        "feature_set": validation["feature_set"],
        "feature_count_before_filtering": validation["feature_count_before_filtering"],
        "dropped_all_nan_count": len(validation["dropped_all_nan"]),
        "dropped_constant_count": len(validation["dropped_constant"]),
        "near_constant_count": len(validation["near_constant"]),
        "splits": {
            split: {field: statistics[split].get(field) for field in _COHORT_FIELDS}
            for split in ("train", "dev", "test")
        },
        "exclusions": {
            split: {field: exclusions[split].get(field) for field in _EXCLUSION_FIELDS}
            for split in sorted(exclusions)
        },
        "test_predictions_generated": test_predictions,
    }


def _class_summary(target: pd.Series) -> dict[str, int]:
    values = target.astype(int)
    return {
        "participants": len(values),
        "depressed": int(values.sum()),
        "non_depressed": int((values == 0).sum()),
    }


def _scalar_values(prefix: str, value: Any) -> dict[str, float | int | bool | str]:
    if isinstance(value, Mapping):
        result: dict[str, float | int | bool | str] = {}
        for key, item in value.items():
            result.update(_scalar_values(f"{prefix}/{key}", item))
        return result
    if isinstance(value, bool):
        return {prefix: value}
    if isinstance(value, (int, float)):
        numeric = float(value)
        return {prefix: numeric if math.isfinite(numeric) else float("nan")}
    if isinstance(value, str):
        return {prefix: value}
    return {}


def _safe_model_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return _redact_paths(
        {
            key: metadata[key]
            for key in ("model_name", "package", "package_version", "checkpoint", "parameters")
            if key in metadata
        }
    )


@dataclass
class WandbTracker:
    artifacts: RunArtifacts
    enabled: bool
    mode: str
    project: str
    entity: str | None
    study_id: str
    data_card: dict[str, Any]
    config: dict[str, Any]
    run: Any | None = None
    sdk: Any | None = None
    artifact_references: list[dict[str, str]] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        config: Mapping[str, Any],
        artifacts: RunArtifacts,
        validation: Mapping[str, Any],
        dataset_cache_key: str,
    ) -> WandbTracker:
        settings = config["tracking"]["wandb"]
        study_id = _study_id(config)
        tracker = cls(
            artifacts=artifacts,
            enabled=bool(settings["enabled"]),
            mode=str(settings["mode"]),
            project=str(settings["project"]),
            entity=settings["entity"],
            study_id=study_id,
            data_card=_cohort_card(
                validation,
                dataset_cache_key,
                study_id,
                bool(config["evaluation"]["test_predictions"]),
            ),
            config=sanitized_wandb_config(config, dataset_cache_key, study_id),
        )
        if tracker.enabled and tracker.mode != "disabled":
            load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
            import wandb

            experiment = config["experiment"]
            run_id = artifacts.path.name
            experiment_name = str(experiment.get("name") or experiment["feature_set"])
            run_type = "smoke" if config["model"]["parameters"].get("epochs") == 1 else "full"
            tags = [
                str(config["model"]["name"]),
                str(experiment["feature_set"]),
                str(experiment["task"]),
                run_type,
                *settings["tags"],
            ]
            tracker.sdk = wandb
            tracker.run = wandb.init(
                project=tracker.project,
                entity=tracker.entity,
                name=f"{experiment_name}-{run_id}",
                group=tracker.study_id,
                job_type="fine_tuning",
                tags=list(dict.fromkeys(tags)),
                config=tracker.config,
                mode=tracker.mode,
                resume="never",
                dir=artifacts.path,
                settings=wandb.Settings(
                    disable_git=True,
                    save_code=False,
                    x_save_requirements=False,
                ),
            )
        tracker._write_run_record("running" if tracker.run is not None else "disabled")
        tracker._log("data", tracker.data_card)
        return tracker

    def _write_run_record(self, status: str) -> None:
        self.artifacts.json(
            "wandb_run.json",
            {
                "enabled": self.enabled,
                "mode": self.mode,
                "status": status,
                "project": self.project,
                "entity": self.entity,
                "run_id": getattr(self.run, "id", None),
                "url": getattr(self.run, "url", None),
                "study_id": self.study_id,
                "artifact_references": self.artifact_references,
            },
        )

    def _log(self, prefix: str, values: Mapping[str, Any]) -> None:
        if self.run is not None:
            self.run.log(_scalar_values(prefix, values))

    def record_fine_tuning(
        self,
        training_target: pd.Series,
        validation_target: pd.Series,
        feature_count: int,
        fit_seconds: float,
        metadata: Mapping[str, Any],
    ) -> None:
        fine_tuning = {
            "training": _class_summary(training_target),
            "validation": _class_summary(validation_target),
            "feature_count_after_selection": feature_count,
            "fit_seconds": fit_seconds,
            "selection_metric": metadata.get("selection_metric"),
            "best_validation_metric": metadata.get("best_validation_metric"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        }
        self.data_card["fine_tuning"] = _json_value(fine_tuning)
        self.results["fine_tuning"] = _json_value(fine_tuning)
        self._log("fine_tuning", fine_tuning)

    def record_development(self, metrics: Mapping[str, Any]) -> None:
        scalar_metrics = _scalar_values("", metrics)
        development = {key.removeprefix("/"): value for key, value in scalar_metrics.items()}
        self.results["development"] = development
        self._log("development", development)

    def record_bootstrap(self, summary: Mapping[str, Any]) -> None:
        self.results["bootstrap"] = _json_value(summary)
        self._log("bootstrap", summary)

    def record_repeated_holdout(
        self, metrics: pd.DataFrame, summary: Mapping[str, Any]
    ) -> None:
        records = _json_value(metrics.to_dict(orient="records"))
        self.results["repeated_holdout"] = {"metrics": records, "summary": _json_value(summary)}
        self._log("repeated_holdout", summary)
        if self.run is not None and not metrics.empty:
            table = self.sdk.Table(
                columns=[str(column) for column in metrics.columns],
                data=[[_json_value(value) for value in row] for row in metrics.itertuples(index=False)],
            )
            self.run.log({"repeated_holdout/metrics": table})

    def complete(
        self,
        model_metadata: Mapping[str, Any],
        environment: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> None:
        self.results["model"] = _safe_model_metadata(model_metadata)
        self.results["environment"] = _redact_paths(environment)
        self.results["runtime"] = _redact_paths(runtime)
        self._log("runtime", runtime)
        research_directory = self.artifacts.path / "wandb_research"
        research_directory.mkdir(exist_ok=True)
        (research_directory / "data_card.json").write_text(
            json.dumps(self.data_card, indent=2, allow_nan=True), encoding="utf-8"
        )
        (research_directory / "settings.json").write_text(
            json.dumps(self.config, indent=2, allow_nan=True), encoding="utf-8"
        )
        (research_directory / "results.json").write_text(
            json.dumps(self.results, indent=2, allow_nan=True), encoding="utf-8"
        )
        if self.run is not None:
            artifact = self.sdk.Artifact(
                name=f"research-record-{self.artifacts.path.name}",
                type="research-record",
                description="Sanitized cohort-level DAIC-WOZ fine-tuning record",
                metadata={
                    "study_id": self.study_id,
                    "dataset_cache_key": self.data_card["dataset_cache_key"],
                },
            )
            for path in sorted(research_directory.iterdir()):
                artifact.add_file(str(path), name=path.name)
            self.run.log_artifact(artifact)
            self.artifact_references.append({"name": artifact.name, "type": artifact.type})
            self._write_run_record("finished")
            self.run.finish(exit_code=0)
        else:
            self._write_run_record("disabled")

    def fail(self) -> None:
        self._write_run_record("failed")
        if self.run is not None:
            with suppress(Exception):
                self.run.finish(exit_code=1)
