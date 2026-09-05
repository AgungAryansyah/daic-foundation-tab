from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd


class TabICLv2FineTuningError(RuntimeError):
    pass


class _EpochLogger:
    def __init__(self, callback: Callable[[Mapping[str, float]], None]) -> None:
        self._callback = callback

    def setup(self, config: dict[str, Any]) -> None:
        del config

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        del metrics, step

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        del step
        self._callback(metrics)

    def finish(self) -> None:
        pass


class TabICLv2FineTunedModel:
    def __init__(self, model_config: dict[str, Any]) -> None:
        self._model_config = model_config
        self._parameters = dict(model_config["parameters"])
        self._estimator: Any | None = None
        self._checkpoint_directory: Path | None = None

    def _constructor_parameters(self) -> dict[str, Any]:
        parameters = dict(self._parameters)
        if parameters.get("device") != "cuda:0":
            raise TabICLv2FineTuningError("GPU-only TabICLv2 fine-tuning requires device='cuda:0'")
        if parameters.get("amp") is not True:
            raise TabICLv2FineTuningError("GPU-only TabICLv2 fine-tuning requires amp=True")
        parameters["checkpoint_version"] = self._model_config["checkpoint_version"]
        return parameters

    def validate_input(self, features: pd.DataFrame) -> None:
        if features.empty:
            raise TabICLv2FineTuningError("TabICLv2 fine-tuning requires at least one feature column")
        if not features.columns.is_unique:
            raise TabICLv2FineTuningError("TabICLv2 fine-tuning input columns must be unique")

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        checkpoint_directory: Path,
        epoch_callback: Callable[[Mapping[str, float]], None] | None = None,
    ) -> TabICLv2FineTunedModel:
        self.validate_input(features)
        self.validate_input(validation_features)
        if list(features.columns) != list(validation_features.columns):
            raise TabICLv2FineTuningError("Fine-tuning validation columns must match training columns")
        if target.nunique() != 2 or validation_target.nunique() != 2:
            raise TabICLv2FineTuningError(
                "Fine-tuning and early-stopping partitions must both contain two classes"
            )
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        try:
            from tabicl import FinetunedTabICLClassifier

            if epoch_callback is None:
                self._estimator = FinetunedTabICLClassifier(**self._constructor_parameters())
            else:
                epoch_logger = _EpochLogger(epoch_callback)

                class _TrackedFinetunedTabICLClassifier(FinetunedTabICLClassifier):
                    def _make_experiment_logger(self) -> _EpochLogger:
                        # This private TabICL hook is the only available path to its epoch metrics.
                        return epoch_logger

                self._estimator = _TrackedFinetunedTabICLClassifier(
                    **self._constructor_parameters()
                )
            self._estimator.fit(
                features,
                target,
                X_val=validation_features,
                y_val=validation_target,
                output_dir=checkpoint_directory,
            )
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise TabICLv2FineTuningError(
                    "TabICLv2 fine-tuning ran out of memory. Lower n_estimators_finetune and record "
                    "the changed configuration."
                ) from error
            raise
        except ImportError as error:
            raise TabICLv2FineTuningError(
                "TabICLv2 fine-tuning dependencies are unavailable. Run 'uv sync --group dev'."
            ) from error
        self._checkpoint_directory = checkpoint_directory
        return self

    def predict(self, features: pd.DataFrame) -> object:
        if self._estimator is None:
            raise TabICLv2FineTuningError("Call fit before predict")
        self.validate_input(features)
        return self._estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> object:
        if self._estimator is None:
            raise TabICLv2FineTuningError("Call fit before predict_proba")
        self.validate_input(features)
        return self._estimator.predict_proba(features)

    def run_metadata(self) -> dict[str, Any]:
        try:
            package_version = version("tabicl")
        except PackageNotFoundError:
            package_version = None
        return {
            "model_name": "TabICLv2-FT",
            "package": "tabicl",
            "package_version": package_version,
            "checkpoint": self._model_config["checkpoint_version"],
            "parameters": self._constructor_parameters(),
        }

    def finetune_metadata(self) -> dict[str, Any]:
        if self._estimator is None or self._checkpoint_directory is None:
            raise TabICLv2FineTuningError("Call fit before requesting fine-tuning metadata")
        checkpoint = self._checkpoint_directory / "best.ckpt"
        if not checkpoint.is_file():
            raise TabICLv2FineTuningError(f"Fine-tuning checkpoint was not saved: {checkpoint}")
        hasher = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(chunk)
        return {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hasher.hexdigest(),
            "selection_metric": self._parameters["eval_metric"],
            "best_validation_metric": float(self._estimator._best_metric_),
        }
