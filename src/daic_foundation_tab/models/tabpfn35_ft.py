from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .tabiclv2_ft import _EpochLogger


class TabPFN35FineTuningError(RuntimeError):
    pass


class _TabPFNEpochLogger(_EpochLogger):
    def __init__(self, callback: Callable[[Mapping[str, float]], None]) -> None:
        super().__init__(callback)
        self._learning_rate: float | None = None

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        self._learning_rate = metrics.get("train/lr", self._learning_rate)

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        names = {"val/ROC AUC": "val/roc_auc", "val/MSE": "val/mse"}
        normalized = {names.get(key, key): value for key, value in metrics.items()}
        if self._learning_rate is not None:
            normalized["train/lr"] = self._learning_rate
        super().log_epoch(normalized, step)


class TabPFN35FineTunedModel:
    _regression = False

    def __init__(self, model_config: dict[str, Any]) -> None:
        self._model_config = model_config
        self._parameters = dict(model_config["parameters"])
        self._estimator: Any | None = None
        self._checkpoint: Path | None = None
        self._best_metric: float | None = None
        self._baseline_metric: float | None = None
        self._best_epoch = -1

    def _constructor_parameters(self) -> dict[str, Any]:
        parameters = dict(self._parameters)
        if parameters.get("device") != "cuda:0":
            raise TabPFN35FineTuningError("GPU-only TabPFN fine-tuning requires device='cuda:0'")
        if self._model_config.get("model_version") != "v3.5":
            raise TabPFN35FineTuningError("TabPFN experiments require model_version='v3.5'")
        if parameters.get("early_stopping", True) is not True:
            raise TabPFN35FineTuningError("Early stopping is required to retain selected weights")
        metric = "mse" if self._regression else "roc_auc"
        if parameters.get("eval_metric") != metric:
            raise TabPFN35FineTuningError(f"TabPFN experiment requires eval_metric='{metric}'")
        parameters["model_version"] = "v3.5"
        return parameters

    def validate_input(self, features: pd.DataFrame) -> None:
        if features.empty or not features.columns.is_unique:
            raise TabPFN35FineTuningError("TabPFN requires nonempty features with unique columns")

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        checkpoint_directory: Path,
        epoch_callback: Callable[[Mapping[str, float]], None] | None = None,
    ) -> Self:
        self.validate_input(features)
        self.validate_input(validation_features)
        if list(features.columns) != list(validation_features.columns):
            raise TabPFN35FineTuningError("Validation columns must match training columns")
        for inputs, values in ((features, target), (validation_features, validation_target)):
            if len(inputs) != len(values):
                raise TabPFN35FineTuningError("Targets must align with feature rows")
            if not np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)).all():
                raise TabPFN35FineTuningError("Targets must be finite numeric values")
            if not self._regression and set(values) != {0, 1}:
                raise TabPFN35FineTuningError("Training and validation must both contain classes 0 and 1")
        parameters = self._constructor_parameters()
        if checkpoint_directory.exists() and any(checkpoint_directory.glob("*.pth")):
            raise TabPFN35FineTuningError("Use a fresh checkpoint directory for each experiment")
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self._estimator = None
        self._checkpoint = None
        self._best_metric = None
        self._baseline_metric = None
        self._best_epoch = -1
        checkpoint_metrics: dict[str, float] = {}
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        try:
            from tabpfn.constants import ModelVersion
            from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor
            from tabpfn.finetuning.train_util import get_checkpoint_name
            from tabpfn.model_loading import save_tabpfn_model

            fine_tuner = FinetunedTabPFNRegressor if self._regression else FinetunedTabPFNClassifier
            adapter = self

            class _TrackedFineTuner(fine_tuner):
                def _log_epoch_evaluation(self, epoch, eval_result, mean_train_loss) -> None:
                    nonlocal checkpoint_metrics
                    metric = float(eval_result.primary)
                    # The public logger omits the baseline used for native checkpoint selection.
                    if epoch == -1:
                        if not math.isfinite(metric):
                            raise TabPFN35FineTuningError("Initial validation metric must be finite")
                        adapter._baseline_metric = metric
                    if math.isfinite(metric) and (
                        adapter._best_metric is None
                        or self._is_improvement(metric, adapter._best_metric)
                    ):
                        adapter._best_metric = metric
                        adapter._best_epoch = epoch
                        checkpoint_metrics = self._get_checkpoint_metrics(eval_result)
                    super()._log_epoch_evaluation(epoch, eval_result, mean_train_loss)

            parameters["model_version"] = ModelVersion.V3_5
            if epoch_callback is not None:
                parameters["experiment_logger"] = _TabPFNEpochLogger(epoch_callback)
            estimator = _TrackedFineTuner(**parameters)
            estimator.fit(
                features,
                target.astype(float if self._regression else int),
                X_val=validation_features,
                y_val=validation_target.astype(float if self._regression else int),
                output_dir=checkpoint_directory,
            )
            if self._best_metric is None:
                raise TabPFN35FineTuningError("Fine-tuning produced no finite validation metric")
            checkpoint = checkpoint_directory / get_checkpoint_name(len(features), is_best=True)
            save_tabpfn_model(
                estimator.finetuned_estimator_,
                checkpoint,
                additional_fields={"epoch": self._best_epoch + 1, **checkpoint_metrics},
            )
        except ImportError as error:
            raise TabPFN35FineTuningError(
                "TabPFN dependencies are unavailable. Run 'uv sync --group dev'."
            ) from error
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise TabPFN35FineTuningError(
                    "TabPFN fine-tuning ran out of memory; use a GPU with sufficient VRAM "
                    "for the configured experiment."
                ) from error
            raise
        self._estimator = estimator
        self._checkpoint = checkpoint
        return self

    def predict(self, features: pd.DataFrame) -> object:
        if self._estimator is None:
            raise TabPFN35FineTuningError("Call fit before predict")
        self.validate_input(features)
        return self._estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> object:
        if self._estimator is None:
            raise TabPFN35FineTuningError("Call fit before predict_proba")
        self.validate_input(features)
        return self._estimator.predict_proba(features)

    def run_metadata(self) -> dict[str, Any]:
        try:
            package_version = version("tabpfn")
        except PackageNotFoundError:
            package_version = None
        return {
            "model_name": "TabPFN-3.5-FT" + (" Regressor" if self._regression else ""),
            "package": "tabpfn",
            "package_version": package_version,
            "model_version": "v3.5",
            "checkpoint": self._model_config["checkpoint_version"],
            "parameters": self._constructor_parameters(),
        }

    def finetune_metadata(self) -> dict[str, Any]:
        if self._checkpoint is None or not self._checkpoint.is_file():
            raise TabPFN35FineTuningError("Selected checkpoint is unavailable; call fit first")
        with self._checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "checkpoint_path": str(self._checkpoint),
            "checkpoint_sha256": digest,
            "selection_metric": self._parameters["eval_metric"],
            "best_validation_metric": self._best_metric,
            "baseline_validation_metric": self._baseline_metric,
            "selected_epoch": self._best_epoch + 1,
        }


class TabPFN35FineTunedRegressor(TabPFN35FineTunedModel):
    _regression = True
