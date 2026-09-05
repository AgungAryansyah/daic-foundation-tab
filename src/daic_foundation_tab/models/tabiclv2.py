from __future__ import annotations

import gc
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd


class TabICLv2Error(RuntimeError):
    pass


class TabICLv2Model:
    def __init__(self, model_config: dict[str, Any]) -> None:
        self._model_config = model_config
        self._parameters = dict(model_config["parameters"])
        self._estimator: Any | None = None

    def _constructor_parameters(self) -> dict[str, Any]:
        parameters = dict(self._parameters)
        parameters.pop("auto_recover_oom", None)
        if parameters.get("device") == "auto":
            parameters["device"] = None
        parameters["checkpoint_version"] = self._model_config["checkpoint_version"]
        return parameters

    def validate_input(self, features: pd.DataFrame) -> None:
        if features.empty:
            raise TabICLv2Error("TabICLv2 requires at least one feature column")
        if not features.columns.is_unique:
            raise TabICLv2Error("TabICLv2 input feature columns must be unique")

    def fit(self, features: pd.DataFrame, target: pd.Series) -> TabICLv2Model:
        self.validate_input(features)
        if target.nunique() != 2:
            raise TabICLv2Error("TabICLv2 classification context must contain both classes")
        try:
            from tabicl import TabICLClassifier

            self._estimator = TabICLClassifier(**self._constructor_parameters())
            self._estimator.fit(features, target)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                self._clear_cuda_memory()
                raise TabICLv2Error(
                    "TabICLv2 ran out of memory. Lower batch_size first, then n_estimators, and record "
                    "the changed configuration."
                ) from error
            raise
        except ImportError as error:
            raise TabICLv2Error("The tabicl package is required. Run 'uv sync --group dev'.") from error
        return self

    def predict(self, features: pd.DataFrame) -> object:
        if self._estimator is None:
            raise TabICLv2Error("Call fit before predict")
        self.validate_input(features)
        return self._estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> object:
        if self._estimator is None:
            raise TabICLv2Error("Call fit before predict_proba")
        self.validate_input(features)
        return self._estimator.predict_proba(features)

    def run_metadata(self) -> dict[str, Any]:
        try:
            package_version = version("tabicl")
        except PackageNotFoundError:
            package_version = None
        return {
            "model_name": "TabICLv2",
            "package": "tabicl",
            "package_version": package_version,
            "checkpoint": self._model_config["checkpoint_version"],
            "parameters": self._constructor_parameters(),
        }

    @staticmethod
    def _clear_cuda_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return
