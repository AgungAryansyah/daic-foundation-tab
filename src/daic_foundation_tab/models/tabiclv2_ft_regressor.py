from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .tabiclv2_ft import TabICLv2FineTunedModel, TabICLv2FineTuningError, _EpochLogger


class TabICLv2FineTunedRegressor(TabICLv2FineTunedModel):
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        checkpoint_directory: Path,
        epoch_callback: Callable[[Mapping[str, float]], None] | None = None,
    ) -> TabICLv2FineTunedRegressor:
        self.validate_input(features)
        self.validate_input(validation_features)
        if list(features.columns) != list(validation_features.columns):
            raise TabICLv2FineTuningError(
                "Fine-tuning validation columns must match training columns"
            )
        if len(features) != len(target) or len(validation_features) != len(validation_target):
            raise TabICLv2FineTuningError("Regression targets must align with feature rows")
        for values in (target, validation_target):
            if not np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)).all():
                raise TabICLv2FineTuningError("Regression targets must be finite")
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        try:
            from tabicl import FinetunedTabICLRegressor

            if epoch_callback is None:
                self._estimator = FinetunedTabICLRegressor(**self._constructor_parameters())
            else:
                epoch_logger = _EpochLogger(epoch_callback)

                class _TrackedFinetunedTabICLRegressor(FinetunedTabICLRegressor):
                    def _make_experiment_logger(self) -> _EpochLogger:
                        return epoch_logger

                self._estimator = _TrackedFinetunedTabICLRegressor(**self._constructor_parameters())
            self._estimator.fit(
                features,
                target.astype(float),
                X_val=validation_features,
                y_val=validation_target.astype(float),
                output_dir=checkpoint_directory,
            )
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise TabICLv2FineTuningError(
                    "TabICLv2 regression fine-tuning ran out of memory. Lower "
                    "n_estimators_finetune and record the changed configuration."
                ) from error
            raise
        except ImportError as error:
            raise TabICLv2FineTuningError(
                "TabICLv2 fine-tuning dependencies are unavailable. Run 'uv sync --group dev'."
            ) from error
        self._checkpoint_directory = checkpoint_directory
        return self

    def run_metadata(self) -> dict[str, Any]:
        return {**super().run_metadata(), "model_name": "TabICLv2-FT Regressor"}

    def finetune_metadata(self) -> dict[str, Any]:
        metadata = super().finetune_metadata()
        if metadata["selection_metric"] == "mae":
            metadata["best_validation_metric"] = -metadata["best_validation_metric"]
        return metadata
