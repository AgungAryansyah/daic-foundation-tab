from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, Self

import pandas as pd


class FineTunableClassifier(Protocol):
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        checkpoint_directory: Path,
        epoch_callback: Callable[[Mapping[str, float]], None] | None = None,
    ) -> Self: ...

    def predict(self, features: pd.DataFrame) -> object: ...

    def predict_proba(self, features: pd.DataFrame) -> object: ...

    def validate_input(self, features: pd.DataFrame) -> None: ...

    def run_metadata(self) -> dict[str, Any]: ...

    def finetune_metadata(self) -> dict[str, Any]: ...
