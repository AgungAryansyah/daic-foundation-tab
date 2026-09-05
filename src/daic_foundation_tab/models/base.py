from __future__ import annotations

from typing import Any, Protocol, Self

import pandas as pd


class TabularClassifier(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self: ...

    def predict(self, features: pd.DataFrame) -> object: ...

    def predict_proba(self, features: pd.DataFrame) -> object: ...

    def validate_input(self, features: pd.DataFrame) -> None: ...

    def run_metadata(self) -> dict[str, Any]: ...
