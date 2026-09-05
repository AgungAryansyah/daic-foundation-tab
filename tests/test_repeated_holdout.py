from __future__ import annotations

import numpy as np
import pandas as pd

from daic_foundation_tab.evaluation.repeated_holdout import repeated_holdout


class FakeClassifier:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> FakeClassifier:
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile([0.6, 0.4], (len(features), 1))

    def validate_input(self, features: pd.DataFrame) -> None:
        return None

    def run_metadata(self) -> dict:
        return {}


def test_repeated_holdout_saves_audit_assignments() -> None:
    features = pd.DataFrame({"feature": range(10)})
    target = pd.Series([0, 1] * 5)
    participant_ids = pd.Series([str(index) for index in range(10)])

    metrics, assignments = repeated_holdout(
        features,
        target,
        participant_ids,
        repeats=2,
        validation_fraction=0.2,
        seed_start=0,
        model_factory=FakeClassifier,
    )

    assert metrics["seed"].tolist() == [0, 1]
    assert set(assignments[0]["role"]) == {"context", "validation"}
