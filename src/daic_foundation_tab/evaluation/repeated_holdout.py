from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.models.base import TabularClassifier


def _positive_probability(model: TabularClassifier, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Classifier predict_proba must return two probability columns")
    return probabilities[:, 1]


def repeated_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    participant_ids: pd.Series,
    repeats: int,
    validation_fraction: float,
    seed_start: int,
    model_factory: Callable[[], TabularClassifier],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows: list[dict[str, float | int]] = []
    assignments: dict[int, pd.DataFrame] = {}
    for seed in range(seed_start, seed_start + repeats):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        context_indices, validation_indices = next(splitter.split(features, target))
        context_target = target.iloc[context_indices]
        validation_target = target.iloc[validation_indices]
        if context_target.nunique() != 2 or validation_target.nunique() != 2:
            raise ValueError(f"Repeated-holdout seed {seed} lost a class")

        model = model_factory()
        model.fit(features.iloc[context_indices], context_target)
        prediction = model.predict(features.iloc[validation_indices])
        probability = _positive_probability(model, features.iloc[validation_indices])
        metrics = classification_metrics(validation_target, prediction, probability)
        rows.append({"seed": seed, **{key: value for key, value in metrics.items() if isinstance(value, float)}})
        assignments[seed] = pd.DataFrame(
            {
                "participant_id": participant_ids.iloc[np.r_[context_indices, validation_indices]].to_numpy(),
                "role": ["context"] * len(context_indices) + ["validation"] * len(validation_indices),
                "target": target.iloc[np.r_[context_indices, validation_indices]].to_numpy(),
            }
        )

    return pd.DataFrame(rows), assignments


def repeated_holdout_summary(metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        column: {
            "mean": float(metrics[column].mean()),
            "std": float(metrics[column].std(ddof=0)),
            "median": float(metrics[column].median()),
            "p2_5": float(metrics[column].quantile(0.025)),
            "p97_5": float(metrics[column].quantile(0.975)),
        }
        for column in metrics.columns
        if column != "seed"
    }
