from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import classification_metrics


def bootstrap_metrics(
    target: object,
    prediction: object,
    probability_positive: object,
    iterations: int,
    confidence: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_true = np.asarray(target, dtype=int)
    y_pred = np.asarray(prediction, dtype=int)
    y_prob = np.asarray(probability_positive, dtype=float)
    generator = np.random.default_rng(random_state)
    rows: list[dict[str, float]] = []

    for iteration in range(iterations):
        indices = generator.integers(0, len(y_true), size=len(y_true))
        metrics = classification_metrics(y_true[indices], y_pred[indices], y_prob[indices])
        rows.append(
            {
                "iteration": iteration,
                **{key: value for key, value in metrics.items() if isinstance(value, float)},
            }
        )

    distribution = pd.DataFrame(rows)
    alpha = (1 - confidence) / 2
    summary: dict[str, Any] = {}
    for column in distribution.columns:
        if column == "iteration":
            continue
        values = distribution[column].dropna()
        summary[column] = {
            "estimate": float(classification_metrics(y_true, y_pred, y_prob)[column]),
            "ci_lower": float(values.quantile(alpha)) if not values.empty else None,
            "ci_upper": float(values.quantile(1 - alpha)) if not values.empty else None,
            "valid_bootstrap_samples": len(values),
            "invalid_bootstrap_samples": len(distribution) - len(values),
        }
    return distribution, summary
