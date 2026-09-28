from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
)

PHQ8_DEPRESSION_CUTOFF = 10


def regression_metrics(target: object, prediction: object) -> dict[str, Any]:
    y_true = np.asarray(target, dtype=float)
    y_pred = np.asarray(prediction, dtype=float)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not len(y_true):
        raise ValueError("Regression targets and predictions must be nonempty aligned vectors")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Regression targets and predictions must be finite")
    true_binary = (y_true >= PHQ8_DEPRESSION_CUTOFF).astype(int)
    predicted_binary = (y_pred >= PHQ8_DEPRESSION_CUTOFF).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        true_binary, predicted_binary, labels=[0, 1]
    ).ravel()
    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    sensitivity = (
        float(true_positive / sensitivity_denominator) if sensitivity_denominator else float("nan")
    )
    specificity = (
        float(true_negative / specificity_denominator) if specificity_denominator else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "derived_accuracy": float(np.mean(true_binary == predicted_binary)),
        "derived_balanced_accuracy": float(np.mean([sensitivity, specificity]))
        if sensitivity_denominator and specificity_denominator
        else float("nan"),
        "derived_macro_f1": float(
            f1_score(true_binary, predicted_binary, average="macro", zero_division=0)
        ),
        "derived_depressed_f1": float(
            f1_score(true_binary, predicted_binary, pos_label=1, zero_division=0)
        ),
        "derived_precision_depressed": float(
            precision_score(true_binary, predicted_binary, zero_division=0)
        ),
        "derived_recall_sensitivity": sensitivity,
        "derived_specificity": specificity,
        "derived_confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
        "derived_cutoff": PHQ8_DEPRESSION_CUTOFF,
    }


def regression_predictions(
    participant_ids: pd.Series,
    split: str,
    prediction: object,
    target: pd.Series | None = None,
) -> pd.DataFrame:
    values = np.asarray(prediction, dtype=float)
    if values.ndim != 1 or len(values) != len(participant_ids) or not np.isfinite(values).all():
        raise ValueError("Regression predictions must be finite and aligned with participants")
    if target is not None and len(target) != len(values):
        raise ValueError("Regression targets must align with predictions")
    return pd.DataFrame(
        {
            "participant_id": participant_ids.astype(str).to_numpy(),
            "split": split,
            "phq8_true": target.to_numpy(dtype=float) if target is not None else pd.NA,
            "phq8_pred": values,
            "depressed_true": (target.to_numpy(dtype=float) >= PHQ8_DEPRESSION_CUTOFF).astype(int)
            if target is not None
            else pd.NA,
            "depressed_pred": (values >= PHQ8_DEPRESSION_CUTOFF).astype(int),
        }
    )


def bootstrap_regression_metrics(
    target: object,
    prediction: object,
    iterations: int,
    confidence: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_true = np.asarray(target, dtype=float)
    y_pred = np.asarray(prediction, dtype=float)
    estimate = regression_metrics(y_true, y_pred)
    if iterations < 1 or not 0 < confidence < 1:
        raise ValueError(
            "Bootstrap requires positive iterations and confidence between zero and one"
        )
    generator = np.random.default_rng(random_state)
    rows = []
    for iteration in range(iterations):
        indices = generator.integers(0, len(y_true), size=len(y_true))
        metrics = regression_metrics(y_true[indices], y_pred[indices])
        rows.append(
            {
                "iteration": iteration,
                **{key: value for key, value in metrics.items() if isinstance(value, float)},
            }
        )
    distribution = pd.DataFrame(rows)
    alpha = (1 - confidence) / 2
    summary = {}
    for column in distribution.columns.drop("iteration"):
        values = distribution[column].dropna()
        summary[column] = {
            "estimate": estimate[column],
            "ci_lower": float(values.quantile(alpha)) if not values.empty else None,
            "ci_upper": float(values.quantile(1 - alpha)) if not values.empty else None,
            "valid_bootstrap_samples": len(values),
            "invalid_bootstrap_samples": len(distribution) - len(values),
        }
    return distribution, summary
