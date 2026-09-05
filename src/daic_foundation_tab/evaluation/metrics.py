from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(function: Any, *args: Any, **kwargs: Any) -> float:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return float("nan")


def _expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    assignments = np.minimum((probability * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if not mask.any():
            continue
        error += mask.mean() * abs(target[mask].mean() - probability[mask].mean())
    return float(error)


def classification_metrics(
    target: object, prediction: object, probability_positive: object
) -> dict[str, Any]:
    y_true = np.asarray(target, dtype=int)
    y_pred = np.asarray(prediction, dtype=int)
    y_prob = np.asarray(probability_positive, dtype=float)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    calibration: dict[str, list[float]] = {"observed": [], "predicted": []}
    if has_both_classes:
        observed, predicted = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        calibration = {"observed": observed.tolist(), "predicted": predicted.tolist()}

    return {
        "macro_f1": _safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "depressed_f1": _safe_metric(f1_score, y_true, y_pred, pos_label=1, zero_division=0),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_prob) if has_both_classes else float("nan"),
        "pr_auc": _safe_metric(average_precision_score, y_true, y_prob)
        if has_both_classes
        else float("nan"),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision_depressed": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall_sensitivity": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "specificity": float(true_negative / (true_negative + false_positive))
        if true_negative + false_positive
        else float("nan"),
        "mcc": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "brier_score": _safe_metric(brier_score_loss, y_true, y_prob),
        "expected_calibration_error": _expected_calibration_error(y_true, y_prob)
        if has_both_classes
        else float("nan"),
        "confusion_matrix": matrix.tolist(),
        "calibration_curve": calibration,
    }
