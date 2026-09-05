from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from daic_foundation_tab.evaluation.metrics import classification_metrics


def test_core_metrics_match_sklearn() -> None:
    target = np.array([0, 0, 1, 1])
    prediction = np.array([0, 1, 1, 1])
    probability = np.array([0.1, 0.6, 0.7, 0.9])

    metrics = classification_metrics(target, prediction, probability)

    assert metrics["macro_f1"] == f1_score(target, prediction, average="macro")
    assert metrics["roc_auc"] == roc_auc_score(target, probability)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
