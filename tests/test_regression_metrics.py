from __future__ import annotations

import numpy as np
import pandas as pd

from daic_foundation_tab.evaluation.regression import (
    bootstrap_regression_metrics,
    regression_metrics,
    regression_predictions,
)


def test_regression_metrics_use_raw_scores_and_fixed_binary_cutoff() -> None:
    metrics = regression_metrics([0, 8, 12, 20], [-2, 11, 9, 25])

    assert metrics["mae"] == 3.25
    assert metrics["rmse"] == np.sqrt(11.75)
    assert metrics["derived_cutoff"] == 10
    assert metrics["derived_confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["derived_depressed_f1"] == 0.5


def test_regression_predictions_keep_unclipped_values_and_hide_test_targets() -> None:
    ids = pd.Series(["500", "501"])

    frame = regression_predictions(ids, "test", [-2.5, 26.5])

    assert frame["phq8_pred"].tolist() == [-2.5, 26.5]
    assert frame["phq8_true"].isna().all()
    assert frame["depressed_pred"].tolist() == [0, 1]


def test_regression_bootstrap_resamples_paired_predictions_reproducibly() -> None:
    first, summary = bootstrap_regression_metrics([1, 4, 12, 18], [2, 3, 10, 16], 20, 0.95, 42)
    second, _ = bootstrap_regression_metrics([1, 4, 12, 18], [2, 3, 10, 16], 20, 0.95, 42)

    assert first.equals(second)
    assert summary["mae"]["estimate"] == 1.5
    assert summary["mae"]["valid_bootstrap_samples"] == 20
