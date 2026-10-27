import pandas as pd
import pytest

from daic_foundation_tab.tabular_study.comparisons import paired_bootstrap


def test_paired_bootstrap_preserves_folds_and_returns_reproducible_differences():
    left = pd.DataFrame(
        {
            "participant_id": ["1", "2", "3", "4"],
            "fold": [1, 1, 2, 2],
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 0, 0, 0],
            "prob_depressed": [0.2, 0.2, 0.2, 0.2],
        }
    )
    right = left.copy()
    right["y_pred"] = right["y_true"]
    right["prob_depressed"] = [0.1, 0.9, 0.1, 0.9]
    calls = []
    result = paired_bootstrap(
        left,
        right,
        "classification",
        iterations=20,
        seed=42,
        progress=lambda n, total: calls.append((n, total)),
    )
    assert result == paired_bootstrap(left, right, "classification", iterations=20, seed=42)
    macro = next(row for row in result if row["metric"] == "macro_f1")
    assert macro["delta_ft_minus_icl"] == pytest.approx(2 / 3)
    assert macro["valid_bootstrap_samples"] == 20
    assert calls[0] == (0, 20) and calls[-1] == (20, 20)
    right.loc[0, "fold"] = 2
    with pytest.raises(ValueError, match="different folds"):
        paired_bootstrap(left, right, "classification", iterations=2, seed=42)
