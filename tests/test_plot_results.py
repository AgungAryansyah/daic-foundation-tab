from matplotlib.figure import Figure

from daic_foundation_tab.plot_results import fold_scores, fold_table_panel


def test_fold_table_preserves_partial_fold_positions_and_mean():
    group = ("qwen35_2b", "transcript")
    runs = [
        {
            "model": group[0],
            "input": group[1],
            "dataset": "daic_woz",
            "task": "regression",
            "method": "dora",
            "fold": fold,
            "metrics": {"cv/mae": score},
        }
        for fold, score in [(4, 7.2342), (1, 5.1234)]
    ]
    ax = Figure().subplots()

    scores = fold_scores(runs, "daic_woz", "regression")
    fold_table_panel(ax, scores, [group], ["Qwen3.5-2B"], ["dora"])

    table = ax.tables[0]
    values = [table[1, col].get_text().get_text() for col in range(2, 8)]
    assert values == ["5.123", "—", "—", "7.234", "—", "6.179 (2/5)"]
