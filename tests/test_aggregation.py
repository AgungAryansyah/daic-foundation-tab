from __future__ import annotations

import numpy as np
import pandas as pd

from daic_foundation_tab.data.aggregation import COVAREP_COLUMNS, aggregate_source


def test_covarep_aggregation_uses_population_standard_deviation(tmp_path) -> None:
    source = tmp_path / "participant_COVAREP.csv"
    values = np.zeros((3, len(COVAREP_COLUMNS)))
    values[:, 0] = [0.0, 0.01, 0.02]
    values[:, 1] = [1.0, 2.0, 3.0]
    pd.DataFrame(values).to_csv(source, index=False, header=False)

    result = aggregate_source(
        source,
        "covarep",
        {"parser": "covarep", "modality": "audio"},
        {"statistics": ["mean", "std"], "std_ddof": 0},
    )

    assert result.values["covarep_F0_mean"] == 2.0
    assert result.values["covarep_F0_std"] == np.sqrt(2 / 3)
    assert "covarep_timestamp_mean" not in result.values
