from __future__ import annotations

import random

import numpy as np

from daic_foundation_tab.tracking.seeds import set_global_seed


def test_set_global_seed_reproduces_python_and_numpy_randomness() -> None:
    set_global_seed(42)
    first = (random.random(), np.random.random())

    set_global_seed(42)
    second = (random.random(), np.random.random())

    assert first == second
