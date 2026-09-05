from __future__ import annotations

import time
from typing import Any


def reset_cuda_peak_memory() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def cuda_peak_memory() -> dict[str, float | None]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
            }
    except ImportError:
        pass
    return {"peak_cuda_allocated_mb": None, "peak_cuda_reserved_mb": None}


def timed_call(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started
