from __future__ import annotations

import re
import time
from typing import Any


class CudaRequirementError(RuntimeError):
    pass


def _torch() -> Any:
    import torch

    return torch


def _device_index(device: str) -> int:
    match = re.fullmatch(r"cuda:(\d+)", device)
    if match is None:
        raise CudaRequirementError("GPU-only runs require runtime.device in the form 'cuda:<index>'")
    return int(match.group(1))


def require_cuda_device(device: str) -> str:
    try:
        torch = _torch()
    except ImportError as error:
        raise CudaRequirementError(
            "GPU-only runs require CUDA-enabled PyTorch. Run 'uv sync --group dev' on the remote GPU server."
        ) from error
    index = _device_index(device)
    if not torch.cuda.is_available():
        raise CudaRequirementError(
            "GPU-only runs require torch.cuda.is_available() to be true. Verify the NVIDIA driver, "
            "CUDA-visible device, and CUDA-enabled PyTorch wheel on the remote server."
        )
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise CudaRequirementError("GPU-only runs require at least one visible CUDA device")
    if index >= device_count:
        raise CudaRequirementError(
            f"Configured CUDA device {device!r} is unavailable; visible CUDA device count is {device_count}"
        )
    try:
        torch.cuda.set_device(index)
        torch.cuda.reset_peak_memory_stats(index)
    except RuntimeError as error:
        raise CudaRequirementError(f"Unable to initialize configured CUDA device {device!r}") from error
    return device


def reset_cuda_peak_memory(device: str) -> None:
    torch = _torch()
    torch.cuda.reset_peak_memory_stats(_device_index(device))


def cuda_peak_memory(device: str) -> dict[str, float]:
    torch = _torch()
    index = _device_index(device)
    return {
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(index) / (1024**2),
        "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(index) / (1024**2),
    }


def timed_call(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started
