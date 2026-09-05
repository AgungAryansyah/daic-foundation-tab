from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import psutil


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def environment_metadata(device: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram_bytes": psutil.virtual_memory().total,
        "torch_version": _version("torch"),
        "tabicl_version": _version("tabicl"),
        "cuda_version": None,
        "cuda_device": device,
        "gpu_name": None,
        "total_gpu_vram_bytes": None,
    }
    import torch

    index = int(device.removeprefix("cuda:"))
    metadata["cuda_version"] = torch.version.cuda
    properties = torch.cuda.get_device_properties(index)
    metadata["gpu_name"] = properties.name
    metadata["total_gpu_vram_bytes"] = properties.total_memory
    return metadata
