from __future__ import annotations

import sys
from types import SimpleNamespace

from daic_foundation_tab.tracking.environment import environment_metadata


def test_environment_metadata_records_selected_cuda_device(monkeypatch) -> None:
    properties = SimpleNamespace(name="Remote GPU", total_memory=24 * 1024**3)
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        cuda=SimpleNamespace(get_device_properties=lambda index: properties),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    metadata = environment_metadata("cuda:0")

    assert metadata["cuda_device"] == "cuda:0"
    assert metadata["cuda_version"] == "12.1"
    assert metadata["gpu_name"] == "Remote GPU"
    assert metadata["total_gpu_vram_bytes"] == 24 * 1024**3
