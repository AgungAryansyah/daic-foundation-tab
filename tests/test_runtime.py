from __future__ import annotations

import pytest

from daic_foundation_tab.tracking import runtime


class _FakeCuda:
    def __init__(self, *, available: bool = True, device_count: int = 1) -> None:
        self.available = available
        self.count = device_count
        self.selected = []
        self.resets = []
        self.allocated_requests = []
        self.reserved_requests = []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def set_device(self, index: int) -> None:
        self.selected.append(index)

    def reset_peak_memory_stats(self, index: int) -> None:
        self.resets.append(index)

    def max_memory_allocated(self, index: int) -> int:
        self.allocated_requests.append(index)
        return 3 * 1024**2

    def max_memory_reserved(self, index: int) -> int:
        self.reserved_requests.append(index)
        return 5 * 1024**2


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda


def test_require_cuda_device_selects_configured_gpu(monkeypatch) -> None:
    cuda = _FakeCuda()
    monkeypatch.setattr(runtime, "_torch", lambda: _FakeTorch(cuda))

    device = runtime.require_cuda_device("cuda:0")

    assert device == "cuda:0"
    assert cuda.selected == [0]
    assert cuda.resets == [0]


def test_require_cuda_device_rejects_missing_cuda(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_torch", lambda: _FakeTorch(_FakeCuda(available=False)))

    with pytest.raises(runtime.CudaRequirementError, match="torch.cuda.is_available"):
        runtime.require_cuda_device("cuda:0")


def test_require_cuda_device_rejects_no_visible_devices(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_torch", lambda: _FakeTorch(_FakeCuda(device_count=0)))

    with pytest.raises(runtime.CudaRequirementError, match="at least one visible"):
        runtime.require_cuda_device("cuda:0")


def test_require_cuda_device_rejects_invalid_device_index(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_torch", lambda: _FakeTorch(_FakeCuda(device_count=1)))

    with pytest.raises(runtime.CudaRequirementError, match="visible CUDA device count is 1"):
        runtime.require_cuda_device("cuda:1")


def test_cuda_memory_uses_selected_device(monkeypatch) -> None:
    cuda = _FakeCuda()
    monkeypatch.setattr(runtime, "_torch", lambda: _FakeTorch(cuda))

    runtime.reset_cuda_peak_memory("cuda:0")
    memory = runtime.cuda_peak_memory("cuda:0")

    assert cuda.resets == [0]
    assert cuda.allocated_requests == [0]
    assert cuda.reserved_requests == [0]
    assert memory == {"peak_cuda_allocated_mb": 3.0, "peak_cuda_reserved_mb": 5.0}
