from __future__ import annotations

import os
import resource
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread

import psutil
import yaml

from daic_foundation_tab.data.participants import fingerprint

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def load_host(path: str | Path) -> dict:
    host = yaml.safe_load(Path(path).read_text())
    if not isinstance(host, dict) or not host.get("output_root") or not host.get("host_label"):
        raise ValueError("Transcript host requires output_root and host_label")
    if host.get("device", "cuda:0") != "cuda:0":
        raise ValueError("Select the visible GPU with CUDA_VISIBLE_DEVICES and use cuda:0")
    threads = host.setdefault("cpu_threads", 1)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if host.get("wandb", {}).get("mode", "online") not in {"online", "offline", "disabled"}:
        raise ValueError("Unsupported host W&B mode")
    host["output_root"] = str(Path(host["output_root"]).expanduser().resolve())
    if host.get("test_ground_truth"):
        host["test_ground_truth"] = str(Path(host["test_ground_truth"]).expanduser().resolve())
    return host


def worker_environment(host: dict) -> dict:
    return {**os.environ, **{key: str(host.get("cpu_threads", 1)) for key in THREAD_VARIABLES},
            "MALLOC_ARENA_MAX": "2", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}


def configure_threads(host: dict) -> None:
    import torch

    os.environ.update({key: value for key, value in worker_environment(host).items()
                       if key in {*THREAD_VARIABLES, "MALLOC_ARENA_MAX", "TOKENIZERS_PARALLELISM"}})
    torch.set_num_threads(host.get("cpu_threads", 1))


def memory_snapshot() -> dict:
    usage = psutil.Process().memory_info().rss
    limit, available = None, psutil.virtual_memory().available
    mount = Path("/sys/fs/cgroup")
    if (mount / "memory.max").is_file():
        maximum = (mount / "memory.max").read_text().strip()
        if maximum != "max":
            limit = int(maximum)
            current = int((mount / "memory.current").read_text())
            stats = dict(line.split() for line in (mount / "memory.stat").read_text().splitlines())
            available = min(available, max(0, limit - current + int(stats.get("inactive_file", 0))))
    return {"process_ram_mb": usage / 2**20, "peak_process_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "container_ram_limit_mb": limit / 2**20 if limit is not None else None,
            "available_ram_mb": available / 2**20}


def hardware_identity(host: dict) -> dict:
    import torch

    properties = torch.cuda.get_device_properties(0)
    return {"hostname": socket.gethostname(), "host_label": host["host_label"],
            "gpu_name": properties.name, "gpu_uuid": str(getattr(properties, "uuid", "")),
            "total_gpu_vram_bytes": properties.total_memory, "cpu_threads": host.get("cpu_threads", 1),
            "container_ram_limit_mb": memory_snapshot()["container_ram_limit_mb"]}


def profile_identity(study: dict, audit: dict, host: dict, task: str, stage: str) -> dict:
    return {"study_id": study["study_id"], "software_identity": study["software_identity"],
            "audit_id": audit["audit_id"], "task": task, "stage": stage, "hardware": hardware_identity(host)}


def profile_path(host: dict, identity: dict) -> Path:
    return Path(host["output_root"]) / identity["study_id"] / "preflight" / f"{fingerprint(identity)}.json"


def require_capacity(profile: dict) -> None:
    import torch

    free, _ = torch.cuda.mem_get_info(0)
    if memory_snapshot()["available_ram_mb"] < profile["peak_process_ram_mb"] * 1.2:
        raise RuntimeError("Insufficient system RAM for transcript execution; no job was queued")
    if free / 2**20 < profile["peak_cuda_reserved_mb"] * 1.2:
        raise RuntimeError("Insufficient free GPU memory for transcript execution; no job was queued")


@contextmanager
def observe_resources(logger):
    started = time.perf_counter()
    stopped = Event()
    samples = {"peak_process_ram_mb": memory_snapshot()["peak_process_ram_mb"]}

    def sample():
        snapshot = memory_snapshot()
        samples.update(snapshot, peak_process_ram_mb=max(samples["peak_process_ram_mb"], snapshot["peak_process_ram_mb"]))

    def observe():
        while not stopped.wait(1):
            sample()
            if int(time.perf_counter() - started) % 30 == 0:
                logger.info("phase=resource monitoring elapsed=%.1fs RAM=%.1fMiB peak=%.1fMiB",
                            time.perf_counter() - started, samples["process_ram_mb"], samples["peak_process_ram_mb"])

    sample()
    thread = Thread(target=observe, daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stopped.set()
        thread.join()
        sample()
