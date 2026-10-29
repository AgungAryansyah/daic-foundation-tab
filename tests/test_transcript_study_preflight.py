import json
from types import SimpleNamespace

import pytest

from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.tabular_study.bundle import _json
from daic_foundation_tab.transcript_study import preflight, resources
from daic_foundation_tab.transcript_study.audit import require_audit
from daic_foundation_tab.transcript_study.bundle import read_bundle
from tests.test_transcript_study_audit import prepared_input


def mock_hardware(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(name="test-GPU", uuid="gpu-1", total_memory=80 * 2**30))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (70 * 2**30, 80 * 2**30))
    monkeypatch.setattr(resources, "memory_snapshot", lambda: {"available_ram_mb": 8192, "container_ram_limit_mb": 16384})


def test_preflight_is_bound_to_host_task_stage_audit_and_resource_limits(tmp_path, monkeypatch):
    mock_hardware(monkeypatch)
    bundle = prepared_input(tmp_path)
    context = read_bundle(bundle)
    host = {"output_root": str(tmp_path / "runs"), "host_label": "test", "cpu_threads": 1}
    audit = require_audit(bundle, context[0])
    identity = resources.profile_identity(context[0], audit, host, "classification", "cv")
    path = resources.profile_path(host, identity)
    with pytest.raises(ValueError, match="successful preflight"):
        preflight.require_preflight(bundle, context, host, "classification", "cv")
    profile = {"status": "passed", "identity": identity, "peak_process_ram_mb": 1024, "peak_cuda_reserved_mb": 8192}
    profile["profile_id"] = fingerprint(profile)
    path.parent.mkdir(parents=True)
    _json(path, profile)
    assert preflight.require_preflight(bundle, context, host, "classification", "cv") == profile
    with pytest.raises(ValueError, match="successful preflight"):
        preflight.require_preflight(bundle, context, {**host, "cpu_threads": 2}, "classification", "cv")
    with pytest.raises(ValueError, match="successful preflight"):
        preflight.require_preflight(bundle, context, host, "classification", "final")
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (1024, 80 * 2**30))
    with pytest.raises(RuntimeError, match="no job was queued"):
        preflight.require_preflight(bundle, context, host, "classification", "cv")
    profile["status"] = "failed"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="failed, changed"):
        preflight.require_preflight(bundle, context, host, "classification", "cv")


def test_worker_environment_limits_all_cpu_libraries():
    environment = resources.worker_environment({})
    assert all(environment[key] == "1" for key in resources.THREAD_VARIABLES)
    assert environment["MALLOC_ARENA_MAX"] == "2"
    assert resources.worker_environment({"cpu_threads": 3})["OMP_NUM_THREADS"] == "3"
