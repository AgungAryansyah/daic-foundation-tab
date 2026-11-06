import json
from contextlib import contextmanager
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


def test_preflight_rejects_dependencies_outside_the_lock(monkeypatch):
    monkeypatch.setattr(resources, "version", lambda package: "0.0.0")
    with pytest.raises(ValueError, match="do not match uv.lock"):
        resources.dependency_versions()


def test_nested_cgroups_use_the_most_restrictive_available_capacity(tmp_path):
    mount = tmp_path / "cgroup"
    child = mount / "job/step"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/job/step\n")
    for directory, maximum, current, inactive in ((mount, 16000, 15000, 3000), (child, 8000, 7500, 1000)):
        (directory / "memory.max").write_text(str(maximum))
        (directory / "memory.current").write_text(str(current))
        (directory / "memory.stat").write_text(f"inactive_file {inactive}\n")
    assert resources._cgroup_capacity(mount, membership) == (8000, 1500)


def test_probe_exercises_optimizer_state_residency_and_unlabeled_prediction(tmp_path, monkeypatch):
    import torch

    from daic_foundation_tab.models.text_base import TextPrediction

    bundle = prepared_input(tmp_path)
    context = read_bundle(bundle)
    audit = require_audit(bundle, context[0])
    host = {"output_root": str(tmp_path / "runs"), "host_label": "test", "wandb": {"mode": "disabled"}}
    (tmp_path / "runs").mkdir()
    calls = []
    longest_evaluation = next(
        c for c in reversed(context[0]["conditions"]["cv"])
        if c["task"] == "classification" and c["mode"] == "fine_tuning"
    )
    evaluation = next(row for row in audit["conditions"][longest_evaluation["condition_id"]]["records"]
                      if row["role"] == "evaluation")
    evaluation["total_tokens"] += 1000

    class Model:
        def __init__(self, config):
            self.config = config

        def prepare(self, demonstrations):
            calls.append((self.config["experiment"]["mode"], len(demonstrations)))

        def run_metadata(self):
            return {"checkpoint_revision": self.config["model"]["revision"]}

        def preflight(self, examples):
            assert examples

        def fit(self, training, validation, directory, progress):
            assert self.config["model"]["parameters"]["epochs"] == 2
            assert len(training) == 1 and len(validation) == 2
            assert all(x.target is not None for x in [*training, *validation])
            progress(2, 2)

        def predict(self, examples):
            assert all(x.target is None for x in examples)
            if self.config["experiment"]["mode"] == "fine_tuning":
                assert examples[0].participant_id == evaluation["participant_id"]
            return [TextPrediction(1, .75, "1", 100)]

    @contextmanager
    def monitor(logger):
        yield {"peak_process_ram_mb": 500}

    monkeypatch.setattr(preflight, "create_text_model", Model)
    monkeypatch.setattr(preflight, "observe_resources", monitor)
    monkeypatch.setattr(preflight, "reset_cuda_peak_memory", lambda device: None)
    monkeypatch.setattr(preflight, "cuda_peak_memory", lambda device: {"peak_cuda_reserved_mb": 1000, "peak_cuda_allocated_mb": 900})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    logger = SimpleNamespace(info=lambda *args: None, exception=lambda *args: None)
    peaks = preflight._probe(context, audit, host, "classification", "cv", logger)
    assert peaks["peak_process_ram_mb"] == 500 and peaks["peak_cuda_reserved_mb"] == 1000
    assert calls == [("frozen_icl", 2), ("fine_tuning", 0)]


def test_failed_repeat_probe_invalidates_a_previous_success(tmp_path, monkeypatch):
    from tests.test_transcript_study_groups import host_input

    mock_hardware(monkeypatch)
    bundle, host_path = prepared_input(tmp_path), host_input(tmp_path)
    monkeypatch.setattr(preflight, "require_cuda_device", lambda device: device)
    monkeypatch.setattr(preflight, "require_capacity", lambda profile: None)
    monkeypatch.setattr(preflight, "_probe", lambda *args: {"peak_process_ram_mb": 512, "peak_cuda_reserved_mb": 1024})
    output = preflight.preflight_worker(bundle, host_path, "classification", "cv")
    context, host = read_bundle(bundle), resources.load_host(host_path)
    assert preflight.require_preflight(bundle, context, host, "classification", "cv")["status"] == "passed"

    def fail(*args):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(preflight, "_probe", fail)
    with pytest.raises(RuntimeError, match="probe failed"):
        preflight.preflight_worker(bundle, host_path, "classification", "cv")
    assert json.loads(output.read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="failed, changed"):
        preflight.require_preflight(bundle, context, host, "classification", "cv")


def test_measured_memory_margin_must_pass_before_approving_preflight(tmp_path, monkeypatch):
    from tests.test_transcript_study_groups import host_input

    mock_hardware(monkeypatch)
    bundle, host_path = prepared_input(tmp_path), host_input(tmp_path)
    monkeypatch.setattr(preflight, "require_cuda_device", lambda device: device)
    peaks = {"peak_process_ram_mb": 1024, "peak_cuda_reserved_mb": 60 * 1024}
    monkeypatch.setattr(preflight, "_probe", lambda *args: peaks)
    with pytest.raises(RuntimeError, match="Insufficient free GPU memory"):
        preflight.preflight_worker(bundle, host_path, "classification", "cv")
    context, host = read_bundle(bundle), resources.load_host(host_path)
    identity = resources.profile_identity(context[0], require_audit(bundle, context[0]), host, "classification", "cv")
    profile = json.loads(resources.profile_path(host, identity).read_text())
    assert profile["status"] == "failed"
    assert profile["peak_cuda_reserved_mb"] == peaks["peak_cuda_reserved_mb"]
