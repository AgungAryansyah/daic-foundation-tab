import fcntl
import json

import pytest
import yaml

from daic_foundation_tab import text_runner
from daic_foundation_tab.transcript_study import groups
from daic_foundation_tab.transcript_study.bundle import read_bundle
from daic_foundation_tab.transcript_study.results import verify_result
from tests.test_text_runner import FakeTextModel, mock_gpu
from tests.test_transcript_study_audit import prepared_input


def install_model(monkeypatch):
    mock_gpu(monkeypatch)

    def create(config):
        model = FakeTextModel(config["experiment"]["task"])
        original = model.fit

        def fit(training, validation, directory, *args, **kwargs):
            original(training, validation, directory, *args, **kwargs)
            adapter = directory / "best_adapter"
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-adapter")

        model.fit = fit
        metadata = model.run_metadata
        model.run_metadata = lambda: {**metadata(), "checkpoint_revision": config["model"]["revision"],
                                      "model_id": config["model"]["model_id"],
                                      "model_name": config["model"]["name"],
                                      "base_parameters_frozen": True}
        return model

    monkeypatch.setattr(text_runner, "create_text_model", create)
    monkeypatch.setattr(groups, "require_preflight", lambda *args: {})


def host_input(tmp_path):
    path = tmp_path / "host.yaml"
    path.write_text(yaml.safe_dump({"output_root": str(tmp_path / "runs"), "host_label": "test",
                                    "wandb": {"mode": "disabled"}}))
    return path


def test_group_verifies_results_skips_completed_and_recovers_failure(tmp_path, monkeypatch):
    bundle, host = prepared_input(tmp_path), host_input(tmp_path)
    install_model(monkeypatch)
    calls = []

    def execute(bundle, host_path, condition, host):
        calls.append(condition["condition_id"])
        if len(calls) == 2:
            raise RuntimeError("interrupted worker")
        groups.condition_worker(bundle, host_path, condition["condition_id"], condition["stage"])

    monkeypatch.setattr(groups, "_run_subprocess", execute)
    with pytest.raises(RuntimeError, match="interrupted"):
        groups.run_group(bundle, host, "classification")
    saved = next((tmp_path / "runs").rglob("result.json"))
    before = saved.read_bytes()
    output = groups.run_group(bundle, host, "classification")
    assert saved.read_bytes() == before
    assert len(list(output.rglob("result.json"))) == 10
    state = json.loads((output / "status.json").read_text())
    assert state["status"] == "completed" and state["completed_conditions"] == 10
    context = read_bundle(bundle)
    for path in output.rglob("result.json"):
        verify_result(context, path.parent, require_checkpoint=True)
    count = len(calls)
    groups.run_group(bundle, host, "classification")
    assert len(calls) == count
    saved.with_name("runtime.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact is missing or changed"):
        groups.run_group(bundle, host, "classification")


def test_group_lock_prevents_concurrent_execution(tmp_path, monkeypatch):
    bundle, host = prepared_input(tmp_path), host_input(tmp_path)
    install_model(monkeypatch)
    root = tmp_path / "runs" / read_bundle(bundle)[0]["study_id"] / "cv/qwen35_2b-regression"
    root.mkdir(parents=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            groups.run_group(bundle, host, "regression")


def test_group_requires_preflight_and_worker_must_write_a_result(tmp_path, monkeypatch):
    bundle, host = prepared_input(tmp_path), host_input(tmp_path)
    install_model(monkeypatch)
    monkeypatch.setattr(groups, "require_preflight", lambda *args: (_ for _ in ()).throw(ValueError("preflight required")))
    with pytest.raises(ValueError, match="preflight required"):
        groups.run_group(bundle, host, "regression")
    monkeypatch.setattr(groups, "require_preflight", lambda *args: {})
    monkeypatch.setattr(groups, "_run_subprocess", lambda *args: None)
    with pytest.raises(ValueError, match="without a verified result"):
        groups.run_group(bundle, host, "regression")


def test_worker_reuses_a_completed_attempt_when_only_sealing_was_interrupted(tmp_path, monkeypatch):
    bundle, host = prepared_input(tmp_path), host_input(tmp_path)
    install_model(monkeypatch)
    condition = read_bundle(bundle)[0]["conditions"]["cv"][0]
    output = groups.condition_worker(bundle, host, condition["condition_id"], "cv")
    (output / "result.json").unlink()
    monkeypatch.setattr(text_runner, "create_text_model", lambda config: pytest.fail("Completed inference repeated"))
    assert groups.condition_worker(bundle, host, condition["condition_id"], "cv") == output
    assert (output / "result.json").is_file()
