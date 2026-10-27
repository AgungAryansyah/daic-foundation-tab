import fcntl
import json
import subprocess
import sys

import pytest
import yaml

from daic_foundation_tab.tabular_study import groups
from daic_foundation_tab.tabular_study.bundle import read_bundle
from daic_foundation_tab.tabular_study.condition import run_condition
from daic_foundation_tab.tabular_study.results import write_result

pytest_plugins = ["tests.test_tabular_study_condition"]


def host_file(tmp_path, name):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / name),
                "host_label": name,
                "device": "cuda:0",
                "wandb": {"mode": "disabled"},
            }
        )
    )
    return path


def fake_launcher(launches, fail_at=None):
    def launch(bundle, condition_id, host_path, root, attempt):
        launches.append(condition_id)
        if len(launches) == fail_at:
            raise subprocess.CalledProcessError(1, "synthetic")
        context = read_bundle(bundle)
        condition = next(
            x
            for stage in context[0]["conditions"].values()
            for x in stage
            if x["condition_id"] == condition_id
        )
        host = groups.load_host(host_path)
        host["output_root"] = str(root)
        output = run_condition(bundle, condition, host, attempt)
        write_result(context, condition, output)

    return launch


def test_groups_run_independently_and_resume_completed_conditions(prepared, tmp_path, monkeypatch):
    bundle, _ = prepared
    launches = []
    monkeypatch.setattr(groups, "_run_subprocess", fake_launcher(launches, fail_at=2))
    with pytest.raises(subprocess.CalledProcessError):
        groups.run_group(bundle, "tabpfn35", "audio", host_file(tmp_path, "host-one"))
    status_file = next((tmp_path / "host-one").rglob("status.json"))
    state = json.loads(status_file.read_text())
    assert state["status"] == "failed"
    first = launches[0]
    groups.run_group(bundle, "tabpfn35", "audio", host_file(tmp_path, "host-one"))
    assert len(launches) == 21 and launches.count(first) == 1
    status = json.loads(status_file.read_text())
    assert status["status"] == "completed" and status["completed_conditions"] == 20
    groups.run_group(bundle, "kumo_medium", "visual", host_file(tmp_path, "host-two"))
    assert len(launches) == 41
    assert (
        json.loads(next((tmp_path / "host-two").rglob("status.json")).read_text())["status"]
        == "completed"
    )


def test_group_rejects_concurrent_execution(prepared, tmp_path):
    bundle, _ = prepared
    study = read_bundle(bundle)[0]
    host = host_file(tmp_path, "host")
    root = tmp_path / "host" / study["study_id"] / "cv" / "tabpfn35-audio"
    root.mkdir(parents=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            groups.run_group(bundle, "tabpfn35", "audio", host)


def test_host_rejects_unsupported_gpu_index(tmp_path):
    path = tmp_path / "host.yaml"
    path.write_text(yaml.safe_dump({"output_root": str(tmp_path), "device": "cuda:1"}))
    with pytest.raises(ValueError, match="cuda:0"):
        groups.load_host(path)


def test_worker_writes_a_verified_result(prepared, tmp_path, monkeypatch):
    bundle, _ = prepared
    context = read_bundle(bundle)
    item = context[0]["conditions"]["cv"][0]
    host = host_file(tmp_path, "worker")
    root = tmp_path / "group"
    monkeypatch.setattr(sys, "argv", ["worker", "--study", str(bundle), "--condition",
        item["condition_id"], "--host-config", str(host), "--group-root", str(root),
        "--attempt", "attempt-one"])
    groups.main()
    assert len(list(root.rglob("result.json"))) == 1
