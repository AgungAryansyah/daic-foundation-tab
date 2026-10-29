from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from daic_foundation_tab.runner import _git_commit
from daic_foundation_tab.tabular_study.bundle import _json
from daic_foundation_tab.tracking.logging import configure_run_logger

from .bundle import TASKS, read_bundle
from .condition import run_condition
from .preflight import require_preflight
from .resources import (
    configure_threads,
    load_host,
    memory_snapshot,
    observe_resources,
    worker_environment,
)
from .results import verify_result, write_result


def _existing(context: tuple, root: Path, condition: dict) -> Path | None:
    results = list((root / condition["condition_id"]).glob("*/result.json"))
    if len(results) > 1:
        raise ValueError("Multiple successful transcript attempts exist for one condition")
    if results:
        verify_result(context, results[0].parent, require_checkpoint=True)
        return results[0].parent
    return None


def _resume_candidate(root: Path, condition: dict) -> Path | None:
    attempts = sorted((root / condition["condition_id"]).glob("*/study_run.json"))
    return attempts[-1].parent if attempts else None


def condition_worker(bundle: Path, host_path: Path, condition_id: str, stage: str) -> Path:
    host = load_host(host_path)
    configure_threads(host)
    context = read_bundle(bundle)
    matches = [c for c in context[0]["conditions"].get(stage, []) if c["condition_id"] == condition_id]
    if len(matches) != 1:
        raise ValueError("Transcript worker condition is missing or ambiguous")
    condition = matches[0]
    require_preflight(bundle, context, host, condition["task"], stage)
    root = Path(host["output_root"]) / context[0]["study_id"] / stage / f"qwen35_2b-{condition['task']}"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    host["output_root"] = str(root)
    existing = _existing(context, root, condition)
    if existing:
        return existing
    logger = configure_run_logger(root / "worker.log", "INFO")
    environment = {"host_label": host["host_label"], "cpu_threads": host["cpu_threads"], "git_commit": _git_commit(),
                   "dependency_versions": {name: version(name) for name in ("torch", "transformers", "peft", "numpy", "pandas")},
                   "container_ram_limit_mb": memory_snapshot()["container_ram_limit_mb"]}
    with observe_resources(logger):
        output = run_condition(bundle, condition, host, resume=_resume_candidate(root, condition),
                               extra_environment=environment, runtime_metrics=memory_snapshot)
    write_result(context, condition, output)
    return output


def _run_subprocess(bundle: Path, host_path: Path, condition: dict, host: dict) -> None:
    subprocess.run([sys.executable, "-u", "-m", __name__, "--study", str(bundle), "--host-config", str(host_path),
                    "--condition", condition["condition_id"], "--task", condition["task"], "--stage", condition["stage"]],
                   env=worker_environment(host), check=True)


def run_group(bundle: str | Path, host_path: str | Path, task: str, *, stage: str = "cv") -> Path:
    bundle, host_path = Path(bundle).resolve(), Path(host_path).resolve()
    host = load_host(host_path)
    configure_threads(host)
    context = read_bundle(bundle)
    if stage not in context[0]["conditions"] or task not in TASKS:
        raise ValueError("Unsupported transcript group task or stage")
    root = Path(host["output_root"]) / context[0]["study_id"] / stage / f"qwen35_2b-{task}"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This transcript group is already running") from error
        logger = configure_run_logger(root / "group.log", "INFO")
        conditions = [c for c in context[0]["conditions"][stage] if c["task"] == task]
        state = {"study_id": context[0]["study_id"], "status": "running", "conditions": {},
                 "total_conditions": len(conditions), "completed_conditions": 0}
        path = root / "status.json"
        started = time.perf_counter()
        try:
            for completed, condition in enumerate(conditions):
                identity = condition["condition_id"]
                state.update(active_condition=identity, completed_conditions=completed, elapsed_seconds=time.perf_counter() - started)
                _json(path, state)
                logger.info("conditions=%d/%d (%.1f%%) active=%s elapsed=%.1fs ETA unavailable",
                            completed, len(conditions), 100 * completed / len(conditions), identity[:12], time.perf_counter() - started)
                output = _existing(context, root, condition)
                if output is None:
                    require_preflight(bundle, context, host, task, stage)
                    state["conditions"][identity] = {"status": "running"}
                    _json(path, state)
                    _run_subprocess(bundle, host_path, condition, host)
                    output = _existing(context, root, condition)
                    if output is None:
                        raise ValueError("Transcript worker returned without a verified result")
                state["conditions"][identity] = {"status": "completed", "output": str(output)}
                state["completed_conditions"] = completed + 1
                _json(path, state)
        except BaseException as error:
            state.update(status="failed", error_type=type(error).__name__)
            identity = state.get("active_condition")
            if identity:
                state["conditions"][identity] = {"status": "failed", "error_type": type(error).__name__}
            _json(path, state)
            raise
        state.update(status="completed", active_condition=None, elapsed_seconds=time.perf_counter() - started)
        _json(path, state)
        logger.info("conditions=%d/%d (100%%) status=completed elapsed=%.1fs", len(conditions), len(conditions), state["elapsed_seconds"])
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--host-config", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--stage", choices=("cv", "final"), default="cv")
    args = parser.parse_args()
    print(condition_worker(args.study, args.host_config, args.condition, args.stage))
