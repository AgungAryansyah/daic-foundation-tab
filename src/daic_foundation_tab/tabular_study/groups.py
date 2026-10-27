from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

from daic_foundation_tab.tabular_study.bundle import _json, read_bundle
from daic_foundation_tab.tabular_study.condition import run_condition
from daic_foundation_tab.tabular_study.results import verify_result, write_result
from daic_foundation_tab.tracking.logging import configure_run_logger, phase


def load_host(path: str | Path) -> dict:
    host = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(host, dict)
        or not isinstance(host.get("output_root"), str)
        or not host["output_root"]
    ):
        raise ValueError("Host configuration requires output_root")
    if host.get("device", "cuda:0") != "cuda:0":
        raise ValueError("Use cuda:0 and select the visible GPU through the host environment")
    if host.get("wandb", {}).get("mode", "online") not in {"online", "offline", "disabled"}:
        raise ValueError("Host W&B mode must be online, offline, or disabled")
    host["output_root"] = str(Path(host["output_root"]).expanduser().resolve())
    return host


def _run_subprocess(
    bundle: Path, condition_id: str, host_path: Path, group_root: Path, attempt_id: str
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "daic_foundation_tab.tabular_study.groups",
            "--study",
            str(bundle),
            "--condition",
            condition_id,
            "--host-config",
            str(host_path),
            "--group-root",
            str(group_root),
            "--attempt",
            attempt_id,
        ],
        check=True,
    )


def _existing(context: tuple, group_root: Path, condition: dict) -> Path | None:
    results = sorted((group_root / condition["condition_id"]).glob("*/result.json"))
    if len(results) > 1:
        raise ValueError("Multiple successful attempts exist for one condition")
    if not results:
        return None
    output = results[0].parent
    verify_result(context, output, require_checkpoint=True)
    return output


def run_group(
    bundle: str | Path, model: str, modality: str, host_path: str | Path, *, stage: str = "cv"
) -> Path:
    bundle, host_path = Path(bundle).resolve(), Path(host_path).resolve()
    host = load_host(host_path)
    context = read_bundle(bundle)
    study = context[0]
    if stage not in study["conditions"]:
        raise ValueError("Stage must be cv or final")
    conditions = [
        x for x in study["conditions"][stage] if x["family"] == model and x["modality"] == modality
    ]
    if not conditions:
        raise ValueError("The requested model and modality are not in the study")
    group_root = Path(host["output_root"]) / study["study_id"] / stage / f"{model}-{modality}"
    group_root.mkdir(parents=True, exist_ok=True)
    with (group_root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This execution group is already running") from error
        return _run_group(context, bundle, host_path, group_root, conditions)


def _run_group(context, bundle, host_path, group_root, conditions):
    logger = configure_run_logger(group_root / "group.log", "INFO")
    status_path = group_root / "status.json"
    state = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {"study_id": context[0]["study_id"], "conditions": {}}
    )
    if state["study_id"] != context[0]["study_id"]:
        raise ValueError("Execution group belongs to a different study")
    started = time.perf_counter()
    for completed, condition in enumerate(conditions):
        condition_id = condition["condition_id"]
        state.update(
            status="running",
            active_condition=condition_id,
            completed_conditions=completed,
            total_conditions=len(conditions),
            elapsed_seconds=time.perf_counter() - started,
        )
        _json(status_path, state)
        logger.info(
            "conditions=%d/%d (%.1f%%) active=%s elapsed=%.1fs ETA unavailable",
            completed,
            len(conditions),
            100 * completed / len(conditions),
            condition_id[:12],
            time.perf_counter() - started,
        )
        try:
            output = _existing(context, group_root, condition)
            if output is None:
                attempt = uuid.uuid4().hex
                state["conditions"][condition_id] = {"status": "running", "attempt_id": attempt}
                _json(status_path, state)
                with phase(logger, f"running {condition_id[:12]}"):
                    _run_subprocess(bundle, condition_id, host_path, group_root, attempt)
                output = _existing(context, group_root, condition)
                if output is None:
                    raise ValueError("Worker did not produce a verified result")
            tracking = json.loads((output / "wandb_run.json").read_text())
            state["conditions"][condition_id] = {
                "status": "completed",
                "output": str(output),
                "wandb_url": tracking.get("url"),
                "wandb_run_id": tracking.get("run_id"),
            }
            _json(status_path, state)
        except BaseException as error:
            state.update(status="failed", elapsed_seconds=time.perf_counter() - started)
            state["conditions"][condition_id] = {
                "status": "failed",
                "error_type": type(error).__name__,
            }
            _json(status_path, state)
            raise
    state.update(
        status="completed",
        active_condition=None,
        completed_conditions=len(conditions),
        elapsed_seconds=time.perf_counter() - started,
    )
    _json(status_path, state)
    logger.info(
        "conditions=%d/%d (100%%) status=completed elapsed=%.1fs",
        len(conditions),
        len(conditions),
        time.perf_counter() - started,
    )
    return group_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--host-config", type=Path, required=True)
    parser.add_argument("--group-root", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    context = read_bundle(args.study)
    matches = [
        item
        for stage in context[0]["conditions"].values()
        for item in stage
        if item["condition_id"] == args.condition
    ]
    if len(matches) != 1:
        raise ValueError("Worker condition identity is ambiguous or missing")
    host = load_host(args.host_config)
    host["output_root"] = str(args.group_root.resolve())
    output = run_condition(args.study, matches[0], host, args.attempt)
    write_result(context, matches[0], output)


if __name__ == "__main__":
    main()
