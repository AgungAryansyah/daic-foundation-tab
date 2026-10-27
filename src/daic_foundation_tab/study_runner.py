from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from daic_foundation_tab.config import load_config
from daic_foundation_tab.data.participants import fingerprint, source_matches
from daic_foundation_tab.matched_study import FAMILIES, TASKS, _save_json, verify_prepared_study
from daic_foundation_tab.tracking.logging import configure_run_logger, phase


def _input_identity(config: dict) -> str:
    root = Path(config["data"]["root"])
    manifest = json.loads(Path(config["experiment"]["partition_manifest"]).read_text())
    patterns = [group["pattern"] for group in config["data"]["source_groups"].values()]
    patterns.append(config["data"]["transcripts"]["pattern"])
    paths = {root / name for name in config["data"]["labels"].values()}
    truth = config["data"].get("test_ground_truth")
    if truth:
        paths.add(root / truth)
    for row in manifest["participants"]:
        for pattern in patterns:
            matches = source_matches(root, pattern, str(row["participant_id"]))
            if len(matches) != 1:
                raise ValueError("A prepared study input is missing or ambiguous")
            paths.update(matches)
    return fingerprint(
        [
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            if path.exists()
            else (str(path), None, None)
            for path in sorted(paths)
        ]
    )


def _completed_output(result_path: Path, entry: dict, config: dict, revision: str | None) -> Path:
    result = json.loads(result_path.read_text())
    if result["identity"] != entry["identity"]:
        raise ValueError("Completed condition configuration mismatch")
    output = Path(result["output"]).resolve()
    if output.parent != Path(config["project"]["output_root"]).resolve():
        raise ValueError("Completed condition output directory mismatch")
    record = json.loads((output / "study_run.json").read_text())
    expected = {
        "status": "completed",
        "paired_comparison_verified": True,
        "dataset_id": config["data"]["dataset_id"],
        "task": config["experiment"]["task"],
        "model_name": config["model"]["name"],
        "mode": config["experiment"]["mode"],
        "input_kind": config["input"]["kind"],
        "manifest_id": config["study"]["manifest_id"],
        "cohort_id": config["study"]["cohort_id"],
        "source_revision": revision,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed condition provenance mismatch: {entry['name']}")
    resolved = load_config(output / "config_resolved.yaml")
    if (
        fingerprint({key: value for key, value in resolved.items() if not key.startswith("_")})
        != entry["identity"]
    ):
        raise ValueError("Completed run settings differ from the prepared condition")
    manifest = json.loads(Path(config["experiment"]["partition_manifest"]).read_text())
    for split in ("dev", "test"):
        table = pd.read_csv(output / f"predictions_{split}.csv", dtype={"participant_id": str})
        expected_ids = {
            str(row["participant_id"]) for row in manifest["participants"] if row["split"] == split
        }
        if (
            table["participant_id"].duplicated().any()
            or set(table["participant_id"]) != expected_ids
        ):
            raise ValueError("Completed predictions do not match the shared participants")
        if set(table["split"]) != {split}:
            raise ValueError("Completed predictions have an incorrect split")
        columns = (
            ["y_pred", "prob_depressed"]
            if config["experiment"]["task"] == "classification"
            else ["phq8_pred"]
        )
        if not np.isfinite(table[columns].to_numpy(dtype=float)).all():
            raise ValueError("Completed predictions must be finite")
        if "prob_depressed" in columns and not table["prob_depressed"].between(0, 1).all():
            raise ValueError("Completed classification probabilities must be within [0,1]")
    for filename in ("metrics_dev.json", "runtime.json", "wandb_run.json"):
        if not (output / filename).is_file():
            raise ValueError(f"Completed run is missing {filename}")
    truth = config["data"].get("test_ground_truth")
    if (
        truth
        and (Path(config["data"]["root"]) / truth).is_file()
        and not (output / "metrics_test.json").is_file()
    ):
        raise ValueError("Completed run is missing available test metrics")
    tracking = config["tracking"]["wandb"]
    if tracking["enabled"] and tracking["mode"] != "disabled":
        saved = json.loads((output / "wandb_run.json").read_text())
        if saved.get("status") not in {"finished", "completed"} or not saved.get("run_id"):
            raise ValueError("Completed condition has no completed W&B record")
    return output


def _execute_condition(config_path: Path, result_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "daic_foundation_tab.study_runner",
            "--config",
            str(config_path),
            "--result",
            str(result_path),
        ],
        check=True,
    )


def run_study(destination: str | Path) -> Path:
    destination = Path(destination).resolve()
    with (destination / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This study is already running") from error
        return _run_study(destination)


def _run_study(destination: Path) -> Path:
    from daic_foundation_tab.runner import _git_commit

    logger = configure_run_logger(destination / "run.log", "INFO")
    with phase(logger, "validating prepared study"):
        conditions = verify_prepared_study(destination)
        inputs = {entry["name"]: _input_identity(config) for entry, config in conditions}
    revision = _git_commit()
    identity = fingerprint(
        {
            "conditions": [entry for entry, _ in conditions],
            "inputs": inputs,
            "source_revision": revision,
        }
    )
    state_path = destination / "status.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "identity": identity,
            "source_revision": revision,
            "conditions": {},
        }
    )
    if state["identity"] != identity:
        raise ValueError(
            "Study inputs, source revision, or settings changed. Use a new study directory"
        )
    started = time.perf_counter()
    results = destination / "results"
    results.mkdir(exist_ok=True)
    for completed, (entry, config) in enumerate(conditions):
        name = entry["name"]
        result_path = results / f"{name}.json"
        state.update(
            status="running",
            active_condition=name,
            completed_conditions=completed,
            total_conditions=len(conditions),
            elapsed_seconds=time.perf_counter() - started,
        )
        _save_json(state_path, state)
        logger.info(
            "conditions=%d/%d (%.1f%%) active=%s elapsed=%.1fs ETA unavailable",
            completed,
            len(conditions),
            100 * completed / len(conditions),
            name,
            time.perf_counter() - started,
        )
        try:
            if _input_identity(config) != inputs[name] or _git_commit() != revision:
                raise ValueError("Study inputs or source revision changed during execution")
            if not result_path.exists():
                state["conditions"][name] = {"status": "running"}
                _save_json(state_path, state)
                with phase(logger, f"running {name}"):
                    _execute_condition(destination / entry["config"], result_path)
            output = _completed_output(result_path, entry, config, revision)
            if _input_identity(config) != inputs[name] or _git_commit() != revision:
                raise ValueError("Study inputs or source revision changed during execution")
            tracking = json.loads((output / "wandb_run.json").read_text())
            state["conditions"][name] = {
                "status": "completed",
                "output": str(output),
                "wandb_url": tracking.get("url"),
                "wandb_run_id": tracking.get("run_id"),
            }
        except BaseException as error:
            state["status"] = "failed"
            state["elapsed_seconds"] = time.perf_counter() - started
            state["conditions"][name] = {"status": "failed", "error_type": type(error).__name__}
            _save_json(state_path, state)
            raise
        _save_json(state_path, state)
    state.update(active_condition=None, completed_conditions=len(conditions))
    _save_json(state_path, state)
    try:
        with phase(logger, "summarizing matched study"):
            summarize_study(destination, conditions, revision)
    except BaseException as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.perf_counter() - started,
        )
        _save_json(state_path, state)
        raise
    state.update(
        status="completed",
        active_condition=None,
        completed_conditions=len(conditions),
        elapsed_seconds=time.perf_counter() - started,
    )
    _save_json(state_path, state)
    logger.info(
        "conditions=%d/%d (100%%) status=completed elapsed=%.1fs",
        len(conditions),
        len(conditions),
        time.perf_counter() - started,
    )
    return destination


def summarize_study(destination: Path, conditions: list, revision: str | None) -> None:
    rows = []
    for entry, config in conditions:
        output = _completed_output(
            destination / "results" / f"{entry['name']}.json", entry, config, revision
        )
        row = {
            "condition": entry["name"],
            "task": config["experiment"]["task"],
            "mode": config["experiment"]["mode"],
            "model": config["model"]["name"],
            "run": str(output),
        }
        for prefix, filename in (
            ("development", "metrics_dev.json"),
            ("test", "metrics_test.json"),
            ("runtime", "runtime.json"),
        ):
            path = output / filename
            if path.is_file():
                row.update(
                    {
                        f"{prefix}/{key}": value
                        for key, value in json.loads(path.read_text()).items()
                        if isinstance(value, (int, float))
                    }
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(destination / "metrics.csv", index=False)
    differences = []
    indexed = table.set_index("condition")
    for task in TASKS:
        for family in FAMILIES:
            left = indexed.loc[f"{task}_{family}_frozen_icl"]
            right = indexed.loc[f"{task}_{family}_fine_tuning"]
            for column in table.columns:
                if "/" in column and pd.notna(left[column]) and pd.notna(right[column]):
                    differences.append(
                        {
                            "task": task,
                            "family": family,
                            "metric": column,
                            "frozen_icl": left[column],
                            "fine_tuning": right[column],
                            "delta_ft_minus_icl": right[column] - left[column],
                        }
                    )
    pd.DataFrame(differences).to_csv(destination / "adaptation_differences.csv", index=False)


def main() -> None:
    import argparse

    from daic_foundation_tab.runner import run_experiment

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    output = run_experiment(config)
    _save_json(args.result, {"identity": fingerprint(payload), "output": str(output)})


if __name__ == "__main__":
    main()
