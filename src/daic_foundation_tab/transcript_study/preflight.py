from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.models.registry import create_text_model
from daic_foundation_tab.models.text_prompt import select_demonstrations
from daic_foundation_tab.tabular_study.bundle import _json
from daic_foundation_tab.tracking.logging import configure_run_logger, phase
from daic_foundation_tab.tracking.runtime import (
    cuda_peak_memory,
    require_cuda_device,
    reset_cuda_peak_memory,
)
from daic_foundation_tab.tracking.seeds import set_global_seed

from .audit import require_audit
from .bundle import MODES, TASKS, dataset_for, read_bundle
from .condition import execution_config
from .resources import (
    configure_threads,
    load_host,
    memory_snapshot,
    observe_resources,
    profile_identity,
    profile_path,
    require_capacity,
    worker_environment,
)


def require_preflight(bundle: Path, context: tuple, host: dict, task: str, stage: str) -> dict:
    study = context[0]
    audit = require_audit(bundle, study)
    identity = profile_identity(study, audit, host, task, stage)
    path = profile_path(host, identity)
    if not path.is_file():
        raise ValueError("Run a successful preflight for this study, task, stage, and host first")
    profile = json.loads(path.read_text())
    payload = {key: value for key, value in profile.items() if key != "profile_id"}
    if (profile.get("identity") != identity or profile.get("status") != "passed"
            or profile.get("profile_id") != fingerprint(payload)
            or any(not isinstance(profile.get(key), (int, float)) or not math.isfinite(profile[key]) or profile[key] <= 0
                   for key in ("peak_process_ram_mb", "peak_cuda_reserved_mb"))):
        raise ValueError("Transcript preflight is failed, changed, or incompatible")
    require_capacity(profile)
    return profile


def _longest(dataset, audit: dict, role: str, *, labeled: bool = False):
    lengths = {row["participant_id"]: row["total_tokens"] for row in audit["records"] if row["role"] == role}
    return max(dataset.examples(role, labeled=labeled), key=lambda example: lengths[example.participant_id])


def _probe(context: tuple, audit: dict, host: dict, task: str, stage: str, logger) -> dict:
    import torch

    study = context[0]
    peaks = {"peak_process_ram_mb": 0.0, "peak_cuda_reserved_mb": 0.0, "peak_cuda_allocated_mb": 0.0}
    for completed, mode in enumerate(MODES, 1):
        candidates = [c for c in study["conditions"][stage] if c["task"] == task and c["mode"] == mode]
        relevant_role = "fit" if mode == "fine_tuning" else ("evaluation" if stage == "cv" else "dev")
        condition = max(candidates, key=lambda c: max(row["total_tokens"] for row in audit["conditions"][c["condition_id"]]["records"]
                                                     if row["role"] == relevant_role))
        detail = audit["conditions"][condition["condition_id"]]
        dataset = dataset_for(context, condition)
        config = execution_config(study, condition, host)
        config["tracking"]["wandb"].update(enabled=False, mode="disabled")
        config["model"]["parameters"].update(epochs=1, gradient_accumulation_steps=1, checkpoint_steps=1)
        set_global_seed(condition["model_seed"])
        reset_cuda_peak_memory("cuda:0")
        with phase(logger, f"preflighting {mode}"), observe_resources(logger) as resources:
            model = create_text_model(config)
            demonstrations = select_demonstrations(dataset.examples("fit", labeled=True),
                                                   config["model"]["parameters"]["demonstrations"], condition["model_seed"])
            model.prepare(demonstrations)
            if model.run_metadata()["checkpoint_revision"] != config["model"]["revision"]:
                raise ValueError("Preflight loaded a different model checkpoint")
            if mode == "fine_tuning":
                training = [_longest(dataset, detail, "fit", labeled=True)]
                selection = dataset.examples("selection", labeled=True)
                lengths = {r["participant_id"]: r["total_tokens"] for r in detail["records"]}
                if task == "classification":
                    validation = [max((x for x in selection if x.target == label), key=lambda x: lengths[x.participant_id])
                                  for label in (0, 1)]
                else:
                    validation = [_longest(dataset, detail, "selection", labeled=True)]
                model.preflight([*training, *validation])
                with tempfile.TemporaryDirectory(prefix="transcript-probe-", dir=host["output_root"]) as temporary:
                    model.fit(training, validation, Path(temporary),
                              progress=lambda n, total: logger.info("phase=probe training optimizer_steps=%d/%d", n, total))
            for role in (("evaluation",) if stage == "cv" else ("dev", "test")):
                query = _longest(dataset, detail, role)
                model.preflight([query])
                prediction = model.predict([replace(query, target=None)])
                if len(prediction) != 1 or not math.isfinite(prediction[0].value):
                    raise ValueError("Preflight returned invalid predictions")
            torch.cuda.synchronize()
            metrics = cuda_peak_memory("cuda:0")
        for key, value in peaks.items():
            peaks[key] = max(value, resources.get(key, metrics.get(key, 0)))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("phase=preflight progress=%d/2 methods (%.1f%%)", completed, completed * 50)
    return peaks


def preflight_worker(bundle: Path, host_path: Path, task: str, stage: str) -> Path:
    host = load_host(host_path)
    configure_threads(host)
    context = read_bundle(bundle)
    study = context[0]
    if task not in TASKS or stage not in study["conditions"]:
        raise ValueError("Unsupported transcript preflight task or stage")
    audit = require_audit(bundle, study)
    directory = Path(host["output_root"]) / study["study_id"] / "preflight"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    logger = configure_run_logger(directory / f"{task}-{stage}.log", "INFO")
    status_path = directory / f"{task}-{stage}.status.json"
    _json(status_path, {"status": "running", "task": task, "stage": stage})
    try:
        require_cuda_device("cuda:0")
        require_capacity({"peak_process_ram_mb": 2048 / 1.2, "peak_cuda_reserved_mb": 8192 / 1.2})
        identity = profile_identity(study, audit, host, task, stage)
        with phase(logger, "full-checkpoint transcript preflight"):
            measured = _probe(context, audit, host, task, stage, logger)
        profile = {"version": 1, "status": "passed", "identity": identity, **measured,
                   "memory": memory_snapshot()}
        profile["profile_id"] = fingerprint(profile)
        destination = profile_path(host, identity)
        _json(destination, profile)
        _json(status_path, {"status": "passed", "profile": str(destination)})
        return destination
    except BaseException as error:
        _json(status_path, {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def preflight_study(bundle: Path, host_path: Path, task: str, *, stage: str = "cv") -> None:
    host = load_host(host_path)
    subprocess.run([sys.executable, "-u", "-m", __name__, "--study", str(Path(bundle).resolve()),
                    "--host-config", str(Path(host_path).resolve()), "--task", task, "--stage", stage],
                   env=worker_environment(host), check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--host-config", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--stage", choices=("cv", "final"), default="cv")
    args = parser.parse_args()
    print(preflight_worker(args.study, args.host_config, args.task, args.stage))
