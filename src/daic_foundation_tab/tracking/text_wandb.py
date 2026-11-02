from __future__ import annotations

import math
from pathlib import Path

from dotenv import load_dotenv

from .wandb import _redact_paths, _scalar_values


class TextTracker:
    def __init__(self, config: dict, artifacts, model_metadata: dict, *, environment: dict | None = None):
        self.run = None
        self.artifacts = artifacts
        self.settings = config["tracking"].get("wandb", {})
        if (
            self.settings.get("enabled", False)
            and self.settings.get("mode", "disabled") != "disabled"
        ):
            import wandb

            load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
            self.run = wandb.init(
                project=self.settings.get("project", "daic-foundation-tab"),
                entity=self.settings.get("entity"),
                id=artifacts.path.name,
                resume="allow",
                dir=str(artifacts.path),
                mode=self.settings.get("mode", "offline"),
                name=f"{config['experiment'].get('name', 'qwen')}-{artifacts.path.name}",
                group=self.settings.get("group"),
                job_type=config["experiment"].get("mode", "fine_tuning"),
                tags=list(dict.fromkeys([
                    config["model"]["name"], "transcript", config["experiment"]["task"],
                    config["experiment"].get("mode", "fine_tuning"),
                    *self.settings.get("tags", []),
                ])),
                config={
                    "dataset_id": config["data"].get("dataset_id"),
                    "model_id": model_metadata["model_id"],
                    "checkpoint_revision": model_metadata["checkpoint_revision"],
                    "task": config["experiment"]["task"],
                    "mode": config["experiment"].get("mode", "fine_tuning"),
                    "seed": config["project"]["seed"],
                    "parameters": _redact_paths(config["model"].get("parameters", {})),
                    "study": {
                        key: config.get("study", {})[key]
                        for key in ("campaign_id", "manifest_id", "cohort_id", "required_inputs",
                                    "transcript_study_id", "parent_study_id", "software_identity", "condition_id",
                                    "stage", "fold", "model_seed", "host_label", "execution_group", "cpu_threads")
                        if key in config.get("study", {})
                    },
                    "speaker_policy": config["data"]["transcripts"].get(
                        "speaker_policy", "participant_only"
                    ),
                    "environment": {key: (environment or {})[key] for key in (
                        "git_commit", "host_label", "gpu_name", "total_gpu_vram_bytes", "cuda_version",
                        "cpu_threads", "container_ram_limit_mb", "dependency_versions"
                    ) if key in (environment or {})},
                },
                settings=wandb.Settings(
                    disable_git=True, save_code=False, x_save_requirements=False,
                    console="off" if config.get("study", {}).get("transcript_study_id") else "auto",
                ),
            )
            self.run.define_metric("fine_tuning/epoch", hidden=True)
            self.run.define_metric("fine_tuning/*", step_metric="fine_tuning/epoch")
        self._record("running" if self.run else "disabled")

    def _record(self, status):
        self.artifacts.json(
            "wandb_run.json",
            {
                "enabled": self.settings.get("enabled", False),
                "status": status,
                "run_id": getattr(self.run, "id", None),
                "url": getattr(self.run, "url", None),
            },
        )

    def metrics(self, prefix, values):
        if self.run:
            scalars = {
                key: value for key, value in _scalar_values(prefix, values).items()
                if isinstance(value, (float, int)) and math.isfinite(value)
            }
            self.run.log(scalars)
            self.run.summary.update(scalars)

    def finish(self, failed=False):
        if self.run:
            self.run.finish(exit_code=1 if failed else 0)
        self._record("failed" if failed else "completed")
