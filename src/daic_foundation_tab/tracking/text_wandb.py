from __future__ import annotations

import math


class TextTracker:
    def __init__(self, config: dict, artifacts, model_metadata: dict):
        self.run = None
        self.artifacts = artifacts
        self.settings = config["tracking"].get("wandb", {})
        if (
            self.settings.get("enabled", False)
            and self.settings.get("mode", "disabled") != "disabled"
        ):
            import wandb

            self.run = wandb.init(
                project=self.settings.get("project", "daic-foundation-tab"),
                entity=self.settings.get("entity"),
                id=artifacts.path.name,
                resume="allow",
                dir=str(artifacts.path),
                mode=self.settings.get("mode", "offline"),
                config={
                    "model_id": model_metadata["model_id"],
                    "checkpoint_revision": model_metadata["checkpoint_revision"],
                    "task": config["experiment"]["task"],
                    "mode": config["experiment"].get("mode", "fine_tuning"),
                    "seed": config["project"]["seed"],
                    "speaker_policy": config["data"]["transcripts"].get(
                        "speaker_policy", "participant_only"
                    ),
                },
                settings=wandb.Settings(
                    disable_git=True, save_code=False, x_save_requirements=False
                ),
            )
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
            self.run.log(
                {
                    f"{prefix}/{key}": value
                    for key, value in values.items()
                    if isinstance(value, (float, int)) and math.isfinite(value)
                }
            )

    def finish(self, failed=False):
        if self.run:
            self.run.finish(exit_code=1 if failed else 0)
        self._record("failed" if failed else "completed")
