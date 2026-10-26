from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class RunArtifacts:
    def __init__(self, output_root: str | Path, model_name: str, feature_set: str, seed: int,
                 *, resume_path: str | Path | None = None) -> None:
        if resume_path is not None:
            self.path = Path(resume_path).expanduser().resolve()
            if not self.path.is_dir():
                raise ValueError("Resume directory does not exist")
            return
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = Path(output_root).expanduser().resolve() / (
            f"{timestamp}_{model_name}_{feature_set}_{seed}"
        )
        self.path.mkdir(parents=True, exist_ok=False)

    def json(self, name: str, value: Any) -> Path:
        path = self.path / name
        path.write_text(json.dumps(value, indent=2, default=str, allow_nan=True), encoding="utf-8")
        return path

    def csv(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.path / name
        frame.to_csv(path, index=False)
        return path

    def text(self, name: str, value: str) -> Path:
        path = self.path / name
        path.write_text(value, encoding="utf-8")
        return path
