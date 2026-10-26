from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self


@dataclass(frozen=True)
class TextExample:
    participant_id: str
    text: str
    target: float | None = None


@dataclass(frozen=True)
class TextPrediction:
    value: float
    probability_positive: float | None
    raw_answer: str
    prompt_tokens: int


class TextModel(Protocol):
    def prepare(self, demonstrations: Sequence[TextExample]) -> None: ...

    def preflight(self, examples: Sequence[TextExample]) -> None: ...

    def fit(
        self,
        training: Sequence[TextExample],
        validation: Sequence[TextExample],
        checkpoint_directory: Path,
        progress: Callable[[int, int], None] | None = None,
        resume: bool = False,
    ) -> Self: ...

    def predict(self, examples: Sequence[TextExample]) -> list[TextPrediction]: ...

    def run_metadata(self) -> dict[str, Any]: ...
