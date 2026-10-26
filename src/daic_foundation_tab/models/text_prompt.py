from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from daic_foundation_tab.data.participants import fingerprint

from .text_base import TextExample


def parse_score(answer: str) -> float:
    if not re.fullmatch(r"\s*(?:\d+(?:\.\d+)?|\.\d+)\s*", answer):
        raise ValueError(f"Invalid PHQ-8 numeric answer: {answer!r}")
    value = float(answer)
    if not math.isfinite(value) or not 0 <= value <= 24:
        raise ValueError(f"PHQ-8 answer is outside [0,24]: {answer!r}")
    return value


def answer_text(example: TextExample, task: str) -> str:
    if example.target is None:
        raise ValueError("Training and demonstration examples require targets")
    if task == "classification":
        if example.target not in (0, 1):
            raise ValueError("Binary targets must be zero or one")
        return str(int(example.target))
    return format(parse_score(str(example.target)), "g")


def select_demonstrations(pool: Sequence[TextExample], count: int, seed: int) -> list[TextExample]:
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(pool):
        raise ValueError("Demonstration count exceeds the available context participants")
    ordered = sorted(pool, key=lambda item: item.participant_id)
    if len({item.participant_id for item in ordered}) != len(ordered):
        raise ValueError("Duplicate demonstration participants")
    indices = np.random.default_rng(seed).permutation(len(ordered))[:count]
    return [ordered[index] for index in indices]


class PromptEncoder:
    def __init__(
        self, tokenizer, task: str, parameters: dict, identity: dict, cache_root: Path | None = None
    ):
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = parameters["max_length"]
        self.max_new_tokens = parameters.get("max_new_tokens", 8)
        self.identity = identity
        self.cache_root = cache_root
        self.demonstrations: list[TextExample] = []

    def messages(self, example: TextExample) -> list[dict[str, str]]:
        instruction = (
            "Assess depression from the interview transcript. Return only 0 for PHQ-8 below 10 "
            "or 1 for PHQ-8 at least 10."
            if self.task == "classification"
            else "Estimate the PHQ-8 total from the interview transcript. Return only one numeric "
            "score between 0 and 24."
        )
        messages = [{"role": "system", "content": instruction}]
        for demonstration in self.demonstrations:
            messages.extend(
                [
                    {"role": "user", "content": self._input(demonstration)},
                    {"role": "assistant", "content": answer_text(demonstration, self.task)},
                ]
            )
        messages.append({"role": "user", "content": self._input(example)})
        return messages

    @staticmethod
    def _input(example: TextExample) -> str:
        return f"Interview transcript:\n<transcript>\n{example.text}\n</transcript>"

    def prompt(self, example: TextExample) -> list[int]:
        messages = self.messages(example)
        key = fingerprint(
            {"version": 1, "identity": self.identity, "messages": messages, "thinking": False}
        )
        cache = self.cache_root / f"{key}.json" if self.cache_root else None
        if cache and cache.is_file():
            tokens = json.loads(cache.read_text(encoding="utf-8"))
        else:
            tokens = list(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".tmp")
                temporary.write_text(json.dumps(tokens), encoding="utf-8")
                temporary.replace(cache)
        if not tokens:
            raise ValueError("Chat template produced an empty prompt")
        self.check_length(len(tokens), self.max_new_tokens)
        return tokens

    def completion(self, answer: str) -> list[int]:
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define an EOS token")
        return list(self.tokenizer.encode(answer, add_special_tokens=False)) + [
            self.tokenizer.eos_token_id
        ]

    def check_length(self, prompt_tokens: int, answer_tokens: int) -> None:
        if prompt_tokens + answer_tokens > self.max_length:
            raise ValueError(
                f"Context overflow: {prompt_tokens} prompt + {answer_tokens} answer tokens "
                f"exceeds max_length={self.max_length}; no truncation was applied"
            )

    def training_record(self, example: TextExample) -> dict[str, list[int]]:
        prompt = self.prompt(example)
        completion = self.completion(answer_text(example, self.task))
        self.check_length(len(prompt), len(completion))
        tokens = prompt + completion
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "labels": [-100] * len(prompt) + completion,
        }
