from __future__ import annotations

import math
from collections.abc import Mapping

from daic_foundation_tab.models.capabilities import model_capabilities


def validate_text_config(config: Mapping) -> None:
    from daic_foundation_tab.config import ConfigError

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ConfigError(message)

    for section in ("project", "runtime", "data", "experiment", "model", "evaluation", "tracking"):
        require(
            isinstance(config.get(section), Mapping), f"Missing configuration section: {section}"
        )
    task = config["experiment"].get("task")
    mode = config["experiment"].get("mode", "fine_tuning")
    try:
        capability = model_capabilities(config["model"].get("name", ""))
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error
    require(capability.input_kind == "transcript", "Transcript input requires a text model")
    require(task in capability.tasks, "Unsupported transcript task")
    require(mode in capability.modes, "Unsupported transcript adaptation mode")
    require(config["model"].get("model_id") == "Qwen/Qwen3.5-2B", "Use Qwen/Qwen3.5-2B")
    require(bool(config["data"].get("dataset_id")), "data.dataset_id is required")
    transcript = config["data"].get("transcripts", {})
    require(bool(transcript.get("pattern")), "data.transcripts.pattern is required")
    require(
        transcript.get("speaker_policy", "participant_only") in {"participant_only", "dialogue"},
        "speaker_policy must be participant_only or dialogue",
    )
    require(config["runtime"].get("require_cuda") is True, "Text runs require CUDA")
    require(config["runtime"].get("device") == "cuda:0", "runtime.device must be cuda:0")
    parameters = config["model"].get("parameters", {})
    require(isinstance(parameters, Mapping), "model.parameters must be a mapping")
    require(parameters.get("dtype", "bfloat16") == "bfloat16", "Text profiles require bfloat16")
    require(
        not parameters.get("quantization"), "Quantization is not supported by the initial profile"
    )
    require(parameters.get("enable_thinking", False) is False, "Thinking mode must be disabled")
    for key, minimum in (
        ("max_length", 2),
        ("max_new_tokens", 1),
        ("epochs", 1),
        ("gradient_accumulation_steps", 1),
        ("lora_rank", 1),
        ("checkpoint_steps", 1),
    ):
        value = parameters.get(
            key,
            {
                "max_new_tokens": 8,
                "epochs": 1,
                "gradient_accumulation_steps": 1,
                "lora_rank": 8,
                "checkpoint_steps": 10,
            }.get(key),
        )
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"model.parameters.{key} must be an integer >= {minimum}",
        )
    require(
        parameters.get("max_new_tokens", 8) < parameters["max_length"],
        "max_new_tokens must be smaller than max_length",
    )
    learning_rate = parameters.get("learning_rate", 0.0002)
    require(
        isinstance(learning_rate, (int, float))
        and not isinstance(learning_rate, bool)
        and math.isfinite(learning_rate)
        and learning_rate > 0,
        "Invalid learning_rate",
    )
    shots = parameters.get("demonstrations", 0)
    require(
        isinstance(shots, int) and not isinstance(shots, bool) and shots >= 0,
        "demonstrations must be a nonnegative integer",
    )
    require(mode != "fine_tuning" or shots == 0, "LoRA profiles use zero demonstrations")
    evaluation = config["evaluation"]
    threshold = evaluation.get("threshold")
    require(
        (task == "regression" and threshold is None)
        or (
            task == "classification"
            and isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and 0 <= threshold <= 1
        ),
        "Use a classification threshold in [0,1], or null for regression",
    )
    fraction = evaluation.get("fine_tuning", {}).get("validation_fraction")
    require(
        isinstance(fraction, (int, float)) and 0 < fraction < 1,
        "validation_fraction must be between zero and one",
    )
    require(
        not evaluation.get("repeated_holdout", {}).get("enabled", False),
        "Text campaigns and repeated holdouts are deferred",
    )
    require(
        not config.get("bootstrap", {}).get("enabled", False),
        "Text bootstrap campaigns are deferred",
    )
