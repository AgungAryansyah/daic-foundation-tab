from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    family: str
    input_kind: str
    tasks: tuple[str, ...] = ("classification", "regression")
    modes: tuple[str, ...] = ("frozen_icl", "fine_tuning")


MODEL_FAMILIES = {
    family: ModelCapabilities(family, kind)
    for family, kind in (
        ("tabpfn35", "tabular"),
        ("tabiclv2", "tabular"),
        ("kumo_medium", "tabular"),
        ("qwen35_2b", "transcript"),
    )
}


def model_capabilities(name: str) -> ModelCapabilities:
    if not isinstance(name, str):
        raise ValueError("Model name must be a string")
    family = name.removesuffix("_regressor").removesuffix("_ft").removesuffix("_icl")
    try:
        return MODEL_FAMILIES[family]
    except KeyError as error:
        raise ValueError(f"Unsupported model family: {name}") from error
