from __future__ import annotations

from typing import Any

from .base import TabularClassifier
from .tabiclv2 import TabICLv2Model


class ModelRegistryError(ValueError):
    pass


def create_model(model_config: dict[str, Any]) -> TabularClassifier:
    name = model_config.get("name")
    if name == "tabiclv2":
        return TabICLv2Model(model_config)
    raise ModelRegistryError(f"No model adapter is registered for '{name}'")
