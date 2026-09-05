from __future__ import annotations

from typing import Any

from .base import FineTunableClassifier
from .tabiclv2_ft import TabICLv2FineTunedModel


class ModelRegistryError(ValueError):
    pass


def create_model(model_config: dict[str, Any]) -> FineTunableClassifier:
    name = model_config.get("name")
    if name == "tabiclv2_ft":
        return TabICLv2FineTunedModel(model_config)
    raise ModelRegistryError("Only 'tabiclv2_ft' is supported; the standalone 'tabiclv2' ICL path was retired")
