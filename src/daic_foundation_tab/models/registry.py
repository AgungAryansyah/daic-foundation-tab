from __future__ import annotations

from typing import Any

from .base import FineTunableClassifier, FineTunableRegressor
from .tabiclv2_ft import TabICLv2FineTunedModel
from .tabiclv2_ft_regressor import TabICLv2FineTunedRegressor


class ModelRegistryError(ValueError):
    pass


def create_model(model_config: dict[str, Any]) -> FineTunableClassifier | FineTunableRegressor:
    name = model_config.get("name")
    if name == "tabiclv2_ft":
        return TabICLv2FineTunedModel(model_config)
    if name == "tabiclv2_ft_regressor":
        return TabICLv2FineTunedRegressor(model_config)
    raise ModelRegistryError("Unsupported model; the standalone 'tabiclv2' ICL path was retired")
