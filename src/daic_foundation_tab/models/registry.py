from __future__ import annotations

from typing import Any

from .base import FineTunableClassifier, FineTunableRegressor
from .kumo_medium_ft import KumoMediumFineTunedModel, KumoMediumFineTunedRegressor
from .tabiclv2_ft import TabICLv2FineTunedModel
from .tabiclv2_ft_regressor import TabICLv2FineTunedRegressor
from .tabpfn35_ft import TabPFN35FineTunedModel, TabPFN35FineTunedRegressor


class ModelRegistryError(ValueError):
    pass


def create_model(model_config: dict[str, Any]) -> FineTunableClassifier | FineTunableRegressor:
    name = model_config.get("name")
    if name == "kumo_medium_ft":
        return KumoMediumFineTunedModel(model_config)
    if name == "kumo_medium_ft_regressor":
        return KumoMediumFineTunedRegressor(model_config)
    if name == "tabiclv2_ft":
        return TabICLv2FineTunedModel(model_config)
    if name == "tabiclv2_ft_regressor":
        return TabICLv2FineTunedRegressor(model_config)
    if name == "tabpfn35_ft":
        return TabPFN35FineTunedModel(model_config)
    if name == "tabpfn35_ft_regressor":
        return TabPFN35FineTunedRegressor(model_config)
    raise ModelRegistryError("Unsupported model; the standalone 'tabiclv2' ICL path was retired")
