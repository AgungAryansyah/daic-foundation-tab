from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import torch

from .kumo_medium_ft import CHECKPOINT_REVISION, MODEL_REPOSITORY, MODEL_VERSION, PACKAGE_REVISION


class FrozenICLError(RuntimeError):
    pass


class FrozenICLModel:
    task = "classification"
    family = ""

    def __init__(self, model_config: dict[str, Any]) -> None:
        self._config = model_config
        self._parameters = dict(model_config["parameters"])
        self._estimator: Any = None
        self._checkpoint_path: Path | None = None
        self._weight_digest: str | None = None
        self._columns: list[str] = []
        self._classes: np.ndarray | None = None
        self._context_rows = 0

    def validate_input(self, features: pd.DataFrame) -> None:
        if features.empty or not features.columns.is_unique:
            raise FrozenICLError("Frozen ICL requires nonempty features with unique columns")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
            raise FrozenICLError("Frozen ICL participant features must be numerical")
        if np.isinf(features.to_numpy(dtype=float)).any():
            raise FrozenICLError("Frozen ICL features must not contain infinity")

    def _load_and_fit_context(self, features: pd.DataFrame, target: pd.Series) -> Path:
        raise NotImplementedError

    @staticmethod
    def _state_digest(modules: list[torch.nn.Module]) -> str:
        digest = hashlib.sha256()
        for index, module in enumerate(modules):
            digest.update(str(index).encode())
            for name, value in sorted(module.state_dict().items()):
                digest.update(name.encode())
                digest.update(
                    value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                )
        return digest.hexdigest()

    def _modules(self) -> list[torch.nn.Module]:
        if self._estimator is None:
            raise FrozenICLError("Call fit_context before prediction")
        if self.family == "kumo_medium":
            modules = [self._estimator]
        else:
            fitted_models = getattr(self._estimator, "models_", None)
            if fitted_models is not None:
                modules = (
                    [fitted_models]
                    if isinstance(fitted_models, torch.nn.Module)
                    else list(fitted_models)
                )
            else:
                module = getattr(self._estimator, "model_", None)
                modules = [module] if module is not None else []
            if not modules:
                raise FrozenICLError("The pretrained model weights were not initialized")
        if any(not isinstance(module, torch.nn.Module) for module in modules):
            raise FrozenICLError("The pretrained model does not expose a PyTorch module")
        return modules

    def _freeze(self) -> None:
        modules = self._modules()
        for module in modules:
            module.eval()
            module.requires_grad_(False)
        if any(parameter.requires_grad for module in modules for parameter in module.parameters()):
            raise FrozenICLError("Pretrained model parameters must remain frozen")
        self._weight_digest = self._state_digest(modules)

    def _verify_frozen(self) -> list[torch.nn.Module]:
        modules = self._modules()
        if any(
            module.training or any(parameter.requires_grad for parameter in module.parameters())
            for module in modules
        ):
            raise FrozenICLError("Frozen ICL model left evaluation mode or enabled gradients")
        if self._state_digest(modules) != self._weight_digest:
            raise FrozenICLError("Pretrained model weights changed during inference")
        return modules

    def fit_context(self, features: pd.DataFrame, target: pd.Series) -> Self:
        self.validate_input(features)
        if len(features) != len(target) or not len(features):
            raise FrozenICLError("Context targets must be nonempty and aligned with features")
        numeric_target = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric_target).all():
            raise FrozenICLError("Context targets must be finite")
        if self.task == "classification" and set(np.unique(numeric_target)) != {0.0, 1.0}:
            raise FrozenICLError("Classification context must contain both binary classes only")

        self._columns = list(features.columns)
        self._classes = np.array([0, 1]) if self.family == "kumo_medium" else None
        self._context_rows = len(features)
        self._estimator = None
        self._weight_digest = None
        with torch.inference_mode():
            self._checkpoint_path = self._load_and_fit_context(features, target)
        if self.task == "classification" and self.family != "kumo_medium":
            classes = getattr(self._estimator, "classes_", None)
            if classes is None or set(np.asarray(classes).tolist()) != {0, 1}:
                raise FrozenICLError("Classifier context must initialize classes 0 and 1")
            self._classes = np.asarray(classes)
        self._freeze()
        return self

    def _predict(self, features: pd.DataFrame) -> np.ndarray:
        self._verify_frozen()
        self.validate_input(features)
        if list(features.columns) != self._columns:
            raise FrozenICLError("Prediction columns must match the ICL context columns")
        with torch.inference_mode():
            if self.family == "kumo_medium":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    result = self._kumo_prediction(
                        self._estimator.predict(self._kumo_table(features))
                    )
            elif self.task == "classification":
                result = np.asarray(self._estimator.predict_proba(features), dtype=float)
                if self._classes is None or result.shape != (len(features), len(self._classes)):
                    raise FrozenICLError("Frozen ICL returned an invalid class probability matrix")
                positions = [self._classes.tolist().index(label) for label in (0, 1)]
                result = result[:, positions]
            else:
                result = np.asarray(self._estimator.predict(features), dtype=float).reshape(-1)
        self._verify_frozen()
        expected = (len(features), 2) if self.task == "classification" else (len(features),)
        if result.shape != expected or not np.isfinite(result).all():
            raise FrozenICLError("Frozen ICL returned invalid predictions")
        if self.task == "classification" and (
            result.min() < 0
            or result.max() > 1
            or not np.allclose(result.sum(axis=1), 1, atol=1e-5)
        ):
            raise FrozenICLError("Frozen ICL returned invalid class probabilities")
        return result

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise FrozenICLError("predict_proba is only available for classification")
        return self._predict(features)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        result = self._predict(features)
        return result.argmax(axis=1) if self.task == "classification" else result

    def _kumo_table(self, features: pd.DataFrame):
        import sdm

        return sdm.TableTensor.from_pandas(
            features,
            stypes={column: "numerical" for column in features},
            device=self._parameters["device"],
        )

    def _kumo_prediction(self, output) -> np.ndarray:
        if self.task == "regression":
            result = output.numerical.mean(dim=-1).detach().cpu().numpy()
        else:
            result = (
                torch.stack(
                    [output[str(label)].numerical.squeeze(-1) for label in (0, 1)], dim=-1
                )
                .detach()
                .cpu()
                .numpy()
            )
        return result

    def run_metadata(self) -> dict[str, Any]:
        if self._checkpoint_path is None:
            raise FrozenICLError("Call fit_context before requesting model metadata")
        digest = self._file_digest(self._checkpoint_path)
        model = {
            "tabiclv2": "TabICLv2-ICL",
            "tabpfn35": "TabPFN-3.5-ICL",
            "kumo_medium": "Kumo-Tabular-Medium-ICL",
        }[self.family]
        package = {
            "tabiclv2": "tabicl",
            "tabpfn35": "tabpfn",
            "kumo_medium": "structured-data-models",
        }[self.family]
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = None
        metadata = {
            "model_name": model + (" Regressor" if self.task == "regression" else ""),
            "package": package,
            "package_version": package_version,
            "checkpoint": self._config["checkpoint_version"],
            "checkpoint_path": str(self._checkpoint_path),
            "checkpoint_sha256": digest,
            "frozen": True,
            "parameters": dict(self._parameters),
        }
        if self.family == "tabpfn35":
            metadata["model_version"] = "v3.5"
        if self.family == "kumo_medium":
            metadata.update({
                "model_version": MODEL_VERSION,
                "package_revision": PACKAGE_REVISION,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "model_repository": MODEL_REPOSITORY,
                "size": "medium",
            })
        return metadata

    def icl_metadata(self) -> dict[str, Any]:
        if self._weight_digest is None or self._checkpoint_path is None:
            raise FrozenICLError("Call fit_context before requesting ICL metadata")
        self._verify_frozen()
        return {
            "context_rows": self._context_rows,
            "feature_count": len(self._columns),
            "n_estimators": self._parameters["n_estimators"],
            "weights_before_inference_sha256": self._weight_digest,
            "weights_frozen": True,
            "gradients_enabled": False,
            "optimizer_steps": 0,
            "epochs_completed": 0,
            "checkpoint_sha256": self._file_digest(self._checkpoint_path),
        }

    @staticmethod
    def _file_digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def _package_revision() -> str | None:
        try:
            direct_url = distribution("structured-data-models").read_text("direct_url.json")
        except PackageNotFoundError:
            return None
        return json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")


class FrozenICLClassifier(FrozenICLModel):
    def _load_and_fit_context(self, features: pd.DataFrame, target: pd.Series) -> Path:
        if self.family == "tabiclv2":
            from tabicl import TabICLClassifier

            self._estimator = TabICLClassifier(
                checkpoint_version=self._config["checkpoint_version"],
                n_estimators=self._parameters["n_estimators"],
                device=self._parameters["device"],
                random_state=self._parameters["random_state"],
                kv_cache="kv",
            )
            self._estimator.fit(features, target.astype(int))
            return Path(self._estimator.model_path_)
        if self.family == "tabpfn35":
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion

            self._estimator = TabPFNClassifier.create_default_for_version(
                ModelVersion.V3_5,
                n_estimators=self._parameters["n_estimators"],
                auto_scale_n_estimators=False,
                device=self._parameters["device"],
                random_state=self._parameters["random_state"],
                tuning_config=None,
                show_progress_bar=False,
            )
            self._estimator.fit(features, target.astype(int))
            return Path(self._estimator.model_path)
        self._fit_kumo(features, target.astype(int))
        return self._checkpoint_path

    def _fit_kumo(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._load_kumo()
        import sdm

        x = self._kumo_table(features)
        y = sdm.TableTensor.from_pandas(
            pd.DataFrame({"target": target.to_numpy(dtype=np.int64)}),
            stypes={"target": "categorical"},
            device=self._parameters["device"],
        )
        for parameter in self._estimator.parameters():
            parameter.requires_grad_(False)
        self._estimator.eval()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            self._estimator.fit(
                x,
                y,
                num_estimators=self._parameters["n_estimators"],
                generator=self._generator(),
            )

    def _generator(self) -> torch.Generator:
        return torch.Generator(device=self._parameters["device"]).manual_seed(
            self._parameters["random_state"]
        )

    def _load_kumo(self) -> None:
        import sdm
        from huggingface_hub import hf_hub_download

        if self._package_revision() != PACKAGE_REVISION:
            raise FrozenICLError("Install the pinned structured-data-models revision with uv sync --locked")
        checkpoint = Path(
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=self._config["checkpoint_version"],
                revision=CHECKPOINT_REVISION,
            )
        )
        self._estimator = sdm.models.KumoTabular(
            task=self.task, size="medium", pretrained=False, device=self._parameters["device"]
        )
        self._estimator.models[self.task].load_state_dict(
            torch.load(checkpoint, map_location=self._parameters["device"], weights_only=True),
            strict=True,
        )
        self._checkpoint_path = checkpoint

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return super().predict_proba(features)


class FrozenICLRegressor(FrozenICLClassifier):
    task = "regression"

    def _load_and_fit_context(self, features: pd.DataFrame, target: pd.Series) -> Path:
        if self.family == "tabiclv2":
            from tabicl import TabICLRegressor

            self._estimator = TabICLRegressor(
                checkpoint_version=self._config["checkpoint_version"],
                n_estimators=self._parameters["n_estimators"],
                device=self._parameters["device"],
                random_state=self._parameters["random_state"],
                kv_cache="repr",
            )
            self._estimator.fit(features, target.astype(float))
            return Path(self._estimator.model_path_)
        if self.family == "tabpfn35":
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion

            self._estimator = TabPFNRegressor.create_default_for_version(
                ModelVersion.V3_5,
                n_estimators=self._parameters["n_estimators"],
                auto_scale_n_estimators=False,
                device=self._parameters["device"],
                random_state=self._parameters["random_state"],
                tuning_config=None,
                show_progress_bar=False,
            )
            self._estimator.fit(features, target.astype(float))
            return Path(self._estimator.model_path)
        self._fit_kumo(features, target.astype(float))
        return self._checkpoint_path

    def _fit_kumo(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._load_kumo()
        import sdm

        x = self._kumo_table(features)
        y = sdm.TableTensor.from_pandas(
            pd.DataFrame({"target": target.to_numpy(dtype=float)}),
            stypes={"target": "numerical"},
            device=self._parameters["device"],
        )
        for parameter in self._estimator.parameters():
            parameter.requires_grad_(False)
        self._estimator.eval()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            self._estimator.fit(
                x,
                y,
                num_estimators=self._parameters["n_estimators"],
                generator=self._generator(),
            )


def create_icl_model(config: dict[str, Any]) -> FrozenICLModel:
    name = config.get("name", "")
    regression = name.endswith("_regressor")
    family = name.removesuffix("_regressor").removesuffix("_icl")
    if family not in {"tabiclv2", "tabpfn35", "kumo_medium"}:
        raise FrozenICLError(f"Unsupported frozen ICL model: {name}")
    model = FrozenICLRegressor if regression else FrozenICLClassifier
    instance = model({**config, "family": family})
    instance.family = family
    return instance
