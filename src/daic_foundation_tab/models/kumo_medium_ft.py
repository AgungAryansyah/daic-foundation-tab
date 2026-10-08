from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score

PACKAGE_REVISION = "842c408fe2a8711bdf2e7cff4bfbe54266d6b940"
CHECKPOINT_REVISION = "3c3e10bbdb590ace29e7026847f92db3c603096d"
MODEL_REPOSITORY = "nvidia/Kumo-Tabular"
MODEL_VERSION = "v1.0.1"
_PARAMETERS = {
    "epochs",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "amp",
    "warmup_proportion",
    "n_estimators_finetune",
    "n_estimators_validation",
    "n_estimators_inference",
    "max_data_size",
    "finetune_ctx_query_ratio",
    "patience",
    "min_delta",
    "eval_metric",
    "device",
    "random_state",
}


class KumoMediumFineTuningError(RuntimeError):
    pass


def validate_kumo_config(config: Mapping[str, Any]) -> None:
    regression = config.get("name") == "kumo_medium_ft_regressor"
    checkpoint = f"medium/{'regressor' if regression else 'classifier'}.pt"
    for key, expected in (
        ("model_version", MODEL_VERSION),
        ("size", "medium"),
        ("package_revision", PACKAGE_REVISION),
        ("checkpoint_revision", CHECKPOINT_REVISION),
        ("checkpoint_version", checkpoint),
    ):
        if config.get(key) != expected:
            raise ValueError(f"Kumo Medium requires model.{key}={expected!r}")
    parameters = config["parameters"]
    unknown = set(parameters) - _PARAMETERS
    if unknown or _PARAMETERS - set(parameters):
        raise ValueError(
            "Kumo Medium parameters must match its model preset; unsupported or missing arguments"
        )
    if parameters["device"] != "cuda:0" or parameters["amp"] is not True:
        raise ValueError("Kumo Medium requires cuda:0 and amp=True")
    if parameters["eval_metric"] != ("mse" if regression else "roc_auc"):
        raise ValueError("Kumo Medium requires MSE regression or ROC-AUC classification selection")
    for key in (
        "epochs",
        "patience",
        "n_estimators_finetune",
        "n_estimators_validation",
        "n_estimators_inference",
    ):
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Kumo Medium {key} must be a positive integer")
    maximum = parameters["max_data_size"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 3:
        raise ValueError("Kumo Medium max_data_size must be an integer of at least three")
    seed = parameters["random_state"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Kumo Medium random_state must be a nonnegative integer")
    for key in (
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "min_delta",
        "warmup_proportion",
        "finetune_ctx_query_ratio",
    ):
        value = parameters[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Kumo Medium {key} must be finite")
        if value < 0 or (key in {"learning_rate", "grad_clip"} and value == 0):
            raise ValueError(f"Kumo Medium {key} is out of range")
    if not 0 < parameters["finetune_ctx_query_ratio"] < 1:
        raise ValueError("Kumo Medium query fraction must be between zero and one")
    if not 0 <= parameters["warmup_proportion"] < 1:
        raise ValueError("Kumo Medium warmup proportion must be in [0, 1)")


class KumoMediumFineTunedModel:
    task = "classification"

    def __init__(self, model_config: dict[str, Any]) -> None:
        validate_kumo_config(model_config)
        self._config = model_config
        self._parameters = dict(model_config["parameters"])
        self._device = torch.device(self._parameters["device"])
        self._model: Any = None
        self._metadata: dict[str, Any] | None = None

    def validate_input(self, features: pd.DataFrame) -> None:
        if features.empty or not features.columns.is_unique:
            raise KumoMediumFineTuningError("Kumo requires nonempty features with unique columns")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
            raise KumoMediumFineTuningError("Kumo participant features must be numerical")
        if np.isinf(features.to_numpy(dtype=float)).any():
            raise KumoMediumFineTuningError("Kumo features must not contain infinity")

    def _generator(self) -> torch.Generator:
        return torch.Generator(device=self._device).manual_seed(self._parameters["random_state"])

    def _load_model(self) -> tuple[Any, Path]:
        import sdm
        from huggingface_hub import hf_hub_download

        if self._package_revision() != PACKAGE_REVISION:
            raise KumoMediumFineTuningError(
                "Install the pinned structured-data-models revision with uv sync --locked"
            )

        source = Path(
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=self._config["checkpoint_version"],
                revision=CHECKPOINT_REVISION,
            )
        )
        model = sdm.models.KumoTabular(
            task=self.task, size="medium", pretrained=False, device=self._device
        )
        model.models[self.task].load_state_dict(
            torch.load(source, map_location=self._device, weights_only=True),
            strict=True,
        )
        return model, source

    def _tables(self, features: pd.DataFrame, target: pd.Series | None = None):
        import sdm

        x = sdm.TableTensor.from_pandas(
            features,
            stypes={column: "numerical" for column in features},
            device=self._device,
        )
        if target is None:
            return x
        y = sdm.TableTensor.from_pandas(
            pd.DataFrame(
                {
                    "target": target.to_numpy(
                        dtype=np.int64 if self.task == "classification" else float
                    )
                }
            ),
            stypes={"target": "categorical" if self.task == "classification" else "numerical"},
            device=self._device,
        )
        return x, y

    def _episodes(self, target: pd.Series, random: np.random.Generator):
        chunks = np.array_split(
            random.permutation(len(target)),
            math.ceil(len(target) / self._parameters["max_data_size"]),
        )
        for chunk in chunks:
            if len(chunk) < 3:
                raise KumoMediumFineTuningError("An episode needs at least three participants")
            query_count = min(
                len(chunk) - 2,
                max(1, math.ceil(len(chunk) * self._parameters["finetune_ctx_query_ratio"])),
            )
            reserved = []
            if self.task == "classification":
                for label in (0, 1):
                    candidates = chunk[target.iloc[chunk].to_numpy() == label]
                    if not len(candidates):
                        raise KumoMediumFineTuningError(
                            "Every classification chunk must contain both classes"
                        )
                    reserved.append(random.choice(candidates))
            remaining = chunk[~np.isin(chunk, reserved)]
            query = remaining[:query_count]
            context = np.r_[reserved, remaining[query_count:]].astype(int)
            yield context, query

    def _training_recipe(self):
        import sdm.processing as sp

        recipe = self._model.default_recipe()
        recipe.target = sp.StypeDispatch(
            categorical=[sp.AlignCategories(), sp.ShuffleCategories(method="shift")],
            numerical=sp.Standardize(),
        )
        recipe.output = sp.Identity()
        return recipe

    def _train_loss(self, x, y, context, query, recipe, generator):
        import sdm

        out = self._model(
            x_context=x[context],
            y_context=y[context],
            x_query=x[query],
            recipe=recipe,
            num_estimators=self._parameters["n_estimators_finetune"],
            estimator_batch_size=1,
            generator=generator,
        )
        losses = []
        for estimator in range(self._parameters["n_estimators_finetune"]):
            prediction = out[estimator]
            if self.task == "classification":
                logits, truth = sdm.evaluation.to_class_indices(
                    prediction, y[query], missing_score=-torch.inf
                )
                losses.append(torch.nn.functional.cross_entropy(logits.float(), truth))
            else:
                difference = y[query].numerical - prediction.numerical
                levels = torch.linspace(0.001, 0.999, 999, device=self._device)
                losses.append(torch.maximum(levels * difference, (levels - 1) * difference).mean())
        return torch.stack(losses).mean()

    def _prediction(self, out) -> np.ndarray:
        if self.task == "regression":
            result = out.numerical.mean(dim=-1).detach().cpu().numpy()
        else:
            result = (
                torch.stack([out[str(label)].numerical.squeeze(-1) for label in (0, 1)], dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
        expected = (out.size(0),) if self.task == "regression" else (out.size(0), 2)
        if result.shape != expected or not np.isfinite(result).all():
            raise KumoMediumFineTuningError("Kumo returned invalid predictions")
        if self.task == "classification" and (
            np.any(result < 0)
            or np.any(result > 1)
            or not np.allclose(result.sum(axis=1), 1, atol=1e-5)
        ):
            raise KumoMediumFineTuningError("Kumo returned invalid probabilities")
        return result

    def _evaluate(self, x, y, validation_x, target: pd.Series) -> float:
        self._model.eval()
        self._model.clear()
        with torch.amp.autocast(self._device.type, dtype=torch.float16):
            out = self._model(
                x_context=x,
                y_context=y,
                x_query=validation_x,
                num_estimators=self._parameters["n_estimators_validation"],
                estimator_batch_size=1,
                generator=self._generator(),
            )
        prediction = self._prediction(out)
        metric = (
            mean_squared_error(target, prediction)
            if self.task == "regression"
            else roc_auc_score(target, prediction[:, 1])
        )
        if not math.isfinite(metric):
            raise KumoMediumFineTuningError("Kumo validation metric must be finite")
        return float(metric)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        checkpoint_directory: Path,
        epoch_callback: Callable[[Mapping[str, float]], None] | None = None,
    ):
        self._metadata = None
        self.validate_input(features)
        self.validate_input(validation_features)
        if list(features.columns) != list(validation_features.columns):
            raise KumoMediumFineTuningError("Validation columns must match training columns")
        for x, y in ((features, target), (validation_features, validation_target)):
            if len(x) != len(y) or not np.isfinite(y.to_numpy(dtype=float)).all():
                raise KumoMediumFineTuningError("Targets must be finite and match feature rows")
            if self.task == "classification" and set(y.unique()) != {0, 1}:
                raise KumoMediumFineTuningError(
                    "Both partitions must contain binary classes 0 and 1"
                )
        if checkpoint_directory.exists() and any(checkpoint_directory.iterdir()):
            raise KumoMediumFineTuningError("Use a fresh checkpoint directory")
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self._parameters["random_state"])
        self._columns = list(features.columns)
        self._model, source = self._load_model()
        for parameter in self._model.parameters():
            parameter.requires_grad_(True)
        x, y = self._tables(features, target)
        validation_x = self._tables(validation_features)
        baseline = best = self._evaluate(x, y, validation_x, validation_target)
        checkpoint = checkpoint_directory / "best.pt"
        torch.save(self._model.state_dict(), checkpoint)
        selected_epoch = unsuccessful = steps = completed = 0
        parameters = self._parameters
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=parameters["learning_rate"],
            weight_decay=parameters["weight_decay"],
        )
        total_steps = parameters["epochs"] * math.ceil(len(target) / parameters["max_data_size"])
        warmup = int(total_steps * parameters["warmup_proportion"])

        def learning_rate_factor(step):
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (
                1
                + math.cos(
                    math.pi * min(step - warmup, total_steps - warmup) / (total_steps - warmup)
                )
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
        scaler = torch.amp.GradScaler(self._device.type)
        random = np.random.default_rng(parameters["random_state"])
        generator = self._generator()
        recipe = self._training_recipe()
        for epoch in range(parameters["epochs"]):
            self._model.train()
            self._model.clear()
            losses = []
            for context, query in self._episodes(target, random):
                context = torch.as_tensor(context, device=self._device)
                query = torch.as_tensor(query, device=self._device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(self._device.type, dtype=torch.float16):
                    loss = self._train_loss(x, y, context, query, recipe, generator)
                if not torch.isfinite(loss):
                    raise KumoMediumFineTuningError("Kumo training loss must be finite")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), parameters["grad_clip"], error_if_nonfinite=True
                )
                rate = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                steps += 1
                losses.append(float(loss.detach()))
            metric = self._evaluate(x, y, validation_x, validation_target)
            completed = epoch + 1
            improved = (
                metric < best - parameters["min_delta"]
                if self.task == "regression"
                else metric > best + parameters["min_delta"]
            )
            if improved:
                best, selected_epoch, unsuccessful = metric, completed, 0
                torch.save(self._model.state_dict(), checkpoint)
            else:
                unsuccessful += 1
            if epoch_callback is not None:
                epoch_callback(
                    {
                        "train/epoch": epoch,
                        "train/mean_loss": float(np.mean(losses)),
                        "train/lr": rate,
                        f"val/{parameters['eval_metric']}": metric,
                    }
                )
            if unsuccessful >= parameters["patience"]:
                break
        self._model.load_state_dict(
            torch.load(checkpoint, map_location=self._device, weights_only=True), strict=True
        )
        self._model.eval()
        self._model.clear()
        with torch.amp.autocast(self._device.type, dtype=torch.float16):
            self._model.fit(
                x,
                y,
                num_estimators=parameters["n_estimators_inference"],
                estimator_batch_size=1,
                generator=self._generator(),
            )
        self._metadata = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": self._sha256(checkpoint),
            "selection_metric": parameters["eval_metric"],
            "baseline_validation_metric": baseline,
            "best_validation_metric": best,
            "selected_epoch": selected_epoch,
            "epochs_completed": completed,
            "optimizer_steps": steps,
            "source_checkpoint_sha256": self._sha256(source),
        }
        return self

    @staticmethod
    def _sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def _package_revision() -> str | None:
        try:
            direct_url = distribution("structured-data-models").read_text("direct_url.json")
        except PackageNotFoundError:
            return None
        return json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")

    def _predict(self, features: pd.DataFrame):
        if self._metadata is None:
            raise KumoMediumFineTuningError("Call fit before predicting")
        self.validate_input(features)
        if list(features.columns) != self._columns:
            raise KumoMediumFineTuningError("Prediction columns must match training columns")
        with torch.amp.autocast(self._device.type, dtype=torch.float16):
            return self._prediction(self._model.predict(self._tables(features)))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self._predict(features)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)

    def finetune_metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            raise KumoMediumFineTuningError("Call fit before requesting fine-tuning metadata")
        return dict(self._metadata)

    def run_metadata(self) -> dict[str, Any]:
        try:
            package_version = version("structured-data-models")
        except PackageNotFoundError:
            package_version = None
        return {
            "model_name": f"Kumo-Tabular-Medium-FT{'-Regressor' if self.task == 'regression' else ''}",
            "package": "structured-data-models",
            "package_version": package_version,
            "package_revision": self._package_revision(),
            "model_version": MODEL_VERSION,
            "checkpoint": self._config["checkpoint_version"],
            "checkpoint_revision": CHECKPOINT_REVISION,
            "model_repository": MODEL_REPOSITORY,
            "size": "medium",
            "parameters": dict(self._parameters),
        }


class KumoMediumFineTunedRegressor(KumoMediumFineTunedModel):
    task = "regression"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self._predict(features)
