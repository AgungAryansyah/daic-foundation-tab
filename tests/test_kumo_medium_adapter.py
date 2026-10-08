from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sdm
import torch

from daic_foundation_tab.config import load_config
from daic_foundation_tab.models.kumo_medium_ft import (
    CHECKPOINT_REVISION,
    MODEL_REPOSITORY,
    PACKAGE_REVISION,
    KumoMediumFineTuningError,
)
from daic_foundation_tab.models.registry import create_model


class ToyModel(torch.nn.Module):
    def __init__(self, task, columns):
        super().__init__()
        self.task = task
        self.linear = torch.nn.Linear(columns, 2 if task == "classification" else 1)
        self.cache = None
        self.calls = []

    def default_recipe(self):
        return sdm.models.KumoTabular.default_recipe()

    def forward(self, *, x_context, y_context, x_query, num_estimators, generator, **kwargs):
        self.calls.append(
            (
                self.training,
                x_context,
                y_context,
                x_query,
                num_estimators,
                generator.get_state().clone(),
            )
        )
        values = self.linear(torch.nan_to_num(x_query.numerical).float())
        if self.task == "classification":
            columns = ["1", "0"]
            views = torch.stack(
                [values + values.new_tensor([i * 0.5, 0]) for i in range(num_estimators)]
            )
        else:
            columns = [f"q{i:03d}" for i in range(1, 1000)]
            views = torch.stack(
                [values + 12 + torch.linspace(-3, 3, 999) for _ in range(num_estimators)]
            )
        if not self.training:
            self._last_trained_weight = self.linear.weight.detach().clone()
            values = views.mean(dim=0)
            if self.task == "classification":
                values = values.softmax(dim=-1)
        else:
            values = views
        return sdm.TableTensor.from_tensor(values, columns=columns)

    def clear(self):
        self.cache = None

    def fit(self, x, y, **kwargs):
        self.cache = (x, y, kwargs)

    def predict(self, x):
        context, target, parameters = self.cache
        return self(x_context=context, y_context=target, x_query=x, **parameters)


def model_config(regression=False, **parameters):
    suffix = "regression_" if regression else ""
    config = load_config(
        f"configs/experiments/kumo_medium_ft_{suffix}edaic_audio_visual_smoke.yaml"
    )["model"]
    config["parameters"].update(parameters)
    return config


def install_toy(monkeypatch, tmp_path, metrics=None):
    instances = []
    original_scaler = torch.amp.GradScaler
    monkeypatch.setattr(torch.amp, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        torch.amp, "GradScaler", lambda *args, **kwargs: original_scaler("cpu", enabled=False)
    )

    def load(model):
        model._device = torch.device("cpu")
        toy = ToyModel(model.task, len(model._columns))
        source = tmp_path / f"source_{len(instances)}.pt"
        torch.save(toy.state_dict(), source)
        instances.append((model, toy, source))
        return toy, source

    monkeypatch.setattr(
        "daic_foundation_tab.models.kumo_medium_ft.KumoMediumFineTunedModel._load_model", load
    )
    if metrics is not None:

        def evaluate(model, *args):
            model._model.eval()
            model._model.clear()
            index = getattr(model, "_test_metric_index", 0)
            model._test_metric_index = index + 1
            return metrics[index]

        monkeypatch.setattr(
            "daic_foundation_tab.models.kumo_medium_ft.KumoMediumFineTunedModel._evaluate", evaluate
        )
    return instances


def fit_model(model, directory, callback=None):
    features = pd.DataFrame(np.random.default_rng(42).normal(size=(20, 3)), columns=["a", "b", "c"])
    target = pd.Series([2.0, 12.0] * 10 if model.task == "regression" else [0, 1] * 10)
    model.fit(
        features.iloc[:16],
        target.iloc[:16],
        validation_features=features.iloc[16:],
        validation_target=target.iloc[16:],
        checkpoint_directory=directory,
        epoch_callback=callback,
    )
    return features, target


@pytest.mark.parametrize("regression", [False, True])
def test_constructor_loads_explicit_medium_task_and_pinned_source(
    monkeypatch, tmp_path, regression
):
    import huggingface_hub

    model = create_model(model_config(regression))
    model._device = torch.device("cpu")
    child = torch.nn.Linear(3, 2)
    source = tmp_path / "source.pt"
    torch.save(child.state_dict(), source)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    class Native(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls.append(kwargs)
            self.models = torch.nn.ModuleDict({model.task: torch.nn.Linear(3, 2)})

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(sdm.models, "KumoTabular", Native)
    native, loaded = model._load_model()
    assert calls == [
        {
            "repo_id": MODEL_REPOSITORY,
            "filename": f"medium/{'regressor' if regression else 'classifier'}.pt",
            "revision": CHECKPOINT_REVISION,
        },
        {"task": model.task, "size": "medium", "pretrained": False, "device": torch.device("cpu")},
    ]
    assert loaded == source
    assert torch.equal(native.models[model.task].weight, child.weight)
    assert model.run_metadata()["package_revision"] == PACKAGE_REVISION


@pytest.mark.parametrize("regression", [False, True])
def test_training_updates_weights_and_uses_only_training_context(monkeypatch, tmp_path, regression):
    instances = install_toy(monkeypatch, tmp_path)
    model = create_model(model_config(regression))
    history = []
    features, _ = fit_model(model, tmp_path / "checkpoints", history.append)
    _, native, source = instances[0]
    training = [call for call in native.calls if call[0]]
    assert len(training) == 1 and training[0][4] == 2
    assert training[0][1].size(0) == 12 and training[0][3].size(0) == 4
    context_rows = training[0][1].numerical.numpy()
    query_rows = training[0][3].numerical.numpy()
    assert not any(np.array_equal(a, b) for a in context_rows for b in query_rows)
    assert len(native.cache[0]) == 16 and native.cache[2]["num_estimators"] == 8
    np.testing.assert_allclose(
        native.cache[0].numerical.numpy(), features.iloc[:16].to_numpy(), rtol=1e-6
    )
    assert history[0]["train/epoch"] == 0
    assert np.isfinite(history[0]["train/mean_loss"])
    assert f"val/{'mse' if regression else 'roc_auc'}" in history[0]
    assert not torch.equal(
        torch.load(source, weights_only=True)["linear.weight"], native._last_trained_weight
    )
    metadata = model.finetune_metadata()
    assert metadata["optimizer_steps"] == metadata["epochs_completed"] == 1
    checkpoint = Path(metadata["checkpoint_path"])
    assert checkpoint.name == "best.pt"
    assert metadata["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["source_checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    prediction = model.predict(features.iloc[16:])
    assert prediction.shape == (4,) and np.isfinite(prediction).all()
    if not regression:
        probabilities = model.predict_proba(features.iloc[16:])
        assert probabilities.shape == (4, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    else:
        assert prediction.mean() > 8
    baseline, validation = [call for call in native.calls if not call[0]][:2]
    assert torch.equal(baseline[5], validation[5])


@pytest.mark.parametrize(
    "regression,metrics,selected,completed",
    [
        (False, [0.75, 0.7, 0.75005, 0.6], 0, 2),
        (False, [0.75, 0.8, 0.7, 0.6], 1, 3),
        (True, [3.0, 3.5, 4.0, 5.0], 0, 2),
        (True, [3.0, 2.0, 2.5, 3.0], 1, 3),
    ],
)
def test_baseline_improvement_and_early_stopping_restore_selected_state(
    monkeypatch, tmp_path, regression, metrics, selected, completed
):
    install_toy(monkeypatch, tmp_path)
    model = create_model(model_config(regression, epochs=5, patience=2))
    states = []

    def evaluate(*args):
        states.append(
            {key: value.detach().clone() for key, value in model._model.state_dict().items()}
        )
        return metrics[len(states) - 1]

    monkeypatch.setattr(model, "_evaluate", evaluate)
    history = []
    fit_model(model, tmp_path / "checkpoints", history.append)
    metadata = model.finetune_metadata()
    assert metadata["selected_epoch"] == selected
    assert metadata["epochs_completed"] == len(history) == completed
    assert metadata["best_validation_metric"] == metrics[selected]
    assert any(not torch.equal(states[0][key], states[1][key]) for key in states[0])
    for key, value in model._model.state_dict().items():
        assert torch.equal(value, states[selected][key])


@pytest.mark.parametrize("regression", [False, True])
def test_loss_averages_every_estimator_and_regression_uses_raw_units(
    monkeypatch, tmp_path, regression
):
    install_toy(monkeypatch, tmp_path)
    model = create_model(model_config(regression))
    features, target = fit_model(model, tmp_path / "checkpoints")
    x, y = model._tables(features.iloc[:16], target.iloc[:16])
    context, query = torch.arange(12), torch.arange(12, 16)
    recipe = model._training_recipe()
    model._model.train()
    out = model._model(
        x_context=x[context],
        y_context=y[context],
        x_query=x[query],
        recipe=recipe,
        num_estimators=2,
        generator=model._generator(),
    )
    actual = model._train_loss(x, y, context, query, recipe, model._generator())
    if regression:
        levels = torch.linspace(0.001, 0.999, 999)
        diff = y[query].numerical - out.numerical
        expected = torch.maximum(levels * diff, (levels - 1) * diff).mean()
    else:
        losses = []
        for i in range(2):
            logits, truth = sdm.evaluation.to_class_indices(out[i], y[query])
            losses.append(torch.nn.functional.cross_entropy(logits, truth))
        expected = torch.stack(losses).mean()
        assert not torch.allclose(actual, losses[0])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("metric", [float("nan"), float("inf")])
@pytest.mark.parametrize("epoch", [0, 1])
def test_invalid_validation_metric_fails_instead_of_completing(
    monkeypatch, tmp_path, metric, epoch
):
    install_toy(monkeypatch, tmp_path)
    model = create_model(model_config())
    import daic_foundation_tab.models.kumo_medium_ft as adapter

    metrics = iter(([0.8] if epoch else []) + [metric])
    monkeypatch.setattr(adapter, "roc_auc_score", lambda *args: next(metrics))
    with pytest.raises(KumoMediumFineTuningError, match="validation metric must be finite"):
        fit_model(model, tmp_path / "checkpoints")
    with pytest.raises(KumoMediumFineTuningError, match="Call fit"):
        model.finetune_metadata()


def test_episode_reproducibility_and_reserved_binary_context():
    model = create_model(model_config(max_data_size=10))
    target = pd.Series([0] * 8 + [1] * 2)
    context, query = next(model._episodes(target, np.random.default_rng(42)))
    again = next(model._episodes(target, np.random.default_rng(42)))
    assert set(target.iloc[context]) == {0, 1}
    assert not set(context) & set(query)
    assert len(query) == 2 and len(context) == 8
    assert set(context) | set(query) == set(range(10))
    for a, b in zip((context, query), again, strict=True):
        np.testing.assert_array_equal(a, b)


def test_validation_and_prediction_guards(monkeypatch, tmp_path):
    install_toy(monkeypatch, tmp_path)
    model = create_model(model_config())
    features = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(KumoMediumFineTuningError, match="Call fit"):
        model.predict(features)
    with pytest.raises(KumoMediumFineTuningError, match="infinity"):
        model.validate_input(pd.DataFrame({"a": [float("inf")]}))
    with pytest.raises(KumoMediumFineTuningError, match="numerical"):
        model.validate_input(pd.DataFrame({"a": ["private"]}))
    features, _ = fit_model(model, tmp_path / "checkpoints")
    with pytest.raises(KumoMediumFineTuningError, match="columns must match"):
        model.predict(features.rename(columns={"a": "different"}))
    with pytest.raises(KumoMediumFineTuningError, match="fresh checkpoint"):
        fit_model(model, tmp_path / "checkpoints")


def test_probabilities_are_aligned_to_binary_class_values():
    model = create_model(model_config())
    table = sdm.TableTensor.from_tensor(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), columns=["1", "0"])
    np.testing.assert_allclose(model._prediction(table), [[0.1, 0.9], [0.8, 0.2]])


def test_nonfinite_training_loss_aborts_without_completed_metadata(monkeypatch, tmp_path):
    install_toy(monkeypatch, tmp_path)
    model = create_model(model_config())
    monkeypatch.setattr(model, "_train_loss", lambda *args: torch.tensor(float("nan")))
    with pytest.raises(KumoMediumFineTuningError, match="training loss must be finite"):
        fit_model(model, tmp_path / "checkpoints")
    with pytest.raises(KumoMediumFineTuningError, match="Call fit"):
        model.finetune_metadata()
