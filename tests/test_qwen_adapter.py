import json

import pytest

from daic_foundation_tab.models.qwen import QwenTextModel
from daic_foundation_tab.models.text_base import TextExample
from tests.test_text_config import text_config
from tests.test_text_prompt import TinyTokenizer


def tiny_adapter(tmp_path, *, mode="frozen_icl", task="classification", device="cpu", family="qwen35_2b"):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(7)
    model_config = transformers.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
        eos_token_id=2,
        pad_token_id=2,
    )
    model = transformers.Qwen3_5ForCausalLM(model_config).to(device)
    config = text_config()
    model_id = "Qwen/Qwen3.5-0.8B" if family == "qwen35_0_8b" else "Qwen/Qwen3.5-2B"
    config["model"].update(name=family, model_id=model_id)
    config["project"]["cache_root"] = str(tmp_path / "cache")
    config["experiment"].update(task=task, mode=mode)
    config["evaluation"]["threshold"] = 0.5 if task == "classification" else None
    config["model"]["parameters"].update(checkpoint_steps=1, learning_rate=0.01)
    return QwenTextModel(config, model=model, tokenizer=TinyTokenizer(), revision="tiny-fixture-v1")


def examples():
    return [TextExample("train0", "I feel tired", 0), TextExample("train1", "I feel well", 1)]


@pytest.mark.parametrize("family", ["qwen35_2b", "qwen35_0_8b"])
def test_frozen_scoring_preserves_weights_and_blocks_demonstration_leakage(tmp_path, family):
    import torch

    adapter = tiny_adapter(tmp_path, family=family)
    before = [parameter.detach().clone() for parameter in adapter.model.parameters()]
    adapter.prepare(examples()[:1])
    prediction = adapter.predict([TextExample("dev", "I feel sad")])[0]
    assert 0 <= prediction.probability_positive <= 1
    assert adapter.run_metadata()["model_name"] == family
    assert adapter.run_metadata()["optimizer_steps"] == 0
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())
    assert all(
        torch.equal(old, new) for old, new in zip(before, adapter.model.parameters(), strict=True)
    )
    with pytest.raises(ValueError, match="demonstrations"):
        adapter.predict(examples()[:1])
    with pytest.raises(ValueError, match="Frozen"):
        adapter.fit(examples(), examples(), tmp_path / "checkpoints")


def test_transcript_campaign_source_changes_invalidate_token_cache(tmp_path):
    for identity in ("source-one", "source-two"):
        adapter = tiny_adapter(tmp_path)
        adapter.config["study"] = {"transcript_study_id": "study", "software_identity": identity}
        adapter.prepare([])
        adapter.predict([TextExample("query", "The same interview")])
    assert len(list((tmp_path / "cache/text_tokens").glob("*.json"))) == 2


@pytest.mark.parametrize("family", ["qwen35_2b", "qwen35_0_8b"])
def test_lora_updates_only_adapters_and_reload_reproduces_predictions(tmp_path, family):
    import torch

    adapter = tiny_adapter(tmp_path, mode="fine_tuning", family=family)
    original = [(parameter, parameter.detach().clone()) for parameter in adapter.model.parameters()]
    validation = [TextExample("dev0", "Tired", 0), TextExample("dev1", "Well", 1)]
    history = []
    adapter.fit(examples(), validation, tmp_path / "checkpoints", epoch_callback=history.append)
    assert adapter.optimizer_steps == 2
    assert history[0]["epoch"] == 1
    assert history[0]["optimizer_steps"] == 2
    assert history[0]["train/mean_loss"] > 0
    assert history[0]["train/examples_observed"] == 2
    assert 0 <= history[0]["val/roc_auc"] <= 1
    assert all(torch.equal(parameter, old) for parameter, old in original)
    assert any(
        "lora_B" in name and torch.count_nonzero(parameter) > 0
        for name, parameter in adapter.model.named_parameters()
    )
    before = adapter.predict(validation)
    loaded = tiny_adapter(tmp_path / "reloaded", mode="fine_tuning", family=family)
    loaded.load_adapter(tmp_path / "checkpoints/best_adapter")
    assert [item.probability_positive for item in loaded.predict(validation)] == pytest.approx(
        [item.probability_positive for item in before]
    )


def test_checkpointed_lora_training_keeps_base_weights_frozen(tmp_path):
    import torch

    adapter = tiny_adapter(tmp_path, mode="fine_tuning")
    adapter.parameters["gradient_checkpointing"] = True
    original = [(parameter, parameter.detach().clone()) for parameter in adapter.model.parameters()]
    adapter.fit(examples(), [TextExample("dev0", "Tired", 0), TextExample("dev1", "Well", 1)],
                tmp_path / "checkpoints")
    assert adapter.run_metadata()["gradient_checkpointing"]
    assert adapter.optimizer_steps == 2
    assert all(torch.equal(parameter, old) for parameter, old in original)


def test_resume_checks_identity_and_continues_optimizer_steps(tmp_path):
    adapter = tiny_adapter(tmp_path, mode="fine_tuning")
    validation = [TextExample("dev0", "Tired", 0), TextExample("dev1", "Well", 1)]

    def interrupt(completed, total):
        if completed == 1:
            raise RuntimeError("interrupted")

    directory = tmp_path / "checkpoints"
    with pytest.raises(RuntimeError, match="interrupted"):
        adapter.fit(examples(), validation, directory, progress=interrupt)
    assert json.loads((directory / "latest.json").read_text())["checkpoint"]
    incompatible = tiny_adapter(tmp_path, mode="fine_tuning")
    incompatible.parameters["learning_rate"] = 0.02
    with pytest.raises(ValueError, match="identity"):
        incompatible.fit(examples(), validation, directory, resume=True)
    resumed = tiny_adapter(tmp_path, mode="fine_tuning")
    resumed.fit(examples(), validation, directory, resume=True)
    assert resumed.optimizer_steps == 2


def test_regression_uses_numeric_generation_and_rejects_invalid_output(tmp_path, monkeypatch):
    import torch

    adapter = tiny_adapter(tmp_path, task="regression")
    adapter.tokenizer.decode = lambda *args, **kwargs: "12.5"
    monkeypatch.setattr(
        adapter.model,
        "generate",
        lambda input_ids, **kwargs: torch.cat(
            [input_ids, torch.tensor([[3]], device=input_ids.device)], dim=1
        ),
    )
    adapter.prepare([])
    assert adapter.predict([TextExample("dev", "Tired")])[0].value == 12.5
    adapter.tokenizer.decode = lambda *args, **kwargs: "PHQ score: 12"
    with pytest.raises(ValueError, match="numeric answer"):
        adapter.predict([TextExample("dev", "Tired")])


def test_classification_scores_the_whole_answer_including_eos(tmp_path):
    from types import SimpleNamespace

    import torch

    class UniformModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=4096)

        def forward(self, input_ids, logits_to_keep, **kwargs):
            return SimpleNamespace(logits=torch.zeros(1, logits_to_keep, 10))

    tokenizer = TinyTokenizer()
    tokenizer.encode = lambda answer, **kwargs: [3, 4] if answer == "0" else [5]
    config = text_config()
    config["project"]["cache_root"] = str(tmp_path)
    adapter = QwenTextModel(config, model=UniformModel(), tokenizer=tokenizer, revision="uniform")
    adapter.prepare([])
    prediction = adapter.predict([TextExample("query", "text")])[0]
    assert prediction.probability_positive == pytest.approx(10 / 11)


def test_resume_preserves_accumulation_with_a_partial_final_group(tmp_path):
    training = [*examples(), TextExample("train2", "Sad", 1)]
    validation = [TextExample("dev0", "Tired", 0), TextExample("dev1", "Well", 1)]
    complete = tiny_adapter(tmp_path / "complete", mode="fine_tuning")
    complete.parameters["gradient_accumulation_steps"] = 2
    complete.fit(training, validation, tmp_path / "complete/checkpoints")
    interrupted = tiny_adapter(tmp_path / "interrupted", mode="fine_tuning")
    interrupted.parameters["gradient_accumulation_steps"] = 2

    def stop(completed, total):
        if completed == 1:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted.fit(training, validation, tmp_path / "interrupted/checkpoints", stop)
    resumed = tiny_adapter(tmp_path / "interrupted", mode="fine_tuning")
    resumed.parameters["gradient_accumulation_steps"] = 2
    resumed.fit(training, validation, tmp_path / "interrupted/checkpoints", resume=True)
    assert resumed.optimizer_steps == complete.optimizer_steps == 2
    assert [item.probability_positive for item in resumed.predict(validation)] == pytest.approx(
        [item.probability_positive for item in complete.predict(validation)]
    )


@pytest.mark.parametrize("task,target", [("classification", 1), ("regression", 12.5)])
def test_answer_suffix_loss_matches_full_loss_and_gradients_without_shortening_input(tmp_path, task, target):
    import torch

    adapter = tiny_adapter(tmp_path, mode="fine_tuning", task=task)
    adapter.prepare([])
    adapter.model.requires_grad_(True)
    record = adapter.encoder.training_record(TextExample("fit", "I feel tired " * 20, target))
    batch = {key: adapter._tensor(value) for key, value in record.items()}
    full_loss = adapter.model(**batch, use_cache=False).loss
    full_loss.backward()
    original = {name: parameter.grad.clone() for name, parameter in adapter.model.named_parameters()
                if parameter.grad is not None}
    adapter.model.zero_grad(set_to_none=True)
    projected, input_lengths = [], []
    projection_hook = adapter.model.lm_head.register_forward_pre_hook(
        lambda module, args: projected.append(args[0].shape[1]))
    input_hook = adapter.model.model.register_forward_pre_hook(
        lambda module, args, kwargs: input_lengths.append(kwargs["input_ids"].shape[1]),
        with_kwargs=True)
    try:
        reduced_loss = adapter._training_loss(record)
        reduced_loss.backward()
    finally:
        projection_hook.remove()
        input_hook.remove()
    torch.testing.assert_close(reduced_loss, full_loss)
    for name, parameter in adapter.model.named_parameters():
        if name in original:
            torch.testing.assert_close(parameter.grad, original[name], atol=1e-6, rtol=1e-4)
    assert input_lengths == [len(record["input_ids"])]
    assert projected == [sum(label != -100 for label in record["labels"]) + 1]
    assert projected[0] < input_lengths[0]
