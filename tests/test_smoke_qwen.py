import os

import pytest

from daic_foundation_tab.models.qwen import QwenTextModel
from daic_foundation_tab.models.text_base import TextExample
from tests.test_text_config import text_config
from tests.test_text_prompt import TinyTokenizer


@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("DAIC_QWEN_SMOKE") != "1",
    reason="Set DAIC_QWEN_SMOKE=1 for synthetic CUDA checks",
)
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mode", ["frozen_icl", "fine_tuning"])
def test_synthetic_qwen_cuda(tmp_path, task, mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("A BF16 CUDA device is unavailable")
    torch.manual_seed(42)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
        eos_token_id=2,
        pad_token_id=2,
    )
    model = transformers.Qwen3_5ForCausalLM(config).to(device="cuda:0", dtype=torch.bfloat16)
    tokenizer = TinyTokenizer()
    settings = text_config()
    settings["experiment"].update(task=task, mode=mode)
    settings["project"]["cache_root"] = str(tmp_path / "cache")
    settings["evaluation"]["threshold"] = 0.5 if task == "classification" else None
    settings["model"]["parameters"].update(checkpoint_steps=1)
    adapter = QwenTextModel(
        settings, model=model, tokenizer=tokenizer, revision="synthetic-hybrid-v1"
    )
    if task == "regression":
        original_generate = model.generate

        def numeric_generate(input_ids, **kwargs):
            prompt_length = input_ids.shape[1]
            return original_generate(
                input_ids=input_ids,
                **kwargs,
                prefix_allowed_tokens_fn=lambda batch, tokens: (
                    list(range(51, 61)) if len(tokens) == prompt_length else [2]
                ),
            )

        model.generate = numeric_generate
        tokenizer.decode = lambda tokens, **kwargs: "".join(
            chr(int(token) - 3) for token in tokens if int(token) != 2
        )
    original = [(parameter, parameter.detach().clone()) for parameter in model.parameters()]
    training = [
        TextExample("t0", "Tired", 0 if task == "classification" else 4),
        TextExample("t1", "Sad", 1 if task == "classification" else 12),
    ]
    validation = [
        TextExample("v0", "Well", 0 if task == "classification" else 3),
        TextExample("v1", "Sleepy", 1 if task == "classification" else 13),
    ]
    adapter.prepare(training[:1] if mode == "frozen_icl" else [])
    adapter.preflight([*training, *validation])
    if mode == "fine_tuning":
        adapter.fit(training, validation, tmp_path / "checkpoints")
        assert adapter.optimizer_steps == 2
        assert (tmp_path / "checkpoints/best_adapter/adapter_config.json").is_file()
    else:
        assert adapter.optimizer_steps == 0
    result = adapter.predict([TextExample("query", "Sleep is difficult")])[0]
    assert 0 <= result.value <= (1 if task == "classification" else 24)
    assert all(torch.equal(parameter, old) for parameter, old in original)
    assert adapter.run_metadata()["base_parameters_frozen"]
