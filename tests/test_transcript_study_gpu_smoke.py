import json
import os
from pathlib import Path

import pytest

from daic_foundation_tab import text_runner
from daic_foundation_tab.models.qwen import QwenTextModel
from daic_foundation_tab.transcript_study.bundle import read_bundle
from daic_foundation_tab.transcript_study.condition import run_condition
from daic_foundation_tab.transcript_study.results import write_result
from tests.test_text_prompt import TinyTokenizer
from tests.test_transcript_study_audit import prepared_input


@pytest.mark.real_model
@pytest.mark.skipif(os.environ.get("DAIC_QWEN_CV_SMOKE") != "1", reason="Transcript CUDA smoke tests are opt-in")
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize(("mode", "adapter_method"), [("frozen_icl", None), ("fine_tuning", None), ("fine_tuning", "dora")])
def test_native_hybrid_qwen_fold(tmp_path, monkeypatch, task, mode, adapter_method):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA with BF16 is required")
    torch.set_num_threads(1)
    bundle = prepared_input(tmp_path, adapter_method=adapter_method)
    context = read_bundle(bundle)
    condition = next(c for c in context[0]["conditions"]["cv"] if c["task"] == task and c["mode"] == mode)
    originals = []

    def create(config):
        native_config = transformers.Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
            max_position_embeddings=32768,
            rope_parameters={"rope_type": "default", "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
            eos_token_id=2, pad_token_id=2,
        )
        model = transformers.Qwen3_5ForCausalLM(native_config).to(device="cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            model.model.layers[0].linear_attn.in_proj_qkv.weight[0].zero_()
        tokenizer = TinyTokenizer()
        if task == "regression":
            generate = model.generate

            def numeric_generate(input_ids, **kwargs):
                start = input_ids.shape[1]
                return generate(input_ids=input_ids, **kwargs, prefix_allowed_tokens_fn=lambda batch, tokens:
                                list(range(51, 61)) if len(tokens) == start else [2])

            model.generate = numeric_generate
            tokenizer.decode = lambda tokens, **kwargs: "".join(chr(int(token) - 3) for token in tokens if int(token) != 2)
        originals.extend((parameter, parameter.detach().clone()) for parameter in model.parameters())
        return QwenTextModel(config, model=model, tokenizer=tokenizer, revision=config["model"]["revision"])

    monkeypatch.setattr(text_runner, "create_text_model", create)
    output = run_condition(bundle, condition, {"output_root": str(tmp_path / "runs"), "host_label": "synthetic-cuda",
                                               "wandb": {"mode": "disabled"}})
    result = write_result(context, condition, output)
    assert bool(result["checkpoints"]) == (mode == "fine_tuning")
    if mode == "fine_tuning":
        metadata = json.loads((output / "model_metadata.json").read_text())
        assert metadata["adaptation"] == (adapter_method or "lora")
    assert all(torch.equal(parameter, before) for parameter, before in originals)


@pytest.mark.real_model
@pytest.mark.skipif(os.environ.get("DAIC_QWEN_FULL_PREFLIGHT") != "1", reason="Full-checkpoint preflight is opt-in")
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_full_checkpoint_preflight(task):
    from daic_foundation_tab.transcript_study.preflight import preflight_study

    study = os.environ.get("DAIC_TRANSCRIPT_PREFLIGHT_STUDY")
    host = os.environ.get("DAIC_TRANSCRIPT_PREFLIGHT_HOST")
    if not study or not host:
        pytest.fail("Set DAIC_TRANSCRIPT_PREFLIGHT_STUDY and DAIC_TRANSCRIPT_PREFLIGHT_HOST")
    preflight_study(Path(study), Path(host), task)
