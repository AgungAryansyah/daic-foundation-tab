from __future__ import annotations

import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.evaluation.regression import regression_metrics

from .text_base import TextExample, TextPrediction
from .text_prompt import PromptEncoder, parse_score


class QwenTextModel:
    def __init__(self, config: dict, *, model=None, tokenizer=None, revision: str | None = None):
        self.config = config
        self.parameters = config["model"]["parameters"]
        self.task = config["experiment"]["task"]
        self.mode = config["experiment"].get("mode", "fine_tuning")
        self.model = model
        self.tokenizer = tokenizer
        self.revision = revision
        self.encoder: PromptEncoder | None = None
        self.optimizer_steps = 0
        self.best_metric: float | None = None

    def _load(self) -> None:
        import torch

        if self.model is None:
            from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Qwen profiles require a CUDA device with BF16 support")
            model_id = self.config["model"]["model_id"]
            revision = self.config["model"].get("revision", "main")
            model_config = AutoConfig.from_pretrained(model_id, revision=revision)
            self.revision = model_config._commit_hash
            if not self.revision:
                raise RuntimeError("Cannot resolve an immutable Qwen checkpoint revision")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
            self.model = Qwen3_5ForCausalLM.from_pretrained(
                model_id,
                revision=self.revision,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(self.config["runtime"]["device"])
        if self.tokenizer is None or not self.revision:
            raise ValueError("A tokenizer and immutable model revision are required")
        self.model.requires_grad_(False)
        self.model.eval()
        self.encoder = PromptEncoder(
            self.tokenizer,
            self.task,
            self.parameters,
            {"model_id": self.config["model"]["model_id"], "revision": self.revision},
            Path(self.config["project"]["cache_root"]) / "text_tokens",
        )

    def prepare(self, demonstrations: Sequence[TextExample]) -> None:
        if self.encoder is None:
            self._load()
        if self.mode == "fine_tuning" and demonstrations:
            raise ValueError("LoRA fine-tuning does not use demonstrations")
        self.encoder.demonstrations = list(demonstrations)

    def preflight(self, examples: Sequence[TextExample]) -> None:
        if self.encoder is None:
            raise RuntimeError("Call prepare before preflight")
        limit = getattr(self.model.config, "max_position_embeddings", self.encoder.max_length)
        if self.encoder.max_length > limit:
            raise ValueError("Configured context limit exceeds the model context window")
        for example in examples:
            prompt = self.encoder.prompt(example)
            if self.task == "classification":
                for answer in ("0", "1"):
                    self.encoder.check_length(len(prompt), len(self.encoder.completion(answer)))
            if example.target is not None and self.mode == "fine_tuning":
                self.encoder.training_record(example)

    def _tensor(self, values):
        import torch

        return torch.tensor([values], dtype=torch.long, device=next(self.model.parameters()).device)

    def _candidate_score(self, prompt: list[int], answer: str):
        import torch

        suffix = self.encoder.completion(answer)
        self.encoder.check_length(len(prompt), len(suffix))
        tokens = self._tensor(prompt + suffix)
        logits = self.model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            use_cache=False,
            logits_to_keep=len(suffix) + 1,
        ).logits
        probabilities = logits[0, -(len(suffix) + 1) : -1].float().log_softmax(-1)
        return probabilities.gather(1, self._tensor(suffix).T).sum()

    def predict(self, examples: Sequence[TextExample]) -> list[TextPrediction]:
        import torch

        if self.encoder is None:
            raise RuntimeError("Call prepare before prediction")
        self.model.eval()
        demo_ids = {example.participant_id for example in self.encoder.demonstrations}
        results = []
        with torch.inference_mode():
            for example in examples:
                if example.participant_id in demo_ids:
                    raise ValueError("Evaluation participant appears in ICL demonstrations")
                prompt = self.encoder.prompt(example)
                if self.task == "classification":
                    scores = torch.stack(
                        [self._candidate_score(prompt, str(label)) for label in (0, 1)]
                    )
                    probability = float(scores.softmax(0)[1].item())
                    if not math.isfinite(probability):
                        raise ValueError("Non-finite class answer likelihood")
                    value = int(probability > self.config["evaluation"]["threshold"])
                    results.append(TextPrediction(value, probability, str(value), len(prompt)))
                else:
                    tokens = self._tensor(prompt)
                    output = self.model.generate(
                        input_ids=tokens,
                        attention_mask=torch.ones_like(tokens),
                        do_sample=False,
                        max_new_tokens=self.encoder.max_new_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                    answer = self.tokenizer.decode(
                        output[0, len(prompt) :], skip_special_tokens=True
                    )
                    results.append(TextPrediction(parse_score(answer), None, answer, len(prompt)))
        return results

    def _validation_metric(self, validation):
        predictions = self.predict(validation)
        target = [item.target for item in validation]
        if self.task == "classification":
            return classification_metrics(
                target,
                [item.value for item in predictions],
                [item.probability_positive for item in predictions],
            )["roc_auc"]
        return regression_metrics(target, [item.value for item in predictions])["mae"]

    def fit(self, training, validation, checkpoint_directory: Path, progress=None, resume=False):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model

        if self.mode != "fine_tuning":
            raise ValueError("Frozen models cannot perform optimizer updates")
        if not training or not validation:
            raise ValueError("Training and validation examples are required")
        if self.task == "classification" and {item.target for item in validation} != {0, 1}:
            raise ValueError("ROC-AUC selection requires both validation classes")
        self.prepare([])
        self.preflight([*training, *validation])
        records = [self.encoder.training_record(example) for example in training]
        directory = Path(checkpoint_directory)
        directory.mkdir(parents=True, exist_ok=True)
        identity = fingerprint(
            {
                "revision": self.revision,
                "parameters": self.parameters,
                "task": self.task,
                "seed": self.config["project"]["seed"],
                "training": [(x.participant_id, x.text, x.target) for x in training],
                "validation": [(x.participant_id, x.text, x.target) for x in validation],
            }
        )
        start_epoch, offset = 0, 0
        state = None
        latest_path = directory / "latest.json"
        if resume:
            if not latest_path.is_file():
                raise ValueError("No complete training checkpoint is available to resume")
            latest = directory / json.loads(latest_path.read_text())["checkpoint"]
            state = torch.load(latest / "training.pt", map_location="cpu", weights_only=True)
            if state["identity"] != identity:
                raise ValueError("Training resume identity mismatch")
            self.model = PeftModel.from_pretrained(
                self.model, latest / "adapter", is_trainable=True
            )
            start_epoch, offset = state["epoch"], state["offset"]
            self.optimizer_steps, self.best_metric = state["steps"], state["best_metric"]
        else:
            if latest_path.exists():
                raise ValueError("Training checkpoints already exist; explicit resume is required")
            rank = self.parameters.get("lora_rank", 8)
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=rank,
                    lora_alpha=2 * rank,
                    lora_dropout=0.0,
                    target_modules="all-linear",
                    bias="none",
                ),
            )
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable or any(
            parameter.requires_grad and "lora_" not in name
            for name, parameter in self.model.named_parameters()
        ):
            raise RuntimeError("Only LoRA adapter parameters may be trainable")
        optimizer = torch.optim.AdamW(trainable, lr=self.parameters.get("learning_rate", 0.0002))
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            if torch.cuda.is_available() and state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        epochs = self.parameters.get("epochs", 1)
        accumulation = self.parameters.get("gradient_accumulation_steps", 1)
        total_steps = epochs * math.ceil(len(records) / accumulation)
        save_every = self.parameters.get("checkpoint_steps", 10)

        def save(epoch, batch_offset):
            name = f"step-{self.optimizer_steps}-epoch-{epoch}-offset-{batch_offset}"
            destination = directory / name
            temporary = directory / f"{name}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            self.model.save_pretrained(temporary / "adapter")
            torch.save(
                {
                    "identity": identity,
                    "epoch": epoch,
                    "offset": batch_offset,
                    "steps": self.optimizer_steps,
                    "best_metric": self.best_metric,
                    "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                temporary / "training.pt",
            )
            if destination.exists():
                shutil.rmtree(destination)
            temporary.replace(destination)
            old = (
                json.loads(latest_path.read_text())["checkpoint"] if latest_path.exists() else None
            )
            pointer = directory / "latest.tmp"
            pointer.write_text(json.dumps({"checkpoint": name}))
            pointer.replace(latest_path)
            if old and old != name:
                shutil.rmtree(directory / old)

        if progress:
            progress(self.optimizer_steps, total_steps)
        for epoch in range(start_epoch, epochs):
            order = np.random.default_rng(self.config["project"]["seed"] + epoch).permutation(
                len(records)
            )
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            for position in range(offset, len(order)):
                record = records[order[position]]
                batch = {key: self._tensor(value) for key, value in record.items()}
                group_start = (position // accumulation) * accumulation
                group_size = min(accumulation, len(order) - group_start)
                loss = self.model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite adapter training loss")
                (loss / group_size).backward()
                if (position + 1) % accumulation == 0 or position + 1 == len(order):
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.optimizer_steps += 1
                    if self.optimizer_steps % save_every == 0:
                        save(epoch, position + 1)
                    if progress:
                        progress(self.optimizer_steps, total_steps)
            metric = self._validation_metric(validation)
            if not math.isfinite(metric):
                raise ValueError("Non-finite checkpoint selection metric")
            better = self.best_metric is None or (
                metric > self.best_metric
                if self.task == "classification"
                else metric < self.best_metric
            )
            if better:
                self.best_metric = metric
                self.model.save_pretrained(directory / "best_adapter")
            save(epoch + 1, 0)
            offset = 0
        if not (directory / "best_adapter" / "adapter_config.json").is_file():
            raise ValueError("The selected adapter checkpoint is missing")
        self.model.load_adapter(
            directory / "best_adapter", adapter_name="selected", is_trainable=False
        )
        self.model.set_adapter("selected")
        self.model.requires_grad_(False)
        self.model.eval()
        return self

    def load_adapter(self, path: Path) -> None:
        from peft import PeftModel

        if self.encoder is None:
            self.prepare([])
        self.model = PeftModel.from_pretrained(self.model, path, is_trainable=False)
        self.model.requires_grad_(False)
        self.model.eval()

    def run_metadata(self) -> dict:
        return {
            "model_name": "qwen35_2b",
            "model_id": self.config["model"]["model_id"],
            "checkpoint_revision": self.revision,
            "adaptation": "lora" if self.mode == "fine_tuning" else "frozen_icl",
            "optimizer_steps": self.optimizer_steps,
            "best_validation_metric": self.best_metric,
            "selection_metric": "roc_auc" if self.task == "classification" else "mae",
            "probability_kind": "normalized_answer_likelihood"
            if self.task == "classification"
            else None,
            "base_parameters_frozen": all(
                not parameter.requires_grad
                for name, parameter in self.model.named_parameters()
                if "lora_" not in name
            ),
            "demonstrations": len(self.encoder.demonstrations),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
        }
