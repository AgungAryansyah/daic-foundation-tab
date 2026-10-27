# Shared tabular and transcript architecture

The repository keeps separate model execution paths and shares participant selection, prediction formats, metrics, and run records.
Existing tabular configurations retain their defaults.
The study compares frozen ICL and fine-tuning on fixed participant cohorts for depression classification and PHQ-8 score prediction.

## Model boundaries

| Input | Models | Methods | Tasks |
| --- | --- | --- | --- |
| Numeric features | TabPFN-3.5, TabICLv2, Kumo Medium | Frozen ICL, native fine-tuning | Classification, regression |
| Transcript text | Qwen3.5-2B | Frozen ICL, LoRA | Classification, generated PHQ-8 score |

`models/capabilities.py` declares supported combinations.
`models/base.py` retains the numeric interfaces.
`models/text_base.py` defines `TextExample`, `TextPrediction`, and `TextModel`.
The main runner dispatches using `input.kind`, which defaults to `tabular`.

The Qwen backend uses Transformers' text-only model class and PEFT adapters.
It resolves the checkpoint revision before token caching or execution identity checks.
The initial profile uses BF16, disables thinking, and does not use quantization.

## Participant selection

Set an explicit `data.dataset_id` before preparing a shared manifest.
DAIC-WOZ and E-DAIC must use different identities.
The participant index reads official split assignments and derives binary labels from PHQ-8 >=10.

`study.required_inputs` selects `tabular`, `transcript`, or both.
The index selects their availability intersection and records excluded participants.
Tabular availability requires the configured numeric source groups.
Use `study.feature_set` to select audio, visual, or combined availability.

Each split uses all eligible official training participants, divided into fit/context and internal selection groups.
Classification uses a stratified split. Score prediction uses a seeded random split.
The manifest records `fit`, `selection`, `dev`, and `test` roles.

Prepare one manifest per task, cohort, and seed.
Set `experiment.partition_manifest` to the same file in every corresponding model configuration.
The loader checks its fingerprint, dataset identity, labels, and official splits.
Feature preparation applies the selected cohort before aggregation.

Frozen tabular models use the fit pool as context.
Qwen selects the configured demonstration count from that pool.
Frozen methods reserve internal selection participants without using their labels for checkpoint selection.
Fine-tuning uses selection participants for checkpoint selection.
Run records distinguish the fit pool, selection pool, and actual Qwen demonstration count.

## Transcript and output contracts

Transcript settings define the file pattern, delimiter, encoding, column mapping, and speaker names.
The default reader expects tab-separated files with `speaker` and `value` columns.
It ignores hidden filesystem files and rejects duplicate sources, unknown speakers, and empty selected text.
It preserves source row order.

`participant_only` retains participant speech.
`dialogue` retains both speakers with normalized speaker labels.
Participant identifiers and query labels stay outside prompts.
Cleaned text and tokenized prompts use separate caches.
Content changes, preprocessing changes, and checkpoint revisions invalidate the relevant cache.

The encoder counts complete prompts and answer tokens.
It rejects context overflow without truncating transcripts or removing demonstrations.
The 4096-token examples are validation starting points, not a claim that complete interviews fit.
Chunking and summarization require later study decisions.

Classification scores the complete `0` and `1` answers, including EOS.
It normalizes their likelihoods and applies the configured threshold.
These probabilities are relative answer likelihoods, not assumed calibration.
Existing discrimination and calibration metrics evaluate them.

Score prediction generates one numeric answer.
The parser accepts finite values from 0 to 24 and preserves decimals.
It rejects malformed answers without clipping or imputation.
Training predicts answer tokens only. It does not train a continuous regression head.
Score reports retain MAE, RMSE, R², and classes derived at PHQ-8 >=10.

## Execution and recovery

`study_run.json` records preparation, training, prediction, evaluation, completion, or failure.
The record exists before CUDA checks, so early failures remain observable.
Logs report work units, percentages, elapsed time, throughput, and available ETA estimates.

LoRA updates adapter parameters only.
Classification selects by internal validation ROC-AUC. Score prediction selects by MAE.
The adapter saves optimizer state, random state, progress, and adapter weights at checkpoint intervals and epoch boundaries.
The selected adapter remains separate from the latest recovery checkpoint.

Resume requires the same resolved configuration, checkpoint revision, input content, and participant manifest.
The original configuration and its saved resolved configuration can identify the same checkpoint.
If no complete training checkpoint exists, the runner reports that it must restart adapter training.
Completed runs reject resume.

Local prediction files preserve existing task-specific columns.
Text runs also retain raw answers and prompt token counts locally.
W&B receives an explicit metadata allowlist and finite scalar metrics.
It does not receive transcripts, participant IDs, or raw answers.

`compare-paired` requires completed runs with matching dataset, task, manifest, cohort, and observed targets.
Its bootstrap resamples identical participants for both systems.
Differences use right minus left. Intervals describe uncertainty conditional on the saved predictions.
They do not include variation from retraining.
Legacy results remain readable but lack verified provenance for this command.
Collectors exclude new runs marked incomplete or failed.

## Usage

Install the optional backend:

```bash
rtk proxy uv sync --extra text --group dev --locked
```

The validation examples are under `configs/examples/qwen35_2b/`.
They cover both tasks and methods. They disable campaigns, bootstrap, and W&B by default.
Provide the licensed transcripts before inspecting or validating them.

```bash
rtk proxy uv run --extra text daic-foundation-tab inspect --config configs/examples/qwen35_2b/classification_icl.yaml
rtk proxy uv run --extra text daic-foundation-tab validate --config configs/examples/qwen35_2b/classification_icl.yaml
rtk proxy uv run --extra text daic-foundation-tab prepare-participants --config configs/examples/qwen35_2b/classification_icl.yaml --output outputs/participants.json
```

After setting the shared manifest and choosing the experiment settings, the existing `run` command executes one configuration.
Its optional `--resume` argument identifies a failed text run directory.
`compare-paired` accepts two run directories, `--output`, optional `--split dev|test`, and optional `--iterations`.

## Validation and extension

The default suite uses synthetic transcripts and a tiny native Qwen model.
It checks label isolation, answer masking, weight freezing, adapter reload, identity checks, and checkpoint resume.
Optional CUDA checks use a tiny random model with linear and full attention.
Their score decoder restricts numeric tokens because the random model has no learned task semantics.
These checks verify execution, not depression prediction quality or full-checkpoint memory requirements.

```bash
rtk proxy uv run --extra text pytest -q -m 'not real_model'
rtk proxy env DAIC_QWEN_SMOKE=1 uv run --extra text pytest -q tests/test_smoke_qwen.py
```

To add another language model, implement `TextModel` and register its capabilities and factory.
Keep model-specific loading and formatting inside the adapter.
Reuse the participant manifest, output contracts, metrics, and lifecycle.
Additional experiment matrices, model families, severity bins, and text-feature fusion remain future work.
