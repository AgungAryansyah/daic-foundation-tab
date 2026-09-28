# DAIC-WOZ TabICLv2 Fine-Tuning

This repository is a reproducible research pipeline for fine-tuning TabICLv2 on participant-level DAIC-WOZ and E-DAIC behavioural features. It supports binary depression classification and PHQ-8 score regression with numeric audio, visual, and audio-visual feature sets. Runs use leakage-safe development experiments and record privacy-safe research metadata in Weights & Biases (W&B).

DAIC-WOZ is licensed sensitive data and is not included in this repository. Keep the dataset, feature cache, checkpoints, and experiment outputs outside version control.

## Remote environment

Run fine-tuning only on the remote Linux x86_64 GPU server. The project requires Python 3.12, PyTorch 2.5.1 with CUDA 11.8, and an NVIDIA driver version 450.80.02 or later for CUDA 11.x minor-version compatibility. CPU fine-tuning is intentionally unsupported.

```bash
nvidia-smi
python --version
uv sync --group dev --python 3.12
uv run --python 3.12 python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Every experiment requires `cuda:0`. The runner verifies CUDA before reading data, creating outputs, or initializing W&B, then records the selected device, CUDA runtime, GPU model, VRAM, and peak memory use.

## W&B setup

W&B tracking is enabled by default. Create a local credential file on the remote server:

```bash
cp .env.example .env
```

Set `WANDB_API_KEY` in `.env`. The key is loaded before W&B initialization and an already-exported `WANDB_API_KEY` takes precedence. Credentials are never written to configs, local run metadata, or W&B artifacts.

The supported tracking modes are configured in `tracking.wandb.mode`:

- `online` streams the run to the configured W&B project.
- `offline` stores W&B records within the ignored experiment output for later synchronization.
- `disabled` skips W&B SDK initialization.

## Dataset layout

The supplied flat DAIC-WOZ copy is expected to have this structure:

```text
data/
├── original_labels/
│   ├── train_split_Depression_AVEC2017.csv
│   ├── dev_split_Depression_AVEC2017.csv
│   └── test_split_Depression_AVEC2017.csv
└── data/
    ├── 300_COVAREP.csv
    ├── 300_CLNF_AUs.txt
    ├── 300_CLNF_gaze.txt
    └── 300_CLNF_pose.txt
```

Inspect, build, and validate the selected cohort before starting an experiment:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli inspect \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli build-features \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli validate \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml
```

## Run experiments

Start with the one-epoch smoke configuration. It disables bootstrap and repeated holdouts:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort_smoke.yaml
```

Then run the three full modality ablations:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_visual_complete_cohort.yaml
```

Compare development metrics from completed runs:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli compare \
  outputs/<run-a> outputs/<run-b> outputs/<run-c>
```

## E-DAIC CSV preparation

For the E-DAIC file-index download, keep the extracted `labels/` directory and participant `data/<id>_P/features/` directories together. Prepare the OpenSMILE eGeMAPS and OpenFace 2.1 CSVs for the E-DAIC experiment:

```bash
uv run --python 3.12 python -m daic_foundation_tab.data.prepare_edaic \
  --source /path/to/edaic \
  --output data/edaic

uv run --python 3.12 python -m daic_foundation_tab.cli inspect \
  --config configs/experiments/tabiclv2_ft_edaic_audio_visual.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli build-features \
  --config configs/experiments/tabiclv2_ft_edaic_audio_visual.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli validate \
  --config configs/experiments/tabiclv2_ft_edaic_audio_visual.yaml
```

Preparation checks all 275 participant IDs and both selected feature files before writing the flat dataset. It hard-links files when possible and copies across filesystems. Progress and failures are recorded in `data/edaic/preparation_status.json`.

The supplied E-DAIC binary labels disagree with the project's `PHQ8_Score >= 10` target for some participants. Preparation writes score-only labels, so train, development, and test evaluation derive the target consistently. The status file records disagreement counts by split. The original E-DAIC labels remain in the downloaded source directory.

### Train on E-DAIC

Run these commands from the repository root on the remote GPU server after validation succeeds. Complete the CUDA and W&B setup above first. Start with the one-epoch smoke run, which skips repeated holdouts and bootstrap:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_edaic_audio_visual_smoke.yaml
```

Then run the full E-DAIC experiments. Each command trains a separate model using the selected modality:

```bash
# Audio and visual
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_edaic_audio_visual.yaml

# Audio only
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_edaic_audio.yaml

# Visual only
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_edaic_visual.yaml
```

The full configuration allows up to 50 training epochs, then runs 30 repeated holdouts and 2,000 bootstrap iterations. Each run writes its checkpoint, predictions, metrics, and summary under `outputs/`.

## PHQ-8 score regression

The regression presets fine-tune `FinetunedTabICLRegressor` using the numeric PHQ-8 score. They select `checkpoints/best.ckpt` by validation MAE and save raw, unclipped score predictions. A secondary depression classification uses the fixed PHQ-8 cutoff of 10 on those predictions. No probability or calibration metric is inferred from regression scores.

For DAIC-WOZ, validate the combined feature set, then run the one-epoch smoke preset or any full modality preset:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli validate \
  --config configs/experiments/tabiclv2_ft_regression_audio_visual_complete_cohort.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_regression_audio_visual_complete_cohort_smoke.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_regression_audio_visual_complete_cohort.yaml
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_regression_audio_complete_cohort.yaml
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_regression_visual_complete_cohort.yaml
```

For prepared E-DAIC, use the matching E-DAIC presets:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli validate \
  --config configs/experiments/tabiclv2_ft_regression_edaic_audio_visual.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_regression_edaic_audio_visual_smoke.yaml

uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_regression_edaic_audio_visual.yaml
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_regression_edaic_audio.yaml
uv run --python 3.12 python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_regression_edaic_visual.yaml
```

The smoke presets skip bootstrap and repeated holdouts. Full presets run the same 30 nested train-only holdouts and 2,000 development bootstrap resamples as classification. Regression metrics include MAE, RMSE, R², and cutoff-derived binary metrics. Test ground truth is loaded only after prediction; set `data.test_ground_truth` to null to save unscored test predictions.

## Evaluation protocol

Each participant is represented by one row after temporal mean/std pooling. Frame metadata, participant identifiers, PHQ-8 scores/items, supplied labels, and prediction targets are excluded from the feature matrix. The binary target is derived from `PHQ8_Score >= 10`; supplied binary labels are retained only for auditing.

For each classification run, the official training split is stratified into an 80% fine-tuning partition and 20% early-stopping validation partition. Feature filtering is fit only on the fine-tuning partition. The official development split is reserved for evaluation and never participates in feature selection, fine-tuning, or checkpoint selection. Repeated internal holdouts use nested train-only early-stopping partitions.

Regression uses seeded random training and early-stopping partitions, then fits feature filtering inside the training partition. Nested repeated holdouts follow the same isolation rule. The official development split selects no features or checkpoints, and official test scores remain hidden until evaluation.

Classification hard predictions use `evaluation.threshold` on the depressed-class probability; the default is 0.5, and a score must exceed the threshold to count as depressed. Choose any alternative threshold using training-only validation data before evaluating the official test split. ROC-AUC and PR-AUC use probabilities and do not change with this threshold.

The default classification profile allows up to 50 epochs, selects on validation ROC-AUC, and retains only `checkpoints/best.ckpt`. After fine-tuning, every run saves official test predictions to `predictions_test.csv` using the selected checkpoint. Classification runs include class probabilities; regression runs include raw PHQ-8 scores and cutoff-derived classes. The DAIC-WOZ test split CSV has no labels; when `data/original_labels/full_test_split.csv` is available, its PHQ scores are matched by participant ID only after prediction to calculate `metrics_test.json` and `metrics_test.csv`. The prepared E-DAIC test split contains PHQ scores, which are used only after prediction. Set `data.test_ground_truth` to null to save predictions without test metrics.

## Research records

Each local result directory contains resolved settings, environment metadata, data-validation reports, feature and split provenance, the best checkpoint, development predictions and metrics, official test predictions, test metrics when labels are available, uncertainty results, runtime metadata, and a summary. Classification predictions include probabilities; regression predictions include raw PHQ-8 scores.

When classification test labels are available, scoring updates `outputs/test_evaluations.csv` with one row per run. The CSV includes the experiment, dataset, model, feature set, seed, decision threshold, and all scalar test metrics. Older runs without `decision_threshold.json` used the model's effective 0.5 threshold. Rows reflect saved test evaluations even if a later tracking step fails. Runs without test labels have no row. To rebuild the CSV from existing run directories without training again, use:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli collect-test-results --output-root outputs
```

Regression test scores use a separate `outputs/regression_test_evaluations.csv` with one row per scored regression run. It includes MAE, RMSE, R², and cutoff-derived metrics without a probability threshold column. Rebuild it from saved runs with:

```bash
uv run --python 3.12 python -m daic_foundation_tab.cli collect-regression-test-results --output-root outputs
```

W&B charts TabICLv2's real per-epoch mean training loss and validation metrics, then records model-selection metadata, development and test metrics, bootstrap summaries, repeated-holdout metric rows, and runtime metadata. The immutable `research-record` artifact retains the sanitized configuration, dataset and study fingerprints, cohort-level train/validation/development/test statistics, and fine-tuning history. Participant IDs, raw targets, feature values, manifests, split assignments, predictions, raw inputs, local paths, and checkpoints remain local and are never uploaded.

## Research use

DAIC-WOZ is an extreme-small-N dataset. These experiments are a leakage-safe feasibility study and must not be presented as clinical-use, superiority, or state-of-the-art claims.
