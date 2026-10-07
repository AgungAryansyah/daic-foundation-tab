# DAIC-WOZ and E-DAIC Foundation Model Fine-Tuning

This repository is a reproducible research pipeline for fine-tuning TabICLv2 and TabPFN-3.5 on participant-level DAIC-WOZ and E-DAIC behavioural features. It supports binary depression classification and PHQ-8 score regression with numeric audio, visual, and audio-visual feature sets. Runs use leakage-safe development experiments and record privacy-safe research metadata in Weights & Biases (W&B).

DAIC-WOZ is licensed sensitive data and is not included in this repository. Keep the dataset, feature cache, checkpoints, and experiment outputs outside version control.

## Remote environment

Run fine-tuning only on the remote Linux x86_64 GPU server. The project requires Python 3.12, PyTorch 2.9.1 with CUDA 13.0, and an NVIDIA driver from the R580 series or newer. Linux driver 580.65.06 or newer is recommended for CUDA 13.0; see the [NVIDIA compatibility table](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-driver). The CUDA 13 wheel requires a supported NVIDIA GPU (Turing or newer) and glibc 2.28 or newer. CPU fine-tuning is intentionally unsupported.

```bash
nvidia-smi
python --version
uv sync --group dev --python 3.12
uv run --python 3.12 python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

The [official CUDA 13 PyTorch wheels](https://pytorch.org/get-started/previous-versions/#v2-9-1) supply the CUDA runtime through Python dependencies; installing a separate system CUDA toolkit is unnecessary for these experiments. `nvidia-smi` reports driver capability, while `torch.version.cuda` reports the runtime used by PyTorch. Resync an existing environment with `uv sync --group dev --python 3.12 --locked`.

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

The TabICL regression presets fine-tune `FinetunedTabICLRegressor` using the numeric PHQ-8 score. They select `checkpoints/best.ckpt` by validation MAE and save raw, unclipped score predictions. A secondary depression classification uses the fixed PHQ-8 cutoff of 10 on those predictions. No probability or calibration metric is inferred from regression scores.

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

## TabPFN-3.5 experiments

`tabpfn==9.1.0` is pinned alongside Python 3.12 and `torch==2.9.1+cu130`. The adapters explicitly use `ModelVersion.V3_5`, the full multitask checkpoint `tabpfn-v3.5-20260909.safetensors`, and native `FinetunedTabPFNClassifier` / `FinetunedTabPFNRegressor` training. TabPFN selects CUDA mixed precision automatically.

### Model access and remote validation

Obtain authorized model access before starting a headless run. Follow the [official authentication instructions](https://github.com/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb): sign in at [PriorLabs](https://ux.priorlabs.ai), review and accept the applicable model license yourself, and obtain an access token. Put `TABPFN_TOKEN` in the remote `.env` or export it privately. Exported credentials take precedence. `.env.example` sets `TABPFN_NO_BROWSER=1` so unattended runs fail clearly instead of waiting for login. Previously authorized cached weights can be used without downloading again.

Verify the remote checkout contains this integration, complete the CUDA/W&B setup and E-DAIC preparation above, and check the environment from the repository root:

```bash
uv sync --group dev --python 3.12 --locked
uv run --no-sync python -c "import torch; from importlib.metadata import version; assert version('tabpfn') == '9.1.0'; assert torch.__version__ == '2.9.1+cu130'; assert torch.version.cuda == '13.0'; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
uv run --no-sync python -c "from dotenv import load_dotenv; load_dotenv('.env'); from tabpfn.browser_auth import ensure_license_accepted; ensure_license_accepted(hf_repo_id='tabpfn_3_5')"
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
```

Keep DAIC-WOZ test truth at `data/original_labels/full_test_split.csv` and prepared E-DAIC test labels at `data/edaic/original_labels/test_split.csv`. Individual runs support unscored test predictions, but matrix completion requires development and test results for every preset. Validate all presets before launching:

```bash
for preset in configs/experiments/tabpfn35_ft_*.yaml; do
  uv run --no-sync python -m daic_foundation_tab.cli validate --config "$preset" || exit 1
done
```

### Presets

All filenames are under `configs/experiments/`. Each mirrors its `tabiclv2_ft` counterpart's modality enablement, missing-modality policy, participant cohort, mean/std aggregation, filtering, splits, and evaluation.

| Dataset | Task | Features | Full preset |
| --- | --- | --- | --- |
| DAIC-WOZ | Classification | Audio | `tabpfn35_ft_audio_complete_cohort.yaml` |
| DAIC-WOZ | Classification | Visual | `tabpfn35_ft_visual_complete_cohort.yaml` |
| DAIC-WOZ | Classification | Audio + visual | `tabpfn35_ft_audio_visual_complete_cohort.yaml` |
| E-DAIC | Classification | Audio | `tabpfn35_ft_edaic_audio.yaml` |
| E-DAIC | Classification | Visual | `tabpfn35_ft_edaic_visual.yaml` |
| E-DAIC | Classification | Audio + visual | `tabpfn35_ft_edaic_audio_visual.yaml` |
| DAIC-WOZ | Regression | Audio | `tabpfn35_ft_regression_audio_complete_cohort.yaml` |
| DAIC-WOZ | Regression | Visual | `tabpfn35_ft_regression_visual_complete_cohort.yaml` |
| DAIC-WOZ | Regression | Audio + visual | `tabpfn35_ft_regression_audio_visual_complete_cohort.yaml` |
| E-DAIC | Regression | Audio | `tabpfn35_ft_regression_edaic_audio.yaml` |
| E-DAIC | Regression | Visual | `tabpfn35_ft_regression_edaic_visual.yaml` |
| E-DAIC | Regression | Audio + visual | `tabpfn35_ft_regression_edaic_audio_visual.yaml` |

The four audio-visual smoke presets are:

- `tabpfn35_ft_audio_visual_complete_cohort_smoke.yaml`
- `tabpfn35_ft_edaic_audio_visual_smoke.yaml`
- `tabpfn35_ft_regression_audio_visual_complete_cohort_smoke.yaml`
- `tabpfn35_ft_regression_edaic_audio_visual_smoke.yaml`

Full presets use at most 50 epochs, learning rate `1e-5`, weight decay `0.01`, gradient clipping `1.0`, cosine scheduling with native 10% warmup, patience 10, and minimum improvement `1e-4`. Ensembles are 2/2/8 for training/validation/inference. Context-plus-query chunks contain at most 10,000 participants with a 20% query fraction. Activation checkpointing is enabled; interval checkpoints are disabled. Preprocessing draws are seeded from `random_state: 42`; `use_fixed_preprocessing_seed: false` follows the native recommendation for unequal training and inference ensembles.

The project seed is 42. Early-stopping validation uses 20% of official training participants. Full runs retain classification threshold 0.5, 30 nested train-only holdouts, and 2,000 development bootstrap resamples. Smokes use one epoch and skip holdouts and bootstrap.

**Regression checkpoint selection differs:** TabPFN minimizes native validation **MSE**; TabICL minimizes validation **MAE**. Both report MAE, RMSE, R², raw PHQ-8 predictions, and classes derived at PHQ-8 ≥10. Both classifiers maximize validation ROC-AUC. Keep this selection difference explicit in comparisons.

TabPFN retains native `checkpoints/checkpoint_*_best.pth` names and exports restored selected weights even when the pretrained baseline remains best. `finetune_metadata.json` records the checkpoint path, SHA-256, selection metric, baseline and selected validation values, and selected epoch (zero means baseline). A small observation hook captures selection because the public logger omits baseline evaluation; it preserves native selection and early stopping and is tested against the pinned package contract.

### Run or resume the matrix

The existing per-experiment CLI remains available:

```bash
uv run --no-sync python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabpfn35_ft_audio_visual_complete_cohort_smoke.yaml
```

On the supplied remote GPU target, launch all four smokes followed by twelve full experiments sequentially through the same CLI:

```bash
uv run --no-sync python -m daic_foundation_tab.matrix
uv run --no-sync python -m daic_foundation_tab.matrix --verify-only
```

Each preset runs in a separate process on `cuda:0`. Smoke failures block full experiments; full-run failures are recorded while remaining presets are attempted. The launcher exits unsuccessfully while any preset remains incomplete and never reduces configured ensembles, epochs, holdouts, or bootstrap counts. Run it again to resume: only verified completed runs are skipped; failed, interrupted, or invalid runs receive fresh directories.

`outputs/tabpfn35_matrix_status.json` persists attempt history, status, elapsed time, output directory, and failures, updating running status every 30 seconds. The adjacent `.log` and each run's `run.log` retain phase and heartbeat progress. Participant, epoch, bootstrap, and holdout counts include percentages and throughput. ETAs use observed throughput; training progress uses the maximum epoch budget and may finish sooner through early stopping. Matrix ETA is unavailable because smoke and full workloads differ.

Verification checks resolved settings, package/model version, CUDA device, checkpoint hash, prediction counts and finite values, development/test metrics, bootstrap count, holdout seeds and assignments, and W&B completion. Both aggregate test CSVs are rebuilt and checked for every completed matrix run. `--verify-only` checks without training. Use existing collectors and the comparison command for matching saved TabICL runs; absent baselines remain missing.

### Opt-in real-model checks

Automated tests use synthetic participants and mocked native training. Native constructors are checked without loading weights. Two additional one-epoch GPU checks validate finite predictions and checkpoint reloadability when explicitly enabled on the remote host with an authorized local checkpoint:

```bash
DAIC_RUN_REAL_TABPFN_FINETUNE=1 \
DAIC_TABPFN35_CHECKPOINT=/absolute/path/tabpfn-v3.5-20260909.safetensors \
  uv run --no-sync pytest -q tests/test_smoke_tabpfn35_ft.py -m real_model
```

These technical checks use one estimator per stage; the four experiment smokes preserve 2/2/8 ensembles. Remote experiment execution remains pending until target connection details are supplied.

## Evaluation protocol

Each participant is represented by one row after temporal mean/std pooling. Frame metadata, participant identifiers, PHQ-8 scores/items, supplied labels, and prediction targets are excluded from the feature matrix. The binary target is derived from `PHQ8_Score >= 10`; supplied binary labels are retained only for auditing.

For each classification run, the official training split is stratified into an 80% fine-tuning partition and 20% early-stopping validation partition. Feature filtering is fit only on the fine-tuning partition. The official development split is reserved for evaluation and never participates in feature selection, fine-tuning, or checkpoint selection. Repeated internal holdouts use nested train-only early-stopping partitions.

Regression uses seeded random training and early-stopping partitions, then fits feature filtering inside the training partition. Nested repeated holdouts follow the same isolation rule. The official development split selects no features or checkpoints, and official test scores remain hidden until evaluation.

Classification hard predictions use `evaluation.threshold` on the depressed-class probability; the default is 0.5, and a score must exceed the threshold to count as depressed. Choose any alternative threshold using training-only validation data before evaluating the official test split. ROC-AUC and PR-AUC use probabilities and do not change with this threshold.

The default classification profile allows up to 50 epochs and selects on validation ROC-AUC. TabICL retains `checkpoints/best.ckpt`; TabPFN retains a native `checkpoints/checkpoint_*_best.pth` file. After fine-tuning, every run saves official test predictions to `predictions_test.csv` using the selected checkpoint. Classification runs include class probabilities; regression runs include raw PHQ-8 scores and cutoff-derived classes. The DAIC-WOZ test split CSV has no labels; when `data/original_labels/full_test_split.csv` is available, its PHQ scores are matched by participant ID only after prediction to calculate `metrics_test.json` and `metrics_test.csv`. The prepared E-DAIC test split contains PHQ scores, which are used only after prediction. Set `data.test_ground_truth` to null to save predictions without test metrics.

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

W&B charts each model's real per-epoch mean training loss and validation metrics, then records model-selection metadata, development and test metrics, bootstrap summaries, repeated-holdout metric rows, and runtime metadata. The immutable `research-record` artifact retains the sanitized configuration, dataset and study fingerprints, cohort-level train/validation/development/test statistics, and fine-tuning history. Participant IDs, raw targets, feature values, manifests, split assignments, predictions, raw inputs, local paths, and checkpoints remain local and are never uploaded.

## Research use

DAIC-WOZ is an extreme-small-N dataset. These experiments are a leakage-safe feasibility study and must not be presented as clinical-use, superiority, or state-of-the-art claims.
