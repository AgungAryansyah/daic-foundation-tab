# DAIC-WOZ and E-DAIC Foundation Model Fine-Tuning

This repository is a reproducible research pipeline for fine-tuning TabICLv2 and TabPFN-3.5 on participant-level DAIC-WOZ and E-DAIC behavioural features, and Kumo Tabular Medium on E-DAIC. It supports binary depression classification and PHQ-8 score regression with numeric audio, visual, and audio-visual feature sets. Runs use leakage-safe development experiments and record privacy-safe research metadata in Weights & Biases (W&B).

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

## Kumo Tabular Medium experiments

Kumo runs use NVIDIA's [structured-data-models](https://github.com/NVIDIA/structured-data-models) package pinned to commit `842c408fe2a8711bdf2e7cff4bfbe54266d6b940` (package version `0.2.0rc1`). Python 3.12 and PyTorch `2.9.1+cu130` are unchanged. The adapters explicitly instantiate a single task with `size="medium"` and `pretrained=False`, then strictly load `medium/classifier.pt` or `medium/regressor.pt` from `nvidia/Kumo-Tabular` at immutable revision `3c3e10bbdb590ace29e7026847f92db3c603096d` (`v1.0.1`). The [weights](https://huggingface.co/nvidia/Kumo-Tabular) use OpenMDW-1.1; review its model card and license before use. Source code uses Apache-2.0.

The public Hugging Face checkpoints download automatically and are reused from the local Hub cache. No TabPFN token or Kumo hosted API key is used. If the Hub requires authentication in your environment, configure its standard `HF_TOKEN` privately; do not put credentials in configuration files or launch commands. Participant data stays on the execution machine.

This integration **updates pretrained model weights** with a small PyTorch loop based on [NVIDIA's fine-tuning example](https://github.com/NVIDIA/structured-data-models/blob/842c408fe2a8711bdf2e7cff4bfbe54266d6b940/examples/tabular/finetune.py). Native `KumoTabular.fit()` only caches labeled context examples; it is used after weight fine-tuning for inference. Training uses gradient-enabled `forward()` with disjoint context/query participants. Each epoch shuffles the training partition into chunks of at most 10,000 participants, with a 20% query fraction rounded up and at least two context participants. Binary contexts reserve one participant from each class. E-DAIC fits have one chunk and one optimizer step per epoch.

All parameters are trainable. The training recipe retains NVIDIA's feature preprocessing, disables output postprocessing and regression target flipping, and averages losses across both training estimators. Classification uses cross-entropy; regression uses pinball loss over 999 quantiles in raw target units. Evaluation uses the native output recipe, aligns probability columns to `[0, 1]`, and averages the postprocessed quantiles for the raw PHQ-8 estimate. Validation preprocessing receives the same seed on every epoch. Feature filtering and preprocessing use training participants only.

### Presets and selection

All eight presets are under `configs/experiments/`:

| Task | Features | Full preset |
| --- | --- | --- |
| Classification | Audio | `kumo_medium_ft_edaic_audio.yaml` |
| Classification | Visual | `kumo_medium_ft_edaic_visual.yaml` |
| Classification | Audio + visual | `kumo_medium_ft_edaic_audio_visual.yaml` |
| Regression | Audio | `kumo_medium_ft_regression_edaic_audio.yaml` |
| Regression | Visual | `kumo_medium_ft_regression_edaic_visual.yaml` |
| Regression | Audio + visual | `kumo_medium_ft_regression_edaic_audio_visual.yaml` |

The two one-epoch smokes are `kumo_medium_ft_edaic_audio_visual_smoke.yaml` and `kumo_medium_ft_regression_edaic_audio_visual_smoke.yaml`. They disable bootstrap and repeated holdouts while preserving 2/2/8 training/validation/inference estimators.

Full presets preserve the matching TabICL/TabPFN E-DAIC participant cohort, sources, modality policy, mean/std aggregation, feature filtering and splits. They use seed 42, 20% training-only validation, up to 50 epochs, AdamW at `1e-5`, weight decay `0.01`, gradient clipping `1.0`, CUDA FP16 autocast with gradient scaling, and cosine scheduling with 10% optimizer-step warmup (rounded down; a one-step smoke has no warmup). Early stopping uses patience 10 and strict improvement greater than `1e-4`. No interval checkpoints or additional activation-checkpoint wrappers are used.

Kumo classification selects by validation ROC-AUC. **Kumo regression selects by MSE, matching TabPFN; TabICL regression selects by MAE.** Regression training uses pinball loss, independently of checkpoint selection. All models report MAE, RMSE, R² and classes derived at PHQ-8 ≥10. The classification evaluation threshold remains 0.5.

The pretrained baseline is evaluated and saved before the first optimizer step. `checkpoints/best.pt` contains the restored selected `KumoTabular.state_dict()`, including when the baseline wins. `finetune_metadata.json` records its path and SHA-256, the source checkpoint SHA-256, baseline/best validation values, selected epoch (zero means baseline), completed epochs and optimizer steps. `model.json` and sanitized W&B research metadata retain the model/package source revisions. Predictions, participant data, checkpoints, credentials and local paths remain outside W&B artifacts.

### Validate and launch on the remote GPU

After Git synchronization, verify the intended commit, prepared `data/edaic`, CUDA availability and W&B credentials. Use the remote terminal's native commands:

```bash
uv sync --group dev --python 3.12 --locked
uv run --no-sync python -c "import torch; assert torch.__version__ == '2.9.1+cu130'; assert torch.cuda.is_available(); print(torch.version.cuda, torch.cuda.get_device_name(0))"
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
DAIC_RUN_REAL_KUMO_FINETUNE=1 uv run --no-sync pytest -vv -s tests/test_smoke_kumo_medium_ft.py
```

The opt-in technical checks download both real Medium checkpoints, retain 2/2/8 estimators, confirm an optimizer step changes weights, check finite predictions and strictly reload selected checkpoints with matching predictions. They use small synthetic tables; the two matrix smokes additionally validate the real E-DAIC audio-visual workload.

The existing individual-run CLI works with any Kumo preset. To launch both smokes followed by all six full experiments in a detached remote process:

```bash
mkdir -p outputs
nohup uv run --no-sync python -u -m daic_foundation_tab.matrix --model-family kumo_medium > outputs/kumo_medium_matrix_launcher.log 2>&1 < /dev/null &
echo $!
```

Confirm launch once using its PID and initial log, then leave the process running. The launcher gates full experiments on both successful smokes, runs sequentially on `cuda:0`, and retains failures without changing settings. Resume with the same command only after confirming the previous process ended. A lock prevents duplicate matrix launchers. Only verified complete presets are skipped; retries use fresh directories.

`outputs/kumo_medium_matrix_status.json` records each attempt, elapsed time, output directory and failure, with 30-second updates and an adjacent progress log. Each run also retains phase/epoch/bootstrap/holdout progress and heartbeats. Full presets perform 30 nested train-only holdouts and 2,000 development bootstrap resamples: six full presets require 186 main/holdout fits. Runtime is workload-dependent; the matrix does not estimate an ETA.

Check saved completion without training again:

```bash
uv run --no-sync python -m daic_foundation_tab.matrix --model-family kumo_medium --verify-only
```

Verification covers checkpoint hashes and revisions, CUDA use, finite development/test predictions and metrics, all bootstrap/holdout records, W&B completion and both aggregate result CSVs. Compare only available matching E-DAIC TabICL/TabPFN results, preserving the regression-selection and training-loss differences. The default matrix invocation still runs the original TabPFN matrix.

## Evaluation protocol

Each participant is represented by one row after temporal mean/std pooling. Frame metadata, participant identifiers, PHQ-8 scores/items, supplied labels, and prediction targets are excluded from the feature matrix. The binary target is derived from `PHQ8_Score >= 10`; supplied binary labels are retained only for auditing.

For each classification run, the official training split is stratified into an 80% fine-tuning partition and 20% early-stopping validation partition. Feature filtering is fit only on the fine-tuning partition. The official development split is reserved for evaluation and never participates in feature selection, fine-tuning, or checkpoint selection. Repeated internal holdouts use nested train-only early-stopping partitions.

Regression uses seeded random training and early-stopping partitions, then fits feature filtering inside the training partition. Nested repeated holdouts follow the same isolation rule. The official development split selects no features or checkpoints, and official test scores remain hidden until evaluation.

Classification hard predictions use `evaluation.threshold` on the depressed-class probability; the default is 0.5, and a score must exceed the threshold to count as depressed. Choose any alternative threshold using training-only validation data before evaluating the official test split. ROC-AUC and PR-AUC use probabilities and do not change with this threshold.

The default classification profile allows up to 50 epochs and selects on validation ROC-AUC. TabICL retains `checkpoints/best.ckpt`; TabPFN retains a native `checkpoints/checkpoint_*_best.pth` file; Kumo retains `checkpoints/best.pt`. After fine-tuning, every run saves official test predictions to `predictions_test.csv` using the selected checkpoint. Classification runs include class probabilities; regression runs include raw PHQ-8 scores and cutoff-derived classes. The DAIC-WOZ test split CSV has no labels; when `data/original_labels/full_test_split.csv` is available, its PHQ scores are matched by participant ID only after prediction to calculate `metrics_test.json` and `metrics_test.csv`. The prepared E-DAIC test split contains PHQ scores, which are used only after prediction. Set `data.test_ground_truth` to null to save predictions without test metrics.

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
