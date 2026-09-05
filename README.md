# DAIC-WOZ TabICLv2 Fine-Tuning

This repository is a reproducible research pipeline for fine-tuning TabICLv2 on participant-level DAIC-WOZ behavioural features for binary depression prediction. It builds numeric audio, visual, and audio-visual feature sets, runs leakage-safe development experiments, and records privacy-safe research metadata in Weights & Biases (W&B).

DAIC-WOZ is licensed sensitive data and is not included in this repository. Keep the dataset, feature cache, checkpoints, and experiment outputs outside version control.

## Remote environment

Run fine-tuning only on the remote Linux x86_64 GPU server. The project requires Python 3.12, PyTorch 2.5.1 with CUDA 12.1, and an NVIDIA driver compatible with CUDA 12.2 or later. CPU fine-tuning is intentionally unsupported.

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

## Evaluation protocol

Each participant is represented by one row after temporal mean/std pooling. Frame metadata, participant identifiers, PHQ-8 scores/items, supplied labels, and prediction targets are excluded from the feature matrix. The binary target is derived from `PHQ8_Score >= 10`; supplied binary labels are retained only for auditing.

For each run, the official training split is stratified into an 80% fine-tuning partition and 20% early-stopping validation partition. Feature filtering is fit only on the fine-tuning partition. The official development split is reserved for evaluation and never participates in feature selection, fine-tuning, or checkpoint selection. Repeated internal holdouts use nested train-only early-stopping partitions.

The default profile allows up to 50 epochs, selects on validation ROC-AUC, and retains only `checkpoints/best.ckpt`. Test prediction generation is intentionally disabled pending a separate frozen full-data finalization workflow.

## Research records

Each local result directory contains resolved settings, environment metadata, data-validation reports, feature and split provenance, the best checkpoint, development predictions and metrics, uncertainty results, runtime metadata, and a summary.

W&B charts TabICLv2's real per-epoch mean training loss and validation metrics, then records model-selection metadata, development metrics, bootstrap summaries, repeated-holdout metric rows, and runtime metadata. The immutable `research-record` artifact retains the sanitized configuration, dataset and study fingerprints, cohort-level train/validation/development/test statistics, and fine-tuning history. Participant IDs, raw targets, feature values, manifests, split assignments, predictions, raw inputs, local paths, and checkpoints remain local and are never uploaded.

## Research use

DAIC-WOZ is an extreme-small-N dataset. These experiments are a leakage-safe feasibility study and must not be presented as clinical-use, superiority, or state-of-the-art claims.
