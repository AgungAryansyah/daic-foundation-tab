# DAIC-WOZ Tabular Foundation Models

This project evaluates fine-tuned TabICLv2 on participant-level DAIC-WOZ behavioral features for binary depression prediction. It does not provide a standalone zero-shot/in-context TabICL workflow, conventional baselines, other foundation models, raw audio/video, or text embeddings.

DAIC-WOZ is licensed sensitive data and is not distributed by this repository. Keep it in the local, ignored `data/` directory or set `data.root` in a copied configuration. Never commit data, cached features, checkpoints, or experiment outputs.

## Setup

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/). From the project directory:

```bash
uv sync --group dev
uv run pytest
```

The locked default uses PyTorch's CPU wheel so the project can run on modest local storage. The model chooses CUDA automatically when a CUDA-enabled PyTorch build is installed; otherwise it records CPU fallback in each run's environment metadata.

## W&B research tracking

W&B tracking is enabled by default for remote runs. Create a local `.env` from the tracked example and set the key for the private W&B account approved for this research:

```bash
cp .env.example .env
```

Set `WANDB_API_KEY` in `.env`. The tracker loads that file without overriding an already exported environment variable, and neither form of the key is written to a run configuration, output, or artifact.

Each run is grouped by a stable study fingerprint and tagged by model, modality, task, and smoke/full profile. W&B receives a sanitized configuration, dataset fingerprint, cohort-level train/validation/development/test information, fine-tuning selection metadata, aggregate metrics, uncertainty summaries, repeated-holdout metric rows, runtime, environment, and one immutable `research-record` artifact.

Participant IDs, raw labels or targets, feature values/manifests, split assignments, predictions, raw inputs, local paths, and checkpoints stay in the ignored local run directory and are never uploaded. Test predictions remain disabled; W&B records only test-cohort availability and shape.

The base configuration supports these modes:

- `online` is the default and requires W&B credentials/connectivity.
- `offline` writes W&B records beneath the ignored experiment output for later synchronization.
- `disabled` skips W&B SDK initialization for tests or tracking-free runs.

## Expected local layout

The supplied flat DAIC-WOZ copy is configured as follows:

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

Source patterns, labels, aggregation, and outputs are configuration values. Dataset inspection is discovery-only:

```bash
uv run python -m daic_foundation_tab.cli inspect \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml
```

The strict default requires every requested modality. The current local copy lacks all requested sources for one development participant, so the complete-cohort presets explicitly exclude and audit that participant:

```bash
uv run python -m daic_foundation_tab.cli build-features \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml

uv run python -m daic_foundation_tab.cli validate \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml
```

## Fine-tuning experiments

Start with the non-final one-epoch smoke run. It disables bootstrap and repeated holdouts:

```bash
uv run python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort_smoke.yaml
```

The default fine-tuning profile uses up to 50 epochs, early stopping selected by train-only validation ROC-AUC, and only retains `checkpoints/best.ckpt`. The official development split is never used for training, early stopping, feature selection, or checkpoint selection.

After confirming checkpoint loading, run the full modality ablations:

```bash
uv run python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml

uv run python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml

uv run python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_ft_visual_complete_cohort.yaml
```

Compare saved development results with:

```bash
uv run python -m daic_foundation_tab.cli compare outputs/<run-a> outputs/<run-b> outputs/<run-c>
```

Every participant becomes exactly one row after temporal mean/std pooling. Frame, timestamp, confidence, success, participant ID, PHQ-8 scores/items, labels, and targets never enter the feature matrix. The binary target is derived from `PHQ8_Score >= 10`; supplied binary labels are audited but not used.

Each run creates a reproducible stratified 80/20 split from official train. Feature filtering is fit on the 80% fine-tuning partition only, and its columns are then applied to early-stopping validation, dev, and test data. Repeated internal holdouts use a nested train-only early-stopping split for every outer evaluation split.

Each final result directory contains the resolved config, environment, feature manifest, reconciliation report, validation report, `finetune_split.csv`, `finetune_metadata.json`, the retained best checkpoint, development predictions/metrics, uncertainty artifacts, runtime/VRAM metadata, and a cautious summary. Test prediction generation is intentionally disabled until a separate frozen full-data finalization workflow is implemented.

DAIC-WOZ is an extreme-small-N dataset. Fine-tuning is a leakage-safe feasibility study and must not be presented as a clinical-use, superiority, or state-of-the-art claim.
