# DAIC-WOZ Tabular Foundation Models

This project evaluates pretrained tabular foundation models on participant-level DAIC-WOZ behavioral features for binary depression prediction. Phase 1 implements TabICLv2 only; it does not include baselines, other foundation models, fine-tuning, raw audio/video, or text embeddings.

DAIC-WOZ is licensed sensitive data and is not distributed by this repository. Keep it in the local, ignored `data/` directory or set `data.root` in a copied configuration. Never commit data, cached features, checkpoints, or experiment outputs.

## Setup

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/). From the project directory:

```bash
uv sync --group dev
uv run pytest
```

The locked default uses PyTorch's CPU wheel so the project can run on modest local storage. The model chooses CUDA automatically when a CUDA-enabled PyTorch build is installed; otherwise it records CPU fallback in each run's environment metadata.

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
  --config configs/experiments/tabiclv2_audio_visual.yaml
```

The strict default requires every requested modality. The current local copy lacks all requested sources for one development participant, so the complete-cohort presets explicitly exclude and audit that participant:

```bash
uv run python -m daic_foundation_tab.cli build-features \
  --config configs/experiments/tabiclv2_audio_visual_complete_cohort.yaml

uv run python -m daic_foundation_tab.cli validate \
  --config configs/experiments/tabiclv2_audio_visual_complete_cohort.yaml
```

## Phase 1 experiments

Start with the non-final smoke run. It uses one TabICLv2 estimator without bootstrap or repeated holdouts:

```bash
uv run python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_audio_visual_complete_cohort_smoke.yaml
```

After confirming checkpoint loading, run the primary official train-to-development experiment and the modality ablations:

```bash
uv run python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_audio_visual_complete_cohort.yaml

uv run python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_audio_complete_cohort.yaml

uv run python -m daic_foundation_tab.cli run \
  --config configs/experiments/tabiclv2_visual_complete_cohort.yaml
```

Compare saved development results with:

```bash
uv run python -m daic_foundation_tab.cli compare outputs/<run-a> outputs/<run-b> outputs/<run-c>
```

Every participant becomes exactly one row after temporal mean/std pooling. Frame, timestamp, confidence, success, participant ID, PHQ-8 scores/items, labels, and targets never enter the feature matrix. The binary target is derived from `PHQ8_Score >= 10`; supplied binary labels are audited but not used.

`TabICLClassifier.fit()` stores the context table for pretrained in-context inference. It does not train a foundation model from scratch. Each final result directory contains the resolved config, environment, feature manifest, reconciliation report, validation report, development predictions/metrics, uncertainty artifacts, runtime/VRAM metadata, and a cautious summary. Test mode produces predictions only after a configuration is frozen; it does not compute test metrics.

DAIC-WOZ is an extreme-small-N dataset. Phase 1 is a leakage-safe feasibility study and must not be presented as a clinical-use, superiority, or state-of-the-art claim.
