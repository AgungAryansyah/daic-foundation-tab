# Distributed tabular study

This study compares TabPFN-3.5 and Kumo Medium with frozen in-context learning (ICL) and fine-tuning. It evaluates binary depression classification and PHQ-8 score regression. The feature sets are audio, visual, and audio plus visual. It does not use a label-budget sweep.

The DAIC-WOZ YAML is `configs/studies/daic_woz_tabular_cv.yaml`. The E-DAIC YAML is `configs/studies/edaic_tabular_cv.yaml`. Each specifies five outer folds and one model seed (42). This gives 120 cross-validation conditions per dataset: two models × three feature sets × two methods × two tasks × five folds. Each dataset has six model-and-feature groups with 20 conditions each. There is a separate, optional final stage with 24 conditions per dataset; preparing the study does not run it.

All modalities use the same eligible audio-and-visual participant cohort. For each task, the folds are shared across models, modalities, and methods. Classification folds are stratified by class; regression folds use a separate split. Each outer training fold reserves 20% of its participants for fine-tuning selection. Frozen ICL uses the same remaining participants as its context. Feature selection uses only that fit/context pool. Outer-fold participants are never used for fitting or model selection. The official development and test splits remain outside cross-validation.

## Prepare once

Set up the same repository source and locked Python environment on each system as described in the README. Put the DAIC-WOZ data at the location in the YAML, or override `data.root` in a local configuration that extends the default. Model weights and any required model access must be available on each GPU host.

From the repository root, prepare an empty destination:

```bash
uv run --no-sync daic-foundation-tab prepare-tabular-study \
  --config configs/studies/daic_woz_tabular_cv.yaml \
  --output outputs/tabular-study
```

Preparation creates the shared features, participant assignments, folds, and condition schedule. It reports progress and does not train a model. Copy this bundle privately to each system. It contains participant-level features and labels and must be protected as sensitive data. Its hashes identify the prepared contents and the source and dependency lock. Every system must use matching source files, `pyproject.toml`, `uv.lock`, dependencies, and pretrained checkpoint contents. A source change requires a new bundle or the original source version.

For E-DAIC, use `configs/studies/edaic_tabular_cv.yaml` and a separate destination such as `outputs/edaic-tabular-study`. Its default data root is `data/edaic`, with 163 training, 56 development, and 56 test participants. It uses the existing OpenSMILE eGeMAPS audio features and OpenFace 2.1 visual features. DAIC-WOZ uses COVAREP audio features and CLNF visual features. The same modality name therefore does not imply the same feature extractor across datasets.

Run E-DAIC first, then prepare and run DAIC-WOZ when its data is available. The total is 240 cross-validation conditions with one model seed. Prepare, execute, and collect each dataset separately. Each bundle has its own study identity and fold assignments. W&B records `data.dataset_id` and a separate study group. Compare fine-tuning with ICL within each dataset and report dataset results separately. This procedure does not train on one dataset and evaluate on the other.

## Run independent groups

Create a host-specific YAML file outside version control, for example:

```yaml
output_root: /private/daic-runs
host_label: gpu-node-a
device: cuda:0
wandb:
  mode: online
```

W&B modes are `online`, `offline`, and `disabled`. Use the existing W&B setup in the README for online tracking. Select a physical GPU with `CUDA_VISIBLE_DEVICES`; the process uses its visible `cuda:0`.

Run one group on a system:

```bash
uv run --no-sync daic-foundation-tab run-tabular-study \
  --study /private/tabular-study --model tabpfn35 --modality audio \
  --host-config /private/host.yaml
```

Choose `tabpfn35` or `kumo_medium`, and `audio`, `visual`, or `audio_visual`. Repeat for the other groups, on this system or other systems. A group runs all its tasks, methods, folds, and configured seeds. Assign each group to only one system: its lock is local to that system. The runner keeps a status file and logs, stops a group on failure, and skips verified completed conditions when restarted with the same paths. Each successful fine-tuning condition retains its selected checkpoint on the execution host.

W&B records condition identifiers, settings, aggregate predictive and runtime metrics, training history, and hardware and provenance metadata. The new study disables console capture and does not explicitly upload participant-level predictions. Local bundles, logs, predictions, and checkpoints remain sensitive. Runtime comparisons should be interpreted by GPU type and memory capacity.

## Collect and compare

Copy the private result directories to a system with the same study bundle and source. Checkpoint files may be omitted from the copies; retain them on the execution hosts. Then collect:

```bash
uv run --no-sync daic-foundation-tab collect-tabular-study \
  --study /private/tabular-study --runs /private/results-node-a /private/results-node-b \
  --output /private/tabular-report
```

The collector verifies identities, result files, dependencies, and model checkpoints. It reports missing or conflicting conditions and fails by default when the schedule is incomplete. Add `--allow-partial` for a provisional report. `coverage.json` and `coverage.csv` show condition status. The report includes per-condition metrics and hardware-grouped runtime summaries. Complete five-fold groups also get fold summaries and pooled out-of-fold predictions: one prediction per training participant. Paired FT-minus-ICL comparisons use matching participants and folds. Bootstrap intervals resample the saved predictions within folds; they do not include uncertainty from retraining the models.

## Add model seeds

Create a new YAML file that extends the default and sets, for example, `tabular_study.model_seeds: [42, 0, 1, 2, 3]`. Prepare a new, empty destination with `--reuse-data /private/tabular-study`. The feature contents and folds are reused, and existing condition IDs remain valid. The expanded schedule has 600 cross-validation conditions. Run each group again using the expanded bundle and the same output roots; verified existing conditions are skipped. Collect against the expanded bundle. Do not change other study settings when using `--reuse-data`.

## Optional final stage

After the cross-validation protocol is fixed, pass `--stage final` to the group runner and collector. This stage trains or supplies context from official training participants and evaluates the official development and test participants. Test predictions can be generated without test labels. To score the test split, set `test_ground_truth` in the host YAML to the local absolute path of the same authorized label file on every host. The labels are read only after test predictions. Keep test results for final evaluation, not model selection.

## Optional GPU smoke tests

The opt-in smoke tests use synthetic participants and a one-epoch fine-tuning profile. They cover each model, task, and method through the new fold runner. They require CUDA, the locked environment, and accessible model checkpoints; they may download weights if the model libraries need them. They do not start the research campaign or upload to W&B.

```bash
DAIC_RUN_TABULAR_CV_SMOKE=1 uv run --no-sync pytest -q tests/test_tabular_study_gpu_smoke.py -m real_model
```
