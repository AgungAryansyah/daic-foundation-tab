# Matched DAIC-WOZ adaptation study

The study compares frozen in-context learning (ICL) and fine-tuning within each model family.
ICL predicts from labeled examples without changing model weights.
LoRA changes a small set of adapter parameters in Qwen.
The tabular adapters use their existing native weight-training methods.

The preset is `configs/studies/daic_woz_matched.yaml`.
It defines 16 conditions with seed 42:

| Models | Input | Methods | Tasks |
| --- | --- | --- | --- |
| TabPFN-3.5, TabICLv2, Kumo Medium | Audio and visual numeric features | Frozen ICL, fine-tuning | Binary depression, PHQ-8 score |
| Qwen3.5-2B | Participant-only interview transcripts | Frozen ICL, LoRA | Binary depression, generated PHQ-8 score |

## Shared evaluation

Preparation selects participants with all required numeric files and a nonempty transcript file.
It parses the transcripts before it writes the completed campaign definition.
Hidden metadata files do not qualify as transcripts.
The shared cohort must satisfy `data.max_exclusion_fraction` separately for each official split.
The default limit is 5%.

Preparation saves one participant manifest per task.
A manifest records participant identities, targets, official splits, and internal training roles.
Every model and method for that task uses the same manifest.
All eligible official training participants enter either the fit pool or the internal selection pool.
Classification stratifies this division by class. Regression uses a seeded random division.
The default selection fraction is 20%.
The development and test participants remain separate from both pools.

Frozen tabular models use the entire fit pool as context.
Qwen ICL uses two fixed demonstrations selected from that pool.
Fine-tuning uses the fit pool and selects a checkpoint with the internal selection pool.
Frozen ICL does not use selection labels.
The comparison matches participants and evaluation splits. Each model retains its own training recipe and context size.
Cross-family differences also reflect the different input modalities.

Binary evaluation uses the fixed probability threshold of 0.5.
All classifiers select fine-tuned checkpoints by internal ROC-AUC.
TabPFN and Kumo regression select by internal MSE. TabICL and Qwen select by internal MAE.
The native model profiles retain their checkpoint pins, hyperparameters, and ensemble settings.
Qwen uses the pinned 2B checkpoint, BF16, five LoRA epochs, and no thinking mode.
These settings define an initial experiment. They do not establish optimal hyperparameters.

The campaign disables repeated holdouts and each runner's bootstrap procedure.
The summary reports point differences from one seed, without confidence intervals.
Use the existing `compare-paired` command for participant bootstrap intervals on saved predictions.
Those intervals do not include variation from retraining.

## Prepare and execute

Run these commands from the repository root on the Linux GPU server.
Complete the CUDA, model-access, and W&B setup in the README first.
Provide the licensed numeric files and transcripts under `data/`.
Add `data/data/<participant_id>_TRANSCRIPT.csv` for each participant.
The default transcript format is tab-separated, with `speaker` and `value` columns.

1. Install the text backend.

   ```bash
   rtk proxy uv sync --extra text --group dev --locked
   ```

2. Prepare the participant manifests and 16 resolved configurations.

   ```bash
   rtk proxy uv run --no-sync daic-foundation-tab prepare-matched-study \
     --config configs/studies/daic_woz_matched.yaml \
     --output outputs/daic_woz_matched
   ```

   This command validates the cohort and transcript parsing. It does not load model weights or start training.
   It does not check full-model GPU memory or tokenize every prompt.
   Each Qwen run checks complete prompt lengths before training or prediction.
   The preset allows 32,768 tokens, including demonstrations and the answer.
   A run rejects overflow without truncating transcripts or removing demonstrations.
   Full-model memory requirements still need a check on the execution GPU.

3. Execute the prepared campaign.

   ```bash
   rtk proxy uv run --no-sync daic-foundation-tab run-matched-study \
     --study outputs/daic_woz_matched
   ```

The launcher runs conditions sequentially on `cuda:0` in separate processes.
Logs identify the active condition, completed conditions, elapsed time, and phase progress.
Thirty-second phase messages report activity when model operations remain silent.
The launcher reports no ETA because condition durations differ.
Each model retains its own detailed progress log.

The campaign stops on the first failure.
Repeat the execution command to preserve verified completed conditions and retry the failed condition in a fresh run directory.
The campaign does not automatically resume a partially trained adapter.
The existing single-run `--resume` option remains available for that separate operation.
A file lock prevents concurrent launchers for the same study directory.

Do not edit prepared configurations or participant manifests.
Change the source preset and use a new study directory for a different experiment.
Reuse checks require identical prepared settings, participant manifests, and Git commit.
They also check input paths, sizes, and modification times before and after each condition.
These file metadata checks do not detect content edits that preserve size and modification time.

## Measurements and outputs

W&B is online by default. Credentials load from `.env` or the existing environment.
Set `tracking.wandb.mode: offline` in the source preset to retain tracking records locally.
Both input paths use the same W&B group for a task, seed, and participant manifest.
Tags identify the model, input, task, adaptation method, and matched campaign.
W&B receives aggregate metadata and metrics. Transcripts, participant IDs, and predictions remain local.

| W&B metric prefix | Measurements |
| --- | --- |
| `development/`, `test/` | Classification F1, ROC-AUC, PR-AUC, accuracy, Brier score, calibration error, and other existing task metrics |
| `development/`, `test/` | Regression MAE, RMSE, R², and classes derived at PHQ-8 >=10 |
| `fine_tuning/` | Epoch, training loss, internal validation performance, and checkpoint selection measurements |
| `runtime/` | Fit seconds, prediction seconds, total seconds, peak allocated GPU memory, and peak reserved GPU memory |

Qwen also records optimizer steps, learning rate, and the number of training examples observed per logged epoch.
After an interrupted epoch, its resumed loss mean covers only the remaining examples observed in that attempt.
Qwen's normalized answer likelihoods require calibration evaluation. They are not calibrated clinical probabilities.
Malformed generated scores fail the run instead of receiving an invented score.

Test metrics appear only when `data.test_ground_truth` exists.
Test predictions remain available without test labels.
Runtime totals include each runner's preparation and loading work.
Compare runtime on the same GPU with equivalent cache conditions.
The values do not represent isolated GPU kernel timings.

The study directory contains:

- `campaign.json`: prepared condition identities and task manifests.
- `participants_<task>.json`: the shared local participant assignments.
- `<condition>.yaml`: one resolved configuration per condition.
- `status.json`: active condition, completion counts, failures, output paths, and W&B links.
- `runs/<condition>/<run>/`: predictions, metrics, model metadata, runtime, and logs.
- `metrics.csv`: one row per condition after all 16 conditions pass completion checks.
- `adaptation_differences.csv`: fine-tuning minus frozen ICL for each available scalar metric.

Positive differences favor fine-tuning for metrics such as F1 and ROC-AUC.
Negative differences favor fine-tuning for errors, elapsed time, and memory use.
Use aggregate reports only when `status.json` reports `completed`.
