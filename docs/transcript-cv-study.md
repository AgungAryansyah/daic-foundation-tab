# Matched transcript cross-validation

The transcript campaign compares Qwen3.5-2B frozen in-context learning (ICL) with LoRA fine-tuning. E-DAIC and DAIC-WOZ use separate studies. Each dataset has 20 CV conditions: two tasks × two methods × five folds × seed 42. Classification predicts PHQ-8 below or above the fixed cutoff of 10. Score prediction generates a finite number from 0 to 24.

Use `configs/studies/edaic_transcript_cv.yaml` or `configs/studies/daic_woz_transcript_cv.yaml`. Both retain full participant speech, two seeded ICL demonstrations, BF16, and the pinned Qwen3.5-2B checkpoint. Thinking and quantization remain disabled. LoRA uses five epochs, rank 8, alpha 16, dropout 0, all-linear targets, learning rate 0.0002, one interview per microbatch, and gradient accumulation across four microbatches. Non-reentrant gradient checkpointing reduces activation memory. The base model stays frozen.

## Inputs and preparation

Supply authorized transcripts for every participant in the prepared tabular cohort, including development and test participants. The default pattern is `data/{participant_id}_TRANSCRIPT.csv`, relative to `data.root`. The parser expects tab-separated `speaker` and `value` columns. Set the root, pattern, delimiter, encoding, or column names in a local configuration that extends the dataset preset. E-DAIC numeric feature availability does not establish transcript availability. The implementation does not create transcripts from recordings.

Preparation imports exact participants, labels, task-specific folds, internal selection groups, and final assignments from the tabular bundle. It never regenerates folds or excludes participants with missing transcripts. Frozen ICL selects two demonstrations with the existing seeded selector from each fit pool. Selection is fixed for each task, fold, and seed. This selector does not balance classes. LoRA uses the complete fit pool and the imported selection pool. Outer evaluation participants enter neither pool.

Historical tabular bundles receive read-only content and provenance checks. New transcript source has a separate identity. Do not register this extension as a compatible repair of the old tabular campaign. Completed tabular bundles and results remain unchanged.

These commands run on the remote GPU host. Use native commands there, without RTK. Install the locked optional text backend first:

```bash
uv sync --extra text --group dev --locked
uv run --no-sync daic-foundation-tab prepare-transcript-study \
  --config configs/studies/edaic_transcript_cv.yaml \
  --tabular-study /private/edaic-tabular-study \
  --output /private/edaic-transcript-study
```

Preparation loads the tokenizer without model weights. It parses every transcript and audits all CV and final prompts, including instructions, demonstrations, and answer space. The initial limit is 32,768 tokens. Overflow fails preparation and records the condition in private `audit.json`. It never truncates speech or removes demonstrations. A changed context setting requires a new prepared study and another preflight.

The portable private bundle contains cleaned transcripts, source hashes, participants, folds, the parent bundle, resolved settings, the schedule, and the audit. Directory access starts at mode 0700. Preserve private access when transferring bundles and results. Every host must use matching bundle contents, source files, and locked dependencies. Output and cache paths can differ. Original transcript files are not required after preparation. Changed original sources require new preparation. Changes inside the bundle invalidate it.

## Preflight and independent groups

Copy `configs/examples/transcript-host.yaml` outside version control. Set the output path, cache path, and host label. CPU libraries default to one thread. Workers limit allocator arenas to two. Select a physical GPU with `CUDA_VISIBLE_DEVICES`; execution uses its visible `cuda:0`.

```bash
uv run --no-sync daic-foundation-tab preflight-transcript-study \
  --study /private/edaic-transcript-study \
  --host-config /private/host.yaml --task classification
```

Preflight loads the full pinned checkpoint. It exercises the longest applicable prompts for frozen prediction and two disposable LoRA updates with validation. The second update measures training with optimizer state already allocated. Probe outputs remain separate from research results and W&B. The initial check requires 2 GiB available RAM and 8 GiB free GPU memory. A successful profile records observed process RAM and reserved GPU memory. Execution requires available capacity of at least 120% of these measured peaks. RAM availability respects the container limit and reclaimable inactive file pages. Other jobs can still change capacity after the check.

The profile identifies the study, audit, source, task, stage, host, GPU, CPU thread setting, and container limit. A changed identity requires another preflight. A failed probe leaves a status record without a runnable profile. Capacity failures stop immediately without queuing or changing model settings.

Each dataset has two independent groups: `classification` and `regression`. Each group has ten CV conditions. Run groups on separate systems or sequentially on one system. Assign each group to one system only, because its lock is local.

```bash
uv run --no-sync daic-foundation-tab run-transcript-study \
  --study /private/edaic-transcript-study \
  --host-config /private/host.yaml --task classification
```

Repeat preflight and execution with `--task regression` for PHQ-8 prediction. Use a persistent terminal or detached execution for long work. For example, after preflight passes:

```bash
nohup uv run --no-sync daic-foundation-tab run-transcript-study \
  --study /private/edaic-transcript-study \
  --host-config /private/host.yaml --task classification \
  > /private/classification-launcher.log 2>&1 < /dev/null &
```

Outputs reside under `<output_root>/<study_id>/cv/qwen35_2b-<task>/`. `status.json` records the active condition, counts, outputs, and failures. Logs report phases, work units, elapsed time, throughput, and available ETA estimates. Silent phases emit activity every 30 seconds. RAM monitoring continues during model operations.

The group stops on failure. Repeat its command to skip verified completed conditions. Interrupted LoRA training resumes from its latest complete checkpoint when input and configuration identities match. Without a complete checkpoint, training restarts explicitly. Incomplete prediction work restarts. A completed run that lacks only its result marker is verified and sealed without repeating inference. The selected adapter remains separate from recovery checkpoints.

## Tracking and collection

W&B records dataset and study identities, parent study, condition, task, method, fold, seed, checkpoint, code revision, host settings, training history, predictive metrics, and runtime. Classification includes discrimination and calibration metrics. Score prediction includes MAE, RMSE, R², and binary metrics derived at PHQ-8 >=10. Normalized answer likelihoods are not assumed to be calibrated probabilities.

Runtime separates preparation, loading, prompt validation/tokenization, training, and prediction. It also records participant throughput, input prompt tokens per second, prompt-length summaries, peak process RAM, the container limit, and peak GPU memory. Input prompt throughput is not generated-token throughput. It covers the complete prediction procedure.

Campaign tracking disables console capture and code upload. Transcripts, participant identities, raw answers, demonstrations, and participant predictions remain private local artifacts. Do not upload bundles or prediction directories to W&B.

```bash
uv run --no-sync daic-foundation-tab collect-transcript-study \
  --study /private/edaic-transcript-study \
  --runs /private/results-host-a /private/results-host-b \
  --output /private/edaic-transcript-report
```

Collection checks identities, artifact hashes, model settings, selected adapter hashes, checkpoint revisions, assignments, prediction targets, metrics, and dependency agreement. Keep adapters on execution hosts. Copies for collection may omit adapter directories but must retain recorded hashes. Duplicate successful conditions, changed artifacts, and incomplete schedules fail collection. `--allow-partial` produces a provisional report with incomplete coverage.

Reports contain coverage, condition metrics, hardware-grouped runtime summaries, fold summaries, and pooled out-of-fold predictions. Each training participant appears once per task and method. `cv_paired_oof.csv` reports fine-tuning (LoRA) minus frozen ICL, with 2,000 paired bootstrap resamples within folds. Intervals describe uncertainty conditional on saved predictions. They do not include variation from retraining.

For cross-input CV comparisons, add `--tabular-report /private/edaic-tabular-report`. It must identify the completed parent study. Participants, targets, and folds must match. `cross_input_comparisons.csv` reports Qwen transcript values minus tabular values for corresponding tasks, methods, and seeds. Both model and input differ. These comparisons do not isolate a modality effect.

Interpret aggregate files only when `coverage.json` reports `completed`. Compare runtimes within matching hardware and CPU thread profiles. Prepare and collect DAIC-WOZ separately with its corresponding preset and parent bundle.

## Final stage and tests

The optional final stage contains four conditions per dataset. It uses imported official training assignments for fit and selection, then predicts development and test participants. It does not run automatically. Repeat preflight, execution, and collection with `--stage final`. To score test predictions, set the same authorized label file through the host YAML's absolute `test_ground_truth` path. Test labels are read after prediction. Without labels, predictions remain available and test metrics are omitted.

Local validation uses RTK:

```bash
rtk proxy uv run --extra text --no-sync pytest -q -m 'not real_model'
rtk proxy env DAIC_QWEN_CV_SMOKE=1 uv run --extra text --no-sync pytest -q tests/test_transcript_study_gpu_smoke.py
```

Opt-in CUDA tests use a tiny random native Qwen model with hybrid attention and synthetic transcripts. They check fold execution and LoRA checkpointing without downloading 2B weights or starting a research campaign. They do not establish accuracy or full-model memory requirements. Full-checkpoint preflight remains mandatory on every execution host.
