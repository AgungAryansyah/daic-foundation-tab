# Experiment figures

The plot command reads an aggregate results snapshot and exports seven figures as 300 DPI PNGs.
Run it from the project root.

```bash
uv run --locked daic-foundation-tab plot-results
```

The command reads `outputs/results_snapshot.json` and writes the figures directly to `outputs`.
Use `--data` to select a different snapshot. Use `--output` to select a different destination.

The snapshot contains `wandb` run summaries, `collected` report tables, and `pilot` metrics.
It keeps aggregate scores and source identifiers. The plotting command reads only this snapshot.

| File stem | Contents |
|---|---|
| `01_coverage` | Completed cross-validation conditions by dataset and campaign |
| `02_tabular_cv` | Tabular fold scores and means by model, feature set, and method |
| `03_transcript_cv` | Transcript fold scores and means by Qwen size and method |
| `04_tabular_paired` | Pooled DAIC-WOZ tabular scores and paired intervals |
| `05_transcript_paired` | Pooled Qwen scores and paired intervals from collected reports |
| `06_transcript_runtime` | Median W&B elapsed time per fold by method and GPU |
| `07_pilot_and_standalone` | Earlier development and test scores from pilot and standalone runs |

ROC-AUC measures class ranking. Mean absolute error (MAE) measures score error in PHQ-8 points.
A fold mean averages the individual fold scores. A pooled score uses the combined held-out
predictions. Paired intervals describe the saved predictions at seed 42.
