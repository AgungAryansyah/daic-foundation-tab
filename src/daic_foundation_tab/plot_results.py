"""Render the saved DAIC aggregate results as PNG and PDF figures."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from time import monotonic

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

DATASETS = {"daic_woz": "DAIC-WOZ", "edaic": "E-DAIC"}
MODELS = {
    "tabiclv2": "TabICLv2",
    "tabpfn35": "TabPFN-3.5",
    "kumo_medium": "Kumo Medium",
    "qwen35_0_8b": "Qwen3.5-0.8B",
    "qwen35_2b": "Qwen3.5-2B",
    "qwen35_9b": "Qwen3.5-9B",
}
INPUTS = {"audio": "Audio", "visual": "Face", "audio_visual": "Audio + face"}
METRICS = {"classification": "roc_auc", "regression": "mae"}
TASKS = {"classification": "Classification", "regression": "Score prediction"}
AXIS_LABELS = {"roc_auc": "ROC-AUC  ↑", "mae": "MAE in PHQ-8 points  ↓"}
STYLES = {
    "frozen_icl": ("Frozen ICL", "#0072B2", "o"),
    "fine_tuning": ("Fine-tuning", "#D55E00", "s"),
    "lora": ("LoRA", "#D55E00", "s"),
    "dora": ("DoRA", "#9B508A", "D"),
}
TABULAR_MODELS = ("tabpfn35", "kumo_medium")
QWEN_MODELS = ("qwen35_0_8b", "qwen35_2b", "qwen35_9b")
INK = "#16283C"
MUTED = "#465B72"
GRID = "#DDE5EF"
PALE = "#E9EEF5"
TEAL = "#008269"

FIGURE_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titleweight": "bold",
    "xtick.color": MUTED,
    "ytick.color": INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}


def title(fig, heading, subtitle):
    fig.text(0.035, 0.958, heading, fontsize=25, weight="bold", va="top")
    fig.text(0.036, 0.897, subtitle, fontsize=11.5, color=MUTED, va="top")


def footer(fig, text):
    fig.text(0.036, 0.025, text, fontsize=9.5, color=MUTED, va="bottom")


def legend(fig, handles, y=0.838):
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.032, y),
        ncol=len(handles),
        borderaxespad=0,
        handletextpad=0.45,
        columnspacing=1.55,
    )


def method_handle(method):
    label, color, marker = STYLES[method]
    return Line2D([], [], label=label, color=color, marker=marker, linestyle="none", markersize=8)


def yrows(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.45, -0.55)
    ax.tick_params(axis="y", pad=8, labelsize=10)
    for i in range(len(labels)):
        ax.axhspan(i - 0.48, i + 0.48, color=PALE if i % 2 == 0 else "white", alpha=0.50, zorder=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)


def finished_folds(runs):
    unique = {}
    for run in sorted(runs, key=lambda r: (r.get("created_at") or "", r["id"])):
        if run["stage"] != "cv" or run["state"] != "finished":
            continue
        key = tuple(run[k] for k in ("dataset", "model", "input", "task", "method", "fold", "seed"))
        if any(value is None for value in key):
            raise ValueError(f"Incomplete fold identity for {run['id']}")
        unique[key] = run
    return list(unique.values())


def coverage_figure(folds, date_label):
    groups = [
        ("Tabular · DAIC-WOZ", 120, "daic_woz", TABULAR_MODELS, ("frozen_icl", "fine_tuning")),
        ("Tabular · E-DAIC", 120, "edaic", TABULAR_MODELS, ("frozen_icl", "fine_tuning")),
        ("Transcript · DAIC-WOZ", 40, "daic_woz", QWEN_MODELS[:2], ("frozen_icl", "lora")),
        ("Transcript · E-DAIC", 40, "edaic", QWEN_MODELS[:2], ("frozen_icl", "lora")),
        ("DoRA 0.8B / 2B · DAIC-WOZ", 20, "daic_woz", QWEN_MODELS[:2], ("dora",)),
        ("DoRA 0.8B / 2B · E-DAIC", 20, "edaic", QWEN_MODELS[:2], ("dora",)),
        ("9B ICL / DoRA · DAIC-WOZ", 20, "daic_woz", QWEN_MODELS[2:], ("frozen_icl", "dora")),
        ("9B ICL / DoRA · E-DAIC", 20, "edaic", QWEN_MODELS[2:], ("frozen_icl", "dora")),
    ]
    counts = [
        sum(
            r["dataset"] == dataset and r["model"] in models and r["method"] in methods
            for r in folds
        )
        for _, _, dataset, models, methods in groups
    ]
    if any(done > group[1] for done, group in zip(counts, groups)) or sum(counts) != len(folds):
        raise ValueError("Finished fold counts do not match the study matrix")
    total = sum(group[1] for group in groups)
    fig = Figure(figsize=(11.6, 5.5))
    ax = fig.subplots()
    title(
        fig,
        f"{sum(counts)} of {total} cross-validation conditions complete",
        f"DAIC research progress · {date_label}",
    )
    fig.subplots_adjust(left=0.32, right=0.93, top=0.80, bottom=0.17)
    yrows(ax, [group[0] for group in groups])
    ax.set_xlim(0, 1.17)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.barh(range(len(groups)), [1] * len(groups), color=PALE, height=0.48)
    ax.barh(
        range(len(groups)),
        [done / group[1] for done, group in zip(counts, groups)],
        color=[TEAL if done == group[1] else "#0072B2" for done, group in zip(counts, groups)],
        height=0.48,
    )
    for i, (done, group) in enumerate(zip(counts, groups)):
        ax.text(1.02, i, f"{done}/{group[1]}", va="center", fontsize=10, weight="bold")
    footer(
        fig, "Source · W&B run summaries. Each completed condition represents one held-out fold."
    )
    return fig


def fold_panel(ax, folds, labels, groups, dataset, task, methods, limits):
    metric = METRICS[task]
    lookup = defaultdict(list)
    for run in folds:
        if run["dataset"] == dataset and run["task"] == task:
            value = run["metrics"].get(f"cv/{metric}")
            if value is not None:
                lookup[run["model"], run["input"], run["method"]].append((run["fold"], value))
    yrows(ax, labels)
    for row, (model, input_) in enumerate(groups):
        for i, method in enumerate(methods):
            observations = sorted(lookup[model, input_, method])
            if not observations:
                continue
            _, color, marker = STYLES[method]
            y = row + (i - (len(methods) - 1) / 2) * 0.24
            values = [value for _, value in observations]
            jitter = [(fold - 3) * 0.035 for fold, _ in observations]
            ax.scatter(
                values,
                [y + offset for offset in jitter],
                color=color,
                s=17,
                alpha=0.35,
                edgecolors="none",
                zorder=2,
            )
            average = mean(values)
            complete = len(values) == 5
            ax.scatter(
                average,
                y,
                marker=marker,
                facecolor=color if complete else "white",
                edgecolor="white" if complete else color,
                linewidth=1.3,
                s=79,
                zorder=4,
            )
            if not complete:
                ax.annotate(
                    f"{len(values)}/5",
                    (average, y),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    color=color,
                    fontsize=8,
                    weight="bold",
                )
    ax.set_xlim(*limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_title(f"{DATASETS[dataset]} · {TASKS[task]}", loc="left", fontsize=12, pad=11)
    ax.set_xlabel(AXIS_LABELS[metric], labelpad=8)
    if task == "classification":
        ax.axvline(0.5, color="#8393A6", linewidth=1, linestyle="--", zorder=1)


def fold_figure(folds, transcript):
    if transcript:
        methods = ("frozen_icl", "lora", "dora")
        models = [model for model in QWEN_MODELS if any(r["model"] == model for r in folds)]
        groups = [(model, "transcript") for model in models]
        labels = [MODELS[model] for model in models]
        heading = "Transcript cross-validation"
        note = (
            "Open markers show partial groups. The label gives the number of folds with a metric."
        )
        source = "Source · W&B run summaries. The dashed line marks ROC-AUC = 0.50."
    else:
        methods = ("frozen_icl", "fine_tuning")
        groups = [(model, input_) for model in TABULAR_MODELS for input_ in INPUTS]
        labels = [f"{MODELS[model]} · {INPUTS[input_]}" for model, input_ in groups]
        heading = "Tabular cross-validation"
        note = "The dashed line marks ROC-AUC = 0.50."
        source = (
            "Source · W&B run summaries. Input features represent audio, facial behavior, or both."
        )
    selected = [r for r in folds if r["model"] in (QWEN_MODELS if transcript else TABULAR_MODELS)]
    errors = [r["metrics"]["cv/mae"] for r in selected if "cv/mae" in r["metrics"]]
    upper_mae = max(1, max(errors, default=1) * 1.12)
    fig = Figure(figsize=(13.2, 7.6 if transcript else 9.2))
    axes = fig.subplots(2, 2)
    title(
        fig, heading, "Small dots show individual folds. Large markers show their arithmetic mean."
    )
    fig.subplots_adjust(
        left=0.225 if not transcript else 0.165,
        right=0.968,
        top=0.773,
        bottom=0.13,
        hspace=0.61,
        wspace=0.77 if not transcript else 0.51,
    )
    legend(fig, [method_handle(method) for method in methods])
    for dataset, row in zip(DATASETS, axes):
        for task, ax in zip(TASKS, row):
            fold_panel(
                ax,
                selected,
                labels,
                groups,
                dataset,
                task,
                methods,
                (0, 1) if task == "classification" else (0, upper_mae),
            )
    footer(fig, f"{note}\n{source}")
    return fig


def paired_entries(reports, comparisons):
    tabular = defaultdict(list)
    transcript = defaultdict(list)
    for report in reports:
        dataset = report["dataset"]
        if dataset not in DATASETS:
            continue
        for row in report["paired"]:
            task = row["task"]
            if row["metric"] != METRICS[task]:
                continue
            model = row["family"]
            if model in TABULAR_MODELS:
                label = f"{DATASETS[dataset]} · {MODELS[model]} · {INPUTS[row['modality']]}"
                tabular[task].append(
                    (label, "fine_tuning", row, "frozen_icl", "fine_tuning", "delta_ft_minus_icl")
                )
            elif model in QWEN_MODELS and row.get("delta_ft_minus_icl") not in ("", None):
                label = f"{DATASETS[dataset]} · {MODELS[model]} · LoRA − ICL"
                transcript[task].append(
                    (label, "lora", row, "frozen_icl", "fine_tuning", "delta_ft_minus_icl")
                )
    for comparison in comparisons:
        dataset = comparison["dataset"]
        for row in comparison["rows"]:
            task = row["task"]
            model = row["candidate_family"]
            left = STYLES[row["reference_adaptation"]][0].replace("Frozen ", "")
            right = STYLES[row["candidate_adaptation"]][0]
            label = f"{DATASETS[dataset]} · {MODELS[model]} · {right} − {left}"
            transcript[task].append(
                (
                    label,
                    row["candidate_adaptation"],
                    row,
                    "reference",
                    "candidate",
                    "delta_candidate_minus_reference",
                )
            )
    return tabular, transcript


def paired_figure(entries, transcript):
    title_prefix = "Transcript" if transcript else "DAIC-WOZ tabular"
    row_count = max(map(len, entries.values()))
    fig = Figure(figsize=(13.5, max(7.4, 2.5 + row_count * 0.78)))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.1, 1.48, 1.48, 1.0),
        left=0.28 if transcript else 0.34,
        right=0.97,
        bottom=0.16,
        top=0.77,
        wspace=0.17,
    )
    title(
        fig,
        f"{title_prefix} · paired changes on held-out participants",
        "Each point is a pooled score difference. The line shows its 95% paired bootstrap interval.",
    )
    for start, task in zip((0, 2), TASKS):
        ax = fig.add_subplot(grid[0, start : start + 2])
        rows = sorted(entries[task], key=lambda item: item[0])
        if start == 0:
            yrows(ax, [row[0] for row in rows])
        else:
            yrows(ax, [""] * len(rows))
        if [row[0] for row in rows] != sorted(row[0] for row in entries["classification"]):
            raise ValueError("Paired comparisons must cover the same groups for both tasks")
        ax.axvline(0, color="#6F8094", linewidth=1.2, linestyle="--")
        span = max(abs(float(r[key])) for _, _, r, *_ in rows for key in ("ci_lower", "ci_upper"))
        ax.set_xlim(-span * 1.10, span * 1.10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        metric = METRICS[task]
        ax.set_title(
            "Classification · ROC-AUC" if task == "classification" else "Score prediction · MAE",
            loc="left",
            fontsize=12,
            pad=13,
        )
        ax.set_xlabel(
            "Candidate − reference   ·   higher favors candidate"
            if metric == "roc_auc"
            else "Candidate − reference   ·   lower favors candidate",
            fontsize=9.5,
            labelpad=10,
        )
        for i, (_, method, row, before, after, delta) in enumerate(rows):
            value = float(row[delta])
            low, high = float(row["ci_lower"]), float(row["ci_upper"])
            _, color, marker = STYLES[method]
            ax.hlines(i, low, high, color=color, linewidth=2.2, zorder=3)
            ax.scatter(
                value,
                i,
                color=color,
                marker=marker,
                s=51,
                zorder=4,
                edgecolors="white",
                linewidth=0.8,
            )
            digits = 3 if metric == "roc_auc" else 2
            ax.text(
                0.97,
                i + 0.33,
                f"{float(row[before]):.{digits}f} → {float(row[after]):.{digits}f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                color=MUTED,
                fontsize=8,
            )
    source = (
        "Source · Collected transcript reports and matched DoRA comparisons. Intervals describe saved seed-42 predictions."
        if transcript
        else "Source · Collected tabular reports. Fine-tuning is the candidate in every row."
    )
    footer(fig, source)
    return fig


def compute_figure(folds):
    groups = defaultdict(list)
    for r in folds:
        runtime = r["metrics"].get("_runtime")
        gpu = r.get("gpu")
        if runtime and runtime > 0 and gpu and r["model"] in QWEN_MODELS:
            groups[r["dataset"], r["model"], r["task"], gpu, r["method"]].append(runtime / 60)
    fig = Figure(figsize=(13.1, 6.15))
    axes = fig.subplots(1, 2)
    title(
        fig,
        "Transcript runtime by method",
        "Median elapsed minutes recorded by W&B. The hardware label identifies the GPU.",
    )
    fig.subplots_adjust(left=0.22, right=0.96, top=0.74, bottom=0.18, wspace=0.71)
    methods = ("frozen_icl", "lora", "dora")
    legend(fig, [method_handle(method) for method in methods])
    for ax, task in zip(axes, TASKS):
        keys = sorted({(dataset, model, gpu) for dataset, model, t, gpu, _ in groups if t == task})
        labels = [
            f"{DATASETS[ds]} · {MODELS[model]}\n{gpu.removeprefix('NVIDIA GeForce ')}"
            for ds, model, gpu in keys
        ]
        yrows(ax, labels)
        maximum = max(
            (median(values) for key, values in groups.items() if key[2] == task), default=1
        )
        ax.set_xlim(0, maximum * 1.30)
        for row, (dataset, model, gpu) in enumerate(keys):
            for i, method in enumerate(methods):
                values = groups.get((dataset, model, task, gpu, method), [])
                if not values:
                    continue
                y = row + (i - 1) * 0.22
                _, color, marker = STYLES[method]
                value = median(values)
                ax.hlines(y, 0, value, color=color, alpha=0.4, linewidth=2)
                ax.scatter(
                    value,
                    y,
                    marker=marker,
                    facecolor=color if len(values) == 5 else "white",
                    edgecolor=color,
                    s=45,
                    zorder=3,
                )
                ax.annotate(
                    f"{value:.1f}" + (f"  ({len(values)}/5)" if len(values) < 5 else ""),
                    (value, y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8.5,
                    color=color,
                )
        ax.set_title(TASKS[task], loc="left", fontsize=12, pad=11)
        ax.set_xlabel("Minutes per fold", labelpad=8)
    footer(
        fig,
        "Source · W&B runtime and GPU metadata. Labels on open markers give coverage for partial groups.",
    )
    return fig


def historic_value(run, split, metric):
    for prefix in ("development", "dev") if split == "Development" else ("test",):
        value = run["metrics"].get(f"{prefix}/{metric}")
        if value is not None:
            return value
    return None


def history_figure(runs, pilots):
    selected = [
        r for r in runs if r["stage"] in {"standalone", "final"} and r["state"] == "finished"
    ]
    selected.extend(pilots)
    selected = [
        r
        for r in selected
        if r["model"] in MODELS
        and r["dataset"] in DATASETS
        and r["input"] in INPUTS
        and r["task"] in TASKS
        and r["method"] in STYLES
    ]
    groups = {
        task: sorted(
            {
                (r["dataset"], r["model"], r["input"])
                for r in selected
                if r["task"] == task
                and any(
                    historic_value(r, split, METRICS[task]) is not None
                    for split in ("Development", "Test")
                )
            }
        )
        for task in TASKS
    }
    total_rows = sum(map(len, groups.values()))
    fig = Figure(figsize=(13.3, max(9.8, total_rows * 0.37 + 4)))
    axes = fig.subplots(2, 2, height_ratios=[len(groups[t]) for t in TASKS])
    title(
        fig,
        "Pilot and standalone experiments",
        "Individual run scores from the earlier development and test protocols.",
    )
    fig.subplots_adjust(left=0.33, right=0.96, top=0.78, bottom=0.10, wspace=0.33, hspace=0.34)
    filled = Line2D([], [], marker="o", color=MUTED, linestyle="none", label="Full run")
    opened = Line2D(
        [],
        [],
        marker="o",
        color=MUTED,
        markerfacecolor="white",
        linestyle="none",
        label="Smoke test",
    )
    partial = Line2D([], [], marker="x", color=MUTED, linestyle="none", label="Partial output")
    legend(
        fig, [method_handle("frozen_icl"), method_handle("fine_tuning"), filled, opened, partial]
    )
    for task, panels in zip(TASKS, axes):
        metric = METRICS[task]
        keys = groups[task]
        labels = [
            f"{DATASETS[ds]} · {MODELS[model]} · {INPUTS[input_]}" for ds, model, input_ in keys
        ]
        values = [
            v
            for r in selected
            if r["task"] == task
            for split in ("Development", "Test")
            if (v := historic_value(r, split, metric)) is not None
        ]
        for split, ax in zip(("Development", "Test"), panels):
            yrows(ax, labels if split == "Development" else [""] * len(labels))
            ax.set_title(f"{TASKS[task]} · {split}", loc="left", fontsize=12, pad=8)
            for row, key in enumerate(keys):
                candidates = [
                    r
                    for r in selected
                    if r["task"] == task
                    and (r["dataset"], r["model"], r["input"]) == key
                    and historic_value(r, split, metric) is not None
                ]
                for j, r in enumerate(candidates):
                    offset = (j - (len(candidates) - 1) / 2) * min(0.105, 0.65 / len(candidates))
                    _, color, marker = STYLES[r["method"]]
                    if r["state"].startswith("partial"):
                        ax.scatter(
                            historic_value(r, split, metric),
                            row + offset,
                            marker="x",
                            color=color,
                            linewidth=1.2,
                            s=40,
                            zorder=3,
                        )
                        continue
                    ax.scatter(
                        historic_value(r, split, metric),
                        row + offset,
                        marker=marker,
                        facecolor="white" if r.get("profile") == "smoke" else color,
                        edgecolor=color,
                        linewidth=1.2,
                        s=38,
                        zorder=3,
                    )
            ax.set_xlim(0, 1 if metric == "roc_auc" else max(values, default=1) * 1.10)
            ax.set_xlabel(AXIS_LABELS[metric])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    footer(
        fig,
        "Sources · Local September pilot outputs and W&B. Each marker represents a run under its original protocol.",
    )
    return fig


def render_results(data_path: Path, output: Path) -> None:
    started = monotonic()
    print(f"Reading aggregate results from {data_path}", flush=True)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    runs = data["wandb"]["runs"]
    folds = finished_folds(runs)
    reports = data["collected"]["reports"]
    comparisons = data["collected"]["comparisons"]
    tabular, transcript = paired_entries(reports, comparisons)
    stamp = datetime.fromisoformat(data["wandb"]["captured_at"])
    date_label = stamp.strftime("%d %B %Y")
    tasks = [
        ("01_coverage", coverage_figure, (folds, date_label)),
        ("02_tabular_cv", fold_figure, (folds, False)),
        ("03_transcript_cv", fold_figure, (folds, True)),
        ("04_tabular_paired", paired_figure, (tabular, False)),
        ("05_transcript_paired", paired_figure, (transcript, True)),
        ("06_transcript_runtime", compute_figure, (folds,)),
        ("07_pilot_and_standalone", history_figure, (runs, data["pilot"]["runs"])),
    ]
    output.mkdir(parents=True, exist_ok=True)
    total = len(tasks)
    print(f"Rendering 0/{total} figures (0%) | ETA unavailable", flush=True)
    for count, (name, render, inputs) in enumerate(tasks, 1):
        print(f"Rendering {name}", flush=True)
        with rc_context(FIGURE_STYLE):
            fig = render(*inputs)
            fig.savefig(output / f"{name}.png", dpi=300)
            fig.savefig(output / f"{name}.pdf")
        elapsed = monotonic() - started
        eta = f"{elapsed / count * (total - count):.1f}s estimated" if count > 1 else "unavailable"
        print(
            f"Exported {count}/{total} figures ({count / total:.0%}) | {elapsed:.1f}s elapsed | ETA {eta}",
            flush=True,
        )
    print(f"Figures saved in {output.resolve()}", flush=True)
