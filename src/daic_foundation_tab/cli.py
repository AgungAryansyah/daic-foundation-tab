from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from daic_foundation_tab.config import load_config
from daic_foundation_tab.data import build_or_load_dataset
from daic_foundation_tab.data.discovery import discover_sources
from daic_foundation_tab.data.labels import load_official_labels
from daic_foundation_tab.data.validation import validate_dataset, validate_regression_dataset
from daic_foundation_tab.runner import run_experiment
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase
from daic_foundation_tab.tracking.test_evaluations import (
    collect_regression_test_evaluations,
    collect_test_evaluations,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daic-foundation-tab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "build-features", "validate", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
        if command == "run":
            subparser.add_argument("--resume", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument("--output", type=Path)
    paired = subparsers.add_parser("compare-paired")
    paired.add_argument("runs", nargs=2, type=Path)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--split", choices=["dev", "test"], default="dev")
    paired.add_argument("--iterations", type=int, default=1000)
    prepare = subparsers.add_parser("prepare-participants")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    study = subparsers.add_parser("prepare-matched-study")
    study.add_argument("--config", required=True, type=Path)
    study.add_argument("--output", required=True, type=Path)
    run_study = subparsers.add_parser("run-matched-study")
    run_study.add_argument("--study", required=True, type=Path)
    prepare_tabular = subparsers.add_parser("prepare-tabular-study")
    prepare_tabular.add_argument("--config", required=True, type=Path)
    prepare_tabular.add_argument("--output", required=True, type=Path)
    prepare_tabular.add_argument("--reuse-data", type=Path)
    revision = subparsers.add_parser("register-tabular-revision")
    revision.add_argument("--study", required=True, type=Path)
    revision.add_argument("--baseline-commit", required=True)
    revision.add_argument("--allowed-files", required=True, nargs="+")
    revision.add_argument("--reason", required=True)
    run_tabular = subparsers.add_parser("run-tabular-study")
    run_tabular.add_argument("--study", required=True, type=Path)
    run_tabular.add_argument("--model", required=True, choices=["tabpfn35", "kumo_medium"])
    run_tabular.add_argument("--modality", required=True, choices=["audio", "visual", "audio_visual"])
    run_tabular.add_argument("--host-config", required=True, type=Path)
    run_tabular.add_argument("--stage", default="cv", choices=["cv", "final"])
    collect_tabular = subparsers.add_parser("collect-tabular-study")
    collect_tabular.add_argument("--study", required=True, type=Path)
    collect_tabular.add_argument("--runs", required=True, nargs="+", type=Path)
    collect_tabular.add_argument("--output", required=True, type=Path)
    collect_tabular.add_argument("--stage", default="cv", choices=["cv", "final"])
    collect_tabular.add_argument("--allow-partial", action="store_true")
    collect = subparsers.add_parser("collect-test-results")
    collect.add_argument("--output-root", type=Path, default=Path("outputs"))
    collect_regression = subparsers.add_parser("collect-regression-test-results")
    collect_regression.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser


def _output_path(config: dict, prefix: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path(config["project"]["output_root"]).resolve() / f"{timestamp}_{prefix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _inspect(config: dict) -> Path:
    if config.get("input", {}).get("kind") == "transcript":
        from daic_foundation_tab.data.participants import participant_index

        output = _output_path(config, "transcript_inspection")
        logger = configure_run_logger(output / "run.log", "INFO")
        with phase(logger, "preparing participants") as started:
            index = participant_index(config, lambda n, total: log_progress(
                logger, "preparing participants", n, total, started))
        index.to_csv(output / "participant_inventory.csv", index=False)
        print(output)
        return output
    labels = load_official_labels(config["data"])
    discovery = discover_sources(labels.table, config["data"])
    output = _output_path(config, "dataset_inspection")
    (output / "dataset_inventory.json").write_text(
        json.dumps({**discovery.inventory, "label_audit": labels.audit}, indent=2), encoding="utf-8"
    )
    print(output)
    return output


def _build_features(config: dict) -> None:
    if config.get("input", {}).get("kind") == "transcript":
        raise ValueError("build-features requires tabular input; use validate for transcripts")
    output = _output_path(config, "feature_build")
    logger = configure_run_logger(output / "run.log", config["logging"]["level"])
    with phase(logger, "preparing features") as started:
        dataset = build_or_load_dataset(
            config,
            progress_callback=lambda completed, total: log_progress(
                logger, "preparing features", completed, total, started
            ),
        )
    print(f"cache_key={dataset.cache_key}")
    print(f"participants={len(dataset.table)}")
    print(f"features={len(dataset.manifest)}")


def _validate(config: dict) -> Path:
    if config.get("input", {}).get("kind") == "transcript":
        from daic_foundation_tab.data.transcripts import load_transcripts

        output = _output_path(config, "transcript_validation")
        logger = configure_run_logger(output / "run.log", config.get("logging", {}).get("level", "INFO"))
        with phase(logger, "preparing transcripts") as started:
            dataset = load_transcripts(config, lambda n, total: log_progress(
                logger, "preparing transcripts", n, total, started))
        (output / "participant_manifest.json").write_text(json.dumps(dataset.manifest, indent=2), encoding="utf-8")
        (output / "validation_report.json").write_text(json.dumps({
            "participants": len(dataset.texts), "cache_key": dataset.cache_key,
            "speaker_policy": config["data"]["transcripts"].get("speaker_policy", "participant_only"),
        }, indent=2), encoding="utf-8")
        print(output)
        return output
    dataset = build_or_load_dataset(config)
    validate = (
        validate_regression_dataset
        if config["experiment"]["task"] == "regression"
        else validate_dataset
    )
    report = validate(
        dataset,
        config["experiment"]["feature_set"],
        float(config["data"].get("max_exclusion_fraction", 0.05)),
    )
    output = _output_path(config, "validation")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            *( {"column": column, "reason": "all_nan"} for column in report["dropped_all_nan"] ),
            *( {"column": column, "reason": "constant"} for column in report["dropped_constant"] ),
        ]
    ).to_csv(output / "dropped_features.csv", index=False)
    print(output)
    return output


def _markdown_table(table: pd.DataFrame) -> str:
    columns = [str(column) for column in table.columns]

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        f"| {' | '.join(columns)} |",
        f"| {' | '.join('---' for _ in columns)} |",
    ]
    lines.extend(f"| {' | '.join(cell(value) for value in row)} |" for row in table.itertuples(index=False, name=None))
    return "\n".join(lines)


def _compare(runs: list[Path], output: Path | None) -> Path:
    rows = []
    for run in runs:
        from daic_foundation_tab.tracking.lifecycle import completed_run

        if not completed_run(run):
            raise ValueError(f"Run is incomplete: {run}")
        metrics_path = run / "metrics_dev.json"
        if not metrics_path.is_file():
            raise ValueError(f"Missing development metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"run": run.name, **{key: value for key, value in metrics.items() if isinstance(value, float)}})
    destination = output or Path("outputs") / f"comparison_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    destination.mkdir(parents=True, exist_ok=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(destination / "comparison.csv", index=False)
    (destination / "comparison.md").write_text(_markdown_table(comparison) + "\n", encoding="utf-8")
    print(destination)
    return destination


def main() -> None:
    args = _parser().parse_args()
    if args.command == "register-tabular-revision":
        from daic_foundation_tab.tabular_study.revisions import register_revision

        record = register_revision(args.study, args.baseline_commit, args.allowed_files, args.reason)
        print(f"Registered execution revision {record['revision_id']} at {record['git_commit']}")
        return
    if args.command == "collect-tabular-study":
        from daic_foundation_tab.tabular_study.collection import collect_study

        print(collect_study(args.study, args.runs, args.output, stage=args.stage,
                            allow_partial=args.allow_partial))
        return
    if args.command == "prepare-tabular-study":
        from daic_foundation_tab.tabular_study.bundle import prepare_bundle

        print(prepare_bundle(args.config, args.output, reuse=args.reuse_data))
        return
    if args.command == "run-tabular-study":
        from daic_foundation_tab.tabular_study.groups import run_group

        print(run_group(args.study, args.model, args.modality, args.host_config, stage=args.stage))
        return
    if args.command == "run-matched-study":
        from daic_foundation_tab.study_runner import run_study

        print(run_study(args.study))
        return
    if args.command == "prepare-matched-study":
        from daic_foundation_tab.matched_study import prepare_study

        print(prepare_study(args.config, args.output))
        return
    if args.command == "compare-paired":
        from daic_foundation_tab.evaluation.paired import paired_comparison

        args.output.mkdir(parents=True, exist_ok=True)
        logger = configure_run_logger(args.output / "run.log", "INFO")
        with phase(logger, "bootstrapping paired differences") as started:
            table = paired_comparison(*args.runs, split=args.split, iterations=args.iterations,
                progress=lambda n, total: log_progress(logger, "bootstrapping paired differences", n, total, started))
        table.to_csv(args.output / "comparison.csv", index=False)
        (args.output / "comparison.md").write_text(_markdown_table(table) + "\n", encoding="utf-8")
        print(args.output)
        return
    if args.command == "compare":
        _compare(args.runs, args.output)
        return
    if args.command == "collect-test-results":
        print(collect_test_evaluations(args.output_root))
        return
    if args.command == "collect-regression-test-results":
        print(collect_regression_test_evaluations(args.output_root))
        return
    config = load_config(args.config)
    if args.command == "prepare-participants":
        from daic_foundation_tab.data.participants import prepare_manifest

        args.output.parent.mkdir(parents=True, exist_ok=True)
        logger = configure_run_logger(args.output.with_suffix(".log"), "INFO")
        with phase(logger, "preparing participants") as started:
            manifest = prepare_manifest(config, lambda n, total: log_progress(
                logger, "preparing participants", n, total, started))
        args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(args.output)
    elif args.command == "inspect":
        _inspect(config)
    elif args.command == "build-features":
        _build_features(config)
    elif args.command == "validate":
        _validate(config)
    elif args.command == "run":
        if args.resume is not None:
            if config.get("input", {}).get("kind") != "transcript":
                raise ValueError("Explicit checkpoint resume is currently supported for text runs")
            config["_resume_directory"] = str(args.resume)
        print(run_experiment(config))


if __name__ == "__main__":
    main()
