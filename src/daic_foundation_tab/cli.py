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
from daic_foundation_tab.data.validation import validate_dataset
from daic_foundation_tab.runner import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daic-foundation-tab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "build-features", "validate", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument("--output", type=Path)
    return parser


def _output_path(config: dict, prefix: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path(config["project"]["output_root"]).resolve() / f"{timestamp}_{prefix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _inspect(config: dict) -> Path:
    labels = load_official_labels(config["data"])
    discovery = discover_sources(labels.table, config["data"])
    output = _output_path(config, "dataset_inspection")
    (output / "dataset_inventory.json").write_text(
        json.dumps({**discovery.inventory, "label_audit": labels.audit}, indent=2), encoding="utf-8"
    )
    print(output)
    return output


def _build_features(config: dict) -> None:
    dataset = build_or_load_dataset(config)
    print(f"cache_key={dataset.cache_key}")
    print(f"participants={len(dataset.table)}")
    print(f"features={len(dataset.manifest)}")


def _validate(config: dict) -> Path:
    dataset = build_or_load_dataset(config)
    report = validate_dataset(dataset, config["experiment"]["feature_set"])
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


def _compare(runs: list[Path], output: Path | None) -> Path:
    rows = []
    for run in runs:
        metrics_path = run / "metrics_dev.json"
        if not metrics_path.is_file():
            raise ValueError(f"Missing development metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"run": run.name, **{key: value for key, value in metrics.items() if isinstance(value, float)}})
    destination = output or Path("outputs") / f"comparison_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    destination.mkdir(parents=True, exist_ok=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(destination / "comparison.csv", index=False)
    (destination / "comparison.md").write_text(comparison.to_markdown(index=False) + "\n", encoding="utf-8")
    print(destination)
    return destination


def main() -> None:
    args = _parser().parse_args()
    if args.command == "compare":
        _compare(args.runs, args.output)
        return
    config = load_config(args.config)
    if args.command == "inspect":
        _inspect(config)
    elif args.command == "build-features":
        _build_features(config)
    elif args.command == "validate":
        _validate(config)
    elif args.command == "run":
        print(run_experiment(config))
