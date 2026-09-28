from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import shutil
import time
from pathlib import Path

EXPECTED_COUNTS = {"train": 163, "dev": 56, "test": 56}
FEATURE_FILES = (
    "OpenSMILE2.3.0_egemaps.csv",
    "OpenFace2.1.0_Pose_gaze_AUs.csv",
)


class PreparationError(ValueError):
    pass


def _load_splits(
    source: Path, expected_counts: dict[str, int]
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    splits: dict[str, list[tuple[str, str]]] = {}
    binary_disagreements: dict[str, int] = {}
    seen: set[str] = set()
    for split, expected in expected_counts.items():
        path = source / "labels" / f"{split}_split.csv"
        if not path.is_file():
            raise PreparationError(f"Missing E-DAIC label file: {path}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"Participant_ID", "PHQ_Score", "PHQ_Binary"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise PreparationError(f"Missing label columns in {path}: {sorted(required)}")
            rows: list[tuple[str, str]] = []
            disagreements = 0
            for row in reader:
                participant_id = row["Participant_ID"].strip().removesuffix(".0")
                if not participant_id.isdigit() or participant_id in seen:
                    raise PreparationError(
                        f"Invalid or duplicate participant ID in {path}: {participant_id}"
                    )
                try:
                    score = float(row["PHQ_Score"])
                    binary = int(row["PHQ_Binary"])
                except (TypeError, ValueError) as exc:
                    raise PreparationError(
                        f"Invalid PHQ label for {participant_id} in {path}"
                    ) from exc
                if not 0 <= score <= 24 or binary not in (0, 1):
                    raise PreparationError(f"Invalid PHQ label for {participant_id} in {path}")
                disagreements += int(binary != int(score >= 10))
                rows.append((participant_id, row["PHQ_Score"].strip()))
                seen.add(participant_id)
        if len(rows) != expected:
            raise PreparationError(
                f"Expected {expected} {split} participants, found {len(rows)} in {path}"
            )
        splits[split] = rows
        binary_disagreements[split] = disagreements
    return splits, binary_disagreements


def _check_header(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise PreparationError(f"Missing or empty feature file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = stream.readline()
    try:
        dialect = csv.Sniffer().sniff(header, delimiters=",;\t")
        columns = next(csv.reader([header], dialect))
    except (csv.Error, StopIteration) as exc:
        raise PreparationError(f"Invalid CSV header: {path}") from exc
    if len(columns) < 2:
        raise PreparationError(f"Expected multiple CSV columns in {path}")


def _place(source: Path, destination: Path) -> None:
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if source.samefile(destination) or (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return
        raise PreparationError(f"Prepared file differs from source: {destination}")
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
            raise
        shutil.copy2(source, destination)


def prepare_edaic(
    source: Path, output: Path, expected_counts: dict[str, int] | None = None
) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise PreparationError("Source and output directories must differ")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "preparation_status.json"
    started = time.monotonic()
    status: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "phase": "checking labels",
        "completed_participants": 0,
        "total_participants": 0,
        "percentage": 0,
        "elapsed_seconds": 0,
        "eta": "unavailable",
    }

    def report(phase: str, completed: int, total: int, *, log: bool = True) -> None:
        status.update(
            phase=phase,
            completed_participants=completed,
            total_participants=total,
            percentage=round(100 * completed / total, 1) if total else 0,
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        temporary_status = status_path.with_suffix(".json.tmp")
        temporary_status.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary_status.replace(status_path)
        if log:
            print(
                f"{phase}: {completed}/{total} participants ({status['percentage']}%), "
                f"elapsed {status['elapsed_seconds']}s, ETA unavailable",
                flush=True,
            )

    report("checking labels", 0, 0)
    try:
        splits, binary_disagreements = _load_splits(source, expected_counts or EXPECTED_COUNTS)
        participants = [participant_id for rows in splits.values() for participant_id, _ in rows]
        total = len(participants)
        report("checking feature files", 0, total)
        for index, participant_id in enumerate(participants, 1):
            for feature in FEATURE_FILES:
                path = (
                    source
                    / "data"
                    / f"{participant_id}_P"
                    / "features"
                    / f"{participant_id}_{feature}"
                )
                _check_header(path)
            report("checking feature files", index, total, log=index % 10 == 0 or index == total)

        labels_output = output / "original_labels"
        data_output = output / "data"
        labels_output.mkdir(exist_ok=True)
        data_output.mkdir(exist_ok=True)
        for split, rows in splits.items():
            path = labels_output / f"{split}_split.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Participant_ID", "PHQ8_Score"))
                writer.writerows(rows)

        report("preparing feature files", 0, total)
        for index, participant_id in enumerate(participants, 1):
            for feature in FEATURE_FILES:
                name = f"{participant_id}_{feature}"
                path = source / "data" / f"{participant_id}_P" / "features" / name
                _place(path, data_output / name)
            report("preparing feature files", index, total, log=index % 10 == 0 or index == total)

        status["split_counts"] = {split: len(ids) for split, ids in splits.items()}
        status["supplied_binary_disagreements"] = binary_disagreements
        status["feature_files_per_participant"] = len(FEATURE_FILES)
        report("complete", total, total)
        return status
    except Exception as exc:
        status["error"] = str(exc)
        report("failed", int(status["completed_participants"]), int(status["total_participants"]))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare E-DAIC CSVs for TabICLv2 experiments")
    parser.add_argument(
        "--source", type=Path, required=True, help="Directory created by the E-DAIC download script"
    )
    parser.add_argument("--output", type=Path, default=Path("data/edaic"))
    args = parser.parse_args()
    prepare_edaic(args.source, args.output)


if __name__ == "__main__":
    main()
