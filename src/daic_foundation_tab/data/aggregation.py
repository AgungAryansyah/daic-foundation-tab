from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class AggregationError(ValueError):
    pass


COVAREP_COLUMNS = (
    "timestamp",
    "F0",
    "VUV",
    "NAQ",
    "QOQ",
    "H1H2",
    "PSP",
    "MDQ",
    "peakSlope",
    "Rd",
    "Rd_conf",
    *(f"MCEP_{index}" for index in range(25)),
    *(f"HMPDM_{index}" for index in range(25)),
    *(f"HMPDD_{index}" for index in range(13)),
)

if len(COVAREP_COLUMNS) != 74:
    raise RuntimeError("The COVAREP schema must contain 74 columns")

METADATA_COLUMNS = {
    "frame",
    "frame_number",
    "timestamp",
    "confidence",
    "success",
    "detection_success",
}


@dataclass(frozen=True)
class AggregationResult:
    values: dict[str, float]
    manifest: list[dict[str, str]]
    source_quality: dict[str, Any]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")


def _read_source(path: Path, parser: str) -> pd.DataFrame:
    if parser == "covarep":
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] != len(COVAREP_COLUMNS):
            raise AggregationError(
                f"Expected {len(COVAREP_COLUMNS)} COVAREP columns in {path}, found {frame.shape[1]}"
            )
        frame.columns = COVAREP_COLUMNS
        return frame
    if parser == "openface":
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [str(column).strip() for column in frame.columns]
        if "success" in frame.columns:
            frame = frame.loc[pd.to_numeric(frame["success"], errors="coerce") == 1].copy()
        return frame
    raise AggregationError(f"Unsupported source parser: {parser}")


def _statistics(series: pd.Series, statistics: list[str], ddof: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for statistic in statistics:
        if statistic == "mean":
            result[statistic] = float(series.mean())
        elif statistic == "std":
            result[statistic] = float(series.std(ddof=ddof))
        elif statistic == "median":
            result[statistic] = float(series.median())
        elif statistic == "min":
            result[statistic] = float(series.min())
        elif statistic == "max":
            result[statistic] = float(series.max())
        else:
            raise AggregationError(f"Unsupported aggregation statistic: {statistic}")
    return result


def aggregate_source(
    path: Path,
    source_group: str,
    source_config: dict[str, Any],
    aggregation_config: dict[str, Any],
) -> AggregationResult:
    frame = _read_source(path, source_config["parser"])
    metadata = METADATA_COLUMNS.intersection(frame.columns)
    values: dict[str, float] = {}
    manifest: list[dict[str, str]] = []
    missingness: dict[str, float] = {}
    discarded: list[str] = []

    for column in frame.columns:
        if column in metadata:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() == 0 and frame[column].notna().sum() > 0:
            discarded.append(column)
            continue
        missingness[column] = float(series.isna().mean()) if len(series) else 1.0
        for statistic, value in _statistics(
            series,
            list(aggregation_config["statistics"]),
            int(aggregation_config.get("std_ddof", 0)),
        ).items():
            feature_name = f"{source_group}_{_safe_name(column)}_{statistic}"
            values[feature_name] = value
            manifest.append(
                {
                    "column": feature_name,
                    "modality": source_config["modality"],
                    "source_group": source_group,
                    "source_feature": column,
                    "aggregation": statistic,
                }
            )

    return AggregationResult(
        values=values,
        manifest=manifest,
        source_quality={
            "rows": len(frame),
            "missingness": missingness,
            "discarded_nonnumeric_columns": discarded,
        },
    )
