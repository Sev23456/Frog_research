#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark_frogs import (
    EVENT_FIELDS,
    RUN_METRIC_FIELDS,
    SUMMARY_METRICS,
    TIME_SERIES_FIELDS,
    aggregate_metric_rows,
    aggregate_seed_rows,
    write_csv,
    write_gzip_csv,
    write_json,
)


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_gzip_csv(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_outputs(input_dirs: Iterable[Path], output_dir: Path) -> Path:
    input_dirs = [Path(path) for path in input_dirs]
    output_dir.mkdir(parents=True, exist_ok=False)

    merged_run_metrics: List[Dict[str, Any]] = []
    merged_time_series: List[Dict[str, Any]] = []
    merged_events: List[Dict[str, Any]] = []
    metadatas: List[Dict[str, Any]] = []

    for directory in input_dirs:
        metadatas.append(load_json(directory / "metadata.json"))
        merged_run_metrics.extend(load_csv(directory / "run_metrics.csv"))
        merged_time_series.extend(load_gzip_csv(directory / "time_series.csv.gz"))
        merged_events.extend(load_gzip_csv(directory / "event_log.csv.gz"))

    merged_metadata = {
        "merged_from": [str(directory) for directory in input_dirs],
        "steps": sorted({metadata["steps"] for metadata in metadatas}),
        "spawn_seeds": sorted({seed for metadata in metadatas for seed in metadata.get("spawn_seeds", [])}),
        "repeats": sorted({metadata["repeats"] for metadata in metadatas}),
        "modes": sorted({mode for metadata in metadatas for mode in metadata.get("modes", [])}),
        "architectures": sorted({arch for metadata in metadatas for arch in metadata.get("architectures", [])}),
        "sample_interval": sorted({metadata["sample_interval"] for metadata in metadatas}),
        "competence_catches": sorted({metadata["competence_catches"] for metadata in metadatas}),
        "notes": ["Merged benchmark output"],
    }
    write_json(output_dir / "metadata.json", merged_metadata)

    aggregate_rows = aggregate_metric_rows(merged_run_metrics)
    seed_rows = aggregate_seed_rows(merged_run_metrics)

    write_csv(output_dir / "run_metrics.csv", merged_run_metrics, RUN_METRIC_FIELDS)
    write_json(output_dir / "run_metrics.json", merged_run_metrics)
    write_gzip_csv(output_dir / "time_series.csv.gz", merged_time_series, TIME_SERIES_FIELDS)
    write_gzip_csv(output_dir / "event_log.csv.gz", merged_events, EVENT_FIELDS)

    aggregate_fieldnames = ["arch", "mode", "runs"]
    for metric in SUMMARY_METRICS:
        aggregate_fieldnames.extend(
            [
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_std",
                f"{metric}_ci95",
            ]
        )
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows, aggregate_fieldnames)
    write_json(output_dir / "aggregate_metrics.json", aggregate_rows)

    seed_fieldnames = ["arch", "mode", "spawn_seed", "repeats"]
    for metric in SUMMARY_METRICS:
        seed_fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "seed_summary.csv", seed_rows, seed_fieldnames)
    write_json(output_dir / "seed_summary.json", seed_rows)

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple frog benchmark shards into one output directory.")
    parser.add_argument("output_dir", type=Path, help="Directory where the merged benchmark should be written.")
    parser.add_argument("input_dirs", nargs="+", type=Path, help="Benchmark shard directories to merge.")
    parser.add_argument("--report", action="store_true", help="Generate a markdown report after merging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_dir = merge_outputs(args.input_dirs, args.output_dir)
    if args.report:
        from benchmark_frogs_report import generate_report

        generate_report(merged_dir)
    print(f"Merged benchmark saved to: {merged_dir}")


if __name__ == "__main__":
    main()
