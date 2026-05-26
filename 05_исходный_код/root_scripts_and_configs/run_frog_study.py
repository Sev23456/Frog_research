#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from benchmark_frogs import ARCHITECTURES, MODES, run_benchmark_suite
from benchmark_frogs_merge import merge_outputs
from benchmark_frogs_report import generate_report


def as_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def metric_cell(value, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}{suffix}"


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_study_dir(base_dir: Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path("benchmark_results") / "studies"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = root / f"frog_study_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=False)
    return study_dir


def summarize_shard(output_dir: Path) -> List[str]:
    aggregate_rows = load_csv(output_dir / "aggregate_metrics.csv")
    if not aggregate_rows:
        return [f"## {output_dir.name}", "", "- No aggregate rows found.", ""]
    row = aggregate_rows[0]
    return [
        f"## {row['arch']} / {row['mode']}",
        "",
        f"- Catch rate: `{metric_cell(as_float(row.get('catch_rate_per_minute_mean')))} / min`",
        f"- Capture success: `{metric_cell(as_float(row.get('capture_success_mean')), suffix='')}`",
        f"- Learning AUC: `{metric_cell(as_float(row.get('learning_auc_mean')))} `",
        f"- Flies / energy: `{metric_cell(as_float(row.get('flies_per_energy_spent_mean')))} `",
        f"- Low/high energy catch rate: `{metric_cell(as_float(row.get('catch_rate_low_energy_mean')))} / {metric_cell(as_float(row.get('catch_rate_high_energy_mean')))} `",
        f"- Low/high energy strike rate: `{metric_cell(as_float(row.get('strike_rate_low_energy_mean')))} / {metric_cell(as_float(row.get('strike_rate_high_energy_mean')))} `",
        f"- Ignored visible high energy: `{metric_cell(as_float(row.get('visible_but_ignored_ratio_high_energy_mean')))} `",
        f"- Compute per catch: `{metric_cell(as_float(row.get('compute_per_catch_s_mean')), precision=3, suffix=' s')}`",
        "",
    ]


def summarize_merged(output_dir: Path) -> List[str]:
    aggregate_rows = load_csv(output_dir / "aggregate_metrics.csv")
    lines: List[str] = ["# Final Study Summary", ""]
    for mode in ("adult", "developmental"):
        selected = [row for row in aggregate_rows if row.get("mode") == mode]
        if not selected:
            continue
        catch_winner = max(selected, key=lambda row: as_float(row.get("catch_rate_per_minute_mean")) or float("-inf"))
        energy_winner = max(selected, key=lambda row: as_float(row.get("flies_per_energy_spent_mean")) or float("-inf"))
        compute_winner = min(selected, key=lambda row: as_float(row.get("ms_per_step_mean")) or float("inf"))
        lines.extend(
            [
                f"## {mode.capitalize()}",
                "",
                f"- Best catch rate: `{catch_winner['arch']}` at `{metric_cell(as_float(catch_winner.get('catch_rate_per_minute_mean')))} / min`",
                f"- Best energy efficiency: `{energy_winner['arch']}` at `{metric_cell(as_float(energy_winner.get('flies_per_energy_spent_mean')))} `",
                f"- Cheapest compute: `{compute_winner['arch']}` at `{metric_cell(as_float(compute_winner.get('ms_per_step_mean')), precision=4, suffix=' ms/step')}`",
                "",
            ]
        )
    return lines


def write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full frog architecture study in sharded mode.")
    parser.add_argument("--steps", type=int, default=100_000, help="Simulation steps per run.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of spawn seeds, starting from 0.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per seed.")
    parser.add_argument("--sample-interval", type=int, default=500, help="Sampling interval for time-series logging.")
    parser.add_argument("--competence-catches", type=int, default=10, help="Catch threshold used as competence proxy.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers inside each shard.")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(ARCHITECTURES),
        choices=list(ARCHITECTURES),
        help="Architectures to include in the study.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(MODES.keys()),
        choices=sorted(MODES.keys()),
        help="Modes to include in the study.",
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Optional parent directory for the study.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = make_study_dir(args.output_root)
    shards_root = study_dir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "steps": args.steps,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "sample_interval": args.sample_interval,
        "competence_catches": args.competence_catches,
        "workers": args.workers,
        "architectures": list(args.architectures),
        "modes": list(args.modes),
        "notes": [
            "Current variants are preserved untouched; new variants live in copied packages.",
            "Each architecture x mode combination is benchmarked as a separate shard with its own intermediate report.",
            "Final merged output aggregates all shards into one study-level report.",
        ],
    }
    with (study_dir / "study_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    shard_dirs: List[Path] = []
    intermediate_lines: List[str] = [
        "# Intermediate Study Notes",
        "",
        f"- Steps per run: `{args.steps}`",
        f"- Seeds: `{args.seeds}`",
        f"- Repeats: `{args.repeats}`",
        f"- Workers per shard: `{args.workers}`",
        "",
    ]

    shard_index = 0
    for mode in args.modes:
        for arch in args.architectures:
            shard_index += 1
            logical_parent = shards_root / f"{shard_index:02d}_{arch}_{mode}"
            logical_parent.mkdir(parents=True, exist_ok=True)
            print(f"\n=== Shard {shard_index}: {arch} / {mode} ===")
            output_dir = run_benchmark_suite(
                steps=args.steps,
                spawn_seeds=tuple(range(args.seeds)),
                repeats=args.repeats,
                modes=(mode,),
                architectures=(arch,),
                sample_interval=args.sample_interval,
                competence_catches=args.competence_catches,
                output_dir=logical_parent,
                workers=max(1, int(args.workers)),
            )
            generate_report(output_dir)
            shard_dirs.append(output_dir)
            intermediate_lines.extend(summarize_shard(output_dir))
            write_lines(study_dir / "intermediate_conclusions.md", intermediate_lines)

    merged_dir = study_dir / "merged"
    merge_outputs(shard_dirs, merged_dir)
    generate_report(merged_dir)
    write_lines(study_dir / "final_conclusions.md", summarize_merged(merged_dir))
    print(f"\nStudy completed: {study_dir}")


if __name__ == "__main__":
    main()
