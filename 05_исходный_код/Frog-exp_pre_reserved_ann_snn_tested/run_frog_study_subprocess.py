#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from benchmark_frogs import ARCHITECTURES, MODES
from benchmark_frogs_merge import merge_outputs
from benchmark_frogs_report import generate_report
from run_frog_study import make_study_dir, summarize_merged, summarize_shard, write_lines


@dataclass(frozen=True)
class ShardSpec:
    index: int
    arch: str
    mode: str
    study_dir: Path
    steps: int
    spawn_seeds: Tuple[int, ...]
    repeat_indices: Tuple[int, ...]
    sample_interval: int
    competence_catches: int

    @property
    def shard_name(self) -> str:
        seed_part = compact_range_label("s", self.spawn_seeds)
        repeat_part = compact_range_label("r", self.repeat_indices)
        return f"{self.index:03d}_{self.arch}_{self.mode}_{seed_part}_{repeat_part}"

    @property
    def shard_root(self) -> Path:
        return self.study_dir / "shards" / self.shard_name

    @property
    def stdout_log(self) -> Path:
        return self.shard_root / "stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.shard_root / "stderr.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frog study with shard-level subprocess parallelism.")
    parser.add_argument("--steps", type=int, default=100_000, help="Simulation steps per run.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of spawn seeds, starting from 0.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per seed.")
    parser.add_argument("--seed-batch-size", type=int, default=0, help="Split each architecture/mode into seed batches of this size. 0 means no split.")
    parser.add_argument("--repeat-batch-size", type=int, default=0, help="Split each architecture/mode into repeat-index batches of this size. 0 means no split.")
    parser.add_argument("--sample-interval", type=int, default=500, help="Sampling interval for time-series logging.")
    parser.add_argument("--competence-catches", type=int, default=10, help="Catch threshold used as competence proxy.")
    parser.add_argument("--shard-workers", type=int, default=4, help="Number of shard subprocesses to run in parallel.")
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
    parser.add_argument("--resume-study", type=Path, default=None, help="Resume an existing study directory instead of creating a new one.")
    return parser.parse_args()


def write_manifest(path: Path, args: argparse.Namespace) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "steps": args.steps,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "sample_interval": args.sample_interval,
        "competence_catches": args.competence_catches,
        "shard_workers": args.shard_workers,
        "seed_batch_size": args.seed_batch_size,
        "repeat_batch_size": args.repeat_batch_size,
        "architectures": list(args.architectures),
        "modes": list(args.modes),
        "notes": [
            "Shard-level subprocess parallelism avoids the Windows ProcessPool permission issue in this environment.",
            "Each shard runs benchmark_frogs.py with --workers 1 and writes its own stdout/stderr logs.",
            "All previous benchmark outputs remain untouched for reproducibility.",
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def compact_range_label(prefix: str, values: Sequence[int]) -> str:
    values = tuple(int(value) for value in values)
    if not values:
        return f"{prefix}none"
    if len(values) == 1:
        return f"{prefix}{values[0]}"
    if values == tuple(range(values[0], values[-1] + 1)):
        return f"{prefix}{values[0]}-{values[-1]}"
    return f"{prefix}{'-'.join(str(value) for value in values)}"


def chunked(values: Sequence[int], batch_size: int) -> List[Tuple[int, ...]]:
    values = tuple(int(value) for value in values)
    if batch_size <= 0 or batch_size >= len(values):
        return [values]
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def latest_completed_output(spec: ShardSpec) -> Path | None:
    benchmark_dirs = sorted(spec.shard_root.glob("frog_benchmark_*"))
    for output_dir in reversed(benchmark_dirs):
        if (output_dir / "aggregate_metrics.csv").exists() and (output_dir / "report.md").exists():
            return output_dir
    return None


def build_specs(args: argparse.Namespace, study_dir: Path) -> List[ShardSpec]:
    specs: List[ShardSpec] = []
    index = 1
    seed_batches = chunked(tuple(range(args.seeds)), int(args.seed_batch_size))
    repeat_batches = chunked(tuple(range(args.repeats)), int(args.repeat_batch_size))
    for mode in args.modes:
        for arch in args.architectures:
            for seed_batch in seed_batches:
                for repeat_batch in repeat_batches:
                    specs.append(
                        ShardSpec(
                            index=index,
                            arch=arch,
                            mode=mode,
                            study_dir=study_dir,
                            steps=args.steps,
                            spawn_seeds=seed_batch,
                            repeat_indices=repeat_batch,
                            sample_interval=args.sample_interval,
                            competence_catches=args.competence_catches,
                        )
                )
                    index += 1
    return specs


def run_shard(spec: ShardSpec) -> Tuple[ShardSpec, Path, bool]:
    spec.shard_root.mkdir(parents=True, exist_ok=True)
    existing_output = latest_completed_output(spec)
    if existing_output is not None:
        return spec, existing_output, False
    command = [
        sys.executable,
        "-u",
        "benchmark_frogs.py",
        "--steps",
        str(spec.steps),
        "--spawn-seeds",
        *[str(seed) for seed in spec.spawn_seeds],
        "--repeat-indices",
        *[str(repeat) for repeat in spec.repeat_indices],
        "--sample-interval",
        str(spec.sample_interval),
        "--competence-catches",
        str(spec.competence_catches),
        "--workers",
        "1",
        "--architectures",
        spec.arch,
        "--modes",
        spec.mode,
        "--output-dir",
        str(spec.shard_root),
        "--skip-report",
    ]
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with spec.stdout_log.open("w", encoding="utf-8") as stdout_handle, spec.stderr_log.open("w", encoding="utf-8") as stderr_handle:
        subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent),
            check=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=env,
        )
    benchmark_dirs = sorted(spec.shard_root.glob("frog_benchmark_*"))
    if not benchmark_dirs:
        raise RuntimeError(f"No benchmark output found for shard {spec.shard_name}")
    output_dir = benchmark_dirs[-1]
    generate_report(output_dir)
    return spec, output_dir, True


def append_intermediate_notes(path: Path, lines: Iterable[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def main() -> None:
    args = parse_args()
    if args.resume_study is not None:
        study_dir = Path(args.resume_study)
        (study_dir / "shards").mkdir(parents=True, exist_ok=True)
    else:
        study_dir = make_study_dir(args.output_root)
        (study_dir / "shards").mkdir(parents=True, exist_ok=True)
        write_manifest(study_dir / "study_manifest.json", args)
    intermediate_path = study_dir / "intermediate_conclusions.md"
    if not intermediate_path.exists():
        write_lines(
            intermediate_path,
            [
                "# Intermediate Study Notes",
                "",
                f"- Steps per run: `{args.steps}`",
                f"- Seeds: `{args.seeds}`",
                f"- Repeats: `{args.repeats}`",
                f"- Shard workers: `{args.shard_workers}`",
                f"- Seed batch size: `{args.seed_batch_size or 'all'}`",
                f"- Repeat batch size: `{args.repeat_batch_size or 'all'}`",
                "",
            ],
        )

    specs = build_specs(args, study_dir)
    shard_results: List[Path] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.shard_workers))) as executor:
        future_map = {executor.submit(run_shard, spec): spec for spec in specs}
        for future in concurrent.futures.as_completed(future_map):
            spec = future_map[future]
            completed_spec, output_dir, executed = future.result()
            shard_results.append(output_dir)
            if executed:
                append_intermediate_notes(intermediate_path, summarize_shard(output_dir))
            print(f"Completed shard {completed_spec.shard_name}: {output_dir}")

    shard_results.sort()
    merged_dir = study_dir / "merged"
    if merged_dir.exists():
        merged_dir = study_dir / f"merged_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    merge_outputs(shard_results, merged_dir)
    generate_report(merged_dir)
    write_lines(study_dir / "final_conclusions.md", summarize_merged(merged_dir))
    print(f"Study completed: {study_dir}")


if __name__ == "__main__":
    main()
