#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


ARCH_COLORS = {
    "ANN": "#2f80ed",
    "ANN_FROZEN": "#7fb3ff",
    "SNN": "#27ae60",
    "SNN_FROZEN": "#7bd8a3",
    "BIO": "#e67e22",
    "BIO_DUAL": "#b85c11",
}


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_gzip_csv(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_cell(mean_value: Optional[float], std_value: Optional[float], precision: int = 2, suffix: str = "") -> str:
    if mean_value is None:
        return "n/a"
    if std_value is None:
        return f"{mean_value:.{precision}f}{suffix}"
    return f"{mean_value:.{precision}f} ± {std_value:.{precision}f}{suffix}"


def build_mode_table(rows: Sequence[Dict[str, str]], mode: str) -> List[Dict[str, str]]:
    selected = [row for row in rows if row.get("mode") == mode]
    selected.sort(key=lambda item: item.get("arch", ""))
    table: List[Dict[str, str]] = []
    for row in selected:
        table.append(
            {
                "Architecture": row["arch"],
                "Catch rate / min": metric_cell(as_float(row.get("catch_rate_per_minute_mean")), as_float(row.get("catch_rate_per_minute_std"))),
                "Capture success": metric_cell(as_float(row.get("capture_success_mean")), as_float(row.get("capture_success_std")), suffix=""),
                "First catch (s)": metric_cell(as_float(row.get("time_to_first_catch_s_mean")), as_float(row.get("time_to_first_catch_s_std"))),
                "Learning AUC": metric_cell(as_float(row.get("learning_auc_mean")), as_float(row.get("learning_auc_std"))),
                "Flies / energy": metric_cell(as_float(row.get("flies_per_energy_spent_mean")), as_float(row.get("flies_per_energy_spent_std"))),
                "Move eff. px/catch": metric_cell(as_float(row.get("movement_efficiency_px_per_catch_mean")), as_float(row.get("movement_efficiency_px_per_catch_std"))),
                "ms / step": metric_cell(as_float(row.get("ms_per_step_mean")), as_float(row.get("ms_per_step_std")), precision=4),
            }
        )
    return table


def build_state_dependence_table(rows: Sequence[Dict[str, str]], mode: str) -> List[Dict[str, str]]:
    selected = [row for row in rows if row.get("mode") == mode]
    selected.sort(key=lambda item: item.get("arch", ""))
    table: List[Dict[str, str]] = []
    for row in selected:
        table.append(
            {
                "Architecture": row["arch"],
                "Catch rate low E": metric_cell(as_float(row.get("catch_rate_low_energy_mean")), as_float(row.get("catch_rate_low_energy_std"))),
                "Catch rate high E": metric_cell(as_float(row.get("catch_rate_high_energy_mean")), as_float(row.get("catch_rate_high_energy_std"))),
                "Strike rate low E": metric_cell(as_float(row.get("strike_rate_low_energy_mean")), as_float(row.get("strike_rate_low_energy_std"))),
                "Strike rate high E": metric_cell(as_float(row.get("strike_rate_high_energy_mean")), as_float(row.get("strike_rate_high_energy_std"))),
                "Ignored visible high E": metric_cell(as_float(row.get("visible_but_ignored_ratio_high_energy_mean")), as_float(row.get("visible_but_ignored_ratio_high_energy_std"))),
                "E deficit vs strike r": metric_cell(as_float(row.get("energy_deficit_strike_correlation_mean")), as_float(row.get("energy_deficit_strike_correlation_std"))),
            }
        )
    return table


def markdown_table(rows: Sequence[Dict[str, str]]) -> str:
    if not rows:
        return "_No data._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def determine_winners(rows: Sequence[Dict[str, str]], mode: str) -> Dict[str, str]:
    selected = [row for row in rows if row.get("mode") == mode]
    winners: Dict[str, str] = {}
    if not selected:
        return winners

    def best(metric: str, higher_is_better: bool = True) -> str:
        scored = [(row["arch"], as_float(row.get(metric))) for row in selected]
        scored = [(arch, value) for arch, value in scored if value is not None]
        if not scored:
            return "n/a"
        scored.sort(key=lambda item: item[1], reverse=higher_is_better)
        return scored[0][0]

    winners["catch_rate"] = best("catch_rate_per_minute_mean", True)
    winners["capture_success"] = best("capture_success_mean", True)
    winners["first_catch"] = best("time_to_first_catch_s_mean", False)
    winners["energy_efficiency"] = best("flies_per_energy_spent_mean", True)
    winners["compute"] = best("ms_per_step_mean", False)
    return winners


def plot_metric_bars(rows: Sequence[Dict[str, str]], metric: str, title: str, filename: Path, ylabel: str) -> Optional[Path]:
    if plt is None:
        return None

    modes = sorted({row["mode"] for row in rows})
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * max(1, len(modes)), 4), squeeze=False)
    for idx, mode in enumerate(modes):
        axis = axes[0][idx]
        selected = [row for row in rows if row["mode"] == mode]
        selected.sort(key=lambda item: item["arch"])
        labels = [row["arch"] for row in selected]
        means = [as_float(row.get(f"{metric}_mean")) or 0.0 for row in selected]
        stds = [as_float(row.get(f"{metric}_std")) or 0.0 for row in selected]
        colors = [ARCH_COLORS.get(label, "#888888") for label in labels]
        axis.bar(labels, means, yerr=stds, color=colors, alpha=0.88, capsize=5)
        axis.set_title(mode.capitalize())
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_learning_curves(time_series_rows: Sequence[Dict[str, str]], output_path: Path) -> Optional[Path]:
    if plt is None or not time_series_rows:
        return None

    grouped: Dict[Tuple[str, str], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in time_series_rows:
        arch = row["arch"]
        mode = row["mode"]
        step = int(float(row["step"]))
        value = as_float(row.get("catch_rate_per_minute"))
        if value is not None:
            grouped[(arch, mode)][step].append(value)

    modes = sorted({mode for _, mode in grouped.keys()})
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * max(1, len(modes)), 4), squeeze=False)
    for idx, mode in enumerate(modes):
        axis = axes[0][idx]
        for arch in sorted({arch for arch, current_mode in grouped.keys() if current_mode == mode}):
            step_map = grouped[(arch, mode)]
            steps = sorted(step_map.keys())
            means = [sum(step_map[step]) / len(step_map[step]) for step in steps]
            axis.plot(steps, means, label=arch, color=ARCH_COLORS.get(arch, "#888888"), linewidth=2.2)
        axis.set_title(f"{mode.capitalize()} learning curve")
        axis.set_xlabel("Step")
        axis.set_ylabel("Catch rate / minute")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_development_curves(time_series_rows: Sequence[Dict[str, str]], output_path: Path) -> Optional[Path]:
    if plt is None:
        return None

    developmental_rows = [row for row in time_series_rows if row.get("mode") == "developmental"]
    if not developmental_rows:
        return None

    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in developmental_rows:
        progress = as_float(row.get("juvenile_progress"))
        if progress is None:
            continue
        grouped[row["arch"]][int(float(row["step"]))].append(progress)

    fig, axis = plt.subplots(figsize=(7, 4))
    for arch, step_map in sorted(grouped.items()):
        steps = sorted(step_map.keys())
        means = [sum(step_map[step]) / len(step_map[step]) for step in steps]
        axis.plot(steps, means, label=arch, color=ARCH_COLORS.get(arch, "#888888"), linewidth=2.2)
    axis.set_title("Developmental progress")
    axis.set_xlabel("Step")
    axis.set_ylabel("Juvenile progress / maturity signal")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_report(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    metadata = load_json(output_dir / "metadata.json")
    aggregate_rows = load_csv(output_dir / "aggregate_metrics.csv")
    run_rows = load_csv(output_dir / "run_metrics.csv")
    time_series_rows = load_gzip_csv(output_dir / "time_series.csv.gz")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    catch_rate_plot = plot_metric_bars(
        aggregate_rows,
        "catch_rate_per_minute",
        "Catch Rate per Minute",
        plots_dir / "catch_rate_per_minute.png",
        "Catches / minute",
    )
    capture_success_plot = plot_metric_bars(
        aggregate_rows,
        "capture_success",
        "Capture Success",
        plots_dir / "capture_success.png",
        "Success ratio",
    )
    first_catch_plot = plot_metric_bars(
        aggregate_rows,
        "time_to_first_catch_s",
        "Time to First Catch",
        plots_dir / "time_to_first_catch.png",
        "Seconds",
    )
    energy_plot = plot_metric_bars(
        aggregate_rows,
        "flies_per_energy_spent",
        "Flies Caught per Energy Spent",
        plots_dir / "energy_efficiency.png",
        "Catches / energy spent",
    )
    learning_curve_plot = plot_learning_curves(time_series_rows, plots_dir / "learning_curves.png")
    development_curve_plot = plot_development_curves(time_series_rows, plots_dir / "development_curves.png")

    adult_table = build_mode_table(aggregate_rows, "adult")
    developmental_table = build_mode_table(aggregate_rows, "developmental")
    adult_state_table = build_state_dependence_table(aggregate_rows, "adult")
    developmental_state_table = build_state_dependence_table(aggregate_rows, "developmental")
    adult_winners = determine_winners(aggregate_rows, "adult")
    developmental_winners = determine_winners(aggregate_rows, "developmental")

    lines: List[str] = []
    lines.append("# Frog Benchmark Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(
        f"- Architectures: {', '.join(metadata['architectures'])}\n"
        f"- Modes: {', '.join(metadata['modes'])}\n"
        f"- Spawn seeds: {len(metadata['spawn_seeds'])}\n"
        f"- Repeats per seed: {metadata['repeats']}\n"
        f"- Steps per run: {metadata['steps']}\n"
        f"- Online learning: enabled for current variants; frozen variants disable online updates\n"
        f"- Competence proxy: {metadata['competence_catches']} catches"
    )
    lines.append("")
    lines.append("## Adult Mode Summary")
    lines.append("")
    lines.append(markdown_table(adult_table))
    lines.append("")
    lines.append(
        f"Adult-mode winners: catch rate `{adult_winners.get('catch_rate', 'n/a')}`, "
        f"capture success `{adult_winners.get('capture_success', 'n/a')}`, "
        f"first catch `{adult_winners.get('first_catch', 'n/a')}`, "
        f"energy efficiency `{adult_winners.get('energy_efficiency', 'n/a')}`, "
        f"compute `{adult_winners.get('compute', 'n/a')}`."
    )
    lines.append("")
    lines.append("### Adult State Dependence")
    lines.append("")
    lines.append(markdown_table(adult_state_table))
    lines.append("")
    lines.append("## Developmental Mode Summary")
    lines.append("")
    lines.append(markdown_table(developmental_table))
    lines.append("")
    lines.append(
        f"Developmental-mode winners: catch rate `{developmental_winners.get('catch_rate', 'n/a')}`, "
        f"capture success `{developmental_winners.get('capture_success', 'n/a')}`, "
        f"first catch `{developmental_winners.get('first_catch', 'n/a')}`, "
        f"energy efficiency `{developmental_winners.get('energy_efficiency', 'n/a')}`, "
        f"compute `{developmental_winners.get('compute', 'n/a')}`."
    )
    lines.append("")
    lines.append("### Developmental State Dependence")
    lines.append("")
    lines.append(markdown_table(developmental_state_table))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `adult` corresponds to `training_mode=False`; `developmental` corresponds to `training_mode=True`.")
    lines.append("- Spawn seeds control the fly spawn stream; repeats keep the spawn stream but vary runtime randomness.")
    lines.append("- `ANN_FROZEN` and `SNN_FROZEN` keep the same architectures and sensors but disable online weight updates during the run.")
    lines.append("- `BIO_DUAL` is a copied descendant of the current BIO runtime with an internal slow-vs-fast prey-capture split, not an external controller.")
    lines.append("- ANN and SNN developmental modes are lighter-weight than BIO/BIO_DUAL developmental mode, so those rows should be interpreted as `training-like juvenile conditions`, not as fully symmetric ontogeny.")
    lines.append("- Metrics not tied to the fixed benchmark protocol, such as retention after a pause or robustness to environmental perturbations, were intentionally left out of this report.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Run metrics: `{(output_dir / 'run_metrics.csv').name}`")
    lines.append(f"- Aggregate metrics: `{(output_dir / 'aggregate_metrics.csv').name}`")
    lines.append(f"- Seed summary: `{(output_dir / 'seed_summary.csv').name}`")
    lines.append(f"- Time series: `{(output_dir / 'time_series.csv.gz').name}`")
    lines.append(f"- Event log: `{(output_dir / 'event_log.csv.gz').name}`")
    lines.append("")
    if catch_rate_plot:
        lines.append(f"![Catch rate]({catch_rate_plot.resolve().as_posix()})")
        lines.append("")
    if capture_success_plot:
        lines.append(f"![Capture success]({capture_success_plot.resolve().as_posix()})")
        lines.append("")
    if first_catch_plot:
        lines.append(f"![First catch]({first_catch_plot.resolve().as_posix()})")
        lines.append("")
    if energy_plot:
        lines.append(f"![Energy efficiency]({energy_plot.resolve().as_posix()})")
        lines.append("")
    if learning_curve_plot:
        lines.append(f"![Learning curves]({learning_curve_plot.resolve().as_posix()})")
        lines.append("")
    if development_curve_plot:
        lines.append(f"![Development curves]({development_curve_plot.resolve().as_posix()})")
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown report and plots for frog benchmark outputs.")
    parser.add_argument("output_dir", type=Path, help="Benchmark output directory created by benchmark_frogs.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = generate_report(args.output_dir)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
