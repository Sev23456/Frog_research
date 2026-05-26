#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import gzip
import html
import json
import math
import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from scipy import stats


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "article_assets" / "frog_architecture_study_20260421"


ARCH_COLORS = {
    "ANN": "#2f80ed",
    "ANN_FROZEN": "#7fb3ff",
    "ANN_SOFT_SATED_V2": "#0d356c",
    "ANN_FROZEN_SOFT_SATED_V2": "#5f86c4",
    "ANN_SOFT_SATED_V3": "#081f45",
    "ANN_FROZEN_SOFT_SATED_V3": "#476fa7",
    "ANN_SOFT_SATED_V3_1": "#103260",
    "ANN_FROZEN_SOFT_SATED_V3_1": "#6d91bf",
    "SNN": "#27ae60",
    "SNN_FROZEN": "#7bd8a3",
    "BIO_COMPARE": "#ff9b3d",
    "BIO_DUAL_COMPARE": "#cf7b2f",
    "BIO_FAST_COMPARE": "#ffd08c",
    "BIO_DUAL_FAST_COMPARE": "#a75a1c",
}


@dataclass(frozen=True)
class StudySource:
    key: str
    label: str
    relative_dir: str
    family: str
    role: str


@dataclass(frozen=True)
class ArchMeta:
    arch: str
    label_ru: str
    short_label: str
    folder_name: str
    family: str
    branch: str
    chronology_group: str
    chronology_order: int


STUDY_SOURCES = (
    StudySource(
        key="ann_snn_baseline",
        label="Базовое сравнение ANN/SNN",
        relative_dir=r"benchmark_results\studies_full_subprocess\frog_study_20260418_165410\partial_merged_ann_snn",
        family="baseline",
        role="main_baseline",
    ),
    StudySource(
        key="bio_compare_family",
        label="Семейство compare-биолягушек",
        relative_dir=r"benchmark_results\bio_compare_only_foreground_20260420_run1\frog_study_20260420_193346\merged",
        family="bio",
        role="bio_family",
    ),
    StudySource(
        key="ann_homeostasis_chronology",
        label="Хронология homeostatic ANN: V2 -> V3 -> V3.1",
        relative_dir=r"benchmark_results\ann_soft_sated_v2_v3_v31_full_20260421_run1\frog_study_20260421_174042\merged",
        family="ann_homeostasis",
        role="chronology",
    ),
)


ARCH_META: Dict[str, ArchMeta] = {
    "ANN": ArchMeta("ANN", "ANN (frog_lib_ann)", "ANN", "frog_lib_ann", "ANN", "baseline", "baseline", 10),
    "ANN_FROZEN": ArchMeta("ANN_FROZEN", "ANN frozen (frog_lib_ann_frozen)", "ANN frozen", "frog_lib_ann_frozen", "ANN", "baseline_frozen", "baseline", 11),
    "SNN": ArchMeta("SNN", "SNN (frog_lib_snn)", "SNN", "frog_lib_snn", "SNN", "baseline", "baseline", 20),
    "SNN_FROZEN": ArchMeta("SNN_FROZEN", "SNN frozen (frog_lib_snn_frozen)", "SNN frozen", "frog_lib_snn_frozen", "SNN", "baseline_frozen", "baseline", 21),
    "ANN_SOFT_SATED_V2": ArchMeta(
        "ANN_SOFT_SATED_V2",
        "ANN homeostatic V2 (frog_lib_ann_soft_sated_v2)",
        "ANN V2",
        "frog_lib_ann_soft_sated_v2",
        "ANN",
        "ann_homeostasis",
        "ann_homeostasis",
        30,
    ),
    "ANN_FROZEN_SOFT_SATED_V2": ArchMeta(
        "ANN_FROZEN_SOFT_SATED_V2",
        "ANN frozen homeostatic V2 (frog_lib_ann_frozen_soft_sated_v2)",
        "ANN frozen V2",
        "frog_lib_ann_frozen_soft_sated_v2",
        "ANN",
        "ann_homeostasis_frozen",
        "ann_homeostasis",
        31,
    ),
    "ANN_SOFT_SATED_V3": ArchMeta(
        "ANN_SOFT_SATED_V3",
        "ANN homeostatic V3 (frog_lib_ann_soft_sated_v3)",
        "ANN V3",
        "frog_lib_ann_soft_sated_v3",
        "ANN",
        "ann_homeostasis",
        "ann_homeostasis",
        32,
    ),
    "ANN_FROZEN_SOFT_SATED_V3": ArchMeta(
        "ANN_FROZEN_SOFT_SATED_V3",
        "ANN frozen homeostatic V3 (frog_lib_ann_frozen_soft_sated_v3)",
        "ANN frozen V3",
        "frog_lib_ann_frozen_soft_sated_v3",
        "ANN",
        "ann_homeostasis_frozen",
        "ann_homeostasis",
        33,
    ),
    "ANN_SOFT_SATED_V3_1": ArchMeta(
        "ANN_SOFT_SATED_V3_1",
        "ANN homeostatic V3.1 (frog_lib_ann_soft_sated_v3_1)",
        "ANN V3.1",
        "frog_lib_ann_soft_sated_v3_1",
        "ANN",
        "ann_homeostasis_final",
        "ann_homeostasis",
        34,
    ),
    "ANN_FROZEN_SOFT_SATED_V3_1": ArchMeta(
        "ANN_FROZEN_SOFT_SATED_V3_1",
        "ANN frozen homeostatic V3.1 (frog_lib_ann_frozen_soft_sated_v3_1)",
        "ANN frozen V3.1",
        "frog_lib_ann_frozen_soft_sated_v3_1",
        "ANN",
        "ann_homeostasis_frozen",
        "ann_homeostasis",
        35,
    ),
    "BIO_COMPARE": ArchMeta(
        "BIO_COMPARE",
        "BioFrog compare (Frog_predator_neuro_compare)",
        "BIO compare",
        "Frog_predator_neuro_compare",
        "BIO",
        "bio_compare",
        "bio_family",
        40,
    ),
    "BIO_DUAL_COMPARE": ArchMeta(
        "BIO_DUAL_COMPARE",
        "BioFrog dual compare (Frog_predator_neuro_dual_compare)",
        "BIO dual compare",
        "Frog_predator_neuro_dual_compare",
        "BIO",
        "bio_dual_compare",
        "bio_family",
        41,
    ),
    "BIO_FAST_COMPARE": ArchMeta(
        "BIO_FAST_COMPARE",
        "BioFrog fast compare (Frog_predator_neuro_fast_compare)",
        "BIO fast compare",
        "Frog_predator_neuro_fast_compare",
        "BIO",
        "bio_fast_compare",
        "bio_family",
        42,
    ),
    "BIO_DUAL_FAST_COMPARE": ArchMeta(
        "BIO_DUAL_FAST_COMPARE",
        "BioFrog dual fast compare (Frog_predator_neuro_dual_fast_compare)",
        "BIO dual fast compare",
        "Frog_predator_neuro_dual_fast_compare",
        "BIO",
        "bio_dual_fast_compare",
        "bio_family",
        43,
    ),
}


FINAL_MAIN_ARCHES = ("ANN_SOFT_SATED_V3_1", "SNN", "BIO_DUAL_FAST_COMPARE")
ANN_SNN_BASELINE_ARCHES = ("ANN", "ANN_FROZEN", "SNN", "SNN_FROZEN")
BIO_COMPARE_ARCHES = ("BIO_COMPARE", "BIO_DUAL_COMPARE", "BIO_FAST_COMPARE", "BIO_DUAL_FAST_COMPARE")
ANN_HOMEOSTASIS_ARCHES = (
    "ANN_SOFT_SATED_V2",
    "ANN_FROZEN_SOFT_SATED_V2",
    "ANN_SOFT_SATED_V3",
    "ANN_FROZEN_SOFT_SATED_V3",
    "ANN_SOFT_SATED_V3_1",
    "ANN_FROZEN_SOFT_SATED_V3_1",
)


KEY_METRICS = {
    "catch_rate_per_minute": ("Темп ловли, мух/мин", True),
    "time_to_first_catch_s": ("Время до первой поимки, с", False),
    "flies_per_energy_spent": ("Энергоэффективность, мух/энергию", True),
    "visible_but_ignored_ratio": ("Игнорирование видимой добычи", False),
    "learning_auc": ("Learning AUC", True),
    "capture_success": ("Успешность поимки", True),
    "compute_per_catch_s": ("Вычислительная цена, с/поимку", False),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def as_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def meta_for_arch(arch: str) -> ArchMeta:
    return ARCH_META.get(
        arch,
        ArchMeta(arch, f"{arch} (unknown)", arch, arch, "UNKNOWN", "unknown", "unknown", 999),
    )


def add_row_context(row: Dict[str, str], source: StudySource) -> Dict[str, Any]:
    meta = meta_for_arch(row["arch"])
    enriched = dict(row)
    enriched["study_key"] = source.key
    enriched["study_label"] = source.label
    enriched["study_family"] = source.family
    enriched["study_role"] = source.role
    enriched["arch_label_ru"] = meta.label_ru
    enriched["arch_short_label"] = meta.short_label
    enriched["arch_folder_name"] = meta.folder_name
    enriched["arch_family"] = meta.family
    enriched["arch_branch"] = meta.branch
    enriched["chronology_group"] = meta.chronology_group
    enriched["chronology_order"] = meta.chronology_order

    low = as_float(row.get("catch_rate_low_energy"))
    high = as_float(row.get("catch_rate_high_energy"))
    strike_low = as_float(row.get("strike_rate_low_energy"))
    strike_high = as_float(row.get("strike_rate_high_energy"))
    enriched["homeostatic_catch_delta"] = "" if low is None or high is None else f"{low - high:.12f}"
    enriched["homeostatic_strike_delta"] = "" if strike_low is None or strike_high is None else f"{strike_low - strike_high:.12f}"

    energy_ratio = as_float(row.get("energy_ratio"))
    if energy_ratio is not None:
        enriched["energy_deficit"] = f"{1.0 - energy_ratio:.12f}"
    return enriched


def mean_std_label(mean_value: Optional[float], std_value: Optional[float], precision: int = 2) -> str:
    if mean_value is None:
        return "n/a"
    if std_value is None:
        return f"{mean_value:.{precision}f}"
    return f"{mean_value:.{precision}f} ± {std_value:.{precision}f}"


def mean_std_from_run(rows: Sequence[Dict[str, Any]], metric: str) -> Tuple[Optional[float], Optional[float]]:
    values = [as_float(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "_Нет данных._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def html_table(title: str, rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return f"<h2>{html.escape(title)}</h2><p>Нет данных.</p>"
    headers = list(rows[0].keys())
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_html = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        body_html.append(f"<tr>{tds}</tr>")
    return (
        f"<h2>{html.escape(title)}</h2>"
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_html)}</tbody>"
        "</table>"
    )


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_gzip_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def cliffs_delta(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if not xs or not ys:
        return None
    greater = 0
    lower = 0
    for x in xs:
        for y in ys:
            if x > y:
                greater += 1
            elif x < y:
                lower += 1
    total = len(xs) * len(ys)
    if total == 0:
        return None
    return (greater - lower) / total


def holm_correction(records: List[Dict[str, Any]], p_key: str, out_key: str) -> None:
    ordered = sorted(
        [(index, record) for index, record in enumerate(records) if record.get(p_key) is not None],
        key=lambda item: item[1][p_key],
    )
    m = len(ordered)
    if m == 0:
        return
    adjusted: List[Tuple[int, float]] = []
    running_max = 0.0
    for rank, (index, record) in enumerate(ordered, start=1):
        adjusted_p = (m - rank + 1) * float(record[p_key])
        adjusted_p = min(1.0, adjusted_p)
        running_max = max(running_max, adjusted_p)
        adjusted.append((index, running_max))
    for index, value in adjusted:
        records[index][out_key] = value


def format_p_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def pairwise_tests(
    run_rows: Sequence[Dict[str, Any]],
    arches: Sequence[str],
    mode: str,
    metrics: Dict[str, Tuple[str, bool]],
    group_label: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    selected = [row for row in run_rows if row.get("mode") == mode and row.get("arch") in arches]
    by_arch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_arch[row["arch"]].append(row)

    for metric, (metric_label, higher_is_better) in metrics.items():
        metric_rows: List[Dict[str, Any]] = []
        for left, right in combinations(arches, 2):
            left_values = [as_float(row.get(metric)) for row in by_arch.get(left, [])]
            right_values = [as_float(row.get(metric)) for row in by_arch.get(right, [])]
            left_values = [value for value in left_values if value is not None]
            right_values = [value for value in right_values if value is not None]
            record: Dict[str, Any] = {
                "group": group_label,
                "mode": mode,
                "metric": metric,
                "metric_label_ru": metric_label,
                "left_arch": left,
                "left_label_ru": meta_for_arch(left).label_ru,
                "right_arch": right,
                "right_label_ru": meta_for_arch(right).label_ru,
                "n_left": len(left_values),
                "n_right": len(right_values),
            }
            if left_values and right_values:
                p_raw = stats.mannwhitneyu(left_values, right_values, alternative="two-sided").pvalue
                effect = cliffs_delta(left_values, right_values)
                left_mean = statistics.fmean(left_values)
                right_mean = statistics.fmean(right_values)
                diff = left_mean - right_mean
                preferred = left if diff > 0 else right
                if not higher_is_better:
                    preferred = right if diff > 0 else left
                record.update(
                    {
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                        "mean_difference_left_minus_right": diff,
                        "p_raw": p_raw,
                        "cliffs_delta": effect,
                        "preferred_arch": preferred,
                    }
                )
            else:
                record.update(
                    {
                        "left_mean": None,
                        "right_mean": None,
                        "mean_difference_left_minus_right": None,
                        "p_raw": None,
                        "cliffs_delta": None,
                        "preferred_arch": "n/a",
                    }
                )
            metric_rows.append(record)
        holm_correction(metric_rows, "p_raw", "p_holm")
        for row in metric_rows:
            row["significant_05"] = bool(row.get("p_holm") is not None and row["p_holm"] < 0.05)
        results.extend(metric_rows)
    return results


def correlation_by_run(
    time_rows: Sequence[Dict[str, Any]],
    arches: Sequence[str],
    mode: str,
    x_key: str,
    y_key: str,
    label: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = defaultdict(list)
    for row in time_rows:
        if row.get("mode") != mode or row.get("arch") not in arches:
            continue
        x = as_float(row.get(x_key))
        y = as_float(row.get(y_key))
        if x is None or y is None:
            continue
        grouped[(row["arch"], row.get("spawn_seed", ""), row.get("repeat", ""))].append((x, y))

    per_run_rows: List[Dict[str, Any]] = []
    for (arch, seed, repeat), values in sorted(grouped.items()):
        xs = [item[0] for item in values]
        ys = [item[1] for item in values]
        if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        corr = stats.spearmanr(xs, ys).correlation
        per_run_rows.append(
            {
                "arch": arch,
                "arch_label_ru": meta_for_arch(arch).label_ru,
                "mode": mode,
                "spawn_seed": seed,
                "repeat": repeat,
                "correlation_label": label,
                "correlation": corr,
            }
        )
    return per_run_rows


def aggregate_correlation_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        corr = as_float(row.get("correlation"))
        if corr is None:
            continue
        grouped[(row["arch"], row["mode"])].append(corr)

    result: List[Dict[str, Any]] = []
    for (arch, mode), values in sorted(grouped.items(), key=lambda item: (meta_for_arch(item[0][0]).chronology_order, item[0][1])):
        mean_value = statistics.fmean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(
            {
                "Архитектура": meta_for_arch(arch).label_ru,
                "Короткое имя": meta_for_arch(arch).short_label,
                "Режим": mode,
                "Корреляция, mean ± std": mean_std_label(mean_value, std_value, precision=3),
                "N запусков": len(values),
            }
        )
    return result


def select_rows(rows: Sequence[Dict[str, Any]], arches: Sequence[str], modes: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    mode_set = set(modes) if modes else None
    selected = [row for row in rows if row.get("arch") in arches and (mode_set is None or row.get("mode") in mode_set)]
    order_map = {arch: index for index, arch in enumerate(arches)}
    selected.sort(key=lambda row: (row.get("mode", ""), order_map.get(row["arch"], 999), meta_for_arch(row["arch"]).chronology_order))
    return selected


def build_summary_table(aggregate_rows: Sequence[Dict[str, Any]], arches: Sequence[str], title_prefix: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in select_rows(aggregate_rows, arches):
        rows.append(
            {
                "Группа": title_prefix,
                "Режим": row["mode"],
                "Архитектура": row["arch_label_ru"],
                "Короткое имя": row["arch_short_label"],
                "Темп ловли, mean ± std": mean_std_label(as_float(row.get("catch_rate_per_minute_mean")), as_float(row.get("catch_rate_per_minute_std"))),
                "Первая поимка, с": mean_std_label(as_float(row.get("time_to_first_catch_s_mean")), as_float(row.get("time_to_first_catch_s_std"))),
                "Энергоэффективность": mean_std_label(as_float(row.get("flies_per_energy_spent_mean")), as_float(row.get("flies_per_energy_spent_std")), precision=3),
                "Learning AUC": mean_std_label(as_float(row.get("learning_auc_mean")), as_float(row.get("learning_auc_std"))),
                "Игнорирование видимой добычи": mean_std_label(as_float(row.get("visible_but_ignored_ratio_mean")), as_float(row.get("visible_but_ignored_ratio_std")), precision=3),
                "Low E catch": mean_std_label(as_float(row.get("catch_rate_low_energy_mean")), as_float(row.get("catch_rate_low_energy_std"))),
                "High E catch": mean_std_label(as_float(row.get("catch_rate_high_energy_mean")), as_float(row.get("catch_rate_high_energy_std"))),
                "Delta low-high": mean_std_label(as_float(row.get("homeostatic_catch_delta_mean")), as_float(row.get("homeostatic_catch_delta_std"))),
                "Вычислительная цена, с/поимку": mean_std_label(as_float(row.get("compute_per_catch_s_mean")), as_float(row.get("compute_per_catch_s_std")), precision=2),
            }
        )
    return rows


def svg_escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def wrap_svg(width: int, height: int, body: str, title: str = "") -> str:
    title_block = f"<title>{svg_escape(title)}</title>" if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{svg_escape(title)}">'
        f"{title_block}"
        "<style>"
        "text{font-family:'Segoe UI',Arial,sans-serif;fill:#1f2937}"
        ".title{font-size:22px;font-weight:700}"
        ".subtitle{font-size:13px;fill:#4b5563}"
        ".axis{stroke:#4b5563;stroke-width:1}"
        ".grid{stroke:#d1d5db;stroke-width:1}"
        ".legend{font-size:12px}"
        ".label{font-size:12px}"
        ".tick{font-size:11px;fill:#4b5563}"
        "</style>"
        f"{body}</svg>"
    )


def bar_chart_svg(
    panels: Sequence[Tuple[str, Sequence[Tuple[str, float, float, str]]]],
    title: str,
    subtitle: str,
    ylabel: str,
    path: Path,
) -> None:
    width = 560 * max(1, len(panels))
    height = 440
    panel_width = width / max(1, len(panels))
    top = 70
    bottom = 80
    left = 70
    right = 40
    plot_height = height - top - bottom
    body: List[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="34">{svg_escape(title)}</text>',
        f'<text class="subtitle" x="24" y="56">{svg_escape(subtitle)}</text>',
    ]

    for idx, (panel_title, items) in enumerate(panels):
        plot_left = idx * panel_width + left
        plot_right = (idx + 1) * panel_width - right
        plot_width = plot_right - plot_left
        plot_top = top
        all_values = [max(0.0, value + error) for _, value, error, _ in items] or [1.0]
        max_value = max(all_values)
        if max_value <= 0:
            max_value = 1.0
        rounded_max = max_value * 1.1
        tick_count = 5
        body.append(f'<text class="label" x="{plot_left}" y="{plot_top - 14}">{svg_escape(panel_title)}</text>')
        for tick_index in range(tick_count + 1):
            tick_value = rounded_max * tick_index / tick_count
            y = plot_top + plot_height - (tick_value / rounded_max) * plot_height
            body.append(f'<line class="grid" x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}"/>')
            body.append(f'<text class="tick" x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick_value:.2f}</text>')
        body.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}"/>')
        body.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_right}" y2="{plot_top + plot_height}"/>')
        body.append(
            f'<text class="tick" x="{plot_left - 50}" y="{plot_top + plot_height / 2:.1f}" transform="rotate(-90 {plot_left - 50} {plot_top + plot_height / 2:.1f})">{svg_escape(ylabel)}</text>'
        )

        count = max(1, len(items))
        gap = 18
        bar_width = max(18.0, (plot_width - gap * (count + 1)) / count)
        for item_index, (label, value, error, color) in enumerate(items):
            x = plot_left + gap + item_index * (bar_width + gap)
            bar_height = (max(0.0, value) / rounded_max) * plot_height
            y = plot_top + plot_height - bar_height
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" opacity="0.9"/>')
            err_top = plot_top + plot_height - ((value + error) / rounded_max) * plot_height
            err_bottom = plot_top + plot_height - ((max(0.0, value - error)) / rounded_max) * plot_height
            center = x + bar_width / 2
            body.append(f'<line class="axis" x1="{center:.1f}" y1="{err_top:.1f}" x2="{center:.1f}" y2="{err_bottom:.1f}"/>')
            body.append(f'<line class="axis" x1="{center - 5:.1f}" y1="{err_top:.1f}" x2="{center + 5:.1f}" y2="{err_top:.1f}"/>')
            body.append(f'<line class="axis" x1="{center - 5:.1f}" y1="{err_bottom:.1f}" x2="{center + 5:.1f}" y2="{err_bottom:.1f}"/>')
            body.append(f'<text class="tick" x="{center:.1f}" y="{plot_top + plot_height + 18}" text-anchor="middle">{svg_escape(label)}</text>')
            body.append(f'<text class="tick" x="{center:.1f}" y="{y - 6:.1f}" text-anchor="middle">{value:.2f}</text>')

    save_text(path, wrap_svg(width, height, "".join(body), title))


def quartiles(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    ordered = sorted(values)
    return (
        ordered[0],
        percentile(ordered, 0.25),
        percentile(ordered, 0.50),
        percentile(ordered, 0.75),
        ordered[-1],
    )


def boxplot_svg(
    panels: Sequence[Tuple[str, Sequence[Tuple[str, Sequence[float], str]]]],
    title: str,
    subtitle: str,
    ylabel: str,
    path: Path,
) -> None:
    width = 560 * max(1, len(panels))
    height = 440
    panel_width = width / max(1, len(panels))
    top = 70
    bottom = 80
    left = 70
    right = 40
    plot_height = height - top - bottom
    body: List[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="34">{svg_escape(title)}</text>',
        f'<text class="subtitle" x="24" y="56">{svg_escape(subtitle)}</text>',
    ]
    for idx, (panel_title, items) in enumerate(panels):
        plot_left = idx * panel_width + left
        plot_right = (idx + 1) * panel_width - right
        plot_width = plot_right - plot_left
        plot_top = top
        all_values = [value for _, values, _ in items for value in values]
        max_value = max(all_values) if all_values else 1.0
        min_value = min(all_values) if all_values else 0.0
        if math.isclose(max_value, min_value):
            max_value += 1.0
            min_value -= 1.0
        pad = (max_value - min_value) * 0.1
        max_value += pad
        min_value -= pad
        body.append(f'<text class="label" x="{plot_left}" y="{plot_top - 14}">{svg_escape(panel_title)}</text>')
        for tick_index in range(6):
            tick_value = min_value + (max_value - min_value) * tick_index / 5
            y = plot_top + plot_height - ((tick_value - min_value) / (max_value - min_value)) * plot_height
            body.append(f'<line class="grid" x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}"/>')
            body.append(f'<text class="tick" x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick_value:.2f}</text>')
        body.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}"/>')
        body.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_right}" y2="{plot_top + plot_height}"/>')
        count = max(1, len(items))
        gap = 22
        box_width = max(26.0, (plot_width - gap * (count + 1)) / count)
        for item_index, (label, values, color) in enumerate(items):
            if not values:
                continue
            x = plot_left + gap + item_index * (box_width + gap)
            min_v, q1, med, q3, max_v = quartiles(list(values))

            def y_for(value: float) -> float:
                return plot_top + plot_height - ((value - min_value) / (max_value - min_value)) * plot_height

            y_q1 = y_for(q1)
            y_q3 = y_for(q3)
            y_med = y_for(med)
            y_min = y_for(min_v)
            y_max = y_for(max_v)
            center = x + box_width / 2
            body.append(f'<line class="axis" x1="{center:.1f}" y1="{y_min:.1f}" x2="{center:.1f}" y2="{y_q1:.1f}"/>')
            body.append(f'<line class="axis" x1="{center:.1f}" y1="{y_q3:.1f}" x2="{center:.1f}" y2="{y_max:.1f}"/>')
            body.append(f'<rect x="{x:.1f}" y="{y_q3:.1f}" width="{box_width:.1f}" height="{max(1.0, y_q1 - y_q3):.1f}" fill="{color}" opacity="0.75" stroke="#1f2937"/>')
            body.append(f'<line class="axis" x1="{x:.1f}" y1="{y_med:.1f}" x2="{x + box_width:.1f}" y2="{y_med:.1f}"/>')
            body.append(f'<line class="axis" x1="{x + 4:.1f}" y1="{y_min:.1f}" x2="{x + box_width - 4:.1f}" y2="{y_min:.1f}"/>')
            body.append(f'<line class="axis" x1="{x + 4:.1f}" y1="{y_max:.1f}" x2="{x + box_width - 4:.1f}" y2="{y_max:.1f}"/>')
            body.append(f'<text class="tick" x="{center:.1f}" y="{plot_top + plot_height + 18}" text-anchor="middle">{svg_escape(label)}</text>')
    save_text(path, wrap_svg(width, height, "".join(body), title))


def line_chart_svg(
    panels: Sequence[Tuple[str, Sequence[Tuple[str, Sequence[Tuple[float, float]], str]]]],
    title: str,
    subtitle: str,
    ylabel: str,
    path: Path,
) -> None:
    width = 560 * max(1, len(panels))
    height = 440
    panel_width = width / max(1, len(panels))
    top = 70
    bottom = 70
    left = 70
    right = 40
    plot_height = height - top - bottom
    body: List[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="34">{svg_escape(title)}</text>',
        f'<text class="subtitle" x="24" y="56">{svg_escape(subtitle)}</text>',
    ]
    for idx, (panel_title, series) in enumerate(panels):
        plot_left = idx * panel_width + left
        plot_right = (idx + 1) * panel_width - right
        plot_width = plot_right - plot_left
        plot_top = top
        all_x = [x for _, points, _ in series for x, _ in points]
        all_y = [y for _, points, _ in series for _, y in points]
        if not all_x or not all_y:
            continue
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        if math.isclose(max_x, min_x):
            max_x += 1.0
        if math.isclose(max_y, min_y):
            max_y += 1.0
        y_pad = (max_y - min_y) * 0.1
        max_y += y_pad
        min_y = max(0.0, min_y - y_pad)
        body.append(f'<text class="label" x="{plot_left}" y="{plot_top - 14}">{svg_escape(panel_title)}</text>')
        for tick_index in range(6):
            tick_y = min_y + (max_y - min_y) * tick_index / 5
            y = plot_top + plot_height - ((tick_y - min_y) / (max_y - min_y)) * plot_height
            body.append(f'<line class="grid" x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}"/>')
            body.append(f'<text class="tick" x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick_y:.2f}</text>')
        body.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}"/>')
        body.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_right}" y2="{plot_top + plot_height}"/>')
        legend_x = plot_left
        legend_y = plot_top + plot_height + 24
        for legend_index, (label, points, color) in enumerate(series):
            path_points = []
            for x_value, y_value in points:
                x = plot_left + ((x_value - min_x) / (max_x - min_x)) * plot_width
                y = plot_top + plot_height - ((y_value - min_y) / (max_y - min_y)) * plot_height
                path_points.append((x, y))
            if len(path_points) < 2:
                continue
            d = " ".join(
                [f"M {path_points[0][0]:.1f} {path_points[0][1]:.1f}"]
                + [f"L {x:.1f} {y:.1f}" for x, y in path_points[1:]]
            )
            body.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            lx = legend_x + legend_index * 160
            body.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            body.append(f'<text class="legend" x="{lx + 24}" y="{legend_y + 4}">{svg_escape(label)}</text>')
        for tick_index in range(6):
            tick_x = min_x + (max_x - min_x) * tick_index / 5
            x = plot_left + ((tick_x - min_x) / (max_x - min_x)) * plot_width
            body.append(f'<text class="tick" x="{x:.1f}" y="{plot_top + plot_height + 18}" text-anchor="middle">{int(tick_x)}</text>')
        body.append(
            f'<text class="tick" x="{plot_left - 50}" y="{plot_top + plot_height / 2:.1f}" transform="rotate(-90 {plot_left - 50} {plot_top + plot_height / 2:.1f})">{svg_escape(ylabel)}</text>'
        )
    save_text(path, wrap_svg(width, height, "".join(body), title))


def timeline_svg(path: Path) -> None:
    width = 1400
    height = 620
    body: List[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text class="title" x="24" y="36">Хронология исследовательских версий лягушки</text>',
        '<text class="subtitle" x="24" y="60">Отдельно показаны ветки честности homeostasis для ANN и compare/runtime-ветки для BioFrog.</text>',
        '<line x1="80" y1="170" x2="1320" y2="170" stroke="#cbd5e1" stroke-width="4"/>',
        '<line x1="80" y1="410" x2="1320" y2="410" stroke="#cbd5e1" stroke-width="4"/>',
        '<text class="label" x="80" y="130">ANN / homeostatic fairness</text>',
        '<text class="label" x="80" y="370">BioFrog / compare-runtime ветка</text>',
    ]

    ann_nodes = [
        ("ANN", 140, 170, "#2f80ed", "Базовый RL-ANN"),
        ("SATED", 320, 170, "#1f5fbf", "Жёсткое насыщение"),
        ("SOFT", 500, 170, "#154a94", "Плавное ослабление"),
        ("V2", 700, 170, "#0d356c", "Честнее, но wrong-sign bias"),
        ("V3", 900, 170, "#081f45", "Инстинкт перенесён вверх, но переторможен"),
        ("V3.1", 1120, 170, "#103260", "Смягчённый V3; текущий компромисс"),
    ]
    bio_nodes = [
        ("BIO", 140, 410, "#e67e22", "Исходный биоморфный runtime"),
        ("COMPARE", 340, 410, "#ff9b3d", "Task-set floor для честного сравнения"),
        ("DUAL", 560, 410, "#b85c11", "Двуконтурный prey-capture runtime"),
        ("FAST", 760, 410, "#f2a65a", "Ускоренная версия"),
        ("DUAL_COMPARE", 980, 410, "#cf7b2f", "Сравнимый dual runtime"),
        ("DUAL_FAST_COMPARE", 1220, 410, "#a75a1c", "Текущий compare-ready компромисс"),
    ]

    for label, x, y, color, note in ann_nodes + bio_nodes:
        body.append(f'<circle cx="{x}" cy="{y}" r="22" fill="{color}"/>')
        body.append(f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="11" fill="#ffffff" font-family="Segoe UI,Arial,sans-serif" font-weight="700">{svg_escape(label)}</text>')
        body.append(f'<text class="legend" x="{x}" y="{y + 46}" text-anchor="middle">{svg_escape(note)}</text>')

    body.append('<text class="subtitle" x="80" y="548">Смысл ANN-ветки: постепенно убрать прямой постконтроллер насыщения и сделать comparison более homeostatically честным.</text>')
    body.append('<text class="subtitle" x="80" y="570">Смысл BIO-ветки: удержать биоморфность, но сделать режим охоты и benchmark-сравнение ближе к условиям ANN/SNN.</text>')
    save_text(path, wrap_svg(width, height, "".join(body), "Version chronology"))


def build_time_series_mean(
    rows: Sequence[Dict[str, Any]],
    arches: Sequence[str],
    mode: str,
    value_key: str,
) -> Dict[str, List[Tuple[float, float]]]:
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("mode") != mode or row.get("arch") not in arches:
            continue
        value = as_float(row.get(value_key))
        step = as_int(row.get("step"))
        if value is None or step is None:
            continue
        grouped[row["arch"]][step].append(value)
    result: Dict[str, List[Tuple[float, float]]] = {}
    for arch, step_map in grouped.items():
        series = []
        for step in sorted(step_map):
            values = step_map[step]
            if not values:
                continue
            series.append((float(step), statistics.fmean(values)))
        result[arch] = series
    return result


def compute_aggregate_with_deltas(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        low_mean = as_float(row.get("catch_rate_low_energy_mean"))
        high_mean = as_float(row.get("catch_rate_high_energy_mean"))
        low_std = as_float(row.get("catch_rate_low_energy_std"))
        high_std = as_float(row.get("catch_rate_high_energy_std"))
        strike_low_mean = as_float(row.get("strike_rate_low_energy_mean"))
        strike_high_mean = as_float(row.get("strike_rate_high_energy_mean"))
        strike_low_std = as_float(row.get("strike_rate_low_energy_std"))
        strike_high_std = as_float(row.get("strike_rate_high_energy_std"))
        updated["homeostatic_catch_delta_mean"] = "" if low_mean is None or high_mean is None else low_mean - high_mean
        updated["homeostatic_catch_delta_std"] = "" if low_std is None or high_std is None else math.sqrt(low_std ** 2 + high_std ** 2)
        updated["homeostatic_strike_delta_mean"] = "" if strike_low_mean is None or strike_high_mean is None else strike_low_mean - strike_high_mean
        updated["homeostatic_strike_delta_std"] = "" if strike_low_std is None or strike_high_std is None else math.sqrt(strike_low_std ** 2 + strike_high_std ** 2)
        enriched.append(updated)
    return enriched


def load_sources() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    study_index_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []

    for source in STUDY_SOURCES:
        source_dir = ROOT / source.relative_dir
        metadata = read_json(source_dir / "metadata.json")
        study_index_rows.append(
            {
                "study_key": source.key,
                "study_label_ru": source.label,
                "study_role": source.role,
                "family": source.family,
                "source_dir": str(source_dir),
                "architectures": ", ".join(metadata.get("architectures", [])),
                "modes": ", ".join(metadata.get("modes", [])),
                "steps": ", ".join(str(item) for item in metadata.get("steps", [])),
                "spawn_seeds": len(metadata.get("spawn_seeds", [])),
                "repeats": ", ".join(str(item) for item in metadata.get("repeats", [])),
                "sample_interval": ", ".join(str(item) for item in metadata.get("sample_interval", [])),
                "competence_catches": ", ".join(str(item) for item in metadata.get("competence_catches", [])),
            }
        )
        for row in compute_aggregate_with_deltas([add_row_context(item, source) for item in read_csv(source_dir / "aggregate_metrics.csv")]):
            aggregate_rows.append(row)
        for row in [add_row_context(item, source) for item in read_csv(source_dir / "run_metrics.csv")]:
            run_rows.append(row)
        seed_path = source_dir / "seed_summary.csv"
        if seed_path.exists():
            for row in [add_row_context(item, source) for item in read_csv(seed_path)]:
                seed_rows.append(row)
        time_path = source_dir / "time_series.csv.gz"
        if time_path.exists():
            for row in [add_row_context(item, source) for item in read_gzip_csv(time_path)]:
                time_rows.append(row)
    return study_index_rows, aggregate_rows, run_rows, seed_rows, time_rows


def build_readme() -> str:
    return "\n".join(
        [
            "# Пакет артефактов для статьи по лягушке",
            "",
            "Этот каталог собран автоматически из уже завершённых прогонов и предназначен как стартовый пакет для статьи, техотчёта и приложения.",
            "",
            "## Что внутри",
            "",
            "- `data/`: нормализованные CSV/JSON/TSV-представления с привязкой к исходным study-директориям.",
            "- `tables/`: человекочитаемые таблицы в Markdown и HTML.",
            "- `figures/main/`: более компактный набор графиков для основной статьи.",
            "- `figures/appendix/`: расширенный набор графиков для приложения.",
            "- `workbook/`: Excel-книга со сводными таблицами и индексами.",
            "- `outlines/`: каркасы статьи, техотчёта, приложения и executive summary.",
            "",
            "## Главная рабочая гипотеза текущего пакета",
            "",
            "- Финальное основное сравнение собрано как `ANN_SOFT_SATED_V3_1` vs `SNN` vs `BIO_DUAL_FAST_COMPARE`.",
            "- Это решение принято автоматически как рабочее, потому что `V3.1` была последней ANN-веткой честности, `SNN` был заявлен как основной SNN-эталон, а `BIO_DUAL_FAST_COMPARE` является последним compare-ready потомком dual-fast ветки.",
            "- При необходимости этот выбор можно быстро пересобрать, отредактировав только список `FINAL_MAIN_ARCHES` в `build_frog_article_assets.py`.",
            "",
            "## Важное ограничение",
            "",
            "- Основное сравнение сведено из нескольких завершённых исследований, а не из одного совместного повторного запуска. Для статьи это надо будет явно проговорить в разделе об ограничениях и воспроизводимости.",
        ]
    )


def build_article_outline() -> str:
    return "\n".join(
        [
            "# Каркас основной статьи",
            "",
            "1. Введение",
            "   - постановка задачи: сравнение ANN, SNN и биоморфных архитектур в игре про ловлю мух;",
            "   - почему задача вообще интересна для AI и computational neuroscience.",
            "",
            "2. Краткий обзор сравниваемых архитектур",
            "   - ANN;",
            "   - SNN: как кодируются сенсорные сигналы, где возникают спайки и как из них декодируется моторное действие;",
            "   - биоморфные SNN/runtime-архитектуры.",
            "",
            "3. Что именно считается биоморфностью в этой работе",
            "   - сигнал hunger/reward/predation/task-set;",
            "   - детство и взрослость;",
            "   - сравнение с мозгом и с обычными ИНС;",
            "   - аккуратная формулировка ограничений.",
            "",
            "4. Конкретные архитектуры, участвовавшие в исследовании",
            "   - основной триплет: ANN V3.1, SNN, BIO_DUAL_FAST_COMPARE;",
            "   - базовые ANN/SNN frozen-варианты;",
            "   - отдельный разбор текущей SNN-архитектуры: сенсорное кодирование, спайковая динамика, пластичность, моторный декодер и вычислительная цена;",
            "   - хронология ANN homeostasis: V2 -> V3 -> V3.1;",
            "   - семейство BIO compare-веток.",
            "",
            "5. Игровая среда и протокол прогона",
            "   - правила, состояние мира, сиды, повторы, шаги;",
            "   - различие adult/developmental;",
            "   - какие именно условия фиксировались между архитектурами.",
            "",
            "6. Метрики",
            "   - формула + интерпретация + ограничения + зачем выбрана;",
            "   - почему сравнение вообще оказывается нетривиальным именно на уровне метрик: satiety, homeostasis, developmental-режим, различие в режимах охоты;",
            "   - как вводились fairness-поправки и зачем понадобились отдельные исследовательские ветки V2, V3, V3.1, BIO_COMPARE и BIO_DUAL/FAST_COMPARE;",
            "   - performance, efficiency, state dependence, ignored visible prey, learning/development, correlations.",
            "",
            "7. Основные результаты",
            "   - компактные bar chart + error bars;",
            "   - boxplots по seed/repeat;",
            "   - learning curves;",
            "   - developmental curves;",
            "   - state dependence.",
            "",
            "8. История исследовательских веток",
            "   - зачем появились SATED, SOFT, V2, V3, V3.1;",
            "   - зачем появились COMPARE, DUAL, FAST, DUAL_FAST_COMPARE;",
            "   - где ветки помогли, а где породили перекосы.",
            "",
            "9. Интерпретация результатов через архитектуру",
            "   - почему ANN производительнее;",
            "   - почему текущая SNN-реализация проиграла: где именно её ограничили кодирование, разреженность спайков, схема обучения или декодирование действия;",
            "   - почему BIO понятнее по внутренним сигналам, но дороже и менее стабилен;",
            "   - почему честное сравнение сильно зависит от homeostatic режима.",
            "",
            "10. Общий вывод",
            "   - эффективность;",
            "   - биологическая правдоподобность;",
            "   - интерпретируемость.",
            "",
            "11. Ограничения исследования",
            "   - разные family-ветки сводились из разных прогонов;",
            "   - не все сигналы симметричны между ANN/SNN/BIO;",
            "   - homeostatic fairness сама была предметом исследования.",
        ]
    )


def build_technical_outline() -> str:
    return "\n".join(
        [
            "# Каркас подробного технического отчёта",
            "",
            "1. Инвентаризация всех использованных папок, версий и study-директорий.",
            "2. Полная схема архитектур и соответствие папок реальным экспериментальным веткам.",
            "3. Подробный разбор ANN, SNN и BIO по отдельности: входы, внутренние состояния, обучение, выходы, вычислительная цена.",
            "4. Подробный разбор сигналов, состояний и нейропараметров.",
            "5. Отдельный технический разбор SNN: кодирование, мембранная динамика, спайковая разреженность, decoding bottleneck и причины просадки на benchmark.",
            "6. Подробный протокол benchmark.",
            "7. Подробный список метрик и точные формулы.",
            "8. Полные таблицы `mean ± std` по всем архитектурам и режимам.",
            "9. Pairwise significance tables.",
            "10. Полный разбор learning/developmental/state-dependence графиков.",
            "11. Хронология инженерных решений и промежуточных веток.",
            "12. Ограничения, угрозы валидности и воспроизводимость.",
        ]
    )


def build_appendix_outline() -> str:
    return "\n".join(
        [
            "# Каркас приложения",
            "",
            "1. Полные абсолютные таблицы по всем архитектурам.",
            "2. Все попарные статистические сравнения.",
            "3. Полный figure index.",
            "4. Все appendix-графики.",
            "5. Version chronology с комментариями по веткам.",
            "6. Команды воспроизведения и пути к исходным study-директориям.",
        ]
    )


def build_exec_summary_outline() -> str:
    return "\n".join(
        [
            "# Каркас executive summary",
            "",
            "1. Что сравнивали.",
            "2. Какой агент оказался самым результативным.",
            "3. Какой агент оказался самым биологически правдоподобным.",
            "4. Какой агент оказался наиболее интерпретируемым.",
            "5. Почему сравнение пришлось делать через дополнительные ветки fairness/homeostasis.",
            "6. Ключевой итог: какой компромисс сейчас выглядит лучшим.",
        ]
    )


def build_workbook_sheets(
    study_index_rows: Sequence[Dict[str, Any]],
    main_table_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    bio_rows: Sequence[Dict[str, Any]],
    ann_history_rows: Sequence[Dict[str, Any]],
    significance_rows: Sequence[Dict[str, Any]],
    figure_index_rows: Sequence[Dict[str, Any]],
) -> List[Tuple[str, List[List[Any]]]]:
    def matrix_from_rows(rows: Sequence[Dict[str, Any]]) -> List[List[Any]]:
        if not rows:
            return [["Нет данных"]]
        headers = list(rows[0].keys())
        matrix = [headers]
        for row in rows:
            matrix.append([row.get(header, "") for header in headers])
        return matrix

    return [
        ("StudyIndex", matrix_from_rows(study_index_rows)),
        ("MainCompare", matrix_from_rows(main_table_rows)),
        ("BaselineANN_SNN", matrix_from_rows(baseline_rows)),
        ("BioFamily", matrix_from_rows(bio_rows)),
        ("ANNHistory", matrix_from_rows(ann_history_rows)),
        ("Significance", matrix_from_rows(significance_rows)),
        ("FigureIndex", matrix_from_rows(figure_index_rows)),
    ]


def column_name(index: int) -> str:
    result = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result.append(chr(65 + remainder))
    return "".join(reversed(result))


def build_sheet_xml(sheet_rows: Sequence[Sequence[Any]]) -> str:
    rows_xml: List[str] = []
    max_cols = max((len(row) for row in sheet_rows), default=1)
    for row_index, row in enumerate(sheet_rows, start=1):
        cells: List[str] = []
        for col_index, value in enumerate(row, start=1):
            if value in (None, ""):
                continue
            cell_ref = f"{column_name(col_index)}{row_index}"
            if isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}" t="b"><v>{1 if value else 0}</v></c>')
                continue
            number_value = None
            if isinstance(value, (int, float)):
                number_value = value
            else:
                number_value = as_float(value)
            if number_value is not None and not isinstance(value, str):
                cells.append(f'<c r="{cell_ref}"><v>{number_value}</v></c>')
            else:
                text = html.escape(str(value))
                cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>')
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    dimension = f"A1:{column_name(max_cols)}{max(1, len(sheet_rows))}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        "<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews>"
        "<sheetFormatPr defaultRowHeight=\"15\"/>"
        f"<sheetData>{''.join(rows_xml)}</sheetData>"
        "</worksheet>"
    )


def build_xlsx(path: Path, sheets: Sequence[Tuple[str, List[List[Any]]]]) -> None:
    workbook_sheets = []
    workbook_rels = []
    content_type_overrides = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (sheet_name, rows) in enumerate(sheets, start=1):
            sheet_path = f"xl/worksheets/sheet{index}.xml"
            workbook_sheets.append(
                f'<sheet name="{html.escape(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
            )
            workbook_rels.append(
                f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
            )
            content_type_overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
            archive.writestr(sheet_path, build_sheet_xml(rows))

        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            f"{''.join(content_type_overrides)}</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>',
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<workbookPr/>'
            '<bookViews><workbookView/></bookViews>'
            f"<sheets>{''.join(workbook_sheets)}</sheets>"
            '</workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{''.join(workbook_rels)}"
            '</Relationships>',
        )
        archive.writestr(
            "docProps/core.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>Frog Architecture Study</dc:title>'
            '<dc:creator>Codex</dc:creator>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{datetime.now(UTC).isoformat()}</dcterms:created>'
            '</cp:coreProperties>',
        )
        archive.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            '<Application>Codex</Application>'
            f'<TitlesOfParts><vt:vector size="{len(sheets)}" baseType="lpstr">{"".join(f"<vt:lpstr>{html.escape(name)}</vt:lpstr>" for name, _ in sheets)}</vt:vector></TitlesOfParts>'
            f'<HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>{len(sheets)}</vt:i4></vt:variant></vt:vector></HeadingPairs>'
            '</Properties>',
        )


def validate_xml_file(path: Path) -> None:
    ET.fromstring(path.read_text(encoding="utf-8"))


def build_outputs() -> Dict[str, Path]:
    ensure_dir(OUTPUT_ROOT)
    data_dir = ensure_dir(OUTPUT_ROOT / "data")
    tables_dir = ensure_dir(OUTPUT_ROOT / "tables")
    figures_main_dir = ensure_dir(OUTPUT_ROOT / "figures" / "main")
    figures_appendix_dir = ensure_dir(OUTPUT_ROOT / "figures" / "appendix")
    workbook_dir = ensure_dir(OUTPUT_ROOT / "workbook")
    outlines_dir = ensure_dir(OUTPUT_ROOT / "outlines")

    study_index_rows, aggregate_rows, run_rows, seed_rows, time_rows = load_sources()

    save_csv(data_dir / "study_index.csv", study_index_rows)
    save_csv(data_dir / "combined_aggregate_metrics.csv", aggregate_rows)
    save_csv(data_dir / "combined_run_metrics.csv", run_rows)
    save_csv(data_dir / "combined_seed_summary.csv", seed_rows)
    save_gzip_csv(data_dir / "combined_time_series.csv.gz", time_rows)
    save_text(data_dir / "study_index.json", json.dumps(study_index_rows, ensure_ascii=False, indent=2))

    main_table_rows = build_summary_table(aggregate_rows, FINAL_MAIN_ARCHES, "Основное сравнение")
    baseline_rows = build_summary_table(aggregate_rows, ANN_SNN_BASELINE_ARCHES, "Базовые ANN/SNN")
    bio_rows = build_summary_table(aggregate_rows, BIO_COMPARE_ARCHES, "BIO compare family")
    ann_history_rows = build_summary_table(aggregate_rows, ANN_HOMEOSTASIS_ARCHES, "ANN homeostasis chronology")

    save_csv(tables_dir / "main_comparison_summary.csv", main_table_rows)
    save_csv(tables_dir / "ann_snn_baseline_summary.csv", baseline_rows)
    save_csv(tables_dir / "bio_family_summary.csv", bio_rows)
    save_csv(tables_dir / "ann_homeostasis_summary.csv", ann_history_rows)

    save_text(tables_dir / "main_comparison_summary.md", markdown_table(main_table_rows))
    save_text(tables_dir / "ann_snn_baseline_summary.md", markdown_table(baseline_rows))
    save_text(tables_dir / "bio_family_summary.md", markdown_table(bio_rows))
    save_text(tables_dir / "ann_homeostasis_summary.md", markdown_table(ann_history_rows))

    html_sections = [
        "<html><head><meta charset='utf-8'><style>body{font-family:'Segoe UI',Arial,sans-serif;margin:24px;color:#1f2937}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #d1d5db;padding:8px;vertical-align:top}th{background:#f3f4f6;text-align:left}h1{margin-bottom:8px}p{color:#4b5563}</style></head><body>",
        "<h1>Человекочитаемые сводные таблицы по лягушке</h1>",
        "<p>Таблицы собраны автоматически из завершённых benchmark-прогонов и используют формат mean ± std.</p>",
        html_table("Основное сравнение", main_table_rows),
        html_table("Базовые ANN/SNN", baseline_rows),
        html_table("BIO compare family", bio_rows),
        html_table("ANN homeostasis chronology", ann_history_rows),
        "</body></html>",
    ]
    save_text(tables_dir / "tables_overview.html", "".join(html_sections))

    significance_rows = []
    significance_rows.extend(pairwise_tests(run_rows, FINAL_MAIN_ARCHES, "adult", KEY_METRICS, "final_main"))
    significance_rows.extend(pairwise_tests(run_rows, FINAL_MAIN_ARCHES, "developmental", KEY_METRICS, "final_main"))
    significance_rows.extend(pairwise_tests(run_rows, BIO_COMPARE_ARCHES, "adult", KEY_METRICS, "bio_family"))
    significance_rows.extend(pairwise_tests(run_rows, BIO_COMPARE_ARCHES, "developmental", KEY_METRICS, "bio_family"))
    significance_rows.extend(pairwise_tests(run_rows, ANN_HOMEOSTASIS_ARCHES, "adult", KEY_METRICS, "ann_history"))
    significance_rows.extend(pairwise_tests(run_rows, ANN_HOMEOSTASIS_ARCHES, "developmental", KEY_METRICS, "ann_history"))

    significance_table_rows = [
        {
            "Группа": row["group"],
            "Режим": row["mode"],
            "Метрика": row["metric_label_ru"],
            "Сравнение": f"{row['left_label_ru']} vs {row['right_label_ru']}",
            "Mean diff (L-R)": "" if row["mean_difference_left_minus_right"] is None else f"{row['mean_difference_left_minus_right']:.4f}",
            "Cliff's delta": "" if row["cliffs_delta"] is None else f"{row['cliffs_delta']:.4f}",
            "p raw": format_p_value(row.get("p_raw")),
            "p Holm": format_p_value(row.get("p_holm")),
            "significant<0.05": row.get("significant_05", False),
            "Предпочтительная архитектура": row["preferred_arch"],
        }
        for row in significance_rows
    ]
    save_csv(tables_dir / "pairwise_significance.csv", significance_table_rows)
    save_text(tables_dir / "pairwise_significance.md", markdown_table(significance_table_rows[:120]))

    final_main_agg = select_rows(aggregate_rows, FINAL_MAIN_ARCHES, ("adult", "developmental"))
    catch_panels = []
    first_catch_panels = []
    energy_panels = []
    ignored_panels = []
    state_panels = []
    for mode in ("adult", "developmental"):
        rows = [row for row in final_main_agg if row["mode"] == mode]
        catch_panels.append(
            (
                mode,
                [
                    (
                        row["arch_short_label"],
                        as_float(row.get("catch_rate_per_minute_mean")) or 0.0,
                        as_float(row.get("catch_rate_per_minute_std")) or 0.0,
                        ARCH_COLORS.get(row["arch"], "#888888"),
                    )
                    for row in rows
                ],
            )
        )
        first_catch_panels.append(
            (
                mode,
                [
                    (
                        row["arch_short_label"],
                        as_float(row.get("time_to_first_catch_s_mean")) or 0.0,
                        as_float(row.get("time_to_first_catch_s_std")) or 0.0,
                        ARCH_COLORS.get(row["arch"], "#888888"),
                    )
                    for row in rows
                ],
            )
        )
        energy_panels.append(
            (
                mode,
                [
                    (
                        row["arch_short_label"],
                        as_float(row.get("flies_per_energy_spent_mean")) or 0.0,
                        as_float(row.get("flies_per_energy_spent_std")) or 0.0,
                        ARCH_COLORS.get(row["arch"], "#888888"),
                    )
                    for row in rows
                ],
            )
        )
        ignored_panels.append(
            (
                mode,
                [
                    (
                        row["arch_short_label"],
                        as_float(row.get("visible_but_ignored_ratio_mean")) or 0.0,
                        as_float(row.get("visible_but_ignored_ratio_std")) or 0.0,
                        ARCH_COLORS.get(row["arch"], "#888888"),
                    )
                    for row in rows
                ],
            )
        )
        state_panels.append(
            (
                f"{mode}: low vs high energy catch rate",
                [
                    (
                        f"{row['arch_short_label']} low",
                        as_float(row.get("catch_rate_low_energy_mean")) or 0.0,
                        as_float(row.get("catch_rate_low_energy_std")) or 0.0,
                        ARCH_COLORS.get(row["arch"], "#888888"),
                    )
                    for row in rows
                ]
                + [
                    (
                        f"{row['arch_short_label']} high",
                        as_float(row.get("catch_rate_high_energy_mean")) or 0.0,
                        as_float(row.get("catch_rate_high_energy_std")) or 0.0,
                        "#cbd5e1",
                    )
                    for row in rows
                ],
            )
        )

    bar_chart_svg(catch_panels, "Основное сравнение: темп ловли", "Финальный триплет ANN V3.1 / SNN / BIO dual fast compare", "мух/мин", figures_main_dir / "main_catch_rate.svg")
    bar_chart_svg(first_catch_panels, "Основное сравнение: время до первой поимки", "Чем ниже, тем быстрее архитектура выходит на результат", "секунды", figures_main_dir / "main_first_catch.svg")
    bar_chart_svg(energy_panels, "Основное сравнение: энергоэффективность", "Мух на единицу потраченной энергии", "мух/энергию", figures_main_dir / "main_energy_efficiency.svg")
    bar_chart_svg(ignored_panels, "Основное сравнение: игнорирование видимой добычи", "Ниже лучше", "доля", figures_main_dir / "main_ignored_visible_prey.svg")
    bar_chart_svg(state_panels, "Основное сравнение: state dependence", "Low/high energy сравниваются отдельно", "мух/мин", figures_main_dir / "main_state_dependence.svg")

    final_main_learning = []
    for mode in ("adult", "developmental"):
        series = build_time_series_mean(time_rows, FINAL_MAIN_ARCHES, mode, "catch_rate_per_minute")
        final_main_learning.append(
            (
                mode,
                [
                    (meta_for_arch(arch).short_label, points, ARCH_COLORS.get(arch, "#888888"))
                    for arch, points in sorted(series.items(), key=lambda item: meta_for_arch(item[0]).chronology_order)
                ],
            )
        )
    line_chart_svg(final_main_learning, "Learning curves: основное сравнение", "Средний темп ловли по времени", "мух/мин", figures_main_dir / "main_learning_curves.svg")

    bio_dev_series = []
    for mode in ("developmental",):
        series = build_time_series_mean(time_rows, BIO_COMPARE_ARCHES, mode, "juvenile_progress")
        bio_dev_series.append(
            (
                mode,
                [
                    (meta_for_arch(arch).short_label, points, ARCH_COLORS.get(arch, "#888888"))
                    for arch, points in sorted(series.items(), key=lambda item: meta_for_arch(item[0]).chronology_order)
                ],
            )
        )
    line_chart_svg(
        bio_dev_series,
        "Developmental curves: BIO family",
        "Средняя динамика juvenile_progress по шагам",
        "juvenile_progress",
        figures_main_dir / "bio_developmental_curves.svg",
    )

    box_panels = []
    for mode in ("adult", "developmental"):
        items = []
        for arch in FINAL_MAIN_ARCHES:
            values = [as_float(row.get("catch_rate_per_minute")) for row in run_rows if row.get("arch") == arch and row.get("mode") == mode]
            values = [value for value in values if value is not None]
            items.append((meta_for_arch(arch).short_label, values, ARCH_COLORS.get(arch, "#888888")))
        box_panels.append((mode, items))
    boxplot_svg(box_panels, "Распределения catch rate по seed/repeat", "Boxplot по полным run-level метрикам", "мух/мин", figures_main_dir / "main_boxplot_catch_rate.svg")

    hunger_corr_runs = correlation_by_run(time_rows, BIO_COMPARE_ARCHES, "adult", "hunger_bias", "strike_drive", "hunger_bias vs strike_drive")
    hunger_corr_runs += correlation_by_run(time_rows, BIO_COMPARE_ARCHES, "developmental", "hunger_bias", "strike_drive", "hunger_bias vs strike_drive")
    hunger_corr_table = aggregate_correlation_rows(hunger_corr_runs)
    save_csv(tables_dir / "bio_hunger_strike_correlations.csv", hunger_corr_table)
    save_text(tables_dir / "bio_hunger_strike_correlations.md", markdown_table(hunger_corr_table))
    corr_panels = []
    for mode in ("adult", "developmental"):
        items = []
        for row in hunger_corr_table:
            if row["Режим"] != mode:
                continue
            arch_name = next(
                arch
                for arch, meta in ARCH_META.items()
                if meta.label_ru == row["Архитектура"]
            )
            mean_part = str(row["Корреляция, mean ± std"]).split(" ± ")[0]
            items.append((meta_for_arch(arch_name).short_label, float(mean_part), 0.0, ARCH_COLORS.get(arch_name, "#888888")))
        corr_panels.append((mode, items))
    bar_chart_svg(
        corr_panels,
        "BIO family: корреляция hunger_bias / strike_drive",
        "Используется Spearman по run-level временным рядам",
        "rho",
        figures_appendix_dir / "bio_hunger_strike_correlations.svg",
    )

    bio_family_panels = []
    for mode in ("adult", "developmental"):
        rows = [row for row in select_rows(aggregate_rows, BIO_COMPARE_ARCHES) if row["mode"] == mode]
        bio_family_panels.append(
            (
                mode,
                [
                    (
                        row["arch_short_label"],
                        as_float(row.get("catch_rate_per_minute_mean")) or 0.0,
                        as_float(row.get("catch_rate_per_minute_std")) or 0.0,
                        ARCH_COLORS.get(row["arch"], "#888888"),
                    )
                    for row in rows
                ],
            )
        )
    bar_chart_svg(bio_family_panels, "BIO family: темп ловли", "Сравнение compare-вариантов", "мух/мин", figures_appendix_dir / "bio_family_catch_rate.svg")

    ann_history_panels = []
    for mode in ("adult", "developmental"):
        rows = [row for row in select_rows(aggregate_rows, ANN_HOMEOSTASIS_ARCHES) if row["mode"] == mode]
        ann_history_panels.append(
            (
                mode,
                [
                    (
                        row["arch_short_label"],
                        as_float(row.get("catch_rate_per_minute_mean")) or 0.0,
                        as_float(row.get("catch_rate_per_minute_std")) or 0.0,
                        ARCH_COLORS.get(row["arch"], "#888888"),
                    )
                    for row in rows
                ],
            )
        )
    bar_chart_svg(
        ann_history_panels,
        "ANN homeostasis chronology: темп ловли",
        "V2 -> V3 -> V3.1 и frozen-аналоги",
        "мух/мин",
        figures_appendix_dir / "ann_homeostasis_catch_rate.svg",
    )

    ann_history_delta_panels = []
    for mode in ("adult", "developmental"):
        rows = [row for row in select_rows(aggregate_rows, ANN_HOMEOSTASIS_ARCHES) if row["mode"] == mode]
        ann_history_delta_panels.append(
            (
                mode,
                [
                    (
                        row["arch_short_label"],
                        as_float(row.get("homeostatic_catch_delta_mean")) or 0.0,
                        as_float(row.get("homeostatic_catch_delta_std")) or 0.0,
                        ARCH_COLORS.get(row["arch"], "#888888"),
                    )
                    for row in rows
                ],
            )
        )
    bar_chart_svg(
        ann_history_delta_panels,
        "ANN homeostasis chronology: low-high energy delta",
        "Положительное значение означает более активную охоту при дефиците энергии",
        "delta",
        figures_appendix_dir / "ann_homeostasis_state_delta.svg",
    )

    timeline_svg(figures_main_dir / "version_chronology.svg")

    figure_index_rows = []
    for figure_path in sorted((OUTPUT_ROOT / "figures").rglob("*.svg")):
        figure_index_rows.append(
            {
                "Файл": figure_path.name,
                "Путь": str(figure_path),
                "Раздел": "main" if "figures\\main" in str(figure_path) else "appendix",
            }
        )
    save_csv(data_dir / "figure_index.csv", figure_index_rows)
    save_text(tables_dir / "figure_index.md", markdown_table(figure_index_rows))

    workbook_path = workbook_dir / "frog_study_tables.xlsx"
    workbook_sheets = build_workbook_sheets(
        study_index_rows,
        main_table_rows,
        baseline_rows,
        bio_rows,
        ann_history_rows,
        significance_table_rows,
        figure_index_rows,
    )
    build_xlsx(workbook_path, workbook_sheets)

    outlines = {
        outlines_dir / "article_outline.md": build_article_outline(),
        outlines_dir / "technical_report_outline.md": build_technical_outline(),
        outlines_dir / "appendix_outline.md": build_appendix_outline(),
        outlines_dir / "executive_summary_outline.md": build_exec_summary_outline(),
    }
    for outline_path, content in outlines.items():
        save_text(outline_path, content)

    save_text(OUTPUT_ROOT / "README.md", build_readme())

    index_html = (
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:'Segoe UI',Arial,sans-serif;margin:24px;color:#1f2937}"
        "a{color:#0f5cb6;text-decoration:none}"
        "a:hover{text-decoration:underline}"
        "img{max-width:100%;border:1px solid #d1d5db;margin:16px 0}"
        "table{border-collapse:collapse;width:100%;margin-top:16px}"
        "th,td{border:1px solid #d1d5db;padding:8px;text-align:left}"
        "th{background:#f3f4f6}"
        "</style></head><body>"
        "<h1>Пакет артефактов для статьи по лягушке</h1>"
        "<p>Это стартовый набор таблиц, графиков и индексов для статьи, техотчёта и приложения.</p>"
        "<ul>"
        f"<li><a href='./tables/tables_overview.html'>HTML-таблицы</a></li>"
        f"<li><a href='./workbook/{workbook_path.name}'>Excel-книга</a></li>"
        f"<li><a href='./outlines/article_outline.md'>Каркас статьи</a></li>"
        "</ul>"
        "<h2>Ключевые графики</h2>"
        f"<img src='./figures/main/main_catch_rate.svg' alt='main catch rate'/>"
        f"<img src='./figures/main/main_energy_efficiency.svg' alt='main energy efficiency'/>"
        f"<img src='./figures/main/main_learning_curves.svg' alt='main learning curves'/>"
        f"<img src='./figures/main/version_chronology.svg' alt='version chronology'/>"
        "</body></html>"
    )
    save_text(OUTPUT_ROOT / "index.html", index_html)

    # Quick XML sanity checks for the generated SVG/HTML-oriented assets.
    for svg_path in sorted((OUTPUT_ROOT / "figures").rglob("*.svg")):
        validate_xml_file(svg_path)

    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_root": str(OUTPUT_ROOT),
        "sources": [source.relative_dir for source in STUDY_SOURCES],
        "final_main_arches": list(FINAL_MAIN_ARCHES),
        "outputs": {
            "readme": str(OUTPUT_ROOT / "README.md"),
            "index_html": str(OUTPUT_ROOT / "index.html"),
            "workbook": str(workbook_path),
            "tables_html": str(tables_dir / "tables_overview.html"),
            "main_catch_rate": str(figures_main_dir / "main_catch_rate.svg"),
            "version_chronology": str(figures_main_dir / "version_chronology.svg"),
        },
    }
    save_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    return {
        "output_root": OUTPUT_ROOT,
        "workbook": workbook_path,
        "index_html": OUTPUT_ROOT / "index.html",
        "main_table_md": tables_dir / "main_comparison_summary.md",
        "significance_csv": tables_dir / "pairwise_significance.csv",
        "main_catch_rate": figures_main_dir / "main_catch_rate.svg",
        "learning_curves": figures_main_dir / "main_learning_curves.svg",
        "version_chronology": figures_main_dir / "version_chronology.svg",
    }


def main() -> None:
    outputs = build_outputs()
    print("Article assets generated:")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
