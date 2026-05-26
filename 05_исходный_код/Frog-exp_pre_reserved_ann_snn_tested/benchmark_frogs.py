#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None

from frog_lib_ann.simulation import ANNFlyCatchingSimulation
from frog_lib_ann_frozen.simulation import ANNFlyCatchingSimulation as ANNFrozenSimulation
from frog_lib_snn.simulation import SNNFlyCatchingSimulation
from frog_lib_snn_frozen.simulation import SNNFlyCatchingSimulation as SNNFrozenSimulation
from Frog_predator_neuro.simulation import Simulation as BioSimulation
from Frog_predator_neuro_dual.simulation import Simulation as BioDualSimulation


ARCHITECTURES: Tuple[str, ...] = ("ANN", "ANN_FROZEN", "SNN", "SNN_FROZEN", "BIO", "BIO_DUAL")
MODES: Dict[str, bool] = {
    "adult": False,
    "developmental": True,
}

RUN_METRIC_FIELDS: List[str] = [
    "arch",
    "mode",
    "training_mode",
    "spawn_seed",
    "repeat",
    "steps_requested",
    "steps_executed",
    "dt",
    "sim_time_s",
    "wall_clock_s",
    "ms_per_step",
    "catches",
    "strike_attempts",
    "successful_strike_episodes",
    "captures_per_strike_attempt",
    "capture_success",
    "false_strike_rate",
    "catch_rate_per_1k_steps",
    "catch_rate_per_minute",
    "time_to_first_catch_s",
    "time_to_competence_s",
    "learning_auc",
    "ttc_mean_s",
    "ttc_median_s",
    "reaction_latency_mean_s",
    "reaction_latency_median_s",
    "energy_initial",
    "energy_final",
    "energy_spent_est",
    "catch_rate_low_energy",
    "catch_rate_mid_energy",
    "catch_rate_high_energy",
    "strike_rate_low_energy",
    "strike_rate_mid_energy",
    "strike_rate_high_energy",
    "flies_per_energy_spent",
    "net_energy_balance_per_catch",
    "path_length_px",
    "movement_efficiency_px_per_catch",
    "avg_speed_px_s",
    "avg_alignment",
    "avg_controller_signal",
    "avg_neural_activity",
    "avg_acceleration_px_s2",
    "avg_jerk_px_s3",
    "tongue_usage_ratio",
    "visible_time_s",
    "visible_but_ignored_ratio",
    "visible_but_ignored_ratio_low_energy",
    "visible_but_ignored_ratio_high_energy",
    "energy_deficit_strike_correlation",
    "strike_opportunity_count",
    "strike_opportunity_conversion",
    "shot_distance_mean_px",
    "shot_distance_std_px",
    "shot_distance_p10_px",
    "shot_distance_p50_px",
    "shot_distance_p90_px",
    "pre_shot_positioning_error_mean_px",
    "pre_shot_positioning_error_p90_px",
    "survived",
    "compute_per_catch_s",
    "avg_visible_target_count",
    "seeded_spawn_stream",
    "model_seed",
    "runtime_seed",
    "adult_reached",
    "adult_step",
    "adult_time_s",
    "catches_before_maturity",
    "avg_juvenile_progress",
    "avg_maturity_readiness",
    "avg_maturity_stability",
    "avg_food_prediction_error",
    "avg_hunger_bias",
    "avg_reward_seek_bias",
    "avg_predation_bias",
    "avg_prey_permission",
    "avg_fast_target_lock",
    "avg_fast_loop_gate",
    "avg_fast_strike_drive",
    "avg_effective_motor_gate",
    "avg_bg_gating_signal",
    "avg_spike_rate",
    "avg_spike_count",
    "burst_ratio",
    "silent_ratio",
    "spikes_per_catch",
    "avg_eligibility_norm",
    "avg_learning_reward",
    "avg_actor_advantage",
    "avg_value_estimate",
]

TIME_SERIES_FIELDS: List[str] = [
    "arch",
    "mode",
    "spawn_seed",
    "repeat",
    "step",
    "time_s",
    "catches",
    "catch_rate_per_minute",
    "energy",
    "energy_ratio",
    "controller_signal",
    "neural_activity",
    "alignment",
    "speed_px_s",
    "target_distance_px",
    "visibility",
    "visible_target_count",
    "strike_drive",
    "strike_intent",
    "strike_readiness",
    "tongue_extended",
    "caught_step",
    "is_juvenile",
    "juvenile_progress",
    "maturity_readiness",
    "maturity_stability",
    "spike_count",
    "eligibility_norm",
    "learning_reward",
    "actor_advantage",
    "value_estimate",
    "food_prediction_error",
    "hunger_bias",
    "reward_seek_bias",
    "predation_bias",
    "prey_permission",
    "fast_target_lock",
    "fast_loop_gate",
    "fast_strike_drive",
    "bg_gating_signal",
    "effective_motor_gate",
]

EVENT_FIELDS: List[str] = [
    "arch",
    "mode",
    "spawn_seed",
    "repeat",
    "step",
    "time_s",
    "event_type",
    "catches",
    "energy",
    "target_distance_px",
    "controller_signal",
    "strike_drive",
    "visibility",
    "visible_target_count",
    "is_juvenile",
    "detail",
]

SUMMARY_METRICS: Tuple[str, ...] = (
    "catches",
    "catch_rate_per_minute",
    "capture_success",
    "time_to_first_catch_s",
    "time_to_competence_s",
    "learning_auc",
    "flies_per_energy_spent",
    "net_energy_balance_per_catch",
    "movement_efficiency_px_per_catch",
    "visible_but_ignored_ratio",
    "avg_acceleration_px_s2",
    "avg_jerk_px_s3",
    "tongue_usage_ratio",
    "reaction_latency_mean_s",
    "ttc_mean_s",
    "compute_per_catch_s",
    "catch_rate_low_energy",
    "catch_rate_mid_energy",
    "catch_rate_high_energy",
    "strike_rate_low_energy",
    "strike_rate_mid_energy",
    "strike_rate_high_energy",
    "wall_clock_s",
    "adult_time_s",
    "avg_spike_rate",
    "avg_spike_count",
    "spikes_per_catch",
    "avg_food_prediction_error",
    "avg_hunger_bias",
    "avg_reward_seek_bias",
    "avg_predation_bias",
    "avg_prey_permission",
    "avg_fast_target_lock",
    "avg_fast_loop_gate",
    "avg_fast_strike_drive",
    "avg_maturity_readiness",
    "avg_maturity_stability",
)


@dataclass(frozen=True)
class BenchmarkTask:
    arch: str
    mode: str
    spawn_seed: int
    repeat: int
    steps: int
    sample_interval: int
    competence_catches: int


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def configure_numpy_and_torch(seed: int) -> None:
    np.random.seed(seed & 0xFFFFFFFF)
    if torch is not None:
        torch.manual_seed(seed & 0xFFFFFFFF)
        if torch.cuda.is_available():  # pragma: no cover - GPU is optional
            torch.cuda.manual_seed_all(seed & 0xFFFFFFFF)


def safe_mean(values: Sequence[float]) -> Optional[float]:
    usable = [
        float(value)
        for value in values
        if value not in (None, "", "None") and not math.isnan(float(value))
    ]
    if not usable:
        return None
    return float(statistics.fmean(usable))


def safe_median(values: Sequence[float]) -> Optional[float]:
    usable = [
        float(value)
        for value in values
        if value not in (None, "", "None") and not math.isnan(float(value))
    ]
    if not usable:
        return None
    return float(statistics.median(usable))


def safe_std(values: Sequence[float]) -> Optional[float]:
    usable = [
        float(value)
        for value in values
        if value not in (None, "", "None") and not math.isnan(float(value))
    ]
    if len(usable) < 2:
        return 0.0 if usable else None
    return float(statistics.pstdev(usable))


def safe_quantile(values: Sequence[float], q: float) -> Optional[float]:
    usable = sorted(
        float(value)
        for value in values
        if value not in (None, "", "None") and not math.isnan(float(value))
    )
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]
    position = (len(usable) - 1) * q
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return usable[left]
    fraction = position - left
    return float(usable[left] * (1.0 - fraction) + usable[right] * fraction)


def confidence_interval_95(values: Sequence[float]) -> Optional[float]:
    usable = [
        float(value)
        for value in values
        if value not in (None, "", "None") and not math.isnan(float(value))
    ]
    if len(usable) < 2:
        return 0.0 if usable else None
    return float(1.96 * statistics.pstdev(usable) / math.sqrt(len(usable)))


def bool_to_float(value: bool) -> float:
    return 1.0 if value else 0.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def safe_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    usable: List[Tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        x_value = float(x)
        y_value = float(y)
        if math.isnan(x_value) or math.isnan(y_value):
            continue
        usable.append((x_value, y_value))
    if len(usable) < 2:
        return None
    x_values = np.array([pair[0] for pair in usable], dtype=float)
    y_values = np.array([pair[1] for pair in usable], dtype=float)
    if float(np.std(x_values)) <= 1e-9 or float(np.std(y_values)) <= 1e-9:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def energy_bucket_name(energy_ratio: float) -> str:
    if energy_ratio < 0.40:
        return "low"
    if energy_ratio < 0.80:
        return "mid"
    return "high"


def architecture_family(arch: str) -> str:
    if arch in {"ANN", "ANN_FROZEN"}:
        return "ANN"
    if arch in {"SNN", "SNN_FROZEN"}:
        return "SNN"
    if arch in {"BIO", "BIO_DUAL"}:
        return "BIO"
    return arch


def instantiate_simulation(arch: str, training_mode: bool, model_seed: int):
    configure_numpy_and_torch(model_seed)
    if arch == "ANN":
        return ANNFlyCatchingSimulation(headless=True, training_mode=training_mode)
    if arch == "ANN_FROZEN":
        return ANNFrozenSimulation(headless=True, training_mode=training_mode)
    if arch == "SNN":
        return SNNFlyCatchingSimulation(headless=True, training_mode=training_mode)
    if arch == "SNN_FROZEN":
        return SNNFrozenSimulation(headless=True, training_mode=training_mode)
    if arch == "BIO":
        return BioSimulation(headless=True, training_mode=training_mode, brain_seed=model_seed)
    if arch == "BIO_DUAL":
        return BioDualSimulation(headless=True, training_mode=training_mode, brain_seed=model_seed)
    raise ValueError(f"Unsupported architecture: {arch}")


def primary_agent(sim: Any, arch: str) -> Any:
    if arch in {"ANN", "ANN_FROZEN", "SNN", "SNN_FROZEN"}:
        return sim.frog
    return sim.frogs[0]


def patch_spawn_stream(sim: Any, spawn_state: object) -> None:
    spawn_rng = random.Random()
    spawn_rng.setstate(spawn_state)

    def _spawn_position() -> Tuple[float, float]:
        margin_x = min(100, max(20, sim.width // 5))
        margin_y = min(100, max(20, sim.height // 5))
        return (
            spawn_rng.uniform(margin_x, max(margin_x, sim.width - margin_x)),
            spawn_rng.uniform(margin_y, max(margin_y, sim.height - margin_y)),
        )

    sim._spawn_position = _spawn_position


def catch_distance_for_agent(agent: Any) -> float:
    if hasattr(agent, "catch_distance"):
        return float(agent.catch_distance)
    shape_radius = float(getattr(getattr(agent, "shape", None), "radius", 0.0))
    tongue_reach = float(getattr(agent, "tongue_reach", 0.0))
    hit_radius = float(getattr(agent, "hit_radius", 0.0))
    return shape_radius + tongue_reach + hit_radius


def step_simulation(sim: Any, arch: str) -> Dict[str, Any]:
    if arch in {"BIO", "BIO_DUAL"}:
        result = sim.step()
        agents = result.get("agents", [])
        if agents:
            return dict(agents[0])
        frog = sim.frogs[0]
        return {
            "position": np.array(frog.body.position, dtype=float),
            "velocity": np.zeros(2, dtype=float),
            "energy": float(frog.energy),
            "caught_flies": int(frog.caught_flies),
            "caught_fly": None,
            "controller_signal": 0.0,
            "alignment": 0.0,
            "target_distance": None,
            "neural_activity": 0.0,
            "tongue_extended": False,
            "tongue_length": 0.0,
            "strike_drive": 0.0,
            "strike_commitment": 0,
            "strike_cooldown": 0.0,
            "strike_readiness": 0.0,
            "visible_target_count": 0,
            "focus_x": None,
            "focus_y": None,
            "focus_vx": None,
            "focus_vy": None,
            "focus_distance": None,
            "food_prediction_error": float(frog.last_brain_output.get("food_prediction_error", 0.0)),
            "motivation_context": dict(frog.last_brain_output.get("motivation_context", {}) or {}),
            "is_juvenile": bool(frog.brain.is_juvenile),
            "juvenile_progress": float(frog.last_brain_output.get("juvenile_progress", 0.0)),
            "maturity_readiness": float(frog.last_brain_output.get("maturity_readiness", 0.0)),
            "maturity_stability": float(frog.last_brain_output.get("maturity_stability", 0.0)),
            "bg_gating_signal": float(frog.last_brain_output.get("bg_gating_signal", 0.0)),
            "effective_motor_gate": float(frog.last_brain_output.get("effective_motor_gate", 0.0)),
            "spike_count": None,
            "eligibility_norm": None,
            "learning_reward": None,
            "actor_advantage": None,
            "value_estimate": None,
        }
    return dict(sim.step())


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_gzip_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def aggregate_metric_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["arch"], row["mode"]), []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for (arch, mode), members in sorted(grouped.items()):
        aggregate: Dict[str, Any] = {
            "arch": arch,
            "mode": mode,
            "runs": len(members),
        }
        for metric in SUMMARY_METRICS:
            values = [member.get(metric) for member in members]
            aggregate[f"{metric}_mean"] = safe_mean(values)
            aggregate[f"{metric}_median"] = safe_median(values)
            aggregate[f"{metric}_std"] = safe_std(values)
            aggregate[f"{metric}_ci95"] = confidence_interval_95(values)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def aggregate_seed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["arch"], row["mode"], int(row["spawn_seed"])), []).append(row)

    seed_rows: List[Dict[str, Any]] = []
    for (arch, mode, spawn_seed), members in sorted(grouped.items()):
        record: Dict[str, Any] = {
            "arch": arch,
            "mode": mode,
            "spawn_seed": spawn_seed,
            "repeats": len(members),
        }
        for metric in SUMMARY_METRICS:
            values = [member.get(metric) for member in members]
            record[f"{metric}_mean"] = safe_mean(values)
            record[f"{metric}_std"] = safe_std(values)
        seed_rows.append(record)
    return seed_rows


def run_single_task(task: BenchmarkTask) -> Dict[str, Any]:
    training_mode = MODES[task.mode]
    model_seed = stable_seed("model", architecture_family(task.arch))
    runtime_seed = stable_seed("runtime", task.arch, task.mode, task.spawn_seed, task.repeat)

    random.seed(task.spawn_seed)
    configure_numpy_and_torch(model_seed)
    sim = instantiate_simulation(task.arch, training_mode, model_seed)
    spawn_state = random.getstate()
    patch_spawn_stream(sim, spawn_state)

    random.seed(runtime_seed)
    configure_numpy_and_torch(runtime_seed)

    agent = primary_agent(sim, task.arch)
    dt = float(getattr(sim, "dt", 0.01))
    initial_energy = float(getattr(agent, "energy", 0.0))
    max_energy = max(initial_energy, float(getattr(agent, "max_energy", initial_energy or 1.0)))
    catch_distance = catch_distance_for_agent(agent)

    catches = 0
    strike_attempts = 0
    successful_strike_episodes = 0
    tongue_catches = 0
    visible_steps = 0
    ignored_visible_steps = 0
    strike_window_episodes = 0
    bucket_seconds = {"low": 0.0, "mid": 0.0, "high": 0.0}
    bucket_catches = {"low": 0, "mid": 0, "high": 0}
    bucket_strikes = {"low": 0, "mid": 0, "high": 0}
    bucket_visible_steps = {"low": 0, "mid": 0, "high": 0}
    bucket_ignored_steps = {"low": 0, "mid": 0, "high": 0}
    first_catch_step: Optional[int] = None
    competence_step: Optional[int] = None
    adult_step: Optional[int] = None
    catches_before_maturity: Optional[int] = None
    detection_start_step: Optional[int] = None
    strike_window_start_step: Optional[int] = None
    previous_tongue_extended = False
    previous_in_strike_window = False
    strike_episode_active = False
    strike_episode_succeeded = False
    previous_velocity = np.zeros(2, dtype=float)
    previous_acceleration = np.zeros(2, dtype=float)

    path_length = 0.0
    acceleration_samples: List[float] = []
    jerk_samples: List[float] = []
    ttc_samples: List[float] = []
    reaction_samples: List[float] = []
    shot_distances: List[float] = []
    positioning_errors: List[float] = []
    visible_target_counts: List[float] = []
    alignment_samples: List[float] = []
    controller_samples: List[float] = []
    neural_samples: List[float] = []
    juvenile_progress_samples: List[float] = []
    maturity_readiness_samples: List[float] = []
    maturity_stability_samples: List[float] = []
    food_prediction_error_samples: List[float] = []
    effective_motor_gate_samples: List[float] = []
    bg_gating_signal_samples: List[float] = []
    spike_rate_samples: List[float] = []
    spike_count_samples: List[float] = []
    eligibility_samples: List[float] = []
    learning_reward_samples: List[float] = []
    actor_advantage_samples: List[float] = []
    value_estimate_samples: List[float] = []
    hunger_bias_samples: List[float] = []
    reward_seek_bias_samples: List[float] = []
    predation_bias_samples: List[float] = []
    prey_permission_samples: List[float] = []
    fast_target_lock_samples: List[float] = []
    fast_loop_gate_samples: List[float] = []
    fast_strike_drive_samples: List[float] = []
    energy_deficit_samples: List[float] = []
    strike_propensity_samples: List[float] = []
    burst_steps = 0
    silent_steps = 0

    time_series_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    start_wall_clock = time.perf_counter()
    try:
        for step in range(1, task.steps + 1):
            state = step_simulation(sim, task.arch)
            current_time_s = step * dt

            velocity = np.array(state.get("velocity", (0.0, 0.0)), dtype=float)
            speed = float(np.linalg.norm(velocity))
            path_length += speed * dt

            acceleration = (velocity - previous_velocity) / max(dt, 1e-9)
            acceleration_samples.append(float(np.linalg.norm(acceleration)))
            if step > 1:
                jerk = (acceleration - previous_acceleration) / max(dt, 1e-9)
                jerk_samples.append(float(np.linalg.norm(jerk)))
            previous_velocity = velocity
            previous_acceleration = acceleration

            target_distance_raw = state.get("target_distance")
            target_distance = float(target_distance_raw) if target_distance_raw is not None else math.nan
            visible_target_count = int(state.get("visible_target_count", 0) or 0)
            current_energy = float(state.get("energy", 0.0))
            energy_ratio = clamp(current_energy / max(max_energy, 1e-9), 0.0, 1.2)
            energy_bucket = energy_bucket_name(energy_ratio)
            bucket_seconds[energy_bucket] += dt
            visibility = 1.0 if (visible_target_count > 0 or target_distance_raw is not None or state.get("visibility", 0.0) > 0.0) else 0.0
            if visibility > 0.0:
                visible_steps += 1
                bucket_visible_steps[energy_bucket] += 1
            if visibility > 0.0 and float(state.get("controller_signal", 0.0)) < 0.15 and not bool(state.get("tongue_extended", False)) and float(state.get("strike_drive", 0.0)) < 0.35:
                ignored_visible_steps += 1
                bucket_ignored_steps[energy_bucket] += 1

            if visibility > 0.0 and detection_start_step is None:
                detection_start_step = step
                event_rows.append(
                    {
                        "arch": task.arch,
                        "mode": task.mode,
                        "spawn_seed": task.spawn_seed,
                        "repeat": task.repeat,
                        "step": step,
                        "time_s": current_time_s,
                        "event_type": "first_visible",
                        "catches": catches,
                        "energy": float(state.get("energy", 0.0)),
                        "target_distance_px": target_distance_raw,
                        "controller_signal": float(state.get("controller_signal", 0.0)),
                        "strike_drive": float(state.get("strike_drive", 0.0)),
                        "visibility": visibility,
                        "visible_target_count": visible_target_count,
                        "is_juvenile": bool(state.get("is_juvenile", False)),
                        "detail": "",
                    }
                )
            elif visibility <= 0.0:
                detection_start_step = None

            catch_distance = catch_distance_for_agent(primary_agent(sim, task.arch))
            in_strike_window = visibility > 0.0 and not math.isnan(target_distance) and target_distance <= catch_distance * 1.10
            if in_strike_window and not previous_in_strike_window:
                strike_window_episodes += 1
                strike_window_start_step = step
            elif not in_strike_window:
                strike_window_start_step = None
            previous_in_strike_window = in_strike_window

            tongue_extended = bool(state.get("tongue_extended", False))
            caught = state.get("caught_fly") is not None
            strike_attempt = (tongue_extended and not previous_tongue_extended) or (caught and not previous_tongue_extended)
            if strike_attempt:
                strike_attempts += 1
                strike_episode_active = True
                strike_episode_succeeded = False
                bucket_strikes[energy_bucket] += 1
                if not math.isnan(target_distance):
                    shot_distances.append(target_distance)
                    positioning_errors.append(max(0.0, target_distance - catch_distance))
                if strike_window_start_step is not None:
                    reaction_samples.append((step - strike_window_start_step) * dt)
                event_rows.append(
                    {
                        "arch": task.arch,
                        "mode": task.mode,
                        "spawn_seed": task.spawn_seed,
                        "repeat": task.repeat,
                        "step": step,
                        "time_s": current_time_s,
                        "event_type": "strike_attempt",
                        "catches": catches,
                        "energy": float(state.get("energy", 0.0)),
                        "target_distance_px": target_distance_raw,
                        "controller_signal": float(state.get("controller_signal", 0.0)),
                        "strike_drive": float(state.get("strike_drive", 0.0)),
                        "visibility": visibility,
                        "visible_target_count": visible_target_count,
                        "is_juvenile": bool(state.get("is_juvenile", False)),
                        "detail": "",
                    }
                )
                strike_window_start_step = None

            if caught:
                catches += 1
                bucket_catches[energy_bucket] += 1
                if strike_episode_active and not strike_episode_succeeded:
                    successful_strike_episodes += 1
                    strike_episode_succeeded = True
                if first_catch_step is None:
                    first_catch_step = step
                if competence_step is None and catches >= task.competence_catches:
                    competence_step = step
                if detection_start_step is not None:
                    ttc_samples.append((step - detection_start_step) * dt)
                detection_start_step = None
                strike_window_start_step = None
                if tongue_extended or previous_tongue_extended:
                    tongue_catches += 1
                event_rows.append(
                    {
                        "arch": task.arch,
                        "mode": task.mode,
                        "spawn_seed": task.spawn_seed,
                        "repeat": task.repeat,
                        "step": step,
                        "time_s": current_time_s,
                        "event_type": "catch",
                        "catches": catches,
                        "energy": float(state.get("energy", 0.0)),
                        "target_distance_px": target_distance_raw,
                        "controller_signal": float(state.get("controller_signal", 0.0)),
                        "strike_drive": float(state.get("strike_drive", 0.0)),
                        "visibility": visibility,
                        "visible_target_count": visible_target_count,
                        "is_juvenile": bool(state.get("is_juvenile", False)),
                        "detail": "",
                    }
                )

            current_is_juvenile = bool(state.get("is_juvenile", False))
            if task.mode == "developmental" and adult_step is None and not current_is_juvenile:
                adult_step = step
                catches_before_maturity = catches
                event_rows.append(
                    {
                        "arch": task.arch,
                        "mode": task.mode,
                        "spawn_seed": task.spawn_seed,
                        "repeat": task.repeat,
                        "step": step,
                        "time_s": current_time_s,
                        "event_type": "maturity",
                        "catches": catches,
                        "energy": float(state.get("energy", 0.0)),
                        "target_distance_px": target_distance_raw,
                        "controller_signal": float(state.get("controller_signal", 0.0)),
                        "strike_drive": float(state.get("strike_drive", 0.0)),
                        "visibility": visibility,
                        "visible_target_count": visible_target_count,
                        "is_juvenile": current_is_juvenile,
                        "detail": "",
                    }
                )

            visible_target_counts.append(float(visible_target_count))
            alignment_samples.append(float(state.get("alignment", 0.0)))
            controller_samples.append(float(state.get("controller_signal", 0.0)))
            neural_samples.append(float(state.get("neural_activity", 0.0)))
            juvenile_progress_samples.append(float(state.get("juvenile_progress", 1.0 if not current_is_juvenile else 0.0)))
            maturity_readiness_samples.append(float(state.get("maturity_readiness", 0.0)))
            maturity_stability_samples.append(float(state.get("maturity_stability", 0.0)))
            food_prediction_error_samples.append(float(state.get("food_prediction_error", 0.0)))
            effective_motor_gate_samples.append(float(state.get("effective_motor_gate", 0.0)))
            bg_gating_signal_samples.append(float(state.get("bg_gating_signal", 0.0)))
            motivation_context = dict(state.get("motivation_context", {}) or {})
            hunger_bias = motivation_context.get("hunger_bias")
            reward_seek_bias = motivation_context.get("reward_seek_bias")
            predation_bias = motivation_context.get("predation_bias")
            if hunger_bias is not None:
                hunger_bias_samples.append(float(hunger_bias))
            if reward_seek_bias is not None:
                reward_seek_bias_samples.append(float(reward_seek_bias))
            if predation_bias is not None:
                predation_bias_samples.append(float(predation_bias))
            prey_permission = state.get("prey_permission")
            fast_target_lock = state.get("fast_target_lock")
            fast_loop_gate = state.get("fast_loop_gate")
            fast_strike_drive = state.get("fast_strike_drive")
            if prey_permission is not None:
                prey_permission_samples.append(float(prey_permission))
            if fast_target_lock is not None:
                fast_target_lock_samples.append(float(fast_target_lock))
            if fast_loop_gate is not None:
                fast_loop_gate_samples.append(float(fast_loop_gate))
            if fast_strike_drive is not None:
                fast_strike_drive_samples.append(float(fast_strike_drive))
            energy_deficit_samples.append(clamp(1.0 - energy_ratio, 0.0, 1.0))
            strike_propensity_samples.append(1.0 if strike_attempt else 0.0)

            spike_count_value = state.get("spike_count")
            if spike_count_value is not None:
                spike_count = float(spike_count_value)
                spike_count_samples.append(spike_count)
                spike_rate_samples.append(float(state.get("neural_activity", 0.0)))
                eligibility_samples.append(float(state.get("eligibility_norm", 0.0) or 0.0))
                if spike_count == 0.0:
                    silent_steps += 1
                if spike_count >= 2.0:
                    burst_steps += 1

            learning_reward = state.get("learning_reward")
            if learning_reward is not None:
                learning_reward_samples.append(float(learning_reward))
            actor_advantage = state.get("actor_advantage")
            if actor_advantage is not None:
                actor_advantage_samples.append(float(actor_advantage))
            value_estimate = state.get("value_estimate")
            if value_estimate is not None:
                value_estimate_samples.append(float(value_estimate))

            if step == 1 or step == task.steps or step % max(1, task.sample_interval) == 0:
                time_series_rows.append(
                    {
                        "arch": task.arch,
                        "mode": task.mode,
                        "spawn_seed": task.spawn_seed,
                        "repeat": task.repeat,
                        "step": step,
                        "time_s": current_time_s,
                        "catches": catches,
                        "catch_rate_per_minute": catches / max(current_time_s / 60.0, 1e-9),
                        "energy": current_energy,
                        "energy_ratio": energy_ratio,
                        "controller_signal": float(state.get("controller_signal", 0.0)),
                        "neural_activity": float(state.get("neural_activity", 0.0)),
                        "alignment": float(state.get("alignment", 0.0)),
                        "speed_px_s": speed,
                        "target_distance_px": target_distance_raw,
                        "visibility": visibility,
                        "visible_target_count": visible_target_count,
                        "strike_drive": float(state.get("strike_drive", 0.0)),
                        "strike_intent": float(state.get("strike_intent", 0.0) or 0.0),
                        "strike_readiness": float(state.get("strike_readiness", 0.0) or 0.0),
                        "tongue_extended": tongue_extended,
                        "caught_step": caught,
                        "is_juvenile": current_is_juvenile,
                        "juvenile_progress": float(state.get("juvenile_progress", 1.0 if not current_is_juvenile else 0.0)),
                        "maturity_readiness": float(state.get("maturity_readiness", 0.0)),
                        "maturity_stability": float(state.get("maturity_stability", 0.0)),
                        "spike_count": spike_count_value,
                        "eligibility_norm": state.get("eligibility_norm"),
                        "learning_reward": learning_reward,
                        "actor_advantage": actor_advantage,
                        "value_estimate": value_estimate,
                        "food_prediction_error": state.get("food_prediction_error"),
                        "hunger_bias": hunger_bias,
                        "reward_seek_bias": reward_seek_bias,
                        "predation_bias": predation_bias,
                        "prey_permission": prey_permission,
                        "fast_target_lock": fast_target_lock,
                        "fast_loop_gate": fast_loop_gate,
                        "fast_strike_drive": fast_strike_drive,
                        "bg_gating_signal": state.get("bg_gating_signal"),
                        "effective_motor_gate": state.get("effective_motor_gate"),
                    }
                )

            if strike_episode_active and not tongue_extended and previous_tongue_extended:
                strike_episode_active = False
                strike_episode_succeeded = False
            previous_tongue_extended = tongue_extended

        wall_clock_s = time.perf_counter() - start_wall_clock
        final_energy = float(primary_agent(sim, task.arch).energy)
    finally:
        sim.close()

    sim_time_s = task.steps * dt
    catches_float = float(catches)
    energy_spent_est = catches_float * 5.0 + initial_energy - final_energy
    ms_per_step = wall_clock_s * 1000.0 / max(1, task.steps)
    learning_curve_times = [float(row["time_s"]) for row in time_series_rows]
    learning_curve_values = [float(row["catch_rate_per_minute"]) for row in time_series_rows]
    if len(learning_curve_times) >= 2:
        learning_auc = float(np.trapezoid(learning_curve_values, learning_curve_times) / learning_curve_times[-1])
    elif learning_curve_values:
        learning_auc = float(learning_curve_values[0])
    else:
        learning_auc = 0.0

    metrics = {
        "arch": task.arch,
        "mode": task.mode,
        "training_mode": training_mode,
        "spawn_seed": task.spawn_seed,
        "repeat": task.repeat,
        "steps_requested": task.steps,
        "steps_executed": task.steps,
        "dt": dt,
        "sim_time_s": sim_time_s,
        "wall_clock_s": wall_clock_s,
        "ms_per_step": ms_per_step,
        "catches": catches,
        "strike_attempts": strike_attempts,
        "successful_strike_episodes": successful_strike_episodes,
        "captures_per_strike_attempt": catches_float / strike_attempts if strike_attempts > 0 else 0.0,
        "capture_success": successful_strike_episodes / strike_attempts if strike_attempts > 0 else 0.0,
        "false_strike_rate": max(0.0, strike_attempts - successful_strike_episodes) / strike_attempts if strike_attempts > 0 else 0.0,
        "catch_rate_per_1k_steps": catches_float / max(task.steps / 1000.0, 1e-9),
        "catch_rate_per_minute": catches_float / max(sim_time_s / 60.0, 1e-9),
        "time_to_first_catch_s": first_catch_step * dt if first_catch_step is not None else None,
        "time_to_competence_s": competence_step * dt if competence_step is not None else None,
        "learning_auc": learning_auc,
        "ttc_mean_s": safe_mean(ttc_samples),
        "ttc_median_s": safe_median(ttc_samples),
        "reaction_latency_mean_s": safe_mean(reaction_samples),
        "reaction_latency_median_s": safe_median(reaction_samples),
        "energy_initial": initial_energy,
        "energy_final": final_energy,
        "energy_spent_est": energy_spent_est,
        "catch_rate_low_energy": bucket_catches["low"] / max(bucket_seconds["low"] / 60.0, 1e-9) if bucket_seconds["low"] > 0 else None,
        "catch_rate_mid_energy": bucket_catches["mid"] / max(bucket_seconds["mid"] / 60.0, 1e-9) if bucket_seconds["mid"] > 0 else None,
        "catch_rate_high_energy": bucket_catches["high"] / max(bucket_seconds["high"] / 60.0, 1e-9) if bucket_seconds["high"] > 0 else None,
        "strike_rate_low_energy": bucket_strikes["low"] / max(bucket_seconds["low"] / 60.0, 1e-9) if bucket_seconds["low"] > 0 else None,
        "strike_rate_mid_energy": bucket_strikes["mid"] / max(bucket_seconds["mid"] / 60.0, 1e-9) if bucket_seconds["mid"] > 0 else None,
        "strike_rate_high_energy": bucket_strikes["high"] / max(bucket_seconds["high"] / 60.0, 1e-9) if bucket_seconds["high"] > 0 else None,
        "flies_per_energy_spent": catches_float / energy_spent_est if energy_spent_est > 0 else None,
        "net_energy_balance_per_catch": (final_energy - initial_energy) / catches_float if catches > 0 else None,
        "path_length_px": path_length,
        "movement_efficiency_px_per_catch": path_length / catches_float if catches > 0 else None,
        "avg_speed_px_s": path_length / max(sim_time_s, 1e-9),
        "avg_alignment": safe_mean(alignment_samples),
        "avg_controller_signal": safe_mean(controller_samples),
        "avg_neural_activity": safe_mean(neural_samples),
        "avg_acceleration_px_s2": safe_mean(acceleration_samples),
        "avg_jerk_px_s3": safe_mean(jerk_samples),
        "tongue_usage_ratio": tongue_catches / catches_float if catches > 0 else None,
        "visible_time_s": visible_steps * dt,
        "visible_but_ignored_ratio": ignored_visible_steps / visible_steps if visible_steps > 0 else None,
        "visible_but_ignored_ratio_low_energy": bucket_ignored_steps["low"] / bucket_visible_steps["low"] if bucket_visible_steps["low"] > 0 else None,
        "visible_but_ignored_ratio_high_energy": bucket_ignored_steps["high"] / bucket_visible_steps["high"] if bucket_visible_steps["high"] > 0 else None,
        "energy_deficit_strike_correlation": safe_correlation(energy_deficit_samples, strike_propensity_samples),
        "strike_opportunity_count": strike_window_episodes,
        "strike_opportunity_conversion": catches_float / strike_window_episodes if strike_window_episodes > 0 else None,
        "shot_distance_mean_px": safe_mean(shot_distances),
        "shot_distance_std_px": safe_std(shot_distances),
        "shot_distance_p10_px": safe_quantile(shot_distances, 0.10),
        "shot_distance_p50_px": safe_quantile(shot_distances, 0.50),
        "shot_distance_p90_px": safe_quantile(shot_distances, 0.90),
        "pre_shot_positioning_error_mean_px": safe_mean(positioning_errors),
        "pre_shot_positioning_error_p90_px": safe_quantile(positioning_errors, 0.90),
        "survived": bool_to_float(final_energy > 0.0),
        "compute_per_catch_s": wall_clock_s / catches_float if catches > 0 else None,
        "avg_visible_target_count": safe_mean(visible_target_counts),
        "seeded_spawn_stream": True,
        "model_seed": model_seed,
        "runtime_seed": runtime_seed,
        "adult_reached": bool_to_float(adult_step is not None or not training_mode),
        "adult_step": 0 if (not training_mode) else adult_step,
        "adult_time_s": 0.0 if (not training_mode) else (adult_step * dt if adult_step is not None else None),
        "catches_before_maturity": 0 if (not training_mode) else catches_before_maturity,
        "avg_juvenile_progress": safe_mean(juvenile_progress_samples),
        "avg_maturity_readiness": safe_mean(maturity_readiness_samples),
        "avg_maturity_stability": safe_mean(maturity_stability_samples),
        "avg_food_prediction_error": safe_mean(food_prediction_error_samples),
        "avg_hunger_bias": safe_mean(hunger_bias_samples),
        "avg_reward_seek_bias": safe_mean(reward_seek_bias_samples),
        "avg_predation_bias": safe_mean(predation_bias_samples),
        "avg_prey_permission": safe_mean(prey_permission_samples),
        "avg_fast_target_lock": safe_mean(fast_target_lock_samples),
        "avg_fast_loop_gate": safe_mean(fast_loop_gate_samples),
        "avg_fast_strike_drive": safe_mean(fast_strike_drive_samples),
        "avg_effective_motor_gate": safe_mean(effective_motor_gate_samples),
        "avg_bg_gating_signal": safe_mean(bg_gating_signal_samples),
        "avg_spike_rate": safe_mean(spike_rate_samples),
        "avg_spike_count": safe_mean(spike_count_samples),
        "burst_ratio": burst_steps / task.steps if task.steps > 0 else None,
        "silent_ratio": silent_steps / task.steps if task.steps > 0 else None,
        "spikes_per_catch": (sum(spike_count_samples) / catches_float) if catches > 0 and spike_count_samples else None,
        "avg_eligibility_norm": safe_mean(eligibility_samples),
        "avg_learning_reward": safe_mean(learning_reward_samples),
        "avg_actor_advantage": safe_mean(actor_advantage_samples),
        "avg_value_estimate": safe_mean(value_estimate_samples),
    }

    return {
        "task": asdict(task),
        "metrics": metrics,
        "time_series": time_series_rows,
        "events": event_rows,
    }


def make_output_dir(base_dir: Optional[Path] = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path("benchmark_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / f"frog_benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_benchmark_suite(
    *,
    steps: int = 100_000,
    spawn_seeds: Sequence[int] = tuple(range(10)),
    repeats: int = 5,
    repeat_indices: Optional[Sequence[int]] = None,
    modes: Sequence[str] = ("adult", "developmental"),
    architectures: Sequence[str] = ARCHITECTURES,
    sample_interval: int = 200,
    competence_catches: int = 10,
    output_dir: Optional[Path] = None,
    workers: int = 1,
) -> Path:
    output_dir = make_output_dir(output_dir)
    selected_repeats = tuple(int(repeat) for repeat in repeat_indices) if repeat_indices is not None else tuple(range(repeats))
    tasks: List[BenchmarkTask] = [
        BenchmarkTask(
            arch=arch,
            mode=mode,
            spawn_seed=int(spawn_seed),
            repeat=int(repeat),
            steps=int(steps),
            sample_interval=int(sample_interval),
            competence_catches=int(competence_catches),
        )
        for mode in modes
        for arch in architectures
        for spawn_seed in spawn_seeds
        for repeat in selected_repeats
    ]

    run_metrics: List[Dict[str, Any]] = []
    time_series_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    metadata = {
        "steps": steps,
        "spawn_seeds": list(spawn_seeds),
        "repeats": repeats,
        "repeat_indices": list(selected_repeats),
        "modes": list(modes),
        "architectures": list(architectures),
        "sample_interval": sample_interval,
        "competence_catches": competence_catches,
        "workers": workers,
        "created_at": datetime.now().isoformat(),
        "notes": [
            "spawn_seed controls initial fly spawn stream; runtime randomness varies per repeat",
            "online learning remains enabled for ANN/SNN/BIO current variants",
            "ANN_FROZEN and SNN_FROZEN preserve architecture but disable online weight updates",
            "BIO_DUAL is a copied dual-loop descendant of the current BIO runtime",
            "developmental mode maps to training_mode=True",
        ],
    }
    write_json(output_dir / "metadata.json", metadata)

    print(
        f"Benchmark start | runs={len(tasks)} | steps={steps} | seeds={len(spawn_seeds)} | "
        f"repeats={len(selected_repeats)} | workers={workers}"
    )

    completed = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_single_task, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                result = future.result()
                run_metrics.append(result["metrics"])
                time_series_rows.extend(result["time_series"])
                event_rows.extend(result["events"])
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] {task.arch} {task.mode} "
                    f"spawn={task.spawn_seed} repeat={task.repeat} catches={result['metrics']['catches']}"
                )
    else:
        for task in tasks:
            result = run_single_task(task)
            run_metrics.append(result["metrics"])
            time_series_rows.extend(result["time_series"])
            event_rows.extend(result["events"])
            completed += 1
            print(
                f"[{completed}/{len(tasks)}] {task.arch} {task.mode} "
                f"spawn={task.spawn_seed} repeat={task.repeat} catches={result['metrics']['catches']}"
            )

    aggregate_rows = aggregate_metric_rows(run_metrics)
    seed_rows = aggregate_seed_rows(run_metrics)

    write_csv(output_dir / "run_metrics.csv", run_metrics, RUN_METRIC_FIELDS)
    write_json(output_dir / "run_metrics.json", run_metrics)
    write_gzip_csv(output_dir / "time_series.csv.gz", time_series_rows, TIME_SERIES_FIELDS)
    write_gzip_csv(output_dir / "event_log.csv.gz", event_rows, EVENT_FIELDS)

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

    print(f"Benchmark outputs saved to: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-form benchmark for ANN, SNN and BIO frog variants.")
    parser.add_argument("--steps", type=int, default=100_000, help="Simulation steps per run.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of spawn seeds starting from 0.")
    parser.add_argument(
        "--spawn-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit spawn seed list. If provided, overrides --seeds.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per spawn seed.")
    parser.add_argument(
        "--repeat-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit repeat indices. If provided, overrides the repeat range implied by --repeats.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["adult", "developmental"],
        choices=sorted(MODES.keys()),
        help="Simulation modes to benchmark.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(ARCHITECTURES),
        choices=list(ARCHITECTURES),
        help="Architectures to benchmark.",
    )
    parser.add_argument("--sample-interval", type=int, default=200, help="Sampling interval for time-series logging.")
    parser.add_argument("--competence-catches", type=int, default=10, help="Catch threshold used as competence proxy.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional parent directory for outputs.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count.")
    parser.add_argument("--skip-report", action="store_true", help="Skip markdown report and plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spawn_seeds = tuple(int(seed) for seed in args.spawn_seeds) if args.spawn_seeds is not None else tuple(range(args.seeds))
    repeat_indices = tuple(int(repeat) for repeat in args.repeat_indices) if args.repeat_indices is not None else None
    output_dir = run_benchmark_suite(
        steps=args.steps,
        spawn_seeds=spawn_seeds,
        repeats=args.repeats,
        repeat_indices=repeat_indices,
        modes=tuple(args.modes),
        architectures=tuple(args.architectures),
        sample_interval=args.sample_interval,
        competence_catches=args.competence_catches,
        output_dir=args.output_dir,
        workers=max(1, int(args.workers)),
    )
    if not args.skip_report:
        try:
            from benchmark_frogs_report import generate_report

            generate_report(output_dir)
        except Exception as exc:  # pragma: no cover - report generation is secondary
            print(f"Report generation failed: {exc}")


if __name__ == "__main__":
    main()
