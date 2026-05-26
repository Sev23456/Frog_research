#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
from statistics import mean
from typing import Dict, List

import numpy as np

from frog_lib_ann.simulation import ANNFlyCatchingSimulation
from frog_lib_snn.simulation import SNNFlyCatchingSimulation
from Frog_predator_neuro.simulation import Simulation as BioSimulation


def run_variant(name: str, steps: int, seed: int) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)

    if name == "ANN":
        sim = ANNFlyCatchingSimulation(headless=True, training_mode=False)
    elif name == "SNN":
        sim = SNNFlyCatchingSimulation(headless=True, training_mode=False)
    elif name == "BIO":
        sim = BioSimulation(headless=True, training_mode=False)
    else:
        raise ValueError(f"Unknown variant: {name}")

    try:
        for _ in range(steps):
            sim.step()
        stats = sim.get_statistics()
        return {
            "caught_flies": float(stats["caught_flies"]),
            "final_energy": float(stats["final_energy"]),
            "avg_alignment": float(stats.get("avg_alignment", 0.0)),
            "avg_distance_to_target": float(stats.get("avg_distance_to_target", float("nan"))),
            "avg_speed": float(stats.get("avg_speed", 0.0)),
            "success_rate": float(stats.get("success_rate", 0.0)),
        }
    finally:
        sim.close()


def format_metric(values: List[float]) -> str:
    rounded = [round(value, 3) for value in values]
    return f"{rounded} | avg {round(mean(values), 3)}"


def main():
    parser = argparse.ArgumentParser(description="Compare ANN, SNN and BIO frog variants under the same seeds.")
    parser.add_argument("--steps", type=int, default=2000, help="Steps per seed")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds starting from 0")
    args = parser.parse_args()

    variants = ["ANN", "SNN", "BIO"]
    results: Dict[str, List[Dict[str, float]]] = {name: [] for name in variants}

    for seed in range(args.seeds):
        for name in variants:
            results[name].append(run_variant(name, args.steps, seed))

    print(f"Frog comparison | steps={args.steps} | seeds={args.seeds}")
    for name in variants:
        catches = [entry["caught_flies"] for entry in results[name]]
        energy = [entry["final_energy"] for entry in results[name]]
        alignment = [entry["avg_alignment"] for entry in results[name]]
        speed = [entry["avg_speed"] for entry in results[name]]
        success = [entry["success_rate"] for entry in results[name]]

        print(name)
        print("  catches       ", format_metric(catches))
        print("  final_energy  ", format_metric(energy))
        print("  avg_alignment ", format_metric(alignment))
        print("  avg_speed     ", format_metric(speed))
        print("  success_rate  ", format_metric(success))


if __name__ == "__main__":
    main()
