#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from frog_lib_ann_frozen_soft_sated.simulation import ANNFlyCatchingSimulation
else:
    from .simulation import ANNFlyCatchingSimulation


def main():
    parser = argparse.ArgumentParser(description="Run standalone frozen soft-satiation ANN frog simulation")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--num-flies", type=int, default=15)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--training", action="store_true", help="Enable online supervision")
    parser.add_argument("--plot", action="store_true", help="Show behavior plots after the run")
    parser.add_argument("--save-state", action="store_true", help="Save statistics to JSON")
    args = parser.parse_args()

    sim = ANNFlyCatchingSimulation(
        width=args.width,
        height=args.height,
        num_flies=args.num_flies,
        headless=args.headless,
        training_mode=args.training,
    )

    try:
        sim.run_simulation(args.steps)
        if args.plot:
            sim.plot_results()
        if args.save_state:
            sim.save_state()
        print(sim.get_statistics())
    finally:
        sim.close()


if __name__ == "__main__":
    main()
