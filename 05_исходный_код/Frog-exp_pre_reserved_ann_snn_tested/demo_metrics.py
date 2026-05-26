#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using the Metrics Framework for AI Comparison
========================================================
Demonstrates how to use MetricsCollector to compare ANN, SNN, and BioFrog agents.
"""

import sys
sys.path.insert(0, '/workspace')

from metrics_framework import MetricsCollector, compare_architectures


def simulate_agent_step(agent_type: str, step_num: int):
    """
    Simulate agent step data for demonstration.
    In real usage, this would come from actual simulation.
    """
    import random
    import math
    
    base_catch_prob = {
        "ANN": 0.04,
        "SNN": 0.035,
        "BioFrog": 0.025,
    }
    
    # Simulate catch
    caught = random.random() < base_catch_prob.get(agent_type, 0.03)
    
    # Simulate energy dynamics
    energy_drain = 0.1 + random.uniform(0, 0.05)
    if caught:
        energy_gain = 5.0
    else:
        energy_gain = 0
    
    # Simulate velocity and position
    speed = random.uniform(0.3, 0.8) * 65.0  # max_speed = 65
    angle = random.uniform(0, 2 * math.pi)
    velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
    
    # Simulate alignment
    alignment = random.uniform(0.4, 0.95)
    
    # Simulate target distance
    target_distance = random.uniform(10, 40) if random.random() > 0.2 else None
    
    # Bio-specific: spike counts
    if agent_type == "SNN":
        spike_count = random.choices([0, 1, 2, 3, 4], weights=[0.4, 0.3, 0.2, 0.08, 0.02])[0]
    elif agent_type == "BioFrog":
        spike_count = random.choices([0, 1, 2, 3], weights=[0.5, 0.3, 0.15, 0.05])[0]
    else:
        spike_count = 0
    
    # BioFrog juvenile phase
    is_juvenile = agent_type == "BioFrog" and step_num < 5000
    
    return {
        "position": (random.uniform(100, 700), random.uniform(100, 500)),
        "velocity": velocity,
        "energy": 30.0 - step_num * energy_drain + (5.0 if caught else 0),
        "alignment": alignment,
        "target_distance": target_distance,
        "caught_fly": "fly_object" if caught else None,
        "spike_count": spike_count,
        "neural_activity": spike_count / 8.0 if agent_type in ["SNN", "BioFrog"] else 0.0,
        "is_juvenile": is_juvenile,
    }


def run_demo_simulation():
    """Run demo simulation for all three architectures."""
    import random
    random.seed(42)  # For reproducibility
    
    print("🚀 Starting Demo Simulation with Metrics Collection")
    print("="*80)
    
    architectures = ["ANN", "SNN", "BioFrog"]
    collectors = {}
    
    for arch in architectures:
        print(f"\n⏳ Simulating {arch} agent for 2000 steps...")
        
        collector = MetricsCollector()
        collector.initial_energy = 30.0
        collector.current_energy = 30.0
        
        for step in range(2000):
            state = simulate_agent_step(arch, step)
            collector.record_step(state, dt=0.01)
            
            # Record tongue shots occasionally
            if state["target_distance"] and random.random() < 0.1:
                success = random.random() < 0.8
                collector.record_tongue_shot(state["target_distance"], success)
        
        collectors[arch] = collector
        print(f"   ✅ {arch}: {collector.total_catches} catches, "
              f"energy={collector.current_energy:.1f}")
    
    # Print individual reports
    print("\n\n" + "="*80)
    print("📋 INDIVIDUAL METRICS REPORTS")
    print("="*80)
    
    for arch, collector in collectors.items():
        collector.print_summary(arch)
    
    # Print comparison
    print(compare_architectures(list(collectors.items())))
    
    # Export to dict for further analysis
    print("\n📊 EXPORTING DATA FOR ANALYSIS")
    print("-"*80)
    
    comparison_data = {}
    for arch, collector in collectors.items():
        comparison_data[arch] = collector.to_dict()
        print(f"{arch}: {len(comparison_data[arch])} metrics collected")
    
    # Show key metrics side-by-side
    print("\n🔑 KEY METRICS COMPARISON:")
    key_metrics = [
        "task_performance_catch_rate",
        "task_performance_flies_per_energy",
        "bio_plausibility_sparsity_ratio",
        "survival_energy_balance",
    ]
    
    print(f"{'Metric':<40} | {'ANN':>10} | {'SNN':>10} | {'BioFrog':>10}")
    print("-"*80)
    
    for metric in key_metrics:
        values = []
        for arch in architectures:
            val = comparison_data[arch].get(metric, 0)
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        print(f"{metric:<40} | {values[0]:>10} | {values[1]:>10} | {values[2]:>10}")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 To use in your simulations:")
    print("   1. Import MetricsCollector from metrics_framework")
    print("   2. Create collector = MetricsCollector()")
    print("   3. Call collector.record_step(agent_state, dt) each frame")
    print("   4. Call collector.record_tongue_shot(distance, success) when shooting")
    print("   5. Call collector.print_summary('YourAgentName') for report")
    print("   6. Use compare_architectures() for multi-agent comparison")


if __name__ == "__main__":
    run_demo_simulation()
