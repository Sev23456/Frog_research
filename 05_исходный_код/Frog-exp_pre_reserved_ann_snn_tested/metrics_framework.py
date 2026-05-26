#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Metrics Collection Framework for AI Comparison
======================================================
Comprehensive metrics for comparing ANN, SNN, and BioFrog architectures
in the fly-catching simulation.

Metric Categories:
1. Task Performance - Overall effectiveness
2. Strategy & Mechanics - Tactical behavior analysis  
3. Bio-Plausibility - Biological realism metrics
4. Error Cost - Penalty for mistakes
5. Survival Metrics - Long-term viability
6. Developmental Metrics - Ontogeny analysis (BioFrog only)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MetricsCollector:
    """
    Unified metrics collector for all AI architectures.
    
    Collects, computes, and formats comprehensive performance metrics
    for fair comparison between ANN, SNN, and BioFrog agents.
    """
    
    # Time window for recent statistics
    window_size: int = 500
    
    # History buffers
    energy_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    catch_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    distance_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    alignment_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    smoothness_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    
    # Bio-specific metrics
    spike_count_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    neural_activity_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    reaction_latency_history: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Tongue/capture metrics
    tongue_shot_distances: deque = field(default_factory=lambda: deque(maxlen=500))
    tongue_success_history: deque = field(default_factory=lambda: deque(maxlen=500))
    recovery_time_history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    # State tracking
    total_steps: int = 0
    total_catches: int = 0
    total_energy_spent: float = 0.0
    last_catch_step: int = -999
    last_miss_step: int = -999
    current_energy: float = 30.0
    initial_energy: float = 30.0
    
    # For developmental metrics (BioFrog)
    juvenile_steps: int = 0
    competence_achieved: bool = False
    competence_step: Optional[int] = None
    is_juvenile: bool = False
    
    # Movement tracking
    total_distance_traveled: float = 0.0
    last_position: Optional[Tuple[float, float]] = None
    
    def update_position(self, position: Tuple[float, float]):
        """Track movement for distance calculations."""
        if self.last_position is not None:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            self.total_distance_traveled += math.hypot(dx, dy)
        self.last_position = position
    
    def record_step(self, agent_state: Dict[str, Any], dt: float = 0.01):
        """
        Record one simulation step with comprehensive metrics.
        
        Args:
            agent_state: Dictionary containing agent state from update()
            dt: Time step duration
        """
        self.total_steps += 1
        
        # Extract common fields
        energy = agent_state.get("energy", self.current_energy)
        velocity = agent_state.get("velocity", (0.0, 0.0))
        position = agent_state.get("position", self.last_position or (0.0, 0.0))
        alignment = agent_state.get("alignment", 0.0)
        target_distance = agent_state.get("target_distance")
        caught_fly = agent_state.get("caught_fly")
        
        # Update position tracking
        self.update_position(position)
        
        # Energy tracking
        energy_delta = self.current_energy - energy
        if energy_delta > 0:
            self.total_energy_spent += energy_delta
        self.current_energy = energy
        self.energy_history.append(energy)
        
        # Speed and smoothness
        speed = math.hypot(velocity[0], velocity[1]) if hasattr(velocity, '__iter__') else 0.0
        self.speed_history.append(speed)
        
        if len(self.speed_history) >= 2:
            prev_speed = self.speed_history[-2]
            if speed > 1e-6 and prev_speed > 1e-6:
                smoothness = 1.0  # Will be computed properly in get_statistics
            else:
                smoothness = 1.0
        else:
            smoothness = 1.0
        self.smoothness_history.append(smoothness)
        
        # Distance and alignment
        if target_distance is not None:
            self.distance_history.append(target_distance)
        self.alignment_history.append(alignment)
        
        # Catch tracking
        if caught_fly is not None:
            self.catch_history.append(1)
            self.total_catches += 1
            self.last_catch_step = self.total_steps
        else:
            self.catch_history.append(0)
        
        # Bio-specific metrics
        spike_count = agent_state.get("spike_count", 0)
        neural_activity = agent_state.get("neural_activity", 0.0)
        self.spike_count_history.append(spike_count)
        self.neural_activity_history.append(neural_activity)
        
        # Juvenile tracking
        is_juv = agent_state.get("is_juvenile", False)
        if is_juv:
            self.is_juvenile = True
            self.juvenile_steps += 1
            if not self.competence_achieved and energy > self.initial_energy:
                self.competence_achieved = True
                self.competence_step = self.total_steps
    
    def record_tongue_shot(self, distance: float, success: bool):
        """Record tongue shot for tactical analysis."""
        self.tongue_shot_distances.append(distance)
        self.tongue_success_history.append(1 if success else 0)
        
        if not success:
            self.last_miss_step = self.total_steps
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics from collected data.
        
        Returns:
            Dictionary with all computed metrics organized by category
        """
        stats = {}
        
        # ============================================================
        # 1. TASK PERFORMANCE METRICS
        # ============================================================
        stats["task_performance"] = self._compute_task_performance()
        
        # ============================================================
        # 2. STRATEGY & MECHANICS METRICS
        # ============================================================
        stats["strategy_mechanics"] = self._compute_strategy_metrics()
        
        # ============================================================
        # 3. BIO-PLAUSIBILITY METRICS
        # ============================================================
        stats["bio_plausibility"] = self._compute_bio_metrics()
        
        # ============================================================
        # 4. ERROR COST METRICS
        # ============================================================
        stats["error_cost"] = self._compute_error_cost()
        
        # ============================================================
        # 5. SURVIVAL METRICS
        # ============================================================
        stats["survival"] = self._compute_survival_metrics()
        
        # ============================================================
        # 6. DEVELOPMENTAL METRICS (if applicable)
        # ============================================================
        stats["developmental"] = self._compute_developmental_metrics()
        
        return stats
    
    def _compute_task_performance(self) -> Dict[str, Any]:
        """Compute task effectiveness metrics."""
        total_steps = max(1, self.total_steps)
        
        # Catch Rate (CR) - catches per 100 steps
        catch_rate = (self.total_catches / total_steps) * 100 if total_steps > 0 else 0.0
        
        # Time-to-Capture (TTC) - average steps between catches
        if self.total_catches > 0:
            avg_ttc = total_steps / self.total_catches
        else:
            avg_ttc = float('inf')
        
        # Flies Caught per Energy - key efficiency metric
        energy_efficiency = (
            self.total_catches / (self.total_energy_spent + 1e-6)
            if self.total_energy_spent > 0
            else float('inf') if self.total_catches > 0 else 0.0
        )
        
        return {
            "catch_rate": catch_rate,
            "catch_rate_comment": "Catches per 100 steps (higher is better)",
            "time_to_capture_avg": avg_ttc,
            "time_to_capture_comment": "Average steps between catches (lower is better)",
            "flies_per_energy": energy_efficiency,
            "flies_per_energy_comment": "Key metric: catches per energy unit (metabolic efficiency)",
            "total_catches": self.total_catches,
            "total_energy_spent": self.total_energy_spent,
        }
    
    def _compute_strategy_metrics(self) -> Dict[str, Any]:
        """Compute tactical behavior metrics."""
        # Tongue Usage Ratio
        total_shots = len(self.tongue_success_history)
        if total_shots > 0:
            tongue_success_rate = sum(self.tongue_success_history) / total_shots
            avg_shot_distance = sum(self.tongue_shot_distances) / total_shots
        else:
            tongue_success_rate = 0.0
            avg_shot_distance = 0.0
        
        # Shot distance distribution
        if self.tongue_shot_distances:
            shot_distances = list(self.tongue_shot_distances)
            min_shot = min(shot_distances)
            max_shot = max(shot_distances)
            median_shot = sorted(shot_distances)[len(shot_distances) // 2]
        else:
            min_shot = max_shot = median_shot = 0.0
        
        # Movement Efficiency
        movement_efficiency = (
            self.total_distance_traveled / (self.total_catches + 1e-6)
            if self.total_catches > 0
            else self.total_distance_traveled
        )
        
        # Average alignment during pursuit
        avg_alignment = (
            sum(self.alignment_history) / len(self.alignment_history)
            if self.alignment_history
            else 0.0
        )
        
        return {
            "tongue_success_rate": tongue_success_rate,
            "tongue_success_comment": "% of successful tongue shots (should be ~75-92% based on config)",
            "avg_shot_distance": avg_shot_distance,
            "shot_distance_range": f"{min_shot:.1f} - {max_shot:.1f}",
            "median_shot_distance": median_shot,
            "shot_distance_comment": "Optimal: close enough to hit, far enough to not spook fly",
            "movement_efficiency": movement_efficiency,
            "movement_efficiency_comment": "Total path length per catch (lower = better positioning)",
            "avg_alignment": avg_alignment,
            "alignment_comment": "How well velocity aligns with prey direction (0-1, higher is better)",
            "total_shots": total_shots,
            "total_distance": self.total_distance_traveled,
        }
    
    def _compute_bio_metrics(self) -> Dict[str, Any]:
        """Compute biological plausibility metrics."""
        # Spike/Activity Efficiency
        total_spikes = sum(self.spike_count_history) if self.spike_count_history else 0
        spikes_per_catch = total_spikes / (self.total_catches + 1e-6)
        avg_spike_rate = (
            sum(self.spike_count_history) / len(self.spike_count_history)
            if self.spike_count_history
            else 0.0
        )
        
        # Neural activity sparsity
        silent_steps = sum(1 for s in self.spike_count_history if s == 0)
        burst_steps = sum(1 for s in self.spike_count_history if s >= 2)
        total_neural_steps = len(self.spike_count_history)
        
        sparsity_ratio = silent_steps / (total_neural_steps + 1e-6)
        burst_ratio = burst_steps / (total_neural_steps + 1e-6)
        
        # Decision stability (variance in similar states)
        if len(self.alignment_history) > 10:
            recent_alignments = list(self.alignment_history)[-100:]
            alignment_variance = (
                sum((a - sum(recent_alignments)/len(recent_alignments))**2 
                    for a in recent_alignments) / len(recent_alignments)
            )
        else:
            alignment_variance = 0.0
        
        return {
            "total_spikes": total_spikes,
            "spikes_per_catch": spikes_per_catch,
            "spikes_per_catch_comment": "Neural efficiency: fewer spikes per catch = more efficient coding",
            "avg_spike_rate": avg_spike_rate,
            "sparsity_ratio": sparsity_ratio,
            "sparsity_comment": "Fraction of silent steps (biological systems aim for sparse coding)",
            "burst_ratio": burst_ratio,
            "burst_comment": "Fraction of burst steps (≥2 spikes, indicates salient events)",
            "decision_stability": 1.0 - min(1.0, alignment_variance),
            "stability_comment": "Consistency in similar states (1.0 = perfectly stable)",
        }
    
    def _compute_error_cost(self) -> Dict[str, Any]:
        """Compute cost of mistakes metrics."""
        # Recovery time after misses
        recovery_times = []
        for i, catch in enumerate(self.catch_history):
            if catch == 1 and i > 0:
                # Find previous miss
                for j in range(i - 1, -1, -1):
                    if self.catch_history[j] == 0:
                        recovery_times.append(i - j)
                        break
        
        avg_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else 0.0
        
        # Opportunity cost - flies missed during recovery
        # Simplified: count misses in windows after other misses
        opportunity_windows = 0
        for i, catch in enumerate(self.catch_history):
            if catch == 0 and i + 10 < len(self.catch_history):
                missed_in_window = sum(1 for j in range(i + 1, min(i + 11, len(self.catch_history))) 
                                      if self.catch_history[j] == 0)
                opportunity_windows += missed_in_window
        
        return {
            "avg_recovery_time": avg_recovery,
            "recovery_comment": "Steps to recover after a miss (lower = faster recovery)",
            "opportunity_cost_estimate": opportunity_windows,
            "opportunity_comment": "Estimated flies missed during recovery periods",
        }
    
    def _compute_survival_metrics(self) -> Dict[str, Any]:
        """Compute survival and energy balance metrics."""
        # Energy balance stability
        if self.energy_history:
            avg_energy = sum(self.energy_history) / len(self.energy_history)
            min_energy = min(self.energy_history)
            max_energy = max(self.energy_history)
            energy_variance = sum((e - avg_energy)**2 for e in self.energy_history) / len(self.energy_history)
        else:
            avg_energy = min_energy = max_energy = energy_variance = 0.0
        
        # Energy balance (gain vs loss)
        energy_gain = self.total_catches * 5.0  # 5.0 energy per fly (from config)
        energy_balance = energy_gain - self.total_energy_spent
        
        # Starvation risk
        critical_threshold = 6.0  # AGENT_CRITICAL_THRESHOLD from config
        starvation_risk = 1.0 if min_energy <= critical_threshold else 0.0
        
        return {
            "avg_energy": avg_energy,
            "min_energy": min_energy,
            "max_energy": max_energy,
            "energy_variance": energy_variance,
            "energy_balance": energy_balance,
            "energy_balance_comment": "Net energy: gain (from food) - spent (movement/metabolism)",
            "survival_status": "ALIVE" if self.current_energy > 0 else "STARVED",
            "starvation_risk": starvation_risk,
            "risk_comment": "1.0 = experienced critical energy levels, 0.0 = safe",
        }
    
    def _compute_developmental_metrics(self) -> Dict[str, Any]:
        """Compute ontogeny/development metrics (BioFrog specific)."""
        if not self.is_juvenile:
            return {
                "applicable": False,
                "comment": "Not applicable for non-developmental agents (ANN/SNN)",
            }
        
        # Time to Competence
        ttc = self.competence_step if self.competence_achieved else self.total_steps
        
        # Developmental cost
        developmental_cost = self.juvenile_steps
        
        # Skill retention (simplified: are we still catching as adults?)
        if self.competence_achieved and self.total_steps > self.competence_step:
            adult_steps = self.total_steps - self.competence_step
            adult_catches = self.total_catches  # Simplified
            retention_score = adult_catches / (adult_steps + 1e-6) * 100
        else:
            retention_score = 0.0
        
        return {
            "applicable": True,
            "time_to_competence": ttc,
            "competence_achieved": self.competence_achieved,
            "ttc_comment": "Steps needed to achieve positive energy balance",
            "developmental_cost": developmental_cost,
            "cost_comment": "Total juvenile steps (investment in development)",
            "skill_retention_score": retention_score,
            "retention_comment": "Catches per 100 adult steps (skill maintenance)",
        }
    
    def print_summary(self, architecture_name: str = "Agent"):
        """
        Print formatted metrics summary to console.
        
        Args:
            architecture_name: Name of the architecture (ANN/SNN/BioFrog)
        """
        metrics = self.compute_metrics()
        
        print("\n" + "="*80)
        print(f"🐸 {architecture_name.upper()} - COMPREHENSIVE METRICS REPORT")
        print("="*80)
        
        # Task Performance
        tp = metrics["task_performance"]
        print(f"\n📊 TASK PERFORMANCE:")
        print(f"   Catch Rate:           {tp['catch_rate']:.2f}% ({tp['catch_rate_comment']})")
        print(f"   Avg Time-to-Capture:  {tp['time_to_capture_avg']:.1f} steps ({tp['time_to_capture_comment']})")
        print(f"   Flies/Energy:         {tp['flies_per_energy']:.3f} ⭐ ({tp['flies_per_energy_comment']})")
        print(f"   Total Catches:        {tp['total_catches']}")
        print(f"   Total Energy Spent:   {tp['total_energy_spent']:.2f}")
        
        # Strategy & Mechanics
        sm = metrics["strategy_mechanics"]
        print(f"\n🎯 STRATEGY & TACTICS:")
        print(f"   Tongue Success Rate:  {sm['tongue_success_rate']*100:.1f}% ({sm['tongue_success_comment']})")
        print(f"   Avg Shot Distance:    {sm['avg_shot_distance']:.1f}px (range: {sm['shot_distance_range']})")
        print(f"   Median Shot Distance: {sm['median_shot_distance']:.1f}px ({sm['shot_distance_comment']})")
        print(f"   Movement Efficiency:  {sm['movement_efficiency']:.2f}px/catch ({sm['movement_efficiency_comment']})")
        print(f"   Avg Alignment:        {sm['avg_alignment']:.3f} ({sm['alignment_comment']})")
        
        # Bio-Plausibility
        bp = metrics["bio_plausibility"]
        print(f"\n🧬 BIOLOGICAL PLAUSIBILITY:")
        print(f"   Total Spikes:         {bp['total_spikes']}")
        print(f"   Spikes/Catch:         {bp['spikes_per_catch']:.2f} ({bp['spikes_per_catch_comment']})")
        print(f"   Sparsity Ratio:       {bp['sparsity_ratio']:.3f} ({bp['sparsity_comment']})")
        print(f"   Burst Ratio:          {bp['burst_ratio']:.3f} ({bp['burst_comment']})")
        print(f"   Decision Stability:   {bp['decision_stability']:.3f} ({bp['stability_comment']})")
        
        # Error Cost
        ec = metrics["error_cost"]
        print(f"\n⚠️  ERROR COST:")
        print(f"   Avg Recovery Time:    {ec['avg_recovery_time']:.1f} steps ({ec['recovery_comment']})")
        print(f"   Opportunity Cost:     ~{ec['opportunity_cost_estimate']} flies ({ec['opportunity_comment']})")
        
        # Survival
        sv = metrics["survival"]
        print(f"\n❤️ SURVIVAL STATUS:")
        print(f"   Status:               {sv['survival_status']} {'✅' if sv['survival_status'] == 'ALIVE' else '❌'}")
        print(f"   Avg Energy:           {sv['avg_energy']:.2f}/{self.initial_energy:.1f}")
        print(f"   Energy Range:         {sv['min_energy']:.2f} - {sv['max_energy']:.2f}")
        print(f"   Net Energy Balance:   {sv['energy_balance']:+.2f} ({sv['energy_balance_comment']})")
        print(f"   Starvation Risk:      {sv['starvation_risk']:.1f} ({sv['risk_comment']})")
        
        # Developmental
        dev = metrics["developmental"]
        if dev["applicable"]:
            print(f"\n🌱 DEVELOPMENTAL ONTOGENY:")
            print(f"   Time to Competence:   {dev['time_to_competence']} steps ({dev['ttc_comment']})")
            print(f"   Developmental Cost:   {dev['developmental_cost']} steps ({dev['cost_comment']})")
            print(f"   Skill Retention:      {dev['skill_retention_score']:.3f} ({dev['retention_comment']})")
        else:
            print(f"\n🌱 DEVELOPMENTAL ONTOGENY:")
            print(f"   {dev['comment']}")
        
        print("\n" + "="*80)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return all metrics as a flat dictionary for export/comparison."""
        metrics = self.compute_metrics()
        flat = {
            "total_steps": self.total_steps,
            "total_catches": self.total_catches,
        }
        
        for category, submetrics in metrics.items():
            for key, value in submetrics.items():
                if isinstance(value, (int, float, bool, str)):
                    flat[f"{category}_{key}"] = value
        
        return flat


def compare_architectures(metrics_list: List[Tuple[str, MetricsCollector]]) -> str:
    """
    Generate comparison report for multiple architectures.
    
    Args:
        metrics_list: List of (architecture_name, MetricsCollector) tuples
    
    Returns:
        Formatted comparison string
    """
    if not metrics_list:
        return "No architectures to compare."
    
    lines = []
    lines.append("\n" + "="*100)
    lines.append("🏆 ARCHITECTURE COMPARISON SUMMARY")
    lines.append("="*100)
    
    # Header
    header = f"{'Architecture':<15} | {'Catches':>8} | {'Catch Rate':>11} | {'Flies/E':>9} | {'Sparsity':>9} | {'Energy Bal':>10} | {'Status':>8}"
    lines.append(header)
    lines.append("-"*100)
    
    for name, collector in metrics_list:
        metrics = collector.compute_metrics()
        tp = metrics["task_performance"]
        bp = metrics["bio_plausibility"]
        sv = metrics["survival"]
        
        status_symbol = "✅" if sv["survival_status"] == "ALIVE" else "❌"
        
        row = (
            f"{name:<15} | "
            f"{tp['total_catches']:>8} | "
            f"{tp['catch_rate']:>10.2f}% | "
            f"{tp['flies_per_energy']:>9.3f} | "
            f"{bp['sparsity_ratio']:>9.3f} | "
            f"{sv['energy_balance']:>+10.2f} | "
            f"{status_symbol} {sv['survival_status']}"
        )
        lines.append(row)
    
    lines.append("="*100)
    lines.append("\nLegend:")
    lines.append("  • Catch Rate: % of steps resulting in catch (higher = better)")
    lines.append("  • Flies/E: Flies caught per energy unit (KEY METRIC - metabolic efficiency)")
    lines.append("  • Sparsity: Fraction of silent neural steps (bio-realism)")
    lines.append("  • Energy Bal: Net energy gain/loss (positive = sustainable)")
    lines.append("="*100 + "\n")
    
    return "\n".join(lines)
