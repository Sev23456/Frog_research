# -*- coding: utf-8 -*-
"""
Biologically informed neuron models (LIF, pyramidal, fast-spiking interneuron).

Adapted from Frog-main to provide consistent, low-level neural dynamics
for the Frog_predator_neuro_fast agents.
"""

import math
import numpy as np
from typing import Optional


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron."""

    def __init__(self, rest_potential: float = -70.0, threshold: float = -40.0,
                 tau_membrane: float = 20.0, tau_refractory: float = 2.0,
                 max_firing_rate: float = 200.0):
        self.rest_potential = rest_potential
        self.threshold_baseline = threshold + float(np.random.uniform(-1.0, 1.0))
        self.threshold = self.threshold_baseline
        self.membrane_potential = rest_potential
        self.tau_membrane = tau_membrane * float(np.random.uniform(0.95, 1.05))  # ms-like units
        self.tau_refractory = tau_refractory * float(np.random.uniform(0.95, 1.05))  # ms-like units
        self.max_firing_rate = max_firing_rate
        self.intrinsic_gain = float(np.random.uniform(0.94, 1.06))
        self.homeostatic_activity_trace = 0.0
        self.target_activity = 0.14

        self.spike_output = 0.0
        self.refractory_counter = 0.0
        self.last_spike_time = -1000.0
        self.membrane_potential_history = []
        self.store_history = False

    def integrate(self, dt: float, input_current: float, homeostatic_factor: float = 1.0):
        """Integrate membrane potential for a timestep dt.

        dt is expected to be a small timestep (e.g. 0.01).
        """
        current_activity = self.spike_output
        self.homeostatic_activity_trace = 0.95 * self.homeostatic_activity_trace + 0.05 * current_activity
        homeostatic_adjustment = (self.homeostatic_activity_trace - self.target_activity) * 4.0 * homeostatic_factor
        self.threshold = self.threshold_baseline + homeostatic_adjustment

        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        decay = math.exp(-dt / self.tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay
        self.membrane_potential += input_current * self.intrinsic_gain * dt / self.tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = self.tau_refractory
            self.last_spike_time = 0.0
        else:
            self.spike_output = 0.0

        self.last_spike_time += dt
        if self.store_history:
            self.membrane_potential_history.append(self.membrane_potential)
            if len(self.membrane_potential_history) > 1000:
                del self.membrane_potential_history[:-1000]

    def reset(self):
        self.membrane_potential = self.rest_potential
        self.spike_output = 0.0
        self.refractory_counter = 0.0
        self.last_spike_time = -1000.0
        self.membrane_potential_history = []

    def activity_level(self):
        """Normalized activity level for compatibility with existing code.

        Returns a value in [0, 1] representing recent activity or depolarization.
        """
        span = max(1e-6, self.threshold - self.rest_potential)
        depolarization = (self.membrane_potential - self.rest_potential) / span
        smoothed = 0.8 * depolarization + 0.2 * self.spike_output
        return _clamp01(smoothed)


class PyramidalNeuron(LIFNeuron):
    """Pyramidal neuron with simple apical/basal dendritic integration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apical_dendrite_input = 0.0
        self.basal_dendrite_input = 0.0
        self.soma_input = 0.0
        self.dendritic_plateau_potential = 0.0
        self.dendritic_spike_threshold = -30.0

    def integrate(self, dt: float, basal_input: float, apical_input: Optional[float] = None, homeostatic_factor: float = 1.0):
        if apical_input is None:
            apical_input = 0.0

        current_activity = self.spike_output
        self.homeostatic_activity_trace = 0.95 * self.homeostatic_activity_trace + 0.05 * current_activity
        homeostatic_adjustment = (self.homeostatic_activity_trace - self.target_activity) * 4.0 * homeostatic_factor
        self.threshold = self.threshold_baseline + homeostatic_adjustment

        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        self.basal_dendrite_input = 0.9 * self.basal_dendrite_input + 0.1 * basal_input
        self.apical_dendrite_input = 0.9 * self.apical_dendrite_input + 0.1 * apical_input

        if apical_input > 0.5 and self.basal_dendrite_input > 0.3:
            self.dendritic_plateau_potential = 1.0
        else:
            self.dendritic_plateau_potential *= math.exp(-dt / 50.0)

        soma_input = basal_input + 0.3 * apical_input * self.dendritic_plateau_potential

        decay = math.exp(-dt / self.tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay
        self.membrane_potential += soma_input * self.intrinsic_gain * dt / self.tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = self.tau_refractory
            self.last_spike_time = 0.0
        else:
            self.spike_output = 0.0

        self.last_spike_time += dt
        if self.store_history:
            self.membrane_potential_history.append(self.membrane_potential)
            if len(self.membrane_potential_history) > 1000:
                del self.membrane_potential_history[:-1000]


class FastSpikingInterneuron(LIFNeuron):
    """Fast-spiking interneuron with an adaptation current."""

    def __init__(self, **kwargs):
        super().__init__(tau_membrane=5.0, tau_refractory=1.0, **kwargs)
        self.adaptation_current = 0.0
        self.adaptation_tau = 100.0
        self.spike_count = 0

    def integrate(self, dt: float, input_current: float, homeostatic_factor: float = 1.0):
        current_activity = self.spike_output
        self.homeostatic_activity_trace = 0.95 * self.homeostatic_activity_trace + 0.05 * current_activity
        homeostatic_adjustment = (self.homeostatic_activity_trace - self.target_activity) * 4.0 * homeostatic_factor
        self.threshold = self.threshold_baseline + homeostatic_adjustment

        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        decay = math.exp(-dt / self.adaptation_tau)
        self.adaptation_current *= decay

        decay_soma = math.exp(-dt / self.tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay_soma

        net_input = input_current - 0.5 * self.adaptation_current
        self.membrane_potential += net_input * self.intrinsic_gain * dt / self.tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = self.tau_refractory
            self.last_spike_time = 0.0
            self.adaptation_current += 5.0
            self.spike_count += 1
        else:
            self.spike_output = 0.0

        self.last_spike_time += dt
        if self.store_history:
            self.membrane_potential_history.append(self.membrane_potential)
            if len(self.membrane_potential_history) > 1000:
                del self.membrane_potential_history[:-1000]

