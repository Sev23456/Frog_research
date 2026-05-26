"""Biological synapse models with plasticity and neuromodulation."""

import numpy as np


class BiologicalSynapse:
    def __init__(self, max_weight=1.0, initial_weight=None, min_weight=0.0, stdp_window=50.0, stdp_amplitude=0.01):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.weight = initial_weight if initial_weight is not None else np.random.uniform(0.2, 0.8)
        self.stdp_window = stdp_window
        self.stdp_amplitude = stdp_amplitude
        self.facilitation_state = 0.0
        self.depression_state = 0.0
        self.facilitation_tau = 50.0
        self.depression_tau = 200.0
        self.current_efficacy = 1.0
        self.dopamine_modulation = 0.5
        self.serotonin_modulation = 0.5
        self.acetylcholine_level = 0.3

    def apply_stdp(self, presynaptic_spike_time, postsynaptic_spike_time):
        if presynaptic_spike_time is None or postsynaptic_spike_time is None:
            return
        delta_t = postsynaptic_spike_time - presynaptic_spike_time
        if abs(delta_t) < self.stdp_window:
            if delta_t > 0:
                weight_change = self.stdp_amplitude * np.exp(-delta_t / self.stdp_window)
            else:
                weight_change = -self.stdp_amplitude * np.exp(delta_t / self.stdp_window)
            modulation = 0.7 * self.dopamine_modulation + 0.3 * self.serotonin_modulation
            self.weight = np.clip(self.weight + weight_change * modulation, self.min_weight, self.max_weight)

    def apply_short_term_plasticity(self, dt, spike):
        if spike:
            self.facilitation_state = 1.0
            self.depression_state = 1.0
        else:
            self.facilitation_state *= np.exp(-dt / self.facilitation_tau)
            self.depression_state *= np.exp(-dt / self.depression_tau)
        self.current_efficacy = 1.0 + 0.3 * self.facilitation_state - 0.5 * self.depression_state
        self.current_efficacy = np.clip(self.current_efficacy, 0.1, 2.0)

    def transmit(self, presynaptic_output):
        return presynaptic_output * self.weight * self.current_efficacy

    def update_modulators(self, dopamine, serotonin, acetylcholine):
        self.dopamine_modulation = dopamine
        self.serotonin_modulation = serotonin
        self.acetylcholine_level = acetylcholine


class DynamicSynapse(BiologicalSynapse):
    def __init__(self, synapse_type="depressing", **kwargs):
        super().__init__(**kwargs)
        self.synapse_type = synapse_type
        if synapse_type == "depressing":
            self.facilitation_tau = 30.0
            self.depression_tau = 800.0
        else:
            self.facilitation_tau = 200.0
            self.depression_tau = 100.0

    def transmit_dynamic(self, presynaptic_output, available_resources):
        utilization = presynaptic_output * self.current_efficacy
        return utilization * available_resources
