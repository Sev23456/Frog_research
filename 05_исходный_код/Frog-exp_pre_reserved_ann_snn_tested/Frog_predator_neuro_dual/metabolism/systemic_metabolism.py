"""Systemic and per-neuron metabolism models."""

import numpy as np


class NeuronMetabolism:
    def __init__(self, baseline_atp=1.0, max_atp=2.0):
        self.atp_level = baseline_atp
        self.max_atp = max_atp
        self.baseline_atp = baseline_atp
        self.atp_cost_spike = 0.1
        self.atp_cost_rest = 0.01
        self.atp_recovery_rate = 0.02
        self.excitability_modifier = 1.0

    def consume_energy(self, spiked, firing_rate, dt):
        if spiked:
            self.atp_level -= self.atp_cost_spike
        self.atp_level -= self.atp_cost_rest * dt
        self.atp_level -= firing_rate * 0.01 * dt
        self.atp_level = np.clip(self.atp_level, 0.0, self.max_atp)

    def recover_energy(self, dt, oxygen_level=1.0, glucose_level=1.0):
        recovery = self.atp_recovery_rate * oxygen_level * glucose_level * dt
        self.atp_level = np.clip(self.atp_level + recovery, 0.0, self.max_atp)

    def affects_excitability(self):
        if self.atp_level < self.baseline_atp * 0.5:
            excitability = 0.3 + 0.7 * (self.atp_level / (self.baseline_atp * 0.5))
        else:
            excitability = 1.0 + 0.5 * ((self.atp_level - self.baseline_atp) / self.baseline_atp)
        self.excitability_modifier = float(np.clip(excitability, 0.1, 2.0))
        return self.excitability_modifier


class SystemicMetabolism:
    def __init__(self):
        self.glucose_level = 1.0
        self.oxygen_level = 1.0
        self.lactate_level = 0.1
        self.glucose_consumption_rate = 0.001
        self.oxygen_consumption_rate = 0.0015
        self.glucose_recovery_rate = 0.0005
        self.oxygen_recovery_rate = 0.001
        self.neural_activity_level = 0.0
        self.movement_activity_level = 0.0
        self.circadian_phase = 0.0
        self.fatigue_level = 0.0

    def update(self, dt, movement_intensity=0.0, neural_activity=0.0):
        self.neural_activity_level = 0.9 * self.neural_activity_level + 0.1 * neural_activity
        self.movement_activity_level = 0.9 * self.movement_activity_level + 0.1 * movement_intensity
        total_activity = self.neural_activity_level + self.movement_activity_level
        self.glucose_level -= self.glucose_consumption_rate * (1.0 + total_activity) * dt
        self.oxygen_level -= self.oxygen_consumption_rate * (1.0 + total_activity) * dt
        recovery_factor = 1.0 - 0.5 * self.fatigue_level
        self.glucose_level += self.glucose_recovery_rate * recovery_factor * dt
        self.oxygen_level += self.oxygen_recovery_rate * recovery_factor * dt
        self.lactate_level += total_activity * 0.01 * dt
        self.lactate_level *= np.exp(-dt / 100.0)
        self.circadian_phase = (self.circadian_phase + dt / 10000.0) % (2 * np.pi)
        time_of_day_fatigue = 0.3 * (1.0 - np.cos(self.circadian_phase)) / 2.0
        activity_fatigue = 0.2 * total_activity
        resource_fatigue = 0.3 * (1.0 - (self.glucose_level + self.oxygen_level) / 2.0)
        self.fatigue_level = float(np.clip(time_of_day_fatigue + activity_fatigue + resource_fatigue, 0.0, 1.0))
        self.glucose_level = float(np.clip(self.glucose_level, 0.0, 1.5))
        self.oxygen_level = float(np.clip(self.oxygen_level, 0.0, 1.5))
        return self.get_metabolic_state()

    def get_metabolic_state(self):
        return {
            "glucose": self.glucose_level,
            "oxygen": self.oxygen_level,
            "lactate": self.lactate_level,
            "fatigue": self.fatigue_level,
            "neural_activity": self.neural_activity_level,
            "movement_activity": self.movement_activity_level,
        }
