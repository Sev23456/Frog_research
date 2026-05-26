"""Astrocyte and glial modulation models."""

import numpy as np


class Astrocyte:
    def __init__(self, position, influence_radius=50.0):
        self.position = np.array(position, dtype=float)
        self.influence_radius = influence_radius
        self.calcium_level = 0.1
        self.calcium_resting = 0.1
        self.calcium_peak = 2.0
        self.calcium_decay_tau = 500.0
        self.gliotransmitter_release = 0.0
        self.gliotransmitter_threshold = 0.3
        self.glutamate_level = 0.0
        self.atp_level = 0.0

    def respond_to_neural_activity(self, neural_activity_map, neural_positions, dt):
        if len(neural_positions) == 0:
            return
        distances = np.linalg.norm(neural_positions - self.position, axis=1)
        nearby = distances < self.influence_radius
        local_activity = float(np.mean(neural_activity_map[nearby])) if np.any(nearby) else 0.0

        if local_activity > 0.1:
            self.calcium_level += (self.calcium_peak - self.calcium_level) * 0.1
        else:
            decay = np.exp(-dt / self.calcium_decay_tau)
            self.calcium_level = self.calcium_resting + (self.calcium_level - self.calcium_resting) * decay

        if self.calcium_level > self.gliotransmitter_threshold:
            self.gliotransmitter_release = (self.calcium_level - self.gliotransmitter_threshold) / (self.calcium_peak - self.gliotransmitter_threshold)
            self.glutamate_level = 0.3 * self.gliotransmitter_release
            self.atp_level = 0.5 * self.gliotransmitter_release
        else:
            self.gliotransmitter_release = 0.0
            self.glutamate_level *= np.exp(-dt / 100.0)
            self.atp_level *= np.exp(-dt / 150.0)

    def modulate_synapses(self, synapses, weight_modifier=0.05):
        if self.gliotransmitter_release <= 0:
            return
        weight_change = self.atp_level * weight_modifier
        for synapse in synapses:
            synapse.current_efficacy = min(2.0, synapse.current_efficacy + weight_change)


class GlialNetwork:
    def __init__(self, num_astrocytes=24, brain_size=(400.0, 400.0)):
        grid_size = int(np.sqrt(num_astrocytes))
        self.astrocytes = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) * brain_size[0] / grid_size
                y = (j + 0.5) * brain_size[1] / grid_size
                self.astrocytes.append(Astrocyte((x, y), influence_radius=brain_size[0] / (2 * grid_size)))
        self.average_gliotransmitter = 0.0
        self.brain_state = "resting"

    def update(self, neural_activity_map, neural_positions, dt):
        for astrocyte in self.astrocytes:
            astrocyte.respond_to_neural_activity(neural_activity_map, neural_positions, dt)
        self.average_gliotransmitter = np.mean([astro.gliotransmitter_release for astro in self.astrocytes]) if self.astrocytes else 0.0
        if self.average_gliotransmitter > 0.5:
            self.brain_state = "excited"
        elif self.average_gliotransmitter > 0.2:
            self.brain_state = "active"
        else:
            self.brain_state = "resting"

    def get_local_modulation(self, position):
        modulation = {"dopamine": 0.5, "serotonin": 0.5, "acetylcholine": 0.3}
        for astrocyte in self.astrocytes:
            distance = np.linalg.norm(position - astrocyte.position)
            if distance < astrocyte.influence_radius:
                influence = (1.0 - distance / astrocyte.influence_radius) * astrocyte.gliotransmitter_release
                modulation["acetylcholine"] += influence * 0.1
        return modulation

