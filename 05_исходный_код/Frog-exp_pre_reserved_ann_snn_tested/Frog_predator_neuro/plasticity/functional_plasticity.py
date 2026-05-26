"""Functional and structural plasticity managers."""

import numpy as np


class StructuralPlasticityManager:
    def __init__(self):
        self.synapse_creation_threshold = 0.1
        self.synapse_elimination_threshold = 0.01
        self.synapse_creation_rate = 0.0001
        self.synapse_elimination_rate = 0.00005

    def update_structure(self, synapses, neural_activity, dt):
        for synapse in synapses:
            if synapse.weight > self.synapse_creation_threshold:
                synapse.weight = min(synapse.weight + self.synapse_creation_rate * synapse.weight * dt, synapse.max_weight)
            elif synapse.weight < self.synapse_elimination_threshold:
                if np.random.random() < self.synapse_elimination_rate * dt:
                    synapse.weight = 0.0


class FunctionalPlasticityManager:
    def __init__(self):
        self.homeostatic_target = 0.3
        self.homeostatic_learning_rate = 0.0001
        self.synaptic_scaling_enabled = True
        self.intrinsic_plasticity_enabled = True

    def apply_homeostatic_scaling(self, neurons, dt):
        if not self.synaptic_scaling_enabled:
            return
        for neuron in neurons:
            if len(neuron.membrane_potential_history) == 0:
                continue
            recent_activity = np.mean([1 if v > neuron.threshold else 0 for v in neuron.membrane_potential_history[-100:]])
            if recent_activity > self.homeostatic_target * 1.5:
                neuron.threshold += self.homeostatic_learning_rate * dt
            elif recent_activity < self.homeostatic_target * 0.5:
                neuron.threshold -= self.homeostatic_learning_rate * dt

    def apply_intrinsic_plasticity(self, neurons, dt):
        if not self.intrinsic_plasticity_enabled:
            return
        for neuron in neurons:
            if len(neuron.membrane_potential_history) == 0:
                continue
            mean_potential = np.mean(neuron.membrane_potential_history[-100:])
            if mean_potential > neuron.threshold * 0.95:
                neuron.threshold += self.homeostatic_learning_rate * dt * 0.5
            elif mean_potential < neuron.rest_potential + 5.0:
                neuron.threshold -= self.homeostatic_learning_rate * dt * 0.5
