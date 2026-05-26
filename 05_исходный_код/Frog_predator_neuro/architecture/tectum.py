"""Tectum-like directional salience processor."""

import numpy as np

from Frog_predator_neuro.core.biological_neuron import FastSpikingInterneuron, PyramidalNeuron
from Frog_predator_neuro.core.synapse_models import BiologicalSynapse


class TectalColumn:
    def __init__(self, preferred_direction):
        self.preferred_direction = preferred_direction
        self.pyramidal_neurons = [PyramidalNeuron() for _ in range(8)]
        self.interneurons = [FastSpikingInterneuron() for _ in range(4)]
        self.output_neurons = [PyramidalNeuron() for _ in range(2)]
        self.synapses = [BiologicalSynapse() for _ in range(6)]
        self.output = 0.0

    def process_salience(self, visual_input, salience_vector):
        vector_norm = np.linalg.norm(salience_vector)
        if vector_norm > 1e-6:
            salience_direction = np.arctan2(salience_vector[1], salience_vector[0])
            direction_difference = np.abs(salience_direction - self.preferred_direction)
            direction_difference = np.min([direction_difference, 2 * np.pi - direction_difference])
            selectivity = np.exp(-(direction_difference ** 2) / (2 * 0.55 ** 2))
        else:
            selectivity = 0.2

        input_current = visual_input * 8.0 * selectivity + vector_norm * 12.0 * selectivity
        for neuron in self.pyramidal_neurons:
            neuron.integrate(0.01, input_current)

        pyramidal_activity = float(np.mean([n.activity_level() for n in self.pyramidal_neurons]))
        inter_input = pyramidal_activity * 42.0
        for neuron in self.interneurons:
            neuron.integrate(0.01, inter_input)

        exc_input = pyramidal_activity * 38.0
        inh_input = float(np.mean([n.activity_level() for n in self.interneurons])) * 18.0
        for neuron in self.output_neurons:
            neuron.integrate(0.01, exc_input - inh_input)
        self.output = float(np.mean([n.activity_level() for n in self.output_neurons]))
        return self.output


class Tectum:
    def __init__(self, columns=16):
        self.num_columns = columns
        self.columns = [TectalColumn((i / columns) * 2 * np.pi) for i in range(columns)]
        self.tectal_output = np.zeros(columns)
        self.dominant_direction = 0.0

    def process(self, retinal_input, salience_vectors):
        self.tectal_output = np.zeros(self.num_columns)
        mean_visual_input = np.mean(retinal_input) if len(retinal_input) > 0 else 0.0
        aggregated_salience = np.mean(salience_vectors, axis=0) if len(salience_vectors) > 0 else np.zeros(2)
        for idx, column in enumerate(self.columns):
            self.tectal_output[idx] = column.process_salience(mean_visual_input, aggregated_salience)
        if np.sum(self.tectal_output) > 0:
            dominant_idx = int(np.argmax(self.tectal_output))
            self.dominant_direction = (dominant_idx / self.num_columns) * 2 * np.pi
        return self.tectal_output

    def get_movement_command(self):
        if np.sum(self.tectal_output) <= 0:
            return (0.0, 0.0)
        magnitude = float(np.sum(self.tectal_output) / self.num_columns)
        return (magnitude * np.cos(self.dominant_direction), magnitude * np.sin(self.dominant_direction))
