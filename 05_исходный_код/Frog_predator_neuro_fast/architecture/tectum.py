"""Tectum-like directional salience processor."""

import math
import numpy as np

from Frog_predator_neuro_fast.core.biological_neuron import FastSpikingInterneuron, PyramidalNeuron
from Frog_predator_neuro_fast.core.synapse_models import BiologicalSynapse


class TectalColumn:
    def __init__(self, preferred_direction):
        self.preferred_direction = preferred_direction
        self.pyramidal_neurons = [PyramidalNeuron() for _ in range(8)]
        self.interneurons = [FastSpikingInterneuron() for _ in range(4)]
        self.output_neurons = [PyramidalNeuron() for _ in range(2)]
        self.synapses = [BiologicalSynapse() for _ in range(6)]
        self.output = 0.0

    def process_salience(self, visual_input, salience_vector):
        sx = float(salience_vector[0])
        sy = float(salience_vector[1])
        vector_norm = math.hypot(sx, sy)
        if vector_norm > 1e-6:
            salience_direction = math.atan2(sy, sx)
            direction_difference = abs(salience_direction - self.preferred_direction)
            direction_difference = min(direction_difference, 2.0 * math.pi - direction_difference)
            selectivity = math.exp(-(direction_difference ** 2) / (2.0 * 0.55 ** 2))
        else:
            selectivity = 0.2

        input_current = visual_input * 8.0 * selectivity + vector_norm * 12.0 * selectivity
        for neuron in self.pyramidal_neurons:
            neuron.integrate(0.01, input_current)

        pyramidal_activity = sum(n.activity_level() for n in self.pyramidal_neurons) / len(self.pyramidal_neurons)
        inter_input = pyramidal_activity * 42.0
        for neuron in self.interneurons:
            neuron.integrate(0.01, inter_input)

        exc_input = pyramidal_activity * 38.0
        inh_input = (sum(n.activity_level() for n in self.interneurons) / len(self.interneurons)) * 18.0
        for neuron in self.output_neurons:
            neuron.integrate(0.01, exc_input - inh_input)
        self.output = sum(n.activity_level() for n in self.output_neurons) / len(self.output_neurons)
        return self.output


class Tectum:
    def __init__(self, columns=16):
        self.num_columns = columns
        self.columns = [TectalColumn((i / columns) * 2.0 * math.pi) for i in range(columns)]
        self.tectal_output = np.zeros(columns)
        self.dominant_direction = 0.0

    def process(self, retinal_input, salience_vectors):
        self.tectal_output.fill(0.0)
        mean_visual_input = float(np.mean(retinal_input)) if len(retinal_input) > 0 else 0.0
        if len(salience_vectors) > 0:
            sx = sum(float(vector[0]) for vector in salience_vectors) / len(salience_vectors)
            sy = sum(float(vector[1]) for vector in salience_vectors) / len(salience_vectors)
            aggregated_salience = (sx, sy)
        else:
            aggregated_salience = (0.0, 0.0)
        for idx, column in enumerate(self.columns):
            self.tectal_output[idx] = column.process_salience(mean_visual_input, aggregated_salience)
        if float(np.sum(self.tectal_output)) > 0:
            dominant_idx = int(np.argmax(self.tectal_output))
            self.dominant_direction = (dominant_idx / self.num_columns) * 2.0 * math.pi
        return self.tectal_output

    def get_movement_command(self):
        tectal_sum = float(np.sum(self.tectal_output))
        if tectal_sum <= 0:
            return (0.0, 0.0)
        magnitude = tectal_sum / self.num_columns
        return (magnitude * math.cos(self.dominant_direction), magnitude * math.sin(self.dominant_direction))

