"""Hierarchical motor controller adapted for maze locomotion."""

import numpy as np

from Frog_predator_neuro_dual_fast.core.biological_neuron import FastSpikingInterneuron, PyramidalNeuron


class MotorHierarchy:
    def __init__(self):
        self.executive_neurons = [PyramidalNeuron() for _ in range(4)]
        self.coordination_interneurons = [FastSpikingInterneuron() for _ in range(8)]
        self.motor_neurons = [PyramidalNeuron() for _ in range(12)]
        self.muscle_activation = np.zeros(12)
        self.tonic_drive = 0.0
        self.motor_phase = 0.0

    def execute_movement_command(self, command, proprioceptive_feedback, tonic_drive=0.0, energy_level=1.0):
        self.tonic_drive = tonic_drive
        self.motor_phase = (self.motor_phase + 0.10 + tonic_drive * 0.18) % (2.0 * np.pi)
        cpg_gate = 0.72 + 0.28 * max(0.0, float(np.sin(self.motor_phase)))
        homeostatic_factor = max(0.55, float(energy_level))
        magnitude = np.linalg.norm(command)
        if magnitude > 0.01:
            input_angle = np.arctan2(command[1], command[0])
            angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            for idx, neuron in enumerate(self.executive_neurons):
                angle_diff = np.abs(input_angle - angles[idx])
                angle_diff = np.min([angle_diff, 2 * np.pi - angle_diff])
                selectivity = np.exp(-(angle_diff ** 2) / (2 * 0.5 ** 2))
                neuron.integrate(0.01, magnitude * selectivity * (15.0 + tonic_drive * 10.0), homeostatic_factor=homeostatic_factor)
        else:
            for neuron in self.executive_neurons:
                neuron.integrate(0.01, 0.0, homeostatic_factor=homeostatic_factor)

        exec_activity = float(np.mean([n.activity_level() for n in self.executive_neurons]))
        for neuron in self.coordination_interneurons:
            neuron.integrate(0.01, exec_activity * 46.0, homeostatic_factor=homeostatic_factor)

        inter_activity = float(np.mean([n.activity_level() for n in self.coordination_interneurons]))
        for idx, neuron in enumerate(self.motor_neurons):
            feedback = proprioceptive_feedback[idx] * 5.0 if idx < len(proprioceptive_feedback) else 0.0
            neuron.integrate(0.01, inter_activity * 36.0 - feedback, homeostatic_factor=homeostatic_factor)
            target_activation = neuron.activity_level() * cpg_gate
            self.muscle_activation[idx] = 0.85 * self.muscle_activation[idx] + 0.15 * target_activation
        return self.muscle_activation

    def get_velocity_vector(self, command):
        magnitude = np.linalg.norm(command)
        if magnitude < 1e-6:
            return (0.0, 0.0)
        muscle_gain = float(np.mean(self.muscle_activation[:4])) if len(self.muscle_activation) else 0.0
        locomotor_tone = max(muscle_gain, 0.10 + self.tonic_drive * 0.55)
        return tuple((np.array(command) / magnitude) * magnitude * locomotor_tone)

    def all_neurons(self):
        return self.executive_neurons + self.coordination_interneurons + self.motor_neurons


