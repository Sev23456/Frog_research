"""Retinal-style sensory preprocessing adapted for egocentric maze vision."""

import numpy as np

from Frog_predator_neuro.core.biological_neuron import LIFNeuron


class CenterSurroundFilter:
    def __init__(self, position, center_size=1.2, surround_size=2.8, filter_type="on_center"):
        self.position = np.array(position, dtype=float)
        self.center_size = center_size
        self.surround_size = surround_size
        self.filter_type = filter_type
        self.center_neuron = LIFNeuron()
        self.output = 0.0
        self.adaptation_trace = 0.0
        self.previous_drive = 0.0
        self.current_drive = 0.0

    def process(self, stimulus):
        center_distance = np.linalg.norm(stimulus - self.position)
        center_response = np.exp(-(center_distance ** 2) / (2 * self.center_size ** 2))
        surround_response = np.exp(-(center_distance ** 2) / (2 * self.surround_size ** 2))
        if self.filter_type == "on_center":
            raw_drive = center_response - 0.3 * surround_response
        else:
            raw_drive = surround_response - 0.3 * center_response
        raw_drive = float(np.clip(raw_drive, 0.0, 1.0))
        motion_drive = abs(raw_drive - self.previous_drive)
        self.adaptation_trace = 0.92 * self.adaptation_trace + 0.08 * raw_drive
        habituated_drive = raw_drive * np.exp(-1.8 * self.adaptation_trace)
        combined_drive = np.clip(0.20 * habituated_drive + 0.80 * motion_drive, 0.0, 1.0)
        self.current_drive = float(combined_drive)
        input_current = combined_drive * 20.0
        self.center_neuron.integrate(0.01, input_current)
        self.output = self.center_neuron.activity_level()
        self.previous_drive = raw_drive
        return self.output


class RetinalProcessing:
    def __init__(self, visual_field_size=(5.0, 5.0), num_filters_per_side=6):
        self.visual_field_size = visual_field_size
        self.num_filters_per_side = num_filters_per_side
        self.filters = []
        for i in range(num_filters_per_side):
            for j in range(num_filters_per_side):
                x = (i + 0.5) * visual_field_size[0] / num_filters_per_side
                y = (j + 0.5) * visual_field_size[1] / num_filters_per_side
                filter_type = "on_center" if (i + j) % 2 == 0 else "off_center"
                self.filters.append(CenterSurroundFilter((x, y), filter_type=filter_type))
        self.retinal_output = np.zeros(len(self.filters))
        self.processed_image = None

    def process_visual_input(self, visual_scene):
        self.retinal_output = np.zeros(len(self.filters))
        for idx, filter_cell in enumerate(self.filters):
            total_response = 0.0
            for obj_x, obj_y, brightness in visual_scene:
                stimulus = np.array([obj_x, obj_y], dtype=float)
                total_response += filter_cell.process(stimulus) * brightness
            self.retinal_output[idx] = total_response
        self.processed_image = self.retinal_output.reshape((self.num_filters_per_side, self.num_filters_per_side))
        return self.retinal_output

    def get_spatial_attention_map(self):
        if self.processed_image is None:
            return np.zeros((self.num_filters_per_side, self.num_filters_per_side))
        attention_map = np.clip(self.processed_image, 0.0, 1.0)
        kernel = np.array([0.25, 0.5, 0.25])
        flat = np.convolve(attention_map.flatten(), kernel, mode="same")
        return flat.reshape(self.processed_image.shape)
