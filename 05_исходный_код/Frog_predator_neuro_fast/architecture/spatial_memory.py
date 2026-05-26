"""Distributed place-cell memory without explicit tile-map planning."""

import math
import random
from dataclasses import dataclass

import numpy as np

from Frog_predator_neuro_fast.config import (
    ASSOCIATION_DECAY,
    HEAD_DIRECTION_CELLS,
    HEAD_DIRECTION_GAIN,
    MAP_HEIGHT,
    MAP_WIDTH,
    MEMORY_LOOKAHEAD_PX,
    MEMORY_PROBE_DIRECTIONS,
    MEMORY_RECURRENT_K,
    MEMORY_REPLAY_GAIN,
    MEMORY_VECTOR_MOMENTUM,
    NOVELTY_RECOVERY,
    OCCUPANCY_DECAY,
    OPENNESS_GAIN,
    PLACE_CELL_COUNT,
    PLACE_CELL_DECAY,
    PLACE_CELL_WIDTH_MAX,
    PLACE_CELL_WIDTH_MIN,
    PLACE_FIELD_LEARNING_RATE,
    REPLAY_TRANSITION_DECAY,
)
from Frog_predator_neuro_fast.core.biological_neuron import PyramidalNeuron
from Frog_predator_neuro_fast.core.synapse_models import BiologicalSynapse
from Frog_predator_neuro_fast.utils import clamp, normalize_vector


@dataclass
class PlaceCell:
    preferred_position: np.ndarray
    width: float
    neuron: PyramidalNeuron
    novelty: float = 1.0
    food_association: float = 0.0
    occupancy_trace: float = 0.0
    replay_trace: float = 0.0
    last_activation: float = 0.0


@dataclass
class HeadDirectionCell:
    preferred_angle: float
    neuron: PyramidalNeuron
    activation: float = 0.0


@dataclass
class RecurrentLink:
    source_idx: int
    target_idx: int
    synapse: BiologicalSynapse


class SpatialMemory:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.place_cells = []
        for _ in range(PLACE_CELL_COUNT):
            preferred_position = np.array(
                [
                    self.rng.uniform(0.0, MAP_WIDTH),
                    self.rng.uniform(0.0, MAP_HEIGHT),
                ],
                dtype=float,
            )
            width = self.rng.uniform(PLACE_CELL_WIDTH_MIN, PLACE_CELL_WIDTH_MAX)
            self.place_cells.append(PlaceCell(preferred_position, width, PyramidalNeuron()))
        self.preferred_positions = np.array([cell.preferred_position for cell in self.place_cells], dtype=float)
        self.place_widths = np.array([cell.width for cell in self.place_cells], dtype=float)

        self.head_direction_cells = [
            HeadDirectionCell((idx / HEAD_DIRECTION_CELLS) * 2.0 * math.pi, PyramidalNeuron())
            for idx in range(HEAD_DIRECTION_CELLS)
        ]
        self.recurrent_links = []
        self.incoming_links = {index: [] for index in range(len(self.place_cells))}
        self._build_recurrent_links()

        self.theta_phase = 0.0
        self.wander_angle = self.rng.uniform(0.0, 2.0 * math.pi)
        self.last_vector = np.zeros(2, dtype=float)
        self.last_position = np.array([MAP_WIDTH * 0.5, MAP_HEIGHT * 0.5], dtype=float)
        self.last_heading = 0.0
        self.last_activations = np.zeros(len(self.place_cells), dtype=float)
        self.replay_gate = 0.0
        self.external_drive = 0.0
        self.food_expectation = 0.0
        self.food_prediction_error = 0.0
        self.reorientation_drive = 0.0
        self.reorientation_vector = np.zeros(2, dtype=float)
        self.last_policy_metrics = {
            "vigor": 0.35,
            "search_burst": 0.18,
            "confidence": 0.22,
            "conflict": 0.35,
            "habit_pressure": 0.0,
            "selected_openness": 0.0,
            "loop_pressure": 0.0,
            "reorientation_drive": 0.0,
        }

    def _build_recurrent_links(self):
        for index, cell in enumerate(self.place_cells):
            deltas = self.preferred_positions - cell.preferred_position
            distances = np.linalg.norm(deltas, axis=1)
            order = np.argsort(distances)
            neighbors = [neighbor for neighbor in order if neighbor != index][:MEMORY_RECURRENT_K]
            for neighbor in neighbors:
                distance = max(distances[neighbor], 1.0)
                weight = float(np.clip(0.92 - distance / max(MAP_WIDTH, MAP_HEIGHT), 0.08, 0.65))
                synapse = BiologicalSynapse(initial_weight=weight, max_weight=1.4)
                link = RecurrentLink(source_idx=index, target_idx=neighbor, synapse=synapse)
                self.recurrent_links.append(link)
                self.incoming_links[neighbor].append(link)

    def all_neurons(self):
        neurons = [cell.neuron for cell in self.place_cells]
        neurons.extend(cell.neuron for cell in self.head_direction_cells)
        return neurons

    def all_synapses(self):
        return [link.synapse for link in self.recurrent_links]

    def _gaussian_projection(self, position):
        position = np.array(position, dtype=float)
        deltas = position - self.preferred_positions
        distances_sq = np.sum(deltas * deltas, axis=1)
        return np.exp(-distances_sq / (2.0 * self.place_widths * self.place_widths))

    def _spatial_overlap(self, position, width_scale=1.0):
        projection = self._gaussian_projection(position)
        width_scale = max(0.35, width_scale)
        return np.clip(projection * width_scale, 0.0, 1.0)

    def _theta_drive(self, fatigue):
        base_frequency = 5.5 + 2.2 * max(0.0, 1.0 - fatigue)
        self.theta_phase = (self.theta_phase + 0.01 * base_frequency) % (2.0 * math.pi)
        return 0.5 + 0.5 * math.sin(self.theta_phase)

    def _update_head_direction_cells(self, heading_angle):
        for cell in self.head_direction_cells:
            difference = abs(heading_angle - cell.preferred_angle)
            difference = min(difference, 2.0 * math.pi - difference)
            selectivity = math.exp(-(difference ** 2) / (2.0 * 0.55 ** 2))
            cell.neuron.integrate(0.01, selectivity * 12.0)
            cell.activation = cell.neuron.activity_level()

    def _signal_projection(self, positions, sigma_scale=1.0):
        if not positions:
            return np.zeros(len(self.place_cells), dtype=float)
        accumulation = np.zeros(len(self.place_cells), dtype=float)
        for position, strength in positions:
            accumulation += self._spatial_overlap(position, width_scale=sigma_scale) * strength
        return np.clip(accumulation, 0.0, None)

    def _project_vector_to_open_space(self, vector, open_vectors):
        norm = np.linalg.norm(vector)
        if norm <= 1e-6:
            return np.zeros(2, dtype=float)
        if not open_vectors:
            return vector / norm

        target = vector / norm
        projected = np.zeros(2, dtype=float)
        best_open = max(open_vectors, key=lambda item: item[2])
        for vector_x, vector_y, weight in open_vectors:
            open_unit = np.array([vector_x, vector_y], dtype=float)
            dot_product = max(0.0, float(np.dot(target, open_unit)))
            projected += open_unit * (weight * dot_product * dot_product)

        if np.linalg.norm(projected) <= 1e-6:
            projected = np.array([best_open[0], best_open[1]], dtype=float)
        projected_norm = np.linalg.norm(projected)
        if projected_norm <= 1e-6:
            return target
        return projected / projected_norm

    def observe(
        self,
        position,
        heading_angle,
        reward,
        visible_food_sources,
        visible_peer_sources,
        heard_signals,
        fatigue,
        social_need,
    ):
        position = np.array(position, dtype=float)
        previous_food_expectation = float(self.food_expectation)
        sensory_projection = self._gaussian_projection(position)
        theta_drive = self._theta_drive(fatigue)
        self._update_head_direction_cells(heading_angle)

        food_projection = self._signal_projection(visible_food_sources, sigma_scale=1.15)

        visible_food_support = float(np.max(food_projection)) if len(food_projection) > 0 else 0.0
        external_drive = clamp(visible_food_support, 0.0, 2.5)
        self.external_drive = external_drive
        self.replay_gate = clamp(0.18 + fatigue * 0.55 - external_drive * 0.25, 0.0, 1.0)
        food_prediction_error = clamp(reward * 1.2 + visible_food_support * 0.9 - previous_food_expectation, -1.2, 1.2)
        self.food_prediction_error = 0.88 * self.food_prediction_error + 0.12 * food_prediction_error

        activations = np.zeros(len(self.place_cells), dtype=float)
        for index, cell in enumerate(self.place_cells):
            recurrent_input = 0.0
            for link in self.incoming_links[index]:
                recurrent_input += link.synapse.transmit(self.last_activations[link.source_idx])
            basal_input = sensory_projection[index] * (8.0 + 6.0 * theta_drive)
            basal_input += recurrent_input * (1.25 + self.replay_gate * MEMORY_REPLAY_GAIN)
            apical_input = reward * 9.0 + food_projection[index] * 2.8
            cell.neuron.integrate(0.01, basal_input, apical_input)
            activation = cell.neuron.activity_level() * sensory_projection[index]
            activations[index] = activation
            cell.last_activation = activation
            cell.occupancy_trace = OCCUPANCY_DECAY * cell.occupancy_trace + activation
            cell.replay_trace = PLACE_CELL_DECAY * cell.replay_trace + recurrent_input * 0.08
            cell.novelty = clamp(cell.novelty * (1.0 - 0.08 * activation) + NOVELTY_RECOVERY, 0.05, 1.3)

            cell.food_association = clamp(
                cell.food_association * ASSOCIATION_DECAY
                + PLACE_FIELD_LEARNING_RATE
                * activation
                * (reward * 1.6 + food_projection[index] * 0.55 + max(0.0, self.food_prediction_error) * 0.35),
                -0.9,
                3.2,
            )
            if reward <= 0.0 and visible_food_support < 0.16:
                extinction = 0.018 + max(0.0, 0.16 - visible_food_support) * 0.16 + max(0.0, -self.food_prediction_error) * 0.08
                cell.food_association = max(-0.95, cell.food_association - activation * extinction)

        for link in self.recurrent_links:
            presynaptic = self.place_cells[link.source_idx].neuron
            postsynaptic = self.place_cells[link.target_idx].neuron
            link.synapse.apply_stdp(presynaptic.last_spike_time, postsynaptic.last_spike_time)
            link.synapse.update_modulators(
                dopamine=clamp(0.5 + reward * 0.35, 0.0, 1.0),
                serotonin=clamp(0.45, 0.0, 1.0),
                acetylcholine=clamp(0.22 + external_drive * 0.18, 0.0, 1.0),
            )
            link.synapse.apply_short_term_plasticity(0.01, self.last_activations[link.source_idx] > 0.08)
            link.synapse.weight *= REPLAY_TRANSITION_DECAY

        food_values = np.array([cell.food_association for cell in self.place_cells], dtype=float)
        occupancy_values = np.array([cell.occupancy_trace for cell in self.place_cells], dtype=float)
        novelty_values = np.array([cell.novelty for cell in self.place_cells], dtype=float)
        self.food_expectation = float(np.dot(activations, food_values))
        local_occupancy = float(
            np.dot(activations, occupancy_values)
            / max(1e-6, float(np.sum(activations)))
        )
        local_novelty = float(
            np.dot(activations, novelty_values)
            / max(1e-6, float(np.sum(activations)))
        )
        if self.external_drive < 0.35:
            self.wander_angle = (
                self.wander_angle
                + self.rng.uniform(-0.12, 0.12)
                + (0.48 - local_novelty) * 0.22
                + local_occupancy * 0.08
            ) % (2.0 * math.pi)
        else:
            self.wander_angle = heading_angle
        self.last_activations = activations
        self.last_position = position
        self.last_heading = heading_angle

    def _candidate_directions(self, open_vectors):
        candidates = []
        for angle_index in range(MEMORY_PROBE_DIRECTIONS):
            angle = (angle_index / MEMORY_PROBE_DIRECTIONS) * 2.0 * math.pi
            unit = np.array([math.cos(angle), math.sin(angle)], dtype=float)
            openness = 0.0
            for vector_x, vector_y, weight in open_vectors:
                dot_product = max(0.0, vector_x * unit[0] + vector_y * unit[1])
                openness = max(openness, dot_product * weight)
            if not open_vectors:
                openness = 1.0
            if openness < 0.18:
                continue
            candidates.append((unit, openness))

        if not candidates:
            candidates.append((np.array([math.cos(self.last_heading), math.sin(self.last_heading)], dtype=float), 0.35))
            candidates.append((np.array([1.0, 0.0], dtype=float), 0.2))
            candidates.append((np.array([-1.0, 0.0], dtype=float), 0.2))
        return candidates

    def memory_vector(
        self,
        position,
        heading_angle,
        open_vectors,
        hunger_drive,
        social_need,
        distress_bias,
        curiosity_drive,
        frustration,
        social_comfort,
        reward_confidence,
        restlessness,
        stall_pressure,
    ):
        position = np.array(position, dtype=float)
        current_heading_vector = np.array([math.cos(heading_angle), math.sin(heading_angle)], dtype=float)
        wander_vector = np.array([math.cos(self.wander_angle), math.sin(self.wander_angle)], dtype=float)
        previous_vector = self.last_vector.copy()
        candidates = self._candidate_directions(open_vectors)
        food_values = np.array([cell.food_association for cell in self.place_cells], dtype=float)
        novelty_values = np.array([cell.novelty for cell in self.place_cells], dtype=float)
        occupancy_values = np.array([cell.occupancy_trace for cell in self.place_cells], dtype=float)
        replay_values = self.last_activations + np.array([cell.replay_trace for cell in self.place_cells], dtype=float)

        score_trace = []
        candidate_units = []
        candidate_openness = []
        candidate_habit = []
        candidate_novelty = []
        candidate_heading = []
        candidate_momentum = []
        for unit, openness in candidates:
            sample_position = position + unit * MEMORY_LOOKAHEAD_PX
            field = self._gaussian_projection(sample_position)
            food = float(np.dot(field, food_values))
            novelty = float(np.dot(field, novelty_values))
            occupancy = float(np.dot(field, occupancy_values))
            replay = float(np.dot(field, replay_values))
            heading_alignment = float(np.dot(unit, current_heading_vector))
            momentum_alignment = float(np.dot(unit, previous_vector)) if np.linalg.norm(previous_vector) > 1e-6 else 0.0
            wander_alignment = float(np.dot(unit, wander_vector))
            explore_bias = max(0.0, 0.95 - self.external_drive)
            ventral_value = hunger_drive * (food * 1.38 + replay * (0.05 + reward_confidence * 0.08))
            hippocampal_value = novelty * (0.22 + curiosity_drive * 0.64 + restlessness * 0.26)
            habit_cost = occupancy * (0.34 + frustration * 0.24 + restlessness * 0.22)

            score = ventral_value + hippocampal_value
            score += heading_alignment * HEAD_DIRECTION_GAIN
            # FIXED: Boost momentum influence (was 0.10Г—0.78=0.078, now 0.35Г—0.78=0.273)
            # Helps frogs stick to previously chosen direction instead of constantly recalculating
            score += momentum_alignment * MEMORY_VECTOR_MOMENTUM * 0.35
            # FIXED: Reduce wander exploration bias when searching (was always active)
            score += wander_alignment * explore_bias * (0.18 + curiosity_drive * 0.22 + restlessness * 0.12)
            score += openness * openness * (OPENNESS_GAIN * 2.3)
            score -= habit_cost

            score_trace.append(score)
            candidate_units.append(unit)
            candidate_openness.append(openness)
            candidate_habit.append(habit_cost)
            candidate_novelty.append(novelty)
            candidate_heading.append(heading_alignment)
            candidate_momentum.append(momentum_alignment)

        score_array = np.array(score_trace, dtype=float)
        if len(score_array) == 0:
            self.last_policy_metrics = {
                "vigor": 0.22,
                "search_burst": 0.30,
                "confidence": 0.0,
                "conflict": 1.0,
                "habit_pressure": 0.0,
                "selected_openness": 0.0,
                "loop_pressure": 1.0,
                "reorientation_drive": 0.55,
            }
            return tuple(normalize_vector((float(current_heading_vector[0]), float(current_heading_vector[1])))), 0.0

        best_index = int(np.argmax(score_array))
        best_score = float(score_array[best_index])
        if len(score_array) > 1:
            second_score = float(np.partition(score_array, -2)[-2])
        else:
            second_score = best_score - 0.25
        score_gap = best_score - second_score
        confidence = clamp(0.18 + score_gap * 0.45, 0.0, 1.0)
        conflict = clamp(1.0 - score_gap / (0.45 + abs(best_score) + abs(second_score) * 0.35), 0.0, 1.0)

        temperature = clamp(0.78 + curiosity_drive * 0.38 + restlessness * 0.18 - reward_confidence * 0.10, 0.60, 1.45)
        stable_scores = (score_array - np.max(score_array)) / max(0.25, temperature)
        probabilities = np.exp(stable_scores)
        probabilities /= np.sum(probabilities)

        combined_vector = np.zeros(2, dtype=float)
        for idx, probability in enumerate(probabilities):
            combined_vector += candidate_units[idx] * probability * (0.42 + candidate_openness[idx] * 0.58)
        combined_vector += candidate_units[best_index] * (0.30 + confidence * 0.42)

        candidate_novelty_arr = np.array(candidate_novelty, dtype=float)
        candidate_heading_arr = np.array(candidate_heading, dtype=float)
        candidate_momentum_arr = np.array(candidate_momentum, dtype=float)
        if np.linalg.norm(combined_vector) <= 1e-6:
            combined_vector = current_heading_vector

        habit_pressure = clamp(float(np.dot(probabilities, np.array(candidate_habit, dtype=float))), 0.0, 1.0)
        loop_pressure = clamp(
            stall_pressure * 0.62
            + habit_pressure * 0.34
            + conflict * 0.26
            + max(0.0, 0.12 - best_score) * 0.10
            + max(0.0, 0.28 - reward_confidence) * 0.08
            + max(0.0, 0.35 - self.external_drive) * 0.08,
            0.0,
            1.35,
        )
        reorientation_drive = clamp(
            loop_pressure * (0.38 + curiosity_drive * 0.22 + restlessness * 0.18)
            - social_need * 0.10
            - max(0.0, best_score) * 0.04,
            0.0,
            1.0,
        )
        if reorientation_drive > 0.02:
            reverse_heading = np.maximum(0.0, -candidate_heading_arr)
            reverse_momentum = np.maximum(0.0, -candidate_momentum_arr)
            reorientation_scores = (
                candidate_novelty_arr * (0.82 + curiosity_drive * 0.24)
                + np.array(candidate_openness, dtype=float) * (0.54 + restlessness * 0.18)
                + reverse_heading * 0.26
                + reverse_momentum * 0.22
                - np.array(candidate_habit, dtype=float) * (0.48 + frustration * 0.22)
            )
            reorient_index = int(np.argmax(reorientation_scores))
            self.reorientation_vector = candidate_units[reorient_index].copy()
            combined_vector = combined_vector * (1.0 - reorientation_drive * 0.46) + self.reorientation_vector * (
                0.18 + reorientation_drive * 0.92
            )
        else:
            self.reorientation_vector = np.zeros(2, dtype=float)

        normalized = np.array(normalize_vector((float(combined_vector[0]), float(combined_vector[1]))), dtype=float)
        normalized = self._project_vector_to_open_space(normalized, open_vectors)
        if np.linalg.norm(normalized) > 1e-6:
            # FIXED: Aggressive persistence boost for stable path-following
            # Old: 0.20-0.42 (forgot direction in 4-5 frames) в†’ causes circling
            # New: 0.60-0.82 (remembers direction 20+ frames) в†’ long-term path commitment
            persistence = 0.60 + max(0.0, 0.28 - reorientation_drive) * 0.22
            self.last_vector = previous_vector * persistence + normalized * (1.0 - persistence)
        else:
            self.last_vector = previous_vector * 0.80

        search_burst = clamp(
            conflict * (0.24 + frustration * 0.44 + restlessness * 0.26)
            + habit_pressure * (0.18 + restlessness * 0.18)
            + reorientation_drive * (0.18 + curiosity_drive * 0.12)
            + max(0.0, 0.40 - self.external_drive) * (0.14 + curiosity_drive * 0.18)
            + max(0.0, 0.22 - confidence) * 0.24,
            0.0,
            1.15,
        )
        vigor = clamp(
            0.24
            + hunger_drive * 0.30
            + curiosity_drive * 0.16
            + restlessness * 0.14
            + search_burst * 0.28
            + max(0.0, best_score) * 0.05
            - self.replay_gate * 0.04,
            0.18,
            1.35,
        )
        self.last_policy_metrics = {
            "vigor": float(vigor),
            "search_burst": float(search_burst),
            "confidence": float(confidence),
            "conflict": float(conflict),
            "habit_pressure": float(habit_pressure),
            "selected_openness": float(candidate_openness[best_index]),
            "loop_pressure": float(loop_pressure),
            "reorientation_drive": float(reorientation_drive),
        }
        self.reorientation_drive = float(reorientation_drive)
        return tuple(normalize_vector((float(self.last_vector[0]), float(self.last_vector[1])))), best_score

    def snapshot(self):
        return {
            "food": [cell.food_association for cell in self.place_cells],
            "novelty": [cell.novelty for cell in self.place_cells],
            "occupancy": [cell.occupancy_trace for cell in self.place_cells],
            "food_prediction_error": float(self.food_prediction_error),
        }

