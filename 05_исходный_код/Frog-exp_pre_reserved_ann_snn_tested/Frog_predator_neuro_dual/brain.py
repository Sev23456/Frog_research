"""Biological frog brain with distributed spatial memory and predatory hunting."""

import copy
import math
import random
from dataclasses import dataclass

import numpy as np

from Frog_predator_neuro_dual.architecture import BasalGanglia, MotorHierarchy, RetinalProcessing, SpatialMemory, Tectum
from Frog_predator_neuro_dual.config import (
    AGENT_BODY_ENERGY_MAX,
    AGENT_CRITICAL_THRESHOLD,
    JUVENILE_STEPS,
    RETINA_FILTERS_PER_SIDE,
    TECTUM_COLUMNS,
    TILE_SIZE,
)
from Frog_predator_neuro_dual.core import NeurotransmitterDiffusion, PyramidalNeuron
from Frog_predator_neuro_dual.metabolism.systemic_metabolism import NeuronMetabolism, SystemicMetabolism
from Frog_predator_neuro_dual.plasticity.functional_plasticity import FunctionalPlasticityManager, StructuralPlasticityManager
from Frog_predator_neuro_dual.utils import add_vectors, clamp, normalize_vector, scale_vector, tile_center


@dataclass
class BrainProfile:
    exploration_gain: float = 1.0
    memory_gain: float = 1.0
    food_signal_gain: float = 1.0
    metabolism_efficiency: float = 1.0
    persistence_gain: float = 1.0


@dataclass
class AffectiveState:
    arousal: float = 0.32
    curiosity: float = 0.55
    frustration: float = 0.08
    reward_confidence: float = 0.28
    restlessness: float = 0.34


class BioFrogBrain:
    def __init__(self, juvenile_mode=True, rng=None):
        self.dt = 0.01
        self.rng = rng or random.Random()
        self.is_juvenile = juvenile_mode
        self.juvenile_duration = JUVENILE_STEPS
        self.juvenile_age = 0
        self.maturity_readiness = 0.08 if juvenile_mode else 1.0
        self.maturity_stability = 0.02 if juvenile_mode else 1.0
        self.self_selected_maturity_age = None

        self.visual_system = RetinalProcessing(visual_field_size=(5.0, 5.0), num_filters_per_side=RETINA_FILTERS_PER_SIDE)
        self.tectum = Tectum(columns=TECTUM_COLUMNS)
        self.basal_ganglia = BasalGanglia()
        self.motor_hierarchy = MotorHierarchy()
        self.spatial_memory = SpatialMemory(rng=self.rng)

        self.neurotransmitter_system = NeurotransmitterDiffusion(space_size=(360, 360), grid_resolution=28)
        self.metabolism = SystemicMetabolism()
        self.neuron_metabolism = NeuronMetabolism()
        self.functional_plasticity = FunctionalPlasticityManager()
        self.structural_plasticity = StructuralPlasticityManager()

        self.num_astrocytes = 12
        self.profile = BrainProfile()
        self.affect = AffectiveState(
            arousal=0.40 if juvenile_mode else 0.32,
            curiosity=0.68 if juvenile_mode else 0.52,
            frustration=0.04 if juvenile_mode else 0.08,
            reward_confidence=0.24,
            restlessness=0.46 if juvenile_mode else 0.34,
        )

        self.dopamine_level = 0.85 if juvenile_mode else 0.5
        self.serotonin_level = 0.75 if juvenile_mode else 0.5
        self.acetylcholine_level = 0.3
        self.steps = 0
        self.last_reward = 0.0
        self.last_memory_score = 0.0
        self.last_velocity = (0.0, 0.0)
        self.last_desired_vector = (0.0, 0.0)
        self.last_gating_signal = 0.0
        self.last_action_vigor = 0.28
        self.last_search_burst = 0.16
        self.last_locomotor_tone = 0.18
        self.last_lateral_inhibition = 0.0
        self.last_memory_vector = (0.0, 0.0)
        self.last_motivation_context = {}
        self.scan_phase = 0.0
        self.last_food_vector = (0.0, 0.0)
        self.prey_permission = 0.0
        self.fast_target_lock = 0.0
        self.fast_orient_gain = 0.0
        self.fast_loop_gate = 0.0
        self.fast_strike_drive = 0.0
        self.slow_hunt_drive = 0.0

        self._neuron_cache = []
        self._synapse_cache = []
        self._neural_positions = np.zeros((0, 2), dtype=float)
        self.refresh_caches()

    def refresh_caches(self):
        neurons = []
        neurons.extend(filter_cell.center_neuron for filter_cell in self.visual_system.filters)
        for column in self.tectum.columns:
            neurons.extend(column.pyramidal_neurons)
            neurons.extend(column.interneurons)
            neurons.extend(column.output_neurons)
        neurons.extend(self.basal_ganglia.all_neurons())
        neurons.extend(self.motor_hierarchy.all_neurons())
        neurons.extend(self.spatial_memory.all_neurons())
        self._neuron_cache = list(neurons)

        synapses = []
        for column in self.tectum.columns:
            synapses.extend(column.synapses)
        synapses.extend(self.spatial_memory.all_synapses())
        self._synapse_cache = synapses

        grid_width = int(math.ceil(math.sqrt(max(1, len(self._neuron_cache)))))
        self._neural_positions = np.array(
            [[(index % grid_width) * 12.0, (index // grid_width) * 12.0] for index in range(len(self._neuron_cache))],
            dtype=float,
        )

    def clone(self, heterogeneous=False, rng=None):
        rng = rng or random.Random()
        clone = copy.deepcopy(self)
        if heterogeneous:
            clone.apply_variation(rng)
        return clone

    def apply_variation(self, rng):
        self.profile.exploration_gain *= rng.uniform(0.82, 1.18)
        self.profile.memory_gain *= rng.uniform(0.82, 1.18)
        self.profile.food_signal_gain *= rng.uniform(0.82, 1.20)
        self.profile.metabolism_efficiency *= rng.uniform(0.85, 1.15)
        self.profile.persistence_gain *= rng.uniform(0.82, 1.18)

        self.metabolism.glucose_consumption_rate *= rng.uniform(0.9, 1.1)
        self.metabolism.oxygen_consumption_rate *= rng.uniform(0.9, 1.1)
        self.functional_plasticity.homeostatic_learning_rate *= rng.uniform(0.85, 1.20)

        for neuron in self.all_neurons():
            neuron.threshold += rng.uniform(-2.2, 2.2)
            neuron.tau_membrane *= rng.uniform(0.92, 1.08)

    def update_affective_state(
        self,
        perception,
        body_energy,
        reward,
        observational_reward,
        memory_score,
        exploration_bonus,
        neural_activity,
    ):
        visible_food_count = len(perception["visible_food_tiles"])
        fatigue = self.metabolism.fatigue_level
        expectation_gap = clamp(max(0.0, memory_score) * 0.38 + self.spatial_memory.food_expectation * 0.05, 0.0, 1.2)
        unrewarded_probe = 1.0 if reward <= 0.0 and visible_food_count == 0 and memory_score > 0.18 else 0.0
        novelty_pressure = clamp(1.0 - self.spatial_memory.external_drive + exploration_bonus * 2.5, 0.0, 1.4)
        food_salience = clamp(visible_food_count * 0.45, 0.0, 1.0)
        normalized_activity = clamp(neural_activity, 0.0, 1.0)
        hunger_pressure = clamp(max(0.0, 0.68 - body_energy) / 0.68, 0.0, 1.0)
        energy_surplus = clamp((body_energy - 0.72) / 0.20, 0.0, 1.0)
        desperation = clamp((0.60 - body_energy) / 0.30, 0.0, 1.0)
        recent_progress = clamp(perception.get("recent_progress", 0.0), 0.0, 1.0)

        target_frustration = clamp(
            0.03
            + unrewarded_probe * (0.18 + expectation_gap * 0.18)
            + fatigue * 0.08
            - reward * 0.40
            - observational_reward * 0.26
            - food_salience * 0.12,
            0.0,
            1.0,
        )
        target_reward_confidence = clamp(
            0.16
            + reward * 0.90
            + observational_reward * 0.34
            + food_salience * 0.32
            + max(0.0, self.spatial_memory.food_expectation) * 0.08
            - target_frustration * 0.10
            - hunger_pressure * 0.08,
            0.0,
            1.0,
        )
        target_curiosity = clamp(
            0.18
            + novelty_pressure * 0.35
            + target_reward_confidence * 0.12
            + energy_surplus * 0.28
            + desperation * 0.10
            + observational_reward * 0.10
            - target_frustration * 0.25
            - fatigue * 0.18,
            0.0,
            1.0,
        )
        target_arousal = clamp(
            0.18
            + normalized_activity * 0.38
            + food_salience * 0.24
            + reward * 0.20
            + observational_reward * 0.08
            + energy_surplus * 0.06
            + desperation * 0.26
            - fatigue * 0.14,
            0.0,
            1.0,
        )
        target_restlessness = clamp(
            0.14
            + max(0.0, 0.55 - recent_progress) * 0.58
            + novelty_pressure * 0.12
            + max(0.0, 0.24 - reward) * 0.04
            + observational_reward * 0.06
            + energy_surplus * 0.28
            + desperation * 0.24
            - food_salience * 0.42
            - fatigue * 0.18,
            0.0,
            1.0,
        )

        # Increased update rates for faster emotional response
        self.affect.frustration = clamp(self.affect.frustration * 0.920 + target_frustration * 0.080, 0.0, 1.0)
        self.affect.reward_confidence = clamp(self.affect.reward_confidence * 0.920 + target_reward_confidence * 0.080, 0.08, 0.92)
        self.affect.curiosity = clamp(self.affect.curiosity * 0.920 + target_curiosity * 0.080, 0.10, 0.92)
        self.affect.arousal = clamp(self.affect.arousal * 0.920 + target_arousal * 0.080, 0.10, 0.84)
        self.affect.restlessness = clamp(self.affect.restlessness * 0.920 + target_restlessness * 0.080, 0.06, 0.88)

    def all_neurons(self):
        return self._neuron_cache

    def all_synapses(self):
        return self._synapse_cache

    def local_affordance(self, direction, open_vectors):
        if not open_vectors:
            return 0.0
        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            return 0.0
        unit = direction / norm
        best = 0.0
        for vector_x, vector_y, weight in open_vectors:
                dot_product = max(0.0, vector_x * unit[0] + vector_y * unit[1])
                best = max(best, dot_product * clamp(weight, 0.0, 1.0))
        return clamp(best, 0.0, 1.0)

    def build_motivation_context(self, perception, body_energy):
        visible_food_count = len(perception.get("visible_food_tiles", []))
        memory_bias = clamp(max(0.0, self.last_memory_score), 0.0, 1.0)
        energy_ratio = clamp(body_energy / max(1e-6, AGENT_BODY_ENERGY_MAX), 0.0, 1.0)
        food_prediction_error = float(self.spatial_memory.food_prediction_error)

        hunger_bias = clamp(
            max(0.0, 1.0 - energy_ratio) * 0.92
            + self.affect.frustration * 0.12
            + (1.0 - self.affect.reward_confidence) * 0.10
            + max(0.0, -food_prediction_error) * 0.10,
            0.0,
            1.35,
        )
        vigilance_bias = clamp(
            self.affect.arousal * 0.40
            + self.affect.frustration * 0.24
            + self.affect.restlessness * 0.16
            + max(0.0, -food_prediction_error) * 0.08,
            0.0,
            1.0,
        )
        exploration_bias = clamp(
            self.affect.curiosity * 0.46
            + self.affect.restlessness * 0.30
            + max(0.0, 0.30 - self.spatial_memory.external_drive) * 0.28
            + max(0.0, 0.20 - memory_bias) * 0.12
            - visible_food_count * 0.08
            - max(0.0, food_prediction_error) * 0.04,
            0.0,
            1.15,
        )
        reward_seek_bias = clamp(
            self.affect.reward_confidence * 0.40
            + memory_bias * 0.14
            + max(0.0, self.spatial_memory.food_expectation) * 0.14
            + max(0.0, food_prediction_error) * 0.18
            + visible_food_count * 0.05,
            0.0,
            1.0,
        )
        locomotor_bias = clamp(
            0.18
            + self.affect.arousal * 0.16
            + exploration_bias * 0.14
            + hunger_bias * 0.12
            - self.affect.frustration * 0.06,
            0.10,
            1.0,
        )
        predation_bias = clamp(
            0.16
            + hunger_bias * 0.30
            + vigilance_bias * 0.20
            + reward_seek_bias * 0.18
            + visible_food_count * 0.06,
            0.0,
            1.20,
        )
        context = {
            "hunger_bias": float(hunger_bias),
            "social_bias": 0.0,
            "vigilance_bias": float(vigilance_bias),
            "exploration_bias": float(exploration_bias),
            "reward_seek_bias": float(reward_seek_bias),
            "locomotor_bias": float(locomotor_bias),
            "predation_bias": float(predation_bias),
        }
        self.last_motivation_context = context
        return context

    def arbitrate_action_selection(
        self,
        *,
        tectum_vector,
        food_vector,
        memory_vector,
        novelty_vector,
        open_vector,
        reorientation_vector,
        body_energy,
        tectum_strength,
        food_directness,
        memory_strength,
        memory_score,
        novelty_strength,
        open_strength,
        policy_confidence,
        policy_conflict,
        habit_pressure,
        loop_pressure,
        reorientation_drive,
    ):
        module_drive_strength = clamp(
            tectum_strength * 0.18
            + food_directness * 0.26
            + memory_strength * 0.20
            + novelty_strength * 0.14
            + reorientation_drive * 0.14
            + open_strength * 0.08,
            0.0,
            1.0,
        )
        goal_commitment = clamp(
            food_directness * 0.30
            + max(0.0, memory_score) * 0.20
            + policy_confidence * 0.22
            + reorientation_drive * 0.14
            + tectum_strength * 0.10,
            0.0,
            1.0,
        )
        search_pressure = clamp(
            policy_conflict * 0.32
            + habit_pressure * 0.22
            + loop_pressure * 0.14
            + novelty_strength * 0.16
            + max(0.0, 0.24 - food_directness) * 0.14
            + max(0.0, 0.18 - self.spatial_memory.external_drive) * 0.14,
            0.0,
            1.0,
        )
        action_vigor = clamp(
            0.22
            + module_drive_strength * 0.42
            + goal_commitment * 0.28
            + max(0.0, 0.10 - search_pressure) * 0.18,
            0.18,
            1.15,
        )
        search_burst = clamp(
            0.06
            + search_pressure * 0.72
            + max(0.0, 0.28 - goal_commitment) * 0.12,
            0.0,
            1.10,
        )
        action_vigor = self.last_action_vigor * 0.82 + action_vigor * 0.18
        search_burst = self.last_search_burst * 0.84 + search_burst * 0.16
        self.last_action_vigor = float(action_vigor)
        self.last_search_burst = float(search_burst)

        raw_lateral_inhibition = clamp(food_directness * 1.45, 0.0, 1.0)
        inhibition_alpha = 0.16
        lateral_inhibition = raw_lateral_inhibition * inhibition_alpha + self.last_lateral_inhibition * (1.0 - inhibition_alpha)
        self.last_lateral_inhibition = float(lateral_inhibition)
        conflict_damping = clamp(1.0 - policy_conflict * 0.24 - loop_pressure * 0.10, 0.66, 1.0)

        tectum_scaled = scale_vector(tectum_vector, 0.70 + tectum_strength * 0.24 + food_directness * 0.20)
        food_scaled = scale_vector(food_vector, 0.46 + food_directness * 1.10)
        novelty_scaled = scale_vector(
            novelty_vector,
            (0.16 + novelty_strength * (0.54 + (0.18 if self.is_juvenile else 0.0)))
            * (1.0 - lateral_inhibition * 0.60)
            * conflict_damping,
        )
        memory_scaled = scale_vector(
            memory_vector,
            (0.42 + memory_strength * 0.22 + max(0.0, memory_score) * 0.12)
            * (1.0 - food_directness * 0.40)
            * (1.0 - lateral_inhibition * 0.68)
            * conflict_damping,
        )
        reorientation_scaled = scale_vector(reorientation_vector, 0.08 + reorientation_drive * 0.82)
        open_scaled = scale_vector(
            open_vector,
            (0.06 + open_strength * 0.20 + max(0.0, 0.35 - food_directness) * 0.14)
            * (1.0 - lateral_inhibition * 0.50)
            * conflict_damping,
        )
        velocity_scaled = scale_vector(self.last_velocity, (0.10 + policy_confidence * 0.18 + max(0.0, memory_strength) * 0.08) * (1.0 - lateral_inhibition * 0.30))

        desired_vector = add_vectors(
            tectum_scaled,
            food_scaled,
            novelty_scaled,
            memory_scaled,
            reorientation_scaled,
            open_scaled,
            velocity_scaled,
        )
        desired_vector = normalize_vector(desired_vector)

        integrated_command = add_vectors(
            tectum_scaled,
            food_scaled,
            novelty_scaled,
            memory_scaled,
            reorientation_scaled,
            open_scaled,
        )
        pre_normalized_strength = clamp(float(np.linalg.norm(integrated_command)), 0.0, 1.6)
        desired_alpha = clamp(
            0.12
            + food_directness * 0.12
            + reorientation_drive * 0.18
            + policy_confidence * 0.10
            - policy_conflict * 0.06,
            0.10,
            0.34,
        )
        smoothed_desired = add_vectors(
            scale_vector(self.last_desired_vector, 1.0 - desired_alpha),
            scale_vector(desired_vector, desired_alpha),
        )
        if np.linalg.norm(smoothed_desired) > 1e-6:
            desired_vector = normalize_vector(smoothed_desired)
        self.last_desired_vector = desired_vector if np.linalg.norm(desired_vector) > 1e-6 else scale_vector(self.last_desired_vector, 0.94)

        cortical_drives = np.array(
            [
                tectum_strength,
                food_directness,
                memory_strength,
                novelty_strength,
                reorientation_drive,
                clamp(open_strength * 0.50 + novelty_strength * 0.50, 0.0, 1.0),
                clamp(policy_confidence * 0.60 + goal_commitment * 0.40, 0.0, 1.0),
                clamp(search_pressure * 0.45 + habit_pressure * 0.25 + loop_pressure * 0.30, 0.0, 1.0),
            ],
            dtype=float,
        )
        bg_result = self.basal_ganglia.select_action(np.clip(cortical_drives, 0.0, 1.0), self.dopamine_level, can_inhibit=True)
        gating_signal = bg_result["gating_signal"]
        decision_confidence = bg_result["confidence"]
        gating_alpha = clamp(
            0.10 + decision_confidence * 0.10 + food_directness * 0.08 + reorientation_drive * 0.10,
            0.10,
            0.30,
        )
        gating_signal = gating_signal * gating_alpha + self.last_gating_signal * (1.0 - gating_alpha)
        self.last_gating_signal = float(gating_signal)

        command_strength = clamp(
            pre_normalized_strength * 0.55 + goal_commitment * 0.30 + module_drive_strength * 0.15,
            0.0,
            1.4,
        )
        locomotor_tone = clamp(
            0.16
            + command_strength * 0.24
            + decision_confidence * 0.16
            + food_directness * 0.12
            + reorientation_drive * 0.10,
            0.12,
            1.0,
        )
        locomotor_tone = self.last_locomotor_tone * 0.80 + locomotor_tone * 0.20
        self.last_locomotor_tone = float(locomotor_tone)

        return {
            "desired_vector": desired_vector,
            "action_vigor": float(action_vigor),
            "search_burst": float(search_burst),
            "goal_commitment": float(goal_commitment),
            "search_pressure": float(search_pressure),
            "module_drive_strength": float(module_drive_strength),
            "command_strength": float(command_strength),
            "locomotor_tone": float(locomotor_tone),
            "bg_gating_signal": float(gating_signal),
            "bg_confidence": float(decision_confidence),
            "policy_confidence": float(policy_confidence),
            "policy_conflict": float(policy_conflict),
            "lateral_inhibition": float(lateral_inhibition),
            "conflict_damping": float(conflict_damping),
            "bg_result": bg_result,
        }

    def build_visual_scene(self, perception):
        current_tile = perception["current_tile"]
        scene = []
        for tile in perception["visible_walls"]:
            rel = (tile[0] - current_tile[0] + 2.5, tile[1] - current_tile[1] + 2.5, 0.45)
            scene.append(rel)
        for tile in perception["visible_food_tiles"]:
            rel = (tile[0] - current_tile[0] + 2.5, tile[1] - current_tile[1] + 2.5, 1.0)
            scene.append(rel)
        return scene

    def compute_open_drive(self, perception, memory_vector, motivation_context=None):
        open_vectors = perception.get("open_vectors", [])
        if not open_vectors:
            return (0.0, 0.0)

        motivation_context = motivation_context or self.last_motivation_context
        heading_angle = perception["heading_angle"]
        heading_vector = np.array([math.cos(heading_angle), math.sin(heading_angle)], dtype=float)
        wander_vector = np.array(
            [math.cos(self.spatial_memory.wander_angle), math.sin(self.spatial_memory.wander_angle)],
            dtype=float,
        )
        memory_np = np.array(memory_vector, dtype=float) if memory_vector != (0.0, 0.0) else np.zeros(2, dtype=float)
        exploration_bias = float(motivation_context.get("exploration_bias", 0.35))
        reward_seek_bias = float(motivation_context.get("reward_seek_bias", 0.20))
        vigilance_bias = float(motivation_context.get("vigilance_bias", 0.20))

        combined = np.zeros(2, dtype=float)
        for open_x, open_y, weight in open_vectors:
            unit = np.array([open_x, open_y], dtype=float)
            heading_align = max(0.0, float(np.dot(unit, heading_vector)))
            wander_align = max(0.0, float(np.dot(unit, wander_vector)))
            memory_align = max(0.0, float(np.dot(unit, memory_np))) if np.linalg.norm(memory_np) > 1e-6 else 0.0
            score = weight * (
                0.28
                + heading_align * (0.18 + reward_seek_bias * 0.06)
                + wander_align * (0.24 + exploration_bias * 0.24)
                + memory_align * (0.28 + reward_seek_bias * 0.18)
                - vigilance_bias * 0.04
            )
            combined += unit * score

        norm = np.linalg.norm(combined)
        if norm <= 1e-6:
            strongest = max(open_vectors, key=lambda item: item[2])
            return normalize_vector((float(strongest[0]), float(strongest[1])))
        return tuple((combined / norm).tolist())

    def build_salience_vectors(self, perception, body_energy, motivation_context=None):
        current_px = np.array(perception["position"], dtype=float)
        vectors = []
        motivation_context = motivation_context or self.last_motivation_context
        arousal_gain = 0.82 + self.affect.arousal * 0.35 + motivation_context.get("locomotor_bias", 0.2) * 0.10
        open_vectors = perception.get("open_vectors", [])
        hunger_signal = clamp(float(motivation_context.get("hunger_bias", max(0.0, 1.0 - body_energy / max(1e-6, AGENT_BODY_ENERGY_MAX)))), 0.0, 1.4)
        vigilance_bias = float(motivation_context.get("vigilance_bias", 0.2))
        reward_seek_bias = float(motivation_context.get("reward_seek_bias", 0.2))

        for tile in perception["visible_food_tiles"]:
            target_px = np.array(tile_center(tile), dtype=float)
            direction = target_px - current_px
            distance = np.linalg.norm(direction)
            if distance > 0:
                affordance = self.local_affordance(direction, open_vectors)
                if affordance < 0.12 and distance > TILE_SIZE * 0.85:
                    continue
                motor_affordance = affordance ** 1.45
                local_capture = clamp(1.0 - distance / (1.05 * TILE_SIZE), 0.0, 1.0)
                gain = 0.20 + motor_affordance * 1.32 + local_capture * 1.10
                food_drive = (
                    0.14
                    + hunger_signal * hunger_signal * 1.95
                    + local_capture * (0.32 + vigilance_bias * 0.12)
                    + reward_seek_bias * 0.14
                )
                vectors.append((direction / distance) * food_drive * arousal_gain * gain)

        for tile in perception["visible_walls"]:
            target_px = np.array(tile_center(tile), dtype=float)
            direction = current_px - target_px
            distance = np.linalg.norm(direction)
            if distance > 0:
                vectors.append((direction / distance) * max(0.0, 1.15 - distance / (2.25 * TILE_SIZE)) * (0.9 + self.affect.arousal * 0.2))

        if not vectors:
            vectors.append(np.zeros(2))
        return vectors

    def compute_food_capture_vector(self, perception, body_energy, motivation_context=None):
        if not perception["visible_food_tiles"]:
            return (0.0, 0.0), 0.0

        current_px = np.array(perception["position"], dtype=float)
        open_vectors = perception.get("open_vectors", [])
        motivation_context = motivation_context or self.last_motivation_context
        hunger_signal = clamp(float(motivation_context.get("hunger_bias", max(0.0, 1.0 - body_energy / max(1e-6, AGENT_BODY_ENERGY_MAX)))), 0.0, 1.4)
        reward_seek_bias = float(motivation_context.get("reward_seek_bias", 0.2))
        vigilance_bias = float(motivation_context.get("vigilance_bias", 0.2))
        best_direction = np.zeros(2, dtype=float)
        best_score = 0.0

        for tile in perception["visible_food_tiles"]:
            target_px = np.array(tile_center(tile), dtype=float)
            direction = target_px - current_px
            distance = np.linalg.norm(direction)
            if distance <= 1e-6:
                continue
            affordance = self.local_affordance(direction, open_vectors)
            proximity = clamp(1.0 - distance / (1.40 * TILE_SIZE), 0.0, 1.0)
            score = affordance * (0.70 + reward_seek_bias * 0.14)
            score += proximity * (0.78 + hunger_signal * 0.18)
            score += hunger_signal * 0.16
            score += vigilance_bias * 0.04
            if score > best_score:
                best_score = score
                best_direction = direction / distance

        if best_score <= 0.0:
            return (0.0, 0.0), 0.0
        return (float(best_direction[0]), float(best_direction[1])), clamp(best_score, 0.0, 1.4)

    def compute_developmental_novelty_vector(self, perception, body_energy, motivation_context=None):
        if perception["visible_food_tiles"]:
            return (0.0, 0.0), 0.0

        current_px = np.array(perception["position"], dtype=float)
        open_vectors = perception.get("open_vectors", [])
        visible_unvisited = perception.get("visible_unvisited_tiles", set())
        time_since_reward = float(perception.get("time_since_reward", 999.0))
        food_collected = perception.get("food_collected", 0)
        motivation_context = motivation_context or self.last_motivation_context
        exploration_bias = float(motivation_context.get("exploration_bias", 0.35))
        vigilance_bias = float(motivation_context.get("vigilance_bias", 0.20))
        energy_ratio = clamp(body_energy / max(1e-6, AGENT_BODY_ENERGY_MAX), 0.0, 1.0)
        novelty_pressure = clamp(
            (0.70 if self.is_juvenile else 0.22)
            + clamp((time_since_reward - 4.0) / 18.0, 0.0, 1.0) * (0.62 if self.is_juvenile else 0.18)
            + max(0.0, 0.78 - energy_ratio) * 0.12
            + max(0.0, 1 - food_collected) * (0.26 if self.is_juvenile else 0.08)
            + exploration_bias * 0.52
            - vigilance_bias * 0.10,
            0.0,
            1.6,
        )
        if novelty_pressure <= 0.02:
            return (0.0, 0.0), 0.0

        best_direction = np.zeros(2, dtype=float)
        best_score = 0.0
        weighted = np.zeros(2, dtype=float)

        for tile in visible_unvisited:
            target_px = np.array(tile_center(tile), dtype=float)
            direction = target_px - current_px
            distance = np.linalg.norm(direction)
            if distance <= 1e-6:
                continue
            affordance = self.local_affordance(direction, open_vectors)
            if affordance <= 0.04:
                continue
            normalized_distance = clamp(distance / (2.4 * TILE_SIZE), 0.18, 1.0)
            heading_alignment = max(
                0.0,
                float(
                    np.dot(
                        direction / distance,
                        np.array([math.cos(perception["heading_angle"]), math.sin(perception["heading_angle"])], dtype=float),
                    )
                ),
            )
            score = affordance * (0.44 + normalized_distance * 0.46 + heading_alignment * 0.10)
            weighted += (direction / distance) * score
            if score > best_score:
                best_score = score
                best_direction = direction / distance

        if np.linalg.norm(weighted) > 1e-6:
            direction = normalize_vector((float(weighted[0]), float(weighted[1])))
        elif best_score > 0.0:
            direction = (float(best_direction[0]), float(best_direction[1]))
        elif open_vectors:
            breakout = max(open_vectors, key=lambda item: item[2] * (1.0 - abs(item[0] * self.last_velocity[0] + item[1] * self.last_velocity[1])))
            direction = normalize_vector((float(breakout[0]), float(breakout[1])))
            best_score = breakout[2]
        else:
            return (0.0, 0.0), 0.0

        return direction, clamp(best_score * novelty_pressure, 0.0, 1.3)

    def update_spatial_memory(self, perception, reward, body_energy, motivation_context=None):
        current_position = np.array(perception["position"], dtype=float)
        open_vectors = perception.get("open_vectors", [])
        motivation_context = motivation_context or self.last_motivation_context
        hunger_bias = float(motivation_context.get("hunger_bias", 0.35))
        reward_seek_bias = float(motivation_context.get("reward_seek_bias", 0.20))

        visible_food_sources = []
        for tile in perception["visible_food_tiles"]:
            target_position = tile_center(tile)
            direction = np.array(target_position, dtype=float) - current_position
            distance = np.linalg.norm(direction)
            affordance = self.local_affordance(direction, open_vectors)
            if distance <= TILE_SIZE * 0.75:
                strength = 1.0
            else:
                strength = clamp((0.08 + affordance * 0.92) ** 1.65, 0.03, 1.0)
                if affordance < 0.12 and distance > TILE_SIZE * 0.85:
                    strength *= 0.12
            strength = clamp(strength * (0.84 + hunger_bias * 0.22 + reward_seek_bias * 0.10), 0.03, 1.0)
            visible_food_sources.append((target_position, strength))

        hunger_drive = clamp(
            hunger_bias * self.profile.memory_gain
            + self.affect.frustration * 0.06
            + (1.0 - self.affect.reward_confidence) * 0.08,
            0.05,
            1.6,
        )
        memory_vector, memory_score = self.spatial_memory.memory_vector(
            position=perception["position"],
            heading_angle=perception["heading_angle"],
            open_vectors=perception["open_vectors"],
            hunger_drive=hunger_drive,
            social_need=0.0,
            distress_bias=0.0,
            curiosity_drive=self.affect.curiosity,
            frustration=self.affect.frustration,
            social_comfort=0.0,
            reward_confidence=self.affect.reward_confidence,
            restlessness=self.affect.restlessness,
            stall_pressure=clamp(perception.get("stall_time", 0.0) / 3.2, 0.0, 1.2),
        )
        self.spatial_memory.observe(
            position=perception["position"],
            heading_angle=perception["heading_angle"],
            reward=reward,
            visible_food_sources=visible_food_sources,
            visible_peer_sources=[],
            heard_signals=[],
            fatigue=self.metabolism.fatigue_level,
            social_need=0.0,
        )
        self.last_memory_score = memory_score
        self.last_memory_vector = memory_vector
        return memory_vector, memory_score

    def update_neuromodulation(self, reward, observational_reward, exploration_bonus, neural_activity):
        dopamine_base = 0.85 if self.is_juvenile else 0.5
        serotonin_base = 0.75 if self.is_juvenile else 0.5
        target_dopamine = np.clip(
            dopamine_base
            + 0.32 * reward
            + 0.14 * observational_reward
            + 0.12 * exploration_bonus
            + 0.08 * self.affect.curiosity
            - 0.08 * self.affect.frustration
            - 0.1 * self.metabolism.fatigue_level,
            0.22 if self.is_juvenile else 0.20,
            0.86 if self.is_juvenile else 0.80,
        )
        self.dopamine_level = float(np.clip(self.dopamine_level * 0.97 + target_dopamine * 0.03, 0.20, 0.86))
        target_serotonin = np.clip(
            serotonin_base
            + 0.08 * self.metabolism.glucose_level
            + 0.03 * observational_reward
            + 0.05 * self.affect.reward_confidence
            - 0.12 * self.metabolism.fatigue_level,
            0.24,
            0.84,
        )
        self.serotonin_level = float(np.clip(self.serotonin_level * 0.97 + target_serotonin * 0.03, 0.24, 0.84))
        target_acetylcholine = np.clip(
            0.30
            + neural_activity * 0.08
            + self.spatial_memory.replay_gate * 0.06
            + self.affect.arousal * 0.06
            + self.affect.curiosity * 0.04,
            0.10,
            0.68,
        )
        self.acetylcholine_level = float(np.clip(self.acetylcholine_level * 0.96 + target_acetylcholine * 0.04, 0.10, 0.68))
        if reward > 0:
            self.neurotransmitter_system.release((180.0, 180.0), reward, "dopamine")
        self.neurotransmitter_system.diffuse(self.dt)

    def update_maturation_state(
        self,
        perception,
        body_energy,
        reward,
        observational_reward,
        memory_score,
        *,
        stall_scale=1.0,
    ):
        exploration_competence = clamp((perception.get("visited_count", 1) - 1) / 10.0, 0.0, 1.0)
        foraging_competence = clamp(perception.get("food_collected", 0) / 3.0, 0.0, 1.0)
        energy_stability = clamp((body_energy - (AGENT_CRITICAL_THRESHOLD + 0.08)) / 0.55, 0.0, 1.0)
        memory_competence = clamp(
            max(0.0, memory_score) * 0.85
            + max(0.0, self.spatial_memory.food_expectation) * 0.10,
            0.0,
            1.0,
        )
        progress_competence = clamp(perception.get("recent_progress", 0.0), 0.0, 1.0)
        reward_signal = clamp(reward * 1.4 + observational_reward * 0.55 + self.affect.reward_confidence * 0.7, 0.0, 1.0)
        stall_penalty = clamp((perception.get("stall_time", 0.0) * stall_scale) / 4.0, 0.0, 1.0)
        fatigue_penalty = clamp(self.metabolism.fatigue_level, 0.0, 1.0)
        instability_penalty = clamp(
            self.affect.frustration * 0.58 + fatigue_penalty * 0.12,
            0.0,
            1.0,
        )
        positive_prediction = clamp(max(0.0, self.spatial_memory.food_prediction_error), 0.0, 1.0)
        hunting_mastery = clamp(
            foraging_competence * 0.46
            + reward_signal * 0.20
            + memory_competence * 0.16
            + clamp(max(0.0, self.affect.reward_confidence - 0.18) / 0.82, 0.0, 1.0) * 0.12
            + positive_prediction * 0.06,
            0.0,
            1.0,
        )

        readiness_target = clamp(
            0.02
            + hunting_mastery * 0.52
            + exploration_competence * 0.08
            + energy_stability * 0.16
            + progress_competence * 0.10
            - stall_penalty * 0.12
            - instability_penalty * 0.16,
            0.0,
            1.0,
        )
        stability_target = clamp(
            readiness_target * 0.58
            + hunting_mastery * 0.22
            + energy_stability * 0.20
            + progress_competence * 0.10
            - stall_penalty * 0.18
            - instability_penalty * 0.16,
            0.0,
            1.0,
        )

        self.maturity_readiness = clamp(self.maturity_readiness * 0.992 + readiness_target * 0.008, 0.0, 1.0)
        self.maturity_stability = clamp(self.maturity_stability * 0.994 + stability_target * 0.006, 0.0, 1.0)
        maturity_signal = clamp(self.maturity_readiness * 0.58 + self.maturity_stability * 0.42, 0.0, 1.0)

        if (
            self.is_juvenile
            and maturity_signal > 0.60
            and self.maturity_stability > 0.54
            and foraging_competence > 0.15
            and hunting_mastery > 0.34
            and energy_stability > 0.35
        ):
            self.is_juvenile = False
            self.self_selected_maturity_age = self.juvenile_age

        return maturity_signal

    def update(self, perception, body_energy, reward, time_s):
        self.steps += 1
        self.juvenile_age += 1

        motivation_context = self.build_motivation_context(perception, body_energy)
        visual_scene = self.build_visual_scene(perception)
        retinal_output = self.visual_system.process_visual_input(visual_scene)
        salience_vectors = self.build_salience_vectors(perception, body_energy, motivation_context=motivation_context)
        self.tectum.process(retinal_output, salience_vectors)
        tectum_vector = self.tectum.get_movement_command()

        memory_vector, memory_score = self.update_spatial_memory(
            perception,
            reward,
            body_energy,
            motivation_context=motivation_context,
        )
        policy_metrics = self.spatial_memory.last_policy_metrics
        action_vigor = float(policy_metrics.get("vigor", 0.35))
        search_burst = float(policy_metrics.get("search_burst", 0.18))
        policy_confidence = float(policy_metrics.get("confidence", 0.22))
        policy_conflict = float(policy_metrics.get("conflict", 0.35))
        habit_pressure = float(policy_metrics.get("habit_pressure", 0.0))
        loop_pressure = float(policy_metrics.get("loop_pressure", 0.0))
        reorientation_drive = float(policy_metrics.get("reorientation_drive", 0.0))
        reorientation_vector = tuple(self.spatial_memory.reorientation_vector.tolist()) if np.linalg.norm(self.spatial_memory.reorientation_vector) > 1e-6 else (0.0, 0.0)
        food_capture_vector, food_capture_strength = self.compute_food_capture_vector(
            perception,
            body_energy,
            motivation_context=motivation_context,
        )
        developmental_novelty_vector, developmental_novelty_strength = self.compute_developmental_novelty_vector(
            perception,
            body_energy,
            motivation_context=motivation_context,
        )
        food_directness = clamp(food_capture_strength, 0.0, 1.0)
        exploration_bonus = max(0.0, memory_score) * 0.1 + max(0.0, 1.0 - self.spatial_memory.external_drive) * 0.06
        open_drive = self.compute_open_drive(perception, memory_vector, motivation_context=motivation_context)
        memory_affordance = self.local_affordance(memory_vector, perception.get("open_vectors", []))
        gated_memory_vector = scale_vector(memory_vector, 0.06 + 0.94 * (memory_affordance ** 1.35))
        tectum_strength = clamp(float(np.linalg.norm(tectum_vector)), 0.0, 1.0)
        memory_strength = clamp(float(np.linalg.norm(gated_memory_vector)), 0.0, 1.0)
        open_strength = clamp(float(np.linalg.norm(open_drive)), 0.0, 1.0)
        decision = self.arbitrate_action_selection(
            tectum_vector=tectum_vector,
            food_vector=food_capture_vector,
            memory_vector=gated_memory_vector,
            novelty_vector=developmental_novelty_vector,
            open_vector=open_drive,
            reorientation_vector=reorientation_vector,
            body_energy=body_energy,
            tectum_strength=tectum_strength,
            food_directness=food_directness,
            memory_strength=memory_strength,
            memory_score=memory_score,
            novelty_strength=developmental_novelty_strength,
            open_strength=open_strength,
            policy_confidence=policy_confidence,
            policy_conflict=policy_conflict,
            habit_pressure=habit_pressure,
            loop_pressure=loop_pressure,
            reorientation_drive=reorientation_drive,
        )
        desired_vector = decision["desired_vector"]
        action_vigor = decision["action_vigor"]
        search_burst = decision["search_burst"]
        focus_closeness = clamp(1.0 - focused_target_distance / max(1.0, visual_span * 0.65), 0.0, 1.0)
        slow_hunt_drive = clamp(
            0.10
            + predation_bias * 0.34
            + reward_seek_bias * 0.14
            + decision["goal_commitment"] * 0.18
            + max(0.0, memory_score) * 0.08
            - decision["search_pressure"] * 0.10,
            0.0,
            1.0,
        )
        prey_permission_target = clamp(
            slow_hunt_drive * (0.78 + self.affect.reward_confidence * 0.18)
            - max(0.0, energy_ratio - 0.90) * 0.06
            - self.affect.frustration * 0.04,
            0.0,
            1.0,
        )
        self.slow_hunt_drive = self.slow_hunt_drive * 0.78 + slow_hunt_drive * 0.22
        self.prey_permission = self.prey_permission * 0.80 + prey_permission_target * 0.20
        target_lock_target = clamp(
            0.04
            + strongest_target_strength * 0.44
            + focus_closeness * 0.24
            + focused_target_motion * 0.12
            + max(0.0, decision["goal_commitment"] - decision["search_pressure"] * 0.40) * 0.16,
            0.0,
            1.0,
        )
        self.fast_target_lock = self.fast_target_lock * 0.68 + target_lock_target * 0.32
        fast_orient_target = clamp(
            0.02
            + self.prey_permission * 0.22
            + self.fast_target_lock * 0.42
            + focus_closeness * 0.18
            + focused_target_motion * 0.10
            - decision["search_pressure"] * 0.08,
            0.0,
            1.0,
        )
        self.fast_orient_gain = self.fast_orient_gain * 0.72 + fast_orient_target * 0.28

        proprio_feedback = np.zeros(12)
        self.motor_hierarchy.execute_movement_command(
            desired_vector,
            proprio_feedback,
            tonic_drive=decision["locomotor_tone"],
            energy_level=clamp(body_energy / max(1e-6, AGENT_BODY_ENERGY_MAX), 0.0, 1.0),
        )
        velocity = scale_vector(
            self.motor_hierarchy.get_velocity_vector(desired_vector),
            (0.68 + decision["command_strength"] * 0.26 + decision["bg_confidence"] * 0.14)
            * (0.88 + decision["conflict_damping"] * 0.12),
        )
        velocity = scale_vector(velocity, decision["bg_gating_signal"])
        neural_activity = float(np.mean(np.abs(retinal_output))) if len(retinal_output) > 0 else 0.0
        neural_activity += float(np.mean(self.tectum.tectal_output)) * 0.35
        neural_activity += float(np.mean(self.motor_hierarchy.muscle_activation)) * 0.18
        neural_activity += float(np.mean(self.spatial_memory.last_activations)) * 0.45

        movement_intensity = float(np.linalg.norm(velocity))
        self.metabolism.update(self.dt, movement_intensity, neural_activity)
        self.neuron_metabolism.consume_energy(False, neural_activity * 10.0, self.dt)
        self.neuron_metabolism.recover_energy(self.dt, self.metabolism.oxygen_level, self.metabolism.glucose_level)
        excitability = self.neuron_metabolism.affects_excitability()

        neural_map = np.array([neuron.activity_level() for neuron in self._neuron_cache], dtype=float)
        self.update_affective_state(
            perception,
            body_energy,
            reward,
            0.0,
            memory_score,
            exploration_bonus,
            neural_activity,
        )
        maturity_signal = self.update_maturation_state(perception, body_energy, reward, 0.0, memory_score)
        self.update_neuromodulation(
            reward,
            0.0,
            exploration_bonus,
            neural_activity,
        )
        if self.steps % 4 == 0:
            self.functional_plasticity.apply_homeostatic_scaling(self._neuron_cache, self.dt * 4.0)
            self.functional_plasticity.apply_intrinsic_plasticity(self._neuron_cache, self.dt * 4.0)
        if self.steps % 6 == 0:
            self.structural_plasticity.update_structure(self._synapse_cache, neural_map, self.dt * 6.0)
        for synapse in self._synapse_cache:
            synapse.update_modulators(self.dopamine_level, self.serotonin_level, self.acetylcholine_level)
            synapse.apply_short_term_plasticity(self.dt, reward > 0)

        self.last_reward = reward
        # Minimal velocity smoothing: 50% old + 50% new
        smoothed_velocity_x = velocity[0] * 0.50 + self.last_velocity[0] * 0.50
        smoothed_velocity_y = velocity[1] * 0.50 + self.last_velocity[1] * 0.50
        self.last_velocity = (smoothed_velocity_x, smoothed_velocity_y)
        energy_surplus = clamp((body_energy - 0.72) / 0.20, 0.0, 1.0)
        desperation = clamp((0.60 - body_energy) / 0.30, 0.0, 1.0)

        return {
            "velocity": self.last_velocity,
            "memory_vector": gated_memory_vector,
            "memory_score": memory_score,
            "dopamine": self.dopamine_level,
            "serotonin": self.serotonin_level,
            "acetylcholine": self.acetylcholine_level,
            "fatigue": self.metabolism.fatigue_level,
            "glucose": self.metabolism.glucose_level,
            "neural_activity": neural_activity,
            "replay_gate": self.spatial_memory.replay_gate,
            "arousal": float(self.affect.arousal),
            "curiosity": float(self.affect.curiosity),
            "frustration": float(self.affect.frustration),
            "reward_confidence": float(self.affect.reward_confidence),
            "restlessness": float(self.affect.restlessness),
            "motivation_context": self.last_motivation_context.copy(),
            "food_prediction_error": float(self.spatial_memory.food_prediction_error),
            "energy_surplus": float(energy_surplus),
            "desperation": float(desperation),
            "action_vigor": action_vigor,
            "search_burst": search_burst,
            "policy_confidence": policy_confidence,
            "policy_conflict": policy_conflict,
            "habit_pressure": habit_pressure,
            "loop_pressure": loop_pressure,
            "reorientation_drive": reorientation_drive,
            "food_directness": food_directness,
            "developmental_novelty": float(developmental_novelty_strength),
            "goal_commitment": decision["goal_commitment"],
            "search_pressure": decision["search_pressure"],
            "command_strength": decision["command_strength"],
            "bg_gating_signal": decision["bg_gating_signal"],
            "bg_confidence": decision["bg_confidence"],
            "observational_reward": 0.0,
            "is_juvenile": self.is_juvenile,
            "juvenile_progress": 1.0 if not self.is_juvenile else float(maturity_signal),
            "maturity_readiness": float(self.maturity_readiness),
            "maturity_stability": float(self.maturity_stability),
            "self_selected_maturity_age": self.self_selected_maturity_age,
            "reward": reward,
        }

    def update_predator_mode(
        self,
        position,
        heading_angle,
        visible_targets,
        body_energy,
        reward,
        visual_range_px,
        *,
        food_collected=0,
        stall_time=0.0,
        time_since_reward=999.0,
        visited_count=1,
    ):
        """Open-field predator mode used by the fly-catching simulation.

        The maze-oriented `update()` path expects tiled affordances and wall maps.
        For the fly-catching environment we keep the same biological subsystems,
        but feed them a continuous open-field perception built from visible flies.
        """
        self.steps += 1
        self.juvenile_age += 1

        energy_ratio = clamp(body_energy / max(1e-6, AGENT_BODY_ENERGY_MAX), 0.0, 1.0)
        synthetic_perception = {
            "visible_food_tiles": {("fly", idx) for idx, _ in enumerate(visible_targets)},
            "recent_progress": clamp(float(np.linalg.norm(np.array(self.last_velocity, dtype=float))), 0.0, 1.0),
            "food_collected": int(food_collected),
            "stall_time": float(stall_time),
            "time_since_reward": float(time_since_reward),
            "visited_count": int(visited_count),
        }
        motivation_context = self.build_motivation_context(synthetic_perception, body_energy)
        visual_scene = []
        salience_vectors = []
        visible_food_sources = []
        strongest_target_unit = np.zeros(2, dtype=float)
        strongest_target_strength = 0.0
        weighted_target_sum = np.zeros(2, dtype=float)
        focused_target_distance = visual_span = max(1.0, float(visual_range_px))
        focused_target_position = None
        focused_target_motion = 0.0

        current_position = np.array(position, dtype=float)
        predation_bias = float(motivation_context.get("predation_bias", 0.35))
        reward_seek_bias = float(motivation_context.get("reward_seek_bias", 0.20))
        vigilance_bias = float(motivation_context.get("vigilance_bias", 0.20))
        exploration_bias = float(motivation_context.get("exploration_bias", 0.30))
        locomotor_bias = float(motivation_context.get("locomotor_bias", 0.20))
        for target in visible_targets:
            target_vector = np.array(target["vector"], dtype=float)
            distance = float(np.linalg.norm(target_vector))
            if distance <= 1e-6:
                continue

            brightness = clamp(float(target.get("brightness", 0.0)), 0.0, 1.0)
            motion_signal = clamp(float(target.get("motion", brightness)), 0.0, 1.0)
            facing_bias = clamp((float(target.get("facing", 0.0)) + 1.0) * 0.5, 0.0, 1.0)
            target_unit = target_vector / distance
            field_x = 2.5 + clamp(target_vector[0] / visual_span * 2.2, -2.2, 2.2)
            field_y = 2.5 + clamp(target_vector[1] / visual_span * 2.2, -2.2, 2.2)
            visual_scene.append((field_x, field_y, brightness))

            salience_gain = 0.08 + brightness * (1.00 + predation_bias * 0.20) + motion_signal * (0.22 + vigilance_bias * 0.16)
            salience_vectors.append(target_unit * salience_gain)
            visible_food_sources.append(
                (
                    tuple((current_position + target_vector).tolist()),
                    clamp(0.12 + brightness * (0.70 + predation_bias * 0.18) + reward_seek_bias * 0.08, 0.08, 1.0),
                )
            )

            strength = clamp(
                brightness
                * (0.42 + clamp(1.0 - distance / visual_span, 0.0, 1.0) * 0.76)
                * (0.58 + facing_bias * 0.42)
                * (0.70 + motion_signal * 0.30),
                0.0,
                1.4,
            )
            strength = clamp(strength * (0.74 + reward_seek_bias * 0.18 + predation_bias * 0.12), 0.0, 1.4)
            if strength > strongest_target_strength:
                strongest_target_strength = strength
                strongest_target_unit = target_unit.copy()
                focused_target_distance = distance
                focused_target_motion = motion_signal
                focused_target_position = tuple(target.get("position", (current_position + target_vector).tolist()))
            weighted_target_sum += target_unit * strength

        if not salience_vectors:
            salience_vectors = [np.zeros(2, dtype=float)]

        retinal_output = self.visual_system.process_visual_input(visual_scene)
        self.tectum.process(retinal_output, salience_vectors)
        tectum_vector = self.tectum.get_movement_command()

        open_vectors = [
            (math.cos(angle), math.sin(angle), 1.0)
            for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        ]
        hunger_drive = clamp(
            motivation_context.get("hunger_bias", 1.0 - energy_ratio) * self.profile.memory_gain
            + self.affect.frustration * 0.06
            + (1.0 - self.affect.reward_confidence) * 0.08,
            0.05,
            1.6,
        )
        self.spatial_memory.observe(
            position=position,
            heading_angle=heading_angle,
            reward=reward,
            visible_food_sources=visible_food_sources,
            visible_peer_sources=[],
            heard_signals=[],
            fatigue=self.metabolism.fatigue_level,
            social_need=0.0,
        )
        memory_vector, memory_score = self.spatial_memory.memory_vector(
            position=position,
            heading_angle=heading_angle,
            open_vectors=open_vectors,
            hunger_drive=hunger_drive,
            social_need=0.0,
            distress_bias=0.0,
            curiosity_drive=self.affect.curiosity,
            frustration=self.affect.frustration,
            social_comfort=0.0,
            reward_confidence=self.affect.reward_confidence,
            restlessness=self.affect.restlessness,
            stall_pressure=0.0,
        )
        self.last_memory_vector = memory_vector
        self.last_memory_score = memory_score

        food_focus = weighted_target_sum if np.linalg.norm(weighted_target_sum) > 1e-6 else strongest_target_unit
        if np.linalg.norm(food_focus) > 1e-6:
            food_focus = food_focus / np.linalg.norm(food_focus)
        lateral_scan_gain = max(0.0, 0.20 - strongest_target_strength) * (0.16 + self.affect.restlessness * 0.18)
        self.scan_phase = (self.scan_phase + 0.12 + self.affect.curiosity * 0.05 + lateral_scan_gain * 0.10) % (2.0 * math.pi)
        scan_vector = (
            math.cos(heading_angle + self.scan_phase),
            math.sin(heading_angle + self.scan_phase),
        )
        if np.linalg.norm(food_focus) > 1e-6:
            lateral_food = np.array([-food_focus[1], food_focus[0]], dtype=float)
            exploratory_food = food_focus * (0.92 + strongest_target_strength * 0.06) + lateral_food * lateral_scan_gain
            exploratory_food = exploratory_food / max(1e-6, float(np.linalg.norm(exploratory_food)))
            food_capture_vector = (float(exploratory_food[0]), float(exploratory_food[1]))
        else:
            food_capture_vector = (0.0, 0.0)
        food_directness = clamp(strongest_target_strength, 0.0, 1.0)
        exploration_bonus = max(0.0, memory_score) * 0.08 + max(0.0, 1.0 - self.spatial_memory.external_drive) * 0.04
        scan_strength = clamp(
            max(0.0, 0.30 - food_directness) * (0.28 + exploration_bias * 0.34 + self.affect.restlessness * 0.18),
            0.0,
            1.0,
        )
        tectum_strength = clamp(float(np.linalg.norm(tectum_vector)), 0.0, 1.0)
        memory_strength = clamp(float(np.linalg.norm(memory_vector)), 0.0, 1.0)
        open_strength = clamp(scan_strength * 0.75, 0.0, 1.0)
        policy_confidence = clamp(
            0.24
            + strongest_target_strength * 0.34
            + max(0.0, memory_score) * 0.16
            + max(0.0, self.affect.reward_confidence - 0.2) * 0.12
            + focused_target_motion * 0.08,
            0.0,
            1.0,
        )
        policy_conflict = clamp(
            0.08
            + max(0.0, 0.34 - strongest_target_strength) * 0.26
            + max(0.0, 0.10 - memory_score) * 0.10
            + scan_strength * 0.14,
            0.0,
            0.85,
        )
        habit_pressure = clamp(
            max(0.0, 0.16 - self.spatial_memory.external_drive) * 0.20 + max(0.0, -memory_score) * 0.05,
            0.0,
            0.65,
        )
        loop_pressure = clamp(
            max(0.0, 0.18 - strongest_target_strength) * 0.22
            + max(0.0, 0.10 - memory_score) * 0.12
            + scan_strength * 0.12,
            0.0,
            0.70,
        )
        reorientation_drive = clamp(
            loop_pressure * (0.34 + exploration_bias * 0.18 + self.affect.restlessness * 0.12)
            - food_directness * 0.08,
            0.0,
            1.0,
        )
        reorientation_vector = scan_vector if reorientation_drive > 0.02 else (0.0, 0.0)
        decision = self.arbitrate_action_selection(
            tectum_vector=tectum_vector,
            food_vector=food_capture_vector,
            memory_vector=memory_vector,
            novelty_vector=scan_vector,
            open_vector=scan_vector,
            reorientation_vector=reorientation_vector,
            body_energy=energy_ratio,
            tectum_strength=tectum_strength,
            food_directness=food_directness,
            memory_strength=memory_strength,
            memory_score=memory_score,
            novelty_strength=scan_strength,
            open_strength=open_strength,
            policy_confidence=policy_confidence,
            policy_conflict=policy_conflict,
            habit_pressure=habit_pressure,
            loop_pressure=loop_pressure,
            reorientation_drive=reorientation_drive,
        )
        desired_vector = decision["desired_vector"]
        action_vigor = decision["action_vigor"]
        search_burst = decision["search_burst"]
        focus_closeness = clamp(1.0 - focused_target_distance / max(1.0, visual_span * 0.65), 0.0, 1.0)
        slow_hunt_drive = clamp(
            0.10
            + predation_bias * 0.34
            + reward_seek_bias * 0.14
            + decision["goal_commitment"] * 0.18
            + max(0.0, memory_score) * 0.08
            - decision["search_pressure"] * 0.10,
            0.0,
            1.0,
        )
        prey_permission_target = clamp(
            slow_hunt_drive * (0.78 + self.affect.reward_confidence * 0.18)
            - max(0.0, energy_ratio - 0.90) * 0.06
            - self.affect.frustration * 0.04,
            0.0,
            1.0,
        )
        self.slow_hunt_drive = self.slow_hunt_drive * 0.78 + slow_hunt_drive * 0.22
        self.prey_permission = self.prey_permission * 0.80 + prey_permission_target * 0.20
        target_lock_target = clamp(
            0.04
            + strongest_target_strength * 0.44
            + focus_closeness * 0.24
            + focused_target_motion * 0.12
            + max(0.0, decision["goal_commitment"] - decision["search_pressure"] * 0.40) * 0.16,
            0.0,
            1.0,
        )
        self.fast_target_lock = self.fast_target_lock * 0.68 + target_lock_target * 0.32
        fast_orient_target = clamp(
            0.02
            + self.prey_permission * 0.22
            + self.fast_target_lock * 0.42
            + focus_closeness * 0.18
            + focused_target_motion * 0.10
            - decision["search_pressure"] * 0.08,
            0.0,
            1.0,
        )
        self.fast_orient_gain = self.fast_orient_gain * 0.72 + fast_orient_target * 0.28

        proprio_feedback = np.zeros(12)
        self.motor_hierarchy.execute_movement_command(
            desired_vector,
            proprio_feedback,
            tonic_drive=decision["locomotor_tone"],
            energy_level=energy_ratio,
        )
        velocity = scale_vector(
            self.motor_hierarchy.get_velocity_vector(desired_vector),
            (0.68 + decision["command_strength"] * 0.26 + decision["bg_confidence"] * 0.14)
            * (0.88 + decision["conflict_damping"] * 0.12),
        )
        if np.linalg.norm(strongest_target_unit) > 1e-6 and self.fast_orient_gain > 0.02:
            near_field_pull = scale_vector(
                tuple(strongest_target_unit.tolist()),
                self.fast_orient_gain * (0.08 + focus_closeness * 0.24 + self.prey_permission * 0.10),
            )
            velocity = normalize_vector(add_vectors(velocity, near_field_pull))
        predator_motor_floor = clamp(
            0.14
            + decision["locomotor_tone"] * 0.32
            + decision["command_strength"] * 0.16
            + strongest_target_strength * 0.18
            + decision["goal_commitment"] * 0.12
            - decision["search_pressure"] * 0.08,
            0.16,
            0.86,
        )
        fast_loop_gate_target = clamp(
            0.10
            + self.prey_permission * 0.24
            + self.fast_target_lock * 0.30
            + focus_closeness * 0.18
            + focused_target_motion * 0.08
            - decision["search_pressure"] * 0.10,
            0.0,
            1.0,
        )
        self.fast_loop_gate = self.fast_loop_gate * 0.74 + fast_loop_gate_target * 0.26
        effective_motor_gate = max(
            decision["bg_gating_signal"],
            predator_motor_floor * (0.58 + decision["bg_confidence"] * 0.42),
            self.fast_loop_gate * (0.54 + self.prey_permission * 0.22),
        )
        velocity = scale_vector(velocity, effective_motor_gate)
        juvenile_predator_reflex = 0.0
        if self.is_juvenile:
            juvenile_predator_reflex = clamp(
                0.03
                + strongest_target_strength * 0.08
                + focus_closeness * 0.10
                + focused_target_motion * 0.06,
                0.0,
                0.18,
            )
        fast_strike_target = clamp(
            0.02
            + self.prey_permission * 0.12
            + self.fast_target_lock * 0.36
            + focus_closeness * 0.24
            + focused_target_motion * 0.08
            + decision["goal_commitment"] * 0.10
            - self.affect.restlessness * 0.03,
            0.0,
            1.0,
        )
        self.fast_strike_drive = self.fast_strike_drive * 0.70 + fast_strike_target * 0.30
        motor_gate_scale = (
            0.58 + effective_motor_gate * 0.42
            if self.is_juvenile
            else 0.48 + effective_motor_gate * 0.52
        )
        slow_strike_component = (
            0.04
            + strongest_target_strength * 0.30
            + focus_closeness * 0.18
            + decision["goal_commitment"] * 0.16
            + action_vigor * 0.08
            + self.affect.reward_confidence * 0.05
            + focused_target_motion * 0.07
            + juvenile_predator_reflex
            - self.affect.restlessness * 0.03
        )
        strike_readiness = clamp(
            (slow_strike_component * 0.58 + self.fast_strike_drive * 0.42)
            * (0.34 + max(effective_motor_gate, self.fast_loop_gate) * 0.66)
            * motor_gate_scale,
            0.0,
            1.0,
        )

        neural_activity = float(np.mean(np.abs(retinal_output))) if len(retinal_output) > 0 else 0.0
        neural_activity += float(np.mean(self.tectum.tectal_output)) * 0.35
        neural_activity += float(np.mean(self.motor_hierarchy.muscle_activation)) * 0.18
        neural_activity += float(np.mean(self.spatial_memory.last_activations)) * 0.38

        movement_intensity = float(np.linalg.norm(velocity))
        self.metabolism.update(self.dt, movement_intensity, neural_activity)
        self.neuron_metabolism.consume_energy(False, neural_activity * 10.0, self.dt)
        self.neuron_metabolism.recover_energy(self.dt, self.metabolism.oxygen_level, self.metabolism.glucose_level)
        excitability = self.neuron_metabolism.affects_excitability()
        metabolic_drive = 0.35 + 0.65 * energy_ratio
        velocity = scale_vector(velocity, excitability * metabolic_drive)

        predator_progress = clamp(
            max(
                movement_intensity,
                strongest_target_strength * 0.62,
                clamp(float(food_collected) / 3.0, 0.0, 1.0) * 0.45,
            ),
            0.0,
            1.0,
        )
        predator_stall = clamp(float(stall_time) * 0.18, 0.0, 6.0)
        synthetic_perception = {
            "visible_food_tiles": {("fly", idx) for idx, _ in enumerate(visible_targets)},
            "recent_progress": predator_progress,
            "food_collected": int(food_collected),
            "stall_time": predator_stall,
            "time_since_reward": float(time_since_reward),
            "visited_count": int(visited_count + max(0, int(food_collected)) * 2),
        }
        self.update_affective_state(
            synthetic_perception,
            energy_ratio,
            reward,
            0.0,
            memory_score,
            exploration_bonus,
            neural_activity,
        )
        maturity_signal = self.update_maturation_state(
            synthetic_perception,
            body_energy,
            reward,
            0.0,
            memory_score,
            stall_scale=0.45,
        )
        self.update_neuromodulation(
            reward,
            0.0,
            exploration_bonus,
            neural_activity,
        )
        if self.steps % 4 == 0:
            self.functional_plasticity.apply_homeostatic_scaling(self._neuron_cache, self.dt * 4.0)
            self.functional_plasticity.apply_intrinsic_plasticity(self._neuron_cache, self.dt * 4.0)
        if self.steps % 6 == 0:
            neural_map = np.array([neuron.activity_level() for neuron in self._neuron_cache], dtype=float)
            self.structural_plasticity.update_structure(self._synapse_cache, neural_map, self.dt * 6.0)
        for synapse in self._synapse_cache:
            synapse.update_modulators(self.dopamine_level, self.serotonin_level, self.acetylcholine_level)
            synapse.apply_short_term_plasticity(self.dt, reward > 0)

        smoothed_velocity_x = velocity[0] * 0.50 + self.last_velocity[0] * 0.50
        smoothed_velocity_y = velocity[1] * 0.50 + self.last_velocity[1] * 0.50
        self.last_velocity = (smoothed_velocity_x, smoothed_velocity_y)
        self.last_food_vector = food_capture_vector
        return {
            "velocity": self.last_velocity,
            "memory_vector": memory_vector,
            "memory_score": memory_score,
            "dopamine": self.dopamine_level,
            "serotonin": self.serotonin_level,
            "acetylcholine": self.acetylcholine_level,
            "fatigue": self.metabolism.fatigue_level,
            "glucose": self.metabolism.glucose_level,
            "neural_activity": neural_activity,
            "excitability": float(excitability),
            "replay_gate": self.spatial_memory.replay_gate,
            "arousal": float(self.affect.arousal),
            "curiosity": float(self.affect.curiosity),
            "frustration": float(self.affect.frustration),
            "reward_confidence": float(self.affect.reward_confidence),
            "restlessness": float(self.affect.restlessness),
            "motivation_context": self.last_motivation_context.copy(),
            "food_prediction_error": float(self.spatial_memory.food_prediction_error),
            "action_vigor": float(action_vigor),
            "search_burst": float(search_burst),
            "policy_confidence": float(policy_confidence),
            "policy_conflict": float(policy_conflict),
            "habit_pressure": float(habit_pressure),
            "loop_pressure": float(loop_pressure),
            "slow_hunt_drive": float(self.slow_hunt_drive),
            "prey_permission": float(self.prey_permission),
            "fast_target_lock": float(self.fast_target_lock),
            "fast_orient_gain": float(self.fast_orient_gain),
            "fast_loop_gate": float(self.fast_loop_gate),
            "fast_strike_drive": float(self.fast_strike_drive),
            "reorientation_drive": float(reorientation_drive),
            "food_directness": float(food_directness),
            "goal_commitment": decision["goal_commitment"],
            "search_pressure": decision["search_pressure"],
            "command_strength": decision["command_strength"],
            "bg_gating_signal": decision["bg_gating_signal"],
            "bg_confidence": decision["bg_confidence"],
            "effective_motor_gate": float(effective_motor_gate),
            "locomotor_tone": float(decision["locomotor_tone"]),
            "focus_vector": tuple(strongest_target_unit.tolist()) if np.linalg.norm(strongest_target_unit) > 1e-6 else (0.0, 0.0),
            "focus_distance": float(focused_target_distance if focused_target_position is not None else 0.0),
            "focus_position": focused_target_position,
            "strike_readiness": float(strike_readiness),
            "developmental_novelty": float(scan_strength),
            "is_juvenile": self.is_juvenile,
            "juvenile_progress": 1.0 if not self.is_juvenile else float(maturity_signal),
            "maturity_readiness": float(self.maturity_readiness),
            "maturity_stability": float(self.maturity_stability),
            "self_selected_maturity_age": self.self_selected_maturity_age,
            "reward": float(reward),
        }

    def state_vector(self):
        return {
            "is_juvenile": self.is_juvenile,
            "juvenile_age": self.juvenile_age,
            "dopamine": self.dopamine_level,
            "serotonin": self.serotonin_level,
            "acetylcholine": self.acetylcholine_level,
            "glucose": self.metabolism.glucose_level,
            "fatigue": self.metabolism.fatigue_level,
            "arousal": float(self.affect.arousal),
            "curiosity": float(self.affect.curiosity),
            "frustration": float(self.affect.frustration),
            "reward_confidence": float(self.affect.reward_confidence),
            "restlessness": float(self.affect.restlessness),
            "maturity_readiness": float(self.maturity_readiness),
            "maturity_stability": float(self.maturity_stability),
            "self_selected_maturity_age": self.self_selected_maturity_age,
        }

