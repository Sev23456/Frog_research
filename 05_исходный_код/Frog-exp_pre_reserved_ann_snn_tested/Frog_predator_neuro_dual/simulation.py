"""
Biorealistic Frog Predator Neuro - Fly Catching Simulation
Adapted from frog_lib_ann with BioFrogBrain neural architecture
"""

import json
import random
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pymunk

from Frog_predator_neuro_dual.brain import BioFrogBrain
from Frog_predator_neuro_dual.config import (
    AGENT_BODY_ENERGY_MAX,
    AGENT_BODY_ENERGY_START,
    AGENT_MAX_SPEED,
    AGENT_IDLE_DRAIN,
    AGENT_MOVE_DRAIN,
    CYAN,
    GRAY,
    HUD_BG,
    HUD_PANEL,
    PREDATOR_DETECTION_THRESHOLD,
    PREDATOR_VISUAL_RANGE_PX,
    TILE_SIZE,
    HUD_WIDTH,
    WHITE,
    YELLOW,
)
from Frog_predator_neuro_dual.utils import clamp

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

FROG_RENDER_SCALE = 2.7
FROG_RENDER_RADIUS_MIN = 24
FROG_INNER_SCALE = 0.84
FROG_OUTER_COLOR = (20, 135, 40)
FROG_INNER_COLOR = (40, 185, 60)
FROG_TONGUE_COLOR = (240, 90, 90)


def to_pymunk_vec(value) -> Tuple[float, float]:
    if isinstance(value, np.ndarray):
        return float(value[0]), float(value[1])
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return 0.0, 0.0


def to_pygame_vec(value) -> Tuple[int, int]:
    if isinstance(value, np.ndarray):
        return int(value[0]), int(value[1])
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return 0, 0


def add_world_boundaries(space: pymunk.Space, width: int, height: int, margin: float = 8.0):
    body = space.static_body
    segments = [
        pymunk.Segment(body, (margin, margin), (width - margin, margin), margin),
        pymunk.Segment(body, (margin, height - margin), (width - margin, height - margin), margin),
        pymunk.Segment(body, (margin, margin), (margin, height - margin), margin),
        pymunk.Segment(body, (width - margin, margin), (width - margin, height - margin), margin),
    ]
    for segment in segments:
        segment.elasticity = 0.96
        segment.friction = 0.9
    space.add(*segments)
    return segments


class Fly:
    """Simple fly physics object"""
    def __init__(self, space: pymunk.Space, position: Tuple[float, float]):
        self.space = space
        self.alive = True
        moment = pymunk.moment_for_circle(0.1, 0, 5.0)
        self.body = pymunk.Body(0.1, moment)
        self.body.position = to_pymunk_vec(position)
        self.body.velocity = (random.uniform(-45, 45), random.uniform(-45, 45))
        self.shape = pymunk.Circle(self.body, 5.0)
        self.shape.elasticity = 0.9
        self.shape.friction = 0.5
        self.space.add(self.body, self.shape)

    def update(self, dt: float, width: int, height: int):
        if not self.alive:
            return
        pos = self.body.position
        vel = self.body.velocity
        if pos.x < 10 or pos.x > width - 10:
            vel = (-vel[0], vel[1])
        if pos.y < 10 or pos.y > height - 10:
            vel = (vel[0], -vel[1])
        if random.random() < 0.08:
            vel = (random.uniform(-55, 55), random.uniform(-55, 55))
        self.body.velocity = vel

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        pos = to_pygame_vec(self.body.position)
        pygame.draw.circle(surface, (150, 150, 255), pos, 9)
        pygame.draw.circle(surface, (70, 70, 170), pos, 5)

    def remove(self):
        try:
            if self.shape in self.space.shapes and self.body in self.space.bodies:
                self.space.remove(self.body, self.shape)
        except Exception:
            pass
        self.alive = False


class BioFrogPredatorAgent:
    """Biorealistic frog with BioFrogBrain for fly catching"""
    
    def __init__(
        self,
        agent_id: int = 1,
        space: pymunk.Space = None,
        position: Tuple[float, float] = (400, 300),
        training_mode: bool = False,
        brain_rng_seed: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.space = space or pymunk.Space()
        self.training_mode = training_mode
        
        # Physics body
        moment = pymunk.moment_for_circle(1.0, 0, 9.0)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = to_pymunk_vec(position)
        self.body.velocity = (0, 0)
        self.shape = pymunk.Circle(self.body, 9.0)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7
        self.space.add(self.body, self.shape)
        
        # Brain
        brain_rng = random.Random(brain_rng_seed) if brain_rng_seed is not None else None
        self.brain = BioFrogBrain(juvenile_mode=training_mode, rng=brain_rng)
        
        # Energy and state
        self.energy = AGENT_BODY_ENERGY_START
        self.caught_flies = 0
        self.alive = True
        
        # Developmental capture profile: juvenile starts conservative, adult matches the normalized variants.
        self.juvenile_hit_radius = 24.0
        self.adult_hit_radius = 34.0
        self.juvenile_success_prob = 0.75
        self.adult_success_prob = 0.92
        self.juvenile_tongue_reach = 52.0
        self.adult_tongue_reach = 60.0
        
        # History
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.last_velocity = np.array([0.0, 0.0], dtype=float)
        self.last_step_distance = 0.0
        self.tongue_extended = False
        self.tongue_target = None
        self.color = (80, 200, 100)
        self.strike_drive = 0.0
        self.strike_commitment = 0
        self.strike_cooldown = 0.0
        self.pending_reward_signal = 0.0
        self.visual_range_px = float(PREDATOR_VISUAL_RANGE_PX)
        self._visible_targets: List[Dict[str, Any]] = []
        self.elapsed_time_s = 0.0
        self.last_reward_time_s = 0.0
        self.stall_time = 0.0
        self.visited_zones = {self._visit_key(position)}
        self.last_brain_output: Dict[str, Any] = {
            "dopamine": 0.0,
            "serotonin": 0.0,
            "acetylcholine": 0.0,
            "glucose": 1.0,
            "fatigue": 0.0,
            "excitability": 1.0,
            "neural_activity": 0.0,
            "replay_gate": 0.0,
            "arousal": 0.0,
            "curiosity": 0.0,
            "frustration": 0.0,
            "reward_confidence": 0.0,
            "restlessness": 0.0,
            "food_prediction_error": 0.0,
            "memory_score": 0.0,
            "food_directness": 0.0,
            "policy_confidence": 0.0,
            "policy_conflict": 0.0,
            "bg_gating_signal": 0.0,
            "bg_confidence": 0.0,
            "effective_motor_gate": 0.0,
            "goal_commitment": 0.0,
            "search_pressure": 0.0,
            "command_strength": 0.0,
            "strike_readiness": 0.0,
            "action_vigor": 0.0,
            "search_burst": 0.0,
            "focus_distance": 0.0,
            "motivation_context": {},
            "developmental_novelty": 0.0,
            "is_juvenile": self.brain.is_juvenile,
            "juvenile_progress": 0.0,
            "maturity_readiness": 0.0,
            "maturity_stability": 0.0,
            "self_selected_maturity_age": None,
        }
        self._apply_developmental_stage()

    @property
    def body_energy(self):
        return float(self.energy)

    @property
    def food_collected(self):
        return int(self.caught_flies)

    @property
    def x(self):
        return float(self.body.position[0])

    @property
    def y(self):
        return float(self.body.position[1])

    def _get_fly_observations(self, flies: List[Fly], width: int, height: int) -> Tuple[np.ndarray, Optional[Dict[str, Any]], float]:
        """Compute biomorphic prey observations from the local visual field."""
        if not flies:
            self._visible_targets = []
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=float), None, 0.0

        frog_pos = np.array(self.body.position, dtype=float)
        alive_flies = [f for f in flies if f.alive]
        if not alive_flies:
            self._visible_targets = []
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=float), None, 0.0

        heading = np.array(self.last_velocity, dtype=float)
        if np.linalg.norm(heading) <= 1e-6:
            heading = np.array([1.0, 0.0], dtype=float)
        heading = heading / max(1e-6, float(np.linalg.norm(heading)))

        visible_targets = []
        for fly in alive_flies:
            target_pos = np.array(fly.body.position, dtype=float)
            target_vector = target_pos - frog_pos
            target_distance = float(np.linalg.norm(target_vector))
            if target_distance <= 1e-6 or target_distance > self.visual_range_px:
                continue

            target_unit = target_vector / target_distance
            facing = float(np.dot(heading, target_unit))
            peripheral_gain = 0.40 + 0.60 * clamp((facing + 1.0) * 0.5, 0.0, 1.0)
            relative_motion = np.array(fly.body.velocity, dtype=float) - np.array(self.body.velocity, dtype=float)
            motion_signal = clamp(float(np.linalg.norm(relative_motion)) / 95.0, 0.0, 1.0)
            distance_gain = clamp(1.0 - target_distance / max(1.0, self.visual_range_px), 0.0, 1.0)
            brightness = clamp(distance_gain * (0.35 + motion_signal * 0.65) * peripheral_gain, 0.0, 1.0)
            if brightness < PREDATOR_DETECTION_THRESHOLD:
                continue

            visible_targets.append(
                {
                    "fly": fly,
                    "vector": tuple(target_vector.tolist()),
                    "distance": target_distance,
                    "brightness": brightness,
                    "motion": motion_signal,
                    "facing": facing,
                    "position": tuple(target_pos.tolist()),
                }
            )

        self._visible_targets = visible_targets
        if not visible_targets:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=float), None, 0.0

        focus_target = max(
            visible_targets,
            key=lambda item: (item["brightness"] + item["motion"] * 0.20, -item["distance"]),
        )
        rel_pos_norm = np.array(focus_target["vector"], dtype=float) / max(1e-6, focus_target["distance"])
        alignment = float(focus_target["facing"])

        obs = np.array([
            float(rel_pos_norm[0]),
            float(rel_pos_norm[1]),
            min(1.0, focus_target["distance"] / max(1.0, self.visual_range_px)),
            max(-1.0, min(1.0, alignment))
        ], dtype=float)

        return obs, focus_target, float(focus_target["distance"])

    def _visit_key(self, position: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(float(position[0]) // max(1.0, TILE_SIZE * 1.5)),
            int(float(position[1]) // max(1.0, TILE_SIZE * 1.5)),
        )

    def _apply_developmental_stage(self):
        if self.brain.is_juvenile:
            self.hit_radius = self.juvenile_hit_radius
            self.success_prob = self.juvenile_success_prob
            self.tongue_reach = self.juvenile_tongue_reach
            self.color = (96, 224, 118)
        else:
            self.hit_radius = self.adult_hit_radius
            self.success_prob = self.adult_success_prob
            self.tongue_reach = self.adult_tongue_reach
            self.color = (80, 200, 100)

    def _select_strike_target(self, brain_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not getattr(self, "_visible_targets", None):
            return None

        focus_vector = np.array(brain_output.get("focus_vector", (0.0, 0.0)), dtype=float)
        focus_norm = float(np.linalg.norm(focus_vector))
        if focus_norm <= 1e-6:
            return max(
                self._visible_targets,
                key=lambda item: (item["brightness"] + item["motion"] * 0.20, -item["distance"]),
            )

        focus_unit = focus_vector / focus_norm
        return max(
            self._visible_targets,
            key=lambda item: (
                float(np.dot(np.array(item["vector"], dtype=float) / max(1e-6, item["distance"]), focus_unit)) * 0.75
                + item["brightness"] * 0.45
                + item["motion"] * 0.15
                - item["distance"] / max(1.0, self.visual_range_px) * 0.20
            ),
        )
    
    def update(self, dt: float, flies: List[Fly], width: int, height: int) -> Dict[str, Any]:
        """Update frog brain & physics"""
        self.elapsed_time_s += dt
        obs, sensed_target, actual_distance = self._get_fly_observations(flies, width, height)
        alignment = obs[3]
        self.strike_cooldown = max(0.0, self.strike_cooldown - dt)
        reward_signal = self.pending_reward_signal
        self.pending_reward_signal = 0.0

        visible_targets = []
        visual_range = self.visual_range_px
        for target in self._visible_targets:
            visible_targets.append({key: value for key, value in target.items() if key != "fly"})

        frog_pos = np.array(self.body.position, dtype=float)
        heading_angle = float(np.arctan2(self.last_velocity[1], self.last_velocity[0])) if np.linalg.norm(self.last_velocity) > 1e-6 else 0.0
        brain_output = self.brain.update_predator_mode(
            position=tuple(frog_pos.tolist()),
            heading_angle=heading_angle,
            visible_targets=visible_targets,
            body_energy=self.energy,
            reward=reward_signal,
            visual_range_px=visual_range,
            food_collected=self.caught_flies,
            stall_time=self.stall_time,
            time_since_reward=max(0.0, self.elapsed_time_s - self.last_reward_time_s),
            visited_count=len(self.visited_zones),
        )
        self.last_brain_output = dict(brain_output)
        self._apply_developmental_stage()
        self.velocity = np.array(brain_output["velocity"], dtype=float)
        self.body.velocity = to_pymunk_vec(self.velocity * AGENT_MAX_SPEED)
        self.last_step_distance = float(np.linalg.norm(np.array(self.body.velocity, dtype=float)) * dt)

        focus_target = self._select_strike_target(brain_output)
        actual_distance = float(focus_target["distance"]) if focus_target is not None else 0.0
        alignment = float(focus_target["facing"]) if focus_target is not None else float(alignment)

        # Brain-led tongue/catch
        caught_fly = None
        self.tongue_extended = False
        catch_distance = self.shape.radius + self.tongue_reach + self.hit_radius
        strike_signal = 0.0
        strike_readiness = clamp(float(brain_output.get("strike_readiness", 0.0)), 0.0, 1.0)
        if focus_target is not None:
            closeness = max(0.0, min(1.0, 1.0 - actual_distance / max(1e-6, catch_distance)))
            strike_signal = clamp(
                0.10 + closeness * 0.62 + strike_readiness * 0.56 + max(0.0, alignment) * 0.12,
                0.0,
                1.0,
            )
        self.strike_drive = self.strike_drive * 0.64 + strike_signal * 0.36
        if strike_signal > 0.54:
            self.strike_commitment += 1
        else:
            self.strike_commitment = max(0, self.strike_commitment - 1)

        juvenile_mode = bool(brain_output.get("is_juvenile", self.brain.is_juvenile))
        readiness_threshold = 0.20 if juvenile_mode else 0.28
        drive_threshold = 0.54 if juvenile_mode else 0.60
        ready_to_strike = (
            focus_target is not None
            and actual_distance < catch_distance
            and self.strike_cooldown <= 0.0
            and strike_readiness > readiness_threshold
            and self.strike_drive > drive_threshold
            and self.strike_commitment >= 2
        )
        if ready_to_strike:
            self.tongue_extended = True
            self.tongue_target = to_pygame_vec(focus_target["position"])
            self.strike_cooldown = 0.08
            if random.random() < self.success_prob:
                caught_fly = focus_target["fly"]
                self.caught_flies += 1
                self.energy = min(AGENT_BODY_ENERGY_MAX, self.energy + 5.0)
                self.pending_reward_signal = 1.0
                self.last_reward_time_s = self.elapsed_time_s
                self.strike_drive = 0.15
                self.strike_commitment = 0

        # Energy drain (match frog_lib_ann/snn)
        speed = np.linalg.norm(self.velocity)
        energy_drain = AGENT_IDLE_DRAIN + AGENT_MOVE_DRAIN * speed
        self.energy = max(0.0, self.energy - energy_drain * dt)

        current_zone = self._visit_key(self.body.position)
        self.visited_zones.add(current_zone)
        progress_floor = max(0.12, AGENT_MAX_SPEED * dt * 0.12)
        if self.last_step_distance < progress_floor and not self._visible_targets:
            self.stall_time += dt
        else:
            recovery = 0.9 + clamp(self.last_step_distance / max(1e-6, AGENT_MAX_SPEED * dt), 0.0, 2.0)
            recovery += float(brain_output.get("search_burst", 0.0)) * 0.20
            recovery += float(brain_output.get("effective_motor_gate", 0.0)) * 0.12
            self.stall_time = max(0.0, self.stall_time - dt * recovery)

        if self.energy <= 0:
            self.alive = False

        self.last_velocity = self.velocity.copy()

        return {
            "position": np.array(self.body.position, dtype=float),
            "velocity": self.velocity,
            "alignment": alignment,
            "target_distance": actual_distance if focus_target is not None else None,
            "caught_fly": caught_fly,
            "caught_flies": self.caught_flies,
            "energy": self.energy,
            "controller_signal": float(np.linalg.norm(self.velocity)),
            "strike_drive": float(self.strike_drive),
            "strike_commitment": int(self.strike_commitment),
            "strike_cooldown": float(self.strike_cooldown),
            "strike_readiness": float(brain_output.get("strike_readiness", 0.0)),
            "memory_score": brain_output.get("memory_score", 0.0),
            "bg_gating_signal": float(brain_output.get("bg_gating_signal", 0.0)),
            "effective_motor_gate": float(brain_output.get("effective_motor_gate", 0.0)),
            "developmental_novelty": float(brain_output.get("developmental_novelty", 0.0)),
            "dopamine": brain_output.get("dopamine", 0.0),
            "serotonin": brain_output.get("serotonin", 0.0),
            "acetylcholine": brain_output.get("acetylcholine", 0.0),
            "neural_activity": brain_output.get("neural_activity", 0.0),
            "excitability": brain_output.get("excitability", 1.0),
            "tongue_extended": bool(self.tongue_extended),
            "tongue_length": float(self.tongue_reach if self.tongue_extended else 0.0),
            "visible_target_count": int(len(self._visible_targets)),
            "focus_x": float(focus_target["position"][0]) if focus_target is not None else None,
            "focus_y": float(focus_target["position"][1]) if focus_target is not None else None,
            "focus_vx": (
                float(getattr(getattr(focus_target["fly"], "body", focus_target["fly"]), "velocity", (0.0, 0.0))[0])
                if focus_target is not None
                else None
            ),
            "focus_vy": (
                float(getattr(getattr(focus_target["fly"], "body", focus_target["fly"]), "velocity", (0.0, 0.0))[1])
                if focus_target is not None
                else None
            ),
            "focus_distance": float(brain_output.get("focus_distance", actual_distance if focus_target is not None else 0.0)),
            "food_prediction_error": float(brain_output.get("food_prediction_error", 0.0)),
            "motivation_context": dict(brain_output.get("motivation_context", {}) or {}),
            "slow_hunt_drive": float(brain_output.get("slow_hunt_drive", 0.0)),
            "prey_permission": float(brain_output.get("prey_permission", 0.0)),
            "fast_target_lock": float(brain_output.get("fast_target_lock", 0.0)),
            "fast_orient_gain": float(brain_output.get("fast_orient_gain", 0.0)),
            "fast_loop_gate": float(brain_output.get("fast_loop_gate", 0.0)),
            "fast_strike_drive": float(brain_output.get("fast_strike_drive", 0.0)),
            "is_juvenile": brain_output.get("is_juvenile", self.brain.is_juvenile),
            "juvenile_progress": brain_output.get("juvenile_progress", 0.0),
            "maturity_readiness": brain_output.get("maturity_readiness", 0.0),
            "maturity_stability": brain_output.get("maturity_stability", 0.0),
        }

    def compact_status_line(self) -> str:
        state = "alive" if self.alive else "dead"
        stage = "juvenile" if self.brain.is_juvenile else "adult"
        if self.brain.is_juvenile:
            stage = f"{stage} {self.last_brain_output.get('juvenile_progress', 0.0):.0%}"
        return (
            f"Frog {self.agent_id} {state} {stage} | catches {self.caught_flies} | "
            f"energy {self.energy:.2f}"
        )

    def development_badge_text(self) -> str:
        if self.brain.is_juvenile:
            progress = self.last_brain_output.get("juvenile_progress", 0.0)
            return f"JUV {progress:.0%}"
        return "ADULT"

    def render_outer_radius(self) -> int:
        return max(FROG_RENDER_RADIUS_MIN, int(round(float(self.shape.radius) * FROG_RENDER_SCALE)))

    def neuro_panel_sections(self) -> List[Tuple[str, List[str]]]:
        output = self.last_brain_output or {}
        motivation = output.get("motivation_context", {}) or {}
        visible_count = len(getattr(self, "_visible_targets", []))
        focus_distance = float(output.get("focus_distance", 0.0) or 0.0)
        focus_text = f"{focus_distance:.1f}px" if visible_count > 0 else "none"
        return [
            (
                "State",
                [
                    self.compact_status_line(),
                    f"Visible flies {visible_count} | focus {focus_text}",
                    f"Strike drive {self.strike_drive:.2f} | cooldown {self.strike_cooldown:.2f}s",
                    f"Move {self.last_step_distance:.2f}px | speed {np.linalg.norm(self.velocity):.2f}",
                ],
            ),
            (
                "Development",
                [
                    f"Stage {'juvenile' if output.get('is_juvenile', True) else 'adult'} | progress {output.get('juvenile_progress', 0.0):.0%}",
                    f"Readiness {output.get('maturity_readiness', 0.0):.2f} | stability {output.get('maturity_stability', 0.0):.2f}",
                    f"Age {self.brain.juvenile_age} | self-select {output.get('self_selected_maturity_age', None)}",
                    f"Visited zones {len(self.visited_zones)} | stall {self.stall_time:.2f}s",
                ],
            ),
            (
                "Neurochemistry",
                [
                    f"DA {output.get('dopamine', 0.0):.2f} | 5-HT {output.get('serotonin', 0.0):.2f}",
                    f"ACh {output.get('acetylcholine', 0.0):.2f} | glucose {output.get('glucose', 0.0):.2f}",
                    f"Fatigue {output.get('fatigue', 0.0):.2f} | excitability {output.get('excitability', 1.0):.2f}",
                    f"Neural activity {output.get('neural_activity', 0.0):.2f} | replay {output.get('replay_gate', 0.0):.2f}",
                ],
            ),
            (
                "Affect",
                [
                    f"Arousal {output.get('arousal', 0.0):.2f} | curiosity {output.get('curiosity', 0.0):.2f}",
                    f"Frustration {output.get('frustration', 0.0):.2f} | reward conf {output.get('reward_confidence', 0.0):.2f}",
                    f"Restlessness {output.get('restlessness', 0.0):.2f} | food PE {output.get('food_prediction_error', 0.0):.2f}",
                ],
            ),
            (
                "Motivation Context",
                [
                    f"Hunger {motivation.get('hunger_bias', 0.0):.2f} | predation {motivation.get('predation_bias', 0.0):.2f}",
                    f"Reward seek {motivation.get('reward_seek_bias', 0.0):.2f} | vigilance {motivation.get('vigilance_bias', 0.0):.2f}",
                    f"Explore {motivation.get('exploration_bias', 0.0):.2f} | locomotor {motivation.get('locomotor_bias', 0.0):.2f}",
                ],
            ),
            (
                "Action Selection",
                [
                    f"Memory {output.get('memory_score', 0.0):.2f} | food dir {output.get('food_directness', 0.0):.2f}",
                    f"Dev novelty {output.get('developmental_novelty', 0.0):.2f} | focus {output.get('focus_distance', 0.0):.1f}px",
                    f"Policy conf {output.get('policy_confidence', 0.0):.2f} | conflict {output.get('policy_conflict', 0.0):.2f}",
                    f"BG gate {output.get('bg_gating_signal', 0.0):.2f} | BG conf {output.get('bg_confidence', 0.0):.2f}",
                    f"Motor gate {output.get('effective_motor_gate', 0.0):.2f} | strike ready {output.get('strike_readiness', 0.0):.2f}",
                    f"Prey perm {output.get('prey_permission', 0.0):.2f} | target lock {output.get('fast_target_lock', 0.0):.2f}",
                    f"Fast gate {output.get('fast_loop_gate', 0.0):.2f} | fast strike {output.get('fast_strike_drive', 0.0):.2f}",
                    f"Goal {output.get('goal_commitment', 0.0):.2f} | search {output.get('search_pressure', 0.0):.2f} | cmd {output.get('command_strength', 0.0):.2f}",
                    f"Vigor {output.get('action_vigor', 0.0):.2f} | burst {output.get('search_burst', 0.0):.2f}",
                ],
            ),
        ]
    
    def draw(self, surface: pygame.Surface):
        """Draw frog and tongue"""
        frog_pos = to_pygame_vec(self.body.position)
        outer_radius = self.render_outer_radius()
        inner_radius = max(3, int(round(outer_radius * FROG_INNER_SCALE)))
        pygame.draw.circle(surface, FROG_OUTER_COLOR, frog_pos, outer_radius)
        pygame.draw.circle(surface, FROG_INNER_COLOR, frog_pos, inner_radius)
        
        if self.tongue_extended and self.tongue_target is not None:
            pygame.draw.line(surface, FROG_TONGUE_COLOR, frog_pos, self.tongue_target, 3)
    
    def remove(self):
        """Clean up physics body"""
        try:
            if self.shape in self.space.shapes and self.body in self.space.bodies:
                self.space.remove(self.body, self.shape)
        except Exception:
            pass
        self.alive = False


class BiorealisticFlyCatchingSimulation:
    """Frog predator neuro - biorealistic fly catching simulation"""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        num_flies: int = 15,
        headless: bool = False,
        training_mode: bool = False,
        agent_count: int = 1,
        brain_seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.num_flies = num_flies
        self.headless = headless
        self.training_mode = training_mode
        self.agent_count = agent_count

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.dt = 0.01
        self.world_boundaries = add_world_boundaries(self.space, width, height)

        # Create single agent
        self.frogs = [
            BioFrogPredatorAgent(
                agent_id=1,
                space=self.space,
                position=(width // 2, height // 2),
                training_mode=training_mode,
                brain_rng_seed=brain_seed,
            )
        ]
        self.agents = self.frogs
        
        self.flies: List[Fly] = []
        self.spawn_flies(num_flies)

        pygame.init()
        if not headless:
            self.screen = pygame.display.set_mode((width + HUD_WIDTH, height))
            pygame.display.set_caption("Biorealistic Frog Predator Neuro")
            self.clock = pygame.time.Clock()
            self.title_font = pygame.font.Font(None, 24)
            self.font = pygame.font.Font(None, 18)
            self.small_font = pygame.font.Font(None, 16)
        else:
            # Headless mode: dummy display
            self.screen = None
            self.clock = None
            self.title_font = None
            self.font = None
            self.small_font = None

        self.step_count = 0
        self.running = True
        self.paused = False
        self.catch_history = deque(maxlen=2000)
        self.energy_history = deque(maxlen=2000)
        self.distance_history = deque(maxlen=2000)
        self.alignment_history = deque(maxlen=2000)
        self.speed_history = deque(maxlen=2000)
        self.controller_history = deque(maxlen=2000)
        self.neural_history = deque(maxlen=2000)

    def _spawn_position(self) -> Tuple[float, float]:
        margin_x = min(100, max(20, self.width // 5))
        margin_y = min(100, max(20, self.height // 5))
        return (
            random.uniform(margin_x, max(margin_x, self.width - margin_x)),
            random.uniform(margin_y, max(margin_y, self.height - margin_y)),
        )

    def spawn_flies(self, count: int):
        for _ in range(count):
            self.flies.append(Fly(self.space, self._spawn_position()))

    def respawn_flies(self):
        self.flies = [fly for fly in self.flies if fly.alive]
        missing = self.num_flies - len(self.flies)
        if missing > 0:
            self.spawn_flies(missing)

    def handle_events(self):
        """Process input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def step(self) -> Dict[str, Any]:
        """Update one time step"""
        for fly in self.flies:
            fly.update(self.dt, self.width, self.height)

        agents_state = []
        for frog in self.frogs:
            if not frog.alive:
                continue
            state = frog.update(self.dt, self.flies, self.width, self.height)
            agents_state.append(state)

            if state["caught_fly"] is not None and getattr(state["caught_fly"], "alive", False):
                state["caught_fly"].remove()
                self.catch_history.append(1)
            else:
                self.catch_history.append(0)

            velocity = np.array(state["velocity"], dtype=float)
            self.energy_history.append(float(state["energy"]))
            self.distance_history.append(state["target_distance"] if state["target_distance"] is not None else float("nan"))
            self.alignment_history.append(float(state["alignment"]))
            self.speed_history.append(float(np.linalg.norm(velocity)))
            self.controller_history.append(float(state["controller_signal"]))
            self.neural_history.append(float(state["neural_activity"]))

        self.space.step(self.dt)
        self.respawn_flies()
        self.step_count += 1
        
        return {"agents": agents_state, "step": self.step_count}

    def update(self, dt: float = None):
        """Update simulation"""
        if not self.paused:
            self.step()

    def _draw_panel_section(self, title: str, lines: List[str], x: int, y: int, max_y: int) -> int:
        if self.screen is None or self.small_font is None or self.font is None:
            return y

        title_surface = self.font.render(title, True, CYAN)
        self.screen.blit(title_surface, (x, y))
        y += 20
        for line in lines:
            if y > max_y:
                break
            surface = self.small_font.render(line, True, WHITE)
            self.screen.blit(surface, (x, y))
            y += 15
        return y + 6

    def _draw_stage_badge(self, frog: BioFrogPredatorAgent):
        if self.screen is None or self.small_font is None or not self.training_mode:
            return

        text = frog.development_badge_text()
        if frog.brain.is_juvenile:
            fill = (63, 71, 28)
            border = YELLOW
            text_color = YELLOW
        else:
            fill = (19, 64, 50)
            border = CYAN
            text_color = CYAN

        label = self.small_font.render(text, True, text_color)
        padding_x = 8
        padding_y = 4
        badge_width = label.get_width() + padding_x * 2
        badge_height = label.get_height() + padding_y * 2

        frog_x, frog_y = to_pygame_vec(frog.body.position)
        badge_rect = pygame.Rect(0, 0, badge_width, badge_height)
        badge_rect.centerx = frog_x
        badge_rect.bottom = frog_y - frog.render_outer_radius() - 8
        badge_rect.x = max(6, min(self.width - badge_rect.width - 6, badge_rect.x))
        badge_rect.y = max(6, badge_rect.y)

        pygame.draw.rect(self.screen, fill, badge_rect, border_radius=10)
        pygame.draw.rect(self.screen, border, badge_rect, 1, border_radius=10)
        self.screen.blit(label, (badge_rect.x + padding_x, badge_rect.y + padding_y))

    def _draw_hud(self):
        if self.screen is None or self.font is None or self.title_font is None:
            return

        hud_rect = pygame.Rect(self.width, 0, HUD_WIDTH, self.height)
        pygame.draw.rect(self.screen, HUD_BG, hud_rect)
        panel_rect = hud_rect.inflate(-16, -16)
        pygame.draw.rect(self.screen, HUD_PANEL, panel_rect, border_radius=14)
        pygame.draw.rect(self.screen, GRAY, panel_rect, 1, border_radius=14)

        x = panel_rect.x + 12
        y = panel_rect.y + 10
        max_y = panel_rect.bottom - 18

        title = self.title_font.render("Frog Neurobiology", True, YELLOW)
        self.screen.blit(title, (x, y))
        y += 26

        flies_alive = len([fly for fly in self.flies if fly.alive])
        focus_frog = next((frog for frog in self.frogs if frog.alive), self.frogs[0] if self.frogs else None)
        summary_lines = [
            f"Step {self.step_count} | flies {flies_alive}",
            f"Paused {'yes' if self.paused else 'no'} | dt {self.dt:.3f}s",
        ]
        if focus_frog is not None:
            stage = "juvenile" if focus_frog.brain.is_juvenile else "adult"
            progress = focus_frog.last_brain_output.get("juvenile_progress", 1.0 if not focus_frog.brain.is_juvenile else 0.0)
            readiness = focus_frog.last_brain_output.get("maturity_readiness", 0.0)
            stability = focus_frog.last_brain_output.get("maturity_stability", 0.0)
            summary_lines.append(
                f"Stage {stage} | prog {progress:.0%} | R {readiness:.2f} | S {stability:.2f}"
            )
        y = self._draw_panel_section("Runtime", summary_lines, x, y, max_y)

        for frog in self.frogs:
            y = self._draw_panel_section(f"Frog {frog.agent_id} Summary", [frog.compact_status_line()], x, y, max_y)
            if y > max_y:
                break

        if focus_frog is None:
            return

        for title, lines in focus_frog.neuro_panel_sections():
            y = self._draw_panel_section(title, lines, x, y, max_y)
            if y > max_y:
                break

    def draw(self):
        """Render frame"""
        if self.headless or self.screen is None:
            return

        self.screen.fill(HUD_BG)
        pygame.draw.rect(self.screen, (238, 242, 220), (0, 0, self.width, self.height))

        for fly in self.flies:
            fly.draw(self.screen)

        for frog in self.frogs:
            if frog.alive:
                frog.draw(self.screen)
                self._draw_stage_badge(frog)

        self._draw_hud()

        pygame.display.flip()

    def run(self, max_steps: int = 6000):
        """Main game loop"""
        print(f"BioFrog simulation starting | steps={max_steps} | headless={self.headless}")
        
        while self.running and self.step_count < max_steps:
            dt = 1.0 / 60.0
            
            if self.headless:
                pygame.event.pump()
            else:
                self.handle_events()
            
            if not self.paused:
                self.update(dt)
            
            if not self.headless:
                self.draw()
                if self.clock:
                    self.clock.tick(60)

        stats = self.get_statistics()
        print(f"BioFrog summary | frogs_alive={stats['frogs_alive']} | total_catches={stats['total_catches']}")
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Collect final statistics"""
        valid_distances = [value for value in self.distance_history if not np.isnan(value)]
        total_catches = sum(f.caught_flies for f in self.frogs)
        avg_energy = float(np.mean([f.energy for f in self.frogs]) if self.frogs else 0.0)
        avg_maturity_readiness = float(
            np.mean([f.last_brain_output.get("maturity_readiness", 0.0) for f in self.frogs]) if self.frogs else 0.0
        )
        avg_maturity_stability = float(
            np.mean([f.last_brain_output.get("maturity_stability", 0.0) for f in self.frogs]) if self.frogs else 0.0
        )
        return {
            "total_steps": self.step_count,
            "frogs_alive": len([f for f in self.frogs if f.alive]),
            "juveniles_alive": len([f for f in self.frogs if f.alive and f.brain.is_juvenile]),
            "adults_alive": len([f for f in self.frogs if f.alive and not f.brain.is_juvenile]),
            "total_catches": total_catches,
            "caught_flies": total_catches,
            "success_rate": (sum(self.catch_history) / len(self.catch_history)) if self.catch_history else 0.0,
            "avg_energy": avg_energy,
            "final_energy": avg_energy,
            "avg_alignment": float(np.mean(self.alignment_history)) if self.alignment_history else 0.0,
            "avg_distance_to_target": float(np.mean(valid_distances)) if valid_distances else float("nan"),
            "avg_speed": float(np.mean(self.speed_history)) if self.speed_history else 0.0,
            "avg_controller_signal": float(np.mean(self.controller_history)) if self.controller_history else 0.0,
            "avg_neural_activity": float(np.mean(self.neural_history)) if self.neural_history else 0.0,
            "avg_maturity_readiness": avg_maturity_readiness,
            "avg_maturity_stability": avg_maturity_stability,
            "architecture": "biorealistic neural with fly catching",
            "architecture_signature": "biorealistic gated predator loop",
        }

    def save_state(self, filename: str = "biofrog_state.json"):
        """Save statistics"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.get_statistics(), f, indent=2, ensure_ascii=False)

    def close(self):
        """Cleanup"""
        if self.screen is not None:
            pygame.quit()
        for frog in self.frogs:
            frog.remove()
        for fly in self.flies:
            fly.remove()


# Aliases for compatibility
Simulation = BiorealisticFlyCatchingSimulation

