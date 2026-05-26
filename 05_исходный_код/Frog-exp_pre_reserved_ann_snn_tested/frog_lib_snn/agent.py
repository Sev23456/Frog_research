#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymunk


SNN_VISUAL_RANGE = 180.0
SNN_STRIKE_RANGE = 60.0
SNN_DETECTION_THRESHOLD = 0.06


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def to_pymunk_vec(value: np.ndarray | List[float] | Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]), float(value[1])
    return 0.0, 0.0


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros(2, dtype=float)
    return vector / norm


class SNNBrain:
    """Spike-driven decoder with reward-modulated readout plasticity."""

    def __init__(self, neurons_per_feature: int = 8, lr: float = 0.018):
        self.input_dim = 11
        self.action_dim = 3
        self.neurons_per_feature = neurons_per_feature
        self.n_neurons = self.input_dim * neurons_per_feature
        self.lr = lr

        self.w_in = np.zeros((self.n_neurons, self.input_dim), dtype=float)
        self.w_out = np.zeros((self.action_dim, self.n_neurons), dtype=float)

        for feature in range(self.input_dim):
            start = feature * neurons_per_feature
            stop = start + neurons_per_feature
            self.w_in[start:stop, feature] = np.random.uniform(0.92, 1.38, neurons_per_feature)
            self.w_in[start:stop] += np.random.uniform(-0.06, 0.06, (neurons_per_feature, self.input_dim))

        self._init_readout()

        self.bias = np.random.uniform(0.04, 0.09, self.n_neurons)
        self.v = np.zeros((self.n_neurons,), dtype=float)
        self.threshold = 0.15
        self.tau = 0.085
        self.refractory_time = 0.012
        self.refractory = np.zeros((self.n_neurons,), dtype=float)

        self.pre_trace = np.zeros((self.n_neurons,), dtype=float)
        self.e_out = np.zeros_like(self.w_out)
        self.tau_pre = 0.08
        self.tau_e = 0.35

        self.is_juvenile = True
        self.juvenile_age = 0
        self.juvenile_duration = 5000

    def _init_readout(self):
        n = self.neurons_per_feature
        self.w_out[0, 0:n] = 0.62
        self.w_out[0, n : 2 * n] = -0.62
        self.w_out[1, 2 * n : 3 * n] = 0.62
        self.w_out[1, 3 * n : 4 * n] = -0.62

        strike_row = self.w_out[2]
        strike_row[4 * n : 5 * n] = 0.52
        strike_row[5 * n : 6 * n] = 0.46
        strike_row[6 * n : 7 * n] = 0.38
        strike_row[7 * n : 8 * n] = 0.18
        strike_row[9 * n : 10 * n] = 0.12
        strike_row[10 * n : 11 * n] = 0.20

        self.w_out[0, 6 * n : 7 * n] += 0.10
        self.w_out[1, 6 * n : 7 * n] += 0.10
        self.w_out[:, 9 * n : 10 * n] += 0.08

    def encode(
        self,
        target_vec: np.ndarray,
        visual_range: float,
        energy_ratio: float,
        brightness: float = 1.0,
        motion: float = 1.0,
        facing: float = 1.0,
        visibility: float = 1.0,
    ) -> np.ndarray:
        distance = max(1.0, float(np.linalg.norm(target_vec)))
        direction = target_vec / distance if distance > 0 else np.zeros(2, dtype=float)
        proximity = max(0.0, 1.0 - distance / visual_range)
        return np.array(
            [
                max(direction[0], 0.0),
                max(-direction[0], 0.0),
                max(direction[1], 0.0),
                max(-direction[1], 0.0),
                proximity,
                clamp(brightness, 0.0, 1.0),
                clamp(motion, 0.0, 1.0),
                max(facing, 0.0),
                max(-facing, 0.0),
                clamp(energy_ratio, 0.0, 1.0),
                clamp(visibility, 0.0, 1.0),
            ],
            dtype=float,
        )

    def forward(self, x: np.ndarray, dt: float = 0.01):
        decay = np.exp(-dt / self.tau)
        pre_decay = np.exp(-dt / self.tau_pre)
        e_decay = np.exp(-dt / self.tau_e)

        active = self.refractory <= 0.0
        self.refractory = np.maximum(0.0, self.refractory - dt)

        input_current = self.w_in.dot(x * 1.22) + self.bias
        self.v[active] = self.v[active] * decay + input_current[active] * (1.0 - decay)
        self.v[~active] = 0.0

        spikes = (self.v >= self.threshold).astype(float)
        if spikes.any():
            spike_mask = spikes.astype(bool)
            self.v[spike_mask] = 0.0
            self.refractory[spike_mask] = self.refractory_time

        out_raw = self.w_out.dot(spikes)
        out = np.tanh(out_raw * 1.25)

        self.pre_trace = self.pre_trace * pre_decay + spikes
        self.e_out = self.e_out * e_decay + np.outer(out, self.pre_trace)

        return out, float(spikes.mean()), int(spikes.sum()), spikes

    def reward_update(self, reward: float):
        if abs(reward) < 1e-6:
            return
        self.w_out += self.lr * reward * self.e_out
        self.w_out *= 0.9995
        self.w_out = np.clip(self.w_out, -1.6, 1.6)


class FrogSNNAgent:
    """Spike-driven frog without direct pursuit-controller mixing."""

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        training_mode: bool = False,
    ):
        self.space = space
        self.position = np.array(position, dtype=float)
        self.training_mode = training_mode

        self.radius = 9.0
        self.max_speed = 65.0
        self.visual_range = SNN_VISUAL_RANGE

        moment = pymunk.moment_for_circle(1.0, 0, self.radius)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = tuple(self.position)
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7
        self.space.add(self.body, self.shape)

        self.brain = SNNBrain()
        self.brain.is_juvenile = bool(training_mode)
        self.brain.juvenile_age = 0 if training_mode else self.brain.juvenile_duration

        self.max_energy = 30.0
        self.energy = float(self.max_energy)
        self.caught_flies = 0
        self.steps = 0
        self.last_catch_time = 0
        self.catch_cooldown = 12

        self.hit_radius = 24.0 if training_mode else 34.0
        self.success_prob = 0.75 if training_mode else 0.92
        self.tongue_reach = SNN_STRIKE_RANGE
        self.catch_distance = self.radius + self.tongue_reach + self.hit_radius

        self.tongue_extended = False
        self.tongue_length = 0.0
        self.tongue_target: Optional[np.ndarray] = None
        self.attached_fly = None

        self._visible_targets: List[Dict[str, Any]] = []
        self.last_velocity = np.zeros(2, dtype=float)
        self.last_spike_count = 0
        self.last_focus_distance = self.visual_range
        self.last_reward_signal = 0.0
        self.strike_drive = 0.0
        self.strike_commitment = 0
        self.strike_cooldown = 0.0

    def remove(self):
        try:
            if self.shape in self.space.shapes and self.body in self.space.bodies:
                self.space.remove(self.body, self.shape)
        except Exception:
            pass

    def detect_flies(self, flies: List[Any]):
        heading = unit_vector(self.last_velocity)
        if np.linalg.norm(heading) <= 1e-6:
            heading = np.array([1.0, 0.0], dtype=float)

        current_velocity = np.array(self.body.velocity, dtype=float)
        visible_targets: List[Dict[str, Any]] = []
        self._visible_targets = []
        for fly in flies:
            if hasattr(fly, "alive") and not fly.alive:
                continue
            fly_pos = np.array(fly.body.position if hasattr(fly, "body") else fly, dtype=float)
            vector = fly_pos - self.position
            distance = float(np.linalg.norm(vector))
            if distance <= 1e-6 or distance > self.visual_range:
                continue

            direction = vector / distance
            facing = float(np.dot(heading, direction))
            peripheral_gain = 0.40 + 0.60 * clamp((facing + 1.0) * 0.5, 0.0, 1.0)
            fly_velocity = np.array(getattr(getattr(fly, "body", fly), "velocity", (0.0, 0.0)), dtype=float)
            relative_motion = fly_velocity - current_velocity
            motion_signal = clamp(float(np.linalg.norm(relative_motion)) / 95.0, 0.0, 1.0)
            distance_gain = clamp(1.0 - distance / max(1.0, self.visual_range), 0.0, 1.0)
            brightness = clamp(distance_gain * (0.35 + motion_signal * 0.65) * peripheral_gain, 0.0, 1.0)
            if brightness < SNN_DETECTION_THRESHOLD:
                continue

            visible_targets.append(
                {
                    "fly": fly,
                    "vector": vector,
                    "distance": distance,
                    "brightness": brightness,
                    "motion": motion_signal,
                    "facing": facing,
                    "position": fly_pos,
                }
            )

        self._visible_targets = visible_targets
        return visible_targets

    def _focus_target(self) -> Optional[Dict[str, Any]]:
        if not self._visible_targets:
            return None
        return max(
            self._visible_targets,
            key=lambda item: (
                item["brightness"] * 0.58
                + item["motion"] * 0.24
                + max(0.0, item["facing"]) * 0.08
                + clamp(1.0 - item["distance"] / self.visual_range, 0.0, 1.0) * 0.10
            ),
        )

    def extend_tongue(self, target_position: np.ndarray):
        if not self.tongue_extended:
            self.tongue_extended = True
            self.tongue_target = np.array(target_position, dtype=float)
            self.tongue_length = 0.0

    def retract_tongue(self):
        self.tongue_extended = False
        self.tongue_length = 0.0
        self.attached_fly = None
        self.tongue_target = None

    def _shaped_reward(self, reward: float, focus_target: Optional[Dict[str, Any]], strike_intent: float) -> float:
        current_distance = focus_target["distance"] if focus_target is not None else self.visual_range
        approach = clamp((self.last_focus_distance - current_distance) / max(1.0, self.visual_range), -0.45, 0.45)
        total_reward = reward * 2.2 + approach * 0.52
        if focus_target is not None:
            total_reward += focus_target["brightness"] * 0.035 + focus_target["motion"] * 0.020
            if strike_intent > 0.15 and current_distance < self.catch_distance * 1.10:
                total_reward += 0.04
        elif strike_intent > 0.15:
            total_reward -= 0.03
        return float(total_reward - 0.002)

    def update(self, dt: float, flies: List[Any]) -> Dict[str, Any]:
        self.steps += 1
        self.position = np.array(self.body.position, dtype=float)
        if self.training_mode and self.brain.is_juvenile:
            self.brain.juvenile_age += 1
            hunting_progress = clamp(self.caught_flies / 6.0, 0.0, 1.0)
            if hunting_progress >= 1.0 and self.energy / self.max_energy > 0.35:
                self.brain.is_juvenile = False
                self.brain.juvenile_age = self.brain.juvenile_duration

        self.strike_cooldown = max(0.0, self.strike_cooldown - dt)
        visible_targets = self.detect_flies(flies)
        focus_target = self._focus_target()

        reward = 0.0
        if self.attached_fly is not None:
            reward = 1.0
            self.energy = min(self.max_energy, self.energy + 5.0)
            self.attached_fly = None

        self.energy = max(0.0, self.energy - (0.08 + 0.02 * np.linalg.norm(self.last_velocity)) * dt)
        energy_ratio = clamp(self.energy / self.max_energy, 0.0, 1.0)

        if focus_target is not None:
            encoded = self.brain.encode(
                np.array(focus_target["vector"], dtype=float),
                self.visual_range,
                energy_ratio,
                brightness=focus_target["brightness"],
                motion=focus_target["motion"],
                facing=focus_target["facing"],
                visibility=1.0,
            )
        else:
            encoded = self.brain.encode(
                np.array([0.0, 0.0], dtype=float),
                self.visual_range,
                energy_ratio,
                brightness=0.0,
                motion=0.0,
                facing=0.0,
                visibility=0.0,
            )

        micro_outs = []
        micro_activities = []
        total_spikes = 0
        for _ in range(4):
            micro_out, micro_activity, micro_spike_count, _ = self.brain.forward(encoded, dt=dt / 4.0)
            micro_outs.append(micro_out)
            micro_activities.append(micro_activity)
            total_spikes += micro_spike_count

        out = np.mean(micro_outs, axis=0)
        neural_activity = float(np.mean(micro_activities))
        spike_count = int(total_spikes)

        movement = np.array(out[:2], dtype=float)
        move_norm = float(np.linalg.norm(movement))
        if move_norm > 1.0:
            movement = movement / move_norm
        velocity = 0.42 * self.last_velocity + 0.58 * movement
        if np.linalg.norm(velocity) > 1.0:
            velocity = velocity / max(1e-6, np.linalg.norm(velocity))

        self.body.velocity = to_pymunk_vec(velocity * self.max_speed)
        self.last_velocity = velocity
        self.last_spike_count = spike_count

        target_distance = focus_target["distance"] if focus_target is not None else None
        alignment = 0.0
        if focus_target is not None and np.linalg.norm(velocity) > 1e-6:
            alignment = float(np.dot(unit_vector(velocity), unit_vector(np.array(focus_target["vector"], dtype=float))))

        strike_intent = float(out[2])
        strike_signal = 0.0
        if focus_target is not None:
            closeness = clamp(1.0 - focus_target["distance"] / max(1e-6, self.catch_distance), 0.0, 1.0)
            burst_gain = clamp(spike_count / max(1.0, self.brain.neurons_per_feature * 1.8), 0.0, 1.0)
            strike_signal = clamp(
                0.08
                + clamp((strike_intent + 1.0) * 0.5, 0.0, 1.0) * 0.54
                + closeness * 0.26
                + focus_target["brightness"] * 0.08
                + burst_gain * 0.10,
                0.0,
                1.0,
            )

        self.strike_drive = self.strike_drive * 0.60 + strike_signal * 0.40
        if strike_signal > 0.52:
            self.strike_commitment += 1
        else:
            self.strike_commitment = max(0, self.strike_commitment - 1)

        ready_to_strike = (
            focus_target is not None
            and focus_target["distance"] < self.catch_distance
            and self.strike_cooldown <= 0.0
            and strike_intent > -0.08
            and self.strike_drive > 0.58
            and self.strike_commitment >= 2
            and (self.steps - self.last_catch_time) > self.catch_cooldown
        )
        if ready_to_strike:
            self.extend_tongue(np.array(focus_target["position"], dtype=float))
            self.strike_cooldown = 0.08

        caught_fly = None
        if self.tongue_extended and self.tongue_target is not None:
            direction = self.tongue_target - self.position
            distance = float(np.linalg.norm(direction))
            if distance > 0:
                direction = direction / distance
            self.tongue_length += 320.0 * dt
            if self.tongue_length >= 150.0 or distance < self.tongue_length:
                self.retract_tongue()
            if self.attached_fly is None and (self.steps - self.last_catch_time) > self.catch_cooldown:
                tongue_end = self.position + direction * self.tongue_length
                for target in self._visible_targets:
                    fly = target["fly"]
                    fly_pos = np.array(fly.body.position, dtype=float)
                    if np.linalg.norm(fly_pos - tongue_end) < self.hit_radius and random.random() < self.success_prob:
                        self.attached_fly = fly
                        caught_fly = fly
                        self.caught_flies += 1
                        self.last_catch_time = self.steps
                        self.strike_drive = 0.18
                        self.strike_commitment = 0
                        break

        reward_signal = self._shaped_reward(reward, focus_target, strike_intent)
        self.brain.reward_update(reward_signal)
        self.last_reward_signal = reward_signal
        self.last_focus_distance = focus_target["distance"] if focus_target is not None else self.visual_range

        return {
            "position": self.position,
            "velocity": velocity,
            "energy": self.energy,
            "caught_flies": self.caught_flies,
            "fatigue": 1.0 - (self.energy / self.max_energy),
            "is_juvenile": self.brain.is_juvenile,
            "juvenile_progress": (
                min(1.0, self.brain.juvenile_age / self.brain.juvenile_duration)
                if self.training_mode and self.brain.is_juvenile
                else 1.0
            ),
            "tongue_extended": self.tongue_extended,
            "tongue_length": self.tongue_length,
            "controller_signal": float(np.linalg.norm(movement)),
            "neural_activity": neural_activity,
            "spike_count": spike_count,
            "reward": reward,
            "learning_reward": float(self.last_reward_signal),
            "caught_fly": caught_fly,
            "target_distance": target_distance,
            "alignment": alignment,
            "eligibility_norm": float(np.linalg.norm(self.brain.e_out)),
            "avg_membrane": float(np.mean(self.brain.v)),
            "refractory_fraction": float(np.mean((self.brain.refractory > 0.0).astype(float))),
            "strike_drive": float(self.strike_drive),
            "strike_intent": float(strike_intent),
            "strike_commitment": int(self.strike_commitment),
            "strike_cooldown": float(self.strike_cooldown),
            "visibility": 1.0 if focus_target is not None else 0.0,
            "focus_brightness": float(focus_target["brightness"]) if focus_target is not None else 0.0,
            "focus_motion": float(focus_target["motion"]) if focus_target is not None else 0.0,
            "focus_facing": float(focus_target["facing"]) if focus_target is not None else 0.0,
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
            "visual_range": self.visual_range,
        }
