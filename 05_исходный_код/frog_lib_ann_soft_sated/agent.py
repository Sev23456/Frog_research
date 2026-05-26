#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymunk

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


ANN_VISUAL_RANGE = 180.0
ANN_STRIKE_RANGE = 60.0
ANN_DETECTION_THRESHOLD = 0.06
ANN_SOFT_SATIETY_PENALTY_START_RATIO = 0.55
ANN_SOFT_SATIETY_PENALTY_FULL_RATIO = 0.95
ANN_SOFT_STRIKE_SUPPRESSION = 0.52
ANN_SOFT_STRIKE_RELIEF = 0.20
ANN_SOFT_CAPTURE_REWARD_SUPPRESSION = 0.78


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


def normal_log_prob(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    variance = np.maximum(std ** 2, 1e-6)
    return float(-0.5 * np.sum(((action - mean) ** 2) / variance + np.log(2.0 * math.pi * variance)))


if TORCH_AVAILABLE:
    class TorchActorCriticNet(nn.Module):
        def __init__(self, input_dim: int = 10, action_dim: int = 3):
            super().__init__()
            self.feature = nn.Linear(input_dim, input_dim)
            self.actor = nn.Linear(input_dim, action_dim)
            self.value_head = nn.Linear(input_dim, 1)
            self.log_std = nn.Parameter(torch.tensor([-1.05, -1.05, -0.80], dtype=torch.float32))
            self._init_weights()

        def _init_weights(self):
            with torch.no_grad():
                self.feature.weight.copy_(torch.eye(self.feature.in_features))
                self.feature.bias.zero_()
                self.actor.weight.zero_()
                self.actor.bias.zero_()
                self.actor.weight[0, 0] = 1.30
                self.actor.weight[0, 7] = 0.28
                self.actor.weight[1, 1] = 1.30
                self.actor.weight[1, 8] = 0.28
                self.actor.weight[2, 2] = 0.90
                self.actor.weight[2, 3] = 0.65
                self.actor.weight[2, 4] = 0.55
                self.actor.weight[2, 5] = 0.18
                self.actor.weight[2, 9] = 0.12
                self.value_head.weight.zero_()
                self.value_head.bias.zero_()

        def forward(self, x: torch.Tensor):
            hidden = torch.tanh(self.feature(x))
            mean = torch.tanh(self.actor(hidden))
            value = self.value_head(hidden).squeeze(-1)
            return hidden, mean, value


    class TorchActorCriticBrain:
        def __init__(
            self,
            input_dim: int = 10,
            actor_lr: float = 7e-4,
            gamma: float = 0.985,
            value_coef: float = 0.65,
            entropy_beta: float = 0.002,
            device: Optional[str] = None,
        ):
            self.device = torch.device(device or "cpu")
            self.net = TorchActorCriticNet(input_dim=input_dim).to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=actor_lr)
            self.gamma = gamma
            self.value_coef = value_coef
            self.entropy_beta = entropy_beta

        def act(self, obs: np.ndarray, exploration_scale: float = 1.0) -> Dict[str, Any]:
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            hidden, mean, value = self.net(x)
            std = torch.clamp(self.net.log_std.exp() * exploration_scale, 0.08, 0.60)
            sample = mean + torch.randn_like(mean) * std
            action = torch.clamp(sample, -1.0, 1.0)
            log_prob = -0.5 * ((((sample - mean) / std) ** 2) + 2.0 * torch.log(std) + math.log(2.0 * math.pi)).sum(dim=-1)
            entropy = (0.5 * (1.0 + math.log(2.0 * math.pi)) + torch.log(std)).sum(dim=-1)
            mean_np = mean.detach().cpu().numpy().reshape(-1)
            action_np = action.detach().cpu().numpy().reshape(-1)
            return {
                "obs": obs.copy(),
                "action": action_np,
                "mean_action": mean_np,
                "value_tensor": value.squeeze(0),
                "value_estimate": float(value.item()),
                "log_prob": log_prob.squeeze(0),
                "entropy": entropy.squeeze(0),
                "focus_distance": None,
                "strike_intent": float(action_np[2]),
                "exploration_scale": float(exploration_scale),
            }

        def learn(self, transition: Dict[str, Any], reward: float, next_obs: np.ndarray, done: bool = False) -> float:
            next_x = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, _, next_value = self.net(next_x)
                target = float(reward) + (0.0 if done else self.gamma * float(next_value.item()))

            value_tensor = transition["value_tensor"]
            advantage = target - float(value_tensor.item())
            actor_loss = -transition["log_prob"] * advantage - self.entropy_beta * transition["entropy"]
            value_loss = 0.5 * (value_tensor - target) ** 2
            loss = actor_loss + self.value_coef * value_loss

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.5)
            self.opt.step()
            return float(advantage)


class NumpyActorCriticBrain:
    def __init__(
        self,
        input_dim: int = 10,
        action_dim: int = 3,
        actor_lr: float = 3e-3,
        critic_lr: float = 4.5e-3,
        gamma: float = 0.985,
        value_coef: float = 0.60,
    ):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.value_coef = value_coef

        self.w1 = np.eye(input_dim, dtype=float) + np.random.randn(input_dim, input_dim) * 0.01
        self.b1 = np.zeros((input_dim,), dtype=float)
        self.w_actor = np.zeros((action_dim, input_dim), dtype=float)
        self.b_actor = np.zeros((action_dim,), dtype=float)
        self.w_value = np.random.randn(input_dim) * 0.01
        self.b_value = 0.0
        self.std_base = np.array([0.34, 0.34, 0.42], dtype=float)

        self.w_actor[0, 0] = 1.30
        self.w_actor[0, 7] = 0.28
        self.w_actor[1, 1] = 1.30
        self.w_actor[1, 8] = 0.28
        self.w_actor[2, 2] = 0.90
        self.w_actor[2, 3] = 0.65
        self.w_actor[2, 4] = 0.55
        self.w_actor[2, 5] = 0.18
        self.w_actor[2, 9] = 0.12

    def _forward(self, obs: np.ndarray):
        z1 = self.w1.dot(obs) + self.b1
        hidden = np.tanh(z1)
        mean = np.tanh(self.w_actor.dot(hidden) + self.b_actor)
        value = float(self.w_value.dot(hidden) + self.b_value)
        return z1, hidden, mean, value

    def act(self, obs: np.ndarray, exploration_scale: float = 1.0) -> Dict[str, Any]:
        z1, hidden, mean, value = self._forward(obs)
        std = np.clip(self.std_base * exploration_scale, 0.08, 0.60)
        action = np.clip(mean + np.random.randn(self.action_dim) * std, -1.0, 1.0)
        return {
            "obs": obs.copy(),
            "z1": z1,
            "hidden": hidden,
            "mean": mean,
            "action": action,
            "std": std,
            "value": value,
            "value_estimate": float(value),
            "mean_action": mean.copy(),
            "log_prob": normal_log_prob(action, mean, std),
            "focus_distance": None,
            "strike_intent": float(action[2]),
            "exploration_scale": float(exploration_scale),
        }

    def learn(self, transition: Dict[str, Any], reward: float, next_obs: np.ndarray, done: bool = False) -> float:
        _, _, _, next_value = self._forward(next_obs)
        target = float(reward) + (0.0 if done else self.gamma * next_value)
        value = float(transition["value"])
        advantage = target - value

        hidden = transition["hidden"]
        mean = transition["mean"]
        action = transition["action"]
        std = np.maximum(transition["std"], 0.08)
        obs = transition["obs"]
        z1 = transition["z1"]

        value_error = value - target
        grad_w_value = value_error * hidden
        grad_b_value = value_error

        grad_actor = -advantage * (action - mean) / (std ** 2)
        grad_actor_raw = grad_actor * (1.0 - mean ** 2)

        grad_w_actor = np.outer(grad_actor_raw, hidden)
        grad_b_actor = grad_actor_raw

        hidden_grad = self.w_actor.T.dot(grad_actor_raw) + self.value_coef * self.w_value * value_error
        z1_grad = hidden_grad * (1.0 - np.tanh(z1) ** 2)
        grad_w1 = np.outer(z1_grad, obs)
        grad_b1 = z1_grad

        self.w_actor -= self.actor_lr * np.clip(grad_w_actor, -0.8, 0.8)
        self.b_actor -= self.actor_lr * np.clip(grad_b_actor, -0.8, 0.8)
        self.w_value -= self.critic_lr * np.clip(grad_w_value, -1.2, 1.2)
        self.b_value -= self.critic_lr * float(np.clip(grad_b_value, -1.2, 1.2))
        self.w1 -= self.actor_lr * np.clip(grad_w1, -0.6, 0.6)
        self.b1 -= self.actor_lr * np.clip(grad_b1, -0.6, 0.6)
        return float(advantage)


class FrogANNAgent:
    """ANN frog with actor-critic RL and no hand-crafted pursuit controller."""

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        training_mode: bool = False,
        use_torch: Optional[bool] = None,
    ):
        self.space = space
        self.position = np.array(position, dtype=float)
        self.training_mode = training_mode
        self.learning_enabled = True

        self.radius = 9.0
        self.max_speed = 65.0
        self.visual_range = ANN_VISUAL_RANGE

        moment = pymunk.moment_for_circle(1.0, 0, self.radius)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = tuple(self.position)
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7
        self.space.add(self.body, self.shape)

        if use_torch is None:
            use_torch = False
        self.brain = TorchActorCriticBrain() if use_torch and TORCH_AVAILABLE else NumpyActorCriticBrain()

        self.max_energy = 30.0
        self.energy = float(self.max_energy)
        self.satiety_penalty_start_ratio = ANN_SOFT_SATIETY_PENALTY_START_RATIO
        self.satiety_penalty_full_ratio = ANN_SOFT_SATIETY_PENALTY_FULL_RATIO
        self.hunting_enabled = True
        self.satiety_pressure = 0.0
        self.strike_permission = 1.0
        self.capture_reward_scale = 1.0
        self.caught_flies = 0
        self.steps = 0
        self.last_catch_time = 0
        self.catch_cooldown = 12

        self.hit_radius = 24.0 if training_mode else 34.0
        self.success_prob = 0.75 if training_mode else 0.92
        self.tongue_reach = ANN_STRIKE_RANGE
        self.catch_distance = self.radius + self.tongue_reach + self.hit_radius

        self.tongue_extended = False
        self.tongue_length = 0.0
        self.tongue_target: Optional[np.ndarray] = None
        self.attached_fly = None

        self._visible_targets: List[Dict[str, Any]] = []
        self.last_velocity = np.zeros(2, dtype=float)
        self.prev_transition: Optional[Dict[str, Any]] = None
        self.last_advantage = 0.0
        self.last_reward_signal = 0.0
        self.last_focus_distance = self.visual_range
        self.last_focus_brightness = 0.0
        self.last_focus_motion = 0.0
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

        visible_targets: List[Dict[str, Any]] = []
        self._visible_targets = []
        current_velocity = np.array(self.body.velocity, dtype=float)
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
            if brightness < ANN_DETECTION_THRESHOLD:
                continue

            target = {
                "fly": fly,
                "vector": vector,
                "distance": distance,
                "brightness": brightness,
                "motion": motion_signal,
                "facing": facing,
                "position": fly_pos,
            }
            visible_targets.append(target)

        self._visible_targets = visible_targets
        return visible_targets

    def _focus_target(self) -> Optional[Dict[str, Any]]:
        if not self._visible_targets:
            return None
        return max(
            self._visible_targets,
            key=lambda item: (
                item["brightness"] * 0.58
                + item["motion"] * 0.22
                + max(0.0, item["facing"]) * 0.10
                + clamp(1.0 - item["distance"] / self.visual_range, 0.0, 1.0) * 0.10
            ),
        )

    def _observation(self, focus_target: Optional[Dict[str, Any]]) -> np.ndarray:
        obs = np.zeros((10,), dtype=float)
        if focus_target is not None:
            direction = unit_vector(np.array(focus_target["vector"], dtype=float))
            obs[0] = float(direction[0])
            obs[1] = float(direction[1])
            obs[2] = clamp(1.0 - focus_target["distance"] / max(1.0, self.visual_range), 0.0, 1.0)
            obs[3] = clamp(focus_target["brightness"], 0.0, 1.0)
            obs[4] = clamp(focus_target["motion"], 0.0, 1.0)
            obs[5] = clamp(focus_target["facing"], -1.0, 1.0)
            obs[9] = 1.0
        obs[6] = clamp(self.energy / self.max_energy, 0.0, 1.0)
        obs[7] = clamp(self.last_velocity[0], -1.0, 1.0)
        obs[8] = clamp(self.last_velocity[1], -1.0, 1.0)
        return obs

    def _transition_reward(
        self,
        reward: float,
        focus_target: Optional[Dict[str, Any]],
        reward_scale: float,
        reward_satiety_pressure: float,
    ) -> float:
        current_distance = focus_target["distance"] if focus_target is not None else self.visual_range
        current_brightness = focus_target["brightness"] if focus_target is not None else 0.0
        current_motion = focus_target["motion"] if focus_target is not None else 0.0
        approach = clamp((self.last_focus_distance - current_distance) / max(1.0, self.visual_range), -0.45, 0.45)

        total_reward = reward * 2.4 * reward_scale + approach * 0.60
        total_reward += current_brightness * 0.04 + current_motion * 0.02
        if focus_target is None:
            total_reward -= 0.01

        if self.prev_transition is not None:
            strike_intent = float(self.prev_transition.get("strike_intent", 0.0))
            if strike_intent > 0.20 and focus_target is not None and current_distance < self.catch_distance * 1.10:
                total_reward += 0.05
            elif strike_intent > 0.20 and (focus_target is None or current_distance > self.catch_distance * 1.35):
                total_reward -= 0.03

        if reward > 0.0 and reward_satiety_pressure > 0.0:
            total_reward -= 0.18 * reward_satiety_pressure

        return float(total_reward - 0.002)

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

    def _satiety_pressure(self, energy_ratio: float) -> float:
        return clamp(
            (energy_ratio - self.satiety_penalty_start_ratio)
            / max(self.satiety_penalty_full_ratio - self.satiety_penalty_start_ratio, 1e-6),
            0.0,
            1.0,
        )

    def _capture_reward_scale(self, satiety_pressure: float) -> float:
        return clamp(1.0 - ANN_SOFT_CAPTURE_REWARD_SUPPRESSION * satiety_pressure, 0.22, 1.0)

    def _homeostatic_modulators(
        self,
        focus_target: Optional[Dict[str, Any]],
    ) -> Tuple[bool, float, float, float, float]:
        energy_ratio = clamp(self.energy / max(self.max_energy, 1e-6), 0.0, 1.0)
        satiety_pressure = self._satiety_pressure(energy_ratio)
        hunger_drive = 1.0 - satiety_pressure
        closeness = 0.0
        if focus_target is not None:
            closeness = clamp(1.0 - focus_target["distance"] / max(1e-6, self.catch_distance), 0.0, 1.0)
        strike_permission = clamp(
            1.0 - ANN_SOFT_STRIKE_SUPPRESSION * satiety_pressure + ANN_SOFT_STRIKE_RELIEF * closeness,
            0.50,
            1.0,
        )
        capture_reward_scale = self._capture_reward_scale(satiety_pressure)
        hunting_enabled = strike_permission >= 0.60 or hunger_drive >= 0.45
        return (
            bool(hunting_enabled),
            float(hunger_drive),
            float(satiety_pressure),
            float(strike_permission),
            float(capture_reward_scale),
        )

    def update(self, dt: float, flies: List[Any]) -> Dict[str, Any]:
        self.steps += 1
        self.position = np.array(self.body.position, dtype=float)
        self.strike_cooldown = max(0.0, self.strike_cooldown - dt)

        visible_targets = self.detect_flies(flies)
        focus_target = self._focus_target()
        obs = self._observation(focus_target)

        reward = 0.0
        reward_energy_ratio = clamp(self.energy / max(self.max_energy, 1e-6), 0.0, 1.0)
        reward_satiety_pressure = self._satiety_pressure(reward_energy_ratio)
        reward_scale = self._capture_reward_scale(reward_satiety_pressure)
        if self.attached_fly is not None:
            reward = 1.0
            self.energy = min(self.max_energy, self.energy + 5.0)
            self.attached_fly = None

        if self.prev_transition is not None and self.learning_enabled:
            learning_reward = self._transition_reward(reward, focus_target, reward_scale, reward_satiety_pressure)
            self.last_advantage = self.brain.learn(self.prev_transition, learning_reward, obs, done=False)
            self.last_reward_signal = learning_reward
        else:
            self.last_advantage = 0.0
            self.last_reward_signal = float(reward)

        self.energy = max(0.0, self.energy - (0.08 + 0.02 * np.linalg.norm(self.last_velocity)) * dt)
        hunting_enabled, hunger_drive, satiety_gate, strike_permission, capture_reward_scale = self._homeostatic_modulators(
            focus_target
        )
        self.hunting_enabled = hunting_enabled
        self.satiety_pressure = satiety_gate
        self.strike_permission = strike_permission
        self.capture_reward_scale = capture_reward_scale

        visibility = 1.0 if focus_target is not None else 0.0
        exploration_scale = 0.85 if visibility > 0.5 else 1.10
        if not self.training_mode:
            exploration_scale *= 0.45
        policy_step = self.brain.act(obs, exploration_scale=exploration_scale)
        policy_step["focus_distance"] = focus_target["distance"] if focus_target is not None else self.visual_range
        self.prev_transition = policy_step

        raw_action = np.array(policy_step["action"], dtype=float)
        movement = raw_action[:2]
        move_norm = float(np.linalg.norm(movement))
        if move_norm > 1.0:
            movement = movement / move_norm

        velocity = 0.46 * self.last_velocity + 0.54 * movement
        speed = float(np.linalg.norm(velocity))
        if speed > 1.0:
            velocity = velocity / speed

        self.body.velocity = to_pymunk_vec(velocity * self.max_speed)
        self.last_velocity = velocity

        caught_fly = None
        target_distance = focus_target["distance"] if focus_target is not None else None
        alignment = 0.0
        strike_intent = float(raw_action[2])
        if focus_target is not None and np.linalg.norm(velocity) > 1e-6:
            alignment = float(np.dot(unit_vector(velocity), unit_vector(np.array(focus_target["vector"], dtype=float))))

        strike_signal = 0.0
        if focus_target is not None:
            closeness = clamp(1.0 - focus_target["distance"] / max(1e-6, self.catch_distance), 0.0, 1.0)
            strike_signal = clamp(
                0.08
                + clamp((strike_intent + 1.0) * 0.5, 0.0, 1.0) * 0.56
                + closeness * 0.28
                + focus_target["brightness"] * 0.08
                + max(0.0, alignment) * 0.06,
                0.0,
                1.0,
            )
        strike_signal *= strike_permission
        self.strike_drive = self.strike_drive * 0.58 + strike_signal * 0.42
        if strike_signal > 0.54:
            self.strike_commitment += 1
        else:
            self.strike_commitment = max(0, self.strike_commitment - 1)

        ready_to_strike = (
            focus_target is not None
            and focus_target["distance"] < self.catch_distance
            and self.strike_cooldown <= 0.0
            and strike_intent > -0.05
            and self.strike_drive > 0.60
            and self.strike_commitment >= 2
            and (self.steps - self.last_catch_time) > self.catch_cooldown
        )
        if ready_to_strike:
            self.extend_tongue(np.array(focus_target["position"], dtype=float))
            self.strike_cooldown = 0.08

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

        self.last_focus_distance = focus_target["distance"] if focus_target is not None else self.visual_range
        self.last_focus_brightness = focus_target["brightness"] if focus_target is not None else 0.0
        self.last_focus_motion = focus_target["motion"] if focus_target is not None else 0.0

        if self.energy <= 0.0:
            self.body.velocity = (0.0, 0.0)

        controller_signal = float(np.linalg.norm(movement))
        return {
            "position": self.position,
            "velocity": velocity,
            "energy": self.energy,
            "caught_flies": self.caught_flies,
            "fatigue": 1.0 - (self.energy / self.max_energy),
            "tongue_extended": self.tongue_extended,
            "tongue_length": self.tongue_length,
            "controller_signal": controller_signal,
            "neural_activity": controller_signal,
            "reward": reward,
            "learning_reward": float(self.last_reward_signal),
            "actor_advantage": float(self.last_advantage),
            "value_estimate": float(policy_step.get("value_estimate", 0.0)),
            "learning_enabled": bool(self.learning_enabled),
            "exploration_scale": float(policy_step.get("exploration_scale", 0.0)),
            "hunting_enabled": bool(hunting_enabled),
            "homeostatic_hunger_drive": float(hunger_drive),
            "satiety_gate": float(satiety_gate),
            "strike_permission": float(strike_permission),
            "capture_reward_scale": float(capture_reward_scale),
            "sampled_action_x": float(raw_action[0]),
            "sampled_action_y": float(raw_action[1]),
            "sampled_action_strike": float(raw_action[2]),
            "mean_action_x": float(policy_step.get("mean_action", np.zeros(3))[0]),
            "mean_action_y": float(policy_step.get("mean_action", np.zeros(3))[1]),
            "mean_action_strike": float(policy_step.get("mean_action", np.zeros(3))[2]),
            "caught_fly": caught_fly,
            "target_distance": target_distance,
            "alignment": alignment,
            "strike_drive": float(self.strike_drive),
            "strike_intent": float(strike_intent),
            "strike_commitment": int(self.strike_commitment),
            "strike_cooldown": float(self.strike_cooldown),
            "visibility": float(visibility),
            "focus_brightness": float(self.last_focus_brightness),
            "focus_motion": float(self.last_focus_motion),
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
