#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import random
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pymunk

from .agent import FrogANNAgent

HUD_WIDTH = 430
HUD_BG = (15, 19, 28)
HUD_PANEL = (22, 28, 38)
WHITE = (236, 240, 241)
GRAY = (120, 131, 146)
CYAN = (110, 242, 255)
YELLOW = (255, 211, 94)
FROG_RENDER_SCALE = 2.7
FROG_RENDER_RADIUS_MIN = 24
FROG_INNER_SCALE = 0.84
FROG_OUTER_COLOR = (20, 135, 40)
FROG_INNER_COLOR = (40, 185, 60)
FROG_TONGUE_COLOR = (240, 90, 90)


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


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


class ANNFlyCatchingSimulation:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        num_flies: int = 15,
        headless: bool = False,
        training_mode: bool = False,
    ):
        self.width = width
        self.height = height
        self.num_flies = num_flies
        self.headless = headless
        self.training_mode = training_mode

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.dt = 0.01
        self.physics_dt = 0.01
        self.world_boundaries = add_world_boundaries(self.space, width, height)

        self.frog = FrogANNAgent(self.space, position=(width // 2, height // 2), training_mode=training_mode)
        self.flies: List[Fly] = []
        self.spawn_flies(num_flies)

        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width + HUD_WIDTH, height))
            pygame.display.set_caption("ANN Frog - homeostatic actor-critic predator")
            self.clock = pygame.time.Clock()
            self.title_font = pygame.font.Font(None, 24)
            self.font = pygame.font.Font(None, 20)
            self.small_font = pygame.font.Font(None, 16)
        else:
            self.screen = None
            self.clock = None
            self.title_font = None
            self.font = None
            self.small_font = None

        self.step_count = 0
        self.caught_count = 0
        self.energy_history = deque(maxlen=2000)
        self.catch_history = deque(maxlen=2000)
        self.distance_history = deque(maxlen=2000)
        self.alignment_history = deque(maxlen=2000)
        self.speed_history = deque(maxlen=2000)
        self.smoothness_history = deque(maxlen=2000)
        self.controller_history = deque(maxlen=2000)
        self._last_velocity = np.zeros(2, dtype=float)
        self.last_agent_state: Dict[str, Any] = {
            "energy": float(self.frog.energy),
            "caught_flies": 0,
            "alignment": 0.0,
            "target_distance": None,
            "controller_signal": 0.0,
            "learning_reward": 0.0,
            "actor_advantage": 0.0,
            "value_estimate": 0.0,
            "exploration_scale": 0.0,
            "sampled_action_x": 0.0,
            "sampled_action_y": 0.0,
            "sampled_action_strike": 0.0,
            "mean_action_x": 0.0,
            "mean_action_y": 0.0,
            "mean_action_strike": 0.0,
            "strike_drive": 0.0,
            "strike_intent": 0.0,
            "strike_commitment": 0,
            "strike_cooldown": 0.0,
            "visibility": 0.0,
            "focus_brightness": 0.0,
            "focus_motion": 0.0,
            "visual_range": float(self.frog.visual_range),
            "tongue_extended": False,
            "tongue_length": 0.0,
            "reward": 0.0,
            "fatigue": 0.0,
            "hunting_enabled": False,
            "homeostatic_hunger_drive": 0.0,
            "satiety_gate": 1.0,
        }

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

    def _nearest_live_fly_distance(self) -> Optional[float]:
        alive = [fly for fly in self.flies if fly.alive]
        if not alive:
            return None
        frog_pos = np.array(self.frog.body.position, dtype=float)
        return min(float(np.linalg.norm(np.array(fly.body.position, dtype=float) - frog_pos)) for fly in alive)

    def step(self) -> Dict[str, Any]:
        for fly in self.flies:
            fly.update(self.dt, self.width, self.height)

        agent_state = self.frog.update(self.dt, self.flies)
        self.last_agent_state = dict(agent_state)
        self.space.step(self.dt)

        if agent_state["caught_fly"] is not None and getattr(agent_state["caught_fly"], "alive", False):
            agent_state["caught_fly"].remove()
            self.caught_count += 1
            self.catch_history.append(1)
        else:
            self.catch_history.append(0)

        velocity = np.array(agent_state["velocity"], dtype=float)
        speed = float(np.linalg.norm(velocity))
        previous_speed = float(np.linalg.norm(self._last_velocity))
        smoothness = 1.0
        if speed > 1e-6 and previous_speed > 1e-6:
            smoothness = float(np.dot(velocity, self._last_velocity) / (speed * previous_speed))

        self.energy_history.append(float(agent_state["energy"]))
        self.distance_history.append(agent_state["target_distance"] if agent_state["target_distance"] is not None else float("nan"))
        self.alignment_history.append(float(agent_state["alignment"]))
        self.speed_history.append(speed)
        self.smoothness_history.append(smoothness)
        self.controller_history.append(float(agent_state["controller_signal"]))
        self._last_velocity = velocity

        self.respawn_flies()
        self.step_count += 1
        return agent_state

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

    def _neuro_panel_sections(self) -> List[Tuple[str, List[str]]]:
        state = self.last_agent_state or {}
        distance = state.get("target_distance")
        distance_text = f"{distance:.1f}px" if distance is not None else "none"
        return [
            (
                "State",
                [
                    f"Steps {self.step_count} | catches {self.frog.caught_flies}",
                    f"Energy {state.get('energy', self.frog.energy):.2f} | fatigue {state.get('fatigue', 0.0):.2f}",
                    f"Target {distance_text} | align {state.get('alignment', 0.0):.2f}",
                    f"Tongue {'out' if state.get('tongue_extended', False) else 'in'} | len {state.get('tongue_length', 0.0):.1f}",
                ],
            ),
            (
                "Actor-Critic",
                [
                    f"Value {state.get('value_estimate', 0.0):.3f} | adv {state.get('actor_advantage', 0.0):.3f}",
                    f"Learn reward {state.get('learning_reward', 0.0):.3f} | ext reward {state.get('reward', 0.0):.2f}",
                    f"Explore scale {state.get('exploration_scale', 0.0):.2f} | ctrl {state.get('controller_signal', 0.0):.2f}",
                ],
            ),
            (
                "Policy Output",
                [
                    f"Sample ax {state.get('sampled_action_x', 0.0):.2f} ay {state.get('sampled_action_y', 0.0):.2f}",
                    f"Sample strike {state.get('sampled_action_strike', 0.0):.2f}",
                    f"Mean ax {state.get('mean_action_x', 0.0):.2f} ay {state.get('mean_action_y', 0.0):.2f}",
                    f"Mean strike {state.get('mean_action_strike', 0.0):.2f}",
                ],
            ),
            (
                "Sensory Focus",
                [
                    f"Visibility {state.get('visibility', 0.0):.2f} | range {state.get('visual_range', 0.0):.0f}px",
                    f"Brightness {state.get('focus_brightness', 0.0):.2f} | motion {state.get('focus_motion', 0.0):.2f}",
                ],
            ),
            (
                "Homeostasis",
                [
                    f"Hunt gate {'on' if state.get('hunting_enabled', False) else 'off'} | satiety {state.get('satiety_gate', 0.0):.2f}",
                    f"Hunger drive {state.get('homeostatic_hunger_drive', 0.0):.2f}",
                ],
            ),
            (
                "Strike Loop",
                [
                    f"Strike drive {state.get('strike_drive', 0.0):.2f} | intent {state.get('strike_intent', 0.0):.2f}",
                    f"Commitment {state.get('strike_commitment', 0)} | cooldown {state.get('strike_cooldown', 0.0):.2f}s",
                ],
            ),
        ]

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

        title = self.title_font.render("ANN Homeostatic Panel", True, YELLOW)
        self.screen.blit(title, (x, y))
        y += 26

        runtime_lines = [
            f"Flies alive {len([fly for fly in self.flies if fly.alive])}",
            f"Avg speed {np.mean(self.speed_history) if self.speed_history else 0.0:.2f} | smooth {np.mean(self.smoothness_history) if self.smoothness_history else 0.0:.2f}",
        ]
        y = self._draw_panel_section("Runtime", runtime_lines, x, y, max_y)
        for title, lines in self._neuro_panel_sections():
            y = self._draw_panel_section(title, lines, x, y, max_y)
            if y > max_y:
                break

    def draw(self):
        if self.headless or self.screen is None:
            return
        self.screen.fill(HUD_BG)
        pygame.draw.rect(self.screen, (238, 242, 220), (0, 0, self.width, self.height))
        for fly in self.flies:
            fly.draw(self.screen)

        frog_pos = to_pygame_vec(self.frog.body.position)
        outer_radius = max(FROG_RENDER_RADIUS_MIN, int(round(float(self.frog.shape.radius) * FROG_RENDER_SCALE)))
        inner_radius = max(3, int(round(outer_radius * FROG_INNER_SCALE)))
        pygame.draw.circle(self.screen, FROG_OUTER_COLOR, frog_pos, outer_radius)
        pygame.draw.circle(self.screen, FROG_INNER_COLOR, frog_pos, inner_radius)

        if self.frog.tongue_extended and self.frog.tongue_target is not None:
            pygame.draw.line(self.screen, FROG_TONGUE_COLOR, frog_pos, to_pygame_vec(self.frog.tongue_target), 3)

        self._draw_hud()
        pygame.display.flip()

    def run_simulation(self, max_steps: int = 2000):
        print(f"ANN homeostatic simulation start: steps={max_steps}, headless={self.headless}, training={self.training_mode}")
        for _ in range(max_steps):
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
            self.step()
            if not self.headless:
                self.draw()
                self.clock.tick(60)

        stats = self.get_statistics()
        print(
            "ANN homeostatic summary | catches={caught_flies} success={success_rate:.3f} "
            "alignment={avg_alignment:.3f} smoothness={movement_smoothness:.3f}".format(**stats)
        )

    def get_statistics(self) -> Dict[str, Any]:
        valid_distances = [value for value in self.distance_history if not np.isnan(value)]
        return {
            "total_steps": self.step_count,
            "caught_flies": self.frog.caught_flies,
            "success_rate": (sum(self.catch_history) / len(self.catch_history)) if self.catch_history else 0.0,
            "final_energy": float(self.frog.energy),
            "avg_alignment": float(np.mean(self.alignment_history)) if self.alignment_history else 0.0,
            "avg_distance_to_target": float(np.mean(valid_distances)) if valid_distances else float("nan"),
            "avg_speed": float(np.mean(self.speed_history)) if self.speed_history else 0.0,
            "movement_smoothness": float(np.mean(self.smoothness_history)) if self.smoothness_history else 1.0,
            "avg_controller_signal": float(np.mean(self.controller_history)) if self.controller_history else 0.0,
            "architecture_signature": "actor-critic visual predator with satiety-gated strike loop",
        }

    def save_state(self, filename: str = "ann_homeostatic_frog_state.json"):
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(self.get_statistics(), handle, indent=2, ensure_ascii=False)

    def plot_results(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Plotting is unavailable: {exc}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle("ANN homeostatic frog behavior summary")
        axes[0, 0].plot(list(self.energy_history))
        axes[0, 0].set_title("Energy")
        axes[0, 1].plot(list(self.alignment_history))
        axes[0, 1].set_title("Pursuit alignment")
        axes[1, 0].plot(list(self.speed_history))
        axes[1, 0].set_title("Speed")
        axes[1, 1].plot(list(self.controller_history))
        axes[1, 1].set_title("Controller signal")
        for row in axes:
            for axis in row:
                axis.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def close(self):
        if self.screen is not None:
            pygame.quit()
        self.frog.remove()
        for fly in self.flies:
            fly.remove()
