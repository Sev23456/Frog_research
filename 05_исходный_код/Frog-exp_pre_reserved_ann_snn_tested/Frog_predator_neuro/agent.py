import math
import random

import pygame

from Frog_predator_neuro.brain import BioFrogBrain
from Frog_predator_neuro.config import (
    AGENT_BODY_ENERGY_MAX,
    AGENT_BODY_ENERGY_START,
    AGENT_CRITICAL_THRESHOLD,
    AGENT_IDLE_DRAIN,
    AGENT_MAX_SPEED,
    AGENT_MOVE_DRAIN,
    AGENT_RADIUS,
    CARDINALS,
    CYAN,
    DEBUG_MAGENTA,
    DEAD_COLOR,
    FOOD_ENERGY,
    GRAY,
    GREEN,
    HUD_BG,
    ORANGE,
    RED,
    TILE_SIZE,
    VISION_RADIUS_PX,
    VISION_RADIUS_TILES,
    WALL_REPULSION_GAIN,
    WHITE,
    YELLOW,
)
from Frog_predator_neuro.utils import clamp, format_tile, in_bounds, pixel_distance, pixel_to_tile, tile_center


class BioFrogAgent:
    def __init__(self, agent_id, spawn_tile, color, brain=None, rng=None):
        self.agent_id = agent_id
        self.color = color
        self.radius = AGENT_RADIUS
        self.max_speed = AGENT_MAX_SPEED
        self.max_energy = AGENT_BODY_ENERGY_MAX
        self.rng = rng or random.Random()
        self.brain = brain if brain is not None else BioFrogBrain(juvenile_mode=True, rng=self.rng)
        # Start with full energy plus random variation (+/- 1.5)
        baseline_energy = AGENT_BODY_ENERGY_START + self.rng.uniform(-1.5, 1.5)
        self.body_energy = clamp(baseline_energy, AGENT_CRITICAL_THRESHOLD + 1.0, self.max_energy)
        self.x, self.y = tile_center(spawn_tile)
        self.visited_tiles = {spawn_tile}
        self.heading_angle = self.rng.uniform(0.0, 2.0 * math.pi)
        self.motion_trace = (math.cos(self.heading_angle), math.sin(self.heading_angle))

        self.alive = True
        self.dead_reason = ""
        self.food_collected = 0
        self.total_distance = 0.0
        self.pending_reward = 0.0
        self.last_reward_time = -999.0
        self.last_step_distance = 0.0
        self.stall_time = 0.0
        self.bubble_text = ""
        self.bubble_until = 0.0
        self.state = "juvenile" if self.brain.is_juvenile else "adult"
        self.state_detail = "orienting"
        self.last_perception = None
        self.last_output = {
            "velocity": (0.0, 0.0),
            "memory_vector": (0.0, 0.0),
            "dopamine": 0.0,
            "serotonin": 0.0,
            "acetylcholine": 0.0,
            "fatigue": 0.0,
            "glucose": 1.0,
            "neural_activity": 0.0,
            "replay_gate": 0.0,
            "arousal": 0.0,
            "curiosity": 0.0,
            "frustration": 0.0,
            "reward_confidence": 0.0,
            "restlessness": 0.0,
            "energy_surplus": 0.0,
            "desperation": 0.0,
            "juvenile_progress": 0.0,
            "is_juvenile": self.brain.is_juvenile,
            "action_vigor": 0.3,
            "search_burst": 0.0,
            "policy_confidence": 0.0,
            "policy_conflict": 0.0,
            "habit_pressure": 0.0,
            "reward": 0.0,
            "memory_score": 0.0,
        }

    @property
    def tile(self):
        return pixel_to_tile(self.x, self.y)

    @property
    def energy_ratio(self):
        return self.body_energy / self.max_energy if self.max_energy > 0 else 0.0

    def energy_surplus_drive(self):
        return clamp((self.energy_ratio - 0.72) / 0.20, 0.0, 1.0)

    def desperation_drive(self):
        return clamp((0.60 - self.energy_ratio) / 0.30, 0.0, 1.0)

    def set_bubble(self, world_time, text, duration=1.6):
        self.bubble_text = text
        self.bubble_until = world_time + duration

    def build_open_vectors(self, current_tile, visible_passable_tiles):
        vectors = []
        passable = set(visible_passable_tiles)
        for dx, dy in CARDINALS:
            nxt = (current_tile[0] + dx, current_tile[1] + dy)
            if nxt not in passable:
                continue
            weight = 0.62
            second = (current_tile[0] + dx * 2, current_tile[1] + dy * 2)
            if second in passable:
                weight += 0.14
            weight = max(0.08, weight)
            vectors.append((float(dx), float(dy), weight))

        diagonals = [
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        for dx, dy in diagonals:
            nxt = (current_tile[0] + dx, current_tile[1] + dy)
            if nxt not in passable:
                continue
            if (current_tile[0] + dx, current_tile[1]) not in passable:
                continue
            if (current_tile[0], current_tile[1] + dy) not in passable:
                continue
            vectors.append((dx / math.sqrt(2.0), dy / math.sqrt(2.0), 0.18))
        return vectors

    def wall_vectors(self, visible_walls):
        vectors = []
        for tile in visible_walls:
            wall_x, wall_y = tile_center(tile)
            delta_x = self.x - wall_x
            delta_y = self.y - wall_y
            distance = math.hypot(delta_x, delta_y)
            if distance <= 1e-6:
                continue
            weight = clamp(1.0 - distance / (VISION_RADIUS_PX * 0.9), 0.0, 1.0) * WALL_REPULSION_GAIN
            if weight > 0.0:
                vectors.append((delta_x / distance, delta_y / distance, weight))
        return vectors

    def direction_affordance(self, direction, open_vectors):
        if not open_vectors:
            return 0.0
        direction_x, direction_y = direction
        magnitude = math.hypot(direction_x, direction_y)
        if magnitude <= 1e-6:
            return 0.0
        unit_x = direction_x / magnitude
        unit_y = direction_y / magnitude
        best = 0.0
        for open_x, open_y, weight in open_vectors:
            dot_product = max(0.0, unit_x * open_x + unit_y * open_y)
            best = max(best, dot_product * clamp(weight, 0.0, 1.0))
        return clamp(best, 0.0, 1.0)

    def project_drive_to_open_space(self, drive_x, drive_y, open_vectors, food_visible=False):
        if not open_vectors:
            return drive_x, drive_y
        magnitude = math.hypot(drive_x, drive_y)
        if magnitude <= 1e-6:
            return 0.0, 0.0

        unit_x = drive_x / magnitude
        unit_y = drive_y / magnitude
        projected_x = 0.0
        projected_y = 0.0
        for open_x, open_y, weight in open_vectors:
            dot_product = max(0.0, unit_x * open_x + unit_y * open_y)
            projected_x += open_x * weight * dot_product
            projected_y += open_y * weight * dot_product

        projected_magnitude = math.hypot(projected_x, projected_y)
        if projected_magnitude <= 1e-6:
            best_open = max(open_vectors, key=lambda item: item[2])
            return best_open[0] * magnitude, best_open[1] * magnitude
        raw_affordance = self.direction_affordance((drive_x, drive_y), open_vectors)
        base_open_bias = 0.82 - raw_affordance * 0.40
        open_bias = base_open_bias * (0.50 if food_visible else 1.0)
        self_bias = 1.0 - open_bias
        blended_x = projected_x / projected_magnitude * open_bias + unit_x * self_bias
        blended_y = projected_y / projected_magnitude * open_bias + unit_y * self_bias
        blended_magnitude = math.hypot(blended_x, blended_y)
        if blended_magnitude <= 1e-6:
            return projected_x, projected_y
        scale = magnitude / blended_magnitude
        return blended_x * scale, blended_y * scale

    def lane_centering_adjustment(self, preferred_vector, open_vectors=None):
        current_center_x, current_center_y = tile_center(self.tile)
        offset_x = clamp((current_center_x - self.x) / TILE_SIZE, -1.0, 1.0)
        offset_y = clamp((current_center_y - self.y) / TILE_SIZE, -1.0, 1.0)
        has_horizontal = False
        has_vertical = False
        if open_vectors:
            for open_x, open_y, _ in open_vectors:
                if abs(open_x) > 0.7:
                    has_horizontal = True
                if abs(open_y) > 0.7:
                    has_vertical = True

        if has_vertical and not has_horizontal:
            return (offset_x * 0.75, 0.0)
        if has_horizontal and not has_vertical:
            return (0.0, offset_y * 0.75)
        if abs(preferred_vector[0]) >= abs(preferred_vector[1]):
            return (0.0, offset_y * 0.55)
        return (offset_x * 0.55, 0.0)

    def collect_perception(self, world):
        current_tile = self.tile
        visible_passable_tiles = set()
        visible_walls = set()

        for dx in range(-VISION_RADIUS_TILES, VISION_RADIUS_TILES + 1):
            for dy in range(-VISION_RADIUS_TILES, VISION_RADIUS_TILES + 1):
                if max(abs(dx), abs(dy)) > VISION_RADIUS_TILES:
                    continue
                tile = (current_tile[0] + dx, current_tile[1] + dy)
                if not in_bounds(tile):
                    continue
                if world.maze[tile[1]][tile[0]] == 0:
                    visible_passable_tiles.add(tile)
                else:
                    visible_walls.add(tile)

        visible_food_tiles = {
            food.tile
            for food in world.foods
            if max(abs(food.tile[0] - current_tile[0]), abs(food.tile[1] - current_tile[1])) <= VISION_RADIUS_TILES
        }
        visible_unvisited_tiles = {
            tile for tile in visible_passable_tiles if tile not in self.visited_tiles and tile != current_tile
        }

        recent_progress = clamp(self.last_step_distance / max(1e-6, self.max_speed / 60.0), 0.0, 1.0)
        return {
            "position": (self.x, self.y),
            "current_tile": current_tile,
            "heading_angle": self.heading_angle,
            "visible_passable_tiles": visible_passable_tiles,
            "visible_walls": visible_walls,
            "visible_food_tiles": visible_food_tiles,
            "visible_unvisited_tiles": visible_unvisited_tiles,
            "open_vectors": self.build_open_vectors(current_tile, visible_passable_tiles),
            "wall_vectors": self.wall_vectors(visible_walls),
            "recent_progress": recent_progress,
            "visited_count": len(self.visited_tiles),
            "food_collected": self.food_collected,
            "stall_time": self.stall_time,
            "time_since_reward": max(0.0, world.time_s - self.last_reward_time),
        }

    def update(self, world, dt):
        if not self.alive:
            return

        if world.time_s > self.bubble_until:
            self.bubble_text = ""

        old_tile = self.tile
        perception = self.collect_perception(world)
        reward = self.pending_reward
        self.pending_reward = 0.0
        self.last_perception = perception
        self.last_output = self.brain.update(perception, self.body_energy, reward, world.time_s)
        self.apply_motion(world, self.last_output["velocity"], perception, dt)
        self.update_energy(dt, self.last_output["fatigue"])
        self.update_state(perception)
        self.visited_tiles.add(self.tile)
        tile_changed = self.tile != old_tile
        progress_threshold = 0.10
        if self.last_output.get("energy_surplus", 0.0) > 0.15 or self.last_output.get("desperation", 0.0) > 0.15:
            progress_threshold = 0.08
        if (not tile_changed and self.last_step_distance < progress_threshold) and not perception["visible_food_tiles"]:
            self.stall_time += dt
        else:
            movement_recovery = clamp(self.last_step_distance / 0.45, 0.0, 1.8)
            decay_rate = (
                0.9
                + movement_recovery
                + self.last_output.get("search_burst", 0.0) * 0.28
                + self.last_output.get("energy_surplus", 0.0) * 0.35
                + self.last_output.get("desperation", 0.0) * 0.25
            )
            self.stall_time = max(0.0, self.stall_time - dt * decay_rate)

        if self.body_energy <= 0.0:
            self.die(world, "starved")

    def apply_motion(self, world, velocity, perception, dt):
        wall_vectors = perception["wall_vectors"]
        drive_x, drive_y = velocity
        energy_surplus = self.last_output.get("energy_surplus", self.energy_surplus_drive())
        desperation = self.last_output.get("desperation", self.desperation_drive())
        action_vigor = self.last_output.get("action_vigor", 0.3)
        search_burst = self.last_output.get("search_burst", 0.0)
        policy_confidence = self.last_output.get("policy_confidence", 0.0)
        memory_vector = self.last_output.get("memory_vector", (0.0, 0.0))
        reorientation_drive = self.last_output.get("reorientation_drive", 0.0)
        loop_pressure = self.last_output.get("loop_pressure", 0.0)
        food_visible = bool(perception["visible_food_tiles"])
        # Track diagonal corner pressure for enhanced escape
        corner_pressure_x = 0.0
        corner_pressure_y = 0.0
        for wall_x, wall_y, weight in wall_vectors:
            # Adaptive wall repulsion: stronger when very close to walls (high weight = close)
            # Base 0.30, increases to 0.50 when weight > 0.7 (very close)
            adaptive_gain = 0.30 + max(0.0, (weight - 0.5) * 0.4)
            drive_x += wall_x * weight * adaptive_gain
            drive_y += wall_y * weight * adaptive_gain
            # Track diagonal corner pressure (when both components ≠ 0)
            if abs(wall_x) > 0.5 and abs(wall_y) > 0.5:
                corner_pressure_x += wall_x * weight * 0.12
                corner_pressure_y += wall_y * weight * 0.12
        
        # Special escape response for diagonal corners
        if (abs(corner_pressure_x) > 0.15 and abs(corner_pressure_y) > 0.15) and self.stall_time > 0.35:
            # Strong diagonal repulsion boost
            escape_magnitude = math.hypot(corner_pressure_x, corner_pressure_y)
            if escape_magnitude > 1e-6:
                escape_direction = (corner_pressure_x / escape_magnitude, corner_pressure_y / escape_magnitude)
                escape_strength = min(0.8, 0.30 + self.stall_time * 0.20)
                drive_x += escape_direction[0] * escape_strength
                drive_y += escape_direction[1] * escape_strength

        preferred_vector = memory_vector if memory_vector != (0.0, 0.0) else (drive_x, drive_y)
        center_dx, center_dy = self.lane_centering_adjustment(preferred_vector, perception["open_vectors"])
        centering_gain = 0.42 + action_vigor * 0.12 + search_burst * 0.16
        if food_visible:
            centering_gain += 0.14
        drive_x += center_dx * centering_gain
        drive_y += center_dy * centering_gain
        has_horizontal = any(abs(open_x) > 0.7 for open_x, _, _ in perception["open_vectors"])
        has_vertical = any(abs(open_y) > 0.7 for _, open_y, _ in perception["open_vectors"])
        if self.stall_time > 0.45:
            reflex_gain = 0.36 + min(0.42, self.stall_time * 0.10)
            if has_horizontal and not has_vertical:
                drive_y += center_dy * reflex_gain
                drive_x *= max(0.42, 0.92 - abs(center_dy) * 0.42)
            elif has_vertical and not has_horizontal:
                drive_x += center_dx * reflex_gain
                drive_y *= max(0.42, 0.92 - abs(center_dx) * 0.42)

        trace_weight = 0.22 + policy_confidence * 0.10 - search_burst * 0.12
        if food_visible:
            motion_magnitude = math.hypot(self.motion_trace[0], self.motion_trace[1])
            if motion_magnitude > 0.3:
                trace_weight = 0.40
            else:
                trace_weight = 0.08
        drive_x = trace_weight * self.motion_trace[0] + (1.0 - trace_weight) * drive_x
        drive_y = trace_weight * self.motion_trace[1] + (1.0 - trace_weight) * drive_y
        drive_affordance = self.direction_affordance((drive_x, drive_y), perception["open_vectors"])
        if food_visible and drive_affordance < 0.30:
            pass
        else:
            drive_x, drive_y = self.project_drive_to_open_space(drive_x, drive_y, perception["open_vectors"], food_visible=food_visible)
        if perception["open_vectors"] and drive_affordance < 0.18:
            guide_x, guide_y = memory_vector if memory_vector != (0.0, 0.0) else self.motion_trace
            strongest_open = max(
                perception["open_vectors"],
                key=lambda item: item[2] * (0.26 + max(0.0, item[0] * guide_x + item[1] * guide_y)),
            )
            reflex_gain = 0.28 + action_vigor * 0.12 + search_burst * 0.10
            drive_x += strongest_open[0] * reflex_gain
            drive_y += strongest_open[1] * reflex_gain
            drive_x, drive_y = self.project_drive_to_open_space(drive_x, drive_y, perception["open_vectors"], food_visible=food_visible)
        magnitude = math.hypot(drive_x, drive_y)
        if not food_visible and magnitude > 1e-6:
            locomotor_floor = 0.10 + action_vigor * 0.18 + search_burst * 0.22
            if magnitude < locomotor_floor:
                drive_x = drive_x / magnitude * locomotor_floor
                drive_y = drive_y / magnitude * locomotor_floor
                magnitude = locomotor_floor
        if magnitude <= 1e-6:
            self.last_step_distance = 0.0
            self.motion_trace = (self.motion_trace[0] * 0.85, self.motion_trace[1] * 0.85)
            return

        speed_cap = 0.56 + action_vigor * 0.75 + search_burst * 0.20
        if food_visible:
            speed_cap += 0.55
        speed_scale = clamp(magnitude, 0.0, speed_cap)
        move_x = drive_x / magnitude * self.max_speed * speed_scale * dt
        move_y = drive_y / magnitude * self.max_speed * speed_scale * dt

        old_x, old_y = self.x, self.y
        world.try_offset(self, move_x, move_y)
        self.last_step_distance = pixel_distance(old_x, old_y, self.x, self.y)
        micro_stall_threshold = 0.05 + search_burst * 0.06 + energy_surplus * 0.03 + desperation * 0.03
        if self.last_step_distance < micro_stall_threshold and self.stall_time > 0.8:
            center_x, center_y = tile_center(self.tile)
            recenter_dx = center_x - self.x
            recenter_dy = center_y - self.y
            recenter_norm = math.hypot(recenter_dx, recenter_dy)
            if recenter_norm > 1e-6:
                recenter_speed = self.max_speed * (
                    0.10 + reorientation_drive * 0.28 + min(0.24, self.stall_time * 0.05)
                ) * dt
                world.try_offset(
                    self,
                    recenter_dx / recenter_norm * recenter_speed,
                    recenter_dy / recenter_norm * recenter_speed,
                )
                self.last_step_distance = pixel_distance(old_x, old_y, self.x, self.y)
        if self.last_step_distance < micro_stall_threshold and perception["open_vectors"] and (search_burst > 0.16 or reorientation_drive > 0.28):
            guide_x, guide_y = memory_vector if memory_vector != (0.0, 0.0) else self.motion_trace
            best_open = max(
                perception["open_vectors"],
                key=lambda item: (
                    item[2] * (
                        0.24
                        + max(0.0, item[0] * guide_x + item[1] * guide_y) * (0.24 + search_burst * 0.18)
                        + max(0.0, -(item[0] * self.motion_trace[0] + item[1] * self.motion_trace[1])) * (reorientation_drive * 0.55)
                        + (1.0 - abs(item[0] * self.motion_trace[0] + item[1] * self.motion_trace[1])) * (0.12 + loop_pressure * 0.16)
                    )
                ),
            )
            fallback_speed = self.max_speed * (
                0.12
                + search_burst * 0.34
                + action_vigor * 0.12
                + reorientation_drive * 0.32
                + min(0.30, self.stall_time * 0.08)
            ) * dt
            world.try_offset(self, best_open[0] * fallback_speed, best_open[1] * fallback_speed)
            self.last_step_distance = pixel_distance(old_x, old_y, self.x, self.y)
            if self.last_step_distance < micro_stall_threshold and self.stall_time > 0.8:
                self.motion_trace = (best_open[0], best_open[1])
                lateral_open = max(
                    perception["open_vectors"],
                    key=lambda item: item[2] * (1.0 - abs(item[0] * best_open[0] + item[1] * best_open[1])),
                )
                escape_speed = self.max_speed * (0.08 + reorientation_drive * 0.26 + min(0.22, self.stall_time * 0.05)) * dt
                world.try_offset(self, lateral_open[0] * escape_speed, lateral_open[1] * escape_speed)
                self.last_step_distance = pixel_distance(old_x, old_y, self.x, self.y)
        self.total_distance += self.last_step_distance

        if self.last_step_distance > 0.3:
            self.heading_angle = math.atan2(self.y - old_y, self.x - old_x)
            step_norm = max(1e-6, math.hypot(self.x - old_x, self.y - old_y))
            self.motion_trace = ((self.x - old_x) / step_norm, (self.y - old_y) / step_norm)
        else:
            stall_factor = 0.85 if self.stall_time > 0.5 else 0.92
            self.motion_trace = (self.motion_trace[0] * stall_factor, self.motion_trace[1] * stall_factor)

    def update_energy(self, dt, fatigue):
        expected_step = max(1e-6, self.max_speed * dt)
        movement_ratio = clamp(self.last_step_distance / expected_step, 0.0, 1.0)
        metabolic_efficiency = max(0.45, self.brain.profile.metabolism_efficiency)
        drain = (AGENT_IDLE_DRAIN + AGENT_MOVE_DRAIN * movement_ratio) / metabolic_efficiency
        drain *= 1.0 + 0.35 * fatigue
        self.body_energy = clamp(self.body_energy - drain * dt, 0.0, self.max_energy)

    def nearest_tile(self, tiles):
        return min(tiles, key=lambda tile: pixel_distance(self.x, self.y, *tile_center(tile)))

    def focus_tile_from_vector(self, vector, reach_tiles=1.6):
        if vector == (0.0, 0.0):
            return self.tile
        target_x = self.x + vector[0] * TILE_SIZE * reach_tiles
        target_y = self.y + vector[1] * TILE_SIZE * reach_tiles
        target_tile = pixel_to_tile(target_x, target_y)
        if not in_bounds(target_tile):
            return self.tile
        return target_tile

    def update_state(self, perception):
        memory_score = self.last_output.get("memory_score", 0.0)
        memory_vector = self.last_output.get("memory_vector", (0.0, 0.0))
        if not self.alive:
            self.state = "dead"
            self.state_detail = self.dead_reason
            return
        if self.last_output.get("is_juvenile"):
            self.state = "juvenile"
            self.state_detail = f"maturing {self.last_output.get('juvenile_progress', 0.0):.0%}"
        elif self.body_energy <= AGENT_CRITICAL_THRESHOLD:
            self.state = "distressed"
            self.state_detail = "energy collapse"
        elif perception["visible_food_tiles"]:
            target = self.nearest_tile(perception["visible_food_tiles"])
            self.state = "foraging"
            self.state_detail = f"food {format_tile(target)}"
        elif self.last_output.get("reorientation_drive", 0.0) > 0.38:
            self.state = "reorienting"
            self.state_detail = "breaking local loop"
        elif memory_score > 0.30 and memory_vector != (0.0, 0.0):
            self.state = "navigating"
            self.state_detail = f"field {format_tile(self.focus_tile_from_vector(memory_vector))}"
        else:
            self.state = "scouting"
            self.state_detail = "exploring"

    def eat(self, world, food):
        self.food_collected += 1
        self.body_energy = clamp(self.body_energy + FOOD_ENERGY, 0.0, self.max_energy)
        self.pending_reward = clamp(self.pending_reward + FOOD_ENERGY * 2.0, 0.0, 1.0)
        self.last_reward_time = world.time_s
        self.set_bubble(world.time_s, "feed", duration=1.2)
        world.logger.log(
            world.time_s,
            f"A{self.agent_id} fed at {format_tile(food.tile)} | energy={self.body_energy:.2f} food={self.food_collected}.",
        )

    def die(self, world, reason):
        if not self.alive:
            return
        self.alive = False
        self.dead_reason = reason
        self.state = "dead"
        self.state_detail = reason
        self.set_bubble(world.time_s, "silent", duration=3.0)
        world.logger.log(
            world.time_s,
            f"A{self.agent_id} died at {format_tile(self.tile)} | reason={reason} food={self.food_collected} place_cells={len(self.brain.spatial_memory.place_cells)}.",
        )

    def compact_status(self):
        return f"A{self.agent_id} E={self.body_energy:.2f} F={self.food_collected:02d} {self.state}"

    def biometrics_lines(self):
        lines = [
            f"A{self.agent_id} {self.state}",
            f"Energy: {self.body_energy:.2f}",
            f"Food: {self.food_collected}",
            f"Dopamine: {self.last_output.get('dopamine', 0.0):.2f}",
            f"Serotonin: {self.last_output.get('serotonin', 0.0):.2f}",
            f"ACh: {self.last_output.get('acetylcholine', 0.0):.2f}",
            f"Fatigue: {self.last_output.get('fatigue', 0.0):.2f}",
            f"Arousal: {self.last_output.get('arousal', 0.0):.2f}",
            f"Curiosity: {self.last_output.get('curiosity', 0.0):.2f}",
            f"Frustration: {self.last_output.get('frustration', 0.0):.2f}",
            f"Reward conf: {self.last_output.get('reward_confidence', 0.0):.2f}",
            f"Restless: {self.last_output.get('restlessness', 0.0):.2f}",
            f"Energy surplus: {self.last_output.get('energy_surplus', 0.0):.2f}",
            f"Desperation: {self.last_output.get('desperation', 0.0):.2f}",
            f"Visited: {len(self.visited_tiles)}",
            f"Glucose: {self.last_output.get('glucose', 0.0):.2f}",
            f"Activity: {self.last_output.get('neural_activity', 0.0):.2f}",
            f"Replay: {self.last_output.get('replay_gate', 0.0):.2f}",
            f"Juvenile: {self.last_output.get('juvenile_progress', 0.0):.0%}",
            f"Place cells: {len(self.brain.spatial_memory.place_cells)}",
        ]
        vector = self.last_output.get("memory_vector", (0.0, 0.0))
        if vector != (0.0, 0.0):
            lines.append(f"Field: {format_tile(self.focus_tile_from_vector(vector))}")
        return lines

    def draw(self, screen, small_font, tiny_font, debug=False, selected=False):
        draw_x = int(self.x)
        draw_y = int(self.y)
        if self.alive:
            pygame.draw.circle(screen, self.color, (draw_x, draw_y), self.radius)
            outline = YELLOW if selected else WHITE
            pygame.draw.circle(screen, outline, (draw_x, draw_y), self.radius, 2)
        else:
            pygame.draw.circle(screen, DEAD_COLOR, (draw_x, draw_y), self.radius)
            pygame.draw.line(screen, WHITE, (draw_x - 7, draw_y - 7), (draw_x + 7, draw_y + 7), 2)
            pygame.draw.line(screen, WHITE, (draw_x + 7, draw_y - 7), (draw_x - 7, draw_y + 7), 2)

        label = tiny_font.render(str(self.agent_id), True, (0, 0, 0) if self.alive else WHITE)
        screen.blit(label, label.get_rect(center=(draw_x, draw_y)))

        bar_width = 30
        bar_height = 5
        bar_x = draw_x - bar_width // 2
        bar_y = draw_y - self.radius - 12
        pygame.draw.rect(screen, GRAY, (bar_x, bar_y, bar_width, bar_height), border_radius=2)
        fill_width = int(bar_width * self.energy_ratio)
        # Energy level thresholds: green (>50%), orange (20-50%), red (<20%)
        if self.body_energy > 15.0:  # >50%
            fill_color = GREEN
        elif self.body_energy > AGENT_CRITICAL_THRESHOLD:  # >20%
            fill_color = ORANGE
        else:  # ≤20%
            fill_color = RED
        pygame.draw.rect(screen, fill_color, (bar_x, bar_y, fill_width, bar_height), border_radius=2)
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 1, border_radius=2)

        if self.bubble_text:
            bubble_color = CYAN if self.alive else GRAY
            text = tiny_font.render(self.bubble_text, True, bubble_color)
            bubble_rect = text.get_rect(center=(draw_x, draw_y - self.radius - 24))
            bubble_bg = bubble_rect.inflate(10, 4)
            pygame.draw.rect(screen, HUD_BG, bubble_bg, border_radius=6)
            pygame.draw.rect(screen, bubble_color, bubble_bg, 1, border_radius=6)
            screen.blit(text, bubble_rect)

        if debug and self.alive:
            pygame.draw.circle(screen, DEBUG_MAGENTA, (draw_x, draw_y), int(VISION_RADIUS_PX), 1)
            vector = self.last_output.get("memory_vector", (0.0, 0.0))
            if vector != (0.0, 0.0):
                end_x = int(self.x + vector[0] * TILE_SIZE * 1.8)
                end_y = int(self.y + vector[1] * TILE_SIZE * 1.8)
                pygame.draw.line(screen, DEBUG_MAGENTA, (draw_x, draw_y), (end_x, end_y), 2)
            heading_x = int(self.x + math.cos(self.heading_angle) * TILE_SIZE * 0.8)
            heading_y = int(self.y + math.sin(self.heading_angle) * TILE_SIZE * 0.8)
            pygame.draw.line(screen, CYAN, (draw_x, draw_y), (heading_x, heading_y), 2)
            debug_text = tiny_font.render(self.state_detail[:18], True, WHITE)
            screen.blit(debug_text, (draw_x - 18, draw_y + self.radius + 4))
