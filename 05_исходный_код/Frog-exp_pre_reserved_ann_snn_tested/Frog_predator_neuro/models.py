from collections import deque
from dataclasses import dataclass, field

import pygame

from Frog_predator_neuro.config import CYAN, FOOD_RADIUS, MAX_LOG_LINES, TILE_SIZE
from Frog_predator_neuro.utils import tile_center


@dataclass
class Room:
    room_id: int
    tiles: list
    spawns_food: bool
    color: tuple
    tile_set: set = field(init=False)
    x1: int = field(init=False)
    y1: int = field(init=False)
    x2: int = field(init=False)
    y2: int = field(init=False)

    def __post_init__(self):
        self.tile_set = set(self.tiles)
        xs = [tile[0] for tile in self.tiles]
        ys = [tile[1] for tile in self.tiles]
        self.x1 = min(xs)
        self.x2 = max(xs)
        self.y1 = min(ys)
        self.y2 = max(ys)

    def random_floor_tile(self, occupied=None):
        import random

        occupied = occupied or set()
        candidates = [tile for tile in self.tiles if tile not in occupied]
        if candidates:
            return random.choice(candidates)
        return random.choice(self.tiles)

    def draw_debug(self, screen, font):
        rect = pygame.Rect(
            self.x1 * TILE_SIZE,
            self.y1 * TILE_SIZE,
            (self.x2 - self.x1 + 1) * TILE_SIZE,
            (self.y2 - self.y1 + 1) * TILE_SIZE,
        )
        pygame.draw.rect(screen, self.color, rect, 2)
        label = "FOOD" if self.spawns_food else "SAFE"
        screen.blit(font.render(label, True, CYAN), (self.x1 * TILE_SIZE + 4, self.y1 * TILE_SIZE + 4))


@dataclass
class Food:
    tile: tuple
    room_id: int
    radius: int = FOOD_RADIUS

    @property
    def x(self):
        return tile_center(self.tile)[0]

    @property
    def y(self):
        return tile_center(self.tile)[1]


class WorldLogger:
    def __init__(self, path):
        self.path = path
        self.visible_lines = deque(maxlen=MAX_LOG_LINES)
        self.total_lines = 0
        self.path.write_text("", encoding="utf-8")
        self.log_plain("=== Frog_predator_neuro Biological Simulation ===")

    def log_plain(self, text):
        self.visible_lines.append(text)
        self.total_lines += 1
        print(text)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def log(self, time_s, text):
        self.log_plain(f"[{time_s:06.1f}s] {text}")
