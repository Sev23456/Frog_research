from collections import deque

from Frog_predator_neuro_dual.config import BLUE, CARDINALS, GREEN, MAZE_WIDTH, PASSABLE_CHARS, SPAWNABLE_CHARS
from Frog_predator_neuro_dual.models import Room


def generate_maze_from_layout(layout):
    maze = []
    room_candidates = {"R": [], "S": []}
    spawn_tiles = []
    player_start = None

    for y, row in enumerate(layout):
        if len(row) != MAZE_WIDTH:
            raise ValueError(f"Layout row {y} has width {len(row)} instead of {MAZE_WIDTH}.")
        maze_row = []
        for x, char in enumerate(row):
            maze_row.append(0 if char in PASSABLE_CHARS else 1)
            if char in SPAWNABLE_CHARS:
                spawn_tiles.append((x, y))
            if char == "R":
                room_candidates["R"].append((x, y))
            elif char == "S":
                room_candidates["S"].append((x, y))
            elif char == "P":
                player_start = (x, y)
        maze.append(maze_row)

    rooms = []
    room_id = 1

    def cluster_room_tiles(source_tiles, spawns_food, color):
        nonlocal room_id
        visited = set()
        source_set = set(source_tiles)
        for start in source_tiles:
            if start in visited:
                continue
            queue = deque([start])
            cluster = []
            visited.add(start)
            while queue:
                tile = queue.popleft()
                cluster.append(tile)
                for dx, dy in CARDINALS:
                    nxt = (tile[0] + dx, tile[1] + dy)
                    if nxt in source_set and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            rooms.append(Room(room_id=room_id, tiles=cluster, spawns_food=spawns_food, color=color))
            room_id += 1

    cluster_room_tiles(room_candidates["R"], True, GREEN)
    cluster_room_tiles(room_candidates["S"], False, BLUE)
    return maze, rooms, spawn_tiles, player_start

