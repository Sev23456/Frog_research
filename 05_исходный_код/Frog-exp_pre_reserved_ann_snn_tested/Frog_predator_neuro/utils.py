import math

from Frog_predator_neuro.config import MAZE_HEIGHT, MAZE_WIDTH, TILE_SIZE


def tile_center(tile):
    tx, ty = tile
    return tx * TILE_SIZE + TILE_SIZE // 2, ty * TILE_SIZE + TILE_SIZE // 2


def pixel_to_tile(x, y):
    return int(x // TILE_SIZE), int(y // TILE_SIZE)


def in_bounds(tile):
    tx, ty = tile
    return 0 <= tx < MAZE_WIDTH and 0 <= ty < MAZE_HEIGHT


def tile_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pixel_distance(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_vector(vec):
    length = math.hypot(vec[0], vec[1])
    if length < 1e-8:
        return (0.0, 0.0)
    return (vec[0] / length, vec[1] / length)


def add_vectors(*vectors):
    x = sum(vec[0] for vec in vectors)
    y = sum(vec[1] for vec in vectors)
    return (x, y)


def scale_vector(vec, scalar):
    return (vec[0] * scalar, vec[1] * scalar)


def format_tile(tile):
    return f"({tile[0]},{tile[1]})"


def check_collision(maze, x, y, radius):
    # Compute the range of tiles that the agent's bounding box could overlap
    left = x - radius
    right = x + radius
    top = y - radius
    bottom = y + radius

    min_grid_x = int(left // TILE_SIZE)
    max_grid_x = int(right // TILE_SIZE)
    min_grid_y = int(top // TILE_SIZE)
    max_grid_y = int(bottom // TILE_SIZE)

    # If any tile in that range is outside the maze, treat as collision
    if min_grid_x < 0 or max_grid_x >= MAZE_WIDTH or min_grid_y < 0 or max_grid_y >= MAZE_HEIGHT:
        return True

    # For each potentially overlapping tile, perform circle-vs-rectangle test
    for gy in range(min_grid_y, max_grid_y + 1):
        for gx in range(min_grid_x, max_grid_x + 1):
            if maze[gy][gx] == 1:
                rect_left = gx * TILE_SIZE
                rect_top = gy * TILE_SIZE
                rect_right = rect_left + TILE_SIZE
                rect_bottom = rect_top + TILE_SIZE

                # Find nearest point on the rectangle to the circle center
                nearest_x = clamp(x, rect_left, rect_right)
                nearest_y = clamp(y, rect_top, rect_bottom)

                dx = x - nearest_x
                dy = y - nearest_y
                if dx * dx + dy * dy < radius * radius:
                    return True
    return False
