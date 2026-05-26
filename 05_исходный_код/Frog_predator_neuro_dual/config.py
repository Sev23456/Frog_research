"""Configuration for Frog Predator Neuro - Biorealistic Frog with Neural Architecture"""

from pathlib import Path


MAZE_LAYOUT = [
    "##############################",
    "#RRR.....##....##....##R#R#S##",
    "#RRR#.##.##.##.##.##.##R#R#S##",
    "#RRR#....##....##....##R#R#S##",
    "##.##.##..####..####..##.#####",
    "#....#.S#....##....#S#..#...##",
    "#.##.#####.##.##.##.#####.####",
    "#........###.#..S#.#....#...##",
    "#####.##.#SSSSS.####..##.#####",
    "#....#....SSPSS....#....#...##",
    "#.##.##.##SSSSS#.##.##.##.####",
    "#S#......#######..#....#R#R###",
    "#S#.##..####.##.#####.##R#R###",
    "#S#..........##....##....R#R##",
    "#####.##..####..####..##.#####",
    "#....#..S#....##....#S#.....##",
    "#.##.#####.##.##.##.#####.####",
    "#....#....#..S#..S#.#....#..##",
    "#....##....##....##....##...##",
    "##############################",
]

TILE_SIZE = 40
MAZE_WIDTH = len(MAZE_LAYOUT[0])
MAZE_HEIGHT = len(MAZE_LAYOUT)
MAP_WIDTH = MAZE_WIDTH * TILE_SIZE
MAP_HEIGHT = MAZE_HEIGHT * TILE_SIZE
HUD_WIDTH = 430
SCREEN_WIDTH = MAP_WIDTH + HUD_WIDTH
SCREEN_HEIGHT = MAP_HEIGHT

# ============================================================================
# FROG PHYSIOLOGY (adapted from Wolf - large frog variant)
# ============================================================================
AGENT_COUNT = 1  # Single frog (juvenile training, then adult)
AGENT_RADIUS = 9  # Medium frog (0.75x wolf size)
AGENT_MAX_SPEED = 65.0  # Fast jumps but shorter duration (0.74x wolf speed)

# Energy parameters: match frog_lib_ann and frog_lib_snn
AGENT_BODY_ENERGY_MAX = 30.0  # Match frog_lib_ann/snn
AGENT_BODY_ENERGY_START = 30.0  # Start at full energy (same as frog_lib_ann/snn)
AGENT_IDLE_DRAIN = 0.08  # Match frog_lib_snn base drain
AGENT_MOVE_DRAIN = 0.02  # Match frog_lib_ann/snn velocity-based drain

# Frog-specific: no social energy sharing
AGENT_CRITICAL_THRESHOLD = 6.0  # Critical at 20% of max (0.20 * 30.0)
DESPERATION_THRESHOLD = 12.0  # Desperation at 40% of max

# Food (insects) are smaller for frogs
FOOD_RADIUS = 6  # Smaller prey
FOOD_ENERGY = 6.0  # Meal energy (consistent with fly catch reward in simulation)
FOOD_ROOM_CAPACITY = 3  # More food items in same space
FOOD_RESPAWN_MIN = 3.0
FOOD_RESPAWN_MAX = 6.0

# ============================================================================
# SENSORY SYSTEM (frogs optimized for predation)
# ============================================================================
VISION_RADIUS_TILES = 1  # Good vision but biased toward movement detection
VISION_RADIUS_PX = int(VISION_RADIUS_TILES * TILE_SIZE)
PREDATOR_VISUAL_RANGE_PX = int(TILE_SIZE * 4.5)  # Open-field prey sensing is longer than maze tile vision
PREDATOR_DETECTION_THRESHOLD = 0.06  # Ignore prey that does not cross the motion/brightness threshold

# ============================================================================
# NEURAL ARCHITECTURE (reduced from wolf - smaller brain)
# ============================================================================
RETINA_FILTERS_PER_SIDE = 7  # Fewer visual channels (6 from wolf)
TECTUM_COLUMNS = 18  # Smaller optic tectum (18 from wolf 24)
PLACE_CELL_COUNT = 70  # Fewer place cells (70 from wolf 96)
HEAD_DIRECTION_CELLS = 10  # Fewer direction cells (10 from wolf 12)
MEMORY_PROBE_DIRECTIONS = 14
MEMORY_LOOKAHEAD_PX = TILE_SIZE * 1.2  # Shorter lookahead
MEMORY_RECURRENT_K = 4  # Fewer recurrent connections
PLACE_CELL_WIDTH_MIN = TILE_SIZE * 0.95
PLACE_CELL_WIDTH_MAX = TILE_SIZE * 1.95

# ============================================================================
# MEMORY AND LEARNING (shared with wolf - but reduced timescales)
# ============================================================================
PLACE_CELL_DECAY = 0.995
ASSOCIATION_DECAY = 0.999
OCCUPANCY_DECAY = 0.994
NOVELTY_RECOVERY = 0.0018
PLACE_FIELD_LEARNING_RATE = 0.026  # Slightly faster learning (predator)
REPLAY_TRANSITION_DECAY = 0.989
MEMORY_REPLAY_GAIN = 0.32
MEMORY_VECTOR_MOMENTUM = 0.75  # Slightly less momentum (less coordinated)
HEAD_DIRECTION_GAIN = 0.28
OPENNESS_GAIN = 0.20
WALL_REPULSION_GAIN = 1.10

# ============================================================================
# DEVELOPMENT
# ============================================================================
JUVENILE_STEPS = 5000
TRAINING_MIN_FOOD = 2  # More aggressive learning
TRAINING_MIN_ENERGY = 0.14
TRAINING_MIN_VISITED_TILES = 2

# ============================================================================
# DISPLAY AND LOGGING
# ============================================================================
MAX_LOG_LINES = 18
LOG_PATH = Path(__file__).resolve().parent.parent / "Frog_predator_neuro_dual_log.txt"

# ============================================================================
# MAZE PARAMETERS (shared)
# ============================================================================
PASSABLE_CHARS = {".", "R", "S", "P"}
SPAWNABLE_CHARS = {".", "R", "S"}
CARDINALS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
HEAD_DIRECTIONS = 8

# ============================================================================
# COLOR SCHEME
# ============================================================================
BLACK = (9, 12, 16)
WHITE = (236, 240, 241)
WALL = (224, 229, 236)
GRAY = (120, 131, 146)
GREEN = (76, 214, 122)
RED = (255, 91, 91)
BLUE = (90, 174, 255)
YELLOW = (255, 211, 94)
ORANGE = (255, 167, 92)
CYAN = (110, 242, 255)
SAFE_FLOOR = (34, 38, 52)
FOOD_FLOOR = (25, 51, 34)
PASSAGE_FLOOR = (20, 25, 33)
HUD_BG = (15, 19, 28)
HUD_PANEL = (22, 28, 38)
DEAD_COLOR = (90, 96, 110)
DEBUG_MAGENTA = (255, 110, 220)

AGENT_COLORS = [
    (255, 125, 125),
    (90, 174, 255),
    (120, 222, 132),
    (255, 206, 92),
    (198, 143, 255),
    (80, 232, 227),
]

