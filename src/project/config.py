# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Constants (grouped by section separator):
#   Screen Capture:
#     SENSOR_WIDTH, SENSOR_HEIGHT, SCREEN_CAPTURE_QUEUE_SIZE, PERIODIC_UPDATE_MS
#
#   Node / Connection Type Enums:
#     NODE_TYPE_SENSORY, NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE
#     CONN_TYPE_EXCITATORY through CONN_TYPE_CAPACITIVE, NUM_CONN_TYPES
#
#   Binary Node State Encoding:
#     64-bit packed int64 layout:
#       [ALIVE:1][NODE_TYPE:2][CONN_TYPE:3][DNA[0..7]:8x5=40][RSVD:15][MODALITY:3]
#     state == 0 means DEAD
#     BINARY_ALIVE_BIT, BINARY_NODE_TYPE_SHIFT, BINARY_CONN_TYPE_SHIFT,
#     BINARY_DNA_BASE_SHIFT, BINARY_DNA_BITS_PER_NEIGHBOR, BINARY_DNA_MAX_VALUE,
#     BINARY_DNA_MASK, BINARY_TYPE_MASK, BINARY_CONN_TYPE_MASK
#
#   DNA Modality Keys:
#     MODALITY_NEUTRAL through MODALITY_AUDIO_RIGHT, MODALITY_SHIFT, MODALITY_MASK
#     DNA micro-instruction encoding: 5-bit slot = [MODE:1][PARAM:4]
#     DNA_MODE_CLASSIC, DNA_MODE_SPECIAL, DNA_SPECIAL_*, DNA_MUTATION_RATE
#     REVERSE_DIRECTION (8-neighbor 2D Moore neighbourhood)
#
#   3D Neighbor Geometry:
#     26-neighbor Moore neighbourhood (all 27 cells in +/-1 cube minus center)
#     NUM_NEIGHBORS_3D, NEIGHBOR_OFFSETS_3D, REVERSE_DIRECTION_3D
#     DNA packing: 26 slots x 5 bits = 130 bits in 3 int64 words
#     DNA_SLOT_WORD, DNA_SLOT_BIT
#
#   Spawn Rate Tiers:
#     SPAWN_TIER_1/2/3_THRESHOLD, SPAWN_TIER_1/2/3/4_LIMIT
#
#   Legacy Energy Constants:
#     ADR-001 corrected: NODE_ENERGY_CAP=255, NODE_DEATH_THRESHOLD=1.0
#     NODE_SPAWN_THRESHOLD, NODE_ENERGY_SPAWN_COST, MAX_NODE_BIRTHS_PER_STEP,
#     DYNAMIC_NODE_ENERGY_DECAY, CONN_ENERGY_TRANSFER_CAPACITY, CONN_MAINTENANCE_COST
#
# Module-level Functions:
#   _compute_reverse_3d() -> tuple
#     Compute reverse direction mapping for 26-neighbor 3D Moore neighbourhood
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [critical] Add cube constants: 6-neighbor offsets, int32 DNA layout (6 slots ×
#   5 bits), uint8 energy bounds; migrate off 26-neighbor / legacy binary layout
#   per cube architecture spec.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Project configuration constants."""

import math

# =============================================================================
# Screen Capture (used by vision.py)
# =============================================================================
SENSOR_WIDTH = 256
SENSOR_HEIGHT = 144
SCREEN_CAPTURE_QUEUE_SIZE = 100
PERIODIC_UPDATE_MS = 200

# =============================================================================
# Node / Connection Type Enums (used by taichi_engine, energy_calculator,
# simulation_validator)
# =============================================================================
NODE_TYPE_SENSORY = 0
NODE_TYPE_DYNAMIC = 1
NODE_TYPE_WORKSPACE = 2

CONN_TYPE_EXCITATORY = 0
CONN_TYPE_INHIBITORY = 1
CONN_TYPE_GATED = 2
CONN_TYPE_PLASTIC = 3
CONN_TYPE_ANTI_PLASTIC = 4
CONN_TYPE_DAMPED = 5
CONN_TYPE_RESONANT = 6
CONN_TYPE_CAPACITIVE = 7

NUM_CONN_TYPES = 8

# =============================================================================
# Binary Node State Encoding (64-bit packed int64 per node)
# Layout: [ALIVE:1][NODE_TYPE:2][CONN_TYPE:3][DNA[0..7]:8x5=40][RSVD:15][MODALITY:3]
# state == 0 means DEAD — all DNA wiped, disconnected from all math.
# =============================================================================
BINARY_ALIVE_BIT = 63
BINARY_NODE_TYPE_SHIFT = 61
BINARY_CONN_TYPE_SHIFT = 58
BINARY_DNA_BASE_SHIFT = 18
BINARY_DNA_BITS_PER_NEIGHBOR = 5
BINARY_DNA_MAX_VALUE = 31
BINARY_DNA_MASK = 0x1F
BINARY_TYPE_MASK = 0x3
BINARY_CONN_TYPE_MASK = 0x7

# =============================================================================
# DNA Modality Keys (bits 2-0 of the reserved range bits 17-0)
# These tag sensory/workspace nodes with a channel identity and are inherited
# by dynamic children during spawn. The transfer kernel is NOT affected.
# =============================================================================
MODALITY_NEUTRAL     = 0
MODALITY_VISUAL      = 1
MODALITY_AUDIO_LEFT  = 2
MODALITY_AUDIO_RIGHT = 3
MODALITY_SHIFT       = 0
MODALITY_MASK        = 0b111

DNA_MODE_CLASSIC = 0
DNA_MODE_SPECIAL = 1
DNA_SPECIAL_THRESHOLD = 0
DNA_SPECIAL_INVERT = 1
DNA_SPECIAL_PULSE = 2
DNA_SPECIAL_ABSORB = 3

DNA_MUTATION_RATE = 0.1

REVERSE_DIRECTION = (1, 0, 3, 2, 7, 6, 5, 4)

# =============================================================================
# 3D Neighbor Geometry (26-neighbor Moore neighbourhood)
# =============================================================================
NUM_NEIGHBORS_3D = 26

NEIGHBOR_OFFSETS_3D = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
)

def _compute_reverse_3d():
    offsets = NEIGHBOR_OFFSETS_3D
    lookup = {o: i for i, o in enumerate(offsets)}
    return tuple(lookup[(-dz, -dy, -dx)] for dz, dy, dx in offsets)

REVERSE_DIRECTION_3D = _compute_reverse_3d()

DNA_SLOT_WORD = tuple(n // 12 for n in range(26))
DNA_SLOT_BIT  = tuple((n % 12) * 5 for n in range(26))

# =============================================================================
# Spawn Rate Tiers (used by taichi_engine.py)
# Controls how many nodes may be born per step based on current population.
# =============================================================================
SPAWN_TIER_1_THRESHOLD = 300_000
SPAWN_TIER_2_THRESHOLD = 600_000
SPAWN_TIER_3_THRESHOLD = 1_000_000
SPAWN_TIER_1_LIMIT = 5_000
SPAWN_TIER_2_LIMIT = 2_000
SPAWN_TIER_3_LIMIT = 1_000
SPAWN_TIER_4_LIMIT = 200

# =============================================================================
# Legacy Energy Constants (used by energy_calculator.py, simulation_validator.py)
# These are from the PyG connection-based system.  The Taichi engine reads its
# thresholds from pyg_config.json -> hybrid section instead.
#
# ADR-001: corrected to match the 0-255 design range used by the Taichi engine.
#   NODE_ENERGY_CAP was 244 (now 255 — matches 8-bit pixel intensity)
#   NODE_DEATH_THRESHOLD was -10.0 (now 1.0 — nodes die at near-zero energy)
# =============================================================================
NODE_ENERGY_CAP = 255
NODE_DEATH_THRESHOLD = 1.0
NODE_SPAWN_THRESHOLD = NODE_ENERGY_CAP * 0.09
NODE_ENERGY_SPAWN_COST = NODE_ENERGY_CAP * 0.08
MAX_NODE_BIRTHS_PER_STEP = 80

DYNAMIC_NODE_ENERGY_DECAY = 0.005
CONN_ENERGY_TRANSFER_CAPACITY = 0.3
CONN_MAINTENANCE_COST = NODE_ENERGY_CAP * 0.0005
