"""
Project configuration constants.

This module contains ONLY the compile-time constants that are imported by other
modules.  All runtime-configurable values live in ``pyg_config.json`` and are
managed by ``utils.config_manager.ConfigManager``.
"""
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

NUM_CONN_TYPES = 8  # 3 bits required

# =============================================================================
# Binary Node State Encoding (64-bit packed int64 per node)
# Layout: [ALIVE:1][NODE_TYPE:2][CONN_TYPE:3][DNA[0..7]:8×5=40][RSVD:15][MODALITY:3]
# state == 0 means DEAD — all DNA wiped, disconnected from all math.
# =============================================================================
BINARY_ALIVE_BIT = 63
BINARY_NODE_TYPE_SHIFT = 61
BINARY_CONN_TYPE_SHIFT = 58           # 3 bits at positions 60-58
BINARY_DNA_BASE_SHIFT = 18            # shifted down by 1 (was 19)
BINARY_DNA_BITS_PER_NEIGHBOR = 5
BINARY_DNA_MAX_VALUE = 31             # 2^5 - 1
BINARY_DNA_MASK = 0x1F                # 5 bits
BINARY_TYPE_MASK = 0x3                # 2 bits (node type)
BINARY_CONN_TYPE_MASK = 0x7           # 3 bits (connection type)

# =============================================================================
# DNA Modality Keys (bits 2–0 of the reserved range bits 17–0)
# These tag sensory/workspace nodes with a channel identity and are inherited
# by dynamic children during spawn. The transfer kernel is NOT affected.
# =============================================================================
MODALITY_NEUTRAL     = 0   # dynamic nodes / unassigned
MODALITY_VISUAL      = 1   # desktop sensory input / visual workspace output
MODALITY_AUDIO_LEFT  = 2   # left audio channel sensory / workspace
MODALITY_AUDIO_RIGHT = 3   # right audio channel sensory / workspace
MODALITY_SHIFT       = 0   # bit position within the 64-bit node state
MODALITY_MASK        = 0b111  # 3 bits → supports 8 modalities

# DNA micro-instruction encoding: each 5-bit slot = [MODE:1][PARAM:4]
DNA_MODE_CLASSIC = 0                  # MODE bit = 0: param/15 = probability
DNA_MODE_SPECIAL = 1                  # MODE bit = 1: [SPECIAL:2][SPARAM:2]
DNA_SPECIAL_THRESHOLD = 0
DNA_SPECIAL_INVERT = 1
DNA_SPECIAL_PULSE = 2
DNA_SPECIAL_ABSORB = 3

# Mutation rate for DNA heredity (probability per slot per spawn)
DNA_MUTATION_RATE = 0.1

# Reverse direction mapping for lock-and-key (Moore neighborhood)
# Direction indices: 0=up, 1=down, 2=left, 3=right, 4=UL, 5=UR, 6=DL, 7=DR
REVERSE_DIRECTION = (1, 0, 3, 2, 7, 6, 5, 4)

# =============================================================================
# 3D Neighbor Geometry (26-neighbor Moore neighbourhood)
# =============================================================================
NUM_NEIGHBORS_3D = 26   # all 27 cells in ±1 cube minus center

# 3D neighbor offsets, ordered by (dz, dy, dx) ∈ {-1,0,1}³ \ {(0,0,0)}.
# n = (dz+1)*9 + (dy+1)*3 + (dx+1) for the raw 3×3×3 index.
# Center is at raw index 13, so slots 0-12 → raw 0-12, slots 13-25 → raw 14-26.
NEIGHBOR_OFFSETS_3D = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
)   # tuple of (dz, dy, dx) tuples, length 26

# Reverse direction: slot n points toward (dz, dy, dx), reverse points toward (-dz, -dy, -dx).
def _compute_reverse_3d():
    offsets = NEIGHBOR_OFFSETS_3D
    lookup = {o: i for i, o in enumerate(offsets)}
    return tuple(lookup[(-dz, -dy, -dx)] for dz, dy, dx in offsets)

REVERSE_DIRECTION_3D = _compute_reverse_3d()

# DNA packing: 26 slots × 5 bits = 130 bits → packed into 3 int64 words (192 bits).
# Word 0 holds slots 0-11 (bits 0-59), word 1 holds slots 12-23, word 2 holds slots 24-25.
DNA_SLOT_WORD = tuple(n // 12 for n in range(26))   # (0,0,...,0, 1,1,...,1, 2,2)
DNA_SLOT_BIT  = tuple((n % 12) * 5 for n in range(26))  # bit offset within word

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
# thresholds from pyg_config.json → hybrid section instead.
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
