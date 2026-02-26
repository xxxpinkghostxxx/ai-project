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
# Layout: [ALIVE:1][NODE_TYPE:2][CONN_TYPE:3][DNA[0..7]:8×5=40][RSVD:18]
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
# =============================================================================
NODE_ENERGY_CAP = 244
NODE_DEATH_THRESHOLD = -10.0
NODE_SPAWN_THRESHOLD = NODE_ENERGY_CAP * 0.09
NODE_ENERGY_SPAWN_COST = NODE_ENERGY_CAP * 0.08
MAX_NODE_BIRTHS_PER_STEP = 80

DYNAMIC_NODE_ENERGY_DECAY = 0.005
CONN_ENERGY_TRANSFER_CAPACITY = 0.3
CONN_MAINTENANCE_COST = NODE_ENERGY_CAP * 0.0005
