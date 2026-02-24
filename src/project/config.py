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

# Connection type → weight profile lookup table (index by CONN_TYPE 0..3)
CONN_TYPE_WEIGHT_TABLE = (
    (0.7, 0.1, 0.1, 0.1),  # 0: excitatory-dominant
    (0.1, 0.7, 0.1, 0.1),  # 1: inhibitory-dominant
    (0.1, 0.1, 0.7, 0.1),  # 2: gated-dominant
    (0.1, 0.1, 0.1, 0.7),  # 3: plastic-dominant
)

# =============================================================================
# Binary Node State Encoding (64-bit packed int64 per node)
# Layout: [ALIVE:1][NODE_TYPE:2][CONN_TYPE:2][DNA[0..7]:8×5=40][ENERGY_Q:16][RSVD:3]
# state == 0 means DEAD — all DNA wiped, disconnected from all math.
# =============================================================================
BINARY_ALIVE_BIT = 63
BINARY_NODE_TYPE_SHIFT = 61
BINARY_CONN_TYPE_SHIFT = 59
BINARY_DNA_BASE_SHIFT = 19
BINARY_DNA_BITS_PER_NEIGHBOR = 5
BINARY_DNA_MAX_VALUE = 31          # 2^5 - 1
BINARY_DNA_MASK = 0x1F             # 5 bits
BINARY_TYPE_MASK = 0x3             # 2 bits

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
