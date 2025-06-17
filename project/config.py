"""
Configuration module for the Energy-Based Neural System.

This module contains all configuration parameters for the neural system,
including sensory processing, energy management, node/connection parameters,
and visualization settings.
"""

# Sensory Layer (used for all sensory grid sizing)
SENSOR_WIDTH = 256
SENSOR_HEIGHT = 144

# Processing Layer
INITIAL_PROCESSING_NODES = 30
MAX_PROCESSING_NODES = 2_000_000
MIN_PROCESSING_NODES = 10

# Energy parameters (unified)
NODE_ENERGY_CAP = 244  # Max energy for dynamic nodes (matches visualization cap)
SENSOR_NODE_ENERGY_CAP = 244  # Max energy for sensory nodes  
WORKSPACE_NODE_ENERGY_CAP = 244  # Max energy for workspace nodes

# Energy costs and generation
NODE_ENERGY_SPAWN_COST = 0.2  # Lower cost to spawn nodes
NODE_ENERGY_CONN_COST = 0.02  # Lower cost to create connections
NODE_ENERGY_IDLE_COST = 0.0002  # Lower idle cost
NODE_ENERGY_GEN_RATE = 3.0  # Higher energy generation per node
NODE_ENERGY_INIT_RANGE = (8, 12)

# Connection parameters
CONN_MAINTENANCE_COST = 0.0002  # Lower maintenance cost
CONN_ENERGY_TRANSFER_CAPACITY = 0.3
MAX_CONNECTIONS_PER_NODE = 14  # Lower cap for sparser, more adaptive network
MIN_CONNECTION_WEIGHT = -1.0
MAX_CONNECTION_WEIGHT = 1.0

# Growth/Pruning thresholds  
NODE_SPAWN_THRESHOLD = 1.0  # Lower: nodes can spawn at much lower energy
CONN_SPAWN_THRESHOLD = 2.0
NODE_DEATH_THRESHOLD = -20.0  # Lower: nodes only die at very low energy

# Safety limits
MAX_NODE_BIRTHS_PER_STEP = 200  # Allow many more births per step
MAX_CONN_BIRTHS_PER_STEP = 100  # Allow many more connections per step
MAX_TOTAL_CONNECTIONS = 2_000_000
MAX_TOTAL_NODES = 10000

# Workspace
WORKSPACE_SIZE = (16, 16)

# Update intervals (milliseconds)
PERIODIC_UPDATE_MS = 200
UPDATE_INTERVAL_MS = 100
ENERGY_UPDATE_INTERVAL_MS = 50
CONNECTION_UPDATE_INTERVAL_MS = 200
VISUALIZATION_UPDATE_INTERVAL_MS = 500

# Monitoring and history
ENERGY_HISTORY_LENGTH = 200

# Visualization settings
DRAW_GRID_SIZE = 16
DRAW_WINDOW_SIZE = 320
DRAW_UPSCALE = 20

# Screen capture settings
SCREEN_CAPTURE_QUEUE_SIZE = 100

# Connection weights by type
SENSORY_TO_DYNAMIC_CONN_WEIGHT = 0.3  # Weight of sensory to dynamic connections
WORKSPACE_TO_DYNAMIC_CONN_WEIGHT = 0.3  # Weight of workspace to dynamic connections
DYNAMIC_TO_DYNAMIC_CONN_WEIGHT = 0.2  # Weight of dynamic to dynamic connections
DYNAMIC_TO_SENSORY_CONN_WEIGHT = 0.1
HIGHWAY_CONN_WEIGHT = 0.4  # Weight of highway connections

# Initial connection setup
INITIAL_CONN_PER_NODE = 3  # Fewer initial connections per node
INITIAL_SENSORY_TO_DYNAMIC_CONN = 3
INITIAL_WORKSPACE_TO_DYNAMIC_CONN = 2
INITIAL_DYNAMIC_TO_DYNAMIC_CONN = 2
INITIAL_DYNAMIC_TO_SENSORY_CONN = 1

# Energy generation rates by node type
BASE_ENERGY_GEN = 0.2  # Higher base energy generation per tick
BASE_ENERGY_CONSUMPTION = 0.05
SENSORY_NODE_ENERGY_GEN_RATE = 5.0  # Higher energy input from sensory nodes
WORKSPACE_NODE_ENERGY_GEN_RATE = 5.0  # Higher energy for workspace nodes

# Energy decay rates
DYNAMIC_NODE_ENERGY_DECAY = 0.005
SENSORY_NODE_ENERGY_DECAY = 0.005
WORKSPACE_NODE_ENERGY_DECAY = 0.005

# Performance settings
BATCH_SIZE = 1000
USE_GPU = True  # Set False to force CPU mode
GPU_MEMORY_FRACTION = 0.8

# Resource limits
MAX_MEMORY_USAGE_MB = 1024
MAX_CPU_USAGE_PERCENT = 80
MAX_GPU_USAGE_PERCENT = 80

# System state
INITIAL_SUSPENDED = False
INITIAL_SENSORY_ENABLED = True
INITIAL_WORKSPACE_ENABLED = True

# Debug settings
DEBUG_MODE = False  # Set to True to enable debug logging
DEBUG_ENERGY_DISTRIBUTION = True
DEBUG_NODE_LIFECYCLE = True
DEBUG_CONNECTION_CHANGES = True
DEBUG_ON_ZERO_DYNAMIC_NODES = True  # Special debug logging when dynamic nodes reach zero

# Logging configuration
LOG_DIR = 'logs'
LOG_LEVEL = "INFO"
LOG_FILE = "system.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Safety thresholds
EMERGENCY_SHUTDOWN_THRESHOLD = 0.8
MIN_NODE_ENERGY = 0.0
MIN_CONN_WEIGHT = 0.01
MAX_CONN_AGE = 10000

# Derived calculations
ENERGY_UPDATE_STEPS = UPDATE_INTERVAL_MS // ENERGY_UPDATE_INTERVAL_MS
CONNECTION_UPDATE_STEPS = UPDATE_INTERVAL_MS // CONNECTION_UPDATE_INTERVAL_MS
VISUALIZATION_UPDATE_STEPS = UPDATE_INTERVAL_MS // VISUALIZATION_UPDATE_INTERVAL_MS

# Optimal connection count for base energy generation
OPTIMAL_CONN = 10  # More outgoing connections for max base gen

# Export settings
DASH_EXPORT_PATH = 'live_data.pkl'
