# config.py - Project configuration parameters
# (Ensure this file is present in the project root for imports)

# Configuration for Energy-Based Neural System

# Sensory Layer (used for all sensory grid sizing)
SENSOR_WIDTH = 256
SENSOR_HEIGHT = 144

# Processing Layer
INITIAL_PROCESSING_NODES = 30
MAX_PROCESSING_NODES = 2_000_000
NODE_ENERGY_INIT_RANGE = (8, 12)
NODE_ENERGY_SPAWN_COST = 0.2
NODE_ENERGY_CONN_COST = 0.02
NODE_ENERGY_IDLE_COST = 0.0002
NODE_ENERGY_GEN_RATE = 3.0
NODE_ENERGY_CAP = 244  # Max energy for dynamic nodes (matches visualization cap)

# Connection
CONN_MAINTENANCE_COST = 0.0002
CONN_ENERGY_TRANSFER_CAPACITY = 0.3

# Growth/Pruning
NODE_SPAWN_THRESHOLD = 1.0
CONN_SPAWN_THRESHOLD = 2.0
NODE_DEATH_THRESHOLD = -20.0

# Workspace
WORKSPACE_SIZE = (16, 16)

# Monitoring
ENERGY_HISTORY_LENGTH = 200

# Safety
MAX_NODE_BIRTHS_PER_STEP = 200
MAX_CONN_BIRTHS_PER_STEP = 100

# Visualization
DRAW_GRID_SIZE = 16
MAX_TOTAL_CONNECTIONS = 2_000_000

# Logging and UI config
LOG_DIR = 'logs'
DRAW_WINDOW_SIZE = 320
DRAW_UPSCALE = 20
DASH_EXPORT_PATH = 'live_data.pkl'

# Increase screen capture queue size for less frame drop
SCREEN_CAPTURE_QUEUE_SIZE = 100
PERIODIC_UPDATE_MS = 100

# --- Advanced Node/Connection Growth ---
MIN_PROCESSING_NODES = 10  # Minimum number of dynamic nodes
MAX_CONNECTIONS_PER_NODE = 14
MIN_CONNECTION_WEIGHT = -1.0
MAX_CONNECTION_WEIGHT = 1.0
SENSORY_TO_DYNAMIC_CONN_WEIGHT = 0.1  # Default initial weight
WORKSPACE_TO_DYNAMIC_CONN_WEIGHT = 0.1
DYNAMIC_TO_DYNAMIC_CONN_WEIGHT = 0.1
DYNAMIC_TO_SENSORY_CONN_WEIGHT = 0.1
INITIAL_CONN_PER_NODE = 3  # Used for initial random connection setup
INITIAL_SENSORY_TO_DYNAMIC_CONN = 3
INITIAL_WORKSPACE_TO_DYNAMIC_CONN = 2
INITIAL_DYNAMIC_TO_DYNAMIC_CONN = 2
INITIAL_DYNAMIC_TO_SENSORY_CONN = 1

# --- Energy Economy ---
# Base energy generation and consumption rates
BASE_ENERGY_GEN = 0.1  # Base energy generated per tick
BASE_ENERGY_CONSUMPTION = 0.05  # Base energy consumed per tick

# Node energy parameters
NODE_ENERGY_CAP = 100.0  # Maximum energy a node can hold
NODE_DEATH_THRESHOLD = -10.0  # Node dies if energy falls below this
NODE_SPAWN_THRESHOLD = 20.0  # Node can spawn new connections when above this
NODE_ENERGY_INIT = 10.0  # Initial energy for new nodes

# Connection parameters
CONNECTION_MAINTENANCE_COST = 0.01  # Energy cost per connection per tick
CONNECTION_FORMATION_COST = 5.0  # Energy cost to form a new connection
CONNECTION_ENERGY_TRANSFER = 0.2  # Maximum energy transfer per connection per tick

# Growth parameters
MAX_NODES_PER_UPDATE = 10  # Maximum number of new nodes per update
MAX_CONNECTIONS_PER_UPDATE = 20  # Maximum number of new connections per update
MAX_CONNECTIONS_PER_NODE = 8  # Maximum number of connections per node

# Safety parameters
EMERGENCY_SHUTDOWN_THRESHOLD = 0.8  # System shuts down if energy std dev exceeds 80% of cap
MAX_TOTAL_NODES = 10000  # Maximum total nodes in the system
MAX_TOTAL_CONNECTIONS = 50000  # Maximum total connections in the system

# Update intervals
UPDATE_INTERVAL_MS = 100  # System update interval in milliseconds
ENERGY_UPDATE_INTERVAL_MS = 50  # Energy update interval in milliseconds
CONNECTION_UPDATE_INTERVAL_MS = 200  # Connection update interval in milliseconds

# Visualization parameters
VISUALIZATION_UPDATE_INTERVAL_MS = 500  # How often to update visualizations
ENERGY_HISTORY_LENGTH = 100  # Number of energy history points to keep

# Debug parameters
DEBUG_MODE = False  # Enable debug logging
DEBUG_ENERGY_DISTRIBUTION = True  # Log energy distribution statistics
DEBUG_NODE_LIFECYCLE = True  # Log node birth/death events
DEBUG_CONNECTION_CHANGES = True  # Log connection formation/removal events

# Performance parameters
BATCH_SIZE = 1000  # Number of nodes to process in each batch
USE_GPU = True  # Whether to use GPU acceleration
GPU_MEMORY_FRACTION = 0.8  # Maximum fraction of GPU memory to use

# System state parameters
INITIAL_SUSPENDED = False  # Whether the system starts suspended
INITIAL_SENSORY_ENABLED = True  # Whether sensory input starts enabled
INITIAL_WORKSPACE_ENABLED = True  # Whether workspace manipulation starts enabled

# Resource limits
MAX_MEMORY_USAGE_MB = 1024  # Maximum memory usage in MB
MAX_CPU_USAGE_PERCENT = 80  # Maximum CPU usage percentage
MAX_GPU_USAGE_PERCENT = 80  # Maximum GPU usage percentage

# Logging parameters
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE = "system.log"  # Log file path
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Log format

# --- Derived parameters ---
# These are calculated from the above parameters
ENERGY_UPDATE_STEPS = UPDATE_INTERVAL_MS // ENERGY_UPDATE_INTERVAL_MS
CONNECTION_UPDATE_STEPS = UPDATE_INTERVAL_MS // CONNECTION_UPDATE_INTERVAL_MS
VISUALIZATION_UPDATE_STEPS = UPDATE_INTERVAL_MS // VISUALIZATION_UPDATE_INTERVAL_MS

# Energy thresholds for different node types
SENSORY_NODE_ENERGY_CAP = NODE_ENERGY_CAP * 0.5  # Sensory nodes have lower energy cap
WORKSPACE_NODE_ENERGY_CAP = NODE_ENERGY_CAP * 0.7  # Workspace nodes have medium energy cap
HIGHWAY_NODE_ENERGY_CAP = NODE_ENERGY_CAP  # Highway nodes have full energy cap

# Energy generation rates for different node types
SENSORY_NODE_ENERGY_GEN = BASE_ENERGY_GEN * 0.5  # Sensory nodes generate less energy
WORKSPACE_NODE_ENERGY_GEN = BASE_ENERGY_GEN * 0.7  # Workspace nodes generate medium energy
HIGHWAY_NODE_ENERGY_GEN = BASE_ENERGY_GEN  # Highway nodes generate full energy

# Energy consumption rates for different node types
SENSORY_NODE_ENERGY_CONSUMPTION = BASE_ENERGY_CONSUMPTION * 0.5  # Sensory nodes consume less
WORKSPACE_NODE_ENERGY_CONSUMPTION = BASE_ENERGY_CONSUMPTION * 0.7  # Workspace nodes consume medium
HIGHWAY_NODE_ENERGY_CONSUMPTION = BASE_ENERGY_CONSUMPTION  # Highway nodes consume full

# Connection parameters for different node types
SENSORY_TO_DYNAMIC_CONN_WEIGHT = 0.3  # Weight of sensory to dynamic connections
WORKSPACE_TO_DYNAMIC_CONN_WEIGHT = 0.3  # Weight of workspace to dynamic connections
DYNAMIC_TO_DYNAMIC_CONN_WEIGHT = 0.2  # Weight of dynamic to dynamic connections
HIGHWAY_CONN_WEIGHT = 0.4  # Weight of highway connections

# Growth parameters for different node types
SENSORY_NODE_SPAWN_THRESHOLD = NODE_SPAWN_THRESHOLD * 0.5  # Sensory nodes spawn at lower energy
WORKSPACE_NODE_SPAWN_THRESHOLD = NODE_SPAWN_THRESHOLD * 0.7  # Workspace nodes spawn at medium energy
HIGHWAY_NODE_SPAWN_THRESHOLD = NODE_SPAWN_THRESHOLD  # Highway nodes spawn at full energy

# Death thresholds for different node types
SENSORY_NODE_DEATH_THRESHOLD = NODE_DEATH_THRESHOLD * 0.5  # Sensory nodes die at higher energy
WORKSPACE_NODE_DEATH_THRESHOLD = NODE_DEATH_THRESHOLD * 0.7  # Workspace nodes die at medium energy
HIGHWAY_NODE_DEATH_THRESHOLD = NODE_DEATH_THRESHOLD  # Highway nodes die at full energy

# --- Pruning/Death ---
MIN_NODE_ENERGY = 0.0  # Node dies if below
MIN_CONN_WEIGHT = 0.01  # Prune connections below this weight
MAX_CONN_AGE = 10000  # Max age before forced pruning (ticks)

# --- Other ---
# (Add more as needed for experiments)

# --- Balanced & Growth-Encouraging Settings ---
# These settings encourage node turnover (death and birth) for emergent adaptation.
CONN_MAINTENANCE_COST = 0.0002  # Lower cost to allow more connections
SENSORY_NODE_ENERGY_GEN_RATE = 5.0  # Higher energy input from sensory nodes
DYNAMIC_NODE_ENERGY_DECAY = 0.01  # Lower decay for dynamic nodes
NODE_ENERGY_IDLE_COST = 0.0002  # Lower idle cost
INITIAL_CONN_PER_NODE = 3  # Fewer initial connections per node
MAX_CONNECTIONS_PER_NODE = 14  # Lower cap for sparser, more adaptive network
NODE_DEATH_THRESHOLD = -20.0  # Higher threshold: nodes die more easily, encouraging turnover
NODE_SPAWN_THRESHOLD = 1.0  # Slightly higher: only high-energy nodes spawn
MAX_NODE_BIRTHS_PER_STEP = 200  # Allow more births per step for adaptation

# (Keep other advanced/experimental settings as is)

USE_GPU = True  # Set False to force CPU mode

# --- Node base energy generation parameters ---
BASE_GEN = 0.2  # Base energy generated per tick (per second)
OPTIMAL_CONN = 10  # Optimal number of outgoing connections for max base gen

DEBUG_MODE = False  # Set to True to enable debug logging
DEBUG_ON_ZERO_DYNAMIC_NODES = True  # Set to True to enable special debug logging when dynamic nodes reach zero 

# MAX_PROCESSING_NODES and MAX_TOTAL_CONNECTIONS are set very high to effectively remove the cap. 

# --- RAPID GROWTH SETTINGS ---
# These settings are tuned to encourage rapid node and connection growth.
NODE_SPAWN_THRESHOLD = 1.0  # Lower: nodes can spawn at much lower energy
NODE_DEATH_THRESHOLD = -20.0  # Lower: nodes only die at very low energy
MAX_NODE_BIRTHS_PER_STEP = 200  # Allow many more births per step
MAX_CONN_BIRTHS_PER_STEP = 100  # Allow many more connections per step
NODE_ENERGY_SPAWN_COST = 0.2  # Lower cost to spawn nodes
NODE_ENERGY_CONN_COST = 0.02  # Lower cost to create connections
NODE_ENERGY_IDLE_COST = 0.0002  # Lower idle cost
NODE_ENERGY_GEN_RATE = 3.0  # Higher energy generation per node
CONN_MAINTENANCE_COST = 0.0002  # Lower maintenance cost
SENSORY_NODE_ENERGY_GEN_RATE = 5.0  # Higher energy input from sensory nodes
WORKSPACE_NODE_ENERGY_GEN_RATE = 5.0  # Higher energy for workspace nodes
BASE_GEN = 0.2  # Higher base energy generation per tick
OPTIMAL_CONN = 10  # More outgoing connections for max base gen 

# --- Energy Caps (Unified) ---
NODE_ENERGY_CAP = 244  # Max energy for dynamic nodes (matches visualization cap)
SENSOR_NODE_ENERGY_CAP = 244  # Max energy for sensory nodes
WORKSPACE_NODE_ENERGY_CAP = 244  # Max energy for workspace nodes

# --- Cost Scaling (percentages of cap) ---
DYNAMIC_SPAWN_COST_PCT = 0.08   # 8% of cap to spawn a node (lowered for easier growth)
DYNAMIC_CONN_COST_PCT = 0.008   # 0.8% of cap to create a connection (lowered)
DYNAMIC_IDLE_COST_PCT = 0.0005  # 0.05% of cap per tick (idle, very low)
DYNAMIC_GEN_RATE_PCT = 0.12     # 12% of cap generated per tick (moderate)
CONN_MAINTENANCE_COST_PCT = 0.0005  # 0.05% of cap per connection per tick (very low)

# --- Derived absolute values (from cap) ---
NODE_ENERGY_SPAWN_COST = NODE_ENERGY_CAP * DYNAMIC_SPAWN_COST_PCT      # 19.52
NODE_ENERGY_CONN_COST = NODE_ENERGY_CAP * DYNAMIC_CONN_COST_PCT        # 1.952
NODE_ENERGY_IDLE_COST = NODE_ENERGY_CAP * DYNAMIC_IDLE_COST_PCT        # 0.122
NODE_ENERGY_GEN_RATE = NODE_ENERGY_CAP * DYNAMIC_GEN_RATE_PCT          # 29.28
CONN_MAINTENANCE_COST = NODE_ENERGY_CAP * CONN_MAINTENANCE_COST_PCT    # 0.122

WORKSPACE_NODE_ENERGY_GEN_RATE = WORKSPACE_NODE_ENERGY_CAP * 0.18      # 43.92
SENSORY_NODE_ENERGY_GEN_RATE = SENSOR_NODE_ENERGY_CAP * 0.18           # 43.92

# --- Growth/Pruning (relative to cap) ---
NODE_SPAWN_THRESHOLD = NODE_ENERGY_CAP * 0.09      # 9% of cap
NODE_DEATH_THRESHOLD = -10.0    # Nodes only die if very negative (forgiving, but less extreme)
MAX_NODE_BIRTHS_PER_STEP = 80   # Allow healthy growth, but not runaway
MAX_CONN_BIRTHS_PER_STEP = 40

# --- Decay ---
DYNAMIC_NODE_ENERGY_DECAY = 0.005
SENSORY_NODE_ENERGY_DECAY = 0.005
WORKSPACE_NODE_ENERGY_DECAY = 0.005

# --- Connection ---
CONN_ENERGY_TRANSFER_CAPACITY = 0.3

# --- Pruning/Death ---
MIN_NODE_ENERGY = 0.0
MIN_CONN_WEIGHT = 0.01
MAX_CONN_AGE = 10000

# --- Other ---
OPTIMAL_CONN = 10
BASE_GEN = DYNAMIC_GEN_RATE_PCT

# --- Visualization/Update Tuning ---
PERIODIC_UPDATE_MS = 200

# --- Remove any remaining duplicate/conflicting assignments below ---
# ... existing code ... 
