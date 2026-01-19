# Configuration Management API

Centralized configuration management for the entire neural system with comprehensive parameter control.

## Module: `config.py`

### Configuration Structure

The configuration system organizes parameters into logical categories:

#### Sensory Layer Parameters

- `SENSOR_WIDTH = 256`: Width of sensory input grid
- `SENSOR_HEIGHT = 144`: Height of sensory input grid

#### Processing Layer Parameters

- `INITIAL_PROCESSING_NODES = 30`: Initial number of dynamic nodes
- `MAX_PROCESSING_NODES = 2_000_000`: Maximum allowed dynamic nodes
- `MIN_PROCESSING_NODES = 10`: Minimum dynamic nodes for system stability

#### Workspace Parameters

- `WORKSPACE_SIZE = (16, 16)`: Dimensions of workspace grid

#### Node and Connection Parameters

- `MAX_CONNECTIONS_PER_NODE = 14`: Maximum connections per node
- `MIN_CONNECTION_WEIGHT = -1.0`: Minimum connection weight
- `MAX_CONNECTION_WEIGHT = 1.0`: Maximum connection weight
- `INITIAL_CONN_PER_NODE = 3`: Initial connections per node

#### Energy Parameters

- `BASE_ENERGY_GEN = 0.2`: Base energy generation rate
- `BASE_ENERGY_CONSUMPTION = 0.05`: Base energy consumption rate
- `NODE_ENERGY_CAP = 244`: Maximum energy per node
- `NODE_SPAWN_THRESHOLD = 21.96`: Energy threshold for node creation (9% of NODE_ENERGY_CAP)
- `NODE_DEATH_THRESHOLD = -10.0`: Energy threshold for node removal

#### Connection Parameters

- `CONNECTION_FORMATION_COST = 5.0`: Energy cost to form connections
- `CONNECTION_ENERGY_TRANSFER = 0.2`: Energy transfer per connection
- `CONN_ENERGY_TRANSFER_CAPACITY = 0.3`: Maximum energy transfer capacity

#### Growth Parameters

- `MAX_NODES_PER_UPDATE = 10`: Maximum nodes added per update
- `MAX_CONNECTIONS_PER_UPDATE = 20`: Maximum connections added per update
- `MAX_NODE_BIRTHS_PER_STEP = 80`: Maximum node births per step
- `MAX_CONN_BIRTHS_PER_STEP = 40`: Maximum connection births per step

#### System Parameters

- `PERIODIC_UPDATE_MS = 200`: Main update loop interval (ms)
- `UPDATE_INTERVAL_MS = 100`: System update interval (ms)
- `ENERGY_UPDATE_INTERVAL_MS = 50`: Energy update interval (ms)
- `CONNECTION_UPDATE_INTERVAL_MS = 200`: Connection update interval (ms)
- `VISUALIZATION_UPDATE_INTERVAL_MS = 500`: Visualization update interval (ms)

#### Resource Limits

- `MAX_TOTAL_NODES = 10000`: Maximum total nodes in system
- `max_total_connections = 50000`: Maximum total connections
- `MAX_MEMORY_USAGE_MB = 1024`: Maximum memory usage (MB)
- `MAX_CPU_USAGE_PERCENT = 80`: Maximum CPU usage (%)
- `MAX_GPU_USAGE_PERCENT = 80`: Maximum GPU usage (%)

#### Debug and Performance Parameters

- `DEBUG_MODE = True`: Enable debug logging
- `BATCH_SIZE = 1000`: Batch size for operations
- `USE_GPU = True`: Enable GPU acceleration
- `GPU_MEMORY_FRACTION = 0.8`: GPU memory allocation fraction

### Usage Patterns

#### Basic Configuration Access

```python
from project.config import SENSOR_WIDTH, SENSOR_HEIGHT

# Access configuration values directly
width = SENSOR_WIDTH
height = SENSOR_HEIGHT
```

#### Configuration Validation

```python
from project.config import (
    INITIAL_PROCESSING_NODES,
    MAX_PROCESSING_NODES,
    NODE_ENERGY_CAP
)

# Validate configuration ranges
if INITIAL_PROCESSING_NODES > MAX_PROCESSING_NODES:
    raise ValueError("Initial nodes exceeds maximum")

if NODE_ENERGY_CAP <= 0:
    raise ValueError("Energy cap must be positive")
```

#### Parameter Calculation

```python
from project.config import (
    NODE_ENERGY_CAP,
    DYNAMIC_SPAWN_COST_PCT,
    DYNAMIC_CONN_COST_PCT
)

# Calculate derived parameters
spawn_cost = NODE_ENERGY_CAP * DYNAMIC_SPAWN_COST_PCT
conn_cost = NODE_ENERGY_CAP * DYNAMIC_CONN_COST_PCT
```

### Advanced Configuration

#### Dynamic Parameter Adjustment

```python
# Note: config.py uses module-level constants, not a config object
# Access values directly:
from project.config import PERIODIC_UPDATE_MS, MAX_PROCESSING_NODES

# Adjust parameters at runtime
config['PERIODIC_UPDATE_MS'] = 150  # Faster updates
config['MAX_PROCESSING_NODES'] = 5000  # More nodes
```

#### Configuration Profiles

```python
# Define configuration profiles
PERFORMANCE_PROFILE = {
    'BATCH_SIZE': 2000,
    'USE_GPU': True,
    'MAX_PROCESSING_NODES': 10000
}

STABILITY_PROFILE = {
    'BATCH_SIZE': 500,
    'MAX_PROCESSING_NODES': 2000,
    'NODE_SPAWN_THRESHOLD': 25.0
}

# Apply profile
for key, value in PERFORMANCE_PROFILE.items():
    if hasattr(config, key):
        setattr(config, key, value)
```

### Best Practices

1. **Validation**: Always validate configuration parameters before use
2. **Documentation**: Document non-standard parameter values
3. **Performance**: Adjust parameters based on system capabilities
4. **Stability**: Start with conservative values and increase gradually
5. **Monitoring**: Track the impact of parameter changes on system behavior

### Parameter Tuning Guide

#### Performance Optimization

- **Increase `BATCH_SIZE`**: For better throughput on powerful hardware
- **Enable `USE_GPU`**: For GPU-accelerated computations
- **Adjust `UPDATE_INTERVAL_MS`**: For smoother real-time performance

#### Stability Enhancement

- **Reduce `MAX_PROCESSING_NODES`**: For more stable operation
- **Increase `NODE_SPAWN_THRESHOLD`**: For slower, more controlled growth
- **Lower `CONNECTION_ENERGY_TRANSFER`**: For more conservative energy flow

#### Memory Management

- **Limit `MAX_TOTAL_NODES`**: To control memory usage
- **Adjust `MAX_MEMORY_USAGE_MB`**: Based on available system resources
- **Monitor `max_total_connections`**: To prevent memory exhaustion

### Configuration Reference

#### Energy System Parameters

- `DYNAMIC_SPAWN_COST_PCT = 0.08`: Spawn cost as % of energy cap
- `DYNAMIC_CONN_COST_PCT = 0.008`: Connection cost as % of energy cap
- `DYNAMIC_IDLE_COST_PCT = 0.0005`: Idle cost as % of energy cap
- `CONN_MAINTENANCE_COST_PCT = 0.0005`: Connection maintenance cost

#### Node Type Parameters

- `SENSORY_NODE_ENERGY_CAP = 122`: Sensory node energy capacity (50% of NODE_ENERGY_CAP)
- `workspace_node_energy_cap = 170`: Workspace node energy capacity (70% of NODE_ENERGY_CAP)

#### Decay and Growth Parameters

- `DYNAMIC_NODE_ENERGY_DECAY = 0.005`: Dynamic node energy decay rate
- `SENSORY_NODE_ENERGY_DECAY = 0.005`: Sensory node energy decay rate
- `WORKSPACE_NODE_ENERGY_DECAY = 0.005`: Workspace node energy decay rate

### Troubleshooting

Common configuration issues and solutions:

- **Performance Problems**: Reduce node counts and connection density
- **Memory Issues**: Lower maximum node and connection limits
- **Stability Issues**: Increase energy thresholds and reduce growth rates
- **Parameter Conflicts**: Validate all parameters work together harmoniously