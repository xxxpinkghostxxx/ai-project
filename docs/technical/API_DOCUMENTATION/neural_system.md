# Neural System API

The core neural system implementation using PyTorch Geometric with advanced features for energy-based neural networks.

## Module: `pyg_neural_system.py`

### Main Class: `PyGNeuralSystem`

#### Constructor

```python
class PyGNeuralSystem:
    def __init__(self, sensory_width: int, sensory_height: int, n_dynamic: int,
                 workspace_size: tuple[int, int] = (16, 16), device: str = 'cpu')
```

**Parameters:**

- `sensory_width`: Width of the sensory input grid
- `sensory_height`: Height of the sensory input grid
- `n_dynamic`: Initial number of dynamic nodes
- `workspace_size`: Tuple specifying workspace dimensions (default: (16, 16))
- `device`: Compute device ('cpu' or 'cuda', default: 'cpu')

### Node Types and Constants

The system supports multiple node types:

- `NODE_TYPE_SENSORY = 0`: Sensory input nodes
- `NODE_TYPE_DYNAMIC = 1`: Dynamic processing nodes
- `NODE_TYPE_WORKSPACE = 2`: Workspace nodes
- `NODE_TYPE_HIGHWAY = 3`: Highway nodes (special high-capacity nodes)

### Connection Types

- `CONN_TYPE_EXCITATORY = 0`: Standard positive-weight connections
- `CONN_TYPE_INHIBITORY = 1`: Negative-weight connections
- `CONN_TYPE_GATED = 2`: Conditional connections based on thresholds
- `CONN_TYPE_PLASTIC = 3`: Adaptive connections with learning capabilities

### Key Methods

#### `update()`

Main update method that orchestrates the entire neural system update cycle:

- Processes node energy updates
- Handles node birth/death cycles
- Manages connection growth and pruning
- Applies connection worker results
- Includes comprehensive error handling and recovery

```python
neural_system = PyGNeuralSystem(sensory_width=256, sensory_height=144, n_dynamic=30)
neural_system.update()
```

#### `update_sensory_nodes(sensory_input: np.ndarray | torch.Tensor)`

Updates sensory nodes with new input data:

- Validates input shape and type
- Converts numpy arrays to tensors automatically
- Applies input to sensory node energies

```python
# Capture screen and update sensory nodes
screen_data = capture_screen()
neural_system.update_sensory_nodes(screen_data)
```

#### `get_metrics() -> Dict[str, int | float]`

Returns comprehensive system metrics including:

- Total energy in the system
- Node counts by type (sensory, dynamic, workspace)
- Average dynamic node energy
- Node birth/death statistics
- Connection statistics
- Performance metrics

```python
metrics = neural_system.get_metrics()
print(f"Total energy: {metrics['total_energy']:.2f}")
print(f"Dynamic nodes: {metrics['dynamic_node_count']}")
```

#### `cleanup()`

Cleans up system resources and stops background threads:

```python
# Clean up when done
neural_system.cleanup()
```

**Note:** The `PyGNeuralSystem` class does not have a `reset()` method. To reset the system, create a new instance.

### Advanced Features

#### Connection Management

The system includes sophisticated connection management:

- **Connection Workers**: Thread-based connection processing for performance
- **Batch Processing**: Efficient handling of large connection sets
- **Directional Connections**: Support for one-way and bidirectional connections
- **Plastic Connections**: Learning connections that adapt over time

```python
# Start connection worker
neural_system.start_connection_worker(batch_size=25)

# Queue connection tasks
neural_system.queue_connection_growth()
neural_system.queue_cull()
```

#### Energy Management

Comprehensive energy management system:

- **Energy Transfer**: Vectorized energy calculations for performance
- **Energy Caps**: Node energy limits with automatic clamping
- **Energy-Based Spawning**: New nodes created based on energy thresholds
- **Decay Mechanisms**: Automatic energy decay for all node types

#### Recovery and Error Handling

Robust error recovery mechanisms:

- **Graph State Validation**: Continuous validation of system state
- **Automatic Recovery**: Self-healing from common error conditions
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Memory Management**: Automatic cleanup and resource optimization

### System Architecture

The neural system follows a modular architecture:

1. **Sensory Layer**: Input processing and feature extraction
2. **Dynamic Layer**: Core processing with adaptive nodes
3. **Workspace Layer**: Higher-level pattern recognition
4. **Highway System**: Fast information routing between layers

### Performance Optimization

Key performance features:

- **Vectorized Operations**: All computations use PyTorch vectorized operations
- **GPU Acceleration**: Automatic CUDA support when available
- **Memory Optimization**: Tensor memory management and cleanup
- **Batch Processing**: Efficient handling of large node/connection sets

### Usage Examples

#### Basic System Setup

```python
from project.pyg_neural_system import PyGNeuralSystem
from project.vision import ThreadedScreenCapture

# Initialize system
system = PyGNeuralSystem(
    sensory_width=256,
    sensory_height=144,
    n_dynamic=30,
    workspace_size=(16, 16),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Start connection worker
system.start_connection_worker()

# Main update loop
while running:
    # Capture and process sensory input
    screen_data = capture_screen()
    system.update_sensory_nodes(screen_data)

    # Update neural system
    system.update()

    # Get and display metrics
    metrics = system.get_metrics()
    print(f"Nodes: {metrics['dynamic_node_count']}, Energy: {metrics['total_energy']:.1f}")
```

#### Advanced Configuration

```python
# Configure system parameters
config = {
    'NODE_SPAWN_THRESHOLD': 20.0,
    'NODE_DEATH_THRESHOLD': -10.0,
    'NODE_ENERGY_CAP': 244.0,
    'MAX_NODE_BIRTHS_PER_STEP': 80,
    'MAX_CONN_BIRTHS_PER_STEP': 40
}

# Apply configuration to system
for key, value in config.items():
    if hasattr(system, key):
        setattr(system, key, value)
```

### Best Practices

1. **Resource Management**: Always call `cleanup()` when done with the system
2. **Error Handling**: Wrap system operations in try-catch blocks
3. **Performance Monitoring**: Use the metrics system to track performance
4. **Configuration**: Validate all configuration parameters before use
5. **Thread Safety**: Use the provided locking mechanisms for thread-safe operations

### Troubleshooting

Common issues and solutions:

- **Memory Issues**: Reduce node counts or use smaller workspace sizes
- **Performance Problems**: Use GPU acceleration and reduce connection density
- **Stability Issues**: Adjust energy thresholds and decay rates
- **Connection Problems**: Validate graph state and use recovery mechanisms