# System Architecture

## Overview

The PyTorch Geometric Neural System uses a **hybrid grid-graph architecture** that combines a continuous energy field (grid layer) with discrete node mechanics (node layer). This achieves billions of parallel operations while preserving individual node identity, spawn/death mechanics, and connection logic.

---

## Two-Layer Design

```
+-----------------------------------------------------------+
|  NODE LAYER (Discrete Logic)                              |
|  - Individual node identities                              |
|  - Spawn when energy > threshold (default 8.0)             |
|  - Die when energy < threshold (default 1.0)               |
|  - Node types (Sensory, Dynamic, Workspace)               |
|  - Connection topology                                     |
+-----------------------------------------------------------+
                          | Sync
+-----------------------------------------------------------+
|  GRID LAYER (Continuous Physics)                           |
|  - Energy field [2560x1920] (tiled, 512x512 tiles)         |
|  - FFT diffusion (5+ billion ops/sec)                     |
|  - Reaction-diffusion PDEs                                 |
|  - Bulk energy propagation                                 |
+-----------------------------------------------------------+
```

### How It Works

1. **Grid Layer** handles bulk energy propagation (billions of parallel ops via FFT convolution)
2. **Node Layer** applies spawn/death/type-specific rules
3. **Synchronization** happens efficiently (only nodes update, grid is continuous)

---

## Node Types

| Node Type | Immortal | Can Spawn | Can Die | Feed | Display |
|-----------|----------|-----------|---------|------|---------|
| Sensory (0) | Yes | No | No | Desktop pixels | No |
| Dynamic (1) | No | Yes (E>threshold) | Yes (E<threshold) | No | No |
| Workspace (2) | Yes | No | No | No | UI grid |

### Node Interaction Rules

**Allowed interactions:**
```
Sensory  -> Dynamic    (inject energy into dynamic field)
Dynamic  -> Dynamic    (Markov chains via convolution)
Dynamic <-> Workspace  (bidirectional feedback)
```

**Forbidden interactions:**
```
Sensory  -> Workspace  (no direct connection)
Workspace -> Sensory   (no direct connection)
Sensory  -> Sensory    (sensory nodes don't connect to each other)
Workspace -> Workspace (workspace nodes don't connect to each other)
```

---

## Data Flow

```
Desktop Pixels (1920x1080)
    | (capture & downsample)
Sensory Nodes (top region of grid)
    | (probabilistic excitatory transfer)
Dynamic Field (energy_field)
    | Dynamic -> Dynamic Markov Chains
    | (long chains via convolution)
    |
Dynamic -> Workspace (bidirectional)
    |
Workspace Nodes (bottom region of grid, 128x128)
    | (read for display)
UI Grid (workspace visualization)
```

---

## Connection Logic: Probabilistic Neighborhood Model

Instead of explicit per-edge connections, the system uses **probabilistic 8-neighbor transfer** on the grid:

```python
# Each grid cell has 8 neighbors (Moore neighborhood)
# Connection types map to probability weights:
EXCITATORY (60%):  dst.energy += transfer   # Positive
INHIBITORY (20%):  dst.energy -= transfer   # Negative
GATED      (10%):  transfer if E > threshold
PLASTIC    (10%):  Adaptive weights
```

### How Convolution Enables Markov Chains

```python
# Single convolution step: Dynamic -> Dynamic (1 hop)
neighbor_activity = F.conv2d(node_activity, flow_kernel)

# Multiple steps = long chains
for step in range(num_steps):
    neighbor_activity = F.conv2d(neighbor_activity, flow_kernel)
    # Energy propagates: Dynamic -> Dynamic -> Dynamic -> ...
```

### Dirac-Compressed Probabilities

Connection probabilities are represented as sparse field operations rather than explicit edges:
- **Compressed**: Sparse field operations instead of edge lists
- **Probabilistic**: Each neighbor has a probability weight (1/8 for equal distribution)
- **Vectorized**: All connections processed in parallel via convolution

---

## Hybrid Engine Integration

The hybrid engine integrates into `pyg_main.py` via the `HybridNeuralSystemAdapter` class and can be enabled with a single config change:

```json
{
    "hybrid": {
        "enabled": true,
        "grid_size": [2560, 1920],
        "tile_size": [512, 512],
        "tile_mode": true,
        "toroidal": true,
        "excitatory_prob": 0.6,
        "inhibitory_prob": 0.2,
        "gated_prob": 0.1,
        "num_diffusion_steps": 1,
        "diffusion_coeff": 0.2,
        "node_spawn_threshold": 8.0,
        "node_death_threshold": 1.0,
        "node_energy_cap": 1000.0
    }
}
```

### System Modes

| Feature | Traditional Mode | Hybrid Mode |
|---------|-----------------|-------------|
| Speed | ~49K updates/sec | ~5B ops/sec |
| Node Types | All 3 | All 3 |
| Desktop Feed | Yes | Yes |
| Workspace UI | Yes | Yes |
| Spawn/Death | Yes | Yes |
| Connection Logic | Explicit edges | Probabilistic |
| Backward Compatible | - | 100% |

### Usage

```python
from project.system.hybrid_grid_engine import HybridGridGraphEngine

engine = HybridGridGraphEngine(
    grid_size=(2560, 1920),
    device='cuda',
    node_spawn_threshold=8.0,
    node_death_threshold=1.0
)

# Run with probabilistic connections
engine.step(
    num_diffusion_steps=1,
    use_probabilistic_transfer=True,
    excitatory_prob=0.6,
    inhibitory_prob=0.2
)
```

---

## Mathematical Foundation

### Why Grid Operations Are Faster

| Method | Complexity | Example |
|--------|-----------|---------|
| Node-to-node (explicit edges) | O(E) | ~10,000 ops |
| Grid FFT diffusion | O(N log N) | Millions of ops (parallel on GPU) |

Grid diffusion approximates connection-based transfer:
```
Traditional:  E_dst += E_src * weight * capacity
Grid:         dE/dt = D * nabla^2(E)
```

With proper diffusion coefficient D, both produce similar energy flow patterns.

---

## Key Files

- `src/project/system/hybrid_grid_engine.py` - Hybrid grid-graph engine
- `src/project/system/probabilistic_field_engine.py` - Probabilistic field engine
- `src/project/system/sparse_hybrid_engine.py` - Sparse hybrid engine
- `src/project/pyg_main.py` - Main application with mode selection
- `src/project/config.py` - Configuration management
