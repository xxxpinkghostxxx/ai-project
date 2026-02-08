# Node Interaction Architecture

## Critical Constraint

**Workspace nodes and Sensory nodes NEVER interact directly!**

Both node types ONLY interact with Dynamic nodes.

## Allowed Interactions

```
Sensory → Dynamic ✅
Dynamic → Dynamic ✅ (Markov chains!)
Dynamic → Workspace ✅
Workspace → Dynamic ✅ (bidirectional feedback)
```

## Forbidden Interactions

```
Sensory → Workspace ❌
Workspace → Sensory ❌
Sensory → Sensory ❌ (sensory nodes don't connect to each other)
Workspace → Workspace ❌ (workspace nodes don't connect to each other)
```

## Data Flow Chain

```
Desktop Pixels
    ↓
Sensory Nodes (inject into dynamic field)
    ↓
Dynamic Field (energy_field)
    ↓
Dynamic → Dynamic Markov Chains
    ↓
    (Long chains via convolution)
    ↓
Dynamic → Workspace (bidirectional)
    ↓
Workspace Nodes (display to UI)
```

## Dynamic Node Markov Chains

Dynamic nodes form **long Markov chains** with **Dirac-compressed connection probabilities**:

1. **8-Neighbor Convolution**: Each dynamic node connects to 8 neighbors via convolution
2. **Probabilistic Transfer**: Connection probabilities are represented as field operations (Dirac-compressed)
3. **Chain Propagation**: Energy flows through chains via repeated convolution steps
4. **Arbitrary Length**: Chains can be arbitrarily long (limited only by grid size)

### How Convolution Enables Chains

```python
# Single convolution step: Dynamic → Dynamic (1 hop)
neighbor_activity = F.conv2d(node_activity, flow_kernel)

# Multiple steps = long chains
for step in range(num_steps):
    neighbor_activity = F.conv2d(neighbor_activity, flow_kernel)
    # Energy propagates through: Dynamic → Dynamic → Dynamic → ...
```

### Dirac-Compressed Probabilities

Instead of storing explicit edges, connection probabilities are:
- **Compressed**: Represented as sparse field operations
- **Probabilistic**: Each neighbor has a probability weight (1/8 for equal distribution)
- **Dirac-like**: Point masses (sparse) rather than continuous distributions
- **Vectorized**: All connections processed in parallel via convolution

## Implementation Details

### Sensory Injection
- `inject_sensory_data()`: Injects desktop pixels into `energy_field` (dynamic field)
- **NO workspace feedback**: Sensory nodes only receive external input
- Location: Sensory region of grid (top portion)

### Dynamic Flow
- `step()`: Convolution-based neighbor flow between dynamic nodes
- Enables Markov chains of arbitrary length
- Connection probabilities encoded in `flow_kernel` (3×3, 8 neighbors)

### Workspace Interaction
- `step()`: Workspace nodes pull/push from dynamic field only
- Samples dynamic field at positions above workspace nodes
- Bidirectional: Workspace can feed back into dynamic field
- **NO direct sensory interaction**

## Benefits

1. **Clean Separation**: Sensory and Workspace are isolated
2. **Scalable Chains**: Dynamic nodes can form chains of any length
3. **Fast**: Convolution is highly optimized on GPU
4. **Probabilistic**: Dirac-compressed representation is memory-efficient
