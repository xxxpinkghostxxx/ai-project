# 3D Volume with Island Clusters — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the neural simulation from a 2D `[2560, 1920]` grid to a 3D `[512, 512, 8]` volume where sensory and workspace nodes form scattered island clusters, and dynamic nodes navigate the full XYZ space via ±1 spawn drift.

**Architecture:** Taichi module-level fields gain `_node_pos_z` and `_node_dna[MAX_NODES, 3]`. The 26-neighbor 3D Moore neighborhood replaces the current 8-neighbor 2D Moore. `energy_field` and `grid_node_id` expand to `[H, W, D]`. Cluster placement in `main.py` replaces the strict horizontal band regions.

**Tech Stack:** Taichi 1.7.x, PyTorch CUDA, Python 3.x. Tests use pytest + `torch.cuda.is_available()` guards.

**Design doc:** `docs/plans/2026-03-03-3d-volume-island-clusters-design.md`

**Key Taichi constraint:** `from __future__ import annotations` must NOT appear in `taichi_engine.py` (breaks `@ti.kernel` annotation resolution). `continue` is not allowed in non-static `if` inside `ti.static for` — use nested `if` guards instead.

---

## Task 1: Config — grid_size and cluster section

**Files:**
- Modify: `src/project/pyg_config.json`
- Modify: `src/project/config.py`

### Step 1: Write the failing test

```python
# tests/test_3d_config.py
import json, os, pytest

def test_grid_size_is_3d():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'project', 'pyg_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg['hybrid']['grid_size'] == [512, 512, 8], "grid_size must be [512, 512, 8]"

def test_clusters_section_exists():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'project', 'pyg_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    cl = cfg.get('clusters', {})
    assert 'sensory_count' in cl
    assert 'workspace_count' in cl
    assert 'min_cluster_separation' in cl
```

### Step 2: Run test to verify it fails

```
pytest tests/test_3d_config.py -v
```
Expected: FAIL — grid_size is currently `[2560, 1920]`, no `clusters` section.

### Step 3: Update `pyg_config.json`

In the `hybrid` section, change:
```json
"grid_size": [512, 512, 8]
```

Add a top-level `clusters` section:
```json
"clusters": {
    "sensory_count": 10,
    "sensory_nodes_each": 30,
    "sensory_radius": 3,
    "workspace_count": 8,
    "workspace_nodes_each": 32,
    "workspace_radius": 3,
    "min_cluster_separation": 8
}
```

### Step 4: Add constants to `config.py`

Add after the existing `REVERSE_DIRECTION` constant:

```python
# =============================================================================
# 3D Neighbor Geometry (26-neighbor Moore neighbourhood)
# =============================================================================
NUM_NEIGHBORS_3D = 26   # all 27 cells in ±1 cube minus center

# 3D neighbor offsets, ordered by (dz, dy, dx) ∈ {-1,0,1}³ \ {(0,0,0)}.
# n = (dz+1)*9 + (dy+1)*3 + (dx+1) for the raw 3×3×3 index.
# Center is at raw index 13, so slots 0-12 → raw 0-12, slots 13-25 → raw 14-26.
_RAW_3D = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
]
NEIGHBOR_OFFSETS_3D = _RAW_3D   # list of (dz, dy, dx) tuples, length 26

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
```

Update the import in `taichi_engine.py` to include these new constants (done in Task 2).

### Step 5: Run tests to verify they pass

```
pytest tests/test_3d_config.py -v
```
Expected: PASS

### Step 6: Commit

```bash
git add src/project/pyg_config.json src/project/config.py tests/test_3d_config.py
git commit -m "feat: update grid_size to [512,512,8] and add 3D neighbor constants + clusters config"
```

---

## Task 2: Engine — field declarations and `__init__` for 3D

**Files:**
- Modify: `src/project/system/taichi_engine.py` (module-level fields section + `__init__`)

> **Warning:** All module-level Taichi fields must be declared at import time. Changes to field shapes require touching declarations, lookup table initialization in `__init__`, and all kernels that reference those fields. This task only updates declarations and `__init__`; kernel updates follow in Tasks 3-7.

### Step 1: Write the failing test

```python
# tests/test_3d_engine_init.py
import pytest, torch

def test_engine_creates_3d_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine
    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        assert engine.energy_field.shape == (32, 32, 4)
        assert engine.grid_node_id.shape == (32, 32, 4)
        assert engine.D == 4
    finally:
        TaichiNeuralEngine._instance = None
        del engine

def test_engine_3d_has_node_pos_z():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_pos_z
    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        assert _node_pos_z.shape[0] > 0
    finally:
        TaichiNeuralEngine._instance = None
        del engine
```

### Step 2: Run tests to verify they fail

```
pytest tests/test_3d_engine_init.py -v
```
Expected: FAIL — `engine.D` attribute missing, `energy_field.shape` is `(32, 32)` not `(32, 32, 4)`, `_node_pos_z` not defined.

### Step 3: Update module-level field declarations

In `taichi_engine.py`, update the **constants** section and **module-level fields**:

```python
MAX_NODES = 2_000_000   # [512, 512, 8] = 2.1M cells; reduced from 4M

# 3D Moore neighbourhood — 26 directions, ordered by (dz, dy, dx)
from project.config import (
    NEIGHBOR_OFFSETS_3D,
    REVERSE_DIRECTION_3D,
    DNA_SLOT_WORD,
    DNA_SLOT_BIT,
    NUM_NEIGHBORS_3D,
    # ... keep existing imports ...
)

# Remove old 2D neighborhood constants:
#   _NEIGHBOR_OFFSETS, _DNA_SHIFTS  (now replaced by NEIGHBOR_OFFSETS_3D / DNA_SLOT_*)
```

In the module-level field declarations block, **replace** the existing fields:

```python
# Node data
_node_state  = ti.field(dtype=ti.i64, shape=MAX_NODES)   # packed binary (no DNA bits)
_node_energy = ti.field(dtype=ti.f32, shape=MAX_NODES)   # working energy
_node_pos_y  = ti.field(dtype=ti.i32, shape=MAX_NODES)   # Y grid coordinate
_node_pos_x  = ti.field(dtype=ti.i32, shape=MAX_NODES)   # X grid coordinate
_node_pos_z  = ti.field(dtype=ti.i32, shape=MAX_NODES)   # Z grid coordinate (NEW)
_node_count  = ti.field(dtype=ti.i32, shape=())
_node_charge = ti.field(dtype=ti.f32, shape=MAX_NODES)
_node_dna    = ti.field(dtype=ti.i64, shape=(MAX_NODES, 3))  # 26×5=130 bits in 3 int64s (NEW)

# Per-step counters (unchanged)
_deaths_count = ti.field(dtype=ti.i32, shape=())
_spawns_count = ti.field(dtype=ti.i32, shape=())

# On-GPU reductions (unchanged)
_dyn_energy_sum = ti.field(dtype=ti.f64, shape=())
_dyn_node_count = ti.field(dtype=ti.i32, shape=())

# Constant lookup tables for 3D neighbour kernel — filled at engine init
_neighbor_dz   = ti.field(dtype=ti.i32, shape=26)   # NEW (replaces old shape=8)
_neighbor_dy   = ti.field(dtype=ti.i32, shape=26)   # extended to 26
_neighbor_dx   = ti.field(dtype=ti.i32, shape=26)   # extended to 26
_reverse_dir   = ti.field(dtype=ti.i32, shape=26)   # extended to 26
_dna_slot_word = ti.field(dtype=ti.i32, shape=26)   # NEW: which int64 word per slot
_dna_slot_bit  = ti.field(dtype=ti.i32, shape=26)   # NEW: bit offset within word

# Remove old: _dna_shifts = ti.field(..., shape=8) — no longer needed
```

### Step 4: Update `TaichiNeuralEngine.__init__`

```python
def __init__(
    self,
    grid_size: Tuple = (512, 512, 8),   # (H, W, D) — D defaults to 8
    # ... rest unchanged ...
):
    # ...
    if len(grid_size) == 2:
        self.H, self.W = grid_size
        self.D = 8
    else:
        self.H, self.W, self.D = grid_size
    self.grid_size = (self.H, self.W, self.D)

    # 3D energy field and grid map
    self.energy_field = torch.zeros(
        self.H, self.W, self.D, dtype=torch.float32, device=self.device
    )
    self.grid_node_id = torch.full(
        (self.H, self.W, self.D), MAX_NODES, dtype=torch.int32, device=self.device
    )

    # Initialize 3D lookup tables
    for n, (dz, dy, dx) in enumerate(NEIGHBOR_OFFSETS_3D):
        _neighbor_dz[n]   = dz
        _neighbor_dy[n]   = dy
        _neighbor_dx[n]   = dx
        _reverse_dir[n]   = REVERSE_DIRECTION_3D[n]
        _dna_slot_word[n] = DNA_SLOT_WORD[n]
        _dna_slot_bit[n]  = DNA_SLOT_BIT[n]

    # Remove the old 2D loop that filled _neighbor_dy[d] for d in range(8)
```

Also update the `grid_operations_per_step` estimate:
```python
# 26 neighbors × 2 DNA reads + energy reads + field ops ≈ 60 per node
self.grid_operations_per_step = self.H * self.W * self.D * 60
```

### Step 5: Run tests

```
pytest tests/test_3d_engine_init.py -v
```
Expected: PASS (fields exist, shapes correct, `engine.D` set)

### Step 6: Commit

```bash
git add src/project/system/taichi_engine.py tests/test_3d_engine_init.py
git commit -m "feat: add 3D Taichi fields (_node_pos_z, _node_dna, 26-neighbor lookup tables)"
```

---

## Task 3: Engine — helper functions and `add_nodes_batch` for 3D

**Files:**
- Modify: `src/project/system/taichi_engine.py` (helper functions + `add_nodes_batch` + `_write_nodes_kernel`)

### Step 1: Write the failing test

```python
# tests/test_3d_add_nodes.py
import pytest, torch

def test_add_node_stores_z_position():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_pos_z, _node_dna

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        ids = engine.add_nodes_batch(
            positions=[(10, 12, 3)],
            energies=[5.0],
            node_types=[1],
        )
        assert _node_pos_z[ids[0]] == 3, f"Expected z=3, got {_node_pos_z[ids[0]]}"
    finally:
        TaichiNeuralEngine._instance = None
        del engine

def test_add_node_stores_dna_in_separate_field():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_state, _node_dna
    from project.config import BINARY_DNA_BASE_SHIFT

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        ids = engine.add_nodes_batch(
            positions=[(5, 5, 1)],
            energies=[10.0],
            node_types=[1],
        )
        nid = ids[0]
        # _node_state should have zero in old DNA bit range (bits 57-18)
        state = int(_node_state[nid])
        dna_bits = (state >> 18) & ((1 << 40) - 1)
        assert dna_bits == 0, f"DNA bits should be 0 in _node_state, got {dna_bits}"
        # _node_dna should be nonzero (random DNA was written)
        dna_w0 = int(_node_dna[nid, 0])
        dna_w1 = int(_node_dna[nid, 1])
        assert (dna_w0 | dna_w1) != 0, "DNA should be nonzero"
    finally:
        TaichiNeuralEngine._instance = None
        del engine
```

### Step 2: Run tests to verify they fail

```
pytest tests/test_3d_add_nodes.py -v
```
Expected: FAIL — positions are still expected as 2-tuples, `_node_pos_z` not written, DNA still in `_node_state`.

### Step 3: Update `_pack_state_batch` — remove DNA

```python
def _pack_state_batch(alive: torch.Tensor, node_type: torch.Tensor,
                      conn_type: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
    """Pack N nodes into int64 binary states — DNA is now in _node_dna field."""
    state = (alive.to(torch.int64) << BINARY_ALIVE_BIT)
    state |= (node_type.to(torch.int64) << BINARY_NODE_TYPE_SHIFT)
    state |= (conn_type.to(torch.int64) << BINARY_CONN_TYPE_SHIFT)
    state |= modality.to(torch.int64)   # bits 2-0
    return state
    # NOTE: No DNA bits written here — DNA lives in _node_dna[MAX_NODES, 3]
```

### Step 4: Replace `_random_dna` with 3D version

```python
def _random_dna_3d(n: int, device: torch.device) -> torch.Tensor:
    """Random 5-bit DNA for 26 neighbours packed into [N, 3] int64 words."""
    slots = torch.randint(0, 32, (n, NUM_NEIGHBORS_3D), device=device, dtype=torch.int64)
    dna = torch.zeros(n, 3, device=device, dtype=torch.int64)
    for slot_n in range(NUM_NEIGHBORS_3D):
        word = DNA_SLOT_WORD[slot_n]
        bit  = DNA_SLOT_BIT[slot_n]
        dna[:, word] |= (slots[:, slot_n] << bit)
    return dna  # shape [N, 3]
```

### Step 5: Add `_write_dna_kernel` and update `_write_nodes_kernel`

```python
@ti.kernel
def _write_nodes_kernel(
    states:   ti.types.ndarray(),
    energies: ti.types.ndarray(),
    pos_y:    ti.types.ndarray(),
    pos_x:    ti.types.ndarray(),
    pos_z:    ti.types.ndarray(),   # NEW
    start:    int,
    n:        int,
):
    """Bulk-write N new nodes starting at slot `start`."""
    for i in range(n):
        _node_state[start + i]  = states[i]
        _node_energy[start + i] = energies[i]
        _node_pos_y[start + i]  = pos_y[i]
        _node_pos_x[start + i]  = pos_x[i]
        _node_pos_z[start + i]  = pos_z[i]   # NEW
        _node_charge[start + i] = 0.0


@ti.kernel
def _write_dna_kernel(
    dna_data: ti.types.ndarray(),   # shape [N, 3] int64
    start: int,
    n: int,
):
    """Write DNA words for N nodes starting at slot `start`."""
    for i in range(n):
        _node_dna[start + i, 0] = dna_data[i, 0]
        _node_dna[start + i, 1] = dna_data[i, 1]
        _node_dna[start + i, 2] = dna_data[i, 2]
```

### Step 6: Update `add_nodes_batch`

Change the signature and body to handle 3-tuples:

```python
def add_nodes_batch(
    self,
    positions:  List[Tuple],           # (y, x, z) — 3-tuples; (y, x) still accepted (z=0)
    energies:   List[float],
    node_types: List[int],
    modalities: Optional[List[int]] = None,
) -> List[int]:
    n = len(positions)
    if n == 0:
        return []

    # Unpack positions — support both 2-tuple (y, x) and 3-tuple (y, x, z)
    ys, xs, zs = [], [], []
    for p in positions:
        if len(p) == 3:
            ys.append(p[0]); xs.append(p[1]); zs.append(p[2])
        else:
            ys.append(p[0]); xs.append(p[1]); zs.append(0)

    dev = self.device
    new_ys     = torch.tensor(ys, device=dev, dtype=torch.int32) % self.H
    new_xs     = torch.tensor(xs, device=dev, dtype=torch.int32) % self.W
    new_zs     = torch.tensor(zs, device=dev, dtype=torch.int32) % self.D   # NEW
    new_e      = torch.tensor(energies, device=dev, dtype=torch.float32)
    new_types  = torch.tensor(node_types, device=dev, dtype=torch.int64)
    new_conn   = _random_conn_type(n, dev)
    new_dna    = _random_dna_3d(n, dev)    # [N, 3] int64

    if modalities is not None:
        new_modality = torch.tensor(modalities, device=dev, dtype=torch.int64)
    else:
        new_modality = torch.zeros(n, device=dev, dtype=torch.int64)

    # Workspace nodes: excitatory, max DNA (all PARAM=15, MODE=0 → value=15)
    ws_mask = (new_types == 2)
    if ws_mask.any():
        new_conn[ws_mask] = 0    # excitatory
        # Pack value=15 (0b01111) into all 26 slots for workspace nodes
        ws_dna = torch.zeros(n, 3, device=dev, dtype=torch.int64)
        for slot_n in range(NUM_NEIGHBORS_3D):
            word = DNA_SLOT_WORD[slot_n]
            bit  = DNA_SLOT_BIT[slot_n]
            ws_dna[:, word] |= torch.tensor(15, dtype=torch.int64, device=dev) << bit
        new_dna[ws_mask] = ws_dna[ws_mask]

    alive_bits = torch.ones(n, device=dev, dtype=torch.int64)
    new_state = _pack_state_batch(alive_bits, new_types, new_conn, new_modality)

    start = self._count
    end   = start + n
    if end > MAX_NODES:
        logger.error("add_nodes_batch: truncating to MAX_NODES")
        n = MAX_NODES - start
        end = MAX_NODES
        if n <= 0:
            return []
        new_state = new_state[:n]; new_e = new_e[:n]
        new_ys = new_ys[:n]; new_xs = new_xs[:n]; new_zs = new_zs[:n]
        new_dna = new_dna[:n]

    _write_nodes_kernel(new_state, new_e, new_ys, new_xs, new_zs, start, n)
    _write_dna_kernel(new_dna.contiguous(), start, n)

    _node_count[None] = end
    self._count = end

    # Stamp initial energy into 3D field
    self.energy_field[new_ys.long(), new_xs.long(), new_zs.long()] = torch.maximum(
        self.energy_field[new_ys.long(), new_xs.long(), new_zs.long()], new_e
    )

    if 2 in node_types:
        with self._workspace_cache_lock:
            self._workspace_cache_valid = False

    return list(range(start, end))
```

Also update the single-node wrapper:
```python
def add_node(self, position: Tuple, energy: float, node_type: int = 1) -> int:
    return self.add_nodes_batch([position], [energy], [node_type])[0]
```

### Step 7: Run tests

```
pytest tests/test_3d_add_nodes.py -v
```
Expected: PASS

### Step 8: Commit

```bash
git add src/project/system/taichi_engine.py tests/test_3d_add_nodes.py
git commit -m "feat: add_nodes_batch accepts (y,x,z) positions, DNA moves to _node_dna field"
```

---

## Task 4: Engine — update grid map kernels for 3D

**Files:**
- Modify: `src/project/system/taichi_engine.py` (`_clear_grid_map`, `_build_grid_map`)

### Step 1: Write the failing test

```python
# tests/test_3d_grid_map.py
import pytest, torch

def test_grid_map_places_node_at_correct_z():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, MAX_NODES

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        ids = engine.add_nodes_batch(
            positions=[(10, 12, 3)],
            energies=[5.0],
            node_types=[1],
        )
        # Force a grid rebuild by running step (grid is rebuilt each step)
        # Or call kernels directly:
        from project.system.taichi_engine import _clear_grid_map, _build_grid_map
        _clear_grid_map(engine.grid_node_id)
        _build_grid_map(engine.grid_node_id)

        # grid_node_id[10, 12, 3] should contain the node ID
        val = int(engine.grid_node_id[10, 12, 3])
        assert val == ids[0], f"Expected node {ids[0]} at [10,12,3], got {val}"
        # Other Z layers should be empty (MAX_NODES)
        for z in range(4):
            if z != 3:
                val_other = int(engine.grid_node_id[10, 12, z])
                assert val_other == MAX_NODES, f"Expected empty at z={z}"
    finally:
        TaichiNeuralEngine._instance = None
        del engine
```

### Step 2: Run test to verify it fails

```
pytest tests/test_3d_grid_map.py -v
```
Expected: FAIL — grid_node_id is `[32, 32]` not `[32, 32, 4]`, kernels index with 2D.

### Step 3: Update `_clear_grid_map`

```python
@ti.kernel
def _clear_grid_map(grid_node_id: ti.types.ndarray()):
    """Fill grid_node_id with MAX_NODES (no valid node). Works for any shape."""
    for y, x, z in ti.ndrange(
        grid_node_id.shape[0], grid_node_id.shape[1], grid_node_id.shape[2]
    ):
        grid_node_id[y, x, z] = MAX_NODES
```

### Step 4: Update `_build_grid_map`

```python
@ti.kernel
def _build_grid_map(grid_node_id: ti.types.ndarray()):
    """Scatter alive nodes into the 3D grid map. Lowest node ID wins per cell."""
    for i in range(_node_count[None]):
        if _node_state[i] != 0:
            ti.atomic_min(
                grid_node_id[_node_pos_y[i], _node_pos_x[i], _node_pos_z[i]], i
            )
```

### Step 5: Run test

```
pytest tests/test_3d_grid_map.py -v
```
Expected: PASS

### Step 6: Commit

```bash
git add src/project/system/taichi_engine.py tests/test_3d_grid_map.py
git commit -m "feat: update _clear_grid_map and _build_grid_map for 3D [H,W,D] grid"
```

---

## Task 5: Engine — update `_dna_transfer_kernel` for 26 neighbours

**Files:**
- Modify: `src/project/system/taichi_engine.py` (`_dna_transfer_kernel`, `_sync_energy_from_field`, `_clamp_kernel`)

### Step 1: Write the failing test

```python
# tests/test_3d_dna_transfer.py
import pytest, torch

def test_energy_transfers_across_z_layers():
    """A node at z=0 should transfer energy to a node at z=1."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4), transfer_dt=0.1)
    try:
        # Sender at (10, 10, 0) — high energy
        # Receiver at (10, 10, 1) — zero energy
        ids = engine.add_nodes_batch(
            positions=[(10, 10, 0), (10, 10, 1)],
            energies=[100.0, 0.0],
            node_types=[1, 2],  # workspace = max compatibility
        )
        # Run one step
        engine.step()
        # Receiver should have gained energy from sender
        e_recv = float(engine.energy_field[10, 10, 1])
        assert e_recv > 0.0, f"Energy at z=1 should be positive after transfer, got {e_recv}"
    finally:
        TaichiNeuralEngine._instance = None
        del engine
```

### Step 2: Run test to verify it fails

```
pytest tests/test_3d_dna_transfer.py -v
```
Expected: FAIL or ERROR — kernel still uses 2D energy_field indexing.

### Step 3: Rewrite `_dna_transfer_kernel`

Replace the entire kernel with the 3D version:

```python
@ti.kernel
def _dna_transfer_kernel(
    energy_field:  ti.types.ndarray(),   # [H, W, D] float32
    grid_node_id:  ti.types.ndarray(),   # [H, W, D] int32
    H: int, W: int, D: int,              # grid dimensions (D is new)
    dt: float,
    gate_threshold: float,
    frame: int,
    strength: float,
):
    """
    3D DNA transfer: 26-neighbour Moore neighbourhood.

    DNA is read from _node_dna[i, word][bit:bit+5] using precomputed
    _dna_slot_word and _dna_slot_bit lookup tables.
    Lock-and-key and connection type logic unchanged from 2D version.
    """
    for i in range(_node_count[None]):
        state = _node_state[i]
        if state == 0:
            continue

        py = _node_pos_y[i]
        px = _node_pos_x[i]
        pz = _node_pos_z[i]
        energy = _node_energy[i]
        conn_type = int((state >> 58) & 7)
        gate_fire = 1.0 if energy > gate_threshold else 0.0
        damping = ti.exp(-ti.abs(energy) / 50.0)

        total = 0.0
        for d in ti.static(range(26)):
            ny = (py + _neighbor_dy[d] + H) % H
            nx = (px + _neighbor_dx[d] + W) % W
            nz = (pz + _neighbor_dz[d] + D) % D

            # Read my DNA for this direction from _node_dna
            my_word = _dna_slot_word[d]
            my_bit  = _dna_slot_bit[d]
            my_dna_raw = int((_node_dna[i, my_word] >> my_bit) & 31)
            my_mode  = (my_dna_raw >> 4) & 1
            my_param = my_dna_raw & 15

            # Lock-and-key: look up neighbour's DNA for reverse direction
            neighbor_id = grid_node_id[ny, nx, nz]
            compat = 0.0
            if neighbor_id < MAX_NODES:
                n_state = _node_state[neighbor_id]
                if n_state != 0:
                    rd = _reverse_dir[d]
                    their_word = _dna_slot_word[rd]
                    their_bit  = _dna_slot_bit[rd]
                    their_dna_raw = int((_node_dna[neighbor_id, their_word] >> their_bit) & 31)
                    their_param = their_dna_raw & 15
                    compat = float(my_param & their_param) / 15.0

            if compat > 0.0:
                delta = energy_field[ny, nx, nz] - energy

                dna_strength = compat
                skip = 0
                if my_mode == 1:
                    special = (my_param >> 2) & 3
                    sparam  = my_param & 3
                    if special == 0:   # THRESHOLD
                        if ti.abs(delta) <= float(sparam) * 16.0:
                            skip = 1
                    elif special == 1: # INVERT
                        delta = -delta
                        dna_strength = float(sparam) / 3.0
                    elif special == 2: # PULSE
                        if frame % (sparam + 1) != 0:
                            skip = 1
                    elif special == 3: # ABSORB
                        if delta < 0.0:
                            skip = 1
                        dna_strength = float(sparam) / 3.0

                if skip == 0:
                    contrib = 0.0
                    if conn_type == 0:
                        contrib = delta * dna_strength * strength * dt
                    elif conn_type == 1:
                        contrib = -delta * dna_strength * strength * dt
                    elif conn_type == 2:
                        contrib = delta * dna_strength * strength * dt * gate_fire
                    elif conn_type == 3:
                        contrib = dna_strength * strength * dt
                    elif conn_type == 4:
                        contrib = -dna_strength * strength * dt
                    elif conn_type == 5:
                        contrib = delta * dna_strength * strength * dt * damping
                    elif conn_type == 6:
                        osc = ti.sin(float(frame) * dna_strength * ti.math.pi)
                        contrib = delta * osc * strength * dt
                    elif conn_type == 7:
                        ti.atomic_add(_node_charge[i], ti.abs(delta) * dna_strength * dt)
                        contrib = 0.0

                    contrib = ti.min(ti.max(contrib, -100.0), 100.0)
                    if contrib != 0.0:
                        total += contrib
                        ti.atomic_sub(energy_field[ny, nx, nz], contrib)

        # Capacitive burst
        if conn_type == 7 and _node_charge[i] > gate_threshold:
            burst = _node_charge[i]
            _node_charge[i] = 0.0
            total += burst

        if total != 0.0:
            ti.atomic_add(energy_field[py, px, pz], total)
```

### Step 4: Update `_sync_energy_from_field` for 3D

```python
@ti.kernel
def _sync_energy_from_field(energy_field: ti.types.ndarray(), grid_node_id: ti.types.ndarray()):
    """Sync each grid-map-owner node's energy from the 3D energy field."""
    for i in range(_node_count[None]):
        if _node_state[i] != 0:
            y = _node_pos_y[i]
            x = _node_pos_x[i]
            z = _node_pos_z[i]
            if grid_node_id[y, x, z] == i:
                _node_energy[i] = energy_field[y, x, z]
```

### Step 5: Update `_clamp_kernel` for 3D

```python
@ti.kernel
def _clamp_kernel(energy_field: ti.types.ndarray(), lo: float, hi: float):
    """Clamp every cell of 3D energy_field to [lo, hi]."""
    for y, x, z in ti.ndrange(
        energy_field.shape[0], energy_field.shape[1], energy_field.shape[2]
    ):
        v = energy_field[y, x, z]
        if v < lo:
            energy_field[y, x, z] = lo
        elif v > hi:
            energy_field[y, x, z] = hi
```

### Step 6: Update `step()` call to pass D

In `TaichiNeuralEngine.step()`, update the `_dna_transfer_kernel` call:

```python
_dna_transfer_kernel(
    self.energy_field,
    self.grid_node_id,
    self.H, self.W, self.D,   # add self.D
    self.transfer_dt,
    self.gate_threshold,
    self.frame_counter,
    self.transfer_strength,
)
```

### Step 7: Run test

```
pytest tests/test_3d_dna_transfer.py -v
```
Expected: PASS

### Step 8: Commit

```bash
git add src/project/system/taichi_engine.py tests/test_3d_dna_transfer.py
git commit -m "feat: rewrite _dna_transfer_kernel for 26-neighbour 3D Moore neighbourhood"
```

---

## Task 6: Engine — update `_spawn_kernel` for 3D

**Files:**
- Modify: `src/project/system/taichi_engine.py` (`_spawn_kernel`)

### Step 1: Write the failing test

```python
# tests/test_3d_spawn.py
import pytest, torch

def test_spawn_creates_child_with_z_coordinate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_pos_z

    engine = TaichiNeuralEngine(grid_size=(32, 32, 8), node_spawn_threshold=5.0)
    try:
        initial_count = engine._count
        ids = engine.add_nodes_batch(
            positions=[(10, 10, 4)],
            energies=[100.0],
            node_types=[1],
        )
        engine.step()
        # If spawning happened, count should have increased
        new_count = engine._count
        if new_count > initial_count + 1:
            # Check all dynamic nodes have z in [0, 8)
            for nid in range(new_count):
                from project.system.taichi_engine import _node_state
                if int(_node_state[nid]) != 0:
                    z = int(_node_pos_z[nid])
                    assert 0 <= z < 8, f"Node {nid} has z={z} out of range"
    finally:
        TaichiNeuralEngine._instance = None
        del engine
```

### Step 2: Run test to verify it fails

```
pytest tests/test_3d_spawn.py -v
```
Expected: FAIL — children spawned without Z coordinate, or Z out of bounds.

### Step 3: Rewrite `_spawn_kernel`

```python
@ti.kernel
def _spawn_kernel(
    H: int, W: int, D: int,            # grid dimensions — D is new
    spawn_threshold: float,
    spawn_cost: float,
    child_energy: float,
    max_spawns: int,
    n_existing: int,
):
    for i in range(n_existing):
        state = _node_state[i]
        if state == 0:
            continue
        node_type = int((state >> 61) & 3)
        if node_type != 1:
            continue
        if _node_energy[i] < spawn_threshold:
            continue

        my_spawn = ti.atomic_add(_spawns_count[None], 1)
        if my_spawn >= max_spawns:
            ti.atomic_sub(_spawns_count[None], 1)
            continue

        slot = ti.atomic_add(_node_count[None], 1)
        if slot >= MAX_NODES:
            ti.atomic_sub(_node_count[None], 1)
            ti.atomic_sub(_spawns_count[None], 1)
            continue

        # Random ±1 offset in all three axes (toroidal wrap)
        offset_y = int(ti.random() * 3.0) - 1
        offset_x = int(ti.random() * 3.0) - 1
        offset_z = int(ti.random() * 3.0) - 1   # NEW
        child_y  = (_node_pos_y[i] + offset_y + H) % H
        child_x  = (_node_pos_x[i] + offset_x + W) % W
        child_z  = (_node_pos_z[i] + offset_z + D) % D   # NEW

        # Inherit DNA from parent with 10% per-slot mutation (26 slots)
        for d in ti.static(range(26)):
            word = _dna_slot_word[d]
            bit  = _dna_slot_bit[d]
            parent_dna = int((_node_dna[i, word] >> bit) & 31)
            if ti.random() < 0.1:
                flip_bit = ti.min(int(ti.random() * 5.0), 4)
                parent_dna = parent_dna ^ (1 << flip_bit)
            # Clear the slot in child DNA word, then write inherited value
            mask = ti.i64(31) << bit
            _node_dna[slot, word] = (_node_dna[slot, word] & ~mask) | (ti.i64(parent_dna) << bit)

        conn_type = ti.min(int(ti.random() * 8.0), 7)
        parent_modality = state & ti.i64(7)
        new_state = (
            (ti.i64(1) << 63)
          | (ti.i64(1) << 61)
          | (ti.i64(conn_type) << 58)
          | parent_modality
        )

        _node_state[slot]  = new_state
        _node_energy[slot] = child_energy
        _node_pos_y[slot]  = child_y
        _node_pos_x[slot]  = child_x
        _node_pos_z[slot]  = child_z   # NEW
        _node_charge[slot] = 0.0

        _node_energy[i] -= spawn_cost
```

> **Note on DNA inheritance in `_spawn_kernel`:** The loop writes slot-by-slot using atomic bit operations. The child slot was freshly allocated (all zeros from field init), so the `& ~mask` clears only the bits for this slot before writing the inherited value. This avoids race conditions because only one parent writes to each new slot.

### Step 4: Update `step()` to pass D to `_spawn_kernel`

```python
_spawn_kernel(
    self.H, self.W, self.D,   # add self.D
    effective_spawn_threshold, effective_spawn_cost, effective_child_energy,
    self._spawn_limit(),
    self._count,
)
```

### Step 5: Run test

```
pytest tests/test_3d_spawn.py -v
```
Expected: PASS

### Step 6: Commit

```bash
git add src/project/system/taichi_engine.py tests/test_3d_spawn.py
git commit -m "feat: extend _spawn_kernel to XYZ ±1 drift with 26-slot DNA inheritance"
```

---

## Task 7: Engine — update `_inject_sensory_kernel` and `_death_kernel` for 3D

**Files:**
- Modify: `src/project/system/taichi_engine.py`

### Step 1: Update `_inject_sensory_kernel`

The kernel signature gains a `z` parameter to inject into a specific Z layer:

```python
@ti.kernel
def _inject_sensory_kernel(
    energy_field: ti.types.ndarray(),
    data:         ti.types.ndarray(),   # [H, W] float32
    y0: int, x0: int, h: int, w: int,
    z: int,                              # NEW: target Z layer
):
    """Write data values as energy into a specific Z layer of energy_field."""
    for dy, dx in ti.ndrange(h, w):
        energy_field[y0 + dy, x0 + dx, z] = data[dy, dx]
```

Update `inject_sensory_data` method signature to accept a `z` parameter:

```python
def inject_sensory_data(
    self,
    pixel_data: torch.Tensor,
    region: Tuple[int, int, int, int],
    z: int = 0,   # NEW: which Z layer to inject into
) -> None:
    y0, y1, x0, x1 = region
    h = min(y1 - y0, pixel_data.shape[0])
    w = min(x1 - x0, pixel_data.shape[1])
    data = pixel_data[:h, :w].to(dtype=torch.float32, device=self.device).contiguous()
    _inject_sensory_kernel(self.energy_field, data, y0, x0, h, w, z)
    self._injection_counter += 1
```

### Step 2: Update `_death_kernel` — clear `_node_dna` on death

Add DNA clearing to the death logic:

```python
@ti.kernel
def _death_kernel(death_threshold: float):
    """Kill dynamic nodes below threshold; also clear their DNA."""
    for i in range(_node_count[None]):
        state = _node_state[i]
        if state == 0:
            continue
        node_type = int((state >> 61) & 3)
        if node_type == 1 and _node_energy[i] < death_threshold:
            _node_state[i]  = 0
            _node_energy[i] = 0.0
            _node_charge[i] = 0.0
            _node_dna[i, 0] = 0    # NEW: clear DNA on death
            _node_dna[i, 1] = 0
            _node_dna[i, 2] = 0
            ti.atomic_add(_deaths_count[None], 1)
```

### Step 3: Write a quick test for Z-layer injection

```python
# tests/test_3d_injection.py
import pytest, torch

def test_inject_targets_correct_z_layer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        data = torch.ones(32, 32, dtype=torch.float32, device=engine.device) * 42.0
        engine.inject_sensory_data(data, region=(0, 32, 0, 32), z=2)

        # Z=2 should be 42.0
        val_z2 = float(engine.energy_field[5, 5, 2])
        # Z=0 should be 0.0
        val_z0 = float(engine.energy_field[5, 5, 0])
        assert abs(val_z2 - 42.0) < 0.01, f"Expected 42.0 at z=2, got {val_z2}"
        assert abs(val_z0) < 0.01, f"Expected 0.0 at z=0, got {val_z0}"
    finally:
        TaichiNeuralEngine._instance = None
        del engine
```

### Step 4: Run tests

```
pytest tests/test_3d_injection.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add src/project/system/taichi_engine.py tests/test_3d_injection.py
git commit -m "feat: update inject_sensory_kernel for Z-layer targeting, clear DNA on death"
```

---

## Task 8: Fix existing tests for 3D API

**Files:**
- Modify: `tests/test_modality_engine.py`
- Modify: `tests/test_workspace_system.py` (if it adds nodes)
- Modify: other tests that reference grid_size or node positions

### Step 1: Run the full test suite and identify failures

```
pytest tests/ -v 2>&1 | grep FAIL
```

### Step 2: Update `test_modality_engine.py`

In both test functions, `TaichiNeuralEngine(grid_size=(64, 64))` stays valid (backward compat handles it). But positions need updating:

```python
# Before:  positions=[(60, 10), (60, 20)]
# After:   positions=[(60, 10, 0), (60, 20, 0)]  — explicit Z=0
ids = engine.add_nodes_batch(
    positions=[(60, 10, 0), (60, 20, 0)],
    energies=[10.0, 10.0],
    node_types=[2, 2],
    modalities=[MODALITY_VISUAL, MODALITY_AUDIO_LEFT],
)
```

Also: `engine.energy_field` stamping now uses 3D indexing — ensure any test that reads `energy_field[y, x]` is updated to `energy_field[y, x, z]`.

### Step 3: Update any other failing tests

For each failing test: update 2-tuple positions to 3-tuples, update `energy_field` indexing to 3D, and update `grid_node_id` indexing to 3D. Pattern:

```python
# OLD: engine.energy_field[y, x]
# NEW: engine.energy_field[y, x, 0]   (or appropriate z)

# OLD: engine.grid_node_id[y, x]
# NEW: engine.grid_node_id[y, x, 0]
```

### Step 4: Run the full test suite

```
pytest tests/ -v
```
Expected: All existing + new tests PASS.

### Step 5: Commit

```bash
git add tests/
git commit -m "fix: update existing tests for 3D node positions and energy_field indexing"
```

---

## Task 9: `main.py` — replace band regions with cluster initialization

**Files:**
- Modify: `src/project/main.py`

### Step 1: Write a test for cluster placement

```python
# tests/test_cluster_init.py
import pytest, torch

def test_clusters_produce_nodes_at_scattered_positions():
    """Sensory and workspace nodes should not all be at Z=0."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Import the cluster placement helper directly
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from project.main import _place_clusters

    positions = _place_clusters(
        grid_H=32, grid_W=32, grid_D=4,
        count=4, nodes_each=5, radius=2, min_separation=6,
    )
    # Should have 4 clusters × 5 nodes = 20 positions (approximately)
    assert len(positions) > 0
    # All positions in bounds
    for y, x, z in positions:
        assert 0 <= y < 32
        assert 0 <= x < 32
        assert 0 <= z < 4
    # Not all at Z=0 (with 4 clusters and D=4, at least one cluster will be at z>0)
    z_values = {z for _, _, z in positions}
    assert len(z_values) > 1, f"Expected multiple Z values, got {z_values}"
```

### Step 2: Run test to verify it fails

```
pytest tests/test_cluster_init.py -v
```
Expected: FAIL — `_place_clusters` does not exist.

### Step 3: Add `_place_clusters` helper to `main.py`

Add this function near the top of `main.py` (before the class definition):

```python
import random
import math

def _place_clusters(
    grid_H: int, grid_W: int, grid_D: int,
    count: int, nodes_each: int, radius: int, min_separation: int,
) -> list:
    """
    Place `count` node clusters scattered through a [H, W, D] 3D volume.

    Each cluster has `nodes_each` nodes within a sphere of `radius` cells.
    Centers are chosen at random with a minimum pairwise distance of
    `min_separation` cells. Returns a flat list of (y, x, z) tuples.

    Placement is random; run a few times if the first attempt fails
    (can happen when count × radius² exceeds grid size).
    """
    centers = []
    attempts = 0
    while len(centers) < count and attempts < count * 1000:
        attempts += 1
        cy = random.randint(0, grid_H - 1)
        cx = random.randint(0, grid_W - 1)
        cz = random.randint(0, grid_D - 1)
        # Check minimum separation from existing centers
        too_close = any(
            math.sqrt((cy - oy)**2 + (cx - ox)**2 + (cz - oz)**2) < min_separation
            for oy, ox, oz in centers
        )
        if not too_close:
            centers.append((cy, cx, cz))

    positions = []
    for cy, cx, cz in centers:
        placed = 0
        node_attempts = 0
        while placed < nodes_each and node_attempts < nodes_each * 100:
            node_attempts += 1
            dy = random.randint(-radius, radius)
            dx = random.randint(-radius, radius)
            dz = random.randint(-radius, radius)
            if dy**2 + dx**2 + dz**2 <= radius**2:
                ny = max(0, min(grid_H - 1, cy + dy))
                nx = max(0, min(grid_W - 1, cx + dx))
                nz = max(0, min(grid_D - 1, cz + dz))
                positions.append((ny, nx, nz))
                placed += 1

    return positions
```

### Step 4: Update main initialization to use clusters

Find the existing region seeding code in `main.py` (around lines 580-625 where it seeds sensory/dynamic/workspace nodes). Replace with cluster-based init:

```python
# NEW cluster-based initialization
H, W, D = grid_size  # now 3 elements

clusters_cfg = config_manager.get('clusters', {})
s_count    = clusters_cfg.get('sensory_count', 10)
s_each     = clusters_cfg.get('sensory_nodes_each', 30)
s_radius   = clusters_cfg.get('sensory_radius', 3)
ws_count   = clusters_cfg.get('workspace_count', 8)
ws_each    = clusters_cfg.get('workspace_nodes_each', 32)
ws_radius  = clusters_cfg.get('workspace_radius', 3)
min_sep    = clusters_cfg.get('min_cluster_separation', 8)

# Place sensory clusters
sensory_positions = _place_clusters(H, W, D, s_count, s_each, s_radius, min_sep)
if sensory_positions:
    engine.add_nodes_batch(
        positions=sensory_positions,
        energies=[10.0] * len(sensory_positions),
        node_types=[NODE_TYPE_SENSORY] * len(sensory_positions),
        modalities=[MODALITY_VISUAL] * len(sensory_positions),
    )

# Place workspace clusters (enforce separation from sensory too if desired)
ws_positions = _place_clusters(H, W, D, ws_count, ws_each, ws_radius, min_sep)
if ws_positions:
    engine.add_nodes_batch(
        positions=ws_positions,
        energies=[5.0] * len(ws_positions),
        node_types=[NODE_TYPE_WORKSPACE] * len(ws_positions),
        modalities=[MODALITY_VISUAL] * len(ws_positions),
    )

# Seed dynamic nodes: one per (x, z) column at mid-Y
mid_y = H // 2
dynamic_positions = [(mid_y, x, z) for z in range(D) for x in range(W)]
engine.add_nodes_batch(
    positions=dynamic_positions[:min(len(dynamic_positions), 10_000)],
    energies=[15.0] * min(len(dynamic_positions), 10_000),
    node_types=[NODE_TYPE_DYNAMIC] * min(len(dynamic_positions), 10_000),
)
```

### Step 5: Update grid_size usage

Everywhere `main.py` unpacks `grid_size` as `(H, W)`, update to `(H, W, D)`:
```python
# Before: H, W = grid_size
# After:  H, W, D = grid_size  (or:  H, W = grid_size[0], grid_size[1])
```

### Step 6: Run test

```
pytest tests/test_cluster_init.py -v
```
Expected: PASS

### Step 7: Commit

```bash
git add src/project/main.py tests/test_cluster_init.py
git commit -m "feat: replace band regions with 3D island cluster initialization in main.py"
```

---

## Task 10: Update sensory energy injection for clusters

**Files:**
- Modify: `src/project/main.py` (the per-frame sensory injection loop)

### Step 1: Find and update the sensory injection call

Currently `main.py` calls `engine.inject_sensory_data(pixel_data, region=(0, 1080, 0, 1920))` (or similar). Replace with per-cluster injection.

Add a `_cluster_centers` list stored at init time (store alongside sensory positions), then inject at each cluster's Z layer each frame:

```python
# At init time, also store sensory cluster centers:
self._sensory_cluster_centers = []   # list of (cy, cx, cz)
# ... populate when calling _place_clusters ...
# Store the centers (first `s_count` elements of sensory_positions if using cluster center logic)
# Or compute centers separately:
self._sensory_cluster_centers = centers_from_place_clusters(...)  # track centers

# Per-frame injection (in the update/step loop):
if pixel_data is not None:
    # Downsample from 1920×1080 to 512×512
    import torch.nn.functional as F
    small = F.interpolate(
        pixel_data.unsqueeze(0).unsqueeze(0),
        size=(self.H, self.W), mode='bilinear', align_corners=False
    ).squeeze()
    for cy, cx, cz in self._sensory_cluster_centers:
        r = s_radius
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        patch = small[y0:y1, x0:x1]
        engine.inject_sensory_data(patch, region=(y0, y1, x0, x1), z=cz)
```

> **Simplification note:** The `_place_clusters` function doesn't separately return centers. Modify it to return both centers and node positions, or add a companion `_cluster_centers` helper. Keep it simple — refactor `_place_clusters` to return `(centers, positions)`.

### Step 2: Update `_place_clusters` to return centers

```python
def _place_clusters(
    grid_H, grid_W, grid_D, count, nodes_each, radius, min_separation
) -> tuple:   # returns (centers, positions)
    centers = [...]   # same logic as before
    positions = [...]
    return centers, positions   # return both
```

Update call sites accordingly.

### Step 3: Write a simple smoke test

```python
# tests/test_cluster_injection.py
import pytest, torch

def test_per_cluster_injection_updates_specific_z():
    """Injecting into cluster centers should update energy at the cluster's Z layer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        data = torch.ones(4, 4, dtype=torch.float32, device=engine.device) * 99.0
        engine.inject_sensory_data(data, region=(14, 18, 14, 18), z=2)
        val = float(engine.energy_field[15, 15, 2])
        assert val > 0.0, f"Expected energy at cluster Z=2, got {val}"
        assert float(engine.energy_field[15, 15, 0]) == 0.0
    finally:
        TaichiNeuralEngine._instance = None
        del engine
```

### Step 4: Run test

```
pytest tests/test_cluster_injection.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add src/project/main.py tests/test_cluster_injection.py
git commit -m "feat: per-cluster sensory injection with Z-layer targeting"
```

---

## Task 11: Update workspace system for 3D coordinates

**Files:**
- Modify: `src/project/workspace/workspace_node.py`
- Modify: `src/project/workspace/workspace_system.py`
- Modify: `src/project/workspace/mapping.py`

### Step 1: Update `workspace_node.py`

Add `z` coordinate to the `WorkspaceNode` dataclass or class:

```python
# Find the WorkspaceNode class and add z field.
# Before:
class WorkspaceNode:
    def __init__(self, node_id, y, x, ...):
        self.y = y
        self.x = x
# After:
class WorkspaceNode:
    def __init__(self, node_id, y, x, z=0, ...):
        self.y = y
        self.x = x
        self.z = z
```

### Step 2: Update `workspace_system.py`

The system currently places workspace nodes at a fixed 16×16 bottom strip. Update to use cluster-provided positions:

```python
# Accept pre-computed positions from main.py cluster placement.
# The workspace system is initialized with (y, x, z) tuples.
class WorkspaceNodeSystem:
    def __init__(self, positions: list, engine, ...):
        self.nodes = [
            WorkspaceNode(node_id=i, y=y, x=x, z=z)
            for i, (y, x, z) in enumerate(positions)
        ]
```

Remove hardcoded `workspace_height = 16`, `workspace_width = 16` region logic.

### Step 3: Update `mapping.py`

The sensory→workspace mapping currently uses 2D ratio-based aggregation. Update to 3D:
- Workspace nodes now have (y, x, z) positions
- Sensory clusters have (y, x, z) positions
- Keep ratio-based mapping but extend to include Z proximity

For minimal change: map by (y, x) ratio as before, ignoring Z. If workspace nodes and sensory nodes happen to be at similar Z levels, energy flows naturally. This is the simplest correct behavior.

### Step 4: Write a test

```python
# tests/test_3d_workspace_system.py
import pytest

def test_workspace_system_stores_z():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from project.workspace.workspace_system import WorkspaceNodeSystem
    from project.workspace.workspace_node import WorkspaceNode

    positions = [(10, 5, 3), (20, 15, 1)]
    system = WorkspaceNodeSystem(positions=positions, engine=None)
    assert system.nodes[0].z == 3
    assert system.nodes[1].z == 1
```

### Step 5: Run test

```
pytest tests/test_3d_workspace_system.py -v
```
Expected: PASS

### Step 6: Commit

```bash
git add src/project/workspace/ tests/test_3d_workspace_system.py
git commit -m "feat: workspace system and nodes support 3D (y, x, z) positions"
```

---

## Task 12: Visualization — max-pool Z projection

**Files:**
- Modify: `src/project/workspace/visualization.py` (or wherever GGUI rendering happens)
- Modify: `src/project/ui/taichi_gui_manager.py` (TaichiGUIManager energy display)

### Step 1: Find where `energy_field` is read for display

Grep for `energy_field` in the visualization/GGUI files:

```
grep -r "energy_field" src/project/workspace/ src/project/ui/
```

### Step 2: Add `project_energy_field_to_2d` helper

Add this PyTorch helper in `visualization.py` (or equivalent):

```python
def project_energy_field_to_2d(energy_field: torch.Tensor) -> torch.Tensor:
    """
    Max-pool a 3D energy_field [H, W, D] to a 2D display image [H, W].
    Each pixel shows the brightest Z layer — preserves visual feel of the 2D version.
    """
    if energy_field.dim() == 2:
        return energy_field   # backward compat for 2D fields
    return energy_field.max(dim=2).values   # [H, W]
```

### Step 3: Update GGUI render calls

Find where `self.energy_field` (or `engine.energy_field`) is passed to GGUI for display.
Add the projection call before rendering:

```python
# Before: display_img = engine.energy_field
# After:
display_img = project_energy_field_to_2d(engine.energy_field)
# Then pass display_img to the GGUI window
```

### Step 4: Write a test

```python
# tests/test_projection.py
import torch
from project.workspace.visualization import project_energy_field_to_2d

def test_max_pool_projection():
    field = torch.zeros(8, 8, 4)
    field[3, 3, 2] = 99.0
    proj = project_energy_field_to_2d(field)
    assert proj.shape == (8, 8)
    assert float(proj[3, 3]) == 99.0
    assert float(proj[0, 0]) == 0.0

def test_2d_passthrough():
    field = torch.ones(8, 8)
    proj = project_energy_field_to_2d(field)
    assert proj.shape == (8, 8)
```

### Step 5: Run test

```
pytest tests/test_projection.py -v
```
Expected: PASS

### Step 6: Commit

```bash
git add src/project/workspace/ src/project/ui/ tests/test_projection.py
git commit -m "feat: max-pool Z projection for 3D energy_field visualization"
```

---

## Task 13: End-to-end smoke test

**Files:**
- Create: `tests/test_3d_end_to_end.py`

### Step 1: Write the end-to-end test

```python
# tests/test_3d_end_to_end.py
import pytest, torch

def test_3d_simulation_runs_10_steps():
    """Full simulation: clusters initialized, 10 steps run, no errors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine
    from project.config import NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        # Seed workspace clusters
        ws_positions = [(28, 5, 1), (28, 15, 2), (28, 25, 3)]
        engine.add_nodes_batch(
            positions=ws_positions,
            energies=[50.0] * 3,
            node_types=[NODE_TYPE_WORKSPACE] * 3,
        )
        # Seed dynamic nodes
        dyn_positions = [(10, x, z) for z in range(4) for x in range(0, 32, 4)]
        engine.add_nodes_batch(
            positions=dyn_positions,
            energies=[30.0] * len(dyn_positions),
            node_types=[NODE_TYPE_DYNAMIC] * len(dyn_positions),
        )
        # Inject sensory data at z=0
        data = torch.rand(32, 32, device=engine.device) * 20
        engine.inject_sensory_data(data, region=(0, 32, 0, 32), z=0)

        for step_i in range(10):
            result = engine.step()
            assert isinstance(result, dict)
            assert result['num_nodes'] > 0
            # Energy field should be finite (no NaN/Inf)
            assert torch.isfinite(engine.energy_field).all(), \
                f"NaN/Inf in energy_field at step {step_i}"

        print(f"Final node count: {engine._count}")
    finally:
        TaichiNeuralEngine._instance = None
        del engine
```

### Step 2: Run the test

```
pytest tests/test_3d_end_to_end.py -v
```
Expected: PASS

### Step 3: Run the full test suite

```
pytest tests/ -v
```
Expected: All tests PASS.

### Step 4: Commit

```bash
git add tests/test_3d_end_to_end.py
git commit -m "test: add 3D end-to-end smoke test (10 steps, cluster init, energy finite)"
```

---

## Task 14: Run the full application

### Step 1: Start the application

```
cd src/project && python main.py
```

Watch for:
- No import errors
- Taichi initializes on CUDA
- Grid [512, 512, 8] reported in logs
- GGUI windows open (showing max-pooled XY projection)
- Nodes spawning/dying in the visualization

### Step 2: Verify cluster visualization

Check that the energy display shows scattered hotspots (cluster locations) rather than a solid band at top or bottom. The display should look organic — islands of activity with dynamic nodes flowing between them.

### Step 3: Fix any runtime issues

If any errors appear, check:
- Memory allocation (4GB VRAM — with [512, 512, 8] total fields should be ~100MB)
- Kernel compilation errors (check Taichi output for shape mismatches)
- `_node_count` sync (Python-side `_count` vs Taichi scalar field)

### Step 4: Final commit

```bash
git add -A
git commit -m "feat: 3D volume with island clusters — complete implementation"
```

---

## Summary of Changes

| File | Change |
|---|---|
| `src/project/pyg_config.json` | `grid_size: [512,512,8]`, add `clusters` section |
| `src/project/config.py` | Add `NEIGHBOR_OFFSETS_3D`, `REVERSE_DIRECTION_3D`, `DNA_SLOT_WORD/BIT` |
| `src/project/system/taichi_engine.py` | Full 3D rewrite: new fields, 26-neighbor kernels, 3D spawn |
| `src/project/main.py` | Cluster placement replacing band regions |
| `src/project/workspace/workspace_node.py` | Add `z` coordinate |
| `src/project/workspace/workspace_system.py` | Accept cluster positions |
| `src/project/workspace/mapping.py` | Extend sensory→workspace mapping |
| `src/project/workspace/visualization.py` | Max-pool Z projection |
| `src/project/ui/taichi_gui_manager.py` | Use projected 2D image for GGUI |
| `tests/` | New tests for each 3D feature, existing tests updated |
