# Enhanced DNA System & 8 Connection Types — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand from 4 connection types to 8, add DNA-DNA lock-and-key interactions, hereditary mutation, and behavioral micro-instructions to the Taichi neural engine.

**Architecture:** Update bit layout constants in config.py, add new Taichi fields for grid-node mapping and capacitive charge, rewrite the DNA transfer kernel with 8 connection type branches and lock-and-key compatibility, update spawn kernel for hereditary DNA with mutation, and update the heatmap renderer for 8 colors.

**Tech Stack:** Python 3.12, Taichi 1.7.x (module-level kernels only), PyTorch CUDA, Qt6

**Design doc:** `docs/plans/2026-02-26-enhanced-dna-connection-types-design.md`

**CRITICAL Taichi constraints:**
- `from __future__ import annotations` MUST NOT be used — breaks `@ti.kernel` annotation resolution
- All kernels must be module-level functions (not class methods) — `@ti.data_oriented` class kernels do NOT support `ti.types.ndarray()` in Taichi 1.7.x
- Taichi fields are global — shapes are fixed at `ti.init` time and cannot be resized

---

## Task 1: Update Bit Layout Constants in config.py

**Files:**
- Modify: `src/project/config.py:26-51`

**Step 1: Update connection type enums and add new types**

Replace lines 26-37 with:

```python
CONN_TYPE_EXCITATORY = 0
CONN_TYPE_INHIBITORY = 1
CONN_TYPE_GATED = 2
CONN_TYPE_PLASTIC = 3
CONN_TYPE_ANTI_PLASTIC = 4
CONN_TYPE_DAMPED = 5
CONN_TYPE_RESONANT = 6
CONN_TYPE_CAPACITIVE = 7

NUM_CONN_TYPES = 8  # 3 bits required
```

Remove the `CONN_TYPE_WEIGHT_TABLE` entirely — it is replaced by per-type branching in the kernel.

**Step 2: Update binary layout constants**

Replace lines 39-51 with:

```python
# =============================================================================
# Binary Node State Encoding (64-bit packed int64 per node)
# Layout: [ALIVE:1][NODE_TYPE:2][CONN_TYPE:3][DNA[0..7]:8x5=40][RSVD:18]
# state == 0 means DEAD — all DNA wiped, disconnected from all math.
# =============================================================================
BINARY_ALIVE_BIT = 63
BINARY_NODE_TYPE_SHIFT = 61
BINARY_CONN_TYPE_SHIFT = 58           # was 59 — now 3 bits at positions 60-58
BINARY_DNA_BASE_SHIFT = 18            # was 19 — shifted down by 1
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
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/project/config.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/project/config.py
git commit -m "feat: update bit layout for 8 connection types and DNA micro-instructions"
```

---

## Task 2: Update Taichi Engine Imports and Fields

**Files:**
- Modify: `src/project/system/taichi_engine.py:19-24` (docstring bit layout)
- Modify: `src/project/system/taichi_engine.py:36-53` (imports)
- Modify: `src/project/system/taichi_engine.py:78-98` (fields)

**Step 1: Update docstring bit layout comment**

Update lines 19-24 to reflect the new layout:

```python
Bit layout (64-bit int64 per node):
    bit 63      : alive flag
    bits 62-61  : node_type  (0=sensory, 1=dynamic, 2=workspace)
    bits 60-58  : conn_type  (0-7, eight connection types)
    bits 57-18  : DNA x 8 neighbors, 5 bits each [MODE:1][PARAM:4]
    bits 17-0   : reserved
```

**Step 2: Update imports**

Replace lines 36-53 to import new constants and remove `CONN_TYPE_WEIGHT_TABLE`:

```python
from project.config import (
    BINARY_ALIVE_BIT,
    BINARY_NODE_TYPE_SHIFT,
    BINARY_CONN_TYPE_SHIFT,
    BINARY_DNA_BASE_SHIFT,
    BINARY_DNA_BITS_PER_NEIGHBOR,
    BINARY_DNA_MAX_VALUE,
    BINARY_DNA_MASK,
    BINARY_TYPE_MASK,
    BINARY_CONN_TYPE_MASK,
    CONN_TYPE_CAPACITIVE,
    DNA_MUTATION_RATE,
    REVERSE_DIRECTION,
    SPAWN_TIER_1_THRESHOLD,
    SPAWN_TIER_2_THRESHOLD,
    SPAWN_TIER_3_THRESHOLD,
    SPAWN_TIER_1_LIMIT,
    SPAWN_TIER_2_LIMIT,
    SPAWN_TIER_3_LIMIT,
    SPAWN_TIER_4_LIMIT,
)
```

**Step 3: Add new fields and update existing ones**

After the existing fields (line 98), the fields section should become:

```python
# Node data
_node_state  = ti.field(dtype=ti.i64, shape=MAX_NODES)  # packed binary state
_node_energy = ti.field(dtype=ti.f32, shape=MAX_NODES)  # working energy
_node_pos_y  = ti.field(dtype=ti.i32, shape=MAX_NODES)  # Y grid coordinate
_node_pos_x  = ti.field(dtype=ti.i32, shape=MAX_NODES)  # X grid coordinate
_node_count  = ti.field(dtype=ti.i32, shape=())          # high-water mark
_node_charge = ti.field(dtype=ti.f32, shape=MAX_NODES)   # capacitive charge accumulator

# Per-step event counters (reset before each step)
_deaths_count = ti.field(dtype=ti.i32, shape=())
_spawns_count = ti.field(dtype=ti.i32, shape=())

# Constant lookup tables (filled once at engine init)
_neighbor_dy   = ti.field(dtype=ti.i32, shape=8)
_neighbor_dx   = ti.field(dtype=ti.i32, shape=8)
_dna_shifts    = ti.field(dtype=ti.i32, shape=8)
_reverse_dir   = ti.field(dtype=ti.i32, shape=8)  # reverse direction for lock-and-key
```

Remove `_weight_table` — no longer needed.

Note: `_grid_node_id` cannot be a fixed-shape field because H and W are not known until engine init. It will be created as a PyTorch tensor on the engine instance and passed to kernels via `ti.types.ndarray()`. See Task 4.

**Step 4: Update engine __init__ to populate new lookup tables**

In `TaichiNeuralEngine.__init__` (around line 436-443), add initialization:

```python
for d, (dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
    _neighbor_dy[d] = dy
    _neighbor_dx[d] = dx
    _dna_shifts[d]  = _DNA_SHIFTS[d]
    _reverse_dir[d] = REVERSE_DIRECTION[d]
```

Remove the `_weight_table` initialization loop.

**Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/project/system/taichi_engine.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/project/system/taichi_engine.py
git commit -m "feat: add new Taichi fields and imports for 8 conn types"
```

---

## Task 3: Write Grid Map Kernels

**Files:**
- Modify: `src/project/system/taichi_engine.py` (add kernels after existing kernels, before the class)

**Step 1: Add grid map clear and scatter kernels**

Add these module-level kernels:

```python
@ti.kernel
def _clear_grid_map(grid_node_id: ti.types.ndarray()):
    """Fill grid_node_id with -1 (no node)."""
    for y, x in ti.ndrange(grid_node_id.shape[0], grid_node_id.shape[1]):
        grid_node_id[y, x] = -1


@ti.kernel
def _build_grid_map(grid_node_id: ti.types.ndarray()):
    """Scatter alive nodes into the grid map. Last-writer-wins for shared cells."""
    for i in range(_node_count[None]):
        if _node_state[i] != 0:
            grid_node_id[_node_pos_y[i], _node_pos_x[i]] = i
```

**Step 2: Create grid_node_id tensor in engine __init__**

In `TaichiNeuralEngine.__init__`, after the `energy_field` creation (around line 418):

```python
# Grid-to-node mapping for DNA lock-and-key neighbor lookups
self.grid_node_id = torch.full(
    (self.H, self.W), -1, dtype=torch.int32, device=self.device
)
```

**Step 3: Add grid map rebuild to step()**

In the `step()` method, insert before the DNA transfer call (before line 479):

```python
# Step 0: Rebuild grid-node map for lock-and-key DNA lookups
_clear_grid_map(self.grid_node_id)
if self._count > 0:
    _build_grid_map(self.grid_node_id)
```

**Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/project/system/taichi_engine.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add src/project/system/taichi_engine.py
git commit -m "feat: add grid-to-node map kernels for DNA lock-and-key lookups"
```

---

## Task 4: Rewrite DNA Transfer Kernel

**Files:**
- Modify: `src/project/system/taichi_engine.py:104-162` (replace `_dna_transfer_kernel`)

This is the most complex task. The kernel must handle:
- Lock-and-key DNA compatibility via `_grid_node_id` → neighbor state lookup
- 8 connection type branches
- DNA micro-instruction decoding (MODE/PARAM/SPECIAL)
- Capacitive charge accumulation
- Frame counter for resonant type

**Step 1: Replace the transfer kernel**

Replace `_dna_transfer_kernel` (lines 104-162) with the new kernel. The kernel signature adds `grid_node_id` and `frame` parameters:

```python
@ti.kernel
def _dna_transfer_kernel(
    energy_field:  ti.types.ndarray(),   # PyTorch CUDA tensor, zero-copy
    grid_node_id:  ti.types.ndarray(),   # [H, W] int32, node index at each cell
    H: int,
    W: int,
    dt: float,
    gate_threshold: float,
    frame: int,
    strength: float,
):
    """
    Enhanced DNA transfer with lock-and-key compatibility, 8 connection types,
    and DNA micro-instructions.

    For every live node: compute energy exchange with its 8 DNA-weighted neighbors.
    Transfer depends on BOTH the sender's and receiver's DNA (lock-and-key).
    """
    for i in range(_node_count[None]):
        state = _node_state[i]
        if state == 0:
            continue

        py = _node_pos_y[i]
        px = _node_pos_x[i]
        energy = _node_energy[i]
        conn_type = int((state >> 58) & 7)
        gate_fire = 1.0 if energy > gate_threshold else 0.0
        damping = ti.exp(-ti.abs(energy) / 50.0)

        total = 0.0
        for d in ti.static(range(8)):
            ny = (py + _neighbor_dy[d] + H) % H
            nx = (px + _neighbor_dx[d] + W) % W

            # --- Extract my DNA for direction d ---
            my_dna_raw = int((state >> _dna_shifts[d]) & 31)
            my_mode  = (my_dna_raw >> 4) & 1
            my_param = my_dna_raw & 15

            # --- Lock-and-key: look up neighbor's DNA for reverse direction ---
            neighbor_id = grid_node_id[ny, nx]
            compat = 0.0
            their_param = 0
            if neighbor_id >= 0:
                n_state = _node_state[neighbor_id]
                if n_state != 0:
                    rd = _reverse_dir[d]
                    their_dna_raw = int((n_state >> _dna_shifts[rd]) & 31)
                    their_param = their_dna_raw & 15
                    compat = float(my_param & their_param) / 15.0
            # else: no neighbor node at this cell — compat stays 0.0

            if compat <= 0.0:
                continue

            delta = energy_field[ny, nx] - energy

            # --- Apply DNA micro-instruction mode ---
            dna_strength = compat
            skip = 0
            if my_mode == 1:
                special = (my_param >> 2) & 3
                sparam  = my_param & 3
                if special == 0:      # THRESHOLD
                    if ti.abs(delta) <= float(sparam) * 16.0:
                        skip = 1
                elif special == 1:    # INVERT
                    delta = -delta
                    dna_strength = float(sparam) / 3.0
                elif special == 2:    # PULSE
                    if frame % (sparam + 1) != 0:
                        skip = 1
                elif special == 3:    # ABSORB
                    if delta < 0.0:
                        skip = 1
                    dna_strength = float(sparam) / 3.0

            if skip == 1:
                continue

            # --- Connection type determines transfer formula ---
            contrib = 0.0
            if conn_type == 0:        # Excitatory
                contrib = delta * dna_strength * strength * dt
            elif conn_type == 1:      # Inhibitory
                contrib = -delta * dna_strength * strength * dt
            elif conn_type == 2:      # Gated
                contrib = delta * dna_strength * strength * dt * gate_fire
            elif conn_type == 3:      # Plastic
                contrib = dna_strength * strength * dt
            elif conn_type == 4:      # Anti-Plastic
                contrib = -dna_strength * strength * dt
            elif conn_type == 5:      # Damped
                contrib = delta * dna_strength * strength * dt * damping
            elif conn_type == 6:      # Resonant
                osc = ti.sin(float(frame) * dna_strength * 3.14159265)
                contrib = delta * osc * strength * dt
            elif conn_type == 7:      # Capacitive
                # Accumulate |delta| in charge; burst-discharge handled below
                ti.atomic_add(_node_charge[i], ti.abs(delta) * dna_strength * dt)
                contrib = 0.0         # no immediate transfer

            if contrib != 0.0:
                total += contrib
                ti.atomic_sub(energy_field[ny, nx], contrib)

        # Capacitive burst discharge
        if conn_type == 7 and _node_charge[i] > gate_threshold:
            burst = _node_charge[i]
            _node_charge[i] = 0.0
            total += burst

        # Write accumulated transfer to energy field
        if total != 0.0:
            ti.atomic_add(energy_field[py, px], total)
```

**Step 2: Update the step() call to pass new arguments**

In the `step()` method, update the `_dna_transfer_kernel` call (around line 479):

```python
_dna_transfer_kernel(
    self.energy_field,
    self.grid_node_id,
    self.H, self.W,
    self.transfer_dt,
    self.gate_threshold,
    self.frame_counter,
    0.7,  # strength constant
)
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/project/system/taichi_engine.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/project/system/taichi_engine.py
git commit -m "feat: rewrite DNA transfer kernel with 8 conn types, lock-and-key, micro-instructions"
```

---

## Task 5: Update Spawn Kernel for Hereditary DNA

**Files:**
- Modify: `src/project/system/taichi_engine.py` (`_spawn_kernel`, around lines 185-253)

**Step 1: Replace random DNA generation with hereditary mutation**

In `_spawn_kernel`, replace the DNA generation block. The child should inherit parent DNA with 10% per-slot bit-flip mutation. The conn_type for children is still random (8 types now).

Key changes in the spawn kernel:
- Read parent DNA from `state` (parent's packed state)
- For each direction: copy parent's 5-bit DNA, with 10% chance flip 1 random bit
- Connection type: `ti.min(int(ti.random() * 8.0), 7)` (was 4, now 8)
- Bit shifts use new `_dna_shifts` (which now use `BINARY_DNA_BASE_SHIFT = 18`)
- Conn type shift is now 58 (was 59)

Replace the DNA + state packing section:

```python
        # Hereditary DNA: inherit parent's DNA with 10% mutation per slot
        dna_packed = ti.i64(0)
        for d in ti.static(range(8)):
            parent_dna = int((state >> _dna_shifts[d]) & 31)
            if ti.random() < 0.1:   # mutation
                bit = int(ti.random() * 5.0)
                bit = ti.min(bit, 4)
                parent_dna = parent_dna ^ (1 << bit)
            dna_packed |= ti.i64(parent_dna) << _dna_shifts[d]

        conn_type = ti.min(int(ti.random() * 8.0), 7)

        new_state = (
            (ti.i64(1) << 63)           # alive bit
          | (ti.i64(1) << 61)           # node type = dynamic (1)
          | (ti.i64(conn_type) << 58)   # connection type (3 bits)
          | dna_packed
        )
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/project/system/taichi_engine.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/project/system/taichi_engine.py
git commit -m "feat: hereditary DNA with mutation in spawn kernel"
```

---

## Task 6: Update Batch Node Init and Heatmap Renderer

**Files:**
- Modify: `src/project/system/taichi_engine.py` — `_random_conn_type`, `_pack_state_batch`, `add_nodes_batch`, `render_connection_heatmap`

**Step 1: Update _random_conn_type for 8 types**

In `_random_conn_type` (line ~326):

```python
def _random_conn_type(n: int, device: torch.device) -> torch.Tensor:
    """Uniformly random conn type: 12.5% each of 8 types."""
    return torch.randint(0, 8, (n,), device=device, dtype=torch.int64)
```

**Step 2: Update _pack_state_batch for 3-bit conn_type**

In `_pack_state_batch` (line ~315), the shift was imported from config.py so it auto-updates. But verify the mask used in render_connection_heatmap is updated.

**Step 3: Update workspace DNA override in add_nodes_batch**

In `add_nodes_batch` (around line 625-629), workspace nodes should have MODE=0, PARAM=15 (all classic, max probability):

```python
# Workspace nodes (type=2) are energy buckets: excitatory + max classic DNA
ws_mask = (new_types == 2)
if ws_mask.any():
    new_conn[ws_mask] = 0                  # excitatory
    new_dna[ws_mask]  = 15                 # MODE=0, PARAM=15 → probability 1.0
```

Note: `15` not `31` — because in the new encoding, the classic max is `0b0_1111 = 15` (MODE=0, PARAM=15). The old value `31 = 0b1_1111` would set MODE=1 (special mode) which is wrong for workspace nodes.

**Step 4: Update _random_dna for new encoding**

In `_random_dna` (line ~331), keep generating full 5-bit random values (0-31). This naturally creates ~50% classic (MODE=0) and ~50% special (MODE=1) nodes, which is the desired distribution:

```python
def _random_dna(n: int, device: torch.device) -> torch.Tensor:
    """Random 5-bit DNA for 8 neighbors. Returns [N, 8] int64.
    Each value is [MODE:1][PARAM:4] — ~50% classic, ~50% special."""
    return torch.randint(0, BINARY_DNA_MAX_VALUE + 1, (n, 8),
                         device=device, dtype=torch.int64)
```

**Step 5: Update render_connection_heatmap**

Replace the heatmap rendering section (lines ~889-905) with 8 colors:

```python
        conn_type = torch.tensor(
            ((alive_state >> BINARY_CONN_TYPE_SHIFT) & BINARY_CONN_TYPE_MASK),
            dtype=torch.long, device=self.device,
        )
        energy_t     = torch.tensor(alive_e, dtype=torch.float32, device=self.device)
        energy_range = self.energy_cap - self.death_threshold
        brightness   = ((energy_t.clamp(self.death_threshold, self.energy_cap)
                         - self.death_threshold) / energy_range
                        if energy_range > 0 else torch.zeros_like(energy_t))

        # 8 connection type colors
        m = conn_type == 0; heatmap[alive_y[m], alive_x[m], 1] = brightness[m]              # Excitatory: green
        m = conn_type == 1; heatmap[alive_y[m], alive_x[m], 0] = brightness[m]              # Inhibitory: red
        m = conn_type == 2; heatmap[alive_y[m], alive_x[m], 2] = brightness[m]              # Gated: blue
        m = conn_type == 3                                                                    # Plastic: yellow
        heatmap[alive_y[m], alive_x[m], 0] = brightness[m]
        heatmap[alive_y[m], alive_x[m], 1] = brightness[m]
        m = conn_type == 4                                                                    # Anti-Plastic: magenta
        heatmap[alive_y[m], alive_x[m], 0] = brightness[m]
        heatmap[alive_y[m], alive_x[m], 2] = brightness[m]
        m = conn_type == 5                                                                    # Damped: cyan
        heatmap[alive_y[m], alive_x[m], 1] = brightness[m]
        heatmap[alive_y[m], alive_x[m], 2] = brightness[m]
        m = conn_type == 6                                                                    # Resonant: white
        heatmap[alive_y[m], alive_x[m], 0] = brightness[m]
        heatmap[alive_y[m], alive_x[m], 1] = brightness[m]
        heatmap[alive_y[m], alive_x[m], 2] = brightness[m]
        m = conn_type == 7                                                                    # Capacitive: orange
        heatmap[alive_y[m], alive_x[m], 0] = brightness[m]
        heatmap[alive_y[m], alive_x[m], 1] = brightness[m] * 0.5
```

**Step 6: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/project/system/taichi_engine.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git add src/project/system/taichi_engine.py
git commit -m "feat: update batch init, random DNA, and heatmap for 8 connection types"
```

---

## Task 7: Update main.py Adapter and Region Setup

**Files:**
- Modify: `src/project/main.py`

**Step 1: Remove CONN_TYPE_WEIGHT_TABLE usage if any**

Search for any remaining references to `CONN_TYPE_WEIGHT_TABLE` in main.py. There should be none (it's only used in taichi_engine.py), but verify.

**Step 2: Verify adapter compatibility**

The `HybridNeuralSystemAdapter` passes through to the engine and does not reference conn types or DNA directly. No changes needed to the adapter.

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/project/main.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit (if any changes)**

```bash
git add src/project/main.py
git commit -m "chore: verify main.py compatibility with new bit layout"
```

---

## Task 8: Clean Up Removed References

**Files:**
- Modify: `src/project/system/taichi_engine.py` — remove any remaining `_weight_table` references
- Modify: `src/project/system/taichi_engine.py` — remove `CONN_TYPE_WEIGHT_TABLE` from imports

**Step 1: Search for stale references**

Run: `grep -rn "CONN_TYPE_WEIGHT_TABLE\|_weight_table" src/project/`

Remove all found references.

**Step 2: Search for old conn_type shift value (59)**

Run: `grep -rn ">> 59\|<< 59" src/project/`

All should now use 58 (via the imported constant or the new hardcoded value in kernels).

**Step 3: Search for old BINARY_TYPE_MASK used as conn mask**

The old code used `BINARY_TYPE_MASK` (0x3) for both node type AND conn type. With 3-bit conn type, conn type extraction must use `BINARY_CONN_TYPE_MASK` (0x7). Search and fix:

Run: `grep -rn "BINARY_TYPE_MASK" src/project/`

Verify each usage: if extracting node_type → keep 0x3. If extracting conn_type → change to `BINARY_CONN_TYPE_MASK`.

**Step 4: Verify syntax on all modified files**

Run: `python -c "import ast; ast.parse(open('src/project/config.py').read()); print('config OK')"` and same for all changed files.

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: clean up stale references to old 4-type weight table and 2-bit conn mask"
```

---

## Task 9: Integration Test — Manual Smoke Test

**Step 1: Run the application**

```bash
cd src && python -m project.main --log-level DEBUG
```

**Step 2: Verify in logs**

Check `pyg_system.log` for:
- "TaichiNeuralEngine initialized" — no errors
- Spawn/death counts appearing in step metrics
- No Taichi kernel compilation errors
- Grid map rebuild happening each step

**Step 3: Verify heatmap**

Open the visualization window. The connection heatmap should show 8 different colors instead of 4.

**Step 4: Verify DNA heredity**

In DEBUG logs, after several hundred frames, check that spawn counts are nonzero and death counts are nonzero — the population is turning over. The lock-and-key mechanism should create visible clustering in the heatmap (similar-color nodes grouping together over time).

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration test fixes for enhanced DNA system"
```
