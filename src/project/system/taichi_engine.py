"""
TaichiNeuralEngine — Probabilistic energy-field neural cellular automaton on GPU.

Uses Taichi for explicit, readable GPU kernels and PyTorch for the shared
CUDA energy_field tensor — zero-copy, no sync needed.

Architecture note (Taichi 1.7.x):
    `@ti.data_oriented` class kernels do NOT support `ti.types.ndarray()` arguments.
    All kernels that need to accept PyTorch tensors (energy_field) must be
    module-level functions. The TaichiNeuralEngine class is a plain Python manager
    that calls these module-level kernels.

    Module-level Taichi fields (node data) are accessed by name inside the kernels.
    Only one engine instance is supported per process.

Node capacity: pre-allocated at MAX_NODES (2 million) at startup.
Dead nodes (state == 0) contribute zero energy automatically.

Bit layout (64-bit int64 per node, matches config.py):
    bit 63      : alive flag
    bits 62-61  : node_type  (0=sensory, 1=dynamic, 2=workspace) — informational only; physics is uniform
    bits 60-58  : conn_type  (0-7, eight connection types)
    bits 57-3   : reserved (DNA moved to _node_dna field)
    bits 2-0    : modality  (0=neutral, 1=visual, 2=audio_L, 3=audio_R)

3D DNA layout (separate _node_dna field, 3 int64 words per node):
    26 neighbor slots × 5 bits = 130 bits packed into 3 int64 words.
    Word 0: slots 0-11 (bits 0-59), word 1: slots 12-23, word 2: slots 24-25.
    Each 5-bit slot: [MODE:1][PARAM:4].
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import taichi as ti

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
    MODALITY_SHIFT,
    MODALITY_MASK,
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
    NEIGHBOR_OFFSETS_3D,
    REVERSE_DIRECTION_3D,
    DNA_SLOT_WORD,
    DNA_SLOT_BIT,
    NUM_NEIGHBORS_3D,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants for values used inside Taichi kernels and step().
# Taichi kernels read these as Python globals at compile time.
# ---------------------------------------------------------------------------
DAMPING_SCALE = 50.0            # exp(-|energy|/DAMPING_SCALE) for damped connections
SPECIAL_PARAM_MAX = 3.0         # max value of 2-bit sparam field (0-3)
THRESHOLD_ENERGY_SCALE = 16.0   # THRESHOLD micro-instruction: sparam × this
CONTRIB_CLAMP_FRACTION = 0.08   # per-neighbor clamp = energy_cap × this (~20 for cap=244, ~40 for cap=500)
SPAWN_ABOVE_AVG_FACTOR = 1.10   # spawn threshold = avg energy × this
SPAWN_COST_FRACTION = 0.75      # spawn cost = threshold × this
SPAWN_CAP_FRACTION = 0.80       # spawn threshold capped at energy_cap × this


def project_energy_field_to_2d(energy_field: torch.Tensor) -> torch.Tensor:
    """Max-pool a 3D energy_field [H, W, D] to a 2D display image [H, W].

    Each pixel shows the brightest Z layer.  2D inputs are returned unchanged.
    """
    if energy_field.dim() == 2:
        return energy_field
    return energy_field.max(dim=2).values


# =============================================================================
# Taichi initialization — lazy, so config can override device/memory settings
# =============================================================================
_taichi_initialized = False

def init_taichi(device: str = 'auto', device_memory_fraction: float = 0.6) -> None:
    """Initialize Taichi runtime. Must be called before creating TaichiNeuralEngine.

    Args:
        device: 'auto', 'cuda', or 'cpu'
        device_memory_fraction: fraction of GPU VRAM to reserve (0.0-1.0)
    """
    global _taichi_initialized
    if _taichi_initialized:
        logger.warning("Taichi already initialized, skipping")
        return
    if device == 'auto':
        arch = ti.cuda if torch.cuda.is_available() else ti.cpu
    elif device == 'cuda':
        arch = ti.cuda
    else:
        arch = ti.cpu
    ti.init(arch=arch, device_memory_fraction=device_memory_fraction)
    _taichi_initialized = True
    logger.info("Taichi initialized: arch=%s, memory_fraction=%.2f", arch, device_memory_fraction)

# =============================================================================
# Constants
# =============================================================================

MAX_NODES = 2_000_000   # [512, 512, 8] = 2.1M cells; reduced from 4M

# Legacy Python helper — still used by _pack_state_batch (updated in Task 5).
# DNA bits in _node_state bits 57-18 are being phased out in favour of _node_dna.
_DNA_SHIFTS = [BINARY_DNA_BASE_SHIFT + d * BINARY_DNA_BITS_PER_NEIGHBOR
               for d in range(8)]

# =============================================================================
# Module-level Taichi fields
# (globally accessible by all kernels below; one global engine per process)
# =============================================================================

# Ensure Taichi is initialized before defining fields (import-time fallback)
if not _taichi_initialized:
    init_taichi()

# Node data
_node_state  = ti.field(dtype=ti.i64, shape=MAX_NODES)       # packed binary (no DNA bits in slots 57-18)
_node_energy = ti.field(dtype=ti.f32, shape=MAX_NODES)       # working energy
_node_pos_y  = ti.field(dtype=ti.i32, shape=MAX_NODES)       # Y grid coordinate
_node_pos_x  = ti.field(dtype=ti.i32, shape=MAX_NODES)       # X grid coordinate
_node_pos_z  = ti.field(dtype=ti.i32, shape=MAX_NODES)       # Z grid coordinate (NEW)
_node_count  = ti.field(dtype=ti.i32, shape=())               # high-water mark
_node_charge = ti.field(dtype=ti.f32, shape=MAX_NODES)        # capacitive charge
_node_dna    = ti.field(dtype=ti.i64, shape=(MAX_NODES, 3))  # 26×5=130 bits in 3 int64s (NEW)

# =============================================================================
# Region registry (ADR-001) — sensory and workspace as named regions of the grid.
#
# Each entry describes a 2-D bounding box (all Z layers) with one behavioural
# flag:
#   _region_spawn[r]    — 1 if nodes inside this region may reproduce, else 0
#
# All nodes are mortal — no immortality flags exist.  Any node that drops below
# the death threshold is removed regardless of region type.
#
# Regions are registered by the adapter via TaichiNeuralEngine.register_region()
# before the first step().  Kernels call _get_node_region() to look up a node's
# region index, then gate spawning on the region flag.
#
# Up to MAX_REGIONS entries; entries 0..(_region_count-1) are valid.
# =============================================================================
MAX_REGIONS = 16

_region_count    = ti.field(dtype=ti.i32, shape=())
_region_y0       = ti.field(dtype=ti.i32, shape=MAX_REGIONS)
_region_y1       = ti.field(dtype=ti.i32, shape=MAX_REGIONS)
_region_x0       = ti.field(dtype=ti.i32, shape=MAX_REGIONS)
_region_x1       = ti.field(dtype=ti.i32, shape=MAX_REGIONS)
_region_type     = ti.field(dtype=ti.i32, shape=MAX_REGIONS)   # 0=sensory,1=dynamic,2=workspace
_region_spawn    = ti.field(dtype=ti.i32, shape=MAX_REGIONS)   # 1 = spawning allowed

# Per-step event counters (reset before each step)
_deaths_count = ti.field(dtype=ti.i32, shape=())
_spawns_count = ti.field(dtype=ti.i32, shape=())

# On-GPU reduction accumulators
_dyn_energy_sum = ti.field(dtype=ti.f64, shape=())
_dyn_node_count = ti.field(dtype=ti.i32, shape=())

# Constant lookup tables for 3D neighbour kernels — filled at engine init
_neighbor_dz   = ti.field(dtype=ti.i32, shape=26)   # Z offsets (NEW)
_neighbor_dy   = ti.field(dtype=ti.i32, shape=26)   # extended from 8 to 26
_neighbor_dx   = ti.field(dtype=ti.i32, shape=26)   # extended from 8 to 26
_reverse_dir   = ti.field(dtype=ti.i32, shape=26)   # extended from 8 to 26
_dna_slot_word = ti.field(dtype=ti.i32, shape=26)   # which int64 word per slot (NEW)
_dna_slot_bit  = ti.field(dtype=ti.i32, shape=26)   # bit offset within word (NEW)
# Note: _dna_shifts removed (DNA no longer packed in _node_state)

# =============================================================================
# Taichi kernels (module-level — support ti.types.ndarray() for PyTorch tensors)
# =============================================================================

@ti.kernel
def _clear_grid_map(grid_node_id: ti.types.ndarray()):
    """Fill 3D grid_node_id [H, W, D] with MAX_NODES (no valid node)."""
    for y, x, z in ti.ndrange(
        grid_node_id.shape[0], grid_node_id.shape[1], grid_node_id.shape[2]
    ):
        grid_node_id[y, x, z] = MAX_NODES


@ti.kernel
def _build_grid_map(grid_node_id: ti.types.ndarray()):
    """Scatter alive nodes into the 3D grid map. Lowest node ID wins per cell."""
    for i in range(_node_count[None]):
        if _node_state[i] != 0:
            ti.atomic_min(
                grid_node_id[_node_pos_y[i], _node_pos_x[i], _node_pos_z[i]], i
            )


@ti.kernel
def _dna_transfer_kernel(
    energy_field:  ti.types.ndarray(),   # PyTorch CUDA tensor [H, W, D], zero-copy
    grid_node_id:  ti.types.ndarray(),   # [H, W, D] int32, node index at each cell
    H: int,
    W: int,
    D: int,
    dt: float,
    gate_threshold: float,
    frame: int,
    strength: float,
    contrib_clamp: float,
):
    """
    3D DNA transfer with lock-and-key compatibility, 26 neighbours, 8 connection
    types, and DNA micro-instructions.

    For every live node: compute energy exchange with its 26 DNA-weighted neighbors.
    Transfer depends on BOTH the sender's and receiver's DNA (lock-and-key).
    DNA is read from _node_dna[node, word] using _dna_slot_word / _dna_slot_bit
    lookup tables. Each 5-bit slot: [MODE:1][PARAM:4].
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
        damping = ti.exp(-ti.abs(energy) / DAMPING_SCALE)

        total = 0.0
        for d in ti.static(range(26)):
            ny = (py + _neighbor_dy[d] + H) % H
            nx = (px + _neighbor_dx[d] + W) % W
            nz = (pz + _neighbor_dz[d] + D) % D

            # --- Extract my DNA for direction d from _node_dna ---
            my_dna_raw = int((_node_dna[i, _dna_slot_word[d]] >> _dna_slot_bit[d]) & 31)
            my_mode  = (my_dna_raw >> 4) & 1
            my_param = my_dna_raw & 15

            # --- Lock-and-key: look up neighbor's DNA for reverse direction ---
            # NOTE: Taichi 1.7 disallows `continue` in non-static if inside
            # ti.static for. All logic must be guarded with nested ifs instead.
            neighbor_id = grid_node_id[ny, nx, nz]
            compat = 0.0
            if neighbor_id < MAX_NODES:
                n_state = _node_state[neighbor_id]
                if n_state != 0:
                    rd = _reverse_dir[d]
                    their_dna_raw = int((_node_dna[neighbor_id, _dna_slot_word[rd]] >> _dna_slot_bit[rd]) & 31)
                    their_param = their_dna_raw & 15
                    compat = float(my_param & their_param) / 15.0

            if compat > 0.0:
                delta = energy_field[ny, nx, nz] - energy

                # --- Apply DNA micro-instruction mode ---
                dna_strength = compat
                skip = 0
                if my_mode == 1:
                    special = (my_param >> 2) & 3
                    sparam  = my_param & 3
                    if special == 0:      # THRESHOLD
                        if ti.abs(delta) <= float(sparam) * THRESHOLD_ENERGY_SCALE:
                            skip = 1
                    elif special == 1:    # INVERT
                        delta = -delta
                        dna_strength = float(sparam) / SPECIAL_PARAM_MAX
                    elif special == 2:    # PULSE
                        if frame % (sparam + 1) != 0:
                            skip = 1
                    elif special == 3:    # ABSORB
                        if delta < 0.0:
                            skip = 1
                        dna_strength = float(sparam) / SPECIAL_PARAM_MAX

                if skip == 0:
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
                        osc = ti.sin(float(frame) * dna_strength * ti.math.pi)
                        contrib = delta * osc * strength * dt
                    elif conn_type == 7:      # Capacitive
                        ti.atomic_add(_node_charge[i], ti.abs(delta) * dna_strength * dt)
                        contrib = 0.0         # no immediate transfer

                    # Clamp per-neighbor contribution to prevent runaway transfer.
                    # Field-level clamp follows after all transfers complete.
                    contrib = ti.min(ti.max(contrib, -contrib_clamp), contrib_clamp)
                    if contrib != 0.0:
                        total += contrib
                        ti.atomic_sub(energy_field[ny, nx, nz], contrib)

        # Capacitive burst discharge
        if conn_type == 7 and _node_charge[i] > gate_threshold:
            burst = _node_charge[i]
            _node_charge[i] = 0.0
            total += burst

        # Write accumulated transfer to energy field
        if total != 0.0:
            ti.atomic_add(energy_field[py, px, pz], total)


@ti.func
def _get_node_region(py: int, px: int) -> int:
    """Return region index [0..MAX_REGIONS-1] for (y, x), or -1 if unregistered.

    Iterates over all MAX_REGIONS slots; only registered entries
    (r < _region_count) are checked.  Returns the first matching region.
    Called from inside kernels — compiled as an inlined GPU function.
    """
    result = -1
    n = _region_count[None]
    for r in ti.static(range(MAX_REGIONS)):
        if r < n and result < 0:
            if (py >= _region_y0[r] and py < _region_y1[r] and
                    px >= _region_x0[r] and px < _region_x1[r]):
                result = r
    return result


@ti.kernel
def _death_kernel(death_threshold: float):
    """
    Kill every node whose energy falls below the death threshold.

    All nodes are mortal — there is no immortality exception for any region
    or node type.  Sensory and workspace nodes that stop receiving energy will
    deplete and die naturally, freeing their slots for dynamic regrowth.  This
    is the intended thermodynamic behaviour.

    Zeroing the state wipes alive, type, and conn fields in one operation.
    DNA words in _node_dna are also cleared on death.
    Death count is accumulated atomically in _deaths_count.
    """
    for i in range(_node_count[None]):
        if _node_state[i] == 0:
            continue
        if _node_energy[i] >= death_threshold:
            continue
        _node_state[i]  = 0
        _node_energy[i] = 0.0
        _node_charge[i] = 0.0
        _node_dna[i, 0] = 0
        _node_dna[i, 1] = 0
        _node_dna[i, 2] = 0
        ti.atomic_add(_deaths_count[None], 1)


@ti.kernel
def _spawn_kernel(
    H: int,
    W: int,
    D: int,
    spawn_threshold: float,
    spawn_cost: float,
    child_energy: float,
    max_spawns: int,
    n_existing: int,
):
    """
    Each live dynamic node (type == 1) with sufficient energy spawns one child.

    Child slot is claimed by atomically incrementing _node_count. If the buffer
    is full the parent is skipped. Child position is parent +/- 1 cell (toroidal
    in Y, X, and Z). Child DNA is inherited from _node_dna (26 slots, 3 int64
    words) with 10% per-slot bit-flip mutation. Connection type is random (8
    types).

    Spawn count accumulated atomically in _spawns_count.

    n_existing: snapshot of node count *before* this kernel -- prevents newly
    spawned children from being visited (and chain-spawning) in the same step.
    """
    for i in range(n_existing):
        state = _node_state[i]
        if state == 0:
            continue
        if _node_energy[i] < spawn_threshold:
            continue

        # Region-aware spawn gate (ADR-001): check the region registry first.
        # Fall back to node_type == 1 check for nodes placed before any
        # regions were registered (backward-compat).
        py = _node_pos_y[i]
        px = _node_pos_x[i]
        region = _get_node_region(py, px)

        can_spawn = 0
        if region >= 0:
            can_spawn = _region_spawn[region]
        else:
            # No region registered — only dynamic (type 1) may spawn (legacy)
            node_type = int((state >> 61) & 3)
            if node_type == 1:
                can_spawn = 1

        if can_spawn == 0:
            continue

        # Atomically claim a spawn slot BEFORE doing work — prevents overshoot
        my_spawn = ti.atomic_add(_spawns_count[None], 1)
        if my_spawn >= max_spawns:
            ti.atomic_sub(_spawns_count[None], 1)  # undo — limit reached
            continue

        # Claim next node slot atomically — parallel-safe
        slot = ti.atomic_add(_node_count[None], 1)
        if slot >= MAX_NODES:
            ti.atomic_sub(_node_count[None], 1)    # undo — buffer full
            ti.atomic_sub(_spawns_count[None], 1)  # undo spawn claim too
            continue

        # Random ±1 offset from parent (toroidal grid wrap in Y, X, Z)
        offset_y = int(ti.random() * 3.0) - 1
        offset_x = int(ti.random() * 3.0) - 1
        offset_z = int(ti.random() * 3.0) - 1
        child_y  = (_node_pos_y[i] + offset_y + H) % H
        child_x  = (_node_pos_x[i] + offset_x + W) % W
        child_z  = (_node_pos_z[i] + offset_z + D) % D

        # Hereditary DNA: inherit parent's 26-slot DNA with 10% mutation per slot
        # Clear child DNA words first
        _node_dna[slot, 0] = ti.i64(0)
        _node_dna[slot, 1] = ti.i64(0)
        _node_dna[slot, 2] = ti.i64(0)

        for d in ti.static(range(26)):
            word = _dna_slot_word[d]
            bit  = _dna_slot_bit[d]
            parent_dna = int((_node_dna[i, word] >> bit) & 31)
            if ti.random() < 0.1:
                flip_bit = ti.min(int(ti.random() * 5.0), 4)
                parent_dna = parent_dna ^ (1 << flip_bit)
            _node_dna[slot, word] = _node_dna[slot, word] | (ti.i64(parent_dna) << bit)

        conn_type = ti.min(int(ti.random() * 8.0), 7)  # 8 connection types

        # Pack child state: alive=1, type=1 (dynamic), conn_type + parent modality
        # DNA is stored separately in _node_dna — NOT packed in _node_state
        parent_modality = state & ti.i64(7)   # bits 2-0 of parent state
        new_state = (
            (ti.i64(1) << 63)           # alive bit
          | (ti.i64(1) << 61)           # node type = dynamic (1)
          | (ti.i64(conn_type) << 58)   # connection type (3 bits)
          | parent_modality             # inherit parent modality (bits 2-0)
        )

        _node_state[slot]  = new_state
        _node_energy[slot] = child_energy
        _node_pos_y[slot]  = child_y
        _node_pos_x[slot]  = child_x
        _node_pos_z[slot]  = child_z
        _node_charge[slot] = 0.0  # Reset capacitive charge for new node

        _node_energy[i] -= spawn_cost


@ti.kernel
def _clamp_kernel(energy_field: ti.types.ndarray(), lo: float, hi: float):
    """Clamp every cell of energy_field to [lo, hi]. Parallel over all H×W×D cells."""
    for y, x, z in ti.ndrange(
        energy_field.shape[0], energy_field.shape[1], energy_field.shape[2]
    ):
        v = energy_field[y, x, z]
        if v < lo:
            energy_field[y, x, z] = lo
        elif v > hi:
            energy_field[y, x, z] = hi


@ti.kernel
def _inject_sensory_kernel(
    energy_field: ti.types.ndarray(),
    data:         ti.types.ndarray(),
    y0: int, x0: int, h: int, w: int,
    z: int,
):
    """Write raw data values as energy into a specific Z layer of energy_field.

    Used for audio injection (direct overwrite).
    """
    for dy, dx in ti.ndrange(h, w):
        energy_field[y0 + dy, x0 + dx, z] = data[dy, dx]


@ti.kernel
def _inject_sensory_delta_kernel(
    energy_field: ti.types.ndarray(),
    grid_node_id: ti.types.ndarray(),
    data:         ti.types.ndarray(),
    y0: int, x0: int, h: int, w: int,
    z: int,
):
    """Node-targeted sensory injection.

    For each pixel in [y0:y0+h, x0:x0+w] at depth z, if a live node
    occupies that cell, drive its energy toward the pixel value:

        delta = pixel_value − node_energy
        _node_energy[id] += delta × 0.5
        energy_field[y, x, z] = _node_energy[id]   (keep field in sync)

    Only nodes receive energy — the field is a mirror of node state,
    never an independent energy source.  Empty cells are untouched.
    """
    for dy, dx in ti.ndrange(h, w):
        py = y0 + dy
        px = x0 + dx
        node_id = grid_node_id[py, px, z]
        if node_id < MAX_NODES:
            state = _node_state[node_id]
            if state != 0:
                pixel_val = data[dy, dx]
                node_e    = _node_energy[node_id]
                new_e     = node_e + (pixel_val - node_e) * 0.5
                _node_energy[node_id] = new_e
                energy_field[py, px, z] = new_e


@ti.kernel
def _sync_energy_from_field(energy_field: ti.types.ndarray(), grid_node_id: ti.types.ndarray()):
    """
    Sync each live node's working energy from its 3D position in energy_field.
    Called after DNA transfer so birth/death decisions see the current values.

    Only the grid-map owner of each cell (lowest node ID) syncs from the field.
    Non-owner nodes sharing a cell keep their previous energy, preventing
    multiple nodes from getting identical energy and making synchronized
    spawn/death decisions.
    """
    for i in range(_node_count[None]):
        if _node_state[i] != 0:
            y = _node_pos_y[i]
            x = _node_pos_x[i]
            z = _node_pos_z[i]
            if grid_node_id[y, x, z] == i:
                _node_energy[i] = energy_field[y, x, z]


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
        _node_charge[start + i] = 0.0  # Reset capacitive charge for new node


@ti.kernel
def _write_dna_kernel(
    dna_data: ti.types.ndarray(),   # [N, 3] int64
    start: int,
    n: int,
):
    """Write DNA words for N nodes starting at slot `start`."""
    for i in range(n):
        _node_dna[start + i, 0] = dna_data[i, 0]
        _node_dna[start + i, 1] = dna_data[i, 1]
        _node_dna[start + i, 2] = dna_data[i, 2]


@ti.kernel
def _reduce_dynamic_energy():
    """Compute sum and count of alive dynamic nodes on-GPU (no CPU transfer)."""
    _dyn_energy_sum[None] = 0.0
    _dyn_node_count[None] = 0
    ti.loop_config(block_dim=256)
    for i in range(_node_count[None]):
        state = _node_state[i]
        if state != 0:
            node_type = (state >> 61) & 3
            if node_type == 1:  # dynamic
                ti.atomic_add(_dyn_energy_sum[None], ti.cast(_node_energy[i], ti.f64))
                ti.atomic_add(_dyn_node_count[None], 1)


# =============================================================================
# Module-level helpers (pure PyTorch, used by add_nodes_batch)
# =============================================================================

def _pack_state_batch(alive: torch.Tensor, node_type: torch.Tensor,
                      conn_type: torch.Tensor,
                      modality: torch.Tensor) -> torch.Tensor:
    """Pack N nodes into int64 binary states — DNA is now in _node_dna field."""
    state = (alive.to(torch.int64) << BINARY_ALIVE_BIT)
    state |= (node_type.to(torch.int64) << BINARY_NODE_TYPE_SHIFT)
    state |= (conn_type.to(torch.int64) << BINARY_CONN_TYPE_SHIFT)
    state |= modality.to(torch.int64)   # bits 2-0 = MODALITY (MODALITY_SHIFT == 0)
    return state


def _random_conn_type(n: int, device: torch.device) -> torch.Tensor:
    """Uniformly random conn type: 12.5% each of 8 types."""
    return torch.randint(0, 8, (n,), device=device, dtype=torch.int64)


def _random_dna(n: int, device: torch.device) -> torch.Tensor:
    """Random 5-bit DNA for 8 neighbors. Returns [N, 8] int64."""
    return torch.randint(0, BINARY_DNA_MAX_VALUE + 1, (n, 8),
                         device=device, dtype=torch.int64)


def _random_dna_3d(n: int, device: torch.device) -> torch.Tensor:
    """Random 5-bit DNA for 26 neighbours packed into [N, 3] int64 words."""
    slots = torch.randint(0, 32, (n, NUM_NEIGHBORS_3D), device=device, dtype=torch.int64)
    dna = torch.zeros(n, 3, device=device, dtype=torch.int64)
    for slot_n in range(NUM_NEIGHBORS_3D):
        word = DNA_SLOT_WORD[slot_n]
        bit  = DNA_SLOT_BIT[slot_n]
        dna[:, word] |= (slots[:, slot_n] << bit)
    return dna  # shape [N, 3]


# =============================================================================
# TaichiNeuralEngine — Python manager class
# =============================================================================

class TaichiNeuralEngine:
    """
    GPU-accelerated neural simulation engine.

    This class manages configuration, PyTorch state (energy_field),
    and the Python-side node count. All heavy GPU work is done by the module-level
    Taichi kernels above, which access module-level Taichi fields directly.

    Only one instance is supported per process (Taichi fields are global).

    Lifecycle
    ---------
    Nodes occupy slots 0.._count-1 in the module-level Taichi fields.
    Dead nodes (state == 0) are simply skipped in every kernel — their energy
    contributes nothing automatically. No compaction needed: 2M slots give
    massive headroom.

    Energy Field
    ------------
    `energy_field` is a PyTorch CUDA tensor. It is passed to Taichi kernels as
    a `ti.types.ndarray()` argument — zero-copy on CUDA.

    Public API
    ------------------------------------------
    step(), inject_sensory_data(), add_node(), add_nodes_batch(),
    get_node_data(), get_energy_field(), add_energy_at(),
    read_workspace_energies(), render_connection_heatmap(), get_metrics()
    """

    _instance: "Optional[TaichiNeuralEngine]" = None

    def __init__(
        self,
        grid_size: Tuple = (512, 512, 8),   # (H, W, D) — backward compat: 2-tuple → D=8
        node_spawn_threshold: float = 20.0,
        node_death_threshold: float = -10.0,
        node_energy_cap: float = 244.0,
        spawn_cost: float = 19.52,
        gate_threshold: float = 0.5,
        transfer_dt: float = 0.1,
        child_energy_fraction: float = 0.5,
        transfer_strength: float = 0.7,
        device: str = "cuda",
    ):
        if TaichiNeuralEngine._instance is not None:
            raise RuntimeError(
                "Only one TaichiNeuralEngine per process (Taichi fields are global). "
                "Destroy the existing instance before creating a new one."
            )
        TaichiNeuralEngine._instance = self

        try:
            if not _taichi_initialized:
                init_taichi()  # fallback: init with defaults if caller forgot

            if len(grid_size) == 2:
                self.H, self.W = grid_size
                self.D = 8
            else:
                self.H, self.W, self.D = grid_size
            self.grid_size = (self.H, self.W, self.D)
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

            # Node lifecycle thresholds
            self.spawn_threshold       = node_spawn_threshold
            self.death_threshold       = node_death_threshold
            self.energy_cap            = node_energy_cap
            self.spawn_cost            = spawn_cost           # static fallback only
            self.child_energy_fraction = child_energy_fraction
            self.gate_threshold      = gate_threshold
            self.transfer_dt         = transfer_dt
            self.transfer_strength   = transfer_strength

            # Statistics
            self.total_spawns  = 0
            self.total_deaths  = 0
            self.frame_counter = 0
            self.grid_operations_per_step = self.H * self.W * self.D * 60

            # ---------------------------------------------------------------
            # PyTorch energy field (shared with Taichi kernels via ndarray)
            # ---------------------------------------------------------------
            self.energy_field = torch.zeros(
                self.H, self.W, self.D, dtype=torch.float32, device=self.device
            )

            # Grid-to-node mapping for DNA lock-and-key neighbor lookups
            # Sentinel = MAX_NODES (no valid node); used with ti.atomic_min for determinism
            self.grid_node_id = torch.full(
                (self.H, self.W, self.D), MAX_NODES, dtype=torch.int32, device=self.device
            )

            # Python-side count (mirrors _node_count Taichi field)
            self._count = 0

            # Workspace read cache (retired — read_workspace_energies uses direct field slice)
            self._workspace_local_y: Optional[torch.Tensor] = None
            self._workspace_local_x: Optional[torch.Tensor] = None
            self._workspace_cache_valid = False
            self._workspace_cached_region: Optional[Tuple[int, int, int, int]] = None
            self._workspace_cache_lock = threading.Lock()

            # Lock to serialise step() and read_workspace_energies() across threads
            self.step_lock = threading.Lock()

            # Sensory injection counter
            self._injection_counter = 0

            # ---------------------------------------------------------------
            # Initialize constant lookup tables in module-level Taichi fields
            # ---------------------------------------------------------------
            for n, (dz, dy, dx) in enumerate(NEIGHBOR_OFFSETS_3D):
                _neighbor_dz[n]   = dz
                _neighbor_dy[n]   = dy
                _neighbor_dx[n]   = dx
                _reverse_dir[n]   = REVERSE_DIRECTION_3D[n]
                _dna_slot_word[n] = DNA_SLOT_WORD[n]
                _dna_slot_bit[n]  = DNA_SLOT_BIT[n]

            logger.info(
                "TaichiNeuralEngine initialized: grid=%s, MAX_NODES=%s, device=%s",
                grid_size, f"{MAX_NODES:,}", self.device,
            )
        except Exception:
            TaichiNeuralEngine._instance = None
            raise

    def __del__(self):
        TaichiNeuralEngine._instance = None

    @property
    def node_count(self) -> int:
        """Current node count (high-water mark, no GPU roundtrip)."""
        return self._count

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Region registry (ADR-001)
    # -----------------------------------------------------------------------

    def register_region(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        region_type: int,
        spawn: bool = False,
        immortal: bool = False,   # ignored — all nodes are mortal (kept for API compat)
    ) -> int:
        """Register a 2-D spatial region with uniform behavioural flags.

        Args:
            y0, y1: Row bounds [y0, y1) in grid coordinates.
            x0, x1: Column bounds [x0, x1) in grid coordinates.
            region_type: 0 = sensory, 1 = dynamic, 2 = workspace (informational).
            spawn:    If True, nodes inside this region may reproduce.
            immortal: Deprecated — ignored.  All nodes are mortal; any node
                      that drops below the death threshold is removed regardless
                      of region type.  The parameter is kept for call-site
                      backward compatibility only.

        Returns:
            Region index (0-based). Raises RuntimeError if registry is full.

        Note:
            Must be called before the first step().  The Z dimension is not
            part of region bounds — all Z layers of (y0:y1, x0:x1) share the
            same flags, consistent with how cluster-based placement works.
        """
        if immortal:
            logger.warning(
                "register_region(): immortal=True is ignored — all nodes are mortal."
            )
        idx = int(_region_count[None])
        if idx >= MAX_REGIONS:
            raise RuntimeError(
                f"Region registry full (MAX_REGIONS={MAX_REGIONS}). "
                "Increase MAX_REGIONS if you need more regions."
            )
        _region_y0[idx]       = int(y0)
        _region_y1[idx]       = int(y1)
        _region_x0[idx]       = int(x0)
        _region_x1[idx]       = int(x1)
        _region_type[idx]     = int(region_type)
        _region_spawn[idx]    = 1 if spawn else 0
        _region_count[None]   = idx + 1
        logger.info(
            "Region %d registered: type=%d y=[%d,%d) x=[%d,%d) spawn=%s",
            idx, region_type, y0, y1, x0, x1, spawn,
        )
        return idx

    def get_region_info(self) -> List[Dict[str, Any]]:
        """Return a list of dicts describing all registered regions.

        Useful for monitoring, logging, and test assertions.
        """
        n = int(_region_count[None])
        regions = []
        for r in range(n):
            regions.append({
                'index':    r,
                'type':     int(_region_type[r]),
                'y0':       int(_region_y0[r]),
                'y1':       int(_region_y1[r]),
                'x0':       int(_region_x0[r]),
                'x1':       int(_region_x1[r]),
                'spawn':    bool(_region_spawn[r]),
            })
        return regions

    def clear_regions(self) -> None:
        """Remove all registered regions (e.g., before a grid resize).

        Warning: call this only when no step() is running.
        """
        _region_count[None] = 0
        logger.info("Region registry cleared.")

    # -----------------------------------------------------------------------
    # Public API — runtime parameter updates (hot-reload from config)
    # -----------------------------------------------------------------------

    def update_parameters(self, **kwargs) -> None:
        """Update engine runtime parameters without restart.

        Accepts any subset of: transfer_strength, transfer_dt, gate_threshold,
        spawn_threshold, death_threshold, energy_cap, child_energy_fraction.
        Unknown keys are silently ignored.

        Thread safety: each parameter is a simple Python float assignment
        (atomic under the GIL). However, a multi-parameter update is NOT
        atomic — step() may observe a partially-applied set if called from
        another thread between individual setattr() calls. In the Qt event
        loop (single-threaded), this is not an issue.
        """
        _MUTABLE = {
            'transfer_strength', 'transfer_dt', 'gate_threshold',
            'spawn_threshold', 'death_threshold', 'energy_cap',
            'child_energy_fraction',
        }
        changed = []
        for key, value in kwargs.items():
            if key in _MUTABLE and hasattr(self, key):
                old = getattr(self, key)
                if old != value:
                    setattr(self, key, value)
                    changed.append(f"{key}: {old} -> {value}")
        if changed:
            logger.info("Engine parameters updated: %s", "; ".join(changed))

    # -----------------------------------------------------------------------
    # Public API — simulation
    # -----------------------------------------------------------------------

    def step(self, **kwargs) -> Dict[str, Any]:
        """
        One full simulation step:
            0. Rebuild grid-node map for lock-and-key   (Taichi kernel)
            1. DNA-based neighborhood energy transfer   (Taichi kernel)
            2. Sync node energies from field            (Taichi kernel)
            3. Death                                    (Taichi kernel)
            4. Spawn                                    (Taichi kernel)
            5. Clamp energy field                       (Taichi kernel)
        """
        with self.step_lock:
            t_start = time.time()

            # Ensure prior Taichi kernels (e.g. inject_sensory_data) have finished
            # writing to energy_field before we read it in DNA transfer.
            ti.sync()

            # Step 0: Rebuild grid-node map for lock-and-key DNA lookups
            _clear_grid_map(self.grid_node_id)
            if self._count > 0:
                _build_grid_map(self.grid_node_id)

            # Step 1: DNA-based energy transfer
            t0 = time.time()
            if self._count > 0:
                _dna_transfer_kernel(
                    self.energy_field,
                    self.grid_node_id,
                    self.H, self.W, self.D,
                    self.transfer_dt,
                    self.gate_threshold,
                    self.frame_counter,
                    self.transfer_strength,
                    self.energy_cap * CONTRIB_CLAMP_FRACTION,
                )
            transfer_time = time.time() - t0

            # Step 2: Sync node energies from the field
            # Only the grid-map owner of each cell syncs — prevents multi-occupancy
            # nodes from getting identical energy and making synchronized decisions.
            if self._count > 0:
                _sync_energy_from_field(self.energy_field, self.grid_node_id)

            # Steps 4 & 5: Death + Spawn (reset per-step counters first)
            t0 = time.time()
            _deaths_count[None] = 0
            _spawns_count[None] = 0

            if self._count > 0:
                _death_kernel(self.death_threshold)

                # Adaptive spawn threshold and cost, both derived from avg dynamic node energy.
                #   birth_threshold = avg_dyn_e × 1.10
                #     → only nodes with above-average energy can reproduce
                #   birth_cost = birth_threshold × 0.75
                #     → parent keeps 25% of the threshold surplus after spawning
                #   child_energy = birth_cost × child_energy_fraction
                #
                # Recomputed every 10 steps (on-GPU reduction — reads 2 scalars instead of ~48MB arrays).
                if self.frame_counter % 10 == 0 or not hasattr(self, '_cached_avg_dyn_e'):
                    _reduce_dynamic_energy()
                    dyn_count = int(_dyn_node_count[None])
                    if dyn_count > 0:
                        self._cached_avg_dyn_e = float(_dyn_energy_sum[None]) / dyn_count
                    else:
                        self._cached_avg_dyn_e = self.spawn_threshold
                avg_dyn_e = self._cached_avg_dyn_e
                # Clamp spawn threshold to energy_cap × SPAWN_CAP_FRACTION — otherwise
                # at high avg energy the threshold exceeds cap and nodes can never spawn.
                effective_spawn_threshold = min(avg_dyn_e * SPAWN_ABOVE_AVG_FACTOR,
                                                self.energy_cap * SPAWN_CAP_FRACTION)
                effective_spawn_cost      = effective_spawn_threshold * SPAWN_COST_FRACTION
                effective_child_energy    = effective_spawn_cost * self.child_energy_fraction

                _spawn_kernel(
                    self.H, self.W, self.D,
                    effective_spawn_threshold, effective_spawn_cost, effective_child_energy,
                    self._spawn_limit(),
                    self._count,   # snapshot before spawn — prevents chain-spawn
                )

            deaths = int(_deaths_count[None])
            spawns = int(_spawns_count[None])

            # Keep Python-side count in sync with the Taichi scalar field.
            # Taichi field access (field[None]) automatically synchronizes the GPU.
            self._count = int(_node_count[None])
            rules_time = time.time() - t0

            # Step 6: Clamp field to the hard 0–255 energy range.
            # Fixed constants — not self.death_threshold / self.energy_cap — so
            # a misconfigured runtime value can never push a node above 255 or
            # below 0 regardless of config drift.
            _clamp_kernel(self.energy_field, 0.0, 255.0)

            # Warn at 90% capacity: gives operators ~400k slots of headroom before hard stop
            if self._count > MAX_NODES * 0.9:
                logger.warning(
                    "Node buffer at %.0f%% capacity (%s / %s)",
                    100 * self._count / MAX_NODES, f"{self._count:,}", f"{MAX_NODES:,}",
                )

            self.total_spawns += spawns
            self.total_deaths += deaths
            self.frame_counter += 1

            total_time = time.time() - t_start

            if self.frame_counter % 90 == 0:
                logger.info(
                    "STEP | Total: %.1fms | Transfer: %.1fms | Rules: %.1fms | "
                    "Nodes: %s | Spawns: %d | Deaths: %d",
                    total_time * 1000, transfer_time * 1000, rules_time * 1000,
                    f"{self._count:,}", spawns, deaths,
                )

            total_grid_ops = self.grid_operations_per_step
            ops_per_sec    = total_grid_ops / total_time if total_time > 0 else 0.0

            return {
                "total_time":            total_time,
                "transfer_time":         transfer_time,
                "rules_time":            rules_time,
                "spawns":                spawns,
                "deaths":                deaths,
                "num_nodes":             self._count,
                "total_spawns":          self.total_spawns,
                "total_deaths":          self.total_deaths,
                "grid_operations":       total_grid_ops,
                "operations_per_second": ops_per_sec,
                "avg_energy":            float(self.energy_field.mean().item()),
                "max_energy":            float(self.energy_field.max().item()),
                "transfer_mode":         "dna_taichi",
            }

    def _spawn_limit(self) -> int:
        """Rate-limit spawning per step based on current node count.

        Tiered limits prevent runaway population explosions while still allowing
        rapid recovery from small populations:
          < 300k  → 5 000 spawns/step  (fast growth from low density)
          < 600k  → 2 000 spawns/step  (moderate growth)
          < 1M    → 1 000 spawns/step  (slow growth near mid-capacity)
          ≥ 1M    →   200 spawns/step  (near-stable maintenance only)
        """
        n = self._count
        if n < SPAWN_TIER_1_THRESHOLD: return SPAWN_TIER_1_LIMIT
        if n < SPAWN_TIER_2_THRESHOLD: return SPAWN_TIER_2_LIMIT
        if n < SPAWN_TIER_3_THRESHOLD: return SPAWN_TIER_3_LIMIT
        return SPAWN_TIER_4_LIMIT

    # -----------------------------------------------------------------------
    # Public API — node management
    # -----------------------------------------------------------------------

    def add_nodes_batch(
        self,
        positions:  List[Tuple],
        energies:   List[float],
        node_types: List[int],
        modalities: Optional[List[int]] = None,
        dna: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Add N nodes at once with 3D positions (y, x, z).
        DNA is written to _node_dna[slot, 0..2], not packed in _node_state.
        2-tuple positions are backward-compatible (z defaults to 0).

        If *dna* is provided it must be an [N, 3] int64 tensor on the correct
        device; otherwise random DNA is generated.
        """
        n = len(positions)
        if n == 0:
            return []

        # Unpack positions — support both (y, x) and (y, x, z)
        ys, xs, zs = [], [], []
        for p in positions:
            if len(p) == 3:
                ys.append(p[0]); xs.append(p[1]); zs.append(p[2])
            else:
                ys.append(p[0]); xs.append(p[1]); zs.append(0)

        dev = self.device

        new_ys    = torch.tensor(ys, device=dev, dtype=torch.int32) % self.H
        new_xs    = torch.tensor(xs, device=dev, dtype=torch.int32) % self.W
        new_zs    = torch.tensor(zs, device=dev, dtype=torch.int32) % self.D
        new_e     = torch.tensor(energies, device=dev, dtype=torch.float32)
        new_types = torch.tensor(node_types, device=dev, dtype=torch.int64)
        new_conn  = _random_conn_type(n, dev)
        new_dna   = dna if dna is not None else _random_dna_3d(n, dev)  # [N, 3] int64

        if modalities is not None:
            new_modality = torch.tensor(modalities, device=dev, dtype=torch.int64)
        else:
            new_modality = torch.zeros(n, device=dev, dtype=torch.int64)

        # All nodes use the same random DNA and connection type — workspace and
        # sensory are spatial regions of the same grid, not special node subtypes.

        alive_bits = torch.ones(n, device=dev, dtype=torch.int64)
        new_state  = _pack_state_batch(alive_bits, new_types, new_conn, new_modality)

        start = self._count
        end   = start + n
        if end > MAX_NODES:
            logger.error(
                "add_nodes_batch: %d nodes would exceed MAX_NODES=%d; truncating", n, MAX_NODES
            )
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

    def add_node(self, position: Tuple, energy: float, node_type: int = 1) -> int:
        """Add a single node. Prefer add_nodes_batch() for bulk adds."""
        return self.add_nodes_batch([position], [energy], [node_type])[0]

    # -----------------------------------------------------------------------
    # Public API — energy injection
    # -----------------------------------------------------------------------

    def inject_sensory_data(
        self,
        pixel_data:  torch.Tensor,
        region:      Tuple[int, int, int, int],
        z:           int = 0,
    ) -> None:
        """Inject pixel data as a delta signal into live sensory nodes (ADR-001).

        Uses node-gated delta injection: for each grid cell in *region* at
        depth *z*, if a live node occupies that cell, the field receives
        ``pixel_value − node_energy`` (driving the node toward the pixel).
        Cells without a live node get no injection — their field energy
        drains naturally to zero.

        This replaces the previous hard-overwrite approach so that:
          - Sensory coverage follows live-node density (organic resolution)
          - Starved / dead sensory cells produce black output automatically
          - No energy is injected into cells the network hasn't populated

        Note: ``grid_node_id`` is populated by step()'s grid-map rebuild.
        The very first injection (before step() runs) is a no-op; the signal
        takes effect from frame 2 onward once the map is populated.

        pixel_data: [H, W] float32, values in 0–255 (raw pixel intensities).
        region:     (y0, y1, x0, x1) slice into energy_field.
        z:          Z layer index (default 0).
        """
        y0, y1, x0, x1 = region
        h = min(y1 - y0, pixel_data.shape[0])
        w = min(x1 - x0, pixel_data.shape[1])

        data = pixel_data[:h, :w].to(dtype=torch.float32, device=self.device).contiguous()

        _inject_sensory_delta_kernel(
            self.energy_field, self.grid_node_id, data, y0, x0, h, w, z,
        )

        self._injection_counter += 1
        if self._injection_counter % 60 == 0:
            logger.info("SENSORY | %dx%d z=%d | mean=%.1f", h, w, z,
                        float(data.mean().item()))

    def add_energy_at(self, position, amount: float) -> None:
        """Add energy at a single grid cell (clamped to grid bounds).

        *position* may be a 2-tuple ``(y, x)`` (backward compat, z defaults
        to 0) or a 3-tuple ``(y, x, z)``.
        """
        y = max(0, min(self.H - 1, position[0]))
        x = max(0, min(self.W - 1, position[1]))
        z = max(0, min(self.D - 1, position[2])) if len(position) > 2 else 0
        self.energy_field[y, x, z] = min(
            float(self.energy_field[y, x, z].item()) + amount, self.energy_cap
        )

    # -----------------------------------------------------------------------
    # Public API — audio region helpers
    # -----------------------------------------------------------------------

    def inject_audio_data(
        self,
        spectrum_2d: torch.Tensor,
        region: Tuple[int, int, int, int],
    ) -> None:
        """Inject a 2-D audio spectrum into an audio sensory region.

        Data values are written directly as energy — no gain or bias.

        Parameters
        ----------
        spectrum_2d : torch.Tensor
            Shape ``(rows, cols)`` float32.
        region : tuple
            ``(y0, y1, x0, x1)`` slice into ``energy_field``.
        """
        y0, y1, x0, x1 = region
        h = min(y1 - y0, spectrum_2d.shape[0])
        w = min(x1 - x0, spectrum_2d.shape[1])

        data = spectrum_2d[:h, :w].to(dtype=torch.float32, device=self.device).contiguous()
        _inject_sensory_kernel(
            self.energy_field, data, y0, x0, h, w, 0,
        )

    def read_audio_workspace_energies(
        self,
        region: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Read energy field slice for an audio workspace region.

        Returns a CPU float32 tensor of shape ``(h, w)``.
        Unlike ``read_workspace_energies()`` this is a direct field slice —
        no workspace node cache needed because audio workspace nodes occupy a
        dense regular grid.
        """
        y0, y1, x0, x1 = region
        return self.energy_field[y0:y1, x0:x1, :].sum(dim=-1).cpu()

    # -----------------------------------------------------------------------
    # Public API — data access
    # -----------------------------------------------------------------------

    def get_node_data(self) -> Dict[str, Any]:
        """
        Return positions, energies, and types for all alive nodes (CPU numpy).

        Reads Taichi fields via to_numpy() — a single GPU→CPU transfer.
        Expensive; cache externally if calling every frame.
        """
        n = self._count
        if n == 0:
            return {"positions": [], "energies": np.array([]),
                    "types": np.array([]), "num_nodes": 0}

        state_np  = _node_state.to_numpy()[:n]
        energy_np = _node_energy.to_numpy()[:n]
        pos_y_np  = _node_pos_y.to_numpy()[:n]
        pos_x_np  = _node_pos_x.to_numpy()[:n]

        alive_mask  = state_np != 0
        alive_state = state_np[alive_mask]
        types       = ((alive_state >> BINARY_NODE_TYPE_SHIFT)
                       & BINARY_TYPE_MASK).astype(np.int8)

        return {
            "positions": list(zip(pos_y_np[alive_mask].tolist(),
                                  pos_x_np[alive_mask].tolist())),
            "energies":  energy_np[alive_mask],
            "types":     types,
            "num_nodes": int(alive_mask.sum()),
        }

    def get_energy_field(self) -> torch.Tensor:
        """Return the current energy field as a CPU tensor (copy)."""
        return self.energy_field.cpu()

    def read_workspace_energies(self, region: Tuple[int, int, int, int]) -> torch.Tensor:
        """Read workspace region energy as a direct field slice (ADR-001).

        Returns a [h, w] float32 CPU tensor where each value is the maximum
        energy across Z layers at that (y, x) position.

        No node scanning, no cache, no Taichi→CPU field reads.

        Why this works:
          All nodes are mortal.  A workspace cell that has a live node will
          hold energy through DNA transfer and reflect that node's activity.
          A workspace cell whose node has starved (or was never populated)
          receives no delta injection (sensory delta injection is node-gated)
          and drains to zero through field clamping — it appears black in the
          display or silent in audio without any special-casing.  This is the
          intended thermodynamic behaviour: dead regions naturally go dark.
        """
        y0, y1, x0, x1 = region
        # max over Z: display pixel shows the brightest depth layer
        return self.energy_field[y0:y1, x0:x1, :].max(dim=-1).values.cpu()

    def _build_workspace_cache(self, y0: int, y1: int, x0: int, x1: int) -> None:
        """RETIRED (ADR-001) — read_workspace_energies now uses a direct field slice.

        Kept as a no-op so that any lingering call sites don't crash.
        """
        logger.debug("_build_workspace_cache() called but is retired (ADR-001); no-op.")

    def render_connection_heatmap(self) -> torch.Tensor:
        """
        Render [H, W, 3] uint8 RGB heatmap of node connection types.

            Excitatory   (0) → green
            Inhibitory   (1) → red
            Gated        (2) → blue
            Plastic      (3) → yellow
            Anti-Plastic (4) → magenta
            Damped       (5) → cyan
            Resonant     (6) → white
            Capacitive   (7) → orange
            Dead              → black

        Returns a CPU tensor.
        """
        heatmap = torch.zeros(self.H, self.W, 3, dtype=torch.float32, device=self.device)

        n = self._count
        if n == 0:
            return (heatmap * 255).to(torch.uint8).cpu()

        state_np  = _node_state.to_numpy()[:n]
        energy_np = _node_energy.to_numpy()[:n]
        pos_y_np  = _node_pos_y.to_numpy()[:n]
        pos_x_np  = _node_pos_x.to_numpy()[:n]

        alive_mask  = state_np != 0
        alive_state = state_np[alive_mask]
        alive_e     = energy_np[alive_mask]
        alive_y     = torch.tensor(pos_y_np[alive_mask], dtype=torch.long, device=self.device)
        alive_x     = torch.tensor(pos_x_np[alive_mask], dtype=torch.long, device=self.device)

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

        return (heatmap * 255).to(torch.uint8).cpu()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return engine metrics. Field stats from PyTorch; node type counts
        from a single numpy read of Taichi fields.
        """
        total_energy = float(self.energy_field.sum().item())
        avg_energy   = float(self.energy_field.mean().item())

        n = self._count
        if n > 0:
            state_np = _node_state.to_numpy()[:n]
            alive    = state_np != 0
            types    = ((state_np[alive] >> BINARY_NODE_TYPE_SHIFT)
                        & BINARY_TYPE_MASK).astype(np.int64)
            counts   = np.bincount(types, minlength=3)
            sensory_count   = int(counts[0])
            dynamic_count   = int(counts[1])
            workspace_count = int(counts[2])
            alive_count     = int(alive.sum())
        else:
            sensory_count = dynamic_count = workspace_count = alive_count = 0

        return {
            'alive_count':          alive_count,
            'sensory_count':        sensory_count,
            'dynamic_count':        dynamic_count,
            'workspace_count':      workspace_count,
            'total_spawns':         self.total_spawns,
            'total_deaths':         self.total_deaths,
            'total_energy':         total_energy,
            'avg_energy':           avg_energy,
            'workspace_energy_avg': 0.0,
            'workspace_energy_min': 0.0,
            'workspace_energy_max': 0.0,
        }

    # -----------------------------------------------------------------------
    # Backward-compatible property wrappers
    # (main.py adapter accesses these for its cached metrics)
    # -----------------------------------------------------------------------

    @property
    def node_state(self) -> torch.Tensor:
        """Live node states as a CPU tensor ([:_count] slice)."""
        return torch.from_numpy(_node_state.to_numpy()[:self._count])

    @property
    def node_energy(self) -> torch.Tensor:
        """Live node energies as a CPU tensor ([:_count] slice)."""
        return torch.from_numpy(_node_energy.to_numpy()[:self._count])

    @property
    def node_positions_y(self) -> torch.Tensor:
        return torch.from_numpy(_node_pos_y.to_numpy()[:self._count])

    @property
    def node_positions_x(self) -> torch.Tensor:
        return torch.from_numpy(_node_pos_x.to_numpy()[:self._count])


# =============================================================================
# Standalone helpers — used by main.py's HybridNeuralSystemAdapter
# (mirrors helpers previously in hybrid_grid_engine.py)
# =============================================================================

def is_alive(state: torch.Tensor) -> torch.Tensor:
    """Bool mask: True for alive nodes (state != 0)."""
    return state != 0


def extract_node_type(state: torch.Tensor) -> torch.Tensor:
    """Extract 2-bit node type (0=sensory, 1=dynamic, 2=workspace)."""
    return ((state >> BINARY_NODE_TYPE_SHIFT) & BINARY_TYPE_MASK).to(torch.int8)
