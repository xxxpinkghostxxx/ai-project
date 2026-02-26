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

Node capacity: pre-allocated at MAX_NODES (4 million) at startup.
Dead nodes (state == 0) contribute zero energy automatically.

Bit layout (64-bit int64 per node, unchanged from config.py):
    bit 63      : alive flag
    bits 62-61  : node_type  (0=sensory, 1=dynamic, 2=workspace)
    bits 60-59  : conn_type  (0=excitatory, 1=inhibitory, 2=gated, 3=plastic)
    bits 58-19  : DNA × 8 neighbors, 5 bits each
    bits 18-0   : reserved
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
    CONN_TYPE_WEIGHT_TABLE,
    SPAWN_TIER_1_THRESHOLD,
    SPAWN_TIER_2_THRESHOLD,
    SPAWN_TIER_3_THRESHOLD,
    SPAWN_TIER_1_LIMIT,
    SPAWN_TIER_2_LIMIT,
    SPAWN_TIER_3_LIMIT,
    SPAWN_TIER_4_LIMIT,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Taichi initialization — must happen before any field/kernel definition
# =============================================================================
_TAICHI_ARCH = ti.cuda if torch.cuda.is_available() else ti.cpu
ti.init(arch=_TAICHI_ARCH, device_memory_fraction=0.6)
logger.info("Taichi initialized: arch=%s", _TAICHI_ARCH)

# =============================================================================
# Constants
# =============================================================================

MAX_NODES = 4_000_000   # Pre-allocated ceiling. ~144 MB VRAM for all 4 fields.

# Moore neighborhood offsets (8 directions): up, down, left, right, 4 diagonals
_NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

# DNA bit shifts for each of the 8 neighbor directions
_DNA_SHIFTS = [BINARY_DNA_BASE_SHIFT + d * BINARY_DNA_BITS_PER_NEIGHBOR
               for d in range(8)]

# =============================================================================
# Module-level Taichi fields
# (globally accessible by all kernels below; one global engine per process)
# =============================================================================

# Node data
_node_state  = ti.field(dtype=ti.i64, shape=MAX_NODES)  # packed binary state
_node_energy = ti.field(dtype=ti.f32, shape=MAX_NODES)  # working energy
_node_pos_y  = ti.field(dtype=ti.i32, shape=MAX_NODES)  # Y grid coordinate
_node_pos_x  = ti.field(dtype=ti.i32, shape=MAX_NODES)  # X grid coordinate
_node_count  = ti.field(dtype=ti.i32, shape=())          # high-water mark

# Per-step event counters (reset before each step)
_deaths_count = ti.field(dtype=ti.i32, shape=())
_spawns_count = ti.field(dtype=ti.i32, shape=())

# Constant lookup tables (filled once at engine init)
_neighbor_dy  = ti.field(dtype=ti.i32, shape=8)
_neighbor_dx  = ti.field(dtype=ti.i32, shape=8)
_dna_shifts   = ti.field(dtype=ti.i32, shape=8)
_weight_table = ti.field(dtype=ti.f32, shape=(4, 4))   # conn_type × (exc, inh, gate, plastic)

# =============================================================================
# Taichi kernels (module-level — support ti.types.ndarray() for PyTorch tensors)
# =============================================================================

@ti.kernel
def _dna_transfer_kernel(
    energy_field: ti.types.ndarray(),   # PyTorch CUDA tensor, zero-copy
    H: int,
    W: int,
    dt: float,
    gate_threshold: float,
):
    """
    For every live node: compute energy exchange with its 8 DNA-weighted neighbors.

    Each of the 8 neighbor directions d has a 5-bit probability p_d packed into
    the node's int64 state (bits 19-58, 5 bits each). Transfer formula:

        transfer_d = (neighbor_energy - node_energy)
                   × (p_d / 31.0)          ← DNA probability
                   × combined_weight        ← connection type weighting
                   × dt

    combined_weight: excitatory adds flow, inhibitory subtracts, gated fires only
    above threshold, plastic adds a constant baseline.

    Dead nodes (state == 0) are skipped entirely.
    ti.atomic_add handles multiple nodes sharing the same grid cell.
    energy_field is the PyTorch CUDA tensor — accessed zero-copy.
    """
    for i in range(_node_count[None]):
        state = _node_state[i]
        if state == 0:
            continue

        py = _node_pos_y[i]
        px = _node_pos_x[i]
        energy = _node_energy[i]

        conn_type = int((state >> 59) & 3)
        gate_fire = 1.0 if energy > gate_threshold else 0.0

        combined = (
            _weight_table[conn_type, 0]                   # excitatory  (pull energy in)
          - _weight_table[conn_type, 1]                   # inhibitory  (push energy away)
          + _weight_table[conn_type, 2] * gate_fire       # gated       (only fires above threshold)
          + _weight_table[conn_type, 3]                   # plastic     (constant baseline)
        )

        total = 0.0
        for d in ti.static(range(8)):   # unrolled at compile time — no loop overhead at runtime
            ny = (py + _neighbor_dy[d] + H) % H
            nx = (px + _neighbor_dx[d] + W) % W
            dna_prob = float((state >> _dna_shifts[d]) & 31) / 31.0
            delta    = energy_field[ny, nx] - energy
            total   += delta * dna_prob * combined * dt

        _node_energy[i] += total
        ti.atomic_add(energy_field[py, px], total)


@ti.kernel
def _death_kernel(death_threshold: float):
    """
    Kill dynamic nodes (type == 1) whose energy falls below threshold.

    Workspace nodes (type == 2) are immortal — never killed here.
    Zeroing the state wipes alive, DNA, type, and conn fields in one operation.
    Death count is accumulated atomically in _deaths_count.
    """
    for i in range(_node_count[None]):
        state = _node_state[i]
        if state == 0:
            continue
        node_type = int((state >> 61) & 3)
        if node_type == 1 and _node_energy[i] < death_threshold:
            _node_state[i]  = 0
            _node_energy[i] = 0.0
            ti.atomic_add(_deaths_count[None], 1)


@ti.kernel
def _spawn_kernel(
    H: int,
    W: int,
    spawn_threshold: float,
    spawn_cost: float,
    child_energy: float,
    max_spawns: int,
):
    """
    Each live dynamic node (type == 1) with sufficient energy spawns one child.

    Child slot is claimed by atomically incrementing _node_count. If the buffer
    is full the parent is skipped. Child position is parent ± 1 cell (toroidal).
    Child DNA and connection type are random.

    Spawn count accumulated atomically in _spawns_count.
    """
    for i in range(_node_count[None]):
        state = _node_state[i]
        if state == 0:
            continue
        node_type = int((state >> 61) & 3)
        if node_type != 1:
            continue
        if _node_energy[i] < spawn_threshold:
            continue
        if _spawns_count[None] >= max_spawns:
            continue

        # Claim next slot atomically — parallel-safe
        slot = ti.atomic_add(_node_count[None], 1)
        if slot >= MAX_NODES:
            ti.atomic_sub(_node_count[None], 1)   # undo — buffer full
            continue

        # Random ±1 offset from parent (toroidal grid wrap)
        offset_y = int(ti.random() * 3.0) - 1
        offset_x = int(ti.random() * 3.0) - 1
        child_y  = (_node_pos_y[i] + offset_y + H) % H
        child_x  = (_node_pos_x[i] + offset_x + W) % W

        # Random 5-bit DNA for all 8 neighbor directions, packed into 40 bits
        dna_packed = ti.i64(0)
        for d in ti.static(range(8)):
            dna_val    = ti.min(int(ti.random() * 32.0), 31)   # 0..31
            dna_packed |= ti.i64(dna_val) << (19 + d * 5)

        conn_type = ti.min(int(ti.random() * 4.0), 3)

        # Pack child state: alive=1, type=1 (dynamic), random conn and DNA
        new_state = (
            (ti.i64(1) << 63)           # alive bit
          | (ti.i64(1) << 61)           # node type = dynamic (1)
          | (ti.i64(conn_type) << 59)   # connection type
          | dna_packed
        )

        _node_state[slot]  = new_state
        _node_energy[slot] = child_energy
        _node_pos_y[slot]  = child_y
        _node_pos_x[slot]  = child_x

        _node_energy[i] -= spawn_cost
        ti.atomic_add(_spawns_count[None], 1)


@ti.kernel
def _clamp_kernel(energy_field: ti.types.ndarray(), lo: float, hi: float):
    """Clamp every cell of energy_field to [lo, hi]. Parallel over all H×W cells."""
    for y, x in ti.ndrange(energy_field.shape[0], energy_field.shape[1]):
        v = energy_field[y, x]
        if v < lo:
            energy_field[y, x] = lo
        elif v > hi:
            energy_field[y, x] = hi


@ti.kernel
def _inject_sensory_kernel(
    energy_field: ti.types.ndarray(),
    data:         ti.types.ndarray(),
    y0: int, x0: int, h: int, w: int,
):
    """Write raw data values as energy into the sensory region of energy_field."""
    for dy, dx in ti.ndrange(h, w):
        energy_field[y0 + dy, x0 + dx] = data[dy, dx]


@ti.kernel
def _sync_energy_from_field(energy_field: ti.types.ndarray()):
    """
    Sync each live node's working energy from its position in energy_field.
    Called after DNA transfer so birth/death decisions see the current values.
    """
    for i in range(_node_count[None]):
        if _node_state[i] != 0:
            _node_energy[i] = energy_field[_node_pos_y[i], _node_pos_x[i]]


@ti.kernel
def _write_nodes_kernel(
    states:   ti.types.ndarray(),
    energies: ti.types.ndarray(),
    pos_y:    ti.types.ndarray(),
    pos_x:    ti.types.ndarray(),
    start:    int,
    n:        int,
):
    """Bulk-write N new nodes starting at slot `start`."""
    for i in range(n):
        _node_state[start + i]  = states[i]
        _node_energy[start + i] = energies[i]
        _node_pos_y[start + i]  = pos_y[i]
        _node_pos_x[start + i]  = pos_x[i]


# =============================================================================
# Module-level helpers (pure PyTorch, used by add_nodes_batch)
# =============================================================================

def _pack_state_batch(alive: torch.Tensor, node_type: torch.Tensor,
                      conn_type: torch.Tensor, dna_q: torch.Tensor) -> torch.Tensor:
    """Pack N nodes into int64 binary states (vectorized, no Python loop)."""
    state = (alive.to(torch.int64) << BINARY_ALIVE_BIT)
    state |= (node_type.to(torch.int64) << BINARY_NODE_TYPE_SHIFT)
    state |= (conn_type.to(torch.int64) << BINARY_CONN_TYPE_SHIFT)
    shifts = torch.tensor(_DNA_SHIFTS, device=dna_q.device, dtype=torch.int64)
    state |= (dna_q.to(torch.int64) << shifts.unsqueeze(0)).sum(dim=1)
    return state


def _random_conn_type(n: int, device: torch.device) -> torch.Tensor:
    """Uniformly random conn type: 25% each of excitatory/inhibitory/gated/plastic."""
    return torch.randint(0, 4, (n,), device=device, dtype=torch.int64)


def _random_dna(n: int, device: torch.device) -> torch.Tensor:
    """Random 5-bit DNA for 8 neighbors. Returns [N, 8] int64."""
    return torch.randint(0, BINARY_DNA_MAX_VALUE + 1, (n, 8),
                         device=device, dtype=torch.int64)


# =============================================================================
# TaichiNeuralEngine — Python manager class
# =============================================================================

class TaichiNeuralEngine:
    """
    GPU-accelerated neural simulation engine.

    This class manages configuration, PyTorch state (energy_field, FFT kernels),
    and the Python-side node count. All heavy GPU work is done by the module-level
    Taichi kernels above, which access module-level Taichi fields directly.

    Only one instance is supported per process (Taichi fields are global).

    Lifecycle
    ---------
    Nodes occupy slots 0.._count-1 in the module-level Taichi fields.
    Dead nodes (state == 0) are simply skipped in every kernel — their energy
    contributes nothing automatically. No compaction needed: 4M slots give
    massive headroom.

    FFT Diffusion
    -------------
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
        grid_size: Tuple[int, int] = (512, 512),
        node_spawn_threshold: float = 20.0,
        node_death_threshold: float = -10.0,
        node_energy_cap: float = 244.0,
        spawn_cost: float = 19.52,
        gate_threshold: float = 0.5,
        transfer_dt: float = 0.1,
        child_energy_fraction: float = 0.5,
        device: str = "cuda",
    ):
        if TaichiNeuralEngine._instance is not None:
            raise RuntimeError(
                "Only one TaichiNeuralEngine per process (Taichi fields are global). "
                "Destroy the existing instance before creating a new one."
            )
        TaichiNeuralEngine._instance = self

        self.grid_size = grid_size
        self.H, self.W = grid_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Node lifecycle thresholds
        self.spawn_threshold       = node_spawn_threshold
        self.death_threshold       = node_death_threshold
        self.energy_cap            = node_energy_cap
        self.spawn_cost            = spawn_cost           # static fallback only
        self.child_energy_fraction = child_energy_fraction
        self.child_energy          = spawn_cost * child_energy_fraction
        self.gate_threshold  = gate_threshold
        self.transfer_dt     = transfer_dt

        # Statistics
        self.total_spawns  = 0
        self.total_deaths  = 0
        self.frame_counter = 0
        # 20 ≈ 8 DNA neighbor reads + 8 energy_field reads + 4 field ops per cell
        self.grid_operations_per_step = self.H * self.W * 20

        # ---------------------------------------------------------------
        # PyTorch energy field (shared with Taichi kernels via ndarray)
        # ---------------------------------------------------------------
        self.energy_field = torch.zeros(
            self.H, self.W, dtype=torch.float32, device=self.device
        )

        # Python-side count (mirrors _node_count Taichi field)
        self._count = 0

        # Workspace read cache — built once on first call (workspace nodes are immortal)
        self._workspace_local_y: Optional[torch.Tensor] = None
        self._workspace_local_x: Optional[torch.Tensor] = None
        self._workspace_cache_valid = False
        self._workspace_cache_lock = threading.Lock()

        # Sensory injection counter
        self._injection_counter = 0

        # ---------------------------------------------------------------
        # Initialize constant lookup tables in module-level Taichi fields
        # ---------------------------------------------------------------
        for d, (dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
            _neighbor_dy[d] = dy
            _neighbor_dx[d] = dx
            _dna_shifts[d]  = _DNA_SHIFTS[d]

        for i, row in enumerate(CONN_TYPE_WEIGHT_TABLE):
            for j, val in enumerate(row):
                _weight_table[i, j] = val

        logger.info(
            "TaichiNeuralEngine initialized: grid=%s, MAX_NODES=%s, device=%s",
            grid_size, f"{MAX_NODES:,}", self.device,
        )

    def __del__(self):
        TaichiNeuralEngine._instance = None

    # -----------------------------------------------------------------------
    # Public API — simulation
    # -----------------------------------------------------------------------

    def step(self, **kwargs) -> Dict[str, Any]:
        """
        One full simulation step:
            1. DNA-based neighborhood energy transfer  (Taichi kernel)
            2. Sync node energies from field           (Taichi kernel)
            3. Death                                   (Taichi kernel)
            4. Spawn                                   (Taichi kernel)
            5. Clamp energy field                      (Taichi kernel)
        """
        t_start = time.time()

        # Step 1: DNA-based energy transfer
        t0 = time.time()
        if self._count > 0:
            _dna_transfer_kernel(
                self.energy_field,
                self.H, self.W,
                self.transfer_dt,
                self.gate_threshold,
            )
        transfer_time = time.time() - t0

        # Step 2: Sync node energies from the field
        # After DNA transfer the field already holds the correct per-cell values;
        # this call propagates any field contributions (e.g. multiple nodes sharing
        # a cell) back into each node's working energy scalar.
        if self._count > 0:
            _sync_energy_from_field(self.energy_field)

        # Steps 4 & 5: Death + Spawn (reset per-step counters first)
        t0 = time.time()
        _deaths_count[None] = 0
        _spawns_count[None] = 0

        if self._count > 0:
            _death_kernel(self.death_threshold)

            # Adaptive spawn threshold and cost, both derived from avg dynamic node energy.
            #   spawn_threshold = avg_dyn_e + 10% of energy_cap
            #     → self-regulating: rises when population is energy-rich (suppresses
            #       over-spawning), falls when scarce (encourages recovery).
            #   spawn_cost = 110% of avg_dyn_e
            #     → ensures only well-above-average nodes can afford to spawn,
            #       and the cost scales with current population health.
            #   child_energy = spawn_cost × child_energy_fraction
            #     → child starts with a proportional share of the spent cost.
            #
            # PERFORMANCE: Taichi→CPU numpy read is expensive at large node counts.
            # Spawn parameters change slowly, so recompute only every 10 steps;
            # use the cached value on other steps. This cuts the CPU roundtrip
            # overhead by ~10× without meaningfully affecting spawn behaviour.
            if self.frame_counter % 10 == 0 or not hasattr(self, '_cached_avg_dyn_e'):
                state_np  = _node_state.to_numpy()[:self._count]
                energy_np = _node_energy.to_numpy()[:self._count]
                dyn_mask  = (state_np != 0) & (((state_np >> 61) & 3) == 1)
                dyn_e     = energy_np[dyn_mask]
                self._cached_avg_dyn_e = (
                    float(dyn_e.mean()) if len(dyn_e) > 0 else self.spawn_threshold
                )
            avg_dyn_e = self._cached_avg_dyn_e
            effective_spawn_threshold = avg_dyn_e + 0.10 * self.energy_cap
            effective_spawn_cost      = 1.10 * avg_dyn_e
            effective_child_energy    = effective_spawn_cost * self.child_energy_fraction

            _spawn_kernel(
                self.H, self.W,
                effective_spawn_threshold, effective_spawn_cost, effective_child_energy,
                self._spawn_limit(),
            )

        deaths = int(_deaths_count[None])
        spawns = int(_spawns_count[None])

        # Keep Python-side count in sync with the Taichi scalar field.
        # Taichi field access (field[None]) automatically synchronizes the GPU.
        self._count = int(_node_count[None])
        rules_time = time.time() - t0

        # Step 6: Clamp field to valid energy range
        _clamp_kernel(self.energy_field, self.death_threshold, self.energy_cap)

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
        positions:  List[Tuple[int, int]],
        energies:   List[float],
        node_types: List[int],
    ) -> List[int]:
        """
        Add N nodes at once, packing each into a binary int64 state.
        Writes directly into the module-level Taichi fields via bulk-write kernel.
        """
        n = len(positions)
        if n == 0:
            return []

        ys, xs = zip(*positions)
        dev = self.device

        new_ys     = torch.tensor(ys, device=dev, dtype=torch.int32) % self.H
        new_xs     = torch.tensor(xs, device=dev, dtype=torch.int32) % self.W
        new_e      = torch.tensor(energies, device=dev, dtype=torch.float32)
        new_types  = torch.tensor(node_types, device=dev, dtype=torch.int64)
        new_conn   = _random_conn_type(n, dev)
        new_dna    = _random_dna(n, dev)

        # Workspace nodes (type=2) are energy buckets: excitatory + max DNA
        ws_mask = (new_types == 2)
        if ws_mask.any():
            new_conn[ws_mask] = 0                       # excitatory (absorb energy)
            new_dna[ws_mask]  = BINARY_DNA_MAX_VALUE    # max probability all 8 directions

        alive_bits = torch.ones(n, device=dev, dtype=torch.int64)

        new_state = _pack_state_batch(alive_bits, new_types, new_conn, new_dna)

        start = self._count
        end   = start + n
        if end > MAX_NODES:
            logger.error(
                "add_nodes_batch: %d nodes would exceed MAX_NODES=%d; truncating", n, MAX_NODES
            )
            n         = MAX_NODES - start
            end       = MAX_NODES
            if n <= 0:
                return []
            new_state  = new_state[:n]
            new_e      = new_e[:n]
            new_ys     = new_ys[:n]
            new_xs     = new_xs[:n]

        _write_nodes_kernel(new_state, new_e, new_ys, new_xs, start, n)

        # Sync Taichi scalar and Python counter
        _node_count[None] = end
        self._count = end

        # Stamp initial energy into the field so FFT sees it immediately
        self.energy_field[new_ys.long(), new_xs.long()] = torch.maximum(
            self.energy_field[new_ys.long(), new_xs.long()], new_e
        )

        # Invalidate workspace cache if workspace nodes were added
        if 2 in node_types:
            with self._workspace_cache_lock:
                self._workspace_cache_valid = False

        return list(range(start, end))

    def add_node(self, position: Tuple[int, int], energy: float, node_type: int = 1) -> int:
        """Add a single node. Prefer add_nodes_batch() for bulk adds."""
        return self.add_nodes_batch([position], [energy], [node_type])[0]

    # -----------------------------------------------------------------------
    # Public API — energy injection
    # -----------------------------------------------------------------------

    def inject_sensory_data(
        self,
        pixel_data:  torch.Tensor,
        region:      Tuple[int, int, int, int],
    ) -> None:
        """
        Write pixel data as energy into the sensory region.

        Data values are written directly — no gain, bias, or normalization.
        Sensory nodes are the inverse of workspace nodes: they output their
        assigned data value as energy into attached dynamic nodes.

        pixel_data: [H, W] float32.
        region:     (y0, y1, x0, x1) slice into energy_field.
        """
        y0, y1, x0, x1 = region
        h = min(y1 - y0, pixel_data.shape[0])
        w = min(x1 - x0, pixel_data.shape[1])

        data = pixel_data[:h, :w].to(dtype=torch.float32, device=self.device).contiguous()

        _inject_sensory_kernel(
            self.energy_field, data, y0, x0, h, w,
        )

        self._injection_counter += 1
        if self._injection_counter % 60 == 0:
            logger.info("SENSORY | %dx%d | mean=%.1f", h, w,
                        float(data.mean().item()))

    def add_energy_at(self, position: Tuple[int, int], amount: float) -> None:
        """Add energy at a single grid cell (clamped to grid bounds)."""
        y = max(0, min(self.H - 1, position[0]))
        x = max(0, min(self.W - 1, position[1]))
        self.energy_field[y, x] = min(
            float(self.energy_field[y, x].item()) + amount, self.energy_cap
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
            self.energy_field, data, y0, x0, h, w,
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
        return self.energy_field[y0:y1, x0:x1].cpu()

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
        """
        Read workspace node (type == 2) energies in the given region.

        Workspace nodes are immortal — positions never change.
        Position cache built once on first call; subsequent calls are a single
        indexed tensor read, no Taichi scan.

        Returns: [h, w] float32 CPU tensor (zeros where no workspace node).
        """
        y0, y1, x0, x1 = region
        h, w = y1 - y0, x1 - x0

        with self._workspace_cache_lock:
            if not self._workspace_cache_valid:
                self._build_workspace_cache(y0, y1, x0, x1)

        if self._workspace_local_y is None or len(self._workspace_local_y) == 0:
            return torch.zeros((h, w), dtype=torch.float32)

        energy_grid = torch.zeros((h, w), device=self.device, dtype=torch.float32)
        energy_grid[self._workspace_local_y, self._workspace_local_x] = (
            self.energy_field[
                self._workspace_local_y + y0,
                self._workspace_local_x + x0,
            ]
        )
        return energy_grid.cpu()

    def _build_workspace_cache(self, y0: int, y1: int, x0: int, x1: int) -> None:
        """Scan Taichi fields once to find all workspace nodes in the region."""
        n = self._count
        if n == 0:
            self._workspace_local_y = torch.tensor([], dtype=torch.long, device=self.device)
            self._workspace_local_x = torch.tensor([], dtype=torch.long, device=self.device)
            self._workspace_cache_valid = True
            return

        state_np = _node_state.to_numpy()[:n]
        pos_y_np = _node_pos_y.to_numpy()[:n]
        pos_x_np = _node_pos_x.to_numpy()[:n]

        is_ws = ((state_np != 0) &
                 (((state_np >> BINARY_NODE_TYPE_SHIFT) & BINARY_TYPE_MASK) == 2))
        ws_y  = pos_y_np[is_ws]
        ws_x  = pos_x_np[is_ws]
        in_r  = (ws_y >= y0) & (ws_y < y1) & (ws_x >= x0) & (ws_x < x1)

        self._workspace_local_y = torch.tensor(
            ws_y[in_r] - y0, dtype=torch.long, device=self.device
        )
        self._workspace_local_x = torch.tensor(
            ws_x[in_r] - x0, dtype=torch.long, device=self.device
        )
        self._workspace_cache_valid = True

        logger.info(
            "Workspace cache: %d nodes in region [%d:%d, %d:%d]",
            int(in_r.sum()), y0, y1, x0, x1,
        )

    def render_connection_heatmap(self) -> torch.Tensor:
        """
        Render [H, W, 3] uint8 RGB heatmap of node connection types.

            Excitatory (0) → green
            Inhibitory (1) → red
            Gated      (2) → blue
            Plastic    (3) → yellow (R + G)
            Dead             → black

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
            ((alive_state >> BINARY_CONN_TYPE_SHIFT) & BINARY_TYPE_MASK),
            dtype=torch.long, device=self.device,
        )
        energy_t     = torch.tensor(alive_e, dtype=torch.float32, device=self.device)
        energy_range = self.energy_cap - self.death_threshold
        brightness   = ((energy_t.clamp(self.death_threshold, self.energy_cap)
                         - self.death_threshold) / energy_range
                        if energy_range > 0 else torch.zeros_like(energy_t))

        m = conn_type == 0; heatmap[alive_y[m], alive_x[m], 1] = brightness[m]  # green
        m = conn_type == 1; heatmap[alive_y[m], alive_x[m], 0] = brightness[m]  # red
        m = conn_type == 2; heatmap[alive_y[m], alive_x[m], 2] = brightness[m]  # blue
        m = conn_type == 3                                                         # yellow
        heatmap[alive_y[m], alive_x[m], 0] = brightness[m]
        heatmap[alive_y[m], alive_x[m], 1] = brightness[m]

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
