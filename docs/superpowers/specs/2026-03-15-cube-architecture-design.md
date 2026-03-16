# Cube Architecture — Design Specification

**Date:** 2026-03-15
**Status:** Approved
**Replaces:** Current rectangular grid engine (`taichi_engine.py`, `workspace/`)

---

## 1. Overview

Replace the existing 2048×2048×16 rectangular grid engine with a symmetric N×N×N cubic simulation volume. The cube uses face-stamped I/O panels for sensory input and workspace output, integer energy arithmetic, 6-neighbor DNA transfer, and sparse memory allocation.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid shape | N³ cubic, N=2048 default (configurable, power of 2) | Axis symmetry, no implicit gravity |
| Energy dtype | uint8 (0–255) storage, int16 transfer deltas | 4× memory savings, deterministic integer math |
| Neighbor count | 6 (±X, ±Y, ±Z) | Simpler DNA, less memory, faster kernels |
| DNA encoding | Single int32 per node (6 slots × 5 bits = 30 bits) | Down from 3 × int64 with 26 neighbors |
| Memory strategy | Taichi block-sparse SNode tree | Dense N=2048 field = 32 GB; sparse ≈ 0.25–0.5 GB |
| I/O panels | Arbitrary rectangles on outermost face layer | Shape driven by data source, not cube geometry |
| Z-axis rule | Sensory panels: z < N/2. Workspace panels: z ≥ N/2 | Universal intake-to-output direction |
| Spawn threshold | Dynamic: avg_energy * 1.10 | Self-regulating population |
| Engine replacement | Clean replacement of taichi_engine.py | No coexistence with old engine |
| Visualization | Full rethink — face projections, slices, 3D scatter | Not just workspace grid anymore |

---

## 2. Sparse Cubic Grid (`cube_grid.py`)

### Responsibility
Owns the N³ energy field, toroidal indexing, and clamping. Nothing else.

### Sparse SNode Tree

At N=2048, a dense float32 field = 2048³ × 4 bytes = 32 GB. Instead:

```
Root
 └─ pointer[N/B, N/B, N/B]     ← coarse blocks (B=64 → 32³ pointers)
     └─ dense[B, B, B]          ← 64³ = 262K cells per block
         └─ place(energy: u8)
```

- Blocks allocated on-demand when a node spawns into them or an I/O panel covers them
- Deactivated when all cells in a block reach zero energy and contain no alive nodes
- At typical occupancy (<5% of volume active), memory ≈ 0.25–0.5 GB

### Toroidal Wrap
N is always a power of 2: `wrapped = coord & (N - 1)`. Bitwise AND, no modulo.

### Clamp Kernel
```python
@ti.kernel
def _clamp_kernel(field):
    for x, y, z in field:  # only iterates active blocks
        field[x, y, z] = ti.math.clamp(field[x, y, z], 0, 255)
```

### API
```python
class CubeGrid:
    def __init__(self, N: int, block_size: int = 64)
    def clamp(self)
    def wrap(self, x, y, z) -> (int, int, int)
    def read_cell(self, x, y, z) -> int
    def write_cell(self, x, y, z, val: int)
    def activate_block(self, bx, by, bz)
    def get_energy_field()  # raw field reference for kernels
```

---

## 3. DNA Transfer Engine (`cube_dna.py`)

### Responsibility
6-neighbor lock-and-key energy exchange. No spawn, no death, no I/O.

### 6-Neighbor Offsets
```python
NEIGHBOR_OFFSETS_6 = [
    (-1, 0, 0), (+1, 0, 0),   # ±X
    ( 0,-1, 0), ( 0,+1, 0),   # ±Y
    ( 0, 0,-1), ( 0, 0,+1),   # ±Z
]
```

### DNA Structure
6 neighbors × 5 bits = 30 bits. Single int32 per node.

```
Bits [4:0]   → slot 0 (-X)   [MODE:1][PARAM:4]
Bits [9:5]   → slot 1 (+X)
Bits [14:10] → slot 2 (-Y)
Bits [19:15] → slot 3 (+Y)
Bits [24:20] → slot 4 (-Z)
Bits [29:25] → slot 5 (+Z)
```

### Transfer Kernel
```python
@ti.kernel
def dna_transfer(energy, node_state, node_dna, node_pos,
                 grid_map, N: int):
    for i in alive_list:
        x, y, z = node_pos[i]
        my_energy = energy[x, y, z]

        for slot in range(6):
            dx, dy, dz = OFFSETS_6[slot]
            nx = (x + dx) & (N - 1)
            ny = (y + dy) & (N - 1)
            nz = (z + dz) & (N - 1)

            neighbor_id = grid_map[nx, ny, nz]
            if neighbor_id == SENTINEL:
                continue

            my_dna = (node_dna[i] >> (slot * 5)) & 0x1F
            reverse_slot = slot ^ 1  # 0↔1, 2↔3, 4↔5
            neighbor_dna = (node_dna[neighbor_id] >> (reverse_slot * 5)) & 0x1F

            my_mode = (my_dna >> 4) & 1
            my_param = my_dna & 0xF
            n_mode = (neighbor_dna >> 4) & 1
            n_param = neighbor_dna & 0xF

            if my_mode != n_mode or my_param != n_param:
                continue

            transfer = compute_transfer(my_energy, my_mode, my_param)
            ti.atomic_add(energy[nx, ny, nz], transfer)
            ti.atomic_sub(energy[x, y, z], transfer)
```

### Connection Types
8 types encoded in MODE+PARAM bits: excitatory, inhibitory, gated, plastic, anti-plastic, damped, resonant, capacitive. `compute_transfer` maps these to integer transfer amounts.

---

## 4. Panel Registry (`cube_panels.py`)

### Responsibility
Register, inject, and read I/O panels on cube faces. Enforces the Z-axis placement rule.

### Panel Definition
```python
@dataclass
class Panel:
    name: str       # e.g. "visual_sensory"
    face: str       # "NEAR", "FAR", "LEFT", "RIGHT", "TOP", "BOTTOM"
    role: str       # "sensory" or "workspace"
    origin: (int, int)  # (u, v) on face's 2D coordinate system
    width: int
    height: int
```

### Face Coordinate Mapping

| Face | Fixed | Free axes | (u, v) → (x, y, z) |
|------|-------|-----------|---------------------|
| NEAR | z=0 | X, Y | (u, v, 0) |
| FAR | z=N-1 | X, Y | (u, v, N-1) |
| LEFT | x=0 | Y, Z | (0, u, v) |
| RIGHT | x=N-1 | Y, Z | (N-1, u, v) |
| BOTTOM | y=0 | X, Z | (u, 0, v) |
| TOP | y=N-1 | X, Z | (u, N-1, v) |

### Z-Axis Placement Rule (enforced at registration)
- **Sensory panels**: cells must have z < N/2
- **Workspace panels**: cells must have z ≥ N/2
- Violation raises an error

### Overlap Detection
Registry maintains a set of claimed (x, y, z) cells. Registration checks for conflicts.

### Pre-computed Face Maps
On registration, each panel's (row, col) → (x, y, z) mapping is computed once and stored as a Taichi ndarray. Kernels index into this map — no per-frame coordinate math.

### Inject Kernel (Sensory)
```python
@ti.kernel
def inject_panel(energy, data: ti.types.ndarray(dtype=ti.u8, ndim=2),
                 face_map: ti.types.ndarray(dtype=ti.i32, ndim=3)):
    for row, col in ti.ndrange(data.shape[0], data.shape[1]):
        idx = row * data.shape[1] + col
        x, y, z = face_map[idx, 0], face_map[idx, 1], face_map[idx, 2]
        energy[x, y, z] = data[row, col]  # overwrite, no blending
```

### Read Kernel (Workspace)
```python
@ti.kernel
def read_panel(energy, output: ti.types.ndarray(dtype=ti.u8, ndim=2),
               face_map: ti.types.ndarray(dtype=ti.i32, ndim=3)):
    for row, col in ti.ndrange(output.shape[0], output.shape[1]):
        idx = row * output.shape[1] + col
        x, y, z = face_map[idx, 0], face_map[idx, 1], face_map[idx, 2]
        output[row, col] = energy[x, y, z]
```

### API
```python
class PanelRegistry:
    def __init__(self, grid: CubeGrid)
    def register(self, panel: Panel) -> int
    def unregister(self, panel_id: int)
    def inject(self, panel_id: int, data: np.ndarray)   # sensory only
    def read(self, panel_id: int) -> np.ndarray          # workspace only
    def list_panels(self) -> list[Panel]
```

---

## 5. Step Pipeline (`cube_pipeline.py`)

### Responsibility
Per-tick execution order. Owns alive-node list, spawn, and death.

### Tick Order
```
1. INJECT       — PanelRegistry.inject() for all sensory panels
2. DNA TRANSFER — dna_transfer() over alive-list
3. CLAMP        — CubeGrid.clamp() → all cells to [0, 255]
4. SYNC         — each alive node reads its cell's clamped energy
5. DEATH        — energy < death_threshold → node dies
6. SPAWN        — energy > dynamic spawn_threshold → emit child
7. READ         — PanelRegistry.read() for all workspace panels
```

### Alive-Node List
```python
alive_list: ti.field(dtype=ti.i32, shape=MAX_NODES)
alive_count: ti.field(dtype=ti.i32, shape=())
```
Rebuilt each tick after death/spawn by a compaction kernel.

### Node State (struct of arrays)
```
pos_x[i]: int32        # grid position
pos_y[i]: int32
pos_z[i]: int32
energy[i]: int16       # node's own energy tracker
dna[i]: int32          # 6 slots × 5 bits = 30 bits
alive[i]: uint8        # 0 or 1
node_type[i]: uint8    # 0=sensory, 1=dynamic, 2=workspace
```

### Spawn Economics (Dynamic Threshold)
```
spawn_threshold  = avg_energy_all_nodes * 1.10
spawn_cost       = spawn_threshold * 0.50
child_energy     = spawn_cost * 0.80
parent_deduction = spawn_cost
```

The 20% leak (cost minus child energy) acts as a natural population brake. Higher average energy → higher threshold → harder to spawn. Self-regulating.

```python
@ti.kernel
def compute_avg_energy(energy, alive_list, alive_count: int) -> int:
    total = 0
    for idx in range(alive_count):
        i = alive_list[idx]
        total += energy[i]
    return total // alive_count

# In step():
avg = compute_avg_energy(...)
spawn_threshold = avg + (avg // 10)
spawn_cost = spawn_threshold // 2
child_energy = (spawn_cost * 4) // 5
```

All integer arithmetic.

### Death Kernel
```python
@ti.kernel
def death_kernel(alive, energy, node_type, threshold: int):
    for i in alive_list:
        if node_type[i] == 1 and energy[i] < threshold:
            alive[i] = 0
```

### Spawn Kernel
```python
@ti.kernel
def spawn_kernel(alive, energy, pos, dna,
                 threshold: int, cost: int, child_energy: int, N: int):
    for i in alive_list:
        if energy[i] > threshold:
            child_idx = ti.atomic_add(alive_count[None], 1)
            if child_idx >= MAX_NODES:
                continue
            offset = random_offset_6()
            child_x = (pos_x[i] + offset.x) & (N - 1)
            child_y = (pos_y[i] + offset.y) & (N - 1)
            child_z = (pos_z[i] + offset.z) & (N - 1)
            child_dna = mutate_dna(dna[i])
            energy[i] -= cost
            # write child fields...
```

### Grid Map Rebuild
```python
@ti.kernel
def build_grid_map(grid_map, pos, alive):
    for x, y, z in grid_map:
        grid_map[x, y, z] = SENTINEL
    for i in alive_list:
        ti.atomic_min(grid_map[pos_x[i], pos_y[i], pos_z[i]], i)
```

### API
```python
class CubePipeline:
    def __init__(self, grid: CubeGrid, dna: DnaTransfer,
                 panels: PanelRegistry, config: dict)
    def step(self)
    def get_alive_count(self) -> int
    def get_node_stats(self) -> dict
```

### Configuration
```python
{
    "death_threshold": 1,
    "max_nodes": 4_000_000,
    "dna_mutation_rate": 0.10
}
```

Spawn threshold/cost/child_energy are computed dynamically each tick — not configured.

---

## 6. Visualization (`cube_vis.py`)

### Responsibility
Render cube state for PyQt6 UI. Multiple view modes.

### View Modes

**1. Face Projection View**
Six faces arranged in cube net layout. Each rendered at reduced resolution (512×512) with energy→color mapping. Panel outlines overlaid.

```
         [TOP]
  [LEFT] [NEAR] [RIGHT] [FAR]
         [BOTTOM]
```

**2. Z-Slice View**
Slider selects z-layer (0 to N-1). Shows full X×Y plane at that depth as heatmap. Energy flow from sensory (low z) toward workspace (high z) visible.

**3. Axis Slice Triplet**
Three simultaneous orthogonal slices (XY, XZ, YZ). Medical-imaging style.

**4. Node Population View**
3D scatter plot of alive nodes colored by energy. Downsampled for performance.

**5. Panel Data View**
Raw display of each panel's current data — injected sensory alongside read workspace. Side-by-side input/output comparison.

### Sampling Strategy
- Face projections and slices computed by Taichi kernels that downsample on GPU
- Only active view mode computed each frame
- Panel data views already small — no downsampling needed

### API
```python
class CubeVisualizer:
    def __init__(self, grid: CubeGrid, panels: PanelRegistry,
                 pipeline: CubePipeline)
    def get_face_projection(self, face: str, resolution: int = 512) -> np.ndarray
    def get_z_slice(self, z: int) -> np.ndarray
    def get_axis_slices(self, x: int, y: int, z: int) -> tuple[np.ndarray, ...]
    def get_node_scatter(self, max_points: int = 50000) -> np.ndarray
    def get_panel_data(self, panel_id: int) -> np.ndarray
```

### New UI Widgets
- `CubeFaceWidget` — face projection heatmaps
- `CubeSliceWidget` — axis slices with slider controls
- `CubeScatterWidget` — 3D node scatter
- `CubePanelWidget` — raw panel I/O display

---

## 7. Integration & File Layout

### New Files
```
src/project/cube/
├── __init__.py
├── cube_grid.py           # sparse N³ field, clamp, toroidal wrap
├── cube_dna.py            # 6-neighbor transfer kernel
├── cube_panels.py         # panel registry, inject/read kernels
├── cube_pipeline.py       # step orchestration, spawn/death, alive-list
└── cube_vis.py            # face projections, slices, scatter, panel views

src/project/ui/
├── cube_face_widget.py    # NEW — face projection heatmap
├── cube_slice_widget.py   # NEW — axis slice + slider
├── cube_scatter_widget.py # NEW — 3D node scatter
└── cube_panel_widget.py   # NEW — raw panel I/O display
```

### Reworked Files
- **`main.py`** — creates CubeGrid → DnaTransfer → PanelRegistry → CubePipeline → CubeVisualizer, registers panels from config, wires vision/audio, starts UI
- **`config.py`** — 6-neighbor offsets, int32 DNA layout, uint8 energy constants
- **`pyg_config.json`** — cube config schema
- **`modern_main_window.py`** — new layout with cube visualization widgets
- **`modern_config_panel.py`** — cube-relevant controls

### Kept As-Is
- `vision.py` — screen capture → feeds panel inject
- `audio_capture.py` — FFT spectrum → feeds panel inject
- `audio_output.py` — reads workspace panel → playback
- `system/state_manager.py` — runtime toggles, metrics
- `system/global_storage.py`
- `utils/` — all utility modules

### Deleted
- `system/taichi_engine.py` — replaced by `cube/` modules
- `workspace/` (all files) — replaced by cube_panels + cube_vis
- `visualization/taichi_gui_manager.py` — replaced by new UI widgets

### Startup Sequence
```
1.  ti.init(arch=ti.cuda)
2.  grid = CubeGrid(N=2048)
3.  dna = DnaTransfer(grid)
4.  panels = PanelRegistry(grid)
5.  panels.register(Panel("visual_sensory",         "NEAR",  "sensory",   (0,0), 1920, 1080))
6.  panels.register(Panel("visual_workspace",       "FAR",   "workspace", (0,0), 512,  512))
7.  panels.register(Panel("audio_left_sensory",     "LEFT",  "sensory",   (0,0), 256,  256))
8.  panels.register(Panel("audio_left_workspace",   "LEFT",  "workspace", (0,0), 256,  256))
9.  panels.register(Panel("audio_right_sensory",    "RIGHT", "sensory",   (0,0), 256,  256))
10. panels.register(Panel("audio_right_workspace",  "RIGHT", "workspace", (0,0), 256,  256))
11. pipeline = CubePipeline(grid, dna, panels, config)
12. vis = CubeVisualizer(grid, panels, pipeline)
13. seed initial node clusters in interior
14. start vision/audio capture threads
15. start PyQt6 event loop with periodic pipeline.step()
```

### Data Flow
```
vision.py (screen capture)
    ↓ numpy uint8 array (1920×1080)
main.py → panels.inject("visual_sensory", frame)
    ↓
cube_panels.py stamps onto NEAR face (z=0)
    ↓
cube_pipeline.step()
    1. inject all sensory panels
    2. dna_transfer over alive-list
    3. clamp [0, 255]
    4. sync node energies
    5. death
    6. spawn (dynamic threshold)
    7. read all workspace panels
    ↓
cube_panels.py reads from FAR face (z=N-1)
    ↓ numpy uint8 array
cube_vis.py → UI widgets
audio_output.py → synthesize from workspace panel
```

---

## 8. What NOT To Do

| DO NOT | WHY |
|--------|-----|
| Place workspace panel at z < N/2 | Workspace lives in high-Z half only |
| Place sensory panel at z ≥ N/2 | Sensory lives in low-Z half only |
| Use z=1 or z=N-2 for any panel | Interior cells — not on a face |
| Offset stereo channels in Z or Y | Stereo is X-axis only |
| Place audio L and R on same X face | Collapses stereo |
| Inject to FAR face | That is a workspace surface |
| Read workspace from NEAR face | That is a sensory surface |
| Apply gain or bias to injected values | Corrupts absolute 0–255 scale |
| Scale or interpolate before injection | Destroys 1:1 spatial mapping |
| Different energy caps per modality | Breaks cross-modal DNA compatibility |
| Dense grid at N ≥ 1024 | Memory explosion — use sparse SNode |
| Non-cubic grid | Axis asymmetry, implicit gravity |
| FFT diffusion while nodes exist | Smears DNA-transfer energy gradients |
| Float arithmetic in energy transfer | Use integer math throughout |
| 26-neighbor Moore neighborhood | Spec requires 6-neighbor only (±X/Y/Z) |
