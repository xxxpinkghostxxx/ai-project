# 3D Volume with Island Clusters — Design

**Date:** 2026-03-03
**Status:** Approved

## Overview

Extend the AI neural simulation from a 2D flat grid to a true 3D volume, replacing strict band-based regions (sensory top / dynamic middle / workspace bottom) with scattered 3D island clusters. Dynamic nodes navigate the volume in XYZ space, evolving toward and between sensory and workspace clusters via spawn drift.

## Approach

Approach C — Downsized full 3D:
- Grid shrinks from `[2560, 1920]` to `[512, 512, 8]` (H×W×D)
- Full 3D energy field and node positions
- Separate DNA field (`_node_dna`) to accommodate 26 neighbors × 5 bits = 130-bit DNA
- Sensory and workspace nodes form scattered 3D island clusters throughout the volume
- Dynamic nodes drift ±1 in all three axes (X, Y, Z) symmetrically

## Hardware Context

GTX 1650 Max-Q (4 GB VRAM). The grid downsize actually reduces `energy_field` and `grid_node_id` from ~19MB each to ~8.4MB each, freeing VRAM headroom for the new `_node_dna` field.

---

## Section 1 — Grid & Fields

### Grid dimensions

```
H = 512    (rows / Y)
W = 512    (columns / X)
D = 8      (depth / Z)
```

Total cells: 512 × 512 × 8 = 2,097,152 (vs 4,915,200 previously).

### Field changes

| Field | Old | New |
|---|---|---|
| `energy_field` | `torch.f32[2560, 1920]` | `torch.f32[512, 512, 8]` |
| `grid_node_id` | `torch.i32[2560, 1920]` | `torch.i32[512, 512, 8]` |
| `_node_pos_y` | `ti.i32[MAX_NODES]` | unchanged |
| `_node_pos_x` | `ti.i32[MAX_NODES]` | unchanged |
| `_node_pos_z` | — | **new** `ti.i32[MAX_NODES]` |
| `_node_state` | `ti.i64[MAX_NODES]` (includes DNA bits 57-18) | `ti.i64[MAX_NODES]` (DNA bits freed → reserved) |
| `_node_dna` | — | **new** `ti.i64[MAX_NODES, 3]` |

### `_node_state` bit layout (updated)

```
bit 63       = alive flag
bits 62-61   = node_type  (0=sensory, 1=dynamic, 2=workspace)
bits 60-58   = conn_type  (0-7, eight connection types)
bits 57-3    = reserved
bits 2-0     = modality   (0=neutral, 1=visual, 2=audio_L, 3=audio_R)
```

DNA is removed from `_node_state` and moved to `_node_dna`.

### `_node_dna` layout

Three `ti.i64` words per node = 192 bits. Holds 26 neighbor DNA slots × 5 bits each = 130 bits (62 bits spare).

Neighbor index ordering (slot `n`, 0–25):

```
n = (dz+1)*9 + (dy+1)*3 + (dx+1)   where dz,dy,dx ∈ {-1,0,1}
skip n=13 (center, dz=dy=dx=0); remap above 13 down by 1
```

Each slot: `[MODE:1][PARAM:4]` — identical semantics to current 5-bit DNA slots.

Bit access: `slot_word = n // 12`, `slot_bit_offset = (n % 12) * 5`. Read 5 bits from `_node_dna[i, slot_word]`.

### Constant lookup tables (new/updated)

- `_reverse_dir`: `ti.i32[26]` — reverse direction index for each of 26 neighbor slots
- `_neighbor_dz`, `_neighbor_dy`, `_neighbor_dx`: `ti.i32[26]` — offsets per slot index
- `_dna_slot_word`: `ti.i32[26]` — precomputed word index per slot
- `_dna_slot_bit`: `ti.i32[26]` — precomputed bit offset per slot

---

## Section 2 — Cluster Placement

Strict horizontal band regions are replaced with scattered 3D island clusters. Cluster parameters are stored in `pyg_config.json` under a new `clusters` section.

### Config schema

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

### Placement algorithm (startup, `main.py`)

1. Randomly pick `sensory_count` center positions `(y, x, z)` from the full `[512, 512, 8]` volume
2. Randomly pick `workspace_count` centers, enforcing `min_cluster_separation` cells between any two cluster centers (sensory or workspace)
3. For each cluster center, place `N` nodes within a sphere of `radius` cells (random offset, clamp to grid bounds)
4. Sensory nodes initialized with `node_type=0`, workspace with `node_type=2`
5. Dynamic nodes seeded in the space between clusters: one node per `(x, z)` pair at Y mid-depth

### Energy injection (sensory clusters)

- Screen capture (1920×1080) downsampled to 512×512 (XY mapping)
- Energy injected into `energy_field[y, x, z]` only at cells belonging to a sensory cluster, weighted by `exp(-dist² / radius²)` from cluster center
- Not a full-band scan — per-cluster injection only

### Audio clusters (if `AUDIO_AVAILABLE`)

- Two additional cluster groups (audio_L, audio_R) with modality set accordingly
- Placed with same separation enforcement as sensory/workspace clusters
- FFT energy injected at audio cluster cells by frequency band

---

## Section 3 — Spawn & Movement

### 3D spawn offsets

```python
# All three axes: ±1 uniform random
offset_y = int(ti.random() * 3.0) - 1
offset_x = int(ti.random() * 3.0) - 1
offset_z = int(ti.random() * 3.0) - 1   # NEW

child_y = (py + offset_y + H) % H   # toroidal wrap
child_x = (px + offset_x + W) % W
child_z = (pz + offset_z + D) % D   # NEW, toroidal in Z
```

Z is fully symmetric with XY — no bias, no preferred direction.

### grid_node_id update

`_build_grid_kernel` now writes to `grid_node_id[y, x, z]`. Collision resolution (lowest node ID via `ti.atomic_min`) unchanged.

### Workspace nodes

Placed at computed cluster cell positions. Immortal and infertile (no spawn, no death). Have real `(y, x, z)` coordinates in the 3D volume.

### Death

Unchanged — nodes die when `_node_energy < death_threshold`. No Z-specific logic.

### Spawn tiers

Unchanged — adaptive rate limiting based on alive node count.

---

## Section 4 — DNA Transfer (3D)

### Neighbor enumeration

For sender at `(y, x, z)`, iterate all 26 Moore neighbors:

```python
for n in range(26):
    dz = _neighbor_dz[n]
    dy = _neighbor_dy[n]
    dx = _neighbor_dx[n]
    ny = (y + dy + H) % H
    nx = (x + dx + W) % W
    nz = (z + dz + D) % D
    j = grid_node_id[ny, nx, nz]
    if j < 0: continue   # empty cell
    # lock-and-key check, energy transfer...
```

### Lock-and-key (26-slot)

- Sender slot `n` → read PARAM bits from `_node_dna[i, slot_word][slot_bit : slot_bit+4]`
- Receiver reverse slot `r = _reverse_dir[n]` → read PARAM bits from `_node_dna[j, ...]`
- Compatible if `sender_PARAM & receiver_PARAM != 0` (bitwise AND nonzero)
- If compatible: transfer energy from `energy_field[y, x, z]` to `energy_field[ny, nx, nz]` scaled by conn_type modifier

### MODE=1 special behaviors

Unchanged (threshold, invert, pulse, absorb) — slot semantics identical, just more slots.

### DNA heredity

Child inherits all three `_node_dna` words from parent, then independently mutates each of 26 slots with 10% probability (single random bit flip within the 5-bit slot). Mutation rate unchanged.

### Reverse direction table

Precomputed `_reverse_dir[26]`: for slot `n` with offset `(dz, dy, dx)`, reverse is slot with offset `(-dz, -dy, -dx)`.

---

## Section 5 — Visualization

### Default view (max-pool projection)

```python
display[y, x] = max(energy_field[y, x, z] for z in range(D))
```

Existing three GGUI windows (energy, nodes, connections) show the max-pooled XY projection. The 3D volume appears as a top-down composite, with bright regions wherever any Z-layer is active.

### Z-slice view (Qt UI addition)

A Z-depth slider in the Qt config panel (or Taichi GGUI controls) switches each window between max-pool mode and single-Z-layer display. Allows "looking through" different depth layers.

### Cluster overlay

Cluster centers drawn as filled circles on the energy display:
- Sensory clusters: cyan
- Workspace clusters: orange

Overlay rendered in the Taichi GGUI visualization pass, not post-processed in PyTorch.

---

## Files to Modify

| File | Change |
|---|---|
| `system/taichi_engine.py` | Add `_node_pos_z`, `_node_dna[,3]`, `_neighbor_dz/dy/dx[26]`, `_reverse_dir[26]`, `_dna_slot_word/bit[26]`; rewrite `_dna_transfer_kernel`, `_spawn_kernel`, `_build_grid_kernel`, `_death_kernel` for 3D; remove DNA from `_node_state` |
| `main.py` | Replace band-based region init with cluster placement algorithm; 3D energy injection at cluster cells; update grid_size to `[512, 512, 8]` |
| `pyg_config.json` | `grid_size: [512, 512, 8]`; add `clusters` section |
| `config.py` | Update constants if needed (H, W, D references) |
| `workspace/workspace_node.py` | Update to work with 3D coordinates |
| `workspace/workspace_system.py` | Update cluster-based init |
| `workspace/mapping.py` | Update sensory→workspace mapping for 3D cluster-to-cluster |
| `ui/modern_main_window.py` | Add Z-slice slider / cluster overlay toggle |
| `workspace/visualization.py` | Add max-pool projection, cluster overlay rendering |

---

## Memory Estimate (GTX 1650 Max-Q, 4 GB VRAM)

| Field | Size |
|---|---|
| `energy_field [512, 512, 8]` | 8.4 MB |
| `grid_node_id [512, 512, 8]` | 8.4 MB |
| `_node_dna [MAX_NODES, 3]` (at MAX_NODES=500K) | 12 MB |
| `_node_pos_z [MAX_NODES]` | 2 MB |
| All other node fields (state, energy, pos_y/x, charge) | ~15 MB |
| **Total new footprint** | **~46 MB** |

Previous `energy_field + grid_node_id` alone were ~38 MB. Net change is small positive.
