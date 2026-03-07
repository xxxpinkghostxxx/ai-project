# ADR-001: Sensory and Workspace as Regions of the Singular Dynamic Node Grid

**Status:** Accepted
**Date:** 2026-03-07
**Deciders:** Lilac Initiative Core

---

## Context

The TaichiNeuralEngine operates on a 3D grid (`H × W × D`) where nodes carry a packed 64-bit state that encodes `node_type` (sensory=0, dynamic=1, workspace=2) in bits 62–61. Historically these three types evolved from a PyG graph system where sensory and workspace were entirely separate subsystems with their own lifecycle managers, energy ranges, and connection graphs.

After migrating to the Taichi engine the *data* was unified — all nodes share the same `_node_state`, `_node_energy`, `_node_dna`, and `_node_pos_{y,x,z}` fields — but the *behavioral rules* remained type-hardcoded inside kernels:

- `_death_kernel` only kills `node_type == 1`; sensory nodes never starve.
- `_spawn_kernel` skips any `node_type != 1`; sensory/workspace never reproduce.
- `SENSORY_REGION` and `WORKSPACE_REGION` are bare tuples living in the adapter, invisible to the engine.
- `pulse_energy()` injects energy globally, bypassing all node economics.
- Legacy energy caps (`NODE_ENERGY_CAP = 244`, `node_death_threshold = -10.0`) don't match the 0–255 design range.
- `mapping.py` (sensory→workspace node ID mapping) is dead code from the removed PyG layer.
- `WorkspaceNodeSystem._create_sensory_mapping()` calls that dead mapping on every startup.

The net effect: "sensory" and "workspace" are special-cased shadows of the dynamic grid rather than true first-class citizens of it.

---

## Decision

**Formalise sensory and workspace as named regions of the single dynamic node grid.**

Specifically:

1. Add a **region registry** (`_region_count`, `_region_y0/y1/x0/x1`, `_region_type`, `_region_spawn`, `_region_immortal`) as module-level Taichi fields, populated via `TaichiNeuralEngine.register_region()`.
2. **Refactor `_death_kernel`** to kill any node whose energy falls below threshold *unless* the node belongs to an immortal region — removing the `node_type == 1` hardcode.
3. **Refactor `_spawn_kernel`** to gate spawning on `_region_spawn` lookup — removing the `node_type != 1` hardcode.
4. **Call `engine.register_region()`** during `create_hybrid_neural_system()` so the engine is always self-describing.
5. **Remove `pulse_energy()`** from `HybridNeuralSystemAdapter` (external energy injection violates node economics).
6. **Fix legacy energy constants**: `NODE_ENERGY_CAP → 255`, `NODE_DEATH_THRESHOLD → 1.0` in `config.py`; `node_death_threshold → 1.0` in `pyg_config.json`.
7. **Deprecate `mapping.py`** and remove the dead `_create_sensory_mapping()` call from `WorkspaceNodeSystem`.

---

## Options Considered

### Option A: Node-type flags (status quo)
Keep `node_type == 1` hardcodes in kernels; never formally register regions.

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Extensibility | Low — adding a 4th type requires kernel edits |
| Correctness | Medium — sensory nodes that starve cannot die, causing phantom energy |
| Coupling | High — adapter holds region geometry; engine is blind |

**Pros:** No new fields; no kernel changes.
**Cons:** Adding audio regions, nested regions, or variable-immortality requires type-enum expansion and kernel rewrites. Sensory-node starvation is silently masked. Region geometry is duplicated across callers.

---

### Option B: Region registry in the engine (chosen)
Add a small (16-entry) region table as Taichi fields; kernels look up region properties instead of branching on type.

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Extensibility | High — new region types require only a `register_region()` call |
| Correctness | High — all nodes follow unified physics; immortality is explicit |
| Coupling | Low — engine is self-describing; adapter becomes a thin coordinator |

**Pros:** Sensory nodes can starve and die like dynamic nodes (realistic energy economics). Future regions (e.g., secondary audio workspace, inhibitory border zone) need zero kernel changes. Engine exposes `get_region_info()` for monitoring/debug.
**Cons:** 16-entry table adds ~640 bytes of Taichi field memory; `_get_node_region()` adds one `@ti.func` call per node per step (≈ 16 comparisons — negligible vs. the DNA transfer loop).

---

### Option C: Encode region in node modality bits
Overload the 3-bit modality field (bits 2–0) to also carry region identity.

| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Extensibility | Low — 3 bits → 8 region IDs maximum |
| Correctness | Medium — conflates signal channel with spatial region |
| Coupling | High — region semantics baked into bit encoding |

**Pros:** No new fields.
**Cons:** Modality already carries channel identity (visual/audio_L/audio_R). Conflating it with region type would prevent e.g. "audio-left sensory region" from being distinct from "visual sensory region."

---

## Trade-off Analysis

The region registry adds a 16-comparison lookup per live node per step. With 2M nodes at 60 fps this is ~120M comparisons/second — well within Taichi GPU throughput (billions of ops/sec). The benefit is that:

- **Immortality** becomes a registry property, not a type enum. The same workspace *type* could be mortal in one region and immortal in another if the design evolves.
- **Spawn control** becomes spatial rather than type-based. Future plans (e.g., restricted-growth border zones) become `register_region(..., spawn=False)` rather than new kernel branches.
- **Energy correctness** is restored: sensory nodes that stop receiving pixel input will deplete and die naturally, freeing slots for dynamic growth. This is the intended thermodynamic behaviour.

---

## Consequences

**Easier after this change:**
- Adding new region types (audio, inhibitory, modulator zones) without touching kernels.
- Monitoring region-level energy statistics (the registry provides bounds for GPU slices).
- Reasoning about the system — the engine is self-describing; no need to cross-reference adapter tuples.

**Harder after this change:**
- Engine startup order: `register_region()` must be called *before* the first `step()`.
- `TaichiNeuralEngine` is no longer stateless at module level — region state is mutable after init. Tests must reset `_region_count` between test runs.

**Will need revisiting:**
- If `grid_size` changes at runtime, registered region bounds may become invalid. A `clear_regions()` / `re-register` pattern should be documented.
- Workspace `WorkspaceNodeSystem` still maintains a Python-level shadow grid (16×16 `WorkspaceNode` objects). When workspace nodes can now be spawned inside a workspace region by relaxing `spawn=False`, the shadow grid will need resizing logic.

---

## Action Items

1. [x] Add `MAX_REGIONS = 16` constant and `_region_*` Taichi fields to `taichi_engine.py`.
2. [x] Add `@ti.func _get_node_region(py, px)` helper.
3. [x] Refactor `_death_kernel` — replace `node_type == 1` with region immortality check.
4. [x] Refactor `_spawn_kernel` — replace `node_type != 1` with region spawn check.
5. [x] Add `TaichiNeuralEngine.register_region()` and `get_region_info()` methods.
6. [x] Call `engine.register_region()` for sensory, dynamic, and workspace regions in `create_hybrid_neural_system()`.
7. [x] Remove `HybridNeuralSystemAdapter.pulse_energy()`.
8. [x] Fix `NODE_ENERGY_CAP` and `NODE_DEATH_THRESHOLD` in `config.py`.
9. [x] Fix `node_death_threshold` in `pyg_config.json`.
10. [x] Add deprecation warning to `mapping.py`; remove dead call from `WorkspaceNodeSystem._create_sensory_mapping()`.
