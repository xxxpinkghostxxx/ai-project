# File Header Template Refactoring — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply a consistent three-section `#` comment header to all 25 Python source files in `src/project/`, remove all inline `#` comments, and preserve all `"""docstrings"""`.

**Architecture:** Each file gets a header block (CODE STRUCTURE / TODOS / KNOWN BUGS) followed by a rule line, then a trimmed module docstring, then imports and code body with no `#` comments. Tool pragmas (`# type: ignore`, `# noqa`, `# pylint:`, `# fmt:`, `# pragma:`) are retained.

**Tech Stack:** Python only. No new dependencies. Validation via import checks.

**Spec:** `docs/superpowers/specs/2026-03-15-file-header-template-design.md`

---

## Standard Process Per File

Every file follows the same procedure. Task 1 demonstrates it in full detail; subsequent tasks follow the same steps but are written concisely.

**For each file:**
1. Read the entire file
2. Catalog: all module-level functions, classes, methods, properties (with signatures)
3. Collect: any TODO/FIXME/NOTE/BUG comments and their text
4. Write the three-section `#` header at the top (see spec for template)
5. Add the rule line: `# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.`
6. Trim the module docstring to 1-2 lines
7. Remove all `#` comments in the code body — but **retain** all `# type: ignore`, `# noqa`, `# pylint:`, `# fmt:`, `# pragma:` directives
8. If a shebang (`#!/usr/bin/env python3`) exists, keep it as line 1 before the header
9. Validate the file still imports: `python -c "from project.<module> import *"` (adjust path per file)
10. Commit

**Special case — `config.py`:** Retains `# ===` section separators in the code body for readability. Section 1 lists constant *groups* rather than individual constants.

---

## Chunk 1: Phase 1 — taichi_engine.py (exemplar)

### Task 1: Apply header template to taichi_engine.py

**Files:**
- Modify: `src/project/system/taichi_engine.py`

This is the exemplar task — fully detailed to establish the pattern.

- [ ] **Step 1: Write the three-section header at the top of the file**

Replace the current content before the imports with the following header block. The header goes before the module docstring.

```python
# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Architecture:
#   GPU-accelerated neural cellular automaton. Taichi handles explicit GPU kernels;
#   PyTorch provides the shared CUDA energy_field tensor (zero-copy).
#   @ti.data_oriented class kernels do NOT support ti.types.ndarray() in Taichi 1.7.x,
#   so all kernels accepting PyTorch tensors are module-level functions.
#   Only one engine instance per process.
#
#   Bit layout (64-bit int64 per node):
#     bit 63: alive flag
#     bits 62-61: node_type (0=sensory, 1=dynamic, 2=workspace)
#     bits 60-58: conn_type (0-7)
#     bits 57-3: reserved (DNA in separate _node_dna field)
#     bits 2-0: modality (0=neutral, 1=visual, 2=audio_L, 3=audio_R)
#
#   3D DNA layout (_node_dna field, 3 int64 words per node):
#     26 neighbor slots x 5 bits = 130 bits packed into 3 int64 words.
#     Each 5-bit slot: [MODE:1][PARAM:4].
#
# Module-level Constants:
#   DAMPING_SCALE = 50.0
#   SPECIAL_PARAM_MAX = 3.0
#   THRESHOLD_ENERGY_SCALE = 16.0
#   CONTRIB_CLAMP_FRACTION = 0.08
#   SPAWN_ABOVE_AVG_FACTOR = 1.10
#   SPAWN_COST_FRACTION = 0.75
#   SPAWN_CAP_FRACTION = 0.80
#
# Module-level Functions:
#   project_energy_field_to_2d(energy_field: torch.Tensor) -> torch.Tensor
#     Max-pool 3D energy_field [H,W,D] to 2D [H,W]; 2D inputs pass through
#
#   init_taichi(device: str, device_memory_fraction: float) -> None
#     Lazy Taichi runtime init; sets arch, memory fraction, random seed
#
#   _clear_grid_map(grid_node_id: ti.types.ndarray())    @ti.kernel
#     Reset grid-to-node map to sentinel (-1)
#
#   _build_grid_map(grid_node_id: ti.types.ndarray())    @ti.kernel
#     Populate grid-to-node map from alive node positions
#
#   _dna_transfer_kernel(energy_field: ti.types.ndarray(),    @ti.kernel
#       grid_node_id: ti.types.ndarray(), H, W, D, dt, gate_threshold,
#       frame, strength, contrib_clamp)
#     Core energy transfer: for each alive node, read DNA micro-instructions per
#     neighbor, compute contributions by conn_type (excitatory/inhibitory/gated/
#     plastic/anti-plastic/damped/resonant/capacitive), write to energy field.
#     Taichi 1.7 disallows `continue` in non-static if — all logic uses nested ifs.
#
#   _get_node_region(py: int, px: int) -> int    @ti.func
#     Return region ID for a grid position, or -1 if unregistered
#
#   _death_kernel(death_threshold: float)    @ti.kernel
#     Kill nodes with energy below death_threshold (all nodes are mortal)
#
#   _spawn_kernel(H, W, D, spawn_threshold, spawn_cost,    @ti.kernel
#       child_energy, max_spawns, n_existing)
#     Spawn children from high-energy nodes: region-aware gate, atomic slot claim,
#     random offset from parent (toroidal wrap), hereditary DNA with 10% mutation
#
#   _clamp_kernel(energy_field: ti.types.ndarray(), lo: float, hi: float)    @ti.kernel
#     Clamp energy field values to [lo, hi]
#
#   _inject_sensory_kernel(energy_field: ti.types.ndarray(),    @ti.kernel
#       data: ti.types.ndarray(), y0, x0, h, w, z)
#     Inject sensory input into energy field at region position
#
#   _inject_sensory_delta_kernel(energy_field: ti.types.ndarray(),    @ti.kernel
#       grid_node_id: ti.types.ndarray(), data: ti.types.ndarray(),
#       y0, x0, h, w, z)
#     Delta-mode sensory injection: add frame difference to energy field
#
#   _sync_energy_from_field(energy_field: ti.types.ndarray(),    @ti.kernel
#       grid_node_id: ti.types.ndarray())
#     Sync per-node energy scalars from the 3D energy field
#
#   _write_nodes_kernel(states, energies, pos_y, pos_x, pos_z: ti.types.ndarray(),    @ti.kernel
#       start: int, n: int)
#     Bulk-write node state/position/energy into Taichi fields
#
#   _write_dna_kernel(dna_data: ti.types.ndarray(), start: int, n: int)    @ti.kernel
#     Bulk-write DNA words into _node_dna field
#
#   _reduce_dynamic_energy()    @ti.kernel
#     Sum energy of alive dynamic-region nodes into _dyn_energy_sum/_dyn_node_count
#
#   _pack_state_batch(alive: torch.Tensor, node_type: torch.Tensor,
#       conn_type: torch.Tensor, modality: torch.Tensor) -> torch.Tensor
#     Pack node state int64 from components (pure PyTorch)
#
#   _random_conn_type(n: int, device: torch.device) -> torch.Tensor
#     Generate n random connection types [0, NUM_CONN_TYPES)
#
#   _random_dna(n: int, device: torch.device) -> torch.Tensor
#     Generate n random 8-neighbor DNA values (legacy 2D)
#
#   _random_dna_3d(n: int, device: torch.device) -> torch.Tensor
#     Generate n random 26-neighbor DNA packed into 3 int64 words per node
#
#   is_alive(state: torch.Tensor) -> torch.Tensor
#     Check alive bit of packed node state (tensor-level operation)
#
#   extract_node_type(state: torch.Tensor) -> torch.Tensor
#     Extract 2-bit node_type from packed node state (tensor-level operation)
#
# Classes:
#   TaichiNeuralEngine:
#     __init__(grid_size: Tuple, node_spawn_threshold, node_death_threshold,
#         node_energy_cap, spawn_cost, gate_threshold, transfer_dt,
#         child_energy_fraction, transfer_strength, device) -> None
#       Allocate Taichi fields, PyTorch energy field, init constant lookup tables
#
#     __del__(self) -> None
#       Destructor logging
#
#     node_count -> int    @property
#       Current alive node count (Python-side)
#
#     register_region(y0, y1, x0, x1, region_type: int,
#         spawn: bool, immortal: bool) -> int
#       Register a region in the GPU region table (ADR-001);
#       returns region_id [0..MAX_REGIONS-1]
#
#     get_region_info() -> List[Dict[str, Any]]
#       Return metadata for all registered regions
#
#     clear_regions() -> None
#       Reset all region registry entries
#
#     update_parameters(**kwargs) -> None
#       Update runtime parameters (energy_cap, dt, death/spawn thresholds, etc.)
#
#     step(**kwargs) -> Dict[str, Any]
#       One simulation tick: rebuild grid map -> DNA transfer -> sync energies
#       -> death + spawn -> clamp -> return metrics dict
#
#     _spawn_limit() -> int
#       Tiered spawn rate limit based on current population density
#
#     add_nodes_batch(positions: List[Tuple], energies: List[float],
#         node_types: List[int], modalities, dna) -> List[int]
#       Bulk-add nodes via GPU kernel; returns list of assigned indices
#
#     add_node(position: Tuple, energy: float, node_type: int) -> int
#       Add single node (delegates to add_nodes_batch)
#
#     inject_sensory_data(pixel_data: torch.Tensor,
#         region: Tuple[int,int,int,int], z: int) -> None
#       Inject visual sensory input into energy field at region position
#
#     add_energy_at(position, amount: float) -> None
#       Add energy at a specific grid position
#
#     inject_audio_data(spectrum_2d: torch.Tensor,
#         region: Tuple[int,int,int,int]) -> None
#       Inject audio spectrum into a region of the energy field
#
#     read_audio_workspace_energies(region: Tuple[int,int,int,int]) -> torch.Tensor
#       Read energy values from an audio workspace region
#
#     get_node_data() -> Dict[str, Any]
#       Read node state/position/energy for all alive nodes
#
#     get_energy_field() -> torch.Tensor
#       Return reference to the shared energy field tensor
#
#     read_workspace_energies(region: Tuple[int,int,int,int]) -> torch.Tensor
#       Read energy values from workspace region; max over Z for 3D
#
#     _build_workspace_cache(y0, y1, x0, x1) -> None
#       Internal cache builder for workspace energy reads
#
#     render_connection_heatmap() -> torch.Tensor
#       Render RGBA heatmap of connection types across the grid
#
#     get_metrics() -> Dict[str, Any]
#       Return simulation metrics (node count, energy stats, spawn/death counts)
#
#     node_state -> torch.Tensor    @property
#       Backward-compatible access to _node_state field
#
#     node_energy -> torch.Tensor    @property
#       Backward-compatible access to _node_energy field
#
#     node_positions_y -> torch.Tensor    @property
#       Backward-compatible access to _node_y field
#
#     node_positions_x -> torch.Tensor    @property
#       Backward-compatible access to _node_x field
#
# =============================================================================
# TODOS
# =============================================================================
#
# None
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.
```

- [ ] **Step 2: Trim the module docstring**

Replace the current 30-line module docstring with:

```python
"""TaichiNeuralEngine — Probabilistic energy-field neural cellular automaton on GPU."""
```

The architecture notes, bit layout, and DNA encoding docs have been moved to the Architecture preamble in Section 1.

- [ ] **Step 3: Remove all inline `#` comments from the code body**

Remove every `#` comment below the rule line. This file has ~60 inline comments including:
- Section separators (`# ===...===` blocks at lines 72, 95-97, 195, 625, 662, etc.)
- Named constant annotations (lines 76-82)
- Step comments in `step()` method (lines 934, 938, 943, 958, 964, etc.)
- Architecture notes (lines 127-128, 134-135, 154-163, 264-265, etc.)
- ADR-001 region registry comments (lines 417-419, 428, 800-803)

**Do NOT remove:**
- Any `"""docstrings"""` (function, method, or class level)
- Any `# type: ignore` or `# pylint:` pragmas (this file has none, but check)

- [ ] **Step 4: Validate the file imports correctly**

Run:
```bash
cd src && python -c "from project.system.taichi_engine import TaichiNeuralEngine, project_energy_field_to_2d, is_alive, extract_node_type; print('OK')"
```

Expected: `OK` (Taichi init may print GPU info, that's fine)

If import fails, fix the issue (likely an accidentally deleted line or broken string).

- [ ] **Step 5: Commit**

```bash
git add src/project/system/taichi_engine.py
git commit -m "refactor: apply file header template to taichi_engine.py

Add three-section header (Code Structure, TODOs, Known Bugs), trim module
docstring, remove all inline # comments per file header template spec."
```

---

## Chunk 2: Phase 2 — Core system files

### Task 2: Apply header template to main.py

**Files:**
- Modify: `src/project/main.py` (1,186 lines)

**Key details for this file:**
- 8 module-level functions: `_place_clusters`, `_build_neutral_dna`, `resolve_device`, `managed_resources` (@contextlib.contextmanager), `create_hybrid_neural_system`, `initialize_system`, `_attempt_system_recovery`, `main`
- 1 class: `HybridNeuralSystemAdapter` with 23 methods (including `cleanup`/`stop` aliases for `shutdown`)
- **Heavy tool pragmas:** 37 `# type: ignore` and 31 `# pylint:` directives (11 lines have both) — ALL must be retained
- No TODO/FIXME/NOTE comments
- Module docstring is 17 lines — trim to 1-2 lines

- [ ] **Step 1: Read file, write header, trim docstring**
- [ ] **Step 2: Remove inline `#` comments (retain all `# type: ignore` and `# pylint:` pragmas)**
- [ ] **Step 3: Validate import**

```bash
cd src && python -c "from project.main import main, HybridNeuralSystemAdapter; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/project/main.py
git commit -m "refactor: apply file header template to main.py"
```

### Task 3: Apply header template to config.py (relaxed variant)

**Files:**
- Modify: `src/project/config.py` (137 lines)

**Key details for this file:**
- **Relaxed variant:** Retains `# ===` section separators in code body
- Constants-focused file with one private helper `_compute_reverse_3d()`
- Section 1 lists constant *groups* (Screen Capture, Node/Connection Type Enums, Binary Node State Encoding, DNA Modality Keys, 3D Neighbor Geometry, etc.) rather than individual constants
- No tool pragmas, no TODO/FIXME

- [ ] **Step 1: Read file, write header (list constant groups), trim docstring**
- [ ] **Step 2: Remove all `#` comments that are NOT `# ===...===` separator lines** (this includes explanatory text lines that follow separators, e.g., the bit layout description lines)
- [ ] **Step 3: Validate import**

```bash
cd src && python -c "from project.config import BINARY_ALIVE_BIT, NEIGHBOR_OFFSETS_3D; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/project/config.py
git commit -m "refactor: apply file header template to config.py (relaxed variant)"
```

### Task 4: Apply header template to state_manager.py

**Files:**
- Modify: `src/project/system/state_manager.py`

- [ ] **Step 1: Read file, write header, trim docstring**
- [ ] **Step 2: Remove inline `#` comments (retain tool pragmas if any)**
- [ ] **Step 3: Validate import**

```bash
cd src && python -c "from project.system.state_manager import StateManager; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/project/system/state_manager.py
git commit -m "refactor: apply file header template to state_manager.py"
```

### Task 5: Apply header template to global_storage.py

**Files:**
- Modify: `src/project/system/global_storage.py`

- [ ] **Step 1: Read file, write header, trim docstring**
- [ ] **Step 2: Remove inline `#` comments (retain tool pragmas if any)**
- [ ] **Step 3: Validate import**

```bash
cd src && python -c "from project.system.global_storage import GlobalStorage; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/project/system/global_storage.py
git commit -m "refactor: apply file header template to global_storage.py"
```

---

## Chunk 3: Phase 3 — UI files

### Task 6: Apply header template to modern_main_window.py

**Files:**
- Modify: `src/project/ui/modern_main_window.py` (1,973 lines — largest file)

**Key details for this file:**
- Tool pragmas: 5 `# type: ignore` and 19 `# pylint:` directives — ALL retained
- Module-level standalone pragma lines (`# pylint: disable=no-name-in-module`, `# pylint: disable=import-error`, `# pylint: disable=broad-exception-caught,too-many-lines`) must be retained
- **Important:** Documentary `#` comment lines adjacent to pragma lines must be REMOVED while the pragma lines themselves are retained. For example, lines like `# PyQt6 classes are dynamically loaded...` (explaining a pragma) are documentary and should be removed.
- ~159 inline comments to remove

- [ ] **Step 1: Read file, write header, trim docstring**
- [ ] **Step 2: Remove inline `#` comments (retain all tool pragmas — this file has many)**
- [ ] **Step 3: Validate import**

```bash
cd src && python -c "from project.ui.modern_main_window import ModernMainWindow; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/project/ui/modern_main_window.py
git commit -m "refactor: apply file header template to modern_main_window.py"
```

### Task 7: Apply header template to modern_config_panel.py

**Files:**
- Modify: `src/project/ui/modern_config_panel.py` (916 lines)

**Key details:**
- Multiple `# type: ignore[reportUnknownMemberType]` pragmas on signal connections — retain all
- 1 `# pylint: disable=broad-exception-caught` — retain

- [ ] **Step 1: Read file, write header, trim docstring**
- [ ] **Step 2: Remove inline `#` comments (retain tool pragmas)**
- [ ] **Step 3: Validate import**

```bash
cd src && python -c "from project.ui.modern_config_panel import ModernConfigPanel; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/project/ui/modern_config_panel.py
git commit -m "refactor: apply file header template to modern_config_panel.py"
```

---

## Chunk 4: Phase 4 — Workspace files

### Task 8: Apply header template to workspace_system.py

**Files:**
- Modify: `src/project/workspace/workspace_system.py`

- [ ] **Step 1: Read file, write header, trim docstring**
- [ ] **Step 2: Remove inline `#` comments (retain tool pragmas if any)**
- [ ] **Step 3: Validate import**

```bash
cd src && python -c "from project.workspace.workspace_system import WorkspaceNodeSystem; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/project/workspace/workspace_system.py
git commit -m "refactor: apply file header template to workspace_system.py"
```

### Task 9: Apply header template to workspace_node.py

**Files:**
- Modify: `src/project/workspace/workspace_node.py`

- [ ] **Step 1-4: Standard process (read, header, trim, remove comments, validate, commit)**

```bash
cd src && python -c "from project.workspace.workspace_node import WorkspaceNode; print('OK')"
git add src/project/workspace/workspace_node.py
git commit -m "refactor: apply file header template to workspace_node.py"
```

### Task 10: Apply header template to renderer.py

**Files:**
- Modify: `src/project/workspace/renderer.py`

- [ ] **Step 1-4: Standard process**

```bash
cd src && python -c "from project.workspace.renderer import *; print('OK')"
git add src/project/workspace/renderer.py
git commit -m "refactor: apply file header template to renderer.py"
```

### Task 11: Apply header template to workspace/visualization.py

**Files:**
- Modify: `src/project/workspace/visualization.py`

- [ ] **Step 1-4: Standard process**

```bash
cd src && python -c "from project.workspace.visualization import *; print('OK')"
git add src/project/workspace/visualization.py
git commit -m "refactor: apply file header template to visualization.py"
```

### Task 12: Apply header template to visualization_integration.py

**Files:**
- Modify: `src/project/workspace/visualization_integration.py`

**Key detail:** Contains `# TODO: Implement actual image export via visualization_window` at line 167 — move to TODOS section as `[hanging]`.

- [ ] **Step 1-4: Standard process (move the TODO to Section 2)**

```bash
cd src && python -c "from project.workspace.visualization_integration import *; print('OK')"
git add src/project/workspace/visualization_integration.py
git commit -m "refactor: apply file header template to visualization_integration.py"
```

### Task 13: Apply header template to workspace/mapping.py

**Files:**
- Modify: `src/project/workspace/mapping.py`

- [ ] **Step 1-4: Standard process**

```bash
cd src && python -c "from project.workspace.mapping import *; print('OK')"
git add src/project/workspace/mapping.py
git commit -m "refactor: apply file header template to mapping.py"
```

### Task 14: Apply header template to pixel_shading.py

**Files:**
- Modify: `src/project/workspace/pixel_shading.py`

- [ ] **Step 1-4: Standard process**

```bash
cd src && python -c "from project.workspace.pixel_shading import *; print('OK')"
git add src/project/workspace/pixel_shading.py
git commit -m "refactor: apply file header template to pixel_shading.py"
```

### Task 15: Apply header template to workspace/config.py

**Files:**
- Modify: `src/project/workspace/config.py`

- [ ] **Step 1-4: Standard process**

```bash
cd src && python -c "from project.workspace.config import EnergyReadingConfig; print('OK')"
git add src/project/workspace/config.py
git commit -m "refactor: apply file header template to workspace/config.py"
```

---

## Chunk 5: Phase 5 — Utility files

### Task 16: Apply header template to config_manager.py

**Files:**
- Modify: `src/project/utils/config_manager.py`

**Key details:**
- Has `# type: ignore[assignment]`, `# type: ignore[reportUnnecessaryIsInstance]`, `# type: ignore[union-attr,return-value]` — retain all

- [ ] **Step 1-4: Standard process (retain tool pragmas)**

```bash
cd src && python -c "from project.utils.config_manager import ConfigManager; print('OK')"
git add src/project/utils/config_manager.py
git commit -m "refactor: apply file header template to config_manager.py"
```

### Task 17: Apply header template to error_handler.py

**Files:**
- Modify: `src/project/utils/error_handler.py`

- [ ] **Step 1-4: Standard process**

```bash
cd src && python -c "from project.utils.error_handler import ErrorHandler; print('OK')"
git add src/project/utils/error_handler.py
git commit -m "refactor: apply file header template to error_handler.py"
```

### Task 18: Apply header template to performance_utils.py

**Files:**
- Modify: `src/project/utils/performance_utils.py`

**Key details:**
- Has a shebang line (`#!/usr/bin/env python3` at line 1) — keep it as line 1 before the header
- Has `# type: ignore[assignment]` — retain

- [ ] **Step 1-4: Standard process (keep shebang as line 1, retain pragmas)**

```bash
cd src && python -c "from project.utils.performance_utils import *; print('OK')"
git add src/project/utils/performance_utils.py
git commit -m "refactor: apply file header template to performance_utils.py"
```

### Task 19: Apply header template to security_utils.py

**Files:**
- Modify: `src/project/utils/security_utils.py`

**Key details:**
- Has a shebang line (`#!/usr/bin/env python3` at line 1) — keep it as line 1 before the header

- [ ] **Step 1-4: Standard process (keep shebang as line 1)**

```bash
cd src && python -c "from project.utils.security_utils import *; print('OK')"
git add src/project/utils/security_utils.py
git commit -m "refactor: apply file header template to security_utils.py"
```

### Task 20: Apply header template to shutdown_utils.py

**Files:**
- Modify: `src/project/utils/shutdown_utils.py`

**Key details:**
- Has `# type: ignore[comparison-overlap]` — retain

- [ ] **Step 1-4: Standard process (retain pragmas)**

```bash
cd src && python -c "from project.utils.shutdown_utils import *; print('OK')"
git add src/project/utils/shutdown_utils.py
git commit -m "refactor: apply file header template to shutdown_utils.py"
```

---

## Chunk 6: Phase 6 — I/O and visualization files

### Task 21: Apply header template to audio_capture.py

**Files:**
- Modify: `src/project/audio_capture.py`

**Key details:**
- Has `# type: ignore[assignment]` and `# type: ignore[return-value]` — retain

- [ ] **Step 1-4: Standard process (retain pragmas)**

```bash
cd src && python -c "from project.audio_capture import AudioCapture; print('OK')"
git add src/project/audio_capture.py
git commit -m "refactor: apply file header template to audio_capture.py"
```

### Task 22: Apply header template to audio_output.py

**Files:**
- Modify: `src/project/audio_output.py`

**Key details:**
- Has `# type: ignore[assignment]` — retain

- [ ] **Step 1-4: Standard process (retain pragmas)**

```bash
cd src && python -c "from project.audio_output import AudioOutput; print('OK')"
git add src/project/audio_output.py
git commit -m "refactor: apply file header template to audio_output.py"
```

### Task 23: Apply header template to optimized_capture.py

**Files:**
- Modify: `src/project/optimized_capture.py`

**Key details:**
- Has `# type: ignore[import-not-found]` — retain

- [ ] **Step 1-4: Standard process (retain pragmas)**

```bash
cd src && python -c "from project.optimized_capture import *; print('OK')"
git add src/project/optimized_capture.py
git commit -m "refactor: apply file header template to optimized_capture.py"
```

### Task 24: Apply header template to vision.py

**Files:**
- Modify: `src/project/vision.py`

**Key details:**
- Multiple `# type: ignore` pragmas on cv2 calls and ImageGrab — retain all

- [ ] **Step 1-4: Standard process (retain pragmas)**

```bash
cd src && python -c "from project.vision import ThreadedScreenCapture; print('OK')"
git add src/project/vision.py
git commit -m "refactor: apply file header template to vision.py"
```

### Task 25: Apply header template to taichi_gui_manager.py

**Files:**
- Modify: `src/project/visualization/taichi_gui_manager.py`

- [ ] **Step 1-4: Standard process**

```bash
cd src && python -c "from project.visualization.taichi_gui_manager import *; print('OK')"
git add src/project/visualization/taichi_gui_manager.py
git commit -m "refactor: apply file header template to taichi_gui_manager.py"
```

---

## Final Validation

After all 25 files are done:

- [ ] **Step 1: Run full import check across all modules**

```bash
cd src && python -c "
import project.config
import project.main
import project.vision
import project.audio_capture
import project.audio_output
import project.optimized_capture
import project.system.taichi_engine
import project.system.state_manager
import project.system.global_storage
import project.ui.modern_main_window
import project.ui.modern_config_panel
import project.workspace.workspace_system
import project.workspace.workspace_node
import project.workspace.renderer
import project.workspace.visualization
import project.workspace.visualization_integration
import project.workspace.mapping
import project.workspace.pixel_shading
import project.workspace.config
import project.utils.config_manager
import project.utils.error_handler
import project.utils.performance_utils
import project.utils.security_utils
import project.utils.shutdown_utils
import project.visualization.taichi_gui_manager
print('All 25 modules import OK')
"
```

- [ ] **Step 2: Verify no stray documentary `#` comments remain (automated check)**

```bash
cd src && grep -rn "^\s*#" project/ --include="*.py" | grep -v "__init__" | grep -v "# ===" | grep -v "# type: ignore" | grep -v "# pylint:" | grep -v "# noqa" | grep -v "# fmt:" | grep -v "# pragma:" | grep -v "# DO NOT ADD PROJECT NOTES" | grep -v "# CODE STRUCTURE\|# TODOS\|# KNOWN BUGS\|# None\|# Module-level\|# Classes:\|# Architecture:"
```

Any output indicates a stray documentary comment that was missed. Fix and re-commit.

- [ ] **Step 3: Run existing test suite**

```bash
pytest tests/ -x -q
```
