# Taichi GUI Visualization + DNA Modality Keys + Qt UI Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three Taichi GGUI visualization windows, a DNA modality key system for node channel identity, and a full Qt UI redesign with a 3-column layout.

**Architecture:** Three standalone `ti.ui.Window` (GGUI/Vulkan) windows render slices of `energy_field` on the GPU at 60/30 FPS. A 3-bit modality field is packed into reserved bits 2–0 of the 64-bit node state, assigning VISUAL/AUDIO_LEFT/AUDIO_RIGHT identity to workspace nodes and inheriting it into dynamic children during spawn. The Qt UI becomes a 3-column layout: live preview left, Taichi window controls center, engine config right.

**Tech Stack:** Python 3.11, Taichi 1.7.x (GGUI/ti.ui), PyTorch CUDA, PyQt6, threading

**Design doc:** `docs/plans/2026-03-03-taichi-gui-modality-overhaul-design.md`

---

## Key constraints (read before touching any file)

- **NEVER add `from __future__ import annotations`** to `taichi_engine.py` or `taichi_gui_manager.py` — it breaks `@ti.kernel` annotation resolution in Taichi 1.7.
- **All Taichi fields must be module-level**, not class-level — Taichi 1.7 constraint. Define them at the top of the file outside any class.
- **`energy_field` is a PyTorch CUDA tensor** (`torch.float32`, shape `[2560, 1920]`). Pass it to Taichi kernels as `ti.types.ndarray()` — zero-copy on CUDA, no `.numpy()` calls.
- Modality is a **label only** — do NOT modify the `_dna_transfer_kernel`. The transfer math is unchanged.
- `continue` is **not allowed** inside non-static `if` inside `ti.static` for — use nested `if` guards instead (Taichi 1.7 limitation).
- Only **one TaichiNeuralEngine per process** (fields are global). The GUI manager reads those same global fields.
- Grid: `H=2560, W=1920`. Sensory region: `[0:1080, 0:1920]`. Audio strip: `[y0:2560, 256:1280]`. Workspace: `[2432:2560, 0:128]` (bottom 128 rows, left 128 cols).

---

## Task 1: DNA Modality Constants

**Files:**
- Modify: `src/project/config.py`

### Step 1: Write the failing test

Create `tests/test_modality_constants.py`:

```python
"""Tests for DNA modality constants and bit-packing."""
import pytest


def test_modality_constants_exist():
    from project.config import (
        MODALITY_NEUTRAL, MODALITY_VISUAL,
        MODALITY_AUDIO_LEFT, MODALITY_AUDIO_RIGHT,
        MODALITY_SHIFT, MODALITY_MASK,
    )
    assert MODALITY_NEUTRAL == 0
    assert MODALITY_VISUAL == 1
    assert MODALITY_AUDIO_LEFT == 2
    assert MODALITY_AUDIO_RIGHT == 3
    assert MODALITY_SHIFT == 0
    assert MODALITY_MASK == 0b111


def test_modality_mask_extracts_bits():
    from project.config import MODALITY_VISUAL, MODALITY_SHIFT, MODALITY_MASK
    # Simulate packing: VISUAL=1 in bits 2-0
    packed_state = MODALITY_VISUAL << MODALITY_SHIFT
    extracted = (packed_state >> MODALITY_SHIFT) & MODALITY_MASK
    assert extracted == MODALITY_VISUAL


def test_modality_does_not_overlap_dna():
    """Bits 2-0 must not overlap DNA range bits 57-18."""
    from project.config import MODALITY_SHIFT, MODALITY_MASK, BINARY_DNA_BASE_SHIFT
    modality_top_bit = MODALITY_SHIFT + MODALITY_MASK.bit_length() - 1
    assert modality_top_bit < BINARY_DNA_BASE_SHIFT  # bits 2-0 are below bit 18
```

### Step 2: Run test to verify it fails

```
cd c:/Users/chris/Documents/ai-project/ai-project
python -m pytest tests/test_modality_constants.py -v
```

Expected: `ImportError: cannot import name 'MODALITY_NEUTRAL' from 'project.config'`

### Step 3: Add constants to config.py

Open `src/project/config.py`. After the `BINARY_CONN_TYPE_MASK` line (~line 50), add:

```python
# =============================================================================
# DNA Modality Keys (bits 2–0 of the reserved range bits 17–0)
# These tag sensory/workspace nodes with a channel identity and are inherited
# by dynamic children during spawn. The transfer kernel is NOT affected.
# =============================================================================
MODALITY_NEUTRAL     = 0   # dynamic nodes / unassigned
MODALITY_VISUAL      = 1   # desktop sensory input / visual workspace output
MODALITY_AUDIO_LEFT  = 2   # left audio channel sensory / workspace
MODALITY_AUDIO_RIGHT = 3   # right audio channel sensory / workspace
MODALITY_SHIFT       = 0   # bit position within the 64-bit node state
MODALITY_MASK        = 0b111  # 3 bits → supports 8 modalities
```

Also update the docstring comment at line 39 (`Layout: ...`) to:
```python
# Layout: [ALIVE:1][NODE_TYPE:2][CONN_TYPE:3][DNA[0..7]:8×5=40][RSVD:15][MODALITY:3]
```

### Step 4: Run tests to verify they pass

```
python -m pytest tests/test_modality_constants.py -v
```

Expected: `3 passed`

### Step 5: Commit

```bash
git add src/project/config.py tests/test_modality_constants.py
git commit -m "feat: add DNA modality constants (bits 2-0, VISUAL/AUDIO_L/AUDIO_R)"
```

---

## Task 2: Engine Modality Packing + Spawn Inheritance

**Files:**
- Modify: `src/project/system/taichi_engine.py` (3 changes)
- Modify: `src/project/main.py` (pass modalities in initialization)
- Test: `tests/test_modality_engine.py`

### Step 1: Write the failing tests

Create `tests/test_modality_engine.py`:

```python
"""Tests for modality bit packing and spawn inheritance."""
import pytest
import torch


def test_pack_state_includes_modality():
    """add_nodes_batch must embed modality bits into the packed state."""
    # Skip if no CUDA — engine requires it
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import taichi as ti
    # Guard: don't re-init Taichi if already done in another test
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass

    from project.system.taichi_engine import TaichiNeuralEngine, _node_state, MAX_NODES
    from project.config import MODALITY_VISUAL, MODALITY_AUDIO_LEFT, MODALITY_SHIFT, MODALITY_MASK

    engine = TaichiNeuralEngine(grid_size=(64, 64))
    try:
        # Add 2 workspace nodes: one VISUAL, one AUDIO_LEFT
        ids = engine.add_nodes_batch(
            positions=[(60, 10), (60, 20)],
            energies=[10.0, 10.0],
            node_types=[2, 2],
            modalities=[MODALITY_VISUAL, MODALITY_AUDIO_LEFT],
        )
        # Read back packed states
        state0 = int(_node_state[ids[0]])
        state1 = int(_node_state[ids[1]])
        mod0 = (state0 >> MODALITY_SHIFT) & MODALITY_MASK
        mod1 = (state1 >> MODALITY_SHIFT) & MODALITY_MASK
        assert mod0 == MODALITY_VISUAL
        assert mod1 == MODALITY_AUDIO_LEFT
    finally:
        del engine


def test_modality_neutral_when_omitted():
    """Nodes added without modality arg should have MODALITY_NEUTRAL (0)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass

    from project.system.taichi_engine import TaichiNeuralEngine, _node_state
    from project.config import MODALITY_NEUTRAL, MODALITY_SHIFT, MODALITY_MASK

    engine = TaichiNeuralEngine(grid_size=(64, 64))
    try:
        ids = engine.add_nodes_batch(
            positions=[(30, 30)],
            energies=[50.0],
            node_types=[1],
        )
        state = int(_node_state[ids[0]])
        mod = (state >> MODALITY_SHIFT) & MODALITY_MASK
        assert mod == MODALITY_NEUTRAL
    finally:
        del engine
```

### Step 2: Run tests to verify they fail

```
python -m pytest tests/test_modality_engine.py -v
```

Expected: `TypeError: add_nodes_batch() got an unexpected keyword argument 'modalities'`

### Step 3: Update `_pack_state_batch()` in `taichi_engine.py`

Find `_pack_state_batch` (~line 445). Change its signature and body to accept modality:

**Old:**
```python
def _pack_state_batch(alive: torch.Tensor, node_type: torch.Tensor,
                      conn_type: torch.Tensor, dna_q: torch.Tensor) -> torch.Tensor:
    """Pack N nodes into int64 binary states (vectorized, no Python loop)."""
    state = (alive.to(torch.int64) << BINARY_ALIVE_BIT)
    state |= (node_type.to(torch.int64) << BINARY_NODE_TYPE_SHIFT)
    state |= (conn_type.to(torch.int64) << BINARY_CONN_TYPE_SHIFT)
    shifts = torch.tensor(_DNA_SHIFTS, device=dna_q.device, dtype=torch.int64)
    state |= (dna_q.to(torch.int64) << shifts.unsqueeze(0)).sum(dim=1)
    return state
```

**New:**
```python
def _pack_state_batch(alive: torch.Tensor, node_type: torch.Tensor,
                      conn_type: torch.Tensor, dna_q: torch.Tensor,
                      modality: torch.Tensor) -> torch.Tensor:
    """Pack N nodes into int64 binary states (vectorized, no Python loop)."""
    state = (alive.to(torch.int64) << BINARY_ALIVE_BIT)
    state |= (node_type.to(torch.int64) << BINARY_NODE_TYPE_SHIFT)
    state |= (conn_type.to(torch.int64) << BINARY_CONN_TYPE_SHIFT)
    shifts = torch.tensor(_DNA_SHIFTS, device=dna_q.device, dtype=torch.int64)
    state |= (dna_q.to(torch.int64) << shifts.unsqueeze(0)).sum(dim=1)
    state |= modality.to(torch.int64)   # bits 2-0 = MODALITY (MODALITY_SHIFT == 0)
    return state
```

### Step 4: Update `add_nodes_batch()` in `taichi_engine.py`

Find `add_nodes_batch` (~line 785). Update the signature and the `_pack_state_batch` call:

**Change the method signature** (add `modalities` parameter after `node_types`):
```python
def add_nodes_batch(
    self,
    positions:  List[Tuple[int, int]],
    energies:   List[float],
    node_types: List[int],
    modalities: Optional[List[int]] = None,
) -> List[int]:
```

**After the line `new_dna = _random_dna(n, dev)`, add:**
```python
        if modalities is not None:
            new_modality = torch.tensor(modalities, device=dev, dtype=torch.int64)
        else:
            new_modality = torch.zeros(n, device=dev, dtype=torch.int64)
```

**Change the `_pack_state_batch` call** (the existing call ~line 818):
```python
        new_state = _pack_state_batch(alive_bits, new_types, new_conn, new_dna, new_modality)
```

Also add the import at the top of the `taichi_engine.py` imports section (the `from project.config import (` block):
```python
    MODALITY_SHIFT,
    MODALITY_MASK,
```

### Step 5: Update `_spawn_kernel` to inherit parent modality

Find `_spawn_kernel` (~line 287). Inside the kernel, after the line computing `conn_type`, find the block that packs `new_state` (~line 351):

```python
        # Pack child state: alive=1, type=1 (dynamic), inherited DNA
        new_state = (
            (ti.i64(1) << 63)           # alive bit
          | (ti.i64(1) << 61)           # node type = dynamic (1)
          | (ti.i64(conn_type) << 58)   # connection type (3 bits)
          | dna_packed
        )
```

Replace with:
```python
        # Pack child state: alive=1, type=1 (dynamic), inherited DNA + parent modality
        parent_modality = state & ti.i64(7)   # bits 2-0 of parent state
        new_state = (
            (ti.i64(1) << 63)           # alive bit
          | (ti.i64(1) << 61)           # node type = dynamic (1)
          | (ti.i64(conn_type) << 58)   # connection type (3 bits)
          | dna_packed
          | parent_modality             # inherit parent modality (bits 2-0)
        )
```

### Step 6: Update module docstring bit layout comment

At the top of `taichi_engine.py` (~line 22), change:
```
    bits 17-0   : reserved
```
to:
```
    bits 17-3   : reserved
    bits 2-0    : modality  (0=neutral, 1=visual, 2=audio_L, 3=audio_R)
```

### Step 7: Update `main.py` to pass modalities during initialization

In `main.py`, find the workspace initialization block (~line 668). After building `workspace_positions`, add modality computation:

```python
    # Workspace modality: column-split into thirds (AUDIO_LEFT | VISUAL | AUDIO_RIGHT)
    from project.config import MODALITY_VISUAL, MODALITY_AUDIO_LEFT, MODALITY_AUDIO_RIGHT
    ws_col_third = workspace_width // 3  # ~42 for 128-wide workspace

    def _ws_modality(x: int) -> int:
        if x < ws_col_third:
            return MODALITY_AUDIO_LEFT
        elif x < ws_col_third * 2:
            return MODALITY_VISUAL
        else:
            return MODALITY_AUDIO_RIGHT

    workspace_modalities = [
        _ws_modality(x)
        for y in range(workspace_height)
        for x in range(workspace_width)
    ]
```

Also add modality to audio workspace nodes. After computing `audio_ws_positions`, add:

```python
    audio_ws_modalities: list[int] = []
    if audio_enabled:
        for region, mod in [
            (AUDIO_WORKSPACE_L_REGION, MODALITY_AUDIO_LEFT),
            (AUDIO_WORKSPACE_R_REGION, MODALITY_AUDIO_RIGHT),
        ]:
            ry0, ry1, rx0, rx1 = region
            for y in range(ry0, ry1):
                for x in range(rx0, rx1):
                    audio_ws_modalities.append(mod)
```

Update the `add_nodes_batch` call (~line 701):

```python
    all_modalities = (
        [0] * initial_dynamic           # dynamic: MODALITY_NEUTRAL
        + workspace_modalities
        + audio_ws_modalities
    )
    engine.add_nodes_batch(all_positions, all_energies, all_types, all_modalities)
```

### Step 8: Run tests to verify they pass

```
python -m pytest tests/test_modality_engine.py tests/test_modality_constants.py -v
```

Expected: `5 passed`

### Step 9: Commit

```bash
git add src/project/system/taichi_engine.py src/project/main.py tests/test_modality_engine.py
git commit -m "feat: add DNA modality bit packing and spawn inheritance (bits 2-0)"
```

---

## Task 3: Taichi GGUI Visualization Module

**Files:**
- Create: `src/project/visualization/__init__.py`
- Create: `src/project/visualization/taichi_gui_manager.py`
- Modify: `src/project/pyg_config.json`

### Step 1: Add visualization section to pyg_config.json

Open `src/project/pyg_config.json`. Add the following after the `"ui"` section (before the closing `}`):

```json
    "visualization": {
        "workspace_window_enabled": false,
        "full_ai_window_enabled": false,
        "sensory_window_enabled": false,
        "workspace_fps": 60,
        "full_ai_fps": 30,
        "sensory_fps": 30
    }
```

### Step 2: Create the visualization package

Create `src/project/visualization/__init__.py` (empty file — just a package marker):

```python
"""Taichi GGUI visualization windows for the neural simulation."""
```

### Step 3: Write the TaichiGUIManager module

Create `src/project/visualization/taichi_gui_manager.py`:

> **CRITICAL:** Do NOT add `from __future__ import annotations` to this file. Taichi kernel annotations must resolve at definition time.

```python
"""
TaichiGUIManager — Three standalone GGUI visualization windows.

Renders slices of the shared energy_field PyTorch CUDA tensor
into GPU-resident display fields using Taichi GGUI (ti.ui.Window).

Three windows:
  - workspace:  512×512  B/W — bottom workspace strip (128×128 upsampled)
  - full_ai:    320×240  heat-colored — full 2560×1920 at 1:8 downsample
  - sensory:    480×270  B/W — top sensory region 1920×1080 downsampled

Each window runs in its own Python thread. Windows can be launched and
closed independently while the Qt UI stays running.

Module-level Taichi fields (required by Taichi 1.7 — no class-level fields).
Only the fields defined here are allocated; they are written by the display
kernels and read by ti.ui canvas.set_image().

IMPORTANT: Do not call ti.init() here. The engine already called it.
"""

import logging
import threading
import time
from typing import Optional

import taichi as ti
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display field sizes
# ---------------------------------------------------------------------------
_WORKSPACE_W  = 512
_WORKSPACE_H  = 512
_FULL_AI_W    = 320
_FULL_AI_H    = 240
_SENSORY_W    = 480
_SENSORY_H    = 270

# ---------------------------------------------------------------------------
# Module-level Taichi display fields
# ti.ui (GGUI) canvas.set_image() requires shape (H, W, 3) float32 fields.
# ---------------------------------------------------------------------------
_display_workspace = ti.field(dtype=ti.f32, shape=(_WORKSPACE_H, _WORKSPACE_W, 3))
_display_full_ai   = ti.field(dtype=ti.f32, shape=(_FULL_AI_H, _FULL_AI_W, 3))
_display_sensory   = ti.field(dtype=ti.f32, shape=(_SENSORY_H, _SENSORY_W, 3))

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@ti.kernel
def _fill_workspace_display(
    energy_field: ti.types.ndarray(),   # [2560, 1920] float32 CUDA tensor
    ws_y0: int,                          # workspace strip y-start row
    ws_x0: int,                          # workspace strip x-start col (0)
    ws_h:  int,                          # workspace strip height (128)
    ws_w:  int,                          # workspace strip width  (128)
    e_lo:  float,                        # energy min for normalization
    e_hi:  float,                        # energy max for normalization
):
    """
    Sample the workspace strip from energy_field, normalize to [0,1],
    upsample to 512×512, and write as B/W RGB into _display_workspace.
    """
    for py, px in ti.ndrange(_WORKSPACE_H, _WORKSPACE_W):
        # Map display pixel → source cell (nearest-neighbour)
        src_y = ws_y0 + int(float(py) / _WORKSPACE_H * ws_h)
        src_x = ws_x0 + int(float(px) / _WORKSPACE_W * ws_w)
        e = energy_field[src_y, src_x]
        v = (e - e_lo) / (e_hi - e_lo + 1e-6)
        v = ti.min(ti.max(v, 0.0), 1.0)
        _display_workspace[py, px, 0] = v
        _display_workspace[py, px, 1] = v
        _display_workspace[py, px, 2] = v


@ti.kernel
def _fill_full_ai_display(
    energy_field: ti.types.ndarray(),   # [H, W] float32
    H: int,
    W: int,
    e_lo: float,
    e_hi: float,
):
    """
    Downsample the full energy_field to 320×240.
    Map normalized energy to a heat colormap: blue(0) → white(0.5) → red(1).
    """
    for py, px in ti.ndrange(_FULL_AI_H, _FULL_AI_W):
        src_y = int(float(py) / _FULL_AI_H * H)
        src_x = int(float(px) / _FULL_AI_W * W)
        e = energy_field[src_y, src_x]
        v = (e - e_lo) / (e_hi - e_lo + 1e-6)
        v = ti.min(ti.max(v, 0.0), 1.0)
        # Heat colormap: v<0.5 → blue→white; v>0.5 → white→red
        if v < 0.5:
            t = v * 2.0
            _display_full_ai[py, px, 0] = t        # R: 0→1
            _display_full_ai[py, px, 1] = t        # G: 0→1
            _display_full_ai[py, px, 2] = 1.0      # B: constant
        else:
            t = (v - 0.5) * 2.0
            _display_full_ai[py, px, 0] = 1.0      # R: constant
            _display_full_ai[py, px, 1] = 1.0 - t  # G: 1→0
            _display_full_ai[py, px, 2] = 1.0 - t  # B: 1→0


@ti.kernel
def _fill_sensory_display(
    energy_field: ti.types.ndarray(),   # [H, W] float32
    sen_y0: int,    # sensory region y-start (0)
    sen_x0: int,    # sensory region x-start (0)
    sen_h:  int,    # sensory region height (1080)
    sen_w:  int,    # sensory region width  (1920)
    e_lo:   float,
    e_hi:   float,
):
    """Downsample the sensory region to 480×270, B/W."""
    for py, px in ti.ndrange(_SENSORY_H, _SENSORY_W):
        src_y = sen_y0 + int(float(py) / _SENSORY_H * sen_h)
        src_x = sen_x0 + int(float(px) / _SENSORY_W * sen_w)
        e = energy_field[src_y, src_x]
        v = (e - e_lo) / (e_hi - e_lo + 1e-6)
        v = ti.min(ti.max(v, 0.0), 1.0)
        _display_sensory[py, px, 0] = v
        _display_sensory[py, px, 1] = v
        _display_sensory[py, px, 2] = v


# ---------------------------------------------------------------------------
# TaichiGUIManager class
# ---------------------------------------------------------------------------

class TaichiGUIManager:
    """
    Manages three GGUI visualization windows alongside the Qt UI.

    Each window runs in its own daemon thread. The engine's energy_field
    tensor is passed to each kernel frame — zero-copy on CUDA.

    Usage:
        mgr = TaichiGUIManager(engine)
        mgr.open_workspace_window()
        mgr.open_full_ai_window()
        mgr.open_sensory_window()
        # ...
        mgr.close_all()
    """

    def __init__(self, engine: "TaichiNeuralEngine"):
        """
        Args:
            engine: The running TaichiNeuralEngine instance. Its energy_field
                    tensor is used directly (no copy).
        """
        self._engine = engine
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._fps: dict[str, float] = {
            "workspace": 0.0, "full_ai": 0.0, "sensory": 0.0
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_workspace_window(self) -> None:
        """Launch the workspace B/W grid GGUI window (512×512, 60 FPS target)."""
        self._launch("workspace", self._workspace_loop, target_fps=60)

    def open_full_ai_window(self) -> None:
        """Launch the full AI structure heat-map window (320×240, 30 FPS target)."""
        self._launch("full_ai", self._full_ai_loop, target_fps=30)

    def open_sensory_window(self) -> None:
        """Launch the sensory input B/W window (480×270, 30 FPS target)."""
        self._launch("sensory", self._sensory_loop, target_fps=30)

    def close_workspace_window(self) -> None:
        self._stop("workspace")

    def close_full_ai_window(self) -> None:
        self._stop("full_ai")

    def close_sensory_window(self) -> None:
        self._stop("sensory")

    def close_all(self) -> None:
        for key in list(self._stop_flags.keys()):
            self._stop(key)

    def is_open(self, name: str) -> bool:
        """Return True if the named window thread is alive."""
        t = self._threads.get(name)
        return t is not None and t.is_alive()

    def get_fps(self, name: str) -> float:
        """Return the measured FPS of the named window (or 0 if closed)."""
        return self._fps.get(name, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch(self, name: str, loop_fn, target_fps: int) -> None:
        with self._lock:
            if self.is_open(name):
                logger.warning("Window '%s' is already open", name)
                return
            stop_event = threading.Event()
            self._stop_flags[name] = stop_event
            t = threading.Thread(
                target=loop_fn,
                args=(stop_event, target_fps),
                name=f"ggui-{name}",
                daemon=True,
            )
            self._threads[name] = t
            t.start()
            logger.info("Launched GGUI window: %s", name)

    def _stop(self, name: str) -> None:
        evt = self._stop_flags.pop(name, None)
        if evt:
            evt.set()
        t = self._threads.pop(name, None)
        if t:
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning("GGUI window '%s' did not stop cleanly", name)

    def _workspace_loop(self, stop: threading.Event, target_fps: int) -> None:
        frame_budget = 1.0 / target_fps
        try:
            window = ti.ui.Window(
                "Workspace Grid",
                res=(_WORKSPACE_W, _WORKSPACE_H),
                show_window=True,
            )
            canvas = window.get_canvas()
            engine = self._engine
            # Workspace strip: bottom 128 rows, left 128 cols
            ws_h   = 128
            ws_w   = 128
            ws_y0  = engine.H - ws_h
            ws_x0  = 0
            while not stop.is_set() and window.running:
                t0 = time.perf_counter()
                e_lo = float(engine.energy_field[ws_y0:ws_y0+ws_h, ws_x0:ws_x0+ws_w].min())
                e_hi = float(engine.energy_field[ws_y0:ws_y0+ws_h, ws_x0:ws_x0+ws_w].max()) + 1e-6
                _fill_workspace_display(engine.energy_field, ws_y0, ws_x0, ws_h, ws_w, e_lo, e_hi)
                canvas.set_image(_display_workspace)
                window.show()
                elapsed = time.perf_counter() - t0
                sleep_t = frame_budget - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                self._fps["workspace"] = 1.0 / max(time.perf_counter() - t0, 1e-6)
        except Exception:
            logger.exception("GGUI workspace window error")

    def _full_ai_loop(self, stop: threading.Event, target_fps: int) -> None:
        frame_budget = 1.0 / target_fps
        try:
            window = ti.ui.Window(
                "Full AI Structure",
                res=(_FULL_AI_W, _FULL_AI_H),
                show_window=True,
            )
            canvas = window.get_canvas()
            engine = self._engine
            while not stop.is_set() and window.running:
                t0 = time.perf_counter()
                e_lo = float(engine.energy_field.min())
                e_hi = float(engine.energy_field.max()) + 1e-6
                _fill_full_ai_display(engine.energy_field, engine.H, engine.W, e_lo, e_hi)
                canvas.set_image(_display_full_ai)
                window.show()
                elapsed = time.perf_counter() - t0
                sleep_t = frame_budget - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                self._fps["full_ai"] = 1.0 / max(time.perf_counter() - t0, 1e-6)
        except Exception:
            logger.exception("GGUI full AI window error")

    def _sensory_loop(self, stop: threading.Event, target_fps: int) -> None:
        frame_budget = 1.0 / target_fps
        try:
            window = ti.ui.Window(
                "Sensory Input",
                res=(_SENSORY_W, _SENSORY_H),
                show_window=True,
            )
            canvas = window.get_canvas()
            engine = self._engine
            sen_h  = min(1080, engine.H)
            sen_w  = min(1920, engine.W)
            while not stop.is_set() and window.running:
                t0 = time.perf_counter()
                e_lo = float(engine.energy_field[:sen_h, :sen_w].min())
                e_hi = float(engine.energy_field[:sen_h, :sen_w].max()) + 1e-6
                _fill_sensory_display(engine.energy_field, 0, 0, sen_h, sen_w, e_lo, e_hi)
                canvas.set_image(_display_sensory)
                window.show()
                elapsed = time.perf_counter() - t0
                sleep_t = frame_budget - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                self._fps["sensory"] = 1.0 / max(time.perf_counter() - t0, 1e-6)
        except Exception:
            logger.exception("GGUI sensory window error")
```

### Step 4: Write a smoke test for TaichiGUIManager

Add to `tests/test_taichi_gui_manager.py`:

```python
"""Smoke tests for TaichiGUIManager (no display required — just lifecycle)."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_manager_instantiates():
    """TaichiGUIManager should be constructable given an engine."""
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine
    from project.visualization.taichi_gui_manager import TaichiGUIManager

    engine = TaichiNeuralEngine(grid_size=(64, 64))
    try:
        mgr = TaichiGUIManager(engine)
        assert not mgr.is_open("workspace")
        assert not mgr.is_open("full_ai")
        assert not mgr.is_open("sensory")
    finally:
        del engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_display_fields_have_correct_shapes():
    """Module-level display fields must have the expected shapes."""
    from project.visualization.taichi_gui_manager import (
        _display_workspace, _display_full_ai, _display_sensory,
        _WORKSPACE_H, _WORKSPACE_W, _FULL_AI_H, _FULL_AI_W, _SENSORY_H, _SENSORY_W,
    )
    assert _display_workspace.shape == (_WORKSPACE_H, _WORKSPACE_W, 3)
    assert _display_full_ai.shape   == (_FULL_AI_H, _FULL_AI_W, 3)
    assert _display_sensory.shape   == (_SENSORY_H, _SENSORY_W, 3)
```

### Step 5: Run smoke tests

```
python -m pytest tests/test_taichi_gui_manager.py -v
```

Expected: `2 passed` (or skipped if no CUDA)

### Step 6: Commit

```bash
git add src/project/visualization/ tests/test_taichi_gui_manager.py src/project/pyg_config.json
git commit -m "feat: add TaichiGUIManager with 3 GGUI visualization windows"
```

---

## Task 4: Qt UI Overhaul — `modern_main_window.py`

**Files:**
- Modify: `src/project/ui/modern_main_window.py` (full rewrite of `__init__` and layout methods)

This task does **not** change any business logic. All existing handler methods (`_handle_start`, `_handle_stop`, `_toggle_audio`, etc.) are preserved exactly. Only the layout construction code and stylesheet change.

### Step 1: Read the full current file before starting

Read `src/project/ui/modern_main_window.py` completely to understand what methods exist and what `self.*` attributes are referenced from handlers. Do not remove any existing method. Do not change any existing method signature.

### Step 2: Replace `__init__` layout construction

Replace everything from `super().__init__()` to the end of `__init__` (keeping the same structure but building the 3-column layout). The full replacement for `__init__`:

```python
    def __init__(self, config_manager: ConfigManager, state_manager: StateManager) -> None:
        super().__init__()
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.frame_counter = 0
        self.system: Any | None = None
        self.capture: Any | None = None
        self.workspace_system: Any | None = None
        self._workspace_observer_added = False
        self._gui_manager: Any | None = None   # TaichiGUIManager, set by start_system()

        self.setWindowTitle("Neural Engine")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(self._get_dark_theme_stylesheet())

        # ── Root layout ──────────────────────────────────────────────
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────────
        self._header_bar = self._build_header_bar()
        root_layout.addWidget(self._header_bar)

        # ── 3-column body ────────────────────────────────────────────
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(8)
        root_layout.addWidget(body, stretch=1)

        # Left column (live views + metrics)
        left_col = self._build_left_column()
        body_layout.addWidget(left_col, stretch=3)

        # Center column (Taichi windows + modality legend)
        center_col = self._build_center_column()
        body_layout.addWidget(center_col, stretch=2)

        # Right column (engine config + controls)
        right_col = self._build_right_column()
        body_layout.addWidget(right_col, stretch=2)

        # ── Status bar ───────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "color: #9999bb; background-color: #0d0d0d; "
            "font-family: 'Consolas'; font-size: 11px; border-top: 1px solid #2a2a4e;"
        )
        self.setStatusBar(self.status_bar)

        # ── Audio state ──────────────────────────────────────────────
        self.audio_capture: Any | None = None
        self.audio_output:  Any | None = None

        # ── Timers ───────────────────────────────────────────────────
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        self.resource_stats_timer = QTimer()
        self.resource_stats_timer.timeout.connect(self._update_resource_stats_display)

        # ── Performance throttling (unchanged from original) ─────────
        self.last_update_time = 0
        self.last_sensory_canvas_update = 0
        self.last_workspace_canvas_update = 0.0
        self.last_dynamic_canvas_update = 0.0
        self.sensory_canvas_update_interval = 0.5
        self.dynamic_canvas_update_interval = 0.5
        self.canvas_frame_counter = 0
        self.min_update_interval = 0.001
        self.frame_skip_counter = 0
        self.frame_skip_threshold = 100
        self.node_read_skip_counter = 0
        self.node_read_interval = 2

        # ── Thread safety ─────────────────────────────────────────────
        self._ui_update_lock = threading.Lock()
        self._resource_access_lock = threading.Lock()

        # ── State observer ───────────────────────────────────────────
        self.state_manager.add_observer(self)
```

### Step 3: Add the layout builder methods

Add these new private methods to the class (after `__init__`):

```python
    # ──────────────────────────────────────────────────────────────────
    # Layout builders
    # ──────────────────────────────────────────────────────────────────

    def _build_header_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(40)
        bar.setStyleSheet(
            "background-color: #12122a; border-bottom: 1px solid #2a2a4e;"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)

        title = QLabel("● NEURAL ENGINE")
        title.setStyleSheet("color: #44ff88; font-weight: bold; font-family: 'Consolas';")
        layout.addWidget(title)
        layout.addStretch()

        self._header_fps   = QLabel("FPS: —")
        self._header_nodes = QLabel("Nodes: —")
        self._header_energy = QLabel("Energy: —")
        self._header_status = QLabel("Idle")
        for lbl in (self._header_fps, self._header_nodes, self._header_energy, self._header_status):
            lbl.setStyleSheet("color: #9999cc; font-family: 'Consolas'; font-size: 12px; margin: 0 10px;")
            layout.addWidget(lbl)
        return bar

    def _build_left_column(self) -> QFrame:
        col = QFrame()
        col.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        layout = QVBoxLayout(col)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Workspace canvas
        self.workspace_view = QGraphicsView()
        self.workspace_view.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #2a2a4e; border-radius: 4px;"
        )
        self.workspace_scene = QGraphicsScene()
        self.workspace_view.setScene(self.workspace_scene)
        self.workspace_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        layout.addWidget(self.workspace_view, stretch=3)

        # Sensory thumbnail
        self.sensory_view = QGraphicsView()
        self.sensory_view.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #2a2a4e; border-radius: 4px;"
        )
        self.sensory_scene = QGraphicsScene()
        self.sensory_view.setScene(self.sensory_scene)
        self.sensory_view.setMaximumHeight(150)
        self._sensory_pixmap_item = None
        layout.addWidget(self.sensory_view, stretch=1)

        # Audio spectrum
        self.audio_view = QGraphicsView()
        self.audio_view.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #2a2a4e; border-radius: 4px;"
        )
        self.audio_scene = QGraphicsScene()
        self.audio_view.setScene(self.audio_scene)
        self.audio_view.setMaximumHeight(90)
        self.audio_view.setVisible(False)
        layout.addWidget(self.audio_view)

        # Metrics
        self.metrics_panel = QFrame()
        self.metrics_panel.setStyleSheet(
            "background-color: #12122a; border-radius: 4px; border: 1px solid #2a2a4e;"
        )
        self.metrics_panel.setMaximumHeight(160)
        m_layout = QVBoxLayout(self.metrics_panel)
        m_layout.setContentsMargins(8, 6, 8, 6)
        self.metrics_label = QLabel()
        self.metrics_label.setStyleSheet(
            "color: #c0c0e0; font-family: 'Consolas'; font-size: 11px;"
        )
        self.metrics_label.setWordWrap(True)
        m_layout.addWidget(self.metrics_label)
        layout.addWidget(self.metrics_panel)

        return col

    def _build_center_column(self) -> QFrame:
        col = QFrame()
        col.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        col.setMaximumWidth(280)
        layout = QVBoxLayout(col)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Section: Taichi Windows
        sec_title = QLabel("TAICHI WINDOWS")
        sec_title.setStyleSheet(
            "color: #666699; font-family: 'Consolas'; font-size: 10px; letter-spacing: 1px;"
        )
        layout.addWidget(sec_title)

        # Window toggle buttons + FPS labels
        self._btn_workspace_win = QPushButton("▶  Workspace Grid")
        self._btn_workspace_win.setCheckable(True)
        self._btn_workspace_win.setStyleSheet(self._window_toggle_style(False))
        self._btn_workspace_win.clicked.connect(self._toggle_workspace_window)
        layout.addWidget(self._btn_workspace_win)

        self._lbl_workspace_fps = QLabel("  FPS: —")
        self._lbl_workspace_fps.setStyleSheet("color: #666699; font-family: 'Consolas'; font-size: 10px;")
        layout.addWidget(self._lbl_workspace_fps)

        self._btn_full_ai_win = QPushButton("▶  Full AI Structure")
        self._btn_full_ai_win.setCheckable(True)
        self._btn_full_ai_win.setStyleSheet(self._window_toggle_style(False))
        self._btn_full_ai_win.clicked.connect(self._toggle_full_ai_window)
        layout.addWidget(self._btn_full_ai_win)

        self._lbl_full_ai_fps = QLabel("  FPS: —")
        self._lbl_full_ai_fps.setStyleSheet("color: #666699; font-family: 'Consolas'; font-size: 10px;")
        layout.addWidget(self._lbl_full_ai_fps)

        self._btn_sensory_win = QPushButton("▶  Sensory Input")
        self._btn_sensory_win.setCheckable(True)
        self._btn_sensory_win.setStyleSheet(self._window_toggle_style(False))
        self._btn_sensory_win.clicked.connect(self._toggle_sensory_window)
        layout.addWidget(self._btn_sensory_win)

        self._lbl_sensory_fps = QLabel("  FPS: —")
        self._lbl_sensory_fps.setStyleSheet("color: #666699; font-family: 'Consolas'; font-size: 10px;")
        layout.addWidget(self._lbl_sensory_fps)

        # Divider
        layout.addSpacing(8)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #2a2a4e;")
        layout.addWidget(divider)
        layout.addSpacing(4)

        # Section: Modality Legend
        legend_title = QLabel("MODALITY ZONES")
        legend_title.setStyleSheet(
            "color: #666699; font-family: 'Consolas'; font-size: 10px; letter-spacing: 1px;"
        )
        layout.addWidget(legend_title)

        for dot_color, name, attr in [
            ("#44ff88", "Visual",     "_lbl_visual_energy"),
            ("#4488ff", "Audio Left", "_lbl_audio_l_energy"),
            ("#ff4466", "Audio Right","_lbl_audio_r_energy"),
            ("#888888", "Neutral",    "_lbl_neutral_energy"),
        ]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {dot_color}; font-size: 14px;")
            row_layout.addWidget(dot)
            label = QLabel(name)
            label.setStyleSheet("color: #c0c0e0; font-family: 'Consolas'; font-size: 11px;")
            row_layout.addWidget(label)
            row_layout.addStretch()
            energy_lbl = QLabel("—")
            energy_lbl.setStyleSheet("color: #666699; font-family: 'Consolas'; font-size: 10px;")
            row_layout.addWidget(energy_lbl)
            setattr(self, attr, energy_lbl)
            layout.addWidget(row)

        layout.addStretch()
        return col

    def _build_right_column(self) -> QFrame:
        col = QFrame()
        col.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        col.setMaximumWidth(260)
        layout = QVBoxLayout(col)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # ── Simulation controls ───────────────────────────────────────
        self._section_label(layout, "SIMULATION")

        self.start_button = QPushButton("▶  Start")
        self.start_button.setStyleSheet(self._button_style("#1a4d1a", "#22662a", "#0d330d"))
        self.start_button.clicked.connect(self._handle_start)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("■  Stop")
        self.stop_button.setStyleSheet(self._button_style("#4d1a1a", "#662222", "#330d0d"))
        self.stop_button.clicked.connect(self._handle_stop)
        layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("↺  Reset Map")
        self.reset_button.setStyleSheet(self._button_style("#2a2a55", "#3a3a77", "#1a1a33"))
        self.reset_button.clicked.connect(self._handle_reset)
        layout.addWidget(self.reset_button)

        layout.addSpacing(4)
        self._section_label(layout, "ACTIONS")

        self.suspend_button = QPushButton("Drain && Suspend")
        self.suspend_button.setStyleSheet(self._button_style("#882222", "#993333", "#661111"))
        self.suspend_button.clicked.connect(self._toggle_suspend)
        layout.addWidget(self.suspend_button)

        self.pulse_button = QPushButton("Pulse +10 Energy")
        self.pulse_button.setStyleSheet(self._button_style("#225577", "#3377aa", "#113355"))
        self.pulse_button.clicked.connect(self._pulse_energy)
        layout.addWidget(self.pulse_button)

        self.sensory_button = QPushButton("Disable Sensory")
        self.sensory_button.setStyleSheet(self._button_style("#228822", "#33aa33", "#115511"))
        self.sensory_button.clicked.connect(self._toggle_sensory)
        layout.addWidget(self.sensory_button)

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.setStyleSheet(self._button_style("#444444", "#555555", "#333333"))
        self.clear_log_button.clicked.connect(self._handle_clear_log)
        layout.addWidget(self.clear_log_button)

        self.test_button = QPushButton("Test Rules")
        self.test_button.setStyleSheet(self._button_style("#7744aa", "#9955cc", "#553388"))
        self.test_button.clicked.connect(self._handle_test_rules)
        layout.addWidget(self.test_button)

        layout.addSpacing(4)
        self._section_label(layout, "AUDIO")

        self.audio_toggle_button = QPushButton("Enable Audio")
        self.audio_toggle_button.setStyleSheet(self._button_style("#664488", "#8855aa", "#443366"))
        self.audio_toggle_button.clicked.connect(self._toggle_audio)
        layout.addWidget(self.audio_toggle_button)

        self.audio_source_button = QPushButton("Source: Loopback")
        self.audio_source_button.setStyleSheet(self._button_style("#446688", "#5577aa", "#334466"))
        self.audio_source_button.clicked.connect(self._toggle_audio_source)
        self.audio_source_button.setVisible(False)
        layout.addWidget(self.audio_source_button)

        layout.addSpacing(4)
        self._section_label(layout, "CONFIG")

        self.config_button = QPushButton("Open Config Panel")
        self.config_button.setStyleSheet(self._button_style("#888822", "#aaa933", "#666611"))
        self.config_button.clicked.connect(self._open_config_panel)
        layout.addWidget(self.config_button)

        layout.addSpacing(4)
        self._section_label(layout, "UPDATE INTERVAL")

        # Re-use existing interval slider builder
        self._create_interval_slider_in(layout)

        layout.addStretch()
        return col

    @staticmethod
    def _section_label(layout: QVBoxLayout, text: str) -> None:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #666699; font-family: 'Consolas'; font-size: 10px; "
            "letter-spacing: 1px; margin-top: 2px;"
        )
        layout.addWidget(lbl)

    @staticmethod
    def _window_toggle_style(active: bool) -> str:
        if active:
            return (
                "QPushButton { background-color: #1a3320; color: #44ff88; "
                "border: 1px solid #44ff88; border-radius: 4px; padding: 6px; "
                "font-family: 'Consolas'; font-size: 11px; text-align: left; }"
                "QPushButton:hover { background-color: #224428; }"
            )
        return (
            "QPushButton { background-color: #1a1a2e; color: #9999cc; "
            "border: 1px solid #2a2a4e; border-radius: 4px; padding: 6px; "
            "font-family: 'Consolas'; font-size: 11px; text-align: left; }"
            "QPushButton:hover { background-color: #22223a; }"
        )
```

### Step 4: Add `_create_interval_slider_in()` and GGUI toggle handlers

Find the existing `_create_interval_slider()` method. Keep it as-is (it may be called from other places). Add a new companion method:

```python
    def _create_interval_slider_in(self, layout: QVBoxLayout) -> None:
        """Build the update interval slider into the given layout."""
        self.interval_label = QLabel("Update: 16 ms")
        self.interval_label.setStyleSheet(
            "color: #9999cc; font-family: 'Consolas'; font-size: 11px;"
        )
        layout.addWidget(self.interval_label)

        self.interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.interval_slider.setMinimum(1)
        self.interval_slider.setMaximum(1000)
        self.interval_slider.setValue(16)
        self.interval_slider.valueChanged.connect(self._update_interval)
        layout.addWidget(self.interval_slider)
```

Add the GGUI window toggle handlers:

```python
    def _toggle_workspace_window(self) -> None:
        if self._gui_manager is None:
            return
        if self._gui_manager.is_open("workspace"):
            self._gui_manager.close_workspace_window()
            self._btn_workspace_win.setText("▶  Workspace Grid")
            self._btn_workspace_win.setStyleSheet(self._window_toggle_style(False))
        else:
            self._gui_manager.open_workspace_window()
            self._btn_workspace_win.setText("■  Workspace Grid")
            self._btn_workspace_win.setStyleSheet(self._window_toggle_style(True))

    def _toggle_full_ai_window(self) -> None:
        if self._gui_manager is None:
            return
        if self._gui_manager.is_open("full_ai"):
            self._gui_manager.close_full_ai_window()
            self._btn_full_ai_win.setText("▶  Full AI Structure")
            self._btn_full_ai_win.setStyleSheet(self._window_toggle_style(False))
        else:
            self._gui_manager.open_full_ai_window()
            self._btn_full_ai_win.setText("■  Full AI Structure")
            self._btn_full_ai_win.setStyleSheet(self._window_toggle_style(True))

    def _toggle_sensory_window(self) -> None:
        if self._gui_manager is None:
            return
        if self._gui_manager.is_open("sensory"):
            self._gui_manager.close_sensory_window()
            self._btn_sensory_win.setText("▶  Sensory Input")
            self._btn_sensory_win.setStyleSheet(self._window_toggle_style(False))
        else:
            self._gui_manager.open_sensory_window()
            self._btn_sensory_win.setText("■  Sensory Input")
            self._btn_sensory_win.setStyleSheet(self._window_toggle_style(True))
```

### Step 5: Wire `_gui_manager` in `start_system()`

Find the `start_system()` method (search for `def start_system`). At the end of it, after the engine is running, add:

```python
        # Set up TaichiGUIManager so center column toggle buttons work
        try:
            from project.visualization.taichi_gui_manager import TaichiGUIManager
            if hasattr(system, 'engine'):
                self._gui_manager = TaichiGUIManager(system.engine)
            elif hasattr(system, '_engine'):
                self._gui_manager = TaichiGUIManager(system._engine)
        except Exception as e:
            logger.warning("TaichiGUIManager unavailable: %s", e)
```

### Step 6: Update `periodic_update()` to refresh header bar and GGUI FPS labels

Find `periodic_update()`. Add at the start (before any existing update calls):

```python
        # Update header bar
        if hasattr(self, '_header_fps') and self.system is not None:
            try:
                metrics = self.system.get_metrics() if hasattr(self.system, 'get_metrics') else {}
                fps_val = getattr(self, '_last_fps', 0)
                nodes   = metrics.get('num_nodes', 0)
                energy  = metrics.get('avg_energy', 0.0)
                self._header_fps.setText(f"FPS: {fps_val:.0f}")
                self._header_nodes.setText(f"Nodes: {nodes:,}")
                self._header_energy.setText(f"Energy: {energy:.1f}")
                self._header_status.setText("Running")
            except Exception:
                pass

        # Update GGUI FPS labels
        if self._gui_manager is not None:
            for name, lbl in [
                ("workspace", self._lbl_workspace_fps),
                ("full_ai",   self._lbl_full_ai_fps),
                ("sensory",   self._lbl_sensory_fps),
            ]:
                fps = self._gui_manager.get_fps(name)
                lbl.setText(f"  FPS: {fps:.0f}" if fps > 0 else "  FPS: —")
```

### Step 7: Remove the old `_create_interval_slider()` reference from `__init__`

The original `__init__` called `self._create_interval_slider()`. Since the new `__init__` calls `self._create_interval_slider_in(layout)` from `_build_right_column()`, the old method call is no longer in `__init__`. Keep the old `_create_interval_slider()` method intact (it may be called elsewhere), just ensure it is not called twice.

### Step 8: Update the dark theme stylesheet

Replace `_get_dark_theme_stylesheet()` with the updated version using the new color scheme:

```python
    def _get_dark_theme_stylesheet(self) -> str:
        return """
        QMainWindow, QWidget {
            background-color: #0d0d0d;
        }
        QFrame {
            background-color: #1a1a2e;
        }
        QPushButton {
            background-color: #2a2a4e;
            color: #c0c0e0;
            border: 1px solid #3a3a5e;
            border-radius: 4px;
            padding: 7px 10px;
            font-family: 'Segoe UI';
            font-size: 11px;
        }
        QPushButton:hover { background-color: #3a3a5e; }
        QPushButton:pressed { background-color: #1a1a3e; }
        QPushButton:disabled { color: #555566; }
        QSlider::groove:horizontal {
            height: 5px;
            background: #2a2a4e;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            width: 12px; height: 12px;
            margin: -3px 0;
            border-radius: 6px;
            background: #4488ff;
        }
        QSlider::handle:horizontal:hover { background: #66aaff; }
        QLabel { color: #c0c0e0; }
        QTabWidget::pane {
            border: 1px solid #2a2a4e;
            background: #1a1a2e;
        }
        QTabBar::tab {
            background: #2a2a4e;
            color: #c0c0e0;
            padding: 5px 10px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }
        QTabBar::tab:selected { background: #3a3a5e; }
        QLineEdit, QCheckBox {
            color: #c0c0e0;
            background-color: #2a2a4e;
            border: 1px solid #3a3a5e;
            padding: 3px 6px;
            border-radius: 3px;
        }
        QStatusBar { color: #9999bb; background-color: #0d0d0d; }
        QScrollBar:vertical {
            background: #1a1a2e; width: 8px;
        }
        QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 4px; }
        """
```

### Step 9: Smoke-test the UI launches

```
cd c:/Users/chris/Documents/ai-project/ai-project
python -c "
from PyQt6.QtWidgets import QApplication
import sys
from project.utils.config_manager import ConfigManager
from project.system.state_manager import StateManager
from project.ui.modern_main_window import ModernMainWindow
app = QApplication(sys.argv)
cm = ConfigManager()
sm = StateManager()
w = ModernMainWindow(cm, sm)
w.show()
print('UI launched OK')
app.quit()
"
```

Expected: `UI launched OK` (no exceptions)

### Step 10: Commit

```bash
git add src/project/ui/modern_main_window.py
git commit -m "feat: Qt UI 3-column overhaul with header bar and Taichi window controls"
```

---

## Task 5: Integration — Wire TaichiGUIManager into `main.py` Startup

**Files:**
- Modify: `src/project/main.py` (pass gui_manager reference to window)

The `TaichiGUIManager` needs the engine's `energy_field` to be fully initialized before any window is opened. Wire it during system startup.

### Step 1: In `initialize_system()` in `main.py`, after the engine is created

Find the call to `HybridNeuralSystemAdapter(...)` (~line 721). After the adapter is created:

```python
    # Store engine reference on adapter for GUI manager access
    # (adapter.engine is already set inside HybridNeuralSystemAdapter.__init__)
    adapter = HybridNeuralSystemAdapter(
        engine, SENSORY_REGION, WORKSPACE_REGION, sensory_width, sensory_height,
        audio_sensory_L=AUDIO_SENSORY_L_REGION if audio_enabled else None,
        audio_sensory_R=AUDIO_SENSORY_R_REGION if audio_enabled else None,
        audio_workspace_L=AUDIO_WORKSPACE_L_REGION if audio_enabled else None,
        audio_workspace_R=AUDIO_WORKSPACE_R_REGION if audio_enabled else None,
    )
    return adapter
```

No change needed here — the `ModernMainWindow.start_system()` already accesses `system.engine` or `system._engine`. Verify by checking `HybridNeuralSystemAdapter` stores the engine as `self.engine`.

### Step 2: Find HybridNeuralSystemAdapter and verify engine attribute

```
grep -n "self.engine\|self._engine" src/project/main.py
```

If the adapter stores it as `self.engine`, the `start_system()` change in Task 4 already handles it. If stored differently, update the `start_system()` code accordingly.

### Step 3: Manual integration test

Start the full application:

```
python -m project.main
```

1. The Qt window should open with the new 3-column layout
2. Start the simulation
3. Click "▶ Workspace Grid" in the center column → a 512×512 Taichi GGUI window should open showing the workspace energy
4. Click again → it closes
5. Click "▶ Full AI Structure" → a 320×240 heat-map window opens
6. Click "▶ Sensory Input" → a 480×270 window shows desktop capture energy

### Step 4: Commit

```bash
git add src/project/main.py
git commit -m "feat: wire TaichiGUIManager into system startup for GGUI window launch"
```

---

## Task 6: GGUI Multi-Window Fallback (if needed)

**Only do this task if Task 5 integration test fails** because `ti.ui.Window` cannot run in separate threads on this Windows machine.

If you see errors like `RuntimeError: Cannot create window in non-main thread` or Vulkan crashes, do the following:

### Fallback: Single tiled 1280×480 window

Create a new module-level field:
```python
_display_tiled = ti.field(dtype=ti.f32, shape=(480, 1280, 3))
```

Write a tiling kernel that blits all three displays side by side:
```python
@ti.kernel
def _fill_tiled():
    for py, px in ti.ndrange(480, 1280):
        if px < 512:                     # left: workspace (scaled from 512×512)
            sy = int(py / 480 * 512)
            sx = px
            for c in ti.static(range(3)):
                _display_tiled[py, px, c] = _display_workspace[sy, sx, c]
        elif px < 512 + 320:             # center: full_ai (scaled to fill)
            sy = int(py / 480 * 240)
            sx = px - 512
            for c in ti.static(range(3)):
                _display_tiled[py, px, c] = _display_full_ai[sy, sx, c]
        else:                             # right: sensory (scaled)
            sy = int(py / 480 * 270)
            sx = px - 512 - 320
            for c in ti.static(range(3)):
                _display_tiled[py, px, c] = _display_sensory[sy, sx, c]
```

Replace the three thread loops with a single main-thread loop. The `TaichiGUIManager` would need to run its loop on a non-Qt background thread with the single window.

Commit as: `fix: collapse GGUI to single tiled window (multi-thread unavailable on platform)`

---

## Task 7: Final Validation

### Step 1: Run full test suite

```
python -m pytest tests/ -v --tb=short
```

Expected: all existing tests still pass + new modality tests pass.

### Step 2: Full application smoke test

```
python -m project.main
```

Verify:
- [ ] Qt window opens with 3-column layout
- [ ] Header bar shows FPS/nodes/energy once simulation starts
- [ ] Workspace canvas (left column) still shows live updates
- [ ] Center column: 3 toggle buttons for Taichi windows
- [ ] Modality legend shows 4 colored dots
- [ ] Right column: all original controls present and functional
- [ ] Taichi workspace window opens and shows grayscale grid
- [ ] Taichi full AI window opens and shows heat-map
- [ ] Taichi sensory window opens and shows grayscale
- [ ] All 3 windows can be toggled open/closed independently
- [ ] Audio controls still work
- [ ] Config panel still opens

### Step 3: Commit docs update

```bash
git add docs/
git commit -m "docs: mark Taichi GUI + modality overhaul plan as complete"
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ti.lang.exception_utils.TaichiCompilationError` on `_fill_workspace_display` | `from __future__ import annotations` added | Remove it from taichi_gui_manager.py |
| `RuntimeError: fields must be module-level` | Display field inside a class | Move `_display_*` fields to module level |
| GGUI window flickers / crashes on close | Taichi GGUI + Qt event loop conflict | Increase `thread.join(timeout)` to 5s; close window before joining |
| `AttributeError: 'NoneType' has no attribute 'open_workspace_window'` | `_gui_manager` not set because system not started | Enable buttons only after simulation starts |
| Modality bits read back as 0 despite passing `modalities=[1,2,3]` | Old `_pack_state_batch` call site missed | Search for all calls to `_pack_state_batch` and add the new `modality` arg |
| Energy normalization gives black screen | `e_hi - e_lo ≈ 0` when workspace is empty | The `+1e-6` guard in the kernel handles this — verify it was not removed |
