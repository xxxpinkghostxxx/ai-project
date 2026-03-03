# Taichi GUI Visualization + DNA Modality Keys + Qt UI Overhaul

**Date:** 2026-03-03
**Status:** Approved
**Scope:** Three Taichi GGUI visualization windows, DNA modality key system, full Qt UI redesign

---

## Overview

Three parallel additions to the neural simulation:

1. **Taichi GGUI visualization windows** — three standalone GPU-rendered windows showing the workspace grid, full AI structure, and sensory input in real time
2. **DNA modality key system** — 3-bit channel tag added to the reserved bits of each node's packed state, encoding which sensory/workspace modality the node belongs to (visual, audio-left, audio-right, neutral)
3. **Qt UI overhaul** — full redesign of `modern_main_window.py` with a 3-column layout, modality legend, per-modality energy stats, and Taichi window controls

---

## 1. Taichi GGUI Windows

### Design Choice: `ti.ui` (GGUI) over `ti.GUI`

`ti.ui.Window` (GGUI) renders via Vulkan on the GPU. On GTX 1650 + Windows 11, Vulkan is available. This avoids the CUDA→CPU readback that `ti.GUI` would require for each frame on a 2560×1920 float32 tensor.

**Fallback:** If multi-window multi-thread GGUI proves unstable on Windows, collapse all three views into a single `ti.ui.Window` with a tiled 1280×480 layout.

### Three Windows

| Window | Content | Display Size | FPS Target | Color |
|--------|---------|-------------|-----------|-------|
| A — Workspace Grid | `energy_field[workspace strip]` upsampled | 512×512 | 60 | B&W grayscale |
| B — Full AI Structure | `energy_field[2560×1920]` at 1:8 downsample | 320×240 | 30 | Modality-color-coded |
| C — Sensory Input | `energy_field[0:1080, 0:1920]` downsampled | 480×270 | 30 | B&W grayscale |

### Module-Level Display Fields (pre-allocated, global)

Following Taichi's requirement that fields be module-level (not class-level):

```python
# src/project/visualization/taichi_gui_manager.py
_display_workspace  = ti.field(dtype=ti.f32, shape=(512, 512, 3))
_display_full_ai    = ti.field(dtype=ti.f32, shape=(320, 240, 3))
_display_sensory    = ti.field(dtype=ti.f32, shape=(480, 270, 3))
```

### Kernels (module-level)

- `_fill_workspace_display()` — samples workspace strip from `energy_field`, normalizes to [0,1], writes R=G=B (B&W)
- `_fill_full_ai_display()` — 8× downsample of full `energy_field`; maps modality to color: VISUAL=green, AUDIO_LEFT=blue, AUDIO_RIGHT=red, NEUTRAL=gray
- `_fill_sensory_display()` — samples top sensory strip from `energy_field`, B&W

All kernels read from the shared `energy_field` PyTorch CUDA tensor via Taichi's zero-copy CUDA interop.

### Thread Model

```
Qt Main Thread
  └─ TaichiGUIManager.start()
       ├─ Thread-A: window_a loop — fill_workspace(); canvas.set_image(_display_workspace); window_a.show()
       ├─ Thread-B: window_b loop — fill_full_ai(); canvas.set_image(_display_full_ai); window_b.show()
       └─ Thread-C: window_c loop — fill_sensory(); canvas.set_image(_display_sensory); window_c.show()
```

Each thread has a rate limiter (`time.sleep()` budget) to avoid GPU saturation. Windows can be launched and closed independently via Qt buttons.

### New File

`src/project/visualization/taichi_gui_manager.py`
`src/project/visualization/__init__.py`

---

## 2. DNA Modality Key System

### Bit Layout Update

The 64-bit node state (bits 17–0 previously all reserved) gains a 3-bit MODALITY field:

```
Bit 63:        ALIVE
Bits 62–61:    NODE_TYPE (2 bits)
Bits 60–58:    CONN_TYPE (3 bits)
Bits 57–18:    DNA × 8 neighbors (40 bits)
Bits 17–3:     Reserved (still free)
Bits 2–0:      MODALITY (3 bits)  ← NEW
```

### Modality Constants (added to `config.py`)

```python
MODALITY_NEUTRAL     = 0   # dynamic nodes / unassigned
MODALITY_VISUAL      = 1   # desktop sensory input / visual workspace output
MODALITY_AUDIO_LEFT  = 2   # left audio channel
MODALITY_AUDIO_RIGHT = 3   # right audio channel
MODALITY_SHIFT       = 0   # bit position in reserved range
MODALITY_MASK        = 0b111
```

### Modality Assignment at Initialization

**Sensory nodes (type 0):**
| Grid region | Modality |
|------------|---------|
| `y ∈ [0, 1080)` (desktop capture area) | VISUAL |
| `y ∈ [2432, 2560)`, `x ∈ [256, 768)` (audio strip left) | AUDIO_LEFT |
| `y ∈ [2432, 2560)`, `x ∈ [768, 1280)` (audio strip right) | AUDIO_RIGHT |

**Workspace nodes (type 2):**
Column-split of the 128-wide workspace grid:
| Columns | Modality |
|---------|---------|
| `x ∈ [0, 43)` | AUDIO_LEFT |
| `x ∈ [43, 85)` | VISUAL |
| `x ∈ [85, 128)` | AUDIO_RIGHT |

**Dynamic nodes (type 1):**
- Start as `MODALITY_NEUTRAL` on initial seeding
- Inherit parent's modality during spawn (modality bits copied, no mutation)
- Over time, nodes near visual sensory regions inherit VISUAL modality; audio regions inherit audio modality

### Routing Affinity (Spatial, Not Mechanical)

No changes to the transfer kernel. The affinity is emergent from spatial proximity: visual sensory nodes (top of grid) spawn visual-tagged dynamic nodes that naturally flow toward visual workspace nodes (center column, bottom strip). The modality tag is used for:
- Color-coding in the GGUI windows
- Per-modality energy stats in the Qt UI
- Future: transfer strength multiplier (not in this implementation)

### Engine Changes

- `taichi_engine.py` — `_init_nodes()`: set MODALITY bits on sensory/workspace node initialization
- `taichi_engine.py` — `_spawn_kernel()`: copy parent modality bits to child (single `|=` addition)
- `config.py` — add 4 modality constants + MODALITY_SHIFT + MODALITY_MASK

---

## 3. Qt UI Overhaul

### New 3-Column Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  ● NEURAL ENGINE           FPS: 60   Nodes: 1.2M   Energy: 847K    │
├──────────────────┬──────────────────────┬───────────────────────────┤
│  LIVE VIEWS      │  TAICHI WINDOWS      │  ENGINE & CONFIG          │
│                  │                      │                           │
│  Workspace Grid  │  [■ Workspace]  60Hz │  ▶ Start  ■ Stop  ↺ Reset │
│  128×128 canvas  │  [■ AI Structure] 30H│                           │
│                  │  [■ Sensory]    30Hz │  ── Modality Zones ─────  │
│  Sensory Feed    │                      │  ● Visual    123.4 avg    │
│  (desktop thumb) │  Modality Legend     │  ● Audio L    89.1 avg    │
│                  │  ● Visual (green)    │  ● Audio R    91.2 avg    │
│  Audio Spectrum  │  ● Audio Left (blue) │                           │
│  FFT bars        │  ● Audio Right (red) │  ── Parameters ─────────  │
│                  │  ● Neutral (gray)    │  Spawn thresh  [slider]   │
│  ── Metrics ───  │                      │  Death thresh  [slider]   │
│  Alive: 1.24M    │                      │  Transfer str  [slider]   │
│  Dynamic: 980K   │                      │  Update ms     [slider]   │
│  Spawns/s: 12K   │                      │                           │
│  Deaths/s: 11K   │                      │  ── Actions ────────────  │
│                  │                      │  Pulse  Drain  Suspend    │
│                  │                      │  Config  Audio  Reset Map │
├──────────────────┴──────────────────────┴───────────────────────────┤
│  Status: Running • CUDA OK • Audio: loopback • Modality: active     │
└─────────────────────────────────────────────────────────────────────┘
```

### Color Scheme

| Element | Color |
|---------|-------|
| Background | `#0d0d0d` |
| Panels | `#1a1a2e` |
| Panel borders | `#2a2a4e` |
| Visual modality accent | `#44ff88` (green) |
| Audio Left accent | `#4488ff` (blue) |
| Audio Right accent | `#ff4466` (red) |
| Neutral | `#888888` (gray) |
| Header bar | `#12122a` |
| Text primary | `#e0e0ff` |
| Text secondary | `#888899` |

### Header Bar (new)

Always-visible strip at top showing: FPS, alive node count, total energy, CUDA status, audio status.

### Center Column (new)

- Three toggle buttons (one per Taichi window) with real-time FPS readout from each window's thread
- Modality color legend (colored dots + labels)
- Powered by `TaichiGUIManager` callbacks

### Left Column (updated)

Keeps existing QGraphicsView panels for workspace canvas, sensory thumbnail, audio spectrum — these become lightweight Qt previews; GGUI windows are the main visualization. Metrics moved below.

### Right Column (reorganized)

Existing buttons reorganized into labeled groups: Simulation controls, Modality zone stats, Parameter sliders, Actions.

### Files Changed

- `src/project/ui/modern_main_window.py` — full rewrite
- `src/project/ui/modern_config_panel.py` — add modality section
- `src/project/pyg_config.json` — add `visualization` section

### New `visualization` Config Section

```json
"visualization": {
  "taichi_workspace_enabled": false,
  "taichi_full_ai_enabled": false,
  "taichi_sensory_enabled": false,
  "workspace_fps": 60,
  "full_ai_fps": 30,
  "sensory_fps": 30
}
```

---

## Files Changed / Created

| File | Change |
|------|--------|
| `src/project/config.py` | Add MODALITY_* constants |
| `src/project/system/taichi_engine.py` | Modality init + spawn inheritance |
| `src/project/visualization/__init__.py` | New module |
| `src/project/visualization/taichi_gui_manager.py` | New — GGUI window manager |
| `src/project/ui/modern_main_window.py` | Full rewrite (3-column layout) |
| `src/project/ui/modern_config_panel.py` | Add modality config tab |
| `src/project/pyg_config.json` | Add visualization section |

---

## Risk Notes

- **GGUI multi-window threading on Windows:** `ti.ui.Window` on Windows has been reported as main-thread-only in some Taichi versions. If multi-thread fails, collapse to a single tiled 1280×480 window.
- **Taichi CUDA + Vulkan interop:** `energy_field` is a PyTorch CUDA tensor. GGUI needs it as a Taichi field. The zero-copy copy from PyTorch to the display Taichi field is safe and is the standard pattern.
- **No Qt GUI changes affect the simulation logic** — all engine changes are additive (new bits, new kernels, no existing kernel modification except spawn).
