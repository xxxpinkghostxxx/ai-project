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

from project.system.taichi_engine import project_energy_field_to_2d

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
                ef2d = project_energy_field_to_2d(engine.energy_field)
                e_lo = float(ef2d[ws_y0:ws_y0+ws_h, ws_x0:ws_x0+ws_w].min())
                e_hi = float(ef2d[ws_y0:ws_y0+ws_h, ws_x0:ws_x0+ws_w].max()) + 1e-6
                _fill_workspace_display(ef2d, ws_y0, ws_x0, ws_h, ws_w, e_lo, e_hi)
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
                ef2d = project_energy_field_to_2d(engine.energy_field)
                e_lo = float(ef2d.min())
                e_hi = float(ef2d.max()) + 1e-6
                _fill_full_ai_display(ef2d, engine.H, engine.W, e_lo, e_hi)
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
                ef2d = project_energy_field_to_2d(engine.energy_field)
                e_lo = float(ef2d[:sen_h, :sen_w].min())
                e_hi = float(ef2d[:sen_h, :sen_w].max()) + 1e-6
                _fill_sensory_display(ef2d, 0, 0, sen_h, sen_w, e_lo, e_hi)
                canvas.set_image(_display_sensory)
                window.show()
                elapsed = time.perf_counter() - t0
                sleep_t = frame_budget - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                self._fps["sensory"] = 1.0 / max(time.perf_counter() - t0, 1e-6)
        except Exception:
            logger.exception("GGUI sensory window error")
