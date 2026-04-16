# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Classes:
#   CubeVisualizer:
#     __init__(self, grid: CubeGrid, panels: PanelRegistry, pipeline: CubePipeline) -> None
#       Holds references for future Taichi sampling kernels.
#     get_face_projection(self, face: str, resolution: int = 512) -> np.ndarray
#       Returns zeros (resolution, resolution, 3) float32 RGB placeholder.
#     get_z_slice(self, z: int) -> np.ndarray
#       Returns zeros (N, N) uint8 for XY plane at depth z.
#     get_axis_slices(self, x: int, y: int, z: int) -> tuple[np.ndarray, ...]
#       Returns (xy, xz, yz) each (N, N) uint8 at the given slice indices.
#     get_node_scatter(self, max_points: int = 50000) -> np.ndarray
#       Returns empty (0, 4) float32 until node positions exist; columns x, y, z, energy.
#     get_panel_data(self, panel_id: int) -> np.ndarray
#       Workspace: delegates to PanelRegistry.read; sensory: zeros (height, width) uint8.
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [critical] Taichi kernels for face downsample, slice extraction, scatter sample.
# - [minor] Energy→colormap and panel outline overlays for PyQt widgets.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Cube visualization helpers for PyQt6 (stub arrays until GPU sampling exists)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .cube_grid import CubeGrid
from .cube_panels import PanelRegistry
from .cube_pipeline import CubePipeline

_VALID_FACES = frozenset({"NEAR", "FAR", "LEFT", "RIGHT", "TOP", "BOTTOM"})


class CubeVisualizer:
    """Face projections, slices, scatter, and panel I/O views (stub implementations)."""

    def __init__(
        self,
        grid: CubeGrid,
        panels: PanelRegistry,
        pipeline: CubePipeline,
    ) -> None:
        self._grid = grid
        self._panels = panels
        self._pipeline = pipeline

    def get_face_projection(self, face: str, resolution: int = 512) -> np.ndarray:
        if face not in _VALID_FACES:
            raise ValueError(f"Unknown face: {face!r}")
        out = np.zeros((resolution, resolution, 3), dtype=np.float32)
        return out

    def get_z_slice(self, z: int) -> np.ndarray:
        n = self._grid.N
        if not (0 <= z < n):
            raise ValueError(f"z must be in [0, N), got z={z}, N={n}")
        return np.zeros((n, n), dtype=np.uint8)

    def get_axis_slices(self, x: int, y: int, z: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self._grid.N
        for label, v in (("x", x), ("y", y), ("z", z)):
            if not (0 <= v < n):
                raise ValueError(f"{label} must be in [0, N), got {v}, N={n}")
        xy = np.zeros((n, n), dtype=np.uint8)
        xz = np.zeros((n, n), dtype=np.uint8)
        yz = np.zeros((n, n), dtype=np.uint8)
        return (xy, xz, yz)

    def get_node_scatter(self, max_points: int = 50000) -> np.ndarray:
        _ = max_points
        return np.empty((0, 4), dtype=np.float32)

    def get_panel_data(self, panel_id: int) -> np.ndarray:
        panel = self._panels.get_panel(panel_id)
        if panel is None:
            raise KeyError(f"Unknown panel_id {panel_id}")
        if panel.role == "workspace":
            return self._panels.read(panel_id)
        return np.zeros((panel.height, panel.width), dtype=np.uint8)
