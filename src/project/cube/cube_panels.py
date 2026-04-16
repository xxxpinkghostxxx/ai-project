# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Functions:
#   face_cell_to_xyz(face: str, u: int, v: int, N: int) -> tuple[int, int, int]
#     Maps one face-local (u, v) cell to cubic (x, y, z) per spec §4 table.
#   iter_panel_cells(panel: Panel, N: int) -> Iterator[tuple[int, int, int]]
#     Yields every (x, y, z) covered by panel origin/size on its face.
#   validate_panel_z_rule(panel: Panel, N: int) -> None
#     Ensures sensory z < N/2 and workspace z >= N/2 for all cells; raises ValueError.
#
# Classes:
#   Panel:
#     dataclass: name, face, role, origin, width, height
#   PanelRegistry:
#     __init__(self, grid: CubeGrid) -> None
#     register(self, panel: Panel) -> int
#       Validates bounds, Z-axis rule, overlap; returns panel id.
#     unregister(self, panel_id: int) -> None
#     inject(self, panel_id: int, data: np.ndarray) -> None
#       Sensory only; writes face-mapped cells (stub path uses grid.write_cell).
#     read(self, panel_id: int) -> np.ndarray
#       Workspace only; reads face-mapped cells into ndarray.
#     get_panel(self, panel_id: int) -> Panel | None
#     list_panels(self) -> list[Panel]
#     panels_with_ids(self) -> list[tuple[int, Panel]]
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [critical] Pre-computed Taichi face_map ndarrays and inject_panel/read_panel
#   kernels per spec §4.
# - [hanging] Panel data dtype flexibility beyond uint8.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Face-stamped I/O panels: registration, Z-axis rule, inject/read stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from .cube_grid import CubeGrid

_VALID_FACES = frozenset({"NEAR", "FAR", "LEFT", "RIGHT", "TOP", "BOTTOM"})
_VALID_ROLES = frozenset({"sensory", "workspace"})


def face_cell_to_xyz(face: str, u: int, v: int, N: int) -> Tuple[int, int, int]:
    if face == "NEAR":
        return (u, v, 0)
    if face == "FAR":
        return (u, v, N - 1)
    if face == "LEFT":
        return (0, u, v)
    if face == "RIGHT":
        return (N - 1, u, v)
    if face == "BOTTOM":
        return (u, 0, v)
    if face == "TOP":
        return (u, N - 1, v)
    raise ValueError(f"Unknown face: {face!r}")


def iter_panel_cells(panel: "Panel", N: int) -> Iterator[Tuple[int, int, int]]:
    u0, v0 = panel.origin
    for du in range(panel.width):
        for dv in range(panel.height):
            u = u0 + du
            v = v0 + dv
            yield face_cell_to_xyz(panel.face, u, v, N)


def validate_panel_z_rule(panel: Panel, N: int) -> None:
    half = N / 2.0
    for x, y, z in iter_panel_cells(panel, N):
        if panel.role == "sensory":
            if not (z < half):
                raise ValueError(
                    f"Sensory panel {panel.name!r} violates z < N/2 at cell {(x, y, z)} (N={N})"
                )
        elif panel.role == "workspace":
            if not (z >= half):
                raise ValueError(
                    f"Workspace panel {panel.name!r} violates z >= N/2 at cell {(x, y, z)} (N={N})"
                )


def _validate_bounds(panel: Panel, N: int) -> None:
    u0, v0 = panel.origin
    for du in range(panel.width):
        for dv in range(panel.height):
            u = u0 + du
            v = v0 + dv
            x, y, z = face_cell_to_xyz(panel.face, u, v, N)
            if not (0 <= x < N and 0 <= y < N and 0 <= z < N):
                raise ValueError(
                    f"Panel {panel.name!r} cell out of bounds: {(x, y, z)} for N={N}"
                )


@dataclass
class Panel:
    """Rectangle on a cube face: sensory intake or workspace output."""

    name: str
    face: str
    role: str
    origin: Tuple[int, int]
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.face not in _VALID_FACES:
            raise ValueError(f"Invalid face: {self.face!r}")
        if self.role not in _VALID_ROLES:
            raise ValueError(f"Invalid role: {self.role!r}")
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be >= 1")


class PanelRegistry:
    """Register panels, enforce Z rule and overlap; inject/read via grid stub."""

    def __init__(self, grid: CubeGrid) -> None:
        self._grid = grid
        self._panels: Dict[int, Panel] = {}
        self._claimed: Set[Tuple[int, int, int]] = set()
        self._next_id = 0

    def register(self, panel: Panel) -> int:
        N = self._grid.N
        _validate_bounds(panel, N)
        validate_panel_z_rule(panel, N)
        cells = list(iter_panel_cells(panel, N))
        overlap = [c for c in cells if c in self._claimed]
        if overlap:
            raise ValueError(f"Panel overlaps existing cells (example {overlap[0]})")
        for c in cells:
            self._claimed.add(c)
        pid = self._next_id
        self._next_id += 1
        self._panels[pid] = panel
        return pid

    def unregister(self, panel_id: int) -> None:
        panel = self._panels.pop(panel_id, None)
        if panel is None:
            return
        for c in iter_panel_cells(panel, self._grid.N):
            self._claimed.discard(c)

    def get_panel(self, panel_id: int) -> Optional[Panel]:
        return self._panels.get(panel_id)

    def list_panels(self) -> List[Panel]:
        return list(self._panels.values())

    def panels_with_ids(self) -> List[Tuple[int, Panel]]:
        return sorted(self._panels.items(), key=lambda kv: kv[0])

    def inject(self, panel_id: int, data: np.ndarray) -> None:
        panel = self._panels.get(panel_id)
        if panel is None:
            raise KeyError(f"Unknown panel_id {panel_id}")
        if panel.role != "sensory":
            raise ValueError("inject is only valid for sensory panels")
        if data.ndim != 2:
            raise ValueError("data must be 2D (row, col)")
        if data.shape != (panel.height, panel.width):
            raise ValueError(
                f"data shape {data.shape} != panel (height,width)=({panel.height},{panel.width})"
            )
        u0, v0 = panel.origin
        for row in range(panel.height):
            for col in range(panel.width):
                u = u0 + col
                v = v0 + row
                x, y, z = face_cell_to_xyz(panel.face, u, v, self._grid.N)
                val = int(data[row, col])
                self._grid.write_cell(x, y, z, val)

    def read(self, panel_id: int) -> np.ndarray:
        panel = self._panels.get(panel_id)
        if panel is None:
            raise KeyError(f"Unknown panel_id {panel_id}")
        if panel.role != "workspace":
            raise ValueError("read is only valid for workspace panels")
        out = np.zeros((panel.height, panel.width), dtype=np.uint8)
        u0, v0 = panel.origin
        for row in range(panel.height):
            for col in range(panel.width):
                u = u0 + col
                v = v0 + row
                x, y, z = face_cell_to_xyz(panel.face, u, v, self._grid.N)
                out[row, col] = np.uint8(self._grid.read_cell(x, y, z))
        return out
