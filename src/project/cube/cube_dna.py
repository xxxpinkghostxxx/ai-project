# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Constants:
#   NEIGHBOR_OFFSETS_6
#     Six axis offsets (±X, ±Y, ±Z) for 6-neighbor DNA transfer.
#
# Classes:
#   DnaTransfer:
#     __init__(self, grid: CubeGrid) -> None
#       Holds reference to CubeGrid for future kernels; no Taichi init required.
#     dna_transfer(self) -> None
#       Placeholder step: no energy exchange until lock-and-key kernel exists.
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [critical] Implement Taichi dna_transfer kernel (alive_list, grid_map, int16
#   deltas) per architecture spec §3.
# - [minor] Wire node_dna int32 packing and compute_transfer for 8 connection types.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""6-neighbor DNA transfer engine (stubs until kernels land)."""

from __future__ import annotations

from typing import List, Tuple

from .cube_grid import CubeGrid

NEIGHBOR_OFFSETS_6: List[Tuple[int, int, int]] = [
    (-1, 0, 0),
    (+1, 0, 0),
    (0, -1, 0),
    (0, +1, 0),
    (0, 0, -1),
    (0, 0, +1),
]


class DnaTransfer:
    """6-neighbor lock-and-key energy exchange (no spawn/death/I/O)."""

    def __init__(self, grid: CubeGrid) -> None:
        self._grid = grid

    @property
    def grid(self) -> CubeGrid:
        return self._grid

    def dna_transfer(self) -> None:
        """One DNA transfer sub-step; stub until Taichi kernel is implemented."""
        _ = self._grid
