# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Constants:
#   DEFAULT_MAX_DENSE_N = 256
#     Upper bound on N for allocating a full N×N×N dense uint8 buffer unless
#     CUBE_GRID_MAX_DENSE_N overrides; larger N uses a sparse dict store.
#
# Classes:
#   CubeGrid:
#     __init__(self, N: int, block_size: int = 64) -> None
#       Stores N, block_size; allocates numpy (N,N,N) uint8 when N <= max_dense,
#       else uses per-cell dict storage. No Taichi; sparse SNode tree is deferred.
#     wrap(self, x: int, y: int, z: int) -> tuple[int, int, int]
#       Toroidal wrap via (coord & (N - 1)) for power-of-two N; else modulo.
#     clamp(self) -> None
#       Clamps stored energy values to [0, 255].
#     read_cell(self, x: int, y: int, z: int) -> int
#       Returns uint8 energy at wrapped coordinates.
#     write_cell(self, x: int, y: int, z: int, val: int) -> None
#       Writes clamped uint8 at wrapped coordinates.
#     activate_block(self, bx: int, by: int, bz: int) -> None
#       Records block as active for future sparse allocation hooks.
#     get_energy_field(self) -> numpy.ndarray | None
#       Dense mode: ndarray view; sparse mode: None (no contiguous field).
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [critical] Replace dense/dict stub with Taichi block-sparse SNode tree per
#   architecture spec; avoid O(N³) dense at production N=2048.
# - [minor] Optional Taichi-backed path when grid fits in GPU memory for dev.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""N³ energy grid: toroidal wrap, clamp, and storage (dense or sparse stub)."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_MAX_DENSE_N = 256


def _max_dense_n() -> int:
    raw = os.environ.get("CUBE_GRID_MAX_DENSE_N", str(DEFAULT_MAX_DENSE_N))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_DENSE_N


class CubeGrid:
    """Owns the N³ energy field, toroidal indexing, and clamping (stub storage)."""

    def __init__(self, N: int, block_size: int = 64) -> None:
        if N < 1:
            raise ValueError("N must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.N = N
        self.block_size = block_size
        self._is_power_of_two = N > 0 and (N & (N - 1)) == 0
        self._mask = N - 1 if self._is_power_of_two else None
        max_dense = _max_dense_n()
        self._dense: Optional[np.ndarray] = None
        self._sparse: Dict[Tuple[int, int, int], int] = {}
        self._active_blocks: set[Tuple[int, int, int]] = set()
        if N <= max_dense:
            self._dense = np.zeros((N, N, N), dtype=np.uint8)
        else:
            self._dense = None

    def wrap(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        if self._mask is not None:
            return (x & self._mask, y & self._mask, z & self._mask)
        return (x % self.N, y % self.N, z % self.N)

    def clamp(self) -> None:
        if self._dense is not None:
            np.clip(self._dense, 0, 255, out=self._dense)
        else:
            for k in list(self._sparse.keys()):
                self._sparse[k] = int(np.clip(self._sparse[k], 0, 255))

    def read_cell(self, x: int, y: int, z: int) -> int:
        wx, wy, wz = self.wrap(x, y, z)
        if self._dense is not None:
            return int(self._dense[wx, wy, wz])
        return int(self._sparse.get((wx, wy, wz), 0))

    def write_cell(self, x: int, y: int, z: int, val: int) -> None:
        wx, wy, wz = self.wrap(x, y, z)
        v = int(np.clip(val, 0, 255))
        if self._dense is not None:
            self._dense[wx, wy, wz] = np.uint8(v)
        else:
            if v == 0:
                self._sparse.pop((wx, wy, wz), None)
            else:
                self._sparse[(wx, wy, wz)] = v

    def activate_block(self, bx: int, by: int, bz: int) -> None:
        self._active_blocks.add((bx, by, bz))

    def get_energy_field(self) -> Optional[np.ndarray]:
        """Return dense ndarray for kernels, or None when using sparse dict storage."""
        return self._dense
