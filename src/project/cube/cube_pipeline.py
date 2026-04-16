# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Constants:
#   DEFAULT_CONFIG
#     Default death_threshold, max_nodes, dna_mutation_rate for stub stats.
#
# Classes:
#   CubePipeline:
#     __init__(self, grid: CubeGrid, dna: DnaTransfer, panels: PanelRegistry,
#              config: dict) -> None
#       Stores references and merged config; alive_count stub at 0 until node lists exist.
#     step(self) -> None
#       Ordered stub: inject (zeros) → dna_transfer → clamp → sync → death → spawn → read.
#     get_alive_count(self) -> int
#       Stub alive count (0) until alive_list is implemented.
#     get_node_stats(self) -> dict
#       Minimal placeholder metrics for UI hooks.
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [critical] Implement alive_list fields, sync/death/spawn kernels, dynamic spawn
#   threshold, and grid_map rebuild per spec §5.
# - [minor] Pass real sensory buffers into inject instead of zeros.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Per-tick pipeline: inject → DNA → clamp → sync → death → spawn → read (stub)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .cube_dna import DnaTransfer
from .cube_grid import CubeGrid
from .cube_panels import PanelRegistry

DEFAULT_CONFIG: Dict[str, Any] = {
    "death_threshold": 1,
    "max_nodes": 4_000_000,
    "dna_mutation_rate": 0.10,
}


class CubePipeline:
    """Owns tick order; alive/spawn/death are stubs until Taichi node state exists."""

    def __init__(
        self,
        grid: CubeGrid,
        dna: DnaTransfer,
        panels: PanelRegistry,
        config: dict,
    ) -> None:
        self._grid = grid
        self._dna = dna
        self._panels = panels
        self._config = {**DEFAULT_CONFIG, **config}
        self._alive_count = 0
        self._tick = 0

    def step(self) -> None:
        for pid, panel in self._panels.panels_with_ids():
            if panel.role == "sensory":
                blank = np.zeros((panel.height, panel.width), dtype=np.uint8)
                self._panels.inject(pid, blank)

        self._dna.dna_transfer()
        self._grid.clamp()
        self._sync_stub()
        self._death_stub()
        self._spawn_stub()

        for pid, panel in self._panels.panels_with_ids():
            if panel.role == "workspace":
                _ = self._panels.read(pid)

        self._tick += 1

    def _sync_stub(self) -> None:
        _ = self._grid

    def _death_stub(self) -> None:
        _ = self._config.get("death_threshold", 1)

    def _spawn_stub(self) -> None:
        _ = self._config.get("max_nodes", 4_000_000)

    def get_alive_count(self) -> int:
        return self._alive_count

    def get_node_stats(self) -> dict:
        return {
            "alive_count": self._alive_count,
            "tick": self._tick,
            "death_threshold": self._config.get("death_threshold"),
            "max_nodes": self._config.get("max_nodes"),
            "dna_mutation_rate": self._config.get("dna_mutation_rate"),
        }
