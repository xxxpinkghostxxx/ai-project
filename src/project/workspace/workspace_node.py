# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Constants:
#   DEFAULT_SMOOTHING_FACTOR = 0.1
#
# Classes:
#   WorkspaceNode:
#     __init__(self, node_id: int, grid_x: int, grid_y: int, grid_z: int = 0)
#
#     add_associated_sensory_node(self, sensory_node_id: int)
#       Add a sensory node to this workspace node's association list
#
#     update_energy(self, new_energy: float)
#       Update energy level with EMA smoothing
#
#     get_smoothed_energy(self, smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR)
#         -> float
#       Get smoothed energy value using exponential moving average (O(1))
#
#     get_energy_trend(self) -> str
#       Get energy trend (increasing, decreasing, stable)
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

"""WorkspaceNode representing individual nodes in the workspace grid."""

from collections import deque
from typing import List

DEFAULT_SMOOTHING_FACTOR = 0.1


class WorkspaceNode:
    """Represents a single workspace node in the 16x16 grid."""

    def __init__(self, node_id: int, grid_x: int, grid_y: int, grid_z: int = 0):
        """
        Initialize a workspace node.

        Args:
            node_id: Unique identifier for the node
            grid_x: X coordinate in the workspace grid
            grid_y: Y coordinate in the workspace grid
            grid_z: Z coordinate (depth layer, default 0)
        """
        self.node_id = node_id
        self.grid_position = (grid_x, grid_y)
        self.z = grid_z
        self.associated_sensory_nodes = []
        self.current_energy = 0.0
        self.energy_history = deque(maxlen=100)
        self.ui_pixel_position = (grid_x, grid_y)
        self.visual_state = 0
        self.last_update_time = 0.0
        self._ema: float = 0.0

    def add_associated_sensory_node(self, sensory_node_id: int):
        """Add a sensory node to this workspace node's association list."""
        if sensory_node_id not in self.associated_sensory_nodes:
            self.associated_sensory_nodes.append(sensory_node_id)

    def update_energy(self, new_energy: float):
        """Update the node's energy level with smoothing."""
        if not self.energy_history:
            self._ema = new_energy
        else:
            self._ema += DEFAULT_SMOOTHING_FACTOR * (new_energy - self._ema)
        self.current_energy = new_energy
        self.energy_history.append(new_energy)

    def get_smoothed_energy(self, smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR) -> float:
        """Get smoothed energy value using exponential moving average (O(1))."""
        if not self.energy_history:
            return 0.0
        if smoothing_factor == DEFAULT_SMOOTHING_FACTOR:
            return self._ema
        smoothed = self.energy_history[0]
        for i in range(1, len(self.energy_history)):
            smoothed = smoothed + smoothing_factor * (self.energy_history[i] - smoothed)
        return smoothed

    def get_energy_trend(self) -> str:
        """Get energy trend (increasing, decreasing, stable)."""
        if len(self.energy_history) < 3:
            return "stable"

        recent_avg = sum(self.energy_history[-3:]) / 3
        if len(self.energy_history) >= 6:
            older_avg = sum(self.energy_history[-6:-3]) / 3
        else:
            older_entries = self.energy_history[:-3]
            older_avg = sum(older_entries) / len(older_entries) if older_entries else recent_avg

        if recent_avg > older_avg * 1.05:
            return "increasing"
        elif recent_avg < older_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
