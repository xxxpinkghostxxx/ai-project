"""
Workspace Node Implementation

This module defines the WorkspaceNode class that represents individual nodes
in the 16x16 workspace grid.
"""

import time
from typing import List


class WorkspaceNode:
    """Represents a single workspace node in the 16x16 grid."""
    
    def __init__(self, node_id: int, grid_x: int, grid_y: int):
        """
        Initialize a workspace node.
        
        Args:
            node_id: Unique identifier for the node
            grid_x: X coordinate in the 16x16 grid (0-15)
            grid_y: Y coordinate in the 16x16 grid (0-15)
        """
        self.node_id = node_id
        self.grid_position = (grid_x, grid_y)
        self.associated_sensory_nodes = []  # List of sensory node IDs
        self.current_energy = 0.0
        self.energy_history = []  # Time series data for smoothing
        self.ui_pixel_position = (grid_x, grid_y)
        self.visual_state = 0  # 0-255 for grayscale
        self.last_update_time = 0.0
    
    def add_associated_sensory_node(self, sensory_node_id: int):
        """Add a sensory node to this workspace node's association list."""
        if sensory_node_id not in self.associated_sensory_nodes:
            self.associated_sensory_nodes.append(sensory_node_id)
    
    def update_energy(self, new_energy: float):
        """Update the node's energy level with smoothing."""
        self.current_energy = new_energy
        self.energy_history.append(new_energy)
        
        # Keep only last 100 readings to prevent memory growth
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)
        
        self.last_update_time = time.time()
    
    def get_smoothed_energy(self, smoothing_factor: float = 0.1) -> float:
        """Get smoothed energy value using exponential moving average."""
        if not self.energy_history:
            return 0.0
        
        if len(self.energy_history) == 1:
            return self.energy_history[0]
        
        # Exponential moving average
        smoothed = self.energy_history[0]
        for i in range(1, len(self.energy_history)):
            smoothed = smoothed + smoothing_factor * (self.energy_history[i] - smoothed)
        
        return smoothed
    
    def get_energy_trend(self) -> str:
        """Get energy trend (increasing, decreasing, stable)."""
        if len(self.energy_history) < 3:
            return "stable"
        
        recent_avg = sum(self.energy_history[-3:]) / 3
        older_avg = sum(self.energy_history[-6:-3]) / 3 if len(self.energy_history) >= 6 else self.energy_history[-4]
        
        if recent_avg > older_avg * 1.05:
            return "increasing"
        elif recent_avg < older_avg * 0.95:
            return "decreasing"
        else:
            return "stable"