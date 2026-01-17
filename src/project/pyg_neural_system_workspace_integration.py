"""
Workspace System Integration for PyG Neural System

This module provides the integration methods for the workspace node system
with the existing PyG neural system.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _init_workspace_system(self: Any) -> None:
    """Initialize workspace node system integration."""
    try:
        # Import workspace system
        from project.workspace.workspace_system import WorkspaceNodeSystem
        from project.workspace.config import EnergyReadingConfig
        
        # Create configuration
        config = EnergyReadingConfig()
        config.grid_size = self.workspace_size
        config.reading_interval_ms = 50  # 20 Hz update rate
        
        # Initialize workspace system
        self.workspace_system = WorkspaceNodeSystem(self, config)
        
        # Add visualization observer if main window is available
        if hasattr(self, 'main_window') and self.main_window:
            from project.workspace.visualization import WorkspaceVisualization
            visualization = WorkspaceVisualization(self.main_window, self.workspace_system)
            self.workspace_system.add_observer(visualization)
        
        logger.info("Workspace system initialized successfully")
        
    except ImportError as e:
        logger.warning(f"Workspace system not available: {e}")
        self.workspace_system = None
    except Exception as e:
        logger.error(f"Failed to initialize workspace system: {e}")
        self.workspace_system = None


def get_node_energy(self: Any, node_id: int) -> float:
    """Get energy level for a specific node."""
    if not hasattr(self, 'g') or self.g is None:
        return 0.0
    
    if not hasattr(self.g, 'energy') or self.g.energy is None:
        return 0.0
    
    if node_id < 0 or node_id >= len(self.g.energy):
        return 0.0
    
    try:
        return float(self.g.energy[node_id].item())
    except Exception as e:
        logger.warning(f"Failed to get energy for node {node_id}: {e}")
        return 0.0


def get_batch_energies(self: Any, node_ids: list[int]) -> list[float]:
    """Get energy levels for multiple nodes efficiently."""
    if not hasattr(self, 'g') or self.g is None:
        return [0.0] * len(node_ids)
    
    if not hasattr(self.g, 'energy') or self.g.energy is None:
        return [0.0] * len(node_ids)
    
    energies = []
    for node_id in node_ids:
        if 0 <= node_id < len(self.g.energy):
            try:
                energies.append(float(self.g.energy[node_id].item()))
            except Exception:
                energies.append(0.0)
        else:
            energies.append(0.0)
    
    return energies


def update_with_workspace(self: Any) -> None:
    """
    Enhanced update method with workspace system integration.
    """
    try:
        # ... existing update logic ...
        
        # Update workspace system if initialized
        if hasattr(self, 'workspace_system') and self.workspace_system:
            try:
                self.workspace_system.update()
            except Exception as e:
                logger.error(f"Workspace system update failed: {e}")
    
    except Exception as e:
        logger.error("Critical error in update: %s", str(e))
        # ... existing error handling ...