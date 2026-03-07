"""
Workspace Mapping Utilities

DEPRECATED (ADR-001): This module was written for the PyG graph-based system
where sensory and workspace were discrete node-ID spaces that required an
explicit cross-graph mapping.  Under the unified dynamic node grid those spaces
are *regions* of the same grid, so their spatial relationship is implicit in
the cluster layout — no ID-level mapping table is needed.

These functions are preserved only for backward compatibility.  They will be
removed once all call sites have been updated to use
``TaichiNeuralEngine.register_region()`` and direct field slices.
"""

import logging
import warnings
from typing import Dict, List, Tuple
import numpy as np

_deprecated_logger = logging.getLogger(__name__)


def map_sensory_to_workspace(sensory_width: int, sensory_height: int,
                            workspace_size: Tuple[int, int] = (16, 16)) -> Dict[int, List[int]]:
    """Map sensory nodes to workspace nodes using spatial aggregation.

    .. deprecated::
        ADR-001 — sensory and workspace are now regions of the unified dynamic
        node grid.  Spatial coupling is implicit in grid topology; this
        explicit ID mapping is no longer used in the active simulation path.
        Use ``TaichiNeuralEngine.register_region()`` instead.

    Args:
        sensory_width: Width of sensory grid (e.g., 256)
        sensory_height: Height of sensory grid (e.g., 144)
        workspace_size: Size of workspace grid (16, 16)

    Returns:
        Dictionary mapping workspace node IDs to lists of sensory node IDs
    """
    warnings.warn(
        "map_sensory_to_workspace() is deprecated (ADR-001). "
        "Sensory and workspace are now regions of the unified dynamic node grid. "
        "Use TaichiNeuralEngine.register_region() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _deprecated_logger.debug("map_sensory_to_workspace() called — deprecated (ADR-001)")
    workspace_to_sensory = {}
    
    # Calculate mapping ratio
    x_ratio = sensory_width / workspace_size[0]
    y_ratio = sensory_height / workspace_size[1]
    
    for wx in range(workspace_size[0]):
        for wy in range(workspace_size[1]):
            workspace_id = wy * workspace_size[0] + wx
            
            # Calculate corresponding sensory region
            sensory_x_start = int(wx * x_ratio)
            sensory_x_end = int((wx + 1) * x_ratio)
            sensory_y_start = int(wy * y_ratio)
            sensory_y_end = int((wy + 1) * y_ratio)
            
            # Collect associated sensory nodes
            associated_sensory = []
            for sx in range(sensory_x_start, min(sensory_x_end, sensory_width)):
                for sy in range(sensory_y_start, min(sensory_y_end, sensory_height)):
                    sensory_id = sy * sensory_width + sx
                    associated_sensory.append(sensory_id)
            
            workspace_to_sensory[workspace_id] = associated_sensory
    
    return workspace_to_sensory


def calculate_energy_aggregation(sensory_energies: List[float], method: str = 'average') -> float:
    """
    Calculate aggregated energy from sensory node energies.
    
    Args:
        sensory_energies: List of energy values from associated sensory nodes
        method: Aggregation method ('average', 'weighted', 'maximum')
    
    Returns:
        Aggregated energy value
    """
    if not sensory_energies:
        return 0.0
    
    if method == 'average':
        return sum(sensory_energies) / len(sensory_energies)
    
    elif method == 'weighted':
        # Simple spatial weighting (center nodes have higher weight)
        weights = [1.0] * len(sensory_energies)
        center_weight = 2.0
        if len(sensory_energies) > 0:
            weights[len(sensory_energies) // 2] = center_weight
        
        weighted_sum = sum(energy * weight for energy, weight in zip(sensory_energies, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight
    
    elif method == 'maximum':
        return max(sensory_energies)
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def create_adaptive_mapping(sensory_energies: np.ndarray,
                          workspace_size: Tuple[int, int] = (16, 16)) -> Dict[int, List[int]]:
    """Create adaptive mapping based on energy distribution.

    NOTE: Currently delegates to the basic map_sensory_to_workspace().
    The energy_density analysis is not yet used. Implement clustering-based
    adaptive logic here when needed, or use map_sensory_to_workspace() directly.
    """
    sensory_height, sensory_width = sensory_energies.shape
    return map_sensory_to_workspace(sensory_width, sensory_height, workspace_size)