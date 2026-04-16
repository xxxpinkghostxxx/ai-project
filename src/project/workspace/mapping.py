# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Functions:
#   map_sensory_to_workspace(sensory_width: int, sensory_height: int,
#       workspace_size: Tuple[int, int] = (16, 16)) -> Dict[int, List[int]]
#     Deprecated (ADR-001): map sensory nodes to workspace nodes via spatial
#     aggregation. Sensory/workspace are now regions of the unified grid.
#
#   calculate_energy_aggregation(sensory_energies: List[float],
#       method: str = 'average') -> float
#     Calculate aggregated energy from sensory node energies
#
#   create_adaptive_mapping(sensory_energies: np.ndarray,
#       workspace_size: Tuple[int, int] = (16, 16)) -> Dict[int, List[int]]
#     Create adaptive mapping based on energy distribution (delegates to basic)
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [hanging] Implement clustering-based adaptive logic in create_adaptive_mapping()
# - [critical] Deprecated for cube: panel-based I/O removes explicit sensory→workspace
#   mapping; delete after migration.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Workspace mapping utilities (deprecated under ADR-001)."""

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

    x_ratio = sensory_width / workspace_size[0]
    y_ratio = sensory_height / workspace_size[1]

    for wx in range(workspace_size[0]):
        for wy in range(workspace_size[1]):
            workspace_id = wy * workspace_size[0] + wx

            sensory_x_start = int(wx * x_ratio)
            sensory_x_end = int((wx + 1) * x_ratio)
            sensory_y_start = int(wy * y_ratio)
            sensory_y_end = int((wy + 1) * y_ratio)

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
