"""
Workspace Mapping Utilities

This module provides functions for mapping sensory nodes to workspace nodes.
"""

from typing import Dict, List, Tuple
import numpy as np


def map_sensory_to_workspace(sensory_width: int, sensory_height: int, 
                           workspace_size: Tuple[int, int] = (16, 16)) -> Dict[int, List[int]]:
    """
    Map sensory nodes to workspace nodes using spatial aggregation.
    
    Args:
        sensory_width: Width of sensory grid (e.g., 256)
        sensory_height: Height of sensory grid (e.g., 144)
        workspace_size: Size of workspace grid (16, 16)
    
    Returns:
        Dictionary mapping workspace node IDs to lists of sensory node IDs
    """
    workspace_to_sensory = {}
    
    # Calculate mapping ratio
    x_ratio = sensory_width / workspace_size[0]
    y_ratio = sensory_height / workspace_size[1]
    
    for wx in range(workspace_size[0]):
        for wy in range(workspace_size[1]):
            workspace_id = wx * workspace_size[1] + wy
            
            # Calculate corresponding sensory region
            sensory_x_start = int(wx * x_ratio)
            sensory_x_end = int((wx + 1) * x_ratio)
            sensory_y_start = int(wy * y_ratio)
            sensory_y_end = int((wy + 1) * y_ratio)
            
            # Collect associated sensory nodes
            associated_sensory = []
            for sx in range(sensory_x_start, min(sensory_x_end, sensory_width)):
                for sy in range(sensory_y_start, min(sensory_y_end, sensory_height)):
                    sensory_id = sx * sensory_height + sy
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
    """
    Create adaptive mapping based on energy distribution.
    
    Args:
        sensory_energies: 2D array of sensory node energies
        workspace_size: Target workspace grid size
    
    Returns:
        Adaptive mapping dictionary
    """
    sensory_height, sensory_width = sensory_energies.shape
    
    # Analyze energy distribution
    energy_mean = np.mean(sensory_energies)
    energy_std = np.std(sensory_energies)
    
    # Create energy density map
    energy_density = sensory_energies > (energy_mean + 0.5 * energy_std)
    
    # Adaptive grid sizing based on energy concentration
    # This is a simplified version - could be enhanced with clustering algorithms
    return map_sensory_to_workspace(sensory_width, sensory_height, workspace_size)