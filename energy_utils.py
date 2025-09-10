"""
energy_utils.py

Centralized energy calculation utilities to eliminate code duplication.
Provides consistent energy calculations, node property updates, and energy management.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from torch_geometric.data import Data
from config_manager import get_system_constants
from logging_utils import log_step


def get_node_energy_cap() -> float:
    """Get node energy cap from configuration."""
    constants = get_system_constants()
    return constants.get('node_energy_cap', 244.0)


def get_energy_cap_255() -> float:
    """Get energy cap 255 from configuration."""
    constants = get_system_constants()
    return constants.get('energy_cap_255', 255.0)


def calculate_total_energy(graph: Data) -> float:
    """
    Calculate total energy across all nodes in the graph.
    
    Args:
        graph: PyTorch Geometric graph
        
    Returns:
        Total energy value
    """
    if not hasattr(graph, 'x') or graph.x is None:
        return 0.0
    
    return float(torch.sum(graph.x[:, 0]).item())


def calculate_energy_statistics(graph: Data) -> Dict[str, float]:
    """
    Calculate comprehensive energy statistics for the graph using optimized calculations.
    
    Args:
        graph: PyTorch Geometric graph
        
    Returns:
        Dictionary containing energy statistics
    """
    # Use optimized energy calculations
    from performance_optimizer_v3 import optimize_energy_calculations
    return optimize_energy_calculations(graph)


def _calculate_energy_entropy(energy_values: np.ndarray) -> float:
    """Calculate energy entropy for distribution analysis."""
    if len(energy_values) <= 1:
        return 0.0
    
    # Normalize energy values to probabilities
    total_energy = np.sum(energy_values)
    if total_energy <= 0:
        return 0.0
    
    probabilities = energy_values / total_energy
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return float(entropy)


def update_node_energy_safe(graph: Data, node_id: int, new_energy: float) -> bool:
    """
    Safely update node energy with bounds checking.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: Node index
        new_energy: New energy value
        
    Returns:
        True if successful, False otherwise
    """
    if not hasattr(graph, 'x') or graph.x is None:
        return False
    
    if node_id >= graph.x.shape[0]:
        return False
    
    # Clamp energy to valid range
    clamped_energy = max(0.0, min(new_energy, get_node_energy_cap()))
    graph.x[node_id, 0] = clamped_energy
    
    return True


def update_node_property_safe(graph: Data, node_id: int, property_name: str, value: Any) -> bool:
    """
    Safely update a node property.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: Node index
        property_name: Property name to update
        value: New property value
        
    Returns:
        True if successful, False otherwise
    """
    if not hasattr(graph, 'node_labels') or node_id >= len(graph.node_labels):
        return False
    
    graph.node_labels[node_id][property_name] = value
    return True


def get_node_property_safe(graph: Data, node_id: int, property_name: str, default: Any = None) -> Any:
    """
    Safely get a node property.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: Node index
        property_name: Property name to get
        default: Default value if property not found
        
    Returns:
        Property value or default
    """
    if not hasattr(graph, 'node_labels') or node_id >= len(graph.node_labels):
        return default
    
    return graph.node_labels[node_id].get(property_name, default)


def calculate_membrane_potential(energy: float) -> float:
    """
    Calculate membrane potential from energy value.
    
    Args:
        energy: Energy value
        
    Returns:
        Membrane potential (0.0 to 1.0)
    """
    return min(energy / get_node_energy_cap(), 1.0)


def calculate_energy_transfer(source_energy: float, transfer_rate: float = 0.1) -> float:
    """
    Calculate energy transfer amount.
    
    Args:
        source_energy: Source node energy
        transfer_rate: Transfer rate (0.0 to 1.0)
        
    Returns:
        Energy transfer amount
    """
    return source_energy * transfer_rate


def apply_energy_decay(energy: float, decay_rate: float = 0.95) -> float:
    """
    Apply energy decay.
    
    Args:
        energy: Current energy
        decay_rate: Decay rate (0.0 to 1.0)
        
    Returns:
        Decayed energy
    """
    return energy * decay_rate


def clamp_energy(energy: float) -> float:
    """
    Clamp energy to valid range.
    
    Args:
        energy: Energy value to clamp
        
    Returns:
        Clamped energy value
    """
    return max(0.0, min(energy, get_node_energy_cap()))


def calculate_energy_balance(graph: Data) -> Dict[str, Any]:
    """
    Calculate energy balance and conservation metrics.
    
    Args:
        graph: PyTorch Geometric graph
        
    Returns:
        Dictionary containing energy balance metrics
    """
    stats = calculate_energy_statistics(graph)
    
    # Check for energy conservation violations
    total_energy = stats['total_energy']
    expected_energy = len(graph.node_labels) * get_node_energy_cap() * 0.5  # Assume 50% average
    
    conservation_ratio = total_energy / expected_energy if expected_energy > 0 else 0.0
    conservation_violation = abs(conservation_ratio - 1.0)
    
    # Energy distribution analysis
    energy_values = graph.x[:, 0].cpu().numpy() if hasattr(graph, 'x') else np.array([])
    
    if len(energy_values) > 1:
        skewness = _calculate_skewness(energy_values)
        kurtosis = _calculate_kurtosis(energy_values)
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    return {
        **stats,
        'conservation_ratio': conservation_ratio,
        'conservation_violation': conservation_violation,
        'energy_skewness': skewness,
        'energy_kurtosis': kurtosis,
        'is_conserved': conservation_violation < 0.1  # Within 10% tolerance
    }


def _calculate_skewness(values: np.ndarray) -> float:
    """Calculate skewness of energy distribution."""
    if len(values) <= 2:
        return 0.0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if std_val == 0:
        return 0.0
    
    skewness = np.mean(((values - mean_val) / std_val) ** 3)
    return float(skewness)


def _calculate_kurtosis(values: np.ndarray) -> float:
    """Calculate kurtosis of energy distribution."""
    if len(values) <= 2:
        return 0.0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if std_val == 0:
        return 0.0
    
    kurtosis = np.mean(((values - mean_val) / std_val) ** 4) - 3
    return float(kurtosis)


def batch_update_energies(graph: Data, node_indices: list, energy_updates: list) -> bool:
    """
    Batch update multiple node energies efficiently.
    
    Args:
        graph: PyTorch Geometric graph
        node_indices: List of node indices to update
        energy_updates: List of energy values to set
        
    Returns:
        True if all updates successful, False otherwise
    """
    if not hasattr(graph, 'x') or graph.x is None:
        return False
    
    if len(node_indices) != len(energy_updates):
        return False
    
    try:
        # Convert to numpy for batch operations
        x_np = graph.x.cpu().numpy()
        
        for i, (node_idx, energy) in enumerate(zip(node_indices, energy_updates)):
            if node_idx < x_np.shape[0]:
                x_np[node_idx, 0] = clamp_energy(energy)
        
        # Update graph tensor
        graph.x = torch.tensor(x_np, dtype=graph.x.dtype)
        return True
        
    except Exception as e:
        log_step("Batch energy update failed", error=str(e))
        return False


def validate_energy_bounds(graph: Data) -> Dict[str, Any]:
    """
    Validate that all node energies are within valid bounds.
    
    Args:
        graph: PyTorch Geometric graph
        
    Returns:
        Dictionary containing validation results
    """
    if not hasattr(graph, 'x') or graph.x is None:
        return {
            'valid': False,
            'violations': 0,
            'min_energy': 0.0,
            'max_energy': 0.0,
            'out_of_bounds_nodes': []
        }
    
    energy_values = graph.x[:, 0].cpu().numpy()
    min_energy = float(np.min(energy_values))
    max_energy = float(np.max(energy_values))
    
    # Check for violations
    violations = np.sum((energy_values < 0) | (energy_values > get_node_energy_cap()))
    out_of_bounds_nodes = np.where((energy_values < 0) | (energy_values > get_node_energy_cap()))[0].tolist()
    
    return {
        'valid': violations == 0,
        'violations': int(violations),
        'min_energy': min_energy,
        'max_energy': max_energy,
        'out_of_bounds_nodes': out_of_bounds_nodes
    }
