"""
connection_logic.py

This module will contain all logic for managing connections (edges) between nodes in the energy-based neural system graph.
Designed for modularity and future extension.
"""

import torch
import random
import numpy as np
import logging
from main_graph import select_nodes_by_type, select_nodes_by_state, select_nodes_by_behavior
from config_manager import get_system_constants
# Import connection constants to replace magic numbers
from energy_constants import ConnectionConstants

# Configuration values now accessed directly from config_manager
# Removed hardcoded constants - using config_manager instead

def get_max_dynamic_energy():
    """Get max dynamic energy from configuration."""
    constants = get_system_constants()
    return constants.get('max_dynamic_energy', 1.0)

def get_dynamic_energy_threshold():
    """Get dynamic energy threshold from configuration."""
    return ConnectionConstants.DYNAMIC_ENERGY_THRESHOLD_FRACTION * get_max_dynamic_energy()

# Edge type constants (using ConnectionConstants)
EDGE_TYPES = ConnectionConstants.EDGE_TYPES
DEFAULT_EDGE_WEIGHT = ConnectionConstants.DEFAULT_EDGE_WEIGHT
DEFAULT_EDGE_DELAY = ConnectionConstants.DEFAULT_EDGE_DELAY


class EnhancedEdge:
    """
    Enhanced edge structure with weights, types, delays, and plasticity features.
    This replaces the basic edge_index approach with rich edge attributes.
    """
    
    def __init__(self, source, target, weight=1.0, edge_type='excitatory'):
        self.source = source
        self.target = target
        self.weight = weight
        self.type = edge_type  # 'excitatory', 'inhibitory', 'modulatory'
        self.delay = ConnectionConstants.EDGE_DELAY_DEFAULT  # Transmission delay
        self.plasticity_tag = False  # For learning
        self.eligibility_trace = ConnectionConstants.ELIGIBILITY_TRACE_DEFAULT  # STDP-like mechanism
        self.last_activity = ConnectionConstants.LAST_ACTIVITY_DEFAULT  # For timing-based updates
        self.strength_history = []  # Track weight changes over time
        self.creation_time = 0  # When this edge was created
        self.activation_count = 0  # How many times this edge has been used
    
    def update_eligibility_trace(self, delta_eligibility):
        """Update the eligibility trace for learning."""
        self.eligibility_trace += delta_eligibility
        # Decay eligibility trace over time
        self.eligibility_trace *= ConnectionConstants.ELIGIBILITY_TRACE_DECAY
    
    def record_activation(self, timestamp):
        """Record when this edge was activated."""
        self.last_activity = timestamp
        self.activation_count += 1
    
    def get_effective_weight(self):
        """Get the effective weight considering edge type."""
        if self.type == 'inhibitory':
            return -abs(self.weight)
        elif self.type == 'modulatory':
            return self.weight * ConnectionConstants.MODULATORY_WEIGHT
        else:  # excitatory
            return abs(self.weight)
    
    def to_dict(self):
        """Convert edge to dictionary for serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'type': self.type,
            'delay': self.delay,
            'plasticity_tag': self.plasticity_tag,
            'eligibility_trace': self.eligibility_trace,
            'last_activity': self.last_activity,
            'activation_count': self.activation_count,
            'creation_time': self.creation_time
        }


def create_weighted_connection(graph, source_id, target_id, weight, edge_type='excitatory'):
    """
    Create a weighted connection with enhanced attributes using node IDs.
    
    Args:
        graph: PyTorch Geometric graph
        source_id: Source node ID
        target_id: Target node ID
        weight: Connection weight
        edge_type: Type of connection ('excitatory', 'inhibitory', 'modulatory')
    
    Returns:
        Modified graph with new edge
    """
    # Import ID manager to convert IDs to indices
    from node_id_manager import get_id_manager
    id_manager = get_id_manager()
    
    # Convert node IDs to array indices
    source_index = id_manager.get_node_index(source_id)
    target_index = id_manager.get_node_index(target_id)
    
    if source_index is None or target_index is None:
        logging.warning(f"Invalid node IDs for connection: source={source_id}, target={target_id}")
        return graph
    
    # Create enhanced edge with IDs
    edge = EnhancedEdge(source_id, target_id, weight, edge_type)
    
    # Add to edge_index using array indices
    new_edge = torch.tensor([[source_index], [target_index]], dtype=torch.long)
    if graph.edge_index.numel() == 0:
        graph.edge_index = new_edge
    else:
        graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    
    # Store edge attributes (we'll use a simple approach for now)
    # In a more sophisticated implementation, this would use edge_attr tensor
    if not hasattr(graph, 'edge_attributes'):
        graph.edge_attributes = []
    
    graph.edge_attributes.append(edge)
    
    return graph


def get_edge_attributes(graph, edge_idx):
    """
    Get edge attributes for a given edge index.
    
    Args:
        graph: PyTorch Geometric graph
        edge_idx: Index of the edge
    
    Returns:
        EnhancedEdge object or None
    """
    if hasattr(graph, 'edge_attributes') and edge_idx < len(graph.edge_attributes):
        return graph.edge_attributes[edge_idx]
    return None


def apply_weight_change(graph, edge_idx, weight_change):
    """
    Apply a weight change to an edge.
    
    Args:
        graph: PyTorch Geometric graph
        edge_idx: Index of the edge
        weight_change: Amount to change the weight
    
    Returns:
        Modified graph
    """
    if hasattr(graph, 'edge_attributes') and edge_idx < len(graph.edge_attributes):
        edge = graph.edge_attributes[edge_idx]
        edge.weight = max(ConnectionConstants.ELIGIBILITY_TRACE_DEFAULT, edge.weight + weight_change)  # Ensure non-negative
        edge.strength_history.append(edge.weight)
        
        # Keep only recent history
        if len(edge.strength_history) > 100:
            edge.strength_history = edge.strength_history[-100:]
    
    return graph


def intelligent_connection_formation(graph):
    """
    Create intelligent connections based on node behaviors and states.
    This replaces the basic random connection approach with behavior-aware routing.
    Uses optimized vectorized connection formation for better performance.
    """
    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        return graph
    
    # Use optimized connection formation
    from performance_optimizer_v3 import optimize_connection_formation
    return optimize_connection_formation(graph)


def update_connection_weights(graph, learning_rate=ConnectionConstants.LEARNING_RATE_DEFAULT):
    """
    Implement weight adaptation based on activity patterns.
    Use eligibility traces for gradual weight changes.
    
    Args:
        graph: PyTorch Geometric graph
        learning_rate: Rate of weight adaptation
    
    Returns:
        Modified graph with updated weights
    """
    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return graph
    
    for edge_idx, edge in enumerate(graph.edge_attributes):
        source = edge.source
        target = edge.target
        
        # Get source and target node activities
        source_activity = graph.node_labels[source].get('last_activation', 0)
        target_activity = graph.node_labels[target].get('last_activation', 0)
        
        # Calculate weight change based on node activities
        if source_activity > 0 and target_activity > 0:
            # Both nodes are active - strengthen connection
            weight_change = learning_rate * (source_activity + target_activity) / 2
            edge.weight = min(edge.weight + weight_change, ConnectionConstants.WEIGHT_CAP_MAX)  # Cap at max
            
            # Update eligibility trace
            edge.eligibility_trace += weight_change * ConnectionConstants.WEIGHT_CHANGE_FACTOR
        
        elif source_activity > 0 and target_activity == 0:
            # Only source active - slight weakening
            weight_change = -learning_rate * ConnectionConstants.WEIGHT_CHANGE_FACTOR
            edge.weight = max(edge.weight + weight_change, ConnectionConstants.WEIGHT_MIN)  # Minimum weight
        
        # Decay eligibility trace
        edge.update_eligibility_trace(0)
    
    return graph


def add_dynamic_connections(graph):
    """
    Legacy function - now calls intelligent connection formation.
    Maintains backward compatibility.
    """
    return intelligent_connection_formation(graph)


def add_connections(graph, connection_strategy=None):
    """
    Add or update connections (edges) in the graph according to the specified strategy.
    
    Args:
        graph: The graph to modify.
        connection_strategy (callable or None): Optional function to determine connection policy.
    
    Returns:
        Modified graph with updated edges.
    """
    if connection_strategy is None:
        connection_strategy = intelligent_connection_formation
    
    return connection_strategy(graph)


# AI/Human: Extend this file with more sophisticated connection strategies as needed.
