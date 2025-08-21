"""
connection_logic.py

This module will contain all logic for managing connections (edges) between nodes in the energy-based neural system graph.
Designed for modularity and future extension.
"""

import torch
import random
import numpy as np
from main_graph import select_nodes_by_type, select_nodes_by_state, select_nodes_by_behavior

MAX_DYNAMIC_ENERGY = 1.0  # Define the maximum energy for dynamic nodes
DYNAMIC_ENERGY_THRESHOLD = 0.8 * MAX_DYNAMIC_ENERGY  # 80% threshold

# Edge type constants
EDGE_TYPES = ['excitatory', 'inhibitory', 'modulatory']
DEFAULT_EDGE_WEIGHT = 1.0
DEFAULT_EDGE_DELAY = 0.0


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
        self.delay = 0.0  # Transmission delay
        self.plasticity_tag = False  # For learning
        self.eligibility_trace = 0.0  # STDP-like mechanism
        self.last_activity = 0.0  # For timing-based updates
        self.strength_history = []  # Track weight changes over time
        self.creation_time = 0  # When this edge was created
        self.activation_count = 0  # How many times this edge has been used
    
    def update_eligibility_trace(self, delta_eligibility):
        """Update the eligibility trace for learning."""
        self.eligibility_trace += delta_eligibility
        # Decay eligibility trace over time
        self.eligibility_trace *= 0.95
    
    def record_activation(self, timestamp):
        """Record when this edge was activated."""
        self.last_activity = timestamp
        self.activation_count += 1
    
    def get_effective_weight(self):
        """Get the effective weight considering edge type."""
        if self.type == 'inhibitory':
            return -abs(self.weight)
        elif self.type == 'modulatory':
            return self.weight * 0.5  # Modulatory connections are weaker
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


def create_weighted_connection(graph, source, target, weight, edge_type='excitatory'):
    """
    Create a weighted connection with enhanced attributes.
    
    Args:
        graph: PyTorch Geometric graph
        source: Source node index
        target: Target node index
        weight: Connection weight
        edge_type: Type of connection ('excitatory', 'inhibitory', 'modulatory')
    
    Returns:
        Modified graph with new edge
    """
    # Create enhanced edge
    edge = EnhancedEdge(source, target, weight, edge_type)
    
    # Add to edge_index
    new_edge = torch.tensor([[source], [target]], dtype=torch.long)
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
        edge.weight = max(0.0, edge.weight + weight_change)  # Ensure non-negative
        edge.strength_history.append(edge.weight)
        
        # Keep only recent history
        if len(edge.strength_history) > 100:
            edge.strength_history = edge.strength_history[-100:]
    
    return graph


def intelligent_connection_formation(graph):
    """
    Create intelligent connections based on node behaviors and states.
    This replaces the basic random connection approach with behavior-aware routing.
    """
    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        return graph
    
    # Clear existing connections for fresh formation
    graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    if hasattr(graph, 'edge_attributes'):
        graph.edge_attributes = []
    
    num_nodes = len(graph.node_labels)
    
    # Get nodes by behavior type
    oscillator_indices = select_nodes_by_behavior(graph, 'oscillator')
    integrator_indices = select_nodes_by_behavior(graph, 'integrator')
    relay_indices = select_nodes_by_behavior(graph, 'relay')
    highway_indices = select_nodes_by_behavior(graph, 'highway')
    dynamic_indices = select_nodes_by_behavior(graph, 'dynamic')
    
    # Strategy 1: Oscillators connect to integrators for rhythmic input
    for osc_idx in oscillator_indices:
        # Connect to 2 integrator nodes
        for target in integrator_indices[:2]:
            if target != osc_idx:
                create_weighted_connection(graph, osc_idx, target, 1.0, 'excitatory')
    
    # Strategy 2: Integrators connect to relay nodes for output
    for int_idx in integrator_indices:
        # Connect to 1 relay node
        for target in relay_indices[:1]:
            if target != int_idx:
                create_weighted_connection(graph, int_idx, target, 1.0, 'excitatory')
    
    # Strategy 3: Relay nodes connect to highways for energy distribution
    for relay_idx in relay_indices:
        # Connect to 1 highway node
        for target in highway_indices[:1]:
            if target != relay_idx:
                create_weighted_connection(graph, relay_idx, target, 1.0, 'excitatory')
    
    # Strategy 4: Highways connect back to dynamic nodes for regulation
    for highway_idx in highway_indices:
        # Connect to 3 dynamic nodes for regulation
        dynamic_targets = [idx for idx in dynamic_indices if idx != highway_idx][:3]
        for target in dynamic_targets:
            create_weighted_connection(graph, highway_idx, target, 0.8, 'modulatory')
    
    # Strategy 5: Dynamic nodes connect to each other for basic connectivity
    for dyn_idx in dynamic_indices:
        # Connect to 2 other dynamic nodes
        other_dynamic = [idx for idx in dynamic_indices if idx != dyn_idx][:2]
        for target in other_dynamic:
            create_weighted_connection(graph, dyn_idx, target, 0.6, 'excitatory')
    
    return graph


def update_connection_weights(graph, learning_rate=0.01):
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
            edge.weight = min(edge.weight + weight_change, 5.0)  # Cap at 5.0
            
            # Update eligibility trace
            edge.eligibility_trace += weight_change * 0.1
        
        elif source_activity > 0 and target_activity == 0:
            # Only source active - slight weakening
            weight_change = -learning_rate * 0.1
            edge.weight = max(edge.weight + weight_change, 0.1)  # Minimum 0.1
        
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
