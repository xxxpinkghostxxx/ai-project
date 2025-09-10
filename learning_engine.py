"""
learning_engine.py

This module implements the learning and plasticity system for the energy-based neural system.
It includes STDP-like learning, eligibility traces, and memory formation mechanisms
adapted from research concepts to work with the current energy dynamics framework.
"""

import time
import numpy as np
import torch
from logging_utils import log_step, log_node_state

# Import configuration manager
from config_manager import get_learning_config

# Learning constants with configuration fallbacks
# Removed get_learning_config_values() - using config_manager directly


class LearningEngine:
    """
    Centralized learning and plasticity management system.
    Implements STDP-like learning, eligibility traces, and memory formation
    adapted to the energy-based architecture.
    """
    
    def __init__(self):
        """Initialize the learning engine with learning parameters from configuration."""
        config = get_learning_config()
        self.learning_rate = config.get('plasticity_rate', 0.01)
        self.eligibility_decay = config.get('eligibility_decay', 0.95)
        self.stdp_window = config.get('stdp_window', 20.0)
        self.ltp_rate = config.get('ltp_rate', 0.02)
        self.ltd_rate = config.get('ltd_rate', 0.01)
        self.plasticity_threshold = 0.1  # Default threshold
        self.consolidation_threshold = 0.5  # Default threshold
        
        # Learning statistics
        self.learning_stats = {
            'stdp_events': 0,
            'weight_changes': 0,
            'consolidation_events': 0,
            'memory_traces_formed': 0,
            'total_weight_change': 0.0
        }
        
        # Memory traces for persistent patterns
        self.memory_traces = {}
        self.memory_decay_rate = 0.99
    
    def apply_timing_learning(self, pre_node, post_node, edge, delta_t):
        """
        Apply timing-based learning (STDP-like) adapted to energy system.
        
        Instead of membrane potentials, we use energy timing and activation patterns.
        When pre_node activates before post_node, strengthen connection (LTP).
        When post_node activates before pre_node, weaken connection (LTD).
        
        Args:
            pre_node: Pre-synaptic node label
            post_node: Post-synaptic node label
            edge: EnhancedEdge object
            delta_t: Time difference (pre_time - post_time) in seconds
        
        Returns:
            float: Weight change amount
        """
        # Check if timing is within STDP window
        if abs(delta_t) <= self.stdp_window / 1000.0:  # Convert ms to seconds
            if delta_t > 0:  # Pre before post (LTP)
                # Exponential decay based on timing
                weight_change = self.ltp_rate * np.exp(-delta_t / 0.01)
                self.learning_stats['stdp_events'] += 1
                log_step("LTP applied", 
                        pre_id=pre_node.get('id', '?'), 
                        post_id=post_node.get('id', '?'),
                        delta_t=delta_t,
                        weight_change=weight_change)
            else:  # Post before pre (LTD)
                # Exponential decay based on timing
                weight_change = -self.ltd_rate * np.exp(delta_t / 0.01)
                self.learning_stats['stdp_events'] += 1
                log_step("LTD applied", 
                        pre_id=pre_node.get('id', '?'), 
                        post_id=post_node.get('id', '?'),
                        delta_t=delta_t,
                        weight_change=weight_change)
            
            # Update eligibility trace
            edge.eligibility_trace += weight_change
            return weight_change
        
        return 0.0
    
    def consolidate_connections(self, graph):
        """
        Implement connection strength consolidation based on eligibility traces.
        Only synapses with high eligibility and active plasticity tags undergo lasting changes.
        
        Args:
            graph: PyTorch Geometric graph
        
        Returns:
            Modified graph with consolidated weights
        """
        if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
            return graph
        
        consolidated_count = 0
        total_weight_change = 0.0
        
        for edge_idx, edge in enumerate(graph.edge_attributes):
            # Check if edge has sufficient eligibility for consolidation
            if edge.eligibility_trace > self.consolidation_threshold:
                # Apply weight change
                weight_change = edge.eligibility_trace * self.learning_rate
                
                # Ensure weight stays within bounds
                new_weight = edge.weight + weight_change
                new_weight = max(0.1, min(5.0, new_weight))
                
                actual_change = new_weight - edge.weight
                edge.weight = new_weight
                
                # Reset eligibility trace after consolidation
                edge.eligibility_trace *= 0.5
                
                # Update statistics
                consolidated_count += 1
                total_weight_change += actual_change
                self.learning_stats['consolidation_events'] += 1
                
                log_step("Connection consolidated", 
                        edge_idx=edge_idx,
                        source=edge.source,
                        target=edge.target,
                        weight_change=actual_change,
                        new_weight=new_weight)
        
        # Update global statistics
        self.learning_stats['weight_changes'] += consolidated_count
        self.learning_stats['total_weight_change'] += total_weight_change
        
        return graph
    
    def form_memory_traces(self, graph):
        """
        Create persistent connection patterns based on successful information flow.
        This implements memory formation by identifying stable activation patterns.
        
        Args:
            graph: PyTorch Geometric graph
        
        Returns:
            Modified graph with memory traces
        """
        if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
            return graph
        
        memory_traces_formed = 0
        
        for node_idx, node in enumerate(graph.node_labels):
            if node.get('behavior') in ['integrator', 'relay']:
                # Check if node has stable activation pattern
                if self._has_stable_pattern(node, graph):
                    self._create_memory_trace(node_idx, graph)
                    memory_traces_formed += 1
        
        self.learning_stats['memory_traces_formed'] += memory_traces_formed
        
        if memory_traces_formed > 0:
            log_step("Memory traces formed", count=memory_traces_formed)
        
        return graph
    
    def _has_stable_pattern(self, node, graph):
        """
        Check if a node has a consistent activation pattern over time.
        
        Args:
            node: Node label dictionary
            graph: PyTorch Geometric graph
        
        Returns:
            bool: True if node has stable pattern
        """
        # Check if node has been active recently
        last_activation = node.get('last_activation', 0)
        current_time = time.time()
        
        # Node is considered stable if it activated within last 10 seconds
        # and has sufficient energy
        if (current_time - last_activation < 10.0 and 
            node.get('energy', 0.0) > 0.3):
            return True
        
        return False
    
    def _create_memory_trace(self, node_idx, graph):
        """
        Store successful connection pattern as a memory trace.
        
        Args:
            node_idx: Index of the node
            graph: PyTorch Geometric graph
        """
        # Find incoming connections to this node
        incoming_edges = []
        if hasattr(graph, 'edge_attributes'):
            for edge in graph.edge_attributes:
                if edge.target == node_idx:
                    incoming_edges.append({
                        'source': edge.source,
                        'weight': edge.weight,
                        'type': edge.type
                    })
        
        # Store memory trace
        self.memory_traces[node_idx] = {
            'connections': incoming_edges,
            'strength': 1.0,
            'formation_time': time.time(),
            'activation_count': 0
        }
    
    def apply_memory_influence(self, graph):
        """
        Apply memory traces to influence current connection formation.
        This allows the system to remember and reproduce successful patterns.
        
        Args:
            graph: PyTorch Geometric graph
        
        Returns:
            Modified graph with memory influence
        """
        if not self.memory_traces:
            return graph
        
        # Decay memory traces over time
        current_time = time.time()
        to_remove = []
        
        for node_idx, memory_trace in self.memory_traces.items():
            # Decay memory strength
            age = current_time - memory_trace['formation_time']
            memory_trace['strength'] *= (self.memory_decay_rate ** (age / 60.0))  # Decay per minute
            
            # Remove very weak memories
            if memory_trace['strength'] < 0.1:
                to_remove.append(node_idx)
                continue
            
            # Apply memory influence to current connections
            if node_idx < len(graph.node_labels):
                self._reinforce_memory_pattern(node_idx, memory_trace, graph)
        
        # Remove decayed memories
        for node_idx in to_remove:
            del self.memory_traces[node_idx]
        
        return graph
    
    def _reinforce_memory_pattern(self, node_idx, memory_trace, graph):
        """
        Reinforce connections based on memory trace.
        
        Args:
            node_idx: Index of the node
            memory_trace: Memory trace dictionary
            graph: PyTorch Geometric graph
        """
        if not hasattr(graph, 'edge_attributes'):
            return
        
        # Find existing connections that match memory pattern
        for edge in graph.edge_attributes:
            if edge.target == node_idx:
                # Check if this connection matches a memory pattern
                for mem_conn in memory_trace['connections']:
                    if (edge.source == mem_conn['source'] and 
                        edge.type == mem_conn['type']):
                        # Reinforce this connection based on memory strength
                        reinforcement = memory_trace['strength'] * 0.1
                        edge.weight = min(edge.weight + reinforcement, 5.0)
                        break
    
    def get_learning_statistics(self):
        """Get current learning statistics for monitoring."""
        return self.learning_stats.copy()
    
    def reset_statistics(self):
        """Reset learning statistics."""
        self.learning_stats = {
            'stdp_events': 0,
            'weight_changes': 0,
            'consolidation_events': 0,
            'memory_traces_formed': 0,
            'total_weight_change': 0.0
        }
    
    def get_memory_trace_count(self):
        """Get the number of active memory traces."""
        return len(self.memory_traces)


# Utility functions for learning analysis
def calculate_learning_efficiency(graph):
    """
    Calculate learning efficiency based on weight changes and activity.
    
    Args:
        graph: PyTorch Geometric graph
    
    Returns:
        float: Learning efficiency score (0-1)
    """
    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return 0.0
    
    total_weights = sum(edge.weight for edge in graph.edge_attributes)
    avg_weight = total_weights / len(graph.edge_attributes)
    
    # Higher average weights indicate better learning
    efficiency = min(avg_weight / 2.5, 1.0)  # Normalize to 0-1
    
    return efficiency


def detect_learning_patterns(graph):
    """
    Detect emerging learning patterns in the network.
    
    Args:
        graph: PyTorch Geometric graph
    
    Returns:
        dict: Pattern analysis results
    """
    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return {'patterns_detected': 0}
    
    # Analyze weight distribution
    weights = [edge.weight for edge in graph.edge_attributes]
    weight_variance = np.var(weights) if len(weights) > 1 else 0
    
    # Analyze connection types
    edge_types = {}
    for edge in graph.edge_attributes:
        edge_types[edge.type] = edge_types.get(edge.type, 0) + 1
    
    # Detect patterns based on weight variance and type distribution
    patterns_detected = 0
    if weight_variance > 0.5:  # High variance indicates learning
        patterns_detected += 1
    
    if len(edge_types) > 1:  # Multiple edge types indicate sophistication
        patterns_detected += 1
    
    return {
        'patterns_detected': patterns_detected,
        'weight_variance': weight_variance,
        'edge_type_distribution': edge_types,
        'total_connections': len(graph.edge_attributes)
    }


