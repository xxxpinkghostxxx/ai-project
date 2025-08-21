"""
memory_system.py

This module implements the memory formation and persistence system for the energy-based neural system.
It handles memory traces, pattern consolidation, and memory recall mechanisms.
"""

import time
import numpy as np
import torch
from logging_utils import log_step, log_node_state

# Memory constants
DEFAULT_MEMORY_STRENGTH = 1.0
DEFAULT_MEMORY_DECAY_RATE = 0.99
DEFAULT_CONSOLIDATION_THRESHOLD = 0.8
DEFAULT_RECALL_THRESHOLD = 0.3
DEFAULT_MAX_MEMORY_TRACES = 1000


class MemorySystem:
    """
    Memory formation and persistence system for the neural network.
    Implements memory traces, consolidation, and pattern recall mechanisms.
    """
    
    def __init__(self):
        """Initialize the memory system with memory parameters."""
        self.memory_traces = {}  # Node ID -> memory pattern
        self.consolidation_threshold = DEFAULT_CONSOLIDATION_THRESHOLD
        self.memory_decay_rate = DEFAULT_MEMORY_DECAY_RATE
        self.recall_threshold = DEFAULT_RECALL_THRESHOLD
        self.max_memory_traces = DEFAULT_MAX_MEMORY_TRACES
        
        # Memory statistics
        self.memory_stats = {
            'traces_formed': 0,
            'traces_consolidated': 0,
            'traces_decayed': 0,
            'patterns_recalled': 0,
            'total_memory_strength': 0.0
        }
        
        # Pattern recognition cache
        self.pattern_cache = {}
        self.cache_decay_rate = 0.95
    
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
                    # Check if we can form a new memory trace
                    if len(self.memory_traces) < self.max_memory_traces:
                        self._create_memory_trace(node_idx, graph)
                        memory_traces_formed += 1
                    else:
                        # Replace weakest memory trace
                        self._replace_weakest_memory(node_idx, graph)
        
        self.memory_stats['traces_formed'] += memory_traces_formed
        
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
                        'type': edge.type,
                        'source_behavior': graph.node_labels[edge.source].get('behavior', 'unknown')
                    })
        
        # Store memory trace
        self.memory_traces[node_idx] = {
            'connections': incoming_edges,
            'strength': DEFAULT_MEMORY_STRENGTH,
            'formation_time': time.time(),
            'activation_count': 0,
            'last_accessed': time.time(),
            'pattern_type': self._classify_pattern(incoming_edges, graph)
        }
        
        log_step("Memory trace created", 
                node_id=node_idx,
                pattern_type=self.memory_traces[node_idx]['pattern_type'],
                connection_count=len(incoming_edges))
    
    def _classify_pattern(self, connections, graph):
        """
        Classify the type of pattern based on connection characteristics.
        
        Args:
            connections: List of connection dictionaries
            graph: PyTorch Geometric graph
        
        Returns:
            str: Pattern classification
        """
        if not connections:
            return 'isolated'
        
        # Analyze connection types
        excitatory_count = sum(1 for conn in connections if conn['type'] == 'excitatory')
        inhibitory_count = sum(1 for conn in connections if conn['type'] == 'inhibitory')
        modulatory_count = sum(1 for conn in connections if conn['type'] == 'modulatory')
        
        # Analyze source behaviors
        source_behaviors = [conn['source_behavior'] for conn in connections]
        behavior_counts = {}
        for behavior in source_behaviors:
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        # Classify based on dominant characteristics
        if excitatory_count > inhibitory_count + modulatory_count:
            if 'oscillator' in behavior_counts:
                return 'rhythmic_excitatory'
            elif 'integrator' in behavior_counts:
                return 'integrative_excitatory'
            else:
                return 'excitatory_dominant'
        elif inhibitory_count > excitatory_count:
            return 'inhibitory_dominant'
        elif modulatory_count > 0:
            return 'modulatory_influenced'
        else:
            return 'balanced'
    
    def consolidate_memories(self, graph):
        """
        Consolidate memory traces based on repeated activation and stability.
        Stronger memories become more persistent, weaker ones decay faster.
        
        Args:
            graph: PyTorch Geometric graph
        
        Returns:
            Modified graph with consolidated memories
        """
        current_time = time.time()
        consolidated_count = 0
        
        for node_idx, memory_trace in list(self.memory_traces.items()):
            # Check if node is still active and stable
            if node_idx < len(graph.node_labels):
                node = graph.node_labels[node_idx]
                if self._has_stable_pattern(node, graph):
                    # Strengthen memory through consolidation
                    memory_trace['strength'] = min(memory_trace['strength'] * 1.1, 2.0)
                    memory_trace['activation_count'] += 1
                    memory_trace['last_accessed'] = current_time
                    consolidated_count += 1
                    
                    log_step("Memory consolidated", 
                            node_id=node_idx,
                            new_strength=memory_trace['strength'],
                            activation_count=memory_trace['activation_count'])
        
        self.memory_stats['traces_consolidated'] += consolidated_count
        return graph
    
    def decay_memories(self):
        """
        Apply time-based decay to memory traces.
        Weaker memories decay faster, stronger ones persist longer.
        
        Returns:
            int: Number of memories removed due to decay
        """
        current_time = time.time()
        to_remove = []
        
        for node_idx, memory_trace in self.memory_traces.items():
            # Calculate age-based decay
            age = current_time - memory_trace['formation_time']
            base_decay = self.memory_decay_rate ** (age / 60.0)  # Decay per minute
            
            # Stronger memories decay slower
            strength_factor = memory_trace['strength'] / 2.0
            effective_decay = base_decay * (1.0 - strength_factor * 0.5)
            
            # Apply decay
            memory_trace['strength'] *= effective_decay
            
            # Remove very weak memories
            if memory_trace['strength'] < 0.1:
                to_remove.append(node_idx)
        
        # Remove decayed memories
        for node_idx in to_remove:
            del self.memory_traces[node_idx]
            self.memory_stats['traces_decayed'] += 1
        
        return len(to_remove)
    
    def recall_patterns(self, graph, target_node_idx):
        """
        Recall memory patterns that match the current context.
        This allows the system to reproduce successful past patterns.
        
        Args:
            graph: PyTorch Geometric graph
            target_node_idx: Index of the target node
        
        Returns:
            list: List of recalled patterns that could be applied
        """
        recalled_patterns = []
        
        for memory_idx, memory_trace in self.memory_traces.items():
            # Check if this memory pattern could be relevant
            if self._is_pattern_relevant(memory_trace, target_node_idx, graph):
                # Calculate relevance score
                relevance = self._calculate_pattern_relevance(memory_trace, target_node_idx, graph)
                
                if relevance > self.recall_threshold:
                    recalled_patterns.append({
                        'memory_id': memory_idx,
                        'pattern': memory_trace,
                        'relevance': relevance
                    })
        
        # Sort by relevance
        recalled_patterns.sort(key=lambda x: x['relevance'], reverse=True)
        
        if recalled_patterns:
            self.memory_stats['patterns_recalled'] += len(recalled_patterns)
            log_step("Patterns recalled", 
                    target_node=target_node_idx,
                    pattern_count=len(recalled_patterns))
        
        return recalled_patterns
    
    def _is_pattern_relevant(self, memory_trace, target_node_idx, graph):
        """
        Check if a memory pattern is relevant to the current context.
        
        Args:
            memory_trace: Memory trace dictionary
            target_node_idx: Index of the target node
            graph: PyTorch Geometric graph
        
        Returns:
            bool: True if pattern is relevant
        """
        if target_node_idx >= len(graph.node_labels):
            return False
        
        target_node = graph.node_labels[target_node_idx]
        target_behavior = target_node.get('behavior', 'unknown')
        
        # Check if target node has similar characteristics to memory source
        memory_pattern_type = memory_trace.get('pattern_type', 'unknown')
        
        # Simple relevance check based on behavior and pattern type
        if target_behavior in ['integrator', 'relay']:
            if memory_pattern_type in ['integrative_excitatory', 'excitatory_dominant']:
                return True
        
        return False
    
    def _calculate_pattern_relevance(self, memory_trace, target_node_idx, graph):
        """
        Calculate how relevant a memory pattern is to the current context.
        
        Args:
            memory_trace: Memory trace dictionary
            target_node_idx: Index of the target node
            graph: PyTorch Geometric graph
        
        Returns:
            float: Relevance score (0-1)
        """
        if target_node_idx >= len(graph.node_labels):
            return 0.0
        
        target_node = graph.node_labels[target_node_idx]
        
        # Base relevance on memory strength
        base_relevance = memory_trace['strength'] / 2.0
        
        # Boost relevance for recently accessed memories
        current_time = time.time()
        last_accessed = memory_trace['last_accessed']
        recency_boost = 1.0 / (1.0 + (current_time - last_accessed) / 60.0)  # Decay per minute
        
        # Boost relevance for memories with high activation count
        activation_boost = min(memory_trace['activation_count'] / 10.0, 0.5)
        
        # Calculate final relevance
        relevance = base_relevance + (recency_boost * 0.3) + (activation_boost * 0.2)
        
        return min(relevance, 1.0)
    
    def _replace_weakest_memory(self, new_node_idx, graph):
        """
        Replace the weakest memory trace with a new one.
        
        Args:
            new_node_idx: Index of the new node
            graph: PyTorch Geometric graph
        """
        if not self.memory_traces:
            return
        
        # Find weakest memory
        weakest_idx = min(self.memory_traces.keys(), 
                         key=lambda k: self.memory_traces[k]['strength'])
        
        # Remove weakest memory
        del self.memory_traces[weakest_idx]
        
        # Create new memory trace
        self._create_memory_trace(new_node_idx, graph)
        
        log_step("Memory trace replaced", 
                removed_node=weakest_idx,
                new_node=new_node_idx)
    
    def get_memory_statistics(self):
        """Get current memory statistics for monitoring."""
        # Calculate total memory strength
        total_strength = sum(trace['strength'] for trace in self.memory_traces.values())
        self.memory_stats['total_memory_strength'] = total_strength
        
        return self.memory_stats.copy()
    
    def reset_statistics(self):
        """Reset memory statistics."""
        self.memory_stats = {
            'traces_formed': 0,
            'traces_consolidated': 0,
            'traces_decayed': 0,
            'patterns_recalled': 0,
            'total_memory_strength': 0.0
        }
    
    def get_memory_trace_count(self):
        """Get the number of active memory traces."""
        return len(self.memory_traces)
    
    def get_memory_summary(self):
        """Get a summary of all memory traces."""
        summary = []
        for node_idx, trace in self.memory_traces.items():
            summary.append({
                'node_id': node_idx,
                'pattern_type': trace['pattern_type'],
                'strength': trace['strength'],
                'activation_count': trace['activation_count'],
                'age_minutes': (time.time() - trace['formation_time']) / 60.0
            })
        
        return summary


# Utility functions for memory analysis
def analyze_memory_distribution(memory_system):
    """
    Analyze the distribution of memory traces by pattern type.
    
    Args:
        memory_system: MemorySystem instance
    
    Returns:
        dict: Distribution analysis
    """
    summary = memory_system.get_memory_summary()
    
    pattern_distribution = {}
    strength_distribution = []
    
    for memory in summary:
        pattern_type = memory['pattern_type']
        pattern_distribution[pattern_type] = pattern_distribution.get(pattern_type, 0) + 1
        strength_distribution.append(memory['strength'])
    
    return {
        'pattern_distribution': pattern_distribution,
        'avg_strength': np.mean(strength_distribution) if strength_distribution else 0.0,
        'strength_variance': np.var(strength_distribution) if len(strength_distribution) > 1 else 0.0,
        'total_memories': len(summary)
    }


def calculate_memory_efficiency(memory_system):
    """
    Calculate memory system efficiency based on various metrics.
    
    Args:
        memory_system: MemorySystem instance
    
    Returns:
        float: Efficiency score (0-1)
    """
    stats = memory_system.get_memory_statistics()
    
    # Calculate efficiency based on:
    # 1. Memory formation rate
    # 2. Consolidation success
    # 3. Pattern recall effectiveness
    
    formation_efficiency = min(stats['traces_formed'] / 10.0, 1.0)  # Normalize
    consolidation_efficiency = min(stats['traces_consolidated'] / 5.0, 1.0)  # Normalize
    recall_efficiency = min(stats['patterns_recalled'] / 3.0, 1.0)  # Normalize
    
    # Weighted average
    efficiency = (formation_efficiency * 0.4 + 
                 consolidation_efficiency * 0.3 + 
                 recall_efficiency * 0.3)
    
    return efficiency


# Example usage and testing
if __name__ == "__main__":
    # Test memory system
    memory_system = MemorySystem()
    
    print("Memory System initialized successfully!")
    print(f"Max memory traces: {memory_system.max_memory_traces}")
    print(f"Memory decay rate: {memory_system.memory_decay_rate}")
    print(f"Consolidation threshold: {memory_system.consolidation_threshold}")
    
    # Test statistics
    stats = memory_system.get_memory_statistics()
    print(f"Initial statistics: {stats}")
    
    print("\nMemory System is ready for integration!")
