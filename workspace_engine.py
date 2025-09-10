"""
Workspace Engine for managing internal workspace nodes.

This module implements the workspace system that provides a special place for
"imagination" and flexible thinking, where the system can combine perceptions
and plan actions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from logging_utils import log_runtime, log_step
from config_manager import get_workspace_config


class WorkspaceEngine:
    """
    Manages workspace nodes for imagination and flexible thinking.
    
    Workspace nodes serve as an internal workspace where the system can:
    - Combine multiple sensory inputs and concepts
    - Plan future actions
    - Synthesize new ideas from existing knowledge
    - Maintain focus on complex tasks
    """
    
    def __init__(self):
        """Initialize the workspace engine with configuration."""
        config = get_workspace_config()
        self.workspace_capacity = config.get('workspace_capacity', 5.0)
        self.workspace_creativity = config.get('workspace_creativity', 1.5)
        self.workspace_focus = config.get('workspace_focus', 3.0)
        self.workspace_threshold = config.get('workspace_threshold', 0.6)
        self.workspace_creation_rate = config.get('workspace_creation_rate', 0.1)
        self.workspace_energy_cost = config.get('workspace_energy_cost', 0.3)
        self.workspace_synthesis_rate = config.get('workspace_synthesis_rate', 0.05)
        
        # Workspace state tracking
        self.active_workspaces = []
        self.workspace_concepts = {}
        self.synthesis_history = []
        self.creativity_metrics = {
            'total_syntheses': 0,
            'successful_combinations': 0,
            'focus_duration': 0.0
        }
    
    @log_runtime
    def update_workspace_nodes(self, graph, step: int) -> Dict:
        """
        Update all workspace nodes in the graph.
        
        Args:
            graph: PyTorch Geometric graph with node_labels
            step: Current simulation step
            
        Returns:
            Dictionary with update statistics
        """
        if not hasattr(graph, 'node_labels') or not graph.node_labels:
            return {'workspace_updates': 0, 'syntheses': 0, 'focus_changes': 0}
        
        workspace_updates = 0
        syntheses = 0
        focus_changes = 0
        
        for node_idx, node_label in enumerate(graph.node_labels):
            if node_label.get('type') == 'workspace':
                result = self._update_workspace_node(node_label, graph, step)
                workspace_updates += 1
                syntheses += result.get('synthesis', 0)
                focus_changes += result.get('focus_change', 0)
        
        return {
            'workspace_updates': workspace_updates,
            'syntheses': syntheses,
            'focus_changes': focus_changes
        }
    
    def _update_workspace_node(self, node: Dict, graph, step: int) -> Dict:
        """
        Update a single workspace node.
        
        Args:
            node: Workspace node label dictionary
            graph: PyTorch Geometric graph
            step: Current simulation step
            
        Returns:
            Dictionary with update results
        """
        log_step(f"Updating workspace node {node.get('id', 'unknown')}")
        
        # Get current workspace parameters
        capacity = node.get('workspace_capacity', self.workspace_capacity)
        creativity = node.get('workspace_creativity', self.workspace_creativity)
        focus = node.get('workspace_focus', self.workspace_focus)
        threshold = node.get('threshold', self.workspace_threshold)
        
        # Update membrane potential based on incoming energy
        membrane_potential = node.get('membrane_potential', 0.0)
        energy = node.get('energy', 0.0)
        
        # Calculate energy transfer from connections
        incoming_energy = self._calculate_incoming_energy(node, graph)
        membrane_potential = min(1.0, membrane_potential + incoming_energy * 0.1)
        
        # Update refractory timer
        refractory_timer = max(0.0, node.get('refractory_timer', 0.0) - 0.1)
        
        # Check if workspace should activate
        synthesis_result = {'synthesis': 0, 'focus_change': 0}
        
        if membrane_potential >= threshold and refractory_timer <= 0.0:
            synthesis_result = self._perform_workspace_synthesis(
                node, graph, capacity, creativity, focus, step
            )
            
            # Reset refractory timer
            refractory_timer = 1.0 / creativity  # Higher creativity = faster cycles
            
            # Update last activation
            node['last_activation'] = step
        
        # Update node state
        if synthesis_result['synthesis'] > 0:
            node['state'] = 'synthesizing'
        elif membrane_potential > threshold * 0.8:
            node['state'] = 'planning'
        elif membrane_potential > threshold * 0.5:
            node['state'] = 'imagining'
        else:
            node['state'] = 'active'
        
        # Update node attributes
        node['membrane_potential'] = membrane_potential
        node['refractory_timer'] = refractory_timer
        node['last_update'] = step
        
        return synthesis_result
    
    def _calculate_incoming_energy(self, node: Dict, graph) -> float:
        """
        Calculate total incoming energy from connections.
        
        Args:
            node: Workspace node
            graph: PyTorch Geometric graph
            
        Returns:
            Total incoming energy
        """
        if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
            return 0.0
        
        node_idx = self._find_node_index(node, graph)
        if node_idx is None:
            return 0.0
        
        # Find incoming edges
        edge_index = graph.edge_index.cpu().numpy()
        target_edges = edge_index[1] == node_idx
        
        total_energy = 0.0
        for edge_idx in np.where(target_edges)[0]:
            source_idx = edge_index[0, edge_idx]
            if source_idx < len(graph.node_labels):
                source_node = graph.node_labels[source_idx]
                source_energy = source_node.get('energy', 0.0)
                total_energy += source_energy * 0.1  # 10% transfer rate
        
        return total_energy
    
    def _find_node_index(self, node: Dict, graph) -> Optional[int]:
        """
        Find the index of a node in the graph.
        
        Args:
            node: Node label dictionary
            graph: PyTorch Geometric graph
            
        Returns:
            Node index or None if not found
        """
        for idx, label in enumerate(graph.node_labels):
            if (label.get('type') == node.get('type') and 
                label.get('id') == node.get('id')):
                return idx
        return None
    
    def _perform_workspace_synthesis(
        self, 
        node: Dict, 
        graph, 
        capacity: float, 
        creativity: float, 
        focus: float, 
        step: int
    ) -> Dict:
        """
        Perform workspace synthesis operation.
        
        Args:
            node: Workspace node
            graph: PyTorch Geometric graph
            capacity: Workspace capacity
            creativity: Creativity multiplier
            focus: Focus strength
            step: Current step
            
        Returns:
            Dictionary with synthesis results
        """
        log_step(f"Workspace synthesis: capacity={capacity}, creativity={creativity}, focus={focus}")
        
        # Find available concepts to combine
        available_concepts = self._find_available_concepts(graph, capacity)
        
        if len(available_concepts) < 2:
            return {'synthesis': 0, 'focus_change': 0}
        
        # Calculate synthesis probability based on creativity and focus
        synthesis_prob = min(1.0, creativity * focus * self.workspace_synthesis_rate)
        
        if np.random.random() < synthesis_prob:
            # Perform synthesis
            combined_concept = self._combine_concepts(available_concepts, creativity)
            
            # Store synthesis result
            synthesis_id = len(self.synthesis_history)
            self.synthesis_history.append({
                'step': step,
                'workspace_id': node.get('id'),
                'concepts': available_concepts[:2],  # Use first 2 concepts
                'result': combined_concept,
                'creativity': creativity,
                'focus': focus
            })
            
            # Update creativity metrics
            self.creativity_metrics['total_syntheses'] += 1
            if combined_concept['strength'] > 0.5:
                self.creativity_metrics['successful_combinations'] += 1
            
            # Update focus duration
            self.creativity_metrics['focus_duration'] += focus
            
            return {'synthesis': 1, 'focus_change': 1}
        
        return {'synthesis': 0, 'focus_change': 0}
    
    def _find_available_concepts(self, graph, capacity: float) -> List[Dict]:
        """
        Find available concepts for synthesis.
        
        Args:
            graph: PyTorch Geometric graph
            capacity: Maximum number of concepts to consider
            
        Returns:
            List of available concepts
        """
        concepts = []
        
        if not hasattr(graph, 'node_labels'):
            return concepts
        
        # Look for active nodes with high energy as potential concepts
        for node_label in graph.node_labels:
            if (node_label.get('state') == 'active' and 
                node_label.get('energy', 0.0) > 0.5):
                
                concept = {
                    'type': node_label.get('type', 'unknown'),
                    'energy': node_label.get('energy', 0.0),
                    'behavior': node_label.get('behavior', 'unknown'),
                    'strength': min(1.0, node_label.get('energy', 0.0) / 255.0)
                }
                concepts.append(concept)
                
                if len(concepts) >= int(capacity):
                    break
        
        return concepts
    
    def _combine_concepts(self, concepts: List[Dict], creativity: float) -> Dict:
        """
        Combine multiple concepts into a new synthesis.
        
        Args:
            concepts: List of concepts to combine
            creativity: Creativity multiplier
            
        Returns:
            Combined concept dictionary
        """
        if len(concepts) < 2:
            return {'strength': 0.0, 'type': 'incomplete'}
        
        # Calculate combined strength
        base_strength = sum(c['strength'] for c in concepts) / len(concepts)
        creativity_boost = min(1.0, creativity * 0.2)
        combined_strength = min(1.0, base_strength + creativity_boost)
        
        # Determine combined type
        concept_types = [c['type'] for c in concepts]
        if 'sensory' in concept_types and 'dynamic' in concept_types:
            combined_type = 'sensory_dynamic_synthesis'
        elif 'sensory' in concept_types:
            combined_type = 'sensory_synthesis'
        elif 'dynamic' in concept_types:
            combined_type = 'dynamic_synthesis'
        else:
            combined_type = 'general_synthesis'
        
        return {
            'strength': combined_strength,
            'type': combined_type,
            'components': len(concepts),
            'creativity_boost': creativity_boost
        }
    
    def create_workspace_node(self, node_id: int, step: int) -> Dict:
        """
        Create a new workspace node.
        
        Args:
            node_id: Unique identifier for the workspace node
            step: Current simulation step
            
        Returns:
            Workspace node label dictionary
        """
        return {
            'type': 'workspace',
            'behavior': 'workspace',
            'id': node_id,
            'energy': 0.5,  # Start with moderate energy
            'state': 'active',
            'membrane_potential': 0.3,
            'threshold': self.workspace_threshold,
            'refractory_timer': 0.0,
            'last_activation': step,
            'plasticity_enabled': True,
            'eligibility_trace': 0.0,
            'last_update': step,
            'workspace_capacity': self.workspace_capacity,
            'workspace_creativity': self.workspace_creativity,
            'workspace_focus': self.workspace_focus
        }
    
    def get_workspace_metrics(self) -> Dict:
        """
        Get current workspace metrics.
        
        Returns:
            Dictionary with workspace performance metrics
        """
        return {
            'active_workspaces': len(self.active_workspaces),
            'total_syntheses': self.creativity_metrics['total_syntheses'],
            'successful_combinations': self.creativity_metrics['successful_combinations'],
            'focus_duration': self.creativity_metrics['focus_duration'],
            'synthesis_history_length': len(self.synthesis_history),
            'recent_syntheses': self.synthesis_history[-10:] if self.synthesis_history else []
        }
    
    def should_create_workspace(self, graph, step: int) -> bool:
        """
        Determine if a new workspace node should be created.
        
        Args:
            graph: PyTorch Geometric graph
            step: Current simulation step
            
        Returns:
            True if workspace should be created
        """
        if not hasattr(graph, 'node_labels'):
            return False
        
        # Count existing workspace nodes
        workspace_count = sum(1 for label in graph.node_labels 
                            if label.get('type') == 'workspace')
        
        # Check if we need more workspaces
        if workspace_count == 0:
            return True
        
        # Check creation rate
        if step % 100 == 0:  # Check every 100 steps
            return np.random.random() < self.workspace_creation_rate
        
        return False


# Utility functions
def create_workspace_engine() -> WorkspaceEngine:
    """Create and return a new workspace engine instance."""
    return WorkspaceEngine()


def quick_workspace_analysis(graph) -> Dict:
    """
    Quick analysis of workspace nodes in the graph.
    
    Args:
        graph: PyTorch Geometric graph
        
    Returns:
        Dictionary with workspace analysis
    """
    if not hasattr(graph, 'node_labels'):
        return {'workspace_count': 0, 'workspace_states': {}}
    
    workspace_nodes = [label for label in graph.node_labels 
                      if label.get('type') == 'workspace']
    
    workspace_states = {}
    for node in workspace_nodes:
        state = node.get('state', 'unknown')
        workspace_states[state] = workspace_states.get(state, 0) + 1
    
    return {
        'workspace_count': len(workspace_nodes),
        'workspace_states': workspace_states,
        'total_capacity': sum(node.get('workspace_capacity', 0) for node in workspace_nodes),
        'total_creativity': sum(node.get('workspace_creativity', 0) for node in workspace_nodes)
    }
