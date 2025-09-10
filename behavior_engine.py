"""
behavior_engine.py

This module implements behavior-specific update logic for different node types in the energy-based neural system.
Each behavior type has its own update function that modifies node properties based on their specific characteristics.
"""

import time
import numpy as np
import torch
from logging_utils import log_step, log_node_state

# Import configuration manager
from config_manager import get_enhanced_nodes_config

# Constants for behavior parameters
NODE_ENERGY_CAP = 244.0  # Should match death_and_birth_logic.py

# Get configuration values with defaults
def get_enhanced_config():
    config = get_enhanced_nodes_config()
    return {
        'oscillator_frequency': config.get('oscillator_frequency', 0.1),
        'integrator_threshold': config.get('integrator_threshold', 0.8),
        'relay_amplification': config.get('relay_amplification', 1.5),
        'highway_energy_boost': config.get('highway_energy_boost', 2.0)
    }

# Time constants
TIME_STEP = 0.01  # seconds per simulation step
REFRACTORY_PERIOD = 0.1  # seconds


class BehaviorEngine:
    """
    Centralized behavior management for all node types.
    Each behavior type has its own update function that implements
    the specific dynamics for that node type.
    """
    
    def __init__(self):
        """Initialize the behavior engine with behavior handlers."""
        self.behavior_handlers = {
            'sensory': self.update_sensory_node,
            'dynamic': self.update_dynamic_node,
            'oscillator': self.update_oscillator_node,
            'integrator': self.update_integrator_node,
            'relay': self.update_relay_node,
            'highway': self.update_highway_node,
            'workspace': self.update_workspace_node
        }
        
        # Track behavior statistics
        self.behavior_stats = {
            'oscillator_activations': 0,
            'integrator_activations': 0,
            'relay_transfers': 0,
            'highway_regulations': 0,
            'workspace_syntheses': 0
        }
    
    def update_node_behavior(self, node, graph, step):
        """
        Update a node's behavior based on its behavior type.
        
        Args:
            node: Node label dictionary
            graph: PyTorch Geometric graph
            step: Current simulation step
        
        Returns:
            Updated node label dictionary
        """
        behavior = node.get('behavior', 'dynamic')
        handler = self.behavior_handlers.get(behavior, self.update_dynamic_node)
        
        log_step(f"Updating {behavior} node behavior", node_id=node.get('id', '?'), step=step)
        
        try:
            updated_node = handler(node, graph, step)
            # Update last_update field
            updated_node['last_update'] = step
            return updated_node
        except Exception as e:
            log_step(f"Error updating {behavior} node", error=str(e), node_id=node.get('id', '?'))
            # Return original node if update fails
            node['last_update'] = step
            return node
    
    def update_sensory_node(self, node, graph, step):
        """
        Update sensory node behavior.
        Sensory nodes maintain their energy values and don't change behavior.
        """
        # Sensory nodes are static - just update membrane potential to match energy
        energy = node.get('energy', 0.0)
        node['membrane_potential'] = min(energy / 255.0, 1.0)
        node['state'] = 'active'  # Sensory nodes are always active
        return node
    
    def update_dynamic_node(self, node, graph, step):
        """
        Update basic dynamic node behavior.
        Dynamic nodes have simple energy dynamics and state transitions.
        """
        # Update membrane potential based on current energy
        energy = node.get('energy', 0.0)
        node['membrane_potential'] = min(energy / NODE_ENERGY_CAP, 1.0)
        
        # Update state based on energy level
        threshold = node.get('threshold', 0.3)
        if energy > threshold:
            node['state'] = 'active'
        else:
            node['state'] = 'inactive'
        
        # Update refractory timer
        refractory_timer = node.get('refractory_timer', 0.0)
        if refractory_timer > 0:
            node['refractory_timer'] = max(0.0, refractory_timer - TIME_STEP)
        
        return node
    
    def update_oscillator_node(self, node, graph, step):
        """
        Update oscillator node behavior.
        Oscillators emit periodic energy pulses when their membrane potential exceeds threshold.
        """
        # Get oscillator parameters from configuration
        config = get_enhanced_config()
        oscillation_freq = node.get('oscillation_freq', config['oscillator_frequency'])
        threshold = node.get('threshold', 0.8)
        refractory_timer = node.get('refractory_timer', 0.0)
        membrane_potential = node.get('membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            node['refractory_timer'] = max(0.0, refractory_timer - TIME_STEP)
            return node  # Can't oscillate during refractory period
        
        # Accumulate energy in membrane potential (oscillator behavior)
        # This simulates the natural oscillation of biological oscillators
        energy_increment = oscillation_freq * TIME_STEP * 0.1  # Scale factor for reasonable oscillation
        membrane_potential += energy_increment
        
        # Check if threshold is reached
        if membrane_potential >= threshold:
            # Emit energy pulse
            node['last_activation'] = time.time()
            node['refractory_timer'] = REFRACTORY_PERIOD
            node['membrane_potential'] = 0.0  # Reset after activation
            node['state'] = 'active'
            
            # Update statistics
            self.behavior_stats['oscillator_activations'] += 1
            
            log_step("Oscillator activated", 
                    node_id=node.get('id', '?'), 
                    frequency=oscillation_freq,
                    step=step)
        else:
            node['state'] = 'inactive'
        
        # Clamp membrane potential
        node['membrane_potential'] = min(membrane_potential, 1.0)
        
        return node
    
    def update_integrator_node(self, node, graph, step):
        """
        Update integrator node behavior.
        Integrators accumulate incoming energy and activate when threshold is reached.
        """
        # Get integrator parameters from configuration
        config = get_enhanced_config()
        integration_rate = node.get('integration_rate', 0.5)
        threshold = node.get('threshold', config['integrator_threshold'])
        refractory_timer = node.get('refractory_timer', 0.0)
        membrane_potential = node.get('membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            node['refractory_timer'] = max(0.0, refractory_timer - TIME_STEP)
            return node  # Can't integrate during refractory period
        
        # Integrators accumulate energy from incoming connections
        # For now, we'll simulate this with a small constant increment
        # In the future, this should be based on actual incoming connections
        energy_increment = integration_rate * TIME_STEP * 0.05  # Small accumulation per step
        membrane_potential += energy_increment
        
        # Check if threshold is reached
        if membrane_potential >= threshold:
            # Integrator activates
            node['last_activation'] = time.time()
            node['refractory_timer'] = REFRACTORY_PERIOD
            node['membrane_potential'] = 0.0  # Reset after activation
            node['state'] = 'consolidating'  # Special state for integrators
            
            # Update statistics
            self.behavior_stats['integrator_activations'] += 1
            
            log_step("Integrator activated", 
                    node_id=node.get('id', '?'), 
                    rate=integration_rate,
                    step=step)
        else:
            node['state'] = 'active' if membrane_potential > threshold * 0.5 else 'inactive'
        
        # Clamp membrane potential
        node['membrane_potential'] = min(membrane_potential, 1.0)
        
        return node
    
    def update_relay_node(self, node, graph, step):
        """
        Update relay node behavior.
        Relays transfer energy with amplification and can learn from successful transfers.
        """
        # Get relay parameters from configuration
        config = get_enhanced_config()
        relay_amplification = node.get('relay_amplification', config['relay_amplification'])
        threshold = node.get('threshold', 0.4)
        refractory_timer = node.get('refractory_timer', 0.0)
        membrane_potential = node.get('membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            node['refractory_timer'] = max(0.0, refractory_timer - TIME_STEP)
            return node  # Can't relay during refractory period
        
        # Relays have moderate energy accumulation
        energy_increment = 0.02 * TIME_STEP  # Moderate accumulation
        membrane_potential += energy_increment
        
        # Check if threshold is reached
        if membrane_potential >= threshold:
            # Relay activates and transfers energy
            node['last_activation'] = time.time()
            node['refractory_timer'] = REFRACTORY_PERIOD * 0.5  # Shorter refractory for relays
            node['membrane_potential'] = 0.0  # Reset after activation
            node['state'] = 'active'
            
            # Update statistics
            self.behavior_stats['relay_transfers'] += 1
            
            log_step("Relay activated", 
                    node_id=node.get('id', '?'), 
                    amplification=relay_amplification,
                    step=step)
        else:
            # Check if node has active connections to determine state
            node['state'] = 'pending' if membrane_potential > threshold * 0.3 else 'inactive'
        
        # Clamp membrane potential
        node['membrane_potential'] = min(membrane_potential, 1.0)
        
        return node
    
    def update_highway_node(self, node, graph, step):
        """
        Update highway node behavior.
        Highway nodes provide high-capacity energy distribution and network regulation.
        """
        # Get highway parameters from configuration
        config = get_enhanced_config()
        highway_energy_boost = node.get('highway_energy_boost', config['highway_energy_boost'])
        threshold = node.get('threshold', 0.2)
        refractory_timer = node.get('refractory_timer', 0.0)
        membrane_potential = node.get('membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            node['refractory_timer'] = max(0.0, refractory_timer - TIME_STEP)
            return node  # Can't regulate during refractory period
        
        # Highway nodes have low threshold and high energy capacity
        energy = node.get('energy', 0.0)
        node['membrane_potential'] = min(energy / NODE_ENERGY_CAP, 1.0)
        
        # Highway nodes can pull energy from dynamic nodes when active
        if membrane_potential >= threshold and refractory_timer <= 0:
            # Boost energy using highway multiplier
            boosted_energy = min(NODE_ENERGY_CAP, energy * highway_energy_boost)
            node['energy'] = boosted_energy
            
            # Set refractory timer for regulation cycle
            node['refractory_timer'] = 0.5  # 0.5 second regulation cycle
            node['last_activation'] = step
            node['state'] = 'regulating'
            
            # Update statistics
            self.behavior_stats['highway_regulations'] += 1
        else:
            node['state'] = 'active'
        
        return node
    
    def update_workspace_node(self, node, graph, step):
        """
        Update workspace node behavior.
        Workspace nodes provide internal workspace for imagination and flexible thinking.
        """
        # Get workspace parameters
        workspace_capacity = node.get('workspace_capacity', 5.0)
        workspace_creativity = node.get('workspace_creativity', 1.5)
        workspace_focus = node.get('workspace_focus', 3.0)
        threshold = node.get('threshold', 0.6)
        refractory_timer = node.get('refractory_timer', 0.0)
        membrane_potential = node.get('membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            node['refractory_timer'] = max(0.0, refractory_timer - TIME_STEP)
            return node  # Can't synthesize during refractory period
        
        # Update membrane potential based on energy
        energy = node.get('energy', 0.0)
        node['membrane_potential'] = min(energy / NODE_ENERGY_CAP, 1.0)
        
        # Workspace nodes can synthesize concepts when active
        if membrane_potential >= threshold and refractory_timer <= 0:
            # Check if workspace has enough capacity for synthesis
            if workspace_capacity >= 2.0:  # Need at least 2 concepts to combine
                # Perform synthesis (simplified version - full logic in workspace_engine.py)
                synthesis_success = np.random.random() < (workspace_creativity * workspace_focus * 0.1)
                
                if synthesis_success:
                    node['state'] = 'synthesizing'
                    node['refractory_timer'] = 1.0 / workspace_creativity  # Higher creativity = faster cycles
                    node['last_activation'] = step
                    
                    # Update statistics
                    self.behavior_stats['workspace_syntheses'] += 1
                else:
                    node['state'] = 'planning'
            else:
                node['state'] = 'imagining'
        elif membrane_potential > threshold * 0.8:
            node['state'] = 'planning'
        elif membrane_potential > threshold * 0.5:
            node['state'] = 'imagining'
        else:
            node['state'] = 'active'
        
        return node
    
    def get_behavior_statistics(self):
        """Get current behavior statistics for monitoring."""
        return self.behavior_stats.copy()
    
    def reset_statistics(self):
        """Reset behavior statistics."""
        self.behavior_stats = {
            'oscillator_activations': 0,
            'integrator_activations': 0,
            'relay_transfers': 0,
            'highway_regulations': 0,
            'workspace_syntheses': 0
        }


# Utility functions for behavior analysis
def should_transition_to_learning(node):
    """
    Determine if a node should transition to learning state.
    
    Args:
        node: Node label dictionary
    
    Returns:
        bool: True if node should transition to learning
    """
    # Check if node has high recent activity
    last_activation = node.get('last_activation', 0)
    current_time = time.time()
    
    # If node activated recently (within last 5 seconds) and has plasticity enabled
    if (current_time - last_activation < 5.0 and 
        node.get('plasticity_enabled', False) and
        node.get('eligibility_trace', 0.0) > 0.1):
        return True
    
    return False


def energy_above_threshold(node):
    """
    Check if node energy is above its threshold.
    
    Args:
        node: Node label dictionary
    
    Returns:
        bool: True if energy above threshold
    """
    energy = node.get('energy', 0.0)
    threshold = node.get('threshold', 0.5)
    return energy > threshold


def has_active_connections(node, graph):
    """
    Check if node has active connections.
    
    Args:
        node: Node label dictionary
        graph: PyTorch Geometric graph
    
    Returns:
        bool: True if node has active connections
    """
    if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
        return False
    
    # Get node index
    node_id = node.get('id', -1)
    if node_id < 0 or node_id >= len(graph.node_labels):
        return False
    
    # Check if node has any outgoing or incoming connections
    has_outgoing = torch.any(graph.edge_index[0] == node_id)
    has_incoming = torch.any(graph.edge_index[1] == node_id)
    
    # Also check if node has high energy (indicating activity)
    energy = node.get('energy', 0.0)
    has_energy = energy > 0.3
    
    # Node has active connections if it has graph connections AND energy
    return (has_outgoing or has_incoming) and has_energy


# Example usage and testing
if __name__ == "__main__":
    # Test behavior engine
    engine = BehaviorEngine()
    
    # Create test nodes
    test_nodes = [
        {
            'id': 1, 'behavior': 'oscillator', 'energy': 0.5,
            'membrane_potential': 0.6, 'threshold': 0.8, 'refractory_timer': 0.0,
            'last_activation': 0, 'plasticity_enabled': True, 'eligibility_trace': 0.0,
            'oscillation_freq': 2.0
        },
        {
            'id': 2, 'behavior': 'integrator', 'energy': 0.7,
            'membrane_potential': 0.75, 'threshold': 0.8, 'refractory_timer': 0.0,
            'last_activation': 0, 'plasticity_enabled': True, 'eligibility_trace': 0.0,
            'integration_rate': 0.5
        }
    ]
    
    # Test behavior updates
    for i, node in enumerate(test_nodes):
        print(f"Before update {i+1}: {node}")
        updated_node = engine.update_node_behavior(node, None, 1)
        print(f"After update {i+1}: {updated_node}")
        print()
    
    # Show statistics
    print("Behavior Statistics:", engine.get_behavior_statistics())
