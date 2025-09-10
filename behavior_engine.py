"""
behavior_engine.py

This module implements behavior-specific update logic for different node types in the energy-based neural system.
Each behavior type has its own update function that modifies node properties based on their specific characteristics.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, Optional
from torch_geometric.data import Data
from logging_utils import log_step, log_node_state

# Import configuration manager
from config_manager import get_enhanced_nodes_config, get_system_constants

# Configuration values now accessed directly from config_manager
# Removed hardcoded constants - using config_manager instead

# Time constants
TIME_STEP = 0.01  # seconds per simulation step
REFRACTORY_PERIOD = 0.1  # seconds

def get_node_energy_cap() -> float:
    """Get node energy cap from configuration."""
    constants = get_system_constants()
    return constants.get('node_energy_cap', 244.0)

def get_energy_cap_255() -> float:
    """Get energy cap 255 from configuration."""
    constants = get_system_constants()
    return constants.get('energy_cap_255', 255.0)


class BehaviorEngine:
    """
    Centralized behavior management for all node types.
    Each behavior type has its own update function that implements
    the specific dynamics for that node type.
    """
    
    def __init__(self) -> None:
        """Initialize the behavior engine with behavior handlers."""
        self.behavior_handlers: Dict[str, callable] = {
            'sensory': self.update_sensory_node,
            'dynamic': self.update_dynamic_node,
            'oscillator': self.update_oscillator_node,
            'integrator': self.update_integrator_node,
            'relay': self.update_relay_node,
            'highway': self.update_highway_node,
            'workspace': self.update_workspace_node
        }
        
        # Track behavior statistics
        self.behavior_stats: Dict[str, int] = {
            'oscillator_activations': 0,
            'integrator_activations': 0,
            'relay_transfers': 0,
            'highway_regulations': 0,
            'workspace_syntheses': 0
        }
    
    def update_node_behavior(self, node_id: str, graph: Data, step: int) -> bool:
        """
        Update a node's behavior based on its behavior type using ID-based access.
        
        Args:
            node_id: Unique node ID
            graph: PyTorch Geometric graph
            step: Current simulation step
        
        Returns:
            True if update successful, False otherwise
        """
        # Import ID-based access layer
        from node_access_layer import NodeAccessLayer
        access_layer = NodeAccessLayer(graph)
        
        node = access_layer.get_node_by_id(node_id)
        if node is None:
            return False
        
        behavior = node.get('behavior', 'dynamic')
        handler = self.behavior_handlers.get(behavior, self.update_dynamic_node)
        
        log_step(f"Updating {behavior} node behavior", node_id=node_id, step=step)
        
        try:
            updated_node = handler(node_id, graph, step)
            # Update last_update field
            access_layer.update_node_property(node_id, 'last_update', step)
            return True
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            log_step(f"Error updating {behavior} node", error=str(e), node_id=node_id)
            # Update last_update field even if update fails
            access_layer.update_node_property(node_id, 'last_update', step)
            return False
        except Exception as e:
            log_step(f"Unexpected error updating {behavior} node", error=str(e), node_id=node_id)
            # Update last_update field even if update fails
            access_layer.update_node_property(node_id, 'last_update', step)
            return False
    
    def update_sensory_node(self, node_id: str, graph: Data, step: int) -> None:
        """
        Update sensory node behavior.
        Sensory nodes maintain their energy values and don't change behavior.
        """
        # Import ID-based access layer
        from node_access_layer import NodeAccessLayer
        access_layer = NodeAccessLayer(graph)
        
        # Sensory nodes are static - just update membrane potential to match energy
        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            access_layer.update_node_property(node_id, 'membrane_potential', min(energy / get_energy_cap_255(), 1.0))
            access_layer.update_node_property(node_id, 'state', 'active')  # Sensory nodes are always active
    
    def update_dynamic_node(self, node_id: str, graph: Data, step: int) -> None:
        """
        Update basic dynamic node behavior.
        Dynamic nodes have simple energy dynamics and state transitions.
        """
        # Import ID-based access layer
        from node_access_layer import NodeAccessLayer
        access_layer = NodeAccessLayer(graph)
        
        # Update membrane potential based on current energy
        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            access_layer.update_node_property(node_id, 'membrane_potential', min(energy / get_node_energy_cap(), 1.0))
            
            # Update state based on energy level
            threshold = access_layer.get_node_property(node_id, 'threshold', 0.3)
            if energy > threshold:
                access_layer.update_node_property(node_id, 'state', 'active')
            else:
                access_layer.update_node_property(node_id, 'state', 'inactive')
            
            # Update refractory timer
            refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
            if refractory_timer > 0:
                access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - TIME_STEP))
    
    def update_oscillator_node(self, node_id: str, graph: Data, step: int) -> None:
        """
        Update oscillator node behavior.
        Oscillators emit periodic energy pulses when their membrane potential exceeds threshold.
        """
        # Import ID-based access layer
        from node_access_layer import NodeAccessLayer
        access_layer = NodeAccessLayer(graph)
        
        # Get oscillator parameters from configuration
        config = get_enhanced_config()
        oscillation_freq = access_layer.get_node_property(node_id, 'oscillation_freq', config['oscillator_frequency'])
        threshold = access_layer.get_node_property(node_id, 'threshold', 0.8)
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - TIME_STEP))
            return  # Can't oscillate during refractory period
        
        # Accumulate energy in membrane potential (oscillator behavior)
        # This simulates the natural oscillation of biological oscillators
        energy_increment = oscillation_freq * TIME_STEP * 0.1  # Scale factor for reasonable oscillation
        membrane_potential += energy_increment
        
        # Check if threshold is reached
        if membrane_potential >= threshold:
            # Emit energy pulse
            access_layer.update_node_property(node_id, 'last_activation', time.time())
            access_layer.update_node_property(node_id, 'refractory_timer', REFRACTORY_PERIOD)
            access_layer.update_node_property(node_id, 'membrane_potential', 0.0)  # Reset after activation
            access_layer.update_node_property(node_id, 'state', 'active')
            
            # Update statistics
            self.behavior_stats['oscillator_activations'] += 1
            
            log_step("Oscillator activated", 
                    node_id=node_id, 
                    frequency=oscillation_freq,
                    step=step)
        else:
            access_layer.update_node_property(node_id, 'state', 'inactive')
        
        # Clamp membrane potential
        access_layer.update_node_property(node_id, 'membrane_potential', min(membrane_potential, 1.0))
    
    def update_integrator_node(self, node_id: str, graph: Data, step: int) -> None:
        """
        Update integrator node behavior.
        Integrators accumulate incoming energy and activate when threshold is reached.
        """
        # Import ID-based access layer
        from node_access_layer import NodeAccessLayer
        access_layer = NodeAccessLayer(graph)
        
        # Get integrator parameters from configuration
        config = get_enhanced_config()
        integration_rate = access_layer.get_node_property(node_id, 'integration_rate', 0.5)
        threshold = access_layer.get_node_property(node_id, 'threshold', config['integrator_threshold'])
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - TIME_STEP))
            return  # Can't integrate during refractory period
        
        # Integrators accumulate energy from incoming connections
        # For now, we'll simulate this with a small constant increment
        # In the future, this should be based on actual incoming connections
        energy_increment = integration_rate * TIME_STEP * 0.05  # Small accumulation per step
        membrane_potential += energy_increment
        
        # Check if threshold is reached
        if membrane_potential >= threshold:
            # Integrator activates
            access_layer.update_node_property(node_id, 'last_activation', time.time())
            access_layer.update_node_property(node_id, 'refractory_timer', REFRACTORY_PERIOD)
            access_layer.update_node_property(node_id, 'membrane_potential', 0.0)  # Reset after activation
            access_layer.update_node_property(node_id, 'state', 'consolidating')  # Special state for integrators
            
            # Update statistics
            self.behavior_stats['integrator_activations'] += 1
            
            log_step("Integrator activated", 
                    node_id=node_id, 
                    rate=integration_rate,
                    step=step)
        else:
            state = 'active' if membrane_potential > threshold * 0.5 else 'inactive'
            access_layer.update_node_property(node_id, 'state', state)
        
        # Clamp membrane potential
        access_layer.update_node_property(node_id, 'membrane_potential', min(membrane_potential, 1.0))
    
    def update_relay_node(self, node_id: str, graph: Data, step: int) -> None:
        """
        Update relay node behavior.
        Relays transfer energy with amplification and can learn from successful transfers.
        """
        # Import ID-based access layer
        from node_access_layer import NodeAccessLayer
        access_layer = NodeAccessLayer(graph)
        
        # Get relay parameters from configuration
        config = get_enhanced_config()
        relay_amplification = access_layer.get_node_property(node_id, 'relay_amplification', config['relay_amplification'])
        threshold = access_layer.get_node_property(node_id, 'threshold', 0.4)
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - TIME_STEP))
            return  # Can't relay during refractory period
        
        # Relays have moderate energy accumulation
        energy_increment = 0.02 * TIME_STEP  # Moderate accumulation
        membrane_potential += energy_increment
        
        # Check if threshold is reached
        if membrane_potential >= threshold:
            # Relay activates and transfers energy
            access_layer.update_node_property(node_id, 'last_activation', time.time())
            access_layer.update_node_property(node_id, 'refractory_timer', REFRACTORY_PERIOD * 0.5)  # Shorter refractory for relays
            access_layer.update_node_property(node_id, 'membrane_potential', 0.0)  # Reset after activation
            access_layer.update_node_property(node_id, 'state', 'active')
            
            # Update statistics
            self.behavior_stats['relay_transfers'] += 1
            
            log_step("Relay activated", 
                    node_id=node_id, 
                    amplification=relay_amplification,
                    step=step)
        else:
            # Check if node has active connections to determine state
            state = 'pending' if membrane_potential > threshold * 0.3 else 'inactive'
            access_layer.update_node_property(node_id, 'state', state)
        
        # Clamp membrane potential
        access_layer.update_node_property(node_id, 'membrane_potential', min(membrane_potential, 1.0))
    
    def update_highway_node(self, node_id: str, graph: Data, step: int) -> None:
        """
        Update highway node behavior.
        Highway nodes provide high-capacity energy distribution and network regulation.
        """
        # Import ID-based access layer
        from node_access_layer import NodeAccessLayer
        access_layer = NodeAccessLayer(graph)
        
        # Get highway parameters from configuration
        config = get_enhanced_config()
        highway_energy_boost = access_layer.get_node_property(node_id, 'highway_energy_boost', config['highway_energy_boost'])
        threshold = access_layer.get_node_property(node_id, 'threshold', 0.2)
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - TIME_STEP))
            return  # Can't regulate during refractory period
        
        # Highway nodes have low threshold and high energy capacity
        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            access_layer.update_node_property(node_id, 'membrane_potential', min(energy / get_node_energy_cap(), 1.0))
            
            # Highway nodes can pull energy from dynamic nodes when active
            if membrane_potential >= threshold and refractory_timer <= 0:
                # Boost energy using highway multiplier
                boosted_energy = min(get_node_energy_cap(), energy * highway_energy_boost)
                access_layer.set_node_energy(node_id, boosted_energy)
                
                # Set refractory timer for regulation cycle
                access_layer.update_node_property(node_id, 'refractory_timer', 0.5)  # 0.5 second regulation cycle
                access_layer.update_node_property(node_id, 'last_activation', step)
                access_layer.update_node_property(node_id, 'state', 'regulating')
                
                # Update statistics
                self.behavior_stats['highway_regulations'] += 1
            else:
                access_layer.update_node_property(node_id, 'state', 'active')
    
    def update_workspace_node(self, node_id: str, graph: Data, step: int) -> None:
        """
        Update workspace node behavior.
        Workspace nodes provide internal workspace for imagination and flexible thinking.
        """
        # Import ID-based access layer
        from node_access_layer import NodeAccessLayer
        access_layer = NodeAccessLayer(graph)
        
        # Get workspace parameters
        workspace_capacity = access_layer.get_node_property(node_id, 'workspace_capacity', 5.0)
        workspace_creativity = access_layer.get_node_property(node_id, 'workspace_creativity', 1.5)
        workspace_focus = access_layer.get_node_property(node_id, 'workspace_focus', 3.0)
        threshold = access_layer.get_node_property(node_id, 'threshold', 0.6)
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        
        # Update refractory timer
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - TIME_STEP))
            return  # Can't synthesize during refractory period
        
        # Update membrane potential based on energy
        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            access_layer.update_node_property(node_id, 'membrane_potential', min(energy / get_node_energy_cap(), 1.0))
            
            # Workspace nodes can synthesize concepts when active
            if membrane_potential >= threshold and refractory_timer <= 0:
                # Check if workspace has enough capacity for synthesis
                if workspace_capacity >= 2.0:  # Need at least 2 concepts to combine
                    # Perform synthesis (simplified version - full logic in workspace_engine.py)
                    synthesis_success = np.random.random() < (workspace_creativity * workspace_focus * 0.1)
                    
                    if synthesis_success:
                        access_layer.update_node_property(node_id, 'state', 'synthesizing')
                        access_layer.update_node_property(node_id, 'refractory_timer', 1.0 / workspace_creativity)  # Higher creativity = faster cycles
                        access_layer.update_node_property(node_id, 'last_activation', step)
                        
                        # Update statistics
                        self.behavior_stats['workspace_syntheses'] += 1
                    else:
                        access_layer.update_node_property(node_id, 'state', 'planning')
                else:
                    access_layer.update_node_property(node_id, 'state', 'imagining')
            elif membrane_potential > threshold * 0.8:
                access_layer.update_node_property(node_id, 'state', 'planning')
            elif membrane_potential > threshold * 0.5:
                access_layer.update_node_property(node_id, 'state', 'imagining')
            else:
                access_layer.update_node_property(node_id, 'state', 'active')
    
    def get_behavior_statistics(self) -> Dict[str, int]:
        """Get current behavior statistics for monitoring."""
        return self.behavior_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset behavior statistics."""
        self.behavior_stats = {
            'oscillator_activations': 0,
            'integrator_activations': 0,
            'relay_transfers': 0,
            'highway_regulations': 0,
            'workspace_syntheses': 0
        }


# Utility functions for behavior analysis
def should_transition_to_learning(node: Dict[str, Any]) -> bool:
    """
    Determine if a node should transition to learning state.
    
    Args:
        node: Node label dictionary
    
    Returns:
        True if node should transition to learning
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


def energy_above_threshold(node: Dict[str, Any]) -> bool:
    """
    Check if node energy is above its threshold.
    
    Args:
        node: Node label dictionary
    
    Returns:
        True if energy above threshold
    """
    energy = node.get('energy', 0.0)
    threshold = node.get('threshold', 0.5)
    return energy > threshold


def has_active_connections(node: Dict[str, Any], graph: Data) -> bool:
    """
    Check if node has active connections.
    
    Args:
        node: Node label dictionary
        graph: PyTorch Geometric graph
    
    Returns:
        True if node has active connections
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


