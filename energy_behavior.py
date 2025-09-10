"""
energy_behavior.py

This module implements enhanced energy behavior for the energy-based neural system.
It includes membrane potential integration, refractory periods, plasticity gating,
and energy dynamics adapted from research concepts to work with the current framework.
"""

import time
import numpy as np
import torch
from logging_utils import log_step, log_node_state

# Import configuration manager
from config_manager import get_learning_config, get_system_constants
# Import energy constants to replace magic numbers
from energy_constants import (
    EnergyConstants, OscillatorConstants, IntegratorConstants, 
    RelayConstants, HighwayConstants
)

# Energy behavior constants with configuration fallbacks
# Removed get_energy_config() - using config_manager directly

def get_node_energy_cap():
    """Get node energy cap from configuration."""
    constants = get_system_constants()
    return constants.get('node_energy_cap', get_node_energy_cap())


def update_node_energy_with_learning(graph, node_id, delta_energy):
    """
    Enhanced energy update logic with research-inspired features.
    Implements membrane potential integration, refractory periods, and plasticity gating.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: Index of the node to update
        delta_energy: Change in energy to apply
    
    Returns:
        Modified graph with updated node energy
    """
    if node_id >= len(graph.node_labels):
        return graph
    
    node = graph.node_labels[node_id]
    current_energy = graph.x[node_id, 0].item()
    
    # Basic energy update
    new_energy = current_energy + delta_energy
    
    # Membrane potential integration (adapted to energy system)
    if 'membrane_potential' in node:
        membrane_pot = node['membrane_potential']
        config = get_learning_config()
        threshold = node.get('threshold', EnergyConstants.get_activation_threshold())
        
        if membrane_pot > threshold:
            # Node activates
            node['last_activation'] = time.time()
            node['refractory_timer'] = node.get('refractory_period', EnergyConstants.get_refractory_period())
            
            # Emit energy pulse if oscillator
            if node.get('behavior') == 'oscillator':
                emit_energy_pulse(graph, node_id)
            
            log_step("Node activated", 
                    node_id=node_id,
                    behavior=node.get('behavior', 'unknown'),
                    membrane_potential=membrane_pot,
                    threshold=threshold)
    
    # Refractory period enforcement
    if node.get('refractory_timer', 0) > 0:
        node['refractory_timer'] -= EnergyConstants.TIME_STEP
        # Prevent energy changes during refractory period
        new_energy = current_energy
        
        log_step("Refractory period active", 
                node_id=node_id,
                remaining_time=node['refractory_timer'])
    
    # Plasticity gating
    if not node.get('plasticity_enabled', True):
        # Skip learning-related updates
        log_step("Plasticity disabled", 
                node_id=node_id,
                reason="plasticity_gate_closed")
    
    # Update node energy with bounds
    new_energy = max(0, min(new_energy, get_node_energy_cap()))  # NODE_ENERGY_CAP
    graph.x[node_id, 0] = new_energy
    
    # Update membrane potential based on new energy
    if 'membrane_potential' in node:
        node['membrane_potential'] = min(new_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP)
    
    return graph


def apply_energy_behavior(graph, behavior_params=None):
    """
    Apply behavior-specific energy dynamics to all nodes in the graph.
    Uses ID-based node access instead of array enumeration.
    
    Args:
        graph: PyTorch Geometric graph
        behavior_params: Optional behavior parameters
    
    Returns:
        Modified graph with applied energy behaviors
    """
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return graph
    
    # Import ID-based access layer
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    
    # Iterate over all nodes using ID-based access
    for node_id, node in access_layer.iterate_all_nodes():
        behavior = node.get('behavior', 'dynamic')
        
        # Apply behavior-specific energy dynamics
        if behavior == 'oscillator':
            graph = apply_oscillator_energy_dynamics(graph, node_id)
        elif behavior == 'integrator':
            graph = apply_integrator_energy_dynamics(graph, node_id)
        elif behavior == 'relay':
            graph = apply_relay_energy_dynamics(graph, node_id)
        elif behavior == 'highway':
            graph = apply_highway_energy_dynamics(graph, node_id)
        else:
            # Default dynamic node behavior
            graph = apply_dynamic_energy_dynamics(graph, node_id)
    
    return graph


def apply_oscillator_energy_dynamics(graph, node_id):
    """
    Apply oscillator-specific energy dynamics.
    Oscillators emit periodic energy pulses.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: ID of the oscillator node
    
    Returns:
        Modified graph
    """
    # Import ID-based access layer
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    
    current_time = time.time()
    
    # Check if oscillator should emit pulse
    last_activation = node.get('last_activation', 0)
    oscillation_freq = node.get('oscillation_freq', OscillatorConstants.get_oscillation_frequency())  # Hz
    period = 1.0 / oscillation_freq
    
    if current_time - last_activation >= period:
        # Emit energy pulse
        pulse_energy = access_layer.get_node_energy(node_id) * OscillatorConstants.PULSE_ENERGY_FRACTION
        
        # Distribute pulse to connected nodes
        if hasattr(graph, 'edge_attributes'):
            for edge in graph.edge_attributes:
                if edge.source == node_id:  # Edge stores node IDs, not indices
                    target_id = edge.target
                    if access_layer.get_node_by_id(target_id) is not None:
                        # Transfer energy to target
                        target_energy = access_layer.get_node_energy(target_id)
                        if target_energy is not None:
                            new_target_energy = min(target_energy + pulse_energy, get_node_energy_cap())
                            access_layer.set_node_energy(target_id, new_target_energy)
                            
                            # Update target membrane potential
                            access_layer.update_node_property(target_id, 'membrane_potential', min(new_target_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
        
        # Update oscillator state
        access_layer.update_node_property(node_id, 'last_activation', current_time)
        access_layer.update_node_property(node_id, 'refractory_timer', OscillatorConstants.REFRACTORY_PERIOD_SHORT)
        
        log_step("Oscillator pulse emitted", 
                node_id=node_id,
                pulse_energy=pulse_energy,
                frequency=oscillation_freq)
    
    return graph


def apply_integrator_energy_dynamics(graph, node_id):
    """
    Apply integrator-specific energy dynamics.
    Integrators accumulate energy from inputs and activate when threshold is reached.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: ID of the integrator node
    
    Returns:
        Modified graph
    """
    # Import ID-based access layer
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    
    integration_rate = node.get('integration_rate', IntegratorConstants.get_integration_rate())
    
    # Accumulate energy from incoming connections
    if hasattr(graph, 'edge_attributes'):
        accumulated_energy = EnergyConstants.ELIGIBILITY_TRACE_DEFAULT
        for edge in graph.edge_attributes:
            if edge.target == node_id:  # Edge stores node IDs, not indices
                source_id = edge.source
                if access_layer.get_node_by_id(source_id) is not None:
                    source_energy = access_layer.get_node_energy(source_id)
                    if source_energy is not None:
                        # Weighted accumulation
                        accumulated_energy += source_energy * edge.weight * integration_rate
        
        if accumulated_energy > 0:
            # Update integrator energy
            current_energy = access_layer.get_node_energy(node_id)
            if current_energy is not None:
                new_energy = min(current_energy + accumulated_energy, get_node_energy_cap())
                access_layer.set_node_energy(node_id, new_energy)
                
                # Update membrane potential using centralized utility
                from node_property_utils import update_node_membrane_potential
                from node_id_manager import get_id_manager
                node_index = get_id_manager().get_node_index(node_id)
                if node_index is not None:
                    update_node_membrane_potential(graph, node_index, new_energy)
                
                log_step("Integrator accumulated energy", 
                        node_id=node_id,
                        accumulated=accumulated_energy,
                        new_total=new_energy)
    
    return graph


def apply_relay_energy_dynamics(graph, node_id):
    """
    Apply relay-specific energy dynamics.
    Relays transfer energy with amplification.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: ID of the relay node
    
    Returns:
        Modified graph
    """
    # Import ID-based access layer
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    
    relay_amplification = node.get('relay_amplification', RelayConstants.get_relay_amplification())
    
    # Transfer energy to outgoing connections with amplification
    if hasattr(graph, 'edge_attributes'):
        current_energy = access_layer.get_node_energy(node_id)
        if current_energy is not None:
            transfer_energy = current_energy * RelayConstants.ENERGY_TRANSFER_FRACTION
            
            for edge in graph.edge_attributes:
                if edge.source == node_id:  # Edge stores node IDs, not indices
                    target_id = edge.target
                    if access_layer.get_node_by_id(target_id) is not None:
                        # Amplified energy transfer
                        amplified_transfer = transfer_energy * relay_amplification
                        target_energy = access_layer.get_node_energy(target_id)
                        if target_energy is not None:
                            new_target_energy = min(target_energy + amplified_transfer, get_node_energy_cap())
                            access_layer.set_node_energy(target_id, new_target_energy)
                            
                            # Update target membrane potential
                            access_layer.update_node_property(target_id, 'membrane_potential', min(new_target_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
            
            # Reduce relay energy after transfer
            new_relay_energy = max(current_energy - transfer_energy, 0)
            access_layer.set_node_energy(node_id, new_relay_energy)
            
            # Update relay membrane potential
            access_layer.update_node_property(node_id, 'membrane_potential', min(new_relay_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
            
            log_step("Relay energy transfer", 
                    node_id=node_id,
                    transfer_energy=transfer_energy,
                    amplification=relay_amplification)
    
    return graph


def apply_highway_energy_dynamics(graph, node_id):
    """
    Apply highway-specific energy dynamics.
    Highways provide high-capacity energy distribution.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: ID of the highway node
    
    Returns:
        Modified graph
    """
    # Import ID-based access layer
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    
    highway_energy_boost = node.get('highway_energy_boost', HighwayConstants.get_highway_energy_boost())
    
    # Highway nodes maintain high energy levels and distribute efficiently
    current_energy = access_layer.get_node_energy(node_id)
    if current_energy is not None:
        if current_energy < HighwayConstants.ENERGY_THRESHOLD_LOW:
            boosted_energy = min(current_energy + HighwayConstants.ENERGY_BOOST_AMOUNT, get_node_energy_cap())
            access_layer.set_node_energy(node_id, boosted_energy)
            
            # Update membrane potential
            access_layer.update_node_property(node_id, 'membrane_potential', min(boosted_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
            
            log_step("Highway energy boosted", 
                    node_id=node_id,
                    old_energy=current_energy,
                    new_energy=boosted_energy)
        
        # Efficient energy distribution to multiple targets
        if hasattr(graph, 'edge_attributes'):
            distribution_count = 0
            for edge in graph.edge_attributes:
                if edge.source == node_id:  # Edge stores node IDs, not indices
                    target_id = edge.target
                    if access_layer.get_node_by_id(target_id) is not None:
                        # Efficient distribution with boost
                        distribution_energy = HighwayConstants.DISTRIBUTION_ENERGY_BASE * highway_energy_boost
                        target_energy = access_layer.get_node_energy(target_id)
                        if target_energy is not None:
                            new_target_energy = min(target_energy + distribution_energy, get_node_energy_cap())
                            access_layer.set_node_energy(target_id, new_target_energy)
                            
                            # Update target membrane potential
                            access_layer.update_node_property(target_id, 'membrane_potential', min(new_target_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
                            
                            distribution_count += 1
            
            if distribution_count > 0:
                log_step("Highway energy distribution", 
                        node_id=node_id,
                        targets=distribution_count,
                        energy_per_target=10.0 * highway_energy_boost)
    
    return graph


def apply_dynamic_energy_dynamics(graph, node_id):
    """
    Apply default dynamic node energy dynamics.
    Basic energy management with decay and plasticity.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: ID of the dynamic node
    
    Returns:
        Modified graph
    """
    # Import ID-based access layer
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    
    current_energy = access_layer.get_node_energy(node_id)
    if current_energy is None:
        return graph
    
    # Energy decay over time
    config = get_learning_config()
    decay_rate = EnergyConstants.get_decay_rate()
    decayed_energy = current_energy * decay_rate
    
    # Apply decay
    new_energy = max(current_energy - decayed_energy, 0)
    access_layer.set_node_energy(node_id, new_energy)
    
    # Update membrane potential
    access_layer.update_node_property(node_id, 'membrane_potential', min(new_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
    
    # Update plasticity state based on energy
    if new_energy < EnergyConstants.get_plasticity_threshold():
        access_layer.update_node_property(node_id, 'plasticity_enabled', False)
        log_step("Plasticity disabled", 
                node_id=node_id,
                reason="low_energy",
                energy=new_energy)
    else:
        access_layer.update_node_property(node_id, 'plasticity_enabled', True)
    
    return graph


def emit_energy_pulse(graph, source_node_idx):
    """
    Emit an energy pulse from a source node to connected targets.
    
    Args:
        graph: PyTorch Geometric graph
        source_node_idx: Index of the source node
    
    Returns:
        Modified graph
    """
    if not hasattr(graph, 'edge_attributes'):
        return graph
    
    source_energy = graph.x[source_node_idx, 0].item()
    pulse_energy = source_energy * EnergyConstants.PULSE_ENERGY_FRACTION_LARGE
    
    pulse_targets = []
    
    for edge in graph.edge_attributes:
        if edge.source == source_node_idx:
            target_idx = edge.target
            if target_idx < len(graph.node_labels):
                # Transfer energy to target
                target_energy = graph.x[target_idx, 0].item()
                new_target_energy = min(target_energy + pulse_energy, get_node_energy_cap())
                graph.x[target_idx, 0] = new_target_energy
                
                # Update target membrane potential
                target_node = graph.node_labels[target_idx]
                if 'membrane_potential' in target_node:
                    target_node['membrane_potential'] = min(new_target_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP)
                
                pulse_targets.append(target_idx)
    
    if pulse_targets:
        log_step("Energy pulse emitted", 
                source_node=source_node_idx,
                pulse_energy=pulse_energy,
                targets=pulse_targets)
    
    return graph


def update_membrane_potentials(graph):
    """
    Update membrane potentials for all nodes based on current energy levels.
    
    Args:
        graph: PyTorch Geometric graph
    
    Returns:
        Modified graph with updated membrane potentials
    """
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return graph
    
    for node_idx, node in enumerate(graph.node_labels):
        if 'membrane_potential' in node:
            current_energy = graph.x[node_idx, 0].item()
            # Normalize energy to 0-1 range for membrane potential
            membrane_potential = min(current_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP)
            node['membrane_potential'] = membrane_potential
    
    return graph


def apply_refractory_periods(graph):
    """
    Apply refractory periods to all nodes.
    
    Args:
        graph: PyTorch Geometric graph
    
    Returns:
        Modified graph with updated refractory timers
    """
    if not hasattr(graph, 'node_labels'):
        return graph
    
    current_time = time.time()
    
    for node_idx, node in enumerate(graph.node_labels):
        if 'refractory_timer' in node:
            refractory_timer = node.get('refractory_timer', 0)
            
            if refractory_timer > 0:
                # Decrement refractory timer
                node['refractory_timer'] = max(refractory_timer - EnergyConstants.TIME_STEP, 0)
                
                # Reset membrane potential during refractory period
                if 'membrane_potential' in node:
                    node['membrane_potential'] = EnergyConstants.MEMBRANE_POTENTIAL_RESET
    
    return graph


def couple_sensory_energy_to_channel(graph):
    """
    Couple sensory node energy to the energy channel system.
    This function maintains backward compatibility.
    
    Args:
        graph: PyTorch Geometric graph
    
    Returns:
        Modified graph
    """
    # This function is maintained for backward compatibility
    # The actual energy coupling is now handled by the enhanced energy behavior system
    return graph


def propagate_sensory_energy(graph):
    """
    Propagate sensory energy through the network.
    This function maintains backward compatibility.
    
    Args:
        graph: PyTorch Geometric graph
    
    Returns:
        Modified graph
    """
    # This function is maintained for backward compatibility
    # The actual energy propagation is now handled by the enhanced energy behavior system
    return graph


