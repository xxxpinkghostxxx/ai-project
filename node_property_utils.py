"""
node_property_utils.py

Centralized node property management utilities to eliminate code duplication.
Provides consistent node property updates, state management, and property validation.
"""

from typing import Dict, Any, Optional, List, Union
from torch_geometric.data import Data
from logging_utils import log_step


def update_node_state(graph: Data, node_id: int, new_state: str) -> bool:
    """
    Update node state with validation.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: Node index
        new_state: New state value
        
    Returns:
        True if successful, False otherwise
    """
    valid_states = ['active', 'inactive', 'consolidating', 'synthesizing', 'planning', 'imagining', 'regulating', 'pending']
    
    if new_state not in valid_states:
        log_step(f"Invalid node state: {new_state}", node_id=node_id)
        return False
    
    return update_node_property_safe(graph, node_id, 'state', new_state)


def update_node_membrane_potential(graph: Data, node_id: int, energy: float) -> bool:
    """
    Update node membrane potential based on energy.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: Node index
        energy: Energy value
        
    Returns:
        True if successful, False otherwise
    """
    from energy_utils import calculate_membrane_potential
    membrane_potential = calculate_membrane_potential(energy)
    return update_node_property_safe(graph, node_id, 'membrane_potential', membrane_potential)


def update_node_property_safe(graph: Data, node_id: int, property_name: str, value: Any) -> bool:
    """
    Safely update a node property with bounds checking.
    
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
    
    # Validate property value based on property name
    if not _validate_property_value(property_name, value):
        log_step(f"Invalid property value for {property_name}: {value}", node_id=node_id)
        return False
    
    graph.node_labels[node_id][property_name] = value
    return True


def get_node_property_safe(graph: Data, node_id: int, property_name: str, default: Any = None) -> Any:
    """
    Safely get a node property with error handling.
    
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


def _validate_property_value(property_name: str, value: Any) -> bool:
    """
    Validate property value based on property name.
    
    Args:
        property_name: Name of the property
        value: Value to validate
        
    Returns:
        True if valid, False otherwise
    """
    if property_name == 'state':
        valid_states = ['active', 'inactive', 'consolidating', 'synthesizing', 'planning', 'imagining', 'regulating', 'pending']
        return value in valid_states
    
    elif property_name == 'membrane_potential':
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0
    
    elif property_name == 'threshold':
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0
    
    elif property_name == 'refractory_timer':
        return isinstance(value, (int, float)) and value >= 0.0
    
    elif property_name == 'last_activation':
        return isinstance(value, (int, float)) and value >= 0.0
    
    elif property_name == 'last_update':
        return isinstance(value, (int, float)) and value >= 0.0
    
    elif property_name == 'plasticity_enabled':
        return isinstance(value, bool)
    
    elif property_name == 'eligibility_trace':
        return isinstance(value, (int, float)) and value >= 0.0
    
    elif property_name == 'energy':
        return isinstance(value, (int, float)) and value >= 0.0
    
    elif property_name == 'behavior':
        valid_behaviors = ['sensory', 'dynamic', 'oscillator', 'integrator', 'relay', 'highway', 'workspace']
        return value in valid_behaviors
    
    elif property_name == 'type':
        valid_types = ['sensory', 'dynamic', 'oscillator', 'integrator', 'relay', 'highway', 'workspace']
        return value in valid_types
    
    # For other properties, allow any value
    return True


def batch_update_properties(graph: Data, node_indices: List[int], 
                          property_updates: List[Dict[str, Any]]) -> bool:
    """
    Batch update multiple node properties efficiently.
    
    Args:
        graph: PyTorch Geometric graph
        node_indices: List of node indices to update
        property_updates: List of property dictionaries to apply
        
    Returns:
        True if all updates successful, False otherwise
    """
    if len(node_indices) != len(property_updates):
        return False
    
    success_count = 0
    
    for node_idx, properties in zip(node_indices, property_updates):
        if node_idx >= len(graph.node_labels):
            continue
        
        success = True
        for prop_name, prop_value in properties.items():
            if not update_node_property_safe(graph, node_idx, prop_name, prop_value):
                success = False
                break
        
        if success:
            success_count += 1
    
    return success_count == len(node_indices)


def get_node_statistics(graph: Data) -> Dict[str, Any]:
    """
    Get comprehensive node statistics.
    
    Args:
        graph: PyTorch Geometric graph
        
    Returns:
        Dictionary containing node statistics
    """
    if not hasattr(graph, 'node_labels'):
        return {
            'total_nodes': 0,
            'node_types': {},
            'node_states': {},
            'node_behaviors': {},
            'active_nodes': 0,
            'inactive_nodes': 0
        }
    
    total_nodes = len(graph.node_labels)
    node_types = {}
    node_states = {}
    node_behaviors = {}
    active_nodes = 0
    inactive_nodes = 0
    
    for node in graph.node_labels:
        # Count by type
        node_type = node.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count by state
        node_state = node.get('state', 'unknown')
        node_states[node_state] = node_states.get(node_state, 0) + 1
        
        # Count by behavior
        node_behavior = node.get('behavior', 'unknown')
        node_behaviors[node_behavior] = node_behaviors.get(node_behavior, 0) + 1
        
        # Count active/inactive
        if node_state == 'active':
            active_nodes += 1
        elif node_state == 'inactive':
            inactive_nodes += 1
    
    return {
        'total_nodes': total_nodes,
        'node_types': node_types,
        'node_states': node_states,
        'node_behaviors': node_behaviors,
        'active_nodes': active_nodes,
        'inactive_nodes': inactive_nodes
    }


def validate_node_consistency(graph: Data) -> Dict[str, Any]:
    """
    Validate node consistency and data integrity.
    
    Args:
        graph: PyTorch Geometric graph
        
    Returns:
        Dictionary containing validation results
    """
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return {
            'valid': False,
            'errors': ['Missing node_labels or x tensor'],
            'warnings': []
        }
    
    errors = []
    warnings = []
    
    # Check node count consistency
    if len(graph.node_labels) != graph.x.shape[0]:
        errors.append(f"Node count mismatch: {len(graph.node_labels)} labels vs {graph.x.shape[0]} features")
    
    # Check required properties
    required_properties = ['type', 'behavior', 'state', 'energy']
    
    for i, node in enumerate(graph.node_labels):
        for prop in required_properties:
            if prop not in node:
                errors.append(f"Node {i} missing required property: {prop}")
        
        # Validate property values
        for prop_name, prop_value in node.items():
            if not _validate_property_value(prop_name, prop_value):
                warnings.append(f"Node {i} has invalid {prop_name}: {prop_value}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def reset_node_properties(graph: Data, node_id: int, reset_type: str = 'full') -> bool:
    """
    Reset node properties to default values.
    
    Args:
        graph: PyTorch Geometric graph
        node_id: Node index
        reset_type: Type of reset ('full', 'state', 'timing')
        
    Returns:
        True if successful, False otherwise
    """
    if not hasattr(graph, 'node_labels') or node_id >= len(graph.node_labels):
        return False
    
    node = graph.node_labels[node_id]
    
    if reset_type == 'full':
        # Reset all properties to defaults
        node.update({
            'state': 'inactive',
            'membrane_potential': 0.0,
            'threshold': 0.3,
            'refractory_timer': 0.0,
            'last_activation': 0,
            'last_update': 0,
            'plasticity_enabled': True,
            'eligibility_trace': 0.0
        })
    elif reset_type == 'state':
        # Reset only state-related properties
        node.update({
            'state': 'inactive',
            'membrane_potential': 0.0,
            'refractory_timer': 0.0
        })
    elif reset_type == 'timing':
        # Reset only timing-related properties
        node.update({
            'last_activation': 0,
            'last_update': 0,
            'refractory_timer': 0.0
        })
    
    return True


def copy_node_properties(source_graph: Data, source_node_id: int, 
                        target_graph: Data, target_node_id: int,
                        properties: Optional[List[str]] = None) -> bool:
    """
    Copy node properties from one node to another.
    
    Args:
        source_graph: Source graph
        source_node_id: Source node index
        target_graph: Target graph
        target_node_id: Target node index
        properties: List of properties to copy (None for all)
        
    Returns:
        True if successful, False otherwise
    """
    if not hasattr(source_graph, 'node_labels') or source_node_id >= len(source_graph.node_labels):
        return False
    
    if not hasattr(target_graph, 'node_labels') or target_node_id >= len(target_graph.node_labels):
        return False
    
    source_node = source_graph.node_labels[source_node_id]
    
    if properties is None:
        properties = list(source_node.keys())
    
    for prop_name in properties:
        if prop_name in source_node:
            value = source_node[prop_name]
            if not update_node_property_safe(target_graph, target_node_id, prop_name, value):
                return False
    
    return True
