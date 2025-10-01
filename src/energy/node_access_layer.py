"""
Node Access Layer module for managing neural graph nodes.
This module provides a NodeAccessLayer class and utility functions for accessing and manipulating nodes in a neural graph, including energy management and property updates.
"""
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import numpy as np
from src.energy.energy_constants import get_node_energy_cap
from src.energy.node_id_manager import get_id_manager
from src.utils.logging_utils import log_step
def _update_energy(graph, index, node_id, new_value):
    if not hasattr(graph, 'x') or graph.x is None or index >= graph.x.shape[0]:
        logging.warning("Graph missing valid 'x' attribute or index %s out of range for node %s", index, node_id)
        return False
    energy_cap = get_node_energy_cap()
    clamped_value = max(0.0, min(float(new_value), energy_cap))
    graph.x[index, 0] = clamped_value
    log_step(f"Updated energy for node {node_id} to {clamped_value} (clamped from {new_value})",
             extra={"node_id": node_id, "property": "energy", "old_value": None, "new_value": clamped_value})
    return True
def _update_property(graph, index, node_id, property_name, new_value):
    if not hasattr(graph, 'node_labels') or graph.node_labels is None or index >= len(graph.node_labels):
        logging.warning("Graph missing valid 'node_labels' attribute or index %s out of range for node %s", index, node_id)
        return False
    node = graph.node_labels[index]
    if not isinstance(node, dict):
        logging.warning("Node labels for node %s at index %s is not a dict", node_id, index)
        return False
    old_value = node.get(property_name, None)
    node[property_name] = new_value
    log_step(f"Updated property '{property_name}' for node {node_id} from {old_value} to {new_value}",
             extra={"node_id": node_id, "property": property_name, "old_value": old_value, "new_value": new_value})
    return True
class NodeAccessLayer:
    """
    Provides access to neural graph nodes with caching and validation.
    This class handles node retrieval, energy management, and property updates for neural graphs.
    """
    def __init__(self, graph, id_manager=None):
        """
        Initialize the NodeAccessLayer.
        Args:
            graph: Neural graph
            id_manager: Node ID manager (optional, will use global if not provided)
        """
        self.graph = graph
        if id_manager is None:
            self.id_manager = get_id_manager()
        else:
            self.id_manager = id_manager
        self._node_cache: Dict[int, Dict[str, Any]] = {}
        self._cache_valid = False
        self._invalid_id_count = 0
        log_step("NodeAccessLayer initialized", graph_nodes=len(graph.node_labels) if hasattr(graph, 'node_labels') else 0)
    def get_node_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its ID.
        Args:
            node_id: The ID of the node to retrieve.
        Returns:
            The node dictionary if found, None otherwise.
        """
        if not isinstance(node_id, (int, np.integer)):
            logging.warning("Invalid node ID type: %s", type(node_id))
            return None
        if hasattr(node_id, 'item'):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)
        if not hasattr(self.graph, 'node_labels') or not self.graph.node_labels:
            logging.warning("Graph has no node_labels")
            return None
        index = self.id_manager.get_node_index(node_id)
        if index is None:
            self._invalid_id_count += 1
            if self._invalid_id_count % 1000 == 0:
                logging.warning("Invalid node ID: %s (not found in ID manager) - %s total invalid IDs", node_id, self._invalid_id_count)
            return None
        # Safety check
        if index < 0 or index >= len(self.graph.node_labels):
            logging.warning("Index %s out of range for node ID: %s (max index: %s)", index, node_id, len(self.graph.node_labels)-1)
            return None
        if self._cache_valid and node_id in self._node_cache:
            return self._node_cache[node_id]
        node = self.graph.node_labels[index]
        if not self._cache_valid:
            self._cache_valid = True
            self._node_cache.clear()
        self._node_cache[node_id] = node
        return node
    def get_node_energy(self, node_id: int) -> Optional[float]:
        """
        Get the energy value of a node.
        Args:
            node_id: The ID of the node.
        Returns:
            The energy value if found, None otherwise.
        """
        if hasattr(node_id, 'item'):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)
        index = self.id_manager.get_node_index(node_id)
        if index is None or not hasattr(self.graph, 'x') or index >= self.graph.x.shape[0]:
            return None
        return float(self.graph.x[index, 0].item())
    def set_node_energy(self, node_id: int, energy: float) -> bool:
        """
        Set the energy value of a node.
        Args:
            node_id: The ID of the node.
            energy: The energy value to set.
        Returns:
            True if successful, False otherwise.
        """
        if hasattr(node_id, 'item'):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)
        index = self.id_manager.get_node_index(node_id)
        if index is None or not hasattr(self.graph, 'x') or index >= self.graph.x.shape[0]:
            return False
        self.graph.x[index, 0] = energy
        self._cache_valid = False
        return True
    def update_node_property(self, node_id: int, property_name: str, value: Any) -> bool:
        """
        Update a property of a node.
        Args:
            node_id: The ID of the node.
            property_name: The name of the property.
            value: The value to set.
        Returns:
            True if successful, False otherwise.
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            return False
        node[property_name] = value
        self._cache_valid = False
        return True
    def get_node_property(self, node_id: int, property_name: str, default: Any = None) -> Any:
        """
        Get a property value of a node.
        Args:
            node_id: The ID of the node.
            property_name: The name of the property.
            default: Default value if not found.
        Returns:
            The property value or default.
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            return default
        return node.get(property_name, default)
    def iterate_nodes_by_ids(self, node_ids: List[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Iterate over nodes by their IDs.
        Args:
            node_ids: List of node IDs.
        Yields:
            Tuple of node_id and node dict.
        """
        for node_id in node_ids:
            node = self.get_node_by_id(node_id)
            if node is not None:
                yield node_id, node
    def iterate_all_nodes(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Iterate over all active nodes.
        Yields:
            Tuple of node_id and node dict.
        """
        active_ids = self.id_manager.get_all_active_ids()
        yield from self.iterate_nodes_by_ids(active_ids)
    def select_nodes_by_type(self, node_type: str) -> List[int]:
        """
        Select nodes by type.
        Args:
            node_type: The type of nodes to select.
        Returns:
            List of node IDs.
        """
        return self.id_manager.get_ids_by_type(node_type)
    def select_nodes_by_property(self, property_name: str, property_value: Any) -> List[int]:
        """
        Select nodes by a property value.
        Args:
            property_name: The name of the property.
            property_value: The value to match.
        Returns:
            List of node IDs.
        """
        matching_ids = []
        for node_id, node in self.iterate_all_nodes():
            if node.get(property_name) == property_value:
                matching_ids.append(node_id)
        return matching_ids
    def select_nodes_by_behavior(self, behavior: str) -> List[int]:
        """
        Select nodes by behavior.
        Args:
            behavior: The behavior to match.
        Returns:
            List of node IDs.
        """
        return self.select_nodes_by_property('behavior', behavior)
    def select_nodes_by_state(self, state: str) -> List[int]:
        """
        Select nodes by state.
        Args:
            state: The state to match.
        Returns:
            List of node IDs.
        """
        return self.select_nodes_by_property('state', state)
    def filter_nodes(self, filter_func: Callable[[int, Dict[str, Any]], bool]) -> List[int]:
        """
        Filter nodes using a function.
        Args:
            filter_func: Function that takes node_id and node, returns bool.
        Returns:
            List of node IDs that match.
        """
        matching_ids = []
        for node_id, node in self.iterate_all_nodes():
            if filter_func(node_id, node):
                matching_ids.append(node_id)
        return matching_ids
    def get_node_count(self) -> int:
        """
        Get the total number of active nodes.
        Returns:
            Number of active nodes.
        """
        return len(self.id_manager.get_all_active_ids())
    def is_valid_node_id(self, node_id: int) -> bool:
        """
        Check if a node ID is valid.
        Args:
            node_id: The node ID to check.
        Returns:
            True if valid, False otherwise.
        """
        try:
            if node_id is None or not isinstance(node_id, int):
                return False
            return self.id_manager.is_valid_id(node_id)
        except (TypeError, ValueError, AttributeError):
            return False
    def get_node_count_by_type(self, node_type: str) -> int:
        """
        Get the number of nodes of a specific type.
        Args:
            node_type: The type of nodes.
        Returns:
            Number of nodes of that type.
        """
        return len(self.select_nodes_by_type(node_type))
    def get_node_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the nodes.
        Returns:
            Dictionary with statistics.
        """
        stats = {
            'total_nodes': self.get_node_count(),
            'by_type': {},
            'by_behavior': {},
            'by_state': {},
            'energy_stats': {
                'total_energy': 0.0,
                'average_energy': 0.0,
                'min_energy': float('inf'),
                'max_energy': 0.0
            }
        }
        total_energy = 0.0
        energy_count = 0
        min_energy = float('inf')
        max_energy = 0.0
        for node_id, node in self.iterate_all_nodes():
            node_type = node.get('type', 'unknown')
            stats['by_type'][node_type] = stats['by_type'].get(node_type, 0) + 1
            behavior = node.get('behavior', 'unknown')
            stats['by_behavior'][behavior] = stats['by_behavior'].get(behavior, 0) + 1
            state = node.get('state', 'unknown')
            stats['by_state'][state] = stats['by_state'].get(state, 0) + 1
            energy = self.get_node_energy(node_id)
            if energy is not None:
                total_energy += energy
                energy_count += 1
                min_energy = min(min_energy, energy)
                max_energy = max(max_energy, energy)
        if energy_count > 0:
            stats['energy_stats']['total_energy'] = total_energy
            stats['energy_stats']['average_energy'] = total_energy / energy_count
            stats['energy_stats']['min_energy'] = min_energy
            stats['energy_stats']['max_energy'] = max_energy
        return stats
    def validate_consistency(self) -> Dict[str, Any]:
        """
        Validate graph consistency.
        Returns:
            Dictionary with validation results.
        """
        return self.id_manager.validate_graph_consistency(self.graph)
    def invalidate_cache(self):
        """
        Invalidate the node cache.
        """
        self._cache_valid = False
        self._node_cache.clear()
    def rebuild_cache(self):
        """
        Rebuild the node cache.
        """
        self._cache_valid = True
        self._node_cache.clear()
        for node_id, node in self.iterate_all_nodes():
            self._node_cache[node_id] = node
        log_step("Node cache rebuilt", cached_nodes=len(self._node_cache))
def create_node_access_layer(graph) -> NodeAccessLayer:
    """
    Create a NodeAccessLayer instance.
    Args:
        graph: The neural graph.
    Returns:
        NodeAccessLayer instance.
    """
    return NodeAccessLayer(graph)
def get_node_by_id(graph, node_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a node by ID using a temporary access layer.
    Args:
        graph: The neural graph.
        node_id: The node ID.
    Returns:
        The node dict or None.
    """
    access_layer = NodeAccessLayer(graph)
    return access_layer.get_node_by_id(node_id)
def select_nodes_by_type(graph, node_type: str) -> List[int]:
    """
    Select nodes by type using a temporary access layer.
def _update_energy(graph, index, node_id, new_value):
    if not hasattr(graph, 'x') or graph.x is None or index >= graph.x.shape[0]:
        logging.warning("Graph missing valid 'x' attribute or index %s out of range for node %s", index, node_id)
        return False
    energy_cap = get_node_energy_cap()
    clamped_value = max(0.0, min(float(new_value), energy_cap))
    graph.x[index, 0] = clamped_value
    log_step(f"Updated energy for node {node_id} to {clamped_value} (clamped from {new_value})",
             extra={"node_id": node_id, "property": "energy", "old_value": None, "new_value": clamped_value})
    return True
def _update_property(graph, index, node_id, property_name, new_value):
    if not hasattr(graph, 'node_labels') or graph.node_labels is None or index >= len(graph.node_labels):
        logging.warning("Graph missing valid 'node_labels' attribute or index %s out of range for node %s", index, node_id)
        return False
    node = graph.node_labels[index]
    if not isinstance(node, dict):
        logging.warning("Node labels for node %s at index %s is not a dict", node_id, index)
        return False
    old_value = node.get(property_name, None)
    node[property_name] = new_value
    log_step(f"Updated property '{property_name}' for node {node_id} from {old_value} to {new_value}",
             extra={"node_id": node_id, "property": property_name, "old_value": old_value, "new_value": new_value})
    return True
    Args:
        graph: The neural graph.
        node_type: The type.
    Returns:
        List of node IDs.
    """
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_type(node_type)
def select_nodes_by_behavior(graph, behavior: str) -> List[int]:
    """
    Select nodes by behavior using a temporary access layer.
    Args:
        graph: The neural graph.
        behavior: The behavior.
    Returns:
        List of node IDs.
    """
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_behavior(behavior)
def select_nodes_by_state(_graph, state: str) -> List[int]:
    """
    Select nodes by state using a temporary access layer.
    Args:
        _graph: The neural graph.
        state: The state.
    Returns:
        List of node IDs.
    """
    access_layer = NodeAccessLayer(_graph)
    return access_layer.select_nodes_by_state(state)
def update_node_property(graph, node_id: int, property_name: str, new_value: Any, id_manager=None) -> bool:
    """
    Safely update a property for a specific node in the graph.
    For 'energy', updates graph.x[node_id_index, 0] and clamps to [0, get_node_energy_cap()].
    For other properties (e.g., 'type', 'behavior'), updates graph.node_labels[node_id_index][property_name].
    Handles missing attributes and invalid inputs with logging.
    Args:
        graph: Neural graph
        node_id: Node ID to update
        property_name: Property name to update
        new_value: New value for the property
        id_manager: Node ID manager (optional, will use global if not provided)
    """
    if id_manager is None:
        id_manager = get_id_manager()
    success = False
    try:
        # Convert node_id to int
        if hasattr(node_id, 'item'):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)
        # Validate node_id
        if not id_manager.is_valid_id(node_id):
            logging.warning("Invalid node ID: %s", node_id)
            return False
        index = id_manager.get_node_index(node_id)
        if index is None:
            logging.warning("Could not retrieve index for node_id %s", node_id)
            return False
        if property_name == 'energy':
            success = _update_energy(graph, index, node_id, new_value)
        else:
            success = _update_property(graph, index, node_id, property_name, new_value)
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        logging.error("Error in update_node_property for node %s, property '%s': %s", node_id, property_name, str(e))
        log_step("Failed to update property '%s' for node %s due to error: %s",
                 extra={"node_id": node_id, "property": property_name, "error": str(e)})
        success = False
    return success
