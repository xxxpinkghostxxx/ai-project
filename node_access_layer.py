"""
node_access_layer.py

ID-based node access layer for the ID-based node simulation.
Provides abstracted node access operations using unique IDs instead of array indices.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Iterator, Tuple
from logging_utils import log_step
from node_id_manager import get_id_manager


class NodeAccessLayer:
    """
    Abstracted layer for ID-based node access operations.
    
    Provides:
    - ID-based node lookup and iteration
    - ID-based node selection and filtering
    - Performance-optimized access patterns
    - Type-safe node operations
    """
    
    def __init__(self, graph):
        """
        Initialize the node access layer with a graph.
        
        Args:
            graph: The PyTorch Geometric graph to provide access to
        """
        self.graph = graph
        self.id_manager = get_id_manager()
        
        # Cache for performance optimization
        self._node_cache: Dict[int, Dict[str, Any]] = {}
        self._cache_valid = False
        
        log_step("NodeAccessLayer initialized", graph_nodes=len(graph.node_labels) if hasattr(graph, 'node_labels') else 0)
    
    def get_node_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a node by its unique ID.
        
        Args:
            node_id: The unique node ID
        
        Returns:
            dict: Node label dictionary if found, None otherwise
        """
        if not self.id_manager.is_valid_id(node_id):
            logging.warning(f"Invalid node ID: {node_id}")
            return None
        
        # Try cache first
        if self._cache_valid and node_id in self._node_cache:
            return self._node_cache[node_id]
        
        # Get index and retrieve node
        index = self.id_manager.get_node_index(node_id)
        if index is None:
            logging.warning(f"No index found for node ID: {node_id}")
            return None
        
        if not hasattr(self.graph, 'node_labels') or index >= len(self.graph.node_labels):
            logging.warning(f"Index {index} out of range for node ID: {node_id}")
            return None
        
        node = self.graph.node_labels[index]
        
        # Cache the result
        if self._cache_valid:
            self._node_cache[node_id] = node
        
        return node
    
    def get_node_energy(self, node_id: int) -> Optional[float]:
        """
        Get the energy value for a node by ID.
        
        Args:
            node_id: The unique node ID
        
        Returns:
            float: Energy value if found, None otherwise
        """
        index = self.id_manager.get_node_index(node_id)
        if index is None or not hasattr(self.graph, 'x') or index >= self.graph.x.shape[0]:
            return None
        
        return float(self.graph.x[index, 0].item())
    
    def set_node_energy(self, node_id: int, energy: float) -> bool:
        """
        Set the energy value for a node by ID.
        
        Args:
            node_id: The unique node ID
            energy: The energy value to set
        
        Returns:
            bool: True if successful, False otherwise
        """
        index = self.id_manager.get_node_index(node_id)
        if index is None or not hasattr(self.graph, 'x') or index >= self.graph.x.shape[0]:
            return False
        
        self.graph.x[index, 0] = energy
        
        # Invalidate cache
        self._cache_valid = False
        
        return True
    
    def update_node_property(self, node_id: int, property_name: str, value: Any) -> bool:
        """
        Update a property of a node by ID.
        
        Args:
            node_id: The unique node ID
            property_name: The property name to update
            value: The new value
        
        Returns:
            bool: True if successful, False otherwise
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            return False
        
        node[property_name] = value
        
        # Invalidate cache
        self._cache_valid = False
        
        return True
    
    def get_node_property(self, node_id: int, property_name: str, default: Any = None) -> Any:
        """
        Get a property of a node by ID.
        
        Args:
            node_id: The unique node ID
            property_name: The property name
            default: Default value if property not found
        
        Returns:
            Any: Property value or default
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            return default
        
        return node.get(property_name, default)
    
    def iterate_nodes_by_ids(self, node_ids: List[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Iterate over nodes by their IDs.
        
        Args:
            node_ids: List of node IDs to iterate over
        
        Yields:
            Tuple[int, Dict[str, Any]]: (node_id, node_data) pairs
        """
        for node_id in node_ids:
            node = self.get_node_by_id(node_id)
            if node is not None:
                yield node_id, node
    
    def iterate_all_nodes(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Iterate over all active nodes by ID.
        
        Yields:
            Tuple[int, Dict[str, Any]]: (node_id, node_data) pairs
        """
        active_ids = self.id_manager.get_all_active_ids()
        yield from self.iterate_nodes_by_ids(active_ids)
    
    def select_nodes_by_type(self, node_type: str) -> List[int]:
        """
        Select node IDs by node type.
        
        Args:
            node_type: The node type to filter by
        
        Returns:
            List[int]: List of node IDs of the specified type
        """
        return self.id_manager.get_ids_by_type(node_type)
    
    def select_nodes_by_property(self, property_name: str, property_value: Any) -> List[int]:
        """
        Select node IDs by property value.
        
        Args:
            property_name: The property name to filter by
            property_value: The property value to match
        
        Returns:
            List[int]: List of node IDs matching the criteria
        """
        matching_ids = []
        
        for node_id, node in self.iterate_all_nodes():
            if node.get(property_name) == property_value:
                matching_ids.append(node_id)
        
        return matching_ids
    
    def select_nodes_by_behavior(self, behavior: str) -> List[int]:
        """
        Select node IDs by behavior.
        
        Args:
            behavior: The behavior to filter by
        
        Returns:
            List[int]: List of node IDs with the specified behavior
        """
        return self.select_nodes_by_property('behavior', behavior)
    
    def select_nodes_by_state(self, state: str) -> List[int]:
        """
        Select node IDs by state.
        
        Args:
            state: The state to filter by
        
        Returns:
            List[int]: List of node IDs with the specified state
        """
        return self.select_nodes_by_property('state', state)
    
    def filter_nodes(self, filter_func: Callable[[int, Dict[str, Any]], bool]) -> List[int]:
        """
        Filter nodes using a custom function.
        
        Args:
            filter_func: Function that takes (node_id, node_data) and returns bool
        
        Returns:
            List[int]: List of node IDs that pass the filter
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
            int: Number of active nodes
        """
        return len(self.id_manager.get_all_active_ids())
    
    def get_node_count_by_type(self, node_type: str) -> int:
        """
        Get the number of nodes of a specific type.
        
        Args:
            node_type: The node type to count
        
        Returns:
            int: Number of nodes of the specified type
        """
        return len(self.select_nodes_by_type(node_type))
    
    def get_node_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the nodes.
        
        Returns:
            dict: Node statistics
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
            # Count by type
            node_type = node.get('type', 'unknown')
            stats['by_type'][node_type] = stats['by_type'].get(node_type, 0) + 1
            
            # Count by behavior
            behavior = node.get('behavior', 'unknown')
            stats['by_behavior'][behavior] = stats['by_behavior'].get(behavior, 0) + 1
            
            # Count by state
            state = node.get('state', 'unknown')
            stats['by_state'][state] = stats['by_state'].get(state, 0) + 1
            
            # Energy statistics
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
        Validate the consistency of the node access layer.
        
        Returns:
            dict: Validation results
        """
        return self.id_manager.validate_graph_consistency(self.graph)
    
    def invalidate_cache(self):
        """Invalidate the node cache."""
        self._cache_valid = False
        self._node_cache.clear()
    
    def rebuild_cache(self):
        """Rebuild the node cache for performance."""
        self._cache_valid = True
        self._node_cache.clear()
        
        for node_id, node in self.iterate_all_nodes():
            self._node_cache[node_id] = node
        
        log_step("Node cache rebuilt", cached_nodes=len(self._node_cache))


# Convenience functions for backward compatibility
def create_node_access_layer(graph) -> NodeAccessLayer:
    """Create a node access layer for a graph."""
    return NodeAccessLayer(graph)


def get_node_by_id(graph, node_id: int) -> Optional[Dict[str, Any]]:
    """Get a node by ID using the access layer."""
    access_layer = NodeAccessLayer(graph)
    return access_layer.get_node_by_id(node_id)


def select_nodes_by_type(graph, node_type: str) -> List[int]:
    """Select nodes by type using the access layer."""
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_type(node_type)


def select_nodes_by_behavior(graph, behavior: str) -> List[int]:
    """Select nodes by behavior using the access layer."""
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_behavior(behavior)


def select_nodes_by_state(graph, state: str) -> List[int]:
    """Select nodes by state using the access layer."""
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_state(state)
