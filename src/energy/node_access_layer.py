
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Iterator, Tuple
from src.utils.logging_utils import log_step
from src.energy.node_id_manager import get_id_manager
from src.energy.energy_behavior import get_node_energy_cap


class NodeAccessLayer:

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
        log_step("NodeAccessLayer initialized", graph_nodes=len(graph.node_labels) if hasattr(graph, 'node_labels') else 0)
    def get_node_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
 
        if not isinstance(node_id, (int, np.integer)):
            logging.warning(f"Invalid node ID type: {type(node_id)}")
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
            if not hasattr(self, '_invalid_id_count'):
                self._invalid_id_count = 0
            self._invalid_id_count += 1
            if self._invalid_id_count % 1000 == 0:
                logging.warning(f"Invalid node ID: {node_id} (not found in ID manager) - {self._invalid_id_count} total invalid IDs")
            return None
        
        # Safety check
        if index < 0 or index >= len(self.graph.node_labels):
            logging.warning(f"Index {index} out of range for node ID: {node_id} (max index: {len(self.graph.node_labels)-1})")
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

        if hasattr(node_id, 'item'):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)
        index = self.id_manager.get_node_index(node_id)
        if index is None or not hasattr(self.graph, 'x') or index >= self.graph.x.shape[0]:
            return None
        return float(self.graph.x[index, 0].item())
    def set_node_energy(self, node_id: int, energy: float) -> bool:

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

        node = self.get_node_by_id(node_id)
        if node is None:
            return False
        node[property_name] = value
        self._cache_valid = False
        return True
    def get_node_property(self, node_id: int, property_name: str, default: Any = None) -> Any:

        node = self.get_node_by_id(node_id)
        if node is None:
            return default
        return node.get(property_name, default)
    def iterate_nodes_by_ids(self, node_ids: List[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:

        for node_id in node_ids:
            node = self.get_node_by_id(node_id)
            if node is not None:
                yield node_id, node
    def iterate_all_nodes(self) -> Iterator[Tuple[int, Dict[str, Any]]]:

        active_ids = self.id_manager.get_all_active_ids()
        yield from self.iterate_nodes_by_ids(active_ids)
    def select_nodes_by_type(self, node_type: str) -> List[int]:

        return self.id_manager.get_ids_by_type(node_type)
    def select_nodes_by_property(self, property_name: str, property_value: Any) -> List[int]:

        matching_ids = []
        for node_id, node in self.iterate_all_nodes():
            if node.get(property_name) == property_value:
                matching_ids.append(node_id)
        return matching_ids
    def select_nodes_by_behavior(self, behavior: str) -> List[int]:

        return self.select_nodes_by_property('behavior', behavior)
    def select_nodes_by_state(self, state: str) -> List[int]:

        return self.select_nodes_by_property('state', state)
    def filter_nodes(self, filter_func: Callable[[int, Dict[str, Any]], bool]) -> List[int]:

        matching_ids = []
        for node_id, node in self.iterate_all_nodes():
            if filter_func(node_id, node):
                matching_ids.append(node_id)
        return matching_ids
    def get_node_count(self) -> int:

        return len(self.id_manager.get_all_active_ids())
    def is_valid_node_id(self, node_id: int) -> bool:
        try:
            if node_id is None or not isinstance(node_id, int):
                return False
            return self.id_manager.is_valid_id(node_id)
        except (TypeError, ValueError, AttributeError):
            return False
    def get_node_count_by_type(self, node_type: str) -> int:

        return len(self.select_nodes_by_type(node_type))
    def get_node_statistics(self) -> Dict[str, Any]:

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

        return self.id_manager.validate_graph_consistency(self.graph)
    def invalidate_cache(self):
        self._cache_valid = False
        self._node_cache.clear()
    def rebuild_cache(self):
        self._cache_valid = True
        self._node_cache.clear()
        for node_id, node in self.iterate_all_nodes():
            self._node_cache[node_id] = node
        log_step("Node cache rebuilt", cached_nodes=len(self._node_cache))


def create_node_access_layer(graph) -> NodeAccessLayer:
    return NodeAccessLayer(graph)


def get_node_by_id(graph, node_id: int) -> Optional[Dict[str, Any]]:
    access_layer = NodeAccessLayer(graph)
    return access_layer.get_node_by_id(node_id)


def select_nodes_by_type(graph, node_type: str) -> List[int]:
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_type(node_type)


def select_nodes_by_behavior(graph, behavior: str) -> List[int]:
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_behavior(behavior)


def select_nodes_by_state(graph, state: str) -> List[int]:
    access_layer = NodeAccessLayer(graph)
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
    from src.utils.logging_utils import log_step
    import logging

    if id_manager is None:
        id_manager = get_id_manager()
    
    try:
        # Convert node_id to int
        if hasattr(node_id, 'item'):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)
        
        # Validate node_id
        if not id_manager.is_valid_id(node_id):
            logging.warning(f"Invalid node ID: {node_id}")
            return False
        
        index = id_manager.get_node_index(node_id)
        if index is None:
            logging.warning(f"Could not retrieve index for node_id {node_id}")
            return False
        
        if property_name == 'energy':
            # Handle energy update in graph.x
            if not hasattr(graph, 'x') or graph.x is None or index >= graph.x.shape[0]:
                logging.warning(f"Graph missing valid 'x' attribute or index {index} out of range for node {node_id}")
                return False
            
            energy_cap = get_node_energy_cap()
            clamped_value = max(0.0, min(float(new_value), energy_cap))
            graph.x[index, 0] = clamped_value
            log_step(f"Updated energy for node {node_id} to {clamped_value} (clamped from {new_value})",
                     extra={"node_id": node_id, "property": property_name, "old_value": None, "new_value": clamped_value})
            return True
        else:
            # Handle other properties in node_labels
            if not hasattr(graph, 'node_labels') or graph.node_labels is None or index >= len(graph.node_labels):
                logging.warning(f"Graph missing valid 'node_labels' attribute or index {index} out of range for node {node_id}")
                return False
            
            node = graph.node_labels[index]
            if not isinstance(node, dict):
                logging.warning(f"Node labels for node {node_id} at index {index} is not a dict")
                return False
            
            old_value = node.get(property_name, None)
            node[property_name] = new_value
            log_step(f"Updated property '{property_name}' for node {node_id} from {old_value} to {new_value}",
                     extra={"node_id": node_id, "property": property_name, "old_value": old_value, "new_value": new_value})
            return True
    
    except Exception as e:
        logging.error(f"Error in update_node_property for node {node_id}, property '{property_name}': {str(e)}")
        log_step(f"Failed to update property '{property_name}' for node {node_id} due to error: {str(e)}",
                 extra={"node_id": node_id, "property": property_name, "error": str(e)})
        return False







