
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Iterator, Tuple
from utils.logging_utils import log_step
from energy.node_id_manager import get_id_manager


class NodeAccessLayer:

    def __init__(self, graph):

        self.graph = graph
        self.id_manager = get_id_manager()
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
        if node_id < 0 or node_id >= len(self.graph.node_labels):
            if not hasattr(self, '_invalid_id_count'):
                self._invalid_id_count = 0
            self._invalid_id_count += 1
            if self._invalid_id_count % 1000 == 0:
                logging.warning(f"Invalid node ID: {node_id} (max: {len(self.graph.node_labels)-1}) - {self._invalid_id_count} total invalid IDs")
            return None
        
        # Additional validation: check if ID is actually registered
        if not self.id_manager.is_valid_id(node_id):
            logging.warning(f"Node ID {node_id} is not registered in ID manager")
            return None
        if self._cache_valid and node_id in self._node_cache:
            return self._node_cache[node_id]
        index = node_id
        if index >= len(self.graph.node_labels):
            logging.warning(f"Index {index} out of range for node ID: {node_id}")
            return None
        node = self.graph.node_labels[index]
        if self._cache_valid:
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
            return 0 <= node_id < len(self.node_labels)
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
