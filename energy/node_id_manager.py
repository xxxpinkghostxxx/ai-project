
import threading
import time

from typing import Dict, List, Optional, Set, Any

import logging
from utils.logging_utils import log_step


class NodeIDManager:

    def __init__(self):
        self._lock = threading.RLock()
        self._next_id = 1
        self._active_ids: Set[int] = set()
        self._recycled_ids: List[int] = []
        self._id_to_index: Dict[int, int] = {}
        self._index_to_id: Dict[int, int] = {}
        self._node_type_map: Dict[int, str] = {}
        self._id_metadata: Dict[int, Dict[str, Any]] = {}
        self._max_graph_size = 1000000
        self._stats = {
            'total_ids_generated': 0,
            'active_ids': 0,
            'recycled_ids': 0,
            'lookup_operations': 0,
            'creation_time': time.time()
        }
        log_step("NodeIDManager initialized")
    def set_max_graph_size(self, max_size: int):
        with self._lock:
            self._max_graph_size = max_size
            log_step("Max graph size set", max_size=max_size)
    def generate_unique_id(self, node_type: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> int:

        with self._lock:
            if self._recycled_ids:
                node_id = self._recycled_ids.pop(0)
                if node_id % 10000 == 0:
                    log_step("Recycled ID assigned", node_id=node_id, node_type=node_type)
            else:
                node_id = self._next_id
                self._next_id += 1
                if node_id >= self._max_graph_size:
                    node_id = self._find_next_available_id()
                    if node_id is None:
                        # No available IDs - this is a critical error
                        raise RuntimeError(f"Graph expansion limit reached: {self._max_graph_size} nodes. Cannot generate more IDs.")
                    else:
                        self._next_id = node_id + 1
            if node_id % 10000 == 0:
                log_step("ID generation batch", node_id=node_id, node_type=node_type, batch_size=10000)
            self._active_ids.add(node_id)
            self._node_type_map[node_id] = node_type
            if metadata:
                self._id_metadata[node_id] = metadata.copy()
            self._stats['total_ids_generated'] += 1
            self._stats['active_ids'] = len(self._active_ids)
            return node_id
    def _find_next_available_id(self) -> Optional[int]:
        for i in range(1, self._max_graph_size):
            if i not in self._active_ids:
                return i
        return None
    def register_node_index(self, node_id: int, index: int) -> bool:

        with self._lock:
            if node_id not in self._active_ids:
                logging.warning(f"Attempted to register index for inactive ID: {node_id}")
                return False
            self._id_to_index[node_id] = index
            self._index_to_id[index] = node_id
            if index % 5000 == 0:
                log_step("Node index registered batch", node_id=node_id, index=index, batch_size=5000)
            return True
    def get_node_index(self, node_id: int) -> Optional[int]:

        with self._lock:
            self._stats['lookup_operations'] += 1
            return self._id_to_index.get(node_id)
    def get_node_id(self, index: int) -> Optional[int]:

        with self._lock:
            self._stats['lookup_operations'] += 1
            return self._index_to_id.get(index)
    def is_valid_id(self, node_id: int) -> bool:

        with self._lock:
            return node_id in self._active_ids
    def get_node_type(self, node_id: int) -> Optional[str]:

        with self._lock:
            return self._node_type_map.get(node_id)
    def get_node_metadata(self, node_id: int) -> Optional[Dict[str, Any]]:

        with self._lock:
            return self._id_metadata.get(node_id)
    def recycle_node_id(self, node_id: int) -> bool:

        with self._lock:
            if node_id not in self._active_ids:
                logging.warning(f"Attempted to recycle inactive ID: {node_id}")
                return False
            self._active_ids.remove(node_id)
            index = None
            try:
                if node_id in self._id_to_index:
                    index = self._id_to_index[node_id]
                    del self._id_to_index[node_id]
                    if index is not None and index in self._index_to_id:
                        del self._index_to_id[index]
            except (KeyError, AttributeError):
                pass
            try:
                if node_id in self._node_type_map:
                    del self._node_type_map[node_id]
            except (KeyError, AttributeError):
                pass
            try:
                if node_id in self._id_metadata:
                    del self._id_metadata[node_id]
            except (KeyError, AttributeError):
                pass
            self._recycled_ids.append(node_id)
            self._stats['active_ids'] = len(self._active_ids)
            self._stats['recycled_ids'] = len(self._recycled_ids)
            log_step("Node ID recycled", node_id=node_id)
            return True
    def get_all_active_ids(self) -> List[int]:

        with self._lock:
            return list(self._active_ids)
    def get_ids_by_type(self, node_type: str) -> List[int]:

        with self._lock:
            return [node_id for node_id, ntype in self._node_type_map.items()
                   if ntype == node_type and node_id in self._active_ids]
    def get_statistics(self) -> Dict[str, Any]:

        with self._lock:
            stats = self._stats.copy()
            stats['uptime'] = time.time() - stats['creation_time']
            stats['recycled_ids_count'] = len(self._recycled_ids)
            return stats
    def validate_graph_consistency(self, graph) -> Dict[str, Any]:

        with self._lock:
            validation_results = {
                'is_consistent': True,
                'errors': [],
                'warnings': [],
                'node_count_mismatch': False,
                'missing_ids': [],
                'orphaned_ids': []
            }
            if not hasattr(graph, 'node_labels'):
                validation_results['errors'].append("Graph missing node_labels")
                validation_results['is_consistent'] = False
                return validation_results
            graph_node_count = len(graph.node_labels)
            active_id_count = len(self._active_ids)
            if graph_node_count != active_id_count:
                validation_results['node_count_mismatch'] = True
                validation_results['warnings'].append(
                    f"Node count mismatch: graph has {graph_node_count} nodes, "
                    f"ID manager has {active_id_count} active IDs"
                )
            for i, node_label in enumerate(graph.node_labels):
                node_id = node_label.get('id')
                if node_id is None:
                    validation_results['missing_ids'].append(i)
                    validation_results['errors'].append(f"Node at index {i} missing ID")
                elif not self.is_valid_id(node_id):
                    validation_results['errors'].append(f"Node at index {i} has invalid ID: {node_id}")
                elif self.get_node_index(node_id) != i:
                    validation_results['errors'].append(
                        f"Node ID {node_id} index mismatch: expected {i}, got {self.get_node_index(node_id)}"
                    )
            for node_id in self._active_ids:
                if node_id not in self._id_to_index:
                    validation_results['orphaned_ids'].append(node_id)
                    validation_results['warnings'].append(f"Orphaned ID: {node_id}")
            if validation_results['errors']:
                validation_results['is_consistent'] = False
            return validation_results
    def can_expand_graph(self, additional_nodes: int = 1) -> bool:
        """Check if the graph can expand by the given number of nodes."""
        with self._lock:
            return len(self._active_ids) + additional_nodes <= self._max_graph_size
    
    def get_expansion_capacity(self) -> int:
        """Get the number of additional nodes that can be added before hitting the limit."""
        with self._lock:
            return max(0, self._max_graph_size - len(self._active_ids))
    
    def cleanup_orphaned_ids(self, graph) -> int:
        """Remove IDs that are no longer in the graph and return count of cleaned IDs."""
        with self._lock:
            if not hasattr(graph, 'node_labels'):
                return 0
            
            graph_node_ids = {node.get('id') for node in graph.node_labels if node.get('id') is not None}
            orphaned_ids = self._active_ids - graph_node_ids
            
            for node_id in orphaned_ids:
                self.recycle_node_id(node_id)
            
            log_step(f"Cleaned up {len(orphaned_ids)} orphaned IDs")
            return len(orphaned_ids)
    
    def reset(self):
        with self._lock:
            self._next_id = 1
            self._active_ids.clear()
            self._recycled_ids.clear()
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._node_type_map.clear()
            self._id_metadata.clear()
            self._stats = {
                'total_ids_generated': 0,
                'active_ids': 0,
                'recycled_ids': 0,
                'lookup_operations': 0,
                'creation_time': time.time()
            }
            # Reset max graph size to default
            self._max_graph_size = 1000000  # 1 million nodes default
            log_step("NodeIDManager reset", max_size=self._max_graph_size)
_id_manager_instance = None
_id_manager_lock = threading.Lock()


def get_id_manager() -> NodeIDManager:

    global _id_manager_instance
    if _id_manager_instance is None:
        with _id_manager_lock:
            if _id_manager_instance is None:
                _id_manager_instance = NodeIDManager()
    return _id_manager_instance


def reset_id_manager():
    global _id_manager_instance
    with _id_manager_lock:
        if _id_manager_instance is not None:
            _id_manager_instance.reset()
        else:
            _id_manager_instance = NodeIDManager()

def force_reset_id_manager():
    """Force reset the ID manager, creating a new instance."""
    global _id_manager_instance
    with _id_manager_lock:
        _id_manager_instance = None
        log_step("ID manager force reset - new instance will be created")


def generate_node_id(node_type: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> int:
    return get_id_manager().generate_unique_id(node_type, metadata)


def get_node_index_by_id(node_id: int) -> Optional[int]:
    return get_id_manager().get_node_index(node_id)


def get_node_id_by_index(index: int) -> Optional[int]:
    return get_id_manager().get_node_id(index)


def is_valid_node_id(node_id: int) -> bool:
    return get_id_manager().is_valid_id(node_id)


def recycle_node_id(node_id: int) -> bool:
    return get_id_manager().recycle_node_id(node_id)
