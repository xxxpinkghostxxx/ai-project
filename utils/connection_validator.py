"""
Connection Validator
Centralized validation system for neural graph connections.
"""

import threading
from typing import Dict, List, Any, Optional, Tuple
from utils.logging_utils import log_step


class ConnectionValidationError(Exception):
    """Exception raised for connection validation errors."""
    pass


class ConnectionValidator:
    """Centralized validator for neural graph connections."""

    def __init__(self):
        self._lock = threading.RLock()
        self._validation_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 1000
        self._stats = {
            'validations_performed': 0,
            'cache_hits': 0,
            'errors_found': 0,
            'warnings_issued': 0
        }

    def validate_connection(self, graph, source_id: int, target_id: int,
                          connection_type: str = 'excitatory',
                          weight: float = 1.0) -> Dict[str, Any]:
        """Comprehensive validation of a connection before creation."""
        with self._lock:
            self._stats['validations_performed'] += 1

            result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'suggestions': []
            }

            # Check if IDs exist and are valid
            id_validation = self._validate_node_ids(graph, source_id, target_id)
            result['errors'].extend(id_validation['errors'])
            result['warnings'].extend(id_validation['warnings'])

            # Check connection type validity
            type_validation = self._validate_connection_type(connection_type)
            result['errors'].extend(type_validation['errors'])
            result['warnings'].extend(type_validation['warnings'])

            # Check weight validity
            weight_validation = self._validate_connection_weight(weight, connection_type)
            result['errors'].extend(weight_validation['errors'])
            result['warnings'].extend(weight_validation['warnings'])

            # Check for duplicate connections
            duplicate_check = self._check_duplicate_connection(graph, source_id, target_id)
            result['warnings'].extend(duplicate_check['warnings'])

            # Check graph capacity
            capacity_check = self._check_graph_capacity(graph, source_id, target_id)
            result['warnings'].extend(capacity_check['warnings'])
            result['suggestions'].extend(capacity_check['suggestions'])

            # Check for cycles (potential infinite loops)
            cycle_check = self._check_for_cycles(graph, source_id, target_id)
            result['warnings'].extend(cycle_check['warnings'])

            # Update result validity
            result['is_valid'] = len(result['errors']) == 0

            # Update statistics
            if not result['is_valid']:
                self._stats['errors_found'] += 1
            if result['warnings']:
                self._stats['warnings_issued'] += len(result['warnings'])

            return result

    def _validate_node_ids(self, graph, source_id: int, target_id: int) -> Dict[str, List[str]]:
        """Validate that node IDs exist and are properly registered."""
        result = {'errors': [], 'warnings': []}

        # Check if graph has required attributes
        if not hasattr(graph, 'node_labels'):
            result['errors'].append("Graph missing node_labels attribute")
            return result

        # Check for self-connections
        if source_id == target_id:
            result['errors'].append(f"Self-connection not allowed: source_id={source_id}")
            result['warnings'].append("Self-connections can cause instability in neural networks")
            return result

        # Check if nodes exist in graph
        source_exists = any(node.get('id') == source_id for node in graph.node_labels)
        target_exists = any(node.get('id') == target_id for node in graph.node_labels)

        if not source_exists:
            result['errors'].append(f"Source node {source_id} does not exist in graph")
        if not target_exists:
            result['errors'].append(f"Target node {target_id} does not exist in graph")

        # Check node types compatibility
        if source_exists and target_exists:
            source_node = next((node for node in graph.node_labels if node.get('id') == source_id), None)
            target_node = next((node for node in graph.node_labels if node.get('id') == target_id), None)

            if source_node and target_node:
                source_type = source_node.get('type', 'unknown')
                target_type = target_node.get('type', 'unknown')

                # Check for invalid type combinations
                if source_type == 'sensory' and target_type == 'sensory':
                    result['warnings'].append("Sensory-to-sensory connections may not be meaningful")
                elif source_type == 'workspace' and target_type == 'sensory':
                    result['warnings'].append("Workspace-to-sensory connections are unusual")

        return result

    def _validate_connection_type(self, connection_type: str) -> Dict[str, List[str]]:
        """Validate connection type."""
        result = {'errors': [], 'warnings': []}

        valid_types = ['excitatory', 'inhibitory', 'modulatory', 'plastic', 'burst', 'relay']

        if connection_type not in valid_types:
            result['errors'].append(f"Invalid connection type: {connection_type}. Valid types: {valid_types}")
            result['warnings'].append(f"Connection type '{connection_type}' is not recommended")
        elif connection_type == 'burst':
            result['warnings'].append("Burst connections can cause instability - use with caution")

        return result

    def _validate_connection_weight(self, weight: float, connection_type: str) -> Dict[str, List[str]]:
        """Validate connection weight."""
        result = {'errors': [], 'warnings': []}

        # Check for NaN or infinite values
        if not isinstance(weight, (int, float)) or str(weight).lower() in ['nan', 'inf', '-inf']:
            result['errors'].append(f"Invalid weight value: {weight}")
            return result

        # Check weight ranges based on connection type
        if connection_type == 'inhibitory':
            if weight > 0:
                result['warnings'].append("Inhibitory connections should have negative weights")
        elif connection_type in ['excitatory', 'plastic']:
            if weight < 0:
                result['warnings'].append("Excitatory connections should have positive weights")
        elif connection_type == 'modulatory':
            if abs(weight) > 2.0:
                result['warnings'].append("Modulatory weights should typically be between -2.0 and 2.0")

        # Check for extreme values
        if abs(weight) > 10.0:
            result['warnings'].append(f"Extreme weight value: {weight}. Consider values between -5.0 and 5.0")

        return result

    def _check_duplicate_connection(self, graph, source_id: int, target_id: int) -> Dict[str, List[str]]:
        """Check for existing connections between the same nodes."""
        result = {'warnings': []}

        if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
            return result

        # Check for existing connections
        for edge in graph.edge_attributes:
            if (hasattr(edge, 'source') and hasattr(edge, 'target') and
                edge.source == source_id and edge.target == target_id):
                result['warnings'].append(f"Duplicate connection detected: {source_id} -> {target_id}")
                break

        return result

    def _check_graph_capacity(self, graph, source_id: int, target_id: int) -> Dict[str, List[str]]:
        """Check if adding this connection would exceed graph capacity."""
        result = {'warnings': [], 'suggestions': []}

        if not hasattr(graph, 'edge_attributes'):
            return result

        current_edges = len(graph.edge_attributes) if graph.edge_attributes else 0
        max_recommended_edges = len(graph.node_labels) * 10  # Rough heuristic

        if current_edges >= max_recommended_edges * 0.8:
            result['warnings'].append(f"Graph approaching capacity: {current_edges}/{max_recommended_edges} edges")
            result['suggestions'].append("Consider increasing graph capacity or pruning old connections")

        return result

    def _check_for_cycles(self, graph, source_id: int, target_id: int) -> Dict[str, List[str]]:
        """Check for potential cycles that could cause infinite loops."""
        result = {'warnings': []}

        # Skip cycle check for self-connections
        if source_id == target_id:
            return result

        # Simple cycle detection for small graphs
        if hasattr(graph, 'edge_attributes') and len(graph.edge_attributes) < 1000:
            # Build adjacency list
            adj_list = {}
            for edge in graph.edge_attributes:
                if hasattr(edge, 'source') and hasattr(edge, 'target'):
                    if edge.source not in adj_list:
                        adj_list[edge.source] = []
                    adj_list[edge.source].append(edge.target)

            # Check if target_id can reach source_id (creating a cycle)
            if self._has_path(adj_list, target_id, source_id):
                result['warnings'].append(f"Potential cycle detected: {source_id} -> {target_id} could create a loop")

        return result

    def _has_path(self, adj_list: Dict[int, List[int]], start: int, end: int,
                  visited: Optional[set] = None) -> bool:
        """Check if there's a path from start to end in the graph."""
        if visited is None:
            visited = set()

        if start == end:
            return True

        if start in visited:
            return False

        visited.add(start)

        if start in adj_list:
            for neighbor in adj_list[start]:
                if self._has_path(adj_list, neighbor, end, visited):
                    return True

        visited.remove(start)
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        with self._lock:
            return self._stats.copy()

    def clear_cache(self):
        """Clear the validation cache."""
        with self._lock:
            self._validation_cache.clear()

    def set_cache_size(self, max_size: int):
        """Set the maximum cache size."""
        with self._lock:
            self._cache_max_size = max_size
            # Trim cache if needed
            if len(self._validation_cache) > max_size:
                # Remove oldest entries (simple FIFO)
                items_to_remove = len(self._validation_cache) - max_size
                keys_to_remove = list(self._validation_cache.keys())[:items_to_remove]
                for key in keys_to_remove:
                    del self._validation_cache[key]


# Global instance
_connection_validator_instance = None
_connection_validator_lock = threading.Lock()


def get_connection_validator() -> ConnectionValidator:
    """Get the global connection validator instance."""
    global _connection_validator_instance
    if _connection_validator_instance is None:
        with _connection_validator_lock:
            if _connection_validator_instance is None:
                _connection_validator_instance = ConnectionValidator()
    return _connection_validator_instance