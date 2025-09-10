"""
node_id_manager.py

Centralized unique ID generation and management system for the ID-based node simulation.
Provides thread-safe ID generation, validation, and lookup capabilities.
"""

import threading
import time
import uuid
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
import logging
from logging_utils import log_step


class NodeIDManager:
    """
    Centralized manager for unique node IDs in the ID-based simulation system.
    
    Features:
    - Thread-safe unique ID generation
    - ID validation and lookup
    - ID-to-index mapping for performance
    - ID recycling for deleted nodes
    - Support for different node types
    """
    
    def __init__(self):
        """Initialize the node ID manager."""
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        self._next_id = 1  # Start IDs from 1 (0 reserved for invalid)
        self._active_ids: Set[int] = set()
        self._recycled_ids: List[int] = []
        self._id_to_index: Dict[int, int] = {}  # ID -> array index mapping
        self._index_to_id: Dict[int, int] = {}  # array index -> ID mapping
        self._node_type_map: Dict[int, str] = {}  # ID -> node type mapping
        self._id_metadata: Dict[int, Dict[str, Any]] = {}  # ID -> metadata mapping
        
        # Statistics
        self._stats = {
            'total_ids_generated': 0,
            'active_ids': 0,
            'recycled_ids': 0,
            'lookup_operations': 0,
            'creation_time': time.time()
        }
        
        log_step("NodeIDManager initialized")
    
    def generate_unique_id(self, node_type: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Generate a unique ID for a new node.
        
        Args:
            node_type: Type of node (sensory, dynamic, workspace, etc.)
            metadata: Optional metadata to associate with the ID
        
        Returns:
            int: Unique node ID
        """
        with self._lock:
            # Try to reuse a recycled ID first
            if self._recycled_ids:
                node_id = self._recycled_ids.pop(0)
                log_step("Recycled ID assigned", node_id=node_id, node_type=node_type)
            else:
                # Generate new ID
                node_id = self._next_id
                self._next_id += 1
                log_step("New ID generated", node_id=node_id, node_type=node_type)
            
            # Register the ID
            self._active_ids.add(node_id)
            self._node_type_map[node_id] = node_type
            
            # Store metadata if provided
            if metadata:
                self._id_metadata[node_id] = metadata.copy()
            
            # Update statistics
            self._stats['total_ids_generated'] += 1
            self._stats['active_ids'] = len(self._active_ids)
            
            return node_id
    
    def register_node_index(self, node_id: int, index: int) -> bool:
        """
        Register the array index for a node ID.
        
        Args:
            node_id: The unique node ID
            index: The array index in the graph
        
        Returns:
            bool: True if registration successful, False otherwise
        """
        with self._lock:
            if node_id not in self._active_ids:
                logging.warning(f"Attempted to register index for inactive ID: {node_id}")
                return False
            
            # Update mappings
            self._id_to_index[node_id] = index
            self._index_to_id[index] = node_id
            
            log_step("Node index registered", node_id=node_id, index=index)
            return True
    
    def get_node_index(self, node_id: int) -> Optional[int]:
        """
        Get the array index for a node ID.
        
        Args:
            node_id: The unique node ID
        
        Returns:
            int: Array index if found, None otherwise
        """
        with self._lock:
            self._stats['lookup_operations'] += 1
            return self._id_to_index.get(node_id)
    
    def get_node_id(self, index: int) -> Optional[int]:
        """
        Get the node ID for an array index.
        
        Args:
            index: The array index
        
        Returns:
            int: Node ID if found, None otherwise
        """
        with self._lock:
            self._stats['lookup_operations'] += 1
            return self._index_to_id.get(index)
    
    def is_valid_id(self, node_id: int) -> bool:
        """
        Check if a node ID is valid and active.
        
        Args:
            node_id: The node ID to validate
        
        Returns:
            bool: True if ID is valid and active
        """
        with self._lock:
            return node_id in self._active_ids
    
    def get_node_type(self, node_id: int) -> Optional[str]:
        """
        Get the node type for a given ID.
        
        Args:
            node_id: The node ID
        
        Returns:
            str: Node type if found, None otherwise
        """
        with self._lock:
            return self._node_type_map.get(node_id)
    
    def get_node_metadata(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a given node ID.
        
        Args:
            node_id: The node ID
        
        Returns:
            dict: Metadata if found, None otherwise
        """
        with self._lock:
            return self._id_metadata.get(node_id)
    
    def recycle_node_id(self, node_id: int) -> bool:
        """
        Recycle a node ID for reuse.
        
        Args:
            node_id: The node ID to recycle
        
        Returns:
            bool: True if recycling successful, False otherwise
        """
        with self._lock:
            if node_id not in self._active_ids:
                logging.warning(f"Attempted to recycle inactive ID: {node_id}")
                return False
            
            # Remove from active set
            self._active_ids.remove(node_id)
            
            # Remove from mappings
            if node_id in self._id_to_index:
                index = self._id_to_index[node_id]
                del self._id_to_index[node_id]
                if index in self._index_to_id:
                    del self._index_to_id[index]
            
            # Remove metadata
            if node_id in self._node_type_map:
                del self._node_type_map[node_id]
            if node_id in self._id_metadata:
                del self._id_metadata[node_id]
            
            # Add to recycled list
            self._recycled_ids.append(node_id)
            
            # Update statistics
            self._stats['active_ids'] = len(self._active_ids)
            self._stats['recycled_ids'] = len(self._recycled_ids)
            
            log_step("Node ID recycled", node_id=node_id)
            return True
    
    def get_all_active_ids(self) -> List[int]:
        """
        Get all currently active node IDs.
        
        Returns:
            List[int]: List of active node IDs
        """
        with self._lock:
            return list(self._active_ids)
    
    def get_ids_by_type(self, node_type: str) -> List[int]:
        """
        Get all active node IDs of a specific type.
        
        Args:
            node_type: The node type to filter by
        
        Returns:
            List[int]: List of node IDs of the specified type
        """
        with self._lock:
            return [node_id for node_id, ntype in self._node_type_map.items() 
                   if ntype == node_type and node_id in self._active_ids]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the ID manager.
        
        Returns:
            dict: Statistics dictionary
        """
        with self._lock:
            stats = self._stats.copy()
            stats['uptime'] = time.time() - stats['creation_time']
            stats['recycled_ids_count'] = len(self._recycled_ids)
            return stats
    
    def validate_graph_consistency(self, graph) -> Dict[str, Any]:
        """
        Validate that the ID manager is consistent with the graph structure.
        
        Args:
            graph: The graph to validate against
        
        Returns:
            dict: Validation results
        """
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
            
            # Check node count consistency
            if graph_node_count != active_id_count:
                validation_results['node_count_mismatch'] = True
                validation_results['warnings'].append(
                    f"Node count mismatch: graph has {graph_node_count} nodes, "
                    f"ID manager has {active_id_count} active IDs"
                )
            
            # Check for missing IDs in graph
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
            
            # Check for orphaned IDs
            for node_id in self._active_ids:
                if node_id not in self._id_to_index:
                    validation_results['orphaned_ids'].append(node_id)
                    validation_results['warnings'].append(f"Orphaned ID: {node_id}")
            
            if validation_results['errors']:
                validation_results['is_consistent'] = False
            
            return validation_results
    
    def reset(self):
        """Reset the ID manager to initial state."""
        with self._lock:
            self._next_id = 1
            self._active_ids.clear()
            self._recycled_ids.clear()
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._node_type_map.clear()
            self._id_metadata.clear()
            
            # Reset statistics
            self._stats = {
                'total_ids_generated': 0,
                'active_ids': 0,
                'recycled_ids': 0,
                'lookup_operations': 0,
                'creation_time': time.time()
            }
            
            log_step("NodeIDManager reset")


# Dependency injection support
from dependency_injection import register_service, resolve_service, is_service_registered

def get_id_manager() -> NodeIDManager:
    """
    Get the node ID manager instance from dependency container.
    
    Returns:
        NodeIDManager: The ID manager instance
    """
    if is_service_registered(NodeIDManager):
        return resolve_service(NodeIDManager)
    else:
        # Fallback: create and register a new instance
        id_manager = NodeIDManager()
        register_service(NodeIDManager, instance=id_manager)
        return id_manager


def reset_id_manager():
    """Reset the ID manager instance in the dependency container."""
    if is_service_registered(NodeIDManager):
        id_manager = resolve_service(NodeIDManager)
        id_manager.reset()
        # Re-register a fresh instance
        register_service(NodeIDManager, instance=NodeIDManager())


# Convenience functions for backward compatibility
def generate_node_id(node_type: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> int:
    """Generate a unique node ID."""
    return get_id_manager().generate_unique_id(node_type, metadata)


def get_node_index_by_id(node_id: int) -> Optional[int]:
    """Get array index for a node ID."""
    return get_id_manager().get_node_index(node_id)


def get_node_id_by_index(index: int) -> Optional[int]:
    """Get node ID for an array index."""
    return get_id_manager().get_node_id(index)


def is_valid_node_id(node_id: int) -> bool:
    """Check if a node ID is valid."""
    return get_id_manager().is_valid_id(node_id)


def recycle_node_id(node_id: int) -> bool:
    """Recycle a node ID."""
    return get_id_manager().recycle_node_id(node_id)
