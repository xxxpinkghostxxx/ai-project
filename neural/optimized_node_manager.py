"""
Optimized Node Manager for high-performance neural simulations.
Provides spatial indexing, batch operations, and memory-efficient node management.
"""

import numpy as np
import torch
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import logging
import time

from utils.logging_utils import log_step
from utils.static_allocator import get_static_allocator

class SpatialIndex:
    """Spatial indexing for efficient node queries in large graphs."""
    
    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.grid: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.node_positions: Dict[int, Tuple[float, float]] = {}
        self._lock = threading.RLock()
    
    def add_node(self, node_id: int, x: float, y: float):
        """Add a node to the spatial index."""
        with self._lock:
            grid_x = int(x // self.grid_size)
            grid_y = int(y // self.grid_size)
            self.grid[(grid_x, grid_y)].add(node_id)
            self.node_positions[node_id] = (x, y)
    
    def remove_node(self, node_id: int):
        """Remove a node from the spatial index."""
        with self._lock:
            if node_id in self.node_positions:
                x, y = self.node_positions[node_id]
                grid_x = int(x // self.grid_size)
                grid_y = int(y // self.grid_size)
                self.grid[(grid_x, grid_y)].discard(node_id)
                del self.node_positions[node_id]
    
    def get_nodes_in_radius(self, center_x: float, center_y: float, radius: float) -> Set[int]:
        """Get all nodes within a radius of a point."""
        with self._lock:
            result = set()
            grid_radius = int(radius // self.grid_size) + 1
            
            center_grid_x = int(center_x // self.grid_size)
            center_grid_y = int(center_y // self.grid_size)
            
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    grid_x = center_grid_x + dx
                    grid_y = center_grid_y + dy
                    
                    if (grid_x, grid_y) in self.grid:
                        for node_id in self.grid[(grid_x, grid_y)]:
                            if node_id in self.node_positions:
                                x, y = self.node_positions[node_id]
                                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                                    result.add(node_id)
            
            return result
    
    def get_nearest_nodes(self, x: float, y: float, count: int = 10) -> List[Tuple[int, float]]:
        """Get the nearest N nodes to a point."""
        with self._lock:
            candidates = []
            
            # Check nearby grid cells
            grid_x = int(x // self.grid_size)
            grid_y = int(y // self.grid_size)
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cell_x = grid_x + dx
                    cell_y = grid_y + dy
                    
                    if (cell_x, cell_y) in self.grid:
                        for node_id in self.grid[(cell_x, cell_y)]:
                            if node_id in self.node_positions:
                                nx, ny = self.node_positions[node_id]
                                distance = (nx - x)**2 + (ny - y)**2
                                candidates.append((node_id, distance))
            
            # Sort by distance and return top N
            candidates.sort(key=lambda x: x[1])
            return candidates[:count]

class BatchNodeProcessor:
    """Batch processing system for efficient node operations."""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.node_batches: Dict[str, List[List[int]]] = defaultdict(list)
        self.processing_queues: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()
    
    def add_to_batch(self, operation_type: str, node_ids: List[int]):
        """Add nodes to a batch operation queue."""
        with self._lock:
            queue = self.processing_queues[operation_type]
            for node_id in node_ids:
                queue.append(node_id)
            
            # Create batches when queue is full
            while len(queue) >= self.batch_size:
                batch = []
                for _ in range(self.batch_size):
                    batch.append(queue.popleft())
                self.node_batches[operation_type].append(batch)
    
    def get_batch(self, operation_type: str) -> Optional[List[int]]:
        """Get the next batch for processing."""
        with self._lock:
            if self.node_batches[operation_type]:
                return self.node_batches[operation_type].pop(0)
            return None
    
    def get_batch_count(self, operation_type: str) -> int:
        """Get the number of pending batches for an operation type."""
        with self._lock:
            return len(self.node_batches[operation_type])

class OptimizedNodeManager:
    """High-performance node manager with spatial indexing and batch processing."""
    
    def __init__(self, max_nodes: int = 100000):
        self.max_nodes = max_nodes
        self.spatial_index = SpatialIndex()
        self.batch_processor = BatchNodeProcessor()
        
        # Pre-allocated data structures
        self.node_data = np.zeros((max_nodes, 10), dtype=np.float32)  # energy, membrane_potential, etc.
        self.node_metadata = [None] * max_nodes
        self.active_nodes: Set[int] = set()
        self.node_positions = np.zeros((max_nodes, 2), dtype=np.float32)
        
        # Efficient lookup structures
        self.node_id_to_index: Dict[int, int] = {}
        self.index_to_node_id: Dict[int, int] = {}
        self.node_types: Dict[int, str] = {}
        
        # Performance tracking
        self.stats = {
            'total_nodes': 0,
            'active_nodes': 0,
            'operations_per_second': 0.0,
            'last_update': time.time()
        }
        
        self._lock = threading.RLock()
        log_step("OptimizedNodeManager initialized", max_nodes=max_nodes)
    
    def create_node_batch(self, node_specs: List[Dict[str, Any]]) -> List[int]:
        """Create multiple nodes in a batch operation."""
        with self._lock:
            created_ids = []
            
            for spec in node_specs:
                if len(self.active_nodes) >= self.max_nodes:
                    logging.warning("Node limit reached, cannot create more nodes")
                    break
                
                # Find available index
                index = self._find_available_index()
                if index is None:
                    break
                
                # Generate unique ID
                node_type = spec.get('type', 'dynamic')
                node_id = self._generate_node_id(node_type)
                
                # Initialize node data
                self._initialize_node_data(index, node_id, spec)
                
                # Add to spatial index if position provided
                if 'x' in spec and 'y' in spec:
                    self.spatial_index.add_node(node_id, spec['x'], spec['y'])
                    self.node_positions[index] = [spec['x'], spec['y']]
                
                self.active_nodes.add(node_id)
                self.node_id_to_index[node_id] = index
                self.index_to_node_id[index] = node_id
                self.node_types[node_id] = node_type
                
                created_ids.append(node_id)
            
            self.stats['total_nodes'] += len(created_ids)
            self.stats['active_nodes'] = len(self.active_nodes)
            
            log_step("Batch node creation", count=len(created_ids), total_active=self.stats['active_nodes'])
            return created_ids
    
    def _find_available_index(self) -> Optional[int]:
        """Find an available index for a new node."""
        for i in range(self.max_nodes):
            if i not in self.index_to_node_id:
                return i
        return None
    
    def _generate_node_id(self, node_type: str) -> int:
        """Generate a unique node ID."""
        # Simple ID generation - in production, use a more robust system
        return hash(f"{node_type}_{time.time()}_{len(self.active_nodes)}") % 10000000
    
    def _initialize_node_data(self, index: int, node_id: int, spec: Dict[str, Any]):
        """Initialize node data at the given index."""
        # Initialize basic properties
        self.node_data[index, 0] = spec.get('energy', 1.0)  # energy
        self.node_data[index, 1] = spec.get('membrane_potential', 0.0)  # membrane_potential
        self.node_data[index, 2] = spec.get('threshold', 0.5)  # threshold
        self.node_data[index, 3] = spec.get('refractory_timer', 0.0)  # refractory_timer
        self.node_data[index, 4] = spec.get('last_activation', 0.0)  # last_activation
        self.node_data[index, 5] = spec.get('plasticity_enabled', 1.0)  # plasticity_enabled
        self.node_data[index, 6] = spec.get('eligibility_trace', 0.0)  # eligibility_trace
        self.node_data[index, 7] = spec.get('last_update', time.time())  # last_update
        self.node_data[index, 8] = spec.get('state', 1.0)  # state (1.0 = active)
        self.node_data[index, 9] = spec.get('type_code', 0.0)  # type_code
        
        # Store metadata
        self.node_metadata[index] = {
            'id': node_id,
            'type': spec.get('type', 'dynamic'),
            'behavior': spec.get('behavior', 'dynamic'),
            'state': spec.get('state', 'active'),
            'x': spec.get('x', 0.0),
            'y': spec.get('y', 0.0)
        }
    
    def update_nodes_batch(self, node_ids: List[int], updates: Dict[str, Any]):
        """Update multiple nodes in a batch operation."""
        with self._lock:
            for node_id in node_ids:
                if node_id not in self.node_id_to_index:
                    continue
                
                index = self.node_id_to_index[node_id]
                
                # Update node data
                if 'energy' in updates:
                    self.node_data[index, 0] = updates['energy']
                if 'membrane_potential' in updates:
                    self.node_data[index, 1] = updates['membrane_potential']
                if 'threshold' in updates:
                    self.node_data[index, 2] = updates['threshold']
                if 'refractory_timer' in updates:
                    self.node_data[index, 3] = updates['refractory_timer']
                if 'last_activation' in updates:
                    self.node_data[index, 4] = updates['last_activation']
                if 'plasticity_enabled' in updates:
                    self.node_data[index, 5] = 1.0 if updates['plasticity_enabled'] else 0.0
                if 'eligibility_trace' in updates:
                    self.node_data[index, 6] = updates['eligibility_trace']
                if 'last_update' in updates:
                    self.node_data[index, 7] = updates['last_update']
                if 'state' in updates:
                    self.node_data[index, 8] = 1.0 if updates['state'] == 'active' else 0.0
                
                # Update metadata
                if self.node_metadata[index]:
                    self.node_metadata[index].update(updates)
    
    def get_nodes_in_area(self, center_x: float, center_y: float, radius: float) -> List[int]:
        """Get all nodes within a radius using spatial indexing."""
        return list(self.spatial_index.get_nodes_in_radius(center_x, center_y, radius))
    
    def get_node_data_batch(self, node_ids: List[int]) -> np.ndarray:
        """Get node data for multiple nodes efficiently."""
        with self._lock:
            indices = []
            for node_id in node_ids:
                if node_id in self.node_id_to_index:
                    indices.append(self.node_id_to_index[node_id])
            
            if not indices:
                return np.array([])
            
            return self.node_data[indices].copy()
    
    def process_node_operations_batch(self, operation_type: str, operation_func):
        """Process batched node operations."""
        batch = self.batch_processor.get_batch(operation_type)
        if batch:
            try:
                operation_func(batch)
                log_step(f"Processed {operation_type} batch", count=len(batch))
            except Exception as e:
                logging.error(f"Error processing {operation_type} batch: {e}")
    
    def remove_inactive_nodes(self):
        """Remove nodes that are no longer active."""
        with self._lock:
            inactive_nodes = []
            
            for node_id in list(self.active_nodes):
                index = self.node_id_to_index[node_id]
                if self.node_data[index, 8] < 0.5:  # state != active
                    inactive_nodes.append(node_id)
            
            for node_id in inactive_nodes:
                self._remove_node(node_id)
            
            if inactive_nodes:
                log_step("Removed inactive nodes", count=len(inactive_nodes))
    
    def _remove_node(self, node_id: int):
        """Remove a single node."""
        if node_id not in self.node_id_to_index:
            return
        
        index = self.node_id_to_index[node_id]
        
        # Remove from spatial index
        self.spatial_index.remove_node(node_id)
        
        # Clear data
        self.node_data[index] = 0.0
        self.node_metadata[index] = None
        self.node_positions[index] = [0.0, 0.0]
        
        # Remove from lookup structures
        self.active_nodes.discard(node_id)
        del self.node_id_to_index[node_id]
        del self.index_to_node_id[index]
        if node_id in self.node_types:
            del self.node_types[node_id]
        
        self.stats['active_nodes'] = len(self.active_nodes)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        current_time = time.time()
        time_diff = current_time - self.stats['last_update']
        
        if time_diff > 0:
            self.stats['operations_per_second'] = self.stats.get('operation_count', 0) / time_diff
        
        self.stats['last_update'] = current_time
        
        return {
            'total_nodes': self.stats['total_nodes'],
            'active_nodes': self.stats['active_nodes'],
            'memory_usage_mb': self.node_data.nbytes / (1024 * 1024),
            'operations_per_second': self.stats['operations_per_second'],
            'spatial_index_cells': len(self.spatial_index.grid)
        }
    
    def cleanup(self):
        """Clean up resources."""
        with self._lock:
            self.spatial_index = SpatialIndex()
            self.active_nodes.clear()
            self.node_id_to_index.clear()
            self.index_to_node_id.clear()
            self.node_types.clear()
            
            # Reset arrays
            self.node_data.fill(0.0)
            self.node_positions.fill(0.0)
            for i in range(len(self.node_metadata)):
                self.node_metadata[i] = None

# Global optimized node manager instance
_global_node_manager = None

def get_optimized_node_manager() -> OptimizedNodeManager:
    """Get the global optimized node manager instance."""
    global _global_node_manager
    if _global_node_manager is None:
        _global_node_manager = OptimizedNodeManager()
    return _global_node_manager