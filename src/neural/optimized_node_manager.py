"""
Optimized Node Manager for high-performance neural simulations.
Provides spatial indexing, batch operations, and memory-efficient node management.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.utils.logging_utils import log_step


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
        # Validate max_nodes parameter
        if max_nodes <= 0 or max_nodes > 10000000:  # Reasonable upper limit
            raise ValueError(f"max_nodes must be between 1 and 10,000,000, got {max_nodes}")

        self.max_nodes = max_nodes
        self.spatial_index = SpatialIndex()
        self.batch_processor = BatchNodeProcessor()

        try:
            # Pre-allocated data structures with memory validation
            self.node_data = np.zeros((max_nodes, 10), dtype=np.float32)  # energy, membrane_potential, etc.
            self.node_metadata = [None] * max_nodes
            self.active_nodes: Set[int] = set()
            self.node_positions = np.zeros((max_nodes, 2), dtype=np.float32)

            # Check if memory allocation succeeded
            if self.node_data.nbytes == 0:
                raise MemoryError("Failed to allocate memory for node_data array")

        except MemoryError as e:
            logging.error(f"Memory allocation failed for {max_nodes} nodes: {e}")
            raise

        # Efficient lookup structures
        self.node_id_to_index: Dict[int, int] = {}
        self.index_to_node_id: Dict[int, int] = {}
        self.node_types: Dict[int, str] = {}

        # Performance tracking with thread-safe counters
        self.stats = {
            'total_nodes': 0,
            'active_nodes': 0,
            'operations_per_second': 0.0,
            'last_update': time.time(),
            'operation_count': 0
        }

        self._lock = threading.RLock()
        self._memory_limit_mb = 1024  # 1GB default memory limit
        log_step("OptimizedNodeManager initialized", max_nodes=max_nodes)

    def get_memory_limit(self) -> int:
        """Get the memory limit in MB."""
        return self._memory_limit_mb

    def set_memory_limit_internal(self, limit_mb: int) -> None:
        """Set the memory limit in MB (internal method)."""
        self._memory_limit_mb = limit_mb

    def get_lock(self) -> threading.RLock():
        """Get the internal lock."""
        return self._lock

    def find_available_index_internal(self) -> Optional[int]:
        """Find an available index for a new node (internal method)."""
        return self._find_available_index()

    def generate_node_id_internal(self, node_type: str) -> int:
        """Generate a unique node ID (internal method)."""
        return self._generate_node_id(node_type)

    def initialize_node_data_internal(self, index: int, node_id: int, spec: Dict[str, Any]) -> None:
        """Initialize node data at the given index (internal method)."""
        self._initialize_node_data(index, node_id, spec)

    def validate_float_internal(self, value: Any, min_val: float, max_val: float, field_name: str) -> float:
        """Validate and clamp float values (internal method)."""
        return self._validate_float(value, min_val, max_val, field_name)

    def remove_node_internal(self, node_id: int) -> None:
        """Remove a single node (internal method)."""
        self._remove_node(node_id)
    
    def create_node_batch(self, node_specs: List[Dict[str, Any]]) -> List[int]:
        """Create multiple nodes in a batch operation with validation."""
        if not isinstance(node_specs, list):
            raise TypeError("node_specs must be a list")

        try:
            with self._lock:
                created_ids = []

                for spec in node_specs:
                    if not isinstance(spec, dict):
                        logging.warning("Skipping invalid node spec: not a dictionary")
                        continue

                    if 'type' in spec and not isinstance(spec['type'], str):
                        logging.warning("Skipping invalid node spec: type must be a string")
                        continue

                    # Check memory usage before creating nodes
                    current_memory_mb = self.node_data.nbytes / (1024 * 1024)
                    if current_memory_mb > self.get_memory_limit() * 0.9:  # 90% of limit
                        logging.warning(f"Memory usage high ({current_memory_mb:.1f}MB), limiting node creation")
                        break

                    if len(self.active_nodes) >= self.max_nodes:
                        logging.warning("Node limit reached, cannot create more nodes")
                        break

                    # Find available index
                    index = self.find_available_index_internal()
                    if index is None:
                        logging.warning("No available indices for new nodes")
                        break

                    # Validate index bounds
                    if index < 0 or index >= self.max_nodes:
                        logging.error(f"Invalid index {index}, skipping node creation")
                        continue

                    # Generate unique ID
                    node_type = spec.get('type', 'dynamic')
                    if not isinstance(node_type, str):
                        node_type = 'dynamic'

                    node_id = self.generate_node_id_internal(node_type)

                    # Initialize node data with error handling
                    try:
                        self.initialize_node_data_internal(index, node_id, spec)
                    except Exception as e:
                        logging.error(f"Failed to initialize node data for index {index}: {e}")
                        continue

                    # Add to spatial index if position provided
                    if 'x' in spec and 'y' in spec:
                        try:
                            x, y = float(spec['x']), float(spec['y'])
                            self.spatial_index.add_node(node_id, x, y)
                            self.node_positions[index] = [x, y]
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Invalid position data for node {node_id}: {e}")
                            continue

                    self.active_nodes.add(node_id)
                    self.node_id_to_index[node_id] = index
                    self.index_to_node_id[index] = node_id
                    self.node_types[node_id] = node_type

                    created_ids.append(node_id)

                self.stats['total_nodes'] += len(created_ids)
                self.stats['active_nodes'] = len(self.active_nodes)
                self.stats['operation_count'] += 1

                log_step("Batch node creation", count=len(created_ids), total_active=self.stats['active_nodes'])
                return created_ids
        except Exception as e:
            logging.error(f"Error in create_node_batch: {e}")
            return []
    
    def _find_available_index(self) -> Optional[int]:
        """Find an available index for a new node."""
        for i in range(self.max_nodes):
            if i not in self.index_to_node_id:
                return i
        return None
    
    def _generate_node_id(self, node_type: str) -> int:
        """Generate a unique node ID with collision avoidance."""
        import random

        max_attempts = 100
        for attempt in range(max_attempts):
            # Use timestamp, random component, and attempt counter for uniqueness
            base_id = int(time.time() * 1000000) + random.randint(0, 999999)
            candidate_id = (base_id + attempt * 1234567) % 999999999  # Large prime modulus

            # Ensure uniqueness
            if candidate_id not in self.node_id_to_index:
                return candidate_id

        # Fallback: use a very large number to avoid collisions
        return int(time.time() * 1000000) + len(self.active_nodes) + random.randint(1000000000, 2000000000)
    
    def _initialize_node_data(self, index: int, node_id: int, spec: Dict[str, Any]):
        """Initialize node data at the given index with bounds checking."""
        # Validate index bounds
        if index < 0 or index >= self.max_nodes:
            raise ValueError(f"Index {index} out of bounds (0-{self.max_nodes-1})")

        try:
            # Initialize basic properties with validation
            self.node_data[index, 0] = self.validate_float_internal(spec.get('energy', 1.0), -1000.0, 1000.0, 'energy')
            self.node_data[index, 1] = self.validate_float_internal(spec.get('membrane_potential', 0.0), -100.0, 100.0, 'membrane_potential')
            self.node_data[index, 2] = self.validate_float_internal(spec.get('threshold', 0.5), -10.0, 10.0, 'threshold')
            self.node_data[index, 3] = self.validate_float_internal(spec.get('refractory_timer', 0.0), 0.0, 1000.0, 'refractory_timer')
            self.node_data[index, 4] = self.validate_float_internal(spec.get('last_activation', 0.0), 0.0, float('inf'), 'last_activation')
            self.node_data[index, 5] = 1.0 if spec.get('plasticity_enabled', True) else 0.0
            self.node_data[index, 6] = self.validate_float_internal(spec.get('eligibility_trace', 0.0), -100.0, 100.0, 'eligibility_trace')
            self.node_data[index, 7] = time.time()  # Always use current time for last_update
            self.node_data[index, 8] = 1.0 if spec.get('state', 'active') == 'active' else 0.0
            self.node_data[index, 9] = self.validate_float_internal(spec.get('type_code', 0.0), 0.0, 100.0, 'type_code')

            # Store metadata with validation
            self.node_metadata[index] = {
                'id': node_id,
                'type': str(spec.get('type', 'dynamic'))[:50],  # Limit string length
                'behavior': str(spec.get('behavior', 'dynamic'))[:50],
                'state': str(spec.get('state', 'active'))[:20],
                'x': self.validate_float_internal(spec.get('x', 0.0), -10000.0, 10000.0, 'x'),
                'y': self.validate_float_internal(spec.get('y', 0.0), -10000.0, 10000.0, 'y')
            }

        except Exception as e:
            # Clean up partial initialization
            self.node_data[index] = 0.0
            self.node_metadata[index] = None
            raise RuntimeError(f"Failed to initialize node data at index {index}: {e}")

    def _validate_float(self, value: Any, min_val: float, max_val: float, field_name: str) -> float:
        """Validate and clamp float values."""
        try:
            val = float(value)
            if not (min_val <= val <= max_val):
                logging.warning(f"Value {val} for {field_name} out of range [{min_val}, {max_val}], clamping")
                val = max(min_val, min(max_val, val))
            return val
        except (ValueError, TypeError):
            logging.warning(f"Invalid value {value} for {field_name}, using default")
            return min_val if min_val > 0 else 0.0
    
    def update_nodes_batch(self, node_ids: List[int], updates: Dict[str, Any]):
        """Update multiple nodes in a batch operation with validation."""
        if not isinstance(node_ids, list) or not isinstance(updates, dict):
            raise TypeError("node_ids must be a list and updates must be a dict")

        with self._lock:
            updated_count = 0

            for node_id in node_ids:
                if not isinstance(node_id, int):
                    logging.warning(f"Skipping invalid node_id: {node_id}")
                    continue

                if node_id not in self.node_id_to_index:
                    logging.debug(f"Node {node_id} not found, skipping update")
                    continue

                index = self.node_id_to_index[node_id]

                # Validate index bounds
                if index < 0 or index >= self.max_nodes:
                    logging.error(f"Invalid index {index} for node {node_id}")
                    continue

                try:
                    # Update node data with validation
                    if 'energy' in updates:
                        self.node_data[index, 0] = self.validate_float_internal(updates['energy'], -1000.0, 1000.0, 'energy')
                    if 'membrane_potential' in updates:
                        self.node_data[index, 1] = self.validate_float_internal(updates['membrane_potential'], -100.0, 100.0, 'membrane_potential')
                    if 'threshold' in updates:
                        self.node_data[index, 2] = self.validate_float_internal(updates['threshold'], -10.0, 10.0, 'threshold')
                    if 'refractory_timer' in updates:
                        self.node_data[index, 3] = self.validate_float_internal(updates['refractory_timer'], 0.0, 1000.0, 'refractory_timer')
                    if 'last_activation' in updates:
                        self.node_data[index, 4] = self.validate_float_internal(updates['last_activation'], 0.0, float('inf'), 'last_activation')
                    if 'plasticity_enabled' in updates:
                        self.node_data[index, 5] = 1.0 if updates['plasticity_enabled'] else 0.0
                    if 'eligibility_trace' in updates:
                        self.node_data[index, 6] = self.validate_float_internal(updates['eligibility_trace'], -100.0, 100.0, 'eligibility_trace')
                    if 'last_update' in updates:
                        self.node_data[index, 7] = time.time()  # Always use current time
                    if 'state' in updates:
                        self.node_data[index, 8] = 1.0 if updates['state'] == 'active' else 0.0

                    # Update metadata safely
                    if self.node_metadata[index]:
                        safe_updates = {}
                        for key, value in updates.items():
                            if key in ['type', 'behavior', 'state', 'x', 'y']:
                                if isinstance(value, (str, int, float)):
                                    safe_updates[key] = str(value)[:50]  # Limit string length
                        self.node_metadata[index].update(safe_updates)

                    updated_count += 1

                except Exception as e:
                    logging.error(f"Error updating node {node_id} at index {index}: {e}")
                    continue

            self.stats['operation_count'] += 1

            if updated_count > 0:
                log_step("Batch node update", count=updated_count)
    
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
        """Process batched node operations with comprehensive error handling."""
        if not isinstance(operation_type, str) or not operation_type:
            logging.error("Invalid operation_type provided")
            return

        if not callable(operation_func):
            logging.error("operation_func must be callable")
            return

        with self._lock:
            batch = self.batch_processor.get_batch(operation_type)
            if batch:
                try:
                    # Validate batch contents
                    if not all(isinstance(node_id, int) for node_id in batch):
                        logging.error(f"Invalid node IDs in {operation_type} batch")
                        return

                    operation_func(batch)
                    self.stats['operation_count'] += 1
                    log_step(f"Processed {operation_type} batch", count=len(batch))

                except Exception as e:
                    logging.error(f"Error processing {operation_type} batch: {e}")
                    # Re-queue failed batch for retry if it's recoverable
                    if len(batch) <= self.batch_processor.batch_size // 2:
                        logging.info(f"Re-queuing failed {operation_type} batch for retry")
                        self.batch_processor.add_to_batch(operation_type, batch)
    
    def remove_inactive_nodes(self):
        """Remove nodes that are no longer active."""
        with self._lock:
            inactive_nodes = []
            
            for node_id in list(self.active_nodes):
                index = self.node_id_to_index[node_id]
                if self.node_data[index, 8] < 0.5:  # state != active
                    inactive_nodes.append(node_id)
            
            for node_id in inactive_nodes:
                self.remove_node_internal(node_id)
            
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
        """Get performance statistics with thread safety."""
        with self._lock:
            current_time = time.time()
            time_diff = current_time - self.stats['last_update']

            # Avoid division by zero and ensure positive time difference
            if time_diff > 0.001:  # Minimum 1ms to avoid very small divisions
                operation_count = self.stats.get('operation_count', 0)
                self.stats['operations_per_second'] = operation_count / time_diff
            else:
                self.stats['operations_per_second'] = 0.0

            self.stats['last_update'] = current_time

            # Calculate memory usage safely
            try:
                memory_usage_mb = self.node_data.nbytes / (1024 * 1024)
            except AttributeError:
                memory_usage_mb = 0.0

            return {
                'total_nodes': self.stats['total_nodes'],
                'active_nodes': self.stats['active_nodes'],
                'memory_usage_mb': memory_usage_mb,
                'operations_per_second': self.stats['operations_per_second'],
                'spatial_index_cells': len(self.spatial_index.grid),
                'memory_limit_mb': self.get_memory_limit(),
                'memory_usage_percent': (memory_usage_mb / self.get_memory_limit() * 100) if self.get_memory_limit() > 0 else 0.0
            }
    
    def cleanup(self):
        """Clean up resources efficiently."""
        with self._lock:
            self.spatial_index = SpatialIndex()
            self.active_nodes.clear()
            self.node_id_to_index.clear()
            self.index_to_node_id.clear()
            self.node_types.clear()

            # Reset arrays efficiently
            try:
                self.node_data.fill(0.0)
                self.node_positions.fill(0.0)
                # Use slice assignment for better performance
                self.node_metadata[:] = [None] * len(self.node_metadata)
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
                # Fallback cleanup
                self.node_metadata = [None] * self.max_nodes

            # Reset stats
            self.stats = {
                'total_nodes': 0,
                'active_nodes': 0,
                'operations_per_second': 0.0,
                'last_update': time.time(),
                'operation_count': 0
            }

            log_step("OptimizedNodeManager cleanup completed")

    def set_memory_limit(self, limit_mb: int):
        """Set memory limit in MB."""
        if limit_mb <= 0:
            raise ValueError("Memory limit must be positive")
        with self.get_lock():
            self.set_memory_limit_internal(limit_mb)
            log_step("Memory limit updated", limit_mb=limit_mb)

    def validate_memory_usage(self) -> bool:
        """Validate current memory usage against limits."""
        with self.get_lock():
            current_mb = self.node_data.nbytes / (1024 * 1024)
            if current_mb > self.get_memory_limit():
                logging.warning(f"Memory usage ({current_mb:.1f}MB) exceeds limit ({self.get_memory_limit()}MB)")
                return False
            return True

# Global optimized node manager instance
_global_node_manager = None
_global_lock = threading.Lock()

def get_optimized_node_manager() -> OptimizedNodeManager:
    """Get the global optimized node manager instance with thread safety."""
    global _global_node_manager
    if _global_node_manager is None:
        with _global_lock:
            if _global_node_manager is None:  # Double-check locking
                _global_node_manager = OptimizedNodeManager()
    return _global_node_manager






