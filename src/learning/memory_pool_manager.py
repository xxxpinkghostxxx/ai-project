"""
Memory Pool Manager for efficient object reuse and memory optimization.
Implements object pooling patterns to reduce garbage collection pressure.
"""

import threading
import time
from typing import Dict, List, Any, Optional, TypeVar, Generic
from collections import defaultdict, deque
from dataclasses import dataclass
import weakref
from src.utils.unified_error_handler import ErrorSeverity

T = TypeVar('T')

@dataclass
class PoolStats:
    """Statistics for memory pool usage."""
    total_objects: int = 0
    available_objects: int = 0
    in_use_objects: int = 0
    creation_count: int = 0
    reuse_count: int = 0
    last_cleanup: float = 0.0

class ObjectPool(Generic[T]):
    """Generic object pool for efficient object reuse."""
    
    def __init__(self, factory_func, max_size: int = 1000, cleanup_interval: float = 60.0):
        self.factory_func = factory_func
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.available_objects = deque()
        self.in_use_objects = set()
        self.stats = PoolStats()
        self._lock = threading.RLock()
        self.last_cleanup = time.time()
    
    def get_object(self) -> T:
        """Get an object from the pool or create a new one."""
        with self._lock:
            if self.available_objects:
                obj = self.available_objects.popleft()
                self.stats.reuse_count += 1
            else:
                obj = self.factory_func()
                self.stats.creation_count += 1
            
            self.in_use_objects.add(id(obj))
            self.stats.in_use_objects = len(self.in_use_objects)
            self.stats.available_objects = len(self.available_objects)
            return obj
    
    def return_object(self, obj: T) -> None:
        """Return an object to the pool for reuse."""
        with self._lock:
            obj_id = id(obj)
            if obj_id in self.in_use_objects:
                self.in_use_objects.remove(obj_id)
                
                if len(self.available_objects) < self.max_size:
                    # Reset object state if it has a reset method
                    if hasattr(obj, 'reset'):
                        obj.reset()
                    self.available_objects.append(obj)
                
                self.stats.in_use_objects = len(self.in_use_objects)
                self.stats.available_objects = len(self.available_objects)
    
    def cleanup_expired_objects(self) -> int:
        """Clean up objects that have been in the pool too long."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return 0
        
        cleaned_count = 0
        with self._lock:
            # Remove excess objects beyond max_size
            while len(self.available_objects) > self.max_size // 2:
                self.available_objects.popleft()
                cleaned_count += 1
            
            self.last_cleanup = current_time
            self.stats.last_cleanup = current_time
        
        return cleaned_count
    
    def get_stats(self) -> PoolStats:
        """Get current pool statistics."""
        with self._lock:
            return PoolStats(
                total_objects=len(self.available_objects) + len(self.in_use_objects),
                available_objects=len(self.available_objects),
                in_use_objects=len(self.in_use_objects),
                creation_count=self.stats.creation_count,
                reuse_count=self.stats.reuse_count,
                last_cleanup=self.stats.last_cleanup
            )

class MemoryPoolManager:
    """Centralized memory pool manager for the simulation system."""
    
    def __init__(self):
        self.pools: Dict[str, ObjectPool] = {}
        self._lock = threading.RLock()
        self.cleanup_thread = None
        self.running = False
    
    def create_pool(self, name: str, factory_func, max_size: int = 1000) -> ObjectPool:
        """Create a new object pool."""
        with self._lock:
            if name in self.pools:
                raise ValueError(f"Pool '{name}' already exists")
            
            pool = ObjectPool(factory_func, max_size)
            self.pools[name] = pool
            return pool
    
    def get_pool(self, name: str) -> Optional[ObjectPool]:
        """Get an existing pool by name."""
        with self._lock:
            return self.pools.get(name)
    
    def get_object(self, pool_name: str) -> Any:
        """Get an object from a specific pool."""
        pool = self.get_pool(pool_name)
        if pool is None:
            raise ValueError(f"Pool '{pool_name}' not found")
        return pool.get_object()
    
    def return_object(self, pool_name: str, obj: Any) -> None:
        """Return an object to a specific pool."""
        pool = self.get_pool(pool_name)
        if pool is None:
            raise ValueError(f"Pool '{pool_name}' not found")
        pool.return_object(obj)

    def get_pooled_object(self, pool_name: str) -> 'PooledObject':
        """Get a pooled object with automatic return."""
        obj = self.get_object(pool_name)
        return PooledObject(pool_name, obj)

    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
    
    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=10.0)  # Increased timeout for safety
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.running:
            try:
                self.cleanup_all_pools()
                time.sleep(30)  # Cleanup every 30 seconds
            except Exception as e:
                from src.utils.unified_error_handler import get_error_handler
                error_handler = get_error_handler()
                error_handler.handle_error(e, "memory_pool_cleanup_loop", severity=ErrorSeverity.MEDIUM)
                time.sleep(5)
    
    def cleanup_all_pools(self) -> int:
        """Clean up all pools and return total objects cleaned."""
        total_cleaned = 0
        with self._lock:
            for pool in self.pools.values():
                total_cleaned += pool.cleanup_expired_objects()
        return total_cleaned
    
    def get_all_stats(self) -> Dict[str, PoolStats]:
        """Get statistics for all pools."""
        with self._lock:
            return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    def cleanup(self) -> None:
        """Clean up all pools and stop background threads."""
        self.stop_cleanup_thread()
        with self._lock:
            for pool in self.pools.values():
                pool.available_objects.clear()
                pool.in_use_objects.clear()
            self.pools.clear()

# Global memory pool manager instance
_memory_pool_manager = None

def get_memory_pool_manager() -> MemoryPoolManager:
    """Get the global memory pool manager instance."""
    global _memory_pool_manager
    if _memory_pool_manager is None:
        _memory_pool_manager = MemoryPoolManager()
        _memory_pool_manager.start_cleanup_thread()
    return _memory_pool_manager

def cleanup_memory_pools() -> None:
    """Clean up all memory pools."""
    global _memory_pool_manager
    if _memory_pool_manager is not None:
        _memory_pool_manager.cleanup()
        _memory_pool_manager = None

# Convenience functions for common object types
def create_node_pool(max_size: int = 10000) -> ObjectPool:
    """Create a pool for neural nodes."""
    def create_node():
        return {
            'id': 0,
            'type': 'dynamic',
            'energy': 0.0,
            'state': 'inactive',
            'membrane_potential': 0.0,
            'threshold': 0.5,
            'refractory_timer': 0.0,
            'last_activation': 0,
            'plasticity_enabled': True,
            'eligibility_trace': 0.0,
            'last_update': 0
        }
    
    manager = get_memory_pool_manager()
    return manager.create_pool('neural_nodes', create_node, max_size)

def create_edge_pool(max_size: int = 50000) -> ObjectPool:
    """Create a pool for neural edges."""
    def create_edge():
        return {
            'source': 0,
            'target': 0,
            'weight': 1.0,
            'type': 'excitatory',
            'delay': 0.0,
            'plasticity_tag': False,
            'eligibility_trace': 0.0,
            'last_activity': 0.0,
            'activation_count': 0
        }
    
    manager = get_memory_pool_manager()
    return manager.create_pool('neural_edges', create_edge, max_size)

def create_event_pool(max_size: int = 1000) -> ObjectPool:
    """Create a pool for neural events."""
    def create_event():
        return {
            'event_type': 'spike',
            'timestamp': 0.0,
            'source_node_id': 0,
            'target_node_id': 0,
            'data': {},
            'priority': 0
        }
    
    manager = get_memory_pool_manager()
    return manager.create_pool('neural_events', create_event, max_size)

# Context manager for automatic object return
class PooledObject:
    """Context manager for automatic object return to pool."""
    
    def __init__(self, pool_name: str, obj: Any):
        self.pool_name = pool_name
        self.obj = obj
        self.manager = get_memory_pool_manager()
    
    def __enter__(self):
        return self.obj
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.return_object(self.pool_name, self.obj)

def get_pooled_object(pool_name: str) -> PooledObject:
    """Get a pooled object with automatic return."""
    manager = get_memory_pool_manager()
    obj = manager.get_object(pool_name)
    return PooledObject(pool_name, obj)







