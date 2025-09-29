"""
High-performance caching system for neural simulation optimization.
Provides LRU caching, batch operations, and memory-efficient data structures.
"""

import threading
import time
from typing import Dict, List, Any, Optional, Callable
from collections import OrderedDict

class LRUCache:
    """Thread-safe LRU cache with size limits."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[Any, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get an item from the cache."""
        with self._lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.timestamps[key] > self.ttl:
                    self._remove(key)
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: Any, value: Any):
        """Put an item in the cache."""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key, _ = self.cache.popitem(last=False)
                    if oldest_key in self.timestamps:
                        del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def _remove(self, key: Any):
        """Remove an item from the cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear_expired(self):
        """Clear expired items."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                self._remove(key)
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self.cache)

class BatchOperationCache:
    """Cache for batch operations to reduce redundant computations."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.operation_cache: Dict[str, Dict[str, Any]] = {}
        self.pending_operations: Dict[str, List[tuple]] = {}
        self._lock = threading.RLock()
    
    def add_operation(self, operation_type: str, operation_data: tuple):
        """Add an operation to be batched."""
        with self._lock:
            if operation_type not in self.pending_operations:
                self.pending_operations[operation_type] = []
            
            self.pending_operations[operation_type].append(operation_data)
            
            # Process batch if full
            if len(self.pending_operations[operation_type]) >= self.batch_size:
                self._process_batch(operation_type)
    
    def _process_batch(self, operation_type: str):
        """Process a batch of operations."""
        operations = self.pending_operations[operation_type]
        self.pending_operations[operation_type] = []
        
        # Group similar operations
        grouped_ops = self._group_similar_operations(operations)
        
        for group_key, group_ops in grouped_ops.items():
            # Cache the result
            cache_key = f"{operation_type}:{group_key}"
            if cache_key not in self.operation_cache:
                result = self._execute_batch_operation(operation_type, group_ops)
                self.operation_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time(),
                    'count': len(group_ops)
                }
    
    def _group_similar_operations(self, operations: List[tuple]) -> Dict[str, List[tuple]]:
        """Group similar operations for batch processing."""
        groups = {}
        for op in operations:
            # Simple grouping by first element (can be made more sophisticated)
            group_key = str(op[0]) if op else 'default'
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(op)
        return groups
    
    def _execute_batch_operation(self, operation_type: str, operations: List[tuple]) -> Any:
        """Execute a batch of operations."""
        # This would be implemented based on the specific operation type
        # For now, return a placeholder
        return f"batch_result_{operation_type}_{len(operations)}"
    
    def get_cached_result(self, operation_type: str, group_key: str) -> Optional[Any]:
        """Get a cached batch result."""
        with self._lock:
            cache_key = f"{operation_type}:{group_key}"
            if cache_key in self.operation_cache:
                return self.operation_cache[cache_key]['result']
            return None

class MemoryEfficientDataStore:
    """Memory-efficient data store with compression and deduplication."""
    
    def __init__(self):
        self.data_store: Dict[str, Any] = {}
        self.reference_counts: Dict[str, int] = {}
        self.compressed_data: Dict[str, bytes] = {}
        self._lock = threading.RLock()
    
    def store_data(self, key: str, data: Any, compress: bool = False):
        """Store data with optional compression."""
        with self._lock:
            # Check for duplicate data
            data_hash = self._hash_data(data)
            if data_hash in self.reference_counts:
                # Data already exists, just increment reference
                self.reference_counts[data_hash] += 1
                self.data_store[key] = data_hash
                return
            
            # Store new data
            if compress and isinstance(data, (list, dict)):
                compressed = self._compress_data(data)
                self.compressed_data[data_hash] = compressed
                self.data_store[key] = data_hash
            else:
                self.data_store[key] = data
            
            self.reference_counts[data_hash] = 1
    
    def get_data(self, key: str) -> Optional[Any]:
        """Retrieve data from the store."""
        with self._lock:
            if key not in self.data_store:
                return None
            
            data_ref = self.data_store[key]
            
            if isinstance(data_ref, str) and data_ref in self.compressed_data:
                # Decompress data
                return self._decompress_data(self.compressed_data[data_ref])
            elif isinstance(data_ref, str) and data_ref in self.reference_counts:
                # Return reference (data is shared)
                return None  # Would need to store the actual data somewhere
            else:
                return data_ref
    
    def remove_data(self, key: str):
        """Remove data from the store."""
        with self._lock:
            if key in self.data_store:
                data_ref = self.data_store[key]
                
                if isinstance(data_ref, str) and data_ref in self.reference_counts:
                    self.reference_counts[data_ref] -= 1
                    
                    # If no more references, clean up
                    if self.reference_counts[data_ref] <= 0:
                        if data_ref in self.compressed_data:
                            del self.compressed_data[data_ref]
                        del self.reference_counts[data_ref]
                
                del self.data_store[key]
    
    def _hash_data(self, data: Any) -> str:
        """Generate a hash for data deduplication."""
        import hashlib
        data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for memory efficiency."""
        import pickle
        import zlib
        
        pickled = pickle.dumps(data)
        return zlib.compress(pickled)
    
    def _decompress_data(self, compressed: bytes) -> Any:
        """Decompress data."""
        import pickle
        import zlib
        
        decompressed = zlib.decompress(compressed)
        return pickle.loads(decompressed)

class PerformanceCacheManager:
    """Central manager for all caching systems."""
    
    def __init__(self):
        self.lru_cache = LRUCache(max_size=5000, ttl=600.0)  # 5k items, 10min TTL
        self.batch_cache = BatchOperationCache(batch_size=50)
        self.data_store = MemoryEfficientDataStore()
        
        # Performance tracking
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
            'memory_saved_mb': 0.0
        }
        
        self._lock = threading.RLock()
    
    def get_node_data(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get cached node data."""
        cache_key = f"node_data:{node_id}"
        cached = self.lru_cache.get(cache_key)
        
        if cached:
            self.stats['cache_hits'] += 1
            return cached
        else:
            self.stats['cache_misses'] += 1
            return None
    
    def cache_node_data(self, node_id: int, data: Dict[str, Any]):
        """Cache node data."""
        cache_key = f"node_data:{node_id}"
        self.lru_cache.put(cache_key, data)
    
    def get_connection_data(self, source_id: int, target_id: int) -> Optional[Dict[str, Any]]:
        """Get cached connection data."""
        cache_key = f"connection:{source_id}:{target_id}"
        cached = self.lru_cache.get(cache_key)
        
        if cached:
            self.stats['cache_hits'] += 1
            return cached
        else:
            self.stats['cache_misses'] += 1
            return None
    
    def cache_connection_data(self, source_id: int, target_id: int, data: Dict[str, Any]):
        """Cache connection data."""
        cache_key = f"connection:{source_id}:{target_id}"
        self.lru_cache.put(cache_key, data)
    
    def batch_node_updates(self, updates: List[tuple[int, Dict[str, Any]]]):
        """Batch node update operations."""
        for node_id, update_data in updates:
            self.batch_cache.add_operation('node_update', (node_id, update_data))
        
        self.stats['batch_operations'] += len(updates)
    
    def batch_connection_updates(self, updates: List[tuple[int, int, Dict[str, Any]]]):
        """Batch connection update operations."""
        for source_id, target_id, update_data in updates:
            self.batch_cache.add_operation('connection_update', (source_id, target_id, update_data))
        
        self.stats['batch_operations'] += len(updates)
    
    def store_frequent_data(self, key: str, data: Any, compress: bool = True):
        """Store frequently accessed data with compression."""
        self.data_store.store_data(key, data, compress)
    
    def get_frequent_data(self, key: str) -> Optional[Any]:
        """Retrieve frequently accessed data."""
        return self.data_store.get_data(key)
    
    def cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        self.lru_cache.clear_expired()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_hit_rate': hit_rate,
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'batch_operations': self.stats['batch_operations'],
                'lru_cache_size': self.lru_cache.size(),
                'memory_saved_mb': self.stats['memory_saved_mb']
            }
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self._lock:
            self.stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'batch_operations': 0,
                'memory_saved_mb': 0.0
            }

# Global performance cache manager instance
_global_cache_manager = None

def get_performance_cache_manager() -> PerformanceCacheManager:
    """Get the global performance cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = PerformanceCacheManager()
    return _global_cache_manager

def cached_node_data(func: Callable) -> Callable:
    """Decorator for caching node data operations."""
    def wrapper(*args, **kwargs):
        cache_manager = get_performance_cache_manager()
        
        # Extract node_id from arguments (assuming it's the first argument after self)
        if len(args) > 1 and isinstance(args[1], int):
            node_id = args[1]
            cache_key = f"node_data:{node_id}"
            
            # Try to get from cache first
            cached_result = cache_manager.lru_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Cache the result
        if len(args) > 1 and isinstance(args[1], int):
            node_id = args[1]
            cache_key = f"node_data:{node_id}"
            cache_manager.lru_cache.put(cache_key, result)
        
        return result
    
    return wrapper






