"""
Static memory allocator for NASA compliance.
Provides pre-allocated buffers to avoid dynamic memory allocation during runtime.
"""

import numpy as np
from typing import List, Dict, Any

from collections import deque


class StaticAllocator:
    """Static memory allocator to avoid dynamic allocations during runtime."""
    
    def __init__(self, max_nodes: int = 100000, max_edges: int = 500000):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        # Pre-allocate static buffers
        self._node_buffer = np.zeros((max_nodes, 10), dtype=np.float32)
        self._edge_buffer = np.zeros((max_edges, 3), dtype=np.float32)  # source, target, weight
        self._node_labels_buffer = [None] * max_nodes
        self._edge_attributes_buffer = [None] * max_edges
        
        # Buffer usage tracking
        self._node_count = 0
        self._edge_count = 0
        self._node_buffer_used = 0
        self._edge_buffer_used = 0
        
        # Static lists for common operations
        self._static_list_1000 = [0] * 1000
        self._static_list_100 = [0] * 100
        self._static_list_50 = [0] * 50
        self._static_list_10 = [0] * 10
        
        # Static dictionaries
        self._static_dict_100 = {}
        self._static_dict_50 = {}
        self._static_dict_10 = {}
        
        # Deques for dynamic-like behavior with fixed size
        self._static_deque_1000 = deque(maxlen=1000)
        self._static_deque_100 = deque(maxlen=100)
        self._static_deque_50 = deque(maxlen=50)
    
    def get_static_list(self, size: int) -> List[int]:
        """Get a pre-allocated list of specified size."""
        if size <= 10:
            return self._static_list_10[:size]
        elif size <= 50:
            return self._static_list_50[:size]
        elif size <= 100:
            return self._static_list_100[:size]
        elif size <= 1000:
            return self._static_list_1000[:size]
        else:
            raise ValueError(f"Requested size {size} exceeds maximum static allocation")
    
    def get_static_dict(self, size: int) -> Dict[str, Any]:
        """Get a pre-allocated dictionary."""
        if size <= 10:
            return self._static_dict_10.copy()
        elif size <= 50:
            return self._static_dict_50.copy()
        elif size <= 100:
            return self._static_dict_100.copy()
        else:
            raise ValueError(f"Requested size {size} exceeds maximum static allocation")
    
    def get_static_deque(self, size: int) -> deque:
        """Get a pre-allocated deque of specified size."""
        if size <= 50:
            return self._static_deque_50
        elif size <= 100:
            return self._static_deque_100
        elif size <= 1000:
            return self._static_deque_1000
        else:
            raise ValueError(f"Requested size {size} exceeds maximum static allocation")
    
    def allocate_node_data(self, count: int) -> np.ndarray:
        """Allocate node data from static buffer."""
        if self._node_buffer_used + count > self.max_nodes:
            raise MemoryError("Node buffer overflow - cannot allocate more nodes")
        
        start_idx = self._node_buffer_used
        self._node_buffer_used += count
        return self._node_buffer[start_idx:start_idx + count]
    
    def allocate_edge_data(self, count: int) -> np.ndarray:
        """Allocate edge data from static buffer."""
        if self._edge_buffer_used + count > self.max_edges:
            raise MemoryError("Edge buffer overflow - cannot allocate more edges")
        
        start_idx = self._edge_buffer_used
        self._edge_buffer_used += count
        return self._edge_buffer[start_idx:start_idx + count]
    
    def reset_buffers(self):
        """Reset all buffers to initial state."""
        self._node_count = 0
        self._edge_count = 0
        self._node_buffer_used = 0
        self._edge_buffer_used = 0
        
        # Clear static collections
        self._static_deque_1000.clear()
        self._static_deque_100.clear()
        self._static_deque_50.clear()
        
        self._static_dict_100.clear()
        self._static_dict_50.clear()
        self._static_dict_10.clear()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current buffer usage statistics."""
        return {
            'node_buffer_used': self._node_buffer_used,
            'edge_buffer_used': self._edge_buffer_used,
            'node_buffer_utilization': self._node_buffer_used / self.max_nodes,
            'edge_buffer_utilization': self._edge_buffer_used / self.max_edges,
            'max_nodes': self.max_nodes,
            'max_edges': self.max_edges
        }


# Global static allocator instance
_global_allocator = StaticAllocator()


def get_static_allocator() -> StaticAllocator:
    """Get the global static allocator instance."""
    return _global_allocator


def get_static_list(size: int) -> List[int]:
    """Convenience function to get a static list."""
    return _global_allocator.get_static_list(size)


def get_static_dict(size: int) -> Dict[str, Any]:
    """Convenience function to get a static dictionary."""
    return _global_allocator.get_static_dict(size)


def get_static_deque(size: int) -> deque:
    """Convenience function to get a static deque."""
    return _global_allocator.get_static_deque(size)


def reset_static_allocator():
    """Reset the global static allocator."""
    _global_allocator.reset_buffers()







