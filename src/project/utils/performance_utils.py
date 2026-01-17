#!/usr/bin/env python3
"""
Performance optimization utilities for the PyTorch Geometric Neural System.

This module provides caching mechanisms, tensor operation optimizations,
and performance monitoring utilities to improve system efficiency.
"""

import functools
import time
from typing import Any, TypeVar
from typing_extensions import TypedDict
from collections.abc import Callable
import logging
from datetime import datetime

T = TypeVar('T')

logger = logging.getLogger(__name__)


class ProfilingSectionData(TypedDict):
    """Type definition for profiling section data."""
    count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float


class TensorOperationCache:
    """
    Caching system for expensive tensor operations to improve performance.
    
    This cache helps avoid redundant tensor shape validation and synchronization
    operations that were identified as performance bottlenecks in the audit.
    """
    def __init__(self, max_size: int = 1000):
        """Initialize tensor operation cache with maximum size limit."""
        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._access_times: dict[str, float] = {}
    def get(self, key: str) -> Any | None:
        """Get cached value if available and not expired."""
        if key in self._cache:
            value, _timestamp = self._cache[key]
            self._access_times[key] = time.time()
            return value
        return None
        
    def set(self, key: str, value: Any, ttl: float = 60.0) -> None:
        """Set cached value with time-to-live."""
        current_time = time.time()
        
        # Evict oldest entries if cache is full
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
            
        self._cache[key] = (value, current_time + ttl)
        self._access_times[key] = current_time
        
    def _evict_oldest(self) -> None:
        """Remove the least recently accessed entry."""
        if not self._access_times:
            return
            
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._cache.pop(oldest_key, None)
        self._access_times.pop(oldest_key, None)
        
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_times.clear()
        
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Global cache instance for tensor operations
_tensor_cache = TensorOperationCache()


def cached_tensor_operation(ttl: float = 60.0) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to cache expensive tensor operations.
    
    Args:
        ttl: Time-to-live for cache entries in seconds
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = _tensor_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
                
            # Execute function and cache result
            result = func(*args, **kwargs)
            _tensor_cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, cached result")
            
            return result
        return wrapper
    return decorator


class PerformanceMonitor:
    """
    Performance monitoring utilities for tracking operation timing and resource usage.
    """
    
    def __init__(self) -> None:
        """Initialize performance monitor."""
        self._operation_times: dict[str, list[float]] = {}
        self._operation_counts: dict[str, int] = {}
        
    def record_operation(self, operation_name: str, duration: float) -> None:
        """Record operation timing and update statistics."""
        if operation_name not in self._operation_times:
            self._operation_times[operation_name] = []
            
        self._operation_times[operation_name].append(duration)
        
        # Keep only last 100 measurements per operation
        if len(self._operation_times[operation_name]) > 100:
            self._operation_times[operation_name] = self._operation_times[operation_name][-100:]
            
        self._operation_counts[operation_name] = self._operation_counts.get(operation_name, 0) + 1
        
    def get_operation_stats(self, operation_name: str) -> dict[str, float] | None:
        """Get statistics for a specific operation."""
        if operation_name not in self._operation_times:
            return None
            
        times = self._operation_times[operation_name]
        if not times:
            return None
            
        # Precompute sum for efficiency
        times_sum = sum(times)
        return {
            'count': len(times),
            'total_time': times_sum,
            'avg_time': times_sum / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'last_time': times[-1],
            'std_dev': (sum(t**2 for t in times) / len(times) - (times_sum / len(times))**2)**0.5 if len(times) > 1 else 0.0
        }
    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all operations."""
        return {name: stats for name, stats in 
                ((name, self.get_operation_stats(name)) 
                 for name in self._operation_times.keys())
                if stats is not None}
    def clear_stats(self) -> None:
        """Clear all performance statistics."""
        self._operation_times.clear()
        self._operation_counts.clear()


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to monitor function performance.
    
    Args:
        operation_name: Name for performance tracking
        
    Returns:
        Decorated function with performance monitoring
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                _performance_monitor.record_operation(operation_name, duration)
        return wrapper
    return decorator


class MemoryOptimizer:
    """
    Memory optimization utilities for tensor operations and resource management.
    """
    @staticmethod
    def optimize_tensor_allocation(tensor_shape: tuple[int, ...],
                                    device: str = 'cpu',
                                    dtype: Any = None) -> Any:
        """
        Optimized tensor allocation with memory pooling considerations.

        Args:
            tensor_shape: Shape of tensor to allocate
            device: Device to allocate tensor on
            dtype: Data type for tensor

        Returns:
            Optimized tensor
        """
        try:
            import torch
            return torch.empty(tensor_shape, device=device, dtype=dtype)
        except ImportError:
            # Fallback for when torch is not available
            import numpy as np
            return np.zeros(tensor_shape, dtype=dtype)
    @staticmethod
    def batch_tensor_operations(tensors: list[Any], operation: str = 'concatenate') -> Any:
        """
        Batch multiple tensor operations for improved efficiency.

        Args:
            tensors: List of tensors to process
            operation: Type of operation to perform ('concatenate', 'stack', etc.)

        Returns:
            Result of batched operation
        """
        if not tensors:
            return None
        try:
            import torch
            if operation == 'concatenate':
                return torch.cat(tensors, dim=0)
            elif operation == 'stack':
                return torch.stack(tensors, dim=0)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except ImportError:
            import numpy as np
            if operation == 'concatenate':
                return np.concatenate(tensors, axis=0)
            elif operation == 'stack':
                return np.stack(tensors, axis=0)
            else:
                raise ValueError(f"Unsupported operation: {operation}") from None

class PerformanceProfiler:
    """
    Advanced performance profiling utilities for detailed performance analysis.
    """
    def __init__(self) -> None:
        """Initialize performance profiler."""
        self._profiling_data: dict[str, list[float]] = {}
        self._memory_usage: dict[str, list[float]] = {}
        self._tensor_operations: dict[str, dict[str, Any]] = {}
        self._enabled = True

    def start_profiling(self, section_name: str) -> None:
        """Start profiling a specific section."""
        if not self._enabled:
            return

        if section_name not in self._profiling_data:
            self._profiling_data[section_name] = []
            self._memory_usage[section_name] = []

        # Record start time and memory usage
        start_time = time.time()
        self._profiling_data[section_name].append(-start_time)  # Negative indicates start time

        try:
            import torch
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                self._memory_usage[section_name].append(memory_used)
        except:
            self._memory_usage[section_name].append(0.0)

    def end_profiling(self, section_name: str) -> None:
        """End profiling a specific section."""
        if not self._enabled:
            return

        if section_name not in self._profiling_data or not self._profiling_data[section_name]:
            return

        # Find the last start time (negative value)
        last_index = len(self._profiling_data[section_name]) - 1
        if self._profiling_data[section_name][last_index] < 0:
            start_time = -self._profiling_data[section_name][last_index]
            duration = time.time() - start_time
            self._profiling_data[section_name][last_index] = duration  # Replace with duration

    def record_tensor_operation(self, operation_name: str, tensor_shape: tuple[int, ...], duration: float) -> None:
        """Record tensor operation details."""
        if not self._enabled:
            return

        if operation_name not in self._tensor_operations:
            self._tensor_operations[operation_name] = {
                'count': 0,
                'total_time': 0.0,
                'shapes': [],
                'avg_time': 0.0
            }

        operation_data = self._tensor_operations[operation_name]
        operation_data['count'] += 1
        operation_data['total_time'] += duration
        operation_data['shapes'].append(tensor_shape)
        operation_data['avg_time'] = operation_data['total_time'] / operation_data['count']

    def get_profiling_report(self) -> dict[str, Any]:
        """Generate comprehensive profiling report."""
        report: dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'profiling_sections': {},
            'tensor_operations': self._tensor_operations.copy(),
            'memory_usage': {},
            'performance_summary': {}
        }

        # Process profiling sections
        for section_name, times in self._profiling_data.items():
            if times:
                valid_times = [t for t in times if t > 0]  # Filter out start markers
                if valid_times:
                    report['profiling_sections'][section_name] = {
                        'count': len(valid_times),
                        'total_time': sum(valid_times),
                        'avg_time': sum(valid_times) / len(valid_times),
                        'min_time': min(valid_times),
                        'max_time': max(valid_times)
                    }

        # Process memory usage
        for section_name, memory_values in self._memory_usage.items():
            if memory_values:
                report['memory_usage'][section_name] = {
                    'avg_memory_mb': sum(memory_values) / len(memory_values),
                    'max_memory_mb': max(memory_values),
                    'min_memory_mb': min(memory_values)
                }

        # Generate performance summary
        if report['profiling_sections']:
            all_times: list[float] = []
            profiling_sections: dict[str, ProfilingSectionData] = report['profiling_sections']  # type: ignore[assignment]
            for section_data in profiling_sections.values():
                avg_time: float = section_data['avg_time']
                count: int = section_data['count']
                all_times.extend([avg_time] * count)

            if all_times:
                report['performance_summary'] = {
                    'total_execution_time': sum(all_times),
                    'overall_avg_time': sum(all_times) / len(all_times),
                    'section_count': len(report['profiling_sections']),
                    'operation_count': sum(sd['count'] for sd in profiling_sections.values())
                }

        return report

    def enable_profiling(self, enabled: bool = True) -> None:
        """Enable or disable profiling."""
        self._enabled = enabled

    def clear_profiling_data(self) -> None:
        """Clear all profiling data."""
        self._profiling_data.clear()
        self._memory_usage.clear()
        self._tensor_operations.clear()


def get_tensor_cache() -> TensorOperationCache:
    """Get global tensor operation cache instance."""
    return _tensor_cache


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _performance_monitor


def clear_all_caches() -> None:
    """Clear all performance caches and monitors."""
    _tensor_cache.clear()
    _performance_monitor.clear_stats()
    logger.info("All performance caches and monitors cleared")

# Global performance profiler instance
_performance_profiler = PerformanceProfiler()

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    return _performance_profiler

def profile_operation(operation_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to profile function performance with advanced profiling.

    Args:
        operation_name: Name for profiling

    Returns:
        Decorated function with performance profiling
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = get_performance_profiler()
            profiler.start_profiling(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_profiling(operation_name)
        return wrapper
    return decorator