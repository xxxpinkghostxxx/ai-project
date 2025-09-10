"""
memory_leak_detector.py

Memory leak detection and prevention system for the AI neural project.
Provides tools to detect, monitor, and prevent memory leaks in long-running simulations.
"""

import gc
import sys
import threading
import time
import weakref
from typing import Dict, List, Any, Optional, Set, Callable
from collections import defaultdict
import psutil
import tracemalloc
from logging_utils import log_step


class MemoryLeakDetector:
    """
    Memory leak detection and monitoring system.
    Tracks object references, memory usage, and identifies potential leaks.
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize the memory leak detector.
        
        Args:
            check_interval: Interval in seconds between memory checks
        """
        import threading
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.memory_history = []
        self.object_counts = defaultdict(int)
        self.weak_refs = weakref.WeakSet()
        self.circular_ref_detector = CircularReferenceDetector()
        self._lock = threading.RLock()  # Thread safety lock
        
        # Memory tracking
        self.peak_memory = 0
        self.initial_memory = self._get_memory_usage()
        
        # Start memory tracing
        tracemalloc.start()
        
        log_step("MemoryLeakDetector initialized", 
                check_interval=check_interval,
                initial_memory=self.initial_memory)
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        with self._lock:
            if self.is_monitoring:
                return
            
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            log_step("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        with self._lock:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            
            log_step("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                self._check_memory_usage()
                self._detect_circular_references()
                self._track_object_counts()
                time.sleep(self.check_interval)
            except Exception as e:
                log_step("Memory monitoring error", error=str(e))
                time.sleep(1.0)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _check_memory_usage(self):
        """Check for memory usage anomalies."""
        current_memory = self._get_memory_usage()
        self.memory_history.append(current_memory)
        
        # Keep only last 100 measurements
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        # Update peak memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        # Check for memory growth trend
        if len(self.memory_history) >= 10:
            recent_avg = sum(self.memory_history[-10:]) / 10
            older_avg = sum(self.memory_history[-20:-10]) / 10 if len(self.memory_history) >= 20 else recent_avg
            
            growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            
            if growth_rate > 0.1:  # 10% growth
                log_step("Memory growth detected", 
                        current_memory=current_memory,
                        growth_rate=growth_rate,
                        peak_memory=self.peak_memory)
    
    def _detect_circular_references(self):
        """Detect circular references that could cause memory leaks."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Check for uncollectable objects
            uncollectable = len(gc.garbage)
            if uncollectable > 0:
                log_step("Uncollectable objects detected", count=uncollectable)
                
                # Log details of uncollectable objects
                for obj in gc.garbage[:5]:  # Log first 5
                    log_step("Uncollectable object", 
                            type=type(obj).__name__,
                            id=id(obj))
            
            # Check for reference cycles
            cycles = gc.collect()
            if cycles > 0:
                log_step("Reference cycles detected", cycles=cycles)
                
        except Exception as e:
            log_step("Circular reference detection error", error=str(e))
    
    def _track_object_counts(self):
        """Track object counts to detect unusual growth."""
        try:
            # Count objects by type
            current_counts = defaultdict(int)
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                current_counts[obj_type] += 1
            
            # Compare with previous counts
            for obj_type, count in current_counts.items():
                if obj_type in self.object_counts:
                    growth = count - self.object_counts[obj_type]
                    if growth > 1000:  # Significant growth
                        log_step("Object count growth detected", 
                                type=obj_type,
                                count=count,
                                growth=growth)
            
            self.object_counts = current_counts
            
        except Exception as e:
            log_step("Object count tracking error", error=str(e))
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        current_memory = self._get_memory_usage()
        
        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_growth_mb': current_memory - self.initial_memory,
            'memory_history': self.memory_history.copy(),
            'object_counts': dict(self.object_counts),
            'uncollectable_objects': len(gc.garbage),
            'reference_cycles': len(gc.get_objects()) - len(set(id(obj) for obj in gc.get_objects()))
        }
    
    def force_cleanup(self):
        """Force garbage collection and cleanup."""
        try:
            # Clear weak references
            self.weak_refs.clear()
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            # Clear memory history if it's too large
            if len(self.memory_history) > 50:
                self.memory_history = self.memory_history[-25:]
            
            log_step("Forced memory cleanup completed")
            
        except Exception as e:
            log_step("Memory cleanup error", error=str(e))
    
    def register_weak_reference(self, obj: Any):
        """Register an object with weak reference for leak detection."""
        try:
            self.weak_refs.add(obj)
        except Exception as e:
            log_step("Weak reference registration error", error=str(e))


class CircularReferenceDetector:
    """Detects circular references that could cause memory leaks."""
    
    def __init__(self):
        self.visited = set()
        self.path = []
    
    def detect_cycles(self, obj: Any, max_depth: int = 10) -> List[List[Any]]:
        """
        Detect circular references starting from an object.
        
        Args:
            obj: Object to start detection from
            max_depth: Maximum depth to search
            
        Returns:
            List of circular reference paths
        """
        self.visited.clear()
        self.path.clear()
        cycles = []
        
        self._dfs_detect(obj, max_depth, cycles)
        return cycles
    
    def _dfs_detect(self, obj: Any, depth: int, cycles: List[List[Any]]):
        """Depth-first search for circular references."""
        if depth <= 0 or id(obj) in self.visited:
            return
        
        if obj in self.path:
            # Found a cycle
            cycle_start = self.path.index(obj)
            cycle = self.path[cycle_start:] + [obj]
            cycles.append(cycle)
            return
        
        self.visited.add(id(obj))
        self.path.append(obj)
        
        try:
            # Check object attributes
            if hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    if isinstance(attr_value, (list, tuple, set)):
                        for item in attr_value:
                            self._dfs_detect(item, depth - 1, cycles)
                    elif hasattr(attr_value, '__dict__'):
                        self._dfs_detect(attr_value, depth - 1, cycles)
        except Exception:
            # Skip objects that can't be inspected
            pass
        
        self.path.pop()


class MemoryLeakPrevention:
    """Utilities to prevent memory leaks in common patterns."""
    
    @staticmethod
    def create_weak_callback(callback: Callable) -> Callable:
        """Create a weak reference callback to prevent circular references."""
        def weak_callback(*args, **kwargs):
            try:
                return callback(*args, **kwargs)
            except ReferenceError:
                # Callback object was garbage collected
                pass
        return weak_callback
    
    @staticmethod
    def cleanup_resources(resources: List[Any]):
        """Clean up a list of resources that might hold references."""
        for resource in resources:
            try:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'clear'):
                    resource.clear()
            except Exception as e:
                log_step("Resource cleanup error", error=str(e))
    
    @staticmethod
    def break_circular_references(obj: Any):
        """Break circular references in an object graph."""
        try:
            if hasattr(obj, '__dict__'):
                # Clear references to parent objects
                for attr_name in list(obj.__dict__.keys()):
                    attr_value = obj.__dict__[attr_name]
                    if hasattr(attr_value, '__dict__') and hasattr(attr_value, 'parent'):
                        if attr_value.parent is obj:
                            attr_value.parent = None
        except Exception as e:
            log_step("Circular reference breaking error", error=str(e))


# Global memory leak detector instance
_memory_detector: Optional[MemoryLeakDetector] = None
_detector_lock = threading.Lock()


def get_memory_detector() -> MemoryLeakDetector:
    """Get the global memory leak detector instance."""
    global _memory_detector
    if _memory_detector is None:
        with _detector_lock:
            if _memory_detector is None:
                _memory_detector = MemoryLeakDetector()
    return _memory_detector


def start_memory_monitoring():
    """Start global memory monitoring."""
    detector = get_memory_detector()
    detector.start_monitoring()


def stop_memory_monitoring():
    """Stop global memory monitoring."""
    global _memory_detector
    if _memory_detector:
        _memory_detector.stop_monitoring()


def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics."""
    detector = get_memory_detector()
    return detector.get_memory_stats()


def force_memory_cleanup():
    """Force memory cleanup."""
    detector = get_memory_detector()
    detector.force_cleanup()


# Context manager for memory leak detection
class MemoryLeakContext:
    """Context manager for detecting memory leaks in specific code blocks."""
    
    def __init__(self, description: str = "operation"):
        self.description = description
        self.initial_memory = 0
        self.initial_objects = 0
    
    def __enter__(self):
        # Force garbage collection before starting
        gc.collect()
        
        self.initial_memory = get_memory_stats()['current_memory_mb']
        self.initial_objects = len(gc.get_objects())
        
        log_step(f"Memory leak context started: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Force garbage collection after operation
        gc.collect()
        
        final_memory = get_memory_stats()['current_memory_mb']
        final_objects = len(gc.get_objects())
        
        memory_growth = final_memory - self.initial_memory
        object_growth = final_objects - self.initial_objects
        
        if memory_growth > 10:  # More than 10MB growth
            log_step(f"Potential memory leak detected in {self.description}",
                    memory_growth=memory_growth,
                    object_growth=object_growth)
        
        log_step(f"Memory leak context ended: {self.description}")


# Decorator for automatic memory leak detection
def detect_memory_leaks(description: str = "function"):
    """Decorator to detect memory leaks in functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with MemoryLeakContext(f"{description}: {func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator
