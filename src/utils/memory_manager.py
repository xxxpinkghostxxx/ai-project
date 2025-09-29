"""
Memory Manager
Provides comprehensive memory management for neural simulation systems.
"""

import gc
import psutil
import threading
import time
from typing import Dict, List, Any, Callable
from src.utils.logging_utils import log_step
from src.utils.reader_writer_lock import get_graph_lock


class MemoryMonitor:
    """Monitors memory usage and provides alerts."""

    def __init__(self, warning_threshold_mb: float = 500.0, critical_threshold_mb: float = 1000.0):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self._lock = threading.RLock()
        self._memory_history: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            stats = {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'timestamp': time.time()
            }

            # Add to history
            with self._lock:
                self._memory_history.append(stats)
                # Keep only recent history
                if len(self._memory_history) > 100:
                    self._memory_history = self._memory_history[-100:]

            return stats

        except Exception as e:
            log_step("Error getting memory usage", error=str(e))
            return {'error': str(e)}

    def check_memory_thresholds(self) -> Dict[str, Any]:
        """Check if memory usage exceeds thresholds."""
        stats = self.get_memory_usage()
        if 'error' in stats:
            return stats

        alerts = []
        status = 'normal'

        if stats['rss_mb'] >= self.critical_threshold_mb:
            alerts.append({
                'level': 'critical',
                'message': f"Memory usage critically high: {stats['rss_mb']:.1f} MB",
                'timestamp': stats['timestamp']
            })
            status = 'critical'
        elif stats['rss_mb'] >= self.warning_threshold_mb:
            alerts.append({
                'level': 'warning',
                'message': f"Memory usage high: {stats['rss_mb']:.1f} MB",
                'timestamp': stats['timestamp']
            })
            status = 'warning'

        result = {
            'status': status,
            'alerts': alerts,
            'current_usage_mb': stats['rss_mb']
        }

        # Record alerts
        with self._lock:
            self._alerts.extend(alerts)
            # Keep only recent alerts
            if len(self._alerts) > 50:
                self._alerts = self._alerts[-50:]

        return result

    def get_memory_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent memory usage history."""
        with self._lock:
            return self._memory_history[-limit:]

    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memory alerts."""
        with self._lock:
            return self._alerts[-limit:]


class MemoryOptimizer:
    """Provides memory optimization strategies."""

    def __init__(self):
        self._lock = get_graph_lock()
        self._optimization_stats = {
            'garbage_collections': 0,
            'objects_cleaned': 0,
            'memory_freed_mb': 0.0,
            'optimizations_applied': 0
        }

    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics."""
        start_time = time.time()
        start_objects = len(gc.get_objects())

        # Get memory before GC
        monitor = MemoryMonitor()
        memory_before = monitor.get_memory_usage()

        # Force garbage collection
        collected = gc.collect()

        # Get memory after GC
        memory_after = monitor.get_memory_usage()
        end_objects = len(gc.get_objects())

        memory_freed = memory_before.get('rss_mb', 0) - memory_after.get('rss_mb', 0)
        objects_cleaned = start_objects - end_objects

        stats = {
            'collected_objects': collected,
            'objects_cleaned': objects_cleaned,
            'memory_freed_mb': memory_freed,
            'duration_seconds': time.time() - start_time,
            'memory_before_mb': memory_before.get('rss_mb', 0),
            'memory_after_mb': memory_after.get('rss_mb', 0)
        }

        # Update global stats
        with self._lock.write_lock():
            self._optimization_stats['garbage_collections'] += 1
            self._optimization_stats['objects_cleaned'] += objects_cleaned
            self._optimization_stats['memory_freed_mb'] += memory_freed

        log_step("Garbage collection completed",
                collected=collected, memory_freed=f"{memory_freed:.2f}MB")

        return stats

    def optimize_graph_memory(self, graph) -> Dict[str, Any]:
        """Optimize memory usage of a neural graph."""
        if graph is None:
            return {'error': 'Graph is None'}

        optimizations = []
        memory_saved = 0.0

        with self._lock.write_lock():
            # Clear unused attributes
            if hasattr(graph, 'edge_attributes') and graph.edge_attributes:
                original_count = len(graph.edge_attributes)
                # Remove None values and duplicates
                graph.edge_attributes = [attr for attr in graph.edge_attributes if attr is not None]
                cleaned_count = len(graph.edge_attributes)
                if cleaned_count < original_count:
                    optimizations.append(f"Cleaned {original_count - cleaned_count} edge attributes")
                    memory_saved += (original_count - cleaned_count) * 0.1  # Estimate memory saved

            # Optimize node labels
            if hasattr(graph, 'node_labels') and graph.node_labels:
                for node in graph.node_labels:
                    if isinstance(node, dict):
                        # Remove large unused data
                        keys_to_remove = []
                        for key, value in node.items():
                            if isinstance(value, (list, tuple)) and len(value) > 1000:
                                keys_to_remove.append(key)
                            elif hasattr(value, '__dict__') and len(value.__dict__) > 100:
                                # Clear large object attributes
                                for attr in list(value.__dict__.keys()):
                                    if attr.startswith('_cache') or attr.startswith('_temp'):
                                        try:
                                            delattr(value, attr)
                                        except:
                                            pass

                        for key in keys_to_remove:
                            try:
                                del node[key]
                                memory_saved += 0.05  # Estimate memory saved
                            except:
                                pass

            # Optimize tensor memory
            if hasattr(graph, 'x') and graph.x is not None:
                # Ensure tensor is contiguous for better memory access
                if not graph.x.is_contiguous():
                    graph.x = graph.x.contiguous()
                    optimizations.append("Made tensor contiguous")

            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                if not graph.edge_index.is_contiguous():
                    graph.edge_index = graph.edge_index.contiguous()
                    optimizations.append("Made edge_index contiguous")

        stats = {
            'optimizations_applied': len(optimizations),
            'estimated_memory_saved_mb': memory_saved,
            'optimizations': optimizations
        }

        if optimizations:
            with self._lock.write_lock():
                self._optimization_stats['optimizations_applied'] += len(optimizations)

            log_step("Graph memory optimization completed",
                    optimizations=len(optimizations), memory_saved=f"{memory_saved:.2f}MB")

        return stats

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        with self._lock.read_lock():
            return self._optimization_stats.copy()


class MemoryManager:
    """Central memory management system."""

    def __init__(self):
        self.monitor = MemoryMonitor()
        self.optimizer = MemoryOptimizer()
        self._lock = threading.RLock()
        self._cleanup_callbacks: List[Callable] = []
        self._memory_pressure_handlers: List[Callable] = []

    def register_cleanup_callback(self, callback: Callable):
        """Register a callback to be called during memory cleanup."""
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def register_memory_pressure_handler(self, handler: Callable):
        """Register a handler for memory pressure situations."""
        with self._lock:
            self._memory_pressure_handlers.append(handler)

    def perform_memory_maintenance(self) -> Dict[str, Any]:
        """Perform comprehensive memory maintenance."""
        start_time = time.time()

        # Check memory status
        memory_status = self.monitor.check_memory_thresholds()

        maintenance_actions = []

        # Force garbage collection
        gc_stats = self.optimizer.force_garbage_collection()
        maintenance_actions.append({
            'action': 'garbage_collection',
            'stats': gc_stats
        })

        # Call cleanup callbacks
        cleanup_results = []
        with self._lock:
            for callback in self._cleanup_callbacks:
                try:
                    result = callback()
                    cleanup_results.append(result)
                except Exception as e:
                    log_step("Cleanup callback failed", error=str(e))

        if cleanup_results:
            maintenance_actions.append({
                'action': 'cleanup_callbacks',
                'results': cleanup_results
            })

        # Handle memory pressure if needed
        if memory_status['status'] in ['warning', 'critical']:
            pressure_results = []
            with self._lock:
                for handler in self._memory_pressure_handlers:
                    try:
                        result = handler(memory_status)
                        pressure_results.append(result)
                    except Exception as e:
                        log_step("Memory pressure handler failed", error=str(e))

            if pressure_results:
                maintenance_actions.append({
                    'action': 'memory_pressure_handling',
                    'results': pressure_results
                })

        duration = time.time() - start_time

        result = {
            'duration_seconds': duration,
            'memory_status': memory_status,
            'maintenance_actions': maintenance_actions,
            'timestamp': time.time()
        }

        log_step("Memory maintenance completed",
                duration=f"{duration:.3f}s", status=memory_status['status'])

        return result

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report."""
        current_usage = self.monitor.get_memory_usage()
        memory_history = self.monitor.get_memory_history()
        alerts = self.monitor.get_alerts()
        optimization_stats = self.optimizer.get_optimization_statistics()

        # Calculate trends
        if len(memory_history) >= 2:
            recent = memory_history[-10:]  # Last 10 measurements
            avg_memory = sum(h.get('rss_mb', 0) for h in recent) / len(recent)
            max_memory = max(h.get('rss_mb', 0) for h in recent)
            min_memory = min(h.get('rss_mb', 0) for h in recent)

            trend = 'stable'
            if len(recent) >= 5:
                first_half = sum(h.get('rss_mb', 0) for h in recent[:5]) / 5
                second_half = sum(h.get('rss_mb', 0) for h in recent[5:]) / 5
                if second_half > first_half * 1.1:
                    trend = 'increasing'
                elif second_half < first_half * 0.9:
                    trend = 'decreasing'
        else:
            avg_memory = max_memory = min_memory = current_usage.get('rss_mb', 0)
            trend = 'unknown'

        return {
            'current_usage': current_usage,
            'average_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'min_memory_mb': min_memory,
            'memory_trend': trend,
            'alerts': alerts,
            'optimization_stats': optimization_stats,
            'history_length': len(memory_history),
            'report_timestamp': time.time()
        }


# Global instance
_memory_manager_instance = None
_memory_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        with _memory_manager_lock:
            if _memory_manager_instance is None:
                _memory_manager_instance = MemoryManager()
    return _memory_manager_instance


def force_memory_cleanup():
    """Force immediate memory cleanup."""
    manager = get_memory_manager()
    return manager.perform_memory_maintenance()


def get_memory_report():
    """Get current memory report."""
    manager = get_memory_manager()
    return manager.get_memory_report()






