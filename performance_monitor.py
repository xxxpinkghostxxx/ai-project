"""
Comprehensive Performance Monitoring System

This module provides real-time performance monitoring capabilities for the neural simulation system.
It replaces placeholder performance monitoring with actual system resource tracking.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import gc
import sys
import os
from datetime import datetime, timedelta

# Try to import optional performance monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - using basic performance monitoring")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available - GPU monitoring disabled")

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
    logging.warning("resource module not available - using basic memory monitoring")


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics data structure."""
    
    # Timing metrics
    step_time: float = 0.0
    total_runtime: float = 0.0
    fps: float = 0.0
    
    # Memory metrics
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_percent: float = 0.0
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: float = 0.0
    
    # GPU metrics (if available)
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_temperature: float = 0.0
    
    # Network metrics
    network_activity: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    
    # System health metrics
    error_rate: float = 0.0
    warning_count: int = 0
    system_health_score: float = 100.0
    
    # Simulation-specific metrics
    node_count: int = 0
    edge_count: int = 0
    throughput: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting."""
    
    # Memory thresholds
    memory_warning_mb: float = 1000.0
    memory_critical_mb: float = 2000.0
    memory_percent_warning: float = 80.0
    memory_percent_critical: float = 95.0
    
    # CPU thresholds
    cpu_warning_percent: float = 80.0
    cpu_critical_percent: float = 95.0
    
    # Timing thresholds
    step_time_warning_ms: float = 100.0
    step_time_critical_ms: float = 500.0
    fps_warning: float = 30.0
    fps_critical: float = 15.0
    
    # Error thresholds
    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.1


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Provides real-time monitoring of system resources, simulation performance,
    and health metrics with configurable thresholds and alerting.
    """
    
    def __init__(self, 
                 update_interval: float = 1.0,
                 history_size: int = 100,
                 thresholds: Optional[PerformanceThresholds] = None):
        """
        Initialize performance monitor.
        
        Args:
            update_interval: How often to update metrics (seconds)
            history_size: Number of historical metrics to keep
            thresholds: Performance thresholds for alerting
        """
        self.update_interval = update_interval
        self.history_size = history_size
        self.thresholds = thresholds or PerformanceThresholds()
        
        # Current metrics
        self.current_metrics = PerformanceMetrics()
        self.metrics_history: deque = deque(maxlen=history_size)
        
        # Performance tracking
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.step_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.warning_count = 0
        self.total_steps = 0
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Initialize system info
        self._initialize_system_info()
        
        logging.info(f"PerformanceMonitor initialized - psutil: {PSUTIL_AVAILABLE}, "
                    f"GPUtil: {GPUTIL_AVAILABLE}, resource: {RESOURCE_AVAILABLE}")
    
    def _initialize_system_info(self) -> None:
        """Initialize system information."""
        try:
            if PSUTIL_AVAILABLE:
                self.current_metrics.cpu_count = psutil.cpu_count()
                self.current_metrics.memory_available_mb = psutil.virtual_memory().total / (1024 * 1024)
            else:
                self.current_metrics.cpu_count = os.cpu_count() or 1
                self.current_metrics.memory_available_mb = 0.0
        except Exception as e:
            logging.error(f"Failed to initialize system info: {e}")
            self.current_metrics.cpu_count = 1
            self.current_metrics.memory_available_mb = 0.0
    
    def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        with self._lock:
            if self._monitoring:
                return
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="PerformanceMonitor"
            )
            self._monitor_thread.start()
            logging.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        with self._lock:
            if not self._monitoring:
                return
            
            self._monitoring = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)
            logging.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._update_metrics()
                self._check_thresholds()
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_metrics(self) -> None:
        """Update all performance metrics."""
        current_time = time.time()
        
        # Update timing metrics
        self.current_metrics.total_runtime = current_time - self.start_time
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            self.current_metrics.step_time = avg_step_time
            self.current_metrics.fps = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
        
        # Update memory metrics
        self._update_memory_metrics()
        
        # Update CPU metrics
        self._update_cpu_metrics()
        
        # Update GPU metrics
        self._update_gpu_metrics()
        
        # Update network metrics
        self._update_network_metrics()
        
        # Update system health
        self._update_system_health()
        
        # Update timestamp
        self.current_metrics.timestamp = datetime.now()
        
        # Store in history
        self.metrics_history.append(self.current_metrics)
        
        self.last_update_time = current_time
    
    def _update_memory_metrics(self) -> None:
        """Update memory-related metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # Use psutil for comprehensive memory monitoring
                memory = psutil.virtual_memory()
                process = psutil.Process()
                
                self.current_metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                self.current_metrics.memory_available_mb = memory.available / (1024 * 1024)
                self.current_metrics.memory_percent = memory.percent
                
                # Track peak memory usage
                if self.current_metrics.memory_usage_mb > self.current_metrics.memory_peak_mb:
                    self.current_metrics.memory_peak_mb = self.current_metrics.memory_usage_mb
                    
            elif RESOURCE_AVAILABLE:
                # Use resource module for basic memory monitoring
                memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                self.current_metrics.memory_usage_mb = memory_usage / 1024  # KB to MB
                
            else:
                # Basic fallback
                self.current_metrics.memory_usage_mb = 0.0
                
        except Exception as e:
            logging.error(f"Failed to update memory metrics: {e}")
            self.current_metrics.memory_usage_mb = 0.0
    
    def _update_cpu_metrics(self) -> None:
        """Update CPU-related metrics."""
        try:
            if PSUTIL_AVAILABLE:
                self.current_metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
                self.current_metrics.load_average = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            else:
                # Basic CPU estimation based on timing
                self.current_metrics.cpu_percent = 0.0
                self.current_metrics.load_average = 0.0
                
        except Exception as e:
            logging.error(f"Failed to update CPU metrics: {e}")
            self.current_metrics.cpu_percent = 0.0
    
    def _update_gpu_metrics(self) -> None:
        """Update GPU-related metrics."""
        try:
            if GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self.current_metrics.gpu_usage_percent = gpu.load * 100
                    self.current_metrics.gpu_memory_mb = gpu.memoryUsed
                    self.current_metrics.gpu_temperature = gpu.temperature
                else:
                    self.current_metrics.gpu_usage_percent = 0.0
                    self.current_metrics.gpu_memory_mb = 0.0
                    self.current_metrics.gpu_temperature = 0.0
            else:
                self.current_metrics.gpu_usage_percent = 0.0
                self.current_metrics.gpu_memory_mb = 0.0
                self.current_metrics.gpu_temperature = 0.0
                
        except Exception as e:
            logging.error(f"Failed to update GPU metrics: {e}")
            self.current_metrics.gpu_usage_percent = 0.0
            self.current_metrics.gpu_memory_mb = 0.0
            self.current_metrics.gpu_temperature = 0.0
    
    def _update_network_metrics(self) -> None:
        """Update network-related metrics."""
        try:
            if PSUTIL_AVAILABLE:
                network = psutil.net_io_counters()
                self.current_metrics.network_bytes_sent = network.bytes_sent
                self.current_metrics.network_bytes_recv = network.bytes_recv
                self.current_metrics.network_activity = (network.bytes_sent + network.bytes_recv) / (1024 * 1024)  # MB
            else:
                self.current_metrics.network_bytes_sent = 0
                self.current_metrics.network_bytes_recv = 0
                self.current_metrics.network_activity = 0.0
                
        except Exception as e:
            logging.error(f"Failed to update network metrics: {e}")
            self.current_metrics.network_bytes_sent = 0
            self.current_metrics.network_bytes_recv = 0
            self.current_metrics.network_activity = 0.0
    
    def _update_system_health(self) -> None:
        """Update system health metrics."""
        try:
            # Calculate error rate
            if self.total_steps > 0:
                self.current_metrics.error_rate = self.error_count / self.total_steps
            else:
                self.current_metrics.error_rate = 0.0
            
            self.current_metrics.warning_count = self.warning_count
            
            # Calculate health score (0-100)
            health_score = 100.0
            
            # Deduct points for high memory usage
            if self.current_metrics.memory_percent > self.thresholds.memory_percent_critical:
                health_score -= 30
            elif self.current_metrics.memory_percent > self.thresholds.memory_percent_warning:
                health_score -= 15
            
            # Deduct points for high CPU usage
            if self.current_metrics.cpu_percent > self.thresholds.cpu_critical_percent:
                health_score -= 25
            elif self.current_metrics.cpu_percent > self.thresholds.cpu_warning_percent:
                health_score -= 10
            
            # Deduct points for slow performance
            if self.current_metrics.step_time * 1000 > self.thresholds.step_time_critical_ms:
                health_score -= 20
            elif self.current_metrics.step_time * 1000 > self.thresholds.step_time_warning_ms:
                health_score -= 10
            
            # Deduct points for high error rate
            if self.current_metrics.error_rate > self.thresholds.error_rate_critical:
                health_score -= 25
            elif self.current_metrics.error_rate > self.thresholds.error_rate_warning:
                health_score -= 10
            
            self.current_metrics.system_health_score = max(0.0, health_score)
            
        except Exception as e:
            logging.error(f"Failed to update system health: {e}")
            self.current_metrics.system_health_score = 50.0
    
    def _check_thresholds(self) -> None:
        """Check performance thresholds and trigger alerts."""
        try:
            alerts = []
            
            # Check memory thresholds
            if self.current_metrics.memory_usage_mb > self.thresholds.memory_critical_mb:
                alerts.append(("critical", "memory", {
                    "usage_mb": self.current_metrics.memory_usage_mb,
                    "threshold_mb": self.thresholds.memory_critical_mb
                }))
            elif self.current_metrics.memory_usage_mb > self.thresholds.memory_warning_mb:
                alerts.append(("warning", "memory", {
                    "usage_mb": self.current_metrics.memory_usage_mb,
                    "threshold_mb": self.thresholds.memory_warning_mb
                }))
            
            # Check CPU thresholds
            if self.current_metrics.cpu_percent > self.thresholds.cpu_critical_percent:
                alerts.append(("critical", "cpu", {
                    "usage_percent": self.current_metrics.cpu_percent,
                    "threshold_percent": self.thresholds.cpu_critical_percent
                }))
            elif self.current_metrics.cpu_percent > self.thresholds.cpu_warning_percent:
                alerts.append(("warning", "cpu", {
                    "usage_percent": self.current_metrics.cpu_percent,
                    "threshold_percent": self.thresholds.cpu_warning_percent
                }))
            
            # Check timing thresholds
            step_time_ms = self.current_metrics.step_time * 1000
            if step_time_ms > self.thresholds.step_time_critical_ms:
                alerts.append(("critical", "performance", {
                    "step_time_ms": step_time_ms,
                    "threshold_ms": self.thresholds.step_time_critical_ms
                }))
            elif step_time_ms > self.thresholds.step_time_warning_ms:
                alerts.append(("warning", "performance", {
                    "step_time_ms": step_time_ms,
                    "threshold_ms": self.thresholds.step_time_warning_ms
                }))
            
            # Check FPS thresholds
            if self.current_metrics.fps < self.thresholds.fps_critical:
                alerts.append(("critical", "fps", {
                    "fps": self.current_metrics.fps,
                    "threshold_fps": self.thresholds.fps_critical
                }))
            elif self.current_metrics.fps < self.thresholds.fps_warning:
                alerts.append(("warning", "fps", {
                    "fps": self.current_metrics.fps,
                    "threshold_fps": self.thresholds.fps_warning
                }))
            
            # Check error rate thresholds
            if self.current_metrics.error_rate > self.thresholds.error_rate_critical:
                alerts.append(("critical", "error_rate", {
                    "error_rate": self.current_metrics.error_rate,
                    "threshold_rate": self.thresholds.error_rate_critical
                }))
            elif self.current_metrics.error_rate > self.thresholds.error_rate_warning:
                alerts.append(("warning", "error_rate", {
                    "error_rate": self.current_metrics.error_rate,
                    "threshold_rate": self.thresholds.error_rate_warning
                }))
            
            # Trigger alerts
            for severity, alert_type, data in alerts:
                self._trigger_alert(severity, alert_type, data)
                
        except Exception as e:
            logging.error(f"Failed to check thresholds: {e}")
    
    def _trigger_alert(self, severity: str, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger performance alert."""
        try:
            alert_message = f"Performance {severity.upper()}: {alert_type} - {data}"
            logging.warning(alert_message)
            
            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(severity, alert_type, data)
                except Exception as e:
                    logging.error(f"Alert callback error: {e}")
                    
        except Exception as e:
            logging.error(f"Failed to trigger alert: {e}")
    
    def record_step(self, step_time: float, node_count: int = 0, edge_count: int = 0) -> None:
        """Record simulation step performance."""
        with self._lock:
            self.step_times.append(step_time)
            self.total_steps += 1
            
            if node_count > 0:
                self.current_metrics.node_count = node_count
            if edge_count > 0:
                self.current_metrics.edge_count = edge_count
            
            # Calculate throughput
            if step_time > 0:
                self.current_metrics.throughput = node_count / step_time
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        with self._lock:
            self.error_count += 1
    
    def record_warning(self) -> None:
        """Record a warning occurrence."""
        with self._lock:
            self.warning_count += 1
    
    def add_alert_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """Remove alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            return self.current_metrics
    
    def get_metrics_history(self) -> List[PerformanceMetrics]:
        """Get performance metrics history."""
        with self._lock:
            return list(self.metrics_history)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for reporting."""
        with self._lock:
            return {
                "current_metrics": self.current_metrics,
                "total_runtime": self.current_metrics.total_runtime,
                "total_steps": self.total_steps,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "avg_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0.0,
                "system_health_score": self.current_metrics.system_health_score,
                "monitoring_active": self._monitoring
            }
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        try:
            collected = gc.collect()
            logging.info(f"Forced garbage collection: {collected} objects collected")
        except Exception as e:
            logging.error(f"Failed to force garbage collection: {e}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_monitoring()
        self.alert_callbacks.clear()
        self.metrics_history.clear()
        self.step_times.clear()
        logging.info("PerformanceMonitor cleaned up")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def initialize_performance_monitoring(update_interval: float = 1.0) -> PerformanceMonitor:
    """Initialize global performance monitoring."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(update_interval=update_interval)
        _performance_monitor.start_monitoring()
    return _performance_monitor


def shutdown_performance_monitoring() -> None:
    """Shutdown global performance monitoring."""
    global _performance_monitor
    if _performance_monitor is not None:
        _performance_monitor.cleanup()
        _performance_monitor = None


# Convenience functions for backward compatibility
def get_system_performance_metrics() -> Dict[str, float]:
    """
    Get system performance metrics (backward compatibility).
    
    Returns:
        Dictionary with performance metrics
    """
    monitor = get_performance_monitor()
    metrics = monitor.get_current_metrics()
    
    return {
        'memory_usage': metrics.memory_usage_mb,
        'cpu_usage': metrics.cpu_percent,
        'network_activity': metrics.network_activity,
        'error_rate': metrics.error_rate,
        'gpu_usage': metrics.gpu_usage_percent,
        'fps': metrics.fps,
        'step_time': metrics.step_time,
        'system_health_score': metrics.system_health_score
    }


def record_simulation_step(step_time: float, node_count: int = 0, edge_count: int = 0) -> None:
    """Record simulation step performance."""
    monitor = get_performance_monitor()
    monitor.record_step(step_time, node_count, edge_count)


def record_simulation_error() -> None:
    """Record simulation error."""
    monitor = get_performance_monitor()
    monitor.record_error()


def record_simulation_warning() -> None:
    """Record simulation warning."""
    monitor = get_performance_monitor()
    monitor.record_warning()
