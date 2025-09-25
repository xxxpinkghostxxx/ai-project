"""
PerformanceMonitoringService implementation - Performance monitoring and metrics service.

This module provides the concrete implementation of IPerformanceMonitor,
collecting system performance metrics with minimal overhead while providing
insights into neural simulation performance and resource usage.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime

from ..interfaces.performance_monitor import IPerformanceMonitor, PerformanceMetrics


class PerformanceMonitoringService(IPerformanceMonitor):
    """
    Concrete implementation of IPerformanceMonitor.

    This service provides comprehensive performance monitoring with
    minimal overhead, collecting metrics on CPU, memory, and system resources.
    """

    def __init__(self):
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self._lock = threading.RLock()

        # Performance tracking
        self._step_start_time = 0.0
        self._last_step_duration = 0.0

        # System info
        self._process = psutil.Process()
        self._initial_memory = self._process.memory_info().rss

    def start_monitoring(self) -> bool:
        """
        Start performance monitoring.

        Returns:
            bool: True if monitoring started successfully
        """
        if self._monitoring_active:
            return True

        try:
            self._monitoring_active = True
            self._stop_event.clear()

            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="PerformanceMonitor"
            )
            self._monitoring_thread.start()

            return True

        except Exception as e:
            print(f"Failed to start performance monitoring: {e}")
            self._monitoring_active = False
            return False

    def stop_monitoring(self) -> bool:
        """
        Stop performance monitoring.

        Returns:
            bool: True if monitoring stopped successfully
        """
        if not self._monitoring_active:
            return True

        try:
            self._monitoring_active = False
            self._stop_event.set()

            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=2.0)

            return True

        except Exception as e:
            print(f"Failed to stop performance monitoring: {e}")
            return False

    def get_current_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Returns:
            PerformanceMetrics: Current performance data
        """
        metrics = PerformanceMetrics()

        try:
            # CPU usage
            metrics.cpu_usage = self._process.cpu_percent(interval=0.1)

            # Memory usage
            memory_info = self._process.memory_info()
            metrics.memory_usage = memory_info.rss / (1024 * 1024)  # MB

            # Step time
            metrics.step_time = self._last_step_duration

            # Thread count
            metrics.active_threads = threading.active_count()

            # Network traffic (simplified - would need more complex implementation)
            metrics.network_traffic = 0.0

            # GPU usage (placeholder - would need GPU monitoring library)
            metrics.gpu_usage = None

        except Exception as e:
            print(f"Error collecting performance metrics: {e}")

        return metrics

    def get_historical_metrics(self, time_range: int = 60) -> List[PerformanceMetrics]:
        """
        Get historical performance metrics.

        Args:
            time_range: Time range in seconds (default: 60)

        Returns:
            List of historical performance metrics
        """
        with self._lock:
            current_time = datetime.now()
            recent_metrics = []

            for metrics in self._metrics_history:
                time_diff = (current_time - metrics.timestamp).total_seconds()
                if time_diff <= time_range:
                    recent_metrics.append(metrics)

            return recent_metrics

    def record_step_start(self) -> None:
        """Record the start of a simulation step."""
        self._step_start_time = time.time()

    def record_step_end(self) -> None:
        """Record the end of a simulation step."""
        if self._step_start_time > 0:
            self._last_step_duration = time.time() - self._step_start_time
            self._step_start_time = 0

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance statistics.

        Returns:
            Dict with performance summary
        """
        with self._lock:
            if not self._metrics_history:
                return {"error": "No performance data available"}

            recent_metrics = list(self._metrics_history)[-100:]  # Last 100 measurements

            cpu_usage = [m.cpu_usage for m in recent_metrics if m.cpu_usage > 0]
            memory_usage = [m.memory_usage for m in recent_metrics if m.memory_usage > 0]
            step_times = [m.step_time for m in recent_metrics if m.step_time > 0]

            summary = {
                "total_measurements": len(self._metrics_history),
                "monitoring_active": self._monitoring_active,
                "average_cpu_usage": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                "peak_cpu_usage": max(cpu_usage) if cpu_usage else 0,
                "average_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                "peak_memory_usage": max(memory_usage) if memory_usage else 0,
                "average_step_time": sum(step_times) / len(step_times) if step_times else 0,
                "max_step_time": max(step_times) if step_times else 0,
                "min_step_time": min(step_times) if step_times else 0
            }

            return summary

    def check_performance_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if current performance meets specified thresholds.

        Args:
            thresholds: Dictionary of performance thresholds

        Returns:
            Dict with threshold check results
        """
        current_metrics = self.get_current_metrics()
        results = {"all_passed": True, "violations": []}

        # Check CPU threshold
        if "max_cpu_usage" in thresholds and current_metrics.cpu_usage > thresholds["max_cpu_usage"]:
            results["violations"].append({
                "metric": "cpu_usage",
                "current": current_metrics.cpu_usage,
                "threshold": thresholds["max_cpu_usage"],
                "status": "exceeded"
            })
            results["all_passed"] = False

        # Check memory threshold
        if "max_memory_usage" in thresholds and current_metrics.memory_usage > thresholds["max_memory_usage"]:
            results["violations"].append({
                "metric": "memory_usage",
                "current": current_metrics.memory_usage,
                "threshold": thresholds["max_memory_usage"],
                "status": "exceeded"
            })
            results["all_passed"] = False

        # Check step time threshold
        if "max_step_time" in thresholds and current_metrics.step_time > thresholds["max_step_time"]:
            results["violations"].append({
                "metric": "step_time",
                "current": current_metrics.step_time,
                "threshold": thresholds["max_step_time"],
                "status": "exceeded"
            })
            results["all_passed"] = False

        return results

    def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects metrics periodically."""
        while not self._stop_event.is_set() and self._monitoring_active:
            try:
                # Collect metrics
                metrics = self.get_current_metrics()

                # Store in history
                with self._lock:
                    self._metrics_history.append(metrics)

                # Sleep for a short interval
                time.sleep(0.1)  # 10Hz monitoring

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Sleep longer on error

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.

        Returns:
            Dict with system information
        """
        try:
            system_info = {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
                "memory_available": psutil.virtual_memory().available / (1024**3),  # GB
                "platform": psutil.platform,
                "python_version": psutil.python_version(),
                "process_id": self._process.pid,
                "process_name": self._process.name()
            }
            return system_info
        except Exception as e:
            return {"error": f"Failed to get system info: {e}"}

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_monitoring()
        with self._lock:
            self._metrics_history.clear()






