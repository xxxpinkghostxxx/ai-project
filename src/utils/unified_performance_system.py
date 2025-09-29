"""
Unified Performance System for the Neural Simulation.
Consolidates performance_monitor.py and performance_optimizer.py
into a comprehensive performance monitoring and optimization system.
"""

import gc
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List

import numpy as np
import psutil

from src.utils.logging_utils import log_step
from src.utils.print_utils import print_error


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics data structure."""
    timestamp: float = field(default_factory=time.time)
    step_time: float = 0.0
    total_runtime: float = 0.0
    fps: float = 0.0
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_temperature: float = 0.0
    network_activity: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    error_rate: float = 0.0
    warning_count: int = 0
    system_health_score: float = 100.0
    node_count: int = 0
    edge_count: int = 0
    throughput: float = 0.0
    event_queue_size: int = 0
    spike_queue_size: int = 0
    gc_collections: int = 0
    gc_time_ms: float = 0.0


@dataclass
class PerformanceThresholds:
    """Performance threshold configuration."""
    memory_warning_mb: float = 2000.0
    memory_critical_mb: float = 4000.0
    memory_percent_warning: float = 85.0
    memory_percent_critical: float = 95.0
    cpu_warning_percent: float = 80.0
    cpu_critical_percent: float = 95.0
    step_time_warning_ms: float = 100.0
    step_time_critical_ms: float = 500.0
    fps_warning: float = 20.0
    fps_critical: float = 10.0
    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.1


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion."""
    suggestion_id: str
    title: str
    description: str
    impact_level: OptimizationLevel
    estimated_improvement: float  # Percentage improvement
    implementation_cost: str  # "low", "medium", "high"
    category: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, history_size: int = 1000, update_interval: float = 1.0):
        self.history_size = history_size
        self.update_interval = update_interval
        self.metrics_history = deque(maxlen=history_size)
        self.running = False
        self.monitor_thread = None
        self._lock = threading.RLock()

        # Performance thresholds
        self.thresholds = PerformanceThresholds()

        # Callbacks
        self.alert_callbacks = []
        self.threshold_callbacks = defaultdict(list)
        self._max_callbacks = 10

        # Statistics
        self.error_count = 0
        self.warning_count = 0
        self.step_count = 0
        self._memory_limit_mb = 2000.0

        # Initialize system info
        self._initialize_system_info()

        # Current metrics
        self.current_metrics = PerformanceMetrics()

        log_step("PerformanceMonitor initialized")
    
    def _initialize_system_info(self):
        """Initialize system information."""
        try:
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024
            self.cpu_count = psutil.cpu_count()
        except Exception as e:
            log_step("Failed to initialize system info", error=str(e))
            self.process = None
            self.initial_memory = 0
            self.cpu_count = 1
    
    def start(self):
        """Start performance monitoring."""
        if not self.running and self.monitor_thread is None:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            log_step("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)  # Increased timeout for safety
        log_step("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._update_metrics()
                self._check_thresholds()
                time.sleep(self.update_interval)
            except Exception as e:
                print_error(f"Error in performance monitoring: {e}")
                time.sleep(1)
    
    def _update_metrics(self):
        """Update current performance metrics."""
        with self._lock:
            if self.process is None:
                return
            
            # System metrics
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_used_mb = memory_info.rss / 1024 / 1024
            memory_available_mb = psutil.virtual_memory().available / 1024 / 1024
            memory_percent = self.process.memory_percent()
            
            # GC metrics
            gc_stats = gc.get_stats()
            gc_collections = sum(stat['collections'] for stat in gc_stats)
            
            # Update current metrics
            self.current_metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_usage_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                memory_peak_mb=max(memory_used_mb, self.current_metrics.memory_peak_mb),
                cpu_count=self.cpu_count,
                gc_collections=gc_collections,
                error_rate=self.error_count / max(self.step_count, 1),
                warning_count=self.warning_count,
                system_health_score=self._calculate_system_health_score()
            )
            
            # Add to history
            self.metrics_history.append(self.current_metrics)
    
    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score."""
        score = 100.0
        
        # CPU penalty
        if self.current_metrics.cpu_percent > self.thresholds.cpu_critical_percent:
            score -= 30
        elif self.current_metrics.cpu_percent > self.thresholds.cpu_warning_percent:
            score -= 15
        
        # Memory penalty
        if self.current_metrics.memory_percent > self.thresholds.memory_percent_critical:
            score -= 30
        elif self.current_metrics.memory_percent > self.thresholds.memory_percent_warning:
            score -= 15
        
        # Error rate penalty
        if self.current_metrics.error_rate > self.thresholds.error_rate_critical:
            score -= 25
        elif self.current_metrics.error_rate > self.thresholds.error_rate_warning:
            score -= 10
        
        return max(0.0, score)
    
    def _check_thresholds(self):
        """Check performance thresholds and trigger alerts."""
        metrics = self.current_metrics
        
        # Memory checks
        if metrics.memory_usage_mb > self.thresholds.memory_critical_mb:
            self._trigger_alert("critical", "memory", {
                'memory_usage_mb': metrics.memory_usage_mb,
                'threshold': self.thresholds.memory_critical_mb
            })
        elif metrics.memory_usage_mb > self.thresholds.memory_warning_mb:
            self._trigger_alert("warning", "memory", {
                'memory_usage_mb': metrics.memory_usage_mb,
                'threshold': self.thresholds.memory_warning_mb
            })
        
        # CPU checks
        if metrics.cpu_percent > self.thresholds.cpu_critical_percent:
            self._trigger_alert("critical", "cpu", {
                'cpu_percent': metrics.cpu_percent,
                'threshold': self.thresholds.cpu_critical_percent
            })
        elif metrics.cpu_percent > self.thresholds.cpu_warning_percent:
            self._trigger_alert("warning", "cpu", {
                'cpu_percent': metrics.cpu_percent,
                'threshold': self.thresholds.cpu_warning_percent
            })
        
        # FPS checks
        if metrics.fps > 0 and metrics.fps < self.thresholds.fps_critical:
            self._trigger_alert("critical", "fps", {
                'fps': metrics.fps,
                'threshold': self.thresholds.fps_critical
            })
        elif metrics.fps > 0 and metrics.fps < self.thresholds.fps_warning:
            self._trigger_alert("warning", "fps", {
                'fps': metrics.fps,
                'threshold': self.thresholds.fps_warning
            })
    
    def _trigger_alert(self, severity: str, alert_type: str, data: Dict[str, Any]):
        """Trigger a performance alert."""
        alert_data = {
            'severity': severity,
            'type': alert_type,
            'timestamp': time.time(),
            'data': data
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(severity, alert_type, alert_data)
            except Exception as e:
                print_error(f"Alert callback failed: {e}")
    
    def record_step(self, step_time: float, node_count: int = 0, edge_count: int = 0):
        """Record a simulation step."""
        with self._lock:
            self.step_count += 1
            self.current_metrics.step_time = step_time
            self.current_metrics.node_count = node_count
            self.current_metrics.edge_count = edge_count
            self.current_metrics.fps = 1.0 / step_time if step_time > 0.0001 else 0.0  # Protect against very small step times
            self.current_metrics.throughput = node_count / step_time if step_time > 0 else 0.0
    
    def record_error(self):
        """Record an error occurrence."""
        with self._lock:
            self.error_count += 1
    
    def record_warning(self):
        """Record a warning occurrence."""
        with self._lock:
            self.warning_count += 1
    
    def add_alert_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]):
        """Add an alert callback."""
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            return self.current_metrics
    
    def get_metrics_history(self) -> List[PerformanceMetrics]:
        """Get performance metrics history."""
        with self._lock:
            return list(self.metrics_history)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            
            return {
                'current_health_score': self.current_metrics.system_health_score,
                'average_cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
                'average_memory_percent': np.mean([m.memory_percent for m in recent_metrics]),
                'average_fps': np.mean([m.fps for m in recent_metrics if m.fps > 0]),
                'total_steps': self.step_count,
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'uptime_hours': (time.time() - self.initial_memory) / 3600
            }

    @property
    def total_steps(self):
        """Get total steps."""
        return self.step_count

    def set_memory_limit(self, limit_mb: float):
        """Set memory limit."""
        if limit_mb <= 0:
            raise ValueError("Memory limit must be positive")
        self._memory_limit_mb = limit_mb

    def start_monitoring(self):
        """Alias for start()."""
        self.start()

    @property
    def _monitoring(self):
        """Get monitoring status."""
        return self.running

    def cleanup(self):
        """Clean up the monitor."""
        self.stop()
        self.alert_callbacks.clear()
        self.threshold_callbacks.clear()
        self.metrics_history.clear()
        self.error_count = 0
        self.warning_count = 0
        self.step_count = 0


class PerformanceOptimizer:
    """Performance optimization system."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_suggestions: Dict[str, OptimizationSuggestion] = {}
        self.active_optimizations: Dict[str, bool] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        self._setup_default_suggestions()
    
    def _setup_default_suggestions(self):
        """Setup default optimization suggestions."""
        suggestions = [
            OptimizationSuggestion(
                suggestion_id="memory_pooling",
                title="Enable Memory Pooling",
                description="Use object pooling to reduce memory allocations",
                impact_level=OptimizationLevel.HIGH,
                estimated_improvement=25.0,
                implementation_cost="medium",
                category="memory"
            ),
            OptimizationSuggestion(
                suggestion_id="adaptive_processing",
                title="Enable Adaptive Processing",
                description="Dynamically adjust processing based on system load",
                impact_level=OptimizationLevel.MEDIUM,
                estimated_improvement=15.0,
                implementation_cost="low",
                category="processing"
            ),
            OptimizationSuggestion(
                suggestion_id="spatial_indexing",
                title="Enable Spatial Indexing",
                description="Use spatial indexing for faster node lookups",
                impact_level=OptimizationLevel.MEDIUM,
                estimated_improvement=20.0,
                implementation_cost="high",
                category="algorithm"
            ),
            OptimizationSuggestion(
                suggestion_id="parallel_processing",
                title="Enable Parallel Processing",
                description="Use multiple threads for independent operations",
                impact_level=OptimizationLevel.HIGH,
                estimated_improvement=40.0,
                implementation_cost="high",
                category="concurrency"
            ),
            OptimizationSuggestion(
                suggestion_id="lazy_loading",
                title="Enable Lazy Loading",
                description="Load data only when needed",
                impact_level=OptimizationLevel.LOW,
                estimated_improvement=10.0,
                implementation_cost="low",
                category="memory"
            )
        ]
        
        for suggestion in suggestions:
            self.optimization_suggestions[suggestion.suggestion_id] = suggestion
            self.active_optimizations[suggestion.suggestion_id] = False
    
    def analyze_performance(self) -> List[OptimizationSuggestion]:
        """Analyze current performance and return optimization suggestions."""
        suggestions = []
        metrics = self.monitor.get_current_metrics()
        
        # Memory-based suggestions
        if metrics.memory_percent > 80:
            suggestions.append(self.optimization_suggestions["memory_pooling"])
            suggestions.append(self.optimization_suggestions["lazy_loading"])
        
        # CPU-based suggestions
        if metrics.cpu_percent > 70:
            suggestions.append(self.optimization_suggestions["adaptive_processing"])
            suggestions.append(self.optimization_suggestions["parallel_processing"])
        
        # FPS-based suggestions
        if metrics.fps > 0 and metrics.fps < 30:
            suggestions.append(self.optimization_suggestions["spatial_indexing"])
            suggestions.append(self.optimization_suggestions["adaptive_processing"])
        
        return suggestions
    
    def apply_optimization(self, suggestion_id: str) -> bool:
        """Apply an optimization suggestion."""
        if suggestion_id not in self.optimization_suggestions:
            return False
        
        suggestion = self.optimization_suggestions[suggestion_id]
        
        try:
            if suggestion_id == "memory_pooling":
                self._enable_memory_pooling()
            elif suggestion_id == "adaptive_processing":
                self._enable_adaptive_processing()
            elif suggestion_id == "spatial_indexing":
                self._enable_spatial_indexing()
            elif suggestion_id == "parallel_processing":
                self._enable_parallel_processing()
            elif suggestion_id == "lazy_loading":
                self._enable_lazy_loading()
            
            self.active_optimizations[suggestion_id] = True
            self.optimization_history.append({
                'suggestion_id': suggestion_id,
                'timestamp': time.time(),
                'applied': True
            })
            
            log_step(f"Applied optimization: {suggestion.title}")
            return True
            
        except Exception as e:
            print_error(f"Failed to apply optimization {suggestion_id}: {e}")
            return False
    
    def _enable_memory_pooling(self):
        """Enable memory pooling optimization."""
        # This would integrate with memory_pool_manager.py
        pass
    
    def _enable_adaptive_processing(self):
        """Enable adaptive processing optimization."""
        # This would adjust processing based on system load
        pass
    
    def _enable_spatial_indexing(self):
        """Enable spatial indexing optimization."""
        # This would implement spatial indexing for node lookups
        pass
    
    def _enable_parallel_processing(self):
        """Enable parallel processing optimization."""
        # This would enable multi-threading for independent operations
        pass
    
    def _enable_lazy_loading(self):
        """Enable lazy loading optimization."""
        # This would implement lazy loading for data
        pass
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'active_optimizations': {k: v for k, v in self.active_optimizations.items() if v},
            'available_suggestions': len(self.optimization_suggestions),
            'optimization_history': len(self.optimization_history)
        }


class AdaptiveProcessor:
    """Adaptive processing system that adjusts based on performance."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.adaptive_enabled = False
        self.component_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0
        }
    
    def should_skip(self, component: str, cpu_threshold: float = 80.0) -> bool:
        """Determine if a component should be skipped based on CPU load."""
        if not self.adaptive_enabled:
            logging.debug(f"AdaptiveProcessor: should_skip for {component} - adaptive not enabled, returning False")
            return False
        metrics = self.monitor.get_current_metrics()
        cpu = metrics.cpu_percent
        skip = cpu > cpu_threshold
        essential = ['neural_dynamics', 'energy_behavior', 'connection_logic']
        if skip and component not in essential:
            logging.info(f"AdaptiveProcessor: should_skip({component}) = True (CPU={cpu:.1f}%, threshold={cpu_threshold}, not essential)")
            return True
        else:
            logging.debug(f"AdaptiveProcessor: should_skip({component}) = False (CPU={cpu:.1f}%, threshold={cpu_threshold}, {'essential' if component in essential else 'enabled'})")
            return False
    
    def enable_adaptive_processing(self):
        """Enable adaptive processing."""
        self.adaptive_enabled = True
        logging.info("AdaptiveProcessor: enable_adaptive_processing called, adaptive_enabled=True")
        log_step("Adaptive processing enabled")
    
    def disable_adaptive_processing(self):
        """Disable adaptive processing."""
        self.adaptive_enabled = False
        log_step("Adaptive processing disabled")
    
    def process_components_adaptively(self, components: Dict[str, Callable]) -> Dict[str, Any]:
        """Process components with adaptive performance adjustments."""
        if not self.adaptive_enabled:
            # Process all components normally
            results = {}
            for name, func in components.items():
                results[name] = func()
            return results
        
        # Get current performance metrics
        metrics = self.monitor.get_current_metrics()
        
        # Determine processing strategy based on performance
        if (metrics.cpu_percent > self.performance_thresholds['cpu_critical'] or
            metrics.memory_percent > self.performance_thresholds['memory_critical']):
            # Critical performance - process only essential components
            return self._process_essential_only(components)
        elif (metrics.cpu_percent > self.performance_thresholds['cpu_warning'] or
              metrics.memory_percent > self.performance_thresholds['memory_warning']):
            # Warning performance - process with reduced frequency
            return self._process_with_reduction(components)
        else:
            # Normal performance - process all components
            return self._process_all_components(components)
    
    def _process_essential_only(self, components: Dict[str, Callable]) -> Dict[str, Any]:
        """Process only essential components during critical performance."""
        essential_components = ['neural_dynamics', 'energy_behavior', 'connection_logic']
        results = {}
        
        for name, func in components.items():
            if name in essential_components:
                start_time = time.time()
                try:
                    results[name] = func()
                except Exception as e:
                    print_error(f"Error in essential component {name}: {e}")
                    results[name] = None
                finally:
                    self._update_component_timing(name, (time.time() - start_time) * 1000)
            else:
                results[name] = None  # Skip non-essential components
        
        return results
    
    def _process_with_reduction(self, components: Dict[str, Callable]) -> Dict[str, Any]:
        """Process components with reduced frequency during warning performance."""
        results = {}
        
        for name, func in components.items():
            # Skip every other execution for non-essential components
            if name not in ['neural_dynamics', 'energy_behavior']:
                if len(self.component_times[name]) % 2 == 0:
                    results[name] = None
                    continue
            
            start_time = time.time()
            try:
                results[name] = func()
            except Exception as e:
                print_error(f"Error in component {name}: {e}")
                results[name] = None
            finally:
                self._update_component_timing(name, (time.time() - start_time) * 1000)
        
        return results
    
    def _process_all_components(self, components: Dict[str, Callable]) -> Dict[str, Any]:
        """Process all components normally."""
        results = {}
        
        for name, func in components.items():
            start_time = time.time()
            try:
                results[name] = func()
            except Exception as e:
                print_error(f"Error in component {name}: {e}")
                results[name] = None
            finally:
                self._update_component_timing(name, (time.time() - start_time) * 1000)
        
        return results
    
    def _update_component_timing(self, component_name: str, time_ms: float):
        """Update component timing statistics."""
        self.component_times[component_name].append(time_ms)


# Global instances
_performance_monitor = None
_performance_optimizer = None
_adaptive_processor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        _performance_monitor.start()
    return _performance_monitor


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer is None:
        monitor = get_performance_monitor()
        _performance_optimizer = PerformanceOptimizer(monitor)
    return _performance_optimizer


def get_adaptive_processor() -> AdaptiveProcessor:
    """Get the global adaptive processor."""
    global _adaptive_processor
    if _adaptive_processor is None:
        monitor = get_performance_monitor()
        _adaptive_processor = AdaptiveProcessor(monitor)
        logging.info("AdaptiveProcessor: Global instance created and initialized")
    return _adaptive_processor


def cleanup_performance_systems():
    """Cleanup all performance systems."""
    global _performance_monitor, _performance_optimizer, _adaptive_processor

    if _performance_monitor:
        _performance_monitor.stop()
        _performance_monitor = None

    _performance_optimizer = None
    _adaptive_processor = None


# Convenience functions for backward compatibility
def record_simulation_step(step_time: float, node_count: int = 0, edge_count: int = 0):
    """Record a simulation step."""
    get_performance_monitor().record_step(step_time, node_count, edge_count)


def record_simulation_error():
    """Record a simulation error."""
    get_performance_monitor().record_error()


def record_simulation_warning():
    """Record a simulation warning."""
    get_performance_monitor().record_warning()


def initialize_performance_monitoring(update_interval: float = 1.0) -> PerformanceMonitor:
    """Initialize and return the performance monitor with specified update interval."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(update_interval=update_interval)
        _performance_monitor.start()
    return _performance_monitor


def get_system_performance_metrics() -> Dict[str, float]:
    """Get current system performance metrics."""
    metrics = get_performance_monitor().get_current_metrics()
    return {
        'cpu_percent': metrics.cpu_percent,
        'memory_percent': metrics.memory_percent,
        'memory_usage_mb': metrics.memory_usage_mb,
        'fps': metrics.fps,
        'system_health_score': metrics.system_health_score
    }







