"""
Performance Optimizer for the Neural Simulation System.
Provides real-time performance monitoring, optimization suggestions, and adaptive processing.
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np
import weakref
from utils.unified_error_handler import ErrorSeverity

class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    simulation_fps: float
    step_time_ms: float
    node_count: int
    edge_count: int
    event_queue_size: int
    spike_queue_size: int
    gc_collections: int
    gc_time_ms: float

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
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'fps_warning': 30.0,
            'fps_critical': 15.0,
            'step_time_warning': 50.0,  # ms
            'step_time_critical': 100.0  # ms
        }
        
        # Callbacks
        self.threshold_callbacks = defaultdict(list)
        
        # Initialize process
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
    
    def start(self):
        """Start performance monitoring."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop(self):
        """Stop performance monitoring."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)  # Increased timeout for safety
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self._update_history(metrics)
                self._check_thresholds(metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                from utils.unified_error_handler import get_error_handler
                error_handler = get_error_handler()
                error_handler.handle_error(e, "performance_monitoring_loop", severity=ErrorSeverity.MEDIUM)
                time.sleep(1)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_used_mb = memory_info.rss / 1024 / 1024
        memory_available_mb = psutil.virtual_memory().available / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # GC metrics
        gc_stats = gc.get_stats()
        gc_collections = sum(stat['collections'] for stat in gc_stats)
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            simulation_fps=0.0,  # Will be updated by simulation
            step_time_ms=0.0,    # Will be updated by simulation
            node_count=0,        # Will be updated by simulation
            edge_count=0,        # Will be updated by simulation
            event_queue_size=0,  # Will be updated by simulation
            spike_queue_size=0,  # Will be updated by simulation
            gc_collections=gc_collections,
            gc_time_ms=0.0
        )
    
    def _update_history(self, metrics: PerformanceMetrics):
        """Update metrics history."""
        with self._lock:
            self.metrics_history.append(metrics)
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and trigger callbacks."""
        # CPU threshold checks
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            self._trigger_callback('cpu_critical', metrics)
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            self._trigger_callback('cpu_warning', metrics)
        
        # Memory threshold checks
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            self._trigger_callback('memory_critical', metrics)
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            self._trigger_callback('memory_warning', metrics)
        
        # FPS threshold checks
        if metrics.simulation_fps > 0:
            if metrics.simulation_fps <= self.thresholds['fps_critical']:
                self._trigger_callback('fps_critical', metrics)
            elif metrics.simulation_fps <= self.thresholds['fps_warning']:
                self._trigger_callback('fps_warning', metrics)
        
        # Step time threshold checks
        if metrics.step_time_ms > 0:
            if metrics.step_time_ms >= self.thresholds['step_time_critical']:
                self._trigger_callback('step_time_critical', metrics)
            elif metrics.step_time_ms >= self.thresholds['step_time_warning']:
                self._trigger_callback('step_time_warning', metrics)
    
    def _trigger_callback(self, threshold_name: str, metrics: PerformanceMetrics):
        """Trigger callbacks for threshold violations."""
        for callback in self.threshold_callbacks[threshold_name]:
            try:
                callback(threshold_name, metrics)
            except Exception as e:
                from utils.unified_error_handler import get_error_handler
                error_handler = get_error_handler()
                error_handler.handle_error(e, "threshold_callback", severity=ErrorSeverity.LOW)
    
    def add_threshold_callback(self, threshold_name: str, callback: Callable):
        """Add a callback for threshold violations."""
        self.threshold_callbacks[threshold_name].append(callback)
    
    def update_simulation_metrics(self, fps: float, step_time_ms: float, 
                                 node_count: int, edge_count: int,
                                 event_queue_size: int, spike_queue_size: int):
        """Update simulation-specific metrics."""
        with self._lock:
            if self.metrics_history:
                latest = self.metrics_history[-1]
                latest.simulation_fps = fps
                latest.step_time_ms = step_time_ms
                latest.node_count = node_count
                latest.edge_count = edge_count
                latest.event_queue_size = event_queue_size
                latest.spike_queue_size = spike_queue_size
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, limit: int = None) -> List[PerformanceMetrics]:
        """Get metrics history."""
        with self._lock:
            if limit is None:
                return list(self.metrics_history)
            return list(self.metrics_history)[-limit:]

class PerformanceOptimizer:
    """Performance optimization system."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_level = OptimizationLevel.MEDIUM
        self.active_optimizations = set()
        self.optimization_suggestions = []
        self._setup_default_suggestions()
    
    def _setup_default_suggestions(self):
        """Setup default optimization suggestions."""
        suggestions = [
            OptimizationSuggestion(
                suggestion_id="memory_pool",
                title="Enable Memory Pooling",
                description="Use object pooling to reduce garbage collection pressure",
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
                description="Use spatial indexing for faster node queries",
                impact_level=OptimizationLevel.MEDIUM,
                estimated_improvement=20.0,
                implementation_cost="high",
                category="data_structures"
            ),
            OptimizationSuggestion(
                suggestion_id="parallel_processing",
                title="Enable Parallel Processing",
                description="Process independent components in parallel",
                impact_level=OptimizationLevel.HIGH,
                estimated_improvement=30.0,
                implementation_cost="high",
                category="processing"
            ),
            OptimizationSuggestion(
                suggestion_id="lazy_loading",
                title="Enable Lazy Loading",
                description="Load components only when needed",
                impact_level=OptimizationLevel.LOW,
                estimated_improvement=10.0,
                implementation_cost="low",
                category="memory"
            )
        ]
        
        self.optimization_suggestions = suggestions
    
    def analyze_performance(self) -> List[OptimizationSuggestion]:
        """Analyze current performance and suggest optimizations."""
        current_metrics = self.monitor.get_current_metrics()
        if not current_metrics:
            return []
        
        suggestions = []
        
        # Memory-based suggestions
        if current_metrics.memory_percent > 70:
            suggestions.extend([
                s for s in self.optimization_suggestions 
                if s.category == "memory" and s.impact_level in [OptimizationLevel.HIGH, OptimizationLevel.MEDIUM]
            ])
        
        # CPU-based suggestions
        if current_metrics.cpu_percent > 80:
            suggestions.extend([
                s for s in self.optimization_suggestions 
                if s.category == "processing" and s.impact_level in [OptimizationLevel.HIGH, OptimizationLevel.MEDIUM]
            ])
        
        # FPS-based suggestions
        if current_metrics.simulation_fps > 0 and current_metrics.simulation_fps < 30:
            suggestions.extend([
                s for s in self.optimization_suggestions 
                if s.impact_level in [OptimizationLevel.HIGH, OptimizationLevel.MEDIUM]
            ])
        
        # Remove duplicates and sort by impact
        unique_suggestions = {s.suggestion_id: s for s in suggestions}
        return sorted(unique_suggestions.values(), 
                     key=lambda x: (x.impact_level.value, x.estimated_improvement), 
                     reverse=True)
    
    def apply_optimization(self, suggestion_id: str) -> bool:
        """Apply a specific optimization."""
        suggestion = next((s for s in self.optimization_suggestions if s.suggestion_id == suggestion_id), None)
        if not suggestion:
            return False
        
        try:
            if suggestion_id == "memory_pool":
                self._enable_memory_pooling()
            elif suggestion_id == "adaptive_processing":
                self._enable_adaptive_processing()
            elif suggestion_id == "spatial_indexing":
                self._enable_spatial_indexing()
            elif suggestion_id == "parallel_processing":
                self._enable_parallel_processing()
            elif suggestion_id == "lazy_loading":
                self._enable_lazy_loading()
            
            self.active_optimizations.add(suggestion_id)
            return True
        except Exception as e:
            print(f"Failed to apply optimization {suggestion_id}: {e}")
            return False
    
    def _enable_memory_pooling(self):
        """Enable memory pooling optimization."""
        # This would integrate with the memory pool manager
        pass
    
    def _enable_adaptive_processing(self):
        """Enable adaptive processing optimization."""
        # This would modify the simulation loop to be adaptive
        pass
    
    def _enable_spatial_indexing(self):
        """Enable spatial indexing optimization."""
        # This would add spatial indexing to the graph
        pass
    
    def _enable_parallel_processing(self):
        """Enable parallel processing optimization."""
        # This would enable parallel processing of components
        pass
    
    def _enable_lazy_loading(self):
        """Enable lazy loading optimization."""
        # This would implement lazy loading for components
        pass
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'optimization_level': self.optimization_level.value,
            'active_optimizations': list(self.active_optimizations),
            'available_suggestions': len(self.optimization_suggestions),
            'applied_suggestions': len(self.active_optimizations)
        }

class AdaptiveProcessor:
    """Adaptive processing system that adjusts based on performance."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.processing_budget = 16.67  # 60 FPS target in ms
        self.component_priorities = {
            'spike_system': 1,
            'neural_dynamics': 2,
            'learning_systems': 3,
            'visual_systems': 4,
            'audio_systems': 5
        }
        self.component_times = defaultdict(list)
        self.adaptive_enabled = False
    
    def enable_adaptive_processing(self):
        """Enable adaptive processing."""
        self.adaptive_enabled = True
    
    def disable_adaptive_processing(self):
        """Disable adaptive processing."""
        self.adaptive_enabled = False
    
    def process_components_adaptively(self, components: Dict[str, Callable]) -> Dict[str, Any]:
        """Process components adaptively based on performance."""
        if not self.adaptive_enabled:
            # Process all components normally
            results = {}
            for name, func in components.items():
                results[name] = func()
            return results
        
        # Get current performance metrics
        current_metrics = self.monitor.get_current_metrics()
        if not current_metrics:
            # Fallback to normal processing
            results = {}
            for name, func in components.items():
                results[name] = func()
            return results
        
        # Calculate available time budget
        available_time = self.processing_budget - current_metrics.step_time_ms
        
        # Sort components by priority
        sorted_components = sorted(
            components.items(),
            key=lambda x: self.component_priorities.get(x[0], 999)
        )
        
        results = {}
        remaining_time = available_time
        
        for name, func in sorted_components:
            if remaining_time <= 0:
                # Skip remaining components if time budget exhausted
                results[name] = None
                continue
            
            # Estimate time for this component
            estimated_time = self._estimate_component_time(name)
            
            if estimated_time <= remaining_time:
                start_time = time.time()
                try:
                    results[name] = func()
                    actual_time = (time.time() - start_time) * 1000
                    self._update_component_timing(name, actual_time)
                    remaining_time -= actual_time
                except Exception as e:
                    results[name] = None
                    from utils.unified_error_handler import get_error_handler
                    error_handler = get_error_handler()
                    error_handler.handle_error(e, f"adaptive_processing_{name}", severity=ErrorSeverity.MEDIUM)
            else:
                # Skip this component if not enough time
                results[name] = None
        
        return results
    
    def _estimate_component_time(self, component_name: str) -> float:
        """Estimate processing time for a component."""
        if component_name in self.component_times and self.component_times[component_name]:
            return np.mean(self.component_times[component_name][-10:])  # Average of last 10
        return 1.0  # Default estimate
    
    def _update_component_timing(self, component_name: str, time_ms: float):
        """Update component timing history."""
        self.component_times[component_name].append(time_ms)
        # Keep only last 100 measurements
        if len(self.component_times[component_name]) > 100:
            self.component_times[component_name] = self.component_times[component_name][-100:]

# Global instances
_performance_monitor = None
_performance_optimizer = None
_adaptive_processor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        _performance_monitor.start_monitoring()
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
    return _adaptive_processor

def cleanup_performance_systems():
    """Clean up performance monitoring systems."""
    global _performance_monitor, _performance_optimizer, _adaptive_processor
    
    if _performance_monitor:
        _performance_monitor.stop_monitoring()
        _performance_monitor = None
    
    _performance_optimizer = None
    _adaptive_processor = None
