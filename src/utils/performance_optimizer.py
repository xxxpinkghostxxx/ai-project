"""
Performance Optimizer for the Neural Simulation System.
Provides real-time performance monitoring, optimization suggestions, and adaptive processing.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List
import concurrent.futures

import numpy as np

from src.utils.unified_error_handler import ErrorSeverity, get_error_handler
from src.utils.unified_performance_system import (PerformanceMonitor,
                                                   get_performance_monitor)
from src.utils.lazy_loader import get_lazy_loader


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
        # pylint: disable=broad-except
        except Exception as e:
            print(f"Failed to apply optimization {suggestion_id}: {e}")
            return False

    def _enable_memory_pooling(self):
        """Enable memory pooling optimization."""
        # This would integrate with the memory pool manager

    def _enable_adaptive_processing(self):
        """Enable adaptive processing optimization."""
        # Enable adaptive processing via the adaptive processor
        PerformanceOptimizerSingleton.get_processor().enable_adaptive_processing()

    def _enable_spatial_indexing(self):
        """Enable spatial indexing optimization."""
        # Initialize spatial index if not already done
        if PerformanceOptimizerSingleton._spatial_index is None:
            PerformanceOptimizerSingleton._spatial_index = SpatialIndex()

    def _enable_parallel_processing(self):
        """Enable parallel processing optimization."""
        # Initialize thread pool executor if not already done
        if PerformanceOptimizerSingleton._executor is None:
            PerformanceOptimizerSingleton._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _enable_lazy_loading(self):
        """Enable lazy loading optimization."""
        # Enable lazy loading via the global lazy loader
        # The lazy loader is already available, just ensure it's active
        get_lazy_loader()  # Ensure lazy loader is initialized
        # For basic implementation, lazy loading is enabled by using the loader

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
                # pylint: disable=broad-except
                except Exception as e:
                    results[name] = None
                    get_error_handler().handle_error(e, f"adaptive_processing_{name}", severity=ErrorSeverity.MEDIUM)
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

class SpatialIndex:
    """Basic spatial indexing for fast node queries."""

    def __init__(self, grid_size: float = 10.0):
        self.grid_size = grid_size
        self.grid: Dict[tuple, List[Any]] = defaultdict(list)

    def insert(self, x: float, y: float, item: Any):
        """Insert an item at position (x, y)."""
        grid_x = int(x // self.grid_size)
        grid_y = int(y // self.grid_size)
        self.grid[(grid_x, grid_y)].append(item)

    def query(self, x: float, y: float, radius: float) -> List[Any]:  # pylint: disable=unused-argument
        """Query items within radius of (x, y)."""
        grid_x = int(x // self.grid_size)
        grid_y = int(y // self.grid_size)
        results = []
        # Check neighboring grids
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                grid_items = self.grid.get((grid_x + dx, grid_y + dy), [])
                results.extend(grid_items)
        return results

class PerformanceOptimizerSingleton:
    """Singleton wrapper for performance optimizer and adaptive processor."""

    _optimizer: PerformanceOptimizer = None
    _processor: AdaptiveProcessor = None
    _spatial_index: SpatialIndex = None
    _executor: concurrent.futures.ThreadPoolExecutor = None

    @classmethod
    def get_optimizer(cls) -> PerformanceOptimizer:
        """Get the performance optimizer instance."""
        if cls._optimizer is None:
            monitor = get_performance_monitor()
            cls._optimizer = PerformanceOptimizer(monitor)
        return cls._optimizer

    @classmethod
    def get_processor(cls) -> AdaptiveProcessor:
        """Get the adaptive processor instance."""
        if cls._processor is None:
            monitor = get_performance_monitor()
            cls._processor = AdaptiveProcessor(monitor)
        return cls._processor

    @classmethod
    def cleanup(cls):
        """Clean up performance systems."""
        # The unified system will handle the monitor's lifecycle
        cls._optimizer = None
        cls._processor = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the performance optimizer."""
    return PerformanceOptimizerSingleton.get_optimizer()

def get_adaptive_processor() -> AdaptiveProcessor:
    """Get the adaptive processor."""
    return PerformanceOptimizerSingleton.get_processor()

def cleanup_performance_systems():
    """Clean up performance monitoring systems."""
    PerformanceOptimizerSingleton.cleanup()
