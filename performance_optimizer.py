"""
performance_optimizer.py

Automatic performance tuning and optimization system for the AI neural project.
Provides dynamic performance monitoring, automatic parameter adjustment, and system scaling.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from logging_utils import log_step

# Import configuration manager
from config_manager import get_processing_config


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    step_time: float
    memory_usage: float
    cpu_usage: float
    network_activity: float
    error_rate: float
    timestamp: float


class PerformanceOptimizer:
    """
    Automatic performance tuning and optimization system.
    Monitors system performance and adjusts parameters for optimal operation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance optimizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Performance thresholds
        self.target_step_time = self.config.get('target_step_time', 0.033)  # 30 FPS
        self.max_step_time = self.config.get('max_step_time', 0.1)  # 10 FPS minimum
        self.memory_threshold = self.config.get('memory_threshold', 0.8)  # 80% memory usage
        self.cpu_threshold = self.config.get('cpu_threshold', 0.9)  # 90% CPU usage
        
        # Optimization parameters
        self.optimization_interval = self.config.get('optimization_interval', 10)  # seconds
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)  # 10% change per optimization
        self.min_adaptation_interval = self.config.get('min_adaptation_interval', 5)  # seconds
        
        # Performance history
        self.performance_history: List[PerformanceMetrics] = []
        self.max_history_length = 1000
        
        # Current optimization state
        self.current_optimizations = {
            'sensory_update_interval': 1,
            'memory_update_interval': 50,
            'homeostasis_update_interval': 100,
            'metrics_update_interval': 50,
            'connection_formation_rate': 1.0,
            'node_birth_rate': 1.0,
            'node_death_rate': 1.0
        }
        
        # Optimization callbacks
        self.optimization_callbacks = []
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'last_optimization_time': 0,
            'current_performance_level': 'normal'
        }
        
        log_step("PerformanceOptimizer initialized", 
                target_step_time=self.target_step_time,
                optimization_interval=self.optimization_interval)
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        if self.is_monitoring:
            logging.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self.monitoring_thread.start()
        log_step("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.is_monitoring:
            logging.warning("Performance monitoring not running")
            return
        
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        log_step("Performance monitoring stopped")
    
    def record_performance(self, step_time: float, memory_usage: float = 0.0, 
                          cpu_usage: float = 0.0, network_activity: float = 0.0, 
                          error_rate: float = 0.0):
        """
        Record performance metrics.
        
        Args:
            step_time: Time taken for simulation step
            memory_usage: Memory usage percentage (0-1)
            cpu_usage: CPU usage percentage (0-1)
            network_activity: Network activity level (0-1)
            error_rate: Error rate percentage (0-1)
        """
        metrics = PerformanceMetrics(
            step_time=step_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            network_activity=network_activity,
            error_rate=error_rate,
            timestamp=time.time()
        )
        
        self.performance_history.append(metrics)
        
        # Keep history within limits
        if len(self.performance_history) > self.max_history_length:
            self.performance_history = self.performance_history[-self.max_history_length:]
        
        # Check if optimization is needed
        if self._should_optimize():
            self._perform_optimization()
    
    def _should_optimize(self) -> bool:
        """Check if performance optimization is needed."""
        if len(self.performance_history) < 10:
            return False
        
        current_time = time.time()
        if current_time - self.optimization_stats['last_optimization_time'] < self.min_adaptation_interval:
            return False
        
        # Get recent performance metrics
        recent_metrics = self.performance_history[-10:]
        avg_step_time = sum(m.step_time for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        # Check optimization triggers
        if avg_step_time > self.max_step_time:
            return True
        if avg_memory > self.memory_threshold:
            return True
        if avg_cpu > self.cpu_threshold:
            return True
        if avg_error_rate > 0.1:  # 10% error rate
            return True
        
        return False
    
    def _perform_optimization(self):
        """Perform automatic performance optimization."""
        try:
            self.optimization_stats['total_optimizations'] += 1
            self.optimization_stats['last_optimization_time'] = time.time()
            
            # Analyze current performance
            recent_metrics = self.performance_history[-20:]
            avg_step_time = sum(m.step_time for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            
            # Determine optimization strategy
            optimization_strategy = self._determine_optimization_strategy(
                avg_step_time, avg_memory, avg_cpu, avg_error_rate)
            
            # Apply optimizations
            self._apply_optimizations(optimization_strategy)
            
            # Update performance level
            self._update_performance_level(avg_step_time, avg_memory, avg_cpu, avg_error_rate)
            
            # Call optimization callbacks
            for callback in self.optimization_callbacks:
                try:
                    callback(self.current_optimizations, optimization_strategy)
                except Exception as e:
                    logging.error(f"Optimization callback error: {e}")
            
            self.optimization_stats['successful_optimizations'] += 1
            
            log_step("Performance optimization applied",
                    strategy=optimization_strategy,
                    avg_step_time=avg_step_time,
                    avg_memory=avg_memory)
            
        except Exception as e:
            self.optimization_stats['failed_optimizations'] += 1
            logging.error(f"Performance optimization failed: {e}")
    
    def _determine_optimization_strategy(self, step_time: float, memory: float, 
                                       cpu: float, error_rate: float) -> str:
        """Determine the best optimization strategy based on current performance."""
        if step_time > self.max_step_time:
            return 'reduce_complexity'
        elif memory > self.memory_threshold:
            return 'reduce_memory'
        elif cpu > self.cpu_threshold:
            return 'reduce_cpu'
        elif error_rate > 0.1:
            return 'improve_stability'
        else:
            return 'maintain_performance'
    
    def _apply_optimizations(self, strategy: str):
        """Apply optimizations based on the determined strategy."""
        if strategy == 'reduce_complexity':
            # Reduce update frequencies to improve step time
            self.current_optimizations['sensory_update_interval'] = min(
                self.current_optimizations['sensory_update_interval'] + 1, 5)
            self.current_optimizations['memory_update_interval'] = min(
                self.current_optimizations['memory_update_interval'] + 10, 100)
            self.current_optimizations['homeostasis_update_interval'] = min(
                self.current_optimizations['homeostasis_update_interval'] + 20, 200)
            self.current_optimizations['metrics_update_interval'] = min(
                self.current_optimizations['metrics_update_interval'] + 10, 100)
            
        elif strategy == 'reduce_memory':
            # Reduce memory-intensive operations
            self.current_optimizations['connection_formation_rate'] = max(
                self.current_optimizations['connection_formation_rate'] - 0.1, 0.5)
            self.current_optimizations['node_birth_rate'] = max(
                self.current_optimizations['node_birth_rate'] - 0.1, 0.5)
            
        elif strategy == 'reduce_cpu':
            # Reduce CPU-intensive operations
            self.current_optimizations['sensory_update_interval'] = min(
                self.current_optimizations['sensory_update_interval'] + 2, 10)
            self.current_optimizations['memory_update_interval'] = min(
                self.current_optimizations['memory_update_interval'] + 20, 150)
            
        elif strategy == 'improve_stability':
            # Increase update frequencies for better stability
            self.current_optimizations['sensory_update_interval'] = max(
                self.current_optimizations['sensory_update_interval'] - 1, 1)
            self.current_optimizations['memory_update_interval'] = max(
                self.current_optimizations['memory_update_interval'] - 10, 25)
            self.current_optimizations['homeostasis_update_interval'] = max(
                self.current_optimizations['homeostasis_update_interval'] - 20, 50)
            
        elif strategy == 'maintain_performance':
            # Slight adjustments to maintain optimal performance
            if self.current_optimizations['sensory_update_interval'] > 1:
                self.current_optimizations['sensory_update_interval'] = max(
                    self.current_optimizations['sensory_update_interval'] - 1, 1)
    
    def _update_performance_level(self, step_time: float, memory: float, 
                                cpu: float, error_rate: float):
        """Update the current performance level based on metrics."""
        if step_time > self.max_step_time or memory > self.memory_threshold or cpu > self.cpu_threshold:
            self.optimization_stats['current_performance_level'] = 'degraded'
        elif step_time > self.target_step_time * 1.5:
            self.optimization_stats['current_performance_level'] = 'suboptimal'
        elif step_time < self.target_step_time * 0.8:
            self.optimization_stats['current_performance_level'] = 'excellent'
        else:
            self.optimization_stats['current_performance_level'] = 'normal'
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        log_step("Performance monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Perform periodic optimization check
                if self._should_optimize():
                    self._perform_optimization()
                
                # Sleep for optimization interval
                time.sleep(self.optimization_interval)
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                time.sleep(1)  # Short sleep on error
        
        log_step("Performance monitoring loop ended")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get current optimization recommendations."""
        if len(self.performance_history) < 5:
            return {'status': 'insufficient_data'}
        
        recent_metrics = self.performance_history[-5:]
        avg_step_time = sum(m.step_time for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        
        recommendations = {
            'current_optimizations': self.current_optimizations.copy(),
            'performance_level': self.optimization_stats['current_performance_level'],
            'avg_step_time': avg_step_time,
            'avg_memory_usage': avg_memory,
            'avg_cpu_usage': avg_cpu,
            'target_step_time': self.target_step_time,
            'optimization_stats': self.optimization_stats.copy()
        }
        
        return recommendations
    
    def add_optimization_callback(self, callback: Callable):
        """Add a callback to be called when optimizations are applied."""
        self.optimization_callbacks.append(callback)
    
    def reset_optimizations(self):
        """Reset all optimizations to default values."""
        self.current_optimizations = {
            'sensory_update_interval': 1,
            'memory_update_interval': 50,
            'homeostasis_update_interval': 100,
            'metrics_update_interval': 50,
            'connection_formation_rate': 1.0,
            'node_birth_rate': 1.0,
            'node_death_rate': 1.0
        }
        
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'last_optimization_time': 0,
            'current_performance_level': 'normal'
        }
        
        log_step("Performance optimizations reset to defaults")


# Global performance optimizer instance
_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer

def create_performance_optimizer(config: Optional[Dict[str, Any]] = None) -> PerformanceOptimizer:
    """Create a new performance optimizer instance."""
    return PerformanceOptimizer(config)


# Example usage and testing
if __name__ == "__main__":
    print("PerformanceOptimizer initialized successfully!")
    print("Features include:")
    print("- Automatic performance monitoring")
    print("- Dynamic parameter adjustment")
    print("- System scaling optimization")
    print("- Performance level tracking")
    print("- Optimization recommendations")
    
    # Test basic functionality
    optimizer = PerformanceOptimizer()
    print(f"Optimizer created with target step time: {optimizer.target_step_time}s")
    print("PerformanceOptimizer is ready for integration!")
