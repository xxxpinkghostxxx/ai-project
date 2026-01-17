"""
System Monitor Module.

This module provides comprehensive monitoring and health check functionality for the Energy-Based Neural System,
including real-time system metrics, performance monitoring, health validation, alerting, and diagnostic tools.
"""

import gc
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict, Literal, Any, cast

from project.system.global_storage import GlobalStorage
from project.system.state_manager import StateManager
from project.utils.config_manager import ConfigManager

# TypedDict definitions for type safety
class AlertInfo(TypedDict):
    rule_name: str
    message: str
    severity: Literal['info', 'warning', 'critical']
    metrics: dict[str, float]

class SystemState(TypedDict):
    timestamp: float
    node_count: int
    edge_count: int
    memory_usage_mb: float
    cpu_usage_percent: float
    error: str | None  # Only present in error cases

class BenchmarkData(TypedDict):
    timestamp: float
    benchmark_name: str
    metrics: dict[str, Any]
    system_state: SystemState

class SystemHealthReport(TypedDict):
    timestamp: str
    overall_status: Literal['healthy', 'warning', 'critical']
    component_count: int
    critical_issues: int
    warnings: int
    active_alerts: int
    recent_alerts: int
    performance_summary: dict[str, float] | None
    health_components: list[dict[str, Any]]
    monitoring_status: dict[str, Any]
    latest_metrics: dict[str, float | int] | None
    latest_metrics_timestamp: str | None
    error: str | None  # Only present in error cases

class PerformanceReport(TypedDict):
    timestamp: str
    performance_benchmarks: list[BenchmarkData]
    system_health: SystemHealthReport
    performance_analysis: dict[str, float | int]
    recommendations: list[str]
    error: str | None  # Only present in error cases


logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    frame_rate: float
    update_duration_ms: float
    energy_processing_time_ms: float
    connection_processing_time_ms: float
    gc_collections: int
    gc_time_ms: float

@dataclass 
class HealthStatus:
    """Health status data structure."""
    component: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    message: str
    timestamp: float
    details: dict[str, Any] = field(default_factory=lambda: {})
    resolution_actions: list[str] = field(default_factory=lambda: [])

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: Callable[[PerformanceMetrics], bool]
    threshold_value: float
    severity: str  # 'info', 'warning', 'critical'
    message_template: str
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    last_triggered: float | None = None
    enabled: bool = True

class SystemMonitor:
    """Comprehensive system monitor for the Energy-Based Neural System."""
    
    def __init__(self, config_manager: ConfigManager, state_manager: StateManager) -> None:
        """Initialize SystemMonitor with configuration and state managers."""
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Performance metrics tracking
        self._metrics_history: deque[PerformanceMetrics] = deque(maxlen=1000)  # Keep last 1000 metrics
        self._frame_times: deque[float] = deque(maxlen=60)  # Track frame times for FPS calculation
        self._gc_stats_before = self._get_gc_stats()
        
        # Health monitoring
        self._health_checks: list[Callable[[], HealthStatus]] = []
        self._last_health_check = 0.0
        self._health_check_interval = 30.0  # seconds
        
        # Alert system
        self._alert_rules: list[AlertRule] = []
        self._alert_history: deque[AlertInfo] = deque(maxlen=100)
        self._active_alerts: dict[str, float] = {}  # Track active alerts with timestamps
        
        # Monitoring control
        self._monitoring_enabled = True
        self._monitoring_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
        # Performance thresholds
        self._cpu_warning_threshold = 80.0
        self._cpu_critical_threshold = 95.0
        self._memory_warning_threshold = 512.0  # MB
        self._memory_critical_threshold = 1024.0  # MB
        self._frame_rate_warning_threshold = 10.0  # FPS
        self._update_duration_warning_threshold = 100.0  # ms
        
        # Callbacks for notifications
        self._alert_callbacks: list[Callable[[AlertRule, PerformanceMetrics], None]] = []
        
        # Register for global storage
        GlobalStorage.store('system_monitor', self)
        
        # Initialize alert rules
        self._setup_default_alert_rules()
        
        # Register shutdown cleanup
        # ShutdownDetector.register_cleanup(self._cleanup, "SystemMonitor cleanup")
        
        logger.info("SystemMonitor initialized")
    
    def _get_gc_stats(self) -> tuple[int, int, int]:
        """Get garbage collection statistics."""
        stats = gc.get_stats()
        total_collections = sum(stat['collections'] for stat in stats)
        return (total_collections, gc.get_count()[0], len(gc.get_objects()))
    
    def _setup_default_alert_rules(self) -> None:
        """Set up default alert rules."""
        # CPU usage alerts
        self._alert_rules.append(AlertRule(
            name="high_cpu_usage",
            condition=lambda m: m.cpu_percent > self._cpu_critical_threshold,
            threshold_value=self._cpu_critical_threshold,
            severity="critical",
            message_template="Critical CPU usage: {:.1f}% exceeds threshold {:.1f}%"
        ))
        
        self._alert_rules.append(AlertRule(
            name="high_cpu_warning",
            condition=lambda m: m.cpu_percent > self._cpu_warning_threshold,
            threshold_value=self._cpu_warning_threshold,
            severity="warning",
            message_template="High CPU usage: {:.1f}% exceeds threshold {:.1f}%"
        ))
        
        # Memory usage alerts
        self._alert_rules.append(AlertRule(
            name="high_memory_usage",
            condition=lambda m: m.memory_mb > self._memory_critical_threshold,
            threshold_value=self._memory_critical_threshold,
            severity="critical",
            message_template="Critical memory usage: {:.1f}MB exceeds threshold {:.1f}MB"
        ))
        
        self._alert_rules.append(AlertRule(
            name="high_memory_warning",
            condition=lambda m: m.memory_mb > self._memory_warning_threshold,
            threshold_value=self._memory_warning_threshold,
            severity="warning",
            message_template="High memory usage: {:.1f}MB exceeds threshold {:.1f}MB"
        ))
        
        # Performance alerts
        self._alert_rules.append(AlertRule(
            name="low_frame_rate",
            condition=lambda m: m.frame_rate < self._frame_rate_warning_threshold,
            threshold_value=self._frame_rate_warning_threshold,
            severity="warning",
            message_template="Low frame rate: {:.1f} FPS below threshold {:.1f} FPS"
        ))
        
        self._alert_rules.append(AlertRule(
            name="slow_updates",
            condition=lambda m: m.update_duration_ms > self._update_duration_warning_threshold,
            threshold_value=self._update_duration_warning_threshold,
            severity="warning",
            message_template="Slow update duration: {:.1f}ms exceeds threshold {:.1f}ms"
        ))
    
    def start_monitoring(self) -> None:
        """Start the comprehensive monitoring system."""
        if not self._monitoring_enabled:
            return
            
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self._shutdown_event.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.wait(1.0):  # Check every second
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                if metrics:
                    self._metrics_history.append(metrics)
                    
                    # Check alert conditions
                    self._check_alerts(metrics)
                
                # Periodic health checks
                current_time = time.time()
                if current_time - self._last_health_check >= self._health_check_interval:
                    self.run_health_checks()
                    self._last_health_check = current_time
                    
            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
    
    def _collect_performance_metrics(self) -> PerformanceMetrics | None:
        """Collect current performance metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            # System metrics
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            # Calculate frame rate
            current_time = time.time()
            self._frame_times.append(current_time)
            frame_rate = self._calculate_frame_rate()
            
            # Update duration (estimated from monitoring loop timing)
            update_duration = self._estimate_update_duration()
            
            # Garbage collection metrics
            gc_stats_after = self._get_gc_stats()
            gc_collections = gc_stats_after[0] - self._gc_stats_before[0]
            gc_time_ms = self._estimate_gc_time(gc_collections)
            self._gc_stats_before = gc_stats_after
            
            # Energy and connection processing times (placeholder)
            energy_time = 5.0  # ms - would be collected from neural system
            connection_time = 2.0  # ms - would be collected from neural system
            
            return PerformanceMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                frame_rate=frame_rate,
                update_duration_ms=update_duration,
                energy_processing_time_ms=energy_time,
                connection_processing_time_ms=connection_time,
                gc_collections=gc_collections,
                gc_time_ms=gc_time_ms
            )
            
        except ImportError:
            logger.warning("psutil not available, skipping performance metrics collection")
            return None
        except Exception as e:
            logger.warning(f"Error collecting performance metrics: {e}")
            return None
    
    def _calculate_frame_rate(self) -> float:
        """Calculate current frame rate from frame times."""
        if len(self._frame_times) < 2:
            return 0.0
        
        recent_times = list(self._frame_times)[-10:]  # Use last 10 frames
        if len(recent_times) < 2:
            return 0.0
        
        time_diff = recent_times[-1] - recent_times[0]
        if time_diff <= 0:
            return 0.0
        
        frames = len(recent_times) - 1
        return frames / time_diff
    
    def _estimate_update_duration(self) -> float:
        """Estimate update duration (placeholder)."""
        # This would be calculated from actual system update timing
        return 16.67  # Default to ~60 FPS
    
    def _estimate_gc_time(self, gc_collections: int) -> float:
        """Estimate garbage collection time."""
        # Rough estimation based on collection count
        return gc_collections * 0.1  # Assume 0.1ms per collection
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check metrics against alert rules."""
        current_time = time.time()
        
        for rule in self._alert_rules:
            if not rule.enabled:
                continue
                
            try:
                if rule.condition(metrics):
                    # Check cooldown
                    if (rule.last_triggered and 
                        current_time - rule.last_triggered < rule.cooldown_seconds):
                        continue
                    
                    # Trigger alert
                    message = rule.message_template.format(
                        metrics.cpu_percent if 'cpu' in rule.name else metrics.memory_mb if 'memory' in rule.name else metrics.frame_rate,
                        rule.threshold_value
                    )
                    
                    alert_info: AlertInfo = {
                        'rule_name': rule.name,
                        'message': message,
                        'severity': cast(Literal['info', 'warning', 'critical'], rule.severity),
                        'metrics': {
                            'cpu_percent': metrics.cpu_percent,
                            'memory_mb': metrics.memory_mb,
                            'frame_rate': metrics.frame_rate,
                            'timestamp': metrics.timestamp
                        }
                    }
                    
                    self._alert_history.append(alert_info)
                    self._active_alerts[rule.name] = current_time
                    rule.last_triggered = current_time
                    
                    # Call alert callbacks
                    for callback in self._alert_callbacks:
                        try:
                            callback(rule, metrics)
                        except Exception as e:
                            logger.warning(f"Error in alert callback: {e}")
                    
                    # Log alert
                    if rule.severity == 'critical':
                        logger.critical(f"ALERT: {message}")
                    else:
                        logger.warning(f"ALERT: {message}")
                        
            except Exception as e:
                logger.warning(f"Error checking alert rule '{rule.name}': {e}")
    
    def run_health_checks(self) -> list[HealthStatus]:
        """Run all registered health checks."""
        health_statuses: list[HealthStatus] = []
        
        for check_func in self._health_checks:
            try:
                status = check_func()
                health_statuses.append(status)
                
                # Log critical health issues
                if status.status == 'critical':
                    logger.critical(f"Health check failed: {status.component} - {status.message}")
                elif status.status == 'warning':
                    logger.warning(f"Health check warning: {status.component} - {status.message}")
                    
            except Exception as e:
                error_status = HealthStatus(
                    component="health_check",
                    status="critical",
                    message=f"Health check error: {str(e)}",
                    timestamp=time.time()
                )
                health_statuses.append(error_status)
        
        return health_statuses
    
    def register_health_check(self, check_func: Callable[[], HealthStatus]) -> None:
        """Register a health check function."""
        with self._lock:
            self._health_checks.append(check_func)
    
    def register_alert_callback(self, callback: Callable[[AlertRule, PerformanceMetrics], None]) -> None:
        """Register an alert callback function."""
        with self._lock:
            self._alert_callbacks.append(callback)
    
    def add_custom_alert_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        with self._lock:
            self._alert_rules.append(rule)
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule by name."""
        with self._lock:
            for i, rule in enumerate(self._alert_rules):
                if rule.name == rule_name:
                    del self._alert_rules[i]
                    return True
        return False
    
    def get_performance_metrics(self, count: int = 100) -> list[PerformanceMetrics]:
        """Get recent performance metrics."""
        with self._lock:
            return list(self._metrics_history)[-count:]
    
    def get_alert_history(self, count: int = 50) -> list[AlertInfo]:
        """Get recent alert history."""
        with self._lock:
            return list(self._alert_history)[-count:]
    
    def get_active_alerts(self) -> dict[str, float]:
        """Get currently active alerts."""
        with self._lock:
            return self._active_alerts.copy()
    
    def clear_alert(self, alert_name: str) -> bool:
        """Clear a specific alert."""
        with self._lock:
            if alert_name in self._active_alerts:
                del self._active_alerts[alert_name]
                return True
        return False
    
    def clear_all_alerts(self) -> None:
        """Clear all active alerts."""
        with self._lock:
            self._active_alerts.clear()
    
    def get_system_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report."""
        try:
            current_time = time.time()
            
            # Get recent metrics
            recent_metrics = self.get_performance_metrics(10)
            latest_metrics = recent_metrics[-1] if recent_metrics else None
            
            # Calculate averages
            if recent_metrics:
                avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_mb for m in recent_metrics) / len(recent_metrics)
                avg_fps = sum(m.frame_rate for m in recent_metrics) / len(recent_metrics)
            else:
                avg_cpu = avg_memory = avg_fps = 0.0
            
            # Run health checks
            health_statuses = self.run_health_checks()
            
            # Determine overall health
            critical_count = sum(1 for h in health_statuses if h.status == 'critical')
            warning_count = sum(1 for h in health_statuses if h.status == 'warning')
            
            if critical_count > 0:
                overall_status = 'critical'
            elif warning_count > 0:
                overall_status = 'warning'
            else:
                overall_status = 'healthy'
            
            report: SystemHealthReport = {
                'timestamp': datetime.fromtimestamp(current_time).isoformat(),
                'overall_status': overall_status,
                'component_count': len(health_statuses),
                'critical_issues': critical_count,
                'warnings': warning_count,
                'active_alerts': len(self.get_active_alerts()),
                'recent_alerts': len(self.get_alert_history(10)),
                'performance_summary': {
                    'avg_cpu_percent': round(avg_cpu, 1),
                    'avg_memory_mb': round(avg_memory, 1),
                    'avg_frame_rate': round(avg_fps, 1),
                    'monitoring_duration_minutes': round((current_time - self._metrics_history[0].timestamp) / 60, 1) if self._metrics_history else 0
                } if latest_metrics else None,
                'health_components': [
                    {
                        'component': h.component,
                        'status': h.status,
                        'message': h.message,
                        'timestamp': datetime.fromtimestamp(h.timestamp).isoformat(),
                        'details': h.details,
                        'resolution_actions': h.resolution_actions
                    }
                    for h in health_statuses
                ],
                'monitoring_status': {
                    'enabled': self._monitoring_enabled,
                    'thread_alive': self._monitoring_thread.is_alive() if self._monitoring_thread else False,
                    'metrics_collected': len(self._metrics_history),
                    'alert_rules_active': len([r for r in self._alert_rules if r.enabled])
                },
                'latest_metrics': None,
                'latest_metrics_timestamp': None,
                'error': None
            }
            
            # Add latest metrics if available
            if latest_metrics:
                latest_metrics_dict: dict[str, float | int] = {
                    'cpu_percent': round(latest_metrics.cpu_percent, 1),
                    'memory_mb': round(latest_metrics.memory_mb, 1),
                    'memory_percent': round(latest_metrics.memory_percent, 1),
                    'frame_rate': round(latest_metrics.frame_rate, 1),
                    'update_duration_ms': round(latest_metrics.update_duration_ms, 2),
                    'gc_collections': latest_metrics.gc_collections
                }
                report['latest_metrics'] = latest_metrics_dict
                report['latest_metrics_timestamp'] = datetime.fromtimestamp(latest_metrics.timestamp).isoformat()
            
            return report
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'critical',
                'component_count': 0,
                'critical_issues': 0,
                'warnings': 0,
                'active_alerts': 0,
                'recent_alerts': 0,
                'performance_summary': None,
                'health_components': [],
                'monitoring_status': {},
                'latest_metrics': None,
                'latest_metrics_timestamp': None,
                'error': f'Failed to generate health report: {str(e)}'
            }
    
    def _cleanup(self) -> None:
        """Cleanup resources during shutdown."""
        try:
            self.stop_monitoring()
            self._alert_history.clear()
            self._metrics_history.clear()
            logger.info("SystemMonitor cleanup completed")
        except Exception as e:
            logger.warning(f"Error during SystemMonitor cleanup: {e}")
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self._cleanup()
        except Exception:
            pass  # Avoid exceptions in destructor

    def add_performance_benchmark(self, benchmark_name: str, metrics: dict[str, Any]) -> None:
        """
        Add performance benchmark results to the monitoring system.

        Args:
            benchmark_name: Name of the benchmark
            metrics: Dictionary of performance metrics
        """
        try:
            benchmark_data: BenchmarkData = {
                'timestamp': time.time(),
                'benchmark_name': benchmark_name,
                'metrics': metrics,
                'system_state': self._get_current_system_state()
            }

            # Store in global storage
            benchmarks = GlobalStorage.retrieve('performance_benchmarks', [])
            benchmarks.append(benchmark_data)

            # Keep only the last 100 benchmarks to prevent memory issues
            if len(benchmarks) > 100:
                benchmarks = benchmarks[-100:]

            GlobalStorage.store('performance_benchmarks', benchmarks)
            logger.info(f"Performance benchmark recorded: {benchmark_name}")

        except Exception as e:
            logger.error(f"Error recording performance benchmark: {str(e)}")

    def _get_current_system_state(self) -> SystemState:
        """Get current system state for benchmarking context."""
        try:
            state: SystemState = {
                'timestamp': time.time(),
                'node_count': 0,
                'edge_count': 0,
                'memory_usage_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'error': None
            }

            # Get system metrics if available
            try:
                import psutil
                process = psutil.Process()
                state['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
                state['cpu_usage_percent'] = process.cpu_percent()
            except ImportError:
                pass

            # Get neural system state if available
            try:
                neural_system = GlobalStorage.retrieve('neural_system')
                if neural_system and hasattr(neural_system, 'g') and neural_system.g is not None:
                    state['node_count'] = neural_system.g.num_nodes or 0
                    state['edge_count'] = neural_system.g.num_edges or 0
            except Exception:
                pass

            return state

        except Exception as e:
            logger.warning(f"Error getting current system state: {str(e)}")
            return {
                'timestamp': time.time(),
                'node_count': 0,
                'edge_count': 0,
                'memory_usage_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'error': str(e)
            }

    def get_performance_benchmarks(self, count: int = 50) -> list[BenchmarkData]:
        """
        Get recent performance benchmarks.

        Args:
            count: Number of recent benchmarks to return

        Returns:
            List of benchmark data dictionaries
        """
        try:
            benchmarks = GlobalStorage.retrieve('performance_benchmarks', [])
            return list(benchmarks)[-count:]
        except Exception as e:
            logger.error(f"Error retrieving performance benchmarks: {str(e)}")
            return []

    def generate_performance_report(self) -> PerformanceReport:
        """
        Generate comprehensive performance report with benchmarks and system metrics.

        Returns:
            Dictionary containing performance analysis and recommendations
        """
        try:
            report: PerformanceReport = {
                'timestamp': datetime.now().isoformat(),
                'performance_benchmarks': self.get_performance_benchmarks(20),
                'system_health': self.get_system_health_report(),
                'performance_analysis': {},
                'recommendations': [],
                'error': None
            }

            # Analyze benchmarks if available
            benchmarks = report['performance_benchmarks']
            if benchmarks:
                # Calculate average performance metrics
                total_execution_time = 0.0
                total_memory_usage = 0.0
                total_cpu_usage = 0.0
                benchmark_count = len(benchmarks)

                for benchmark in benchmarks:
                    metrics = benchmark['metrics']
                    if 'execution_time' in metrics:
                        total_execution_time += metrics['execution_time']
                    if 'memory_usage_mb' in metrics:
                        total_memory_usage += metrics['memory_usage_mb']
                    if 'cpu_usage_percent' in metrics:
                        total_cpu_usage += metrics['cpu_usage_percent']

                if benchmark_count > 0:
                    report['performance_analysis'] = {
                        'avg_execution_time': total_execution_time / benchmark_count,
                        'avg_memory_usage_mb': total_memory_usage / benchmark_count,
                        'avg_cpu_usage_percent': total_cpu_usage / benchmark_count,
                        'benchmark_count': benchmark_count,
                        'time_window_seconds': (benchmarks[-1]['timestamp'] - benchmarks[0]['timestamp']) if len(benchmarks) > 1 else 0
                    }

                    # Generate recommendations based on performance analysis
                    if report['performance_analysis']['avg_execution_time'] > 1.0:  # More than 1 second average
                        report['recommendations'].append("Consider optimizing computationally intensive operations - average execution time is high")

                    if report['performance_analysis']['avg_memory_usage_mb'] > 500:  # More than 500MB average
                        report['recommendations'].append("Memory usage is high - consider memory optimization techniques")

                    if report['performance_analysis']['avg_cpu_usage_percent'] > 80:  # More than 80% CPU
                        report['recommendations'].append("CPU usage is high - consider load balancing or optimization")

            return report

        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'performance_benchmarks': [],
                'system_health': {
                    'timestamp': datetime.now().isoformat(),
                    'overall_status': 'critical',
                    'component_count': 0,
                    'critical_issues': 0,
                    'warnings': 0,
                    'active_alerts': 0,
                    'recent_alerts': 0,
                    'performance_summary': None,
                    'health_components': [],
                    'monitoring_status': {},
                    'latest_metrics': None,
                    'latest_metrics_timestamp': None,
                    'error': None
                },
                'performance_analysis': {},
                'recommendations': [],
                'error': f'Failed to generate performance report: {str(e)}'
            }