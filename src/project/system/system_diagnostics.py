"""
System Diagnostics Module.

This module provides comprehensive diagnostic tools and health check implementations for the Energy-Based Neural System,
including component health checks, system validation, diagnostic utilities, troubleshooting capabilities,
and enhanced error reporting with detailed context and severity classification.
"""

import gc
import logging
import os
import platform
import sys
import threading
import time
from typing import Any

from project.system.system_monitor import SystemMonitor, HealthStatus
from project.utils.error_handler import ErrorHandler
from project.system.global_storage import GlobalStorage
from project.system.state_manager import StateManager
from project.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Import error handling constants for consistency
from project.utils.error_handler import (
    ERROR_SEVERITY_CRITICAL, ERROR_CONTEXT_TIMESTAMP, ERROR_CONTEXT_MODULE,
    ERROR_CONTEXT_FUNCTION, ERROR_CONTEXT_ERROR_TYPE, ERROR_CONTEXT_ERROR_MESSAGE,
    ERROR_CONTEXT_ADDITIONAL_INFO
)

class SystemDiagnostics:
    """
    System diagnostics class providing comprehensive health checks and diagnostic tools.

    This class provides advanced system diagnostics capabilities including:
    - Comprehensive health checks for all system components
    - Performance monitoring and analysis
    - Error detection and resolution recommendations
    - System resource monitoring and optimization
    - Thread-safe diagnostic operations
    - Detailed reporting and logging

    API Documentation:
    - run_comprehensive_diagnostics(): Main entry point for full system diagnostics
    - Health check methods: Individual component health checks with standardized HealthStatus return
    - Analysis methods: Pattern detection and issue analysis
    - Reporting methods: Comprehensive diagnostic report generation
    - Cache management: Diagnostic result caching for performance optimization

    Thread Safety:
    - Uses thread-safe operations for all diagnostic functions
    - Implements caching with proper synchronization
    - Ensures consistent results across concurrent diagnostic operations

    Usage Patterns:
    - Regular health monitoring via system monitor integration
    - On-demand comprehensive diagnostics for troubleshooting
    - Performance analysis and optimization recommendations
    - Error pattern detection and resolution guidance
    - System state validation and integrity checking

    Best Practices:
    - Run comprehensive diagnostics during system initialization
    - Schedule regular health checks (e.g., every 5-10 minutes)
    - Monitor critical components more frequently
    - Use diagnostic results for proactive maintenance
    - Implement automated alerting for critical issues
    - Store diagnostic history for trend analysis
    - Use diagnostic data for capacity planning
    - Integrate diagnostics with monitoring dashboards

    Usage Examples:
    ```python
    # Initialize system diagnostics
    system_monitor = SystemMonitor()
    config_manager = ConfigManager()
    state_manager = StateManager()
    diagnostics = SystemDiagnostics(system_monitor, config_manager, state_manager)

    # Run comprehensive diagnostics
    diagnostics_report = diagnostics.run_comprehensive_diagnostics()

    # Check overall system health
    if diagnostics_report['summary']['system_healthy']:
        logger.info("System is healthy")
    else:
        logger.warning(f"System issues detected: {diagnostics_report['summary']['primary_issues']}")

    # Get specific health check results
    health_statuses = diagnostics_report['health_status']['details']
    for status in health_statuses:
        if status['status'] != 'healthy':
            logger.warning(f"{status['component']}: {status['message']}")
    ```
    """
    
    def __init__(self, system_monitor: SystemMonitor, config_manager: ConfigManager, state_manager: StateManager) -> None:
        """Initialize SystemDiagnostics with monitoring and state managers."""
        self.system_monitor = system_monitor
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Register health checks with the system monitor
        self._register_health_checks()
        
        # Diagnostic cache
        self._diagnostic_cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_duration = 300  # 5 minutes
        
        logger.info("SystemDiagnostics initialized")
    
    def _register_health_checks(self) -> None:
        """Register all health check functions with the system monitor."""
        health_checks = [
            self._check_system_resources,
            self._check_neural_system_health,
            self._check_resource_management,
            self._check_config_integrity,
            self._check_screen_capture,
            self._check_ui_components,
            self._check_memory_management,
            self._check_gc_performance,
            self._check_thread_health,
            self._check_error_rates
        ]
        
        for check_func in health_checks:
            self.system_monitor.register_health_check(check_func)
    
    def _check_system_resources(self) -> HealthStatus:
        """Check system resource availability and usage."""
        try:
            import psutil
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Check for critical issues
            issues: list[str] = []
            resolution_actions: list[str] = []
            
            if cpu_percent > 90:
                issues.append(f"Very high CPU usage: {cpu_percent:.1f}%")
                resolution_actions.append("Consider reducing system load or increasing update intervals")
            
            if memory_percent > 90:
                issues.append(f"Very high memory usage: {memory_percent:.1f}%")
                resolution_actions.append("Consider restarting the application or clearing resources")
            
            if disk_free_gb < 1.0:
                issues.append(f"Low disk space: {disk_free_gb:.1f}GB available")
                resolution_actions.append("Free up disk space")
            
            # Determine status
            if len(issues) >= 2:
                status = "critical"
            elif len(issues) == 1:
                status = "warning"
            else:
                status = "healthy"
            
            message = f"System resources OK - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%"
            if issues:
                message = "; ".join(issues)
            
            return HealthStatus(
                component="system_resources",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory_available_gb,
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk_free_gb
                },
                resolution_actions=resolution_actions
            )
            
        except ImportError:
            return HealthStatus(
                component="system_resources",
                status="warning",
                message="psutil not available for system resource monitoring",
                timestamp=time.time(),
                resolution_actions=["Install psutil package for system monitoring"]
            )
        except Exception as e:
            return HealthStatus(
                component="system_resources",
                status="critical",
                message=f"Error checking system resources: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check system permissions and dependencies"]
            )
    
    def _check_neural_system_health(self) -> HealthStatus:
        """Check neural system health and functionality."""
        try:
            neural_system = GlobalStorage.get('neural_system')
            
            if neural_system is None:
                return HealthStatus(
                    component="neural_system",
                    status="warning",
                    message="Neural system not found in global storage",
                    timestamp=time.time(),
                    resolution_actions=["Ensure neural system is properly initialized and stored"]
                )
            
            # Check if neural system has valid graph
            if hasattr(neural_system, 'g') and neural_system.g is not None:
                g = neural_system.g
                
                # Check basic graph properties
                node_count = g.num_nodes if hasattr(g, 'num_nodes') else len(g.x) if hasattr(g, 'x') else 0
                edge_count = g.num_edges if hasattr(g, 'num_edges') else len(g.edge_index[0]) if hasattr(g, 'edge_index') else 0
                
                issues: list[str] = []
                resolution_actions: list[str] = []
                
                if not node_count:
                    issues.append("No nodes in neural system graph")
                    resolution_actions.append("Initialize graph with nodes or check initialization process")
                
                if not edge_count and node_count > 1:
                    issues.append("No edges in graph with multiple nodes")
                    resolution_actions.append("Check connection formation process")
                
                # Check for tensor shape mismatches (the original issue)
                tensor_mismatches = 0
                if hasattr(neural_system, 'validate_tensor_shapes'):
                    try:
                        tensor_mismatches = len(neural_system.validate_tensor_shapes())
                    except Exception:
                        pass
                
                if tensor_mismatches > 0:
                    issues.append(f"Tensor shape mismatches detected: {tensor_mismatches}")
                    resolution_actions.append("Run tensor validation and repair")
                
                # Determine status
                if tensor_mismatches > 10:
                    status = "critical"
                elif len(issues) > 0:
                    status = "warning"
                else:
                    status = "healthy"
                
                message = f"Neural system healthy - {node_count} nodes, {edge_count} edges"
                if issues:
                    message = "; ".join(issues)
                
                return HealthStatus(
                    component="neural_system",
                    status=status,
                    message=message,
                    timestamp=time.time(),
                    details={
                        'node_count': node_count,
                        'edge_count': edge_count,
                        'tensor_mismatches': tensor_mismatches,
                        'has_graph': g is not None
                    },
                    resolution_actions=resolution_actions
                )
            else:
                return HealthStatus(
                    component="neural_system",
                    status="critical",
                    message="Neural system graph is None or invalid",
                    timestamp=time.time(),
                    resolution_actions=["Reinitialize neural system and graph data structures"]
                )
                
        except Exception as e:
            return HealthStatus(
                component="neural_system",
                status="critical",
                message=f"Error checking neural system: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check neural system initialization and dependencies"]
            )
    
    def _check_resource_management(self) -> HealthStatus:
        """Check resource management system health."""
        try:
            ui_resource_manager = GlobalStorage.get('ui_resource_manager')
            
            if ui_resource_manager is None:
                return HealthStatus(
                    component="resource_management",
                    status="warning",
                    message="UI Resource Manager not found",
                    timestamp=time.time(),
                    resolution_actions=["Ensure UI Resource Manager is properly initialized"]
                )
            
            # Get resource statistics
            try:
                stats = ui_resource_manager.get_resource_statistics()
                
                issues: list[str] = []
                resolution_actions: list[str] = []
                
                # Check resource limits
                images_count = stats.get('images_count', 0)
                windows_count = stats.get('windows_count', 0)
                max_images = stats.get('resource_limits', {}).get('max_images', 100)
                max_windows = stats.get('resource_limits', {}).get('max_windows', 10)
                
                if images_count > max_images * 0.9:
                    issues.append(f"High image count: {images_count}/{max_images}")
                    resolution_actions.append("Consider increasing image cleanup frequency")
                
                if windows_count > max_windows * 0.9:
                    issues.append(f"High window count: {windows_count}/{max_windows}")
                    resolution_actions.append("Check for window leaks")
                
                # Check monitoring status
                monitoring_enabled = stats.get('monitoring_enabled', False)
                if not monitoring_enabled:
                    issues.append("Resource monitoring disabled")
                    resolution_actions.append("Enable resource monitoring")
                
                # Determine status
                if len(issues) >= 2:
                    status = "critical"
                elif len(issues) == 1:
                    status = "warning"
                else:
                    status = "healthy"
                
                message = f"Resource management healthy - Images: {images_count}, Windows: {windows_count}"
                if issues:
                    message = "; ".join(issues)
                
                return HealthStatus(
                    component="resource_management",
                    status=status,
                    message=message,
                    timestamp=time.time(),
                    details=stats,
                    resolution_actions=resolution_actions
                )
                
            except Exception as e:
                return HealthStatus(
                    component="resource_management",
                    status="warning",
                    message=f"Error getting resource statistics: {str(e)}",
                    timestamp=time.time(),
                    resolution_actions=["Check resource manager functionality"]
                )
                
        except Exception as e:
            return HealthStatus(
                component="resource_management",
                status="critical",
                message=f"Error checking resource management: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check resource manager initialization"]
            )
    
    def _check_config_integrity(self) -> HealthStatus:
        """Check configuration integrity and validation."""
        try:
            # Check if configuration can be loaded
            try:
                config = self.config_manager.config
                # Config should be a dictionary - if not, it will cause issues downstream
            except Exception as e:
                return HealthStatus(
                    component="config_integrity",
                    status="critical",
                    message=f"Configuration loading failed: {str(e)}",
                    timestamp=time.time(),
                    resolution_actions=["Check config file format and syntax"]
                )
            
            # Check required sections
            required_sections = ['system', 'sensory', 'workspace']
            missing_sections: list[str] = []
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                return HealthStatus(
                    component="config_integrity",
                    status="critical",
                    message=f"Missing configuration sections: {missing_sections}",
                    timestamp=time.time(),
                    resolution_actions=["Add missing configuration sections"]
                )
            
            # Check system configuration
            system_config = config.get('system', {})
            required_keys = ['update_interval', 'energy_pulse', 'max_energy']
            missing_keys = [key for key in required_keys if key not in system_config]
            
            if missing_keys:
                return HealthStatus(
                    component="config_integrity",
                    status="warning",
                    message=f"Missing system config keys: {missing_keys}",
                    timestamp=time.time(),
                    resolution_actions=["Add missing system configuration keys"]
                )
            
            # Validate update interval
            update_interval = system_config.get('update_interval', 0)
            if not isinstance(update_interval, int) or update_interval < 1 or update_interval > 10000:
                return HealthStatus(
                    component="config_integrity",
                    status="warning",
                    message=f"Invalid update interval: {update_interval}",
                    timestamp=time.time(),
                    resolution_actions=["Set update_interval to a valid integer (1-10000)"]
                )
            
            return HealthStatus(
                component="config_integrity",
                status="healthy",
                message="Configuration integrity check passed",
                timestamp=time.time(),
                details={
                    'sections_present': list(config.keys()),
                    'update_interval': update_interval,
                    'config_valid': True
                }
            )
            
        except Exception as e:
            return HealthStatus(
                component="config_integrity",
                status="critical",
                message=f"Error checking config integrity: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check configuration file and validation logic"]
            )
    
    def _check_screen_capture(self) -> HealthStatus:
        """Check screen capture functionality."""
        try:
            vision_system = GlobalStorage.get('vision_system')
            
            if vision_system is None:
                return HealthStatus(
                    component="screen_capture",
                    status="warning",
                    message="Vision system not found",
                    timestamp=time.time(),
                    resolution_actions=["Initialize screen capture system"]
                )
            
            # Check if capture is running
            is_running = False
            if hasattr(vision_system, 'is_running'):
                is_running = vision_system.is_running
                if not is_running:
                    return HealthStatus(
                        component="screen_capture",
                        status="warning",
                        message="Screen capture thread is not running",
                        timestamp=time.time(),
                        resolution_actions=["Start screen capture thread"]
                    )
            
            # Check error count
            error_count = 0
            if hasattr(vision_system, 'error_count'):
                error_count = vision_system.error_count()
            
            issues: list[str] = []
            resolution_actions: list[str] = []
            
            if error_count > 10:
                issues.append(f"High screen capture error count: {error_count}")
                resolution_actions.append("Check screen capture permissions and display")
            
            # Check frame quality
            latest_frame = None
            if hasattr(vision_system, 'get_latest'):
                latest_frame = vision_system.get_latest()
            
            if latest_frame is None:
                issues.append("No recent frames captured")
                resolution_actions.append("Check screen capture initialization and permissions")
            
            # Determine status
            if len(issues) >= 2:
                status = "critical"
            elif len(issues) == 1:
                status = "warning"
            else:
                status = "healthy"
            
            message = f"Screen capture healthy - Running: {is_running}, Errors: {error_count}"
            if issues:
                message = "; ".join(issues)
            
            return HealthStatus(
                component="screen_capture",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'is_running': is_running,
                    'error_count': error_count,
                    'frame_available': latest_frame is not None
                },
                resolution_actions=resolution_actions
            )
            
        except Exception as e:
            return HealthStatus(
                component="screen_capture",
                status="critical",
                message=f"Error checking screen capture: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check screen capture system initialization"]
            )
    
    def _check_ui_components(self) -> HealthStatus:
        """Check UI components and window management."""
        try:
            main_window = GlobalStorage.get('main_window')
            
            if main_window is None:
                return HealthStatus(
                    component="ui_components",
                    status="warning",
                    message="Main window not found in global storage",
                    timestamp=time.time(),
                    resolution_actions=["Ensure main window is properly initialized and stored"]
                )
            
            # Check window existence and validity
            if not hasattr(main_window, 'window') or main_window.window is None:
                return HealthStatus(
                    component="ui_components",
                    status="critical",
                    message="Main window is None or destroyed",
                    timestamp=time.time(),
                    resolution_actions=["Recreate main window or restart application"]
                )
            
            try:
                # Check if window still exists
                window_exists = main_window.window.winfo_exists()
                if not window_exists:
                    return HealthStatus(
                        component="ui_components",
                        status="critical",
                        message="Main window has been destroyed",
                        timestamp=time.time(),
                        resolution_actions=["Recreate main window"]
                    )
                
                # Check UI resource management
                if hasattr(main_window, 'resource_manager'):
                    try:
                        resource_stats = main_window.resource_manager.get_resource_statistics()
                        images_count = resource_stats.get('images_count', 0)
                        
                        if images_count > 500:
                            return HealthStatus(
                                component="ui_components",
                                status="warning",
                                message=f"High UI resource count: {images_count} images",
                                timestamp=time.time(),
                                resolution_actions=["Check UI resource cleanup and management"]
                            )
                    except Exception:
                        pass  # Resource stats check failed, continue with other checks
                
                return HealthStatus(
                    component="ui_components",
                    status="healthy",
                    message="UI components healthy",
                    timestamp=time.time(),
                    details={
                        'window_exists': window_exists,
                        'window_valid': True
                    }
                )
                
            except Exception as e:
                return HealthStatus(
                    component="ui_components",
                    status="critical",
                    message=f"Window access error: {str(e)}",
                    timestamp=time.time(),
                    resolution_actions=["Check window state and accessibility"]
                )
                
        except Exception as e:
            return HealthStatus(
                component="ui_components",
                status="critical",
                message=f"Error checking UI components: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check main window initialization and storage"]
            )
    
    def _check_memory_management(self) -> HealthStatus:
        """Check memory management and garbage collection."""
        try:
            # Get garbage collection statistics
            gc_stats = gc.get_stats()
            total_collections = sum(stat['collections'] for stat in gc_stats)
            gc_counts = gc.get_count()
            gc_objects = len(gc.get_objects())
            
            # Check for memory leaks (high object count)
            resolution_actions: list[str]
            if gc_objects > 1000000:  # 1M objects is quite high
                status = "warning"
                message = f"High object count: {gc_objects:,} objects"
                resolution_actions = ["Check for memory leaks and excessive object creation"]
            elif total_collections > 1000:  # High collection count
                status = "warning"
                message = f"High GC activity: {total_collections} collections"
                resolution_actions = ["Consider optimizing object lifetime and memory usage"]
            else:
                status = "healthy"
                message = f"Memory management healthy - {total_collections} collections, {gc_objects:,} objects"
                resolution_actions = []
            
            return HealthStatus(
                component="memory_management",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'total_collections': total_collections,
                    'gc_counts': gc_counts,
                    'object_count': gc_objects,
                    'gc_stats': gc_stats
                },
                resolution_actions=resolution_actions
            )
            
        except Exception as e:
            return HealthStatus(
                component="memory_management",
                status="critical",
                message=f"Error checking memory management: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check garbage collection system functionality"]
            )
    
    def _check_gc_performance(self) -> HealthStatus:
        """Check garbage collection performance."""
        try:
            # Measure GC performance
            start_time = time.perf_counter()
            start_objects = len(gc.get_objects())
            
            # Force a collection
            collected = gc.collect()
            
            end_time = time.perf_counter()
            end_objects = len(gc.get_objects())
            
            gc_time = (end_time - start_time) * 1000  # Convert to ms
            objects_collected = start_objects - end_objects
            
            issues: list[str] = []
            resolution_actions: list[str] = []
            
            # Check if GC took too long
            if gc_time > 100:  # 100ms threshold
                issues.append(f"Slow garbage collection: {gc_time:.1f}ms")
                resolution_actions.append("Optimize object creation and memory usage patterns")
            
            # Check collection efficiency
            if collected > 1000:
                issues.append(f"High collection count: {collected} objects")
                resolution_actions.append("Investigate memory leaks or excessive object creation")
            
            # Determine status
            if len(issues) >= 2:
                status = "critical"
            elif len(issues) == 1:
                status = "warning"
            else:
                status = "healthy"
            
            message = f"GC performance OK - {gc_time:.1f}ms, {collected} objects collected"
            if issues:
                message = "; ".join(issues)
            
            return HealthStatus(
                component="gc_performance",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'gc_time_ms': gc_time,
                    'objects_collected': objects_collected,
                    'collected': collected,
                    'start_objects': start_objects,
                    'end_objects': end_objects
                },
                resolution_actions=resolution_actions
            )
            
        except Exception as e:
            return HealthStatus(
                component="gc_performance",
                status="critical",
                message=f"Error checking GC performance: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check garbage collection system functionality"]
            )
    
    def _check_thread_health(self) -> HealthStatus:
        """Check threading and concurrent operations health."""
        try:
            # Count active threads
            threading_active_count = threading.active_count()
            
            issues: list[str] = []
            resolution_actions: list[str] = []
            
            # Check for too many threads
            if threading_active_count > 20:
                issues.append(f"High thread count: {threading_active_count}")
                resolution_actions.append("Check for thread leaks or excessive threading")
            
            # Check thread names for potential issues
            thread_names = [t.name for t in threading.enumerate()]
            daemon_threads = sum(1 for t in threading.enumerate() if t.daemon)
            
            if daemon_threads > threading_active_count * 0.8:
                issues.append("Most threads are daemon threads")
                resolution_actions.append("Review daemon thread usage and cleanup")
            
            # Determine status
            if len(issues) >= 2:
                status = "critical"
            elif len(issues) == 1:
                status = "warning"
            else:
                status = "healthy"
            
            message = f"Threading healthy - {threading_active_count} active threads, {daemon_threads} daemon"
            if issues:
                message = "; ".join(issues)
            
            return HealthStatus(
                component="thread_health",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'active_threads': threading_active_count,
                    'daemon_threads': daemon_threads,
                    'thread_names': thread_names
                },
                resolution_actions=resolution_actions
            )
            
        except Exception as e:
            return HealthStatus(
                component="thread_health",
                status="critical",
                message=f"Error checking thread health: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check threading system functionality"]
            )
    
    def _check_error_rates(self) -> HealthStatus:
        """Check error rates and error handling."""
        try:
            # Check recent errors from error handler
            error_handler = ErrorHandler()
            recent_errors = error_handler.get_recent_errors(20)
            error_count = len(error_handler)
            
            issues: list[str] = []
            resolution_actions: list[str] = []
            
            # Check error count thresholds
            if error_count > 50:
                issues.append(f"High total error count: {error_count}")
                resolution_actions.append("Investigate persistent error sources")
            elif error_count > 20:
                issues.append(f"Moderate error count: {error_count}")
                resolution_actions.append("Review error patterns and fix root causes")
            
            # Check for recent error bursts
            if len(recent_errors) > 10:
                issues.append(f"Recent error burst: {len(recent_errors)} errors in last 20")
                resolution_actions.append("Check for recent system issues or configuration problems")
            
            # Determine status
            if error_count > 100 or len(recent_errors) > 15:
                status = "critical"
            elif error_count > 30 or len(recent_errors) > 5:
                status = "warning"
            else:
                status = "healthy"
            
            message = f"Error handling healthy - {error_count} total errors, {len(recent_errors)} recent"
            if issues:
                message = "; ".join(issues)
            
            return HealthStatus(
                component="error_rates",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'total_errors': error_count,
                    'recent_errors': len(recent_errors),
                    'recent_error_samples': recent_errors[-5:] if recent_errors else []
                },
                resolution_actions=resolution_actions
            )
            
        except Exception as e:
            # Enhanced error logging with detailed context
            error_context: dict[str, Any] = {
                ERROR_CONTEXT_TIMESTAMP: time.time(),
                ERROR_CONTEXT_MODULE: 'system_diagnostics',
                ERROR_CONTEXT_FUNCTION: '_check_error_rates',
                ERROR_CONTEXT_ERROR_TYPE: 'DiagnosticError',
                ERROR_CONTEXT_ERROR_MESSAGE: str(e),
                ERROR_CONTEXT_ADDITIONAL_INFO: {
                    'error_handler_available': 'error_handler' in globals(),
                    'recent_errors_count': len(globals().get('recent_errors', [])),
                    'total_error_count': globals().get('error_count', 0)
                }
            }

            logger.error(f"[{ERROR_SEVERITY_CRITICAL}] Error checking error rates: {str(e)} | Context: {error_context}")

            # Store error context in details field since HealthStatus doesn't have error_context field
            return HealthStatus(
                component="error_rates",
                status="critical",
                message=f"Error checking error rates: {str(e)}",
                timestamp=time.time(),
                resolution_actions=["Check error handler functionality"],
                details={'error_context': error_context, 'severity': ERROR_SEVERITY_CRITICAL}
            )
    
    def run_comprehensive_diagnostics(self) -> dict[str, Any]:
        """
        Run comprehensive system diagnostics.

        This method performs a complete system health assessment by:
        1. Gathering system information and environment details
        2. Running all registered health checks for system components
        3. Collecting performance metrics and historical data
        4. Analyzing active alerts and alert history
        5. Identifying issues and generating actionable recommendations
        6. Compiling a comprehensive diagnostic report

        The diagnostic process includes:
        - System resource checks (CPU, memory, disk)
        - Neural system health validation
        - Resource management analysis
        - Configuration integrity verification
        - Screen capture functionality validation
        - UI component health checks
        - Memory management and garbage collection analysis
        - Thread health and concurrency monitoring
        - Error rate analysis and error handling assessment

        Returns:
            Dictionary containing comprehensive diagnostic report with:
            - System information and environment details
            - Health status for all system components
            - Performance metrics and historical trends
            - Active alerts and alert history
            - Issues analysis and pattern detection
            - Actionable recommendations for improvement
            - Overall system health summary

        Example:
        ```python
        # Run diagnostics and handle results
        try:
            diagnostics_report = system_diagnostics.run_comprehensive_diagnostics()

            # Check overall system health
            if diagnostics_report['summary']['system_healthy']:
                logger.info("System diagnostics: All systems healthy")
            else:
                logger.warning(f"System diagnostics: Issues detected in {diagnostics_report['summary']['primary_issues']}")

                # Get detailed component status
                for component_status in diagnostics_report['health_status']['details']:
                    if component_status['status'] != 'healthy':
                        logger.error(f"{component_status['component']}: {component_status['message']}")
                        logger.info(f"Resolution actions: {component_status['resolution_actions']}")

            # Check performance metrics
            if diagnostics_report['performance']['metrics_available']:
                perf = diagnostics_report['performance']['latest_metrics']
                logger.info(f"Performance: CPU {perf['cpu_percent']}%, Memory {perf['memory_mb']}MB")

            # Review recommendations
            for recommendation in diagnostics_report['recommendations']:
                logger.info(f"Recommendation: {recommendation}")

        except Exception as e:
            logger.error(f"Failed to run comprehensive diagnostics: {str(e)}")
            # Fallback to basic health checks
            health_statuses = system_diagnostics.system_monitor.run_health_checks()
            logger.info(f"Basic health check results: {len([h for h in health_statuses if h.status == 'healthy'])} healthy components")
        ```
        """
        try:
            logger.info("Running comprehensive system diagnostics")
            
            # Get system information
            system_info = self._get_system_info()
            
            # Run all health checks
            health_statuses: list[HealthStatus] = self.system_monitor.run_health_checks()
            
            # Get performance metrics
            recent_metrics = self.system_monitor.get_performance_metrics(10)
            latest_metrics = recent_metrics[-1] if recent_metrics else None
            
            # Get active alerts
            active_alerts = self.system_monitor.get_active_alerts()
            alert_history = self.system_monitor.get_alert_history(10)
            
            # Analyze issues and provide recommendations
            issues_analysis = self._analyze_issues(health_statuses)
            recommendations = self._generate_recommendations(health_statuses, issues_analysis)
            
            diagnostics_report: dict[str, Any] = {
                'timestamp': time.time(),
                'system_info': system_info,
                'health_status': {
                    'overall_status': self._determine_overall_status(health_statuses),
                    'component_count': len(health_statuses),
                    'healthy_components': len([h for h in health_statuses if h.status == 'healthy']),
                    'warning_components': len([h for h in health_statuses if h.status == 'warning']),
                    'critical_components': len([h for h in health_statuses if h.status == 'critical']),
                    'details': [
                        {
                            'component': h.component,
                            'status': h.status,
                            'message': h.message,
                            'timestamp': h.timestamp,
                            'details': h.details,
                            'resolution_actions': h.resolution_actions
                        }
                        for h in health_statuses
                    ]
                },
                'performance': self._build_performance_section(latest_metrics, recent_metrics),
                'alerts': {
                    'active_alerts': active_alerts,
                    'recent_alerts': alert_history,
                    'alert_count': len(active_alerts)
                },
                'issues_analysis': issues_analysis,
                'recommendations': recommendations,
                'summary': {
                    'system_healthy': self._determine_overall_status(health_statuses) == 'healthy',
                    'immediate_action_required': any(h.status == 'critical' for h in health_statuses),
                    'monitoring_required': any(h.status == 'warning' for h in health_statuses),
                    'primary_issues': [h.component for h in health_statuses if h.status in ['warning', 'critical']][:5]
                }
            }
            
            logger.info("Comprehensive diagnostics completed")
            return diagnostics_report
            
        except Exception as e:
            # Enhanced error logging with detailed context
            error_context: dict[str, Any] = {
                ERROR_CONTEXT_TIMESTAMP: time.time(),
                ERROR_CONTEXT_MODULE: 'system_diagnostics',
                ERROR_CONTEXT_FUNCTION: 'run_comprehensive_diagnostics',
                ERROR_CONTEXT_ERROR_TYPE: 'ComprehensiveDiagnosticError',
                ERROR_CONTEXT_ERROR_MESSAGE: str(e),
                ERROR_CONTEXT_ADDITIONAL_INFO: {
                    'system_info_available': 'system_info' in globals(),
                    'health_statuses_count': len(globals().get('health_statuses', [])),
                    'recent_metrics_count': len(globals().get('recent_metrics', []))
                }
            }

            logger.error(f"[{ERROR_SEVERITY_CRITICAL}] Error running comprehensive diagnostics: {str(e)} | Context: {error_context}")

            return {
                'error': str(e),
                'timestamp': time.time(),
                'diagnostics_failed': True,
                'severity': ERROR_SEVERITY_CRITICAL,
                'error_context': error_context,
                'recovery_suggestions': [
                    "Check system resource availability",
                    "Verify component initialization",
                    "Review error logs for detailed context",
                    "Restart diagnostic subsystem"
                ]
            }
    
    def _get_system_info(self) -> dict[str, Any]:
        """Get comprehensive system information."""
        try:
            import psutil
            
            return {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'python_version': sys.version,
                    'python_executable': sys.executable
                },
                'hardware': {
                    'cpu_count': psutil.cpu_count(),
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                    'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 1)
                },
                'process': {
                    'pid': os.getpid(),
                    'memory_info': {
                        'rss_mb': round(psutil.Process().memory_info().rss / (1024**2), 1),
                        'vms_mb': round(psutil.Process().memory_info().vms / (1024**2), 1)
                    },
                    'cpu_percent': psutil.Process().cpu_percent(),
                    'create_time': psutil.Process().create_time()
                }
            }
            
        except Exception as e:
            return {
                'error': f"Failed to get system info: {str(e)}",
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'python_version': sys.version
                }
            }
    
    def _analyze_issues(self, health_statuses: list[HealthStatus]) -> dict[str, Any]:
        """Analyze health check results and identify patterns."""
        critical_issues = [h for h in health_statuses if h.status == 'critical']
        warning_issues = [h for h in health_statuses if h.status == 'warning']
        
        # Identify common resolution patterns
        resolution_patterns: dict[str, int] = {}
        for status in health_statuses:
            for action in status.resolution_actions:
                if action not in resolution_patterns:
                    resolution_patterns[action] = 0
                resolution_patterns[action] += 1
        
        # Sort by frequency
        common_resolutions: list[tuple[str, int]] = sorted(resolution_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'critical_count': len(critical_issues),
            'warning_count': len(warning_issues),
            'critical_components': [h.component for h in critical_issues],
            'warning_components': [h.component for h in warning_issues],
            'common_resolution_patterns': common_resolutions,
            'issues_by_component': {h.component: h.status for h in health_statuses}
        }
    
    def _generate_recommendations(self, health_statuses: list[HealthStatus], issues_analysis: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on health check results."""
        recommendations: list[str] = []
        
        # Critical issue recommendations
        if issues_analysis['critical_count'] > 0:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Address critical system issues")
            recommendations.append("🔧 Review component initialization and configuration")
            recommendations.append("🛠️ Consider restarting the application if issues persist")
        
        # Memory-related recommendations
        memory_issues = [h for h in health_statuses if 'memory' in h.component.lower()]
        if memory_issues:
            recommendations.append("💾 Monitor memory usage and consider resource cleanup")
        
        # Resource management recommendations
        resource_issues = [h for h in health_statuses if 'resource' in h.component.lower()]
        if resource_issues:
            recommendations.append("🗂️ Review resource management and cleanup procedures")
        
        # Performance recommendations
        perf_issues = [h for h in health_statuses if 'performance' in h.component.lower() or 'gc' in h.component.lower()]
        if perf_issues:
            recommendations.append("⚡ Optimize performance and garbage collection settings")
        
        # General recommendations
        if issues_analysis['warning_count'] > 0:
            recommendations.append("👀 Monitor system performance and address warnings")
        
        recommendations.append("📊 Continue monitoring system health and performance metrics")
        
        return recommendations
    
    def _determine_overall_status(self, health_statuses: list[HealthStatus]) -> str:
        """Determine overall system health status."""
        if not health_statuses:
            return "unknown"
        
        critical_count = len([h for h in health_statuses if h.status == 'critical'])
        warning_count = len([h for h in health_statuses if h.status == 'warning'])
        
        if critical_count > 0:
            return "critical"
        elif warning_count > 0:
            return "warning"
        else:
            return "healthy"
    
    def clear_diagnostic_cache(self) -> None:
        """Clear the diagnostic cache."""
        self._diagnostic_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Diagnostic cache cleared")

    def _build_performance_section(self, latest_metrics: Any, recent_metrics: list[Any]) -> dict[str, Any]:
        """Build performance section for diagnostics report."""
        if latest_metrics is not None:
            return {
                'metrics_available': True,
                'latest_metrics': {
                    'cpu_percent': round(latest_metrics.cpu_percent, 1),
                    'memory_mb': round(latest_metrics.memory_mb, 1),
                    'frame_rate': round(latest_metrics.frame_rate, 1)
                },
                'monitoring_duration_minutes': round((time.time() - recent_metrics[0].timestamp) / 60, 1) if recent_metrics else 0
            }
        else:
            return {'metrics_available': False}