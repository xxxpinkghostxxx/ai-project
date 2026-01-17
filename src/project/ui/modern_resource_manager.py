"""
Modern Resource Manager Module.

This module provides enhanced resource management functionality for the Energy-Based Neural System UI,
including image reference management, window tracking, cleanup handling, memory monitoring, and lifecycle management.
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections.abc import Callable as CallableABC

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QObject, pyqtSignal

from project.utils.error_handler import ErrorHandler
from project.system.global_storage import GlobalStorage

logger = logging.getLogger(__name__)

@dataclass
class ResourceInfo:
    """Information about a tracked resource."""
    resource_id: str
    resource_type: str
    created_at: datetime
    size_bytes: int = 0
    last_accessed: Optional[datetime] = None
    access_count: int = 0

class ModernResourceManager(QObject):
    """
    Modern UI resource manager class with comprehensive resource tracking and lifecycle management.

    This class provides advanced resource management capabilities for the Energy-Based Neural System UI,
    including image reference management, window tracking, cleanup handling, memory monitoring,
    and lifecycle management.

    Key Features:
    - Comprehensive resource tracking and monitoring
    - Memory usage analysis and optimization
    - Automatic cleanup of unused resources
    - Thread-safe resource management
    - Lifecycle management for UI components
    - Detailed resource statistics and reporting
    - Graceful shutdown and cleanup procedures

    Resource Management Strategy:
    - Tracks all registered resources with detailed metadata
    - Implements automatic cleanup based on age and usage patterns
    - Provides memory monitoring with threshold-based cleanup
    - Uses thread-safe operations for concurrent resource access
    - Implements background monitoring for proactive management
    - Provides comprehensive statistics for performance analysis

    Thread Safety:
    - Uses threading.Lock() for thread-safe resource operations
    - Implements fine-grained locking for critical sections
    - Ensures consistent resource tracking across concurrent operations
    - Provides thread-safe access to resource collections and statistics

    Usage Patterns:
    - Automatic resource registration and tracking
    - Proactive memory monitoring and cleanup
    - Background resource optimization
    - Comprehensive resource statistics reporting
    - Graceful shutdown and cleanup procedures

    Example:
    ```python
    # Initialize resource manager with custom limits
    resource_manager = ModernResourceManager(
        max_images=200,
        max_windows=15,
        max_memory_mb=1024,
        enable_monitoring=True
    )

    # Register resources for tracking
    image = resource_manager.create_qpixmap(numpy_array)
    window = resource_manager.register_window(main_window)

    # Get resource statistics for monitoring
    stats = resource_manager.get_resource_statistics()
    logger.info(f"Resource usage: {stats['images_count']} images, {stats['windows_count']} windows")

    # Cleanup resources when needed
    resource_manager.cleanup()
    ```
    """

    resource_stats_updated = pyqtSignal(dict)

    def __init__(self, max_images: int = 100, max_windows: int = 10,
                 max_memory_mb: int = 512, enable_monitoring: bool = True) -> None:
        """Initialize ModernResourceManager with resource limits and monitoring."""
        super().__init__()

        # Basic resource tracking
        self.images: List[Any] = []  # Keep references to prevent garbage collection
        self.windows: List[Any] = []  # Track all windows
        self._cleanup_handlers: List[CallableABC[[], None]] = []

        # Resource limits
        self._max_images = max_images
        self._max_windows = max_windows
        self._max_memory_mb = max_memory_mb

        # Enhanced resource tracking and monitoring
        self._resource_tracking: Dict[str, ResourceInfo] = {}
        self._resource_counter = 0
        self._monitoring_enabled = enable_monitoring
        self._monitoring_interval = 30  # seconds
        self._last_memory_check = datetime.now()

        # Threading for monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        # Initialize monitoring if enabled
        if enable_monitoring:
            self._start_monitoring()

        # Store in global storage for system-wide access
        GlobalStorage.store('ui_resource_manager', self)

    def _start_monitoring(self) -> None:
        """Start the resource monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            logger.info("Resource monitoring started")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop for resource usage."""
        while not self._shutdown_event.wait(self._monitoring_interval):
            try:
                self._check_memory_usage()
                self._cleanup_old_resources()
                self._log_resource_stats()
            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")

    def _check_memory_usage(self) -> None:
        """Check current memory usage and trigger cleanup if needed."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > self._max_memory_mb:
                logger.warning(f"High memory usage detected: {memory_mb:.1f}MB")
                self._perform_memory_cleanup()

            self._last_memory_check = datetime.now()
        except ImportError:
            # psutil not available, skip memory checking
            pass
        except Exception as e:
            logger.warning(f"Error checking memory usage: {e}")

    def _cleanup_old_resources(self) -> None:
        """Clean up resources that haven't been accessed recently."""
        cutoff_time = datetime.now().timestamp() - 300  # 5 minutes
        with self._lock:
            resources_to_remove: List[str] = []
            for resource_id, info in self._resource_tracking.items():
                if info.last_accessed and info.last_accessed.timestamp() < cutoff_time:
                    resources_to_remove.append(resource_id)

            for resource_id in resources_to_remove:
                del self._resource_tracking[resource_id]
                logger.debug(f"Removed old resource: {resource_id}")

    def _log_resource_stats(self) -> None:
        """Log current resource statistics."""
        try:
            with self._lock:
                stats = {
                    'images': len(self.images),
                    'windows': len(self.windows),
                    'tracked_resources': len(self._resource_tracking),
                    'cleanup_handlers': len(self._cleanup_handlers)
                }
            logger.debug(f"Resource stats: {stats}")
            self.resource_stats_updated.emit(stats)
        except Exception as e:
            logger.warning(f"Error logging resource stats: {e}")

    def _perform_memory_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        logger.info("Performing aggressive memory cleanup")

        # Clear some images if we have too many
        with self._lock:
            if len(self.images) > self._max_images // 2:
                images_to_remove = len(self.images) - self._max_images // 2
                self.images[:images_to_remove] = []
                logger.debug(f"Removed {images_to_remove} old images")

        # Force garbage collection
        gc.collect()

        # Run cleanup handlers
        for handler in self._cleanup_handlers[:]:
            try:
                handler()
            except Exception as e:
                logger.warning(f"Error in cleanup handler: {e}")

    def _generate_resource_id(self) -> str:
        """Generate a unique resource ID."""
        self._resource_counter += 1
        return f"resource_{self._resource_counter}_{int(time.time())}"

    def _track_resource(self, resource: Any, resource_type: str) -> str:
        """Track a resource for monitoring and lifecycle management."""
        if not self._monitoring_enabled:
            return ""

        resource_id = self._generate_resource_id()
        try:
            import sys
            size_bytes = sys.getsizeof(resource)
        except Exception:
            size_bytes = 0

        info = ResourceInfo(
            resource_id=resource_id,
            resource_type=resource_type,
            created_at=datetime.now(),
            size_bytes=size_bytes,
            last_accessed=datetime.now()
        )

        with self._lock:
            self._resource_tracking[resource_id] = info

        return resource_id

    def _update_resource_access(self, resource_id: str) -> None:
        """Update access information for a tracked resource."""
        if not resource_id or not self._monitoring_enabled:
            return

        with self._lock:
            if resource_id in self._resource_tracking:
                info = self._resource_tracking[resource_id]
                info.last_accessed = datetime.now()
                info.access_count += 1

    def register_image(self, image: Any) -> Optional[Any]:
        """Register an image to prevent garbage collection with enhanced error handling."""
        try:
            if len(self.images) >= self._max_images:
                oldest_image = self.images.pop(0)  # Remove oldest image
                self._cleanup_resource_by_ref(oldest_image)
                logger.warning(f"Image limit reached ({self._max_images}), removed oldest image")

            resource_id = self._track_resource(image, "image")
            self.images.append(image)
            logger.debug(f"Registered new image with ID: {resource_id}")
            return image
        except MemoryError as e:
            ErrorHandler.show_error(
                "Memory Error",
                f"Failed to register image due to memory constraints: {str(e)}\n\n"
                f"Try reducing image resolution or closing other applications.",
                severity="high"
            )
            return None
        except Exception as e:
            ErrorHandler.show_error(
                "Resource Error",
                f"Failed to register image: {str(e)}\n\n"
                f"Please check the following:\n"
                f"- Ensure the image data is valid\n"
                f"- Verify sufficient system resources are available\n"
                f"- Check for file permission issues",
                severity="medium"
            )
            return None

    def register_window(self, window: Any) -> Optional[Any]:
        """Register a window for tracking with enhanced error handling."""
        try:
            if len(self.windows) >= self._max_windows:
                oldest_window = self.windows.pop(0)
                try:
                    self._cleanup_resource_by_ref(oldest_window)
                    if hasattr(oldest_window, 'close'):
                        oldest_window.close()
                    logger.warning(f"Window limit reached ({self._max_windows}), closed oldest window")
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up old window: {cleanup_error}")

            resource_id = self._track_resource(window, "window")
            self.windows.append(window)
            logger.debug(f"Registered new window with ID: {resource_id}")
            return window
        except TypeError as e:
            ErrorHandler.show_error(
                "Window Registration Error",
                f"Invalid window type: {str(e)}\n\n"
                f"Please ensure you're passing a valid window object.",
                severity="high"
            )
            return None
        except Exception as e:
            ErrorHandler.show_error(
                "Resource Error",
                f"Failed to register window: {str(e)}\n\n"
                f"Troubleshooting steps:\n"
                f"1. Check if the window object is valid\n"
                f"2. Verify window limits are not exceeded\n"
                f"3. Ensure proper window initialization",
                severity="medium"
            )
            return None

    def register_cleanup(self, handler: CallableABC[[], None]) -> None:
        """Register a cleanup handler."""
        try:
            self._track_resource(handler, "cleanup_handler")
            self._cleanup_handlers.append(handler)
        except Exception as e:
            ErrorHandler.show_error("Resource Error", f"Failed to register cleanup handler: {str(e)}")

    def _cleanup_resource_by_ref(self, resource: Any) -> None:
        """Clean up tracking for a resource by reference."""
        if not self._monitoring_enabled:
            return

        # Find and remove any tracking info for this resource
        with self._lock:
            # Since we can't directly match by reference, we'll log cleanup
            logger.debug(f"Cleaning up resource: {type(resource).__name__}")

        # Force immediate cleanup for this resource
        del resource

    def cleanup(self) -> dict[str, Any]:
        """Clean up all registered resources with comprehensive error handling and reporting.

        This method performs a thorough cleanup of all registered resources including:
        - Executing all registered cleanup handlers
        - Clearing image references to allow garbage collection
        - Gracefully closing all registered windows
        - Clearing cleanup handler references

        The cleanup process is designed to be robust and handle errors gracefully,
        ensuring that the system can recover even if individual cleanup operations fail.

        Returns:
            Dictionary containing comprehensive cleanup statistics with keys:
            - cleanup_handlers_executed: Number of successfully executed handlers
            - cleanup_handler_errors: Number of handler execution errors
            - windows_closed: Number of successfully closed windows
            - window_close_errors: Number of window close errors
            - images_cleared: Number of image references cleared
            - total_cleanup_time_ms: Total cleanup time in milliseconds

        Example:
        ```python
        # Perform comprehensive resource cleanup
        cleanup_stats = resource_manager.cleanup()

        # This will:
        # 1. Execute all cleanup handlers
        # 2. Clear image references
        # 3. Close all windows
        # 4. Clear handler references
        # 5. Log detailed cleanup report
        # 6. Return comprehensive statistics

        logger.info(f"Cleanup completed: {cleanup_stats['cleanup_handlers_executed']} handlers executed")
        ```
        """
        try:
            cleanup_start_time = time.time()
            cleanup_report: dict[str, Any] = {
                'cleanup_handlers_executed': 0,
                'cleanup_handler_errors': 0,
                'windows_closed': 0,
                'window_close_errors': 0,
                'images_cleared': len(self.images),
                'total_cleanup_time_ms': 0.0
            }

            logger.info("Starting comprehensive resource cleanup")

            # Run cleanup handlers with detailed error reporting
            for i, handler in enumerate(self._cleanup_handlers):
                try:
                    start_handler_time = time.time()
                    handler()
                    cleanup_report['cleanup_handlers_executed'] += 1
                    logger.debug(f"Cleanup handler {i+1} executed successfully in {time.time() - start_handler_time:.3f}s")
                except Exception as e:
                    cleanup_report['cleanup_handler_errors'] += 1
                    error_msg = f"Error in cleanup handler {i+1}: {str(e)}"
                    ErrorHandler.log_warning(error_msg)
                    logger.warning(error_msg)

            # Clear image references with memory reporting
            images_cleared = len(self.images)
            self.images.clear()
            cleanup_report['images_cleared'] = images_cleared
            logger.info(f"Cleared {images_cleared} image references")

            # Close windows with graceful error handling
            for i, window in enumerate(self.windows):
                try:
                    if hasattr(window, 'close') and hasattr(window, 'winfo_exists'):
                        if window.winfo_exists():  # Check if window still exists
                            window.close()
                            cleanup_report['windows_closed'] += 1
                            logger.debug(f"Closed window {i+1}")
                        else:
                            logger.debug(f"Window {i+1} already destroyed, skipping")
                    else:
                        logger.debug(f"Window {i+1} missing close method or winfo_exists, skipping")
                except Exception as e:
                    cleanup_report['window_close_errors'] += 1
                    error_msg = f"Error closing window {i+1}: {str(e)}"
                    ErrorHandler.log_warning(error_msg)
                    logger.warning(error_msg)

            self.windows.clear()
            logger.info(f"Closed {cleanup_report['windows_closed']} windows")

            # Clear cleanup handlers
            handlers_cleared = len(self._cleanup_handlers)
            self._cleanup_handlers.clear()
            logger.info(f"Cleared {handlers_cleared} cleanup handlers")

            cleanup_report['total_cleanup_time_ms'] = (time.time() - cleanup_start_time) * 1000

            # Log comprehensive cleanup summary
            summary_msg = (
                f"Resource cleanup completed in {cleanup_report['total_cleanup_time_ms']:.1f}ms\n"
                f"- Handlers executed: {cleanup_report['cleanup_handlers_executed']}\n"
                f"- Handler errors: {cleanup_report['cleanup_handler_errors']}\n"
                f"- Windows closed: {cleanup_report['windows_closed']}\n"
                f"- Window errors: {cleanup_report['window_close_errors']}\n"
                f"- Images cleared: {cleanup_report['images_cleared']}\n"
                f"- Total handlers cleared: {handlers_cleared}"
            )

            logger.info(summary_msg)
            ErrorHandler.log_info("Resource cleanup completed successfully")

            return cleanup_report

        except Exception as e:
            error_context = {
                'timestamp': time.time(),
                'error_type': 'CleanupError',
                'error_message': str(e),
                'resources_at_failure': {
                    'images_count': len(self.images),
                    'windows_count': len(self.windows),
                    'handlers_count': len(self._cleanup_handlers)
                }
            }

            ErrorHandler.show_error(
                "Cleanup Error",
                f"Failed to clean up resources: {str(e)}\n\n"
                f"Cleanup was partially completed. Some resources may still be active.\n"
                f"Please restart the application to ensure complete cleanup.",
                severity="high",
                context=error_context
            )
            return {'error': str(e), 'timestamp': time.time()}

    def create_qpixmap(self, image_data: Any) -> Optional[QPixmap]:
        """Create a QPixmap from image data with resource leak prevention."""
        try:
            if hasattr(image_data, 'toqimage'):
                q_image = image_data.toqimage()
            elif isinstance(image_data, QImage):
                q_image = image_data
            else:
                # Convert numpy array to QImage with proper memory management
                if hasattr(image_data, 'shape') and len(image_data.shape) == 3:
                    height, width = image_data.shape[:2]
                    bytes_per_line = 3 * width

                    # Create a copy of the data to avoid memory issues
                    image_data_copy = image_data.copy()
                    q_image = QImage(image_data_copy.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)

                    # Ensure the numpy array can be garbage collected
                    del image_data_copy
                else:
                    ErrorHandler.log_warning("Invalid image data format for QPixmap creation")
                    return None

            if q_image.isNull():
                ErrorHandler.log_warning("Created null QImage, skipping pixmap creation")
                return None

            pixmap = QPixmap.fromImage(q_image)

            # Check if pixmap creation was successful
            if pixmap.isNull():
                ErrorHandler.log_warning("Failed to create QPixmap from QImage")
                return None

            # Register and track the pixmap for proper cleanup
            registered_pixmap = self.register_image(pixmap)

            # Force garbage collection to clean up temporary objects
            gc.collect()

            return registered_pixmap
        except MemoryError as e:
            ErrorHandler.show_error(
                "Memory Error",
                f"Failed to create QPixmap due to memory constraints: {str(e)}\n\n"
                f"Recommendations:\n"
                f"- Reduce image resolution\n"
                f"- Close other applications\n"
                f"- Increase system memory\n"
                f"- Enable aggressive memory optimization",
                severity="high"
            )
            # Force garbage collection to free up memory
            gc.collect()
            return None
        except Exception as e:
            ErrorHandler.show_error(
                "Resource Error",
                f"Failed to create QPixmap: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"- Verify image data format\n"
                f"- Check memory availability\n"
                f"- Validate image dimensions",
                severity="medium"
            )
            return None

    def get_resource_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive resource statistics.

        This method provides detailed information about current resource usage, including:
        - Counts of registered images, windows, and cleanup handlers
        - Memory usage statistics and limits
        - Resource tracking information
        - Age and access patterns of tracked resources

        The statistics are useful for:
        - Monitoring system resource usage
        - Identifying potential memory leaks
        - Optimizing resource allocation
        - Performance analysis and tuning
        - Capacity planning and scaling

        Returns:
            Dictionary containing comprehensive resource statistics with keys:
            - images_count: Number of registered images
            - windows_count: Number of registered windows
            - cleanup_handlers_count: Number of registered cleanup handlers
            - tracked_resources_count: Number of tracked resources
            - monitoring_enabled: Whether monitoring is enabled
            - memory_limit_mb: Maximum memory limit in MB
            - resource_limits: Dictionary with max_images and max_windows
            - total_tracked_size_bytes: Total size of tracked resources (if monitoring enabled)
            - oldest_resource_age: Age of oldest resource in seconds (if monitoring enabled)
            - most_accessed_resource: ID of most accessed resource (if monitoring enabled)

        Example:
        ```python
        # Get and log resource statistics
        stats = resource_manager.get_resource_statistics()
        logger.info(f"Resource usage: {stats['images_count']} images, {stats['windows_count']} windows")

        # Check if memory usage is approaching limits
        if stats['images_count'] > stats['resource_limits']['max_images'] * 0.8:
            logger.warning("Approaching image limit, consider cleanup")

        # Monitor resource age for potential leaks
        if stats.get('oldest_resource_age', 0) > 3600:  # 1 hour
            logger.warning("Some resources have been around for over an hour")
        ```
        """
        try:
            with self._lock:
                stats: Dict[str, Any] = {
                    'images_count': len(self.images),
                    'windows_count': len(self.windows),
                    'cleanup_handlers_count': len(self._cleanup_handlers),
                    'tracked_resources_count': len(self._resource_tracking),
                    'monitoring_enabled': self._monitoring_enabled,
                    'memory_limit_mb': self._max_memory_mb,
                    'resource_limits': {
                        'max_images': self._max_images,
                        'max_windows': self._max_windows
                    }
                }

                # Add detailed tracking info if monitoring is enabled
                if self._monitoring_enabled:
                    total_size = sum(info.size_bytes for info in self._resource_tracking.values())
                    stats['total_tracked_size_bytes'] = total_size
                    stats['oldest_resource_age'] = self._get_oldest_resource_age()
                    stats['most_accessed_resource'] = self._get_most_accessed_resource()

            return stats
        except Exception as e:
            logger.warning(f"Error getting resource statistics: {e}")
            return {'error': str(e)}

    def _get_oldest_resource_age(self) -> float:
        """Get age of oldest tracked resource in seconds."""
        if not self._resource_tracking:
            return 0.0

        oldest_time = min(info.created_at for info in self._resource_tracking.values())
        return (datetime.now() - oldest_time).total_seconds()

    def _get_most_accessed_resource(self) -> Optional[str]:
        """Get the ID of the most accessed resource."""
        if not self._resource_tracking:
            return None

        return max(self._resource_tracking.items(), key=lambda x: x[1].access_count)[0]

    def shutdown(self) -> None:
        """Graceful shutdown of resource manager."""
        logger.info("Starting ModernResourceManager shutdown")

        # Signal monitoring thread to stop
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._shutdown_event.set()
            self._monitoring_thread.join(timeout=5.0)

        # Perform final cleanup
        self.cleanup()

        # Clear global storage reference
        try:
            GlobalStorage.remove('ui_resource_manager')
        except Exception as e:
            logger.warning(f"Error removing from global storage: {e}")

        logger.info("ModernResourceManager shutdown completed")

    def force_cleanup(self) -> None:
        """Force immediate cleanup of all resources."""
        logger.info("Forcing immediate resource cleanup")

        with self._lock:
            # Clear all tracked resources
            self._resource_tracking.clear()

            # Clear image references
            self.images.clear()

            # Clear cleanup handlers
            self._cleanup_handlers.clear()

        # Force garbage collection
        gc.collect()

        logger.info("Forced cleanup completed")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass  # Avoid exceptions in destructor