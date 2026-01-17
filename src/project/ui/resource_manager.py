"""
Resource Manager Module.

This module provides enhanced resource management functionality for the Energy-Based Neural System UI,
including image reference management, window tracking, cleanup handling, memory monitoring, and lifecycle management.
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from collections.abc import Callable
from PIL import Image, ImageTk
from PIL.Image import Resampling
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
    last_accessed: datetime | None = None
    access_count: int = 0

class UIResourceManager:
    """Enhanced UI resource manager class with comprehensive resource tracking and lifecycle management."""

    def __init__(self: 'UIResourceManager', max_images: int = 100, max_windows: int = 10,
                 max_memory_mb: int = 512, enable_monitoring: bool = True) -> None:
        """Initialize UIResourceManager with resource limits and monitoring."""
        # Basic resource tracking
        self.images: list[Any] = []  # Keep references to prevent garbage collection
        self.windows: list[Any] = []  # Track all windows
        self._cleanup_handlers: list[Callable[[], None]] = []

        # Resource limits
        self._max_images = max_images
        self._max_windows = max_windows
        self._max_memory_mb = max_memory_mb

        # Enhanced resource tracking and monitoring
        self._resource_tracking: dict[str, ResourceInfo] = {}
        self._resource_counter = 0
        self._monitoring_enabled = enable_monitoring
        self._monitoring_interval = 30  # seconds
        self._last_memory_check = datetime.now()

        # Threading for monitoring
        self._monitoring_thread: threading.Thread | None = None
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
            resources_to_remove: list[str] = []
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

    def register_image(self: 'UIResourceManager', image: Any) -> Any | None:
        """Register an image to prevent garbage collection"""
        try:
            if len(self.images) >= self._max_images:
                oldest_image = self.images.pop(0)  # Remove oldest image
                self._cleanup_resource_by_ref(oldest_image)

            self._track_resource(image, "image")
            self.images.append(image)
            return image
        except Exception as e:
            ErrorHandler.show_error("Resource Error", f"Failed to register image: {str(e)}")
            return None

    def register_window(self: 'UIResourceManager', window: Any) -> Any | None:
        """Register a window for tracking"""
        try:
            if len(self.windows) >= self._max_windows:
                oldest_window = self.windows.pop(0)
                try:
                    self._cleanup_resource_by_ref(oldest_window)
                    oldest_window.destroy()
                except Exception:
                    pass

            self._track_resource(window, "window")
            self.windows.append(window)
            return window
        except Exception as e:
            ErrorHandler.show_error("Resource Error", f"Failed to register window: {str(e)}")
            return None

    def register_cleanup(self: 'UIResourceManager', handler: Callable[[], None]) -> None:
        """Register a cleanup handler"""
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

    def cleanup(self: 'UIResourceManager') -> None:
        """Clean up all registered resources"""
        try:
            # Run cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    ErrorHandler.log_warning(f"Error during cleanup: {str(e)}")

            # Clear image references
            self.images.clear()

            # Destroy windows
            for window in self.windows:
                try:
                    # Check if window still exists and hasn't been destroyed
                    if hasattr(window, 'winfo_exists') and window.winfo_exists():
                        window.destroy()
                    else:
                        ErrorHandler.log_warning("Window already destroyed, skipping")
                except Exception as e:
                    ErrorHandler.log_warning(f"Error destroying window: {str(e)}")
            self.windows.clear()

            # Clear cleanup handlers
            self._cleanup_handlers.clear()

            ErrorHandler.log_info("Resource cleanup completed")
        except Exception as e:
            ErrorHandler.show_error("Cleanup Error", f"Failed to clean up resources: {str(e)}")

    def create_tk_image(self: 'UIResourceManager', image_data: Any, size: tuple[int, int] | None = None) -> Any | None:
        """Create a Tkinter image from image data"""
        try:
            if isinstance(image_data, Image.Image):
                if size:
                    image_data = image_data.resize(size, Resampling.NEAREST)
                tk_image = ImageTk.PhotoImage(image_data)
                return self.register_image(tk_image)
            return None
        except Exception as e:
            ErrorHandler.show_error("Resource Error", f"Failed to create Tk image: {str(e)}")
            return None

    def get_resource_statistics(self) -> dict[str, Any]:
        """Get comprehensive resource statistics."""
        try:
            with self._lock:
                stats: dict[str, Any] = {
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

    def _get_most_accessed_resource(self) -> str | None:
        """Get the ID of the most accessed resource."""
        if not self._resource_tracking:
            return None

        return max(self._resource_tracking.items(), key=lambda x: x[1].access_count)[0]

    def shutdown(self) -> None:
        """Graceful shutdown of resource manager."""
        logger.info("Starting UIResourceManager shutdown")

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

        logger.info("UIResourceManager shutdown completed")

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

    def __del__(self: 'UIResourceManager') -> None:
        """Destructor to ensure cleanup"""
        try:
            self.shutdown()
        except Exception:
            pass  # Avoid exceptions in destructor
