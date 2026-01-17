"""
Shutdown Utilities Module.

This module provides enhanced utilities for detecting Python interpreter shutdown
and managing graceful cleanup during system termination with resource management integration.
"""
import sys
import logging
import threading
import atexit
from collections.abc import Callable
from project.system.global_storage import GlobalStorage

logger = logging.getLogger(__name__)

class ShutdownDetector:
    """Enhanced utility class for detecting Python interpreter shutdown and managing resource cleanup."""

    _shutdown_lock = threading.Lock()
    _shutting_down = False
    _cleanup_functions: list[tuple[Callable[[], None], str]] = []
    _registered = False

    @classmethod
    def register_cleanup_function(cls, func: Callable[[], None], name: str) -> None:
        """Register a cleanup function to be called during shutdown."""
        with cls._shutdown_lock:
            cls._cleanup_functions.append((func, name))
            if not cls._registered:
                atexit.register(cls._execute_registered_cleanup)
                cls._registered = True

    @classmethod
    def _execute_registered_cleanup(cls) -> None:
        """Execute all registered cleanup functions."""
        logger.info("Executing registered cleanup functions")

        with cls._shutdown_lock:
            functions = cls._cleanup_functions.copy()
            cls._cleanup_functions.clear()

        for func, name in functions:
            try:
                logger.debug(f"Executing cleanup: {name}")
                func()
            except Exception as e:
                logger.warning(f"Error during cleanup {name}: {str(e)}")

    @classmethod
    def is_shutting_down(cls) -> bool:
        """Check if Python interpreter is shutting down."""
        with cls._shutdown_lock:
            # Check if sys.meta_path is None (indicator of shutdown)
            if not hasattr(sys, 'meta_path') or sys.meta_path is None:  # type: ignore[comparison-overlap]
                cls._shutting_down = True
                return True

            # Check if we're in a cleanup phase
            if cls._shutting_down:
                return True

            return False

    @classmethod
    def set_shutting_down(cls, value: bool = True) -> None:
        """Manually set shutdown flag."""
        with cls._shutdown_lock:
            cls._shutting_down = value

    @classmethod
    def safe_cleanup(cls, cleanup_func: Callable[[], None], name: str = "cleanup") -> None:
        """
        Safely execute cleanup function only if not shutting down.

        Args:
            cleanup_func: The cleanup function to execute
            name: Name of the cleanup operation for logging
        """
        if cls.is_shutting_down():
            logger.info(f"Skipping {name} during Python shutdown")
            return

        try:
            cleanup_func()
        except Exception as e:
            logger.warning(f"Error during {name}: {str(e)}")

    @classmethod
    def register_resource_manager_cleanup(cls) -> None:
        """Register resource manager cleanup for graceful shutdown."""
        def cleanup_resources() -> None:
            try:
                resource_manager = GlobalStorage.retrieve('ui_resource_manager')
                if resource_manager and hasattr(resource_manager, 'shutdown'):
                    resource_manager.shutdown()
            except Exception as e:
                logger.warning(f"Error during resource manager cleanup: {str(e)}")

        cls.register_cleanup_function(cleanup_resources, "Resource Manager")

def is_python_shutting_down() -> bool:
    """Check if Python interpreter is shutting down."""
    return ShutdownDetector.is_shutting_down()

def safe_cleanup(cleanup_func: Callable[[], None], name: str = "cleanup") -> None:
    """Safely execute cleanup function only if not shutting down."""
    ShutdownDetector.safe_cleanup(cleanup_func, name)

def register_cleanup_function(func: Callable[[], None], name: str) -> None:
    """Register a cleanup function to be called during shutdown."""
    ShutdownDetector.register_cleanup_function(func, name)