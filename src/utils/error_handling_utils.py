"""
Enhanced Error Handling Utilities
Provides comprehensive error handling, performance monitoring, and memory management.
"""

import gc
import logging
import socket
import threading
import time
import urllib.error
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

import psutil

from src.utils.print_utils import print_error, print_info, print_warning
from src.utils.unified_error_handler import safe_callback_execution


def safe_graph_access(graph, attribute: str, default_value: Any = None) -> Any:
    """
    Safely accesses a graph attribute, returning default if not found.
    """
    try:
        return getattr(graph, attribute, default_value)
    except Exception:
        message = "Failed to access graph attribute %s: %s"
        logging.warning(message)
        print_warning(message)
        return default_value


def safe_hasattr(obj, *attributes) -> bool:
    """
    Safely checks if an object has multiple attributes.
    """
    try:
        return all(hasattr(obj, attr) for attr in attributes)
    except Exception:
        message = "Failed to check attributes %s: %s"
        logging.warning(message)
        print_warning(message)
        return False


def create_safe_callback(callback: Callable, _context: str = "callback") -> Callable:
    """
    Creates a safe wrapper for a callback function.
    """
    def safe_wrapper(*args, **kwargs):
        return safe_callback_execution(callback, *args, **kwargs)
    return safe_wrapper


# Enhanced Error Handling Classes and Decorators

class ErrorContext:
    """Context manager for error handling with automatic cleanup."""

    def __init__(self, context: str, cleanup_func: Optional[Callable] = None,
                 log_level: str = "warning"):
        self.context = context
        self.cleanup_func = cleanup_func
        self.log_level = log_level
        self.start_time = None
        self.memory_start = None

    def __enter__(self):
        self.start_time = time.time()
        try:
            self.memory_start = psutil.Process().memory_info().rss
        except Exception:
            self.memory_start = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            duration = time.time() - self.start_time
            memory_used = None
            if self.memory_start is not None:
                try:
                    memory_used = psutil.Process().memory_info().rss - self.memory_start
                except Exception:
                    pass

            error_msg = "Error in %s: %s"
            if duration > 1.0:
                error_msg += f" (took {duration:.2f}s)"
            if memory_used is not None:
                error_msg += f" (memory: {memory_used / 1024 / 1024:.1f}MB)"

            if self.log_level == "error":
                logging.error(error_msg)
                print_error(error_msg)
            elif self.log_level == "critical":
                logging.critical(error_msg)
                print_error(f"CRITICAL: {error_msg}")
            else:
                logging.warning(error_msg)
                print_warning(error_msg)

            # Execute cleanup function if provided
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logging.error("Cleanup failed in %s: %s", self.context, cleanup_error)

        return False  # Don't suppress the exception


def handle_errors(context: str = "operation",
                  exceptions: tuple = (Exception,),
                  log_level: str = "warning",
                  cleanup_func: Optional[Callable] = None,
                  re_raise: bool = False):
    """
    Decorator for consistent error handling across functions.

    Args:
        context: Description of the operation for logging
        exceptions: Tuple of exception types to catch
        log_level: Logging level ("debug", "info", "warning", "error", "critical")
        cleanup_func: Function to call on error for cleanup
        re_raise: Whether to re-raise the exception after handling
    """
    def decorator(func: Callable) -> Callable:
        start_time = time.time()  # Capture start time in closure

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions:
                duration = time.time() - start_time

                error_msg = "Error in %s: %s"
                if duration > 0.1:
                    error_msg += f" (took {duration:.3f}s)"

                # Log based on level
                if log_level == "error":
                    logging.error(error_msg)
                    print_error(error_msg)
                elif log_level == "critical":
                    logging.critical(error_msg)
                    print_error(f"CRITICAL: {error_msg}")
                elif log_level == "info":
                    logging.info(error_msg)
                    print_info(error_msg)
                elif log_level == "debug":
                    logging.debug(error_msg)
                else:  # warning
                    logging.warning(error_msg)
                    print_warning(error_msg)

                # Execute cleanup
                if cleanup_func:
                    try:
                        cleanup_func()
                    except Exception as cleanup_error:
                        logging.error("Cleanup failed in %s: %s", context, cleanup_error)

                # Re-raise if requested
                if re_raise:
                    raise

                return None

        return wrapper
    return decorator


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance and memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = None
        try:
            start_memory = psutil.Process().memory_info().rss
        except Exception:
            pass

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            memory_used = None
            if start_memory is not None:
                try:
                    memory_used = psutil.Process().memory_info().rss - start_memory
                except Exception:
                    pass

            # Log performance metrics for slow operations
            if duration > 1.0:  # Only log if operation took more than 1 second
                perf_msg = "Performance: %s took %.3f"
                if memory_used is not None:
                    perf_msg += f", memory: {memory_used / 1024 / 1024:.1f}MB"
                logging.info(perf_msg)

    return wrapper


@contextmanager
def memory_guard(memory_limit_mb: int = 1000, cleanup_func: Optional[Callable] = None):
    """Context manager to monitor and limit memory usage."""
    process = psutil.Process()
    start_memory = process.memory_info().rss

    try:
        yield
    finally:
        current_memory = process.memory_info().rss
        memory_used = current_memory - start_memory

        # Check if memory limit exceeded
        if memory_used > memory_limit_mb * 1024 * 1024:
            warning_msg = "Memory usage exceeded limit: %.1fMB (limit: %sMB)"
            logging.warning(warning_msg)
            print_warning(warning_msg)

            # Force garbage collection
            collected = gc.collect()
            logging.info("Garbage collection freed %s objects", collected)

            # Execute cleanup if provided
            if cleanup_func:
                try:
                    cleanup_func()
                except Exception as e:
                    logging.error("Memory cleanup failed: %s", e)


def thread_safe_operation(lock: threading.Lock, timeout: float = 5.0):
    """Decorator to make operations thread-safe with timeout."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not lock.acquire(timeout=timeout):
                error_msg = "Could not acquire lock for %s within %ss"
                logging.error(error_msg)
                raise TimeoutError(error_msg)

            try:
                return func(*args, **kwargs)
            finally:
                lock.release()

        return wrapper
    return decorator


class ResourceManager:
    """Manages resources with automatic cleanup."""

    def __init__(self):
        self.resources = []
        self.lock = threading.RLock()

    def register_resource(self, resource: Any, cleanup_func: Callable):
        """Register a resource for automatic cleanup."""
        with self.lock:
            self.resources.append((resource, cleanup_func))

    def cleanup_all(self):
        """Clean up all registered resources."""
        with self.lock:
            for resource, cleanup_func in reversed(self.resources):
                try:
                    cleanup_func(resource)
                except Exception as e:
                    logging.error("Failed to cleanup resource: %s", e)
            self.resources.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()


# Utility functions for common error patterns

def safe_tensor_operation(operation: Callable, _context: str = "tensor operation",
                          fallback_value: Any = None) -> Any:
    """Safely execute tensor operations with proper error handling."""
    try:
        return operation()
    except (RuntimeError, ValueError, TypeError):
        error_msg = "Tensor operation failed in %s: %s"
        logging.warning(error_msg)
        print_warning(error_msg)
        return fallback_value
    except Exception:
        error_msg = "Unexpected error in tensor operation %s: %s"
        logging.error(error_msg)
        print_error(error_msg)
        return fallback_value


def safe_file_operation(_file_path: str, operation: Callable,
                        _context: str = "file operation") -> Any:
    """Safely execute file operations with proper error handling."""
    try:
        return operation()
    except (FileNotFoundError, PermissionError, OSError):
        error_msg = "File operation failed in %s for %s: %s"
        logging.error(error_msg)
        print_error(error_msg)
        return None
    except Exception:
        error_msg = "Unexpected error in file operation %s: %s"
        logging.error(error_msg)
        print_error(error_msg)
        return None


def safe_network_operation(operation: Callable, _context: str = "network operation",
                           _timeout: float = 10.0, retries: int = 3) -> Any:
    """Safely execute network operations with retry logic."""

    for attempt in range(retries):
        try:
            return operation()
        except (socket.timeout, socket.gaierror, urllib.error.URLError,
                ConnectionError, TimeoutError) as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.warning("Network operation failed in %s, retrying in %ss: %s",
                               _context, wait_time, e)
                time.sleep(wait_time)
            else:
                error_msg = "Network operation failed in %s after %s attempts: %s"
                logging.error(error_msg)
                print_error(error_msg)
                return None
        except Exception:
            error_msg = "Unexpected error in network operation %s: %s"
            logging.error(error_msg)
            print_error(error_msg)
            return None

    return None  # Ensure consistent return type


# Global resource manager instance
_global_resource_manager = ResourceManager()

def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    return _global_resource_manager


# Additional utility functions from exception_utils.py

def safe_graph_operation(operation: Callable, _graph_context: str = "graph",
                         error_msg: str = "Graph operation failed") -> Any:
    """
    Safely perform graph operations with consistent error handling.
    Replaces repeated graph operation error patterns.
    """
    try:
        return operation()
    except Exception as e:
        logging.warning("%s: %s", error_msg, e)
        print_warning(f"{error_msg}: {e}")
        return None


def handle_critical_error(error: Exception, context: str = "",
                        fallback_action: Optional[Callable] = None) -> None:
    """
    Handle critical errors with consistent logging and fallback.
    Replaces repeated critical error handling patterns.
    """
    error_message = f"Critical error in {context}" if context else "Critical error"
    logging.error("%s: %s", error_message, error)
    print_error(f"{error_message}: {error}")

    if fallback_action:
        try:
            fallback_action()
        except Exception as fallback_e:
            logging.error("Fallback action also failed: %s", fallback_e)
            print_error(f"Fallback action also failed: {fallback_e}")


def log_and_continue(error: Exception, context: str = "",
                    continue_msg: str = "Continuing with fallback") -> None:
    """
    Log error and continue execution with fallback.
    Replaces repeated log-and-continue patterns.
    """
    logging.warning("Error in %s: %s", context, error)
    print_warning(f"Error in {context}: {error}")
    logging.info(continue_msg)
    print_info(continue_msg)

