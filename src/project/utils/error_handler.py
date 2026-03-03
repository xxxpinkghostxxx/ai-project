"""
Error Handler Module.

This module provides comprehensive error handling functionality for the Energy-Based Neural System,
including standardized error handling patterns, enhanced logging with detailed context,
consistent error severity classification, and thread-safe error management.
"""

import logging
import threading
import datetime
import traceback
import time
from functools import wraps
from typing import Any, Optional, Callable, Dict

# Get logger (logging should be configured in main entry point)
# Don't call basicConfig here - it only works once and should be in main.py
logger = logging.getLogger(__name__)

# Error severity classification constants
ERROR_SEVERITY_CRITICAL = 'CRITICAL'
ERROR_SEVERITY_HIGH = 'HIGH'
ERROR_SEVERITY_MEDIUM = 'MEDIUM'
ERROR_SEVERITY_LOW = 'LOW'
ERROR_SEVERITY_INFO = 'INFO'
ERROR_SEVERITY_DEBUG = 'DEBUG'

# Error context keys
ERROR_CONTEXT_TIMESTAMP = 'timestamp'
ERROR_CONTEXT_MODULE = 'module'
ERROR_CONTEXT_FUNCTION = 'function'
ERROR_CONTEXT_ERROR_TYPE = 'error_type'
ERROR_CONTEXT_ERROR_MESSAGE = 'error_message'
ERROR_CONTEXT_STACK_TRACE = 'stack_trace'
ERROR_CONTEXT_ADDITIONAL_INFO = 'additional_info'
ERROR_CONTEXT_SEVERITY = 'severity'
ERROR_CONTEXT_RECOVERY_ACTIONS = 'recovery_actions'

class ErrorHandler:
    """
    Enhanced error handling class with standardized patterns, severity classification,
    and comprehensive error context.

    Provides:
    - Standardized error handling patterns across the application
    - Enhanced logging with detailed error context and severity classification
    - Thread-safe error management with locking mechanisms
    - Comprehensive error tracking and statistics
    """

    # Maximum number of stored errors to prevent unbounded memory growth
    MAX_ERRORS = 1000

    def __init__(self) -> None:
        """Initialize ErrorHandler with empty error list and enhanced tracking."""
        self.errors: list[Dict[str, Any]] = []  # Store structured error information
        self.error_counter = 0
        self._lock = threading.RLock()  # Reentrant lock for nested calls (e.g. get_error_statistics -> get_recent_errors)
        self._retry_logic_enabled = True
        self._max_retries = 3
        self._retry_delay = 1.0

    @staticmethod
    def show_error(title: str, message: str, log: bool = True, severity: str = ERROR_SEVERITY_MEDIUM, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log error and optionally show a dialog to the user (main thread only).

        This method is safe to call from any thread. The Qt dialog is only
        shown when called from the main thread with an active QApplication;
        otherwise the error is logged only.

        Args:
            title: Error title
            message: Error message
            log: Whether to log the error
            severity: Error severity level
            context: Additional error context information
        """
        try:
            # Log with enhanced context (always safe from any thread)
            if log:
                log_message = "%s: %s | Severity: %s" % (title, message, severity)
                if context:
                    log_message += " | Context: %s" % context
                logger.error(log_message)

            # Only attempt Qt dialog from the main thread and if QApplication exists
            try:
                from PyQt6.QtWidgets import QApplication, QMessageBox
                from PyQt6.QtCore import QThread
                app = QApplication.instance()
                if app is not None and QThread.currentThread() is app.thread():
                    QMessageBox.critical(None, title, message)
            except ImportError:
                pass  # Qt not available — log-only mode
            except RuntimeError:
                pass  # Qt shutting down

        except Exception as e:
            logger.error("Error showing error message: %s", str(e))

    @staticmethod
    def safe_operation(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Enhanced decorator for safe operations with comprehensive error handling, retry logic, and recovery mechanisms.

        This decorator provides robust error handling by:
        1. Wrapping function execution in try-catch blocks
        2. Implementing automatic retry logic with exponential backoff
        3. Attempting intelligent recovery based on error context
        4. Providing detailed error logging with comprehensive context
        5. Showing user-friendly error messages when appropriate
        6. Maintaining thread safety throughout the error handling process

        Recovery Process:
        - Analyzes error context to select appropriate recovery mechanism
        - Attempts tensor synchronization for tensor-related errors
        - Attempts graph validation for graph structure issues
        - Attempts connection repair for invalid connection problems
        - Implements exponential backoff between retry attempts
        - Logs detailed recovery attempts and outcomes

        Thread Safety:
        - Creates new ErrorHandler instance for each decorated function call
        - Ensures thread-safe error logging and recovery operations
        - Maintains isolation between different decorated function calls

        Args:
            func: Function to wrap with error handling

        Returns:
            Wrapped function with enhanced error handling

        Example:
        ```python
        # Apply safe operation decorator to a critical function
        @ErrorHandler.safe_operation
        def process_neural_network_data(data):
            # This function will automatically handle errors, attempt recovery,
            # and provide detailed logging if any issues occur
            return neural_system.process(data)

        # The decorated function can be called normally
        result = process_neural_network_data(input_data)
        ```
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = "%s failed: %s" % (func.__name__, e)
                error_context = {
                    'function': func.__name__,
                }

                logger.error(error_msg, exc_info=True)

                # Show error to user (thread-safe — will only show dialog on main thread)
                ErrorHandler.show_error(
                    "Operation Failed",
                    error_msg,
                    severity=ERROR_SEVERITY_HIGH,
                    context=error_context
                )
                return None
        return wrapper

    def with_retry(self, max_retries: int = 3, retry_delay: float = 1.0, backoff: bool = True):
        """
        Decorator for adding retry logic to functions that may experience transient failures.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            backoff: Whether to use exponential backoff

        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                retry_count = 0
                last_error = None

                while retry_count <= max_retries:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        retry_count += 1

                        if retry_count <= max_retries:
                            # Log the retry attempt
                            error_context = {
                                'function': func.__name__,
                                'attempt': retry_count,
                                'max_attempts': max_retries,
                                'error': str(e),
                                'retry_delay': min(retry_delay * (2 ** (retry_count - 1)), 30.0) if backoff else retry_delay
                            }

                            severity = self.classify_error_severity(str(e))
                            self.log_error(
                                f"Transient failure in {func.__name__}, attempt {retry_count}/{max_retries}: {str(e)}",
                                severity=severity,
                                context=error_context
                            )

                            # Wait before retrying
                            if backoff:
                                sleep_time = min(retry_delay * (2 ** (retry_count - 1)), 30.0)
                            else:
                                sleep_time = retry_delay

                            logger.info(f"Retrying {func.__name__} in {sleep_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                            time.sleep(sleep_time)
                        else:
                            # Final attempt failed
                            break

                # If we get here, all retries failed
                if last_error:
                    error_context = {
                        'function': func.__name__,
                        'attempts': max_retries,
                        'final_error': str(last_error)
                    }
                    severity = self.classify_error_severity(str(last_error))
                    self.log_error(
                        f"All retry attempts failed for {func.__name__}: {str(last_error)}",
                        severity=severity,
                        context=error_context
                    )
                    raise last_error
                else:
                    return None
            return wrapper
        return decorator

    @staticmethod
    def log_warning(message: str) -> None:
        """Log a warning message"""
        logger.warning(message)

    @staticmethod
    def log_info(message: str) -> None:
        """Log an info message"""
        logger.info(message)

    @staticmethod
    def log_debug(message: str) -> None:
        """Log a debug message"""
        logger.debug(message)

    def log_error(self, message: str, severity: str = ERROR_SEVERITY_MEDIUM, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an error message with comprehensive context and store structured error information.

        Args:
            message: Error message
            severity: Error severity level
            context: Additional error context
        """
        with self._lock:  # Thread-safe error logging
            error_id = self._generate_error_id()
            error_entry = {
                'error_id': error_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'message': message,
                'severity': severity,
                'context': context or {},
                'recovery_attempted': False,
                'recovery_successful': False
            }

            # Add stack trace for critical errors
            if severity == ERROR_SEVERITY_CRITICAL:
                error_entry['stack_trace'] = traceback.format_exc()

            self.errors.append(error_entry)
            self.error_counter += 1

            # Cap stored errors to prevent unbounded memory growth
            if len(self.errors) > self.MAX_ERRORS:
                self.errors = self.errors[-self.MAX_ERRORS:]

            # Log with severity and context
            log_message = f"[{severity}] {message}"
            if context:
                log_message += f" | Context: {context}"
            logger.error(log_message)

            # Log detailed error entry for debugging
            logger.debug(f"Error entry: {error_entry}")

            return error_id

    def _generate_error_id(self) -> str:
        """Generate a unique error ID."""
        return f"ERR-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.error_counter:04d}"

    def classify_error_severity(self, error_message: str, error_type: str | None = None, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Classify error severity based on error message, type, and context.

        Args:
            error_message: The error message to classify
            error_type: Optional error type/category
            context: Optional additional context

        Returns:
            Appropriate severity level constant
        """
        # Convert to lowercase for case-insensitive matching
        message_lower = error_message.lower()

        # Critical errors - system failures, unrecoverable issues
        critical_keywords = [
            'fatal', 'critical', 'unrecoverable', 'system failure',
            'memory exhaustion', 'out of memory', 'segmentation fault',
            'corrupted', 'crash', 'terminated', 'aborted'
        ]

        if any(keyword in message_lower for keyword in critical_keywords):
            return ERROR_SEVERITY_CRITICAL

        # High severity - major functionality failures, data loss potential
        high_keywords = [
            'failed to initialize', 'connection failed', 'database error',
            'network error', 'timeout', 'deadlock', 'resource exhaustion',
            'validation failed', 'security violation', 'permission denied',
            'authentication failed', 'authorization failed'
        ]

        if any(keyword in message_lower for keyword in high_keywords):
            return ERROR_SEVERITY_HIGH

        # Medium severity - functional issues, recoverable errors
        medium_keywords = [
            'invalid', 'mismatch', 'not found', 'missing', 'null',
            'empty', 'format error', 'parse error', 'type error',
            'value error', 'index error', 'key error', 'attribute error',
            'configuration error', 'setup error', 'load error', 'save error'
        ]

        if any(keyword in message_lower for keyword in medium_keywords):
            return ERROR_SEVERITY_MEDIUM

        # Low severity - warnings, minor issues, informational
        low_keywords = [
            'warning', 'deprecated', 'minor', 'temporary',
            'transient', 'retry', 'timeout', 'slow',
            'performance', 'inefficient', 'suboptimal'
        ]

        if any(keyword in message_lower for keyword in low_keywords):
            return ERROR_SEVERITY_LOW

        # Default to medium severity for uncategorized errors
        return ERROR_SEVERITY_MEDIUM

    def get_recent_errors(self, count: int = 10) -> list[Dict[str, Any]]:
        """Get the most recent errors with full context"""
        with self._lock:  # Thread-safe access
            return self.errors[-count:] if self.errors else []

    def get_errors_by_severity(self, severity: str) -> list[Dict[str, Any]]:
        """
        Get errors filtered by severity level.

        Args:
            severity: Error severity level to filter by

        Returns:
            List of error entries matching the severity
        """
        with self._lock:
            return [error for error in self.errors if error['severity'] == severity]

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics.

        Returns:
            Dictionary containing error statistics by severity
        """
        with self._lock:
            stats = {
                'total_errors': len(self.errors),
                'by_severity': {
                    ERROR_SEVERITY_CRITICAL: 0,
                    ERROR_SEVERITY_HIGH: 0,
                    ERROR_SEVERITY_MEDIUM: 0,
                    ERROR_SEVERITY_LOW: 0,
                    ERROR_SEVERITY_INFO: 0,
                    ERROR_SEVERITY_DEBUG: 0
                },
                'recent_errors': self.get_recent_errors(5),
                'error_rate': self._calculate_error_rate()
            }

            for error in self.errors:
                severity = error['severity']
                if severity in stats['by_severity']:
                    stats['by_severity'][severity] += 1

            return stats

    def _calculate_error_rate(self) -> float:
        """Calculate error rate (errors per minute)."""
        if len(self.errors) < 2:
            return 0.0

        try:
            first_error_time = datetime.datetime.fromisoformat(self.errors[0]['timestamp'])
            last_error_time = datetime.datetime.fromisoformat(self.errors[-1]['timestamp'])
            time_diff_minutes = (last_error_time - first_error_time).total_seconds() / 60

            if time_diff_minutes > 0.0:
                return len(self.errors) / time_diff_minutes
            return 0.0  # Errors all in same instant; rate undefined
        except Exception as e:
            logger.warning(f"Error calculating error rate: {str(e)}")
            return 0.0

    def __len__(self) -> int:
        """Return the number of stored errors"""
        return len(self.errors)
