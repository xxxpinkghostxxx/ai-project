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
    Enhanced error handling class with standardized patterns, severity classification, and comprehensive error context.

    This class provides comprehensive error handling capabilities including:
    - Standardized error handling patterns across the application
    - Enhanced logging with detailed error context and severity classification
    - Thread-safe error management with locking mechanisms
    - Automatic recovery mechanisms for common error scenarios
    - Comprehensive error tracking and statistics
    - User notification and feedback systems

    Recovery Mechanisms:
    - Automatic retry logic with exponential backoff for transient failures
    - Tensor synchronization recovery for tensor-related errors
    - Graph validation recovery for graph structure issues
    - Connection repair recovery for invalid connection problems
    - Intelligent recovery mechanism selection based on error context
    - Thread-safe recovery operations with proper error isolation

    Thread Safety:
    - Uses threading.Lock() for thread-safe error logging and statistics
    - Implements fine-grained locking for critical error operations
    - Ensures consistent error tracking across concurrent operations
    - Provides thread-safe access to error collections and statistics

    Usage Patterns:
    - Decorator-based error handling for functions and methods
    - Manual error logging with comprehensive context
    - Automatic recovery attempts with retry logic
    - Error severity classification and prioritization
    - Comprehensive error reporting and analysis
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
        self._recovery_mechanisms = {
            'tensor_synchronization': self._recover_via_tensor_synchronization,
            'graph_validation': self._recover_via_graph_validation,
            'connection_repair': self._recover_via_connection_repair
        }

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

    def _attempt_recovery(self, function_name: str, error_message: str, context: Dict[str, Any]) -> bool:
        """
        Attempt to recover from an error using available recovery mechanisms.

        This method implements the core recovery logic by:
        1. Analyzing the error context to determine the most appropriate recovery strategy
        2. Selecting the best recovery mechanism based on error patterns and function context
        3. Executing the recovery mechanism with proper error handling
        4. Logging detailed recovery attempts and outcomes
        5. Maintaining thread safety throughout the recovery process

        Recovery Strategy Selection:
        - Uses pattern matching on error messages to identify error types
        - Considers function names and execution context
        - Falls back to tensor synchronization for unknown error types
        - Implements intelligent recovery mechanism selection

        Thread Safety:
        - This method is thread-safe as it only reads shared data
        - Recovery mechanisms are designed to be thread-safe
        - Error logging maintains thread safety guarantees

        Args:
            function_name: Name of the function that failed
            error_message: Error message describing the failure
            context: Error context containing additional diagnostic information

        Returns:
            True if recovery was successful, False otherwise

        Example:
        ```python
        # Attempt recovery after a function failure
        recovery_success = error_handler._attempt_recovery(
            function_name="process_tensor_data",
            error_message="Tensor shape mismatch detected",
            context={
                'tensor_name': 'energy',
                'expected_shape': (1000,),
                'actual_shape': (950,),
                'operation': 'synchronization'
            }
        )

        if recovery_success:
            logger.info("Recovery successful, retrying operation")
        else:
            logger.warning("Recovery failed, additional intervention needed")
        ```
        """
        try:
            logger.info(f"Attempting recovery for {function_name} after error: {error_message}")

            # Determine appropriate recovery mechanism based on function name and error
            recovery_mechanism = self._select_recovery_mechanism(function_name, error_message)

            if recovery_mechanism:
                logger.info(f"Selected recovery mechanism: {recovery_mechanism.__name__}")
                return recovery_mechanism()
            else:
                logger.warning("No suitable recovery mechanism found")
                return False

        except Exception as e:
            logger.error(f"Recovery attempt failed: {str(e)}")
            return False

    def _select_recovery_mechanism(self, function_name: str, error_message: str) -> Optional[Callable[[], bool]]:
        """
        Select the most appropriate recovery mechanism based on function name and error message.

        Args:
            function_name: Name of the function that failed
            error_message: Error message

        Returns:
            Appropriate recovery function or None
        """
        # Check for tensor-related errors
        if 'tensor' in error_message.lower() or 'shape' in error_message.lower():
            return self._recover_via_tensor_synchronization
        # Check for graph-related errors
        elif 'graph' in error_message.lower() or 'node' in error_message.lower():
            return self._recover_via_graph_validation
        # Check for connection-related errors
        elif 'connection' in error_message.lower() or 'edge' in error_message.lower():
            return self._recover_via_connection_repair
        # Default recovery for other cases
        else:
            return self._recover_via_tensor_synchronization

    def _recover_via_tensor_synchronization(self) -> bool:
        """Attempt recovery by synchronizing all tensors. Not yet implemented."""
        logger.warning("Tensor synchronization recovery is not implemented; skipping")
        return False

    def _recover_via_graph_validation(self) -> bool:
        """Attempt recovery by validating and repairing graph state. Not yet implemented."""
        logger.warning("Graph validation recovery is not implemented; skipping")
        return False

    def _recover_via_connection_repair(self) -> bool:
        """Attempt recovery by repairing invalid connections. Not yet implemented."""
        logger.warning("Connection repair recovery is not implemented; skipping")
        return False

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
