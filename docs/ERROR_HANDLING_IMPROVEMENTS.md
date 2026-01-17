# Error Handling and Logging Improvements Documentation

## Overview

This document provides comprehensive documentation for the error handling and logging improvements implemented across the PyTorch Geometric Neural System. These improvements enhance system stability, debugging capabilities, and overall robustness.

## Table of Contents

1. [Standardized Error Handling Patterns](#standardized-error-handling-patterns)
2. [Enhanced Logging with Detailed Context](#enhanced-logging-with-detailed-context)
3. [Improved Recovery Mechanisms](#improved-recovery-mechanisms)
4. [Comprehensive Error Reporting](#comprehensive-error-reporting)
5. [Consistent Error Severity Classification](#consistent-error-severity-classification)
6. [Retry Logic for Transient Failures](#retry-logic-for-transient-failures)
7. [Thread-Safe Error Handling](#thread-safe-error-handling)
8. [Error Context in All Messages](#error-context-in-all-messages)
9. [Usage Examples](#usage-examples)
10. [Best Practices](#best-practices)
11. [Migration Guide](#migration-guide)

## Standardized Error Handling Patterns

### Error Handler Enhancements

The `ErrorHandler` class has been significantly enhanced with:

- **Structured Error Storage**: Errors are now stored as dictionaries with comprehensive context
- **Error Severity Constants**: Standardized severity levels (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`)
- **Error Context Keys**: Standardized context keys for consistent error reporting
- **Thread-Safe Operations**: All error operations are now thread-safe using locking mechanisms

### Key Constants

```python
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
```

### Enhanced Error Storage

```python
class ErrorHandler:
    def __init__(self) -> None:
        self.errors: list[dict] = []  # Store structured error information
        self.error_counter = 0
        self._lock = threading.Lock()  # Thread-safe error handling
        self._retry_logic_enabled = True
        self._max_retries = 3
        self._retry_delay = 1.0
```

## Enhanced Logging with Detailed Context

### Enhanced `show_error` Method

```python
@staticmethod
def show_error(title: str, message: str, log: bool = True, severity: str = ERROR_SEVERITY_MEDIUM, context: dict = None) -> None:
    """
    Show error message to user with enhanced logging and error context.

    Args:
        title: Error title
        message: Error message
        log: Whether to log the error
        severity: Error severity level
        context: Additional error context information
    """
    try:
        # Create comprehensive error context
        error_context = {
            ERROR_CONTEXT_TIMESTAMP: datetime.datetime.now().isoformat(),
            ERROR_CONTEXT_MODULE: 'UI',
            ERROR_CONTEXT_FUNCTION: 'show_error',
            ERROR_CONTEXT_ERROR_TYPE: title,
            ERROR_CONTEXT_ERROR_MESSAGE: message,
            ERROR_CONTEXT_SEVERITY: severity,
            ERROR_CONTEXT_ADDITIONAL_INFO: context or {}
        }

        # Add stack trace for critical errors
        if severity == ERROR_SEVERITY_CRITICAL:
            error_context[ERROR_CONTEXT_STACK_TRACE] = traceback.format_exc()

        # Show error to user
        messagebox.showerror(title, message)

        # Log with enhanced context
        if log:
            log_message = f"{title}: {message} | Severity: {severity}"
            if context:
                log_message += f" | Context: {context}"
            logger.error(log_message)

            # Log detailed error context
            logger.debug(f"Error context: {error_context}")

    except Exception as e:
        logger.error("Error showing error message: %s", str(e))
        print("Error showing error message: %s", str(e))
```

### Enhanced `log_error` Method

```python
def log_error(self, message: str, severity: str = ERROR_SEVERITY_MEDIUM, context: dict = None) -> str:
    """
    Log an error message with comprehensive context and store structured error information.

    Args:
        message: Error message
        severity: Error severity level
        context: Additional error context

    Returns:
        Generated error ID
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

        # Log with severity and context
        log_message = f"[{severity}] {message}"
        if context:
            log_message += f" | Context: {context}"
        logger.error(log_message)

        # Log detailed error entry for debugging
        logger.debug(f"Error entry: {error_entry}")

        return error_id
```

## Improved Recovery Mechanisms

### Enhanced `safe_operation` Decorator

```python
@staticmethod
def safe_operation(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Enhanced decorator for safe operations with comprehensive error handling, retry logic, and recovery mechanisms.

    Args:
        func: Function to wrap with error handling

    Returns:
        Wrapped function with enhanced error handling
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        error_handler = ErrorHandler()
        retry_count = 0
        last_error = None
        error_msg = ""
        error_context = {}

        while retry_count <= error_handler._max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_msg = f"{func.__name__} failed: {str(e)}"
                error_context = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'retry_count': retry_count,
                    'max_retries': error_handler._max_retries
                }

                # Log with enhanced context
                logger.error(error_msg, exc_info=True)
                logger.debug(f"Error context: {error_context}")

                # Attempt recovery if this is a recoverable error
                if retry_count < error_handler._max_retries:
                    recovery_success = error_handler._attempt_recovery(func.__name__, str(e), error_context)
                    if recovery_success:
                        logger.info(f"Recovery successful for {func.__name__}, retrying...")
                        retry_count += 1
                        time.sleep(error_handler._retry_delay * (2 ** retry_count))  # Exponential backoff
                        continue

                # Show error to user
                ErrorHandler.show_error(
                    "Operation Failed",
                    error_msg,
                    severity=ERROR_SEVERITY_HIGH,
                    context=error_context
                )
                break

        # Store error with comprehensive context
        if error_msg and error_context:
            error_handler.log_error(error_msg, severity=ERROR_SEVERITY_HIGH, context=error_context)
        return None
    return wrapper
```

### Recovery Mechanism Selection

```python
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
```

## Comprehensive Error Reporting

### Error Statistics and Reporting

```python
def get_error_statistics(self) -> dict:
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

        if time_diff_minutes > 0:
            return len(self.errors) / time_diff_minutes
        else:
            return len(self.errors)
    except Exception as e:
        logger.warning(f"Error calculating error rate: {str(e)}")
        return 0.0
```

### Tensor Manager Error Reporting

```python
def log_tensor_error(self, error_context: dict) -> str:
    """
    Log tensor-related errors with comprehensive context and update statistics.

    Args:
        error_context: Dictionary containing error context information

    Returns:
        Generated error ID
    """
    with self._lock:  # Thread-safe error logging
        # Generate error ID
        self.error_counter += 1
        error_id = f"TENSOR-ERR-{int(time.time())}-{self.error_counter:04d}"

        # Update error context with ID
        error_context['error_id'] = error_id

        # Update statistics
        self.error_statistics['total_errors'] += 1
        severity = error_context.get(ERROR_CONTEXT_SEVERITY, ERROR_SEVERITY_MEDIUM)
        if severity in self.error_statistics['by_severity']:
            self.error_statistics['by_severity'][severity] += 1

        # Calculate recovery success rate
        if self.error_statistics['total_errors'] > 0:
            self.error_statistics['recovery_success_rate'] = (
                self.successful_recoveries / self.error_statistics['total_errors']
            )

        # Log the error
        severity_tag = error_context.get(ERROR_CONTEXT_SEVERITY, ERROR_SEVERITY_MEDIUM)
        log_message = f"[{severity_tag}] Tensor Error {error_id}: {error_context[ERROR_CONTEXT_ERROR_MESSAGE]}"

        if ERROR_CONTEXT_ADDITIONAL_INFO in error_context:
            log_message += f" | Additional Info: {error_context[ERROR_CONTEXT_ADDITIONAL_INFO]}"

        logger.error(log_message)
        logger.debug(f"Tensor error context: {error_context}")

        # Update last error timestamp
        self.error_statistics['last_error_timestamp'] = error_context[ERROR_CONTEXT_TIMESTAMP]

        return error_id
```

## Consistent Error Severity Classification

### Error Severity Classifier

```python
def classify_error_severity(self, error_message: str, error_type: str = None, context: dict = None) -> str:
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
    type_lower = error_type.lower() if error_type else ''

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
```

### Severity-Based Recommendations

```python
def get_severity_recommendations(self, severity: str) -> dict:
    """
    Get recommendations based on error severity level.

    Args:
        severity: Error severity level

    Returns:
        Dictionary with action recommendations
    """
    recommendations = {
        'actions': [],
        'priority': '',
        'notification_level': ''
    }

    if severity == ERROR_SEVERITY_CRITICAL:
        recommendations['actions'] = [
            "Immediate attention required",
            "Notify system administrators",
            "Attempt automatic recovery if possible",
            "Log detailed diagnostic information",
            "Consider system restart if safe"
        ]
        recommendations['priority'] = "HIGHEST"
        recommendations['notification_level'] = "ALERT"
    elif severity == ERROR_SEVERITY_HIGH:
        recommendations['actions'] = [
            "High priority attention required",
            "Notify technical staff",
            "Attempt automatic recovery",
            "Monitor for recurrence",
            "Log detailed error context"
        ]
        recommendations['priority'] = "HIGH"
        recommendations['notification_level'] = "WARNING"
    elif severity == ERROR_SEVERITY_MEDIUM:
        recommendations['actions'] = [
            "Investigate during next maintenance cycle",
            "Log error for analysis",
            "Monitor for patterns",
            "Consider code improvements"
        ]
        recommendations['priority'] = "MEDIUM"
        recommendations['notification_level'] = "NOTICE"
    elif severity == ERROR_SEVERITY_LOW:
        recommendations['actions'] = [
            "Log for informational purposes",
            "Monitor for recurrence",
            "Consider optimization opportunities"
        ]
        recommendations['priority'] = "LOW"
        recommendations['notification_level'] = "INFO"
    else:
        recommendations['actions'] = [
            "Informational message only",
            "No immediate action required"
        ]
        recommendations['priority'] = "MINIMAL"
        recommendations['notification_level'] = "DEBUG"

    return recommendations
```

## Retry Logic for Transient Failures

### Enhanced Retry Decorator

```python
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
                            'retry_delay': retry_delay * (2 ** (retry_count - 1)) if backoff else retry_delay
                        }

                        severity = self.classify_error_severity(str(e))
                        self.log_error(
                            f"Transient failure in {func.__name__}, attempt {retry_count}/{max_retries}: {str(e)}",
                            severity=severity,
                            context=error_context
                        )

                        # Wait before retrying
                        if backoff:
                            sleep_time = retry_delay * (2 ** (retry_count - 1))
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
```

### System Recovery Mechanism

```python
def attempt_system_recovery() -> bool:
    """
    Attempt to recover the system from initialization failures.

    This function tries to clean up any partially initialized resources
    and reset the system to a clean state for retry.

    Returns:
        True if recovery was successful, False otherwise
    """
    try:
        logger.info("Attempting system recovery...")

        # Clean up global storage
        try:
            from src.project.system.global_storage import GlobalStorage
            GlobalStorage.clear()
            logger.info("Global storage cleared")
        except Exception as e:
            logger.warning(f"Error clearing global storage: {str(e)}")

        # Force garbage collection
        import gc
        gc.collect()
        logger.info("Garbage collection completed")

        # Reset any cached configurations
        try:
            from src.project.utils.config_manager import ConfigManager
            config_manager = ConfigManager()  # Reinitialize to clear any cached state
            logger.info("Configuration cache reset")
        except Exception as e:
            logger.warning(f"Error resetting configuration cache: {str(e)}")

        logger.info("System recovery completed successfully")
        return True

    except Exception as e:
        logger.error(f"System recovery failed: {str(e)}")
        return False
```

## Thread-Safe Error Handling

### Thread-Safe Error Handler

```python
class ErrorHandler:
    def __init__(self) -> None:
        """Initialize ErrorHandler with empty error list and enhanced tracking."""
        self.errors: list[dict] = []  # Store structured error information
        self.error_counter = 0
        self._lock = threading.Lock()  # Thread-safe error handling
        self._retry_logic_enabled = True
        self._max_retries = 3
        self._retry_delay = 1.0
```

### Thread-Safe Methods

```python
def log_error(self, message: str, severity: str = ERROR_SEVERITY_MEDIUM, context: dict = None) -> str:
    """Thread-safe error logging with locking"""
    with self._lock:  # Thread-safe error logging
        # ... implementation ...

def get_recent_errors(self, count: int = 10) -> list[dict]:
    """Thread-safe access to recent errors"""
    with self._lock:  # Thread-safe access
        return self.errors[-count:] if self.errors else []

def get_error_statistics(self) -> dict:
    """Thread-safe statistics generation"""
    with self._lock:
        # ... implementation ...
```

### Thread-Safe Tensor Manager

```python
class TensorManager:
    def __init__(self, neural_system: Any) -> None:
        # ... other initialization ...
        self._lock = threading.Lock()  # Thread-safe locking mechanism

    def validate_tensor_shapes(self) -> Dict[str, bool]:
        """Thread-safe validation"""
        with self._lock:  # Thread-safe validation
            # ... implementation ...

    def synchronize_all_tensors(self) -> Dict[str, bool]:
        """Thread-safe synchronization"""
        with self._lock:  # Thread-safe synchronization
            # ... implementation ...
```

## Error Context in All Messages

### Enhanced Error Context in UI Components

```python
# Example from modern_main_window.py
except Exception as e:
    # Enhanced error reporting with detailed context
    error_context = {
        ERROR_CONTEXT_TIMESTAMP: time.time(),
        ERROR_CONTEXT_MODULE: 'modern_main_window',
        ERROR_CONTEXT_FUNCTION: 'update_workspace_canvas',
        ERROR_CONTEXT_ERROR_TYPE: 'CanvasUpdateError',
        ERROR_CONTEXT_ERROR_MESSAGE: str(e),
        ERROR_CONTEXT_ADDITIONAL_INFO: {
            'workspace_data_shape': workspace_data.shape if workspace_data is not None else 'None',
            'config_available': 'config_available'  # Context information
        }
    }

    severity = ERROR_SEVERITY_MEDIUM
    if 'memory' in str(e).lower() or 'out of' in str(e).lower():
        severity = ERROR_SEVERITY_HIGH

    ErrorHandler.show_error(
        "Canvas Error",
        f"Failed to update workspace: {str(e)}",
        severity=severity,
        context=error_context
    )
```

### Enhanced Error Context in System Diagnostics

```python
# Example from system_diagnostics.py
except Exception as e:
    # Enhanced error logging with detailed context
    error_context = {
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

    # Store error context in details field
    return HealthStatus(
        component="error_rates",
        status="critical",
        message=f"Error checking error rates: {str(e)}",
        timestamp=time.time(),
        resolution_actions=["Check error handler functionality"],
        details={'error_context': error_context, 'severity': ERROR_SEVERITY_CRITICAL}
    )
```

## Usage Examples

### Using the Enhanced Error Handler

```python
# Initialize error handler
error_handler = ErrorHandler()

# Log errors with context
try:
    # Some operation that might fail
    result = risky_operation()
except Exception as e:
    error_context = {
        'input_data': input_data,
        'expected_output': expected_output,
        'operation_type': 'data_processing'
    }
    error_handler.log_error(
        f"Data processing failed: {str(e)}",
        severity=ERROR_SEVERITY_HIGH,
        context=error_context
    )

# Use safe operation decorator
@ErrorHandler.safe_operation
def process_data(data):
    # Data processing logic
    return processed_data

# Use retry decorator
@error_handler.with_retry(max_retries=5, retry_delay=2.0)
def fetch_external_data(url):
    # External data fetching logic
    return fetched_data
```

### Using Tensor Manager with Error Context

```python
# Initialize tensor manager
tensor_manager = TensorManager(neural_system)

# Validate tensors with automatic error logging
try:
    validation_results = tensor_manager.validate_tensor_shapes()
    if invalid_tensors := [k for k, v in validation_results.items() if not v]:
        error_context = {
            'invalid_tensors': invalid_tensors,
            'validation_results': validation_results,
            'recovery_attempted': False
        }
        tensor_manager.log_tensor_error(error_context)
except Exception as e:
    logger.error(f"Tensor validation failed: {str(e)}")
```

## Best Practices

### 1. Consistent Error Severity Classification

- Use the `classify_error_severity()` method to ensure consistent severity levels
- Follow the severity classification guidelines for proper categorization
- Use appropriate severity levels for different types of errors

### 2. Comprehensive Error Context

- Always include relevant context information with errors
- Use the standardized error context keys for consistency
- Include information that helps with debugging and recovery

### 3. Thread-Safe Operations

- Use the built-in locking mechanisms for thread safety
- Ensure all shared resources are accessed in a thread-safe manner
- Use the `@ErrorHandler.safe_operation` decorator for critical operations

### 4. Retry Logic for Transient Failures

- Use the `with_retry()` decorator for operations that may experience transient failures
- Configure appropriate retry parameters (max_retries, retry_delay, backoff)
- Monitor retry attempts and log failures appropriately

### 5. Recovery Mechanisms

- Implement appropriate recovery mechanisms for different error types
- Use the built-in recovery mechanism selection logic
- Monitor recovery success rates and adjust strategies as needed

### 6. Error Monitoring and Reporting

- Regularly review error statistics and trends
- Set up alerts for critical and high-severity errors
- Use the comprehensive error reporting features for analysis

## Migration Guide

### Upgrading from Previous Versions

1. **Update Imports**:
   ```python
   # Old import
   from src.project.utils.error_handler import ErrorHandler

   # New import (add additional constants if needed)
   from src.project.utils.error_handler import (
       ErrorHandler,
       ERROR_SEVERITY_CRITICAL, ERROR_SEVERITY_HIGH, ERROR_SEVERITY_MEDIUM,
       ERROR_SEVERITY_LOW, ERROR_CONTEXT_TIMESTAMP, ERROR_CONTEXT_MODULE,
       ERROR_CONTEXT_FUNCTION, ERROR_CONTEXT_ERROR_TYPE, ERROR_CONTEXT_ERROR_MESSAGE,
       ERROR_CONTEXT_ADDITIONAL_INFO
   )
   ```

2. **Update Error Logging**:
   ```python
   # Old error logging
   ErrorHandler.show_error("Operation Failed", "Error message")

   # New error logging with context
   ErrorHandler.show_error(
       "Operation Failed",
       "Error message",
       severity=ERROR_SEVERITY_HIGH,
       context={'operation': 'data_processing', 'input': input_data}
   )
   ```

3. **Update Error Handling**:
   ```python
   # Old error handling
   try:
       risky_operation()
   except Exception as e:
       logger.error(f"Error: {str(e)}")

   # New error handling with context and severity
   try:
       risky_operation()
   except Exception as e:
       error_handler.log_error(
           f"Operation failed: {str(e)}",
           severity=ERROR_SEVERITY_MEDIUM,
           context={'operation': 'risky_operation', 'input': input_data}
       )
   ```

4. **Add Thread Safety**:
   ```python
   # Ensure thread-safe operations
   with error_handler._lock:
       # Access shared resources
       error_stats = error_handler.get_error_statistics()
   ```

5. **Update Recovery Mechanisms**:
   ```python
   # Use the enhanced recovery mechanisms
   recovery_success = error_handler._attempt_recovery(
       function_name='data_processing',
       error_message=str(e),
       context=error_context
   )
   ```

### Backward Compatibility

The enhanced error handling system maintains backward compatibility with existing code:

- All existing error handler methods continue to work
- New features are additive and don't break existing functionality
- Default parameters ensure existing code continues to function

### Testing Recommendations

1. **Test Error Logging**:
   - Verify that errors are logged with proper context
   - Check that severity levels are correctly classified
   - Ensure thread-safe logging works in multi-threaded environments

2. **Test Recovery Mechanisms**:
   - Test automatic recovery for different error types
   - Verify that recovery attempts are properly logged
   - Check that recovery success rates are tracked

3. **Test Retry Logic**:
   - Test retry behavior for transient failures
   - Verify exponential backoff works correctly
   - Ensure retry limits are respected

4. **Test Thread Safety**:
   - Test concurrent access to error handler methods
   - Verify that locking mechanisms prevent race conditions
   - Check that thread-safe operations don't cause deadlocks

## Summary

This comprehensive error handling and logging improvement provides:

- **Standardized error handling patterns** across all modules
- **Enhanced logging with detailed context** for better debugging
- **Improved recovery mechanisms** with better validation
- **Comprehensive error reporting** with statistics and analysis
- **Consistent error severity classification** for proper prioritization
- **Retry logic for transient failures** with exponential backoff
- **Thread-safe error handling** for multi-threaded environments
- **Error context in all messages** for better troubleshooting

These improvements significantly enhance system stability, debugging capabilities, and overall robustness while maintaining backward compatibility with existing code.