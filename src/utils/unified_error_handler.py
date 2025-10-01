"""
Unified Error Handling System for the Neural Simulation.
Consolidates error_handler.py, enhanced_error_handling.py, and exception_utils.py
into a comprehensive, efficient error handling system.
"""

import functools
import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging_utils import log_step
from src.utils.print_utils import print_error, print_info, print_warning


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    SIMULATION = "simulation"
    MEMORY = "memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    UI = "ui"
    LEARNING = "learning"
    NEURAL = "neural"


@dataclass
class ErrorContext:
    """Context information for an error."""
    component: str
    function: str
    line_number: int
    timestamp: float
    thread_id: int
    user_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    exception: Exception
    context: ErrorContext
    severity: ErrorSeverity
    category: ErrorCategory
    count: int = 1
    first_occurrence: float = field(default_factory=time.time)
    last_occurrence: float = field(default_factory=time.time)
    recovery_attempts: int = 0
    recovery_successful: bool = False


class RecoveryStrategy:
    """Base class for error recovery strategies."""

    def can_handle(self, _error: Exception, _context: ErrorContext) -> bool:
        """Check if this strategy can handle the given error."""
        return False

    def recover(self, _error: Exception, _context: ErrorContext) -> bool:
        """Attempt to recover from the error."""
        return False


class RetryStrategy(RecoveryStrategy):
    """Recovery strategy that retries the operation."""

    def __init__(self, max_retries: int = 3, delay: float = 0.1):
        self.max_retries = max_retries
        self.delay = delay

    def can_handle(self, error: Exception, _context: ErrorContext) -> bool:
        return isinstance(error, (ConnectionError, TimeoutError, OSError))

    def recover(self, _error: Exception, _context: ErrorContext) -> bool:
        time.sleep(self.delay)
        return True


class FallbackStrategy(RecoveryStrategy):
    """Recovery strategy that uses a fallback method."""

    def __init__(self, fallback_func: Callable):
        self.fallback_func = fallback_func

    def can_handle(self, _error: Exception, _context: ErrorContext) -> bool:
        return True  # Can handle any error with fallback

    def recover(self, _error: Exception, _context: ErrorContext) -> bool:
        try:
            self.fallback_func()
            return True
        except Exception:  # pylint: disable=broad-except
            # Broad exception catch necessary as fallback_func is user-provided and may raise any exception
            return False


class CircuitBreakerStrategy(RecoveryStrategy):
    """Circuit breaker pattern for preventing cascading failures."""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def can_handle(self, _error: Exception, _context: ErrorContext) -> bool:
        current_time = time.time()

        if self.state == "open":
            if current_time - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False

        return True

    def recover(self, error: Exception, _context: ErrorContext) -> bool:
        current_time = time.time()

        if self.state == "closed":
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.last_failure_time = current_time
            return True

        if self.state == "half-open":
            if isinstance(error, (ConnectionError, TimeoutError, OSError)):
                self.state = "open"
                self.last_failure_time = current_time
                return False
            self.state = "closed"
            self.failure_count = 0
            return True

        return False


class UnifiedErrorHandler:
    """Unified error handling system combining all error handling capabilities."""

    def __init__(self):
        # Basic error tracking with memory limits
        self.error_counts = {}
        self.recovery_attempts = {}
        self.system_health = 'healthy'
        self.last_error_time = 0

        # Enhanced error tracking with memory limits
        self.error_records: Dict[str, ErrorRecord] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.max_error_records = 5000  # Maximum number of error records to keep
        self.error_record_cleanup_interval = 3600  # Clean up old records every hour
        self.last_cleanup_time = time.time()

        # Callbacks with memory limits
        self.error_callbacks = []
        self.recovery_callbacks = []
        self.max_callbacks = 50  # Maximum number of callbacks per type

        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = []
        self._setup_default_strategies()

        # Configuration with validation
        self.max_errors_per_minute = 10
        self.max_recovery_attempts = 3
        self.error_cooldown = 60
        self.health_levels = {
            'healthy': 0,
            'warning': 1,
            'degraded': 2,
            'critical': 3,
            'failed': 4
        }

        # Thread safety
        self._lock = threading.RLock()

        # Setup logging
        self._setup_logging()

        log_step("UnifiedErrorHandler initialized with memory limits")

    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        self.recovery_strategies = [
            RetryStrategy(max_retries=3, delay=0.1),
            CircuitBreakerStrategy(failure_threshold=5, timeout=60.0)
        ]

    def _setup_logging(self):
        """Setup error logging."""
        self.logger = logging.getLogger('unified_error_handler')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy with validation."""
        if not isinstance(strategy, RecoveryStrategy):
            raise TypeError("Strategy must be an instance of RecoveryStrategy")

        with self._lock:
            if len(self.recovery_strategies) >= 20:  # Reasonable limit
                logging.warning("Maximum recovery strategies reached, cannot add more")
                return
            self.recovery_strategies.append(strategy)

    def add_error_callback(self, callback: Callable):
        """Add an error callback with validation and memory limits."""
        if not callable(callback):
            raise TypeError("Callback must be callable")

        with self._lock:
            if len(self.error_callbacks) >= self.max_callbacks:
                logging.warning("Maximum error callbacks (%s) reached, cannot add more", self.max_callbacks)
                return

            if callback not in self.error_callbacks:
                self.error_callbacks.append(callback)

    def add_recovery_callback(self, callback: Callable):
        """Add a recovery callback with validation and memory limits."""
        if not callable(callback):
            raise TypeError("Callback must be callable")

        with self._lock:
            if len(self.recovery_callbacks) >= self.max_callbacks:
                logging.warning("Maximum recovery callbacks (%s) reached, cannot add more", self.max_callbacks)
                return

            if callback not in self.recovery_callbacks:
                self.recovery_callbacks.append(callback)

    def handle_error(self, error: Exception, context: str = "",  # pylint: disable=too-many-arguments,too-many-positional-arguments
                    recovery_func: Optional[Callable] = None,
                    critical: bool = False,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.SYSTEM) -> bool:
        """Handle an error with comprehensive recovery and logging."""
        # Input validation
        if error is None:
            logging.error("Cannot handle None error")
            return False

        if not isinstance(context, str):
            context = str(context)

        if not isinstance(severity, ErrorSeverity):
            severity = ErrorSeverity.MEDIUM

        if not isinstance(category, ErrorCategory):
            category = ErrorCategory.SYSTEM

        if recovery_func is not None and not callable(recovery_func):
            logging.warning("Recovery function is not callable, ignoring")
            recovery_func = None

        with self._lock:
            # Periodic cleanup
            self._periodic_cleanup()

            # Create error context
            error_context = self._create_error_context(context, error)

            # Generate error ID
            error_id = self._generate_error_id(error, error_context)

            # Update error record
            self._update_error_record(error_id, error, error_context, severity, category)

            # Log error
            self._log_error(error, error_context, severity, category)

            # Update system health
            self._update_system_health(error, error_context, critical)

            # Attempt recovery
            recovery_success = self._attempt_recovery(error, error_context, recovery_func)

            # Notify callbacks
            self._notify_error_callbacks(error, error_context, critical)

            if recovery_success:
                self._notify_recovery_callbacks(error_id, True)
                return True

            if critical:
                self.system_health = 'failed'
                log_step("Critical error - system marked as failed",
                        error_type=type(error).__name__)
                return False

            return True

    def _create_error_context(self, context: str, error: Exception) -> ErrorContext:
        """Create error context from string context and error."""
        frame = traceback.extract_tb(error.__traceback__)[-1] if error.__traceback__ else None

        return ErrorContext(
            component=context.split('.')[0] if '.' in context else context,
            function=frame.name if frame else "unknown",
            line_number=frame.lineno if frame else 0,
            timestamp=time.time(),
            thread_id=threading.get_ident(),
            user_data={'context_string': context}
        )

    def _generate_error_id(self, error: Exception, context: ErrorContext) -> str:
        """Generate unique error ID."""
        error_type = type(error).__name__
        return f"{error_type}_{context.component}_{int(context.timestamp)}"

    def _update_error_record(self, error_id: str, error: Exception,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                           context: ErrorContext, severity: ErrorSeverity,
                           category: ErrorCategory):
        """Update or create error record with memory limits."""
        if error_id in self.error_records:
            record = self.error_records[error_id]
            record.count += 1
            record.last_occurrence = time.time()
        else:
            # Check if we need to clean up old records
            if len(self.error_records) >= self.max_error_records:
                self._cleanup_old_error_records()

            record = ErrorRecord(
                error_id=error_id,
                exception=error,
                context=context,
                severity=severity,
                category=category
            )
            self.error_records[error_id] = record

        # Add to history (deque automatically handles maxlen)
        self.error_history.append(record)

    def _periodic_cleanup(self):
        """Perform periodic cleanup of old error records."""
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.error_record_cleanup_interval:
            self._cleanup_old_error_records()
            self.last_cleanup_time = current_time

    def _cleanup_old_error_records(self):
        """Clean up old error records to prevent memory growth."""
        current_time = time.time()
        max_age = 86400  # 24 hours

        # Remove old error records
        old_records = []
        for error_id, record in self.error_records.items():
            if current_time - record.last_occurrence > max_age:
                old_records.append(error_id)

        for error_id in old_records:
            del self.error_records[error_id]

        if old_records:
            log_step(f"Cleaned up {len(old_records)} old error records")

        # Clean up error_counts and recovery_attempts
        # Keep only recent entries (last 1000 operations)
        if len(self.error_counts) > 1000:
            # Simple cleanup - remove oldest entries
            items_to_remove = len(self.error_counts) - 1000
            keys_to_remove = list(self.error_counts.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.error_counts[key]

        if len(self.recovery_attempts) > 1000:
            items_to_remove = len(self.recovery_attempts) - 1000
            keys_to_remove = list(self.recovery_attempts.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.recovery_attempts[key]

    def _log_error(self, error: Exception, context: ErrorContext,
                  severity: ErrorSeverity, category: ErrorCategory):
        """Log error with appropriate level."""
        error_type = type(error).__name__
        error_msg = "%s in %s.%s: %s"

        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_msg)
            print_error(f"CRITICAL: {error_msg}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(error_msg)
            print_error(f"HIGH: {error_msg}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_msg)
            print_warning(f"MEDIUM: {error_msg}")
        else:
            self.logger.info(error_msg)
            print_info(f"LOW: {error_msg}")

        # Log traceback for debugging
        self.logger.debug("Traceback: %s", traceback.format_exc())

        # Log to step logger
        log_step(f"Error occurred: {error_type}",
                component=context.component,
                function=context.function,
                severity=severity.value,
                category=category.value)

    def _update_system_health(self, _error: Exception, _context: ErrorContext, critical: bool):
        """Update system health based on error."""
        current_time = time.time()

        # Count recent errors
        recent_errors = sum(1 for record in self.error_history
                          if current_time - record.context.timestamp < 60)

        if critical:
            self.system_health = 'failed'
        elif recent_errors > self.max_errors_per_minute:
            self.system_health = 'critical'
        elif recent_errors > self.max_errors_per_minute // 2:
            self.system_health = 'degraded'
        elif recent_errors > 0:
            self.system_health = 'warning'
        else:
            self.system_health = 'healthy'

        self.last_error_time = current_time

    def _attempt_recovery(self, error: Exception, context: ErrorContext,
                         recovery_func: Optional[Callable]) -> bool:
        """Attempt to recover from error using strategies and recovery function."""
        # Try recovery strategies first
        for strategy in self.recovery_strategies:
            if strategy.can_handle(error, context):
                if strategy.recover(error, context):
                    return True

        # Try custom recovery function
        if recovery_func:
            try:
                return recovery_func(error)
            except Exception as recovery_error:  # pylint: disable=broad-except
                # Broad exception catch necessary as recovery_func is user-provided and may raise any exception
                self.logger.warning("Recovery function failed: %s", recovery_error)

        return False

    def _notify_error_callbacks(self, error: Exception, context: ErrorContext, critical: bool):
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error, context, critical)
            except Exception as callback_error:  # pylint: disable=broad-except
                # Broad exception catch necessary as callbacks are user-provided and may raise any exception
                self.logger.error("Error callback failed: %s", callback_error)

    def _notify_recovery_callbacks(self, error_id: str, success: bool):
        """Notify recovery callbacks."""
        for callback in self.recovery_callbacks:
            try:
                callback(error_id, success)
            except Exception as callback_error:  # pylint: disable=broad-except
                # Broad exception catch necessary as callbacks are user-provided and may raise any exception
                self.logger.error("Recovery callback failed: %s", callback_error)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            total_errors = len(self.error_history)
            error_types = defaultdict(int)
            error_categories = defaultdict(int)
            error_severities = defaultdict(int)

            for record in self.error_history:
                error_types[type(record.exception).__name__] += 1
                error_categories[record.category.value] += 1
                error_severities[record.severity.value] += 1

            return {
                'total_errors': total_errors,
                'system_health': self.system_health,
                'error_types': dict(error_types),
                'error_categories': dict(error_categories),
                'error_severities': dict(error_severities),
                'recovery_attempts': sum(self.recovery_attempts.values()),
                'successful_recoveries': sum(1 for r in self.error_records.values()
                                          if r.recovery_successful)
            }

    def get_error_history(self, limit: int = 100) -> List[ErrorRecord]:
        """Get recent error history."""
        with self._lock:
            return list(self.error_history)[-limit:]

    def clear_error_history(self):
        """Clear error history with comprehensive cleanup."""
        with self._lock:
            self.error_history.clear()
            self.error_records.clear()
            self.error_counts.clear()
            self.recovery_attempts.clear()
            self.system_health = 'healthy'
            self.last_error_time = 0
            log_step("Error history cleared")

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        with self._lock:
            return {
                'health_status': self.system_health,
                'health_level': self.health_levels.get(self.system_health, 0),
                'last_error_time': self.last_error_time,
                'recent_error_count': sum(1 for record in self.error_history
                                       if time.time() - record.context.timestamp < 60)
            }


class ErrorHandlerSingleton:  # pylint: disable=too-few-public-methods
    """Singleton wrapper for UnifiedErrorHandler."""

    _instance: UnifiedErrorHandler = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> UnifiedErrorHandler:
        """Get the singleton instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check locking
                    cls._instance = UnifiedErrorHandler()
        return cls._instance


def get_error_handler() -> UnifiedErrorHandler:
    """Get the error handler instance."""
    return ErrorHandlerSingleton.get_instance()


# Convenience functions for backward compatibility
def safe_execute(func: Callable, context: str = "",
                 default_return: Any = None,
                 log_level: str = "warning") -> Any:
    """Safely execute a function with error handling and validation."""
    if not callable(func):
        logging.error("safe_execute: func must be callable")
        return default_return

    if not isinstance(context, str):
        context = str(context)

    try:
        return func()
    except Exception as e:  # pylint: disable=broad-except
        # Broad exception catch necessary in safe wrapper to handle any errors from user function
        error_handler = get_error_handler()
        severity = ErrorSeverity.LOW if log_level == "info" else ErrorSeverity.MEDIUM
        error_handler.handle_error(e, context, severity=severity)
        return default_return


def safe_initialize_component(component_name: str, init_func: Callable,
                            fallback_func: Optional[Callable] = None) -> Any:
    """Safely initialize a component with fallback handling and validation."""
    if not isinstance(component_name, str) or not component_name:
        logging.error("safe_initialize_component: component_name must be a non-empty string")
        return None

    if not callable(init_func):
        logging.error("safe_initialize_component: init_func must be callable")
        return None

    if fallback_func is not None and not callable(fallback_func):
        logging.error("safe_initialize_component: fallback_func must be callable or None")
        fallback_func = None

    try:
        result = init_func()
        logging.info("%s initialized successfully", component_name)
        return result
    except Exception as e:  # pylint: disable=broad-except
        # Broad exception catch necessary in safe wrapper to handle any errors from user init function
        error_handler = get_error_handler()
        if fallback_func:
            try:
                result = fallback_func()
                error_handler.handle_error(e, f"initializing {component_name}",
                                          recovery_func=lambda _: True,
                                          severity=ErrorSeverity.LOW)
                return result
            except Exception as fallback_error:  # pylint: disable=broad-except
                # Broad exception catch necessary as fallback_func is user-provided and may raise any exception
                error_handler.handle_error(fallback_error, f"fallback for {component_name}",
                                          severity=ErrorSeverity.HIGH)
                return None
        else:
            error_handler.handle_error(e, f"initializing {component_name}",
                                      severity=ErrorSeverity.MEDIUM)
            return None


def safe_process_step(process_func: Callable, step_name: str,
                     step_counter: int = 0) -> bool:
    """Safely process a simulation step with validation."""
    if not callable(process_func):
        logging.error("safe_process_step: process_func must be callable")
        return False

    if not isinstance(step_name, str):
        step_name = str(step_name)

    if not isinstance(step_counter, int) or step_counter < 0:
        step_counter = 0

    try:
        result = process_func()
        return result if isinstance(result, bool) else True
    except Exception as e:  # pylint: disable=broad-except
        # Broad exception catch necessary in safe wrapper to handle any errors from user process function
        error_handler = get_error_handler()
        error_handler.handle_error(e, "processing %s (step %s)",
                                  severity=ErrorSeverity.MEDIUM,
                                  category=ErrorCategory.SIMULATION)
        return False


def safe_callback_execution(callback: Callable, *args, **kwargs) -> Any:
    """Safely execute a callback function with validation."""
    if not callable(callback):
        logging.error("safe_callback_execution: callback must be callable")
        return None

    try:
        return callback(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        # Broad exception catch necessary in safe wrapper to handle any errors from user callback
        error_handler = get_error_handler()
        error_handler.handle_error(e, "callback execution",
                                  severity=ErrorSeverity.LOW)
        return None


def handle_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 component: str = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                # Broad exception catch necessary in decorator to handle any errors from decorated function
                error_handler = get_error_handler()
                context = component or func.__name__
                error_handler.handle_error(e, context, severity=severity, category=category)
                return None
        return wrapper
    return decorator


def log_error(error: Exception, context: str = ""):
    """Log an error with the unified error handler."""
    error_handler = get_error_handler()
    error_handler.handle_error(error, context, severity=ErrorSeverity.MEDIUM)


def get_error_statistics() -> Dict[str, Any]:
    """Get error statistics."""
    return get_error_handler().get_error_statistics()


def clear_error_history():
    """Clear error history."""
    get_error_handler().clear_error_history()


# Backward compatibility aliases
ErrorHandler = UnifiedErrorHandler







