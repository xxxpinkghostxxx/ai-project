"""
Enhanced Error Handling System for the Neural Simulation.
Provides comprehensive error recovery, logging, and monitoring capabilities.
"""

import logging
import traceback
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import functools
import weakref

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
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this strategy can handle the given error."""
        return False
    
    def recover(self, error: Exception, context: ErrorContext) -> bool:
        """Attempt to recover from the error."""
        return False

class RetryStrategy(RecoveryStrategy):
    """Recovery strategy that retries the operation."""
    
    def __init__(self, max_retries: int = 3, delay: float = 0.1):
        self.max_retries = max_retries
        self.delay = delay
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        return isinstance(error, (ConnectionError, TimeoutError, OSError))
    
    def recover(self, error: Exception, context: ErrorContext) -> bool:
        time.sleep(self.delay)
        return True

class FallbackStrategy(RecoveryStrategy):
    """Recovery strategy that uses a fallback method."""
    
    def __init__(self, fallback_func: Callable):
        self.fallback_func = fallback_func
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        return True  # Can handle any error with fallback
    
    def recover(self, error: Exception, context: ErrorContext) -> bool:
        try:
            self.fallback_func()
            return True
        except Exception:
            return False

class CircuitBreakerStrategy(RecoveryStrategy):
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        current_time = time.time()
        
        if self.state == "open":
            if current_time - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        
        return True
    
    def recover(self, error: Exception, context: ErrorContext) -> bool:
        current_time = time.time()
        
        if self.state == "closed":
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.last_failure_time = current_time
            return True
        
        elif self.state == "half-open":
            if isinstance(error, Exception):
                self.state = "open"
                self.last_failure_time = current_time
                return False
            else:
                self.state = "closed"
                self.failure_count = 0
                return True
        
        return False

class ErrorHandler:
    """Enhanced error handling system."""
    
    def __init__(self):
        self.error_records: Dict[str, ErrorRecord] = {}
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.error_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        self._lock = threading.RLock()
        self.error_count = 0
        self.recovery_count = 0
        self.max_error_history = 1000
        self.error_history = deque(maxlen=self.max_error_history)
        
        # Setup default recovery strategies
        self._setup_default_strategies()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        self.add_recovery_strategy(RetryStrategy())
        self.add_recovery_strategy(CircuitBreakerStrategy())
    
    def _setup_logging(self):
        """Setup error logging."""
        self.logger = logging.getLogger('neural_simulation.errors')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.ERROR)
    
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy."""
        with self._lock:
            self.recovery_strategies.append(strategy)
    
    def add_error_callback(self, callback: Callable):
        """Add an error callback."""
        with self._lock:
            self.error_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add a recovery callback."""
        with self._lock:
            self.recovery_callbacks.append(callback)
    
    def handle_error(self, error: Exception, context: ErrorContext, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.SYSTEM) -> bool:
        """Handle an error with recovery attempts."""
        with self._lock:
            self.error_count += 1
            
            # Create error record
            error_id = self._generate_error_id(error, context)
            if error_id in self.error_records:
                record = self.error_records[error_id]
                record.count += 1
                record.last_occurrence = time.time()
            else:
                record = ErrorRecord(
                    error_id=error_id,
                    exception=error,
                    context=context,
                    severity=severity,
                    category=category
                )
                self.error_records[error_id] = record
            
            # Add to history
            self.error_history.append(record)
            
            # Log error
            self._log_error(record)
            
            # Notify callbacks
            self._notify_error_callbacks(record)
            
            # Attempt recovery
            recovery_successful = self._attempt_recovery(record)
            
            if recovery_successful:
                self.recovery_count += 1
                self._notify_recovery_callbacks(record)
            
            return recovery_successful
    
    def _generate_error_id(self, error: Exception, context: ErrorContext) -> str:
        """Generate a unique error ID."""
        error_type = type(error).__name__
        error_msg = str(error)[:50]  # First 50 chars
        return f"{error_type}_{context.component}_{context.function}_{hash(error_msg)}"
    
    def _log_error(self, record: ErrorRecord):
        """Log error information."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[record.severity]
        
        self.logger.log(
            log_level,
            f"Error in {record.context.component}.{record.context.function}: "
            f"{record.exception} (count: {record.count})"
        )
    
    def _attempt_recovery(self, record: ErrorRecord) -> bool:
        """Attempt to recover from the error."""
        for strategy in self.recovery_strategies:
            if strategy.can_handle(record.exception, record.context):
                try:
                    if strategy.recover(record.exception, record.context):
                        record.recovery_attempts += 1
                        record.recovery_successful = True
                        return True
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        return False
    
    def _notify_error_callbacks(self, record: ErrorRecord):
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(record)
            except Exception as e:
                self.logger.error(f"Error callback failed: {e}")
    
    def _notify_recovery_callbacks(self, record: ErrorRecord):
        """Notify recovery callbacks."""
        for callback in self.recovery_callbacks:
            try:
                callback(record)
            except Exception as e:
                self.logger.error(f"Recovery callback failed: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = len(self.error_records)
            errors_by_severity = defaultdict(int)
            errors_by_category = defaultdict(int)
            
            for record in self.error_records.values():
                errors_by_severity[record.severity.value] += record.count
                errors_by_category[record.category.value] += record.count
            
            return {
                'total_errors': total_errors,
                'error_count': self.error_count,
                'recovery_count': self.recovery_count,
                'recovery_rate': self.recovery_count / max(self.error_count, 1),
                'errors_by_severity': dict(errors_by_severity),
                'errors_by_category': dict(errors_by_category),
                'recent_errors': len(self.error_history)
            }
    
    def get_error_history(self, limit: int = 100) -> List[ErrorRecord]:
        """Get recent error history."""
        with self._lock:
            return list(self.error_history)[-limit:]
    
    def clear_error_history(self):
        """Clear error history."""
        with self._lock:
            self.error_records.clear()
            self.error_history.clear()
            self.error_count = 0
            self.recovery_count = 0

# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

# Decorator for automatic error handling
def handle_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 component: str = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    component=component or func.__module__,
                    function=func.__name__,
                    line_number=e.__traceback__.tb_lineno if e.__traceback__ else 0,
                    timestamp=time.time(),
                    thread_id=threading.get_ident()
                )
                
                handler = get_error_handler()
                handler.handle_error(e, context, severity, category)
                raise
        
        return wrapper
    return decorator

# Context manager for error handling
class ErrorContext:
    """Context manager for error handling."""
    
    def __init__(self, component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM):
        self.component = component
        self.severity = severity
        self.category = category
        self.handler = get_error_handler()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            context = ErrorContext(
                component=self.component,
                function="<context_manager>",
                line_number=exc_tb.tb_lineno if exc_tb else 0,
                timestamp=time.time(),
                thread_id=threading.get_ident()
            )
            self.handler.handle_error(exc_val, context, self.severity, self.category)
        return False  # Don't suppress the exception

# Utility functions
def log_error(error: Exception, context: str = ""):
    """Log an error with context."""
    handler = get_error_handler()
    error_context = ErrorContext(
        component=context,
        function="<manual_log>",
        line_number=0,
        timestamp=time.time(),
        thread_id=threading.get_ident()
    )
    handler.handle_error(error, error_context)

def get_error_statistics() -> Dict[str, Any]:
    """Get current error statistics."""
    handler = get_error_handler()
    return handler.get_error_statistics()

def clear_error_history():
    """Clear error history."""
    handler = get_error_handler()
    handler.clear_error_history()
