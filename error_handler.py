"""
error_handler.py

Comprehensive error handling and graceful degradation system for the AI neural project.
Provides centralized error management, recovery mechanisms, and system health monitoring.
"""

import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from logging_utils import log_step


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    Provides graceful degradation and automatic recovery mechanisms.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_counts = {}
        self.recovery_attempts = {}
        self.system_health = 'healthy'
        self.last_error_time = 0
        self.error_callbacks = []
        self.recovery_callbacks = []
        
        # Error thresholds
        self.max_errors_per_minute = 10
        self.max_recovery_attempts = 3
        self.error_cooldown = 60  # seconds
        
        # System health levels
        self.health_levels = {
            'healthy': 0,
            'warning': 1,
            'degraded': 2,
            'critical': 3,
            'failed': 4
        }
        
        log_step("ErrorHandler initialized")
    
    def handle_error(self, error: Exception, context: str = "", 
                    recovery_func: Optional[Callable] = None, 
                    critical: bool = False) -> bool:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            recovery_func: Optional function to attempt recovery
            critical: Whether this is a critical error that should stop the system
        
        Returns:
            bool: True if error was handled successfully, False if critical
        """
        error_type = type(error).__name__
        error_key = f"{error_type}:{context}"
        current_time = time.time()
        
        # Update error counts
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        self.error_counts[error_key] += 1
        
        # Log the error
        log_step(f"Error occurred: {error_type}",
                context=context,
                error_count=self.error_counts[error_key],
                critical=critical)
        
        logging.error(f"[ERROR_HANDLER] {error_type} in {context}: {str(error)}")
        logging.debug(f"[ERROR_HANDLER] Traceback: {traceback.format_exc()}")
        
        # Update system health
        self._update_system_health(error_type, critical)
        
        # Attempt recovery if not critical
        if not critical and recovery_func:
            success = self._attempt_recovery(error_key, recovery_func, error)
            if success:
                log_step("Error recovery successful", error_type=error_type, context=context)
                return True
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error, context, critical)
            except Exception as callback_error:
                logging.error(f"Error callback failed: {callback_error}")
        
        # If critical, update system state
        if critical:
            self.system_health = 'failed'
            log_step("Critical error - system marked as failed", error_type=error_type)
            return False
        
        return True
    
    def _update_system_health(self, error_type: str, critical: bool):
        """Update system health based on error type and frequency."""
        current_time = time.time()
        
        # Check error frequency
        recent_errors = sum(1 for count in self.error_counts.values() 
                          if count > 0 and current_time - self.last_error_time < 60)
        
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
    
    def _attempt_recovery(self, error_key: str, recovery_func: Callable, 
                         original_error: Exception) -> bool:
        """Attempt to recover from an error."""
        if error_key not in self.recovery_attempts:
            self.recovery_attempts[error_key] = 0
        
        if self.recovery_attempts[error_key] >= self.max_recovery_attempts:
            log_step("Max recovery attempts reached", error_key=error_key)
            return False
        
        self.recovery_attempts[error_key] += 1
        
        try:
            log_step("Attempting error recovery", 
                    error_key=error_key, 
                    attempt=self.recovery_attempts[error_key])
            
            result = recovery_func(original_error)
            
            if result:
                # Reset error count on successful recovery
                self.error_counts[error_key] = 0
                self.recovery_attempts[error_key] = 0
                
                # Call recovery callbacks
                for callback in self.recovery_callbacks:
                    try:
                        callback(error_key, True)
                    except Exception as callback_error:
                        logging.error(f"Recovery callback failed: {callback_error}")
                
                return True
            else:
                log_step("Recovery attempt failed", error_key=error_key)
                return False
                
        except Exception as recovery_error:
            log_step("Recovery attempt caused new error", 
                    error_key=error_key, 
                    recovery_error=str(recovery_error))
            return False
    
    def add_error_callback(self, callback: Callable):
        """Add a callback to be called when errors occur."""
        self.error_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add a callback to be called when recovery is attempted."""
        self.recovery_callbacks.append(callback)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            'status': self.system_health,
            'level': self.health_levels.get(self.system_health, 0),
            'error_counts': self.error_counts.copy(),
            'recovery_attempts': self.recovery_attempts.copy(),
            'last_error_time': self.last_error_time,
            'is_healthy': self.system_health == 'healthy'
        }
    
    def reset_error_counts(self):
        """Reset all error counts and recovery attempts."""
        self.error_counts.clear()
        self.recovery_attempts.clear()
        self.system_health = 'healthy'
        self.last_error_time = 0
        log_step("Error counts reset")


# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def safe_execute(func: Callable, context: str = "", 
                recovery_func: Optional[Callable] = None,
                critical: bool = False, default_return: Any = None):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        context: Context for error reporting
        recovery_func: Optional recovery function
        critical: Whether errors are critical
        default_return: Value to return if function fails
    
    Returns:
        Function result or default_return if error occurred
    """
    error_handler = get_error_handler()
    
    try:
        return func()
    except Exception as e:
        success = error_handler.handle_error(e, context, recovery_func, critical)
        if not success and critical:
            raise
        return default_return


def error_handler_decorator(context: str = "", recovery_func: Optional[Callable] = None,
                          critical: bool = False, default_return: Any = None):
    """
    Decorator for automatic error handling.
    
    Args:
        context: Context for error reporting
        recovery_func: Optional recovery function
        critical: Whether errors are critical
        default_return: Value to return if function fails
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return safe_execute(
                lambda: func(*args, **kwargs),
                context or func.__name__,
                recovery_func,
                critical,
                default_return
            )
        return wrapper
    return decorator


def graceful_degradation(func: Callable, fallback_func: Callable, 
                        context: str = "") -> Any:
    """
    Execute a function with graceful degradation to a fallback.
    
    Args:
        func: Primary function to execute
        fallback_func: Fallback function if primary fails
        context: Context for error reporting
    
    Returns:
        Result from primary function or fallback function
    """
    error_handler = get_error_handler()
    
    try:
        return func()
    except Exception as e:
        error_handler.handle_error(e, f"{context} (primary)", critical=False)
        
        try:
            log_step("Attempting graceful degradation", context=context)
            return fallback_func()
        except Exception as fallback_error:
            error_handler.handle_error(fallback_error, f"{context} (fallback)", critical=True)
            raise


def system_health_check() -> Dict[str, Any]:
    """Perform a comprehensive system health check."""
    error_handler = get_error_handler()
    health = error_handler.get_system_health()
    
    # Add additional health checks
    health['timestamp'] = time.time()
    health['uptime'] = time.time() - error_handler.last_error_time if error_handler.last_error_time > 0 else 0
    
    return health


# Example usage and testing
if __name__ == "__main__":
    print("ErrorHandler initialized successfully!")
    print("Features include:")
    print("- Centralized error handling")
    print("- Automatic recovery mechanisms")
    print("- System health monitoring")
    print("- Graceful degradation support")
    print("- Error callbacks and notifications")
    
    # Test basic functionality
    handler = ErrorHandler()
    print(f"Handler created with health status: {handler.system_health}")
    print("ErrorHandler is ready for integration!")
