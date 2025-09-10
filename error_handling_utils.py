"""
error_handling_utils.py

Standardized error handling utilities for consistent error reporting across the codebase.
Replaces inconsistent print statements and custom error handlers with unified approach.
"""

import logging
import traceback
from typing import Any, Optional, Callable, Dict
from functools import wraps
from logging_utils import log_step


class ErrorContext:
    """Context manager for error handling with automatic logging."""
    
    def __init__(self, operation: str, context: str = "", 
                 log_level: int = logging.ERROR, 
                 reraise: bool = False,
                 fallback_value: Any = None):
        """
        Initialize error context.
        
        Args:
            operation: Description of the operation being performed
            context: Additional context information
            log_level: Logging level for errors
            reraise: Whether to re-raise exceptions after logging
            fallback_value: Value to return if operation fails
        """
        self.operation = operation
        self.context = context
        self.log_level = log_level
        self.reraise = reraise
        self.fallback_value = fallback_value
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = f"Error in {self.operation}"
            if self.context:
                error_msg += f" ({self.context})"
            error_msg += f": {exc_val}"
            
            # Log the error
            self.logger.log(self.log_level, error_msg)
            
            # Log traceback for debugging
            if self.log_level <= logging.DEBUG:
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Log step for UI integration
            log_step(f"{self.operation} failed", 
                    context=self.context,
                    error=str(exc_val))
            
            # Return True to suppress exception if not reraising
            return not self.reraise
        
        return False


def handle_error(operation: str, context: str = "", 
                log_level: int = logging.ERROR,
                fallback_value: Any = None) -> Callable:
    """
    Decorator for standardized error handling.
    
    Args:
        operation: Description of the operation
        context: Additional context information
        log_level: Logging level for errors
        fallback_value: Value to return if operation fails
    
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {operation}"
                if context:
                    error_msg += f" ({context})"
                error_msg += f": {e}"
                
                # Log the error
                logger = logging.getLogger(__name__)
                logger.log(log_level, error_msg)
                
                # Log step for UI integration
                log_step(f"{operation} failed", 
                        context=context,
                        error=str(e))
                
                return fallback_value
        return wrapper
    return decorator


def safe_execute(operation: str, func: Callable, 
                context: str = "", 
                fallback_value: Any = None,
                reraise: bool = False) -> Any:
    """
    Safely execute a function with standardized error handling.
    
    Args:
        operation: Description of the operation
        func: Function to execute
        context: Additional context information
        fallback_value: Value to return if function fails
        reraise: Whether to re-raise exceptions after logging
    
    Returns:
        Function result or fallback_value if error occurred
    """
    try:
        return func()
    except Exception as e:
        error_msg = f"Error in {operation}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {e}"
        
        # Log the error
        logger = logging.getLogger(__name__)
        logger.error(error_msg)
        
        # Log step for UI integration
        log_step(f"{operation} failed", 
                context=context,
                error=str(e))
        
        if reraise:
            raise
        
        return fallback_value


def log_and_continue(operation: str, context: str = ""):
    """
    Context manager that logs errors but continues execution.
    
    Args:
        operation: Description of the operation
        context: Additional context information
    """
    return ErrorContext(operation, context, logging.WARNING, reraise=False)


def log_and_reraise(operation: str, context: str = ""):
    """
    Context manager that logs errors and re-raises them.
    
    Args:
        operation: Description of the operation
        context: Additional context information
    """
    return ErrorContext(operation, context, logging.ERROR, reraise=True)


def log_ui_error(operation: str, context: str = "", 
                fallback_value: Any = None) -> Callable:
    """
    Decorator for UI operations with error handling.
    
    Args:
        operation: Description of the operation
        context: Additional context information
        fallback_value: Value to return if operation fails
    
    Returns:
        Decorated function with UI error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error for debugging
                logger = logging.getLogger(__name__)
                logger.error(f"UI Error in {operation}: {e}")
                
                # Log step for UI integration
                log_step(f"{operation} failed", 
                        context=context,
                        error=str(e))
                
                # Don't crash the UI, return fallback
                return fallback_value
        return wrapper
    return decorator


def log_system_error(operation: str, context: str = ""):
    """
    Decorator for system operations with error handling.
    
    Args:
        operation: Description of the operation
        context: Additional context information
    
    Returns:
        Decorated function with system error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error for debugging
                logger = logging.getLogger(__name__)
                logger.error(f"System Error in {operation}: {e}")
                
                # Log step for UI integration
                log_step(f"{operation} failed", 
                        context=context,
                        error=str(e))
                
                # Re-raise system errors as they're critical
                raise
        return wrapper
    return decorator


def standardize_error_handling(module_name: str) -> Dict[str, Any]:
    """
    Get standardized error handling configuration for a module.
    
    Args:
        module_name: Name of the module
    
    Returns:
        Dictionary with error handling configuration
    """
    return {
        'ui_operations': {
            'log_level': logging.WARNING,
            'reraise': False,
            'fallback_value': None
        },
        'system_operations': {
            'log_level': logging.ERROR,
            'reraise': True,
            'fallback_value': None
        },
        'data_operations': {
            'log_level': logging.ERROR,
            'reraise': False,
            'fallback_value': {}
        },
        'network_operations': {
            'log_level': logging.WARNING,
            'reraise': False,
            'fallback_value': None
        }
    }


# Convenience functions for common error patterns
def handle_ui_error(operation: str, context: str = "", fallback_value: Any = None):
    """Handle UI errors with appropriate logging and fallback."""
    return handle_error(operation, context, logging.WARNING, fallback_value)


def handle_system_error(operation: str, context: str = ""):
    """Handle system errors with appropriate logging and re-raising."""
    return handle_error(operation, context, logging.ERROR, None)


def handle_data_error(operation: str, context: str = "", fallback_value: Any = {}):
    """Handle data errors with appropriate logging and fallback."""
    return handle_error(operation, context, logging.ERROR, fallback_value)


def handle_network_error(operation: str, context: str = "", fallback_value: Any = None):
    """Handle network errors with appropriate logging and fallback."""
    return handle_error(operation, context, logging.WARNING, fallback_value)
