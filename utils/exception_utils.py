"""
Consolidated exception handling utilities to reduce code duplication.
"""

import logging
from typing import Callable, Any, Optional

from utils.print_utils import print_error, print_warning, print_info



def safe_execute(func: Callable, context: str = "", 
                error_msg: str = "Operation failed", 
                default_return: Any = None,
                log_level: str = "warning") -> Any:
    """
    Safely execute a function with consistent error handling.
    Replaces repeated try-except patterns with logging.
    """
    try:
        return func()
    except Exception as e:
        error_message = f"{error_msg} in {context}" if context else error_msg
        
        if log_level == "error":
            logging.error(f"{error_message}: {e}")
            print_error(error_message, e)
        elif log_level == "warning":
            logging.warning(f"{error_message}: {e}")
            print_warning(f"{error_message}: {e}")
        else:
            logging.info(f"{error_message}: {e}")
            print_info(f"{error_message}: {e}")
        
        return default_return


def safe_initialize_component(component_name: str, init_func: Callable, 
                            fallback_func: Optional[Callable] = None) -> Any:
    """
    Safely initialize a component with fallback handling.
    Replaces repeated component initialization patterns.
    """
    try:
        result = init_func()
        logging.info(f"{component_name} initialized successfully")
        return result
    except Exception as e:
        logging.warning(f"Failed to initialize {component_name}: {e}")
        
        if fallback_func:
            try:
                fallback_result = fallback_func()
                logging.info(f"{component_name} fallback initialized")
                return fallback_result
            except Exception as fallback_e:
                logging.error(f"Fallback for {component_name} also failed: {fallback_e}")
                return None
        return None


def safe_process_step(process_func: Callable, step_name: str, 
                     step_counter: int = 0) -> bool:
    """
    Safely process a simulation step with consistent error handling.
    Replaces repeated step processing patterns.
    """
    try:
        process_func()
        if step_counter % 100 == 0:
            logging.debug(f"{step_name} processed at step {step_counter}")
        return True
    except Exception as e:
        logging.warning(f"{step_name} failed at step {step_counter}: {e}")
        return False


def safe_callback_execution(callback: Callable, *args, **kwargs) -> Any:
    """
    Safely execute callbacks with consistent error handling.
    Replaces repeated callback error patterns.
    """
    try:
        return callback(*args, **kwargs)
    except Exception as e:
        logging.error(f"Callback execution failed: {e}")
        print_error("Callback execution failed", e)
        return None


def safe_file_operation(operation: Callable, file_path: str, 
                       context: str = "file operation") -> Any:
    """
    Safely perform file operations with consistent error handling.
    Replaces repeated file operation error patterns.
    """
    try:
        return operation()
    except Exception as e:
        logging.error(f"File operation failed for {file_path}: {e}")
        print_error(f"File operation failed for {file_path}", e)
        return None


def safe_graph_operation(operation: Callable, graph_context: str = "graph",
                        error_msg: str = "Graph operation failed") -> Any:
    """
    Safely perform graph operations with consistent error handling.
    Replaces repeated graph operation error patterns.
    """
    try:
        return operation()
    except Exception as e:
        logging.warning(f"{error_msg}: {e}")
        print_warning(f"{error_msg}: {e}")
        return None


def handle_critical_error(error: Exception, context: str = "", 
                        fallback_action: Optional[Callable] = None) -> None:
    """
    Handle critical errors with consistent logging and fallback.
    Replaces repeated critical error handling patterns.
    """
    error_message = f"Critical error in {context}" if context else "Critical error"
    logging.error(f"{error_message}: {error}")
    print_error(error_message, error)
    
    if fallback_action:
        try:
            fallback_action()
        except Exception as fallback_e:
            logging.error(f"Fallback action also failed: {fallback_e}")
            print_error("Fallback action also failed", fallback_e)


def log_and_continue(error: Exception, context: str = "", 
                    continue_msg: str = "Continuing with fallback") -> None:
    """
    Log error and continue execution with fallback.
    Replaces repeated log-and-continue patterns.
    """
    logging.warning(f"Error in {context}: {error}")
    print_warning(f"Error in {context}: {error}")
    logging.info(continue_msg)
    print_info(continue_msg)
