"""
Consolidated error handling utilities to reduce code duplication.
"""

import logging
from typing import Any, Callable, Optional, Dict

from config.consolidated_constants import ERROR_MESSAGES
from utils.print_utils import print_error, print_warning, print_info


def safe_execute(func: Callable, context: str = "operation", default_return: Any = None, 
                 critical: bool = False) -> Any:
    """
    Executes a function safely, catching exceptions and logging them.
    If critical is True, logs as error; otherwise, logs as warning.
    """
    try:
        return func()
    except Exception as e:
        message = f"{ERROR_MESSAGES['EXCEPTION_OCCURRED']} in {context}: {e}"
        if critical:
            logging.error(message)
            print_error(message)
        else:
            logging.warning(message)
            print_warning(message)
        return default_return


def safe_initialize_component(component_name: str, init_func: Callable, default_value: Any = None,
                             critical: bool = False) -> Any:
    """
    Safely initializes a component, logging success or failure.
    """
    try:
        component = init_func()
        if component is not None:
            logging.info(f"{component_name} initialized")
            print_info(f"{component_name} initialized")
        return component
    except Exception as e:
        message = f"Failed to initialize component {component_name}: {e}"
        if critical:
            logging.error(message)
            print_error(message)
        else:
            logging.warning(message)
            print_warning(message)
        return default_value


def safe_process_step(process_func: Callable, context: str = "processing step", 
                      critical: bool = False) -> bool:
    """
    Safely executes a processing step, logging success or failure.
    """
    try:
        process_func()
        return True
    except Exception as e:
        message = f"{ERROR_MESSAGES['EXCEPTION_OCCURRED']} during {context}: {e}"
        if critical:
            logging.error(message)
            print_error(message)
        else:
            logging.warning(message)
            print_warning(message)
        return False


def safe_callback_execution(callback: Callable, *args, **kwargs) -> Any:
    """
    Safely executes a callback function, logging any errors.
    """
    try:
        return callback(*args, **kwargs)
    except Exception as e:
        message = f"{ERROR_MESSAGES['CALLBACK_ERROR']}: {e}"
        logging.error(message)
        print_error(message)
        return None


def safe_graph_access(graph, attribute: str, default_value: Any = None) -> Any:
    """
    Safely accesses a graph attribute, returning default if not found.
    """
    try:
        return getattr(graph, attribute, default_value)
    except Exception as e:
        message = f"Failed to access graph attribute {attribute}: {e}"
        logging.warning(message)
        print_warning(message)
        return default_value


def safe_hasattr(obj, *attributes) -> bool:
    """
    Safely checks if an object has multiple attributes.
    """
    try:
        return all(hasattr(obj, attr) for attr in attributes)
    except Exception as e:
        message = f"Failed to check attributes {attributes}: {e}"
        logging.warning(message)
        print_warning(message)
        return False


def create_safe_callback(callback: Callable, context: str = "callback") -> Callable:
    """
    Creates a safe wrapper for a callback function.
    """
    def safe_wrapper(*args, **kwargs):
        return safe_callback_execution(callback, *args, **kwargs)
    return safe_wrapper