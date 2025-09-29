"""
Consolidated print utilities to reduce code duplication.
"""

from config.consolidated_constants import ERROR_MESSAGES, PRINT_PATTERNS


def print_error(message: str, **kwargs):
    """Prints an error message with a consistent prefix."""
    print(f"{PRINT_PATTERNS['ERROR_PREFIX']} {message}", **kwargs)


def print_warning(message: str, **kwargs):
    """Prints a warning message with a consistent prefix."""
    print(f"{PRINT_PATTERNS['WARNING_PREFIX']} {message}", **kwargs)


def print_info(message: str, **kwargs):
    """Prints an informational message with a consistent prefix."""
    print(f"{PRINT_PATTERNS['INFO_PREFIX']} {message}", **kwargs)


def print_debug(message: str, **kwargs):
    """Prints a debug message with a consistent prefix."""
    print(f"{PRINT_PATTERNS['DEBUG_PREFIX']} {message}", **kwargs)


def print_success(message: str, **kwargs):
    """Prints a success message with a consistent prefix."""
    print(f"{PRINT_PATTERNS['SUCCESS_PREFIX']} {message}", **kwargs)


def print_failure(message: str, **kwargs):
    """Prints a failure message with a consistent prefix."""
    print(f"{PRINT_PATTERNS['FAILURE_PREFIX']} {message}", **kwargs)


def print_invalid_slot(slot_number: int):
    """Prints a standardized message for an invalid slot number."""
    print_error(f"{ERROR_MESSAGES['INVALID_SLOT']}: {slot_number}")


def print_ui_error(error: Exception):
    """Prints a standardized UI error message."""
    print_error(f"{ERROR_MESSAGES['UI_ERROR']}: {error}")


def print_processing_error(file_path: str, error: Exception):
    """Prints a standardized processing error message."""
    print_error(f"{ERROR_MESSAGES['PROCESSING_ERROR']} {file_path}: {error}")


def print_simulation_error(error: Exception):
    """Prints a standardized simulation error message."""
    print_error(f"{ERROR_MESSAGES['SIMULATION_ERROR']}: {error}")


def print_exception(context: str, error: Exception):
    """Prints a standardized exception message with context."""
    print_error(f"{context}: {error}")


def print_file_operation_error(operation: str, file_path: str, error: Exception):
    """Prints a standardized file operation error message."""
    print_error(f"Error {operation} {file_path}: {error}")


def print_graph_operation_error(operation: str, error: Exception):
    """Prints a standardized graph operation error message."""
    print_error(f"Error {operation}: {error}")






