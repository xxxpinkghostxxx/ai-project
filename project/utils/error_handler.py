import logging
from tkinter import messagebox
from functools import wraps
from typing import Any, Callable, TypeVar, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='pyg_system.log'
)
logger = logging.getLogger('PyGSystem')

class ErrorHandler:
    def __init__(self) -> None:
        self.errors: List[str] = []

    @staticmethod
    def show_error(title: str, message: str, log: bool = True) -> None:
        """Show error message to user and optionally log it"""
        try:
            messagebox.showerror(title, message)
            if log:
                logger.error(f"{title}: {message}")
        except Exception as e:
            logger.error(f"Error showing error message: {str(e)}")
            print(f"Error showing error message: {str(e)}")

    @staticmethod
    def safe_operation(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for safe operations with proper error handling"""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{func.__name__} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                ErrorHandler.show_error("Operation Failed", error_msg)
                return None
        return wrapper

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

    def log_error(self, message: str) -> None:
        """Log an error message and store it"""
        self.errors.append(message)
        logger.error(message)

    def get_error_summary(self) -> str:
        """Get a summary of all errors"""
        if not self.errors:
            return "No errors recorded"
        return f"Total errors: {len(self.errors)}. Last error: {self.errors[-1] if self.errors else 'None'}"

    def clear_errors(self) -> None:
        """Clear all stored errors"""
        self.errors.clear()

    def get_recent_errors(self, count: int = 10) -> List[str]:
        """Get the most recent errors"""
        return self.errors[-count:] if self.errors else []

    def __len__(self) -> int:
        """Return the number of stored errors"""
        return len(self.errors)
