import logging
from tkinter import messagebox
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dgl_system.log'
)
logger = logging.getLogger('DGLSystem')

class ErrorHandler:
    @staticmethod
    def show_error(title, message, log=True):
        """Show error message to user and optionally log it"""
        try:
            messagebox.showerror(title, message)
            if log:
                logger.error(f"{title}: {message}")
        except Exception as e:
            logger.error(f"Error showing error message: {str(e)}")
            print(f"Error showing error message: {str(e)}")

    @staticmethod
    def safe_operation(func):
        """Decorator for safe operations with proper error handling"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{func.__name__} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                ErrorHandler.show_error("Operation Failed", error_msg)
                return None
        return wrapper

    @staticmethod
    def log_warning(message):
        """Log a warning message"""
        logger.warning(message)

    @staticmethod
    def log_info(message):
        """Log an info message"""
        logger.info(message)

    @staticmethod
    def log_debug(message):
        """Log a debug message"""
        logger.debug(message) 