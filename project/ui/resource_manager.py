from PIL import Image, ImageTk
from PIL.Image import Resampling
from typing import Any, Callable, List, Optional, Tuple
from project.utils.error_handler import ErrorHandler

class UIResourceManager:
    def __init__(self: 'UIResourceManager', max_images: int = 100, max_windows: int = 10) -> None:
        self.images: List[Any] = []  # Keep references to prevent garbage collection
        self.windows: List[Any] = []  # Track all windows
        self._cleanup_handlers: List[Callable[[], None]] = []
        self._max_images = max_images
        self._max_windows = max_windows

    def register_image(self: 'UIResourceManager', image: Any) -> Optional[Any]:
        """Register an image to prevent garbage collection"""
        try:
            if len(self.images) >= self._max_images:
                self.images.pop(0)  # Remove oldest image
            self.images.append(image)
            return image
        except Exception as e:
            ErrorHandler.show_error("Resource Error", f"Failed to register image: {str(e)}")
            return None

    def register_window(self: 'UIResourceManager', window: Any) -> Optional[Any]:
        """Register a window for tracking"""
        try:
            if len(self.windows) >= self._max_windows:
                oldest_window = self.windows.pop(0)
                try:
                    oldest_window.destroy()
                except:
                    pass
            self.windows.append(window)
            return window
        except Exception as e:
            ErrorHandler.show_error("Resource Error", f"Failed to register window: {str(e)}")
            return None

    def register_cleanup(self: 'UIResourceManager', handler: Callable[[], None]) -> None:
        """Register a cleanup handler"""
        try:
            self._cleanup_handlers.append(handler)
        except Exception as e:
            ErrorHandler.show_error("Resource Error", f"Failed to register cleanup handler: {str(e)}")

    def cleanup(self: 'UIResourceManager') -> None:
        """Clean up all registered resources"""
        try:
            # Run cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    ErrorHandler.log_warning(f"Error during cleanup: {str(e)}")

            # Clear image references
            self.images.clear()

            # Destroy windows
            for window in self.windows:
                try:
                    window.destroy()
                except Exception as e:
                    ErrorHandler.log_warning(f"Error destroying window: {str(e)}")
            self.windows.clear()

            # Clear cleanup handlers
            self._cleanup_handlers.clear()

            ErrorHandler.log_info("Resource cleanup completed")
        except Exception as e:
            ErrorHandler.show_error("Cleanup Error", f"Failed to clean up resources: {str(e)}")

    def create_tk_image(self: 'UIResourceManager', image_data: Any, size: Optional[Tuple[int, int]] = None) -> Optional[Any]:
        """Create a Tkinter image from image data"""
        try:
            if isinstance(image_data, Image.Image):
                if size:
                    image_data = image_data.resize(size, Resampling.NEAREST)
                tk_image = ImageTk.PhotoImage(image_data)
                return self.register_image(tk_image)
            return None
        except Exception as e:
            ErrorHandler.show_error("Resource Error", f"Failed to create Tk image: {str(e)}")
            return None

    def __del__(self: 'UIResourceManager') -> None:
        """Destructor to ensure cleanup"""
        self.cleanup()
