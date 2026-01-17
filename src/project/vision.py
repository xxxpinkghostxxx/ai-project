"""Vision module for screen capture and image processing.

This module provides threaded screen capture functionality and image
preprocessing utilities for the neural system. It supports multiple
backends (PIL and mss) with fallback mechanisms and robust error handling.
"""

import threading
import numpy as np
import cv2
import mss
from project.config import SENSOR_WIDTH, SENSOR_HEIGHT
import time
import queue
from project.config import PERIODIC_UPDATE_MS, SCREEN_CAPTURE_QUEUE_SIZE
import logging
from typing import Any, Optional, Tuple
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Pillow import only if needed
try:
    from PIL import ImageGrab
    pil_available = True
except ImportError:
    ImageGrab = None  # type: ignore
    pil_available = False
    logger.warning("PIL.ImageGrab not available, falling back to mss")

class ThreadedScreenCapture:
    """Threaded screen capture class for continuous frame acquisition.
    
    This class provides thread-safe screen capture functionality with
    configurable resolution and frame rate, using a producer-consumer
    pattern for efficient frame handling.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        interval: Optional[float] = None
    ) -> None:
        """Initialize the screen capture with specified dimensions and interval.
        
        Args:
            width: Target width for captured frames
            height: Target height for captured frames
            interval: Capture interval in seconds. If None, uses config default.
        """
        self.width = width
        self.height = height
        # Use adaptive polling interval from config if not specified
        self.interval = (
            interval
            if interval is not None
            else PERIODIC_UPDATE_MS / 1000.0
        )
        self._frame_lock = threading.Lock()
        self._latest_frame: NDArray[Any] = np.zeros(
            (height, width),
            dtype=np.uint8
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        # Async queue for frames with size limit
        self.frame_queue: queue.Queue[NDArray[Any]] = queue.Queue(
            maxsize=int(SCREEN_CAPTURE_QUEUE_SIZE)
        )
        self.frame_counter = 0
        self.drop_counter = 0
        self._error_count = 0
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds

    def __enter__(self) -> 'ThreadedScreenCapture':
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        """Context manager exit"""
        self.stop()

    def start(self) -> None:
        """Start the capture thread"""
        if not self._thread.is_alive():
            self._thread.start()
            logger.info("Screen capture thread started")

    def stop(self) -> None:
        """Stop the capture thread"""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Screen capture thread did not stop gracefully")
        logger.info("Screen capture thread stopped")

    def _capture_loop(self) -> None:
        """Main capture loop with error handling and recovery"""
        with mss.mss() as sct:
            while not self._stop_event.is_set():
                try:
                    # Get primary monitor
                    if len(sct.monitors) <= 1:
                        logger.error("No monitor found")
                        time.sleep(self._retry_delay)
                        continue

                    monitor = sct.monitors[1]  # Primary monitor
                    img = np.array(sct.grab(monitor))
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)  # type: ignore[attr-defined]
                    img = cv2.resize(img, (int(self.width), int(self.height)), interpolation=cv2.INTER_AREA)  # type: ignore[attr-defined]

                    # Update latest frame
                    with self._frame_lock:
                        self._latest_frame = img
                        self.frame_counter += 1

                    # Update frame queue
                    self._update_frame_queue(img)

                    # Reset error count on successful capture
                    self._error_count = 0

                except Exception as e:
                    self._error_count += 1
                    logger.error(f"Screen capture error: {e}")

                    if self._error_count >= self._max_retries:
                        logger.error("Max retries exceeded, stopping capture")
                        self._stop_event.set()
                        break

                    # Exponential backoff
                    time.sleep(self._retry_delay * (2 ** (self._error_count - 1)))

                # Wait for next capture interval
                self._stop_event.wait(self.interval)

    def _update_frame_queue(self, frame: NDArray[Any]) -> None:
        """Update the frame queue with proper error handling.
        
        Clears old frames and adds new frame to the queue.
        Handles queue full scenarios with frame dropping and logging.
        
        Args:
            frame: The frame to add to the queue
        """
        try:
            # Clear old frames to make space for new ones
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            # Add new frame to queue
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.drop_counter += 1
                # Log every 100 dropped frames to avoid log spam
                if not self.drop_counter % 100:
                    logger.warning(
                        f"Dropped {self.drop_counter} frames "
                        f"due to queue full"
                    )

        except Exception as queue_error:
            logger.error(
                f"Error updating frame queue: {queue_error}"
            )

    def get_latest(self) -> NDArray[Any]:
        """Get the latest frame with thread safety"""
        with self._frame_lock:
            return self._latest_frame.copy()

    def get_next_frame(self, timeout: float = 0.01) -> NDArray[Any] | None:
        """Get the next frame from the queue with timeout"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return self.get_latest()

    @property
    def is_running(self) -> bool:
        """Check if the capture thread is running"""
        return self._thread.is_alive()

    @property
    def error_count(self) -> int:
        """Get the current error count"""
        return self._error_count

def capture_screen(
    resolution: Tuple[int, int] = (SENSOR_WIDTH, SENSOR_HEIGHT)
) -> NDArray[Any]:
    """Capture screen with fallback options.
    
    Attempts to capture screen using PIL first, falls back to mss.
    Returns a properly sized RGB image or blank image on failure.
    
    Args:
        resolution: Target resolution as (width, height) tuple
        
    Returns:
        Captured screen image as numpy array in RGB format
    """
    start_time = time.time()
    img: NDArray[Any]
    backend_used = 'none'

    try:
        if pil_available and ImageGrab is not None:
            # Try PIL-based capture first
            pil_img = ImageGrab.grab()
            pil_img = pil_img.resize(resolution)
            img = np.array(pil_img)
            if img.shape[-1] == 4:
                img = img[..., :3]  # RGBA to RGB conversion
            backend_used = 'pillow'
        else:
            # Fallback to mss-based capture
            with mss.mss() as sct:
                if len(sct.monitors) <= 1:
                    logger.warning(
                        "No monitor found, returning blank image"
                    )
                    img = np.zeros(
                        (resolution[1], resolution[0], 3),
                        dtype=np.uint8
                    )
                else:
                    monitor = sct.monitors[1]  # Primary monitor
                    screenshot = sct.grab(monitor)
                    img = np.array(screenshot)
                    img = img[..., :3]  # BGRA to BGR conversion
                    img = cv2.resize(  # type: ignore[attr-defined]
                        img,
                        tuple(resolution)
                    )
            backend_used = 'mss'
    except Exception as capture_error:
        logger.error(f"Screen capture failed: {capture_error}")
        img = np.zeros(
            (resolution[1], resolution[0], 3),
            dtype=np.uint8
        )

    end_time = time.time()
    logger.debug(
        f"capture_screen ({backend_used}): "
        f"shape={img.shape}, "
        f"min={img.min()}, "
        f"max={img.max()}, "
        f"duration={end_time-start_time:.3f}s"
    )
    return img

def preprocess_image(image: NDArray[Any]) -> NDArray[Any]:
    """Preprocess image with error handling.
    
    Converts image to grayscale and resizes to sensor dimensions.
    Handles both RGB and grayscale input images.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed grayscale image resized to sensor dimensions
    """
    try:
        if len(image.shape) == 3:
            # Convert RGB/BGR image to grayscale
            gray = cv2.cvtColor(  # type: ignore[attr-defined]
                image,
                cv2.COLOR_BGR2GRAY
            )
        else:
            # Already grayscale
            gray = image
        
        # Resize to sensor dimensions
        resized = cv2.resize(  # type: ignore[attr-defined]
            gray,
            (int(SENSOR_WIDTH), int(SENSOR_HEIGHT))
        )
        
        logger.debug(
            f"preprocess_image: "
            f"shape={resized.shape}, "
            f"min={resized.min()}, "
            f"max={resized.max()}"
        )
        return resized
    except Exception as preprocessing_error:
        logger.error(
            f"Image preprocessing failed: {preprocessing_error}"
        )
        return np.zeros(
            (int(SENSOR_HEIGHT), int(SENSOR_WIDTH)),
            dtype=np.uint8
        )
