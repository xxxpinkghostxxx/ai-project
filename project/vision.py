import threading
import numpy as np
import cv2
import mss
from config import SENSOR_WIDTH, SENSOR_HEIGHT
import time
import queue
import config
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Toggle: set to 'mss' or 'pillow' to choose capture backend
CAPTURE_BACKEND = 'pillow'  # Change to 'mss' to use mss

# Pillow import only if needed
try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
    logger.warning("PIL.ImageGrab not available, falling back to mss")

class ThreadedScreenCapture:
    def __init__(self, width: int, height: int, interval: Optional[float] = None):
        self.width = width
        self.height = height
        # Use adaptive polling interval from config if not specified
        self.interval = interval if interval is not None else config.PERIODIC_UPDATE_MS / 1000.0
        self._frame_lock = threading.Lock()
        self._latest_frame = np.zeros((height, width), dtype=np.uint8)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        # Async queue for frames with size limit
        self.frame_queue = queue.Queue(maxsize=config.SCREEN_CAPTURE_QUEUE_SIZE)
        self.frame_counter = 0
        self.drop_counter = 0
        self._error_count = 0
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

    def start(self):
        """Start the capture thread"""
        if not self._thread.is_alive():
            self._thread.start()
            logger.info("Screen capture thread started")

    def stop(self):
        """Stop the capture thread"""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Screen capture thread did not stop gracefully")
        logger.info("Screen capture thread stopped")

    def _capture_loop(self):
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
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

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

    def _update_frame_queue(self, frame: np.ndarray):
        """Update the frame queue with proper error handling"""
        try:
            # Clear old frames
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            # Add new frame
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.drop_counter += 1
                if self.drop_counter % 100 == 0:  # Log every 100 drops
                    logger.warning(f"Dropped {self.drop_counter} frames due to queue full")

        except Exception as e:
            logger.error(f"Error updating frame queue: {e}")

    def get_latest(self) -> Optional[np.ndarray]:
        """Get the latest frame with thread safety"""
        with self._frame_lock:
            return self._latest_frame.copy()

    def get_next_frame(self, timeout: float = 0.01) -> Optional[np.ndarray]:
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

def capture_screen(resolution: Tuple[int, int] = (SENSOR_WIDTH, SENSOR_HEIGHT)) -> np.ndarray:
    """Capture screen with fallback options"""
    start = time.time()
    img = None
    backend = 'none'

    try:
        if CAPTURE_BACKEND == 'mss':
            with mss.mss() as sct:
                if len(sct.monitors) <= 1:
                    logger.warning("No monitor found, returning blank image")
                    img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                else:
                    monitor = sct.monitors[1]  # Primary monitor
                    screenshot = sct.grab(monitor)
                    img = np.array(screenshot)
                    img = img[..., :3]  # BGRA to BGR
                    img = cv2.resize(img, resolution)
            backend = 'mss'
        elif CAPTURE_BACKEND == 'pillow' and ImageGrab is not None:
            img = ImageGrab.grab()
            img = img.resize(resolution)
            img = np.array(img)
            if img.shape[-1] == 4:
                img = img[..., :3]  # RGBA to RGB
            backend = 'pillow'
    except Exception as e:
        logger.error(f"Screen capture failed: {e}")
        img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    if img is None:
        img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    end = time.time()
    logger.debug(f"capture_screen ({backend}): shape={img.shape}, min={img.min()}, max={img.max()}, duration={end-start:.3f}s")
    return img

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image with error handling"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        resized = cv2.resize(gray, (SENSOR_WIDTH, SENSOR_HEIGHT))
        logger.debug(f"preprocess_image: shape={resized.shape}, min={resized.min()}, max={resized.max()}")
        return resized
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.uint8) 
