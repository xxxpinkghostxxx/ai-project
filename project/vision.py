import threading
import numpy as np
import cv2
import mss
from config import SENSOR_WIDTH, SENSOR_HEIGHT
import time
import queue
import config

# Toggle: set to 'mss' or 'pillow' to choose capture backend
CAPTURE_BACKEND = 'pillow'  # Change to 'mss' to use mss

# Pillow import only if needed
try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

class ThreadedScreenCapture:
    def __init__(self, width, height, interval=None):
        self.width = width
        self.height = height
        # Use adaptive polling interval from config if not specified
        self.interval = interval if interval is not None else config.PERIODIC_UPDATE_MS / 1000.0
        self.latest_frame = np.zeros((height, width), dtype=np.uint8)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        # Async queue for frames
        self.frame_queue = queue.Queue(maxsize=config.SCREEN_CAPTURE_QUEUE_SIZE)
        self.frame_counter = 0
        self.drop_counter = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _capture_loop(self):
        with mss.mss() as sct:
            while not self._stop_event.is_set():
                try:
                    monitor = sct.monitors[1]  # Primary monitor
                    img = np.array(sct.grab(monitor))
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    self.latest_frame = img
                    self.frame_counter += 1
                    # Always keep only the latest frame in the queue
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    try:
                        self.frame_queue.put_nowait(img)
                    except queue.Full:
                        pass  # Should not happen now
                except Exception as e:
                    pass  # Suppress errors in production
                self._stop_event.wait(self.interval)

    def get_latest(self):
        return self.latest_frame.copy()

    def get_next_frame(self, timeout=0.01):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return self.get_latest()

def capture_screen(resolution=(SENSOR_WIDTH, SENSOR_HEIGHT)):
    start = time.time()
    if CAPTURE_BACKEND == 'mss':
        with mss.mss() as sct:
            if len(sct.monitors) <= 1:
                # No monitor found, return blank image
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
    else:
        img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        backend = 'none'
    end = time.time()
    print(f"[DEBUG] capture_screen ({backend}): shape={img.shape}, min={img.min()}, max={img.max()}, duration={end-start:.3f}s")
    return img

def preprocess_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (SENSOR_WIDTH, SENSOR_HEIGHT))
    print(f"[DEBUG] preprocess_image: shape={resized.shape}, min={resized.min()}, max={resized.max()}")
    return resized 
