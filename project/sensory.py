import numpy as np
import cv2  # pylint: disable=import-error
import mss
from typing import Tuple, Any
from numpy.typing import NDArray
from project.config import SENSOR_WIDTH, SENSOR_HEIGHT

def capture_screen(resolution: Tuple[int, int] = (SENSOR_WIDTH, SENSOR_HEIGHT)) -> NDArray[Any]:
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        # Convert BGRA to BGR
        img = img[..., :3]
        # Resize to desired resolution
        img = cv2.resize(img, resolution)  # type: ignore[attr-defined]
        return img

def preprocess_image(image: NDArray[Any], resolution: Tuple[int, int] = (SENSOR_WIDTH, SENSOR_HEIGHT)) -> NDArray[Any]:
    # Convert to grayscale and resize
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]
    else:
        gray = image
    resized = cv2.resize(gray, resolution)  # type: ignore[attr-defined]
    return resized
