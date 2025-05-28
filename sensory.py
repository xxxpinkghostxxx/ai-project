import numpy as np
import cv2
import mss
from config import EYE_LAYER_WIDTH, EYE_LAYER_HEIGHT

def capture_screen(resolution=(EYE_LAYER_WIDTH, EYE_LAYER_HEIGHT)):
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        # Convert BGRA to BGR
        img = img[..., :3]
        # Resize to desired resolution
        img = cv2.resize(img, resolution)
        return img

def preprocess_image(image, resolution=(EYE_LAYER_WIDTH, EYE_LAYER_HEIGHT)):
    # Convert to grayscale and resize
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, resolution)
    return resized 