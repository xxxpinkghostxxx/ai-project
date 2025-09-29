"""
Screen graph utilities for visual data processing.

This module provides functionality for:
- Screen capture and processing
- Converting pixel data to neural network graphs
- Performance profiling for graph creation
"""

import time

import numpy as np

try:
    import mss

    HAS_MSS = True
except ImportError:
    HAS_MSS = False
from PIL import Image, ImageGrab  # pylint: disable=no-member

try:
    import cv2  # pylint: disable=no-member
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
import torch
from torch_geometric.data import Data

from utils.logging_utils import log_runtime, log_step

RESOLUTION_SCALE = 0.25


def rgb_to_gray(arr):
    """
    Convert RGB array to grayscale.

    Args:
        arr: RGB image array

    Returns:
        Grayscale version of the input array
    """
    gray = np.dot(arr[..., :3], [0.2125, 0.7154, 0.0721]).astype(np.float32)
    assert gray.shape == arr.shape[:2], "Grayscale conversion shape mismatch"
    return gray
@log_runtime


def capture_screen(scale=1.0):
    """
    Capture screen and convert to grayscale array.

    Args:
        scale: Scaling factor for the captured image (0-2.0)

    Returns:
        Grayscale image as numpy array
    """
    if not isinstance(scale, (int, float)):
        raise ValueError("Scale must be a number")
    if scale <= 0 or scale > 2.0:
        raise ValueError("Scale must be between 0 and 2.0")
    t0 = time.perf_counter()
    log_step("capture_screen: start", scale=scale)
    try:
        if HAS_MSS:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)[..., :3]
                img = img[..., ::-1]
                log_step("capture_screen: used mss", shape=img.shape)
        else:
            img = ImageGrab.grab()
            img = img.convert("RGB")
            img = np.array(img)
            log_step("capture_screen: used PIL.ImageGrab", shape=img.shape)
    except (OSError, RuntimeError, ImportError) as e:
        log_step("capture_screen: fallback synthetic frame due to error", error=str(e))
        img = np.zeros((360, 640, 3), dtype=np.uint8)
    t1 = time.perf_counter()
    if scale != 1.0:
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        try:
            if HAS_CV2:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
                log_step("capture_screen: resized with cv2", new_shape=img.shape)
            else:
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)  # pylint: disable=no-member
                img = np.array(pil_img)
                log_step("capture_screen: resized with PIL", new_shape=img.shape)
        except (OSError, ValueError, MemoryError) as e:
            log_step("capture_screen: resize failed, keeping original size", error=str(e))
    t2 = time.perf_counter()
    gray_img = rgb_to_gray(img)
    t3 = time.perf_counter()
    log_step(
        "capture_screen: timing",
        capture_ms=(t1 - t0) * 1000,
        resize_ms=(t2 - t1) * 1000,
        gray_ms=(t3 - t2) * 1000,
        final_shape=gray_img.shape,
    )
    return gray_img
@log_runtime


def create_pixel_gray_graph(image_array):
    """
    Create a PyTorch Geometric graph from grayscale pixel data.

    Args:
        image_array: 2D grayscale image array

    Returns:
        PyTorch Geometric Data object with pixel nodes
    """
    log_step("create_pixel_gray_graph: start", arr_shape=image_array.shape)
    h, w = image_array.shape
    num_nodes = h * w

    # Limit nodes for performance during UI initialization
    if h > 31 or w > 31:
        # Sample a smaller grid for UI initialization
        sample_h = min(h, 31)
        sample_w = min(w, 31)
        step_h = max(1, h // sample_h)
        step_w = max(1, w // sample_w)

        sampled_arr = image_array[::step_h, ::step_w]
        h, w = sampled_arr.shape
        h = min(h, 31)
        w = min(w, 31)
        num_nodes = h * w
        log_step("Reduced graph size for performance", original=(image_array.shape), sampled=(h, w), nodes=num_nodes)

        image_array = sampled_arr

    node_features = image_array.flatten().reshape(-1, 1)
    node_labels = []

    # Use simple sequential IDs for screen capture graphs to avoid expensive ID manager operations
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            energy = image_array[y, x]
            # Use simple sequential ID instead of expensive ID manager
            node_id = idx + 1000000  # Offset to avoid conflicts with regular nodes
            membrane_potential = min(energy / 255.0, 1.0)
            node_labels.append({
                "id": node_id,
                "type": "sensory",
                "behavior": "sensory",
                "x": x,
                "y": y,
                "energy": float(energy),
                "state": "active",
                "membrane_potential": membrane_potential,
                "threshold": 0.5,
                "refractory_timer": 0.0,
                "last_activation": 0,
                "plasticity_enabled": False,
                "eligibility_trace": 0.0,
                "last_update": 0,
                "is_excitatory": True,
                "I_syn": 0.0,
                "IEG_flag": False,
                "plast_enabled": False,
                "theta_burst_counter": 0,
                "v_dend": 0.0
            })

    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    assert len(node_labels) == num_nodes, "Node label count mismatch in create_pixel_gray_graph"
    assert x_tensor.shape[0] == num_nodes, "Node feature count mismatch in create_pixel_gray_graph"
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_index)
    data.node_labels = node_labels
    data.h = h
    data.w = w
    log_step("create_pixel_gray_graph: end", num_nodes=num_nodes, h=h, w=w)
    return data
if __name__ == "__main__":
    import cProfile
    import pstats
    print("Press Ctrl+C to stop.")
    try:
        screen_array = capture_screen(scale=RESOLUTION_SCALE)
        print("Profiling create_pixel_gray_graph...")
        profiler = cProfile.Profile()
        profiler.enable()
        graph = create_pixel_gray_graph(screen_array)
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
        stats.print_stats(20)
        print(
            f"Graph: {len(graph.x)} nodes for {graph.h}x{graph.w} pixels (scale={RESOLUTION_SCALE})"
        )
        print(
            f"First 6 node features/labels: {[{'feature': graph.x[i].item(), **graph.node_labels[i]} for i in range(6)]}"
        )
        time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")

