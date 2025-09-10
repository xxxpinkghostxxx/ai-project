import numpy as np
import time
from logging_utils import log_runtime, log_step

try:
    import mss
    import mss.tools

    HAS_MSS = True
except ImportError:
    HAS_MSS = False
from PIL import ImageGrab

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
import torch
from torch_geometric.data import Data

# --- Config ---
RESOLUTION_SCALE = 0.25  # 1.0 = full res, 0.5 = half, 0.25 = quarter, etc.


def rgb_to_gray(arr):
    """
    Convert an RGB image to grayscale using the standard luminance formula.
    Y = 0.2125 R + 0.7154 G + 0.0721 B
    arr: shape (H, W, 3)
    Returns: (H, W) float32
    """
    gray = np.dot(arr[..., :3], [0.2125, 0.7154, 0.0721]).astype(np.float32)
    # Assertion: output shape matches input height/width
    assert gray.shape == arr.shape[:2], "Grayscale conversion shape mismatch"
    return gray


# --- Screen Capture (Optimized) ---
@log_runtime
def capture_screen(scale=1.0):
    """
    Capture the screen and return a numpy grayscale array, downscaled if needed.
    Uses mss (fastest) if available, else falls back to PIL.ImageGrab.
    Uses OpenCV for resizing if available, else PIL.
    
    Security: Validates scale parameter to prevent resource exhaustion attacks.
    """
    # Validate scale parameter for security
    if not isinstance(scale, (int, float)):
        raise ValueError("Scale must be a number")
    if scale <= 0 or scale > 2.0:  # Limit scale to prevent resource exhaustion
        raise ValueError("Scale must be between 0 and 2.0")
    
    t0 = time.perf_counter()
    log_step("capture_screen: start", scale=scale)
    try:
        if HAS_MSS:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)[..., :3]  # BGRA to BGR
                img = img[..., ::-1]  # BGR to RGB
                log_step("capture_screen: used mss", shape=img.shape)
        else:
            img = ImageGrab.grab()
            img = img.convert("RGB")
            img = np.array(img)
            log_step("capture_screen: used PIL.ImageGrab", shape=img.shape)
    except Exception as e:
        # Safe fallback: generate a synthetic grayscale frame so UI doesn't crash
        log_step("capture_screen: fallback synthetic frame due to error", error=str(e))
        img = np.zeros((360, 640, 3), dtype=np.uint8)
    t1 = time.perf_counter()
    # Downscale if needed
    if scale != 1.0:
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        try:
            if HAS_CV2:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                log_step("capture_screen: resized with cv2", new_shape=img.shape)
            else:
                pil_img = ImageGrab.Image.fromarray(img)
                pil_img = pil_img.resize((new_w, new_h), ImageGrab.Image.LANCZOS)
                img = np.array(pil_img)
                log_step("capture_screen: resized with PIL", new_shape=img.shape)
        except Exception as e:
            log_step("capture_screen: resize failed, keeping original size", error=str(e))
    t2 = time.perf_counter()
    # Convert to grayscale
    gray_img = rgb_to_gray(img)
    t3 = time.perf_counter()
    # Profiling logs
    log_step(
        "capture_screen: timing",
        capture_ms=(t1 - t0) * 1000,
        resize_ms=(t2 - t1) * 1000,
        gray_ms=(t3 - t2) * 1000,
        final_shape=gray_img.shape,
    )
    print(
        f"[PERF] Screen capture: {(t1-t0)*1000:.1f} ms, Resize: {(t2-t1)*1000:.1f} ms, Grayscale: {(t3-t2)*1000:.1f} ms, Shape: {gray_img.shape}"
    )
    return gray_img


# --- Graph Construction ---
@log_runtime
def create_pixel_gray_graph(arr):
    log_step("create_pixel_gray_graph: start", arr_shape=arr.shape)
    # arr: shape (H, W), grayscale
    h, w = arr.shape
    num_nodes = h * w
    node_features = arr.flatten().reshape(-1, 1)  # shape (num_nodes, 1)
    node_labels = []
    
    # Import ID manager for unique ID generation
    from node_id_manager import get_id_manager
    id_manager = get_id_manager()
    
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            energy = arr[y, x]
            
            # Generate unique ID for this sensory node
            node_id = id_manager.generate_unique_id("sensory", {"x": x, "y": y})
            
            # Calculate normalized membrane potential (0-1)
            membrane_potential = min(energy / 255.0, 1.0)
            
            node_labels.append({
                "id": node_id,  # UNIQUE ID: Primary identifier
                "type": "sensory",
                "behavior": "sensory",
                "x": x,  # METADATA: Spatial coordinate preserved as metadata
                "y": y,  # METADATA: Spatial coordinate preserved as metadata
                "energy": float(energy),
                "state": "active",  # Sensory nodes are always active
                "membrane_potential": membrane_potential,
                "threshold": 0.5,  # Activation threshold for downstream connections
                "refractory_timer": 0.0,  # No refractory period for sensory nodes
                "last_activation": 0,  # Sensory nodes don't have traditional activation
                "plasticity_enabled": False,  # Sensory nodes don't learn
                "eligibility_trace": 0.0,  # No learning for sensory nodes
                "last_update": 0
            })
            
            # Register the node index with the ID manager
            id_manager.register_node_index(node_id, idx)
    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    # Assertion: node_labels and x must match num_nodes
    assert len(node_labels) == num_nodes, "Node label count mismatch in create_pixel_gray_graph"
    assert x_tensor.shape[0] == num_nodes, "Node feature count mismatch in create_pixel_gray_graph"
    edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
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
        arr = capture_screen(scale=RESOLUTION_SCALE)
        print("Profiling create_pixel_gray_graph...")
        profiler = cProfile.Profile()
        profiler.enable()
        graph = create_pixel_gray_graph(arr)
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
