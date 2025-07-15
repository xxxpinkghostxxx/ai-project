import numpy as np
import time
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

# --- Screen Capture (Optimized) ---
def capture_screen(scale=1.0):
    """
    Capture the screen and return a numpy RGB array, downscaled if needed.
    Uses mss (fastest) if available, else falls back to PIL.ImageGrab.
    Uses OpenCV for resizing if available, else PIL.
    """
    t0 = time.perf_counter()
    if HAS_MSS:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)[..., :3]  # BGRA to BGR
            img = img[..., ::-1]  # BGR to RGB
    else:
        img = ImageGrab.grab()
        img = img.convert('RGB')
        img = np.array(img)
    t1 = time.perf_counter()
    # Downscale if needed
    if scale != 1.0:
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        if HAS_CV2:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            pil_img = ImageGrab.Image.fromarray(img)
            pil_img = pil_img.resize((new_w, new_h), ImageGrab.Image.LANCZOS)
            img = np.array(pil_img)
    t2 = time.perf_counter()
    # Profiling logs
    print(f"[PERF] Screen capture: {(t1-t0)*1000:.1f} ms, Resize: {(t2-t1)*1000:.1f} ms, Shape: {img.shape}")
    return img

# --- Graph Construction ---
def create_pixel_rgb_graph(arr):
    # arr: shape (H, W, 3)
    h, w, _ = arr.shape
    num_pixels = h * w
    num_nodes = num_pixels * 3  # 3 nodes per pixel (R, G, B)
    node_features = np.zeros((num_nodes, 1), dtype=np.float32)
    node_labels = []
    node_idx = 0
    for y in range(h):
        for x in range(w):
            for c, channel in enumerate(['R', 'G', 'B']):
                node_features[node_idx, 0] = arr[y, x, c]
                node_labels.append({'id': node_idx, 'x': x, 'y': y, 'channel': channel})
                node_idx += 1
    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
    data = Data(x=x_tensor, edge_index=edge_index)
    data.node_labels = node_labels
    data.h = h
    data.w = w
    return data

if __name__ == "__main__":
    import cProfile
    import pstats
    print("Press Ctrl+C to stop.")
    try:
        arr = capture_screen(scale=RESOLUTION_SCALE)
        print("Profiling create_pixel_rgb_graph...")
        profiler = cProfile.Profile()
        profiler.enable()
        graph = create_pixel_rgb_graph(arr)
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
        stats.print_stats(20)
        print(f"Graph: {len(graph.x)} nodes for {graph.h}x{graph.w} pixels (scale={RESOLUTION_SCALE})")
        print(f"First 6 node features/labels: {[{'feature': graph.x[i].item(), **graph.node_labels[i]} for i in range(6)]}")
        time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.") 