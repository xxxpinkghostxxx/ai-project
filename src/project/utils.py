"""utils.py - Common math and utility functions for the neural system project.
Use these helpers to avoid code duplication and centralize reusable logic.
"""
import random
from typing import Any

def clamp(value: int | float, min_value: int | float, max_value: int | float) -> int | float:
    """Clamp a value between min_value and max_value."""
    return max(min_value, min(value, max_value))

def lerp(a: int | float, b: int | float, t: int | float) -> int | float:
    """Linear interpolation between a and b by t (0..1)."""
    return a + (b - a) * t

def random_range(rng: tuple[float, float]) -> float:
    """Return a random float in the given (min, max) tuple."""
    return random.uniform(*rng)

def get_free_ram() -> int | None:
    """Get available RAM in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None

def has_cupy() -> bool:
    """Check if CuPy is available."""
    try:
        import cupy  # type: ignore
        return True
    except ImportError:
        return False

def get_free_gpu_memory() -> int | None:
    """Get free GPU memory in bytes."""
    try:
        import cupy as cp  # type: ignore
        free, _ = cp.cuda.runtime.memGetInfo()  # type: ignore
        return free  # type: ignore[return-value]
    except Exception:
        return None

def get_array_module():
    """Get the best array module (CuPy if available and enabled, else NumPy)."""
    config = {}
    try:
        from project.config import USE_GPU
        config['USE_GPU'] = USE_GPU
    except ImportError:
        config['USE_GPU'] = False

    if config.get('USE_GPU', False):  # type: ignore[union-attr]
        try:
            import cupy as cp  # type: ignore
            return cp
        except ImportError:
            print('[CUDA] CuPy not available, falling back to NumPy.')
            config['USE_GPU'] = False
    return __import__('numpy')

def check_cuda_status() -> None:
    """Check CUDA status and configure device usage."""
    print("[CUDA DIAGNOSTIC] Checking CuPy and CUDA status...")
    try:
        import cupy  # type: ignore
        try:
            print("[CUDA DIAGNOSTIC] CuPy version:", cupy.__version__)  # type: ignore
            print("[CUDA DIAGNOSTIC] CuPy config:")
            print(cupy.show_config())  # type: ignore
            _ = cupy.zeros(1)  # type: ignore
            print("[CUDA DIAGNOSTIC] CuPy can allocate on GPU: SUCCESS")
        except Exception as e:
            print(f"[CUDA DIAGNOSTIC] CuPy import succeeded, but CUDA failed: {e}")
            print("[CUDA DIAGNOSTIC] Falling back to CPU mode.")
    except ImportError:
        print("[CUDA DIAGNOSTIC] CuPy is not installed. Falling back to CPU mode.")
    except Exception as e:
        print(f"[CUDA DIAGNOSTIC] Unexpected error: {e}")

def flatten_nodes(nodes: Any) -> list[Any]:
    """Flatten a list or numpy array of nodes (handles 2D lists/arrays)."""
    import numpy as np
    if isinstance(nodes, np.ndarray):
        return list(nodes.flatten())  # type: ignore
    if isinstance(nodes, list) and nodes and isinstance(nodes[0], list):
        return [n for row in nodes for n in row]  # type: ignore
    return list(nodes)  # type: ignore

def extract_node_attrs(node: Any) -> dict[str, Any]:
    """Extract a dict of non-callable, non-private attributes from a node object."""
    from collections.abc import Callable
    return {k: getattr(node, k, None) for k in dir(node) if not k.startswith('_') and not isinstance(getattr(node, k, None), Callable)}
