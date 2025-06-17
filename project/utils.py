"""
Common utility functions for the neural system project.

This module provides math helpers, profiling tools, system resource
checking, logging setup, and other reusable utility functions to avoid
code duplication across the project.
"""
import random
import time
import config
import logging
from logging.handlers import RotatingFileHandler
import os

def clamp(value, min_value, max_value):
    """Clamp a value between min_value and max_value."""
    return max(min_value, min(value, max_value))

def lerp(a, b, t):
    """Linear interpolation between a and b by t (0..1)."""
    return a + (b - a) * t

def random_range(rng):
    """Return a random float in the given (min, max) tuple."""
    return random.uniform(*rng)

_profile_timings = []
def profile_section(name):
    class Profiler:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start
            _profile_timings.append((name, elapsed))
            print(f"[PROFILE] {name}: {elapsed:.4f}s")
    return Profiler()

def profile_report():
    if not _profile_timings:
        print("[PROFILE] No timings recorded.")
        return
    print("[PROFILE REPORT] Slowest sections:")
    for name, elapsed in sorted(_profile_timings, key=lambda x: -x[1])[:5]:
        print(f"  {name}: {elapsed:.4f}s")
    _profile_timings.clear()

def get_free_ram():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None

def has_cupy():
    try:
        import cupy  # type: ignore
        return True
    except ImportError:
        return False

def get_free_gpu_memory():
    try:
        import cupy as cp # type: ignore
        free, total = cp.cuda.runtime.memGetInfo()
        return free
    except (ImportError, AttributeError, RuntimeError):
        return None

def get_array_module():
    if config.USE_GPU:
        try:
            import cupy as cp # type: ignore
            return cp
        except ImportError:
            print('[CUDA] CuPy not available, falling back to NumPy.')
            config.USE_GPU = False
    return __import__('numpy')

def check_cuda_status():
    print("[CUDA DIAGNOSTIC] Checking CuPy and CUDA status...")
    try:
        import cupy # type: ignore
        try:
            print("[CUDA DIAGNOSTIC] CuPy version:", cupy.__version__)
            print("[CUDA DIAGNOSTIC] CuPy config:")
            print(cupy.show_config())
            a = cupy.zeros(1)
            print("[CUDA DIAGNOSTIC] CuPy can allocate on GPU: SUCCESS")
        except (RuntimeError, MemoryError, AttributeError) as e:
            print(f"[CUDA DIAGNOSTIC] CuPy import succeeded, but CUDA failed: {e}")
            print("[CUDA DIAGNOSTIC] Falling back to CPU mode.")
            config.USE_GPU = False
    except ImportError:
        print("[CUDA DIAGNOSTIC] CuPy is not installed. Falling back to CPU mode.")
        config.USE_GPU = False
    except (AttributeError, RuntimeError) as e:
        print(f"[CUDA DIAGNOSTIC] Unexpected error: {e}")
        config.USE_GPU = False

def setup_logging(log_dir, log_filename, console_log_filename, max_size, backups, debug_mode):
    """
    Set up logging with rotating file handlers and a stream handler for console.
    File handlers get DEBUG/INFO, console gets WARNING+.
    Returns the configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            RotatingFileHandler(log_filename, maxBytes=max_size, backupCount=backups),
            RotatingFileHandler(console_log_filename, maxBytes=max_size, backupCount=backups),
        ]
    )
    logger = logging.getLogger('energy_neural_system')
    # Set StreamHandler (console) to WARNING level
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.WARNING)
    return logger

# Global logger instance (initialized on import)
try:
    LOG_DIR = 'logs'
    os.makedirs(LOG_DIR, exist_ok=True)
    from datetime import datetime
    log_filename = os.path.join(LOG_DIR, f'ai_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    console_log_filename = os.path.join(LOG_DIR, f'console_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    LOG_MAX_SIZE = int(1.8 * 1024 * 1024)
    LOG_BACKUPS = 3
    DEBUG_MODE = getattr(config, 'DEBUG_MODE', False)
    logger = setup_logging(LOG_DIR, log_filename, console_log_filename, LOG_MAX_SIZE, LOG_BACKUPS, DEBUG_MODE)
except (OSError, ValueError, AttributeError) as e:
    print(f"[LOGGING ERROR] Failed to set up logging: {e}")
    logger = None

def get_logger():
    return logger

def flatten_nodes(nodes):
    """Flatten a list or numpy array of nodes (handles 2D lists/arrays)."""
    import numpy as np
    if isinstance(nodes, np.ndarray):
        return list(nodes.flatten())
    if isinstance(nodes, list) and nodes and isinstance(nodes[0], list):
        return [n for row in nodes for n in row]
    return list(nodes)

def extract_node_attrs(node):
    """Extract a dict of non-callable, non-private attributes from a node object."""
    return {k: getattr(node, k, None) for k in dir(node) if not k.startswith('_') and not callable(getattr(node, k, None))} 
