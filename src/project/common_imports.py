"""
Centralized imports for the PyTorch Geometric Neural System project.
This module provides common imports to reduce redundancy across the codebase.
"""

# Standard library imports
import logging
import os
import shutil
import time
import tkinter as tk
from typing import Any

# Third-party imports
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageTk
import cv2
import mss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pyg_system.log'),
        logging.StreamHandler()
    ]
)

# Common type aliases
Tensor = torch.Tensor
ArrayLike = NDArray[np.float64] | Tensor
PathLike = str | os.PathLike[str]

# Common logger instance
logger = logging.getLogger(__name__)

# Utility functions for error handling
def safe_import(module_name: str, fallback: object | None = None) -> object | None:
    """Safely import a module with fallback."""
    try:
        return __import__(module_name)
    except ImportError:
        return fallback

def get_device() -> str:
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

# Profile timing utilities (consolidated from utils.py and utils/profile_section.py)
_profile_timings: list[tuple[str, float]] = []

def profile_section(name: str):
    """Context manager for profiling code sections."""
    class Profiler:
        """Profiler class for timing code sections."""
        def __init__(self):
            self.start = None

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None):
            if self.start is not None:
                elapsed = time.time() - self.start
                _profile_timings.append((name, elapsed))
                logger.debug("[PROFILE] %s: %.4fs", name, elapsed)
    return Profiler()

def profile_report():
    """Generate a profile report of the slowest sections."""
    if not _profile_timings:
        logger.info("[PROFILE] No timings recorded.")
        return
    sorted_timings = sorted(_profile_timings, key=lambda x: -x[1])[:5]
    logger.info("[PROFILE REPORT] Slowest sections:")
    for name, elapsed in sorted_timings:
        logger.info("  %s: %.4fs", name, elapsed)
    _profile_timings.clear()

# File system utilities
def ensure_dir(path: str | os.PathLike[str]) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def safe_remove(path: str | os.PathLike[str]) -> bool:
    """Safely remove a file or directory."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return True
    except OSError as e:
        logger.warning("Failed to remove %s: %s", path, e)
        return False

# Configuration utilities
def get_config_value(config_dict: dict[str, Any], key: str, default: str | int | float | bool | None = None) -> str | int | float | bool | None:
    """Get a value from config dict with fallback."""
    return config_dict.get(key, default)

def validate_config(config_dict: dict[str, Any], required_keys: list[str]) -> bool:
    """Validate that required keys exist in config."""
    return all(key in config_dict for key in required_keys)

# Export commonly used items for easier imports
__all__ = [
    # Types
    'Tensor', 'ArrayLike', 'PathLike',

    # Classes and functions
    'profile_section', 'profile_report', 'safe_import', 'get_device',
    'ensure_dir', 'safe_remove', 'get_config_value', 'validate_config',

    # Modules and types
    'np', 'torch', 'cv2', 'mss', 'Image', 'ImageTk', 'tk',

    # Logging
    'logger'
]
