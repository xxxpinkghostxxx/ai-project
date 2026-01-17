"""
Workspace System Configuration

This module defines configuration parameters for the workspace node system.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnergyReadingConfig:
    """Configuration for energy reading and workspace system."""
    
    # Grid configuration
    grid_size: Tuple[int, int] = (16, 16)
    pixel_size: int = 20
    grid_spacing: int = 2
    
    # Energy reading configuration
    reading_interval_ms: int = 50  # 20 Hz update rate
    energy_smoothing: bool = True
    smoothing_factor: float = 0.1
    energy_threshold_min: float = 0.0
    energy_threshold_max: float = 244.0  # From config.py
    
    # Visualization configuration
    shading_mode: str = 'linear'  # 'linear', 'logarithmic', 'exponential'
    color_scheme: str = 'grayscale'  # 'grayscale', 'heatmap', 'custom'
    animation_enabled: bool = True
    animation_speed: float = 0.1
    
    # Performance configuration
    batch_updates: bool = True
    max_fps: int = 60
    memory_optimization: bool = True
    cache_size: int = 1000
    cache_validity_ms: int = 100
    
    # Error handling
    retry_attempts: int = 3
    retry_delay_ms: int = 10
    error_threshold: float = 0.1  # Maximum acceptable error rate