# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Classes:
#   EnergyReadingConfig:                                           @dataclass
#     grid_size: Tuple[int, int] = (16, 16)
#     pixel_size: int = 20
#     grid_spacing: int = 2
#     reading_interval_ms: int = 50
#     energy_smoothing: bool = True
#     smoothing_factor: float = 0.1
#     energy_threshold_min: float = 0.0
#     energy_threshold_max: float = NODE_ENERGY_CAP
#     shading_mode: str = 'linear'
#     color_scheme: str = 'grayscale'
#     animation_enabled: bool = True
#     animation_speed: float = 0.1
#     batch_updates: bool = True
#     max_fps: int = 60
#     memory_optimization: bool = True
#     cache_size: int = 1000
#     cache_validity_ms: int = 100
#     retry_attempts: int = 3
#     retry_delay_ms: int = 10
#     error_threshold: float = 0.1
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [critical] Merge or replace with cube/panel config (EnergyReadingConfig is
#   rectangular-workspace-specific); align with pyg_config.json cube section.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Configuration parameters for the workspace node system."""

from dataclasses import dataclass
from typing import Tuple

from project.config import NODE_ENERGY_CAP


@dataclass
class EnergyReadingConfig:
    """Configuration for energy reading and workspace system."""

    grid_size: Tuple[int, int] = (16, 16)
    pixel_size: int = 20
    grid_spacing: int = 2

    reading_interval_ms: int = 50
    energy_smoothing: bool = True
    smoothing_factor: float = 0.1
    energy_threshold_min: float = 0.0
    energy_threshold_max: float = float(NODE_ENERGY_CAP)

    shading_mode: str = 'linear'
    color_scheme: str = 'grayscale'
    animation_enabled: bool = True
    animation_speed: float = 0.1

    batch_updates: bool = True
    max_fps: int = 60
    memory_optimization: bool = True
    cache_size: int = 1000
    cache_validity_ms: int = 100

    retry_attempts: int = 3
    retry_delay_ms: int = 10
    error_threshold: float = 0.1
