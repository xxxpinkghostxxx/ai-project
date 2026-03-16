# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Classes:
#   PixelShadingSystem:
#     __init__(self, energy_min: float = NODE_DEATH_THRESHOLD,
#         energy_max: float = NODE_ENERGY_CAP)
#
#     energy_to_pixel_value(self, energy: float) -> int
#       Convert energy level to pixel grayscale value (0-255)
#
#     apply_visual_effects(self, pixel_value: int, energy_trend: str,
#         animation_enabled: bool = True) -> int
#       Apply visual effects based on energy trends and animation settings
#
#     get_color_for_value(self, pixel_value: int) -> Tuple[int, int, int]
#       Get RGB color for a pixel value based on color scheme
#
#     _color_grayscale(pv: int) -> Tuple[int, int, int]    @staticmethod
#     _color_heatmap(pv: int) -> Tuple[int, int, int]      @staticmethod
#     _color_custom(pv: int) -> Tuple[int, int, int]       @staticmethod
#
#     set_shading_mode(self, mode: str)
#       Set shading mode and update cached function reference
#
#     set_color_scheme(self, scheme: str)
#       Set color scheme and update cached function reference
#
# =============================================================================
# TODOS
# =============================================================================
#
# None
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Energy-to-pixel conversion and visual effects for workspace visualization."""

import math
from typing import List, Tuple

from project.config import NODE_DEATH_THRESHOLD, NODE_ENERGY_CAP


class PixelShadingSystem:
    """Handles energy-to-pixel conversion and visual effects."""

    def __init__(self, energy_min: float = NODE_DEATH_THRESHOLD,
                 energy_max: float = NODE_ENERGY_CAP):
        """
        Initialize pixel shading system.

        Args:
            energy_min: Minimum energy value for scaling
            energy_max: Maximum energy value for scaling
        """
        self.energy_min = energy_min
        self.energy_max = energy_max

        self._shading_funcs = {
            'linear': lambda n: int(n * 255),
            'logarithmic': lambda n: int(math.log1p(n * 1000) / math.log1p(1000) * 255),
            'exponential': lambda n: int((n ** 2) * 255),
        }
        self._color_funcs = {
            'grayscale': self._color_grayscale,
            'heatmap': self._color_heatmap,
            'custom': self._color_custom,
        }

        self.shading_mode = 'linear'
        self.color_scheme = 'grayscale'
        self._shading_func = self._shading_funcs['linear']
        self._color_func = self._color_funcs['grayscale']

    def energy_to_pixel_value(self, energy: float) -> int:
        """
        Convert energy level to pixel grayscale value (0-255).

        Args:
            energy: Energy level to convert

        Returns:
            Grayscale pixel value (0-255)
        """
        energy_range = self.energy_max - self.energy_min
        normalized = (energy - self.energy_min) / energy_range if energy_range != 0 else 0.0
        normalized = max(0.0, min(1.0, normalized))

        pixel_value = self._shading_func(normalized)
        return max(0, min(255, pixel_value))

    def apply_visual_effects(self, pixel_value: int, energy_trend: str,
                           animation_enabled: bool = True) -> int:
        """
        Apply visual effects based on energy trends and animation settings.

        Args:
            pixel_value: Base pixel value (0-255)
            energy_trend: Energy trend ('increasing', 'decreasing', 'stable')
            animation_enabled: Whether animations are enabled

        Returns:
            Modified pixel value with effects applied
        """
        if not animation_enabled:
            return pixel_value

        if energy_trend == 'increasing':
            return min(255, pixel_value + 20)
        elif energy_trend == 'decreasing':
            return max(0, pixel_value - 10)
        else:
            return pixel_value

    def get_color_for_value(self, pixel_value: int) -> Tuple[int, int, int]:
        """
        Get RGB color for a pixel value based on color scheme.

        Args:
            pixel_value: Pixel value (0-255)

        Returns:
            RGB color tuple
        """
        return self._color_func(pixel_value)

    @staticmethod
    def _color_grayscale(pv: int) -> Tuple[int, int, int]:
        return (pv, pv, pv)

    @staticmethod
    def _color_heatmap(pv: int) -> Tuple[int, int, int]:
        t = pv / 255.0
        if t < 0.5:
            ratio = t * 2.0
            r = 0
            g = int(255 * ratio)
            b = int(255 * (1.0 - ratio))
        else:
            ratio = (t - 0.5) * 2.0
            r = int(255 * ratio)
            g = int(255 * (1.0 - ratio))
            b = 0
        return (r, g, b)

    @staticmethod
    def _color_custom(pv: int) -> Tuple[int, int, int]:
        return (pv, 255 - pv, pv // 2)

    def set_shading_mode(self, mode: str):
        """Set the shading mode and update cached function reference."""
        if mode not in self._shading_funcs:
            raise ValueError(f"Invalid shading mode: {mode}")
        self.shading_mode = mode
        self._shading_func = self._shading_funcs[mode]

    def set_color_scheme(self, scheme: str):
        """Set the color scheme and update cached function reference."""
        if scheme not in self._color_funcs:
            raise ValueError(f"Invalid color scheme: {scheme}")
        self.color_scheme = scheme
        self._color_func = self._color_funcs[scheme]
