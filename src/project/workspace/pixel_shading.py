"""
Pixel Shading System

This module handles the conversion of energy levels to pixel values
and provides various shading modes for visualization.
"""

import math
from typing import List, Tuple


class PixelShadingSystem:
    """Handles energy-to-pixel conversion and visual effects."""
    
    def __init__(self, energy_min: float = 0.0, energy_max: float = 244.0):
        """
        Initialize pixel shading system.
        
        Args:
            energy_min: Minimum energy value for scaling
            energy_max: Maximum energy value for scaling
        """
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.shading_mode = 'linear'  # 'linear', 'logarithmic', 'exponential'
        self.color_scheme = 'grayscale'  # 'grayscale', 'heatmap', 'custom'
    
    def energy_to_pixel_value(self, energy: float) -> int:
        """
        Convert energy level to pixel grayscale value (0-255).
        
        Args:
            energy: Energy level to convert
            
        Returns:
            Grayscale pixel value (0-255)
        """
        # Normalize energy to 0-1 range
        normalized = (energy - self.energy_min) / (self.energy_max - self.energy_min)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to 0-1
        
        # Apply shading mode
        if self.shading_mode == 'linear':
            pixel_value = int(normalized * 255)
        elif self.shading_mode == 'logarithmic':
            # Logarithmic scaling for better low-energy visibility
            pixel_value = int(math.log1p(normalized * 1000) / math.log1p(1000) * 255)
        elif self.shading_mode == 'exponential':
            # Exponential scaling to emphasize high-energy states
            pixel_value = int((normalized ** 2) * 255)
        else:
            pixel_value = int(normalized * 255)
        
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
            # Add pulsing effect for increasing energy
            return min(255, pixel_value + 20)
        elif energy_trend == 'decreasing':
            # Add dimming effect for decreasing energy
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
        if self.color_scheme == 'grayscale':
            return (pixel_value, pixel_value, pixel_value)
        
        elif self.color_scheme == 'heatmap':
            # Blue (low) -> Green -> Red (high)
            if pixel_value < 85:
                # Blue to Green
                ratio = pixel_value / 85.0
                return (int(255 * (1 - ratio)), int(255 * ratio), 0)
            else:
                # Green to Red
                ratio = (pixel_value - 85) / 170.0
                return (int(255 * ratio), int(255 * (1 - ratio)), 0)
        
        elif self.color_scheme == 'custom':
            # Custom color mapping
            # Implement custom color scheme here
            return (pixel_value, 255 - pixel_value, pixel_value // 2)
        
        else:
            return (pixel_value, pixel_value, pixel_value)
    
    def set_shading_mode(self, mode: str):
        """Set the shading mode."""
        valid_modes = ['linear', 'logarithmic', 'exponential']
        if mode in valid_modes:
            self.shading_mode = mode
        else:
            raise ValueError(f"Invalid shading mode: {mode}")
    
    def set_color_scheme(self, scheme: str):
        """Set the color scheme."""
        valid_schemes = ['grayscale', 'heatmap', 'custom']
        if scheme in valid_schemes:
            self.color_scheme = scheme
        else:
            raise ValueError(f"Invalid color scheme: {scheme}")