"""
Workspace Renderer

This module handles the rendering of the 16x16 workspace grid
using PyQt6 for real-time visualization.
"""

from typing import List, Tuple, Optional
import numpy as np

from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsRectItem
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QBrush, QPen
from PyQt6.QtCore import Qt, QTimer

from .pixel_shading import PixelShadingSystem


class WorkspaceRenderer:
    """Handles rendering of the workspace grid in PyQt6."""
    
    def __init__(self, grid_size: Tuple[int, int] = (16, 16), 
                 pixel_size: int = 20):
        """
        Initialize workspace renderer.
        
        Args:
            grid_size: Size of the workspace grid (width, height)
            pixel_size: Size of each pixel in pixels
        """
        self.grid_size = grid_size
        self.pixel_size = pixel_size
        self.shading_system = PixelShadingSystem()
        
        # UI components
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Grid items
        self.grid_items = []
        self._setup_grid()
    
    def _setup_grid(self):
        """Set up the grid items for rendering."""
        self.scene.clear()
        self.grid_items = []
        
        for y in range(self.grid_size[1]):
            row_items = []
            for x in range(self.grid_size[0]):
                # Create grid item
                rect = QGraphicsRectItem(
                    x * self.pixel_size, y * self.pixel_size,
                    self.pixel_size, self.pixel_size
                )
                
                # Set initial appearance
                rect.setBrush(QBrush(QColor(0, 0, 0)))
                rect.setPen(QPen(Qt.GlobalColor.transparent))
                
                self.scene.addItem(rect)
                row_items.append(rect)
            
            self.grid_items.append(row_items)
    
    def render_grid(self, energy_data: List[List[float]], 
                   energy_trends: Optional[List[List[str]]] = None):
        """
        Render the workspace grid with energy data.
        
        Args:
            energy_data: 2D list of energy values
            energy_trends: 2D list of energy trends (optional)
        """
        if not energy_data or len(energy_data) != self.grid_size[1]:
            return
        
        for y in range(self.grid_size[1]):
            if y >= len(energy_data):
                continue
                
            row_data = energy_data[y]
            if len(row_data) != self.grid_size[0]:
                continue
            
            for x in range(self.grid_size[0]):
                energy = row_data[x]
                
                # Convert energy to pixel value
                pixel_value = self.shading_system.energy_to_pixel_value(energy)
                
                # Apply visual effects if trends are provided
                if energy_trends and y < len(energy_trends) and x < len(energy_trends[y]):
                    trend = energy_trends[y][x]
                    pixel_value = self.shading_system.apply_visual_effects(
                        pixel_value, trend
                    )
                
                # Get color
                color = self.shading_system.get_color_for_value(pixel_value)
                
                # Update grid item
                if y < len(self.grid_items) and x < len(self.grid_items[y]):
                    item = self.grid_items[y][x]
                    item.setBrush(QBrush(QColor(*color)))
    
    def resize_grid(self, new_grid_size: Tuple[int, int], new_pixel_size: int):
        """Resize the grid with new dimensions."""
        self.grid_size = new_grid_size
        self.pixel_size = new_pixel_size
        self._setup_grid()
    
    def set_shading_mode(self, mode: str):
        """Set the shading mode."""
        self.shading_system.set_shading_mode(mode)
    
    def set_color_scheme(self, scheme: str):
        """Set the color scheme."""
        self.shading_system.set_color_scheme(scheme)
    
    def get_view(self) -> QGraphicsView:
        """Get the QGraphicsView for embedding in UI."""
        return self.view