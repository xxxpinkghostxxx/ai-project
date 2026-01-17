# Workspace Node System package
# This makes the workspace directory a proper Python package

"""Workspace Node System package.

This package contains the implementation of the workspace node system
that functions as the inverse of sensory nodes, reading energy levels
from sensory nodes and mapping them to a 16x16 grid visualization.
"""

# Package version information
__version__ = "1.0.0"
__author__ = "Workspace Node System Team"
__license__ = "MIT"

# Import main classes for easy access
from .workspace_node import WorkspaceNode
from .workspace_system import WorkspaceNodeSystem
from .config import EnergyReadingConfig
from .pixel_shading import PixelShadingSystem
from .renderer import WorkspaceRenderer
from .visualization import WorkspaceVisualization

__all__ = [
    'WorkspaceNode',
    'WorkspaceNodeSystem', 
    'EnergyReadingConfig',
    'PixelShadingSystem',
    'WorkspaceRenderer',
    'WorkspaceVisualization'
]