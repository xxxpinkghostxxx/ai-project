"""
IRealTimeVisualization interface - Real-time visualization service for neural simulation.

This interface defines the contract for real-time visualization of neural activity,
energy flow, and system performance in the neural simulation system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class VisualizationData:
    """Represents visualization data for rendering."""

    def __init__(self, data_type: str, timestamp: float):
        self.data_type = data_type  # "neural_activity", "energy_flow", "connections", "performance"
        self.timestamp = timestamp
        self.data = {}
        self.metadata = {}
        self.rendering_hints = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert visualization data to dictionary."""
        return {
            'data_type': self.data_type,
            'timestamp': self.timestamp,
            'data': self.data.copy(),
            'metadata': self.metadata.copy(),
            'rendering_hints': self.rendering_hints.copy()
        }


class VisualizationLayer:
    """Represents a visualization layer with rendering properties."""

    def __init__(self, layer_id: str, layer_type: str):
        self.layer_id = layer_id
        self.layer_type = layer_type  # "neural", "energy", "connections", "performance"
        self.visible = True
        self.opacity = 1.0
        self.z_index = 0
        self.color_scheme = "default"
        self.update_frequency = 30  # FPS
        self.last_update = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert layer to dictionary."""
        return {
            'layer_id': self.layer_id,
            'layer_type': self.layer_type,
            'visible': self.visible,
            'opacity': self.opacity,
            'z_index': self.z_index,
            'color_scheme': self.color_scheme,
            'update_frequency': self.update_frequency,
            'last_update': self.last_update
        }


class CameraController:
    """Controls camera/viewport for 3D visualization."""

    def __init__(self):
        self.position = [0.0, 0.0, 100.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.zoom = 1.0
        self.target = [0.0, 0.0, 0.0]
        self.projection_mode = "perspective"  # "perspective", "orthographic"
        self.field_of_view = 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert camera controller to dictionary."""
        return {
            'position': self.position.copy(),
            'rotation': self.rotation.copy(),
            'zoom': self.zoom,
            'target': self.target.copy(),
            'projection_mode': self.projection_mode,
            'field_of_view': self.field_of_view
        }


class IRealTimeVisualization(ABC):
    """
    Abstract interface for real-time visualization in neural simulation.

    This interface defines the contract for real-time visualization of neural
    activity, energy flow, connections, and system performance with interactive
    3D rendering capabilities.
    """

    @abstractmethod
    def initialize_visualization(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the visualization system with configuration.

        Args:
            config: Visualization configuration parameters

        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    def create_visualization_layer(self, layer_config: Dict[str, Any]) -> str:
        """
        Create a new visualization layer.

        Args:
            layer_config: Configuration for the visualization layer

        Returns:
            str: Layer ID for the created layer
        """
        pass

    @abstractmethod
    def update_visualization_data(self, layer_id: str, data: VisualizationData) -> bool:
        """
        Update visualization data for a specific layer.

        Args:
            layer_id: ID of the layer to update
            data: New visualization data

        Returns:
            bool: True if update successful
        """
        pass

    @abstractmethod
    def render_frame(self) -> Dict[str, Any]:
        """
        Render a single frame of the visualization.

        Returns:
            Dict[str, Any]: Rendered frame data and metadata
        """
        pass

    @abstractmethod
    def get_visualization_snapshot(self, format_type: str = "image") -> bytes:
        """
        Get a snapshot of the current visualization state.

        Args:
            format_type: Format of the snapshot ("image", "data", "json")

        Returns:
            bytes: Snapshot data in requested format
        """
        pass

    @abstractmethod
    def control_camera(self, camera_command: Dict[str, Any]) -> bool:
        """
        Control the camera/viewport for 3D visualization.

        Args:
            camera_command: Camera control command

        Returns:
            bool: True if camera control successful
        """
        pass

    @abstractmethod
    def add_visualization_effect(self, effect_config: Dict[str, Any]) -> str:
        """
        Add a visual effect to the visualization.

        Args:
            effect_config: Configuration for the visual effect

        Returns:
            str: Effect ID for the added effect
        """
        pass

    @abstractmethod
    def create_animation_sequence(self, sequence_config: Dict[str, Any]) -> str:
        """
        Create an animation sequence for visualization.

        Args:
            sequence_config: Configuration for the animation sequence

        Returns:
            str: Sequence ID for the created animation
        """
        pass

    @abstractmethod
    def get_visualization_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the visualization system performance.

        Returns:
            Dict[str, Any]: Visualization performance metrics
        """
        pass

    @abstractmethod
    def export_visualization_data(self, export_config: Dict[str, Any]) -> bool:
        """
        Export visualization data for external analysis.

        Args:
            export_config: Configuration for data export

        Returns:
            bool: True if export successful
        """
        pass