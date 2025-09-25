"""
ISensoryProcessor interface - Sensory input processing service.

This interface defines the contract for processing sensory inputs,
converting external stimuli into neural activation patterns while
maintaining biological plausibility and energy-based processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from torch_geometric.data import Data


class SensoryInput:
    """Represents a sensory input stimulus."""

    def __init__(self, modality: str, data: Any, intensity: float = 1.0,
                 spatial_location: Optional[Tuple[float, float]] = None,
                 temporal_pattern: Optional[List[float]] = None):
        self.modality = modality  # "visual", "auditory", "tactile", etc.
        self.data = data  # Raw sensory data
        self.intensity = intensity
        self.timestamp = 0.0
        self.spatial_location = spatial_location
        self.temporal_pattern = temporal_pattern

    def to_dict(self) -> Dict[str, Any]:
        """Convert sensory input to dictionary."""
        return {
            'modality': self.modality,
            'data': self.data,
            'intensity': self.intensity,
            'timestamp': self.timestamp,
            'spatial_location': self.spatial_location,
            'temporal_pattern': self.temporal_pattern
        }


class ISensoryProcessor(ABC):
    """
    Abstract interface for sensory input processing.

    This interface defines the contract for converting external sensory
    stimuli into neural activation patterns, implementing biologically
    plausible sensory processing pathways.
    """

    @abstractmethod
    def process_sensory_input(self, graph: Data) -> Data:
        """
        Process sensory inputs and update the neural graph.

        Args:
            graph: The current neural graph.

        Returns:
            The updated neural graph with sensory data integrated.
        """
        pass

    @abstractmethod
    def initialize_sensory_pathways(self, graph: Data) -> bool:
        """
        Initialize sensory processing pathways in the neural graph.

        Args:
            graph: Neural graph to initialize sensory pathways in

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def apply_sensory_adaptation(self, graph: Data, adaptation_rate: float) -> Data:
        """
        Apply sensory adaptation to prevent overstimulation.

        Args:
            graph: Current neural graph
            adaptation_rate: Rate of sensory adaptation

        Returns:
            Data: Updated graph with sensory adaptation applied
        """
        pass

    @abstractmethod
    def get_sensory_state(self) -> Dict[str, Any]:
        """
        Get the current state of sensory processing.

        Returns:
            Dict[str, Any]: Current sensory processing state
        """
        pass