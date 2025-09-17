"""
Sensory integration components.

This package contains sensory processing components including:
- Audio and visual processing bridges
- Sensory workspace mapping
"""

from .audio_to_neural_bridge import *
from .visual_energy_bridge import *
from .sensory_workspace_mapper import *

__all__ = [
    'AudioToNeuralBridge',
    'VisualEnergyBridge', 
    'SensoryWorkspaceMapper'
]
