"""
SensoryProcessingService implementation - Sensory input processing service.

This module provides the concrete implementation of ISensoryProcessor,
handling the conversion of external sensory stimuli into neural activation
patterns while maintaining biological plausibility and energy integration.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from torch_geometric.data import Data

from ..interfaces.sensory_processor import ISensoryProcessor, SensoryInput
from ..interfaces.energy_manager import IEnergyManager
from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.event_coordinator import IEventCoordinator


class SensoryProcessingService(ISensoryProcessor):
    """
    Concrete implementation of ISensoryProcessor.

    This service handles the processing of sensory inputs from various modalities
    (visual, auditory, etc.) and converts them into biologically plausible
    neural activation patterns with energy-based processing.
    """

    def __init__(self,
                 energy_manager: IEnergyManager,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator):
        """
        Initialize the SensoryProcessingService.

        Args:
            energy_manager: Service for energy state coordination
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
        """
        self.energy_manager = energy_manager
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator

        # Sensory processing parameters
        self._sensory_buffer_size = 1000
        self._adaptation_rate = 0.95
        self._processing_delay = 0.01

        # Sensory state tracking
        self._sensory_buffer: List[SensoryInput] = []
        self._adaptation_levels: Dict[str, float] = {}
        self._sensory_pathways: Dict[str, List[int]] = {}  # modality -> node_ids

        # Processing statistics
        self._processed_inputs = 0
        self._activation_patterns_generated = 0

    def process_sensory_input(self, graph: Data) -> Data:
        """
        Process sensory inputs and update the neural graph.

        Args:
            graph: The current neural graph.

        Returns:
            The updated neural graph with sensory data integrated.
        """
        try:
            # For now, create a simple sensory input for testing
            # In a full implementation, this would get sensory inputs from a queue or buffer
            sensory_input = SensoryInput(
                modality="visual",
                data={"test": "data"},
                intensity=0.5,
                spatial_location=(0.5, 0.5)
            )
            sensory_input.timestamp = time.time()

            # Apply sensory adaptation
            adapted_intensity = self._apply_adaptation(sensory_input)

            # Generate neural activation pattern
            activation_pattern = self._generate_activation_pattern(sensory_input, adapted_intensity)

            # Apply activations to graph
            self._apply_activations_to_graph(graph, activation_pattern)

            # Update sensory buffer
            self._sensory_buffer.append(sensory_input)
            if len(self._sensory_buffer) > self._sensory_buffer_size:
                self._sensory_buffer.pop(0)

            # Update statistics
            self._processed_inputs += 1
            self._activation_patterns_generated += len(activation_pattern.get('activations', []))

            # Publish sensory processing event
            self.event_coordinator.publish("sensory_input_processed", {
                "modality": sensory_input.modality,
                "intensity": sensory_input.intensity,
                "adapted_intensity": adapted_intensity,
                "activations_generated": len(activation_pattern.get('activations', [])),
                "timestamp": sensory_input.timestamp
            })

            return graph

        except Exception as e:
            print(f"Error processing sensory input: {e}")
            return graph

    def initialize_sensory_pathways(self, graph: Data) -> bool:
        """
        Initialize sensory processing pathways in the neural graph.

        Args:
            graph: Neural graph to initialize sensory pathways in

        Returns:
            bool: True if initialization successful
        """
        try:
            if graph is None or not hasattr(graph, 'node_labels'):
                return False

            # Initialize sensory pathways for different modalities
            modalities = ["visual", "auditory", "tactile", "proprioceptive"]

            for modality in modalities:
                # Create sensory nodes for this modality
                sensory_nodes = self._create_sensory_nodes(graph, modality, 10)  # 10 nodes per modality
                self._sensory_pathways[modality] = sensory_nodes

                # Initialize adaptation levels
                self._adaptation_levels[modality] = 1.0

            # Create connections from sensory to processing nodes
            self._create_sensory_connections(graph)

            return True

        except Exception as e:
            print(f"Error initializing sensory pathways: {e}")
            return False

    def apply_sensory_adaptation(self, graph: Data, adaptation_rate: float) -> Data:
        """
        Apply sensory adaptation to prevent overstimulation.

        Args:
            graph: Current neural graph
            adaptation_rate: Rate of sensory adaptation

        Returns:
            Data: Updated graph with sensory adaptation applied
        """
        if graph is None:
            return graph

        try:
            # Update adaptation levels for all modalities
            for modality in self._adaptation_levels:
                # Decay adaptation level toward baseline
                self._adaptation_levels[modality] *= adaptation_rate
                # Ensure minimum adaptation level
                self._adaptation_levels[modality] = max(0.1, self._adaptation_levels[modality])

            return graph

        except Exception as e:
            print(f"Error applying sensory adaptation: {e}")
            return graph

    def get_sensory_state(self) -> Dict[str, Any]:
        """
        Get the current state of sensory processing.

        Returns:
            Dict[str, Any]: Current sensory processing state
        """
        return {
            "processed_inputs": self._processed_inputs,
            "activation_patterns_generated": self._activation_patterns_generated,
            "buffer_size": len(self._sensory_buffer),
            "adaptation_levels": self._adaptation_levels.copy(),
            "sensory_pathways": {modality: len(nodes) for modality, nodes in self._sensory_pathways.items()},
            "buffer_utilization": len(self._sensory_buffer) / self._sensory_buffer_size if self._sensory_buffer_size > 0 else 0
        }

    def _apply_adaptation(self, sensory_input: SensoryInput) -> float:
        """
        Apply sensory adaptation to input intensity.

        Args:
            sensory_input: The sensory input to adapt

        Returns:
            float: Adapted intensity
        """
        modality = sensory_input.modality
        original_intensity = sensory_input.intensity

        # Get current adaptation level
        adaptation_level = self._adaptation_levels.get(modality, 1.0)

        # Apply adaptation
        adapted_intensity = original_intensity * adaptation_level

        # Update adaptation level based on stimulus intensity
        adaptation_change = original_intensity * 0.1  # Adaptation increases with stimulus strength
        self._adaptation_levels[modality] = min(2.0, adaptation_level + adaptation_change)

        return adapted_intensity

    def _generate_activation_pattern(self, sensory_input: SensoryInput, adapted_intensity: float) -> Dict[str, Any]:
        """
        Generate neural activation pattern from sensory input.

        Args:
            sensory_input: The sensory input
            adapted_intensity: Adapted intensity value

        Returns:
            Dict[str, Any]: Activation pattern
        """
        modality = sensory_input.modality
        sensory_nodes = self._sensory_pathways.get(modality, [])

        if not sensory_nodes:
            return {"activations": []}

        # Generate activation pattern based on modality
        if modality == "visual":
            activations = self._generate_visual_pattern(sensory_input, adapted_intensity, sensory_nodes)
        elif modality == "auditory":
            activations = self._generate_auditory_pattern(sensory_input, adapted_intensity, sensory_nodes)
        else:
            # Generic pattern for other modalities
            activations = self._generate_generic_pattern(sensory_input, adapted_intensity, sensory_nodes)

        return {
            "modality": modality,
            "activations": activations,
            "total_intensity": adapted_intensity,
            "nodes_activated": len(activations)
        }

    def _generate_visual_pattern(self, sensory_input: SensoryInput, intensity: float,
                               sensory_nodes: List[int]) -> List[Dict[str, Any]]:
        """
        Generate visual activation pattern.

        Args:
            sensory_input: Visual sensory input
            intensity: Adapted intensity
            sensory_nodes: Available sensory nodes

        Returns:
            List[Dict[str, Any]]: Activation pattern
        """
        activations = []

        # Simulate retinotopic organization
        if sensory_input.spatial_location:
            x, y = sensory_input.spatial_location
            # Map spatial location to node activation pattern
            for i, node_id in enumerate(sensory_nodes):
                # Distance-based activation
                node_x = (i % 5) * 0.2  # 5x2 grid
                node_y = (i // 5) * 0.5

                distance = np.sqrt((x - node_x)**2 + (y - node_y)**2)
                activation_strength = intensity * np.exp(-distance * 2.0)

                if activation_strength > 0.1:
                    activations.append({
                        "node_id": node_id,
                        "activation_strength": activation_strength,
                        "activation_type": "visual_spatial"
                    })

        return activations

    def _generate_auditory_pattern(self, sensory_input: SensoryInput, intensity: float,
                                 sensory_nodes: List[int]) -> List[Dict[str, Any]]:
        """
        Generate auditory activation pattern.

        Args:
            sensory_input: Auditory sensory input
            intensity: Adapted intensity
            sensory_nodes: Available sensory nodes

        Returns:
            List[Dict[str, Any]]: Activation pattern
        """
        activations = []

        # Simulate tonotopic organization
        if hasattr(sensory_input, 'data') and isinstance(sensory_input.data, dict):
            frequency = sensory_input.data.get('frequency', 1000)  # Hz

            for i, node_id in enumerate(sensory_nodes):
                # Map frequency to node preference
                preferred_freq = 200 * (2 ** (i / 3.0))  # Logarithmic frequency mapping
                tuning_width = preferred_freq * 0.5

                # Gaussian tuning curve
                frequency_distance = abs(np.log2(frequency) - np.log2(preferred_freq))
                activation_strength = intensity * np.exp(-frequency_distance**2 / (2 * 0.5**2))

                if activation_strength > 0.1:
                    activations.append({
                        "node_id": node_id,
                        "activation_strength": activation_strength,
                        "activation_type": "auditory_tonotopic"
                    })

        return activations

    def _generate_generic_pattern(self, sensory_input: SensoryInput, intensity: float,
                                sensory_nodes: List[int]) -> List[Dict[str, Any]]:
        """
        Generate generic activation pattern for other modalities.

        Args:
            sensory_input: Generic sensory input
            intensity: Adapted intensity
            sensory_nodes: Available sensory nodes

        Returns:
            List[Dict[str, Any]]: Activation pattern
        """
        activations = []

        # Simple intensity-based activation
        num_to_activate = min(len(sensory_nodes), max(1, int(intensity * len(sensory_nodes))))

        for i in range(num_to_activate):
            node_id = sensory_nodes[i]
            activation_strength = intensity * (1.0 - i / len(sensory_nodes))

            activations.append({
                "node_id": node_id,
                "activation_strength": activation_strength,
                "activation_type": "generic_intensity"
            })

        return activations

    def _create_sensory_nodes(self, graph: Data, modality: str, count: int) -> List[int]:
        """
        Create sensory nodes for a specific modality.

        Args:
            graph: Neural graph
            modality: Sensory modality
            count: Number of nodes to create

        Returns:
            List[int]: IDs of created sensory nodes
        """
        sensory_nodes = []

        for i in range(count):
            # Create node ID (simplified - in real implementation would use proper ID generation)
            node_id = f"sensory_{modality}_{i}"

            # Add to graph node labels
            if not hasattr(graph, 'node_labels'):
                graph.node_labels = []

            node_label = {
                'id': node_id,
                'type': 'sensory',
                'modality': modality,
                'energy': 0.5,
                'x': i * 10,
                'y': 0,
                'membrane_potential': 0.0,
                'threshold': 0.3,
                'behavior': 'sensory'
            }

            graph.node_labels.append(node_label)
            sensory_nodes.append(node_id)

        return sensory_nodes

    def _create_sensory_connections(self, graph: Data) -> None:
        """
        Create connections from sensory nodes to processing nodes.

        Args:
            graph: Neural graph
        """
        # This would create connections between sensory and processing nodes
        # Simplified implementation for now
        pass

    def _apply_activations_to_graph(self, graph: Data, activation_pattern: Dict[str, Any]) -> None:
        """
        Apply activation pattern to the neural graph.

        Args:
            graph: Neural graph to update
            activation_pattern: Activation pattern to apply
        """
        activations = activation_pattern.get('activations', [])
        for activation in activations:
            node_id = activation.get('node_id')
            strength = activation.get('activation_strength', 0.0)

            # Find node in graph and update its energy/membrane potential
            if hasattr(graph, 'node_labels'):
                for node in graph.node_labels:
                    if node.get('id') == node_id:
                        # Increase energy based on activation strength
                        current_energy = node.get('energy', 0.0)
                        node['energy'] = min(1.0, current_energy + strength * 0.1)
                        break

    def cleanup(self) -> None:
        """Clean up resources."""
        self._sensory_buffer.clear()
        self._adaptation_levels.clear()
        self._sensory_pathways.clear()