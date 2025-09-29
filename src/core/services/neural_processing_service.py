"""
NeuralProcessingService implementation - Focused neural dynamics service.

This module provides the concrete implementation of INeuralProcessor,
handling neural dynamics, spiking mechanisms, and membrane potential updates
while maintaining clean separation from energy management and learning.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch_geometric.data import Data

from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.energy_manager import IEnergyManager
from ..interfaces.event_coordinator import IEventCoordinator
from ..interfaces.neural_processor import (INeuralProcessor, NeuralState,
                                           SpikeEvent)

# Import optimized synaptic calculator
try:
    from ...utils.cpp_extensions import create_synaptic_calculator
    _USE_OPTIMIZED_SYNAPTIC_CALCULATOR = True
except ImportError:
    _USE_OPTIMIZED_SYNAPTIC_CALCULATOR = False


class NeuralProcessingService(INeuralProcessor):
    """
    Concrete implementation of INeuralProcessor.

    This service handles all neural dynamics processing including membrane
    potential updates, spike generation, refractory periods, and neural
    state management while coordinating with energy management for
    biologically plausible behavior.
    """

    def __init__(self,
                 energy_manager: IEnergyManager,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator):
        """
        Initialize the NeuralProcessingService.

        Args:
            energy_manager: Service for energy state coordination
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
        """
        self.energy_manager = energy_manager
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator

        # Neural processing parameters
        self._neural_state = NeuralState()
        self._membrane_time_constant = 10.0  # ms
        self._threshold_potential = -50.0  # mV
        self._reset_potential = -80.0  # mV
        self._refractory_period = 2.0  # ms
        self._resting_potential = -70.0  # mV

        # Statistics tracking
        self._spike_count = 0
        self._last_spike_times: Dict[int, float] = {}
        self._refractory_nodes: Dict[int, float] = {}

        # Initialize optimized synaptic calculator if available
        self._synaptic_calculator = None
        if _USE_OPTIMIZED_SYNAPTIC_CALCULATOR:
            try:
                self._synaptic_calculator = create_synaptic_calculator(time_window=0.1)
                print("Using optimized synaptic calculator")
            except (ImportError, AttributeError, RuntimeError) as e:
                print(f"Failed to initialize optimized synaptic calculator: {e}")
                self._synaptic_calculator = None

    def get_neural_state_internal(self) -> NeuralState:
        """Get the internal neural state."""
        return self._neural_state

    def set_neural_state_internal(self, state: NeuralState) -> None:
        """Set the internal neural state."""
        self._neural_state = state

    def get_membrane_time_constant(self) -> float:
        """Get the membrane time constant."""
        return self._membrane_time_constant

    def get_threshold_potential(self) -> float:
        """Get the threshold potential."""
        return self._threshold_potential

    def get_reset_potential(self) -> float:
        """Get the reset potential."""
        return self._reset_potential

    def get_refractory_period(self) -> float:
        """Get the refractory period."""
        return self._refractory_period

    def get_resting_potential(self) -> float:
        """Get the resting potential."""
        return self._resting_potential

    def get_spike_count(self) -> int:
        """Get the spike count."""
        return self._spike_count

    def increment_spike_count(self) -> None:
        """Increment the spike count."""
        self._spike_count += 1

    def get_last_spike_times(self) -> Dict[int, float]:
        """Get the last spike times dictionary."""
        return self._last_spike_times

    def set_last_spike_time(self, node_id: int, timestamp: float) -> None:
        """Set the last spike time for a node."""
        self._last_spike_times[node_id] = timestamp

    def get_refractory_nodes(self) -> Dict[int, float]:
        """Get the refractory nodes dictionary."""
        return self._refractory_nodes

    def set_refractory_node(self, node_id: int, timestamp: float) -> None:
        """Set a node as refractory for a given time."""
        self._refractory_nodes[node_id] = timestamp

    def remove_refractory_node(self, node_id: int) -> None:
        """Remove a node from refractory state."""
        if node_id in self._refractory_nodes:
            del self._refractory_nodes[node_id]

    def update_refractory_node(self, node_id: int, time_step: float) -> None:
        """Update refractory time for a node."""
        if node_id in self._refractory_nodes:
            self._refractory_nodes[node_id] -= time_step
            if self._refractory_nodes[node_id] <= 0:
                del self._refractory_nodes[node_id]

    def get_synaptic_calculator(self):
        """Get the synaptic calculator."""
        return self._synaptic_calculator

    def calculate_synaptic_inputs_internal(self, graph: Data,
                                           node_energies: Dict[int, float]) -> np.ndarray:
        """Calculate synaptic inputs for all neurons (internal method)."""
        return self._calculate_all_synaptic_inputs(graph, node_energies)

    def calculate_synaptic_input_internal(self, graph: Data, node_id: int) -> float:
        """Calculate synaptic input for a neuron (internal method)."""
        return self._calculate_synaptic_input(graph, node_id)

    def initialize_neural_state(self, graph: Data) -> bool:
        """
        Initialize neural state for the given graph.

        Args:
            graph: Neural graph to initialize

        Returns:
            bool: True if initialization successful
        """
        try:
            if graph is None:
                return False

            # Initialize membrane potentials if not present
            if not hasattr(graph, 'x') or graph.x is None:
                return False

            # Ensure we have node labels
            if not hasattr(graph, 'node_labels') or not graph.node_labels:
                return False

            # Reset statistics
            self._spike_count = 0
            self._last_spike_times.clear()
            self._refractory_nodes.clear()

            # Initialize neural state
            current_state = self.get_neural_state_internal()
            current_state.is_initialized = True
            current_state.total_neurons = len(graph.node_labels)
            current_state.active_neurons = 0

            return True

        except (AttributeError, TypeError, ValueError) as e:
            print(f"Failed to initialize neural state: {e}")
            return False

    def update_membrane_potentials(self, graph: Data, time_step: float) -> Data:
        """
        Update membrane potentials for all neurons.

        Args:
            graph: Neural graph
            time_step: Time step for integration

        Returns:
            Updated neural graph
        """
        if graph is None or not hasattr(graph, 'x') or graph.x is None:
            return graph

        try:
            # Get current energy state for modulation
            energy_state = self.energy_manager.get_energy_state()
            node_energies = (energy_state.node_energies if
                           hasattr(energy_state, 'node_energies') else {})

            # Calculate synaptic inputs for all nodes at once (optimized)
            synaptic_inputs = self.calculate_synaptic_inputs_internal(graph, node_energies)

            # Update membrane potentials
            for i, node in enumerate(graph.node_labels):
                if i >= len(graph.x):
                    continue

                node_id = node.get('id', i)
                current_potential = node.get('membrane_potential', self.get_resting_potential())

                # Skip refractory neurons
                if node_id in self.get_refractory_nodes():
                    self.update_refractory_node(node_id, time_step)
                    continue

                # Get synaptic input for this node
                synaptic_input = synaptic_inputs[i] if i < len(synaptic_inputs) else 0.0

                # Energy-modulated membrane dynamics
                energy_level = node_energies.get(node_id, 1.0)
                effective_time_constant = (self.get_membrane_time_constant() *
                                         (0.5 + 0.5 * energy_level))

                # Update membrane potential
                delta_v = ((synaptic_input - current_potential + self._resting_potential) *
                          (time_step / effective_time_constant))
                new_potential = current_potential + delta_v

                # Clamp to reasonable range
                new_potential = max(-100.0, min(50.0, new_potential))

                # Update node
                node['membrane_potential'] = new_potential
                graph.x[i, 0] = new_potential  # Update tensor

            return graph

        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
            print(f"Error updating membrane potentials: {e}")
            return graph

    def process_spike_generation(self, graph: Data,
                                 current_time: float) -> Tuple[Data, List[SpikeEvent]]:
        """
        Process spike generation based on membrane potentials.

        Args:
            graph: Neural graph
            current_time: Current simulation time

        Returns:
            Tuple of (updated_graph, spike_events)
        """
        spike_events = []

        if graph is None or not hasattr(graph, 'node_labels'):
            return graph, spike_events

        try:
            active_neurons = 0

            for i, node in enumerate(graph.node_labels):
                if i >= len(graph.x):
                    continue

                node_id = node.get('id', i)
                membrane_potential = node.get('membrane_potential', self.get_resting_potential())

                # Check for spike
                if (membrane_potential >= self.get_threshold_potential() and
                    node_id not in self.get_refractory_nodes()):

                    # Generate spike
                    spike_event = SpikeEvent(
                        neuron_id=node_id,
                        timestamp=current_time,
                        membrane_potential=membrane_potential
                    )
                    spike_events.append(spike_event)

                    # Reset membrane potential
                    node['membrane_potential'] = self.get_reset_potential()
                    graph.x[i, 0] = self.get_reset_potential()

                    # Set refractory period
                    self.set_refractory_node(node_id, self.get_refractory_period())
                    self.set_last_spike_time(node_id, current_time)

                    # Update statistics
                    self.increment_spike_count()
                    active_neurons += 1

                    # Publish spike event
                    self.event_coordinator.publish("neuron_spiked", {
                        "neuron_id": node_id,
                        "timestamp": current_time,
                        "membrane_potential": membrane_potential
                    })

            # Update neural state
            current_state = self.get_neural_state_internal()
            current_state.active_neurons = active_neurons
            current_state.total_spikes = self.get_spike_count()

            return graph, spike_events

        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
            print(f"Error processing spike generation: {e}")
            return graph, spike_events

    def update_refractory_periods(self, graph: Data, time_step: float) -> Data:
        """
        Update refractory periods for neurons.

        Args:
            graph: Neural graph
            time_step: Time step for updates

        Returns:
            Updated neural graph
        """
        if graph is None:
            return graph

        try:
            # Refractory period updates are handled in update_membrane_potentials
            # This method is here for interface compliance
            return graph

        except (AttributeError, TypeError) as e:
            print(f"Error updating refractory periods: {e}")
            return graph

    def reset_neural_state(self) -> bool:
        """
        Reset neural state to initial conditions.

        Returns:
            bool: True if reset successful
        """
        try:
            self._spike_count = 0
            self._last_spike_times.clear()
            self._refractory_nodes.clear()
            self.set_neural_state_internal(NeuralState())
            return True
        except (AttributeError, TypeError) as e:
            print(f"Error resetting neural state: {e}")
            return False

    def get_neural_state(self) -> NeuralState:
        """Get current neural state."""
        return self._neural_state

    def get_neural_statistics(self) -> Dict[str, Any]:
        """Get neural processing statistics."""
        current_state = self.get_neural_state_internal()
        return {
            "total_spikes": self.get_spike_count(),
            "active_neurons": current_state.active_neurons,
            "total_neurons": current_state.total_neurons,
            "refractory_neurons": len(self.get_refractory_nodes()),
            "spike_rate": self.get_spike_count() / max(1, len(self.get_last_spike_times()))
        }

    def validate_neural_integrity(self, graph: Optional[Data]) -> Dict[str, Any]:
        """
        Validate neural state integrity.

        Args:
            graph: Neural graph to validate

        Returns:
            Dict with validation results
        """
        issues = []

        if graph is None:
            issues.append("Neural graph is None")
            return {"valid": False, "issues": issues}

        if not hasattr(graph, 'node_labels') or not graph.node_labels:
            issues.append("Missing or empty node_labels")

        if not hasattr(graph, 'x') or graph.x is None:
            issues.append("Missing or empty node features tensor")

        # Check for invalid membrane potentials
        if hasattr(graph, 'x') and graph.x is not None:
            potentials = graph.x[:, 0].numpy()
            invalid_count = np.sum((potentials < -200) | (potentials > 100))
            if invalid_count > 0:
                issues.append(f"Found {invalid_count} neurons with invalid membrane potentials")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_neurons": len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
        }

    def _calculate_all_synaptic_inputs(self, graph: Data,
                                       node_energies: Dict[int, float]) -> np.ndarray:
        """
        Calculate synaptic inputs for all neurons using optimized calculator.

        Args:
            graph: Neural graph
            node_energies: Dictionary of node energy levels

        Returns:
            numpy array of synaptic inputs for all nodes
        """
        if graph is None or not hasattr(graph, 'node_labels'):
            return np.array([])

        num_nodes = len(graph.node_labels)

        # Use optimized synaptic calculator if available
        if self.get_synaptic_calculator() is not None and hasattr(graph, 'edge_attributes'):
            try:
                return self.get_synaptic_calculator().calculate_synaptic_inputs(
                    graph.edge_attributes,
                    node_energies,
                    num_nodes=num_nodes
                )
            except (AttributeError, TypeError, ValueError) as e:
                print(f"Optimized synaptic calculation failed, using fallback: {e}")

        # Fallback: calculate inputs individually
        synaptic_inputs = np.zeros(num_nodes, dtype=np.float64)
        for i in range(num_nodes):
            synaptic_inputs[i] = self._calculate_synaptic_input(graph, i)

        return synaptic_inputs

    def _calculate_synaptic_input(self, graph: Data, node_id: int) -> float:  # pylint: disable=unused-argument
        """
        Calculate synaptic input for a neuron (fallback implementation).

        Args:
            graph: Neural graph containing connectivity information
            node_id: ID of the neuron to calculate inputs for

        Returns:
            float: Total synaptic input
        """
        # This is a simplified implementation
        # In a full implementation, this would calculate weighted inputs from connections
        return np.random.normal(0, 0.1)  # Random noise for demonstration

    def apply_neural_dynamics(self, graph: Data, time_step: float) -> Tuple[Data, List[SpikeEvent]]:
        """
        Apply complete neural dynamics processing for one time step.

        This is the main method that coordinates all neural processing:
        membrane potential updates, spike generation, and refractory period updates.

        Args:
            graph: Current neural graph
            time_step: Time step for the processing

        Returns:
            Tuple[Data, List[SpikeEvent]]: Updated graph and spike events
        """
        # Update membrane potentials
        updated_graph = self.update_membrane_potentials(graph, time_step)

        # Process spike generation
        final_graph, spike_events = self.process_spike_generation(updated_graph, time.time())

        # Update refractory periods
        final_graph = self.update_refractory_periods(final_graph, time_step)

        return final_graph, spike_events

    def calculate_synaptic_inputs(self, graph: Data, neuron_id: int) -> float:
        """
        Calculate total synaptic input for a specific neuron.

        This method computes the combined effect of all synaptic connections
        to the specified neuron, including excitatory and inhibitory inputs.

        Args:
            graph: Current neural graph
            neuron_id: ID of the neuron to calculate inputs for

        Returns:
            float: Total synaptic input current
        """
        if graph is None or not hasattr(graph, 'node_labels'):
            return 0.0

        return self.calculate_synaptic_input_internal(graph, neuron_id)

    def configure_neural_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Configure neural processing parameters.

        Args:
            parameters: Dictionary of neural parameter updates

        Returns:
            bool: True if parameters updated successfully
        """
        try:
            # Note: For configuration parameters, we would need setter methods
            # For now, we'll keep the direct access but this should be refactored
            # to use proper setter methods in a future iteration
            for key, value in parameters.items():
                if key == 'membrane_time_constant':
                    self._membrane_time_constant = float(value)
                elif key == 'threshold_potential':
                    self._threshold_potential = float(value)
                elif key == 'reset_potential':
                    self._reset_potential = float(value)
                elif key == 'refractory_period':
                    self._refractory_period = float(value)
                elif key == 'resting_potential':
                    self._resting_potential = float(value)

            return True
        except (ValueError, TypeError) as e:
            print(f"Error configuring neural parameters: {e}")
            return False

    def get_neural_metrics(self) -> Dict[str, float]:
        """
        Get neural processing metrics and statistics.

        Returns:
            Dict[str, float]: Neural metrics
        """
        current_state = self.get_neural_state_internal()
        return {
            "total_spikes": self.get_spike_count(),
            "active_neurons": current_state.active_neurons,
            "total_neurons": current_state.total_neurons,
            "refractory_neurons": len(self.get_refractory_nodes()),
            "spike_rate": self.get_spike_count() / max(1, len(self.get_last_spike_times())),
            "average_membrane_potential": 0.0  # Would need to calculate from graph
        }

    def process_neural_dynamics(self, graph: Data) -> Tuple[Data, List[SpikeEvent]]:
        """
        Process neural dynamics for a single time step.

        Args:
            graph: Neural graph

        Returns:
            Tuple of (updated_graph, spike_events)
        """
        return self.apply_neural_dynamics(graph, 0.001) # Assuming 1ms timestep for now
def cleanup(self) -> None:
    """Clean up resources."""
    self.get_last_spike_times().clear()
    self.get_refractory_nodes().clear()
