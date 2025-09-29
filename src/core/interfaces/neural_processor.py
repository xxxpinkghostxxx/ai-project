"""
INeuralProcessor interface - Neural dynamics processing service.

This interface defines the contract for neural processing operations,
handling membrane potentials, spiking behavior, and neural state updates
while maintaining biological plausibility and performance requirements.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from torch_geometric.data import Data


class NeuralState:
    """Represents the current neural state of the system."""

    def __init__(self):
        self.membrane_potentials: Dict[int, float] = {}
        self.thresholds: Dict[int, float] = {}
        self.refractory_periods: Dict[int, float] = {}
        self.last_spike_times: Dict[int, float] = {}
        self.neural_activities: Dict[int, float] = {}
        self.total_spikes: int = 0
        self.active_neurons: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert neural state to dictionary for serialization."""
        return {
            'membrane_potentials': self.membrane_potentials.copy(),
            'thresholds': self.thresholds.copy(),
            'refractory_periods': self.refractory_periods.copy(),
            'last_spike_times': self.last_spike_times.copy(),
            'neural_activities': self.neural_activities.copy(),
            'total_spikes': self.total_spikes,
            'active_neurons': self.active_neurons
        }


class SpikeEvent:
    """Represents a neural spike event."""

    def __init__(self, neuron_id: int, timestamp: float, membrane_potential: float):
        self.neuron_id = neuron_id
        self.timestamp = timestamp
        self.membrane_potential = membrane_potential
        self.energy_consumed = 0.0
        self.propagation_targets: List[int] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert spike event to dictionary."""
        return {
            'neuron_id': self.neuron_id,
            'timestamp': self.timestamp,
            'membrane_potential': self.membrane_potential,
            'energy_consumed': self.energy_consumed,
            'propagation_targets': self.propagation_targets.copy()
        }


class INeuralProcessor(ABC):
    """
    Abstract interface for neural processing operations.

    This interface defines the contract for processing neural dynamics,
    including membrane potential updates, spike generation, and neural
    state management while maintaining biological accuracy.
    """

    @abstractmethod
    def initialize_neural_state(self, graph: Data) -> bool:
        """
        Initialize neural state for the given graph.

        Args:
            graph: PyTorch Geometric graph with neural node data

        Returns:
            bool: True if initialization successful, False otherwise
        """

    @abstractmethod
    def update_membrane_potentials(self, graph: Data, time_step: float) -> Data:
        """
        Update membrane potentials for all neurons based on synaptic inputs.

        This method implements biologically plausible membrane dynamics,
        including leak currents, synaptic integration, and refractory periods.

        Args:
            graph: Current neural graph
            time_step: Time step for the update (in seconds)

        Returns:
            Data: Updated graph with new membrane potentials
        """

    @abstractmethod
    def process_spike_generation(self, graph: Data, current_time: float) -> Tuple[
        Data, List[SpikeEvent]
    ]:
        """
        Process spike generation based on membrane potentials and thresholds.

        This method determines which neurons should spike based on their
        current membrane potentials and firing thresholds, implementing
        biologically accurate spiking behavior.

        Args:
            graph: Current neural graph
            current_time: Current simulation time

        Returns:
            Tuple[Data, List[SpikeEvent]]: Updated graph and list of spike events
        """

    @abstractmethod
    def update_refractory_periods(self, graph: Data, time_step: float) -> Data:
        """
        Update refractory periods for neurons that have recently spiked.

        Args:
            graph: Current neural graph
            time_step: Time step for the update

        Returns:
            Data: Updated graph with new refractory periods
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def get_neural_state(self) -> NeuralState:
        """
        Get the current neural state of the system.

        Returns:
            NeuralState: Current neural state information
        """

    @abstractmethod
    def reset_neural_state(self) -> bool:
        """
        Reset neural state to initial conditions.

        Returns:
            bool: True if reset successful, False otherwise
        """

    @abstractmethod
    def validate_neural_integrity(self, graph: Data) -> Dict[str, Any]:
        """
        Validate the integrity of neural state and connections.

        Args:
            graph: Neural graph to validate

        Returns:
            Dict[str, Any]: Validation results with any issues found
        """

    @abstractmethod
    def process_neural_dynamics(self, graph: Data) -> Tuple[Data, List[SpikeEvent]]:
        """
        Process all neural dynamics for a single simulation step.

        Args:
            graph: The current neural graph.

        Returns:
            A tuple containing the updated graph and a list of spike events.
        """

    @abstractmethod
    def get_neural_metrics(self) -> Dict[str, float]:
        """
        Get neural processing metrics and statistics.

        Returns:
            Dict[str, float]: Neural metrics including firing rates, etc.
        """

    @abstractmethod
    def configure_neural_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Configure neural processing parameters.

        Args:
            parameters: Dictionary of neural parameter updates

        Returns:
            bool: True if parameters updated successfully, False otherwise
        """

