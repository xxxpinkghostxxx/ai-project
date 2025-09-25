"""
ISimulationCoordinator interface - Core coordination service for neural simulation.

This interface defines the main coordination contract for the neural simulation system,
providing the primary entry point for simulation execution while maintaining clean
separation of concerns and enabling dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from torch_geometric.data import Data


class SimulationState:
    """Represents the current state of the neural simulation."""

    def __init__(self):
        self.step_count: int = 0
        self.is_running: bool = False
        self.last_step_time: float = 0.0
        self.total_energy: float = 0.0
        self.active_neurons: int = 0
        self.learning_events: int = 0
        self.performance_metrics: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'step_count': self.step_count,
            'is_running': self.is_running,
            'last_step_time': self.last_step_time,
            'total_energy': self.total_energy,
            'active_neurons': self.active_neurons,
            'learning_events': self.learning_events,
            'performance_metrics': self.performance_metrics.copy()
        }


class ISimulationCoordinator(ABC):
    """
    Abstract interface for the main simulation coordinator.

    This interface defines the primary contract for coordinating all neural simulation
    activities while maintaining biological plausibility and performance requirements.
    """

    @abstractmethod
    def initialize_simulation(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the neural simulation with the provided configuration.

        Args:
            config: Optional configuration dictionary for simulation parameters

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def start_simulation(self) -> bool:
        """
        Start the neural simulation execution.

        Returns:
            bool: True if simulation started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop_simulation(self) -> bool:
        """
        Stop the neural simulation execution.

        Returns:
            bool: True if simulation stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def reset_simulation(self) -> bool:
        """
        Reset the neural simulation to initial state.

        Returns:
            bool: True if reset successful, False otherwise
        """
        pass

    @abstractmethod
    def execute_simulation_step(self, step: int) -> bool:
        """
        Execute a single simulation step with energy-based coordination.

        This method coordinates all neural simulation activities for one time step,
        ensuring energy flows properly through neural, learning, and sensory systems
        while maintaining biological plausibility and performance requirements.

        Args:
            step: The current simulation step number

        Returns:
            bool: True if step executed successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_simulation_state(self) -> SimulationState:
        """
        Get the current state of the neural simulation.

        Returns:
            SimulationState: Current simulation state including metrics and status
        """
        pass

    @abstractmethod
    def get_neural_graph(self) -> Optional[Data]:
        """
        Get the current neural graph with all node and connection data.

        Returns:
            Optional[Data]: Current PyTorch Geometric graph, None if not available
        """
        pass

    @abstractmethod
    def update_configuration(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update simulation configuration parameters dynamically.

        Args:
            config_updates: Dictionary of configuration parameter updates

        Returns:
            bool: True if configuration updated successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive performance metrics for the simulation.

        Returns:
            Dict[str, float]: Performance metrics including step times, memory usage, etc.
        """
        pass

    @abstractmethod
    def validate_simulation_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the neural simulation state.

        Returns:
            Dict[str, Any]: Validation results with any issues found
        """
        pass

    @abstractmethod
    def save_simulation_state(self, filepath: str) -> bool:
        """
        Save the current simulation state to file.

        Args:
            filepath: Path to save the simulation state

        Returns:
            bool: True if state saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_simulation_state(self, filepath: str) -> bool:
        """
        Load simulation state from file.

        Args:
            filepath: Path to load the simulation state from

        Returns:
            bool: True if state loaded successfully, False otherwise
        """
        pass