"""
IGPUAccelerator interface - GPU acceleration service for neural simulation.

This interface defines the contract for GPU-accelerated neural processing,
providing high-performance computing capabilities for neural dynamics,
learning, and energy computations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from torch_geometric.data import Data
import torch


class GPUComputeTask:
    """Represents a GPU-accelerated compute task."""

    def __init__(self, task_id: str, task_type: str, data: Any, priority: int = 1):
        self.task_id = task_id
        self.task_type = task_type  # "neural_dynamics", "learning", "energy", "sensory"
        self.data = data
        self.priority = priority
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory_required = 0  # MB
        self.compute_intensity = 1.0  # Relative compute intensity

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'data': self.data,
            'priority': self.priority,
            'device': self.device,
            'memory_required': self.memory_required,
            'compute_intensity': self.compute_intensity
        }


class IGPUAccelerator(ABC):
    """
    Abstract interface for GPU-accelerated neural simulation processing.

    This interface defines the contract for leveraging GPU computing resources
    to accelerate neural dynamics, learning algorithms, and energy computations.
    """

    @abstractmethod
    def initialize_gpu_resources(self, config: Dict[str, Any]) -> bool:
        """
        Initialize GPU computing resources and memory pools.

        Args:
            config: GPU configuration parameters

        Returns:
            bool: True if GPU resources initialized successfully
        """
        raise NotImplementedError()

    @abstractmethod
    def submit_gpu_task(self, task: GPUComputeTask) -> bool:
        """
        Submit a compute task for GPU acceleration.

        Args:
            task: The GPU compute task to execute

        Returns:
            bool: True if task submitted successfully
        """
        raise NotImplementedError()

    @abstractmethod
    def get_gpu_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get the result of a completed GPU task.

        Args:
            task_id: ID of the GPU task

        Returns:
            Optional[Any]: Task result if available
        """
        raise NotImplementedError()

    @abstractmethod
    def accelerate_neural_dynamics(self, graph: Data, time_step: int) -> Data:
        """
        Accelerate neural dynamics computation using GPU.

        Args:
            graph: Neural graph for dynamics computation
            time_step: Current simulation time step

        Returns:
            Data: Updated graph with computed neural dynamics
        """
        raise NotImplementedError()

    @abstractmethod
    def accelerate_learning(self, graph: Data, learning_data: Dict[str, Any]) -> Data:
        """
        Accelerate learning algorithms using GPU.

        Args:
            graph: Neural graph for learning
            learning_data: Learning parameters and data

        Returns:
            Data: Updated graph with applied learning
        """
        raise NotImplementedError()

    @abstractmethod
    def accelerate_energy_computation(self, graph: Data) -> Data:
        """
        Accelerate energy flow computations using GPU.

        Args:
            graph: Neural graph for energy computation

        Returns:
            Data: Updated graph with computed energy flows
        """
        raise NotImplementedError()

    @abstractmethod
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU memory usage information.

        Returns:
            Dict[str, Any]: GPU memory statistics and usage
        """
        raise NotImplementedError()

    @abstractmethod
    def get_gpu_performance_metrics(self) -> Dict[str, float]:
        """
        Get GPU performance metrics and utilization statistics.

        Returns:
            Dict[str, float]: GPU performance metrics
        """
        raise NotImplementedError()

    @abstractmethod
    def optimize_gpu_memory(self) -> Dict[str, Any]:
        """
        Optimize GPU memory usage and perform garbage collection.

        Returns:
            Dict[str, Any]: Memory optimization results
        """
        raise NotImplementedError()

    @abstractmethod
    def synchronize_gpu_operations(self) -> bool:
        """
        Synchronize all pending GPU operations.

        Returns:
            bool: True if synchronization successful
        """
        raise NotImplementedError()
