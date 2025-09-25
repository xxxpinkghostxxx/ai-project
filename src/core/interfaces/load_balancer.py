"""
ILoadBalancer interface - Load balancing service for distributed neural simulation.

This interface defines the contract for load balancing across distributed nodes,
optimizing resource utilization and maintaining performance in multi-node setups.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class LoadMetrics:
    """Metrics for load balancing decisions."""

    def __init__(self):
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.task_count = 0
        self.energy_level = 1.0
        self.network_latency = 0.0
        self.task_completion_rate = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'task_count': self.task_count,
            'energy_level': self.energy_level,
            'network_latency': self.network_latency,
            'task_completion_rate': self.task_completion_rate
        }


class ILoadBalancer(ABC):
    """
    Abstract interface for load balancing in distributed neural simulation.

    This interface defines the contract for distributing workload across
    multiple nodes to optimize performance and resource utilization.
    """

    @abstractmethod
    def assess_node_load(self, node_id: str) -> LoadMetrics:
        """
        Assess the current load of a specific node.

        Args:
            node_id: ID of the node to assess

        Returns:
            LoadMetrics: Current load metrics for the node
        """
        pass

    @abstractmethod
    def calculate_optimal_distribution(self, tasks: List[Dict[str, Any]],
                                     nodes: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate optimal task distribution across nodes.

        Args:
            tasks: List of tasks to distribute
            nodes: List of available node IDs

        Returns:
            Dict[str, List[Dict[str, Any]]]: Task distribution mapping
        """
        pass

    @abstractmethod
    def rebalance_workload(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Rebalance workload across all nodes when imbalance exceeds threshold.

        Args:
            threshold: Load imbalance threshold (0.0-1.0)

        Returns:
            Dict[str, Any]: Rebalancing results
        """
        pass

    @abstractmethod
    def predict_load_changes(self, time_window: int = 60) -> Dict[str, Any]:
        """
        Predict future load changes based on historical data.

        Args:
            time_window: Prediction time window in seconds

        Returns:
            Dict[str, Any]: Load prediction results
        """
        pass

    @abstractmethod
    def get_load_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive load balancing statistics.

        Returns:
            Dict[str, Any]: Load balancing statistics
        """
        pass






