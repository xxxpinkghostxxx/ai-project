"""
ILoadBalancer interface - Load balancing service for distributed neural simulation.

This interface defines the contract for load balancing across distributed nodes,
optimizing resource utilization and maintaining performance in multi-node setups.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class LoadMetrics:
    """Metrics for load balancing decisions."""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    task_count: int = 0
    energy_level: float = 1.0
    network_latency: float = 0.0
    task_completion_rate: float = 0.0

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
        raise NotImplementedError

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
        raise NotImplementedError

    @abstractmethod
    def rebalance_workload(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Rebalance workload across all nodes when imbalance exceeds threshold.

        Args:
            threshold: Load imbalance threshold (0.0-1.0)

        Returns:
            Dict[str, Any]: Rebalancing results
        """
        raise NotImplementedError

    @abstractmethod
    def predict_load_changes(self, time_window: int = 60) -> Dict[str, Any]:
        """
        Predict future load changes based on historical data.

        Args:
            time_window: Prediction time window in seconds

        Returns:
            Dict[str, Any]: Load prediction results
        """
        raise NotImplementedError

    @abstractmethod
    def get_load_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive load balancing statistics.

        Returns:
            Dict[str, Any]: Load balancing statistics
        """
        raise NotImplementedError






