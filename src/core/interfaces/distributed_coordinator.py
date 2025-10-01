"""
IDistributedCoordinator interface - Distributed coordination for multi-node neural simulation.

This interface defines the contract for coordinating neural simulation across multiple
nodes, providing load balancing, fault tolerance, and distributed processing capabilities
while maintaining energy-based coordination and biological plausibility.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from torch_geometric.data import Data


@dataclass
class NodeInfo:
    """Information about a distributed node."""

    node_id: str
    address: str
    capabilities: Dict[str, Any]
    status: str = "unknown"  # "active", "inactive", "failed"
    last_heartbeat: float = 0.0
    workload: float = 0.0
    energy_level: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert node info to dictionary."""
        return {
            'node_id': self.node_id,
            'address': self.address,
            'capabilities': self.capabilities.copy(),
            'status': self.status,
            'last_heartbeat': self.last_heartbeat,
            'workload': self.workload,
            'energy_level': self.energy_level
        }


@dataclass
class DistributedTask:
    """A task that can be distributed across nodes."""

    task_id: str
    task_type: str  # "neural_processing", "learning", "sensory", "energy"
    data: Any
    priority: int = 1
    assigned_node: Optional[str] = None
    status: str = "pending"  # "pending", "running", "completed", "failed"
    created_time: float = 0.0
    completed_time: float = 0.0
    result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'data': self.data,
            'priority': self.priority,
            'assigned_node': self.assigned_node,
            'status': self.status,
            'created_time': self.created_time,
            'completed_time': self.completed_time,
            'result': self.result
        }


class IDistributedCoordinator(ABC):
    """
    Abstract interface for distributed coordination of neural simulation.

    This interface defines the contract for coordinating neural simulation activities
    across multiple nodes, providing load balancing, fault tolerance, and distributed
    processing while maintaining energy-based coordination.
    """

    @abstractmethod
    def initialize_distributed_system(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the distributed neural simulation system.

        Args:
            config: Configuration for distributed system setup

        Returns:
            bool: True if initialization successful
        """

    @abstractmethod
    def register_node(self, node_info: NodeInfo) -> bool:
        """
        Register a new node in the distributed system.

        Args:
            node_info: Information about the node to register

        Returns:
            bool: True if registration successful
        """

    @abstractmethod
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the distributed system.

        Args:
            node_id: ID of the node to unregister

        Returns:
            bool: True if unregistration successful
        """

    @abstractmethod
    def submit_task(self, task: DistributedTask) -> bool:
        """
        Submit a task for distributed processing.

        Args:
            task: The task to submit

        Returns:
            bool: True if task submitted successfully
        """

    @abstractmethod
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get the result of a completed task.

        Args:
            task_id: ID of the task

        Returns:
            Optional[Any]: Task result if available
        """

    @abstractmethod
    def balance_workload(self) -> Dict[str, Any]:
        """
        Balance workload across all active nodes.

        Returns:
            Dict[str, Any]: Workload balancing results
        """

    @abstractmethod
    def handle_node_failure(self, node_id: str) -> bool:
        """
        Handle failure of a distributed node.

        Args:
            node_id: ID of the failed node

        Returns:
            bool: True if failure handled successfully
        """

    @abstractmethod
    def synchronize_state(self, graph: Data) -> bool:
        """
        Synchronize neural graph state across all nodes.

        Args:
            graph: Current neural graph state

        Returns:
            bool: True if synchronization successful
        """

    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the distributed system.

        Returns:
            Dict[str, Any]: System status information
        """

    @abstractmethod
    def optimize_energy_distribution(self) -> Dict[str, Any]:
        """
        Optimize energy distribution across distributed nodes.

        Returns:
            Dict[str, Any]: Energy optimization results
        """

    @abstractmethod
    def migrate_task(self, task_id: str, target_node_id: str) -> bool:
        """
        Migrate a running task to a different node.

        Args:
            task_id: ID of the task to migrate
            target_node_id: ID of the target node

        Returns:
            bool: True if migration successful
        """

