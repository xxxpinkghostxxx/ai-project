"""
IFaultTolerance interface - Fault tolerance service for distributed neural simulation.

This interface defines the contract for handling failures and maintaining
system reliability in distributed neural simulation environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FailureEvent:
    """Represents a system failure event."""

    failure_type: str  # "node_failure", "network_failure", "service_failure"
    affected_component: str
    severity: str = "medium"  # "low", "medium", "high", "critical"
    timestamp: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert failure event to dictionary."""
        return {
            'failure_type': self.failure_type,
            'affected_component': self.affected_component,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'details': self.details.copy(),
            'recovery_actions': self.recovery_actions.copy()
        }


class IFaultTolerance(ABC):
    """
    Abstract interface for fault tolerance in distributed neural simulation.

    This interface defines the contract for detecting, handling, and recovering
    from failures in distributed neural simulation systems.
    """

    @abstractmethod
    def detect_failures(self) -> List[FailureEvent]:
        """
        Detect system failures across all components.

        Returns:
            List[FailureEvent]: List of detected failures
        """

    @abstractmethod
    def handle_node_failure(self, node_id: str) -> Dict[str, Any]:
        """
        Handle failure of a specific node.

        Args:
            node_id: ID of the failed node

        Returns:
            Dict[str, Any]: Recovery actions taken
        """

    @abstractmethod
    def handle_service_failure(
        self, service_name: str, node_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle failure of a specific service.

        Args:
            service_name: Name of the failed service
            node_id: ID of the node where service failed (optional)

        Returns:
            Dict[str, Any]: Recovery actions taken
        """

    @abstractmethod
    def initiate_failover(self, primary_component: str, backup_component: str) -> bool:
        """
        Initiate failover from primary to backup component.

        Args:
            primary_component: Primary component that failed
            backup_component: Backup component to take over

        Returns:
            bool: True if failover successful
        """

    @abstractmethod
    def create_backup(self, component_id: str) -> Dict[str, Any]:
        """
        Create backup for a critical component.

        Args:
            component_id: ID of the component to backup

        Returns:
            Dict[str, Any]: Backup creation results
        """

    @abstractmethod
    def validate_system_integrity(self) -> Dict[str, Any]:
        """
        Validate overall system integrity and identify potential issues.

        Returns:
            Dict[str, Any]: System integrity assessment
        """

    @abstractmethod
    def get_failure_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive failure statistics and trends.

        Returns:
            Dict[str, Any]: Failure statistics and analysis
        """
