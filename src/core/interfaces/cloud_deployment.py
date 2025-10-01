"""
ICloudDeployment interface - Cloud deployment service for neural simulation.

This interface defines the contract for deploying neural simulation systems
to cloud platforms with containerization, orchestration, and scaling capabilities.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DeploymentConfig:
    """Represents a deployment configuration."""

    deployment_id: str
    platform: str  # "aws", "azure", "gcp", "kubernetes"
    region: str = "us-west-2"
    instance_type: str = "t3.medium"
    instance_count: int = 1
    auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10
    container_image: str = ""
    environment_variables: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    network_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: 0.0)  # Will be set by caller
    status: str = "pending"  # "pending", "deploying", "running", "failed", "stopped"

    def to_dict(self) -> Dict[str, Any]:
        """Convert deployment config to dictionary."""
        return {
            'deployment_id': self.deployment_id,
            'platform': self.platform,
            'region': self.region,
            'instance_type': self.instance_type,
            'instance_count': self.instance_count,
            'auto_scaling': self.auto_scaling,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'container_image': self.container_image,
            'environment_variables': self.environment_variables.copy(),
            'volumes': self.volumes.copy(),
            'network_config': self.network_config.copy(),
            'monitoring_config': self.monitoring_config.copy(),
            'created_at': self.created_at,
            'status': self.status
        }


@dataclass
class ScalingPolicy:
    """Represents an auto-scaling policy."""

    policy_id: str
    metric_name: str
    target_value: float = 70.0
    scale_out_threshold: float = 80.0
    scale_in_threshold: float = 30.0
    cooldown_period: int = 300  # seconds
    min_instances: int = 1
    max_instances: int = 10
    enabled: bool = True
    last_scale_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert scaling policy to dictionary."""
        return {
            'policy_id': self.policy_id,
            'metric_name': self.metric_name,
            'target_value': self.target_value,
            'scale_out_threshold': self.scale_out_threshold,
            'scale_in_threshold': self.scale_in_threshold,
            'cooldown_period': self.cooldown_period,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'enabled': self.enabled,
            'last_scale_time': self.last_scale_time
        }


@dataclass
class CloudResource:
    """Represents a cloud resource."""

    resource_id: str
    resource_type: str  # "instance", "load_balancer", "database", "storage"
    platform: str = ""
    region: str = ""
    status: str = "pending"  # "pending", "creating", "running", "failed", "terminated"
    public_ip: str = ""
    private_ip: str = ""
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    cost_per_hour: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert cloud resource to dictionary."""
        return {
            'resource_id': self.resource_id,
            'resource_type': self.resource_type,
            'platform': self.platform,
            'region': self.region,
            'status': self.status,
            'public_ip': self.public_ip,
            'private_ip': self.private_ip,
            'created_at': self.created_at,
            'cost_per_hour': self.cost_per_hour,
            'tags': self.tags.copy()
        }


class ICloudDeployment(ABC):
    """
    Abstract interface for cloud deployment in neural simulation.

    This interface defines the contract for deploying neural simulation systems
    to cloud platforms with containerization, auto-scaling, and multi-cloud support.
    """

    @abstractmethod
    def create_deployment(self, config: Dict[str, Any]) -> str:
        """
        Create a new cloud deployment.

        Args:
            config: Deployment configuration parameters

        Returns:
            str: Deployment ID for tracking
        """

    @abstractmethod
    def update_deployment(self, deployment_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing deployment.

        Args:
            deployment_id: ID of the deployment to update
            updates: Configuration updates

        Returns:
            bool: True if update successful
        """

    @abstractmethod
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get the status of a deployment.

        Args:
            deployment_id: ID of the deployment

        Returns:
            Dict[str, Any]: Deployment status and information
        """

    @abstractmethod
    def scale_deployment(self, deployment_id: str, instance_count: int) -> bool:
        """
        Scale a deployment to a specific number of instances.

        Args:
            deployment_id: ID of the deployment
            instance_count: Target number of instances

        Returns:
            bool: True if scaling successful
        """

    @abstractmethod
    def create_scaling_policy(self, deployment_id: str, policy_config: Dict[str, Any]) -> str:
        """
        Create an auto-scaling policy for a deployment.

        Args:
            deployment_id: ID of the deployment
            policy_config: Scaling policy configuration

        Returns:
            str: Policy ID for tracking
        """

    @abstractmethod
    def get_scaling_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get the scaling status and metrics for a deployment.

        Args:
            deployment_id: ID of the deployment

        Returns:
            Dict[str, Any]: Scaling status and metrics
        """

    @abstractmethod
    def deploy_to_multiple_clouds(self, deployment_config: Dict[str, Any]) -> List[str]:
        """
        Deploy to multiple cloud platforms simultaneously.

        Args:
            deployment_config: Multi-cloud deployment configuration

        Returns:
            List[str]: List of deployment IDs for each cloud
        """

    @abstractmethod
    def get_deployment_costs(
        self, deployment_id: str, time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Get cost information for a deployment.

        Args:
            deployment_id: ID of the deployment
            time_range: Optional time range for cost calculation

        Returns:
            Dict[str, Any]: Cost information and breakdown
        """

    @abstractmethod
    def backup_deployment(self, deployment_id: str, backup_config: Dict[str, Any]) -> str:
        """
        Create a backup of a deployment.

        Args:
            deployment_id: ID of the deployment
            backup_config: Backup configuration

        Returns:
            str: Backup ID for tracking
        """

    @abstractmethod
    def restore_deployment(self, deployment_id: str, backup_id: str) -> bool:
        """
        Restore a deployment from a backup.

        Args:
            deployment_id: ID of the deployment
            backup_id: ID of the backup

        Returns:
            bool: True if restore successful
        """

    @abstractmethod
    def terminate_deployment(self, deployment_id: str) -> bool:
        """
        Terminate a deployment and clean up resources.

        Args:
            deployment_id: ID of the deployment

        Returns:
            bool: True if termination successful
        """
