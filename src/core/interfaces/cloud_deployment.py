"""
ICloudDeployment interface - Cloud deployment service for neural simulation.

This interface defines the contract for deploying neural simulation systems
to cloud platforms with containerization, orchestration, and scaling capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class DeploymentConfig:
    """Represents a deployment configuration."""

    def __init__(self, deployment_id: str, platform: str):
        self.deployment_id = deployment_id
        self.platform = platform  # "aws", "azure", "gcp", "kubernetes"
        self.region = "us-west-2"
        self.instance_type = "t3.medium"
        self.instance_count = 1
        self.auto_scaling = True
        self.min_instances = 1
        self.max_instances = 10
        self.container_image = ""
        self.environment_variables: Dict[str, str] = {}
        self.volumes: List[Dict[str, Any]] = []
        self.network_config: Dict[str, Any] = {}
        self.monitoring_config: Dict[str, Any] = {}
        self.created_at = datetime.now().timestamp()
        self.status = "pending"  # "pending", "deploying", "running", "failed", "stopped"

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


class ScalingPolicy:
    """Represents an auto-scaling policy."""

    def __init__(self, policy_id: str, metric_name: str):
        self.policy_id = policy_id
        self.metric_name = metric_name
        self.target_value = 70.0
        self.scale_out_threshold = 80.0
        self.scale_in_threshold = 30.0
        self.cooldown_period = 300  # seconds
        self.min_instances = 1
        self.max_instances = 10
        self.enabled = True
        self.last_scale_time = 0.0

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


class CloudResource:
    """Represents a cloud resource."""

    def __init__(self, resource_id: str, resource_type: str):
        self.resource_id = resource_id
        self.resource_type = resource_type  # "instance", "load_balancer", "database", "storage"
        self.platform = ""
        self.region = ""
        self.status = "pending"  # "pending", "creating", "running", "failed", "terminated"
        self.public_ip = ""
        self.private_ip = ""
        self.created_at = datetime.now().timestamp()
        self.cost_per_hour = 0.0
        self.tags: Dict[str, str] = {}

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
        pass

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
        pass

    @abstractmethod
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get the status of a deployment.

        Args:
            deployment_id: ID of the deployment

        Returns:
            Dict[str, Any]: Deployment status and information
        """
        pass

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
        pass

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
        pass

    @abstractmethod
    def get_scaling_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get the scaling status and metrics for a deployment.

        Args:
            deployment_id: ID of the deployment

        Returns:
            Dict[str, Any]: Scaling status and metrics
        """
        pass

    @abstractmethod
    def deploy_to_multiple_clouds(self, deployment_config: Dict[str, Any]) -> List[str]:
        """
        Deploy to multiple cloud platforms simultaneously.

        Args:
            deployment_config: Multi-cloud deployment configuration

        Returns:
            List[str]: List of deployment IDs for each cloud
        """
        pass

    @abstractmethod
    def get_deployment_costs(self, deployment_id: str, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Get cost information for a deployment.

        Args:
            deployment_id: ID of the deployment
            time_range: Optional time range for cost calculation

        Returns:
            Dict[str, Any]: Cost information and breakdown
        """
        pass

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
        pass

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
        pass

    @abstractmethod
    def terminate_deployment(self, deployment_id: str) -> bool:
        """
        Terminate a deployment and clean up resources.

        Args:
            deployment_id: ID of the deployment

        Returns:
            bool: True if termination successful
        """
        pass






