"""
IServiceRegistry interface - Service registry and dependency injection.

This interface defines the contract for service registration, resolution,
and lifecycle management in the dependency injection framework.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Defines the lifetime scope of registered services."""
    SINGLETON = "singleton"  # One instance for entire application
    TRANSIENT = "transient"  # New instance each time requested
    SCOPED = "scoped"       # One instance per scope (e.g., per simulation)


class ServiceHealth(Enum):
    """Defines the health status of a service."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceDescriptor:
    """Describes a registered service."""

    def __init__(self, service_type: Type, implementation_type: Type,
                 lifetime: ServiceLifetime = ServiceLifetime.SINGLETON):
        self.service_type = service_type
        self.implementation_type = implementation_type
        self.lifetime = lifetime
        self.instance = None
        self.health = ServiceHealth.UNKNOWN
        self.dependencies: List[Type] = []
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert service descriptor to dictionary."""
        return {
            'service_type': f"{self.service_type.__module__}.{self.service_type.__name__}",
            'implementation_type': (
                f"{self.implementation_type.__module__}."
                f"{self.implementation_type.__name__}"
            ),
            'lifetime': self.lifetime.value,
            'health': self.health.value,
            'dependencies': [f"{dep.__module__}.{dep.__name__}" for dep in self.dependencies],
            'metadata': self.metadata.copy()
        }


class IServiceRegistry(ABC):
    """
    Abstract interface for service registry and dependency injection.

    This interface defines the contract for registering services, resolving
    dependencies, and managing service lifecycles in the neural simulation system.
    """

    @abstractmethod
    def register(self, service_type: Type[T], implementation_type: Type[T],
                   lifetime: ServiceLifetime = ServiceLifetime.SINGLETON) -> None:
        """
        Register a service implementation for a given service type.

        Args:
            service_type: The abstract service interface type
            implementation_type: The concrete implementation type
            lifetime: The lifetime scope for the service
        """
        raise NotImplementedError()

    @abstractmethod
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """
        Register a pre-created service instance.

        Args:
            service_type: The service interface type
            instance: The pre-created service instance
        """
        raise NotImplementedError()

    @abstractmethod
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance for the given service type.

        This method implements the dependency injection resolution logic,
        creating instances as needed based on registered implementations
        and their lifetime scopes.

        Args:
            service_type: The service interface type to resolve

        Returns:
            T: The resolved service instance

        Raises:
            ServiceNotFoundError: If no implementation is registered for the service type
            ServiceResolutionError: If the service cannot be instantiated
        """
        raise NotImplementedError()

    @abstractmethod
    def is_registered(self, service_type: Type) -> bool:
        """
        Check if a service type is registered.

        Args:
            service_type: The service interface type to check

        Returns:
            bool: True if the service type is registered, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_service_descriptor(self, service_type: Type) -> Optional[ServiceDescriptor]:
        """
        Get the service descriptor for a registered service type.

        Args:
            service_type: The service interface type

        Returns:
            Optional[ServiceDescriptor]: The service descriptor, or None if not registered
        """
        raise NotImplementedError()

    @abstractmethod
    def get_registered_services(self) -> List[Type]:
        """
        Get a list of all registered service types.

        Returns:
            List[Type]: List of registered service interface types
        """
        raise NotImplementedError()

    @abstractmethod
    def unregister(self, service_type: Type) -> bool:
        """
        Unregister a service type.

        Args:
            service_type: The service interface type to unregister

        Returns:
            bool: True if the service was unregistered, False if it wasn't registered
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all registered services and reset the registry.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_service_health(self, service_type: Type) -> ServiceHealth:
        """
        Get the health status of a registered service.

        Args:
            service_type: The service interface type

        Returns:
            ServiceHealth: The health status of the service
        """
        raise NotImplementedError()

    @abstractmethod
    def update_service_health(self, service_type: Type, health: ServiceHealth) -> None:
        """
        Update the health status of a registered service.

        Args:
            service_type: The service interface type
            health: The new health status
        """
        raise NotImplementedError()

    @abstractmethod
    def validate_dependencies(self) -> Dict[str, Any]:
        """
        Validate that all service dependencies can be resolved.

        This method checks that all registered services have their
        dependencies properly registered and resolvable.

        Returns:
            Dict[str, Any]: Validation results with any dependency issues
        """
        raise NotImplementedError()

    @abstractmethod
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the dependency graph for all registered services.

        Returns:
            Dict[str, List[str]]: Dependency graph mapping service names to their dependencies
        """
        raise NotImplementedError()

    @abstractmethod
    def create_scope(self) -> 'IServiceRegistry':
        """
        Create a new scoped service registry.

        Scoped registries inherit all services from the parent registry
        but can override them with scoped instances.

        Returns:
            IServiceRegistry: A new scoped service registry
        """
        raise NotImplementedError()

    @abstractmethod
    def dispose_scope(self) -> None:
        """
        Dispose of a scoped service registry and clean up scoped instances.
        """
        raise NotImplementedError()
