"""
ServiceRegistry implementation - Dependency injection container.

This module provides the concrete implementation of IServiceRegistry,
enabling dependency injection and service lifecycle management for the
neural simulation system.
"""

import inspect
from threading import RLock
from typing import Any, Dict, List, Optional, Type, TypeVar

from ..interfaces.service_registry import (IServiceRegistry, ServiceDescriptor,
                                           ServiceHealth, ServiceLifetime)

ServiceType = TypeVar('ServiceType')


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered."""


class ServiceResolutionError(Exception):
    """Raised when a service cannot be resolved or instantiated."""


class ServiceRegistry(IServiceRegistry):
    """
    Concrete implementation of IServiceRegistry with dependency injection.

    This class provides a thread-safe service registry that supports
    singleton, transient, and scoped service lifetimes with automatic
    dependency resolution and health monitoring.
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._lock = RLock()
        self._resolution_stack: List[Type] = []  # Detect circular dependencies

    def register[T](self, service_type: Type[T], implementation_type: Type[T],
                   lifetime: ServiceLifetime = ServiceLifetime.SINGLETON) -> None:
        """
        Register a service implementation.

        Args:
            service_type: The abstract service interface type
            implementation_type: The concrete implementation type
            lifetime: The lifetime scope for the service

        Raises:
            ValueError: If service_type is not a type or implementation is invalid
        """
        if not isinstance(service_type, type):
            raise ValueError("service_type must be a type")
        if not isinstance(implementation_type, type):
            raise ValueError("implementation_type must be a type")
        if not issubclass(implementation_type, service_type):
            raise ValueError(f"{implementation_type.__name__} must implement {service_type.__name__}")

        with self._lock:
            descriptor = ServiceDescriptor(service_type, implementation_type, lifetime)
            descriptor.health = ServiceHealth.HEALTHY  # Assume healthy until proven otherwise
            self._services[service_type] = descriptor
            self._analyze_dependencies(descriptor)

    def register_instance[T](self, service_type: Type[T], instance: T) -> None:
        """
        Register a pre-created service instance.

        Args:
            service_type: The service interface type
            instance: The pre-created service instance
        """
        if not isinstance(service_type, type):
            raise ValueError("service_type must be a type")
        if not isinstance(instance, service_type):
            raise ValueError(f"Instance must be of type {service_type.__name__}")

        with self._lock:
            descriptor = ServiceDescriptor(service_type, type(instance), ServiceLifetime.SINGLETON)
            descriptor.instance = instance
            descriptor.health = ServiceHealth.HEALTHY
            self._services[service_type] = descriptor

    def resolve[T](self, service_type: Type[T]) -> T:
        """
        Resolve a service instance.

        Args:
            service_type: The service interface type to resolve

        Returns:
            T: The resolved service instance

        Raises:
            ServiceNotFoundError: If no implementation is registered
            ServiceResolutionError: If the service cannot be instantiated
        """
        with self._lock:
            # Check for circular dependencies
            if service_type in self._resolution_stack:
                cycle = self._resolution_stack + [service_type]
                raise ServiceResolutionError(f"Circular dependency detected: {' -> '.join(t.__name__ for t in cycle)}")

            if service_type not in self._services:
                raise ServiceNotFoundError(f"No implementation registered for {service_type.__name__}")

            descriptor = self._services[service_type]

            # Return existing instance for singletons
            if descriptor.lifetime == ServiceLifetime.SINGLETON and descriptor.instance is not None:
                return descriptor.instance

            # Detect circular dependencies
            self._resolution_stack.append(service_type)

            try:
                # Create new instance
                instance = self._create_instance(descriptor)

                # Store singleton instances
                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    descriptor.instance = instance
                    descriptor.health = ServiceHealth.HEALTHY

                return instance

            except Exception as e:
                descriptor.health = ServiceHealth.UNHEALTHY
                raise ServiceResolutionError(f"Failed to create instance of {service_type.__name__}: {e}") from e
            finally:
                self._resolution_stack.pop()

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        with self._lock:
            return service_type in self._services

    def get_service_descriptor(self, service_type: Type) -> Optional[ServiceDescriptor]:
        """Get the service descriptor for a registered service type."""
        with self._lock:
            return self._services.get(service_type)

    def get_registered_services(self) -> List[Type]:
        """Get a list of all registered service types."""
        with self._lock:
            return list(self._services.keys())

    def unregister(self, service_type: Type) -> bool:
        """Unregister a service type."""
        with self._lock:
            if service_type in self._services:
                descriptor = self._services[service_type]
                # Clean up singleton instance if it exists
                if descriptor.instance is not None and hasattr(descriptor.instance, 'cleanup'):
                    try:
                        descriptor.instance.cleanup()
                    except (AttributeError, RuntimeError, TypeError):
                        pass  # Ignore cleanup errors for common cleanup issues
                del self._services[service_type]
                return True
            return False

    def clear(self) -> None:
        """Clear all registered services."""
        with self._lock:
            # Clean up all singleton instances
            for descriptor in self._services.values():
                if descriptor.instance is not None and hasattr(descriptor.instance, 'cleanup'):
                    try:
                        descriptor.instance.cleanup()
                    except (AttributeError, RuntimeError, TypeError):
                        pass  # Ignore cleanup errors for common cleanup issues
            self._services.clear()
            self._resolution_stack.clear()

    def get_service_health(self, service_type: Type) -> ServiceHealth:
        """Get the health status of a registered service."""
        with self._lock:
            descriptor = self._services.get(service_type)
            return descriptor.health if descriptor else ServiceHealth.UNKNOWN

    def update_service_health(self, service_type: Type, health: ServiceHealth) -> None:
        """Update the health status of a registered service."""
        with self._lock:
            descriptor = self._services.get(service_type)
            if descriptor:
                descriptor.health = health

    def validate_dependencies(self) -> Dict[str, Any]:
        """
        Validate that all service dependencies can be resolved.

        Returns:
            Dict[str, Any]: Validation results with dependency issues
        """
        issues = []
        warnings = []

        with self._lock:
            for service_type, descriptor in self._services.items():
                try:
                    # Try to resolve the service to check dependencies
                    self.resolve(service_type)
                except ServiceResolutionError as e:
                    issues.append({
                        'service': service_type.__name__,
                        'error': str(e),
                        'type': 'resolution_error'
                    })
                except (ServiceNotFoundError, ValueError, TypeError) as e:
                    issues.append({
                        'service': service_type.__name__,
                        'error': str(e),
                        'type': 'unexpected_error'
                    })

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_services': len(self._services)
        }

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph for all registered services."""
        graph = {}

        with self._lock:
            for service_type, descriptor in self._services.items():
                dependencies = [dep.__name__ for dep in descriptor.dependencies]
                graph[service_type.__name__] = dependencies

        return graph

    def create_scope(self) -> 'IServiceRegistry':
        """Create a new scoped service registry."""
        # For now, return a new instance. In a full implementation,
        # this would create a child registry that inherits from parent
        return ServiceRegistry()

    def dispose_scope(self) -> None:
        """Dispose of a scoped service registry."""
        # Clean up scoped instances
        self.clear()

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a new instance of the service implementation."""
        implementation_type = descriptor.implementation_type

        # Define basic types that should not be injected
        basic_types = (int, str, bool, float, list, dict, tuple, set, bytes, type(None))

        # Get the constructor signature
        constructor = inspect.signature(implementation_type.__init__)
        parameters = constructor.parameters

        # Skip 'self' and variable parameters (*args, **kwargs)
        param_names = [
            name for name, param in parameters.items()
            if name != 'self' and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        if not param_names:
            # No dependencies, create directly
            return implementation_type()

        # Resolve dependencies
        kwargs = {}
        for param_name in param_names:
            param_info = parameters[param_name]
            if param_info.annotation != inspect.Parameter.empty:
                dependency_type = param_info.annotation
                # Skip injection for basic types (assume they have defaults or are not services)
                if dependency_type in basic_types:
                    continue
                try:
                    if dependency_type is IServiceRegistry:
                        kwargs[param_name] = self
                    else:
                        kwargs[param_name] = self.resolve(dependency_type)
                except ServiceNotFoundError as e:
                    raise ServiceResolutionError(
                        f"Cannot resolve dependency '{param_name}' for {implementation_type.__name__}"
                    ) from e
            else:
                raise ServiceResolutionError(
                    f"Parameter '{param_name}' in {implementation_type.__name__} has no type annotation"
                )

        return implementation_type(**kwargs)

    def _analyze_dependencies(self, descriptor: ServiceDescriptor) -> None:
        """Analyze the dependencies of a service implementation."""
        implementation_type = descriptor.implementation_type

        try:
            constructor = inspect.signature(implementation_type.__init__)
            parameters = constructor.parameters

            dependencies = []
            for param_name, param_info in parameters.items():
                if param_name != 'self' and param_info.annotation != inspect.Parameter.empty:
                    dependencies.append(param_info.annotation)

            descriptor.dependencies = dependencies
        except (ValueError, TypeError, AttributeError):
            # If we can't analyze dependencies, leave the list empty
            descriptor.dependencies = []


