"""
dependency_injection.py

Advanced dependency injection system for loose coupling.
Provides service registration, resolution, and lifecycle management.
"""

import threading
from typing import Any, Dict, Type, TypeVar, Callable, Optional, List, Union
from abc import ABC, abstractmethod
from enum import Enum
from logging_utils import log_step
from interfaces import (
    INeuralSystem, IBehaviorEngine, ILearningEngine, IMemorySystem,
    IHomeostasisController, INetworkMetrics, IWorkspaceEngine,
    IEnergyBehavior, IConnectionLogic, IDeathAndBirthLogic,
    IErrorHandler, IPerformanceOptimizer, IConfigurationManager,
    IEventBus, ISimulationState, ICallbackManager, IUIComponent,
    IWindowManager, ISensoryVisualization, ILiveMonitoring,
    ISimulationController
)


class ServiceLifetime(Enum):
    """Service lifetime enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


T = TypeVar('T')


class ServiceDescriptor:
    """Describes a service registration."""
    
    def __init__(self, service_type: Type[T], implementation_type: Type[T] = None,
                 factory: Callable[[], T] = None, instance: T = None,
                 lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
                 dependencies: List[Type] = None):
        """
        Initialize service descriptor.
        
        Args:
            service_type: Interface or base type
            implementation_type: Concrete implementation type
            factory: Factory function to create instances
            instance: Pre-created instance
            lifetime: Service lifetime
            dependencies: List of dependency types
        """
        self.service_type = service_type
        self.implementation_type = implementation_type
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        self.dependencies = dependencies or []
        self.created_at = None


class DependencyInjectionContainer:
    """
    Advanced dependency injection container.
    
    Provides service registration, resolution, and lifecycle management
    with support for different service lifetimes.
    """
    
    def __init__(self):
        """Initialize the dependency injection container."""
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.RLock()
        self._current_scope: Optional[str] = None
        
        log_step("DependencyInjectionContainer initialized")
    
    def register_singleton(self, service_type: Type[T], implementation_type: Type[T] = None,
                          factory: Callable[[], T] = None, instance: T = None) -> 'DependencyInjectionContainer':
        """
        Register a singleton service.
        
        Args:
            service_type: Interface or base type
            implementation_type: Concrete implementation type
            factory: Factory function to create instances
            instance: Pre-created instance
            
        Returns:
            Self for method chaining
        """
        return self._register_service(service_type, implementation_type, factory, 
                                    instance, ServiceLifetime.SINGLETON)
    
    def register_transient(self, service_type: Type[T], implementation_type: Type[T] = None,
                          factory: Callable[[], T] = None) -> 'DependencyInjectionContainer':
        """
        Register a transient service.
        
        Args:
            service_type: Interface or base type
            implementation_type: Concrete implementation type
            factory: Factory function to create instances
            
        Returns:
            Self for method chaining
        """
        return self._register_service(service_type, implementation_type, factory, 
                                    None, ServiceLifetime.TRANSIENT)
    
    def register_scoped(self, service_type: Type[T], implementation_type: Type[T] = None,
                       factory: Callable[[], T] = None) -> 'DependencyInjectionContainer':
        """
        Register a scoped service.
        
        Args:
            service_type: Interface or base type
            implementation_type: Concrete implementation type
            factory: Factory function to create instances
            
        Returns:
            Self for method chaining
        """
        return self._register_service(service_type, implementation_type, factory, 
                                    None, ServiceLifetime.SCOPED)
    
    def _register_service(self, service_type: Type[T], implementation_type: Type[T] = None,
                         factory: Callable[[], T] = None, instance: T = None,
                         lifetime: ServiceLifetime = ServiceLifetime.SINGLETON) -> 'DependencyInjectionContainer':
        """Internal method to register a service."""
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                instance=instance,
                lifetime=lifetime
            )
            
            self._services[service_type] = descriptor
            
            log_step("Service registered", 
                    service_type=service_type.__name__,
                    lifetime=lifetime.value,
                    has_factory=factory is not None,
                    has_instance=instance is not None)
            
            return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance.
        
        Args:
            service_type: Type to resolve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not registered
        """
        with self._lock:
            if service_type not in self._services:
                raise ValueError(f"Service {service_type.__name__} is not registered")
            
            descriptor = self._services[service_type]
            
            # Handle different lifetimes
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                return self._resolve_singleton(descriptor)
            elif descriptor.lifetime == ServiceLifetime.TRANSIENT:
                return self._resolve_transient(descriptor)
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                return self._resolve_scoped(descriptor)
            else:
                raise ValueError(f"Unknown service lifetime: {descriptor.lifetime}")
    
    def _resolve_singleton(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve singleton service."""
        if descriptor.service_type in self._singletons:
            return self._singletons[descriptor.service_type]
        
        instance = self._create_instance(descriptor)
        self._singletons[descriptor.service_type] = instance
        
        log_step("Singleton service resolved", 
                service_type=descriptor.service_type.__name__)
        
        return instance
    
    def _resolve_transient(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve transient service."""
        instance = self._create_instance(descriptor)
        
        log_step("Transient service resolved", 
                service_type=descriptor.service_type.__name__)
        
        return instance
    
    def _resolve_scoped(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve scoped service."""
        if self._current_scope is None:
            raise ValueError("No active scope for scoped service resolution")
        
        if self._current_scope not in self._scoped_instances:
            self._scoped_instances[self._current_scope] = {}
        
        scope_instances = self._scoped_instances[self._current_scope]
        
        if descriptor.service_type in scope_instances:
            return scope_instances[descriptor.service_type]
        
        instance = self._create_instance(descriptor)
        scope_instances[descriptor.service_type] = instance
        
        log_step("Scoped service resolved", 
                service_type=descriptor.service_type.__name__,
                scope=self._current_scope)
        
        return instance
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a service instance."""
        # Use pre-created instance if available
        if descriptor.instance is not None:
            return descriptor.instance
        
        # Use factory if available
        if descriptor.factory is not None:
            return descriptor.factory()
        
        # Use implementation type
        if descriptor.implementation_type is not None:
            return self._create_from_type(descriptor.implementation_type)
        
        # Use service type as fallback
        return self._create_from_type(descriptor.service_type)
    
    def _create_from_type(self, type_class: Type[T]) -> T:
        """Create instance from type with dependency injection."""
        try:
            # Get constructor parameters
            import inspect
            signature = inspect.signature(type_class.__init__)
            parameters = signature.parameters
            
            # Skip 'self' parameter
            param_names = list(parameters.keys())[1:]
            
            # Resolve dependencies
            dependencies = []
            for param_name in param_names:
                param = parameters[param_name]
                param_type = param.annotation
                
                if param_type != inspect.Parameter.empty:
                    try:
                        dependency = self.resolve(param_type)
                        dependencies.append(dependency)
                    except ValueError:
                        # If dependency not registered, use default value
                        if param.default != inspect.Parameter.empty:
                            dependencies.append(param.default)
                        else:
                            raise ValueError(f"Dependency {param_type.__name__} not registered for {type_class.__name__}")
                else:
                    # No type annotation, use default value
                    if param.default != inspect.Parameter.empty:
                        dependencies.append(param.default)
                    else:
                        raise ValueError(f"No type annotation for parameter {param_name} in {type_class.__name__}")
            
            # Create instance
            return type_class(*dependencies)
            
        except Exception as e:
            log_step("Service creation error", 
                    service_type=type_class.__name__,
                    error=str(e))
            raise
    
    def create_scope(self, scope_name: str) -> 'ServiceScope':
        """Create a new service scope."""
        return ServiceScope(self, scope_name)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered."""
        with self._lock:
            return service_type in self._services
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services."""
        with self._lock:
            return self._services.copy()
    
    def clear_scope(self, scope_name: str) -> None:
        """Clear a specific scope."""
        with self._lock:
            if scope_name in self._scoped_instances:
                # Clean up scoped instances
                for instance in self._scoped_instances[scope_name].values():
                    if hasattr(instance, 'cleanup'):
                        try:
                            instance.cleanup()
                        except Exception as e:
                            log_step("Scope cleanup error", error=str(e))
                
                del self._scoped_instances[scope_name]
                log_step("Scope cleared", scope=scope_name)
    
    def clear_all_scopes(self) -> None:
        """Clear all scopes."""
        with self._lock:
            for scope_name in list(self._scoped_instances.keys()):
                self.clear_scope(scope_name)
    
    def cleanup(self) -> None:
        """Clean up all services and scopes."""
        with self._lock:
            # Clean up singletons
            for instance in self._singletons.values():
                if hasattr(instance, 'cleanup'):
                    try:
                        instance.cleanup()
                    except Exception as e:
                        log_step("Singleton cleanup error", error=str(e))
            
            # Clear all scopes
            self.clear_all_scopes()
            
            # Clear registrations
            self._services.clear()
            self._singletons.clear()
            
            log_step("DependencyInjectionContainer cleaned up")


class ServiceScope:
    """Context manager for service scopes."""
    
    def __init__(self, container: DependencyInjectionContainer, scope_name: str):
        """Initialize service scope."""
        self.container = container
        self.scope_name = scope_name
        self._previous_scope = None
    
    def __enter__(self):
        """Enter the scope."""
        with self.container._lock:
            self._previous_scope = self.container._current_scope
            self.container._current_scope = self.scope_name
        
        log_step("Service scope entered", scope=self.scope_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the scope."""
        with self.container._lock:
            self.container._current_scope = self._previous_scope
        
        log_step("Service scope exited", scope=self.scope_name)


class ServiceCollection:
    """Builder pattern for service registration."""
    
    def __init__(self):
        """Initialize service collection."""
        self.container = DependencyInjectionContainer()
    
    def add_singleton(self, service_type: Type[T], implementation_type: Type[T] = None,
                     factory: Callable[[], T] = None, instance: T = None) -> 'ServiceCollection':
        """Add singleton service."""
        self.container.register_singleton(service_type, implementation_type, factory, instance)
        return self
    
    def add_transient(self, service_type: Type[T], implementation_type: Type[T] = None,
                     factory: Callable[[], T] = None) -> 'ServiceCollection':
        """Add transient service."""
        self.container.register_transient(service_type, implementation_type, factory)
        return self
    
    def add_scoped(self, service_type: Type[T], implementation_type: Type[T] = None,
                  factory: Callable[[], T] = None) -> 'ServiceCollection':
        """Add scoped service."""
        self.container.register_scoped(service_type, implementation_type, factory)
        return self
    
    def build(self) -> DependencyInjectionContainer:
        """Build the container."""
        return self.container


# Global container instance
_global_container: Optional[DependencyInjectionContainer] = None
_global_container_lock = threading.Lock()


def get_container() -> DependencyInjectionContainer:
    """Get the global dependency injection container."""
    global _global_container
    
    if _global_container is None:
        with _global_container_lock:
            if _global_container is None:
                _global_container = DependencyInjectionContainer()
    
    return _global_container


def configure_services() -> ServiceCollection:
    """Get a service collection for configuration."""
    return ServiceCollection()


def resolve_service(service_type: Type[T]) -> T:
    """Resolve a service from the global container."""
    return get_container().resolve(service_type)


def register_service(service_type: Type[T], implementation_type: Type[T] = None,
                    factory: Callable[[], T] = None, instance: T = None,
                    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON) -> None:
    """Register a service in the global container."""
    container = get_container()
    
    if lifetime == ServiceLifetime.SINGLETON:
        container.register_singleton(service_type, implementation_type, factory, instance)
    elif lifetime == ServiceLifetime.TRANSIENT:
        container.register_transient(service_type, implementation_type, factory)
    elif lifetime == ServiceLifetime.SCOPED:
        container.register_scoped(service_type, implementation_type, factory)


def is_service_registered(service_type: Type) -> bool:
    """Check if a service is registered in the global container."""
    return get_container().is_registered(service_type)


def create_scope(scope_name: str) -> ServiceScope:
    """Create a scope in the global container."""
    return get_container().create_scope(scope_name)


def cleanup_global_container() -> None:
    """Clean up the global container."""
    global _global_container
    
    if _global_container is not None:
        _global_container.cleanup()
        _global_container = None


# Example usage and testing
if __name__ == "__main__":
    # Test dependency injection
    class TestService:
        def __init__(self, name: str = "default"):
            self.name = name
        
        def get_name(self) -> str:
            return self.name
    
    class TestConsumer:
        def __init__(self, test_service: TestService):
            self.test_service = test_service
        
        def get_service_name(self) -> str:
            return self.test_service.get_name()
    
    # Configure services
    container = configure_services() \
        .add_singleton(TestService, instance=TestService("singleton")) \
        .add_transient(TestConsumer) \
        .build()
    
    # Resolve services
    service = container.resolve(TestService)
    consumer = container.resolve(TestConsumer)
    
    print(f"Service name: {service.get_name()}")
    print(f"Consumer service name: {consumer.get_service_name()}")
    
    print("Dependency injection test completed successfully!")
