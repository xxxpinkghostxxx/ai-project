"""
Comprehensive tests for ServiceRegistry.

This module contains unit tests, integration tests, edge cases, and performance tests
for the ServiceRegistry class, covering all aspects of dependency injection functionality.
"""

import threading
import time
import unittest
from unittest.mock import patch

from src.core.interfaces.service_registry import (ServiceHealth,
                                                   ServiceLifetime)
from src.core.services.service_registry import (ServiceNotFoundError,
                                                ServiceRegistry,
                                                ServiceResolutionError)


class TestServiceRegistry(unittest.TestCase):
    """Unit tests for ServiceRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ServiceRegistry()

    def test_initialization(self):
        """Test registry initialization."""
        self.assertIsInstance(self.registry._services, dict)
        self.assertIsInstance(self.registry._lock, type(threading.RLock()))
        self.assertEqual(len(self.registry._services), 0)

    def test_register_service(self):
        """Test service registration."""
        # Define test service interface and implementation
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.registry.register(ITestService, TestService)

        self.assertTrue(self.registry.is_registered(ITestService))
        descriptor = self.registry.get_service_descriptor(ITestService)
        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.service_type, ITestService)
        self.assertEqual(descriptor.implementation_type, TestService)
        self.assertEqual(descriptor.lifetime, ServiceLifetime.SINGLETON)

    def test_register_with_lifetime(self):
        """Test service registration with custom lifetime."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.registry.register(ITestService, TestService, ServiceLifetime.TRANSIENT)

        descriptor = self.registry.get_service_descriptor(ITestService)
        self.assertEqual(descriptor.lifetime, ServiceLifetime.TRANSIENT)

    def test_register_invalid_service_type(self):
        """Test registration with invalid service type."""
        class TestService:
            pass

        with self.assertRaises(ValueError):
            self.registry.register("not_a_type", TestService)

    def test_register_invalid_implementation_type(self):
        """Test registration with invalid implementation type."""
        class ITestService:
            pass

        with self.assertRaises(ValueError):
            self.registry.register(ITestService, "not_a_type")

    def test_register_non_subclass_implementation(self):
        """Test registration where implementation doesn't inherit from interface."""
        class ITestService:
            pass

        class TestService:
            pass

        with self.assertRaises(ValueError):
            self.registry.register(ITestService, TestService)

    def test_register_instance(self):
        """Test registering a pre-created instance."""
        class ITestService:
            pass

        class TestService(ITestService):
            def __init__(self, value=42):
                self.value = value

        instance = TestService(100)
        self.registry.register_instance(ITestService, instance)

        resolved = self.registry.resolve(ITestService)
        self.assertEqual(resolved, instance)
        self.assertEqual(resolved.value, 100)

    def test_resolve_singleton(self):
        """Test resolving a singleton service."""
        class ITestService:
            pass

        class TestService(ITestService):
            def __init__(self):
                self.created = time.time()

        self.registry.register(ITestService, TestService, ServiceLifetime.SINGLETON)

        instance1 = self.registry.resolve(ITestService)
        instance2 = self.registry.resolve(ITestService)

        self.assertIs(instance1, instance2)  # Same instance
        self.assertIsInstance(instance1, TestService)

    def test_resolve_transient(self):
        """Test resolving a transient service."""
        class ITestService:
            pass

        class TestService(ITestService):
            def __init__(self):
                self.created = time.time()

        self.registry.register(ITestService, TestService, ServiceLifetime.TRANSIENT)

        instance1 = self.registry.resolve(ITestService)
        time.sleep(0.001)  # Small delay
        instance2 = self.registry.resolve(ITestService)

        self.assertIsNot(instance1, instance2)  # Different instances
        self.assertIsInstance(instance1, TestService)
        self.assertIsInstance(instance2, TestService)

    def test_resolve_unregistered_service(self):
        """Test resolving an unregistered service."""
        class ITestService:
            pass

        with self.assertRaises(ServiceNotFoundError):
            self.registry.resolve(ITestService)

    def test_resolve_with_dependencies(self):
        """Test resolving services with dependencies."""
        class ILogger:
            pass

        class IConfig:
            pass

        class LoggerService(ILogger):
            pass

        class ConfigService(IConfig):
            def __init__(self, logger: ILogger):
                self.logger = logger

        class AppService:
            def __init__(self, config: IConfig, logger: ILogger):
                self.config = config
                self.logger = logger

        # Register services
        self.registry.register(ILogger, LoggerService)
        self.registry.register(IConfig, ConfigService)

        # Resolve services
        config = self.registry.resolve(IConfig)
        logger = self.registry.resolve(ILogger)

        self.assertIsInstance(config, ConfigService)
        self.assertIsInstance(logger, LoggerService)
        self.assertIs(config.logger, logger)  # Dependency injection worked

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        class IServiceA:
            pass

        class IServiceB:
            pass

        class ServiceA(IServiceA):
            def __init__(self, service_b: IServiceB):
                self.service_b = service_b

        class ServiceB(IServiceB):
            def __init__(self, service_a: IServiceA):
                self.service_a = service_a

        self.registry.register(IServiceA, ServiceA)
        self.registry.register(IServiceB, ServiceB)

        with self.assertRaises(ServiceResolutionError) as cm:
            self.registry.resolve(IServiceA)

        self.assertIn("Circular dependency", str(cm.exception))

    def test_is_registered(self):
        """Test checking if service is registered."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.assertFalse(self.registry.is_registered(ITestService))

        self.registry.register(ITestService, TestService)
        self.assertTrue(self.registry.is_registered(ITestService))

    def test_get_registered_services(self):
        """Test getting list of registered services."""
        class IServiceA:
            pass

        class IServiceB:
            pass

        class ServiceA(IServiceA):
            pass

        class ServiceB(IServiceB):
            pass

        services = self.registry.get_registered_services()
        self.assertEqual(len(services), 0)

        self.registry.register(IServiceA, ServiceA)
        self.registry.register(IServiceB, ServiceB)

        services = self.registry.get_registered_services()
        self.assertEqual(len(services), 2)
        self.assertIn(IServiceA, services)
        self.assertIn(IServiceB, services)

    def test_unregister_service(self):
        """Test unregistering a service."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.registry.register(ITestService, TestService)
        self.assertTrue(self.registry.is_registered(ITestService))

        result = self.registry.unregister(ITestService)
        self.assertTrue(result)
        self.assertFalse(self.registry.is_registered(ITestService))

    def test_unregister_nonexistent_service(self):
        """Test unregistering a non-existent service."""
        class ITestService:
            pass

        result = self.registry.unregister(ITestService)
        self.assertFalse(result)

    def test_clear_registry(self):
        """Test clearing the registry."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.registry.register(ITestService, TestService)
        self.assertEqual(len(self.registry._services), 1)

        self.registry.clear()
        self.assertEqual(len(self.registry._services), 0)

    def test_get_service_health(self):
        """Test getting service health."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        # Unregistered service
        health = self.registry.get_service_health(ITestService)
        self.assertEqual(health, ServiceHealth.UNKNOWN)

        # Registered service
        self.registry.register(ITestService, TestService)
        health = self.registry.get_service_health(ITestService)
        self.assertEqual(health, ServiceHealth.HEALTHY)

    def test_update_service_health(self):
        """Test updating service health."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.registry.register(ITestService, TestService)

        self.registry.update_service_health(ITestService, ServiceHealth.DEGRADED)
        health = self.registry.get_service_health(ITestService)
        self.assertEqual(health, ServiceHealth.DEGRADED)


class TestServiceRegistryIntegration(unittest.TestCase):
    """Integration tests for ServiceRegistry."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.registry = ServiceRegistry()

    def test_complex_dependency_chain(self):
        """Test resolving a complex dependency chain."""
        # Define service hierarchy
        class ILogger:
            pass

        class IConfig:
            pass

        class IDatabase:
            pass

        class ICache:
            pass

        class LoggerService(ILogger):
            pass

        class ConfigService(IConfig):
            def __init__(self, logger: ILogger):
                self.logger = logger

        class DatabaseService(IDatabase):
            def __init__(self, config: IConfig, logger: ILogger):
                self.config = config
                self.logger = logger

        class CacheService(ICache):
            def __init__(self, config: IConfig):
                self.config = config

        class AppService:
            def __init__(self, database: IDatabase, cache: ICache, logger: ILogger):
                self.database = database
                self.cache = cache
                self.logger = logger

        # Register all services
        self.registry.register(ILogger, LoggerService)
        self.registry.register(IConfig, ConfigService)
        self.registry.register(IDatabase, DatabaseService)
        self.registry.register(ICache, CacheService)

        # Resolve the main service
        app = AppService(
            self.registry.resolve(IDatabase),
            self.registry.resolve(ICache),
            self.registry.resolve(ILogger)
        )

        # Verify dependency injection worked
        self.assertIsInstance(app.database, DatabaseService)
        self.assertIsInstance(app.cache, CacheService)
        self.assertIsInstance(app.logger, LoggerService)

        # Verify nested dependencies
        self.assertIsInstance(app.database.config, ConfigService)
        self.assertIsInstance(app.database.logger, LoggerService)
        self.assertIs(app.database.logger, app.logger)  # Same singleton instance

    def test_validate_dependencies_success(self):
        """Test successful dependency validation."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.registry.register(ITestService, TestService)

        result = self.registry.validate_dependencies()
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['issues']), 0)

    def test_validate_dependencies_failure(self):
        """Test dependency validation with issues."""
        class IServiceA:
            pass

        class IServiceB:
            pass

        class ServiceA(IServiceA):
            def __init__(self, service_b: IServiceB):
                self.service_b = service_b

        # Register only ServiceA, not ServiceB
        self.registry.register(IServiceA, ServiceA)

        result = self.registry.validate_dependencies()
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['issues']), 0)

    def test_get_dependency_graph(self):
        """Test getting dependency graph."""
        class IServiceA:
            pass

        class IServiceB:
            pass

        class ServiceA(IServiceA):
            def __init__(self, service_b: IServiceB):
                self.service_b = service_b

        class ServiceB(IServiceB):
            pass

        self.registry.register(IServiceA, ServiceA)
        self.registry.register(IServiceB, ServiceB)

        graph = self.registry.get_dependency_graph()

        self.assertIn('IServiceA', graph)
        self.assertIn('IServiceB', graph)
        self.assertIn(IServiceB.__name__, graph['IServiceA'])


class TestServiceRegistryEdgeCases(unittest.TestCase):
    """Edge case tests for ServiceRegistry."""

    def setUp(self):
        """Set up edge case test fixtures."""
        self.registry = ServiceRegistry()

    def test_resolve_service_with_init_failure(self):
        """Test resolving service when __init__ fails."""
        class ITestService:
            pass

        class FailingService(ITestService):
            def __init__(self):
                raise ValueError("Initialization failed")

        self.registry.register(ITestService, FailingService)

        with self.assertRaises(ServiceResolutionError):
            self.registry.resolve(ITestService)

        # Service should be marked as unhealthy
        health = self.registry.get_service_health(ITestService)
        self.assertEqual(health, ServiceHealth.UNHEALTHY)

    def test_concurrent_access(self):
        """Test concurrent access to registry."""
        class ITestService:
            pass

        class TestService(ITestService):
            def __init__(self):
                self.value = 0

        self.registry.register(ITestService, TestService)

        results = []

        def worker():
            for _ in range(100):
                service = self.registry.resolve(ITestService)
                service.value += 1
            results.append(service.value)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All threads should have accessed the same singleton instance
        final_service = self.registry.resolve(ITestService)
        self.assertEqual(final_service.value, 500)  # 5 threads * 100 increments

    def test_service_with_no_init_dependencies(self):
        """Test service with no __init__ dependencies."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.registry.register(ITestService, TestService)

        service = self.registry.resolve(ITestService)
        self.assertIsInstance(service, TestService)

    def test_service_with_complex_init_signature(self):
        """Test service with complex __init__ signature."""
        class ITestService:
            pass

        class TestService(ITestService):
            def __init__(self, value: int = 42, name: str = "test", enabled: bool = True):
                self.value = value
                self.name = name
                self.enabled = enabled

        self.registry.register(ITestService, TestService)

        service = self.registry.resolve(ITestService)
        self.assertIsInstance(service, TestService)
        self.assertEqual(service.value, 42)
        self.assertEqual(service.name, "test")
        self.assertTrue(service.enabled)


class TestServiceRegistryPerformance(unittest.TestCase):
    """Performance tests for ServiceRegistry."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.registry = ServiceRegistry()

    def test_resolution_performance(self):
        """Test performance of service resolution."""
        class ITestService:
            pass

        class TestService(ITestService):
            pass

        self.registry.register(ITestService, TestService)

        # Time multiple resolutions
        start_time = time.time()
        for _ in range(1000):
            service = self.registry.resolve(ITestService)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / 1000

        # Should be very fast (< 1ms per resolution)
        self.assertLess(avg_time, 0.001)

    def test_concurrent_resolution_performance(self):
        """Test performance under concurrent load."""
        class ITestService:
            pass

        class TestService(ITestService):
            def __init__(self):
                self.access_count = 0

        self.registry.register(ITestService, TestService)

        def worker():
            for _ in range(100):
                service = self.registry.resolve(ITestService)
                service.access_count += 1

        threads = [threading.Thread(target=worker) for _ in range(10)]

        start_time = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        end_time = time.time()

        total_time = end_time - start_time

        # Should complete in reasonable time
        self.assertLess(total_time, 5.0)

        # Verify singleton behavior
        final_service = self.registry.resolve(ITestService)
        self.assertEqual(final_service.access_count, 1000)  # 10 threads * 100 accesses


if __name__ == '__main__':
    unittest.main()






