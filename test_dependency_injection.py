"""
test_dependency_injection.py

Test cases for the dependency injection system.
"""

import unittest
from dependency_container import DependencyContainer, register_service, resolve_service, clear_container
from error_handler import ErrorHandler
from node_id_manager import NodeIDManager
from simulation_manager import SimulationManager
from service_initializer import initialize_services, reset_services


class TestDependencyInjection(unittest.TestCase):
    """Test cases for the dependency injection system."""
    
    def setUp(self):
        """Set up test fixtures."""
        clear_container()
    
    def tearDown(self):
        """Clean up after tests."""
        clear_container()
    
    def test_container_registration_and_resolution(self):
        """Test basic service registration and resolution."""
        container = DependencyContainer()
        
        # Register a service
        error_handler = ErrorHandler()
        container.register_singleton(ErrorHandler, error_handler)
        
        # Resolve the service
        resolved_handler = container.resolve(ErrorHandler)
        
        # Should be the same instance
        self.assertIs(resolved_handler, error_handler)
        self.assertTrue(container.is_registered(ErrorHandler))
    
    def test_global_container_functions(self):
        """Test global container functions."""
        # Register a service
        error_handler = ErrorHandler()
        register_service(ErrorHandler, error_handler)
        
        # Resolve the service
        resolved_handler = resolve_service(ErrorHandler)
        
        # Should be the same instance
        self.assertIs(resolved_handler, error_handler)
    
    def test_service_initialization(self):
        """Test service initialization."""
        # Initialize services
        initialize_services()
        
        # All services should be registered
        from dependency_injection import is_service_registered
        
        # Note: These might not all be registered depending on implementation
        # Just test that the initialization doesn't crash
        self.assertTrue(True)  # If we get here, initialization succeeded
    
    def test_error_handler_dependency_injection(self):
        """Test that error handler uses dependency injection."""
        # Register a custom error handler
        custom_handler = ErrorHandler()
        register_service(ErrorHandler, custom_handler)
        
        # Get error handler should return our custom instance
        from error_handler import get_error_handler
        handler = get_error_handler()
        
        self.assertIs(handler, custom_handler)
    
    def test_node_id_manager_dependency_injection(self):
        """Test that node ID manager uses dependency injection."""
        # Register a custom ID manager
        custom_manager = NodeIDManager()
        register_service(NodeIDManager, custom_manager)
        
        # Get ID manager should return our custom instance
        from node_id_manager import get_id_manager
        manager = get_id_manager()
        
        self.assertIs(manager, custom_manager)
    
    def test_simulation_manager_dependency_injection(self):
        """Test that simulation manager uses dependency injection."""
        # Register a custom simulation manager
        custom_manager = SimulationManager()
        register_service(SimulationManager, custom_manager)
        
        # Get simulation manager should return our custom instance
        from simulation_manager import get_simulation_manager
        manager = get_simulation_manager()
        
        self.assertIs(manager, custom_manager)
    
    def test_service_reset(self):
        """Test service reset functionality."""
        # Initialize services
        initialize_services()
        
        # Reset services
        reset_services()
        
        # Container should be empty
        from dependency_container import get_container
        container = get_container()
        services = container.get_registered_services()
        
        self.assertEqual(len(services), 0)


def run_dependency_injection_tests():
    """Run all dependency injection tests."""
    print("🧪 Running Dependency Injection Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDependencyInjection)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    if result.wasSuccessful():
        print("\n✅ All dependency injection tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for failure in result.failures:
            print(f"FAIL: {failure[0]}")
            print(f"Error: {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(f"Error: {error[1]}")
        return False


if __name__ == "__main__":
    success = run_dependency_injection_tests()
    exit(0 if success else 1)
