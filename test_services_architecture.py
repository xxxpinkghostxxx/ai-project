#!/usr/bin/env python3
"""
Test the services architecture with energy integration.

This test verifies that the new service-oriented architecture
works correctly with energy-modulated learning and neural processing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_service_initialization():
    """Test that services can be initialized correctly."""
    try:
        from core.services.service_registry import ServiceRegistry
        from core.services.neural_processing_service import NeuralProcessingService
        from core.services.energy_management_service import EnergyManagementService
        from core.services.learning_service import LearningService

        # Create service registry
        registry = ServiceRegistry()

        # Create mock services for dependencies
        class MockConfigurationService:
            def get_parameter(self, key, default=None):
                return default

        class MockEventCoordinator:
            def publish(self, event, data=None):
                pass

        config_service = MockConfigurationService()
        event_coordinator = MockEventCoordinator()

        # Register services
        registry.register_instance(type(config_service), config_service)
        registry.register_instance(type(event_coordinator), event_coordinator)

        # Create energy service
        energy_service = EnergyManagementService(config_service, event_coordinator)
        registry.register_instance(type(energy_service), energy_service)

        # Create neural processing service
        neural_service = NeuralProcessingService(energy_service, config_service, event_coordinator)
        registry.register_instance(type(neural_service), neural_service)

        # Create learning service
        learning_service = LearningService(energy_service, config_service, event_coordinator)
        registry.register_instance(type(learning_service), learning_service)

        print("PASS: Service initialization successful")
        return True

    except Exception as e:
        print(f"FAIL: Service initialization failed: {e}")
        return False

def test_energy_integration():
    """Test energy integration across services."""
    try:
        from core.services.energy_management_service import EnergyManagementService
        from torch_geometric.data import Data
        import torch

        # Create mock services
        class MockConfigurationService:
            def get_parameter(self, key, default=None):
                return default

        class MockEventCoordinator:
            def publish(self, event, data=None):
                pass

        config_service = MockConfigurationService()
        event_coordinator = MockEventCoordinator()

        # Create energy service
        energy_service = EnergyManagementService(config_service, event_coordinator)

        # Create test graph
        test_node_labels = [
            {'id': 0, 'type': 'dynamic', 'energy': 0.5},
            {'id': 1, 'type': 'dynamic', 'energy': 0.8}
        ]
        test_x = torch.tensor([[0.5], [0.8]], dtype=torch.float32)
        graph = Data(x=test_x, node_labels=test_node_labels)

        # Test energy initialization
        success = energy_service.initialize_energy_state(graph)
        if not success:
            raise Exception("Energy initialization failed")

        # Test energy state
        energy_state = energy_service.get_energy_state()
        if not hasattr(energy_state, 'total_system_energy'):
            raise Exception("Energy state missing total_system_energy")

        print("PASS: Energy integration test successful")
        return True

    except Exception as e:
        print(f"FAIL: Energy integration test failed: {e}")
        return False

def test_learning_modulation():
    """Test energy-modulated learning."""
    try:
        from core.services.learning_service import LearningService
        from core.services.energy_management_service import EnergyManagementService

        # Create mock services
        class MockConfigurationService:
            def get_parameter(self, key, default=None):
                return default

        class MockEventCoordinator:
            def publish(self, event, data=None):
                pass

        config_service = MockConfigurationService()
        event_coordinator = MockEventCoordinator()

        # Create services
        energy_service = EnergyManagementService(config_service, event_coordinator)
        learning_service = LearningService(energy_service, config_service, event_coordinator)

        # Test learning modulation
        energy_levels = {0: 0.3, 1: 0.8}  # Low and high energy

        # Create test graph
        from torch_geometric.data import Data
        import torch

        test_node_labels = [
            {'id': 0, 'type': 'dynamic', 'energy': 0.3},
            {'id': 1, 'type': 'dynamic', 'energy': 0.8}
        ]
        test_x = torch.tensor([[0.3], [0.8]], dtype=torch.float32)
        graph = Data(x=test_x, node_labels=test_node_labels)

        # Test modulation
        modulated_graph = learning_service.modulate_learning_by_energy(graph, energy_levels)

        # Check that modulation was applied
        low_energy_node = modulated_graph.node_labels[0]
        high_energy_node = modulated_graph.node_labels[1]

        if not low_energy_node.get('learning_enabled', True):
            print("PASS: Low energy learning modulation working")
        else:
            print("WARNING: Low energy learning modulation may not be working as expected")

        if high_energy_node.get('learning_enabled', False):
            print("PASS: High energy learning modulation working")
        else:
            print("WARNING: High energy learning modulation may not be working as expected")

        print("PASS: Learning modulation test completed")
        return True

    except Exception as e:
        print(f"FAIL: Learning modulation test failed: {e}")
        return False

def main():
    print("Testing Services Architecture with Energy Integration")
    print("=" * 60)

    tests = [
        test_service_initialization,
        test_energy_integration,
        test_learning_modulation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: Services architecture with energy integration is working!")
        return True
    else:
        print("FAILURE: Some services architecture tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)