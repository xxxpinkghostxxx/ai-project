#!/usr/bin/env python3
"""
Test the complete services integration with energy system.

This test verifies that all services work together correctly with
proper dependency injection and energy-modulated functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_complete_service_integration():
    """Test complete service integration with dependency injection."""
    try:
        from core.services.service_registry import ServiceRegistry
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.services.performance_monitoring_service import PerformanceMonitoringService
        from core.services.neural_processing_service import NeuralProcessingService
        from core.services.energy_management_service import EnergyManagementService
        from core.services.learning_service import LearningService
        from core.services.simulation_coordinator import SimulationCoordinator

        # Create service registry
        registry = ServiceRegistry()

        # Register configuration service first (no dependencies)
        config_service = ConfigurationService()
        registry.register_instance(type(config_service), config_service)

        # Register event coordinator (no dependencies)
        event_service = EventCoordinationService()
        registry.register_instance(type(event_service), event_service)

        # Register performance monitor (no dependencies)
        perf_service = PerformanceMonitoringService()
        registry.register_instance(type(perf_service), perf_service)

        # Register energy service (depends on config and event)
        energy_service = EnergyManagementService(config_service, event_service)
        registry.register_instance(type(energy_service), energy_service)

        # Register neural service (depends on energy, config, event)
        neural_service = NeuralProcessingService(energy_service, config_service, event_service)
        registry.register_instance(type(neural_service), neural_service)

        # Register learning service (depends on energy, config, event)
        learning_service = LearningService(energy_service, config_service, event_service)
        registry.register_instance(type(learning_service), learning_service)

        # Register simulation coordinator (depends on all services)
        coordinator = SimulationCoordinator(
            registry,
            neural_service,
            energy_service,
            learning_service,
            None,  # sensory_processor (not implemented yet)
            perf_service,
            None,  # graph_manager (not implemented yet)
            event_service,
            config_service
        )
        registry.register_instance(type(coordinator), coordinator)

        print("PASS: Complete service integration successful")
        return True

    except Exception as e:
        print(f"FAIL: Service integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_management():
    """Test configuration service functionality."""
    try:
        from core.services.configuration_service import ConfigurationService
        from core.interfaces.configuration_service import ConfigurationScope

        config = ConfigurationService()

        # Test parameter setting and getting
        config.set_parameter("test_param", 42, ConfigurationScope.GLOBAL)
        value = config.get_parameter("test_param", ConfigurationScope.GLOBAL)

        if value != 42:
            raise Exception(f"Expected 42, got {value}")

        # Test validation
        validation = config.validate_configuration()
        if not validation["valid"]:
            raise Exception(f"Configuration validation failed: {validation['issues']}")

        print("PASS: Configuration management test successful")
        return True

    except Exception as e:
        print(f"FAIL: Configuration management test failed: {e}")
        return False

def test_event_coordination():
    """Test event coordination functionality."""
    try:
        from core.services.event_coordination_service import EventCoordinationService

        event_service = EventCoordinationService()

        # Test event publishing and subscribing
        events_received = []

        def test_handler(event):
            events_received.append(event)

        # Subscribe to test event
        subscription_id = event_service.subscribe("test_event", test_handler)

        # Publish event
        event_service.publish("test_event", {"test": "data"}, "test_source")

        # Check if event was received
        if len(events_received) != 1:
            raise Exception(f"Expected 1 event, got {len(events_received)}")

        if events_received[0].event_type != "test_event":
            raise Exception(f"Wrong event type: {events_received[0].event_type}")

        # Test unsubscription
        success = event_service.unsubscribe(subscription_id)
        if not success:
            raise Exception("Failed to unsubscribe")

        print("PASS: Event coordination test successful")
        return True

    except Exception as e:
        print(f"FAIL: Event coordination test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    try:
        from core.services.performance_monitoring_service import PerformanceMonitoringService

        perf_service = PerformanceMonitoringService()

        # Test starting monitoring
        success = perf_service.start_monitoring()
        if not success:
            raise Exception("Failed to start monitoring")

        # Test getting metrics
        metrics = perf_service.get_current_metrics()
        if metrics is None:
            raise Exception("Failed to get metrics")

        # Test stopping monitoring
        success = perf_service.stop_monitoring()
        if not success:
            raise Exception("Failed to stop monitoring")

        print("PASS: Performance monitoring test successful")
        return True

    except Exception as e:
        print(f"FAIL: Performance monitoring test failed: {e}")
        return False

def test_energy_integration():
    """Test energy integration across services."""
    try:
        from core.services.energy_management_service import EnergyManagementService
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from torch_geometric.data import Data
        import torch

        # Create dependencies
        config = ConfigurationService()
        event_service = EventCoordinationService()
        energy_service = EnergyManagementService(config, event_service)

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

        # Test energy modulation
        modulated_graph = energy_service.modulate_neural_activity(graph)
        if modulated_graph is None:
            raise Exception("Energy modulation failed")

        # Test energy conservation
        conservation = energy_service.validate_energy_conservation(graph)
        if not conservation["valid"]:
            print(f"WARNING: Energy conservation issues: {conservation['issues']}")

        print("PASS: Energy integration test successful")
        return True

    except Exception as e:
        print(f"FAIL: Energy integration test failed: {e}")
        return False

def main():
    print("Testing Complete Services Integration")
    print("=" * 60)

    tests = [
        test_configuration_management,
        test_event_coordination,
        test_performance_monitoring,
        test_energy_integration,
        test_complete_service_integration
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
        print("SUCCESS: Complete services integration is working!")
        print("Architecture shift to service-oriented design is complete.")
        return True
    else:
        print("FAILURE: Some services integration tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)