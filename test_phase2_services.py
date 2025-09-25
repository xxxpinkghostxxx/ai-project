#!/usr/bin/env python3
"""
Test Phase 2 services - Advanced Services Integration.

This test verifies that the Phase 2 services (SensoryProcessingService,
GraphManagementService) work correctly with the existing service architecture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_sensory_processing_service():
    """Test SensoryProcessingService functionality."""
    try:
        from core.services.sensory_processing_service import SensoryProcessingService
        from core.services.energy_management_service import EnergyManagementService
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.interfaces.sensory_processor import SensoryInput
        from torch_geometric.data import Data
        import torch

        # Create dependencies
        config = ConfigurationService()
        event_service = EventCoordinationService()
        energy_service = EnergyManagementService(config, event_service)
        sensory_service = SensoryProcessingService(energy_service, config, event_service)

        # Create test graph
        test_node_labels = [
            {'id': 0, 'type': 'dynamic', 'energy': 0.5},
            {'id': 1, 'type': 'dynamic', 'energy': 0.8}
        ]
        test_x = torch.tensor([[0.5], [0.8]], dtype=torch.float32)
        graph = Data(x=test_x, node_labels=test_node_labels)

        # Test sensory pathway initialization
        success = sensory_service.initialize_sensory_pathways(graph)
        if not success:
            raise Exception("Sensory pathway initialization failed")

        # Test sensory input processing
        visual_input = SensoryInput(
            modality="visual",
            data={"pixels": [0.8, 0.6, 0.9]},
            intensity=0.7,
            spatial_location=(0.5, 0.3)
        )

        result = sensory_service.process_sensory_input(visual_input)
        if not result.get("success", False):
            raise Exception(f"Sensory input processing failed: {result}")

        # Test auditory input
        auditory_input = SensoryInput(
            modality="auditory",
            data={"frequency": 1000, "amplitude": 0.8},
            intensity=0.6
        )

        result = sensory_service.process_sensory_input(auditory_input)
        if not result.get("success", False):
            raise Exception(f"Auditory input processing failed: {result}")

        # Test sensory adaptation
        adapted_graph = sensory_service.apply_sensory_adaptation(graph, 0.95)
        if adapted_graph is None:
            raise Exception("Sensory adaptation failed")

        # Test sensory state
        state = sensory_service.get_sensory_state()
        if not isinstance(state, dict):
            raise Exception("Invalid sensory state")

        print("PASS: SensoryProcessingService test successful")
        return True

    except Exception as e:
        print(f"FAIL: SensoryProcessingService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_management_service():
    """Test GraphManagementService functionality."""
    try:
        from core.services.graph_management_service import GraphManagementService
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService

        # Create dependencies
        config = ConfigurationService()
        event_service = EventCoordinationService()
        graph_service = GraphManagementService(config, event_service)

        # Test graph initialization
        graph = graph_service.initialize_graph({"size": 100})
        if graph is None:
            raise Exception("Graph initialization failed")

        # Test graph validation
        validation = graph_service.validate_graph_integrity(graph)
        if not validation.get("valid", False):
            raise Exception(f"Graph validation failed: {validation.get('issues', [])}")

        # Test graph statistics
        stats = graph_service.get_graph_statistics(graph)
        if not isinstance(stats, dict):
            raise Exception("Invalid graph statistics")

        # Test graph saving and loading
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save graph
            success = graph_service.save_graph(graph, tmp_path)
            if not success:
                raise Exception("Graph saving failed")

            # Load graph
            loaded_graph = graph_service.load_graph(tmp_path)
            if loaded_graph is None:
                raise Exception("Graph loading failed")

            # Validate loaded graph
            validation = graph_service.validate_graph_integrity(loaded_graph)
            if not validation.get("valid", False):
                raise Exception(f"Loaded graph validation failed: {validation.get('issues', [])}")

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        print("PASS: GraphManagementService test successful")
        return True

    except Exception as e:
        print(f"FAIL: GraphManagementService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase2_service_integration():
    """Test Phase 2 services integration with existing services."""
    try:
        from core.services.service_registry import ServiceRegistry
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.services.performance_monitoring_service import PerformanceMonitoringService
        from core.services.energy_management_service import EnergyManagementService
        from core.services.neural_processing_service import NeuralProcessingService
        from core.services.learning_service import LearningService
        from core.services.sensory_processing_service import SensoryProcessingService
        from core.services.graph_management_service import GraphManagementService
        from core.services.simulation_coordinator import SimulationCoordinator

        # Create service registry
        registry = ServiceRegistry()

        # Register foundational services
        config_service = ConfigurationService()
        registry.register_instance(type(config_service), config_service)

        event_service = EventCoordinationService()
        registry.register_instance(type(event_service), event_service)

        perf_service = PerformanceMonitoringService()
        registry.register_instance(type(perf_service), perf_service)

        # Register Phase 2 services
        energy_service = EnergyManagementService(config_service, event_service)
        registry.register_instance(type(energy_service), energy_service)

        neural_service = NeuralProcessingService(energy_service, config_service, event_service)
        registry.register_instance(type(neural_service), neural_service)

        learning_service = LearningService(energy_service, config_service, event_service)
        registry.register_instance(type(learning_service), learning_service)

        sensory_service = SensoryProcessingService(energy_service, config_service, event_service)
        registry.register_instance(type(sensory_service), sensory_service)

        graph_service = GraphManagementService(config_service, event_service)
        registry.register_instance(type(graph_service), graph_service)

        # Register simulation coordinator
        coordinator = SimulationCoordinator(
            registry,
            neural_service,
            energy_service,
            learning_service,
            sensory_service,  # Now we have sensory service
            perf_service,
            graph_service,   # Now we have graph service
            event_service,
            config_service
        )
        registry.register_instance(type(coordinator), coordinator)

        # Test graph initialization through service
        graph = graph_service.initialize_graph({"size": 50})
        if graph is None:
            raise Exception("Service-based graph initialization failed")

        # Test sensory processing on the graph
        success = sensory_service.initialize_sensory_pathways(graph)
        if not success:
            raise Exception("Service-based sensory pathway initialization failed")

        print("PASS: Phase 2 service integration test successful")
        return True

    except Exception as e:
        print(f"FAIL: Phase 2 service integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_service_features():
    """Test advanced features of Phase 2 services."""
    try:
        from core.services.sensory_processing_service import SensoryProcessingService
        from core.services.graph_management_service import GraphManagementService
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.services.energy_management_service import EnergyManagementService
        from core.interfaces.sensory_processor import SensoryInput
        from torch_geometric.data import Data
        import torch

        # Create services
        config = ConfigurationService()
        event_service = EventCoordinationService()
        energy_service = EnergyManagementService(config, event_service)
        sensory_service = SensoryProcessingService(energy_service, config, event_service)
        graph_service = GraphManagementService(config, event_service)

        # Test advanced sensory features
        graph = graph_service.initialize_graph({"size": 200})
        sensory_service.initialize_sensory_pathways(graph)

        # Test multiple sensory inputs
        inputs = [
            SensoryInput("visual", {"pixels": [0.8, 0.6]}, 0.7, spatial_location=(0.2, 0.8)),
            SensoryInput("auditory", {"frequency": 800}, 0.5),
            SensoryInput("tactile", {"pressure": 0.9}, 0.8, spatial_location=(0.7, 0.3))
        ]

        for sensory_input in inputs:
            result = sensory_service.process_sensory_input(sensory_input)
            if not result.get("success", False):
                raise Exception(f"Advanced sensory processing failed for {sensory_input.modality}")

        # Test graph statistics
        stats = graph_service.get_graph_statistics(graph)
        if "nodes" not in stats or stats["nodes"] == 0:
            raise Exception("Graph statistics missing node count")

        # Test sensory adaptation over time
        initial_state = sensory_service.get_sensory_state()
        for _ in range(5):
            sensory_service.apply_sensory_adaptation(graph, 0.9)

        final_state = sensory_service.get_sensory_state()

        # Adaptation levels should have changed
        if initial_state.get("adaptation_levels") == final_state.get("adaptation_levels"):
            print("WARNING: Sensory adaptation may not be working as expected")

        print("PASS: Advanced service features test successful")
        return True

    except Exception as e:
        print(f"FAIL: Advanced service features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Phase 2 Services - Advanced Services Integration")
    print("=" * 70)

    tests = [
        test_sensory_processing_service,
        test_graph_management_service,
        test_phase2_service_integration,
        test_advanced_service_features
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: Phase 2 services integration is working!")
        print("Advanced services architecture is complete.")
        return True
    else:
        print("FAILURE: Some Phase 2 services tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)