"""
Integration tests for Service-Oriented Architecture (SOA) components.
Tests critical paths and service interactions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import MagicMock, Mock

import pytest

from src.core.interfaces.configuration_service import IConfigurationService
from src.core.interfaces.energy_manager import IEnergyManager
from src.core.interfaces.event_coordinator import IEventCoordinator
from src.core.interfaces.graph_manager import IGraphManager
from src.core.interfaces.learning_engine import ILearningEngine
from src.core.interfaces.neural_processor import INeuralProcessor
from src.core.interfaces.performance_monitor import IPerformanceMonitor
from src.core.interfaces.sensory_processor import ISensoryProcessor
from src.core.interfaces.service_registry import IServiceRegistry
from src.core.interfaces.simulation_coordinator import ISimulationCoordinator
from src.core.services.service_registry import (ServiceNotFoundError,
                                                ServiceRegistry)
from src.core.services.simulation_coordinator import SimulationCoordinator


class TestSOAIntegration:
    """Test SOA integration and critical paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ServiceRegistry()

        # Create mock services
        self.mock_graph_manager = Mock(spec=IGraphManager)
        self.mock_performance_monitor = Mock(spec=IPerformanceMonitor)
        self.mock_event_coordinator = Mock(spec=IEventCoordinator)
        self.mock_neural_processor = Mock(spec=INeuralProcessor)
        self.mock_energy_manager = Mock(spec=IEnergyManager)
        self.mock_learning_engine = Mock(spec=ILearningEngine)
        self.mock_sensory_processor = Mock(spec=ISensoryProcessor)
        self.mock_configuration_service = Mock(spec=IConfigurationService)

        # Register mock services
        self.registry.register_instance(IGraphManager, self.mock_graph_manager)
        self.registry.register_instance(IPerformanceMonitor, self.mock_performance_monitor)
        self.registry.register_instance(IEventCoordinator, self.mock_event_coordinator)
        self.registry.register_instance(INeuralProcessor, self.mock_neural_processor)
        self.registry.register_instance(IEnergyManager, self.mock_energy_manager)
        self.registry.register_instance(ILearningEngine, self.mock_learning_engine)
        self.registry.register_instance(ISensoryProcessor, self.mock_sensory_processor)
        self.registry.register_instance(IConfigurationService, self.mock_configuration_service)

    def test_service_registry_resolution(self):
        """Test that services can be resolved from registry."""
        # Test resolution
        graph_manager = self.registry.resolve(IGraphManager)
        assert graph_manager is not None
        assert graph_manager == self.mock_graph_manager

        performance_monitor = self.registry.resolve(IPerformanceMonitor)
        assert performance_monitor is not None
        assert performance_monitor == self.mock_performance_monitor

    def test_service_registry_has_service(self):
        """Test service existence checking."""
        assert self.registry.is_registered(IGraphManager) == True
        assert self.registry.is_registered(IPerformanceMonitor) == True

        # Non-existent service
        from src.core.interfaces.fault_tolerance import IFaultTolerance
        assert self.registry.is_registered(IFaultTolerance) == False

    def test_simulation_coordinator_initialization(self):
        """Test simulation coordinator initialization with services."""
        # Setup mock returns
        mock_graph = Mock()
        mock_graph.node_labels = []
        self.mock_graph_manager.initialize_graph.return_value = mock_graph
        self.mock_performance_monitor.start_monitoring.return_value = True

        # Create coordinator
        coordinator = SimulationCoordinator(
            self.registry,
            self.mock_neural_processor,
            self.mock_energy_manager,
            self.mock_learning_engine,
            self.mock_sensory_processor,
            self.mock_performance_monitor,
            self.mock_graph_manager,
            self.mock_event_coordinator,
            self.mock_configuration_service
        )

        # Test initialization
        result = coordinator.initialize_simulation()
        assert result == True

        # Verify service calls
        self.mock_graph_manager.initialize_graph.assert_called_once()
        self.mock_performance_monitor.start_monitoring.assert_called_once()

    def test_simulation_step_orchestration(self):
        """Test complete simulation step orchestration."""
        # Setup mocks
        mock_graph = Mock()
        mock_graph.node_labels = []
        self.mock_graph_manager.initialize_graph.return_value = mock_graph
        self.mock_performance_monitor.start_monitoring.return_value = True
        self.mock_neural_processor.process_neural_dynamics.return_value = (mock_graph, [])
        self.mock_energy_manager.update_energy_flows.return_value = (mock_graph, [])
        self.mock_learning_engine.apply_plasticity.return_value = (mock_graph, [])
        self.mock_performance_monitor.get_current_metrics.return_value = Mock()
        self.mock_performance_monitor.record_step_end = Mock()
        self.mock_performance_monitor.record_step_start = Mock()

        # Create and initialize coordinator
        coordinator = SimulationCoordinator(
            self.registry,
            self.mock_neural_processor,
            self.mock_energy_manager,
            self.mock_learning_engine,
            self.mock_sensory_processor,
            self.mock_performance_monitor,
            self.mock_graph_manager,
            self.mock_event_coordinator,
            self.mock_configuration_service
        )
        coordinator.initialize_simulation()

        # Start simulation
        coordinator.start_simulation()

        # Mock the internal graph
        coordinator._graph = mock_graph

        # Run simulation step
        result = coordinator.execute_simulation_step(1)
        assert result == True

        # Verify orchestration order
        self.mock_neural_processor.process_neural_dynamics.assert_called_once_with(mock_graph)
        self.mock_energy_manager.update_energy_flows.assert_called_once_with(mock_graph, [])
        self.mock_learning_engine.apply_plasticity.assert_called_once_with(mock_graph, [])

    def test_simulation_state_management(self):
        """Test simulation state transitions."""
        # Setup mocks
        class MockGraph:
            def __init__(self):
                self.node_labels = []

        mock_graph = MockGraph()
        self.mock_graph_manager.initialize_graph.return_value = mock_graph
        self.mock_performance_monitor.start_monitoring.return_value = True

        coordinator = SimulationCoordinator(
            self.registry,
            self.mock_neural_processor,
            self.mock_energy_manager,
            self.mock_learning_engine,
            self.mock_sensory_processor,
            self.mock_performance_monitor,
            self.mock_graph_manager,
            self.mock_event_coordinator,
            self.mock_configuration_service
        )
        coordinator.initialize_simulation()

        # Test initial state
        state = coordinator.get_simulation_state()
        assert state.is_running == False
        assert state.step_count == 0

        # Start simulation
        coordinator.start_simulation()
        state = coordinator.get_simulation_state()
        assert state.is_running == True

        # Stop simulation
        coordinator.stop_simulation()
        state = coordinator.get_simulation_state()
        assert state.is_running == False

    def test_error_handling_integration(self):
        """Test error handling in service interactions."""
        # Setup mock to raise exception
        self.mock_neural_processor.process_neural_dynamics.side_effect = RuntimeError("Neural processing failed")

        mock_graph = Mock()
        self.mock_graph_manager.initialize_graph.return_value = mock_graph
        self.mock_performance_monitor.start_monitoring.return_value = True

        coordinator = SimulationCoordinator(
            self.registry,
            self.mock_neural_processor,
            self.mock_energy_manager,
            self.mock_learning_engine,
            self.mock_sensory_processor,
            self.mock_performance_monitor,
            self.mock_graph_manager,
            self.mock_event_coordinator,
            self.mock_configuration_service
        )
        coordinator.initialize_simulation()
        coordinator._graph = mock_graph

        # Run step that should handle error gracefully
        result = coordinator.execute_simulation_step(1)
        # Should return False due to error, but not crash
        assert result == False

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        # Setup mocks
        mock_graph = Mock()
        mock_metrics = Mock()
        mock_metrics.system_health = "healthy"

        self.mock_graph_manager.initialize_graph.return_value = mock_graph
        self.mock_performance_monitor.start_monitoring.return_value = True
        self.mock_performance_monitor.get_current_metrics.return_value = mock_metrics
        self.mock_neural_processor.process_neural_dynamics.return_value = (mock_graph, [])
        self.mock_energy_manager.update_energy_flows.return_value = (mock_graph, [])
        self.mock_learning_engine.apply_plasticity.return_value = (mock_graph, [])

        coordinator = SimulationCoordinator(
            self.registry,
            self.mock_neural_processor,
            self.mock_energy_manager,
            self.mock_learning_engine,
            self.mock_sensory_processor,
            self.mock_performance_monitor,
            self.mock_graph_manager,
            self.mock_event_coordinator,
            self.mock_configuration_service
        )
        coordinator.initialize_simulation()
        coordinator._graph = mock_graph

        # Run step
        coordinator.execute_simulation_step(1)

        # Check performance metrics integration
        metrics = coordinator.get_performance_metrics()
        assert metrics is not None

    def test_service_interface_compliance(self):
        """Test that services implement required interfaces."""
        coordinator = SimulationCoordinator(
            self.registry,
            self.mock_neural_processor,
            self.mock_energy_manager,
            self.mock_learning_engine,
            self.mock_sensory_processor,
            self.mock_performance_monitor,
            self.mock_graph_manager,
            self.mock_event_coordinator,
            self.mock_configuration_service
        )

        # Check that coordinator implements ISimulationCoordinator
        assert isinstance(coordinator, ISimulationCoordinator)

        # Check that registry implements IServiceRegistry
        assert isinstance(self.registry, IServiceRegistry)

    def test_event_coordination_integration(self):
        """Test event-driven communication between services."""
        # Setup event expectations
        events_published = []

        def event_handler(event_type, data):
            events_published.append((event_type, data))

        self.mock_event_coordinator.publish = MagicMock(side_effect=lambda e, d=None: event_handler(e, d))

        # Create coordinator and trigger events
        coordinator = SimulationCoordinator(
            self.registry,
            self.mock_neural_processor,
            self.mock_energy_manager,
            self.mock_learning_engine,
            self.mock_sensory_processor,
            self.mock_performance_monitor,
            self.mock_graph_manager,
            self.mock_event_coordinator,
            self.mock_configuration_service
        )

        # Mock some operations that might publish events
        coordinator.start_simulation()
        coordinator.stop_simulation()

        # Verify events were published (this depends on actual implementation)
        # This is a placeholder for actual event testing
        pass


class TestSOAResilience:
    """Test SOA system resilience and fault tolerance."""

    def test_service_unavailability_handling(self):
        """Test handling when services are unavailable."""
        registry = ServiceRegistry()

        # Register only some services
        mock_graph = Mock(spec=IGraphManager)
        registry.register_instance(IGraphManager, mock_graph)

        # Try to create coordinator without all required services
        with pytest.raises(ServiceNotFoundError):  # Should fail gracefully
            coordinator = SimulationCoordinator(
                registry,
                registry.resolve(INeuralProcessor),
                registry.resolve(IEnergyManager),
                registry.resolve(ILearningEngine),
                registry.resolve(ISensoryProcessor),  # Not registered
                registry.resolve(IPerformanceMonitor),
                registry.resolve(IGraphManager),
                registry.resolve(IEventCoordinator),
                registry.resolve(IConfigurationService)  # Not registered
            )
            coordinator.initialize_simulation()

    def test_partial_service_failure(self):
        """Test system behavior when some services fail."""
        registry = ServiceRegistry()

        # Create failing service
        failing_neural = Mock(spec=INeuralProcessor)
        failing_neural.process_neural_dynamics.side_effect = Exception("Service failure")

        # Register services
        registry.register_instance(IGraphManager, Mock(spec=IGraphManager))
        registry.register_instance(IPerformanceMonitor, Mock(spec=IPerformanceMonitor))
        registry.register_instance(IEventCoordinator, Mock(spec=IEventCoordinator))
        registry.register_instance(INeuralProcessor, failing_neural)
        registry.register_instance(IEnergyManager, Mock(spec=IEnergyManager))
        registry.register_instance(ILearningEngine, Mock(spec=ILearningEngine))
        registry.register_instance(ISensoryProcessor, Mock(spec=ISensoryProcessor))
        registry.register_instance(IConfigurationService, Mock(spec=IConfigurationService))

        coordinator = SimulationCoordinator(
            registry,
            registry.resolve(INeuralProcessor),
            registry.resolve(IEnergyManager),
            registry.resolve(ILearningEngine),
            registry.resolve(ISensoryProcessor),
            registry.resolve(IPerformanceMonitor),
            registry.resolve(IGraphManager),
            registry.resolve(IEventCoordinator),
            registry.resolve(IConfigurationService)
        )

        # Setup mocks
        mock_graph = Mock()
        registry.resolve(IGraphManager).initialize_graph.return_value = mock_graph
        registry.resolve(IPerformanceMonitor).start_monitoring.return_value = True

        coordinator.initialize_simulation()
        coordinator._graph = mock_graph

        # Run step - should handle failure gracefully
        result = coordinator.execute_simulation_step(1)
        assert result == False  # Should fail but not crash

    def test_service_recovery(self):
        """Test service recovery mechanisms."""
        registry = ServiceRegistry()

        # Create service that fails then recovers
        recovering_service = Mock(spec=INeuralProcessor)
        recovering_service.process_neural_dynamics.side_effect = [
            Exception("Temporary failure"),
            (Mock(), [])  # Success on retry
        ]

        # Register services
        registry.register_instance(IGraphManager, Mock(spec=IGraphManager))
        registry.register_instance(IPerformanceMonitor, Mock(spec=IPerformanceMonitor))
        registry.register_instance(IEventCoordinator, Mock(spec=IEventCoordinator))
        registry.register_instance(INeuralProcessor, recovering_service)
        registry.register_instance(IEnergyManager, Mock(spec=IEnergyManager))
        registry.register_instance(ILearningEngine, Mock(spec=ILearningEngine))
        registry.register_instance(ISensoryProcessor, Mock(spec=ISensoryProcessor))
        registry.register_instance(IConfigurationService, Mock(spec=IConfigurationService))

        coordinator = SimulationCoordinator(
            registry,
            registry.resolve(INeuralProcessor),
            registry.resolve(IEnergyManager),
            registry.resolve(ILearningEngine),
            registry.resolve(ISensoryProcessor),
            registry.resolve(IPerformanceMonitor),
            registry.resolve(IGraphManager),
            registry.resolve(IEventCoordinator),
            registry.resolve(IConfigurationService)
        )

        # Setup
        mock_graph = Mock()
        registry.resolve(IGraphManager).initialize_graph.return_value = mock_graph
        registry.resolve(IPerformanceMonitor).start_monitoring.return_value = True

        coordinator.initialize_simulation()
        coordinator._graph = mock_graph

        # First attempt should fail
        result1 = coordinator.execute_simulation_step(1)
        assert result1 == False

        # Second attempt should succeed (if retry logic exists)
        # Note: Actual retry logic depends on implementation
        pass


if __name__ == "__main__":
    pytest.main([__file__])






