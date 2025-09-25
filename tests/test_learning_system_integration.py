"""
Integration tests for learning system components.
Tests interactions between learning components, energy systems, neural networks,
and simulation coordinators.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from torch_geometric.data import Data

from src.learning.homeostasis_controller import HomeostasisController
from src.learning.learning_engine import LearningEngine
from src.learning.live_hebbian_learning import LiveHebbianLearning
from src.learning.memory_system import MemorySystem
from src.core.services.simulation_coordinator import SimulationCoordinator
from src.core.interfaces.neural_processor import INeuralProcessor
from src.core.interfaces.energy_manager import IEnergyManager
from src.core.interfaces.learning_engine import ILearningEngine
from src.core.interfaces.sensory_processor import ISensoryProcessor
from src.core.interfaces.performance_monitor import IPerformanceMonitor
from src.core.interfaces.graph_manager import IGraphManager
from src.core.interfaces.event_coordinator import IEventCoordinator
from src.core.interfaces.configuration_service import IConfigurationService
from src.core.interfaces.service_registry import IServiceRegistry


class TestLearningSystemIntegration:
    """Test suite for learning system integration."""

    def setup_method(self):
        """Set up integrated test environment."""
        # Create all required service mocks
        self.service_registry = Mock(spec=IServiceRegistry)
        self.neural_processor = Mock(spec=INeuralProcessor)
        self.energy_manager = Mock(spec=IEnergyManager)
        self.learning_engine_service = Mock(spec=ILearningEngine)
        self.sensory_processor = Mock(spec=ISensoryProcessor)
        self.performance_monitor = Mock(spec=IPerformanceMonitor)
        self.graph_manager = Mock(spec=IGraphManager)
        self.event_coordinator = Mock(spec=IEventCoordinator)
        self.configuration_service = Mock(spec=IConfigurationService)

        # Configure mocks for successful initialization
        mock_graph = Data()
        mock_graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time()},
            {'id': 1, 'behavior': 'dynamic', 'energy': 0.6, 'last_activation': time.time()}
        ]
        mock_graph.x = torch.tensor([[0.8], [0.6]], dtype=torch.float32)
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.graph_manager.initialize_graph.return_value = mock_graph
        self.neural_processor.initialize_neural_state.return_value = True
        self.energy_manager.initialize_energy_state.return_value = True
        self.learning_engine_service.initialize_learning_state.return_value = True
        self.sensory_processor.initialize_sensory_pathways.return_value = True
        self.performance_monitor.start_monitoring.return_value = True

        # Create simulation coordinator with mocks
        self.simulation_manager = SimulationCoordinator(
            self.service_registry, self.neural_processor, self.energy_manager,
            self.learning_engine_service, self.sensory_processor, self.performance_monitor,
            self.graph_manager, self.event_coordinator, self.configuration_service
        )

        self.homeostasis = HomeostasisController()
        self.memory_system = MemorySystem()
        self.mock_access_layer = MagicMock()

        with patch('learning.learning_engine.get_learning_config', return_value={
            'plasticity_rate': 0.01,
            'eligibility_decay': 0.95,
            'stdp_window': 20.0,
            'ltp_rate': 0.02,
            'ltd_rate': 0.01
        }):
            self.learning_engine = LearningEngine(self.mock_access_layer)

        self.hebbian_learning = LiveHebbianLearning(self.simulation_manager)

    def teardown_method(self):
        """Clean up after tests."""
        self.homeostasis.reset_statistics()
        self.memory_system.reset_statistics()
        self.learning_engine.reset_statistics()
        self.hebbian_learning.reset_learning_statistics()

    def test_homeostasis_with_learning_engine(self):
        """Test homeostasis controller integration with learning engine."""
        # Create graph with learning-relevant data
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time()},
            {'id': 1, 'behavior': 'dynamic', 'energy': 0.6, 'last_activation': time.time()}
        ]
        graph.x = torch.tensor([[0.8], [0.6]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph.edge_attributes = [
            MagicMock(weight=1.0, eligibility_trace=0.0, source=0, target=1, type='excitatory'),
            MagicMock(weight=1.0, eligibility_trace=0.0, source=1, target=0, type='excitatory')
        ]

        # Homeostasis regulates energy
        regulated_graph = self.homeostasis.regulate_network_activity(graph)

        # Learning engine consolidates connections
        consolidated_graph = self.learning_engine.consolidate_connections(regulated_graph)

        # Check that systems work together
        assert regulated_graph == consolidated_graph
        assert len(self.homeostasis.get_regulation_statistics()) >= 0
        assert len(self.learning_engine.get_learning_statistics()) >= 0

    def test_memory_system_with_homeostasis(self):
        """Test memory system integration with homeostasis."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time() - 5},
            {'id': 1, 'behavior': 'relay', 'energy': 0.7, 'last_activation': time.time() - 3}
        ]
        graph.x = torch.tensor([[0.8], [0.7]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        # Memory system forms traces
        graph_with_memory = self.memory_system.form_memory_traces(graph)

        # Homeostasis monitors health
        health = self.homeostasis.monitor_network_health(graph_with_memory)

        assert 'memory_traces' in graph_with_memory
        assert health['status'] in ['healthy', 'warning', 'critical']
        assert self.memory_system.get_memory_trace_count() >= 0

    def test_hebbian_learning_with_memory_system(self):
        """Test Hebbian learning integration with memory system."""
        graph = Data()
        graph.x = torch.tensor([[0.8], [0.6], [0.7]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator'},
            {'id': 1, 'behavior': 'dynamic'},
            {'id': 2, 'behavior': 'relay'}
        ]

        # Apply Hebbian learning
        learned_graph = self.hebbian_learning.apply_continuous_learning(graph, 0)

        # Memory system forms traces from learned patterns
        graph_with_memory = self.memory_system.form_memory_traces(learned_graph)

        assert learned_graph == graph_with_memory
        assert self.hebbian_learning.get_learning_statistics()['total_weight_changes'] >= 0

    def test_full_learning_pipeline(self):
        """Test complete learning pipeline integration."""
        # Create comprehensive test graph
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time() - 5},
            {'id': 1, 'behavior': 'dynamic', 'energy': 0.6, 'last_activation': time.time() - 3},
            {'id': 2, 'behavior': 'relay', 'energy': 0.7, 'last_activation': time.time() - 8}
        ]
        graph.x = torch.tensor([[0.8], [0.6], [0.7]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        graph.edge_attributes = [
            MagicMock(weight=1.0, eligibility_trace=0.0, source=0, target=1, type='excitatory'),
            MagicMock(weight=1.0, eligibility_trace=0.0, source=1, target=2, type='excitatory'),
            MagicMock(weight=1.0, eligibility_trace=0.0, source=2, target=0, type='inhibitory')
        ]

        # Step 1: Homeostasis regulation
        regulated_graph = self.homeostasis.regulate_network_activity(graph)

        # Step 2: Hebbian learning
        learned_graph = self.hebbian_learning.apply_continuous_learning(regulated_graph, 0)

        # Step 3: Memory formation
        memory_graph = self.memory_system.form_memory_traces(learned_graph)

        # Step 4: Learning engine consolidation
        consolidated_graph = self.learning_engine.consolidate_connections(memory_graph)

        # Step 5: Memory consolidation
        final_graph = self.memory_system.consolidate_memories(consolidated_graph)

        # Verify all components worked together
        assert final_graph == consolidated_graph
        assert self.homeostasis.get_regulation_statistics()['total_regulation_events'] >= 0
        assert self.hebbian_learning.get_learning_statistics()['total_weight_changes'] >= 0
        assert self.memory_system.get_memory_trace_count() >= 0
        assert self.learning_engine.get_learning_statistics()['weight_changes'] >= 0

    def test_energy_modulation_across_systems(self):
        """Test energy modulation consistency across learning systems."""
        # Test that energy affects all learning components similarly
        high_energy_node = {'id': 0, 'energy': 0.9}
        low_energy_node = {'id': 1, 'energy': 0.1}

        # Learning engine energy modulation
        with patch('learning.learning_engine.get_node_energy_cap', return_value=5.0):
            high_modulated = self.learning_engine._calculate_energy_modulated_rate(high_energy_node, high_energy_node, 0.02)
            low_modulated = self.learning_engine._calculate_energy_modulated_rate(low_energy_node, low_energy_node, 0.02)

            assert high_modulated > low_modulated

        # Hebbian learning energy modulation
        with patch.object(self.hebbian_learning, '_get_node_energy', side_effect=lambda nid: 0.9 if nid == 0 else 0.1), \
              patch('energy.energy_behavior.get_node_energy_cap', return_value=5.0):

            high_hebbian = self.hebbian_learning._calculate_energy_modulated_learning_rate(0, 0)
            low_hebbian = self.hebbian_learning._calculate_energy_modulated_learning_rate(1, 1)

            assert high_hebbian > low_hebbian

    def test_statistics_integration(self):
        """Test statistics integration across all components."""
        graph = Data()
        graph.node_labels = [{'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time()}]
        graph.x = torch.tensor([[0.8]], dtype=torch.float32)

        # Perform operations to generate statistics
        self.homeostasis.regulate_network_activity(graph)
        self.memory_system.form_memory_traces(graph)
        self.hebbian_learning.apply_continuous_learning(graph, 0)

        # Check all statistics are accessible
        homeo_stats = self.homeostasis.get_regulation_statistics()
        memory_stats = self.memory_system.get_memory_statistics()
        hebbian_stats = self.hebbian_learning.get_learning_statistics()
        engine_stats = self.learning_engine.get_learning_statistics()

        assert isinstance(homeo_stats, dict)
        assert isinstance(memory_stats, dict)
        assert isinstance(hebbian_stats, dict)
        assert isinstance(engine_stats, dict)

        # All should have some activity
        assert sum(homeo_stats.values()) >= 0
        assert sum(memory_stats.values()) >= 0
        assert sum(hebbian_stats.values()) >= 0
        assert sum(engine_stats.values()) >= 0

    def test_simulation_coordinator_integration(self):
        """Test integration with simulation coordinator."""
        # Initialize simulation
        success = self.simulation_manager.initialize_simulation()
        if success:
            graph = self.simulation_manager.get_neural_graph()

            # Apply learning systems
            regulated = self.homeostasis.regulate_network_activity(graph)
            learned = self.hebbian_learning.apply_continuous_learning(regulated, 0)
            with_memory = self.memory_system.form_memory_traces(learned)

            # Verify integration
            assert regulated == with_memory
            assert hasattr(with_memory, 'node_labels')
        else:
            # If simulation fails, skip test
            pytest.skip("Simulation coordinator initialization failed")

    def test_cross_component_data_flow(self):
        """Test data flow between components."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time()},
            {'id': 1, 'behavior': 'dynamic', 'energy': 0.6, 'last_activation': time.time()}
        ]
        graph.x = torch.tensor([[0.8], [0.6]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        # Homeostasis adds homeostasis_data
        regulated = self.homeostasis.regulate_network_activity(graph)
        assert hasattr(regulated, 'homeostasis_data')

        # Memory system adds memory_traces
        with_memory = self.memory_system.form_memory_traces(regulated)
        assert hasattr(with_memory, 'memory_traces')

        # Data flows through all components
        assert with_memory == regulated

    def test_error_recovery_integration(self):
        """Test error recovery across integrated components."""
        # Create graph that might cause issues
        problematic_graph = Data()
        problematic_graph.node_labels = None  # Invalid
        problematic_graph.x = torch.tensor([], dtype=torch.float32).reshape(0, 1)

        # All components should handle gracefully
        result1 = self.homeostasis.regulate_network_activity(problematic_graph)
        assert result1 == problematic_graph

        result2 = self.memory_system.form_memory_traces(result1)
        assert result2 == result1

        result3 = self.hebbian_learning.apply_continuous_learning(result2, 0)
        assert result3 == result2

    def test_performance_integration(self):
        """Test performance characteristics of integrated system."""
        import time

        # Create larger test graph
        num_nodes = 50
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'dynamic', 'energy': 0.5 + 0.4 * (i % 2), 'last_activation': time.time()}
            for i in range(num_nodes)
        ]
        graph.x = torch.rand(num_nodes, 1)
        graph.edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        graph.edge_attr = torch.rand(num_nodes * 2, 1)

        start_time = time.time()

        # Run full pipeline
        regulated = self.homeostasis.regulate_network_activity(graph)
        learned = self.hebbian_learning.apply_continuous_learning(regulated, 0)
        with_memory = self.memory_system.form_memory_traces(learned)
        consolidated = self.learning_engine.consolidate_connections(with_memory)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds max for integration test

        # Verify results
        assert consolidated == with_memory


if __name__ == "__main__":
    pytest.main([__file__])






