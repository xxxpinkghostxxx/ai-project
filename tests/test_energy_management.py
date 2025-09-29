"""
Comprehensive tests for EnergyManagementService.

This module contains unit tests, integration tests, edge cases, and performance tests
for the EnergyManagementService class, covering all aspects of energy management functionality.
"""

import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch_geometric.data import Data

from src.core.interfaces.configuration_service import IConfigurationService
from src.core.interfaces.energy_manager import (EnergyFlow, EnergyState,
                                                IEnergyManager)
from src.core.interfaces.event_coordinator import IEventCoordinator
from src.core.services.energy_management_service import EnergyManagementService


class TestEnergyManagementService(unittest.TestCase):
    """Unit tests for EnergyManagementService."""

    def setUp(self):
        """Set up test fixtures."""
        self.configuration_service = Mock(spec=IConfigurationService)
        self.event_coordinator = Mock(spec=IEventCoordinator)

        self.energy_service = EnergyManagementService(
            self.configuration_service, self.event_coordinator
        )

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.energy_service._energy_state, EnergyState)
        self.assertEqual(self.energy_service._energy_cap, 5.0)
        self.assertEqual(self.energy_service._decay_rate, 0.99)
        self.assertEqual(self.energy_service._metabolic_cost_per_spike, 0.1)
        self.assertEqual(len(self.energy_service._node_energies), 0)

    def test_initialize_energy_state_success(self):
        """Test successful energy state initialization."""
        graph = Data()
        graph.node_labels = [
            {"id": 0},
            {"id": 1},
            {"id": 2}
        ]

        result = self.energy_service.initialize_energy_state(graph)

        self.assertTrue(result)
        self.assertTrue(self.energy_service._energy_state.is_initialized)
        self.assertEqual(len(self.energy_service._node_energies), 3)

        # Check that energies were initialized with diversity
        energies = list(self.energy_service._node_energies.values())
        self.assertTrue(all(0.5 <= e <= 0.9 for e in energies))

        # Check that graph was updated
        for node in graph.node_labels:
            self.assertIn("energy", node)
            self.assertGreater(node["energy"], 0)

    def test_initialize_energy_state_invalid_graph(self):
        """Test energy initialization with invalid graph."""
        result = self.energy_service.initialize_energy_state(None)
        self.assertFalse(result)

        graph = Data()  # No node_labels
        result = self.energy_service.initialize_energy_state(graph)
        self.assertFalse(result)

    def test_update_energy_flows_with_spikes(self):
        """Test energy flow updates with spike events."""
        # Initialize energy state
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(3)]
        self.energy_service.initialize_energy_state(graph)

        # Create spike events
        spike_events = [
            Mock(neuron_id=0, timestamp=0.1),
            Mock(neuron_id=1, timestamp=0.2)
        ]

        updated_graph, energy_flows = self.energy_service.update_energy_flows(graph, spike_events)

        self.assertIsNotNone(updated_graph)
        self.assertGreater(len(energy_flows), 0)

        # Check decay flows (one per node)
        decay_flows = [f for f in energy_flows if f.flow_type == "decay"]
        self.assertEqual(len(decay_flows), 3)

        # Check metabolic cost flows (one per spike)
        metabolic_flows = [f for f in energy_flows if f.flow_type == "metabolic_cost"]
        self.assertEqual(len(metabolic_flows), 2)

        # Check that energies decreased
        for node_id, energy in self.energy_service._node_energies.items():
            self.assertLess(energy, 1.0)

    def test_update_energy_flows_no_spikes(self):
        """Test energy flow updates with no spike events."""
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(2)]
        self.energy_service.initialize_energy_state(graph)

        updated_graph, energy_flows = self.energy_service.update_energy_flows(graph, [])

        # Should still have decay flows
        decay_flows = [f for f in energy_flows if f.flow_type == "decay"]
        self.assertEqual(len(decay_flows), 2)

        # Energies should have decayed
        for energy in self.energy_service._node_energies.values():
            self.assertLess(energy, 1.0)

    def test_apply_metabolic_costs(self):
        """Test metabolic cost application."""
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(2)]
        self.energy_service.initialize_energy_state(graph)

        updated_graph = self.energy_service.apply_metabolic_costs(graph, 0.001)

        self.assertIsNotNone(updated_graph)

        # Energies should have decreased due to metabolic costs
        for energy in self.energy_service._node_energies.values():
            self.assertLess(energy, 1.0)

    def test_regulate_energy_homeostasis(self):
        """Test energy homeostasis regulation."""
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(2)]
        self.energy_service.initialize_energy_state(graph)

        # Manually set uneven energies
        self.energy_service._node_energies[0] = 0.1
        self.energy_service._node_energies[1] = 2.0

        updated_graph = self.energy_service.regulate_energy_homeostasis(graph)

        self.assertIsNotNone(updated_graph)

        # Energies should be more balanced
        energies = list(self.energy_service._node_energies.values())
        energy_range = max(energies) - min(energies)
        self.assertLess(energy_range, 1.5)  # Should be more balanced

    def test_modulate_neural_activity_low_energy(self):
        """Test neural activity modulation for low energy nodes."""
        graph = Data()
        graph.node_labels = [
            {"id": 0, "energy": 0.1, "threshold": 0.5, "plasticity_enabled": True},
            {"id": 1, "energy": 0.8, "threshold": 0.5, "plasticity_enabled": True}
        ]
        self.energy_service.initialize_energy_state(graph)

        updated_graph = self.energy_service.modulate_neural_activity(graph)

        # Low energy node should have higher threshold and disabled plasticity
        low_energy_node = next(node for node in updated_graph.node_labels if node["id"] == 0)
        self.assertFalse(low_energy_node["plasticity_enabled"])
        self.assertGreater(low_energy_node["threshold"], 0.5)

        # High energy node should have lower threshold and enabled plasticity
        high_energy_node = next(node for node in updated_graph.node_labels if node["id"] == 1)
        self.assertTrue(high_energy_node["plasticity_enabled"])
        self.assertLess(high_energy_node["threshold"], 0.5)

    def test_reset_energy_state(self):
        """Test energy state reset."""
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(2)]
        self.energy_service.initialize_energy_state(graph)

        # Add some flows
        self.energy_service._energy_flows.append(EnergyFlow(0, 1, 0.1, "test"))

        result = self.energy_service.reset_energy_state()

        self.assertTrue(result)
        self.assertEqual(len(self.energy_service._node_energies), 0)
        self.assertEqual(len(self.energy_service._energy_flows), 0)
        self.assertFalse(self.energy_service._energy_state.is_initialized)

    def test_get_energy_state(self):
        """Test getting energy state."""
        state = self.energy_service.get_energy_state()

        self.assertIsInstance(state, EnergyState)
        self.assertIsNotNone(state.node_energies)

    def test_get_energy_statistics_no_data(self):
        """Test energy statistics with no data."""
        stats = self.energy_service.get_energy_statistics()

        self.assertEqual(stats, {})

    def test_get_energy_statistics_with_data(self):
        """Test energy statistics with data."""
        self.energy_service._node_energies = {0: 1.0, 1: 2.0, 2: 1.5}

        stats = self.energy_service.get_energy_statistics()

        self.assertIn("total_system_energy", stats)
        self.assertIn("average_energy", stats)
        self.assertIn("energy_variance", stats)
        self.assertAlmostEqual(stats["total_system_energy"], 4.5)
        self.assertAlmostEqual(stats["average_energy"], 1.5)

    def test_validate_energy_conservation_good(self):
        """Test energy conservation validation with good conservation."""
        self.energy_service._node_energies = {0: 1.0, 1: 1.0}
        self.energy_service._total_energy_history = [2.0, 2.0, 2.0]

        result = self.energy_service.validate_energy_conservation(None)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)

    def test_validate_energy_conservation_poor(self):
        """Test energy conservation validation with poor conservation."""
        self.energy_service._node_energies = {0: 0.1, 1: 0.1}  # Total: 0.2
        self.energy_service._total_energy_history = [2.0, 1.8, 1.5]  # Large decrease

        result = self.energy_service.validate_energy_conservation(None)

        self.assertFalse(result["valid"])
        self.assertGreater(len(result["issues"]), 0)

    def test_calculate_energy_efficiency_no_data(self):
        """Test energy efficiency calculation with no data."""
        efficiency = self.energy_service.calculate_energy_efficiency(None)

        self.assertEqual(efficiency, 0.0)

    def test_calculate_energy_efficiency_with_data(self):
        """Test energy efficiency calculation with data."""
        self.energy_service._node_energies = {0: 1.0, 1: 1.0, 2: 1.0}

        efficiency = self.energy_service.calculate_energy_efficiency(None)

        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)

    def test_configure_energy_parameters(self):
        """Test energy parameter configuration."""
        params = {
            "energy_cap": 10.0,
            "decay_rate": 0.95,
            "metabolic_cost_per_spike": 0.2
        }

        result = self.energy_service.configure_energy_parameters(params)

        self.assertTrue(result)
        self.assertEqual(self.energy_service._energy_cap, 10.0)
        self.assertEqual(self.energy_service._decay_rate, 0.95)
        self.assertEqual(self.energy_service._metabolic_cost_per_spike, 0.2)

    def test_apply_energy_boost(self):
        """Test energy boost application."""
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(3)]
        self.energy_service.initialize_energy_state(graph)

        updated_graph = self.energy_service.apply_energy_boost(graph, [0, 1], 0.5)

        self.assertIsNotNone(updated_graph)

        # Boosted nodes should have increased energy
        self.assertAlmostEqual(self.energy_service._node_energies[0], 1.5)
        self.assertAlmostEqual(self.energy_service._node_energies[1], 1.5)
        # Non-boosted node should remain the same
        self.assertAlmostEqual(self.energy_service._node_energies[2], 1.0)

    def test_detect_energy_anomalies_no_anomalies(self):
        """Test energy anomaly detection with normal data."""
        self.energy_service._node_energies = {0: 1.0, 1: 1.1, 2: 0.9}

        anomalies = self.energy_service.detect_energy_anomalies(None)

        self.assertEqual(len(anomalies), 0)

    def test_detect_energy_anomalies_with_anomalies(self):
        """Test energy anomaly detection with anomalies."""
        self.energy_service._node_energies = {0: 1.0, 1: 5.0, 2: 0.1}  # Node 1 and 2 are anomalies

        anomalies = self.energy_service.detect_energy_anomalies(None)

        self.assertGreater(len(anomalies), 0)

        # Should detect both high and low energy anomalies
        anomaly_node_ids = [a["node_id"] for a in anomalies]
        self.assertIn(1, anomaly_node_ids)
        self.assertIn(2, anomaly_node_ids)


class TestEnergyManagementServiceIntegration(unittest.TestCase):
    """Integration tests for EnergyManagementService."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.configuration_service = Mock(spec=IConfigurationService)
        self.event_coordinator = Mock(spec=IEventCoordinator)

        self.energy_service = EnergyManagementService(
            self.configuration_service, self.event_coordinator
        )

    def test_full_energy_lifecycle(self):
        """Test complete energy management lifecycle."""
        # Initialize
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(5)]
        result = self.energy_service.initialize_energy_state(graph)
        self.assertTrue(result)

        initial_energies = self.energy_service._node_energies.copy()

        # Simulate some activity
        spike_events = [Mock(neuron_id=i, timestamp=0.1 + i*0.1) for i in range(3)]
        graph, flows1 = self.energy_service.update_energy_flows(graph, spike_events)

        # Apply metabolic costs
        graph = self.energy_service.apply_metabolic_costs(graph, 0.001)

        # Regulate homeostasis
        graph = self.energy_service.regulate_energy_homeostasis(graph)

        # Modulate neural activity
        graph = self.energy_service.modulate_neural_activity(graph)

        # Check that energies changed appropriately
        final_energies = self.energy_service._node_energies
        self.assertNotEqual(initial_energies, final_energies)

        # Validate conservation
        validation = self.energy_service.validate_energy_conservation(graph)
        self.assertIsInstance(validation, dict)

        # Get statistics
        stats = self.energy_service.get_energy_statistics()
        self.assertIn("total_system_energy", stats)

    def test_energy_boost_integration(self):
        """Test energy boost integration with other operations."""
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 0.5} for i in range(3)]
        self.energy_service.initialize_energy_state(graph)

        # Apply boost
        graph = self.energy_service.apply_energy_boost(graph, [0], 0.3)

        # Check boost was applied
        self.assertAlmostEqual(self.energy_service._node_energies[0], 0.8)

        # Run homeostasis - should balance energies
        graph = self.energy_service.regulate_energy_homeostasis(graph)

        # Energies should be more balanced after homeostasis
        energies = list(self.energy_service._node_energies.values())
        energy_range = max(energies) - min(energies)
        initial_range = 0.8 - 0.5  # Before homeostasis
        self.assertLess(energy_range, initial_range)


class TestEnergyManagementServiceEdgeCases(unittest.TestCase):
    """Edge case tests for EnergyManagementService."""

    def setUp(self):
        """Set up edge case test fixtures."""
        self.configuration_service = Mock(spec=IConfigurationService)
        self.event_coordinator = Mock(spec=IEventCoordinator)

        self.energy_service = EnergyManagementService(
            self.configuration_service, self.event_coordinator
        )

    def test_initialize_with_empty_graph(self):
        """Test initialization with empty graph."""
        graph = Data()
        graph.node_labels = []

        result = self.energy_service.initialize_energy_state(graph)
        self.assertTrue(result)  # Should succeed with empty graph

    def test_update_flows_with_invalid_spike_events(self):
        """Test energy flow updates with invalid spike events."""
        graph = Data()
        graph.node_labels = [{"id": 0, "energy": 1.0}]
        self.energy_service.initialize_energy_state(graph)

        # Spike event for non-existent neuron
        spike_events = [Mock(neuron_id=999, timestamp=0.1)]

        updated_graph, flows = self.energy_service.update_energy_flows(graph, spike_events)

        # Should handle gracefully
        self.assertIsNotNone(updated_graph)

    def test_homeostasis_with_single_node(self):
        """Test homeostasis regulation with single node."""
        graph = Data()
        graph.node_labels = [{"id": 0, "energy": 1.0}]
        self.energy_service.initialize_energy_state(graph)

        updated_graph = self.energy_service.regulate_energy_homeostasis(graph)

        # Should work with single node
        self.assertIsNotNone(updated_graph)

    def test_energy_boost_over_capacity(self):
        """Test energy boost that exceeds capacity."""
        graph = Data()
        graph.node_labels = [{"id": 0, "energy": 4.5}]  # Close to capacity
        self.energy_service.initialize_energy_state(graph)

        # Boost that would exceed capacity
        updated_graph = self.energy_service.apply_energy_boost(graph, [0], 2.0)

        # Energy should be capped at capacity
        self.assertLessEqual(self.energy_service._node_energies[0], self.energy_service._energy_cap)

    def test_anomaly_detection_with_identical_energies(self):
        """Test anomaly detection with identical energies."""
        self.energy_service._node_energies = {0: 1.0, 1: 1.0, 2: 1.0}

        anomalies = self.energy_service.detect_energy_anomalies(None)

        # No variance, no anomalies
        self.assertEqual(len(anomalies), 0)

    def test_configuration_with_invalid_parameters(self):
        """Test configuration with invalid parameters."""
        invalid_params = {
            "energy_cap": -1.0,  # Invalid negative value
            "decay_rate": 1.5,   # Invalid > 1.0
        }

        result = self.energy_service.configure_energy_parameters(invalid_params)

        # Should still succeed (no validation in current implementation)
        self.assertTrue(result)

    def test_modulate_activity_with_missing_node_properties(self):
        """Test neural modulation with missing node properties."""
        graph = Data()
        graph.node_labels = [
            {"id": 0, "energy": 0.1},  # Missing threshold and plasticity_enabled
            {"id": 1, "energy": 0.8}
        ]
        self.energy_service.initialize_energy_state(graph)

        # Should handle missing properties gracefully
        updated_graph = self.energy_service.modulate_neural_activity(graph)

        self.assertIsNotNone(updated_graph)


class TestEnergyManagementServicePerformance(unittest.TestCase):
    """Performance tests for EnergyManagementService."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.configuration_service = Mock(spec=IConfigurationService)
        self.event_coordinator = Mock(spec=IEventCoordinator)

        self.energy_service = EnergyManagementService(
            self.configuration_service, self.event_coordinator
        )

    def test_initialization_performance(self):
        """Test performance of energy state initialization."""
        # Create larger graph
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(1000)]

        start_time = time.time()
        result = self.energy_service.initialize_energy_state(graph)
        end_time = time.time()

        self.assertTrue(result)

        init_time = end_time - start_time
        # Should initialize 1000 nodes quickly (< 1 second)
        self.assertLess(init_time, 1.0)

    def test_energy_flow_update_performance(self):
        """Test performance of energy flow updates."""
        # Initialize with many nodes
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(500)]
        self.energy_service.initialize_energy_state(graph)

        # Create many spike events
        spike_events = [Mock(neuron_id=i % 500, timestamp=0.1 + i*0.001) for i in range(100)]

        start_time = time.time()
        updated_graph, flows = self.energy_service.update_energy_flows(graph, spike_events)
        end_time = time.time()

        update_time = end_time - start_time
        # Should process quickly (< 100ms)
        self.assertLess(update_time, 0.1)

    def test_homeostasis_performance(self):
        """Test performance of homeostasis regulation."""
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(1000)]
        self.energy_service.initialize_energy_state(graph)

        start_time = time.time()
        updated_graph = self.energy_service.regulate_energy_homeostasis(graph)
        end_time = time.time()

        homeostasis_time = end_time - start_time
        # Should complete quickly (< 100ms)
        self.assertLess(homeostasis_time, 0.1)

    def test_statistics_calculation_performance(self):
        """Test performance of statistics calculation."""
        # Large energy dataset
        self.energy_service._node_energies = {i: np.random.random() for i in range(10000)}

        start_time = time.time()
        stats = self.energy_service.get_energy_statistics()
        end_time = time.time()

        stats_time = end_time - start_time
        # Should calculate quickly (< 50ms)
        self.assertLess(stats_time, 0.05)

        # Verify stats are correct
        self.assertIn("total_system_energy", stats)
        self.assertAlmostEqual(stats["total_system_energy"], sum(self.energy_service._node_energies.values()), places=5)

    def test_memory_usage_during_operations(self):
        """Test memory usage during energy operations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform multiple operations
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(1000)]
        self.energy_service.initialize_energy_state(graph)

        for _ in range(100):
            spike_events = [Mock(neuron_id=i % 1000, timestamp=0.1) for i in range(50)]
            graph, _ = self.energy_service.update_energy_flows(graph, spike_events)
            graph = self.energy_service.regulate_energy_homeostasis(graph)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for this workload)
        self.assertLess(memory_increase, 100.0)


if __name__ == '__main__':
    unittest.main()






