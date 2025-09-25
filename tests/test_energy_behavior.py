"""
Comprehensive tests for energy behavior components.

This module contains unit tests, integration tests, edge cases, and performance tests
for all energy behavior functions including EnergyCalculator, energy dynamics,
membrane potential updates, and refractory period handling.
"""

import unittest
import time
import numpy as np
import torch
from unittest.mock import Mock, patch
from torch_geometric.data import Data

from energy.energy_behavior import (
    EnergyCalculator, get_node_energy_cap, update_node_energy_with_learning,
    apply_energy_behavior, apply_oscillator_energy_dynamics,
    apply_integrator_energy_dynamics, apply_relay_energy_dynamics,
    apply_highway_energy_dynamics, apply_dynamic_energy_dynamics,
    emit_energy_pulse, update_membrane_potentials, apply_refractory_periods,
    safe_divide
)
from energy.energy_constants import EnergyConstants
from energy.node_access_layer import NodeAccessLayer
from energy.node_id_manager import get_id_manager


class TestEnergyCalculator(unittest.TestCase):
    """Unit tests for EnergyCalculator static methods."""

    def test_calculate_energy_cap(self):
        """Test energy cap calculation."""
        cap = EnergyCalculator.calculate_energy_cap()
        self.assertGreater(cap, 0)
        self.assertIsInstance(cap, float)

    def test_calculate_energy_decay(self):
        """Test energy decay calculation."""
        current_energy = 1.0
        decay_rate = 0.99
        decayed = EnergyCalculator.calculate_energy_decay(current_energy, decay_rate)
        self.assertEqual(decayed, current_energy * decay_rate)

    def test_calculate_energy_transfer(self):
        """Test energy transfer calculation."""
        energy = 1.0
        fraction = 0.2
        transfer = EnergyCalculator.calculate_energy_transfer(energy, fraction)
        self.assertEqual(transfer, energy * fraction)

    def test_calculate_energy_boost(self):
        """Test energy boost calculation."""
        energy = 1.0
        boost = 0.5
        boosted = EnergyCalculator.calculate_energy_boost(energy, boost)
        expected = min(energy + boost, EnergyCalculator.calculate_energy_cap())
        self.assertEqual(boosted, expected)

    def test_calculate_membrane_potential(self):
        """Test membrane potential calculation."""
        energy = 2.5
        potential = EnergyCalculator.calculate_membrane_potential(energy)
        energy_cap = EnergyCalculator.calculate_energy_cap()
        expected = min(energy / energy_cap, 1.0)
        self.assertEqual(potential, expected)

    def test_apply_energy_bounds(self):
        """Test energy bounds application."""
        # Test within bounds
        energy = 2.0
        bounded = EnergyCalculator.apply_energy_bounds(energy)
        self.assertEqual(bounded, energy)

        # Test negative energy
        negative_energy = -1.0
        bounded = EnergyCalculator.apply_energy_bounds(negative_energy)
        self.assertEqual(bounded, 0.0)

        # Test energy over cap
        over_cap = EnergyCalculator.calculate_energy_cap() + 1.0
        bounded = EnergyCalculator.apply_energy_bounds(over_cap)
        self.assertEqual(bounded, EnergyCalculator.calculate_energy_cap())


class TestEnergyBehaviorFunctions(unittest.TestCase):
    """Unit tests for energy behavior functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph = Data()
        self.graph.node_labels = [
            {"type": "dynamic", "energy": 1.0, "membrane_potential": 0.5,
             "threshold": 0.5, "behavior": "dynamic", "plasticity_enabled": True},
            {"type": "oscillator", "energy": 2.0, "membrane_potential": 0.8,
             "threshold": 0.5, "behavior": "oscillator", "last_activation": 0,
             "oscillation_freq": 1.0, "refractory_timer": 0},
            {"type": "integrator", "energy": 1.5, "membrane_potential": 0.6,
             "threshold": 0.5, "behavior": "integrator", "integration_rate": 0.5}
        ]
        self.graph.x = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float32)
        self.graph.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        # Generate and register node IDs with the ID manager
        id_manager = get_id_manager()
        for i, node in enumerate(self.graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type=node.get('type', 'test'))
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

    def test_safe_divide(self):
        """Test safe divide function."""
        # Normal division
        result = safe_divide(10.0, 2.0)
        self.assertEqual(result, 5.0)

        # Division by zero
        result = safe_divide(10.0, 0.0)
        self.assertEqual(result, 0.0)

        # Custom fallback
        result = safe_divide(10.0, 0.0, 1.0)
        self.assertEqual(result, 1.0)

    def test_get_node_energy_cap(self):
        """Test node energy cap retrieval."""
        cap = get_node_energy_cap()
        self.assertGreater(cap, 0)
        self.assertIsInstance(cap, float)

    def test_update_node_energy_with_learning(self):
        """Test energy update with learning."""
        graph = self.graph
        node_id = 0
        delta_energy = 0.5

        result_graph = update_node_energy_with_learning(graph, node_id, delta_energy)

        # Check that energy was updated
        access_layer = NodeAccessLayer(result_graph)
        new_energy = access_layer.get_node_energy(node_id)
        self.assertIsNotNone(new_energy)
        self.assertAlmostEqual(new_energy, 1.5, places=5)

    def test_apply_energy_behavior(self):
        """Test basic energy behavior application."""
        graph = self.graph

        result_graph = apply_energy_behavior(graph)

        # Check that energies decayed
        access_layer = NodeAccessLayer(result_graph)
        for node_id in [0, 1, 2]:
            energy = access_layer.get_node_energy(node_id)
            self.assertIsNotNone(energy)
            self.assertLess(energy, self.graph.x[node_id, 0].item())

    def test_apply_oscillator_energy_dynamics(self):
        """Test oscillator energy dynamics."""
        graph = self.graph
        node_id = 1  # Oscillator node

        result_graph = apply_oscillator_energy_dynamics(graph, node_id)

        # Check that oscillator node properties were updated
        access_layer = NodeAccessLayer(result_graph)
        last_activation = access_layer.get_node_property(node_id, 'last_activation')
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer')

        self.assertIsNotNone(last_activation)
        self.assertIsNotNone(refractory_timer)

    def test_apply_integrator_energy_dynamics(self):
        """Test integrator energy dynamics."""
        graph = self.graph
        node_id = 2  # Integrator node

        result_graph = apply_integrator_energy_dynamics(graph, node_id)

        # Integrator should accumulate energy from inputs
        access_layer = NodeAccessLayer(result_graph)
        energy = access_layer.get_node_energy(node_id)
        self.assertIsNotNone(energy)

    def test_apply_relay_energy_dynamics(self):
        """Test relay energy dynamics."""
        # Create graph with relay node
        graph = Data()
        graph.node_labels = [
            {"behavior": "relay", "relay_amplification": 1.5},
            {"behavior": "dynamic"}
        ]
        graph.x = torch.tensor([[2.0], [1.0]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        # Generate and register node IDs
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type='test')
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        result_graph = apply_relay_energy_dynamics(graph, 0)

        # Check that relay transferred energy
        access_layer = NodeAccessLayer(result_graph)
        relay_energy = access_layer.get_node_energy(0)
        target_energy = access_layer.get_node_energy(1)
        self.assertIsNotNone(relay_energy)
        self.assertIsNotNone(target_energy)

    def test_apply_highway_energy_dynamics(self):
        """Test highway energy dynamics."""
        # Create graph with highway node
        graph = Data()
        graph.node_labels = [
            {"behavior": "highway", "highway_energy_boost": 2.0},
            {"behavior": "dynamic"}
        ]
        graph.x = torch.tensor([[0.5], [1.0]], dtype=torch.float32)  # Low energy highway
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        # Generate and register node IDs
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type='test')
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        result_graph = apply_highway_energy_dynamics(graph, 0)

        # Check that highway boosted energy
        access_layer = NodeAccessLayer(result_graph)
        highway_energy = access_layer.get_node_energy(0)
        self.assertGreater(highway_energy, 0.5)

    def test_apply_dynamic_energy_dynamics(self):
        """Test dynamic energy dynamics."""
        graph = self.graph
        node_id = 0

        result_graph = apply_dynamic_energy_dynamics(graph, node_id)

        # Check plasticity was modulated
        access_layer = NodeAccessLayer(result_graph)
        plasticity = access_layer.get_node_property(node_id, 'plasticity_enabled')
        self.assertIsNotNone(plasticity)

    def test_emit_energy_pulse(self):
        """Test energy pulse emission."""
        graph = self.graph
        source_node_id = 0

        result_graph = emit_energy_pulse(graph, source_node_id)

        # Check that pulse was emitted (energies may have changed)
        access_layer = NodeAccessLayer(result_graph)
        for node_id in [0, 1, 2]:
            energy = access_layer.get_node_energy(node_id)
            self.assertIsNotNone(energy)

    def test_update_membrane_potentials(self):
        """Test membrane potential updates."""
        graph = self.graph

        result_graph = update_membrane_potentials(graph)

        # Check that membrane potentials were updated in node labels
        for node in result_graph.node_labels:
            if 'membrane_potential' in node:
                self.assertIsInstance(node['membrane_potential'], float)

    def test_apply_refractory_periods(self):
        """Test refractory period application."""
        graph = Data()
        graph.node_labels = [
            {"id": 0, "refractory_timer": 0.5, "membrane_potential": 0.8},
            {"id": 1, "refractory_timer": 0.0, "membrane_potential": 0.6}
        ]

        result_graph = apply_refractory_periods(graph)

        # Check that refractory timers were updated
        for node in result_graph.node_labels:
            if 'refractory_timer' in node:
                self.assertLessEqual(node['refractory_timer'], 0.5)


class TestEnergyBehaviorIntegration(unittest.TestCase):
    """Integration tests for energy behavior components."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.graph = Data()
        self.graph.node_labels = []
        self.graph.x = torch.empty((0, 1), dtype=torch.float32)
        self.graph.edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create a small network
        behaviors = ['dynamic', 'oscillator', 'integrator', 'relay', 'highway']
        for i in range(5):
            behavior = behaviors[i]
            node = {
                "type": "dynamic",
                "energy": 1.0 + i * 0.2,
                "membrane_potential": 0.5,
                "threshold": 0.5,
                "behavior": behavior,
                "plasticity_enabled": True,
                "last_activation": 0,
                "refractory_timer": 0
            }
            self.graph.node_labels.append(node)
            self.graph.x = torch.cat([self.graph.x, torch.tensor([[1.0 + i * 0.2]], dtype=torch.float32)], dim=0)

        # Add some connections
        edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
        self.graph.edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Generate and register node IDs with the ID manager
        id_manager = get_id_manager()
        for i, node in enumerate(self.graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type=node.get('type', 'test'))
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

    def test_full_energy_behavior_cycle(self):
        """Test complete energy behavior processing cycle."""
        graph = self.graph

        # Apply all energy behaviors
        graph = apply_energy_behavior(graph)
        graph = update_membrane_potentials(graph)
        graph = apply_refractory_periods(graph)

        # Apply specific dynamics
        for i in range(5):
            if i == 1:  # Oscillator
                graph = apply_oscillator_energy_dynamics(graph, i)
            elif i == 2:  # Integrator
                graph = apply_integrator_energy_dynamics(graph, i)
            elif i == 3:  # Relay
                graph = apply_relay_energy_dynamics(graph, i)
            elif i == 4:  # Highway
                graph = apply_highway_energy_dynamics(graph, i)
            else:  # Dynamic
                graph = apply_dynamic_energy_dynamics(graph, i)

        # Check that graph is still valid
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.node_labels), 5)

        # Check that energies are reasonable
        access_layer = NodeAccessLayer(graph)
        for node_id in range(5):
            energy = access_layer.get_node_energy(node_id)
            self.assertIsNotNone(energy)
            self.assertGreaterEqual(energy, 0)
            self.assertLessEqual(energy, EnergyCalculator.calculate_energy_cap())

    def test_energy_pulse_propagation(self):
        """Test energy pulse propagation through network."""
        graph = self.graph

        # Emit pulse from node 0
        graph = emit_energy_pulse(graph, 0)

        # Check that energies changed
        access_layer = NodeAccessLayer(graph)
        initial_energies = [1.0, 1.2, 1.4, 1.6, 1.8]  # From setup

        for node_id in range(5):
            energy = access_layer.get_node_energy(node_id)
            self.assertIsNotNone(energy)
            # Energy should be different (may increase or decrease)
            self.assertNotEqual(energy, initial_energies[node_id])


class TestEnergyBehaviorEdgeCases(unittest.TestCase):
    """Edge case tests for energy behavior functions."""

    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)

        # Should handle gracefully
        result = apply_energy_behavior(graph)
        self.assertIsNotNone(result)

        result = update_membrane_potentials(graph)
        self.assertIsNotNone(result)

    def test_invalid_node_ids(self):
        """Test handling of invalid node IDs."""
        graph = Data()
        graph.node_labels = [{"id": 0, "energy": 1.0}]
        graph.x = torch.tensor([[1.0]], dtype=torch.float32)

        # Should handle invalid node IDs gracefully
        result = apply_oscillator_energy_dynamics(graph, 999)
        self.assertIsNotNone(result)

        result = emit_energy_pulse(graph, 999)
        self.assertIsNotNone(result)

    def test_missing_node_properties(self):
        """Test handling of missing node properties."""
        graph = Data()
        graph.node_labels = [{"id": 0}]  # Missing energy and other properties
        graph.x = torch.tensor([[1.0]], dtype=torch.float32)

        # Should handle missing properties gracefully
        result = apply_energy_behavior(graph)
        self.assertIsNotNone(result)

        result = update_membrane_potentials(graph)
        self.assertIsNotNone(result)

    def test_zero_energy_handling(self):
        """Test handling of zero energy nodes."""
        graph = Data()
        graph.node_labels = [{"energy": 0.0, "membrane_potential": 0.0}]
        graph.x = torch.tensor([[0.0]], dtype=torch.float32)

        # Generate and register node IDs
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type='test')
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        result = apply_energy_behavior(graph)
        self.assertIsNotNone(result)

        # Energy should remain non-negative
        access_layer = NodeAccessLayer(result)
        energy = access_layer.get_node_energy(0)
        self.assertGreaterEqual(energy, 0)

    def test_extreme_energy_values(self):
        """Test handling of extreme energy values."""
        energy_cap = EnergyCalculator.calculate_energy_cap()

        graph = Data()
        graph.node_labels = [
            {"energy": 0.0},
            {"energy": energy_cap * 2},  # Over cap
            {"energy": -1.0}  # Negative
        ]
        graph.x = torch.tensor([[0.0], [energy_cap * 2], [-1.0]], dtype=torch.float32)

        # Generate and register node IDs
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type='test')
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        result = apply_energy_behavior(graph)
        self.assertIsNotNone(result)

        access_layer = NodeAccessLayer(result)
        for node_id in [0, 1, 2]:
            energy = access_layer.get_node_energy(node_id)
            self.assertGreaterEqual(energy, 0)
            self.assertLessEqual(energy, energy_cap)


# class TestEnergyBehaviorPerformance(unittest.TestCase):
#     """Performance tests for energy behavior functions."""

#     def test_large_graph_performance(self):
        """Test performance with large graphs."""
        # Create large graph
        num_nodes = 100
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)

        for i in range(num_nodes):
            node = {
                "energy": 1.0,
                "membrane_potential": 0.5,
                "behavior": "dynamic"
            }
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[1.0]], dtype=torch.float32)], dim=0)

        # Generate and register node IDs
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type='test')
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        # Test apply_energy_behavior performance
        start_time = time.time()
        result = apply_energy_behavior(graph)
        end_time = time.time()

        self.assertLess(end_time - start_time, 300.0)  # Should complete in < 300 seconds

        # Test update_membrane_potentials performance
        start_time = time.time()
        result = update_membrane_potentials(result)
        end_time = time.time()

        self.assertLess(end_time - start_time, 2.0)  # Should complete in < 2 seconds

    def test_energy_pulse_performance(self):
        """Test energy pulse emission performance."""
        # Create graph with many connections
        num_nodes = 100
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)
        edges = []

        for i in range(num_nodes):
            node = {"energy": 1.0}
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[1.0]], dtype=torch.float32)], dim=0)
            # Connect to next 5 nodes
            for j in range(1, min(6, num_nodes - i)):
                edges.append([i, i + j])

        graph.edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Generate and register node IDs
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type='test')
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        start_time = time.time()
        result = emit_energy_pulse(graph, 0)
        end_time = time.time()

        self.assertLess(end_time - start_time, 1.0)  # Should complete in < 1 second


class TestEnergyBehaviorSimulationScenarios(unittest.TestCase):
    """Simulation scenario tests for energy behavior."""

    def test_oscillator_network_simulation(self):
        """Test oscillator network energy dynamics."""
        # Create oscillator network
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)
        edges = []

        num_oscillators = 5
        for i in range(num_oscillators):
            node = {
                "behavior": "oscillator",
                "oscillation_freq": 1.0 + i * 0.2,
                "energy": 2.0,
                "last_activation": 0,
                "refractory_timer": 0
            }
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[2.0]], dtype=torch.float32)], dim=0)
            # Connect oscillators in ring
            edges.append([i, (i + 1) % num_oscillators])

        graph.edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Generate and register node IDs
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type='test')
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        # Simulate several time steps
        for step in range(10):
            for node_id in range(num_oscillators):
                graph = apply_oscillator_energy_dynamics(graph, node_id)
            graph = apply_energy_behavior(graph)
            graph = update_membrane_potentials(graph)

        # Check that simulation completed
        access_layer = NodeAccessLayer(graph)
        for node_id in range(num_oscillators):
            energy = access_layer.get_node_energy(node_id)
            self.assertIsNotNone(energy)
            self.assertGreaterEqual(energy, 0)

    def test_integrator_chain_simulation(self):
        """Test integrator chain energy accumulation."""
        # Create integrator chain
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)
        edges = []

        chain_length = 5
        for i in range(chain_length):
            node = {
                "behavior": "integrator" if i > 0 else "oscillator",  # First is input
                "energy": 1.0,
                "integration_rate": 0.5
            }
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[1.0]], dtype=torch.float32)], dim=0)
            if i > 0:
                edges.append([i-1, i])  # Chain connection

        graph.edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Generate and register node IDs
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node_type='test')
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        # Simulate input and integration
        for step in range(5):
            # Input pulse
            graph = emit_energy_pulse(graph, 0)
            # Integration
            for node_id in range(1, chain_length):
                graph = apply_integrator_energy_dynamics(graph, node_id)

        # Check that integrators accumulated energy
        access_layer = NodeAccessLayer(graph)
        for node_id in range(1, chain_length):
            energy = access_layer.get_node_energy(node_id)
            self.assertIsNotNone(energy)


if __name__ == '__main__':
    unittest.main()