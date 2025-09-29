"""
Comprehensive tests for NodeAccessLayer.

This module contains unit tests, integration tests, edge cases, and performance tests
for the NodeAccessLayer class, covering all access methods, caching, and graph operations.
"""

import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch_geometric.data import Data

from src.energy.energy_behavior import get_node_energy_cap
from src.energy.node_access_layer import (NodeAccessLayer,
                                          create_node_access_layer)
from src.energy.node_id_manager import NodeIDManager, get_id_manager


class TestNodeAccessLayerInitialization(unittest.TestCase):
    """Unit tests for NodeAccessLayer initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph = Data()
        self.graph.node_labels = [
            {"id": 0, "type": "dynamic", "energy": 1.0, "behavior": "dynamic"},
            {"id": 1, "type": "sensory", "energy": 2.0, "behavior": "sensory"}
        ]
        self.graph.x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        self.graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    def test_initialization_with_graph(self):
        """Test initialization with a graph."""
        access_layer = NodeAccessLayer(self.graph)
        self.assertIsNotNone(access_layer.graph)
        self.assertIsNotNone(access_layer.id_manager)
        self.assertEqual(len(access_layer._node_cache), 0)
        self.assertFalse(access_layer._cache_valid)

    def test_initialization_with_custom_id_manager(self):
        """Test initialization with custom ID manager."""
        custom_id_manager = NodeIDManager()
        access_layer = NodeAccessLayer(self.graph, custom_id_manager)
        self.assertEqual(access_layer.id_manager, custom_id_manager)

    def test_create_node_access_layer_function(self):
        """Test create_node_access_layer function."""
        access_layer = create_node_access_layer(self.graph)
        self.assertIsInstance(access_layer, NodeAccessLayer)
        self.assertEqual(access_layer.graph, self.graph)


class TestNodeAccessLayerBasicOperations(unittest.TestCase):
    """Unit tests for basic NodeAccessLayer operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph = Data()
        self.graph.node_labels = [
            {"id": 0, "type": "dynamic", "energy": 1.0, "behavior": "dynamic", "threshold": 0.5},
            {"id": 1, "type": "sensory", "energy": 2.0, "behavior": "sensory", "threshold": 0.7}
        ]
        self.graph.x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        self.graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        # Register nodes in ID manager
        self.id_manager = get_id_manager()
        self.id_manager.reset()  # Start fresh
        for node in self.graph.node_labels:
            self.id_manager._active_ids.add(node['id'])
        for i, node in enumerate(self.graph.node_labels):
            self.id_manager.register_node_index(node['id'], i)

        self.access_layer = NodeAccessLayer(self.graph, self.id_manager)

    def test_get_node_by_id_valid(self):
        """Test getting node by valid ID."""
        node = self.access_layer.get_node_by_id(0)
        self.assertIsNotNone(node)
        self.assertEqual(node['id'], 0)
        self.assertEqual(node['type'], 'dynamic')

    def test_get_node_by_id_invalid(self):
        """Test getting node by invalid ID."""
        node = self.access_layer.get_node_by_id(999)
        self.assertIsNone(node)

    def test_get_node_by_id_with_caching(self):
        """Test node retrieval with caching."""
        # First access should populate cache
        node1 = self.access_layer.get_node_by_id(0)
        self.assertTrue(self.access_layer._cache_valid)
        self.assertIn(0, self.access_layer._node_cache)

        # Second access should use cache
        node2 = self.access_layer.get_node_by_id(0)
        self.assertEqual(node1, node2)

    def test_get_node_energy_valid(self):
        """Test getting energy for valid node."""
        energy = self.access_layer.get_node_energy(0)
        self.assertIsNotNone(energy)
        self.assertEqual(energy, 1.0)

    def test_get_node_energy_invalid(self):
        """Test getting energy for invalid node."""
        energy = self.access_layer.get_node_energy(999)
        self.assertIsNone(energy)

    def test_set_node_energy_valid(self):
        """Test setting energy for valid node."""
        success = self.access_layer.set_node_energy(0, 3.0)
        self.assertTrue(success)

        # Check that energy was updated
        energy = self.access_layer.get_node_energy(0)
        self.assertEqual(energy, 3.0)

        # Check that cache was invalidated
        self.assertFalse(self.access_layer._cache_valid)

    def test_set_node_energy_invalid(self):
        """Test setting energy for invalid node."""
        success = self.access_layer.set_node_energy(999, 3.0)
        self.assertFalse(success)

    def test_update_node_property_valid(self):
        """Test updating node property for valid node."""
        success = self.access_layer.update_node_property(0, 'threshold', 0.8)
        self.assertTrue(success)

        # Check that property was updated
        threshold = self.access_layer.get_node_property(0, 'threshold')
        self.assertEqual(threshold, 0.8)

    def test_update_node_property_invalid_node(self):
        """Test updating property for invalid node."""
        success = self.access_layer.update_node_property(999, 'threshold', 0.8)
        self.assertFalse(success)

    def test_get_node_property_valid(self):
        """Test getting node property for valid node."""
        threshold = self.access_layer.get_node_property(0, 'threshold')
        self.assertEqual(threshold, 0.5)

    def test_get_node_property_default(self):
        """Test getting node property with default value."""
        missing_prop = self.access_layer.get_node_property(0, 'missing_prop', 'default')
        self.assertEqual(missing_prop, 'default')

    def test_get_node_property_invalid_node(self):
        """Test getting property for invalid node."""
        prop = self.access_layer.get_node_property(999, 'threshold')
        self.assertIsNone(prop)


class TestNodeAccessLayerSelectionMethods(unittest.TestCase):
    """Unit tests for node selection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph = Data()
        self.graph.node_labels = [
            {"id": 0, "type": "dynamic", "behavior": "oscillator"},
            {"id": 1, "type": "sensory", "behavior": "sensory"},
            {"id": 2, "type": "dynamic", "behavior": "integrator"},
            {"id": 3, "type": "sensory", "behavior": "sensory"}
        ]
        self.graph.x = torch.tensor([[1.0], [2.0], [1.5], [2.5]], dtype=torch.float32)

        # Register nodes
        self.id_manager = get_id_manager()
        self.id_manager.reset()
        for node in self.graph.node_labels:
            self.id_manager._active_ids.add(node['id'])
        for i, node in enumerate(self.graph.node_labels):
            self.id_manager.register_node_index(node['id'], i)

        self.access_layer = NodeAccessLayer(self.graph, self.id_manager)

    def test_select_nodes_by_type(self):
        """Test selecting nodes by type."""
        dynamic_nodes = self.access_layer.select_nodes_by_type('dynamic')
        sensory_nodes = self.access_layer.select_nodes_by_type('sensory')

        self.assertEqual(len(dynamic_nodes), 2)
        self.assertEqual(len(sensory_nodes), 2)
        self.assertIn(0, dynamic_nodes)
        self.assertIn(2, dynamic_nodes)
        self.assertIn(1, sensory_nodes)
        self.assertIn(3, sensory_nodes)

    def test_select_nodes_by_behavior(self):
        """Test selecting nodes by behavior."""
        oscillators = self.access_layer.select_nodes_by_behavior('oscillator')
        sensory_nodes = self.access_layer.select_nodes_by_behavior('sensory')
        integrators = self.access_layer.select_nodes_by_behavior('integrator')

        self.assertEqual(len(oscillators), 1)
        self.assertEqual(len(sensory_nodes), 2)
        self.assertEqual(len(integrators), 1)
        self.assertIn(0, oscillators)
        self.assertIn(2, integrators)

    def test_select_nodes_by_property(self):
        """Test selecting nodes by arbitrary property."""
        matching_nodes = self.access_layer.select_nodes_by_property('behavior', 'sensory')
        self.assertEqual(len(matching_nodes), 2)
        self.assertIn(1, matching_nodes)
        self.assertIn(3, matching_nodes)

    def test_filter_nodes(self):
        """Test filtering nodes with custom function."""
        # Filter nodes with energy >= 2.0
        high_energy_nodes = self.access_layer.filter_nodes(
            lambda node_id, node: self.access_layer.get_node_energy(node_id) >= 2.0
        )
        self.assertEqual(len(high_energy_nodes), 2)
        self.assertIn(1, high_energy_nodes)
        self.assertIn(3, high_energy_nodes)

    def test_iterate_nodes_by_ids(self):
        """Test iterating nodes by ID list."""
        node_ids = [0, 2]
        nodes = list(self.access_layer.iterate_nodes_by_ids(node_ids))

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0][0], 0)
        self.assertEqual(nodes[1][0], 2)

    def test_iterate_all_nodes(self):
        """Test iterating all nodes."""
        all_nodes = list(self.access_layer.iterate_all_nodes())
        self.assertEqual(len(all_nodes), 4)

        # Check that all IDs are present
        ids = [node_id for node_id, node in all_nodes]
        self.assertEqual(set(ids), {0, 1, 2, 3})


class TestNodeAccessLayerStatistics(unittest.TestCase):
    """Unit tests for statistics and utility methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph = Data()
        self.graph.node_labels = [
            {"id": 0, "type": "dynamic", "behavior": "oscillator", "state": "active"},
            {"id": 1, "type": "sensory", "behavior": "sensory", "state": "active"},
            {"id": 2, "type": "dynamic", "behavior": "integrator", "state": "inactive"},
            {"id": 3, "type": "sensory", "behavior": "sensory", "state": "active"}
        ]
        self.graph.x = torch.tensor([[1.0], [2.0], [1.5], [2.5]], dtype=torch.float32)

        # Register nodes
        self.id_manager = get_id_manager()
        self.id_manager.reset()
        for node in self.graph.node_labels:
            self.id_manager._active_ids.add(node['id'])
            self.id_manager._node_type_map[node['id']] = node.get('type', 'unknown')
        for i, node in enumerate(self.graph.node_labels):
            self.id_manager.register_node_index(node['id'], i)

        self.access_layer = NodeAccessLayer(self.graph, self.id_manager)

    def test_get_node_count(self):
        """Test getting total node count."""
        count = self.access_layer.get_node_count()
        self.assertEqual(count, 4)

    def test_is_valid_node_id(self):
        """Test node ID validation."""
        self.assertTrue(self.access_layer.is_valid_node_id(0))
        self.assertTrue(self.access_layer.is_valid_node_id(1))
        self.assertFalse(self.access_layer.is_valid_node_id(999))
        self.assertFalse(self.access_layer.is_valid_node_id(None))
        self.assertFalse(self.access_layer.is_valid_node_id(-1))

    def test_get_node_count_by_type(self):
        """Test counting nodes by type."""
        dynamic_count = self.access_layer.get_node_count_by_type('dynamic')
        sensory_count = self.access_layer.get_node_count_by_type('sensory')

        self.assertEqual(dynamic_count, 2)
        self.assertEqual(sensory_count, 2)

    def test_get_node_statistics(self):
        """Test getting comprehensive node statistics."""
        stats = self.access_layer.get_node_statistics()

        self.assertEqual(stats['total_nodes'], 4)
        self.assertEqual(stats['by_type']['dynamic'], 2)
        self.assertEqual(stats['by_type']['sensory'], 2)
        self.assertEqual(stats['by_behavior']['oscillator'], 1)
        self.assertEqual(stats['by_behavior']['sensory'], 2)
        self.assertEqual(stats['by_behavior']['integrator'], 1)
        self.assertEqual(stats['by_state']['active'], 3)
        self.assertEqual(stats['by_state']['inactive'], 1)

        # Check energy statistics
        self.assertAlmostEqual(stats['energy_stats']['total_energy'], 7.0)
        self.assertAlmostEqual(stats['energy_stats']['average_energy'], 1.75)
        self.assertEqual(stats['energy_stats']['min_energy'], 1.0)
        self.assertEqual(stats['energy_stats']['max_energy'], 2.5)

    def test_validate_consistency(self):
        """Test graph consistency validation."""
        consistency = self.access_layer.validate_consistency()
        self.assertIsInstance(consistency, dict)
        self.assertIn('is_consistent', consistency)

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        # Populate cache
        self.access_layer.get_node_by_id(0)
        self.assertTrue(self.access_layer._cache_valid)

        # Invalidate cache
        self.access_layer.invalidate_cache()
        self.assertFalse(self.access_layer._cache_valid)
        self.assertEqual(len(self.access_layer._node_cache), 0)

    def test_rebuild_cache(self):
        """Test cache rebuilding."""
        self.access_layer.rebuild_cache()
        self.assertTrue(self.access_layer._cache_valid)
        self.assertEqual(len(self.access_layer._node_cache), 4)


class TestNodeAccessLayerIntegration(unittest.TestCase):
    """Integration tests for NodeAccessLayer."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.graph = Data()
        self.graph.node_labels = []
        self.graph.x = torch.empty((0, 1), dtype=torch.float32)

        # Create a small network
        for i in range(1, 11):
            behavior = ['oscillator', 'integrator', 'relay', 'highway', 'dynamic'][(i-1) % 5]
            node = {
                "id": i,
                "type": "dynamic",
                "energy": 1.0 + (i-1) * 0.1,
                "behavior": behavior,
                "threshold": 0.5,
                "plasticity_enabled": True
            }
            self.graph.node_labels.append(node)
            self.graph.x = torch.cat([self.graph.x, torch.tensor([[1.0 + (i-1) * 0.1]], dtype=torch.float32)], dim=0)

        # Register all nodes
        self.id_manager = get_id_manager()
        self.id_manager.reset()
        for node in self.graph.node_labels:
            self.id_manager._active_ids.add(node['id'])
        for i, node in enumerate(self.graph.node_labels):
            self.id_manager.register_node_index(node['id'], i)

        self.access_layer = NodeAccessLayer(self.graph, self.id_manager)

    def test_full_node_lifecycle(self):
        """Test complete node lifecycle operations."""
        # Get initial state
        initial_energy = self.access_layer.get_node_energy(1)
        initial_threshold = self.access_layer.get_node_property(1, 'threshold')

        # Update properties
        self.access_layer.update_node_property(1, 'threshold', 0.7)
        self.access_layer.set_node_energy(1, 2.0)

        # Verify updates
        new_energy = self.access_layer.get_node_energy(1)
        new_threshold = self.access_layer.get_node_property(1, 'threshold')

        self.assertNotEqual(initial_energy, new_energy)
        self.assertNotEqual(initial_threshold, new_threshold)
        self.assertEqual(new_energy, 2.0)
        self.assertEqual(new_threshold, 0.7)

    def test_bulk_operations(self):
        """Test bulk node operations."""
        # Update multiple nodes
        for i in range(1, 6):
            self.access_layer.set_node_energy(i, 3.0)
            self.access_layer.update_node_property(i, 'plasticity_enabled', False)

        # Verify all updates
        for i in range(1, 6):
            energy = self.access_layer.get_node_energy(i)
            plasticity = self.access_layer.get_node_property(i, 'plasticity_enabled')

            self.assertEqual(energy, 3.0)
            self.assertFalse(plasticity)

    def test_selection_and_iteration_integration(self):
        """Test integration of selection and iteration methods."""
        # Select dynamic behavior nodes
        dynamic_nodes = self.access_layer.select_nodes_by_behavior('dynamic')

        # Iterate through them and collect data
        energies = []
        for node_id, node in self.access_layer.iterate_nodes_by_ids(dynamic_nodes):
            energy = self.access_layer.get_node_energy(node_id)
            energies.append(energy)

        self.assertEqual(len(energies), len(dynamic_nodes))
        self.assertTrue(all(e > 0 for e in energies))


class TestNodeAccessLayerEdgeCases(unittest.TestCase):
    """Edge case tests for NodeAccessLayer."""

    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)

        # Reset ID manager for empty graph test
        get_id_manager().reset()
        access_layer = NodeAccessLayer(graph)

        # Should handle gracefully
        node = access_layer.get_node_by_id(0)
        self.assertIsNone(node)

        energy = access_layer.get_node_energy(0)
        self.assertIsNone(energy)

        stats = access_layer.get_node_statistics()
        self.assertEqual(stats['total_nodes'], 0)

    def test_missing_graph_attributes(self):
        """Test handling of graphs with missing attributes."""
        graph = Data()
        # No node_labels or x

        access_layer = NodeAccessLayer(graph)

        node = access_layer.get_node_by_id(0)
        self.assertIsNone(node)

        energy = access_layer.get_node_energy(0)
        self.assertIsNone(energy)

    def test_invalid_node_indices(self):
        """Test handling of invalid node indices."""
        graph = Data()
        graph.node_labels = [{"id": 1}]
        graph.x = torch.tensor([[1.0]], dtype=torch.float32)

        # Manually corrupt ID manager to have wrong indices
        id_manager = get_id_manager()
        id_manager.reset()
        id_manager._active_ids.add(1)
        id_manager.register_node_index(1, 999)  # Invalid index

        access_layer = NodeAccessLayer(graph, id_manager)

        node = access_layer.get_node_by_id(1)
        self.assertIsNone(node)

    def test_numpy_integer_ids(self):
        """Test handling of numpy integer node IDs."""
        graph = Data()
        graph.node_labels = [{"id": np.int32(1), "energy": 1.0}]
        graph.x = torch.tensor([[1.0]], dtype=torch.float32)

        id_manager = get_id_manager()
        id_manager.reset()
        id_manager._active_ids.add(1)
        id_manager.register_node_index(1, 0)

        access_layer = NodeAccessLayer(graph, id_manager)

        # Should handle numpy integers
        node = access_layer.get_node_by_id(np.int32(1))
        self.assertIsNotNone(node)

        energy = access_layer.get_node_energy(np.int64(1))
        self.assertIsNotNone(energy)

    def test_torch_tensor_ids(self):
        """Test handling of torch tensor node IDs."""
        graph = Data()
        graph.node_labels = [{"id": 1, "energy": 1.0}]
        graph.x = torch.tensor([[1.0]], dtype=torch.float32)

        id_manager = get_id_manager()
        id_manager.reset()
        id_manager._active_ids.add(1)
        id_manager.register_node_index(1, 0)

        access_layer = NodeAccessLayer(graph, id_manager)

        # Should handle torch tensors
        node = access_layer.get_node_by_id(torch.tensor(1))
        self.assertIsNotNone(node)

    def test_extreme_energy_values(self):
        """Test handling of extreme energy values."""
        graph = Data()
        graph.node_labels = [
            {"id": 1, "energy": 0.0},
            {"id": 2, "energy": get_node_energy_cap() * 2},
            {"id": 3, "energy": -1.0}
        ]
        graph.x = torch.tensor([[0.0], [get_node_energy_cap() * 2], [-1.0]], dtype=torch.float32)

        id_manager = get_id_manager()
        id_manager.reset()
        for node in graph.node_labels:
            id_manager._active_ids.add(node['id'])
        for i in range(3):
            id_manager.register_node_index(i+1, i)

        access_layer = NodeAccessLayer(graph, id_manager)

        # Should handle extreme values
        for i in range(1, 4):
            energy = access_layer.get_node_energy(i)
            self.assertIsNotNone(energy)

    def test_concurrent_cache_access(self):
        """Test cache behavior with concurrent access patterns."""
        graph = Data()
        graph.node_labels = [{"id": i, "energy": 1.0} for i in range(100)]
        graph.x = torch.ones(100, 1, dtype=torch.float32)

        id_manager = get_id_manager()
        id_manager.reset()
        for node in graph.node_labels:
            id_manager._active_ids.add(node['id'])
        for i in range(100):
            id_manager.register_node_index(i, i)

        access_layer = NodeAccessLayer(graph, id_manager)

        # Mix cacheable and non-cacheable operations
        access_layer.get_node_by_id(0)  # Populates cache
        self.assertTrue(access_layer._cache_valid)

        access_layer.set_node_energy(0, 2.0)  # Invalidates cache
        self.assertFalse(access_layer._cache_valid)

        access_layer.get_node_by_id(0)  # Re-populates cache
        self.assertTrue(access_layer._cache_valid)


class TestNodeAccessLayerPerformance(unittest.TestCase):
    """Performance tests for NodeAccessLayer."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.large_graph = Data()
        self.large_graph.node_labels = []
        self.large_graph.x = torch.empty((0, 1), dtype=torch.float32)

        # Create large graph
        num_nodes = 1000
        for i in range(num_nodes):
            node = {"id": i, "energy": 1.0, "behavior": "dynamic"}
            self.large_graph.node_labels.append(node)
            self.large_graph.x = torch.cat([self.large_graph.x, torch.tensor([[1.0]], dtype=torch.float32)], dim=0)

        # Register all nodes
        self.id_manager = get_id_manager()
        self.id_manager.reset()
        for node in self.large_graph.node_labels:
            self.id_manager._active_ids.add(node['id'])
        for i in range(num_nodes):
            self.id_manager.register_node_index(i, i)

        self.access_layer = NodeAccessLayer(self.large_graph, self.id_manager)

    def test_large_graph_node_access_performance(self):
        """Test performance of node access on large graphs."""
        start_time = time.time()

        # Access many nodes
        for i in range(0, 1000, 10):  # Sample every 10th node
            node = self.access_layer.get_node_by_id(i)
            self.assertIsNotNone(node)

        end_time = time.time()
        access_time = end_time - start_time

        # Should complete in reasonable time (< 1 second)
        self.assertLess(access_time, 1.0)

    def test_energy_update_performance(self):
        """Test performance of energy updates."""
        start_time = time.time()

        # Update many energies
        for i in range(0, 1000, 5):  # Update every 5th node
            self.access_layer.set_node_energy(i, 2.0)

        end_time = time.time()
        update_time = end_time - start_time

        # Should complete quickly (< 0.5 seconds)
        self.assertLess(update_time, 0.5)

    def test_statistics_calculation_performance(self):
        """Test performance of statistics calculation."""
        start_time = time.time()
        stats = self.access_layer.get_node_statistics()
        end_time = time.time()

        stats_time = end_time - start_time

        # Should calculate quickly (< 0.1 seconds)
        self.assertLess(stats_time, 0.1)
        self.assertEqual(stats['total_nodes'], 1000)

    def test_cache_performance(self):
        """Test cache performance benefits."""
        # First access (cache miss)
        start_time = time.time()
        node1 = self.access_layer.get_node_by_id(0)
        first_access_time = time.time() - start_time

        # Second access (cache hit)
        start_time = time.time()
        node2 = self.access_layer.get_node_by_id(0)
        second_access_time = time.time() - start_time

        # Cached access should be faster
        self.assertLessEqual(second_access_time, first_access_time)
        self.assertEqual(node1, node2)

    def test_selection_performance(self):
        """Test performance of node selection operations."""
        start_time = time.time()
        dynamic_nodes = self.access_layer.select_nodes_by_behavior('dynamic')
        selection_time = time.time() - start_time

        # Should complete quickly (< 0.05 seconds)
        self.assertLess(selection_time, 0.05)
        self.assertEqual(len(dynamic_nodes), 1000)


class TestNodeAccessLayerRealWorldUsage(unittest.TestCase):
    """Real-world usage tests for NodeAccessLayer."""

    def test_neural_simulation_workflow(self):
        """Test typical neural simulation workflow."""
        # Create a small neural network
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)

        # Input layer
        for i in range(5):
            node = {"id": i, "type": "sensory", "behavior": "sensory", "energy": 0.5}
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[0.5]], dtype=torch.float32)], dim=0)

        # Hidden layer
        for i in range(5, 10):
            node = {"id": i, "type": "dynamic", "behavior": "integrator", "energy": 1.0}
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[1.0]], dtype=torch.float32)], dim=0)

        # Register nodes
        id_manager = get_id_manager()
        id_manager.reset()
        for node in graph.node_labels:
            id_manager._active_ids.add(node['id'])
        for i in range(10):
            id_manager.register_node_index(i, i)

        access_layer = NodeAccessLayer(graph, id_manager)

        # Simulate sensory input
        sensory_nodes = access_layer.select_nodes_by_type('sensory')
        for node_id in sensory_nodes:
            access_layer.set_node_energy(node_id, 2.0)

        # Check that inputs were set
        for node_id in sensory_nodes:
            energy = access_layer.get_node_energy(node_id)
            self.assertEqual(energy, 2.0)

        # Simulate processing
        hidden_nodes = access_layer.select_nodes_by_type('dynamic')
        for node_id in hidden_nodes:
            access_layer.update_node_property(node_id, 'plasticity_enabled', True)

        # Verify processing setup
        for node_id in hidden_nodes:
            plasticity = access_layer.get_node_property(node_id, 'plasticity_enabled')
            self.assertTrue(plasticity)

    def test_energy_monitoring_workflow(self):
        """Test energy monitoring and adjustment workflow."""
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)

        # Create nodes with varying energy levels
        for i in range(20):
            energy = 0.1 + (i * 0.2)  # 0.1 to 4.1
            node = {"id": i, "energy": energy, "behavior": "dynamic"}
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[energy]], dtype=torch.float32)], dim=0)

        # Register nodes
        id_manager = get_id_manager()
        id_manager.reset()
        for node in graph.node_labels:
            id_manager._active_ids.add(node['id'])
        for i in range(20):
            id_manager.register_node_index(i, i)

        access_layer = NodeAccessLayer(graph, id_manager)

        # Monitor energy levels
        stats = access_layer.get_node_statistics()
        self.assertAlmostEqual(stats['energy_stats']['average_energy'], 2.0, places=1)

        # Adjust low energy nodes
        low_energy_nodes = access_layer.filter_nodes(
            lambda node_id, node: access_layer.get_node_energy(node_id) < 1.0
        )

        for node_id in low_energy_nodes:
            current_energy = access_layer.get_node_energy(node_id)
            access_layer.set_node_energy(node_id, current_energy + 0.5)

        # Verify adjustments
        for node_id in low_energy_nodes:
            energy = access_layer.get_node_energy(node_id)
            self.assertGreaterEqual(energy, 0.6)  # Original + 0.5, but some were 0.1


if __name__ == '__main__':
    unittest.main()






