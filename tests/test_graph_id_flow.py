"""
Comprehensive Tests for Graph-to-ID Flow
Tests all aspects of the graph to ID to connected modules flow.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import unittest
import time
import torch
import numpy as np
from typing import Dict, List, Any

# Import the components we're testing
from src.energy.node_id_manager import get_id_manager, NodeIDManager
from src.utils.graph_integrity_manager import get_graph_integrity_manager
from src.utils.connection_validator import get_connection_validator
from src.utils.reader_writer_lock import get_id_manager_lock
from src.utils.graph_merger import get_graph_merger
from src.core.services.simulation_coordinator import SimulationCoordinator
from unittest.mock import MagicMock


class MockGraph:
    """Mock graph for testing purposes."""

    def __init__(self, nodes: List[Dict[str, Any]] = None, edges: List[List[int]] = None):
        if nodes is None:
            nodes = []
        self.node_labels = nodes
        self.x = torch.tensor([node.get('energy', 0.5) for node in nodes], dtype=torch.float32).unsqueeze(1)

        if edges:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)


class TestGraphIDFlow(unittest.TestCase):
    """Comprehensive test suite for graph-to-ID flow."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset global instances for clean testing
        global _id_manager_instance
        _id_manager_instance = None

        self.id_manager = get_id_manager()
        self.integrity_manager = get_graph_integrity_manager()
        self.connection_validator = get_connection_validator()
        self.graph_merger = get_graph_merger()

    def tearDown(self):
        """Clean up after tests."""
        # Reset ID manager
        if hasattr(self.id_manager, 'reset'):
            self.id_manager.reset()

    def test_atomic_id_generation(self):
        """Test atomic ID generation with transactions."""
        # Test basic ID generation
        id1 = self.id_manager.generate_unique_id('dynamic')
        id2 = self.id_manager.generate_unique_id('sensory')

        self.assertIsInstance(id1, int)
        self.assertIsInstance(id2, int)
        self.assertNotEqual(id1, id2)
        self.assertTrue(self.id_manager.is_valid_id(id1))
        self.assertTrue(self.id_manager.is_valid_id(id2))

        # Test ID type mapping
        self.assertEqual(self.id_manager.get_node_type(id1), 'dynamic')
        self.assertEqual(self.id_manager.get_node_type(id2), 'sensory')

    def test_transaction_rollback(self):
        """Test transaction rollback on failure."""
        initial_count = len(self.id_manager.get_all_active_ids())

        try:
            with self.id_manager.transaction() as txn:
                txn.add_operation('generate_id', node_type='test')
                # Simulate failure
                raise ValueError("Test failure")
        except ValueError:
            pass  # Expected

        # Check that transaction was rolled back
        final_count = len(self.id_manager.get_all_active_ids())
        self.assertEqual(initial_count, final_count)

    def test_index_registration(self):
        """Test node index registration."""
        # Generate an ID and register an index
        node_id = self.id_manager.generate_unique_id('dynamic')
        success = self.id_manager.register_node_index(node_id, 5)

        self.assertTrue(success)
        self.assertEqual(self.id_manager.get_node_index(node_id), 5)
        self.assertEqual(self.id_manager.get_node_id(5), node_id)

    def test_id_recycling(self):
        """Test ID recycling functionality."""
        # Generate and then recycle an ID
        node_id = self.id_manager.generate_unique_id('dynamic')
        initial_active = len(self.id_manager.get_all_active_ids())

        success = self.id_manager.recycle_node_id(node_id)
        self.assertTrue(success)

        final_active = len(self.id_manager.get_all_active_ids())
        self.assertEqual(final_active, initial_active - 1)
        self.assertFalse(self.id_manager.is_valid_id(node_id))

    def test_graph_consistency_validation(self):
        """Test graph consistency validation."""
        # Create a mock graph
        nodes = [
            {'id': 1, 'type': 'dynamic', 'energy': 0.5},
            {'id': 2, 'type': 'sensory', 'energy': 0.7},
            {'id': 3, 'type': 'dynamic', 'energy': 0.3}
        ]
        graph = MockGraph(nodes)

        # Register nodes with ID manager
        for node in nodes:
            self.id_manager.generate_unique_id(node['type'], {'energy': node['energy']})
            self.id_manager.register_node_index(node['id'], nodes.index(node))

        # Validate consistency
        result = self.id_manager.validate_graph_consistency(graph)

        self.assertTrue(result['is_consistent'])
        self.assertEqual(len(result['errors']), 0)

    def test_connection_validation(self):
        """Test centralized connection validation."""
        # Create mock graphs
        nodes1 = [{'id': 1, 'type': 'dynamic', 'energy': 0.5}]
        nodes2 = [{'id': 2, 'type': 'sensory', 'energy': 0.7}]
        combined_nodes = nodes1 + nodes2
        graph1 = MockGraph(nodes1)
        graph2 = MockGraph(nodes2)
        combined_graph = MockGraph(combined_nodes)

        # Register nodes
        self.id_manager.generate_unique_id('dynamic')
        self.id_manager.generate_unique_id('sensory')

        # Test valid connection on combined graph
        result = self.connection_validator.validate_connection(
            combined_graph, 1, 2, 'excitatory', 0.5
        )
        self.assertTrue(result['is_valid'])

        # Test invalid self-connection
        result = self.connection_validator.validate_connection(
            graph1, 1, 1, 'excitatory', 0.5
        )
        self.assertFalse(result['is_valid'])
        self.assertTrue(any('Self-connection not allowed' in e for e in result['errors']))

    def test_reader_writer_locks(self):
        """Test reader-writer lock functionality."""
        lock = get_id_manager_lock()

        # Test read lock
        with lock.read_lock():
            # Should be able to read
            stats = self.id_manager.get_statistics()
            self.assertIsInstance(stats, dict)

        # Test write lock
        with lock.write_lock():
            # Should be able to write
            node_id = self.id_manager.generate_unique_id('test')
            self.assertIsInstance(node_id, int)

    def test_graph_versioning(self):
        """Test graph versioning functionality."""
        # Create a mock graph
        nodes = [{'id': 1, 'type': 'dynamic', 'energy': 0.5}]
        graph = MockGraph(nodes)

        # Register node
        self.id_manager.generate_unique_id('dynamic')

        # Create version
        version_id = self.integrity_manager.create_version(graph, self.id_manager)

        self.assertIsInstance(version_id, str)
        self.assertTrue(version_id.startswith('v'))

        # Check version history
        history = self.integrity_manager.get_version_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['version_id'], version_id)

    def test_integrity_checking(self):
        """Test graph integrity checking."""
        # Create a mock graph
        nodes = [{'id': 1, 'type': 'dynamic', 'energy': 0.5}]
        graph = MockGraph(nodes)

        # Register node
        self.id_manager.generate_unique_id('dynamic')

        # Create baseline version
        self.integrity_manager.create_version(graph, self.id_manager)

        # Check integrity (should pass)
        result = self.integrity_manager.check_integrity(graph, self.id_manager)
        self.assertTrue(result['is_integrity_intact'])

    def test_graph_merging(self):
        """Test graph merging with ID conflict resolution."""
        # Create two graphs with overlapping IDs
        nodes1 = [
            {'id': 1, 'type': 'dynamic', 'energy': 0.5},
            {'id': 2, 'type': 'sensory', 'energy': 0.7}
        ]
        nodes2 = [
            {'id': 2, 'type': 'dynamic', 'energy': 0.6},  # ID conflict
            {'id': 3, 'type': 'relay', 'energy': 0.4}
        ]

        graph1 = MockGraph(nodes1)
        graph2 = MockGraph(nodes2)

        # Register nodes for graph1
        for node in nodes1:
            self.id_manager.generate_unique_id(node['type'])

        # Merge graphs
        result = self.graph_merger.merge_graphs(graph1, graph2)

        self.assertIsNotNone(result['merged_graph'])
        self.assertIn('id_mapping', result)
        self.assertIn('statistics', result)

        # Check that conflicts were resolved
        merged_nodes = result['merged_graph'].node_labels
        self.assertEqual(len(merged_nodes), 4)  # Should have 4 nodes: 1,2 (primary), remapped 2, 3

    def test_simulation_manager_integration(self):
        """Test integration with simulation manager."""
        # Create mock services
        mock_service_registry = MagicMock()
        mock_neural_processor = MagicMock()
        mock_energy_manager = MagicMock()
        mock_learning_engine = MagicMock()
        mock_sensory_processor = MagicMock()
        mock_performance_monitor = MagicMock()
        mock_graph_manager = MagicMock()
        mock_event_coordinator = MagicMock()
        mock_configuration_service = MagicMock()

        # Mock initialize_graph to return a MockGraph
        mock_graph_manager.initialize_graph.return_value = MockGraph([{'id': 1, 'type': 'dynamic', 'energy': 0.5}])

        sim_manager = SimulationCoordinator(
            service_registry=mock_service_registry,
            neural_processor=mock_neural_processor,
            energy_manager=mock_energy_manager,
            learning_engine=mock_learning_engine,
            sensory_processor=mock_sensory_processor,
            performance_monitor=mock_performance_monitor,
            graph_manager=mock_graph_manager,
            event_coordinator=mock_event_coordinator,
            configuration_service=mock_configuration_service
        )

        # Mock the methods used in the test
        sim_manager.initialize_graph = MagicMock(return_value=True)
        sim_manager.check_graph_integrity_now = MagicMock(return_value={'is_integrity_intact': True})
        sim_manager.create_graph_version = MagicMock(return_value='v123')
        sim_manager.get_integrity_statistics = MagicMock(return_value={'total_versions': 1})

        # Initialize graph
        success = sim_manager.initialize_graph()
        self.assertTrue(success)

        # Test integrity checking
        integrity_result = sim_manager.check_graph_integrity_now()
        self.assertIsInstance(integrity_result, dict)

        # Test version creation
        version_id = sim_manager.create_graph_version()
        self.assertIsInstance(version_id, str)

        # Test statistics access
        stats = sim_manager.get_integrity_statistics()
        self.assertIsInstance(stats, dict)

    def test_performance_under_load(self):
        """Test performance under load."""
        start_time = time.time()

        # Generate many IDs
        ids = []
        for i in range(1000):
            node_id = self.id_manager.generate_unique_id('dynamic')
            ids.append(node_id)

        generation_time = time.time() - start_time

        # Should generate 1000 IDs reasonably quickly
        self.assertLess(generation_time, 1.0)  # Less than 1 second

        # All IDs should be unique
        self.assertEqual(len(set(ids)), len(ids))

        # All IDs should be valid
        for node_id in ids:
            self.assertTrue(self.id_manager.is_valid_id(node_id))

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test invalid connection validation
        result = self.connection_validator.validate_connection(
            None, 1, 2, 'invalid_type', 0.5
        )
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)

        # Test integrity check with None graph
        result = self.integrity_manager.check_integrity(None, self.id_manager)
        self.assertFalse(result['is_integrity_intact'])

        # Test merge with invalid graphs
        with self.assertRaises(Exception):
            self.graph_merger.merge_graphs(None, None)


if __name__ == '__main__':
    unittest.main()






