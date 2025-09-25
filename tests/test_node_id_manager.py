"""
Comprehensive tests for NodeIDManager.

This module contains unit tests, integration tests, edge cases, and performance tests
for the NodeIDManager class, covering ID generation, indexing, transactions, and integrity checks.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import time
import threading
from unittest.mock import Mock, patch
from torch_geometric.data import Data
import torch

from src.energy.node_id_manager import (
    NodeIDManager, get_id_manager, reset_id_manager, force_reset_id_manager,
    generate_node_id, get_node_index_by_id, get_node_id_by_index,
    is_valid_node_id, recycle_node_id, IDTransaction
)


class TestIDTransaction(unittest.TestCase):
    """Unit tests for IDTransaction class."""

    def test_transaction_initialization(self):
        """Test transaction initialization."""
        txn = IDTransaction("test_txn")
        self.assertEqual(txn.transaction_id, "test_txn")
        self.assertIsInstance(txn.operations, list)
        self.assertEqual(len(txn.operations), 0)
        self.assertGreater(txn.timestamp, 0)

    def test_add_operation(self):
        """Test adding operations to transaction."""
        txn = IDTransaction("test_txn")
        txn.add_operation("generate_id", node_type="dynamic", metadata={"test": True})

        self.assertEqual(len(txn.operations), 1)
        op = txn.operations[0]
        self.assertEqual(op["type"], "generate_id")
        self.assertEqual(op["node_type"], "dynamic")
        self.assertEqual(op["metadata"], {"test": True})
        self.assertIn("timestamp", op)


class TestNodeIDManagerInitialization(unittest.TestCase):
    """Unit tests for NodeIDManager initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        self.assertEqual(self.manager._max_graph_size, 1000000)
        self.assertEqual(self.manager._next_id, 1)
        self.assertEqual(len(self.manager._active_ids), 0)
        self.assertEqual(len(self.manager._recycled_ids), 0)
        self.assertIsInstance(self.manager._lock, type(self.manager._lock))
        self.assertTrue(self.manager._integrity_check_enabled)
        self.assertEqual(self.manager._metadata_size_limit, 1024)

    def test_initialization_thread_safety(self):
        """Test that initialization sets up thread-safe components."""
        self.assertIsNotNone(self.manager._lock)
        self.assertIsNotNone(self.manager._transaction_lock)
        self.assertIsNotNone(self.manager._stats_lock)


class TestNodeIDManagerBasicOperations(unittest.TestCase):
    """Unit tests for basic NodeIDManager operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

    def test_generate_unique_id_basic(self):
        """Test basic ID generation."""
        node_id = self.manager.generate_unique_id("dynamic")
        self.assertIsInstance(node_id, int)
        self.assertGreater(node_id, 0)
        self.assertIn(node_id, self.manager._active_ids)
        self.assertEqual(self.manager._node_type_map[node_id], "dynamic")

    def test_generate_unique_id_with_metadata(self):
        """Test ID generation with metadata."""
        metadata = {"layer": "input", "index": 0}
        node_id = self.manager.generate_unique_id("sensory", metadata)

        self.assertIn(node_id, self.manager._active_ids)
        self.assertEqual(self.manager._id_metadata[node_id], metadata)

    def test_generate_unique_id_large_metadata(self):
        """Test ID generation with large metadata (should be truncated)."""
        large_metadata = {"data": "x" * 2000}  # Over 1KB
        node_id = self.manager.generate_unique_id("dynamic", large_metadata)

        # Should still generate ID but metadata should be modified
        self.assertIn(node_id, self.manager._active_ids)
        stored_metadata = self.manager._id_metadata.get(node_id)
        self.assertIsNotNone(stored_metadata)
        self.assertIn("size_exceeded", stored_metadata)

    def test_register_node_index(self):
        """Test node index registration."""
        node_id = self.manager.generate_unique_id("dynamic")
        success = self.manager.register_node_index(node_id, 5)

        self.assertTrue(success)
        self.assertEqual(self.manager._id_to_index[node_id], 5)
        self.assertEqual(self.manager._index_to_id[5], node_id)

    def test_register_invalid_node_index(self):
        """Test registering index for invalid node ID."""
        success = self.manager.register_node_index(999, 5)
        self.assertFalse(success)

    def test_get_node_index(self):
        """Test getting node index."""
        node_id = self.manager.generate_unique_id("dynamic")
        self.manager.register_node_index(node_id, 10)

        index = self.manager.get_node_index(node_id)
        self.assertEqual(index, 10)

    def test_get_node_index_invalid(self):
        """Test getting index for invalid node ID."""
        index = self.manager.get_node_index(999)
        self.assertIsNone(index)

    def test_get_node_id_by_index(self):
        """Test getting node ID by index."""
        node_id = self.manager.generate_unique_id("dynamic")
        self.manager.register_node_index(node_id, 15)

        retrieved_id = self.manager.get_node_id(15)
        self.assertEqual(retrieved_id, node_id)

    def test_is_valid_id(self):
        """Test ID validation."""
        valid_id = self.manager.generate_unique_id("dynamic")
        self.assertTrue(self.manager.is_valid_id(valid_id))
        self.assertFalse(self.manager.is_valid_id(999))

    def test_get_node_type(self):
        """Test getting node type."""
        node_id = self.manager.generate_unique_id("oscillator")
        node_type = self.manager.get_node_type(node_id)
        self.assertEqual(node_type, "oscillator")

    def test_get_node_metadata(self):
        """Test getting node metadata."""
        metadata = {"test": "value"}
        node_id = self.manager.generate_unique_id("dynamic", metadata)
        retrieved_metadata = self.manager.get_node_metadata(node_id)
        self.assertEqual(retrieved_metadata, metadata)

    def test_recycle_node_id(self):
        """Test node ID recycling."""
        node_id = self.manager.generate_unique_id("dynamic")
        self.manager.register_node_index(node_id, 0)

        success = self.manager.recycle_node_id(node_id)
        self.assertTrue(success)
        self.assertNotIn(node_id, self.manager._active_ids)
        self.assertIn(node_id, self.manager._recycled_ids)
        self.assertNotIn(node_id, self.manager._id_to_index)

    def test_recycle_invalid_node_id(self):
        """Test recycling invalid node ID."""
        success = self.manager.recycle_node_id(999)
        self.assertFalse(success)


class TestNodeIDManagerSelectionMethods(unittest.TestCase):
    """Unit tests for ID manager selection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

        # Create test nodes
        self.node_ids = []
        types = ["dynamic", "sensory", "oscillator", "dynamic", "sensory"]
        for i, node_type in enumerate(types):
            node_id = self.manager.generate_unique_id(node_type)
            self.node_ids.append(node_id)

    def test_get_all_active_ids(self):
        """Test getting all active IDs."""
        active_ids = self.manager.get_all_active_ids()
        self.assertEqual(len(active_ids), 5)
        self.assertEqual(set(active_ids), set(self.node_ids))

    def test_get_ids_by_type(self):
        """Test getting IDs by type."""
        dynamic_ids = self.manager.get_ids_by_type("dynamic")
        sensory_ids = self.manager.get_ids_by_type("sensory")
        oscillator_ids = self.manager.get_ids_by_type("oscillator")

        self.assertEqual(len(dynamic_ids), 2)
        self.assertEqual(len(sensory_ids), 2)
        self.assertEqual(len(oscillator_ids), 1)


class TestNodeIDManagerTransactions(unittest.TestCase):
    """Unit tests for transaction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

    def test_transaction_context_manager(self):
        """Test transaction context manager."""
        with self.manager.transaction("test_txn") as txn:
            self.assertIsInstance(txn, IDTransaction)
            self.assertEqual(txn.transaction_id, "test_txn")

            # Add operations
            txn.add_operation("generate_id", node_type="dynamic")
            txn.add_operation("register_index", node_id=1, index=0)

        # Transaction should be committed
        self.assertIn("test_txn", [t.transaction_id for t in self.manager._completed_transactions])

    def test_transaction_rollback_on_exception(self):
        """Test transaction rollback on exception."""
        initial_active_count = len(self.manager._active_ids)

        try:
            with self.manager.transaction("fail_txn") as txn:
                txn.add_operation("generate_id", node_type="dynamic")
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should not have committed
        self.assertEqual(len(self.manager._active_ids), initial_active_count)
        self.assertNotIn("fail_txn", [t.transaction_id for t in self.manager._completed_transactions])

    def test_transaction_with_operations(self):
        """Test transaction with actual operations."""
        with self.manager.transaction("multi_op_txn") as txn:
            txn.add_operation("generate_id", node_type="dynamic")
            txn.add_operation("generate_id", node_type="sensory")

        # Should have created 2 nodes
        self.assertEqual(len(self.manager._active_ids), 2)

        # Check types
        node_ids = list(self.manager._active_ids)
        types = [self.manager.get_node_type(nid) for nid in node_ids]
        self.assertIn("dynamic", types)
        self.assertIn("sensory", types)


class TestNodeIDManagerStatistics(unittest.TestCase):
    """Unit tests for statistics functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

        # Generate some activity
        for i in range(10):
            self.manager.generate_unique_id("dynamic")

        for i in range(3):
            self.manager.recycle_node_id(list(self.manager._active_ids)[i])

    def test_get_statistics(self):
        """Test getting comprehensive statistics."""
        stats = self.manager.get_statistics()

        self.assertIn("total_ids_generated", stats)
        self.assertIn("active_ids", stats)
        self.assertIn("recycled_ids", stats)
        self.assertIn("memory_usage_mb", stats)
        self.assertIn("uptime", stats)
        self.assertIn("utilization_percent", stats)

        self.assertEqual(stats["total_ids_generated"], 10)
        self.assertEqual(stats["active_ids"], 7)  # 10 - 3 recycled
        self.assertEqual(stats["recycled_ids"], 3)

    def test_get_transaction_history(self):
        """Test getting transaction history."""
        # Create some transactions
        with self.manager.transaction("txn1") as txn:
            txn.add_operation("generate_id", node_type="test")

        with self.manager.transaction("txn2") as txn:
            txn.add_operation("generate_id", node_type="test")

        history = self.manager.get_transaction_history()
        self.assertGreaterEqual(len(history), 2)

        # Check most recent transactions
        recent_txns = [h["transaction_id"] for h in history]
        self.assertIn("txn1", recent_txns)
        self.assertIn("txn2", recent_txns)


class TestNodeIDManagerIntegrity(unittest.TestCase):
    """Unit tests for integrity checking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

        # Create test graph
        self.graph = Data()
        self.graph.node_labels = []

        # Create nodes and register them
        for i in range(5):
            node_id = self.manager.generate_unique_id("dynamic")
            self.manager.register_node_index(node_id, i)
            self.graph.node_labels.append({"id": node_id, "type": "dynamic"})

    def test_validate_graph_consistency_valid(self):
        """Test validation of consistent graph."""
        result = self.manager.validate_graph_consistency(self.graph)

        self.assertTrue(result["is_consistent"])
        self.assertEqual(len(result["errors"]), 0)
        self.assertEqual(len(result["warnings"]), 0)

    def test_validate_graph_consistency_missing_node_labels(self):
        """Test validation with missing node_labels."""
        invalid_graph = Data()  # No node_labels

        result = self.manager.validate_graph_consistency(invalid_graph)
        self.assertFalse(result["is_consistent"])
        self.assertIn("Graph missing node_labels", result["errors"])

    def test_validate_graph_consistency_missing_ids(self):
        """Test validation with nodes missing IDs."""
        invalid_graph = Data()
        invalid_graph.node_labels = [
            {"type": "dynamic"},  # Missing id
            {"id": 1, "type": "dynamic"}
        ]

        result = self.manager.validate_graph_consistency(invalid_graph)
        self.assertFalse(result["is_consistent"])
        self.assertGreater(len(result["missing_ids"]), 0)

    def test_validate_graph_consistency_invalid_ids(self):
        """Test validation with invalid node IDs."""
        invalid_graph = Data()
        invalid_graph.node_labels = [
            {"id": 999, "type": "dynamic"}  # Invalid ID
        ]

        result = self.manager.validate_graph_consistency(invalid_graph)
        self.assertFalse(result["is_consistent"])
        self.assertGreater(len(result["errors"]), 0)

    def test_perform_integrity_check(self):
        """Test integrity check execution."""
        result = self.manager._perform_integrity_check()
        self.assertIsInstance(result, dict)
        self.assertIn("passed", result)
        self.assertIn("errors", result)

    def test_enable_integrity_checks(self):
        """Test enabling/disabling integrity checks."""
        self.manager.enable_integrity_checks(False)
        self.assertFalse(self.manager._integrity_check_enabled)

        self.manager.enable_integrity_checks(True, 120.0)
        self.assertTrue(self.manager._integrity_check_enabled)
        self.assertEqual(self.manager._integrity_check_interval, 120.0)


class TestNodeIDManagerCapacity(unittest.TestCase):
    """Unit tests for capacity management."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

    def test_can_expand_graph(self):
        """Test graph expansion capability."""
        self.assertTrue(self.manager.can_expand_graph(10))

        # Fill up to near capacity
        for i in range(999990):  # Leave room for test
            self.manager.generate_unique_id("test")

        # Should still allow small expansion
        self.assertTrue(self.manager.can_expand_graph(5))

    def test_get_expansion_capacity(self):
        """Test getting expansion capacity."""
        initial_capacity = self.manager.get_expansion_capacity()
        self.assertEqual(initial_capacity, 1000000)

        # Use some capacity
        self.manager.generate_unique_id("test")
        new_capacity = self.manager.get_expansion_capacity()
        self.assertEqual(new_capacity, 999999)

    def test_set_max_graph_size(self):
        """Test setting maximum graph size."""
        self.manager.set_max_graph_size(50000)
        self.assertEqual(self.manager._max_graph_size, 50000)

        capacity = self.manager.get_expansion_capacity()
        self.assertEqual(capacity, 50000)


class TestNodeIDManagerGlobalFunctions(unittest.TestCase):
    """Unit tests for global NodeIDManager functions."""

    def setUp(self):
        """Reset global manager before each test."""
        reset_id_manager()

    def test_get_id_manager_singleton(self):
        """Test singleton behavior of get_id_manager."""
        manager1 = get_id_manager()
        manager2 = get_id_manager()
        self.assertIs(manager1, manager2)

    def test_reset_id_manager(self):
        """Test resetting global ID manager."""
        manager = get_id_manager()
        manager.generate_unique_id("test")
        self.assertEqual(len(manager._active_ids), 1)

        reset_id_manager()
        new_manager = get_id_manager()
        self.assertEqual(len(new_manager._active_ids), 0)

    def test_force_reset_id_manager(self):
        """Test force resetting global ID manager."""
        manager1 = get_id_manager()
        manager1.generate_unique_id("test")

        force_reset_id_manager()
        manager2 = get_id_manager()

        self.assertIsNot(manager1, manager2)
        self.assertEqual(len(manager2._active_ids), 0)

    def test_generate_node_id_function(self):
        """Test global generate_node_id function."""
        node_id = generate_node_id("dynamic")
        self.assertIsInstance(node_id, int)
        self.assertGreater(node_id, 0)

    def test_get_node_index_by_id_function(self):
        """Test global get_node_index_by_id function."""
        manager = get_id_manager()
        node_id = manager.generate_unique_id("dynamic")
        manager.register_node_index(node_id, 5)

        index = get_node_index_by_id(node_id)
        self.assertEqual(index, 5)

    def test_get_node_id_by_index_function(self):
        """Test global get_node_id_by_index function."""
        manager = get_id_manager()
        node_id = manager.generate_unique_id("dynamic")
        manager.register_node_index(node_id, 7)

        retrieved_id = get_node_id_by_index(7)
        self.assertEqual(retrieved_id, node_id)

    def test_is_valid_node_id_function(self):
        """Test global is_valid_node_id function."""
        manager = get_id_manager()
        valid_id = manager.generate_unique_id("dynamic")

        self.assertTrue(is_valid_node_id(valid_id))
        self.assertFalse(is_valid_node_id(999))

    def test_recycle_node_id_function(self):
        """Test global recycle_node_id function."""
        manager = get_id_manager()
        node_id = manager.generate_unique_id("dynamic")

        success = recycle_node_id(node_id)
        self.assertTrue(success)
        self.assertNotIn(node_id, manager._active_ids)


class TestNodeIDManagerEdgeCases(unittest.TestCase):
    """Edge case tests for NodeIDManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

    def test_generate_id_with_invalid_type(self):
        """Test ID generation with invalid type."""
        # Should handle gracefully
        node_id = self.manager.generate_unique_id(123)  # Invalid type
        self.assertIsInstance(node_id, int)

        # Type should be converted to string
        node_type = self.manager.get_node_type(node_id)
        self.assertEqual(node_type, "123")

    def test_generate_id_with_none_metadata(self):
        """Test ID generation with None metadata."""
        node_id = self.manager.generate_unique_id("dynamic", None)
        self.assertIsInstance(node_id, int)
        self.assertNotIn(node_id, self.manager._id_metadata)

    def test_register_duplicate_index(self):
        """Test registering duplicate index."""
        node_id1 = self.manager.generate_unique_id("dynamic")
        node_id2 = self.manager.generate_unique_id("sensory")

        # Register first node to index 0
        self.manager.register_node_index(node_id1, 0)

        # Try to register second node to same index
        success = self.manager.register_node_index(node_id2, 0)
        self.assertTrue(success)  # Should succeed but overwrite

        # Index should now point to second node
        retrieved_id = self.manager.get_node_id(0)
        self.assertEqual(retrieved_id, node_id2)

    def test_recycle_nonexistent_id(self):
        """Test recycling non-existent ID."""
        success = self.manager.recycle_node_id(99999)
        self.assertFalse(success)

    def test_get_statistics_empty_manager(self):
        """Test statistics with empty manager."""
        stats = self.manager.get_statistics()
        self.assertEqual(stats["active_ids"], 0)
        self.assertEqual(stats["total_ids_generated"], 0)

    def test_transaction_with_empty_operations(self):
        """Test transaction with no operations."""
        with self.manager.transaction("empty_txn") as txn:
            pass  # No operations

        # Should still be recorded
        self.assertIn("empty_txn", [t.transaction_id for t in self.manager._completed_transactions])


class TestNodeIDManagerPerformance(unittest.TestCase):
    """Performance tests for NodeIDManager."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.manager = NodeIDManager()

    def test_id_generation_performance(self):
        """Test performance of ID generation."""
        start_time = time.time()

        # Generate many IDs
        for i in range(1000):
            self.manager.generate_unique_id("dynamic")

        end_time = time.time()
        generation_time = end_time - start_time

        # Should complete in reasonable time (< 1 second)
        self.assertLess(generation_time, 1.0)
        self.assertEqual(len(self.manager._active_ids), 1000)

    def test_id_lookup_performance(self):
        """Test performance of ID lookups."""
        # Create many IDs
        node_ids = []
        for i in range(1000):
            node_id = self.manager.generate_unique_id("dynamic")
            self.manager.register_node_index(node_id, i)
            node_ids.append(node_id)

        start_time = time.time()

        # Lookup all IDs
        for node_id in node_ids:
            index = self.manager.get_node_index(node_id)
            self.assertIsNotNone(index)

        end_time = time.time()
        lookup_time = end_time - start_time

        # Should complete quickly (< 0.1 seconds)
        self.assertLess(lookup_time, 0.1)

    def test_transaction_performance(self):
        """Test performance of transactions."""
        start_time = time.time()

        # Perform many transactions
        for i in range(100):
            with self.manager.transaction(f"perf_txn_{i}") as txn:
                for j in range(5):
                    txn.add_operation("generate_id", node_type="test")

        end_time = time.time()
        transaction_time = end_time - start_time

        # Should complete in reasonable time (< 2 seconds)
        self.assertLess(transaction_time, 2.0)
        self.assertEqual(len(self.manager._active_ids), 500)  # 100 txns * 5 IDs each

    def test_statistics_performance(self):
        """Test performance of statistics calculation."""
        # Create some data
        for i in range(100):
            self.manager.generate_unique_id("dynamic")

        start_time = time.time()
        stats = self.manager.get_statistics()
        end_time = time.time()

        stats_time = end_time - start_time

        # Should calculate quickly (< 0.01 seconds)
        self.assertLess(stats_time, 0.01)
        self.assertEqual(stats["active_ids"], 100)

    def test_concurrent_access_simulation(self):
        """Test simulated concurrent access."""
        results = []

        def worker():
            for i in range(50):
                node_id = self.manager.generate_unique_id("test")
                results.append(node_id)

        # Simulate concurrent access with threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have created 250 unique IDs
        self.assertEqual(len(self.manager._active_ids), 250)
        self.assertEqual(len(results), 250)
        self.assertEqual(len(set(results)), 250)  # All unique


class TestNodeIDManagerRealWorldUsage(unittest.TestCase):
    """Real-world usage tests for NodeIDManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = NodeIDManager()

    def test_neural_network_node_creation(self):
        """Test creating nodes for a neural network."""
        # Create different types of nodes
        input_nodes = []
        hidden_nodes = []
        output_nodes = []

        # Input layer
        for i in range(10):
            node_id = self.manager.generate_unique_id("sensory", {"layer": "input", "index": i})
            input_nodes.append(node_id)

        # Hidden layer
        for i in range(20):
            node_id = self.manager.generate_unique_id("dynamic", {"layer": "hidden", "index": i})
            hidden_nodes.append(node_id)

        # Output layer
        for i in range(5):
            node_id = self.manager.generate_unique_id("oscillator", {"layer": "output", "index": i})
            output_nodes.append(node_id)

        # Verify creation
        self.assertEqual(len(self.manager._active_ids), 35)
        self.assertEqual(len(input_nodes), 10)
        self.assertEqual(len(hidden_nodes), 20)
        self.assertEqual(len(output_nodes), 5)

        # Verify types
        for node_id in input_nodes:
            self.assertEqual(self.manager.get_node_type(node_id), "sensory")

        for node_id in hidden_nodes:
            self.assertEqual(self.manager.get_node_type(node_id), "dynamic")

        for node_id in output_nodes:
            self.assertEqual(self.manager.get_node_type(node_id), "oscillator")

    def test_node_lifecycle_management(self):
        """Test complete node lifecycle."""
        # Create nodes
        active_nodes = []
        for i in range(10):
            node_id = self.manager.generate_unique_id("dynamic")
            active_nodes.append(node_id)

        # Register indices
        for i, node_id in enumerate(active_nodes):
            self.manager.register_node_index(node_id, i)

        # Verify active
        for node_id in active_nodes:
            self.assertTrue(self.manager.is_valid_id(node_id))
            self.assertIsNotNone(self.manager.get_node_index(node_id))

        # Recycle some nodes
        recycled_nodes = active_nodes[:3]
        for node_id in recycled_nodes:
            self.manager.recycle_node_id(node_id)

        # Verify recycled
        for node_id in recycled_nodes:
            self.assertFalse(self.manager.is_valid_id(node_id))
            self.assertIsNone(self.manager.get_node_index(node_id))

        # Create new nodes (should reuse recycled IDs)
        new_nodes = []
        for i in range(3):
            node_id = self.manager.generate_unique_id("sensory")
            new_nodes.append(node_id)

        # Verify new nodes are valid
        for node_id in new_nodes:
            self.assertTrue(self.manager.is_valid_id(node_id))

        # Total active should be 10 (10 - 3 + 3)
        self.assertEqual(len(self.manager._active_ids), 10)

    def test_large_scale_network_management(self):
        """Test managing a large-scale network."""
        # Create many nodes
        node_ids = []
        for i in range(1000):
            node_type = ["sensory", "dynamic", "oscillator", "integrator"][i % 4]
            node_id = self.manager.generate_unique_id(node_type)
            node_ids.append(node_id)

        # Register indices
        for i, node_id in enumerate(node_ids):
            self.manager.register_node_index(node_id, i)

        # Verify statistics
        stats = self.manager.get_statistics()
        self.assertEqual(stats["active_ids"], 1000)
        self.assertGreaterEqual(stats["total_ids_generated"], 1000)

        # Test type-based selection
        sensory_ids = self.manager.get_ids_by_type("sensory")
        dynamic_ids = self.manager.get_ids_by_type("dynamic")
        oscillator_ids = self.manager.get_ids_by_type("oscillator")
        integrator_ids = self.manager.get_ids_by_type("integrator")

        self.assertEqual(len(sensory_ids), 250)  # 1000 / 4
        self.assertEqual(len(dynamic_ids), 250)
        self.assertEqual(len(oscillator_ids), 250)
        self.assertEqual(len(integrator_ids), 250)

        # Test graph consistency
        graph = Data()
        graph.node_labels = [{"id": node_id, "type": self.manager.get_node_type(node_id)}
                           for node_id in node_ids]

        consistency = self.manager.validate_graph_consistency(graph)
        self.assertTrue(consistency["is_consistent"])


if __name__ == '__main__':
    unittest.main()






