"""
test_id_based_system.py

Comprehensive test suite for the ID-based node simulation system.
Tests the unique ID manager, node access layer, and integration with the graph system.
"""

import unittest
import threading
import time
import torch
from torch_geometric.data import Data
from node_id_manager import NodeIDManager, get_id_manager, reset_id_manager
from node_access_layer import NodeAccessLayer
import logging


class TestNodeIDManager(unittest.TestCase):
    """Test cases for the NodeIDManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.id_manager = NodeIDManager()
    
    def tearDown(self):
        """Clean up after tests."""
        self.id_manager.reset()
    
    def test_unique_id_generation(self):
        """Test that unique IDs are generated correctly."""
        # Generate multiple IDs
        ids = []
        for i in range(100):
            node_id = self.id_manager.generate_unique_id("test")
            ids.append(node_id)
        
        # Check uniqueness
        self.assertEqual(len(ids), len(set(ids)), "Generated IDs should be unique")
        
        # Check ID range
        self.assertTrue(all(id > 0 for id in ids), "All IDs should be positive")
        self.assertEqual(min(ids), 1, "First ID should be 1")
        self.assertEqual(max(ids), 100, "Last ID should be 100")
    
    def test_id_recycling(self):
        """Test that IDs can be recycled and reused."""
        # Generate and recycle IDs
        id1 = self.id_manager.generate_unique_id("test")
        id2 = self.id_manager.generate_unique_id("test")
        
        # Recycle first ID
        self.assertTrue(self.id_manager.recycle_node_id(id1))
        self.assertFalse(self.id_manager.is_valid_id(id1))
        
        # Generate new ID (should reuse recycled ID)
        id3 = self.id_manager.generate_unique_id("test")
        self.assertEqual(id3, id1, "Recycled ID should be reused")
        self.assertTrue(self.id_manager.is_valid_id(id3))
    
    def test_id_validation(self):
        """Test ID validation functionality."""
        # Generate valid ID
        node_id = self.id_manager.generate_unique_id("test")
        self.assertTrue(self.id_manager.is_valid_id(node_id))
        
        # Test invalid ID
        self.assertFalse(self.id_manager.is_valid_id(999))
        self.assertFalse(self.id_manager.is_valid_id(0))
        self.assertFalse(self.id_manager.is_valid_id(-1))
    
    def test_node_type_tracking(self):
        """Test that node types are tracked correctly."""
        # Generate IDs with different types
        sensory_id = self.id_manager.generate_unique_id("sensory")
        dynamic_id = self.id_manager.generate_unique_id("dynamic")
        workspace_id = self.id_manager.generate_unique_id("workspace")
        
        # Check type retrieval
        self.assertEqual(self.id_manager.get_node_type(sensory_id), "sensory")
        self.assertEqual(self.id_manager.get_node_type(dynamic_id), "dynamic")
        self.assertEqual(self.id_manager.get_node_type(workspace_id), "workspace")
        
        # Check type filtering
        sensory_ids = self.id_manager.get_ids_by_type("sensory")
        self.assertIn(sensory_id, sensory_ids)
        self.assertNotIn(dynamic_id, sensory_ids)
    
    def test_index_mapping(self):
        """Test ID-to-index mapping functionality."""
        # Generate ID and register index
        node_id = self.id_manager.generate_unique_id("test")
        index = 42
        
        self.assertTrue(self.id_manager.register_node_index(node_id, index))
        
        # Test mapping retrieval
        self.assertEqual(self.id_manager.get_node_index(node_id), index)
        self.assertEqual(self.id_manager.get_node_id(index), node_id)
        
        # Test invalid mappings
        self.assertIsNone(self.id_manager.get_node_index(999))
        self.assertIsNone(self.id_manager.get_node_id(999))
    
    def test_metadata_storage(self):
        """Test metadata storage and retrieval."""
        # Generate ID with metadata
        metadata = {"x": 10, "y": 20, "energy": 100.0}
        node_id = self.id_manager.generate_unique_id("test", metadata)
        
        # Retrieve metadata
        retrieved_metadata = self.id_manager.get_node_metadata(node_id)
        self.assertEqual(retrieved_metadata, metadata)
        
        # Test metadata isolation
        node_id2 = self.id_manager.generate_unique_id("test")
        self.assertIsNone(self.id_manager.get_node_metadata(node_id2))
    
    def test_statistics(self):
        """Test statistics tracking."""
        # Generate some IDs
        for i in range(10):
            self.id_manager.generate_unique_id("test")
        
        # Get statistics
        stats = self.id_manager.get_statistics()
        
        self.assertEqual(stats['total_ids_generated'], 10)
        self.assertEqual(stats['active_ids'], 10)
        self.assertEqual(stats['recycled_ids'], 0)
        self.assertGreater(stats['uptime'], 0)
    
    def test_thread_safety(self):
        """Test thread safety of ID generation."""
        results = []
        errors = []
        
        def generate_ids():
            try:
                for i in range(50):
                    node_id = self.id_manager.generate_unique_id("thread_test")
                    results.append(node_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(4):
            thread = threading.Thread(target=generate_ids)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check for errors
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        
        # Check uniqueness across threads
        self.assertEqual(len(results), len(set(results)), "IDs should be unique across threads")
        self.assertEqual(len(results), 200, "Should generate 200 IDs total")


class TestNodeAccessLayer(unittest.TestCase):
    """Test cases for the NodeAccessLayer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test graph
        self.graph = Data(
            x=torch.tensor([[100.0], [200.0], [300.0]], dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            node_labels=[
                {"id": 1, "type": "sensory", "behavior": "sensory", "state": "active", "x": 0, "y": 0},
                {"id": 2, "type": "dynamic", "behavior": "dynamic", "state": "active"},
                {"id": 3, "type": "workspace", "behavior": "workspace", "state": "inactive", "x": 1, "y": 1}
            ]
        )
        
        # Set up ID manager
        self.id_manager = get_id_manager()
        self.id_manager.reset()
        
        # Register node indices
        for i, node in enumerate(self.graph.node_labels):
            self.id_manager.register_node_index(node["id"], i)
        
        # Create access layer
        self.access_layer = NodeAccessLayer(self.graph)
    
    def tearDown(self):
        """Clean up after tests."""
        reset_id_manager()
    
    def test_node_retrieval_by_id(self):
        """Test retrieving nodes by ID."""
        # Test valid ID
        node = self.access_layer.get_node_by_id(1)
        self.assertIsNotNone(node)
        self.assertEqual(node["type"], "sensory")
        
        # Test invalid ID
        node = self.access_layer.get_node_by_id(999)
        self.assertIsNone(node)
    
    def test_energy_operations(self):
        """Test energy get/set operations."""
        # Test getting energy
        energy = self.access_layer.get_node_energy(1)
        self.assertEqual(energy, 100.0)
        
        # Test setting energy
        self.assertTrue(self.access_layer.set_node_energy(1, 150.0))
        energy = self.access_layer.get_node_energy(1)
        self.assertEqual(energy, 150.0)
        
        # Test invalid ID
        self.assertFalse(self.access_layer.set_node_energy(999, 100.0))
    
    def test_property_operations(self):
        """Test node property get/set operations."""
        # Test getting property
        state = self.access_layer.get_node_property(1, "state")
        self.assertEqual(state, "active")
        
        # Test setting property
        self.assertTrue(self.access_layer.update_node_property(1, "state", "inactive"))
        state = self.access_layer.get_node_property(1, "state")
        self.assertEqual(state, "inactive")
        
        # Test default value
        non_existent = self.access_layer.get_node_property(1, "non_existent", "default")
        self.assertEqual(non_existent, "default")
    
    def test_node_selection(self):
        """Test node selection by various criteria."""
        # Test selection by type
        sensory_ids = self.access_layer.select_nodes_by_type("sensory")
        self.assertEqual(len(sensory_ids), 1)
        self.assertIn(1, sensory_ids)
        
        # Test selection by behavior
        dynamic_ids = self.access_layer.select_nodes_by_behavior("dynamic")
        self.assertEqual(len(dynamic_ids), 1)
        self.assertIn(2, dynamic_ids)
        
        # Test selection by state
        active_ids = self.access_layer.select_nodes_by_state("active")
        self.assertEqual(len(active_ids), 2)
        self.assertIn(1, active_ids)
        self.assertIn(2, active_ids)
    
    def test_node_iteration(self):
        """Test node iteration functionality."""
        # Test iteration over all nodes
        all_nodes = list(self.access_layer.iterate_all_nodes())
        self.assertEqual(len(all_nodes), 3)
        
        # Check that all nodes are returned
        node_ids = [node_id for node_id, _ in all_nodes]
        self.assertEqual(set(node_ids), {1, 2, 3})
        
        # Test iteration over specific IDs
        specific_ids = [1, 3]
        specific_nodes = list(self.access_layer.iterate_nodes_by_ids(specific_ids))
        self.assertEqual(len(specific_nodes), 2)
    
    def test_node_statistics(self):
        """Test node statistics generation."""
        stats = self.access_layer.get_node_statistics()
        
        self.assertEqual(stats['total_nodes'], 3)
        self.assertEqual(stats['by_type']['sensory'], 1)
        self.assertEqual(stats['by_type']['dynamic'], 1)
        self.assertEqual(stats['by_type']['workspace'], 1)
        self.assertEqual(stats['by_behavior']['sensory'], 1)
        self.assertEqual(stats['by_behavior']['dynamic'], 1)
        self.assertEqual(stats['by_behavior']['workspace'], 1)
        self.assertEqual(stats['by_state']['active'], 2)
        self.assertEqual(stats['by_state']['inactive'], 1)
        
        # Check energy statistics
        self.assertEqual(stats['energy_stats']['total_energy'], 600.0)
        self.assertEqual(stats['energy_stats']['average_energy'], 200.0)
        self.assertEqual(stats['energy_stats']['min_energy'], 100.0)
        self.assertEqual(stats['energy_stats']['max_energy'], 300.0)
    
    def test_custom_filtering(self):
        """Test custom node filtering."""
        # Filter nodes with energy > 150
        def high_energy_filter(node_id, node):
            energy = self.access_layer.get_node_energy(node_id)
            return energy is not None and energy > 150
        
        high_energy_ids = self.access_layer.filter_nodes(high_energy_filter)
        self.assertEqual(len(high_energy_ids), 2)
        self.assertIn(2, high_energy_ids)
        self.assertIn(3, high_energy_ids)
    
    def test_consistency_validation(self):
        """Test consistency validation."""
        validation = self.access_layer.validate_consistency()
        
        self.assertTrue(validation['is_consistent'])
        self.assertEqual(len(validation['errors']), 0)
        self.assertFalse(validation['node_count_mismatch'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the ID-based system."""
    
    def setUp(self):
        """Set up test fixtures."""
        reset_id_manager()
    
    def tearDown(self):
        """Clean up after tests."""
        reset_id_manager()
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from ID generation to node access."""
        # Create graph with ID-based nodes
        node_labels = []
        x_data = []
        
        # Generate IDs and create nodes
        id_manager = get_id_manager()
        for i in range(5):
            node_id = id_manager.generate_unique_id("test")
            node_labels.append({
                "id": node_id,
                "type": "test",
                "behavior": "test",
                "state": "active",
                "energy": float(100 + i * 50)
            })
            x_data.append([100 + i * 50])
            
            # Register index
            id_manager.register_node_index(node_id, i)
        
        # Create graph
        graph = Data(
            x=torch.tensor(x_data, dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            node_labels=node_labels
        )
        
        # Create access layer
        access_layer = NodeAccessLayer(graph)
        
        # Test operations
        self.assertEqual(access_layer.get_node_count(), 5)
        
        # Test node retrieval
        for i, node in enumerate(node_labels):
            retrieved_node = access_layer.get_node_by_id(node["id"])
            self.assertIsNotNone(retrieved_node)
            self.assertEqual(retrieved_node["id"], node["id"])
        
        # Test energy operations
        for i, node in enumerate(node_labels):
            energy = access_layer.get_node_energy(node["id"])
            self.assertEqual(energy, 100 + i * 50)
        
        # Test statistics
        stats = access_layer.get_node_statistics()
        self.assertEqual(stats['total_nodes'], 5)
        self.assertEqual(stats['by_type']['test'], 5)


def run_performance_tests():
    """Run performance benchmarks for the ID system."""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARKS")
    print("="*50)
    
    # Test ID generation performance
    id_manager = NodeIDManager()
    
    start_time = time.time()
    for i in range(10000):
        id_manager.generate_unique_id("perf_test")
    id_generation_time = time.time() - start_time
    
    print(f"ID Generation (10,000 IDs): {id_generation_time:.4f}s")
    print(f"IDs per second: {10000/id_generation_time:.0f}")
    
    # Test lookup performance
    active_ids = id_manager.get_all_active_ids()
    
    start_time = time.time()
    for i in range(10000):
        node_id = active_ids[i % len(active_ids)]
        id_manager.get_node_index(node_id)
    lookup_time = time.time() - start_time
    
    print(f"ID Lookup (10,000 lookups): {lookup_time:.4f}s")
    print(f"Lookups per second: {10000/lookup_time:.0f}")
    
    # Test statistics
    stats = id_manager.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"Total IDs generated: {stats['total_ids_generated']}")
    print(f"Active IDs: {stats['active_ids']}")
    print(f"Lookup operations: {stats['lookup_operations']}")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run unit tests
    print("Running ID-Based System Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
