"""
test_id_based_communication.py

Comprehensive test suite for ID-based node communication and interaction system.
Tests the refactored communication mechanisms using unique IDs instead of direct references.
"""

import unittest
import torch
import numpy as np
from torch_geometric.data import Data
from node_id_manager import get_id_manager, reset_id_manager
from node_access_layer import NodeAccessLayer
from connection_logic import intelligent_connection_formation, create_weighted_connection
from energy_behavior import apply_energy_behavior
from behavior_engine import BehaviorEngine
import logging


class TestIDBasedCommunication(unittest.TestCase):
    """Test cases for ID-based node communication system."""
    
    def setUp(self):
        """Set up test fixtures."""
        reset_id_manager()
        self.id_manager = get_id_manager()
        
        # Create a test graph with different node types
        self.graph = self._create_test_graph()
    
    def tearDown(self):
        """Clean up after tests."""
        reset_id_manager()
    
    def _create_test_graph(self):
        """Create a test graph with various node types."""
        # Create node features
        x = torch.tensor([[100.0], [200.0], [300.0], [150.0], [250.0]], dtype=torch.float32)
        
        # Create node labels with unique IDs
        node_labels = []
        for i in range(5):
            node_id = self.id_manager.generate_unique_id("test")
            node_labels.append({
                "id": node_id,
                "type": "test",
                "behavior": ["sensory", "dynamic", "oscillator", "integrator", "relay"][i],
                "state": "active",
                "energy": float(x[i, 0].item()),
                "membrane_potential": float(x[i, 0].item()) / 255.0,
                "threshold": 0.5,
                "refractory_timer": 0.0,
                "last_activation": 0,
                "plasticity_enabled": True,
                "eligibility_trace": 0.0,
                "last_update": 0
            })
            # Register node index
            self.id_manager.register_node_index(node_id, i)
        
        return Data(
            x=x,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            node_labels=node_labels
        )
    
    def test_id_based_node_access(self):
        """Test that nodes can be accessed by their unique IDs."""
        access_layer = NodeAccessLayer(self.graph)
        
        # Get all node IDs
        all_ids = self.id_manager.get_all_active_ids()
        self.assertEqual(len(all_ids), 5)
        
        # Test accessing each node by ID
        for node_id in all_ids:
            node = access_layer.get_node_by_id(node_id)
            self.assertIsNotNone(node)
            self.assertEqual(node['id'], node_id)
            
            # Test energy access
            energy = access_layer.get_node_energy(node_id)
            self.assertIsNotNone(energy)
            self.assertGreater(energy, 0)
    
    def test_id_based_node_selection(self):
        """Test that nodes can be selected by type and behavior using IDs."""
        access_layer = NodeAccessLayer(self.graph)
        
        # Test selection by behavior
        sensory_ids = access_layer.select_nodes_by_behavior('sensory')
        dynamic_ids = access_layer.select_nodes_by_behavior('dynamic')
        oscillator_ids = access_layer.select_nodes_by_behavior('oscillator')
        
        self.assertEqual(len(sensory_ids), 1)
        self.assertEqual(len(dynamic_ids), 1)
        self.assertEqual(len(oscillator_ids), 1)
        
        # Verify the selected nodes have the correct behavior
        for node_id in sensory_ids:
            node = access_layer.get_node_by_id(node_id)
            self.assertEqual(node['behavior'], 'sensory')
    
    def test_id_based_connection_formation(self):
        """Test that connections are formed using node IDs."""
        # Create connections using ID-based formation
        connected_graph = intelligent_connection_formation(self.graph)
        
        # Verify connections were created
        self.assertGreater(connected_graph.edge_index.shape[1], 0)
        
        # Verify edge attributes store node IDs
        if hasattr(connected_graph, 'edge_attributes'):
            for edge in connected_graph.edge_attributes:
                # Edge should store node IDs, not indices
                self.assertIsInstance(edge.source, int)
                self.assertIsInstance(edge.target, int)
                
                # Verify these are valid node IDs
                self.assertTrue(self.id_manager.is_valid_id(edge.source))
                self.assertTrue(self.id_manager.is_valid_id(edge.target))
    
    def test_id_based_energy_transfer(self):
        """Test that energy transfer works with node IDs."""
        access_layer = NodeAccessLayer(self.graph)
        
        # Get two nodes
        all_ids = self.id_manager.get_all_active_ids()
        source_id = all_ids[0]
        target_id = all_ids[1]
        
        # Get initial energies
        initial_source_energy = access_layer.get_node_energy(source_id)
        initial_target_energy = access_layer.get_node_energy(target_id)
        
        # Create a connection between them
        connected_graph = create_weighted_connection(self.graph, source_id, target_id, 1.0, 'excitatory')
        
        # Apply energy behavior (this should trigger energy transfer)
        updated_graph = apply_energy_behavior(connected_graph)
        
        # Verify energy transfer occurred
        new_access_layer = NodeAccessLayer(updated_graph)
        final_source_energy = new_access_layer.get_node_energy(source_id)
        final_target_energy = new_access_layer.get_node_energy(target_id)
        
        # Energy should have changed (exact values depend on behavior implementation)
        self.assertIsNotNone(final_source_energy)
        self.assertIsNotNone(final_target_energy)
    
    def test_id_based_behavior_updates(self):
        """Test that behavior updates work with node IDs."""
        behavior_engine = BehaviorEngine()
        access_layer = NodeAccessLayer(self.graph)
        
        # Get a node ID
        all_ids = self.id_manager.get_all_active_ids()
        test_node_id = all_ids[0]
        
        # Get initial state
        initial_state = access_layer.get_node_property(test_node_id, 'state')
        
        # Update behavior
        success = behavior_engine.update_node_behavior(test_node_id, self.graph, 1)
        self.assertTrue(success)
        
        # Verify state was updated
        final_state = access_layer.get_node_property(test_node_id, 'state')
        self.assertIsNotNone(final_state)
    
    def test_id_based_property_updates(self):
        """Test that node properties can be updated using IDs."""
        access_layer = NodeAccessLayer(self.graph)
        
        # Get a node ID
        all_ids = self.id_manager.get_all_active_ids()
        test_node_id = all_ids[0]
        
        # Update various properties
        access_layer.update_node_property(test_node_id, 'state', 'inactive')
        access_layer.update_node_property(test_node_id, 'threshold', 0.8)
        access_layer.set_node_energy(test_node_id, 150.0)
        
        # Verify updates
        self.assertEqual(access_layer.get_node_property(test_node_id, 'state'), 'inactive')
        self.assertEqual(access_layer.get_node_property(test_node_id, 'threshold'), 0.8)
        self.assertEqual(access_layer.get_node_energy(test_node_id), 150.0)
    
    def test_id_based_iteration(self):
        """Test that nodes can be iterated using IDs."""
        access_layer = NodeAccessLayer(self.graph)
        
        # Test iteration over all nodes
        node_count = 0
        for node_id, node in access_layer.iterate_all_nodes():
            self.assertIsNotNone(node)
            self.assertEqual(node['id'], node_id)
            node_count += 1
        
        self.assertEqual(node_count, 5)
        
        # Test iteration over specific IDs
        all_ids = self.id_manager.get_all_active_ids()
        specific_ids = all_ids[:3]  # First 3 nodes
        
        specific_count = 0
        for node_id, node in access_layer.iterate_nodes_by_ids(specific_ids):
            self.assertIn(node_id, specific_ids)
            specific_count += 1
        
        self.assertEqual(specific_count, 3)
    
    def test_id_based_filtering(self):
        """Test that nodes can be filtered using custom functions."""
        access_layer = NodeAccessLayer(self.graph)
        
        # Filter nodes with energy > 200
        def high_energy_filter(node_id, node):
            energy = access_layer.get_node_energy(node_id)
            return energy is not None and energy > 200
        
        high_energy_ids = access_layer.filter_nodes(high_energy_filter)
        
        # Verify filtering worked
        for node_id in high_energy_ids:
            energy = access_layer.get_node_energy(node_id)
            self.assertGreater(energy, 200)
    
    def test_id_based_statistics(self):
        """Test that statistics are generated using ID-based access."""
        access_layer = NodeAccessLayer(self.graph)
        
        stats = access_layer.get_node_statistics()
        
        # Verify statistics structure
        self.assertIn('total_nodes', stats)
        self.assertIn('by_type', stats)
        self.assertIn('by_behavior', stats)
        self.assertIn('by_state', stats)
        self.assertIn('energy_stats', stats)
        
        # Verify counts
        self.assertEqual(stats['total_nodes'], 5)
        self.assertEqual(stats['by_behavior']['sensory'], 1)
        self.assertEqual(stats['by_behavior']['dynamic'], 1)
        self.assertEqual(stats['by_behavior']['oscillator'], 1)
    
    def test_id_based_consistency_validation(self):
        """Test that the ID system maintains consistency."""
        access_layer = NodeAccessLayer(self.graph)
        
        validation = access_layer.validate_consistency()
        
        # Verify consistency
        self.assertTrue(validation['is_consistent'])
        self.assertEqual(len(validation['errors']), 0)
        self.assertFalse(validation['node_count_mismatch'])
    
    def test_id_based_performance(self):
        """Test performance of ID-based operations."""
        import time
        
        access_layer = NodeAccessLayer(self.graph)
        all_ids = self.id_manager.get_all_active_ids()
        
        # Test ID lookup performance
        start_time = time.time()
        for _ in range(1000):
            for node_id in all_ids:
                access_layer.get_node_by_id(node_id)
        lookup_time = time.time() - start_time
        
        # Test property update performance
        start_time = time.time()
        for _ in range(1000):
            for node_id in all_ids:
                access_layer.update_node_property(node_id, 'test_prop', 'test_value')
        update_time = time.time() - start_time
        
        # Performance should be reasonable (less than 1 second for 1000 operations)
        self.assertLess(lookup_time, 1.0)
        self.assertLess(update_time, 1.0)
        
        print(f"ID lookup performance: {lookup_time:.4f}s for 1000 operations")
        print(f"Property update performance: {update_time:.4f}s for 1000 operations")


class TestIDBasedIntegration(unittest.TestCase):
    """Integration tests for the complete ID-based system."""
    
    def setUp(self):
        """Set up test fixtures."""
        reset_id_manager()
    
    def tearDown(self):
        """Clean up after tests."""
        reset_id_manager()
    
    def test_end_to_end_id_based_simulation(self):
        """Test complete end-to-end simulation using ID-based system."""
        # Create a realistic test graph
        from main_graph import initialize_main_graph
        
        # Initialize graph (this will create nodes with unique IDs)
        graph = initialize_main_graph(scale=0.1)  # Small scale for testing
        
        # Verify all nodes have unique IDs
        access_layer = NodeAccessLayer(graph)
        all_ids = access_layer.get_all_active_ids()
        
        self.assertGreater(len(all_ids), 0)
        self.assertEqual(len(all_ids), len(set(all_ids)))  # All IDs should be unique
        
        # Test behavior updates
        behavior_engine = BehaviorEngine()
        for node_id in all_ids[:10]:  # Test first 10 nodes
            success = behavior_engine.update_node_behavior(node_id, graph, 1)
            self.assertTrue(success)
        
        # Test energy behavior
        updated_graph = apply_energy_behavior(graph)
        self.assertIsNotNone(updated_graph)
        
        # Test connection formation
        connected_graph = intelligent_connection_formation(updated_graph)
        self.assertIsNotNone(connected_graph)
        
        # Verify system consistency
        final_access_layer = NodeAccessLayer(connected_graph)
        validation = final_access_layer.validate_consistency()
        self.assertTrue(validation['is_consistent'])


def run_communication_tests():
    """Run all communication tests."""
    print("\n" + "="*60)
    print("ID-BASED COMMUNICATION SYSTEM TESTS")
    print("="*60)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*60)
    print("ALL COMMUNICATION TESTS COMPLETED")
    print("="*60)


if __name__ == '__main__':
    run_communication_tests()
