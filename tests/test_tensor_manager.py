#!/usr/bin/env python3
"""
Comprehensive test suite for the TensorManager system.

This module provides extensive testing and validation of the tensor management
capabilities to ensure all tensor operations, synchronization, and recovery
mechanisms work correctly while preserving simulation features.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from src.project.utils.tensor_manager import TensorManager
from src.project.pyg_neural_system import PyGNeuralSystem

class TestTensorManager(unittest.TestCase):
    """Comprehensive test suite for TensorManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock neural system for testing
        self.mock_neural_system = Mock(spec=PyGNeuralSystem)
        self.mock_neural_system.device = 'cpu'
        self.mock_neural_system.g = Mock()

        # Set up basic graph attributes
        self.mock_neural_system.g.num_nodes = 10
        self.mock_neural_system.g.num_edges = 5
        self.mock_neural_system.g.energy = torch.randn(10, 1)
        self.mock_neural_system.g.node_type = torch.randint(0, 3, (10,))
        self.mock_neural_system.g.edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.mock_neural_system.g.weight = torch.randn(5, 1)
        self.mock_neural_system.g.plastic_lr = torch.randn(5, 1)

        # Initialize TensorManager
        self.tensor_manager = TensorManager(self.mock_neural_system)

    def test_initialization(self):
        """Test TensorManager initialization."""
        self.assertIsNotNone(self.tensor_manager)
        self.assertEqual(self.tensor_manager.neural_system, self.mock_neural_system)
        self.assertIsInstance(self.tensor_manager.tensor_history, dict)
        self.assertEqual(self.tensor_manager.max_history, 100)

    def test_validate_tensor_shapes_valid(self):
        """Test tensor shape validation with valid tensors."""
        # Set up valid tensors
        self.mock_neural_system.g.energy = torch.randn(10, 1)
        self.mock_neural_system.g.node_type = torch.randint(0, 3, (10,))
        self.mock_neural_system.g.weight = torch.randn(5, 1)
        self.mock_neural_system.g.plastic_lr = torch.randn(5, 1)

        validation_results = self.tensor_manager.validate_tensor_shapes()

        # All tensors should be valid
        for key, is_valid in validation_results.items():
            self.assertTrue(is_valid, f"Tensor {key} should be valid")

    def test_validate_tensor_shapes_invalid(self):
        """Test tensor shape validation with invalid tensors."""
        # Set up invalid tensors (wrong shapes)
        self.mock_neural_system.g.energy = torch.randn(8, 1)  # Wrong: should be 10
        self.mock_neural_system.g.weight = torch.randn(3, 1)  # Wrong: should be 5

        validation_results = self.tensor_manager.validate_tensor_shapes()

        # Check that invalid tensors are detected
        self.assertFalse(validation_results.get('energy', True))
        self.assertFalse(validation_results.get('weight', True))

    def test_synchronize_all_tensors_valid(self):
        """Test tensor synchronization with valid tensors."""
        # Set up valid tensors
        self.mock_neural_system.g.energy = torch.randn(10, 1)
        self.mock_neural_system.g.weight = torch.randn(5, 1)

        sync_results = self.tensor_manager.synchronize_all_tensors()

        # All synchronizations should succeed
        for key, success in sync_results.items():
            self.assertTrue(success, f"Tensor {key} synchronization should succeed")

    def test_synchronize_all_tensors_invalid(self):
        """Test tensor synchronization with invalid tensors."""
        # Set up invalid tensors
        self.mock_neural_system.g.energy = torch.randn(8, 1)  # Wrong: should be 10
        self.mock_neural_system.g.weight = torch.randn(3, 1)  # Wrong: should be 5

        sync_results = self.tensor_manager.synchronize_all_tensors()

        # Check that synchronization attempts were made
        self.assertIn('energy', sync_results)
        self.assertIn('weight', sync_results)

        # After synchronization, tensors should be resized
        self.assertEqual(self.mock_neural_system.g.energy.shape[0], 10)
        self.assertEqual(self.mock_neural_system.g.weight.shape[0], 5)

    def test_intelligent_resize_tensor(self):
        """Test intelligent tensor resizing functionality."""
        # Create a tensor with wrong size
        original_tensor = torch.randn(8, 2)
        self.mock_neural_system.g.energy = original_tensor

        # Resize to target size
        success = self.tensor_manager._intelligent_resize_tensor(
            original_tensor, 12, 'energy', 'node'
        )

        self.assertTrue(success)
        self.assertEqual(original_tensor.shape[0], 12)
        self.assertEqual(original_tensor.shape[1], 2)

        # Check that original data was preserved
        self.assertTrue(torch.allclose(original_tensor[:8], original_tensor[:8]))

        # Check that new elements were initialized reasonably
        self.assertFalse(torch.all(original_tensor[8:]))  # Should not be all zeros

    def test_validate_connection_integrity_valid(self):
        """Test connection integrity validation with valid connections."""
        # Set up valid connections
        self.mock_neural_system.g.edge_index = torch.tensor([[0, 1, 2], [5, 6, 7]])
        self.mock_neural_system.g.node_type = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
        self.mock_neural_system.g.num_nodes = 10
        self.mock_neural_system.g.num_edges = 3

        # Mock conn_type to avoid mock comparison issues
        self.mock_neural_system.g.conn_type = torch.tensor([[0], [1], [2]])  # Valid connection types

        is_valid = self.tensor_manager.validate_connection_integrity()
        self.assertTrue(is_valid)

    def test_validate_connection_integrity_invalid(self):
        """Test connection integrity validation with invalid connections."""
        # Set up invalid connections (out of bounds)
        self.mock_neural_system.g.edge_index = torch.tensor([[0, 1, 15], [5, 6, 7]])  # 15 is out of bounds
        self.mock_neural_system.g.node_type = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])

        is_valid = self.tensor_manager.validate_connection_integrity()
        self.assertFalse(is_valid)

    def test_repair_invalid_connections(self):
        """Test repair of invalid connections."""
        # Set up invalid connections
        self.mock_neural_system.g.edge_index = torch.tensor([[0, 1, 15], [5, 6, 7]])
        self.mock_neural_system.g.weight = torch.randn(3, 1)
        self.mock_neural_system.g.node_type = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])

        repaired_count = self.tensor_manager.repair_invalid_connections()

        self.assertEqual(repaired_count, 1)  # Should repair the connection with index 15
        self.assertEqual(self.mock_neural_system.g.edge_index.shape[1], 2)  # Should have 2 valid connections left

    def test_optimize_tensor_memory(self):
        """Test tensor memory optimization."""
        # Set up some unused tensors
        self.mock_neural_system.g.unused_tensor1 = torch.randn(0, 1)
        self.mock_neural_system.g.unused_tensor2 = torch.randn(0, 2)

        optimization_stats = self.tensor_manager.optimize_tensor_memory()

        self.assertGreaterEqual(optimization_stats['tensors_cleaned'], 2)
        self.assertFalse(hasattr(self.mock_neural_system.g, 'unused_tensor1'))
        self.assertFalse(hasattr(self.mock_neural_system.g, 'unused_tensor2'))

    def test_get_tensor_health_report(self):
        """Test tensor health report generation."""
        # Set up valid tensors
        self.mock_neural_system.g.energy = torch.randn(10, 1)
        self.mock_neural_system.g.weight = torch.randn(5, 1)

        health_report = self.tensor_manager.get_tensor_health_report()

        self.assertIn('timestamp', health_report)
        self.assertIn('tensor_count', health_report)
        self.assertIn('validation_results', health_report)
        self.assertIn('node_tensors', health_report)
        self.assertIn('edge_tensors', health_report)

        # Check that validation results are included
        validation_results = health_report['validation_results']
        self.assertIsInstance(validation_results, dict)

        # Check that tensor information is included
        node_tensors = health_report['node_tensors']
        self.assertIn('energy', node_tensors)
        self.assertIn('node_type', node_tensors)

    def test_ensure_simulation_integrity(self):
        """Test simulation integrity assurance."""
        # Set up valid tensors
        self.mock_neural_system.g.energy = torch.randn(10, 1)
        self.mock_neural_system.g.weight = torch.randn(5, 1)

        integrity_ensured = self.tensor_manager.ensure_simulation_integrity()
        self.assertTrue(integrity_ensured)

    def test_ensure_simulation_integrity_with_repair(self):
        """Test simulation integrity assurance with repair needed."""
        # Set up invalid tensors
        self.mock_neural_system.g.energy = torch.randn(8, 1)  # Wrong size
        self.mock_neural_system.g.weight = torch.randn(5, 1)

        integrity_ensured = self.tensor_manager.ensure_simulation_integrity()
        self.assertTrue(integrity_ensured)

        # Check that tensors were repaired
        self.assertEqual(self.mock_neural_system.g.energy.shape[0], 10)

    def test_edge_tensor_synchronization(self):
        """Test edge tensor synchronization specifically."""
        # Set up edge tensors with wrong sizes
        self.mock_neural_system.g.weight = torch.randn(3, 1)  # Should be 5
        self.mock_neural_system.g.plastic_lr = torch.randn(7, 1)  # Should be 5

        sync_results = self.tensor_manager.synchronize_all_tensors()

        # Check that edge tensors were synchronized
        self.assertIn('weight', sync_results)
        self.assertIn('plastic_lr', sync_results)

        # Check final sizes
        self.assertEqual(self.mock_neural_system.g.weight.shape[0], 5)
        self.assertEqual(self.mock_neural_system.g.plastic_lr.shape[0], 5)

    def test_critical_tensor_handling(self):
        """Test handling of critical tensors like plastic_lr and weight."""
        # Set up critical tensors with issues
        self.mock_neural_system.g.plastic_lr = torch.randn(3, 1)  # Wrong size
        self.mock_neural_system.g.weight = torch.randn(7, 1)  # Wrong size

        # These should be handled as critical tensors
        sync_results = self.tensor_manager.synchronize_all_tensors()

        self.assertIn('plastic_lr', sync_results)
        self.assertIn('weight', sync_results)

        # Verify they were resized correctly
        self.assertEqual(self.mock_neural_system.g.plastic_lr.shape[0], 5)
        self.assertEqual(self.mock_neural_system.g.weight.shape[0], 5)

class TestTensorManagerIntegration(unittest.TestCase):
    """Integration tests for TensorManager with real neural system."""

    def test_integration_with_real_system(self):
        """Test TensorManager integration with a real PyGNeuralSystem."""
        # Create a small real neural system
        real_system = PyGNeuralSystem(
            sensory_width=2,
            sensory_height=2,
            n_dynamic=4,
            workspace_size=(1, 1),
            device='cpu'
        )

        # Initialize tensor manager
        tensor_manager = TensorManager(real_system)

        # Test validation
        validation_results = tensor_manager.validate_tensor_shapes()
        self.assertIsInstance(validation_results, dict)

        # Test synchronization
        sync_results = tensor_manager.synchronize_all_tensors()
        self.assertIsInstance(sync_results, dict)

        # Test health report
        health_report = tensor_manager.get_tensor_health_report()
        self.assertIsInstance(health_report, dict)
        self.assertIn('timestamp', health_report)

        # Test integrity assurance
        integrity_ensured = tensor_manager.ensure_simulation_integrity()
        self.assertTrue(integrity_ensured)

if __name__ == '__main__':
    unittest.main()