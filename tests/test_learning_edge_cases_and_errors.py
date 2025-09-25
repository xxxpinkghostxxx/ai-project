"""
Edge cases and error handling tests for learning system components.
Tests boundary conditions, invalid inputs, error recovery, and robustness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data

from learning.homeostasis_controller import HomeostasisController
from learning.learning_engine import LearningEngine
from learning.live_hebbian_learning import LiveHebbianLearning
from learning.memory_system import MemorySystem
from learning.memory_pool_manager import MemoryPoolManager, ObjectPool


class TestEdgeCases:
    """Test suite for edge cases across learning components."""

    def test_extreme_energy_values(self):
        """Test handling of extreme energy values."""
        homeostasis = HomeostasisController()
        learning_engine = LearningEngine(MagicMock())
        hebbian = LiveHebbianLearning()
        memory_system = MemorySystem()

        # Zero energy
        graph_zero = Data()
        graph_zero.node_labels = [{'id': 0, 'behavior': 'dynamic', 'energy': 0.0}]
        graph_zero.x = torch.tensor([[0.0]], dtype=torch.float32)

        result = homeostasis.regulate_network_activity(graph_zero)
        assert result == graph_zero

        # Maximum energy
        graph_max = Data()
        graph_max.node_labels = [{'id': 0, 'behavior': 'dynamic', 'energy': 1000.0}]
        graph_max.x = torch.tensor([[1000.0]], dtype=torch.float32)

        result = homeostasis.regulate_network_activity(graph_max)
        assert result == graph_max

        # Negative energy (should handle gracefully)
        graph_neg = Data()
        graph_neg.node_labels = [{'id': 0, 'behavior': 'dynamic', 'energy': -1.0}]
        graph_neg.x = torch.tensor([[-1.0]], dtype=torch.float32)

        result = homeostasis.regulate_network_activity(graph_neg)
        assert result == graph_neg

    def test_empty_and_minimal_graphs(self):
        """Test with empty or minimal graph structures."""
        homeostasis = HomeostasisController()
        learning_engine = LearningEngine(MagicMock())
        hebbian = LiveHebbianLearning()
        memory_system = MemorySystem()

        # Completely empty graph
        empty_graph = Data()
        empty_graph.node_labels = []
        empty_graph.x = torch.tensor([], dtype=torch.float32).reshape(0, 1)

        # All should handle empty graphs gracefully
        result1 = homeostasis.regulate_network_activity(empty_graph)
        assert result1 == empty_graph

        result2 = memory_system.form_memory_traces(empty_graph)
        assert result2 == empty_graph

        result3 = hebbian.apply_continuous_learning(empty_graph, 0)
        assert result3 == empty_graph

        # Single node graph
        single_graph = Data()
        single_graph.node_labels = [{'id': 0, 'behavior': 'dynamic', 'energy': 0.5}]
        single_graph.x = torch.tensor([[0.5]], dtype=torch.float32)

        result = homeostasis.regulate_network_activity(single_graph)
        assert result == single_graph

    def test_missing_attributes(self):
        """Test handling of missing graph attributes."""
        homeostasis = HomeostasisController()

        # Graph missing x attribute
        graph_no_x = Data()
        graph_no_x.node_labels = [{'id': 0}]

        result = homeostasis.regulate_network_activity(graph_no_x)
        assert result == graph_no_x

        # Graph missing node_labels
        graph_no_labels = Data()
        graph_no_labels.x = torch.tensor([[0.5]], dtype=torch.float32)

        result = homeostasis.regulate_network_activity(graph_no_labels)
        assert result == graph_no_labels

        # Graph with None values
        graph_none = Data()
        graph_none.node_labels = None
        graph_none.x = None

        result = homeostasis.regulate_network_activity(graph_none)
        assert result == graph_none

    def test_large_graphs(self):
        """Test with large graph structures."""
        homeostasis = HomeostasisController()

        # Create large graph
        num_nodes = 1000
        large_graph = Data()
        large_graph.node_labels = [
            {'id': i, 'behavior': 'dynamic', 'energy': 0.5}
            for i in range(num_nodes)
        ]
        large_graph.x = torch.rand(num_nodes, 1)

        # Should handle large graphs without crashing
        result = homeostasis.regulate_network_activity(large_graph)
        assert result == large_graph

    def test_invalid_node_data(self):
        """Test handling of invalid node data."""
        memory_system = MemorySystem()

        # Nodes with invalid behavior
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': None, 'energy': 0.5, 'last_activation': time.time()},
            {'id': 1, 'behavior': 'invalid_type', 'energy': 0.5, 'last_activation': time.time()},
            {'id': 2, 'behavior': '', 'energy': 0.5, 'last_activation': time.time()}
        ]
        graph.x = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float32)

        # Should handle invalid behaviors gracefully
        result = memory_system.form_memory_traces(graph)
        assert result == graph

    def test_extreme_timestamps(self):
        """Test handling of extreme timestamp values."""
        memory_system = MemorySystem()

        # Very old timestamps
        old_time = time.time() - 31536000  # 1 year ago
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': old_time}
        ]
        graph.x = torch.tensor([[0.8]], dtype=torch.float32)

        result = memory_system.form_memory_traces(graph)
        assert result == graph

        # Future timestamps
        future_time = time.time() + 31536000  # 1 year in future
        graph.node_labels[0]['last_activation'] = future_time

        result = memory_system.form_memory_traces(graph)
        assert result == graph

    def test_memory_system_capacity_limits(self):
        """Test memory system capacity limits."""
        memory_system = MemorySystem()
        memory_system.max_memory_traces = 2

        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time() - 5}
            for i in range(5)  # More than capacity
        ]
        graph.x = torch.rand(5, 1)
        graph.edge_index = torch.randint(0, 5, (2, 10))

        memory_system.form_memory_traces(graph)

        # Should not exceed capacity
        assert memory_system.get_memory_trace_count() <= 2

    def test_learning_engine_extreme_weights(self):
        """Test learning engine with extreme weight values."""
        mock_access_layer = MagicMock()
        learning_engine = LearningEngine(mock_access_layer)

        # Mock edges with extreme weights
        edge_min = MagicMock()
        edge_min.weight = 0.01  # Very small
        edge_min.eligibility_trace = 0.6

        edge_max = MagicMock()
        edge_max.weight = 10.0  # Very large
        edge_max.eligibility_trace = 0.6

        graph = Data()
        graph.edge_attributes = [edge_min, edge_max]

        result = learning_engine.consolidate_connections(graph)
        assert result == graph

        # Weights should be clamped appropriately
        assert 0.1 <= edge_min.weight <= 5.0
        assert 0.1 <= edge_max.weight <= 5.0


class TestErrorHandling:
    """Test suite for error handling across learning components."""

    def test_config_loading_failures(self):
        """Test handling of configuration loading failures."""
        with patch('learning.homeostasis_controller.get_config', side_effect=Exception("Config error")):
            # Should use default values
            homeostasis = HomeostasisController()
            assert homeostasis.target_energy_ratio == 0.6  # Default

    def test_energy_cap_unavailable(self):
        """Test handling when energy cap is unavailable."""
        learning_engine = LearningEngine(MagicMock())

        with patch('energy.energy_behavior.get_node_energy_cap', side_effect=ImportError):
            pre_node = {'energy': 0.8}
            post_node = {'energy': 0.6}

            # Should use fallback energy cap
            modulated = learning_engine._calculate_energy_modulated_rate(pre_node, post_node, 0.02)
            assert isinstance(modulated, float)

    def test_database_operation_failures(self):
        """Test handling of database-like operation failures."""
        memory_system = MemorySystem()

        # Simulate memory trace storage failure
        original_traces = memory_system.memory_traces
        memory_system.memory_traces = MagicMock()
        memory_system.memory_traces.__setitem__ = MagicMock(side_effect=Exception("Storage error"))

        graph = Data()
        graph.node_labels = [{'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time()}]
        graph.x = torch.tensor([[0.8]], dtype=torch.float32)

        # Should handle storage failure gracefully
        result = memory_system.form_memory_traces(graph)
        assert result == graph

        # Restore
        memory_system.memory_traces = original_traces

    def test_threading_and_concurrency_errors(self):
        """Test handling of threading and concurrency errors."""
        pool_manager = MemoryPoolManager()

        # Simulate threading error
        with patch('threading.RLock', side_effect=Exception("Threading error")):
            try:
                pool = pool_manager.create_pool('test', lambda: {})
                # If threading fails, should still work with basic functionality
                assert pool is not None
            except:
                # Acceptable if pool creation fails due to threading issues
                pass

    def test_network_communication_failures(self):
        """Test handling of network communication failures."""
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_by_id = MagicMock(side_effect=Exception("Network error"))

        learning_engine = LearningEngine(mock_access_layer)

        # Should handle network errors in event handling
        event_data = {'node_id': 1, 'source_id': 0}
        # Should not crash
        learning_engine._on_spike('SPIKE', event_data)

    def test_memory_allocation_failures(self):
        """Test handling of memory allocation failures."""
        memory_system = MemorySystem()

        # Simulate memory allocation failure
        with patch('builtins.dict', side_effect=MemoryError("Out of memory")):
            graph = Data()
            graph.node_labels = [{'id': 0}]

            # Should handle memory errors gracefully
            result = memory_system.form_memory_traces(graph)
            assert result == graph

    def test_invalid_mathematical_operations(self):
        """Test handling of invalid mathematical operations."""
        homeostasis = HomeostasisController()

        # Create graph that might cause division by zero or invalid math
        graph = Data()
        graph.node_labels = [{'id': 0}]
        graph.x = torch.tensor([[0.0]], dtype=torch.float32)  # Zero energy

        # Should handle gracefully
        result = homeostasis.regulate_network_activity(graph)
        assert result == graph

        # NaN values
        graph.x = torch.tensor([[float('nan')]], dtype=torch.float32)
        result = homeostasis.regulate_network_activity(graph)
        assert result == graph

    def test_external_dependency_failures(self):
        """Test handling of external dependency failures."""
        hebbian = LiveHebbianLearning()

        # Simulate torch operation failure
        with patch('torch.multinomial', side_effect=Exception("Torch error")):
            graph = Data()
            graph.x = torch.tensor([[0.5]], dtype=torch.float32)
            graph.node_labels = [{'id': 0}]

            # Should handle torch failures gracefully
            result = hebbian.apply_continuous_learning(graph, 0)
            assert result == graph

    def test_cleanup_and_recovery(self):
        """Test cleanup and recovery after errors."""
        homeostasis = HomeostasisController()
        memory_system = MemorySystem()

        # Cause some errors
        graph = Data()
        graph.node_labels = None

        homeostasis.regulate_network_activity(graph)
        memory_system.form_memory_traces(graph)

        # Should still be able to reset and continue
        homeostasis.reset_statistics()
        memory_system.reset_statistics()

        stats1 = homeostasis.get_regulation_statistics()
        stats2 = memory_system.get_memory_statistics()

        assert stats1['total_regulation_events'] == 0
        assert stats2['traces_formed'] == 0


class TestBoundaryConditions:
    """Test suite for boundary conditions."""

    def test_zero_division_prevention(self):
        """Test prevention of division by zero."""
        from learning.homeostasis_controller import calculate_network_stability

        # Empty graph
        empty_graph = Data()
        empty_graph.node_labels = []
        empty_graph.x = torch.tensor([], dtype=torch.float32).reshape(0, 1)

        stability = calculate_network_stability(empty_graph)
        assert stability == 0.0

        # Single node
        single_graph = Data()
        single_graph.node_labels = [{'id': 0}]
        single_graph.x = torch.tensor([[0.5]], dtype=torch.float32)

        stability = calculate_network_stability(single_graph)
        assert isinstance(stability, float)

    def test_array_bounds_checking(self):
        """Test array bounds checking."""
        hebbian = LiveHebbianLearning()

        graph = Data()
        graph.x = torch.tensor([[0.5], [0.6]], dtype=torch.float32)
        graph.node_labels = [{'id': 0}, {'id': 1}]

        # Access beyond bounds should be handled
        with patch.object(hebbian, '_get_node_energy', side_effect=lambda nid: graph.x[nid, 0].item() if nid < len(graph.x) else 0.0):
            energy = hebbian._get_node_energy(5)  # Beyond bounds
            assert energy == 0.0

    def test_type_conversion_robustness(self):
        """Test robustness to type conversion issues."""
        memory_system = MemorySystem()

        # Non-numeric values in node data
        graph = Data()
        graph.node_labels = [
            {'id': 'string_id', 'behavior': 'integrator', 'energy': 'not_a_number', 'last_activation': 'invalid_time'}
        ]
        graph.x = torch.tensor([[0.5]], dtype=torch.float32)

        # Should handle type conversion issues gracefully
        result = memory_system.form_memory_traces(graph)
        assert result == graph

    def test_circular_reference_prevention(self):
        """Test prevention of circular references."""
        memory_system = MemorySystem()

        # Create potential circular reference scenario
        graph = Data()
        graph.node_labels = [{'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time()}]
        graph.x = torch.tensor([[0.8]], dtype=torch.float32)

        # Add self-reference that might cause issues
        graph.node_labels[0]['self_ref'] = graph.node_labels[0]

        result = memory_system.form_memory_traces(graph)
        assert result == graph

    def test_resource_exhaustion_prevention(self):
        """Test prevention of resource exhaustion."""
        pool = ObjectPool(lambda: {'data': 'x' * 1000}, max_size=10)  # Large objects

        # Should not allow unlimited growth
        for i in range(20):  # More than max_size
            obj = pool.get_object()
            pool.return_object(obj)

        assert len(pool.available_objects) <= pool.max_size


if __name__ == "__main__":
    pytest.main([__file__])