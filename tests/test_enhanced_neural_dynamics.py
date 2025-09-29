"""
Comprehensive tests for EnhancedNeuralDynamics fixes.
Tests error handling, bounds checking, and thread safety.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.neural.enhanced_neural_dynamics import (
    EnhancedNeuralDynamics, create_enhanced_neural_dynamics)


class TestEnhancedNeuralDynamics:
    """Test suite for EnhancedNeuralDynamics fixes."""

    def setup_method(self):
        """Set up test environment."""
        self.dynamics = EnhancedNeuralDynamics()

    def teardown_method(self):
        """Clean up after tests."""
        self.dynamics.cleanup()

    def test_initialization_validation(self):
        """Test initialization with parameter validation."""
        # Test with valid parameters
        dynamics = EnhancedNeuralDynamics()
        assert dynamics.stdp_window == 20.0
        assert dynamics.ltp_rate == 0.02

        # Test parameter bounds
        assert dynamics._validate_float(1.0, 0.0, 10.0, "test") == 1.0
        assert dynamics._validate_float(-5.0, 0.0, 10.0, "test") == 0.0  # Clamped
        assert dynamics._validate_float(15.0, 0.0, 10.0, "test") == 10.0  # Clamped
        assert dynamics._validate_float("invalid", 0.0, 10.0, "test") == 0.0  # Default

    def test_update_neural_dynamics_validation(self):
        """Test update_neural_dynamics with input validation."""
        # Test with None graph
        result = self.dynamics.update_neural_dynamics(None, 0)
        assert result is None

        # Test with invalid step
        mock_graph = MagicMock()
        mock_graph.node_labels = []
        result = self.dynamics.update_neural_dynamics(mock_graph, -1)
        assert result == mock_graph

        # Test with missing node_labels
        mock_graph.node_labels = None
        result = self.dynamics.update_neural_dynamics(mock_graph, 0)
        assert result == mock_graph

    def test_synaptic_input_calculation_bounds(self):
        """Test synaptic input calculation with bounds checking."""
        # Create mock graph and access layer
        mock_graph = MagicMock()
        mock_edge = MagicMock()
        mock_edge.target = 1
        mock_edge.source = 2
        mock_edge.type = 'excitatory'
        mock_edge.get_effective_weight.return_value = 1.0
        mock_graph.edge_attributes = [mock_edge]

        mock_access_layer = MagicMock()
        mock_source_node = {'last_spike_time': time.time()}
        mock_access_layer.get_node_by_id.return_value = mock_source_node

        # Test with valid data
        result = self.dynamics._calculate_synaptic_input(mock_graph, 1, mock_access_layer)
        assert isinstance(result, float)

        # Test with invalid edge attributes
        mock_graph.edge_attributes = None
        result = self.dynamics._calculate_synaptic_input(mock_graph, 1, mock_access_layer)
        assert result == 0.0

        # Test with invalid edge
        mock_graph.edge_attributes = [None]
        result = self.dynamics._calculate_synaptic_input(mock_graph, 1, mock_access_layer)
        assert result == 0.0

    def test_ei_ratio_calculation(self):
        """Test E/I ratio calculation with division by zero protection."""
        # Create mock graph
        mock_graph = MagicMock()
        mock_edge1 = MagicMock()
        mock_edge1.type = 'excitatory'
        mock_edge1.weight = 2.0
        mock_edge2 = MagicMock()
        mock_edge2.type = 'inhibitory'
        mock_edge2.weight = 1.0
        mock_graph.edge_attributes = [mock_edge1, mock_edge2]

        # Test normal case
        ratio = self.dynamics._calculate_ei_ratio(mock_graph)
        assert ratio == 2.0  # 2.0 / 1.0

        # Test division by zero protection
        mock_edge2.weight = 0.0
        ratio = self.dynamics._calculate_ei_ratio(mock_graph)
        assert ratio == 10.0  # Should return high ratio when inhibitory is zero

        # Test with no inhibitory edges
        mock_graph.edge_attributes = [mock_edge1]
        ratio = self.dynamics._calculate_ei_ratio(mock_graph)
        assert ratio == 10.0

        # Test with invalid data
        mock_graph.edge_attributes = None
        ratio = self.dynamics._calculate_ei_ratio(mock_graph)
        assert ratio == 1.0

    def test_neuromodulator_level_validation(self):
        """Test neuromodulator level setting with validation."""
        # Test valid setting
        self.dynamics.set_neuromodulator_level('dopamine', 0.8)
        assert self.dynamics.dopamine_level == 0.8
        assert self.dynamics.neuromodulators['dopamine'] == 0.8

        # Test clamping
        self.dynamics.set_neuromodulator_level('dopamine', 1.5)
        assert self.dynamics.dopamine_level == 1.0

        self.dynamics.set_neuromodulator_level('dopamine', -0.5)
        assert self.dynamics.dopamine_level == 0.0

        # Test invalid inputs
        self.dynamics.set_neuromodulator_level(123, 0.5)  # Invalid name type
        self.dynamics.set_neuromodulator_level('dopamine', 'invalid')  # Invalid level type

        # Test unknown neuromodulator
        self.dynamics.set_neuromodulator_level('unknown', 0.5)

    def test_statistics_thread_safety(self):
        """Test statistics access thread safety."""
        results = []

        def access_stats():
            stats = self.dynamics.get_statistics()
            results.append(stats)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=access_stats)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert 'total_spikes' in result

    def test_cleanup_comprehensive(self):
        """Test comprehensive cleanup."""
        # Add some test data
        self.dynamics.spike_times[1] = [time.time()]
        self.dynamics.theta_burst_counters[1] = 5
        self.dynamics.ieg_flags[1] = True
        self.dynamics.node_activity_history[1].append(time.time())

        # Set neuromodulators
        self.dynamics.set_neuromodulator_level('dopamine', 0.8)

        # Verify data exists
        assert len(self.dynamics.spike_times) > 0
        assert len(self.dynamics.theta_burst_counters) > 0
        assert self.dynamics.dopamine_level == 0.8

        # Clean up
        self.dynamics.cleanup()

        # Verify cleanup
        assert len(self.dynamics.spike_times) == 0
        assert len(self.dynamics.theta_burst_counters) == 0
        assert len(self.dynamics.ieg_flags) == 0
        assert len(self.dynamics.node_activity_history) == 0
        assert self.dynamics.dopamine_level == 0.0

    def test_thread_safe_operations(self):
        """Test thread safety of operations."""
        errors = []

        def perform_operations():
            try:
                # Test various operations
                self.dynamics.set_neuromodulator_level('dopamine', 0.5)
                stats = self.dynamics.get_statistics()
                self.dynamics.reset_statistics()
                new_stats = self.dynamics.get_statistics()

                # Verify reset worked
                assert new_stats['total_spikes'] == 0

            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=perform_operations)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

    def test_error_recovery_in_update(self):
        """Test error recovery in neural dynamics update."""
        # Create a mock graph that will cause errors
        mock_graph = MagicMock()
        mock_graph.node_labels = [MagicMock()]

        # Mock one of the update methods to raise an exception
        with patch.object(self.dynamics, '_update_membrane_dynamics', side_effect=Exception("Test error")):
            result = self.dynamics.update_neural_dynamics(mock_graph, 0)
            # Should return the graph despite the error
            assert result == mock_graph

    def test_bounds_checking_in_calculations(self):
        """Test bounds checking in various calculations."""
        # Test with extreme values
        assert self.dynamics._validate_float(float('inf'), 0.0, 100.0, "test") == 100.0
        assert self.dynamics._validate_float(float('-inf'), 0.0, 100.0, "test") == 0.0
        assert self.dynamics._validate_float(float('nan'), 0.0, 100.0, "test") == 0.0

    def test_memory_efficiency(self):
        """Test memory efficiency of data structures."""
        # Test that deques have proper maxlen
        assert self.dynamics.node_activity_history[1].maxlen == 1000

        # Add many items to test maxlen enforcement
        for i in range(1500):
            self.dynamics.node_activity_history[1].append(time.time())

        # Should only keep the last 1000 items
        assert len(self.dynamics.node_activity_history[1]) <= 1000

    def test_configuration_fallback(self):
        """Test configuration fallback when config loading fails."""
        # Mock config loading failure
        with patch('neural.enhanced_neural_dynamics.get_learning_config', side_effect=Exception("Config error")):
            dynamics = EnhancedNeuralDynamics()
            # Should still initialize with default values
            assert dynamics.stdp_window == 20.0
            assert dynamics.ltp_rate == 0.02


if __name__ == "__main__":
    pytest.main([__file__])






