"""
Comprehensive tests for EnhancedConnectionSystem fixes.
Tests validation, synchronization, and memory management.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import threading
import time

import pytest

from src.energy.energy_constants import ConnectionConstants
from src.neural.enhanced_connection_system import (EnhancedConnection,
                                                   EnhancedConnectionSystem)


class TestEnhancedConnectionSystem:
    """Test suite for EnhancedConnectionSystem fixes."""

    def setup_method(self):
        """Set up test environment."""
        self.system = EnhancedConnectionSystem()

    def teardown_method(self):
        """Clean up after tests."""
        self.system.cleanup()

    def test_connection_creation_validation(self):
        """Test connection creation with input validation."""
        # Test valid connection
        result = self.system.create_connection(1, 2, 'excitatory', weight=1.0)
        assert result is True

        # Test invalid source_id
        result = self.system.create_connection("invalid", 2, 'excitatory')
        assert result is False

        # Test invalid target_id
        result = self.system.create_connection(1, "invalid", 'excitatory')
        assert result is False

        # Test self-connection
        result = self.system.create_connection(1, 1, 'excitatory')
        assert result is False

        # Test invalid connection type
        result = self.system.create_connection(1, 3, 'invalid_type')
        assert result is False

        # Test duplicate connection
        result = self.system.create_connection(1, 2, 'excitatory')  # Already exists
        assert result is False

    def test_enhanced_connection_validation(self):
        """Test EnhancedConnection input validation."""
        # Test valid connection
        conn = EnhancedConnection(1, 2, 'excitatory', weight=1.0)
        assert conn.source_id == 1
        assert conn.target_id == 2
        assert conn.connection_type == 'excitatory'

        # Test invalid source_id
        with pytest.raises(ValueError):
            EnhancedConnection("invalid", 2, 'excitatory')

        # Test invalid target_id
        with pytest.raises(ValueError):
            EnhancedConnection(1, "invalid", 'excitatory')

        # Test self-connection
        with pytest.raises(ValueError):
            EnhancedConnection(1, 1, 'excitatory')

        # Test invalid connection type
        with pytest.raises(ValueError):
            EnhancedConnection(1, 2, 'invalid_type')

    def test_weight_validation(self):
        """Test weight validation and clamping."""
        conn = EnhancedConnection(1, 2, 'excitatory', weight=1.0)

        # Test valid weight update
        result = conn.update_weight(0.1)
        assert result is True

        # Test invalid weight change
        result = conn.update_weight("invalid")
        assert result is False

        # Test weight clamping
        conn.weight = 100.0  # Set high weight
        conn.update_weight(10.0)  # Try to increase further
        assert conn.weight <= ConnectionConstants.WEIGHT_CAP_MAX

    def test_memory_limits(self):
        """Test memory limits enforcement."""
        # Create connections up to the limit
        for i in range(min(10, self.system.max_connections)):  # Test with smaller number
            result = self.system.create_connection(i, i+100, 'excitatory')
            if i >= self.system.max_connections:
                assert result is False
                break

        # Test connections per node limit
        source_id = 1000
        for i in range(min(5, self.system.max_connections_per_node)):
            result = self.system.create_connection(source_id, 2000+i, 'excitatory')
            if i >= self.system.max_connections_per_node:
                assert result is False
                break

    def test_thread_safe_operations(self):
        """Test thread safety of operations."""
        errors = []
        created_connections = []

        def create_connections():
            try:
                for i in range(5):
                    source_id = threading.get_ident() * 100 + i
                    target_id = source_id + 1000
                    result = self.system.create_connection(source_id, target_id, 'excitatory')
                    if result:
                        created_connections.append((source_id, target_id))
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=create_connections)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have created some connections without errors
        assert len(created_connections) > 0
        assert len(errors) == 0

    def test_connection_removal_validation(self):
        """Test connection removal with validation."""
        # Create a connection first
        result = self.system.create_connection(10, 20, 'excitatory')
        assert result is True

        # Test valid removal
        result = self.system.remove_connection(10, 20)
        assert result is True

        # Test removing non-existent connection
        result = self.system.remove_connection(10, 20)
        assert result is False

        # Test invalid parameters
        result = self.system.remove_connection("invalid", 20)
        assert result is False

        result = self.system.remove_connection(10, "invalid")
        assert result is False

    def test_get_connection_validation(self):
        """Test get_connection with validation."""
        # Create a connection
        self.system.create_connection(30, 40, 'excitatory')

        # Test valid get
        conn = self.system.get_connection(30, 40)
        assert conn is not None
        assert conn.source_id == 30
        assert conn.target_id == 40

        # Test get non-existent connection
        conn = self.system.get_connection(50, 60)
        assert conn is None

        # Test invalid parameters
        conn = self.system.get_connection("invalid", 40)
        assert conn is None

        conn = self.system.get_connection(30, "invalid")
        assert conn is None

    def test_get_connections_for_node_validation(self):
        """Test get_connections_for_node with validation."""
        # Create some connections
        self.system.create_connection(100, 200, 'excitatory')
        self.system.create_connection(100, 201, 'inhibitory')

        # Test valid get
        connections = self.system.get_connections_for_node(100)
        assert len(connections) == 2

        # Test get for node with no connections
        connections = self.system.get_connections_for_node(999)
        assert len(connections) == 0

        # Test invalid node_id
        connections = self.system.get_connections_for_node("invalid")
        assert len(connections) == 0

    def test_neuromodulator_validation(self):
        """Test neuromodulator level setting with validation."""
        # Test valid setting
        self.system.set_neuromodulator_level('dopamine', 0.8)
        assert self.system.neuromodulators['dopamine'] == 0.8

        # Test clamping
        self.system.set_neuromodulator_level('dopamine', 1.5)
        assert self.system.neuromodulators['dopamine'] == 1.0

        self.system.set_neuromodulator_level('dopamine', -0.5)
        assert self.system.neuromodulators['dopamine'] == 0.0

        # Test invalid neuromodulator
        self.system.set_neuromodulator_level(123, 0.5)  # Invalid name type
        self.system.set_neuromodulator_level('dopamine', "invalid")  # Invalid level type

    def test_effective_weight_calculation(self):
        """Test effective weight calculation with neuromodulators."""
        # Create a connection
        self.system.create_connection(300, 400, 'excitatory', weight=1.0)

        # Test without neuromodulators
        weight = self.system.get_effective_weights(300, 400)
        assert weight == 1.0

        # Test with neuromodulators
        self.system.set_neuromodulator_level('dopamine', 0.5)
        weight = self.system.get_effective_weights(300, 400)
        assert weight > 1.0  # Should be increased by dopamine

        # Test invalid parameters
        weight = self.system.get_effective_weights("invalid", 400)
        assert weight == 0.0

        weight = self.system.get_effective_weights(300, "invalid")
        assert weight == 0.0

    def test_statistics_thread_safety(self):
        """Test statistics access thread safety."""
        results = []

        def get_stats():
            stats = self.system.get_connection_statistics()
            results.append(stats)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_stats)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert 'total_connections' in result
            assert 'active_connections' in result

    def test_cleanup_comprehensive(self):
        """Test comprehensive cleanup."""
        # Create some connections
        for i in range(5):
            self.system.create_connection(i, i+10, 'excitatory')

        # Set neuromodulators
        self.system.set_neuromodulator_level('dopamine', 0.8)

        # Verify data exists
        stats = self.system.get_connection_statistics()
        assert stats['total_connections'] == 5
        assert self.system.neuromodulators['dopamine'] == 0.8

        # Clean up
        self.system.cleanup()

        # Verify cleanup
        stats = self.system.get_connection_statistics()
        assert stats['total_connections'] == 0
        assert self.system.neuromodulators['dopamine'] == 0.0
        assert len(self.system.connections) == 0
        assert len(self.system.connection_index) == 0

    def test_activation_recording(self):
        """Test activation recording with history limits."""
        conn = EnhancedConnection(500, 600, 'gated')

        # Record activations
        for i in range(60):  # More than the limit
            conn.record_activation(time.time(), 0.5)

        # History should be limited
        assert len(conn.gate_history) <= 50

        # Activation count should be correct
        assert conn.activation_count == 60

    def test_update_connections_validation(self):
        """Test update_connections with validation."""
        import torch
        from torch_geometric.data import Data

        # Create mock graph
        graph = Data()
        graph.node_labels = [{'id': 1}, {'id': 2}]

        # Test valid update
        result = self.system.update_connections(graph, 0)
        assert result == graph

        # Test with None graph
        result = self.system.update_connections(None, 0)
        assert result is None

        # Test with invalid step
        result = self.system.update_connections(graph, -1)
        assert result == graph

    def test_concurrent_statistics_access(self):
        """Test concurrent access to statistics."""
        # Create some connections first
        for i in range(10):
            self.system.create_connection(i, i+100, 'excitatory')

        errors = []

        def access_stats():
            try:
                for _ in range(10):
                    stats = self.system.get_connection_statistics()
                    assert isinstance(stats, dict)
                    # Simulate some operations
                    self.system.set_neuromodulator_level('dopamine', 0.1)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=access_stats)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])






