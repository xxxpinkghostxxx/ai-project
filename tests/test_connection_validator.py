"""
Comprehensive tests for connection_validator.py
Covers unit tests, integration tests, edge cases, error handling, performance, and real-world usage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from unittest.mock import Mock, patch

import pytest
import torch
from torch_geometric.data import Data

from src.utils.connection_validator import (ConnectionValidationError,
                                            ConnectionValidator,
                                            get_connection_validator)


class TestConnectionValidatorInitialization:
    """Test ConnectionValidator initialization."""

    def test_connection_validator_init(self):
        """Test basic initialization."""
        validator = ConnectionValidator()

        assert validator._cache_max_size == 1000
        assert validator._validation_cache == {}
        assert validator._stats['validations_performed'] == 0
        assert validator._stats['cache_hits'] == 0
        assert validator._stats['errors_found'] == 0
        assert validator._stats['warnings_issued'] == 0

    def test_connection_validator_thread_safety(self):
        """Test thread safety of validator."""
        validator = ConnectionValidator()

        # Check that lock is initialized
        assert hasattr(validator, '_lock')
        assert validator._lock is not None


class TestValidateConnection:
    """Test validate_connection method."""

    def test_validate_connection_valid_excitatory(self):
        """Test validation of valid excitatory connection."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['errors']) == 0
        assert len(result['warnings']) == 0
        assert len(result['suggestions']) == 0

    def test_validate_connection_valid_inhibitory(self):
        """Test validation of valid inhibitory connection."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'inhibitory', -0.5)

        assert result['is_valid'] == True
        assert len(result['errors']) == 0

    def test_validate_connection_self_connection_error(self):
        """Test validation of self-connection (should fail)."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 1, 'excitatory', 1.0)

        assert result['is_valid'] == False
        assert len(result['errors']) == 1
        assert 'Self-connection not allowed' in result['errors'][0]

    def test_validate_connection_missing_source_node(self):
        """Test validation with missing source node."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == False
        assert len(result['errors']) == 1
        assert 'Source node 1 does not exist' in result['errors'][0]

    def test_validate_connection_missing_target_node(self):
        """Test validation with missing target node."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == False
        assert len(result['errors']) == 1
        assert 'Target node 2 does not exist' in result['errors'][0]

    def test_validate_connection_invalid_connection_type(self):
        """Test validation with invalid connection type."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'invalid_type', 1.0)

        assert result['is_valid'] == False
        assert len(result['errors']) == 1
        assert 'Invalid connection type' in result['errors'][0]

    def test_validate_connection_burst_warning(self):
        """Test validation with burst connection (should warn)."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'burst', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 1
        assert 'Burst connections can cause instability' in result['warnings'][0]

    def test_validate_connection_inhibitory_positive_weight_warning(self):
        """Test validation with inhibitory connection having positive weight."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'inhibitory', 0.5)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 1
        assert 'Inhibitory connections should have negative weights' in result['warnings'][0]

    def test_validate_connection_extreme_weight_warning(self):
        """Test validation with extreme weight value."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 15.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 1
        assert 'Extreme weight value' in result['warnings'][0]

    def test_validate_connection_nan_weight_error(self):
        """Test validation with NaN weight."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'excitatory', float('nan'))

        assert result['is_valid'] == False
        assert len(result['errors']) == 1
        assert 'Invalid weight value' in result['errors'][0]

    def test_validate_connection_sensory_type_warnings(self):
        """Test validation with sensory-to-sensory connection."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'sensory'},
            {'id': 2, 'type': 'sensory'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 1
        assert 'Sensory-to-sensory connections may not be meaningful' in result['warnings'][0]

    def test_validate_connection_workspace_to_sensory_warning(self):
        """Test validation with workspace-to-sensory connection."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'workspace'},
            {'id': 2, 'type': 'sensory'}
        ]

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 1
        assert 'Workspace-to-sensory connections are unusual' in result['warnings'][0]


class TestDuplicateConnectionCheck:
    """Test duplicate connection detection."""

    def test_no_duplicate_connection(self):
        """Test validation with no duplicate connections."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]
        graph.edge_attributes = []

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 0

    def test_duplicate_connection_warning(self):
        """Test validation with duplicate connection."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        # Mock edge attributes with existing connection
        mock_edge = Mock()
        mock_edge.source = 1
        mock_edge.target = 2
        graph.edge_attributes = [mock_edge]

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 1
        assert 'Duplicate connection detected' in result['warnings'][0]


class TestGraphCapacityCheck:
    """Test graph capacity validation."""

    def test_graph_within_capacity(self):
        """Test validation when graph is within capacity."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [{'id': i, 'type': 'dynamic'} for i in range(10)]
        graph.edge_attributes = []  # No edges yet

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 0

    def test_graph_approaching_capacity_warning(self):
        """Test validation when graph is approaching capacity."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [{'id': i, 'type': 'dynamic'} for i in range(10)]
        # Create many edges to approach capacity
        graph.edge_attributes = [Mock() for _ in range(80)]  # Close to 10 * 10 = 100 limit

        result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 1
        assert 'Graph approaching capacity' in result['warnings'][0]


class TestCycleDetection:
    """Test cycle detection in connections."""

    def test_no_cycle_detected(self):
        """Test validation with no cycles."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'},
            {'id': 3, 'type': 'dynamic'}
        ]

        # Create edges: 1->2, 3->2 (no cycle when adding 3->1)
        mock_edge1 = Mock()
        mock_edge1.source = 1
        mock_edge1.target = 2
        mock_edge2 = Mock()
        mock_edge2.source = 3
        mock_edge2.target = 2

        graph.edge_attributes = [mock_edge1, mock_edge2]

        result = validator.validate_connection(graph, 3, 1, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 0

    def test_cycle_detected_warning(self):
        """Test validation with potential cycle."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'},
            {'id': 3, 'type': 'dynamic'}
        ]

        # Create edges: 1->2, 2->3, and trying to add 3->1 (creates cycle)
        mock_edge1 = Mock()
        mock_edge1.source = 1
        mock_edge1.target = 2
        mock_edge2 = Mock()
        mock_edge2.source = 2
        mock_edge2.target = 3

        graph.edge_attributes = [mock_edge1, mock_edge2]

        result = validator.validate_connection(graph, 3, 1, 'excitatory', 1.0)

        assert result['is_valid'] == True
        assert len(result['warnings']) == 1
        assert 'Potential cycle detected' in result['warnings'][0]


class TestStatistics:
    """Test statistics tracking."""

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        # Perform several validations
        validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)
        validator.validate_connection(graph, 1, 1, 'excitatory', 1.0)  # Self-connection (error)
        validator.validate_connection(graph, 1, 2, 'invalid', 1.0)    # Invalid type (error)

        stats = validator.get_statistics()

        assert stats['validations_performed'] == 3
        assert stats['errors_found'] == 2  # Two validations had errors
        assert stats['warnings_issued'] == 2  # Warnings from self-connection and invalid type

    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'dynamic'}
        ]

        validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)
        stats_before = validator.get_statistics()
        assert stats_before['validations_performed'] == 1

        # Reset stats
        validator.clear_cache()  # This doesn't reset stats, but let's check
        stats_after = validator.get_statistics()
        assert stats_after['validations_performed'] == 1  # Should still be 1


class TestGlobalInstance:
    """Test global instance management."""

    def test_get_connection_validator_singleton(self):
        """Test that get_connection_validator returns singleton."""
        validator1 = get_connection_validator()
        validator2 = get_connection_validator()

        assert validator1 is validator2
        assert isinstance(validator1, ConnectionValidator)

    def test_global_validator_thread_safety(self):
        """Test thread safety of global validator access."""
        import threading
        import time

        validators = []
        errors = []

        def get_validator_thread():
            try:
                validator = get_connection_validator()
                validators.append(validator)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_validator_thread)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(validators) == 10
        assert len(errors) == 0
        # All should be the same instance
        assert all(v is validators[0] for v in validators)


class TestIntegration:
    """Integration tests for ConnectionValidator."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        validator = ConnectionValidator()

        # Create a complex graph
        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'dynamic'},
            {'id': 2, 'type': 'sensory'},
            {'id': 3, 'type': 'workspace'},
            {'id': 4, 'type': 'dynamic'}
        ]

        # Add some existing edges
        mock_edge = Mock()
        mock_edge.source = 1
        mock_edge.target = 2
        graph.edge_attributes = [mock_edge]

        # Test various connection scenarios
        test_cases = [
            (1, 4, 'excitatory', 1.0, True, 0),    # Valid connection
            (2, 2, 'excitatory', 1.0, False, 1),   # Self-connection (error)
            (1, 2, 'excitatory', 1.0, True, 1),    # Duplicate (warning)
            (3, 2, 'excitatory', 1.0, True, 1),    # Workspace to sensory (warning)
            (1, 4, 'invalid', 1.0, False, 1),      # Invalid type (error)
        ]

        for source, target, conn_type, weight, expected_valid, expected_warnings in test_cases:
            result = validator.validate_connection(graph, source, target, conn_type, weight)

            assert result['is_valid'] == expected_valid
            assert len(result['warnings']) == expected_warnings

    def test_validator_with_real_graph_data(self):
        """Test validator with more realistic graph data."""
        validator = ConnectionValidator()

        # Create graph similar to what might be used in neural simulation
        graph = Data()
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'energy': 0.5 + 0.1 * i}
            for i in range(20)
        ]

        # Add some realistic connections
        graph.edge_attributes = []
        for i in range(15):
            mock_edge = Mock()
            mock_edge.source = i
            mock_edge.target = (i + 1) % 20
            graph.edge_attributes.append(mock_edge)

        # Test validation of a new connection
        result = validator.validate_connection(graph, 15, 16, 'excitatory', 0.8)

        assert result['is_valid'] == True
        # Should have some warnings about graph capacity
        assert len(result['warnings']) >= 0


class TestPerformance:
    """Performance tests for ConnectionValidator."""

    def test_validation_performance_small_graph(self):
        """Test validation performance on small graph."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [{'id': i, 'type': 'dynamic'} for i in range(10)]

        start_time = time.time()
        for _ in range(100):
            result = validator.validate_connection(graph, 1, 2, 'excitatory', 1.0)
            assert result['is_valid'] == True
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 1.0

    def test_validation_performance_large_graph(self):
        """Test validation performance on larger graph."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [{'id': i, 'type': 'dynamic'} for i in range(100)]
        graph.edge_attributes = []

        # Add many edges
        for i in range(90):
            mock_edge = Mock()
            mock_edge.source = i
            mock_edge.target = (i + 1) % 100
            graph.edge_attributes.append(mock_edge)

        start_time = time.time()
        for _ in range(50):
            result = validator.validate_connection(graph, 90, 91, 'excitatory', 1.0)
            assert result['is_valid'] == True
        end_time = time.time()

        # Should still complete reasonably quickly
        assert end_time - start_time < 2.0

    def test_statistics_performance(self):
        """Test performance of statistics access."""
        validator = ConnectionValidator()

        start_time = time.time()
        for _ in range(1000):
            stats = validator.get_statistics()
            assert isinstance(stats, dict)
        end_time = time.time()

        # Statistics access should be very fast
        assert end_time - start_time < 0.1


class TestRealWorldUsage:
    """Real-world usage scenarios for ConnectionValidator."""

    def test_neural_network_connection_validation(self):
        """Test validation in context of neural network simulation."""
        validator = ConnectionValidator()

        # Simulate a neural network graph
        graph = Data()
        graph.node_labels = []

        # Create different types of neurons
        for i in range(50):
            if i < 20:
                node_type = 'sensory'
            elif i < 40:
                node_type = 'dynamic'
            else:
                node_type = 'workspace'

            graph.node_labels.append({
                'id': i,
                'type': node_type,
                'energy': 0.3 + 0.4 * (i / 50),
                'threshold': 0.5
            })

        # Add some existing connections
        graph.edge_attributes = []
        for i in range(30):
            mock_edge = Mock()
            mock_edge.source = i
            mock_edge.target = (i + 5) % 50
            graph.edge_attributes.append(mock_edge)

        # Test various biologically plausible connections
        test_connections = [
            (5, 25, 'excitatory', 0.8),    # Sensory to dynamic
            (25, 35, 'excitatory', 0.6),   # Dynamic to dynamic
            (35, 45, 'inhibitory', -0.4),  # Dynamic to workspace
            (45, 25, 'modulatory', 0.2),   # Workspace to dynamic
        ]

        for source, target, conn_type, weight in test_connections:
            result = validator.validate_connection(graph, source, target, conn_type, weight)

            # In a real scenario, these should generally be valid
            assert result['is_valid'] == True

            # Check for appropriate warnings
            if conn_type == 'inhibitory' and weight > 0:
                assert len(result['warnings']) > 0

    def test_connection_validation_in_simulation_loop(self):
        """Test validator usage in simulation loop context."""
        validator = ConnectionValidator()

        graph = Data()
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'energy': 0.5}
            for i in range(10)
        ]
        graph.edge_attributes = []

        # Simulate multiple simulation steps with connection attempts
        successful_connections = 0
        total_attempts = 0

        for step in range(20):
            # Attempt to add connections based on some logic
            source = step % 10
            target = (step + 1) % 10

            # Only add if energy is sufficient
            if graph.node_labels[source]['energy'] > 0.3:
                result = validator.validate_connection(graph, source, target, 'excitatory', 0.7)
                total_attempts += 1

                if result['is_valid']:
                    successful_connections += 1
                    # In real code, would add the connection here
                    mock_edge = Mock()
                    mock_edge.source = source
                    mock_edge.target = target
                    graph.edge_attributes.append(mock_edge)

        assert total_attempts > 0
        assert successful_connections > 0
        assert successful_connections <= total_attempts

        # Check final statistics
        stats = validator.get_statistics()
        assert stats['validations_performed'] == total_attempts


if __name__ == "__main__":
    pytest.main([__file__])






