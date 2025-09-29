"""
Comprehensive edge case and error handling tests for neural components.
Tests boundary conditions, invalid inputs, extreme values, and error recovery.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.neural.behavior_engine import BehaviorEngine
from src.neural.connection_logic import (create_weighted_connection,
                                         intelligent_connection_formation)
from src.neural.enhanced_neural_dynamics import EnhancedNeuralDynamics
from src.neural.network_metrics import NetworkMetrics
from src.neural.neural_map_persistence import NeuralMapPersistence
from src.neural.spike_queue_system import Spike, SpikeQueueSystem, SpikeType
from src.neural.workspace_engine import WorkspaceEngine


class TestNeuralEdgeCases:
    """Edge case and error handling tests for neural components."""

    def test_behavior_engine_extreme_values(self):
        """Test BehaviorEngine with extreme energy values."""
        engine = BehaviorEngine()
        graph = Data()
        graph.node_labels = [{
            'id': 0,
            'behavior': 'dynamic',
            'energy': float('inf'),  # Infinite energy
            'state': 'active'
        }]

        # Should handle gracefully
        result = engine.update_node_behavior(0, graph, 1)
        assert isinstance(result, bool)

        # Test with negative energy
        graph.node_labels[0]['energy'] = -1000.0
        result = engine.update_node_behavior(0, graph, 1)
        assert isinstance(result, bool)

        # Test with NaN energy
        graph.node_labels[0]['energy'] = float('nan')
        result = engine.update_node_behavior(0, graph, 1)
        assert isinstance(result, bool)

    def test_connection_logic_invalid_nodes(self):
        """Test connection logic with invalid node configurations."""
        graph = Data()
        graph.node_labels = [{'id': 0}, {'id': 1}]
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Test connection with non-existent nodes
        result = create_weighted_connection(graph, 999, 1000, 0.5, 'excitatory')
        assert result == graph  # Should return graph unchanged

        # Test with invalid weight
        result = create_weighted_connection(graph, 0, 1, float('inf'), 'excitatory')
        assert result == graph

        # Test with invalid edge type
        result = create_weighted_connection(graph, 0, 1, 0.5, None)
        assert result == graph

    def test_enhanced_dynamics_extreme_parameters(self):
        """Test EnhancedNeuralDynamics with extreme parameter values."""
        dynamics = EnhancedNeuralDynamics()

        # Test with extreme neuromodulator levels
        dynamics.set_neuromodulator_level('dopamine', float('inf'))
        assert dynamics.dopamine_level == 1.0  # Should be clamped

        dynamics.set_neuromodulator_level('dopamine', -1000.0)
        assert dynamics.dopamine_level == 0.0  # Should be clamped

        # Test with NaN
        dynamics.set_neuromodulator_level('dopamine', float('nan'))
        assert dynamics.dopamine_level == 0.0

    def test_network_metrics_empty_graph(self):
        """Test NetworkMetrics with completely empty graph."""
        metrics = NetworkMetrics()

        graph = Data()
        graph.node_labels = []
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

        # Should handle gracefully
        criticality = metrics.calculate_criticality(graph)
        assert criticality == 0.0

        connectivity = metrics.analyze_connectivity(graph)
        assert connectivity['num_nodes'] == 0
        assert connectivity['num_edges'] == 0

    def test_spike_system_overflow(self):
        """Test SpikeQueueSystem with queue overflow."""
        system = SpikeQueueSystem()

        # Fill queue beyond capacity
        max_size = 100000  # Default max size
        for i in range(max_size + 100):
            system.schedule_spike(i % 100, (i + 1) % 100, SpikeType.EXCITATORY, 1.0, 0.8)

        # Should have dropped some spikes
        stats = system.get_statistics()
        assert stats['propagator_stats']['queue_stats']['dropped_spikes'] > 0

    def test_workspace_engine_zero_capacity(self):
        """Test WorkspaceEngine with zero workspace capacity."""
        engine = WorkspaceEngine()
        graph = Data()
        graph.node_labels = [{
            'id': 0,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 0.0,  # Zero capacity
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }]

        # Should handle zero capacity
        engine._update_workspace_node(graph, 0, 1)
        # Should not crash

    def test_persistence_corrupted_data(self):
        """Test NeuralMapPersistence with corrupted data."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = NeuralMapPersistence(temp_dir)

            # Create corrupted JSON file
            filepath = os.path.join(temp_dir, "neural_map_slot_0.json")
            with open(filepath, 'w') as f:
                f.write("invalid json content")

            # Should handle corruption gracefully
            loaded = persistence.load_neural_map(0)
            assert loaded is None

    def test_behavior_engine_missing_properties(self):
        """Test BehaviorEngine with missing node properties."""
        engine = BehaviorEngine()
        graph = Data()
        graph.node_labels = [{
            'id': 0,
            'behavior': 'dynamic'
            # Missing energy, state, etc.
        }]

        # Should handle missing properties
        result = engine.update_node_behavior(0, graph, 1)
        assert isinstance(result, bool)

    def test_connection_logic_circular_dependencies(self):
        """Test connection logic with potential circular dependencies."""
        graph = Data()
        graph.node_labels = [{'id': i} for i in range(10)]
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Create many connections that might create cycles
        for i in range(9):
            graph = create_weighted_connection(graph, i, i + 1, 0.5, 'excitatory')
            graph = create_weighted_connection(graph, i + 1, i, 0.5, 'inhibitory')  # Reverse connection

        # Should handle without issues
        assert len(graph.edge_attributes) > 0

    def test_dynamics_memory_leak_prevention(self):
        """Test EnhancedNeuralDynamics memory management."""
        dynamics = EnhancedNeuralDynamics()

        # Simulate many processing cycles
        graph = Data()
        graph.node_labels = [{'id': 0, 'enhanced_behavior': True, 'membrane_potential': 0.5}]

        for i in range(2000):  # More than maxlen of 1000
            dynamics.update_neural_dynamics(graph, i)

        # Should not have memory leaks (deques should be limited)
        assert len(dynamics.node_activity_history[0]) <= 1000
        assert len(dynamics.spike_times[0]) <= 1000  # Assuming spikes generated

    def test_metrics_calculation_division_by_zero(self):
        """Test NetworkMetrics division by zero handling."""
        metrics = NetworkMetrics()

        # Test E/I ratio with no inhibitory connections
        graph = Data()
        graph.edge_attributes = [
            MagicMock(type='excitatory', weight=1.0),
            MagicMock(type='excitatory', weight=1.0)
        ]

        ratio = metrics._calculate_ei_ratio(graph)
        assert ratio == 10.0  # Should return high ratio when no inhibitory

        # Test with zero inhibitory weight
        graph.edge_attributes.append(MagicMock(type='inhibitory', weight=0.0))
        ratio = metrics._calculate_ei_ratio(graph)
        assert ratio == 10.0

    def test_spike_timestamp_ordering(self):
        """Test spike ordering with extreme timestamps."""
        system = SpikeQueueSystem()

        # Schedule spikes with extreme timestamps
        system.schedule_spike(0, 1, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=float('inf'))
        system.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=float('-inf'))
        system.schedule_spike(2, 3, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=float('nan'))

        # Should handle without crashing
        processed = system.process_spikes(10)
        assert isinstance(processed, int)

    def test_workspace_engine_extreme_random_values(self):
        """Test WorkspaceEngine with extreme random values."""
        engine = WorkspaceEngine()
        graph = Data()
        graph.node_labels = [{
            'id': 0,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }]

        # Test with random returning extreme values
        with patch('numpy.random.random', return_value=1.0):  # Always above threshold
            engine._update_workspace_node(graph, 0, 1)

        with patch('numpy.random.random', return_value=0.0):  # Always below threshold
            engine._update_workspace_node(graph, 0, 1)

        # Should not crash with extreme random values

    def test_persistence_large_metadata(self):
        """Test NeuralMapPersistence with very large metadata."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = NeuralMapPersistence(temp_dir)

            graph = Data()
            graph.node_labels = [{'id': 0}]

            # Create very large metadata
            large_metadata = {'large_data': 'x' * 1000000}  # 1MB string

            result = persistence.save_neural_map(graph, 0, large_metadata)
            assert result is True

            loaded = persistence.load_neural_map(0)
            assert loaded is not None

    def test_behavior_engine_concurrent_modification(self):
        """Test BehaviorEngine with concurrent graph modification."""
        engine = BehaviorEngine()
        graph = Data()
        graph.node_labels = [{
            'id': 0,
            'behavior': 'dynamic',
            'energy': 0.5,
            'state': 'active'
        }]

        # Simulate concurrent modification
        def modify_graph():
            graph.node_labels[0]['energy'] = 0.9

        with patch('threading.RLock'):  # Disable locking to test concurrency
            import threading
            thread = threading.Thread(target=modify_graph)
            thread.start()

            # Update behavior while graph is being modified
            result = engine.update_node_behavior(0, graph, 1)

            thread.join()
            # Should handle concurrent access

    def test_connection_validation_extreme_weights(self):
        """Test connection validation with extreme weights."""
        from src.utils.connection_validator import get_connection_validator

        validator = get_connection_validator()
        graph = Data()
        graph.node_labels = [{'id': 0}, {'id': 1}]

        # Test with extreme weights
        result = validator.validate_connection(graph, 0, 1, 'excitatory', float('inf'))
        assert not result['is_valid']

        result = validator.validate_connection(graph, 0, 1, 'excitatory', float('-inf'))
        assert not result['is_valid']

        result = validator.validate_connection(graph, 0, 1, 'excitatory', float('nan'))
        assert not result['is_valid']

    def test_dynamics_parameter_validation(self):
        """Test EnhancedNeuralDynamics parameter validation."""
        dynamics = EnhancedNeuralDynamics()

        # Test _validate_float with extreme values
        result = dynamics._validate_float(float('inf'), 0.0, 100.0, 'test')
        assert result == 100.0

        result = dynamics._validate_float(float('-inf'), 0.0, 100.0, 'test')
        assert result == 0.0

        result = dynamics._validate_float(float('nan'), 0.0, 100.0, 'test')
        assert result == 0.0

        result = dynamics._validate_float(None, 0.0, 100.0, 'test')
        assert result == 0.0

    def test_metrics_calculation_with_none_values(self):
        """Test NetworkMetrics with None values in graph."""
        metrics = NetworkMetrics()

        graph = Data()
        graph.node_labels = None
        graph.edge_index = None
        graph.x = None

        # Should handle None values
        criticality = metrics.calculate_criticality(graph)
        assert criticality == 0.0

        connectivity = metrics.analyze_connectivity(graph)
        assert connectivity['num_nodes'] == 0

        energy_balance = metrics.measure_energy_balance(graph)
        assert energy_balance['total_energy'] == 0.0

    def test_spike_queue_corruption_recovery(self):
        """Test SpikeQueue corruption recovery."""
        from src.neural.spike_queue_system import SpikeQueue

        queue = SpikeQueue()

        # Manually corrupt the heap
        queue._queue = [Spike(1, 2, 1.0, SpikeType.EXCITATORY, 1.0),
                       Spike(0, 1, 0.5, SpikeType.EXCITATORY, 1.0)]  # Wrong order

        # Next push should trigger validation and repair
        queue.push(Spike(2, 3, 2.0, SpikeType.EXCITATORY, 1.0))

        # Should have repaired the heap
        assert queue._queue[0].timestamp <= queue._queue[-1].timestamp

    def test_workspace_engine_extreme_parameters(self):
        """Test WorkspaceEngine with extreme parameter values."""
        engine = WorkspaceEngine()

        # Test with extreme workspace parameters
        graph = Data()
        graph.node_labels = [{
            'id': 0,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': float('inf'),
            'workspace_creativity': float('-inf'),
            'workspace_focus': float('nan'),
            'threshold': 0.6,
            'state': 'active'
        }]

        # Should handle extreme values
        engine._update_workspace_node(graph, 0, 1)
        # Should not crash

    def test_persistence_path_traversal_attempt(self):
        """Test NeuralMapPersistence against path traversal attacks."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = NeuralMapPersistence(temp_dir)

            # Attempt path traversal
            malicious_slot = "../../../etc/passwd"

            # Should reject invalid slot numbers
            result = persistence.save_neural_map(Data(), malicious_slot)
            assert result is False

            loaded = persistence.load_neural_map(malicious_slot)
            assert loaded is None

    def test_behavior_engine_refractory_extremes(self):
        """Test BehaviorEngine refractory period extremes."""
        engine = BehaviorEngine()
        graph = Data()
        graph.node_labels = [{
            'id': 0,
            'behavior': 'oscillator',
            'oscillation_freq': 0.1,
            'threshold': 0.8,
            'refractory_timer': float('inf'),  # Infinite refractory
            'membrane_potential': 0.9
        }]

        # Should handle infinite refractory
        result = engine.update_node_behavior(0, graph, 1)
        assert isinstance(result, bool)

    def test_connection_logic_memory_limits(self):
        """Test connection logic with memory constraints."""
        graph = Data()
        graph.node_labels = [{'id': i} for i in range(100000)]  # Very large graph
        graph.x = torch.zeros(100000, 1)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # intelligent_connection_formation should skip large graphs
        result = intelligent_connection_formation(graph)
        assert result == graph  # Should return unchanged for large graphs

    def test_dynamics_statistics_overflow(self):
        """Test EnhancedNeuralDynamics statistics overflow."""
        dynamics = EnhancedNeuralDynamics()

        # Simulate statistics overflow
        dynamics.stats['total_spikes'] = 2**63 - 1  # Max int64

        # Should handle large numbers
        stats = dynamics.get_statistics()
        assert 'total_spikes' in stats

    def test_network_health_with_extreme_metrics(self):
        """Test network health calculation with extreme metric values."""
        metrics = NetworkMetrics()

        # Set extreme metrics
        metrics.last_metrics = {
            'criticality': float('inf'),
            'connectivity': {'density': float('-inf')},
            'energy_balance': {'energy_variance': float('nan')},
            'performance': {'calculation_time': 0.0}
        }

        health = metrics.get_network_health_score()
        assert isinstance(health['score'], float)  # Should handle extremes

    def test_spike_system_timestamp_precision(self):
        """Test SpikeQueueSystem with high-precision timestamps."""
        system = SpikeQueueSystem()

        # Schedule spikes with very close timestamps
        base_time = 1000.0
        for i in range(10):
            system.schedule_spike(i, i+1, SpikeType.EXCITATORY, 1.0, 0.8,
                                timestamp=base_time + i * 1e-9)  # Nanosecond precision

        # Should maintain order
        processed = system.process_spikes(20)
        assert processed >= 0

    def test_workspace_engine_probability_edge_cases(self):
        """Test WorkspaceEngine synthesis probability edge cases."""
        engine = WorkspaceEngine()
        graph = Data()
        graph.node_labels = [{
            'id': 0,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 5.0,
            'workspace_creativity': 0.0,  # Zero creativity
            'workspace_focus': 0.0,  # Zero focus
            'threshold': 0.6,
            'state': 'active'
        }]

        # Probability should be 0.0 * 0.0 * 0.1 = 0.0
        with patch('numpy.random.random', return_value=0.5):  # Any value > 0.0
            engine._update_workspace_node(graph, 0, 1)
            # Should not synthesize due to zero probability

        # Test with infinite creativity/focus
        graph.node_labels[0]['workspace_creativity'] = float('inf')
        graph.node_labels[0]['workspace_focus'] = float('inf')

        with patch('numpy.random.random', return_value=0.5):
            engine._update_workspace_node(graph, 0, 1)
            # Should handle infinite values

    def test_persistence_concurrent_access(self):
        """Test NeuralMapPersistence with concurrent access."""
        import tempfile
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = NeuralMapPersistence(temp_dir)
            graph = Data()
            graph.node_labels = [{'id': 0}]

            errors = []

            def save_operation(slot):
                try:
                    persistence.save_neural_map(graph, slot)
                except Exception as e:
                    errors.append(e)

            threads = []
            for i in range(5):
                thread = threading.Thread(target=save_operation, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            assert len(errors) == 0

    def test_all_components_with_none_graph(self):
        """Test all components with None graph input."""
        # Test that all components handle None graph gracefully
        components = [
            BehaviorEngine(),
            EnhancedNeuralDynamics(),
            NetworkMetrics(),
            SpikeQueueSystem(),
            WorkspaceEngine()
        ]

        for component in components:
            try:
                if hasattr(component, 'update_neural_dynamics'):
                    result = component.update_neural_dynamics(None, 1)
                    assert result is None
                elif hasattr(component, 'calculate_comprehensive_metrics'):
                    result = component.calculate_comprehensive_metrics(None)
                    assert isinstance(result, dict)
                # Other components don't have direct graph processing methods
            except Exception as e:
                pytest.fail(f"Component {component.__class__.__name__} failed with None graph: {e}")


if __name__ == "__main__":
    pytest.main([__file__])






