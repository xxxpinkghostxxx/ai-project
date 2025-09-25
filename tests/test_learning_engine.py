"""
Comprehensive tests for LearningEngine.
Tests STDP learning, memory consolidation, energy modulation, event handling,
statistics tracking, error handling, and integration scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data

from src.learning.learning_engine import LearningEngine, calculate_learning_efficiency, detect_learning_patterns


class TestLearningEngine:
    """Test suite for LearningEngine."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_access_layer = MagicMock()
        with patch('learning.learning_engine.get_learning_config', return_value={
            'plasticity_rate': 0.01,
            'eligibility_decay': 0.95,
            'stdp_window': 20.0,
            'ltp_rate': 0.02,
            'ltd_rate': 0.01
        }):
            self.engine = LearningEngine(self.mock_access_layer)

    def teardown_method(self):
        """Clean up after tests."""
        self.engine.reset_statistics()

    def test_initialization(self):
        """Test LearningEngine initialization."""
        assert self.engine.learning_rate == 0.01
        assert self.engine.eligibility_decay == 0.95
        assert self.engine.stdp_window == 20.0
        assert self.engine.ltp_rate == 0.02
        assert self.engine.ltd_rate == 0.01
        assert self.engine.energy_learning_modulation is True
        assert isinstance(self.engine.learning_stats, dict)
        assert isinstance(self.engine.memory_traces, dict)

    def test_get_node_energy(self):
        """Test node energy retrieval."""
        # Test with energy field
        node_with_energy = {'energy': 0.8}
        energy = self.engine._get_node_energy(node_with_energy)
        assert energy == 0.8

        # Test with membrane_potential fallback
        node_with_mp = {'membrane_potential': 0.6}
        energy = self.engine._get_node_energy(node_with_mp)
        assert energy == 0.6

        # Test with None node
        energy = self.engine._get_node_energy(None)
        assert energy == 0.0

    def test_energy_modulated_rate_calculation(self):
        """Test energy-modulated learning rate calculation."""
        pre_node = {'energy': 0.8}
        post_node = {'energy': 0.6}
        base_rate = 0.02

        with patch('learning.learning_engine.get_node_energy_cap', return_value=5.0):
            modulated_rate = self.engine._calculate_energy_modulated_rate(pre_node, post_node, base_rate)

            # Should be higher than base rate due to high energy
            assert modulated_rate > base_rate
            assert modulated_rate <= base_rate * 1.5  # Max modulation

        # Test with low energy
        low_pre = {'energy': 0.1}
        low_post = {'energy': 0.1}
        low_modulated = self.engine._calculate_energy_modulated_rate(low_pre, low_post, base_rate)

        assert low_modulated < modulated_rate
        assert low_modulated >= base_rate * 0.3  # Min modulation

    def test_apply_timing_learning_ltp(self):
        """Test LTP (Long-Term Potentiation) learning."""
        pre_node = {'id': 1, 'energy': 0.8}
        post_node = {'id': 2, 'energy': 0.6}
        edge = MagicMock()
        edge.eligibility_trace = 0.0
        delta_t = 5.0  # Post after pre, within window

        with patch('learning.learning_engine.get_node_energy_cap', return_value=5.0):
            weight_change = self.engine.apply_timing_learning(pre_node, post_node, edge, delta_t)

            assert weight_change > 0  # LTP should increase weight
            assert edge.eligibility_trace == weight_change
            assert self.engine.learning_stats['stdp_events'] == 1
            assert self.engine.learning_stats['energy_modulated_events'] == 1

    def test_apply_timing_learning_ltd(self):
        """Test LTD (Long-Term Depression) learning."""
        pre_node = {'id': 1, 'energy': 0.8}
        post_node = {'id': 2, 'energy': 0.6}
        edge = MagicMock()
        edge.eligibility_trace = 0.0
        delta_t = -3.0  # Pre after post, within window

        with patch('learning.learning_engine.get_node_energy_cap', return_value=5.0):
            weight_change = self.engine.apply_timing_learning(pre_node, post_node, edge, delta_t)

            assert weight_change < 0  # LTD should decrease weight
            assert edge.eligibility_trace == weight_change
            assert self.engine.learning_stats['stdp_events'] == 1

    def test_apply_timing_learning_outside_window(self):
        """Test learning outside STDP window."""
        pre_node = {'id': 1, 'energy': 0.8}
        post_node = {'id': 2, 'energy': 0.6}
        edge = MagicMock()
        delta_t = 50.0  # Outside window

        weight_change = self.engine.apply_timing_learning(pre_node, post_node, edge, delta_t)

        assert weight_change == 0.0
        assert self.engine.learning_stats['stdp_events'] == 0

    def test_on_spike_event_handling(self):
        """Test spike event handling."""
        self.mock_access_layer.get_node_by_id.side_effect = lambda node_id: {'id': node_id, 'energy': 0.7}

        event_data = {'node_id': 2, 'source_id': 1}

        with patch.object(self.engine, 'emit_learning_update') as mock_emit:
            self.engine._on_spike('SPIKE', event_data)

            mock_emit.assert_called_once_with(1, 2, 0.0)  # delta_t=0, so no change

    def test_consolidate_connections(self):
        """Test connection consolidation."""
        mock_graph = MagicMock()
        mock_graph.edge_attributes = []

        # Create mock edges
        edge1 = MagicMock()
        edge1.eligibility_trace = 0.6  # Above threshold
        edge1.weight = 1.0
        edge1.source = 0
        edge1.target = 1

        edge2 = MagicMock()
        edge2.eligibility_trace = 0.3  # Below threshold
        edge2.weight = 1.0
        edge2.source = 1
        edge2.target = 2

        mock_graph.edge_attributes = [edge1, edge2]
        mock_graph.node_labels = [{'energy': 0.8}, {'energy': 0.6}, {'energy': 0.7}]

        with patch('learning.learning_engine.get_node_energy_cap', return_value=5.0):
            result = self.engine.consolidate_connections(mock_graph)

            assert result == mock_graph
            assert edge1.weight > 1.0  # Should be increased
            assert edge2.weight == 1.0  # Should remain same
            assert self.engine.learning_stats['consolidation_events'] == 1

    def test_form_memory_traces(self):
        """Test memory trace formation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [
            {'behavior': 'integrator', 'last_activation': time.time() - 5, 'energy': 0.8},
            {'behavior': 'dynamic', 'last_activation': time.time() - 15, 'energy': 0.3}
        ]
        mock_graph.edge_attributes = []

        result = self.engine.form_memory_traces(mock_graph)

        assert result == mock_graph
        assert len(self.engine.memory_traces) == 1  # Only integrator with recent activation
        assert self.engine.learning_stats['memory_traces_formed'] == 1

    def test_apply_memory_influence(self):
        """Test memory influence application."""
        # Set up memory trace
        self.engine.memory_traces[1] = {
            'connections': [{'source': 0, 'weight': 1.0, 'type': 'excitatory'}],
            'strength': 0.8,
            'formation_time': time.time() - 10,
            'activation_count': 1,
            'last_accessed': time.time() - 10
        }

        mock_graph = MagicMock()
        mock_graph.edge_attributes = []
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]

        # Create matching edge
        edge = MagicMock()
        edge.target = 1
        edge.source = 0
        edge.type = 'excitatory'
        edge.weight = 1.0
        mock_graph.edge_attributes = [edge]

        result = self.engine.apply_memory_influence(mock_graph)

        assert result == mock_graph
        assert edge.weight > 1.0  # Should be reinforced

    def test_statistics_tracking(self):
        """Test learning statistics tracking."""
        initial_stats = self.engine.get_learning_statistics()

        # Perform some learning
        pre_node = {'id': 1, 'energy': 0.8}
        post_node = {'id': 2, 'energy': 0.6}
        edge = MagicMock()
        edge.eligibility_trace = 0.0

        with patch('learning.learning_engine.get_node_energy_cap', return_value=5.0):
            self.engine.apply_timing_learning(pre_node, post_node, edge, 5.0)

        final_stats = self.engine.get_learning_statistics()

        assert final_stats['stdp_events'] > initial_stats['stdp_events']
        assert final_stats['energy_modulated_events'] > initial_stats['energy_modulated_events']

    def test_reset_statistics(self):
        """Test statistics reset."""
        # Add some stats
        self.engine.learning_stats['stdp_events'] = 5

        self.engine.reset_statistics()

        stats = self.engine.get_learning_statistics()
        assert stats['stdp_events'] == 0
        assert stats['weight_changes'] == 0

    def test_error_handling_energy_modulation(self):
        """Test error handling in energy modulation."""
        pre_node = {'id': 1, 'energy': 'invalid'}  # Invalid energy type
        post_node = {'id': 2, 'energy': 0.6}
        base_rate = 0.02

        # Should fall back to base rate
        modulated_rate = self.engine._calculate_energy_modulated_rate(pre_node, post_node, base_rate)
        assert modulated_rate == base_rate

    def test_edge_cases_empty_graph(self):
        """Test edge cases with empty graphs."""
        empty_graph = MagicMock()
        empty_graph.edge_attributes = None

        result = self.engine.consolidate_connections(empty_graph)
        assert result == empty_graph

        result = self.engine.form_memory_traces(empty_graph)
        assert result == empty_graph

    def test_edge_cases_extreme_values(self):
        """Test edge cases with extreme values."""
        # Very high energy
        high_energy_pre = {'energy': 100.0}
        high_energy_post = {'energy': 100.0}
        base_rate = 0.02

        with patch('learning.learning_engine.get_node_energy_cap', return_value=5.0):
            modulated_rate = self.engine._calculate_energy_modulated_rate(high_energy_pre, high_energy_post, base_rate)

            # Should be clamped to max modulation
            assert modulated_rate == base_rate * 1.5

        # Zero energy
        zero_energy_pre = {'energy': 0.0}
        zero_energy_post = {'energy': 0.0}

        modulated_rate_zero = self.engine._calculate_energy_modulated_rate(zero_energy_pre, zero_energy_post, base_rate)

        # Should be at base modulation (no modulation for zero energy)
        assert modulated_rate_zero == base_rate * 1.0

    def test_memory_decay(self):
        """Test memory trace decay."""
        # Set up old memory trace
        old_time = time.time() - 120  # 2 minutes ago
        self.engine.memory_traces[1] = {
            'connections': [],
            'strength': 1.0,
            'formation_time': old_time,
            'activation_count': 0,
            'last_accessed': old_time
        }

        mock_graph = MagicMock()
        mock_graph.edge_attributes = []

        self.engine.apply_memory_influence(mock_graph)

        # Strength should have decayed
        assert self.engine.memory_traces[1]['strength'] < 1.0

    def test_has_stable_pattern(self):
        """Test stable pattern detection."""
        # Stable pattern
        stable_node = {'last_activation': time.time() - 5, 'energy': 0.8}
        assert self.engine._has_stable_pattern(stable_node, MagicMock()) is True

        # Unstable pattern - old activation
        unstable_node = {'last_activation': time.time() - 20, 'energy': 0.8}
        assert self.engine._has_stable_pattern(unstable_node, MagicMock()) is False

        # Unstable pattern - low energy
        low_energy_node = {'last_activation': time.time() - 5, 'energy': 0.2}
        assert self.engine._has_stable_pattern(low_energy_node, MagicMock()) is False


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_calculate_learning_efficiency(self):
        """Test learning efficiency calculation."""
        mock_graph = MagicMock()
        mock_graph.edge_attributes = [
            MagicMock(weight=1.5),
            MagicMock(weight=2.0),
            MagicMock(weight=1.8)
        ]

        efficiency = calculate_learning_efficiency(mock_graph)
        assert 0.0 <= efficiency <= 1.0

        # Test with no edges
        mock_graph.edge_attributes = []
        efficiency_empty = calculate_learning_efficiency(mock_graph)
        assert efficiency_empty == 0.0

    def test_detect_learning_patterns(self):
        """Test learning pattern detection."""
        mock_graph = MagicMock()
        mock_graph.edge_attributes = []

        # No edges
        patterns = detect_learning_patterns(mock_graph)
        assert patterns['patterns_detected'] == 0
        assert patterns['total_connections'] == 0

        # With edges
        mock_graph.edge_attributes = [
            MagicMock(weight=1.5, type='excitatory'),
            MagicMock(weight=2.0, type='inhibitory'),
            MagicMock(weight=1.8, type='excitatory')
        ]

        patterns = detect_learning_patterns(mock_graph)
        assert patterns['total_connections'] == 3
        assert 'weight_variance' in patterns
        assert 'edge_type_distribution' in patterns


if __name__ == "__main__":
    pytest.main([__file__])






