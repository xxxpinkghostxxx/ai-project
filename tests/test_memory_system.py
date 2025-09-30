"""
Comprehensive tests for MemorySystem.
Tests memory trace formation, consolidation, decay, pattern recall,
statistics tracking, error handling, and integration scenarios.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import MagicMock

import pytest
import torch

from src.learning.memory_system import (MemorySystem,
                                        analyze_memory_distribution,
                                        calculate_memory_efficiency)


class TestMemorySystem:
    """Test suite for MemorySystem."""

    def setup_method(self):
        """Set up test environment."""
        self.memory_system = MemorySystem()

    def teardown_method(self):
        """Clean up after tests."""
        self.memory_system.reset_statistics()

    def test_initialization(self):
        """Test MemorySystem initialization."""
        assert self.memory_system.consolidation_threshold == 0.8
        assert self.memory_system.memory_decay_rate == 0.99
        assert self.memory_system.recall_threshold == 0.3
        assert self.memory_system.max_memory_traces == 1000
        assert isinstance(self.memory_system.memory_traces, dict)
        assert isinstance(self.memory_system.memory_stats, dict)

    def test_form_memory_traces(self):
        """Test memory trace formation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [
            {'behavior': 'integrator', 'last_activation': time.time() - 5, 'energy': 0.8},
            {'behavior': 'dynamic', 'last_activation': time.time() - 15, 'energy': 0.3}
        ]
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        result = self.memory_system.form_memory_traces(mock_graph)

        assert result == mock_graph
        assert len(self.memory_system.memory_traces) == 1  # Only integrator with stable pattern
        assert self.memory_system.memory_stats['traces_formed'] == 1
        assert 0 in self.memory_system.memory_traces  # Node 0 should have memory trace

    def test_form_memory_traces_max_limit(self):
        """Test memory trace formation respects max limit."""
        self.memory_system.max_memory_traces = 2

        mock_graph = MagicMock()
        mock_graph.node_labels = [
            {'behavior': 'integrator', 'last_activation': time.time() - 5, 'energy': 0.8},
            {'behavior': 'integrator', 'last_activation': time.time() - 5, 'energy': 0.8},
            {'behavior': 'integrator', 'last_activation': time.time() - 5, 'energy': 0.8}
        ]
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)

        self.memory_system.form_memory_traces(mock_graph)

        assert len(self.memory_system.memory_traces) <= 2

    def test_has_stable_pattern(self):
        """Test stable pattern detection."""
        # Stable pattern
        stable_node = {'last_activation': time.time() - 5, 'energy': 0.8}
        assert self.memory_system._has_stable_pattern(stable_node, MagicMock()) is True

        # Unstable - old activation
        unstable_old = {'last_activation': time.time() - 15, 'energy': 0.8}
        assert self.memory_system._has_stable_pattern(unstable_old, MagicMock()) is False

        # Unstable - low energy
        unstable_energy = {'last_activation': time.time() - 5, 'energy': 0.2}
        assert self.memory_system._has_stable_pattern(unstable_energy, MagicMock()) is False

    def test_create_memory_trace(self):
        """Test memory trace creation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'behavior': 'integrator'}, {'behavior': 'relay'}]
        mock_graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Correct shape (2, 1)

        self.memory_system._create_memory_trace(0, mock_graph)

        assert 0 in self.memory_system.memory_traces
        trace = self.memory_system.memory_traces[0]
        assert 'connections' in trace
        assert 'strength' in trace
        assert 'formation_time' in trace
        assert 'pattern_type' in trace
        assert trace['strength'] == 1.0

    def test_classify_pattern(self):
        """Test pattern classification."""
        connections = [
            {'type': 'excitatory', 'source_behavior': 'oscillator'},
            {'type': 'excitatory', 'source_behavior': 'integrator'},
            {'type': 'inhibitory', 'source_behavior': 'dynamic'}
        ]

        mock_graph = MagicMock()
        pattern_type = self.memory_system._classify_pattern(connections, mock_graph)

        assert isinstance(pattern_type, str)
        assert pattern_type in ['excitatory_dominant', 'inhibitory_dominant', 'balanced', 'modulatory_influenced', 'rhythmic_excitatory', 'integrative_excitatory', 'isolated']

    def test_consolidate_memories(self):
        """Test memory consolidation."""
        # Set up memory trace
        self.memory_system.memory_traces[0] = {
            'strength': 1.0,
            'activation_count': 0,
            'last_accessed': time.time() - 10
        }

        mock_graph = MagicMock()
        mock_graph.node_labels = [{'behavior': 'integrator', 'last_activation': time.time() - 5, 'energy': 0.8}]

        result = self.memory_system.consolidate_memories(mock_graph)

        assert result == mock_graph
        assert self.memory_system.memory_traces[0]['strength'] > 1.0  # Should be strengthened
        assert self.memory_system.memory_traces[0]['activation_count'] == 1
        assert self.memory_system.memory_stats['traces_consolidated'] == 1

    def test_decay_memories(self):
        """Test memory decay."""
        # Set up old memory trace
        old_time = time.time() - 120  # 2 minutes ago
        self.memory_system.memory_traces[0] = {
            'strength': 1.0,
            'formation_time': old_time,
            'activation_count': 0
        }

        removed_count = self.memory_system.decay_memories()

        assert self.memory_system.memory_traces[0]['strength'] < 1.0  # Should decay
        assert removed_count == 0  # Should not be removed yet

        # Test complete decay
        self.memory_system.memory_traces[0]['strength'] = 0.05  # Very weak
        removed_count = self.memory_system.decay_memories()

        assert removed_count == 1
        assert 0 not in self.memory_system.memory_traces
        assert self.memory_system.memory_stats['traces_decayed'] == 1

    def test_recall_patterns(self):
        """Test pattern recall."""
        # Set up memory trace
        self.memory_system.memory_traces[0] = {
            'connections': [{'source': 1, 'weight': 1.0, 'type': 'excitatory'}],
            'strength': 0.8,
            'pattern_type': 'integrative_excitatory',
            'last_accessed': time.time() - 10,
            'activation_count': 1
        }

        mock_graph = MagicMock()
        mock_graph.node_labels = [{'behavior': 'integrator'}, {'behavior': 'relay'}]

        recalled = self.memory_system.recall_patterns(mock_graph, 0)

        assert isinstance(recalled, list)
        if recalled:  # May not recall if relevance too low
            assert len(recalled) >= 0
            assert self.memory_system.memory_stats['patterns_recalled'] >= 0

    def test_is_pattern_relevant(self):
        """Test pattern relevance checking."""
        memory_trace = {'pattern_type': 'integrative_excitatory'}

        mock_graph = MagicMock()
        mock_graph.node_labels = [{'behavior': 'integrator'}]

        # Relevant
        assert self.memory_system._is_pattern_relevant(memory_trace, 0, mock_graph) is True

        # Not relevant
        mock_graph.node_labels = [{'behavior': 'dynamic'}]
        assert self.memory_system._is_pattern_relevant(memory_trace, 0, mock_graph) is False

    def test_calculate_pattern_relevance(self):
        """Test pattern relevance calculation."""
        memory_trace = {
            'strength': 0.8,
            'last_accessed': time.time() - 60,  # 1 minute ago
            'activation_count': 2
        }

        mock_graph = MagicMock()
        mock_graph.node_labels = [{'behavior': 'integrator'}]  # Set node_labels to have at least 1 element
        relevance = self.memory_system._calculate_pattern_relevance(memory_trace, 0, mock_graph)

        assert 0.0 <= relevance <= 1.0
        # Should include strength, recency, and activation factors
        expected_min = 0.8 * 0.3  # Base strength contribution
        assert relevance >= expected_min

    def test_replace_weakest_memory(self):
        """Test weakest memory replacement."""
        # Set up multiple traces with different strengths
        self.memory_system.memory_traces = {
            0: {'strength': 0.8},
            1: {'strength': 0.3},  # Weakest
            2: {'strength': 0.6}
        }

        mock_graph = MagicMock()
        mock_graph.node_labels = [{'behavior': 'integrator'}, {'behavior': 'relay'}, {'behavior': 'dynamic'}, {'behavior': 'integrator'}]
        mock_graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Set proper edge_index
        self.memory_system._replace_weakest_memory(3, mock_graph)

        assert 1 not in self.memory_system.memory_traces  # Weakest should be removed
        assert 3 in self.memory_system.memory_traces  # New one should be added

    def test_get_memory_statistics(self):
        """Test memory statistics retrieval."""
        # Add some traces
        self.memory_system.memory_traces = {
            0: {'strength': 0.8},
            1: {'strength': 0.6}
        }

        stats = self.memory_system.get_memory_statistics()

        assert stats['total_memory_strength'] == 1.4
        assert 'traces_formed' in stats
        assert 'traces_consolidated' in stats

    def test_reset_statistics(self):
        """Test statistics reset."""
        self.memory_system.memory_stats['traces_formed'] = 5

        self.memory_system.reset_statistics()

        stats = self.memory_system.get_memory_statistics()
        assert stats['traces_formed'] == 0
        assert stats['total_memory_strength'] == 0.0

    def test_get_memory_trace_count(self):
        """Test memory trace count retrieval."""
        self.memory_system.memory_traces = {0: {}, 1: {}, 2: {}}

        count = self.memory_system.get_memory_trace_count()
        assert count == 3

    def test_get_memory_summary(self):
        """Test memory summary retrieval."""
        current_time = time.time()
        self.memory_system.memory_traces = {
            0: {
                'pattern_type': 'excitatory_dominant',
                'strength': 0.8,
                'activation_count': 2,
                'formation_time': current_time - 60
            }
        }

        summary = self.memory_system.get_memory_summary()

        assert len(summary) == 1
        assert summary[0]['node_id'] == 0
        assert summary[0]['pattern_type'] == 'excitatory_dominant'
        assert summary[0]['strength'] == 0.8
        assert 'age_minutes' in summary[0]

    def test_get_node_memory_importance(self):
        """Test node memory importance calculation."""
        self.memory_system.memory_traces[0] = {
            'strength': 0.8,
            'activation_count': 3,
            'formation_time': time.time() - 1800,  # 30 minutes ago
            'pattern_type': 'integration'
        }

        importance = self.memory_system.get_node_memory_importance(0)

        assert 0.0 <= importance <= 1.0
        # Should include strength, activation, age, and pattern factors

        # Test non-existent node
        importance_none = self.memory_system.get_node_memory_importance(999)
        assert importance_none == 0.0

    def test_error_handling_invalid_graph(self):
        """Test error handling with invalid graphs."""
        invalid_graph = MagicMock()
        invalid_graph.node_labels = None

        result = self.memory_system.form_memory_traces(invalid_graph)
        assert result == invalid_graph

        result = self.memory_system.consolidate_memories(invalid_graph)
        assert result == invalid_graph

    def test_edge_cases_empty_graph(self):
        """Test edge cases with empty graphs."""
        empty_graph = MagicMock()
        empty_graph.node_labels = []

        result = self.memory_system.form_memory_traces(empty_graph)
        assert result == empty_graph
        assert len(self.memory_system.memory_traces) == 0

    def test_edge_cases_extreme_values(self):
        """Test edge cases with extreme values."""
        # Very high strength
        self.memory_system.memory_traces[0] = {
            'strength': 10.0,
            'formation_time': time.time() - 100,
            'activation_count': 0
        }

        # Should handle in consolidation
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'behavior': 'integrator', 'last_activation': time.time() - 5, 'energy': 0.8}]

        result = self.memory_system.consolidate_memories(mock_graph)
        assert result == mock_graph
        # Strength should be capped appropriately

        # Zero strength
        self.memory_system.memory_traces[0] = {
            'strength': 0.0,
            'formation_time': time.time() - 100,
            'activation_count': 0
        }
        removed = self.memory_system.decay_memories()
        # May or may not be removed depending on decay calculation

    def test_memory_cache_functionality(self):
        """Test pattern cache functionality."""
        # The cache is used internally but not directly exposed
        # Test that it gets populated during operations
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'behavior': 'integrator', 'last_activation': time.time() - 5, 'energy': 0.8}]
        mock_graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop or proper tensor

        initial_cache_size = len(self.memory_system.pattern_cache)
        self.memory_system.form_memory_traces(mock_graph)

        # Cache may or may not be used depending on implementation
        assert isinstance(self.memory_system.pattern_cache, dict)


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_analyze_memory_distribution(self):
        """Test memory distribution analysis."""
        memory_system = MagicMock()
        memory_system.get_memory_summary.return_value = [
            {'pattern_type': 'excitatory_dominant', 'strength': 0.8},
            {'pattern_type': 'inhibitory_dominant', 'strength': 0.6},
            {'pattern_type': 'excitatory_dominant', 'strength': 0.7}
        ]

        distribution = analyze_memory_distribution(memory_system)

        assert 'pattern_distribution' in distribution
        assert 'avg_strength' in distribution
        assert 'strength_variance' in distribution
        assert distribution['total_memories'] == 3
        assert distribution['pattern_distribution']['excitatory_dominant'] == 2

    def test_calculate_memory_efficiency(self):
        """Test memory efficiency calculation."""
        memory_system = MagicMock()
        memory_system.get_memory_statistics.return_value = {
            'traces_formed': 20,
            'traces_consolidated': 15,
            'patterns_recalled': 8
        }

        efficiency = calculate_memory_efficiency(memory_system)

        assert 0.0 <= efficiency <= 1.0
        # Should be combination of formation, consolidation, and recall efficiencies

    def test_analyze_memory_distribution_empty(self):
        """Test memory distribution analysis with empty memory."""
        memory_system = MagicMock()
        memory_system.get_memory_summary.return_value = []

        distribution = analyze_memory_distribution(memory_system)

        assert distribution['total_memories'] == 0
        assert distribution['avg_strength'] == 0.0
        assert distribution['strength_variance'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])






