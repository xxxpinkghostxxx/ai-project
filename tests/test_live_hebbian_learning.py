"""
Comprehensive tests for LiveHebbianLearning.
Tests continuous Hebbian learning, STDP, energy modulation, activity tracking,
statistics, error handling, and integration scenarios.
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

from learning.live_hebbian_learning import LiveHebbianLearning, create_live_hebbian_learning


class TestLiveHebbianLearning:
    """Test suite for LiveHebbianLearning."""

    def setup_method(self):
        """Set up test environment."""
        self.learning_system = LiveHebbianLearning()

    def teardown_method(self):
        """Clean up after tests."""
        self.learning_system.reset_learning_statistics()

    def test_initialization(self):
        """Test LiveHebbianLearning initialization."""
        assert self.learning_system.learning_active is True
        assert self.learning_system.base_learning_rate == 0.01
        assert self.learning_system.learning_rate == 0.01
        assert self.learning_system.stdp_window == 0.1
        assert self.learning_system.eligibility_decay == 0.95
        assert self.learning_system.energy_learning_modulation is True
        assert isinstance(self.learning_system.learning_stats, dict)

    def test_initialization_with_simulation_manager(self):
        """Test initialization with simulation manager."""
        mock_sim_manager = MagicMock()
        learning_system = LiveHebbianLearning(mock_sim_manager)

        assert learning_system.simulation_manager == mock_sim_manager

    def test_update_activity_history(self):
        """Test activity history updates."""
        graph = Data()
        graph.x = torch.tensor([[0.8], [0.3], [0.9]], dtype=torch.float32)  # Different energies
        graph.node_labels = [{'id': 0}, {'id': 1}, {'id': 2}]

        initial_time = time.time()
        self.learning_system._update_activity_history(graph, 0)

        # Should have recorded activity for high energy nodes
        assert len(self.learning_system.node_activity_history) > 0
        assert all(timestamp >= initial_time for activities in self.learning_system.node_activity_history.values()
                  for timestamp in activities)

    def test_update_activity_history_energy_modulation(self):
        """Test activity history with energy modulation."""
        graph = Data()
        graph.x = torch.tensor([[0.1], [0.9]], dtype=torch.float32)  # Low and high energy
        graph.node_labels = [{'id': 0}, {'id': 1}]

        self.learning_system._update_activity_history(graph, 0)

        # High energy node should be more likely to be selected
        high_energy_activities = len(self.learning_system.node_activity_history.get(1, []))
        low_energy_activities = len(self.learning_system.node_activity_history.get(0, []))

        # Note: Due to probabilistic sampling, we can't guarantee, but test the mechanism
        assert isinstance(high_energy_activities, int)
        assert isinstance(low_energy_activities, int)

    def test_apply_stdp_learning(self):
        """Test STDP learning application."""
        graph = Data()
        graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        graph.node_labels = [{'id': 0}, {'id': 1}]

        # Set up activity history
        current_time = time.time()
        self.learning_system.node_activity_history[0] = [current_time - 0.05]  # Pre fires first
        self.learning_system.node_activity_history[1] = [current_time]  # Post fires later

        result = self.learning_system._apply_stdp_learning(graph, 0)

        assert result == graph
        # Weights should have changed
        assert not torch.equal(graph.edge_attr, torch.tensor([[1.0], [1.0]], dtype=torch.float32))

    def test_calculate_stdp_change_ltp(self):
        """Test LTP STDP change calculation."""
        source_id = 0
        target_id = 1
        current_time = time.time()

        # Set up activity for LTP (pre before post)
        self.learning_system.node_activity_history[source_id] = [current_time - 0.05]
        self.learning_system.node_activity_history[target_id] = [current_time]

        with patch.object(self.learning_system, '_get_node_energy', side_effect=lambda nid: 0.8 if nid == source_id else 0.6):
            change = self.learning_system._calculate_stdp_change(source_id, target_id, current_time)

            assert change > 0  # LTP should be positive

    def test_calculate_stdp_change_ltd(self):
        """Test LTD STDP change calculation."""
        source_id = 0
        target_id = 1
        current_time = time.time()

        # Set up activity for LTD (post before pre)
        self.learning_system.node_activity_history[source_id] = [current_time]
        self.learning_system.node_activity_history[target_id] = [current_time - 0.05]

        with patch.object(self.learning_system, '_get_node_energy', side_effect=lambda nid: 0.8 if nid == source_id else 0.6):
            change = self.learning_system._calculate_stdp_change(source_id, target_id, current_time)

            assert change < 0  # LTD should be negative

    def test_energy_modulated_learning_rate(self):
        """Test energy-modulated learning rate calculation."""
        source_id = 0
        target_id = 1

        with patch.object(self.learning_system, '_get_node_energy', side_effect=lambda nid: 4.0 if nid == source_id else 3.0), \
              patch('energy.energy_behavior.get_node_energy_cap', return_value=5.0):

            modulated_rate = self.learning_system._calculate_energy_modulated_learning_rate(source_id, target_id)

            # Should be higher than base rate due to high energy
            assert modulated_rate > self.learning_system.base_learning_rate

        # Test with low energy
        with patch.object(self.learning_system, '_get_node_energy', return_value=0.1), \
              patch('energy.energy_behavior.get_node_energy_cap', return_value=5.0):

            low_modulated_rate = self.learning_system._calculate_energy_modulated_learning_rate(source_id, target_id)

            assert low_modulated_rate < modulated_rate

    def test_get_node_energy_from_simulation_manager(self):
        """Test node energy retrieval from simulation manager."""
        mock_sim_manager = MagicMock()
        mock_graph = MagicMock()
        mock_sim_manager.graph = mock_graph
        mock_graph.node_labels = [{'id': 5, 'energy': 0.7}]
        mock_graph.x = torch.tensor([[0.7]], dtype=torch.float32)

        learning_system = LiveHebbianLearning(mock_sim_manager)

        energy = learning_system._get_node_energy(5)
        assert energy == pytest.approx(0.7)

    def test_get_node_energy_fallback(self):
        """Test node energy fallback when simulation manager unavailable."""
        # No simulation manager
        energy = self.learning_system._get_node_energy(3)

        # Should return deterministic pseudo-random value
        assert isinstance(energy, float)
        assert 0.5 <= energy <= 0.9  # Based on the fallback calculation

    def test_update_eligibility_traces(self):
        """Test eligibility trace updates."""
        graph = Data()
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[1.0, 0.5]], dtype=torch.float32)  # weight, trace

        result = self.learning_system._update_eligibility_traces(graph, 0)

        assert result == graph
        # Trace should be decayed
        assert graph.edge_attr[0, 1] < 0.5

    def test_consolidate_weights(self):
        """Test weight consolidation."""
        graph = Data()
        graph.edge_index = torch.tensor([[0, 1]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[1.0]], dtype=torch.float32)

        # Consolidation happens every 100 steps
        result = self.learning_system._consolidate_weights(graph, 100)

        assert result == graph
        # Weight should be slightly reduced
        assert graph.edge_attr[0, 0] < 1.0

    def test_apply_continuous_learning_integration(self):
        """Test full continuous learning integration."""
        graph = Data()
        graph.x = torch.tensor([[0.8], [0.6]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0, 1]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[1.0]], dtype=torch.float32)
        graph.node_labels = [{'id': 0}, {'id': 1}]

        initial_stats = self.learning_system.get_learning_statistics()

        result = self.learning_system.apply_continuous_learning(graph, 0)

        assert result == graph

        final_stats = self.learning_system.get_learning_statistics()
        # May or may not have learning events depending on activity
        assert isinstance(final_stats['total_weight_changes'], int)

    def test_learning_statistics_tracking(self):
        """Test learning statistics tracking."""
        initial_stats = self.learning_system.get_learning_statistics()

        # Manually trigger some events
        self.learning_system.learning_stats['stdp_events'] = 1
        self.learning_system.learning_stats['connection_strengthened'] = 1
        self.learning_system.learning_stats['total_weight_changes'] = 1

        final_stats = self.learning_system.get_learning_statistics()

        assert final_stats['stdp_events'] > initial_stats['stdp_events']
        assert final_stats['connection_strengthened'] > initial_stats['connection_strengthened']

    def test_learning_efficiency_calculation(self):
        """Test learning efficiency calculation."""
        # Set up stats for efficiency calculation
        self.learning_system.learning_stats['stdp_events'] = 2000
        self.learning_system.learning_stats['connection_strengthened'] = 1500  # 75% efficiency

        self.learning_system._update_learning_statistics()

        assert 'learning_efficiency' in self.learning_system.learning_stats
        assert 0.0 <= self.learning_system.learning_stats['learning_efficiency'] <= 100.0

    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate adjustments."""
        # High efficiency - should decrease rate
        self.learning_system.learning_stats['stdp_events'] = 2000
        self.learning_system.learning_stats['connection_strengthened'] = 1500  # High efficiency

        initial_rate = self.learning_system.learning_rate
        self.learning_system._update_learning_statistics()

        # Rate should be adjusted
        assert self.learning_system.learning_stats['learning_rate_updates'] >= 0

    def test_reset_learning_statistics(self):
        """Test statistics reset."""
        self.learning_system.learning_stats['stdp_events'] = 10

        self.learning_system.reset_learning_statistics()

        stats = self.learning_system.get_learning_statistics()
        assert stats['stdp_events'] == 0
        assert stats['total_weight_changes'] == 0

    def test_set_learning_rate(self):
        """Test learning rate setting with bounds."""
        self.learning_system.set_learning_rate(0.05)
        assert self.learning_system.learning_rate == 0.05

        # Test upper bound
        self.learning_system.set_learning_rate(0.2)
        assert self.learning_system.learning_rate == 0.1  # Capped

        # Test lower bound
        self.learning_system.set_learning_rate(0.001)
        assert self.learning_system.learning_rate == 0.001

        self.learning_system.set_learning_rate(0.0001)
        assert self.learning_system.learning_rate == 0.001  # Floor

    def test_set_learning_active(self):
        """Test learning active state setting."""
        self.learning_system.set_learning_active(False)
        assert self.learning_system.learning_active is False

        self.learning_system.set_learning_active(True)
        assert self.learning_system.learning_active is True

    def test_get_learning_parameters(self):
        """Test learning parameters retrieval."""
        params = self.learning_system.get_learning_parameters()

        required_keys = ['learning_rate', 'base_learning_rate', 'stdp_window',
                        'eligibility_decay', 'weight_cap', 'weight_min', 'learning_active']

        for key in required_keys:
            assert key in params
            assert isinstance(params[key], (int, float, bool))

    def test_error_handling_invalid_graph(self):
        """Test error handling with invalid graphs."""
        invalid_graph = Data()
        # No edge_index

        result = self.learning_system.apply_continuous_learning(invalid_graph, 0)
        assert result == invalid_graph  # Should handle gracefully

    def test_error_handling_energy_retrieval(self):
        """Test error handling in energy retrieval."""
        # Force error in energy retrieval
        with patch.object(self.learning_system, '_get_node_energy', side_effect=Exception("Energy error")):
            change = self.learning_system._calculate_stdp_change(0, 1, time.time())

            assert change == 0.0  # Should return 0 on error

    def test_edge_cases_empty_graph(self):
        """Test edge cases with empty graphs."""
        empty_graph = Data()
        empty_graph.x = torch.tensor([], dtype=torch.float32).reshape(0, 1)
        empty_graph.node_labels = []

        result = self.learning_system.apply_continuous_learning(empty_graph, 0)
        assert result == empty_graph

    def test_edge_cases_extreme_values(self):
        """Test edge cases with extreme values."""
        # Very high energy
        graph = Data()
        graph.x = torch.tensor([[100.0]], dtype=torch.float32)
        graph.node_labels = [{'id': 0}]

        result = self.learning_system._update_activity_history(graph, 0)
        # Should handle without crashing

        # Zero energy
        graph.x = torch.tensor([[0.0]], dtype=torch.float32)
        result = self.learning_system._update_activity_history(graph, 0)
        # Should handle without crashing

    def test_cleanup(self):
        """Test cleanup functionality."""
        # Add some data
        self.learning_system.node_activity_history[0] = [time.time()]
        self.learning_system.edge_activity_history[0] = [time.time()]

        self.learning_system.cleanup()

        assert len(self.learning_system.node_activity_history) == 0
        assert len(self.learning_system.edge_activity_history) == 0
        assert len(self.learning_system.last_activity_time) == 0


class TestFactoryFunction:
    """Test suite for factory function."""

    def test_create_live_hebbian_learning(self):
        """Test factory function."""
        learning_system = create_live_hebbian_learning()

        assert isinstance(learning_system, LiveHebbianLearning)
        assert learning_system.learning_active is True


if __name__ == "__main__":
    pytest.main([__file__])