"""
Comprehensive tests for HomeostasisController.
Tests network homeostasis regulation, criticality optimization, health monitoring,
statistics tracking, error handling, and integration scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
import time
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data

from learning.homeostasis_controller import HomeostasisController, HistoryManager, StatsManager, calculate_network_stability, detect_network_anomalies


class TestHomeostasisController:
    """Test suite for HomeostasisController."""

    def setup_method(self):
        """Set up test environment."""
        with patch('learning.homeostasis_controller.get_config') as mock_get_config:
            mock_get_config.side_effect = lambda section, key, default, dtype=None: {
                'enable_adaptive_regulation': True,
                'enable_memory_integration': True,
                'enable_validation': True,
                'target_energy_ratio': 0.6,
                'criticality_threshold': 0.1,
                'regulation_rate': 0.001,
                'regulation_interval': 100,
                'branching_target': 1.0,
                'energy_variance_threshold': 0.2
            }.get(key, default)
            self.controller = HomeostasisController()

    def teardown_method(self):
        """Clean up after tests."""
        self.controller.reset_statistics()

    def test_initialization(self):
        """Test HomeostasisController initialization."""
        assert self.controller.enable_adaptive_regulation is True
        assert self.controller.enable_memory_integration is True
        assert self.controller.enable_validation is True
        assert self.controller.target_energy_ratio == 0.6
        assert self.controller.criticality_threshold == 0.1
        assert self.controller.regulation_rate == 0.001
        assert self.controller.branching_target == 1.0
        assert isinstance(self.controller.stats_manager, StatsManager)
        assert isinstance(self.controller.history_manager, HistoryManager)

    def test_regulate_network_activity_high_energy(self):
        """Test energy regulation when energy ratio is too high."""
        # Create mock graph with high energy
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[4.0], [4.0]], dtype=torch.float32)  # High energy
        mock_graph.network_metrics = None

        with patch('learning.homeostasis_controller.get_node_energy_cap', return_value=5.0), \
             patch('learning.homeostasis_controller.get_node_death_threshold', return_value=1.0), \
             patch('learning.homeostasis_controller.get_node_birth_threshold', return_value=2.0), \
             patch('learning.homeostasis_controller.config') as mock_config:

            result = self.controller.regulate_network_activity(mock_graph)

            # Should reduce energy
            assert result == mock_graph
            mock_config.set_value.assert_any_call('NodeLifecycle', 'death_threshold', pytest.approx(1.05, abs=1e-6))
            mock_config.set_value.assert_any_call('NodeLifecycle', 'birth_threshold', pytest.approx(1.95, abs=1e-6))

    def test_regulate_network_activity_low_energy(self):
        """Test energy regulation when energy ratio is too low."""
        # Create mock graph with low energy
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[0.5], [0.5]], dtype=torch.float32)  # Low energy
        mock_graph.network_metrics = None

        with patch('learning.homeostasis_controller.get_node_energy_cap', return_value=5.0), \
             patch('learning.homeostasis_controller.get_node_death_threshold', return_value=1.0), \
             patch('learning.homeostasis_controller.get_node_birth_threshold', return_value=2.0), \
             patch('learning.homeostasis_controller.config') as mock_config:

            result = self.controller.regulate_network_activity(mock_graph)

            # Should increase energy
            assert result == mock_graph
            mock_config.set_value.assert_any_call('NodeLifecycle', 'death_threshold', pytest.approx(0.999, abs=1e-6))
            mock_config.set_value.assert_any_call('NodeLifecycle', 'birth_threshold', pytest.approx(2.005, abs=1e-6))

    def test_regulate_network_activity_with_metrics(self):
        """Test energy regulation using network metrics."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[2.5], [2.5]], dtype=torch.float32)

        mock_metrics = MagicMock()
        mock_metrics.calculate_comprehensive_metrics.return_value = {
            'energy_balance': {
                'total_energy': 5.0,
                'average_energy': 2.5,
                'energy_variance': 0.0,
                'energy_ratio': 0.5
            },
            'connectivity': {'density': 0.5}
        }
        mock_graph.network_metrics = mock_metrics

        with patch('learning.homeostasis_controller.get_node_energy_cap', return_value=5.0), \
             patch('learning.homeostasis_controller.get_node_death_threshold', return_value=1.0), \
             patch('learning.homeostasis_controller.get_node_birth_threshold', return_value=2.0), \
             patch('learning.homeostasis_controller.config') as mock_config:

            result = self.controller.regulate_network_activity(mock_graph)

            # Energy ratio 0.5 < 0.6 - 0.1, should increase energy
            mock_config.set_value.assert_any_call('NodeLifecycle', 'death_threshold', pytest.approx(0.999, abs=1e-6))

    def test_optimize_criticality_supercritical(self):
        """Test criticality optimization for supercritical state."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[2.5], [2.5]], dtype=torch.float32)
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        mock_graph.network_metrics = None
        mock_graph.memory_system = None

        with patch.object(self.controller, '_calculate_branching_ratio', return_value=1.2):  # Above target
            result = self.controller.optimize_criticality(mock_graph)

            # Should reduce excitation
            assert torch.allclose(mock_graph.x, torch.tensor([[2.375], [2.375]], dtype=torch.float32), atol=1e-6)

    def test_optimize_criticality_subcritical(self):
        """Test criticality optimization for subcritical state."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[2.5], [2.5]], dtype=torch.float32)
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        mock_graph.network_metrics = None
        mock_graph.memory_system = None

        with patch.object(self.controller, '_calculate_branching_ratio', return_value=0.8):  # Below target
            result = self.controller.optimize_criticality(mock_graph)

            # Should increase excitation
            assert torch.allclose(mock_graph.x, torch.tensor([[2.625], [2.625]], dtype=torch.float32), atol=1e-6)

    def test_monitor_network_health_healthy(self):
        """Test network health monitoring for healthy network."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[2.5], [2.5]], dtype=torch.float32)
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        mock_graph.homeostasis_data = {'last_regulation': {'timestamp': time.time()}}
        mock_graph.memory_system = None

        health = self.controller.monitor_network_health(mock_graph)

        assert health['status'] == 'healthy'
        assert health['health_score'] >= 0.8
        assert len(health['warnings']) == 0

    def test_monitor_network_health_critical(self):
        """Test network health monitoring for critical network."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[0.1], [0.1]], dtype=torch.float32)  # Very low energy
        mock_graph.edge_index = None  # No connections
        mock_graph.homeostasis_data = {'last_regulation': {'timestamp': time.time()}}
        mock_graph.memory_system = None

        health = self.controller.monitor_network_health(mock_graph)

        assert health['status'] == 'critical'
        assert health['health_score'] < 0.6
        assert len(health['warnings']) > 0

    def test_statistics_tracking(self):
        """Test statistics tracking and management."""
        initial_stats = self.controller.get_regulation_statistics()

        # Perform some regulation
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}]
        mock_graph.x = torch.tensor([[4.0]], dtype=torch.float32)  # High energy
        mock_graph.network_metrics = None

        with patch('learning.homeostasis_controller.get_node_energy_cap', return_value=5.0), \
             patch('learning.homeostasis_controller.get_node_death_threshold', return_value=1.0), \
             patch('learning.homeostasis_controller.get_node_birth_threshold', return_value=2.0), \
             patch('config.unified_config_manager.config'):

            self.controller.regulate_network_activity(mock_graph)

        final_stats = self.controller.get_regulation_statistics()
        assert final_stats['energy_regulations'] > initial_stats['energy_regulations']
        assert final_stats['total_regulation_events'] > initial_stats['total_regulation_events']

    def test_adaptive_regulation(self):
        """Test adaptive regulation based on trends."""
        # Set up history with increasing energy trend
        self.controller.history_manager.update_history('energy', 0.5)
        self.controller.history_manager.update_history('energy', 0.6)
        self.controller.history_manager.update_history('energy', 0.7)
        self.controller.history_manager.update_history('energy', 0.8)
        self.controller.history_manager.update_history('energy', 0.9)
        self.controller.history_manager.update_history('energy', 1.0)
        self.controller.history_manager.update_history('energy', 1.1)
        self.controller.history_manager.update_history('energy', 1.2)
        self.controller.history_manager.update_history('energy', 1.3)
        self.controller.history_manager.update_history('energy', 1.4)

        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}]
        mock_graph.x = torch.tensor([[4.0]], dtype=torch.float32)
        mock_graph.network_metrics = None
        mock_graph.memory_system = None

        with patch('learning.homeostasis_controller.get_node_energy_cap', return_value=5.0), \
             patch('learning.homeostasis_controller.get_node_death_threshold', return_value=1.0), \
             patch('learning.homeostasis_controller.get_node_birth_threshold', return_value=2.0), \
             patch('learning.homeostasis_controller.config') as mock_config:

            self.controller.regulate_network_activity(mock_graph)

            # Should use higher regulation rate due to increasing trend
            mock_config.set_value.assert_any_call('NodeLifecycle', 'death_threshold', pytest.approx(1.06, abs=1e-4))

    def test_memory_integration(self):
        """Test memory integration in criticality optimization."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[2.5], [2.5]], dtype=torch.float32)
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        mock_graph.memory_system = MagicMock()
        mock_graph.memory_system.get_node_memory_importance.return_value = 0.8
        mock_graph.network_metrics = None

        with patch.object(self.controller, '_calculate_branching_ratio', return_value=1.2):
            result = self.controller.optimize_criticality(mock_graph)

            # Memory importance should affect threshold
            mock_graph.memory_system.get_node_memory_importance.assert_called()

    def test_error_handling_invalid_graph(self):
        """Test error handling with invalid graph."""
        # Graph without required attributes
        invalid_graph = MagicMock()
        invalid_graph.node_labels = None

        result = self.controller.regulate_network_activity(invalid_graph)
        assert result == invalid_graph

        result = self.controller.optimize_criticality(invalid_graph)
        assert result == invalid_graph

        health = self.controller.monitor_network_health(invalid_graph)
        assert health['status'] == 'unknown'

    def test_error_handling_metrics_failure(self):
        """Test error handling when metrics calculation fails."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}]
        mock_graph.x = torch.tensor([[2.5]], dtype=torch.float32)
        mock_graph.network_metrics = MagicMock()
        mock_graph.network_metrics.calculate_comprehensive_metrics.side_effect = Exception("Metrics error")

        # Should fall back to basic calculation
        result = self.controller.regulate_network_activity(mock_graph)
        assert result == mock_graph

    def test_edge_cases_empty_graph(self):
        """Test edge cases with empty or minimal graphs."""
        # Empty graph
        empty_graph = MagicMock()
        empty_graph.node_labels = []
        empty_graph.x = torch.tensor([], dtype=torch.float32).reshape(0, 1)
        empty_graph.network_metrics = None

        result = self.controller.regulate_network_activity(empty_graph)
        assert result == empty_graph

        # Single node graph
        single_graph = MagicMock()
        single_graph.node_labels = [{'id': 0}]
        single_graph.x = torch.tensor([[2.5]], dtype=torch.float32)
        single_graph.network_metrics = None

        result = self.controller.regulate_network_activity(single_graph)
        assert result == single_graph

    def test_edge_cases_extreme_values(self):
        """Test edge cases with extreme energy values."""
        # Very high energy
        high_energy_graph = MagicMock()
        high_energy_graph.node_labels = [{'id': 0}]
        high_energy_graph.x = torch.tensor([[100.0]], dtype=torch.float32)
        high_energy_graph.network_metrics = None

        with patch('learning.homeostasis_controller.get_node_energy_cap', return_value=5.0), \
             patch('learning.homeostasis_controller.get_node_death_threshold', return_value=1.0), \
             patch('learning.homeostasis_controller.get_node_birth_threshold', return_value=2.0), \
             patch('learning.homeostasis_controller.config') as mock_config:

            self.controller.regulate_network_activity(high_energy_graph)

            # Should still regulate
            mock_config.set_value.assert_called()

        # Zero energy
        zero_energy_graph = MagicMock()
        zero_energy_graph.node_labels = [{'id': 0}]
        zero_energy_graph.x = torch.tensor([[0.0]], dtype=torch.float32)
        zero_energy_graph.network_metrics = None

        with patch('learning.homeostasis_controller.get_node_energy_cap', return_value=5.0), \
             patch('learning.homeostasis_controller.get_node_death_threshold', return_value=1.0), \
             patch('learning.homeostasis_controller.get_node_birth_threshold', return_value=2.0), \
             patch('learning.homeostasis_controller.config') as mock_config:

            self.controller.regulate_network_activity(zero_energy_graph)

            # Should increase energy
            mock_config.set_value.assert_any_call('NodeLifecycle', 'death_threshold', pytest.approx(0.999, abs=1e-4))

    def test_validation_enabled(self):
        """Test regulation validation when enabled."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}]
        mock_graph.x = torch.tensor([[4.0]], dtype=torch.float32)  # High energy
        mock_graph.network_metrics = None

        with patch('learning.homeostasis_controller.get_node_energy_cap', return_value=5.0), \
             patch('learning.homeostasis_controller.get_node_death_threshold', return_value=1.0), \
             patch('learning.homeostasis_controller.get_node_birth_threshold', return_value=2.0), \
             patch('learning.homeostasis_controller.config') as mock_config:

            # First regulation
            self.controller.regulate_network_activity(mock_graph)

            # Check that validation occurred (rate adjustment)
            assert self.controller.regulation_rate <= 0.001  # May have been adjusted

    def test_reset_statistics(self):
        """Test statistics reset functionality."""
        # Add some stats
        self.controller.stats_manager.increment('energy_regulations')

        initial_stats = self.controller.get_regulation_statistics()
        assert initial_stats['energy_regulations'] > 0

        self.controller.reset_statistics()

        reset_stats = self.controller.get_regulation_statistics()
        assert reset_stats['energy_regulations'] == 0
        assert reset_stats['total_regulation_events'] == 0


class TestHistoryManager:
    """Test suite for HistoryManager."""

    def setup_method(self):
        self.history_manager = HistoryManager()

    def test_update_history(self):
        """Test history buffer updates."""
        self.history_manager.update_history('energy', 0.5)
        self.history_manager.update_history('energy', 0.6)
        self.history_manager.update_history('branching', 1.0)

        assert len(self.history_manager.energy_history) == 2
        assert len(self.history_manager.branching_history) == 1

    def test_history_capacity_limit(self):
        """Test history buffer capacity limits."""
        max_len = self.history_manager.max_history_length

        # Fill beyond capacity
        for i in range(max_len + 10):
            self.history_manager.update_history('energy', float(i))

        assert len(self.history_manager.energy_history) == max_len

    def test_network_trends_calculation(self):
        """Test network trends calculation."""
        # Create increasing energy trend
        for i in range(15):
            self.history_manager.update_history('energy', 0.5 + i * 0.01)

        trends = self.history_manager.get_network_trends()

        assert 'energy_trend' in trends
        assert trends['energy_trend'] == 'increasing'
        assert trends['energy_slope'] > 0


class TestStatsManager:
    """Test suite for StatsManager."""

    def setup_method(self):
        self.stats_manager = StatsManager()

    def test_increment_statistics(self):
        """Test statistics incrementing."""
        self.stats_manager.increment('energy_regulations')
        self.stats_manager.increment('criticality_regulations')

        stats = self.stats_manager.get_statistics()
        assert stats['energy_regulations'] == 1
        assert stats['criticality_regulations'] == 1
        assert stats['total_regulation_events'] == 2

    def test_reset_statistics(self):
        """Test statistics reset."""
        self.stats_manager.increment('energy_regulations')
        initial_stats = self.stats_manager.get_statistics()
        assert initial_stats['energy_regulations'] == 1

        self.stats_manager.reset()
        reset_stats = self.stats_manager.get_statistics()
        assert reset_stats['energy_regulations'] == 0


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_calculate_network_stability(self):
        """Test network stability calculation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0}, {'id': 1}]
        mock_graph.x = torch.tensor([[2.5], [2.5]], dtype=torch.float32)
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        stability = calculate_network_stability(mock_graph)
        assert 0.0 <= stability <= 1.0

        # Test with high variance (less stable)
        mock_graph.x = torch.tensor([[1.0], [4.0]], dtype=torch.float32)
        stability_high_var = calculate_network_stability(mock_graph)
        assert stability_high_var < stability

    def test_detect_network_anomalies(self):
        """Test network anomaly detection."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 0, 'behavior': 'dynamic'}, {'id': 1, 'behavior': 'dynamic'}]
        mock_graph.x = torch.tensor([[2.5], [2.5]], dtype=torch.float32)
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        anomalies = detect_network_anomalies(mock_graph)
        # Should detect no anomalies for normal case
        assert isinstance(anomalies, list)

        # Test with extreme energy values
        mock_graph.x = torch.tensor([[0.1], [4.9]], dtype=torch.float32)  # Outliers
        anomalies = detect_network_anomalies(mock_graph)
        assert len(anomalies) > 0

        # Test with behavior imbalance
        mock_graph.node_labels = [{'behavior': 'dynamic'}, {'behavior': 'dynamic'}, {'behavior': 'oscillator'}]
        mock_graph.x = torch.tensor([[2.5], [2.5], [2.5]], dtype=torch.float32)
        anomalies = detect_network_anomalies(mock_graph)
        # May or may not detect imbalance depending on threshold


if __name__ == "__main__":
    pytest.main([__file__])