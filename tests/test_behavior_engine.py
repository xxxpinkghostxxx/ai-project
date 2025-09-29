"""
Comprehensive tests for BehaviorEngine.
Tests node behavior updates, error handling, thread safety, and integration.
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

from src.neural.behavior_engine import (BehaviorEngine, get_energy_cap_255,
                                        get_enhanced_nodes_config_cached)


class TestBehaviorEngine:
    """Test suite for BehaviorEngine."""

    def setup_method(self):
        """Set up test environment."""
        self.engine = BehaviorEngine()

    def teardown_method(self):
        """Clean up after tests."""
        self.engine.reset_statistics()

    def test_initialization(self):
        """Test BehaviorEngine initialization."""
        assert self.engine.behavior_handlers is not None
        assert 'sensory' in self.engine.behavior_handlers
        assert 'dynamic' in self.engine.behavior_handlers
        assert 'oscillator' in self.engine.behavior_handlers
        assert 'integrator' in self.engine.behavior_handlers
        assert 'relay' in self.engine.behavior_handlers
        assert 'highway' in self.engine.behavior_handlers
        assert 'workspace' in self.engine.behavior_handlers
        assert self.engine.behavior_stats is not None

    def test_update_node_behavior_validation(self):
        """Test node behavior update with input validation."""
        # Create mock graph
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1, 'behavior': 'dynamic', 'energy': 0.5}]

        # Test with invalid node_id - now raises AssertionError
        with pytest.raises(AssertionError, match="Node ID must be non-negative"):
            self.engine.update_node_behavior(-1, mock_graph, 0)

        # Test with None graph - raises AssertionError
        with pytest.raises(AssertionError, match="Graph must not be None"):
            self.engine.update_node_behavior(1, None, 0)

        # Test with invalid step - raises AssertionError
        with pytest.raises(AssertionError, match="Step must be non-negative"):
            self.engine.update_node_behavior(1, mock_graph, -1)

    def test_sensory_node_update(self):
        """Test sensory node behavior update."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_energy.return_value = 0.8
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: default

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_sensory_node(1, mock_graph, 0)

        mock_access_layer.update_node_property.assert_any_call(1, 'membrane_potential', pytest.approx(0.8 / 255.0, abs=1e-6))
        mock_access_layer.update_node_property.assert_any_call(1, 'state', 'active')

    def test_dynamic_node_update(self):
        """Test dynamic node behavior update."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_energy.return_value = 0.6
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: default

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_dynamic_node(1, mock_graph, 0)

        mock_access_layer.update_node_property.assert_any_call(1, 'membrane_potential', pytest.approx(0.6 / 5.0, abs=1e-6))
        mock_access_layer.update_node_property.assert_any_call(1, 'state', 'active')

    def test_oscillator_node_update(self):
        """Test oscillator node behavior update."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: {
            'oscillation_freq': 0.1,
            'threshold': 0.8,
            'refractory_timer': 0.0,
            'membrane_potential': 0.9
        }.get(prop, default)

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_oscillator_node(1, mock_graph, 0)

        # Should activate due to high membrane potential
        mock_access_layer.update_node_property.assert_any_call(1, 'refractory_timer', pytest.approx(0.1, abs=1e-6))
        mock_access_layer.update_node_property.assert_any_call(1, 'membrane_potential', 0.0)
        mock_access_layer.update_node_property.assert_any_call(1, 'state', 'active')

    def test_integrator_node_update(self):
        """Test integrator node behavior update."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: {
            'integration_rate': 0.5,
            'threshold': 0.8,
            'refractory_timer': 0.0,
            'membrane_potential': 0.9
        }.get(prop, default)

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_integrator_node(1, mock_graph, 0)

        # Should activate due to high membrane potential
        mock_access_layer.update_node_property.assert_any_call(1, 'refractory_timer', pytest.approx(0.1, abs=1e-6))
        mock_access_layer.update_node_property.assert_any_call(1, 'membrane_potential', 0.0)
        mock_access_layer.update_node_property.assert_any_call(1, 'state', 'consolidating')

    def test_relay_node_update(self):
        """Test relay node behavior update."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: {
            'relay_amplification': 1.5,
            'threshold': 0.4,
            'refractory_timer': 0.0,
            'membrane_potential': 0.5
        }.get(prop, default)

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_relay_node(1, mock_graph, 0)

        # Should activate due to high membrane potential
        mock_access_layer.update_node_property.assert_any_call(1, 'refractory_timer', pytest.approx(0.05, abs=1e-6))
        mock_access_layer.update_node_property.assert_any_call(1, 'membrane_potential', 0.0)
        mock_access_layer.update_node_property.assert_any_call(1, 'state', 'active')

    def test_highway_node_update(self):
        """Test highway node behavior update."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_energy.return_value = 2.0
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: {
            'highway_energy_boost': 1.2,
            'threshold': 0.2,
            'refractory_timer': 0.0,
            'membrane_potential': 0.5
        }.get(prop, default)

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_highway_node(1, mock_graph, 0)

        # Should regulate due to energy above threshold
        mock_access_layer.set_node_energy.assert_called_with(1, pytest.approx(2.0 * 1.2, abs=1e-6))
        mock_access_layer.update_node_property.assert_any_call(1, 'refractory_timer', 0.5)
        mock_access_layer.update_node_property.assert_any_call(1, 'state', 'regulating')

    def test_workspace_node_update(self):
        """Test workspace node behavior update."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_energy.return_value = 4.0
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: {
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'refractory_timer': 0.0,
            'membrane_potential': 0.8
        }.get(prop, default)

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            # Mock random to ensure synthesis
            with patch('numpy.random.random', return_value=0.01):  # Low value for synthesis success
                self.engine.update_workspace_node(1, mock_graph, 0)

        mock_access_layer.update_node_property.assert_any_call(1, 'state', 'synthesizing')
        assert self.engine.behavior_stats['workspace_syntheses'] == 1

    def test_error_handling(self):
        """Test error handling in behavior updates."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_by_id.return_value = None

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            result = self.engine.update_node_behavior(1, mock_graph, 0)
            assert result is False

    def test_enhanced_behavior_fallback(self):
        """Test fallback to basic behavior when enhanced fails."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1, 'behavior': 'dynamic', 'enhanced_behavior': True}]
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_by_id.return_value = mock_graph.node_labels[0]
        mock_access_layer.get_node_energy.return_value = 0.5

        # Mock enhanced integration to fail
        self.engine.enhanced_integration = MagicMock()
        self.engine.enhanced_integration.node_behavior_system = None

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            result = self.engine.update_node_behavior(1, mock_graph, 0)
            assert result is True  # Should fall back to basic behavior

    def test_thread_safety(self):
        """Test thread safety of behavior updates."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1, 'behavior': 'dynamic', 'energy': 0.5}]
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_by_id.return_value = mock_graph.node_labels[0]
        mock_access_layer.get_node_energy.return_value = 0.5

        errors = []

        def update_behavior():
            try:
                with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
                    self.engine.update_node_behavior(1, mock_graph, 0)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=update_behavior)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0

    def test_statistics_tracking(self):
        """Test behavior statistics tracking."""
        initial_stats = self.engine.get_behavior_statistics()

        # Perform some updates
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1, 'behavior': 'dynamic', 'energy': 0.5}]
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_by_id.return_value = mock_graph.node_labels[0]
        mock_access_layer.get_node_energy.return_value = 0.5

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_node_behavior(1, mock_graph, 0)

        final_stats = self.engine.get_behavior_statistics()
        assert final_stats['basic_updates'] >= initial_stats['basic_updates']

    def test_refractory_period_handling(self):
        """Test refractory period handling across behaviors."""
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: 1.0 if prop == 'refractory_timer' else default

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            # Should not update if in refractory
            self.engine.update_sensory_node(1, mock_graph, 0)
            # Should update refractory timer
            mock_access_layer.update_node_property.assert_called_with(1, 'refractory_timer', pytest.approx(1.0 - 0.01, abs=1e-6))

    def test_energy_thresholds(self):
        """Test energy threshold handling."""
        # Test helper functions
        from src.neural.behavior_engine import (energy_above_threshold,
                                                should_transition_to_learning)

        node_learning = {
            'last_activation': time.time() - 4.0,  # Recent
            'plasticity_enabled': True,
            'eligibility_trace': 0.2
        }
        assert should_transition_to_learning(node_learning)

        node_threshold = {'energy': 0.6, 'threshold': 0.5}
        assert energy_above_threshold(node_threshold)

        node_below = {'energy': 0.3, 'threshold': 0.5}
        assert not energy_above_threshold(node_below)

    def test_connection_activity_check(self):
        """Test has_active_connections function."""
        from src.neural.behavior_engine import has_active_connections

        # Mock graph with connections
        mock_graph = MagicMock()
        mock_graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        mock_graph.node_labels = [{'id': 1}, {'id': 2}]

        node_active = {'id': 1, 'energy': 0.6}
        assert has_active_connections(node_active, mock_graph)

        node_inactive = {'id': 1, 'energy': 0.2}
        assert not has_active_connections(node_inactive, mock_graph)

    def test_configuration_caching(self):
        """Test configuration caching functions."""
        # Test energy cap caching
        cap1 = get_energy_cap_255()
        cap2 = get_energy_cap_255()
        assert cap1 == cap2 == 255.0

        # Test enhanced nodes config caching
        config1 = get_enhanced_nodes_config_cached()
        config2 = get_enhanced_nodes_config_cached()
        assert config1 is config2  # Should be same cached object

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with extreme energy values
        mock_graph = MagicMock()
        mock_access_layer = MagicMock()
        mock_access_layer.get_node_energy.return_value = 1000.0  # Very high energy
        mock_access_layer.get_node_property.side_effect = lambda node_id, prop, default: default

        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_sensory_node(1, mock_graph, 0)

        # Membrane potential should be clamped to 1.0
        mock_access_layer.update_node_property.assert_any_call(1, 'membrane_potential', 1.0)

        # Test with zero energy
        mock_access_layer.get_node_energy.return_value = 0.0
        with patch('energy.node_access_layer.NodeAccessLayer', return_value=mock_access_layer):
            self.engine.update_sensory_node(1, mock_graph, 0)

        mock_access_layer.update_node_property.assert_any_call(1, 'membrane_potential', 0.0)


if __name__ == "__main__":
    pytest.main([__file__])






