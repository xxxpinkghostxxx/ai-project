"""
Comprehensive tests for WorkspaceEngine.
Tests workspace node updates, concept synthesis, and statistics tracking.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data
import torch

from neural.workspace_engine import WorkspaceEngine, create_workspace_engine


class TestWorkspaceEngine:
    """Test suite for WorkspaceEngine."""

    def setup_method(self):
        """Set up test environment."""
        self.engine = WorkspaceEngine()

    def teardown_method(self):
        """Clean up after tests."""
        self.engine.reset_statistics()

    def test_initialization(self):
        """Test WorkspaceEngine initialization."""
        assert self.engine.workspace_capacity == 5.0
        assert self.engine.workspace_creativity == 1.5
        assert self.engine.workspace_focus == 3.0
        assert isinstance(self.engine.workspace_stats, dict)

    def test_update_workspace_nodes(self):
        """Test workspace node updates."""
        # Create graph with workspace nodes
        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'workspace', 'energy': 0.8, 'workspace_capacity': 5.0},
            {'id': 2, 'type': 'dynamic', 'energy': 0.5}
        ]

        result = self.engine.update_workspace_nodes(graph, 1)

        assert result['status'] == 'success'
        assert result['workspace_nodes_updated'] == 1
        assert result['step'] == 1

    def test_update_workspace_nodes_no_workspace_nodes(self):
        """Test update with no workspace nodes."""
        graph = Data()
        graph.node_labels = [{'id': 1, 'type': 'dynamic', 'energy': 0.5}]

        result = self.engine.update_workspace_nodes(graph, 1)

        assert result['status'] == 'no_workspace_nodes'

    def test_update_workspace_nodes_no_nodes(self):
        """Test update with no nodes."""
        graph = Data()
        graph.node_labels = []

        result = self.engine.update_workspace_nodes(graph, 1)

        assert result['status'] == 'no_nodes'

    def test_workspace_node_synthesis(self):
        """Test workspace node concept synthesis."""
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }
        graph.node_labels = [node]

        # Mock random to ensure synthesis
        with patch('numpy.random.random', return_value=0.01):  # Low value for synthesis
            self.engine._update_workspace_node(graph, 0, 1)

        assert node['state'] == 'synthesizing'
        assert self.engine.workspace_stats['syntheses_performed'] == 1
        assert self.engine.workspace_stats['concepts_created'] == 1

    def test_workspace_node_planning(self):
        """Test workspace node planning state."""
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.85,  # Above threshold but not high enough for synthesis
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }
        graph.node_labels = [node]

        self.engine._update_workspace_node(graph, 0, 1)

        assert node['state'] == 'planning'

    def test_workspace_node_imagining(self):
        """Test workspace node imagining state."""
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.65,  # Above 0.5 * threshold
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }
        graph.node_labels = [node]

        self.engine._update_workspace_node(graph, 0, 1)

        assert node['state'] == 'imagining'

    def test_workspace_node_active_state(self):
        """Test workspace node active state."""
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.3,  # Below 0.5 * threshold
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'planning'
        }
        graph.node_labels = [node]

        self.engine._update_workspace_node(graph, 0, 1)

        assert node['state'] == 'active'

    def test_workspace_capacity_constraint(self):
        """Test workspace capacity constraints on synthesis."""
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 1.0,  # Low capacity
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }
        graph.node_labels = [node]

        with patch('numpy.random.random', return_value=0.01):
            self.engine._update_workspace_node(graph, 0, 1)

        # Should not synthesize due to low capacity
        assert node['state'] != 'synthesizing'

    def test_synthesis_probability(self):
        """Test synthesis probability calculation."""
        # Test the probability formula: workspace_creativity * workspace_focus * 0.1
        creativity = 2.0
        focus = 4.0
        expected_prob = creativity * focus * 0.1  # 0.8

        # Create node with these values
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 5.0,
            'workspace_creativity': creativity,
            'workspace_focus': focus,
            'threshold': 0.6,
            'state': 'active'
        }
        graph.node_labels = [node]

        # Mock random to be just below threshold
        with patch('numpy.random.random', return_value=expected_prob - 0.01):
            self.engine._update_workspace_node(graph, 0, 1)
            assert node['state'] == 'synthesizing'

        # Reset and test just above threshold
        node['state'] = 'active'
        with patch('numpy.random.random', return_value=expected_prob + 0.01):
            self.engine._update_workspace_node(graph, 0, 1)
            assert node['state'] != 'synthesizing'

    def test_create_workspace_node(self):
        """Test workspace node creation."""
        node = self.engine.create_workspace_node(1, 0)

        assert node['id'] == 1
        assert node['type'] == 'workspace'
        assert node['behavior'] == 'workspace'
        assert node['state'] == 'active'
        assert node['energy'] == 0.0
        assert node['threshold'] == 0.6
        assert node['workspace_capacity'] == 5.0
        assert node['workspace_creativity'] == 1.5
        assert node['workspace_focus'] == 3.0
        assert node['last_update'] == 0
        assert node['membrane_potential'] == 0.0
        assert node['refractory_timer'] == 0.0
        assert node['plasticity_enabled'] is True
        assert node['eligibility_trace'] == 0.0

    def test_workspace_statistics_tracking(self):
        """Test workspace statistics tracking."""
        initial_stats = self.engine.get_workspace_metrics()

        # Perform synthesis
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }
        graph.node_labels = [node]

        with patch('numpy.random.random', return_value=0.01):
            self.engine._update_workspace_node(graph, 0, 1)

        final_stats = self.engine.get_workspace_metrics()
        assert final_stats['syntheses_performed'] == initial_stats['syntheses_performed'] + 1
        assert final_stats['concepts_created'] == initial_stats['concepts_created'] + 1

    def test_workspace_utilization_calculation(self):
        """Test workspace utilization calculation."""
        graph = Data()
        graph.node_labels = [
            {
                'id': 1,
                'type': 'workspace',
                'workspace_capacity': 10.0,
                'state': 'synthesizing'
            },
            {
                'id': 2,
                'type': 'workspace',
                'workspace_capacity': 10.0,
                'state': 'planning'
            },
            {
                'id': 3,
                'type': 'workspace',
                'workspace_capacity': 10.0,
                'state': 'active'  # Not using capacity
            }
        ]

        self.engine._update_workspace_statistics(graph)

        # Two nodes using capacity (20.0 total), total capacity 30.0
        expected_utilization = 20.0 / 30.0
        assert abs(self.engine.workspace_stats['workspace_utilization'] - expected_utilization) < 1e-6

    def test_statistics_reset(self):
        """Test statistics reset."""
        # Modify stats
        self.engine.workspace_stats['syntheses_performed'] = 5
        self.engine.workspace_stats['concepts_created'] = 3
        self.engine.workspace_stats['workspace_utilization'] = 0.8

        self.engine.reset_statistics()

        assert self.engine.workspace_stats['syntheses_performed'] == 0
        assert self.engine.workspace_stats['concepts_created'] == 0
        assert self.engine.workspace_stats['workspace_utilization'] == 0.0

    def test_cleanup(self):
        """Test cleanup functionality."""
        self.engine.workspace_stats['test'] = 'value'

        self.engine.cleanup()

        assert len(self.engine.workspace_stats) == 0

    def test_error_handling_update_nodes(self):
        """Test error handling in update_workspace_nodes."""
        graph = Data()
        graph.node_labels = None  # This should cause an error

        result = self.engine.update_workspace_nodes(graph, 1)

        assert result['status'] == 'error'
        assert 'error' in result

    def test_error_handling_update_node(self):
        """Test error handling in _update_workspace_node."""
        graph = Data()
        graph.node_labels = [{'id': 1}]  # Missing required fields

        # Should not crash
        self.engine._update_workspace_node(graph, 0, 1)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty graph
        graph = Data()
        graph.node_labels = []

        result = self.engine.update_workspace_nodes(graph, 1)
        assert result['status'] == 'no_nodes'

        # Graph with None node_labels
        graph.node_labels = None
        result = self.engine.update_workspace_nodes(graph, 1)
        assert result['status'] == 'error'

        # Node with extreme values
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 1000.0,  # Extreme energy
            'workspace_capacity': 0.0,  # Zero capacity
            'workspace_creativity': 0.0,  # Zero creativity
            'workspace_focus': 0.0,  # Zero focus
            'threshold': 0.6,
            'state': 'active'
        }
        graph.node_labels = [node]

        # Should handle gracefully
        self.engine._update_workspace_node(graph, 0, 1)

    def test_random_synthesis_variability(self):
        """Test that synthesis depends on random chance."""
        graph = Data()
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }
        graph.node_labels = [node]

        synthesis_count = 0
        trials = 100

        for _ in range(trials):
            node_copy = node.copy()
            graph.node_labels = [node_copy]

            # Random synthesis
            self.engine._update_workspace_node(graph, 0, 1)
            if node_copy['state'] == 'synthesizing':
                synthesis_count += 1

        # Should have some synthesis events (probability ~0.045)
        assert synthesis_count > 0
        assert synthesis_count < trials  # Not all trials

    def test_create_workspace_engine_factory(self):
        """Test factory function."""
        engine = create_workspace_engine()
        assert isinstance(engine, WorkspaceEngine)

    def test_workspace_node_properties_preservation(self):
        """Test that workspace node properties are preserved during updates."""
        graph = Data()
        custom_capacity = 8.0
        custom_creativity = 2.5
        node = {
            'id': 1,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': custom_capacity,
            'workspace_creativity': custom_creativity,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active',
            'custom_property': 'preserved'
        }
        graph.node_labels = [node]

        with patch('numpy.random.random', return_value=0.01):
            self.engine._update_workspace_node(graph, 0, 1)

        # Custom properties should be preserved
        assert node['workspace_capacity'] == custom_capacity
        assert node['workspace_creativity'] == custom_creativity
        assert node['custom_property'] == 'preserved'

    def test_multiple_workspace_nodes(self):
        """Test handling multiple workspace nodes."""
        graph = Data()
        nodes = []
        for i in range(5):
            node = {
                'id': i,
                'type': 'workspace',
                'energy': 0.8,
                'workspace_capacity': 5.0,
                'workspace_creativity': 1.5,
                'workspace_focus': 3.0,
                'threshold': 0.6,
                'state': 'active'
            }
            nodes.append(node)
        graph.node_labels = nodes

        with patch('numpy.random.random', return_value=0.01):
            result = self.engine.update_workspace_nodes(graph, 1)

        assert result['status'] == 'success'
        assert result['workspace_nodes_updated'] == 5

        # At least some nodes should have changed state
        changed_nodes = sum(1 for node in nodes if node['state'] == 'synthesizing')
        assert changed_nodes > 0

    def test_workspace_statistics_aggregation(self):
        """Test workspace statistics aggregation across multiple nodes."""
        graph = Data()
        nodes = [
            {
                'id': 1,
                'type': 'workspace',
                'workspace_capacity': 10.0,
                'state': 'synthesizing'
            },
            {
                'id': 2,
                'type': 'workspace',
                'workspace_capacity': 10.0,
                'state': 'planning'
            },
            {
                'id': 3,
                'type': 'workspace',
                'workspace_capacity': 10.0,
                'state': 'active'
            }
        ]
        graph.node_labels = nodes

        self.engine._update_workspace_statistics(graph)

        # Utilization: (10 + 10) / (10 + 10 + 10) = 20/30 = 2/3
        expected = 20.0 / 30.0
        assert abs(self.engine.workspace_stats['workspace_utilization'] - expected) < 1e-6

    def test_thread_safety(self):
        """Test thread safety of workspace operations."""
        import threading

        graph = Data()
        graph.node_labels = [{
            'id': 1,
            'type': 'workspace',
            'energy': 0.8,
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0,
            'threshold': 0.6,
            'state': 'active'
        }]

        errors = []

        def update_workspace():
            try:
                self.engine.update_workspace_nodes(graph, 1)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=update_workspace)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])