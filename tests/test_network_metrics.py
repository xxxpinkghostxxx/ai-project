"""
Comprehensive tests for NetworkMetrics.
Tests criticality calculation, connectivity analysis, energy balance, and health scoring.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data
import torch
from collections import deque

from neural.network_metrics import NetworkMetrics, create_network_metrics, quick_network_analysis


class TestNetworkMetrics:
    """Test suite for NetworkMetrics."""

    def setup_method(self):
        """Set up test environment."""
        self.metrics = NetworkMetrics()

    def teardown_method(self):
        """Clean up after tests."""
        pass

    def test_initialization(self):
        """Test NetworkMetrics initialization."""
        assert self.metrics.calculation_interval == 50
        assert self.metrics.criticality_target == 1.0
        assert self.metrics.connectivity_target == 0.3
        assert isinstance(self.metrics.metrics_history, deque)
        assert isinstance(self.metrics.activation_patterns, dict)

    def test_calculate_criticality(self):
        """Test criticality calculation."""
        # Create mock graph with activations
        mock_graph = MagicMock()
        mock_graph.node_labels = [
            {'last_activation': 100.0, 'energy': 0.8},
            {'last_activation': 0, 'energy': 0.6},
            {'last_activation': 105.0, 'energy': 0.7}
        ]

        with patch('numpy.random.choice', return_value=[0, 1, 2]):
            criticality = self.metrics.calculate_criticality(mock_graph)

            assert isinstance(criticality, float)
            assert 0.0 <= criticality <= 1.0

    def test_calculate_criticality_no_activations(self):
        """Test criticality with no activations."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'last_activation': 0, 'energy': 0.5}]

        criticality = self.metrics.calculate_criticality(mock_graph)
        assert criticality == 0.0

    def test_analyze_connectivity(self):
        """Test connectivity analysis."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1}, {'id': 2}, {'id': 3}]
        mock_graph.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        connectivity = self.metrics.analyze_connectivity(mock_graph)

        assert 'num_nodes' in connectivity
        assert 'num_edges' in connectivity
        assert 'avg_degree' in connectivity
        assert 'density' in connectivity
        assert connectivity['num_nodes'] == 3
        assert connectivity['num_edges'] == 3
        assert connectivity['avg_degree'] == 2.0  # 2 * 3 / 3

    def test_analyze_connectivity_no_edges(self):
        """Test connectivity analysis with no edges."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1}, {'id': 2}]
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

        connectivity = self.metrics.analyze_connectivity(mock_graph)

        assert connectivity['num_nodes'] == 2
        assert connectivity['num_edges'] == 0
        assert connectivity['avg_degree'] == 0.0
        assert connectivity['density'] == 0.0

    def test_measure_energy_balance(self):
        """Test energy balance measurement."""
        mock_graph = MagicMock()
        mock_graph.x = torch.tensor([[0.5], [0.7], [0.3]], dtype=torch.float32)

        energy_balance = self.metrics.measure_energy_balance(mock_graph)

        assert 'total_energy' in energy_balance
        assert 'energy_variance' in energy_balance
        assert 'energy_entropy' in energy_balance
        assert energy_balance['total_energy'] == 1.5
        assert energy_balance['mean_energy'] == 0.5

    def test_measure_energy_balance_no_features(self):
        """Test energy balance with no node features."""
        mock_graph = MagicMock()
        mock_graph.x = None

        energy_balance = self.metrics.measure_energy_balance(mock_graph)

        assert energy_balance['total_energy'] == 0.0
        assert energy_balance['energy_variance'] == 0.0

    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'last_activation': 100.0, 'energy': 0.8}]
        mock_graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        mock_graph.x = torch.tensor([[0.5]], dtype=torch.float32)

        with patch('time.time', return_value=123456.0), \
             patch('numpy.random.choice', return_value=[0]):

            comprehensive = self.metrics.calculate_comprehensive_metrics(mock_graph)

            assert 'timestamp' in comprehensive
            assert 'criticality' in comprehensive
            assert 'connectivity' in comprehensive
            assert 'energy_balance' in comprehensive
            assert 'performance' in comprehensive

            # Should be added to history
            assert len(self.metrics.metrics_history) == 1

    def test_get_metrics_trends(self):
        """Test metrics trends calculation."""
        # Need at least window_size metrics
        for i in range(15):
            self.metrics.metrics_history.append({
                'criticality': 0.5 + i * 0.01,
                'connectivity': {'density': 0.3 + i * 0.01},
                'energy_balance': {'energy_variance': 0.1 + i * 0.01}
            })

        trends = self.metrics.get_metrics_trends(10)

        assert 'criticality_trend' in trends
        assert 'density_trend' in trends
        assert 'energy_variance_trend' in trends
        assert len(trends['criticality_trend']) == 10

    def test_get_metrics_trends_insufficient_data(self):
        """Test metrics trends with insufficient data."""
        trends = self.metrics.get_metrics_trends(10)
        assert trends == {}

    def test_get_network_health_score(self):
        """Test network health score calculation."""
        # Set up last metrics
        self.metrics.last_metrics = {
            'criticality': 0.9,
            'connectivity': {'density': 0.4},
            'energy_balance': {'energy_variance': 50.0},
            'performance': {'calculation_time': 0.005}
        }

        health = self.metrics.get_network_health_score()

        assert 'score' in health
        assert 'status' in health
        assert 'recommendations' in health
        assert isinstance(health['score'], float)
        assert health['score'] > 0

    def test_get_network_health_score_no_metrics(self):
        """Test health score with no metrics."""
        health = self.metrics.get_network_health_score()

        assert health['score'] == 0.0
        assert health['status'] == 'unknown'

    def test_clustering_coefficient_calculation(self):
        """Test clustering coefficient calculation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1}, {'id': 2}, {'id': 3}]
        mock_graph.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        clustering = self.metrics._calculate_clustering_coefficient(mock_graph)
        assert isinstance(clustering, float)
        assert 0.0 <= clustering <= 1.0

    def test_average_path_length_calculation(self):
        """Test average path length calculation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1}, {'id': 2}]
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

        path_length = self.metrics._calculate_average_path_length(mock_graph)
        assert path_length > 2.0  # Should be high for disconnected graph

    def test_degree_distribution_analysis(self):
        """Test degree distribution analysis."""
        mock_graph = MagicMock()
        mock_graph.edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)

        degree_dist = self.metrics._analyze_degree_distribution(mock_graph)

        assert 'variance' in degree_dist
        assert 'entropy' in degree_dist
        assert 'max_degree' in degree_dist
        assert 'min_degree' in degree_dist

    def test_energy_entropy_calculation(self):
        """Test energy entropy calculation."""
        energy_values = np.array([0.2, 0.3, 0.5])
        entropy = self.metrics._calculate_energy_entropy(energy_values)

        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_energy_entropy_zero_energy(self):
        """Test energy entropy with zero total energy."""
        energy_values = np.array([0.0, 0.0, 0.0])
        entropy = self.metrics._calculate_energy_entropy(energy_values)

        assert entropy == 0.0

    def test_skewness_calculation(self):
        """Test skewness calculation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        skewness = self.metrics._calculate_skewness(values)

        assert isinstance(skewness, float)

    def test_skewness_insufficient_data(self):
        """Test skewness with insufficient data."""
        values = np.array([1.0, 2.0])
        skewness = self.metrics._calculate_skewness(values)

        assert skewness == 0.0

    def test_kurtosis_calculation(self):
        """Test kurtosis calculation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kurtosis = self.metrics._calculate_kurtosis(values)

        assert isinstance(kurtosis, float)

    def test_kurtosis_insufficient_data(self):
        """Test kurtosis with insufficient data."""
        values = np.array([1.0, 2.0, 3.0])
        kurtosis = self.metrics._calculate_kurtosis(values)

        assert kurtosis == 0.0

    def test_energy_conservation_check(self):
        """Test energy conservation checking."""
        # First call should set initial energy
        conservation = self.metrics._check_energy_conservation(None, 100.0)
        assert 'conservation_violation' in conservation
        assert conservation['energy_drift'] == 0.0

        # Second call should check drift
        conservation = self.metrics._check_energy_conservation(None, 95.0)
        assert conservation['energy_drift'] == -5.0
        assert conservation['drift_percentage'] == -5.0

    def test_efficiency_calculation(self):
        """Test efficiency calculation."""
        # No times recorded
        efficiency = self.metrics._calculate_efficiency()
        assert efficiency == 1.0

        # Add some calculation times
        self.metrics.calculation_times.extend([0.001, 0.002, 0.001])
        efficiency = self.metrics._calculate_efficiency()
        assert 0.0 <= efficiency <= 1.0

    def test_activation_pattern_creation(self):
        """Test activation pattern creation."""
        node = {'type': 'sensory', 'energy': 0.7}
        mock_graph = MagicMock()
        mock_graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        pattern = self.metrics._create_activation_pattern(node, mock_graph, 0)
        assert isinstance(pattern, str)
        assert 'sensory' in pattern

    def test_activation_pattern_tracking(self):
        """Test activation pattern tracking."""
        pattern = "test_pattern"

        # First time should be new
        is_new = self.metrics._is_new_activation_pattern(pattern)
        assert is_new
        assert pattern in self.metrics.activation_patterns

        # Second time should not be new
        is_new = self.metrics._is_new_activation_pattern(pattern)
        assert not is_new

    def test_ei_ratio_calculation(self):
        """Test E/I ratio calculation."""
        mock_graph = MagicMock()
        mock_edge1 = MagicMock()
        mock_edge1.type = 'excitatory'
        mock_edge1.weight = 2.0
        mock_edge2 = MagicMock()
        mock_edge2.type = 'inhibitory'
        mock_edge2.weight = 1.0
        mock_graph.edge_attributes = [mock_edge1, mock_edge2]

        ratio = self.metrics._calculate_ei_ratio(mock_graph)
        assert ratio == 2.0

    def test_ei_ratio_division_by_zero(self):
        """Test E/I ratio with zero inhibitory weights."""
        mock_graph = MagicMock()
        mock_edge = MagicMock()
        mock_edge.type = 'excitatory'
        mock_edge.weight = 1.0
        mock_graph.edge_attributes = [mock_edge]

        ratio = self.metrics._calculate_ei_ratio(mock_graph)
        assert ratio == 10.0  # Should return high ratio when no inhibitory

    def test_ei_balance_adjustment(self):
        """Test E/I balance adjustment."""
        mock_graph = MagicMock()
        mock_edge_e = MagicMock()
        mock_edge_e.type = 'excitatory'
        mock_edge_e.weight = 1.0
        mock_edge_i = MagicMock()
        mock_edge_i.type = 'inhibitory'
        mock_edge_i.weight = 1.0
        mock_graph.edge_attributes = [mock_edge_e, mock_edge_i]

        self.metrics._adjust_ei_balance(mock_graph, 2.0)  # Current ratio 1.0, target 0.8

        # Excitatory should be decreased, inhibitory increased
        assert mock_edge_e.weight < 1.0
        assert mock_edge_i.weight > 1.0

    def test_criticality_adjustment(self):
        """Test criticality adjustment."""
        mock_graph = MagicMock()
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

        # Test increasing criticality
        self.metrics._adjust_criticality(mock_graph, 0.5)  # Below target of 1.0

        # Should have added edges
        assert mock_graph.edge_index.shape[1] > 0

    def test_create_network_metrics_custom_interval(self):
        """Test creating NetworkMetrics with custom interval."""
        metrics = create_network_metrics(100)
        assert metrics.calculation_interval == 100

    def test_quick_network_analysis(self):
        """Test quick network analysis function."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'last_activation': 0, 'energy': 0.5}]
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        mock_graph.x = torch.tensor([[0.5]], dtype=torch.float32)

        with patch('time.time', return_value=123456.0), \
             patch('numpy.random.choice', return_value=[0]):

            analysis = quick_network_analysis(mock_graph)

            assert 'criticality' in analysis
            assert 'connectivity' in analysis
            assert 'energy_balance' in analysis

    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        initial_updates = self.metrics.metric_updates

        mock_graph = MagicMock()
        mock_graph.node_labels = [{'last_activation': 0}]
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        mock_graph.x = torch.tensor([[0.5]], dtype=torch.float32)

        with patch('time.time', side_effect=[100.0, 100.005, 100.01]), \
             patch('numpy.random.choice', return_value=[0]):

            self.metrics.calculate_comprehensive_metrics(mock_graph)

            assert self.metrics.metric_updates == initial_updates + 1
            assert len(self.metrics.calculation_times) == 1

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty graph
        mock_graph = MagicMock()
        mock_graph.node_labels = []
        mock_graph.edge_index = None

        criticality = self.metrics.calculate_criticality(mock_graph)
        assert criticality == 0.0

        connectivity = self.metrics.analyze_connectivity(mock_graph)
        assert connectivity['num_nodes'] == 0

        # Graph with None attributes
        mock_graph.node_labels = None
        criticality = self.metrics.calculate_criticality(mock_graph)
        assert criticality == 0.0

    def test_memory_efficiency(self):
        """Test memory efficiency of data structures."""
        # Add many metrics to history
        for i in range(1100):
            self.metrics.metrics_history.append({'test': i})

        # Should only keep last 1000
        assert len(self.metrics.metrics_history) <= 1000

        # Add many calculation times
        for i in range(1100):
            self.metrics.calculation_times.append(0.001)

        # Should only keep last 100
        assert len(self.metrics.calculation_times) <= 100


if __name__ == "__main__":
    pytest.main([__file__])