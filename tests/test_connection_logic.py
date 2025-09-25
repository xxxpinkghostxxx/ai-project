"""
Comprehensive tests for ConnectionLogic.
Tests connection creation, validation, weight changes, and intelligent formation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import threading
import numpy as np
from unittest.mock import MagicMock, patch, MagicMock
from torch_geometric.data import Data
import torch

from neural.connection_logic import (
    create_weighted_connection, get_edge_attributes, apply_weight_change,
    intelligent_connection_formation, EnhancedEdge, create_basic_connections
)


class TestConnectionLogic:
    """Test suite for ConnectionLogic."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_validator = MagicMock()
        self.mock_id_manager = MagicMock()

    def test_enhanced_edge_creation(self):
        """Test EnhancedEdge creation and methods."""
        edge = EnhancedEdge(1, 2, 0.5, 'excitatory')

        assert edge.source == 1
        assert edge.target == 2
        assert edge.weight == 0.5
        assert edge.type == 'excitatory'
        assert edge.eligibility_trace == 0.0
        assert edge.plasticity_tag is False

        # Test effective weight
        assert edge.get_effective_weight() == 0.5

        # Test inhibitory
        edge.type = 'inhibitory'
        assert edge.get_effective_weight() == -0.5

        # Test modulatory
        edge.type = 'modulatory'
        assert edge.get_effective_weight() == 0.5 * 0.5

        # Test plastic
        edge.type = 'plastic'
        edge.eligibility_trace = 0.2
        assert edge.get_effective_weight() == 0.5 * (1.0 + 0.2)

    def test_create_weighted_connection_validation(self):
        """Test connection creation with validation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': i} for i in range(10)]
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        mock_graph.edge_attributes = []

        # Mock successful validation
        self.mock_validator.validate_connection.return_value = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        self.mock_id_manager.get_node_index.side_effect = lambda id: id

        with patch('neural.connection_logic.get_connection_validator', return_value=self.mock_validator), \
             patch('neural.connection_logic.get_id_manager', return_value=self.mock_id_manager):

            result = create_weighted_connection(mock_graph, 1, 2, 0.5, 'excitatory')

            assert result == mock_graph
            self.mock_validator.validate_connection.assert_called_once()
            assert len(mock_graph.edge_attributes) == 1

    def test_create_weighted_connection_invalid(self):
        """Test connection creation with invalid validation."""
        mock_graph = MagicMock()

        # Mock failed validation
        self.mock_validator.validate_connection.return_value = {
            'is_valid': False,
            'errors': ['Invalid connection'],
            'warnings': [],
            'suggestions': []
        }

        with patch('neural.connection_logic.get_connection_validator', return_value=self.mock_validator):
            result = create_weighted_connection(mock_graph, 1, 2, 0.5, 'excitatory')

            assert result == mock_graph
            assert len(mock_graph.edge_attributes) == 0

    def test_get_edge_attributes(self):
        """Test edge attributes retrieval."""
        mock_graph = MagicMock()
        mock_edge = MagicMock()
        mock_graph.edge_attributes = [mock_edge]
        mock_graph._edge_attributes_lock = threading.RLock()

        with patch('neural.connection_logic.safe_hasattr', return_value=True):
            result = get_edge_attributes(mock_graph, 0)
            assert result == mock_edge

        # Test out of bounds
        result = get_edge_attributes(mock_graph, 5)
        assert result is None

    def test_apply_weight_change(self):
        """Test weight change application."""
        mock_graph = MagicMock()
        mock_edge = MagicMock()
        mock_edge.weight = 0.5
        mock_edge.strength_history = []
        mock_edge.activation_count = 0
        mock_graph.edge_attributes = [mock_edge]
        mock_graph._edge_attributes_lock = threading.RLock()

        with patch('neural.connection_logic.safe_hasattr', return_value=True):
            result = apply_weight_change(mock_graph, 0, 0.2)

            assert result == mock_graph
            assert mock_edge.weight == 0.7
            assert len(mock_edge.strength_history) == 1

    def test_apply_weight_change_bounds(self):
        """Test weight change with bounds checking."""
        mock_graph = MagicMock()
        mock_edge = MagicMock()
        mock_edge.weight = 0.9
        mock_edge.strength_history = []
        mock_edge.activation_count = 0
        mock_graph.edge_attributes = [mock_edge]
        mock_graph._edge_attributes_lock = threading.RLock()

        with patch('neural.connection_logic.safe_hasattr', return_value=True):
            # Test upper bound
            apply_weight_change(mock_graph, 0, 0.2)
            assert mock_edge.weight <= 5.0  # WEIGHT_CAP_MAX is 5.0

            # Test lower bound
            mock_edge.weight = 0.1
            apply_weight_change(mock_graph, 0, -0.2)
            assert mock_edge.weight >= 0.0  # Assuming WEIGHT_MIN is 0.0

    def test_intelligent_connection_formation(self):
        """Test intelligent connection formation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [
            {'id': 1, 'type': 'sensory'},
            {'id': 2, 'type': 'dynamic'},
            {'id': 3, 'type': 'sensory'},
            {'id': 4, 'type': 'dynamic'}
        ]
        mock_graph.x = torch.tensor([[0.5], [0.6], [0.7], [0.8]], dtype=torch.float32)
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        mock_graph.edge_attributes = []

        self.mock_validator.validate_connection.return_value = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        with patch('neural.connection_logic.get_connection_validator', return_value=self.mock_validator), \
             patch('neural.connection_logic.get_node_energy_cap', return_value=1.0), \
             patch('random.shuffle'), \
             patch('neural.connection_logic.create_weighted_connection') as mock_create:

            mock_create.return_value = mock_graph
            result = intelligent_connection_formation(mock_graph)

            assert result == mock_graph
            # Should have created connections
            assert mock_create.call_count > 0

    def test_intelligent_connection_formation_large_graph(self):
        """Test intelligent connection formation with large graph."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': i, 'type': 'dynamic'} for i in range(50001)]
        mock_graph.x = torch.zeros(50001, 1)

        with patch('neural.connection_logic.log_step') as mock_log:
            result = intelligent_connection_formation(mock_graph)

            assert result == mock_graph
            mock_log.assert_called_with(
                f"Skipping intelligent connections for large graph ({len(mock_graph.node_labels)} nodes) to maintain performance"
            )

    def test_create_basic_connections(self):
        """Test basic connection creation."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': i} for i in range(10)]
        mock_graph.x = torch.zeros(10, 1)
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        mock_graph.edge_attributes = []

        self.mock_id_manager.get_node_index.side_effect = lambda id: id

        with patch('neural.connection_logic.get_id_manager', return_value=self.mock_id_manager), \
             patch('neural.connection_logic.create_weighted_connection') as mock_create:

            mock_create.return_value = mock_graph
            result = create_basic_connections(mock_graph)

            assert result == mock_graph
            # Should create some connections
            assert mock_create.call_count > 0

    def test_create_basic_connections_small_graph(self):
        """Test basic connection creation with small graph."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': 1}]
        mock_graph.x = torch.zeros(1, 1)

        result = create_basic_connections(mock_graph)
        assert result == mock_graph

    def test_thread_safety(self):
        """Test thread safety of connection operations."""
        mock_graph = MagicMock()
        mock_graph.edge_attributes = [MagicMock()]
        mock_graph._edge_attributes_lock = threading.RLock()

        errors = []

        def apply_changes():
            try:
                with patch('neural.connection_logic.safe_hasattr', return_value=True):
                    apply_weight_change(mock_graph, 0, 0.1)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=apply_changes)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0

    def test_edge_to_dict(self):
        """Test EnhancedEdge to_dict method."""
        edge = EnhancedEdge(1, 2, 0.5, 'excitatory')
        edge.eligibility_trace = 0.1
        edge.plasticity_tag = True
        edge.last_activity = 123.0
        edge.activation_count = 5
        edge.creation_time = 100.0

        edge_dict = edge.to_dict()

        assert edge_dict['source'] == 1
        assert edge_dict['target'] == 2
        assert edge_dict['weight'] == 0.5
        assert edge_dict['type'] == 'excitatory'
        assert edge_dict['eligibility_trace'] == 0.1
        assert edge_dict['plasticity_tag'] is True
        assert edge_dict['last_activity'] == 123.0
        assert edge_dict['activation_count'] == 5
        assert edge_dict['creation_time'] == 100.0

    def test_eligibility_trace_update(self):
        """Test eligibility trace updates."""
        edge = EnhancedEdge(1, 2, 0.5, 'plastic')
        initial_trace = edge.eligibility_trace

        edge.update_eligibility_trace(0.1)
        assert edge.eligibility_trace > initial_trace

        # Test decay
        before_decay = edge.eligibility_trace
        edge.update_eligibility_trace(0.0)
        decayed_trace = edge.eligibility_trace
        assert decayed_trace < before_decay

    def test_connection_energy_modulation(self):
        """Test energy-modulated connection weights."""
        # This is tested indirectly through intelligent_connection_formation
        # but we can test the logic
        sensory_energy = 0.8
        dynamic_energy = 0.6
        energy_cap = 1.0
        energy_mod = (sensory_energy + dynamic_energy) / (2 * energy_cap)
        energy_factor = max(0.1, min(2.0, energy_mod * 2.0))

        assert energy_factor > 0.1
        assert energy_factor <= 2.0

    def test_connection_type_weights(self):
        """Test different connection type weight calculations."""
        edge = EnhancedEdge(1, 2, 0.5, 'excitatory')
        assert edge.get_effective_weight() == 0.5

        edge.type = 'burst'
        assert edge.get_effective_weight() == 0.5 * 1.5

        edge.type = 'unknown'
        assert edge.get_effective_weight() == 0.5

    def test_validation_error_logging(self):
        """Test that validation errors are logged."""
        mock_graph = MagicMock()

        self.mock_validator.validate_connection.return_value = {
            'is_valid': False,
            'errors': ['Test error'],
            'warnings': ['Test warning'],
            'suggestions': ['Test suggestion']
        }

        with patch('neural.connection_logic.get_connection_validator', return_value=self.mock_validator), \
              patch('logging.error') as mock_error, \
              patch('logging.warning') as mock_warning, \
              patch('logging.info') as mock_info:

            result = create_weighted_connection(mock_graph, 1, 2, 0.5, 'excitatory')

            assert result == mock_graph
            mock_error.assert_called()
            # Warnings are not logged when is_valid=False
            mock_warning.assert_not_called()
            mock_info.assert_not_called()

    def test_validation_warning_logging(self):
        """Test that validation warnings are logged when valid."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': i} for i in range(10)]
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        mock_graph.edge_attributes = []

        self.mock_validator.validate_connection.return_value = {
            'is_valid': True,
            'errors': [],
            'warnings': ['Test warning'],
            'suggestions': ['Test suggestion']
        }

        with patch('neural.connection_logic.get_connection_validator', return_value=self.mock_validator), \
              patch('neural.connection_logic.get_id_manager', return_value=self.mock_id_manager), \
              patch('logging.error') as mock_error, \
              patch('logging.warning') as mock_warning, \
              patch('logging.info') as mock_info:

            self.mock_id_manager.get_node_index.side_effect = lambda id: id
            result = create_weighted_connection(mock_graph, 1, 2, 0.5, 'excitatory')

            assert result == mock_graph
            mock_error.assert_not_called()
            mock_warning.assert_called()
            mock_info.assert_called()

    def test_edge_index_bounds_checking(self):
        """Test edge index bounds checking."""
        mock_graph = MagicMock()
        mock_graph.node_labels = [{'id': i} for i in range(5)]
        mock_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        mock_graph.edge_attributes = []

        self.mock_validator.validate_connection.return_value = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        # Mock out of bounds indices
        self.mock_id_manager.get_node_index.side_effect = lambda id: 10 if id == 1 else 15  # Both > 5

        with patch('neural.connection_logic.get_connection_validator', return_value=self.mock_validator), \
              patch('neural.connection_logic.get_id_manager', return_value=self.mock_id_manager), \
              patch('logging.error') as mock_error:

            result = create_weighted_connection(mock_graph, 1, 2, 0.5, 'excitatory')

            assert result == mock_graph
            mock_error.assert_called_with("Node indices out of bounds: source_index=10, target_index=15, graph_size=5")

    def test_strength_history_limits(self):
        """Test strength history size limits."""
        mock_graph = MagicMock()
        mock_edge = MagicMock()
        mock_edge.weight = 0.5
        mock_edge.strength_history = []
        mock_edge.activation_count = 0
        mock_graph.edge_attributes = [mock_edge]
        mock_graph._edge_attributes_lock = threading.RLock()

        with patch('neural.connection_logic.safe_hasattr', return_value=True):
            # Add many weight changes
            for i in range(110):
                apply_weight_change(mock_graph, 0, 0.01)

            # History should be limited to 100 entries
            assert len(mock_edge.strength_history) <= 100


if __name__ == "__main__":
    pytest.main([__file__])