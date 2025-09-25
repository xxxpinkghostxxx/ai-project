"""
Comprehensive tests for NeuralMapPersistence.
Tests saving, loading, and managing neural maps with metadata.
"""
import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data
import torch

from neural.neural_map_persistence import NeuralMapPersistence, create_neural_map_persistence


class TestNeuralMapPersistence:
    """Test suite for NeuralMapPersistence."""

    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = NeuralMapPersistence(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test NeuralMapPersistence initialization."""
        assert self.persistence.save_directory == self.temp_dir
        assert self.persistence.max_slots == 10
        assert self.persistence.current_slot == 0
        assert isinstance(self.persistence.slot_metadata, dict)

    def test_save_neural_map(self):
        """Test saving a neural map."""
        # Create test graph
        graph = Data()
        graph.node_labels = [{'id': 1, 'type': 'sensory'}, {'id': 2, 'type': 'dynamic'}]
        graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph.x = torch.tensor([[0.5], [0.6]], dtype=torch.float32)

        metadata = {'test_key': 'test_value'}

        result = self.persistence.save_neural_map(graph, 0, metadata)

        assert result is True
        assert 0 in self.persistence.slot_metadata

        # Check file exists
        filename = f"neural_map_slot_0.json"
        filepath = os.path.join(self.temp_dir, filename)
        assert os.path.exists(filepath)

        # Check metadata
        saved_metadata = self.persistence.slot_metadata[0]
        assert saved_metadata['test_key'] == 'test_value'
        assert saved_metadata['node_count'] == 2
        assert saved_metadata['edge_count'] == 2

    def test_save_neural_map_invalid_slot(self):
        """Test saving with invalid slot number."""
        graph = Data()
        graph.node_labels = [{'id': 1}]

        result = self.persistence.save_neural_map(graph, -1)
        assert result is False

        result = self.persistence.save_neural_map(graph, 10)
        assert result is False

    def test_load_neural_map(self):
        """Test loading a neural map."""
        # First save a map
        graph = Data()
        graph.node_labels = [{'id': 1, 'type': 'sensory'}]
        graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graph.x = torch.tensor([[0.5]], dtype=torch.float32)

        self.persistence.save_neural_map(graph, 0)

        # Now load it
        loaded_graph = self.persistence.load_neural_map(0)

        assert loaded_graph is not None
        assert len(loaded_graph.node_labels) == 1
        assert loaded_graph.node_labels[0]['id'] == 1
        assert loaded_graph.node_labels[0]['type'] == 'sensory'
        assert torch.equal(loaded_graph.edge_index, torch.tensor([[0], [0]], dtype=torch.long))
        assert torch.equal(loaded_graph.x, torch.tensor([[0.5]], dtype=torch.float32))

    def test_load_neural_map_invalid_slot(self):
        """Test loading with invalid slot number."""
        result = self.persistence.load_neural_map(-1)
        assert result is None

        result = self.persistence.load_neural_map(10)
        assert result is None

    def test_load_neural_map_nonexistent(self):
        """Test loading a map that doesn't exist."""
        result = self.persistence.load_neural_map(5)
        assert result is None

    def test_delete_neural_map(self):
        """Test deleting a neural map."""
        # Save a map first
        graph = Data()
        graph.node_labels = [{'id': 1}]
        self.persistence.save_neural_map(graph, 0)

        # Verify it exists
        assert 0 in self.persistence.slot_metadata
        filepath = os.path.join(self.temp_dir, "neural_map_slot_0.json")
        assert os.path.exists(filepath)

        # Delete it
        result = self.persistence.delete_neural_map(0)
        assert result is True
        assert 0 not in self.persistence.slot_metadata
        assert not os.path.exists(filepath)

    def test_delete_neural_map_invalid_slot(self):
        """Test deleting with invalid slot number."""
        result = self.persistence.delete_neural_map(-1)
        assert result is False

        result = self.persistence.delete_neural_map(10)
        assert result is False

    def test_delete_neural_map_nonexistent(self):
        """Test deleting a map that doesn't exist."""
        result = self.persistence.delete_neural_map(5)
        assert result is False

    def test_list_available_slots(self):
        """Test listing available slots."""
        # Save some maps
        graph = Data()
        graph.node_labels = [{'id': 1}]

        self.persistence.save_neural_map(graph, 0)
        self.persistence.save_neural_map(graph, 2)
        self.persistence.save_neural_map(graph, 5)

        slots = self.persistence.list_available_slots()
        assert 0 in slots
        assert 2 in slots
        assert 5 in slots
        assert len(slots) == 3

    def test_get_slot_info(self):
        """Test getting slot information."""
        # Save a map with metadata
        graph = Data()
        graph.node_labels = [{'id': 1}]
        metadata = {'custom_info': 'test'}

        self.persistence.save_neural_map(graph, 1, metadata)

        info = self.persistence.get_slot_info(1)
        assert info is not None
        assert info['custom_info'] == 'test'
        assert info['node_count'] == 1

        # Test nonexistent slot
        info = self.persistence.get_slot_info(99)
        assert info is None

    def test_set_current_slot(self):
        """Test setting current slot."""
        result = self.persistence.set_current_slot(3)
        assert result is True
        assert self.persistence.current_slot == 3

        result = self.persistence.set_current_slot(-1)
        assert result is False
        assert self.persistence.current_slot == 3  # Unchanged

        result = self.persistence.set_current_slot(10)
        assert result is False

    def test_get_current_slot(self):
        """Test getting current slot."""
        assert self.persistence.get_current_slot() == 0

        self.persistence.set_current_slot(7)
        assert self.persistence.get_current_slot() == 7

    def test_serialization_complex_graph(self):
        """Test serialization of complex graph."""
        graph = Data()
        graph.node_labels = [
            {'id': 1, 'type': 'sensory', 'energy': 0.5},
            {'id': 2, 'type': 'dynamic', 'energy': 0.7}
        ]
        graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[0.8], [0.6]], dtype=torch.float32)
        graph.x = torch.tensor([[0.5, 0.1], [0.7, 0.2]], dtype=torch.float32)
        graph.y = torch.tensor([0, 1], dtype=torch.long)
        graph.pos = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        result = self.persistence.save_neural_map(graph, 0)
        assert result is True

        loaded_graph = self.persistence.load_neural_map(0)
        assert loaded_graph is not None
        assert len(loaded_graph.node_labels) == 2
        assert torch.equal(loaded_graph.edge_index, graph.edge_index)
        assert torch.equal(loaded_graph.edge_attr, graph.edge_attr)
        assert torch.equal(loaded_graph.x, graph.x)
        assert torch.equal(loaded_graph.y, graph.y)
        assert torch.equal(loaded_graph.pos, graph.pos)

    def test_serialization_empty_graph(self):
        """Test serialization of empty graph."""
        graph = Data()
        graph.node_labels = []
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

        result = self.persistence.save_neural_map(graph, 0)
        assert result is True

        loaded_graph = self.persistence.load_neural_map(0)
        assert loaded_graph is not None
        assert len(loaded_graph.node_labels) == 0
        assert loaded_graph.edge_index.shape == (2, 0)

    def test_metadata_persistence(self):
        """Test metadata persistence across sessions."""
        # Save metadata
        graph = Data()
        graph.node_labels = [{'id': 1}]
        self.persistence.save_neural_map(graph, 0, {'session': 'test'})

        # Create new persistence instance (simulating new session)
        new_persistence = NeuralMapPersistence(self.temp_dir)

        # Should load existing metadata
        assert 0 in new_persistence.slot_metadata
        assert new_persistence.slot_metadata[0]['session'] == 'test'

    def test_slot_metadata_file_operations(self):
        """Test slot metadata file operations."""
        metadata_file = os.path.join(self.temp_dir, "slot_metadata.json")

        # Initially no metadata file
        assert not os.path.exists(metadata_file)

        # Save a map, should create metadata file
        graph = Data()
        graph.node_labels = [{'id': 1}]
        self.persistence.save_neural_map(graph, 0)

        assert os.path.exists(metadata_file)

        # Check file contents
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            assert '0' in data
            assert data['0']['node_count'] == 1

    def test_error_handling_save(self):
        """Test error handling during save operations."""
        graph = Data()
        graph.node_labels = [{'id': 1}]

        # Mock json.dump to fail
        with patch('json.dump', side_effect=TypeError("Serialization error")):
            result = self.persistence.save_neural_map(graph, 0)
            assert result is False

        # Mock json.dump to fail
        with patch('json.dump', side_effect=TypeError("Serialization error")):
            result = self.persistence.save_neural_map(graph, 0)
            assert result is False

    def test_error_handling_load(self):
        """Test error handling during load operations."""
        # Create invalid JSON file
        filepath = os.path.join(self.temp_dir, "neural_map_slot_0.json")
        with open(filepath, 'w') as f:
            f.write("invalid json")

        result = self.persistence.load_neural_map(0)
        assert result is None

        # Test missing file
        os.remove(filepath)
        result = self.persistence.load_neural_map(0)
        assert result is None

    def test_error_handling_metadata(self):
        """Test error handling in metadata operations."""
        metadata_file = os.path.join(self.temp_dir, "slot_metadata.json")

        # Create invalid metadata file
        with open(metadata_file, 'w') as f:
            f.write("invalid json")

        # Create new persistence instance
        persistence = NeuralMapPersistence(self.temp_dir)
        # Should handle invalid metadata gracefully
        assert isinstance(persistence.slot_metadata, dict)

    def test_persistence_statistics(self):
        """Test persistence statistics."""
        stats = self.persistence.get_persistence_statistics()

        assert stats['total_slots'] == 10
        assert stats['used_slots'] == 0
        assert stats['available_slots'] == 10
        assert stats['current_slot'] == 0
        assert stats['save_directory'] == self.temp_dir

        # Add some slots
        graph = Data()
        graph.node_labels = [{'id': 1}]
        self.persistence.save_neural_map(graph, 0)
        self.persistence.save_neural_map(graph, 1)

        stats = self.persistence.get_persistence_statistics()
        assert stats['used_slots'] == 2
        assert stats['available_slots'] == 8

    def test_cleanup(self):
        """Test cleanup operation."""
        # Add some data
        self.persistence.slot_metadata[0] = {'test': 'data'}

        self.persistence.cleanup()

        assert len(self.persistence.slot_metadata) == 0

    def test_thread_safety(self):
        """Test thread safety of operations."""
        import threading

        graph = Data()
        graph.node_labels = [{'id': 1}]

        errors = []

        def save_operation(slot):
            try:
                self.persistence.save_neural_map(graph, slot)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            thread = threading.Thread(target=save_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        # Should have saved 3 maps
        assert len(self.persistence.list_available_slots()) == 3

    def test_large_graph_serialization(self):
        """Test serialization of large graphs."""
        # Create a moderately large graph
        num_nodes = 1000
        graph = Data()
        graph.node_labels = [{'id': i, 'type': 'dynamic'} for i in range(num_nodes)]
        graph.x = torch.randn(num_nodes, 2)

        # Create some edges
        edge_list = []
        for i in range(0, num_nodes-1, 2):
            edge_list.extend([[i, i+1], [i+1, i]])
        graph.edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        result = self.persistence.save_neural_map(graph, 0)
        assert result is True

        loaded_graph = self.persistence.load_neural_map(0)
        assert loaded_graph is not None
        assert len(loaded_graph.node_labels) == num_nodes
        assert loaded_graph.edge_index.shape[1] == len(edge_list)

    def test_metadata_timestamp(self):
        """Test metadata timestamp handling."""
        import time

        graph = Data()
        graph.node_labels = [{'id': 1}]

        before_save = time.time()
        self.persistence.save_neural_map(graph, 0)
        after_save = time.time()

        metadata = self.persistence.get_slot_info(0)
        assert metadata is not None
        assert before_save <= metadata['save_time'] <= after_save
        assert 'save_date' in metadata


if __name__ == "__main__":
    pytest.main([__file__])