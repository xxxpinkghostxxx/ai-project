"""
Comprehensive tests for common_utils.py
Covers unit tests, integration tests, edge cases, error handling, performance, and real-world usage.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from unittest.mock import Mock
from torch_geometric.data import Data
import torch

from src.utils.common_utils import (
    safe_hasattr, safe_get_attr, validate_graph_structure,
    safe_graph_access, create_safe_callback, extract_common_constants,
    get_common_error_messages
)


class TestSafeHasattr:
    """Test safe_hasattr function."""

    def test_safe_hasattr_single_attribute(self):
        """Test checking single attribute."""
        class TestObj:
            pass
        obj = TestObj()
        obj.attr1 = "value"
        assert safe_hasattr(obj, "attr1") == True
        assert safe_hasattr(obj, "attr2") == False

    def test_safe_hasattr_multiple_attributes(self):
        """Test checking multiple attributes."""
        class TestObj:
            pass
        obj = TestObj()
        obj.attr1 = "value1"
        obj.attr2 = "value2"
        assert safe_hasattr(obj, "attr1", "attr2") == True
        assert safe_hasattr(obj, "attr1", "attr3") == False

    def test_safe_hasattr_exception_handling(self):
        """Test exception handling in safe_hasattr."""
        class BadObject:
            def __getattr__(self, name):
                raise RuntimeError("Attribute access failed")

        obj = BadObject()
        assert safe_hasattr(obj, "any_attr") == False

    @pytest.mark.parametrize("attrs", [
        ("attr1",),
        ("attr1", "attr2", "attr3"),
        ("nonexistent",),
        ()
    ])
    def test_safe_hasattr_parametrized(self, attrs):
        """Parametrized test for safe_hasattr."""
        class TestObj:
            pass
        obj = TestObj()
        obj.attr1 = "value"
        obj.attr2 = "value"

        expected = all(hasattr(obj, attr) for attr in attrs)
        assert safe_hasattr(obj, *attrs) == expected


class TestSafeGetAttr:
    """Test safe_get_attr function."""

    def test_safe_get_attr_existing_attribute(self):
        """Test getting existing attribute."""
        obj = Mock()
        obj.attr = "value"
        assert safe_get_attr(obj, "attr") == "value"

    def test_safe_get_attr_nonexistent_attribute(self):
        """Test getting nonexistent attribute with default."""
        class TestObj:
            pass
        obj = TestObj()
        assert safe_get_attr(obj, "nonexistent", "default") == "default"

    def test_safe_get_attr_exception_handling(self):
        """Test exception handling in safe_get_attr."""
        class BadObject:
            def __getattribute__(self, name):
                raise AttributeError("Attribute access failed")

        obj = BadObject()
        assert safe_get_attr(obj, "any_attr", "default") == "default"


class TestValidateGraphStructure:
    """Test validate_graph_structure function."""

    def test_validate_graph_structure_valid_graph(self):
        """Test validation of valid graph."""
        graph = Data()
        graph.node_labels = [{"id": 1}, {"id": 2}]
        graph.x = torch.randn(2, 1)
        graph.edge_index = torch.tensor([[0, 1], [1, 0]])

        is_valid, missing = validate_graph_structure(graph)
        assert is_valid == True
        assert missing == []

    def test_validate_graph_structure_missing_attributes(self):
        """Test validation with missing attributes."""
        graph = Data()  # Missing all required attributes

        is_valid, missing = validate_graph_structure(graph)
        assert is_valid == False
        assert set(missing) == {"node_labels", "x", "edge_index"}

    def test_validate_graph_structure_partial_missing(self):
        """Test validation with some missing attributes."""
        graph = Data()
        graph.node_labels = [{"id": 1}]
        # Missing x and edge_index

        is_valid, missing = validate_graph_structure(graph)
        assert is_valid == False
        assert set(missing) == {"x", "edge_index"}

    def test_validate_graph_structure_edge_cases(self):
        """Test edge cases in graph validation."""
        # None graph
        is_valid, missing = validate_graph_structure(None)
        assert is_valid == False
        assert missing == ["node_labels", "x", "edge_index"]

        # Graph with wrong attribute types
        graph = Data()
        graph.node_labels = "not_a_list"
        graph.x = "not_a_tensor"
        graph.edge_index = "not_a_tensor"

        is_valid, missing = validate_graph_structure(graph)
        assert is_valid == False
        assert set(missing) == {"node_labels", "x", "edge_index"}


class TestSafeGraphAccess:
    """Test safe_graph_access function."""

    def test_safe_graph_access_node_count(self):
        """Test getting node count."""
        graph = Data()
        graph.node_labels = [{"id": 1}, {"id": 2}, {"id": 3}]

        result = safe_graph_access(graph, "get_node_count")
        assert result == 3

    def test_safe_graph_access_edge_count(self):
        """Test getting edge count."""
        graph = Data()
        graph.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

        result = safe_graph_access(graph, "get_edge_count")
        assert result == 3

    def test_safe_graph_access_visual_data(self):
        """Test checking visual data presence."""
        graph = Data()
        graph.visual_data = torch.randn(10, 3)

        assert safe_graph_access(graph, "has_visual_data") == True

        graph2 = Data()
        assert safe_graph_access(graph2, "has_visual_data") == False

    def test_safe_graph_access_audio_data(self):
        """Test checking audio data presence."""
        graph = Data()
        graph.audio_data = torch.randn(100, 1)

        assert safe_graph_access(graph, "has_audio_data") == True

        graph2 = Data()
        assert safe_graph_access(graph2, "has_audio_data") == False

    def test_safe_graph_access_invalid_operation(self):
        """Test invalid operation."""
        graph = Data()
        result = safe_graph_access(graph, "invalid_operation")
        assert result is None

    def test_safe_graph_access_exception_handling(self):
        """Test exception handling in safe_graph_access."""
        class BadGraph:
            @property
            def node_labels(self):
                raise RuntimeError("Access failed")

        graph = BadGraph()
        result = safe_graph_access(graph, "get_node_count")
        assert result is None


class TestCreateSafeCallback:
    """Test create_safe_callback function."""

    def test_create_safe_callback_success(self):
        """Test successful callback execution."""
        def test_func(x, y):
            return x + y

        safe_callback = create_safe_callback(test_func)
        result = safe_callback(2, 3)
        assert result == 5

    def test_create_safe_callback_with_error_handler(self):
        """Test callback with custom error handler."""
        error_handler_called = False

        def failing_func():
            raise ValueError("Test error")

        def error_handler(error):
            nonlocal error_handler_called
            error_handler_called = True
            assert isinstance(error, ValueError)

        safe_callback = create_safe_callback(failing_func, error_handler)
        result = safe_callback()

        assert result is None
        assert error_handler_called == True

    def test_create_safe_callback_exception_handling(self):
        """Test exception handling in callback."""
        def failing_func():
            raise RuntimeError("Callback failed")

        safe_callback = create_safe_callback(failing_func)
        result = safe_callback()

        assert result is None

    def test_create_safe_callback_with_args_kwargs(self):
        """Test callback with arguments and keyword arguments."""
        def test_func(a, b=None, c=10):
            return a + (b or 0) + c

        safe_callback = create_safe_callback(test_func)
        result = safe_callback(1, b=2, c=3)
        assert result == 6


class TestExtractCommonConstants:
    """Test extract_common_constants function."""

    def test_extract_common_constants_structure(self):
        """Test that constants have expected structure."""
        constants = extract_common_constants()

        assert isinstance(constants, dict)
        assert len(constants) > 0

        # Check for expected keys
        expected_keys = [
            'SIMULATION_STATUS_RUNNING',
            'SIMULATION_STATUS_STOPPED',
            'NEURAL_SIMULATION_TITLE',
            'MAIN_WINDOW_TAG',
            'STATUS_TEXT_TAG',
            'NODES_TEXT_TAG',
            'EDGES_TEXT_TAG',
            'ENERGY_TEXT_TAG',
            'CONNECTIONS_TEXT_TAG'
        ]

        for key in expected_keys:
            assert key in constants
            assert isinstance(constants[key], str)

    def test_extract_common_constants_values(self):
        """Test that constants have reasonable values."""
        constants = extract_common_constants()

        assert constants['SIMULATION_STATUS_RUNNING'] == 'Running'
        assert constants['SIMULATION_STATUS_STOPPED'] == 'Stopped'
        assert 'Neural' in constants['NEURAL_SIMULATION_TITLE']


class TestGetCommonErrorMessages:
    """Test get_common_error_messages function."""

    def test_get_common_error_messages_structure(self):
        """Test that error messages have expected structure."""
        messages = get_common_error_messages()

        assert isinstance(messages, dict)
        assert len(messages) > 0

        # Check for expected keys
        expected_keys = [
            'GRAPH_NONE',
            'INVALID_NODE_ID',
            'MISSING_ATTRIBUTE',
            'CALLBACK_ERROR',
            'UI_UPDATE_ERROR'
        ]

        for key in expected_keys:
            assert key in messages
            assert isinstance(messages[key], str)

    def test_get_common_error_messages_values(self):
        """Test that error messages have reasonable values."""
        messages = get_common_error_messages()

        assert 'Graph is None' in messages['GRAPH_NONE']
        assert 'Invalid node ID' in messages['INVALID_NODE_ID']
        assert 'Missing required attribute' in messages['MISSING_ATTRIBUTE']


class TestIntegration:
    """Integration tests for common_utils functions."""

    def test_graph_validation_and_access_integration(self):
        """Test integration of graph validation and access functions."""
        # Create a valid graph
        graph = Data()
        graph.node_labels = [{"id": 1}, {"id": 2}]
        graph.x = torch.randn(2, 1)
        graph.edge_index = torch.tensor([[0, 1], [1, 0]])

        # Validate structure
        is_valid, missing = validate_graph_structure(graph)
        assert is_valid == True

        # Access graph properties safely
        node_count = safe_graph_access(graph, "get_node_count")
        edge_count = safe_graph_access(graph, "get_edge_count")

        assert node_count == 2
        assert edge_count == 2

    def test_callback_creation_and_execution_integration(self):
        """Test integration of callback creation and execution."""
        results = []

        def test_callback(value):
            results.append(value)
            return value * 2

        # Create safe callback
        safe_cb = create_safe_callback(test_callback)

        # Execute callback
        result = safe_cb(5)

        assert result == 10
        assert results == [5]

    def test_constants_and_messages_integration(self):
        """Test integration of constants and error messages."""
        constants = extract_common_constants()
        messages = get_common_error_messages()

        # Both should be dictionaries
        assert isinstance(constants, dict)
        assert isinstance(messages, dict)

        # Should have different keys
        assert set(constants.keys()).isdisjoint(set(messages.keys()))


class TestPerformance:
    """Performance tests for common_utils functions."""

    def test_safe_hasattr_performance(self):
        """Test performance of safe_hasattr with many attributes."""
        obj = Mock()
        # Add many attributes
        for i in range(100):
            setattr(obj, f"attr_{i}", f"value_{i}")

        attrs_to_check = [f"attr_{i}" for i in range(100)]

        start_time = time.time()
        for _ in range(1000):
            result = safe_hasattr(obj, *attrs_to_check)
            assert result == True
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0

    def test_safe_graph_access_performance(self):
        """Test performance of safe_graph_access."""
        graph = Data()
        graph.node_labels = [{"id": i} for i in range(1000)]
        graph.edge_index = torch.randint(0, 1000, (2, 2000))

        start_time = time.time()
        for _ in range(1000):
            node_count = safe_graph_access(graph, "get_node_count")
            edge_count = safe_graph_access(graph, "get_edge_count")
            assert node_count == 1000
            assert edge_count == 2000
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0

    def test_callback_performance(self):
        """Test performance of safe callbacks."""
        def simple_callback(x):
            return x + 1

        safe_cb = create_safe_callback(simple_callback)

        start_time = time.time()
        for i in range(10000):
            result = safe_cb(i)
            assert result == i + 1
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0


class TestRealWorldUsage:
    """Real-world usage scenarios for common_utils."""

    def test_graph_processing_workflow(self):
        """Test a complete graph processing workflow."""
        # Create graph
        graph = Data()
        graph.node_labels = [{"id": i, "type": "neuron"} for i in range(10)]
        graph.x = torch.randn(10, 1)
        graph.edge_index = torch.randint(0, 10, (2, 20))

        # Validate graph
        is_valid, missing = validate_graph_structure(graph)
        assert is_valid == True

        # Access properties
        node_count = safe_graph_access(graph, "get_node_count")
        edge_count = safe_graph_access(graph, "get_edge_count")

        assert node_count == 10
        assert edge_count == 20

        # Check for visual/audio data (should be false)
        has_visual = safe_graph_access(graph, "has_visual_data")
        has_audio = safe_graph_access(graph, "has_audio_data")

        assert has_visual == False
        assert has_audio == False

    def test_error_handling_workflow(self):
        """Test error handling workflow with callbacks."""
        error_log = []

        def risky_operation():
            raise ValueError("Operation failed")

        def error_handler(error):
            error_log.append(str(error))

        # Create safe callback
        safe_operation = create_safe_callback(risky_operation, error_handler)

        # Execute and handle error
        result = safe_operation()

        assert result is None
        assert len(error_log) == 1
        assert "Operation failed" in error_log[0]

    def test_configuration_workflow(self):
        """Test configuration workflow with constants."""
        constants = extract_common_constants()
        messages = get_common_error_messages()

        # Simulate using constants in UI setup
        ui_tags = {
            'main_window': constants['MAIN_WINDOW_TAG'],
            'status_text': constants['STATUS_TEXT_TAG'],
            'nodes_text': constants['NODES_TEXT_TAG']
        }

        assert len(ui_tags) == 3
        assert all(isinstance(tag, str) for tag in ui_tags.values())

        # Simulate error handling
        error_scenarios = ['graph_none', 'invalid_node', 'missing_attr']
        for scenario in error_scenarios:
            key = scenario.upper().replace('_', '')
            if key in messages:
                assert isinstance(messages[key], str)


if __name__ == "__main__":
    pytest.main([__file__])






