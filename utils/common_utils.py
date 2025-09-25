"""
Common utility functions to reduce code duplication and improve maintainability.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from torch_geometric.data import Data


def safe_hasattr(obj: Any, *attrs: str) -> bool:
    """
    Check if object has all specified attributes safely.
    Replaces complex 'hasattr(obj, 'attr1') and hasattr(obj, 'attr2')' patterns.
    """
    try:
        return all(hasattr(obj, attr) for attr in attrs)
    except Exception:
        return False


def safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get attribute with default value.
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def validate_graph_structure(graph) -> Tuple[bool, List[str]]:
    """
    Validate graph structure and return validation status and missing attributes.
    Replaces repeated graph validation patterns.
    """
    if graph is None:
        return False, ['node_labels', 'x', 'edge_index']

    required_attrs = ['node_labels', 'x', 'edge_index']
    missing_attrs = []

    # Check node_labels: must be list and not empty
    if not hasattr(graph, 'node_labels') or not isinstance(graph.node_labels, list) or len(graph.node_labels) == 0:
        missing_attrs.append('node_labels')

    # Check x: must be tensor-like (has shape)
    if not hasattr(graph, 'x') or not hasattr(graph.x, 'shape'):
        missing_attrs.append('x')

    # Check edge_index: must be tensor-like (has shape)
    if not hasattr(graph, 'edge_index') or not hasattr(graph.edge_index, 'shape'):
        missing_attrs.append('edge_index')

    return len(missing_attrs) == 0, missing_attrs


def safe_graph_access(graph: Data, operation: str, *args, **kwargs) -> Any:
    """
    Safely access graph with error handling.
    Replaces repeated try-except patterns for graph operations.
    """
    try:
        if operation == 'get_node_count':
            return len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
        elif operation == 'get_edge_count':
            return graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0 else 0
        elif operation == 'has_visual_data':
            return safe_hasattr(graph, 'visual_data') and graph.visual_data is not None
        elif operation == 'has_audio_data':
            return safe_hasattr(graph, 'audio_data') and graph.audio_data is not None
        else:
            return None
    except Exception as e:
        return None


def create_safe_callback(callback_func, error_handler=None):
    """
    Create a safe callback wrapper with error handling.
    Replaces repeated callback error handling patterns.
    """
    def safe_wrapper(*args, **kwargs):
        try:
            return callback_func(*args, **kwargs)
        except Exception as e:
            if error_handler:
                error_handler(e)
            else:
                print(f"Callback error: {e}")
            return None
    return safe_wrapper


def extract_common_constants():
    """
    Extract common string constants to reduce duplication.
    """
    return {
        'SIMULATION_STATUS_RUNNING': 'Running',
        'SIMULATION_STATUS_STOPPED': 'Stopped',
        'NEURAL_SIMULATION_TITLE': 'Neural Simulation',
        'MAIN_WINDOW_TAG': 'main_window',
        'STATUS_TEXT_TAG': 'status_text',
        'NODES_TEXT_TAG': 'nodes_text',
        'EDGES_TEXT_TAG': 'edges_text',
        'ENERGY_TEXT_TAG': 'energy_text',
        'CONNECTIONS_TEXT_TAG': 'connections_text'
    }


def get_common_error_messages():
    """
    Extract common error messages to reduce duplication.
    """
    return {
        'GRAPH_NONE': 'Graph is None',
        'INVALID_NODE_ID': 'Invalid node ID',
        'MISSING_ATTRIBUTE': 'Missing required attribute',
        'CALLBACK_ERROR': 'Callback execution failed',
        'UI_UPDATE_ERROR': 'UI update failed'
    }
