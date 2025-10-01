"""
Graph rendering and visualization functions.

This module contains functions for rendering the neural graph visualization,
including node and edge drawing, viewport management, and UI updates.
"""
# pylint: disable=no-member

import math
import sys

import dearpygui as dpg

from src.ui.ui_constants import ui_state
from src.ui.ui_state import get_latest_graph_for_ui, update_ui_display


def _get_graph_visualization_settings():
    """Get visualization settings from UI controls."""
    return {
        'show_nodes': dpg.get_value("show_nodes"),
        'show_edges': dpg.get_value("show_edges"),
        'color_energy': dpg.get_value("color_energy"),
        'node_size': dpg.get_value("node_size"),
        'edge_thickness': dpg.get_value("edge_thickness"),
        'node_active_color': dpg.get_value("node_active_color"),
        'node_inactive_color': dpg.get_value("node_inactive_color"),
        'edge_color': dpg.get_value("edge_color")
    }


def _get_viewport_dimensions():
    """Get the current viewport dimensions."""
    width = dpg.get_item_rect_size("graph_view")[0]
    height = dpg.get_item_rect_size("graph_view")[1]
    return width, height


def _get_node_position(node, index, total_nodes, width, height, zoom, pan_x, pan_y):
    """Calculate position for a node."""
    if 'pos' in node and node['pos']:
        return node['pos']

    # Improved grid layout
    cols = max(1, int(math.sqrt(total_nodes)))
    rows = math.ceil(total_nodes / cols)
    x = (index % cols) / cols * width * zoom + pan_x
    y = (index // cols) / rows * height * zoom + pan_y
    return x, y


def _calculate_node_color(node, settings):
    """Calculate color for a node based on energy and state."""
    energy = node.get('energy', 0.0)
    state = node.get('state', 'inactive')

    if settings['color_energy']:
        r = int(255 * min(energy / 255.0, 1.0))
        g = 255 - r
        b = 128
        return [r, g, b, 255]
    return settings['node_active_color'] if state == 'active' else settings['node_inactive_color']


def _draw_graph_nodes(graph, settings, viewport_dims, zoom_pan):
    """Draw nodes on the graph visualization."""
    width, height = viewport_dims
    zoom, pan_x, pan_y = zoom_pan

    if not (settings['show_nodes'] and hasattr(graph, 'node_labels') and graph.node_labels):
        return

    num_nodes = len(graph.node_labels)
    for i, node in enumerate(graph.node_labels[:500]):  # Increased limit
        x, y = _get_node_position(node, i, num_nodes, width, height, zoom, pan_x, pan_y)
        color = _calculate_node_color(node, settings)
        dpg.draw_circle([x, y], settings['node_size'] * zoom, color=color, thickness=2)


def _draw_graph_edges(graph, settings, viewport_dims, zoom_pan):
    """Draw edges on the graph visualization."""
    width, height = viewport_dims
    zoom, pan_x, pan_y = zoom_pan

    if not (settings['show_edges'] and hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0):
        return

    num_nodes = len(graph.node_labels)
    for j in range(min(graph.edge_index.shape[1], 200)):  # Increased limit
        src_idx = graph.edge_index[0, j].item()
        tgt_idx = graph.edge_index[1, j].item()

        # Get source position
        if src_idx < len(graph.node_labels) and 'pos' in graph.node_labels[src_idx]:
            src_x, src_y = graph.node_labels[src_idx]['pos']
        else:
            src_x, src_y = _get_node_position(graph.node_labels[src_idx] if src_idx < len(graph.node_labels) else {}, src_idx, num_nodes, width, height, zoom, pan_x, pan_y)

        # Get target position
        if tgt_idx < len(graph.node_labels) and 'pos' in graph.node_labels[tgt_idx]:
            tgt_x, tgt_y = graph.node_labels[tgt_idx]['pos']
        else:
            tgt_x, tgt_y = _get_node_position(graph.node_labels[tgt_idx] if tgt_idx < len(graph.node_labels) else {}, tgt_idx, num_nodes, width, height, zoom, pan_x, pan_y)

        dpg.draw_line([src_x, src_y], [tgt_x, tgt_y], color=settings['edge_color'], thickness=settings['edge_thickness'] * zoom)


def update_graph_visualization():
    """Update the graph visualization drawlist."""
    # Skip DPG calls in test environments to avoid crashes
    skip = 'pytest' in sys.modules
    if skip:
        return

    graph = get_latest_graph_for_ui()
    if not graph:
        return

    # Get settings and viewport info
    settings = _get_graph_visualization_settings()
    width, height = _get_viewport_dimensions()

    # Basic zoom and pan (stored in state)
    zoom = ui_state.get_simulation_state().get('viz_zoom', 1.0)
    pan_x = ui_state.get_simulation_state().get('viz_pan_x', 0.0)
    pan_y = ui_state.get_simulation_state().get('viz_pan_y', 0.0)

    zoom_pan = (zoom, pan_x, pan_y)
    viewport_dims = (width, height)

    dpg.clear_draw_list()  # pylint: disable=no-member

    # Draw nodes and edges separately for better organization
    _draw_graph_nodes(graph, settings, viewport_dims, zoom_pan)
    _draw_graph_edges(graph, settings, viewport_dims, zoom_pan)

    # Note: Full zoom/pan requires mouse handlers; basic scaling here


def update_frame():
    """Update UI and graph every frame."""
    update_ui_display()
    update_graph_visualization()
