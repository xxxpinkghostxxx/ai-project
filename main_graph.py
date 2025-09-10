"""
main_graph.py
Entry point for main graph initialization in the energy-based neural system.
This module constructs the full graph (sensory + dynamic nodes) in a single function for use by the rest of the system.
"""

from screen_graph import capture_screen, create_pixel_gray_graph, RESOLUTION_SCALE
from dynamic_nodes import add_dynamic_nodes
import torch


def create_workspace_grid():
    """
    Creates a 16x16 grid of workspace nodes.
    Each node represents a pixel in the internal workspace grid.
    
    Returns:
        torch_geometric.data.Data: Graph containing only the workspace grid nodes.
    """
    # Import ID manager for unique ID generation
    from node_id_manager import get_id_manager
    id_manager = get_id_manager()
    
    grid_size = 16
    num_nodes = grid_size * grid_size
    
    # Create node features (energy values, initially 0)
    x = torch.zeros((num_nodes, 1), dtype=torch.float32)
    
    # Create node labels for each workspace node
    node_labels = []
    for y in range(grid_size):
        for x_coord in range(grid_size):
            # Generate unique ID for this workspace node
            node_id = id_manager.generate_unique_id("workspace", {"x": x_coord, "y": y})
            
            # Calculate the index in the grid
            node_index = y * grid_size + x_coord
            
            node_label = {
                "id": node_id,  # UNIQUE ID: Primary identifier from ID manager
                "type": "workspace",
                "behavior": "workspace",
                "x": x_coord,  # METADATA: Grid coordinate preserved as metadata
                "y": y,        # METADATA: Grid coordinate preserved as metadata
                "energy": 0.0,
                "state": "inactive",
                "membrane_potential": 0.0,
                "threshold": 0.6,
                "refractory_timer": 0.0,
                "last_activation": 0,
                "plasticity_enabled": True,
                "eligibility_trace": 0.0,
                "last_update": 0,
                "workspace_capacity": 5.0,
                "workspace_creativity": 1.5,
                "workspace_focus": 3.0
            }
            node_labels.append(node_label)
            
            # Register the node index with the ID manager
            id_manager.register_node_index(node_id, node_index)
    
    # Create empty edge index (no connections initially)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create the workspace graph
    from torch_geometric.data import Data
    workspace_graph = Data(
        x=x,
        edge_index=edge_index,
        node_labels=node_labels,
        h=grid_size,
        w=grid_size
    )
    
    return workspace_graph


def merge_graphs(graph1, graph2):
    """
    Merges two PyTorch Geometric graphs by combining their nodes and edges.
    
    Args:
        graph1: First graph
        graph2: Second graph
        
    Returns:
        torch_geometric.data.Data: Merged graph
    """
    from torch_geometric.data import Data
    
    # Combine node features
    combined_x = torch.cat([graph1.x, graph2.x], dim=0)
    
    # Combine node labels
    combined_labels = graph1.node_labels + graph2.node_labels
    
    # Adjust edge indices for the second graph
    if hasattr(graph2, 'edge_index') and graph2.edge_index.numel() > 0:
        offset = graph1.x.size(0)
        adjusted_edge_index = graph2.edge_index + offset
        combined_edge_index = torch.cat([graph1.edge_index, adjusted_edge_index], dim=1)
    else:
        combined_edge_index = graph1.edge_index
    
    # Create merged graph
    merged_graph = Data(
        x=combined_x,
        edge_index=combined_edge_index,
        node_labels=combined_labels
    )
    
    # Copy other attributes
    for attr in ['h', 'w', 'step']:
        if hasattr(graph1, attr):
            setattr(merged_graph, attr, getattr(graph1, attr))
    
    return merged_graph


def initialize_main_graph(scale=RESOLUTION_SCALE):
    """
    Initializes the main graph by capturing the screen, creating the pixel grayscale graph,
    appending dynamic nodes, and adding the workspace grid.

    Args:
        scale (float): Downscaling factor for screen capture.

    Returns:
        torch_geometric.data.Data: The fully constructed graph with node labels.
    """
    # Create the main sensory and dynamic node graph
    arr = capture_screen(scale=scale)
    pixel_graph = create_pixel_gray_graph(arr)
    main_graph = add_dynamic_nodes(pixel_graph)
    
    # Create and merge the workspace grid
    workspace_graph = create_workspace_grid()
    full_graph = merge_graphs(main_graph, workspace_graph)
    
    return full_graph


def select_nodes_by_type(graph, node_type):
    """
    Return a list of node IDs for nodes of the given type (e.g., 'sensory', 'dynamic').
    Uses ID-based selection instead of array indices.
    """
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_type(node_type)

def select_nodes_by_state(graph, state):
    """
    Return a list of node IDs for nodes with the given state (e.g., 'active', 'inactive').
    Uses ID-based selection instead of array indices.
    """
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_state(state)

def select_nodes_by_behavior(graph, behavior):
    """
    Return a list of node IDs for nodes with the given behavior (e.g., 'sensory', 'dynamic', 'oscillator').
    Uses ID-based selection instead of array indices.
    """
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    return access_layer.select_nodes_by_behavior(behavior)

# Example usage (for testing/debugging)
if __name__ == "__main__":
    import time

    print("Press Ctrl+C to stop.")
    try:
        while True:
            graph = initialize_main_graph()
            print(f"Graph: {len(graph.x)} nodes (including dynamic nodes)")
            print(f"First 6 node labels: {graph.node_labels[:6]}")
            print(f"Last 6 node labels: {graph.node_labels[-6:]}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")
