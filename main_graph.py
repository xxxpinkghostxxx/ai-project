"""
main_graph.py
Entry point for main graph initialization in the energy-based neural system.
This module constructs the full graph (sensory + dynamic nodes) in a single function for use by the rest of the system.
"""

from screen_graph import capture_screen, create_pixel_rgb_graph, RESOLUTION_SCALE
from dynamic_nodes import add_dynamic_nodes


def initialize_main_graph(scale=RESOLUTION_SCALE):
    """
    Initializes the main graph by capturing the screen, creating the pixel RGB graph,
    and appending dynamic nodes.

    Args:
        scale (float): Downscaling factor for screen capture.

    Returns:
        torch_geometric.data.Data: The fully constructed graph with node labels.
    """
    arr = capture_screen(scale=scale)
    pixel_graph = create_pixel_rgb_graph(arr)
    full_graph = add_dynamic_nodes(pixel_graph)
    return full_graph

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