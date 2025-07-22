import torch
import numpy as np
from screen_graph import capture_screen, create_pixel_gray_graph, RESOLUTION_SCALE
import time

# Maximum energy for dynamic nodes
MAX_DYNAMIC_ENERGY = 1.0

# Function to add dynamic nodes to the pixel graph
# This function should ONLY be used at initialization to populate the initial dynamic node pool.
# After initialization, dynamic node population should rely on birth, death, and connection logic.


def add_dynamic_nodes(graph, num_dynamic=None):
    """
    Appends dynamic nodes to the graph after all sensory nodes.
    This function is ONLY for initial population at simulation start.
    The number of dynamic nodes is set to 4x the number of sensory nodes by default.
    Each dynamic node is assigned a unique 'id' that follows the last sensory node's id,
    and is initialized with a random energy value between 0 and MAX_DYNAMIC_ENERGY.
    After initialization, use birth, death, and connection logic for further changes.
    Args:
        graph: The graph with sensory nodes already created and labeled.
        num_dynamic: Number of dynamic nodes to add (optional, overrides default 4x sensory).
    Returns:
        The graph with dynamic nodes appended and labeled.
    """
    num_sensory = len(
        [lbl for lbl in graph.node_labels if lbl.get("type", "sensory") == "sensory"]
    )
    if num_dynamic is None:
        num_dynamic = 4 * num_sensory
    # Random energy values for dynamic nodes
    dynamic_energies = np.random.uniform(
        0, MAX_DYNAMIC_ENERGY, size=(num_dynamic, 1)
    ).astype(np.float32)
    dynamic_features = torch.tensor(dynamic_energies, dtype=torch.float32)
    graph.x = torch.cat([graph.x, dynamic_features], dim=0)
    for i in range(num_dynamic):
        node_id = len(graph.node_labels)
        energy = float(dynamic_energies[i, 0])
        graph.node_labels.append({
            "id": node_id,
            "type": "dynamic",
            "energy": energy,
            "behavior": "dynamic",
            "state": "active",
            "last_update": 0
        })
    # Assertion: node_labels and x must match in length
    assert len(graph.node_labels) == graph.x.shape[0], "Node label and feature count mismatch after dynamic node addition"
    return graph


if __name__ == "__main__":
    print("Press Ctrl+C to stop.")
    try:
        while True:
            arr = capture_screen(scale=RESOLUTION_SCALE)
            pixel_graph = create_pixel_gray_graph(arr)
            # Only call add_dynamic_nodes ONCE at initialization in real simulation
            full_graph = add_dynamic_nodes(pixel_graph)
            # Assertion: node_labels and x must match in length
            assert len(full_graph.node_labels) == full_graph.x.shape[0], "Node label and feature count mismatch in main test loop"
            print(f"Graph: {len(full_graph.x)} nodes (including dynamic nodes)")
            print(f"First 6 node labels: {full_graph.node_labels[:6]}")
            print(f"Last 6 node labels: {full_graph.node_labels[-6:]}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")
