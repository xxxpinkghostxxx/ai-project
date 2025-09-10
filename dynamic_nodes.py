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
    Each dynamic node is assigned a unique ID from the ID manager.
    After initialization, use birth, death, and connection logic for further changes.
    Args:
        graph: The graph with sensory nodes already created and labeled.
        num_dynamic: Number of dynamic nodes to add (optional, overrides default 4x sensory).
    Returns:
        The graph with dynamic nodes appended and labeled.
    """
    # Import ID manager for unique ID generation
    from node_id_manager import get_id_manager
    id_manager = get_id_manager()
    
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
        # Generate unique ID for this dynamic node
        node_id = id_manager.generate_unique_id("dynamic")
        energy = float(dynamic_energies[i, 0])
        
        # Calculate normalized membrane potential (0-1)
        membrane_potential = min(energy / MAX_DYNAMIC_ENERGY, 1.0)
        
        # Determine initial state based on energy level
        initial_state = "active" if energy > 0.3 else "inactive"
        
        # Calculate the actual index in the graph
        node_index = len(graph.node_labels)
        
        graph.node_labels.append({
            "id": node_id,  # UNIQUE ID: Primary identifier from ID manager
            "type": "dynamic",
            "behavior": "dynamic",
            "energy": energy,
            "state": initial_state,
            "membrane_potential": membrane_potential,
            "threshold": 0.3,  # Activation threshold for dynamic nodes
            "refractory_timer": 0.0,  # No refractory period initially
            "last_activation": 0,  # No activation history initially
            "plasticity_enabled": True,  # Dynamic nodes can learn
            "eligibility_trace": 0.0,  # No learning history initially
            "last_update": 0
        })
        
        # Register the node index with the ID manager
        id_manager.register_node_index(node_id, node_index)
    # Assertion: node_labels and x must match in length
    assert len(graph.node_labels) == graph.x.shape[0], "Node label and feature count mismatch after dynamic node addition"
    return graph


