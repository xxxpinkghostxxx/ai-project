"""
death_and_birth_logic.py

This module will contain all logic for handling node death (removal) and birth (creation)
in the energy-based neural system graph. Designed for modularity and future extension.
"""

import torch

MAX_DYNAMIC_ENERGY = 1.0  # Should match connection_logic.py
DYNAMIC_BIRTH_THRESHOLD = 0.9 * MAX_DYNAMIC_ENERGY  # 90% threshold
NEW_NODE_ENERGY_FRACTION = 0.4  # 40% of parent node's energy
NODE_ENERGY_CAP = 244.0  # Clamp dynamic node energy to this value
NODE_BIRTH_THRESHOLD = 200.0  # Example threshold for birth (can be set in config)
NODE_BIRTH_COST = 80.0  # Energy cost to parent for spawning a new node
NODE_DEATH_THRESHOLD = 0.0  # Threshold for dynamic node death


def handle_node_death(graph, node_id, strategy=None):
    """
    Handle the removal (death) of a node from the graph according to the specified strategy.
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
        node_id (int): The ID of the node to remove.
        strategy (callable or None): Optional function to determine death policy.
    Returns:
        Modified graph with the node removed (if applicable).
    """
    # TODO: Implement node removal logic here
    pass


def handle_node_birth(graph, birth_params=None):
    """
    Handle the creation (birth) of a new node in the graph according to the specified parameters.
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
        birth_params (dict or None): Parameters for node creation (type, initial energy, etc).
    Returns:
        Modified graph with the new node added (if applicable).
    """
    # TODO: Implement node creation logic here
    pass


def remove_dead_dynamic_nodes(graph):
    """
    Remove all dynamic nodes with energy below NODE_DEATH_THRESHOLD from the graph.
    Frees their id and removes all connections (edges) involving them.
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
    Returns:
        Modified graph with dead dynamic nodes and their edges removed.
    """
    import logging

    if (
        not hasattr(graph, "node_labels")
        or not hasattr(graph, "x")
        or not hasattr(graph, "edge_index")
    ):
        return graph
    # Find indices of dynamic nodes with energy < NODE_DEATH_THRESHOLD
    to_remove = [
        idx
        for idx, label in enumerate(graph.node_labels)
        if label.get("type") == "dynamic" and graph.x[idx].item() < NODE_DEATH_THRESHOLD
    ]
    if not to_remove:
        return graph
    # Log node deaths
    for idx in to_remove:
        logging.info(f"[DEATH] Node {idx} removed (energy={graph.x[idx].item():.2f})")
    # Remove nodes and update node_labels and x
    keep_indices = [i for i in range(len(graph.node_labels)) if i not in to_remove]
    graph.x = graph.x[keep_indices]
    graph.node_labels = [graph.node_labels[i] for i in keep_indices]
    # Remove all edges involving removed nodes
    edge_index = graph.edge_index
    mask = ~(
        (torch.isin(edge_index[0], torch.tensor(to_remove)))
        | (torch.isin(edge_index[1], torch.tensor(to_remove)))
    )
    graph.edge_index = edge_index[:, mask]
    # Optionally, reindex node IDs in node_labels to match new indices
    for new_idx, label in enumerate(graph.node_labels):
        label["id"] = new_idx
    # Assertion: node_labels and x must match in length
    assert len(graph.node_labels) == graph.x.shape[0], "Node label and feature count mismatch after node death"
    # Assertion: all edge indices are valid
    if graph.edge_index.numel() > 0:
        assert torch.all(graph.edge_index < len(graph.node_labels)), "Edge index out of bounds after node death"
    return graph


def birth_new_dynamic_nodes(graph):
    """
    For each dynamic node with energy above NODE_BIRTH_THRESHOLD,
    generate a new dynamic node with a fraction of its parent's energy and deduct a birth cost.
    Clamp energies to [0, NODE_ENERGY_CAP].
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
    Returns:
        Modified graph with new dynamic nodes added.
    """
    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        return graph
    x = graph.x
    node_labels = graph.node_labels
    num_nodes = len(node_labels)
    dynamic_indices = [
        i for i, lbl in enumerate(node_labels) if lbl.get("type") == "dynamic"
    ]
    new_features = []
    new_labels = []
    for idx in dynamic_indices:
        energy = x[idx].item()
        if energy > NODE_BIRTH_THRESHOLD:
            # Deduct birth cost from parent
            new_parent_energy = max(energy - NODE_BIRTH_COST, 0)
            x[idx] = min(new_parent_energy, NODE_ENERGY_CAP)
            # Assign a fraction of parent's energy to new node
            new_energy = min(energy * NEW_NODE_ENERGY_FRACTION, NODE_ENERGY_CAP)
            new_features.append([new_energy])
            new_labels.append({
                "id": num_nodes + len(new_labels),
                "type": "dynamic",
                "energy": new_energy,
                "behavior": "dynamic",
                "state": "active",
                "last_update": 0
            })
            import logging
            logging.info(
                f"[BIRTH] Node {idx} spawned new node with energy {new_energy:.2f} (parent energy now {x[idx].item():.2f})"
            )
    if new_features:
        new_features_tensor = torch.tensor(new_features, dtype=x.dtype)
        graph.x = torch.cat([x, new_features_tensor], dim=0)
        graph.node_labels.extend(new_labels)
        # Assertion: node_labels and x must match in length
        assert len(graph.node_labels) == graph.x.shape[0], "Node label and feature count mismatch after node birth"
    return graph


# AI/Human: Extend this file with actual logic for node death/birth as the system evolves.
