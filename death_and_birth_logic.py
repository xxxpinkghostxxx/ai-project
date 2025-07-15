"""
death_and_birth_logic.py

This module will contain all logic for handling node death (removal) and birth (creation)
in the energy-based neural system graph. Designed for modularity and future extension.
"""

import torch

MAX_DYNAMIC_ENERGY = 1.0  # Should match connection_logic.py
DYNAMIC_BIRTH_THRESHOLD = 0.9 * MAX_DYNAMIC_ENERGY  # 90% threshold
NEW_NODE_ENERGY_FRACTION = 0.4  # 40% of parent node's energy


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
    Remove all dynamic nodes with energy below zero from the graph.
    Frees their id and removes all connections (edges) involving them.
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
    Returns:
        Modified graph with dead dynamic nodes and their edges removed.
    """
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
        return graph
    # Find indices of dynamic nodes with energy < 0
    to_remove = [idx for idx, label in enumerate(graph.node_labels)
                 if label.get('type') == 'dynamic' and graph.x[idx].item() < 0]
    if not to_remove:
        return graph
    # Remove nodes and update node_labels and x
    keep_indices = [i for i in range(len(graph.node_labels)) if i not in to_remove]
    graph.x = graph.x[keep_indices]
    graph.node_labels = [graph.node_labels[i] for i in keep_indices]
    # Remove all edges involving removed nodes
    edge_index = graph.edge_index
    mask = ~((torch.isin(edge_index[0], torch.tensor(to_remove))) |
             (torch.isin(edge_index[1], torch.tensor(to_remove))))
    graph.edge_index = edge_index[:, mask]
    # Optionally, reindex node IDs in node_labels to match new indices
    for new_idx, label in enumerate(graph.node_labels):
        label['id'] = new_idx
    return graph


def birth_new_dynamic_nodes(graph):
    """
    For each dynamic node with energy above 90% of MAX_DYNAMIC_ENERGY,
    generate a new dynamic node with 40% of its parent's energy and no connections between them.
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
    Returns:
        Modified graph with new dynamic nodes added.
    """
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return graph
    new_features = []
    new_labels = []
    for idx, label in enumerate(graph.node_labels):
        if label.get('type') == 'dynamic':
            energy = graph.x[idx].item()
            if energy > DYNAMIC_BIRTH_THRESHOLD:
                new_energy = energy * NEW_NODE_ENERGY_FRACTION
                new_features.append([new_energy])
                new_labels.append({'id': len(graph.node_labels) + len(new_labels), 'type': 'dynamic', 'energy': new_energy})
    if new_features:
        new_features_tensor = torch.tensor(new_features, dtype=graph.x.dtype)
        graph.x = torch.cat([graph.x, new_features_tensor], dim=0)
        graph.node_labels.extend(new_labels)
    return graph

# AI/Human: Extend this file with actual logic for node death/birth as the system evolves. 