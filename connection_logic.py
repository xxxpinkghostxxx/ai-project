"""
connection_logic.py

This module will contain all logic for managing connections (edges) between nodes in the energy-based neural system graph.
Designed for modularity and future extension.
"""

import torch
import random

MAX_DYNAMIC_ENERGY = 1.0  # Define the maximum energy for dynamic nodes
DYNAMIC_ENERGY_THRESHOLD = 0.8 * MAX_DYNAMIC_ENERGY  # 80% threshold


def add_connections(graph, connection_strategy=None):
    """
    Add or update connections (edges) in the graph according to the specified strategy.
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
        connection_strategy (callable or None): Optional function to determine connection policy.
    Returns:
        Modified graph with updated edges.
    """
    # Placeholder for future strategies
    pass


def add_dynamic_connections(graph):
    """
    For each dynamic node with energy above 80% of MAX_DYNAMIC_ENERGY,
    create a connection (edge) to a randomly selected node (excluding itself).
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
    Returns:
        Modified graph with new edges added.
    """
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return graph
    edge_index = graph.edge_index.clone() if graph.edge_index.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
    num_nodes = len(graph.node_labels)
    for idx, label in enumerate(graph.node_labels):
        if label.get('type') == 'dynamic':
            # Assume energy is stored in the node feature (graph.x)
            energy = graph.x[idx].item()
            if energy > DYNAMIC_ENERGY_THRESHOLD:
                # Pick a random target node (excluding self)
                possible_targets = list(range(num_nodes))
                possible_targets.remove(idx)
                if possible_targets:
                    target = random.choice(possible_targets)
                    # Add edge from dynamic node to target
                    new_edge = torch.tensor([[idx], [target]], dtype=torch.long)
                    edge_index = torch.cat([edge_index, new_edge], dim=1)
    graph.edge_index = edge_index
    return graph

# AI/Human: Extend this file with more sophisticated connection strategies as needed. 