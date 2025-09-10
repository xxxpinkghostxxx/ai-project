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
NODE_BIRTH_THRESHOLD = 0.8  # Threshold for birth (80% of max dynamic energy)
NODE_BIRTH_COST = 0.3  # Energy cost to parent for spawning a new node (30% of max energy)
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
    import logging
    
    try:
        if not hasattr(graph, 'node_labels') or node_id >= len(graph.node_labels):
            logging.warning(f"Invalid node_id {node_id} for death handling")
            return graph
        
        node = graph.node_labels[node_id]
        node_type = node.get('type', 'unknown')
        node_energy = graph.x[node_id].item() if hasattr(graph, 'x') else 0.0
        
        # Check if node should be removed based on strategy
        should_remove = False
        
        # Check memory importance if memory system is available
        memory_importance = 0.0
        if hasattr(graph, 'memory_system') and graph.memory_system:
            memory_importance = graph.memory_system.get_node_memory_importance(node_id)
        
        if strategy == 'conservative':
            # Only remove if energy is very low, node is inactive, AND has low memory importance
            should_remove = (node_energy < 0.1 and 
                           node.get('state', 'active') == 'inactive' and
                           memory_importance < 0.2)
        elif strategy == 'aggressive':
            # Remove if energy is low or node is inactive, but still consider memory
            should_remove = ((node_energy < 0.5 or 
                            node.get('state', 'active') == 'inactive') and
                           memory_importance < 0.5)
        elif strategy == 'memory_aware':
            # Memory-aware strategy: protect nodes with high memory importance
            if memory_importance > 0.7:
                should_remove = False  # Protect high-importance nodes
            else:
                should_remove = (node_energy <= NODE_DEATH_THRESHOLD and
                               memory_importance < 0.3)
        elif callable(strategy):
            # Use custom strategy function
            should_remove = strategy(node, graph, node_id)
        else:
            # Default strategy: remove if energy is zero, but consider memory importance
            should_remove = (node_energy <= NODE_DEATH_THRESHOLD and
                           memory_importance < 0.4)
        
        if should_remove:
            # Remove node from graph
            success = remove_node_from_graph(graph, node_id)
            if success:
                logging.info(f"Node {node_id} ({node_type}) removed successfully (energy: {node_energy:.2f})")
            else:
                logging.warning(f"Failed to remove node {node_id}")
        else:
            logging.debug(f"Node {node_id} not removed (strategy: {strategy}, energy: {node_energy:.2f})")
        
        return graph
            
    except Exception as e:
        logging.error(f"Error handling node death for node {node_id}: {e}")
        return graph


def handle_node_birth(graph, birth_params=None):
    """
    Handle the creation (birth) of a new node in the graph according to the specified parameters.
    Args:
        graph (torch_geometric.data.Data): The graph to modify.
        birth_params (dict or None): Parameters for node creation (type, initial energy, etc).
    Returns:
        Modified graph with the new node added (if applicable).
    """
    import logging
    
    try:
        if birth_params is None:
            # Analyze memory patterns to determine optimal node creation
            memory_influenced_params = analyze_memory_patterns_for_birth(graph)
            birth_params = memory_influenced_params
        
        # Create new node
        new_node = create_new_node(graph, birth_params)
        if new_node:
            # Add node to graph
            success = add_node_to_graph(graph, new_node)
            if success:
                logging.info(f"New node created: type={birth_params.get('type')}, energy={birth_params.get('energy')}")
            else:
                logging.warning("Failed to add new node to graph")
        else:
            logging.warning("Failed to create new node")
        
        return graph
            
    except Exception as e:
        logging.error(f"Error handling node birth: {e}")
        return graph


def remove_node_from_graph(graph, node_id):
    """Remove a node from the graph and update all references."""
    import logging
    
    try:
        if not hasattr(graph, 'node_labels') or node_id >= len(graph.node_labels):
            return False
        
        # Remove node from node_labels
        removed_node = graph.node_labels.pop(node_id)
        
        # Update node features (remove corresponding row from x tensor)
        if hasattr(graph, 'x') and graph.x is not None:
            # Create new x tensor without the removed node
            new_x = torch.cat([graph.x[:node_id], graph.x[node_id+1:]], dim=0)
            graph.x = new_x
        
        # Update edge_index to account for removed node
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            # Remove edges connected to this node
            edge_index = graph.edge_index
            # Keep edges where neither source nor target is the removed node
            keep_edges = (edge_index[0] != node_id) & (edge_index[1] != node_id)
            new_edge_index = edge_index[:, keep_edges]
            
            # Adjust indices for nodes that come after the removed node
            new_edge_index[0] = torch.where(new_edge_index[0] > node_id, 
                                          new_edge_index[0] - 1, new_edge_index[0])
            new_edge_index[1] = torch.where(new_edge_index[1] > node_id, 
                                          new_edge_index[1] - 1, new_edge_index[1])
            
            graph.edge_index = new_edge_index
        
        # Update node IDs in remaining labels
        for new_idx, label in enumerate(graph.node_labels):
            label["id"] = new_idx
        
        logging.info(f"Node {node_id} removed from graph")
        return True
        
    except Exception as e:
        logging.error(f"Error removing node {node_id} from graph: {e}")
        return False


def create_new_node(graph, birth_params):
    """Create a new node with the specified parameters."""
    import logging
    
    try:
        # Generate new node ID
        new_node_id = len(graph.node_labels)
        
        # Create node label
        node_label = {
            'id': new_node_id,
            'type': birth_params.get('type', 'dynamic'),
            'behavior': birth_params.get('behavior', 'dynamic'),
            'energy': birth_params.get('energy', 0.5),
            'state': birth_params.get('state', 'active'),
            'membrane_potential': birth_params.get('energy', 0.5) / 255.0,
            'threshold': birth_params.get('threshold', 0.3),
            'refractory_timer': 0.0,
            'last_activation': 0,
            'plasticity_enabled': True,
            'eligibility_trace': 0.0,
            'last_update': 0
        }
        
        # Add behavior-specific parameters
        if birth_params.get('type') == 'oscillator':
            node_label['oscillation_freq'] = birth_params.get('oscillation_freq', 1.0)
        elif birth_params.get('type') == 'integrator':
            node_label['integration_rate'] = birth_params.get('integration_rate', 0.1)
        elif birth_params.get('type') == 'relay':
            node_label['relay_amplification'] = birth_params.get('relay_amplification', 1.5)
        
        return node_label
        
    except Exception as e:
        logging.error(f"Error creating new node: {e}")
        return None


def add_node_to_graph(graph, new_node):
    """Add a new node to the graph."""
    import logging
    
    try:
        # Add node label
        graph.node_labels.append(new_node)
        
        # Add node features
        if hasattr(graph, 'x') and graph.x is not None:
            # Create new feature tensor for the node
            new_features = torch.tensor([[new_node['energy']]], dtype=graph.x.dtype)
            graph.x = torch.cat([graph.x, new_features], dim=0)
        else:
            # Create initial x tensor if it doesn't exist
            graph.x = torch.tensor([[new_node['energy']]], dtype=torch.float32)
        
        logging.info(f"Node {new_node['id']} added to graph")
        return True
        
    except Exception as e:
        logging.error(f"Error adding node to graph: {e}")
        return False


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
    
    import logging
    logging.info(f"[BIRTH] Checking for node birth with threshold {NODE_BIRTH_THRESHOLD}")
    x = graph.x
    node_labels = graph.node_labels
    num_nodes = len(node_labels)
    dynamic_indices = [
        i for i, lbl in enumerate(node_labels) if lbl.get("type") == "dynamic"
    ]
    
    # Log energy levels of first few dynamic nodes
    if dynamic_indices:
        sample_energies = [x[idx].item() for idx in dynamic_indices[:5]]
        logging.info(f"[BIRTH] Sample dynamic node energies: {sample_energies}")
    
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


def analyze_memory_patterns_for_birth(graph):
    """
    Analyze memory patterns to determine optimal node creation parameters.
    
    Args:
        graph: PyTorch Geometric graph with memory system
    
    Returns:
        dict: Birth parameters influenced by memory patterns
    """
    import random
    
    # Default parameters
    birth_params = {
        'type': 'dynamic',
        'behavior': 'dynamic',
        'energy': 0.5,
        'state': 'active'
    }
    
    # If memory system is available, analyze patterns
    if hasattr(graph, 'memory_system') and graph.memory_system:
        memory_system = graph.memory_system
        
        # Get memory statistics
        memory_stats = memory_system.get_memory_statistics()
        
        # If we have many memory traces, create nodes that can help with consolidation
        if memory_stats.get('traces_formed', 0) > 10:
            # Create integrator nodes to help with memory consolidation
            birth_params['behavior'] = 'integrator'
            birth_params['energy'] = 0.7  # Higher energy for important nodes
            birth_params['integration_rate'] = 0.8
            
        # If we have few memory traces, create nodes that can help with formation
        elif memory_stats.get('traces_formed', 0) < 5:
            # Create relay nodes to help with memory formation
            birth_params['behavior'] = 'relay'
            birth_params['energy'] = 0.6
            birth_params['relay_amplification'] = 1.8
            
        # If memory system is very active, create highway nodes for efficiency
        elif memory_stats.get('total_consolidations', 0) > 20:
            birth_params['behavior'] = 'highway'
            birth_params['energy'] = 0.8
            birth_params['highway_energy_boost'] = 2.5
    
    # Add some randomness to prevent uniform node creation
    if random.random() < 0.3:  # 30% chance of special behavior
        behaviors = ['oscillator', 'integrator', 'relay', 'highway']
        birth_params['behavior'] = random.choice(behaviors)
        
        # Add behavior-specific parameters
        if birth_params['behavior'] == 'oscillator':
            birth_params['oscillation_freq'] = random.uniform(0.5, 2.0)
        elif birth_params['behavior'] == 'integrator':
            birth_params['integration_rate'] = random.uniform(0.3, 0.9)
        elif birth_params['behavior'] == 'relay':
            birth_params['relay_amplification'] = random.uniform(1.2, 2.0)
        elif birth_params['behavior'] == 'highway':
            birth_params['highway_energy_boost'] = random.uniform(1.8, 3.0)
    
    return birth_params


# AI/Human: Extend this file with actual logic for node death/birth as the system evolves.
