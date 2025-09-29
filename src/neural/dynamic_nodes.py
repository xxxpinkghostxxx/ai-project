import torch

from src.energy.energy_behavior import get_node_energy_cap
from src.utils.logging_utils import log_step


def add_dynamic_nodes(graph, num_dynamic=None, id_manager=None):
    """
    Add dynamic nodes to the neural graph.

    Args:
        graph: Neural graph to add nodes to
        num_dynamic: Number of dynamic nodes to add (optional)
        id_manager: Node ID manager (optional, will use global if not provided)

    Returns:
        Updated graph with dynamic nodes added
    """
    if id_manager is None:
        from src.energy.node_id_manager import get_id_manager
        id_manager = get_id_manager()
    num_sensory = len(
        [lbl for lbl in graph.node_labels if lbl.get("type", "sensory") == "sensory"]
    )
    if num_dynamic is None:
        num_dynamic = max(5, 4 * num_sensory)  # Ensure at least 5 initial dynamic nodes
    energy_cap = get_node_energy_cap()
    initial_energy = min(100.0, energy_cap)
    dynamic_energies = torch.full((num_dynamic, 1), initial_energy, dtype=torch.float32)
    graph.x = torch.cat([graph.x, dynamic_energies], dim=0)
    for i in range(num_dynamic):
        node_id = id_manager.generate_unique_id("dynamic")
        energy = float(initial_energy)
        membrane_potential = min(energy / energy_cap if energy_cap > 0 else 0.0, 1.0)
        initial_state = "active"
        node_index = len(graph.node_labels)
        graph.node_labels.append({
            "id": node_id,
            "type": "dynamic",
            "behavior": "dynamic",
            "energy": energy,
            "state": initial_state,
            "membrane_potential": membrane_potential,
            "threshold": 0.5,
            "refractory_timer": 0.0,
            "last_activation": 0,
            "plasticity_enabled": True,
            "eligibility_trace": 0.0,
            "last_update": 0,
            "is_excitatory": True,
            "I_syn": 0.0,
            "IEG_flag": False,
            "plast_enabled": True,
            "theta_burst_counter": 0,
            "v_dend": 0.0,
            "feature_position": node_id % 100,
            "feature_rank": (node_id * 7) % 1000,
            "energy_cluster_id": (node_id // 10) % 10,
            "connection_affinity": (node_id * 13) % 100,
            "neighborhood_radius": 5 + (node_id % 10)
        })
        id_manager.register_node_index(node_id, node_index)
    assert len(graph.node_labels) == graph.x.shape[0], "Node label and feature count mismatch after dynamic node addition"
    # Ensure edge_index is set if missing
    if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
        graph.edge_index = torch.empty((2, 0), dtype=torch.long)

    log_step(f"Added {num_dynamic} dynamic nodes with energy {initial_energy}")
    return graph







