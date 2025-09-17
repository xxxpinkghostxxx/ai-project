import torch
import numpy as np


MAX_DYNAMIC_ENERGY = 1.0


def add_dynamic_nodes(graph, num_dynamic=None):

    from node_id_manager import get_id_manager
    id_manager = get_id_manager()
    num_sensory = len(
        [lbl for lbl in graph.node_labels if lbl.get("type", "sensory") == "sensory"]
    )
    if num_dynamic is None:
        num_dynamic = 4 * num_sensory
    dynamic_energies = np.random.uniform(
        0, MAX_DYNAMIC_ENERGY, size=(num_dynamic, 1)
    ).astype(np.float32)
    dynamic_features = torch.tensor(dynamic_energies, dtype=torch.float32)
    graph.x = torch.cat([graph.x, dynamic_features], dim=0)
    for i in range(num_dynamic):
        node_id = id_manager.generate_unique_id("dynamic")
        energy = float(dynamic_energies[i, 0])
        membrane_potential = min(energy / MAX_DYNAMIC_ENERGY, 1.0)
        initial_state = "active" if energy > 0.3 else "inactive"
        node_index = len(graph.node_labels)
        graph.node_labels.append({
            "id": node_id,
            "type": "dynamic",
            "behavior": "dynamic",
            "energy": energy,
            "state": initial_state,
            "membrane_potential": membrane_potential,
            "threshold": 0.3,
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
    return graph
