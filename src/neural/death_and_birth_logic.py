import torch
from typing import Dict, Any, Optional, Union
from torch_geometric.data import Data
from config.unified_config_manager import get_system_constants
import logging
from src.energy.energy_behavior import get_node_energy_cap
from src.energy.node_id_manager import get_id_manager


def get_max_dynamic_energy() -> float:
    return get_node_energy_cap()


def get_dynamic_birth_threshold() -> float:
    return 0.9 * get_max_dynamic_energy()


def get_new_node_energy_fraction() -> float:
    return 0.4


def get_node_birth_threshold() -> float:
    from config.unified_config_manager import get_config
    return get_config('NodeLifecycle', 'birth_threshold', 0.8, float)


def get_node_birth_cost() -> float:
    return 0.3


def get_node_death_threshold() -> float:
    from config.unified_config_manager import get_config
    return get_config('NodeLifecycle', 'death_threshold', 5.0, float)


def get_energy_cap_255() -> float:
    constants = get_system_constants()
    return constants.get('energy_cap_255', 255.0)


def handle_node_death(graph: Data, node_id: int, strategy: Optional[Union[str, callable]] = None) -> Data:

    try:
        if not hasattr(graph, 'node_labels') or node_id >= len(graph.node_labels):
            logging.warning(f"Invalid node_id {node_id} for death handling")
            return graph
        node = graph.node_labels[node_id]
        node_type = node.get('type', 'unknown')
        node_energy = graph.x[node_id].item() if hasattr(graph, 'x') else 0.0
        should_remove = False
        memory_importance = 0.0
        if hasattr(graph, 'memory_system') and graph.memory_system:
            memory_importance = graph.memory_system.get_node_memory_importance(node_id)
        if strategy == 'conservative':
            should_remove = (node_energy < 0.1 and
                           node.get('state', 'active') == 'inactive' and
                           memory_importance < 0.2)
        elif strategy == 'aggressive':
            should_remove = ((node_energy < 0.5 or
                            node.get('state', 'active') == 'inactive') and
                           memory_importance < 0.5)
        elif strategy == 'memory_aware':
            if memory_importance > 0.7:
                should_remove = False
            else:
                should_remove = (node_energy <= get_node_death_threshold() and
                               memory_importance < 0.3)
        elif callable(strategy):
            should_remove = strategy(node, graph, node_id)
        else:
            should_remove = (node_energy <= get_node_death_threshold() and
                           memory_importance < 0.4)
        if should_remove:
            success = remove_node_from_graph(graph, node_id)
            if success:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Node {node_id} ({node_type}) removed successfully (energy: {node_energy:.2f})")
            else:
                logging.warning(f"Failed to remove node {node_id}")
        else:
            logging.debug(f"Node {node_id} not removed (strategy: {strategy}, energy: {node_energy:.2f})")
        return graph
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        logging.error(f"Error handling node death for node {node_id}: {e}")
        return graph
    except Exception as e:
        logging.error(f"Unexpected error handling node death for node {node_id}: {e}")
        return graph


def handle_node_birth(graph: Data, birth_params: Optional[Dict[str, Any]] = None) -> Data:

    try:
        if birth_params is None:
            memory_influenced_params = analyze_memory_patterns_for_birth(graph)
            birth_params = memory_influenced_params
        new_node = create_new_node(graph, birth_params)
        if new_node:
            success = add_node_to_graph(graph, new_node)
            if success:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"New node created: type={birth_params.get('type')}, energy={birth_params.get('energy')}")
            else:
                logging.warning("Failed to add new node to graph")
        else:
            logging.warning("Failed to create new node")
        return graph
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        logging.error(f"Error handling node birth: {e}")
        return graph
    except Exception as e:
        logging.error(f"Unexpected error handling node birth: {e}")
        return graph


def remove_node_from_graph(graph: Data, node_id: int) -> bool:
    try:
        if not hasattr(graph, 'node_labels') or node_id >= len(graph.node_labels):
            return False
        removed_node = graph.node_labels.pop(node_id)
        if hasattr(graph, 'x') and graph.x is not None:
            try:
                if node_id < 0 or node_id >= graph.x.shape[0]:
                    logging.error(f"Invalid node_id {node_id} for tensor of shape {graph.x.shape}")
                    return False
                if node_id + 1 > graph.x.shape[0]:
                    new_x = graph.x[:node_id]
                else:
                    new_x = torch.cat([graph.x[:node_id], graph.x[node_id+1:]], dim=0)
                graph.x = new_x
            except (RuntimeError, IndexError, OverflowError) as e:
                logging.error(f"Tensor slicing failed for node {node_id}: {e}")
                return False
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            edge_index = graph.edge_index
            keep_edges = (edge_index[0] != node_id) & (edge_index[1] != node_id)
            new_edge_index = edge_index[:, keep_edges]
            new_edge_index[0] = torch.where(new_edge_index[0] > node_id,
                                          new_edge_index[0] - 1, new_edge_index[0])
            new_edge_index[1] = torch.where(new_edge_index[1] > node_id,
                                          new_edge_index[1] - 1, new_edge_index[1])
            graph.edge_index = new_edge_index.contiguous()

        # Handle edge_attributes remapping and cleanup
        if hasattr(graph, 'edge_attributes') and graph.edge_attributes:
            import threading
            if not hasattr(graph, '_edge_attributes_lock'):
                graph._edge_attributes_lock = threading.Lock()
            with graph._edge_attributes_lock:
                # Filter out edges connected to removed node
                valid_edges = [e for e in graph.edge_attributes if e.source != node_id and e.target != node_id]
                # Remap surviving edges' indices
                for edge in valid_edges:
                    if edge.source > node_id:
                        edge.source -= 1
                    if edge.target > node_id:
                        edge.target -= 1
                graph.edge_attributes = valid_edges
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"[DEATH] Single removal: remapped {len(valid_edges)} edge attributes")
        # Update ID manager to reflect the new indices
        id_manager = get_id_manager()
        # Full remap: clear old mappings and re-register all
        id_manager._id_to_index.clear()  # Direct access for fix; in production use method
        id_manager._index_to_id.clear()
        for new_idx, label in enumerate(graph.node_labels):
            node_id = label.get("id")
            if node_id is not None:
                id_manager.register_node_index(node_id, new_idx)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[DEATH] Node {node_id} removed and IDs remapped (total nodes: {len(graph.node_labels)})")
        return True
    except (ValueError, TypeError, AttributeError, IndexError, RuntimeError) as e:
        logging.error(f"Error removing node {node_id} from graph: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error removing node {node_id} from graph: {e}")
        return False


def create_new_node(graph: Data, birth_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        id_manager = get_id_manager()
        
        # Check if we can expand the graph
        if not id_manager.can_expand_graph(1):
            logging.warning(f"Cannot create new node: graph expansion limit reached. Capacity: {id_manager.get_expansion_capacity()}")
            return None
            
        new_node_id = id_manager.generate_unique_id(birth_params.get('type', 'dynamic'))
        new_node_index = len(graph.node_labels)
        id_manager.register_node_index(new_node_id, new_node_index)
        node_label = {
            'id': new_node_id,
            'type': birth_params.get('type', 'dynamic'),
            'behavior': birth_params.get('behavior', 'dynamic'),
            'energy': birth_params.get('energy', 0.5),
            'state': birth_params.get('state', 'active'),
            'membrane_potential': birth_params.get('energy', 0.5) / get_energy_cap_255(),
            'threshold': birth_params.get('threshold', 0.3),
            'refractory_timer': 0.0,
            'last_activation': 0,
            'plasticity_enabled': True,
            'eligibility_trace': 0.0,
            'last_update': 0
        }
        if birth_params.get('type') == 'oscillator':
            node_label['oscillation_freq'] = birth_params.get('oscillation_freq', 1.0)
        elif birth_params.get('type') == 'integrator':
            node_label['integration_rate'] = birth_params.get('integration_rate', 0.1)
        elif birth_params.get('type') == 'relay':
            node_label['relay_amplification'] = birth_params.get('relay_amplification', 1.5)
        return node_label
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logging.error(f"Error creating new node: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error creating new node: {e}")
        return None


def add_node_to_graph(graph: Data, new_node: Dict[str, Any]) -> bool:
    try:
        graph.node_labels.append(new_node)
        if hasattr(graph, 'x') and graph.x is not None:
            new_features = torch.tensor([[new_node['energy']]], dtype=graph.x.dtype)
            graph.x = torch.cat([graph.x, new_features], dim=0)
        else:
            graph.x = torch.tensor([[new_node['energy']]], dtype=torch.float32)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Node {new_node['id']} added to graph")
        return True
    except (ValueError, TypeError, AttributeError, RuntimeError) as e:
        logging.error(f"Error adding node to graph: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error adding node to graph: {e}")
        return False


def remove_dead_dynamic_nodes(graph: Data) -> Data:
 
    if (
        not hasattr(graph, "node_labels")
        or not hasattr(graph, "x")
        or not hasattr(graph, "edge_index")
    ):
        logging.debug("[DEATH] Graph missing required attributes, skipping removal")
        return graph
    
    threshold = get_node_death_threshold()
    edge_index = graph.edge_index if hasattr(graph, 'edge_index') else torch.empty((2, 0), dtype=torch.long)
    dynamic_nodes = []
    for idx, label in enumerate(graph.node_labels):
        if label.get("type") == "dynamic":
            energy = graph.x[idx].item() if hasattr(graph, 'x') and graph.x is not None and idx < graph.x.shape[0] else 0.0
            # Compute degree: number of connections
            degree = torch.sum((edge_index[0] == idx) | (edge_index[1] == idx)).item()
            label['degree'] = degree  # Cache for future
            dynamic_nodes.append((idx, energy, degree))
    
    num_dynamic = len(dynamic_nodes)
    if num_dynamic > 0:
        energies = [e for _, e, _ in dynamic_nodes]
        min_energy = min(energies)
        max_energy = max(energies)
        degrees = [d for _, _, d in dynamic_nodes]
        min_degree = min(degrees)
        max_degree = max(degrees)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[DEATH] Checked {num_dynamic} dynamic nodes: min_energy={min_energy:.2f}, max_energy={max_energy:.2f}, min_degree={min_degree}, max_degree={max_degree}, threshold={threshold}")
    else:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[DEATH] No dynamic nodes to check")
    
    min_connections = 3  # Isolation threshold
    to_remove = [
        idx for idx, energy, degree in dynamic_nodes
        if (energy < threshold) or (degree < min_connections)
    ]
    num_low_energy = sum(1 for _, e, _ in dynamic_nodes if e < threshold)
    num_isolated = sum(1 for _, _, d in dynamic_nodes if d < min_connections)
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"[DEATH] Found {len(to_remove)} candidates for removal (low_energy={num_low_energy}, isolated={num_isolated}, threshold={threshold}, min_conn={min_connections})")
    
    if not to_remove:
        logging.debug("[DEATH] No nodes meet removal criteria")
        return graph
    
    for idx, energy, degree in [(idx, graph.x[idx].item(), graph.node_labels[idx].get('degree', 0)) for idx in to_remove]:
        reason = 'low_energy' if energy < threshold else 'isolation'
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[DEATH] Node {idx} removed ({reason}: energy={energy:.2f}, degree={degree})")
    
    to_remove_set = set(to_remove)
    keep_indices = [i for i in range(len(graph.node_labels)) if i not in to_remove_set]
    if hasattr(graph, 'x') and graph.x is not None:
        graph.x = graph.x[keep_indices]
    graph.node_labels = [graph.node_labels[i] for i in keep_indices]

    # Proper remapping for edge_index and edge_attributes in batch removal
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        edge_index = graph.edge_index
        # First, filter edges connected to removed nodes
        remove_mask = (
            torch.isin(edge_index[0], torch.tensor(list(to_remove_set)))
            | torch.isin(edge_index[1], torch.tensor(list(to_remove_set)))
        )
        surviving_edges = ~remove_mask
        filtered_edge_index = edge_index[:, surviving_edges]

        # Create mapping from old surviving index to new index
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

        # Remap source and target indices
        remapped_sources = torch.tensor([old_to_new.get(src.item(), -1) for src in filtered_edge_index[0]])
        remapped_targets = torch.tensor([old_to_new.get(tgt.item(), -1) for tgt in filtered_edge_index[1]])
        valid_remap_mask = (remapped_sources >= 0) & (remapped_targets >= 0)
        graph.edge_index = torch.stack([remapped_sources[valid_remap_mask], remapped_targets[valid_remap_mask]])

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[DEATH] Batch removal: remapped {graph.edge_index.shape[1]} edges from {filtered_edge_index.shape[1]} surviving")

    if hasattr(graph, 'edge_attributes') and graph.edge_attributes:
        import threading
        if not hasattr(graph, '_edge_attributes_lock'):
            graph._edge_attributes_lock = threading.Lock()
        with graph._edge_attributes_lock:
            before_len = len(graph.edge_attributes)
            # Filter edges connected to removed nodes
            valid_edges = [e for e in graph.edge_attributes if e.source not in to_remove_set and e.target not in to_remove_set]
            # Remap surviving edges using old_to_new
            for edge in valid_edges:
                if edge.source in old_to_new:
                    edge.source = old_to_new[edge.source]
                if edge.target in old_to_new:
                    edge.target = old_to_new[edge.target]
            graph.edge_attributes = valid_edges
            after_len = len(graph.edge_attributes)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"[DEATH] Batch removal: cleaned edge_attributes from {before_len} to {after_len}, remapped {len(valid_edges)}")
    else:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[DEATH] No edge_attributes for batch cleanup")
    
    id_manager = get_id_manager()
    # Full remap after death
    id_manager._id_to_index.clear()
    id_manager._index_to_id.clear()
    for new_idx, label in enumerate(graph.node_labels):
        node_id = label.get("id")
        if node_id is not None:
            id_manager.register_node_index(node_id, new_idx)
    
    # Additional diagnostic: Log final edge count and any potential orphans
    final_edge_count = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0
    attr_count = len(graph.edge_attributes) if hasattr(graph, 'edge_attributes') else 0
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"[DEATH] Final graph state: nodes={len(graph.node_labels)}, edges={final_edge_count}, attributes={attr_count}")
    if attr_count != final_edge_count:
        logging.warning(f"[DEATH] Mismatch between edge_index ({final_edge_count}) and edge_attributes ({attr_count})")

    final_nodes = len(graph.node_labels)
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"[DEATH] Removal complete: {final_nodes} nodes remaining (removed {len(to_remove)})")
    
    if hasattr(graph, 'x') and graph.x is not None:
        assert len(graph.node_labels) == graph.x.shape[0], "Node label and feature count mismatch after node death"
    if hasattr(graph, 'edge_index') and graph.edge_index is not None and graph.edge_index.numel() > 0:
        assert torch.all(graph.edge_index < len(graph.node_labels)), "Edge index out of bounds after node death"
    return graph


def birth_new_dynamic_nodes(graph: Data) -> Data:

    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        return graph
    id_manager = get_id_manager()
    
    # Check if we can expand the graph
    if not id_manager.can_expand_graph(1):
        logging.warning(f"[BIRTH] Graph expansion limit reached. Cannot create new nodes. Capacity: {id_manager.get_expansion_capacity()}")
        return graph

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"[BIRTH] Checking for node birth with threshold {get_node_birth_threshold()}")
    x = graph.x
    node_labels = graph.node_labels
    num_nodes = len(node_labels)
    dynamic_indices = []
    for i, lbl in enumerate(node_labels):
        if lbl.get("type") == "dynamic":
            dynamic_indices.append(i)
    if not dynamic_indices:
        return graph
    dynamic_energies = x[dynamic_indices, 0]
    birth_threshold = get_node_birth_threshold()
    birth_candidates = dynamic_energies > birth_threshold
    if dynamic_indices:
        sample_energies = dynamic_energies[:5].tolist()
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[BIRTH] Sample dynamic node energies: {sample_energies}")
    new_features = []
    new_labels = []
    birth_cost = get_node_birth_cost()
    energy_fraction = get_new_node_energy_fraction()
    energy_cap = get_node_energy_cap()
    spawned_from_energy = False
    for i, idx in enumerate(dynamic_indices):
        if birth_candidates[i]:
            energy = dynamic_energies[i].item()
            new_parent_energy = max(energy - birth_cost, 0)
            x[idx] = min(new_parent_energy, energy_cap)
            new_energy = min(energy * energy_fraction, energy_cap)
            new_features.append([new_energy])
            id_manager = get_id_manager()
            new_node_id = id_manager.generate_unique_id("dynamic")
            new_node_index = num_nodes + len(new_labels)
            new_labels.append({
                "id": new_node_id,
                "type": "dynamic",
                "energy": new_energy,
                "behavior": "dynamic",
                "state": "active",
                "last_update": 0,
                "feature_position": new_node_id % 100,
                "feature_rank": (new_node_id * 7) % 1000,
                "energy_cluster_id": (new_node_id // 10) % 10,
                "connection_affinity": (new_node_id * 13) % 100,
                "neighborhood_radius": 5 + (new_node_id % 10),
                "membrane_potential": min(new_energy / get_node_energy_cap(), 1.0),
                "threshold": 0.3,
                "refractory_timer": 0.0,
                "last_activation": 0,
                "plasticity_enabled": True,
                "eligibility_trace": 0.0,
                "is_excitatory": True,
                "I_syn": 0.0,
                "IEG_flag": False,
                "plast_enabled": True,
                "theta_burst_counter": 0,
                "v_dend": 0.0
            })
            id_manager.register_node_index(new_node_id, new_node_index)
            logging.info(
                f"[BIRTH] Node {idx} spawned new node {new_node_id} with energy {new_energy:.2f} (parent energy now {x[idx].item():.2f})"
            )
            spawned_from_energy = True
            
            # Diagnostic: Log new node ID for connection formation tracking
            logging.info(f"[BIRTH] New node ID registered: {new_node_id} at index {new_node_index}")
    # Density-based spawn if low node count and no energy-based spawns
    if not spawned_from_energy and num_nodes < 1000:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[BIRTH] Low density (nodes={num_nodes}), forcing one spawn")
        default_energy = 0.5 * energy_cap
        new_features.append([default_energy])
        id_manager = get_id_manager()
        new_node_id = id_manager.generate_unique_id("dynamic")
        new_node_index = num_nodes + len(new_labels)
        new_labels.append({
            "id": new_node_id,
            "type": "dynamic",
            "energy": default_energy,
            "behavior": "dynamic",
            "state": "active",
            "last_update": 0,
            "feature_position": new_node_id % 100,
            "feature_rank": (new_node_id * 7) % 1000,
            "energy_cluster_id": (new_node_id // 10) % 10,
            "connection_affinity": (new_node_id * 13) % 100,
            "neighborhood_radius": 5 + (new_node_id % 10),
            "membrane_potential": min(default_energy / get_node_energy_cap(), 1.0),
            "threshold": 0.3,
            "refractory_timer": 0.0,
            "last_activation": 0,
            "plasticity_enabled": True,
            "eligibility_trace": 0.0,
            "is_excitatory": True,
            "I_syn": 0.0,
            "IEG_flag": False,
            "plast_enabled": True,
            "theta_burst_counter": 0,
            "v_dend": 0.0
        })
        id_manager.register_node_index(new_node_id, new_node_index)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[BIRTH] Density-spawned node {new_node_id} with energy {default_energy:.2f}")

        # Diagnostic: Log new node ID for connection formation tracking
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[BIRTH] Density new node ID registered: {new_node_id} at index {new_node_index}")
    if new_features:
        new_features_tensor = torch.tensor(new_features, dtype=x.dtype)
        graph.x = torch.cat([x, new_features_tensor], dim=0)
        num_existing = len(graph.node_labels)
        graph.node_labels.extend(new_labels)
        # Connect new nodes to 5 random existing nodes
        from src.neural.connection_logic import create_weighted_connection
        import random
        for new_label in new_labels:
            new_id = new_label['id']
            connections = min(5, num_existing)
            connected = 0
            attempts = 0
            while connected < connections and attempts < num_existing * 2:  # Avoid infinite loop
                target_idx = random.randint(0, num_existing - 1)
                target_label = graph.node_labels[target_idx]
                target_id = target_label.get('id')
                if target_id != new_id:
                    create_weighted_connection(graph, new_id, target_id, weight=0.5, edge_type='excitatory')
                    connected += 1
                attempts += 1
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"[BIRTH] New node {new_id} connected to {connected} existing nodes")
        # Log edge count after birth connections
        edge_count = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[BIRTH] Graph edge count after adding connections for new nodes: {edge_count}")
        assert len(graph.node_labels) == graph.x.shape[0], "Node label and feature count mismatch after node birth"
    return graph


def analyze_memory_patterns_for_birth(graph: Data) -> Dict[str, Any]:

    import random
    birth_params = {
        'type': 'dynamic',
        'behavior': 'dynamic',
        'energy': 0.5,
        'state': 'active'
    }
    if hasattr(graph, 'memory_system') and graph.memory_system:
        memory_system = graph.memory_system
        memory_stats = memory_system.get_memory_statistics()
        if memory_stats.get('traces_formed', 0) > 10:
            birth_params['behavior'] = 'integrator'
            birth_params['energy'] = 0.7
            birth_params['integration_rate'] = 0.8
        elif memory_stats.get('traces_formed', 0) < 5:
            birth_params['behavior'] = 'relay'
            birth_params['energy'] = 0.6
            birth_params['relay_amplification'] = 1.8
        elif memory_stats.get('total_consolidations', 0) > 20:
            birth_params['behavior'] = 'highway'
            birth_params['energy'] = 0.8
            birth_params['highway_energy_boost'] = 2.5
    if random.random() < 0.3:
        behaviors = ['oscillator', 'integrator', 'relay', 'highway']
        birth_params['behavior'] = random.choice(behaviors)
        if birth_params['behavior'] == 'oscillator':
            birth_params['oscillation_freq'] = random.uniform(0.5, 2.0)
        elif birth_params['behavior'] == 'integrator':
            birth_params['integration_rate'] = random.uniform(0.3, 0.9)
        elif birth_params['behavior'] == 'relay':
            birth_params['relay_amplification'] = random.uniform(1.2, 2.0)
        elif birth_params['behavior'] == 'highway':
            birth_params['highway_energy_boost'] = random.uniform(1.8, 3.0)
    return birth_params







