
import torch
import random

import logging
import threading

from utils.logging_utils import log_step
from energy_constants import ConnectionConstants
from utils.common_utils import safe_hasattr
from energy_behavior import get_node_energy_cap


def get_max_dynamic_energy():
    return get_node_energy_cap()


def get_dynamic_energy_threshold():
    return ConnectionConstants.DYNAMIC_ENERGY_THRESHOLD_FRACTION * get_max_dynamic_energy()
# Use constants directly from ConnectionConstants class to reduce global scope


class EnhancedEdge:

    def __init__(self, source, target, weight=1.0, edge_type='excitatory'):
        self.source = source
        self.target = target
        self.weight = weight
        self.type = edge_type
        self.delay = ConnectionConstants.DEFAULT_EDGE_DELAY
        self.plasticity_tag = False
        self.eligibility_trace = 0.0
        self.last_activity = 0.0
        self.strength_history = []
        self.creation_time = 0
        self.activation_count = 0
    def update_eligibility_trace(self, delta_eligibility):
        self.eligibility_trace += delta_eligibility
        self.eligibility_trace *= ConnectionConstants.ELIGIBILITY_TRACE_DECAY
    def record_activation(self, timestamp):
        self.last_activity = timestamp
        self.activation_count += 1
    def get_effective_weight(self):
        if self.type == 'inhibitory':
            return -abs(self.weight)
        elif self.type == 'modulatory':
            return self.weight * ConnectionConstants.MODULATORY_WEIGHT
        elif self.type == 'burst':
            return abs(self.weight) * 1.5
        elif self.type == 'plastic':
            return abs(self.weight) * (1.0 + self.eligibility_trace)
        else:
            return abs(self.weight)
    def to_dict(self):
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'type': self.type,
            'delay': self.delay,
            'plasticity_tag': self.plasticity_tag,
            'eligibility_trace': self.eligibility_trace,
            'last_activity': self.last_activity,
            'activation_count': self.activation_count,
            'creation_time': self.creation_time
        }


def create_weighted_connection(graph, source_id, target_id, weight, edge_type='excitatory'):

    try:
        from energy.node_id_manager import get_id_manager
        id_manager = get_id_manager()
        source_index = id_manager.get_node_index(source_id)
        target_index = id_manager.get_node_index(target_id)
        if source_index is None or target_index is None:
            if not hasattr(create_weighted_connection, '_warning_count'):
                create_weighted_connection._warning_count = 0
            create_weighted_connection._warning_count += 1
            if create_weighted_connection._warning_count <= 5:
                from energy.node_id_manager import get_id_manager
                id_mgr = get_id_manager()
                stats = id_mgr.get_statistics()
                logging.warning(f"Invalid node IDs for connection: source={source_id}, target={target_id}")
                logging.warning(f"ID Manager stats: {stats}")
            elif create_weighted_connection._warning_count == 6:
                logging.warning("Suppressing further connection validation warnings to prevent spam")
            return graph
        
        # Validate indices are within bounds
        if source_index >= len(graph.node_labels) or target_index >= len(graph.node_labels):
            logging.error(f"Node indices out of bounds: source_index={source_index}, target_index={target_index}, graph_size={len(graph.node_labels)}")
            return graph
        edge = EnhancedEdge(source_id, target_id, weight, edge_type)
        new_edge = torch.tensor([[source_index], [target_index]], dtype=torch.long)
        if graph.edge_index.numel() == 0:
            graph.edge_index = new_edge
        else:
            graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
        if not hasattr(graph, 'edge_attributes'):
            graph.edge_attributes = []
        if not hasattr(graph, '_edge_attributes_lock'):
            graph._edge_attributes_lock = threading.RLock()
        with graph._edge_attributes_lock:
            graph.edge_attributes.append(edge)
        return graph
    except Exception as e:
        logging.error(f"Error creating connection between {source_id} and {target_id}: {e}")
        return graph


def get_edge_attributes(graph, edge_idx):

    if safe_hasattr(graph, 'edge_attributes', '_edge_attributes_lock'):
        with graph._edge_attributes_lock:
            if edge_idx < len(graph.edge_attributes):
                return graph.edge_attributes[edge_idx]
    return None


def apply_weight_change(graph, edge_idx, weight_change):

    try:
        if not hasattr(graph, 'edge_attributes'):
            log_step("No edge attributes found in graph")
            return graph
        if not isinstance(edge_idx, int) or edge_idx < 0:
            log_step("Invalid edge index", edge_idx=edge_idx)
            return graph
        if not isinstance(weight_change, (int, float)):
            log_step("Invalid weight change type", weight_change_type=type(weight_change))
            return graph
        if hasattr(graph, '_edge_attributes_lock'):
            with graph._edge_attributes_lock:
                if edge_idx < len(graph.edge_attributes):
                    edge = graph.edge_attributes[edge_idx]
                    old_weight = edge.weight
                    new_weight = old_weight + weight_change
                    new_weight = max(ConnectionConstants.WEIGHT_MIN, new_weight)
                    new_weight = min(ConnectionConstants.WEIGHT_CAP_MAX, new_weight)
                    if abs(new_weight - old_weight) > 1e-6:
                        edge.weight = new_weight
                        edge.strength_history.append(new_weight)
                        if len(edge.strength_history) > 100:
                            edge.strength_history = edge.strength_history[-100:]
                        edge.activation_count += 1
                        if abs(weight_change) > 0.1:
                            logging.debug(f"Edge {edge_idx} weight changed from {old_weight:.3f} to {new_weight:.3f}")
        else:
            if edge_idx < len(graph.edge_attributes):
                edge = graph.edge_attributes[edge_idx]
                old_weight = edge.weight
                new_weight = old_weight + weight_change
                new_weight = max(ConnectionConstants.WEIGHT_MIN, new_weight)
                new_weight = min(ConnectionConstants.WEIGHT_CAP_MAX, new_weight)
                if abs(new_weight - old_weight) > 1e-6:
                    edge.weight = new_weight
                    edge.strength_history.append(new_weight)
                    if len(edge.strength_history) > 100:
                        edge.strength_history = edge.strength_history[-100:]
                    edge.activation_count += 1
                    if abs(weight_change) > 0.1:
                        logging.debug(f"Edge {edge_idx} weight changed from {old_weight:.3f} to {new_weight:.3f}")
    except Exception as e:
        log_step("Error applying weight change", error=str(e), edge_idx=edge_idx)
    return graph


def create_basic_connections(graph):

    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        return graph
    import random
    import torch
    from energy.node_id_manager import get_id_manager
    num_nodes = len(graph.node_labels)
    if num_nodes < 2:
        return graph
    id_manager = get_id_manager()
    num_connections = min(100, num_nodes // 10)
    connections_created = 0
    for _ in range(num_connections):
        source_idx = random.randint(0, num_nodes - 1)
        target_idx = random.randint(0, num_nodes - 1)
        if source_idx != target_idx:
            source_node = graph.node_labels[source_idx]
            target_node = graph.node_labels[target_idx]
            source_id = source_node.get('id')
            target_id = target_node.get('id')
            if source_id and target_id is not None:
                weight = random.uniform(0.1, 0.8)
                edge_type = random.choice(['excitatory', 'inhibitory', 'modulatory'])
                create_weighted_connection(graph, source_id, target_id, weight, edge_type)
                connections_created += 1
    return graph


def intelligent_connection_formation(graph):

    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        return graph
    num_nodes = len(graph.node_labels)
    if num_nodes < 2:
        return graph
    if num_nodes > 50000:
        log_step(f"Skipping intelligent connections for large graph ({num_nodes} nodes) to maintain performance")
        return graph
    max_connections = 50
    sensory_sample_size = 10
    dynamic_sample_size = 20
    import random
    all_indices = list(range(num_nodes))
    random.shuffle(all_indices)
    sensory_nodes = []
    dynamic_nodes = []
    for idx in all_indices[:1000]:
        node = graph.node_labels[idx]
        node_type = node.get('type', 'unknown')
        if node_type == 'sensory' and len(sensory_nodes) < sensory_sample_size:
            sensory_nodes.append(idx)
        elif node_type == 'dynamic' and len(dynamic_nodes) < dynamic_sample_size:
            dynamic_nodes.append(idx)
        if len(sensory_nodes) >= sensory_sample_size and len(dynamic_nodes) >= dynamic_sample_size:
            break
    connections_created = 0
    if sensory_nodes and dynamic_nodes and connections_created < max_connections:
        for sensory_idx in sensory_nodes:
            if connections_created >= max_connections:
                break
            sensory_node = graph.node_labels[sensory_idx]
            sensory_id = sensory_node.get('id')
            if sensory_id is None:
                continue
            sensory_energy = graph.x[sensory_idx, 0].item() if hasattr(graph, 'x') else 0.0
            for dynamic_idx in dynamic_nodes[:2]:
                if sensory_idx != dynamic_idx and connections_created < max_connections:
                    dynamic_node = graph.node_labels[dynamic_idx]
                    dynamic_id = dynamic_node.get('id')
                    if dynamic_id is None:
                        continue
                    connection_types = ['excitatory', 'modulatory', 'plastic']
                    connection_type = connection_types[connections_created % len(connection_types)]
                    weight = 0.3 + (connections_created * 0.1)
                    create_weighted_connection(graph, sensory_id, dynamic_id, weight, connection_type)
                    connections_created += 1
    if len(dynamic_nodes) > 1 and connections_created < max_connections:
        limited_dynamic = dynamic_nodes[:3]
        for i, node1_idx in enumerate(limited_dynamic):
            if connections_created >= max_connections:
                break
            for j, node2_idx in enumerate(limited_dynamic[i+1:], i+1):
                if connections_created >= max_connections:
                    break
                if node1_idx != node2_idx:
                    node1 = graph.node_labels[node1_idx]
                    node2 = graph.node_labels[node2_idx]
                    node1_id = node1.get('id')
                    node2_id = node2.get('id')
                    if node1_id is None or node2_id is None:
                        continue
                    cluster1 = node1.get('energy_cluster_id', 0)
                    cluster2 = node2.get('energy_cluster_id', 0)
                    cluster_similarity = abs(cluster1 - cluster2) <= 2
                    affinity1 = node1.get('connection_affinity', 0)
                    affinity2 = node2.get('connection_affinity', 0)
                    affinity_compatible = abs(affinity1 - affinity2) < 30
                    pos1 = node1.get('feature_position', 0)
                    pos2 = node2.get('feature_position', 0)
                    position_proximity = abs(pos1 - pos2) < 20 or abs(pos1 - pos2) > 80
                    if cluster_similarity and affinity_compatible and position_proximity:
                        connection_types = ['excitatory', 'inhibitory', 'plastic', 'burst']
                        connection_type = connection_types[connections_created % len(connection_types)]
                        weight = 0.2 + (connections_created * 0.05)
                        create_weighted_connection(graph, node1_id, node2_id, weight, connection_type)
                        connections_created += 1
    oscillator_nodes = [i for i, node in enumerate(graph.node_labels) if node.get('behavior') == 'oscillator']
    integrator_nodes = [i for i, node in enumerate(graph.node_labels) if node.get('behavior') == 'integrator']
    if oscillator_nodes and integrator_nodes:
        for osc_idx in oscillator_nodes:
            osc_node = graph.node_labels[osc_idx]
            osc_id = osc_node.get('id')
            if osc_id is None:
                continue
            for int_idx in integrator_nodes[:2]:
                if osc_idx != int_idx:
                    int_node = graph.node_labels[int_idx]
                    int_id = int_node.get('id')
                    if int_id is None:
                        continue
                    create_weighted_connection(graph, osc_id, int_id, 0.4, 'excitatory')
    relay_nodes = [i for i, node in enumerate(graph.node_labels) if node.get('behavior') == 'relay']
    highway_nodes = [i for i, node in enumerate(graph.node_labels) if node.get('behavior') == 'highway']
    if relay_nodes and highway_nodes:
        for relay_idx in relay_nodes:
            relay_node = graph.node_labels[relay_idx]
            relay_id = relay_node.get('id')
            if relay_id is None:
                continue
            for highway_idx in highway_nodes[:2]:
                if relay_idx != highway_idx:
                    highway_node = graph.node_labels[highway_idx]
                    highway_id = highway_node.get('id')
                    if highway_id is None:
                        continue
                    create_weighted_connection(graph, relay_id, highway_id, 0.6, 'excitatory')
    import random
    num_random_connections = min(10, num_nodes // 5)
    for _ in range(num_random_connections):
        source_idx = random.randint(0, num_nodes - 1)
        target_idx = random.randint(0, num_nodes - 1)
        if source_idx != target_idx:
            source_node = graph.node_labels[source_idx]
            target_node = graph.node_labels[target_idx]
            source_id = source_node.get('id')
            target_id = target_node.get('id')
            if source_id and target_id is not None:
                create_weighted_connection(graph, source_id, target_id, 0.2, 'excitatory')
    return graph


def update_connection_weights(graph, learning_rate=ConnectionConstants.LEARNING_RATE_DEFAULT):

    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return graph
    for edge_idx, edge in enumerate(graph.edge_attributes):
        source = edge.source
        target = edge.target
        source_activity = graph.node_labels[source].get('last_activation', 0)
        target_activity = graph.node_labels[target].get('last_activation', 0)
        if source_activity > 0 and target_activity > 0:
            weight_change = learning_rate * (source_activity + target_activity) / 2
            edge.weight = min(edge.weight + weight_change, ConnectionConstants.WEIGHT_CAP_MAX)
            edge.eligibility_trace += weight_change * ConnectionConstants.WEIGHT_CHANGE_FACTOR
        elif source_activity > 0 and target_activity == 0:
            weight_change = -learning_rate * ConnectionConstants.WEIGHT_CHANGE_FACTOR
            edge.weight = max(edge.weight + weight_change, ConnectionConstants.WEIGHT_MIN)
        edge.update_eligibility_trace(0)
    return graph


def add_dynamic_connections(graph):

    return intelligent_connection_formation(graph)


def add_connections(graph, connection_strategy=None):

    if connection_strategy is None:
        connection_strategy = intelligent_connection_formation
    return connection_strategy(graph)
