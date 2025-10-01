
"""
This module handles neural connection logic, including edge creation, weight management, and connection formation strategies.
"""

import logging
import random
import threading

import torch

from src.energy.energy_behavior import get_node_energy_cap
from src.energy.energy_constants import ConnectionConstants, EnergyConstants
from src.energy.node_id_manager import get_id_manager
from src.utils.common_utils import safe_hasattr
from src.utils.connection_validator import get_connection_validator
from src.utils.logging_utils import log_step


def get_max_dynamic_energy():
    """
    Get the maximum dynamic energy for nodes.

    Returns:
        float: The maximum dynamic energy capacity.
    """
    return get_node_energy_cap()


def get_dynamic_energy_threshold():
    """
    Get the dynamic energy threshold for nodes.

    Returns:
        float: The dynamic energy threshold value.
    """
    return EnergyConstants.DYNAMIC_ENERGY_THRESHOLD_FRACTION * get_max_dynamic_energy()
# Use constants directly from ConnectionConstants class to reduce global scope


class EnhancedEdge:
    """
    Represents an enhanced edge in the neural graph with additional attributes for learning and plasticity.

    Attributes:
        source (int): Source node index.
        target (int): Target node index.
        weight (float): Connection weight.
        type (str): Edge type (e.g., 'excitatory', 'inhibitory').
        delay (float): Edge delay.
        plasticity_tag (bool): Indicates if the edge is plastic.
        eligibility_trace (float): Eligibility trace for learning.
        last_activity (float): Timestamp of last activity.
        strength_history (list): History of weight changes.
        creation_time (int): Time of edge creation.
        activation_count (int): Number of activations.
    """

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
        """
        Update the eligibility trace for the edge.

        Args:
            delta_eligibility (float): Change in eligibility.
        """
        self.eligibility_trace += delta_eligibility
        self.eligibility_trace *= ConnectionConstants.ELIGIBILITY_TRACE_DECAY
    def record_activation(self, timestamp):
        """
        Record an activation event for the edge.

        Args:
            timestamp (float): Timestamp of activation.
        """
        self.last_activity = timestamp
        self.activation_count += 1
    def get_effective_weight(self):
        """
        Get the effective weight of the edge based on its type.

        Returns:
            float: The effective weight.
        """
        if self.type == 'inhibitory':
            return -abs(self.weight)
        if self.type == 'modulatory':
            return self.weight * ConnectionConstants.MODULATORY_WEIGHT
        if self.type == 'burst':
            return abs(self.weight) * 1.5
        if self.type == 'plastic':
            return abs(self.weight) * (1.0 + self.eligibility_trace)
        return abs(self.weight)
    def to_dict(self):
        """
        Convert the edge to a dictionary representation.

        Returns:
            dict: Dictionary containing edge attributes.
        """
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


def _get_node_indices(graph, source_id, target_id, id_manager):
    """
    Get node indices for source and target IDs, registering if necessary.

    Args:
        graph: Neural graph
        source_id: Source node ID
        target_id: Target node ID
        id_manager: Node ID manager

    Returns:
        tuple: (source_index, target_index) or (None, None) if failed
    """
    source_index = id_manager.get_node_index(source_id)
    target_index = id_manager.get_node_index(target_id)

    # If indices not found, try to register them from graph.node_labels
    if source_index is None:
        for idx, node in enumerate(graph.node_labels):
            if node.get('id') == source_id:
                id_manager.register_node_index(source_id, idx)
                source_index = idx
                break
    if target_index is None:
        for idx, node in enumerate(graph.node_labels):
            if node.get('id') == target_id:
                id_manager.register_node_index(target_id, idx)
                target_index = idx
                break

    # Double-check indices
    if source_index is None or target_index is None:
        logging.error("ID manager returned None indices: source=%s, target=%s", source_id, target_id)
        return None, None

    # Validate indices are within bounds
    if source_index >= len(graph.node_labels) or target_index >= len(graph.node_labels):
        logging.error("Node indices out of bounds: source_index=%s, target_index=%s, graph_size=%s", source_index, target_index, len(graph.node_labels))
        return None, None

    return source_index, target_index


def create_weighted_connection(graph, source_id, target_id, weight, edge_type='excitatory', *, validator=None, id_manager=None):
    """
    Create a weighted connection with centralized validation.

    Args:
        graph: Neural graph
        source_id: Source node ID
        target_id: Target node ID
        weight: Connection weight
        edge_type: Type of connection
        validator: Connection validator (optional, will use global if not provided)
        id_manager: Node ID manager (optional, will use global if not provided)

    Returns:
        Updated graph
    """
    # Get the centralized validator
    if validator is None:
        validator = get_connection_validator()

    # Perform comprehensive validation
    validation_result = validator.validate_connection(
        graph, source_id, target_id, edge_type, weight
    )

    # Log validation results
    if not validation_result['is_valid']:
        for error in validation_result['errors']:
            logging.error("Connection validation failed: %s", error)
        return graph

    if validation_result['warnings']:
        for warning in validation_result['warnings']:
            logging.warning("Connection validation warning: %s", warning)

    if validation_result['suggestions']:
        for suggestion in validation_result['suggestions']:
            logging.info("Connection suggestion: %s", suggestion)

    try:
        if id_manager is None:
            id_manager = get_id_manager()
        source_index, target_index = _get_node_indices(graph, source_id, target_id, id_manager)
        if source_index is None or target_index is None:
            return graph

        # Create the edge
        edge = EnhancedEdge(source_index, target_index, weight, edge_type)
        new_edge = torch.tensor([[source_index], [target_index]], dtype=torch.long)

        # Add to graph edge index
        if graph.edge_index.numel() == 0:
            graph.edge_index = new_edge
        else:
            graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)

        # Add to edge attributes with thread safety
        if not hasattr(graph, 'edge_attributes'):
            graph.edge_attributes = []
        if not hasattr(graph, '_edge_attributes_lock'):
            graph._edge_attributes_lock = threading.RLock()

        with graph._edge_attributes_lock:
            graph.edge_attributes.append(edge)

        # Diagnostic log for connection creation
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[CONNECTION] Created validated edge: source=%s (idx=%s) to target=%s (idx=%s), type=%s, weight=%s", source_id, source_index, target_id, target_index, edge_type, weight)

        return graph

    except (AttributeError, ValueError, TypeError, RuntimeError) as e:
        logging.error("Error creating validated connection between %s and %s: %s", source_id, target_id, e)
        return graph


def get_edge_attributes(graph, edge_idx):
    """
    Get edge attributes for a given edge index.

    Args:
        graph: Neural graph object.
        edge_idx (int): Edge index.

    Returns:
        EnhancedEdge or None: The edge attributes if found, else None.
    """
    if safe_hasattr(graph, 'edge_attributes', '_edge_attributes_lock'):
        with graph._edge_attributes_lock:
            if edge_idx < len(graph.edge_attributes):
                return graph.edge_attributes[edge_idx]
    return None


def apply_weight_change(graph, edge_idx, weight_change):
    """
    Apply a weight change to an edge.

    Args:
        graph: Neural graph object.
        edge_idx (int): Edge index.
        weight_change (float): Amount to change the weight.

    Returns:
        graph: Updated graph.
    """
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
                            logging.debug("Edge %s weight changed from %.3f to %.3f", edge_idx, old_weight, new_weight)
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
                        logging.debug("Edge %s weight changed from %.3f to %.3f", edge_idx, old_weight, new_weight)
    except (AttributeError, ValueError, TypeError, RuntimeError, IndexError) as e:
        log_step("Error applying weight change", error=str(e), edge_idx=edge_idx)
    return graph


def create_basic_connections(graph, id_manager=None):
    """
    Create basic connections in the neural graph.

    Args:
        graph: Neural graph
        id_manager: Node ID manager (optional, will use global if not provided)

    Returns:
        Updated graph with basic connections
    """
    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        return graph
    if id_manager is None:
        id_manager = get_id_manager()
    num_nodes = len(graph.node_labels)
    if num_nodes < 2:
        return graph
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


def _create_sensory_dynamic_connections(graph, sensory_nodes, dynamic_nodes, validator, max_connections):
    """Create connections between sensory and dynamic nodes."""
    connections_created = 0
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
                dynamic_energy = graph.x[dynamic_idx, 0].item() if hasattr(graph, 'x') else 0.0
                energy_cap = get_node_energy_cap()
                energy_mod = (sensory_energy + dynamic_energy) / (2 * energy_cap) if energy_cap > 0 else 1.0
                energy_factor = max(0.1, min(2.0, energy_mod * 2.0))
                connection_types = ['excitatory', 'modulatory', 'plastic']
                connection_type = connection_types[connections_created % len(connection_types)]
                base_weight = 0.3 + (connections_created * 0.1)
                weight = base_weight * energy_factor

                validation_result = validator.validate_connection(
                    graph, sensory_id, dynamic_id, connection_type, weight
                )

                if validation_result['is_valid']:
                    graph = create_weighted_connection(graph, sensory_id, dynamic_id, weight, connection_type)
                    connections_created += 1
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug("[FORMATION] Sensory-dynamic connection created: %s -> %s, type=%s, weight=%s, energy_mod=%.2f", sensory_id, dynamic_id, connection_type, weight, energy_mod)
                else:
                    for error in validation_result['errors']:
                        logging.warning("Connection validation failed: %s", error)
    return graph, connections_created


def _create_dynamic_dynamic_connections(graph, dynamic_nodes, validator, max_connections, connections_created):
    """Create connections between dynamic nodes."""
    if len(dynamic_nodes) > 1:
        limited_dynamic = dynamic_nodes[:3]
        for i, node1_idx in enumerate(limited_dynamic):
            if connections_created >= max_connections:
                break
            for _, node2_idx in enumerate(limited_dynamic[i+1:], i+1):
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
                        energy1 = graph.x[node1_idx, 0].item() if hasattr(graph, 'x') else 0.0
                        energy2 = graph.x[node2_idx, 0].item() if hasattr(graph, 'x') else 0.0
                        energy_cap = get_node_energy_cap()
                        energy_mod = (energy1 + energy2) / (2 * energy_cap) if energy_cap > 0 else 1.0
                        energy_factor = max(0.1, min(3.0, energy_mod * 3.0))
                        base_weight = 0.2 + (connections_created * 0.05)
                        weight = base_weight * energy_factor

                        validation_result = validator.validate_connection(
                            graph, node1_id, node2_id, connection_type, weight
                        )

                        if validation_result['is_valid']:
                            graph = create_weighted_connection(graph, node1_id, node2_id, weight, connection_type)
                            connections_created += 1
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug("[FORMATION] Dynamic-dynamic connection created: %s -> %s, type=%s, weight=%s, energy_mod=%.2f, criteria: cluster_sim=%s, affinity_compat=%s, pos_prox=%s", node1_id, node2_id, connection_type, weight, energy_mod, cluster_similarity, affinity_compatible, position_proximity)
                        else:
                            for error in validation_result['errors']:
                                logging.warning("Dynamic connection validation failed: %s", error)
    return graph, connections_created


def _create_behavior_connections(graph, validator):
    """Create connections based on node behaviors."""
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
                    osc_energy = graph.x[osc_idx, 0].item() if hasattr(graph, 'x') else 0.0
                    int_energy = graph.x[int_idx, 0].item() if hasattr(graph, 'x') else 0.0
                    energy_cap = get_node_energy_cap()
                    energy_mod = (osc_energy + int_energy) / (2 * energy_cap) if energy_cap > 0 else 1.0
                    weight = 0.4 * max(0.5, energy_mod)

                    validation_result = validator.validate_connection(
                        graph, osc_id, int_id, 'excitatory', weight
                    )

                    if validation_result['is_valid']:
                        graph = create_weighted_connection(graph, osc_id, int_id, weight, 'excitatory')
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            logging.debug("[FORMATION] Oscillator-integrator connection created: %s -> %s, weight=%s, energy_mod=%.2f, excitatory", osc_id, int_id, weight, energy_mod)
                    else:
                        for error in validation_result['errors']:
                            logging.warning("Oscillator-integrator connection validation failed: %s", error)

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
                    relay_energy = graph.x[relay_idx, 0].item() if hasattr(graph, 'x') else 0.0
                    highway_energy = graph.x[highway_idx, 0].item() if hasattr(graph, 'x') else 0.0
                    energy_cap = get_node_energy_cap()
                    energy_mod = (relay_energy + highway_energy) / (2 * energy_cap) if energy_cap > 0 else 1.0
                    weight = 0.6 * max(0.5, energy_mod)

                    validation_result = validator.validate_connection(
                        graph, relay_id, highway_id, 'excitatory', weight
                    )

                    if validation_result['is_valid']:
                        graph = create_weighted_connection(graph, relay_id, highway_id, weight, 'excitatory')
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            logging.debug("[FORMATION] Relay-highway connection created: %s -> %s, weight=%s, energy_mod=%.2f, excitatory", relay_id, highway_id, weight, energy_mod)
                    else:
                        for error in validation_result['errors']:
                            logging.warning("Relay-highway connection validation failed: %s", error)
    return graph


def _create_random_connections(graph, num_nodes, validator):
    """Create random connections."""
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
                source_energy = graph.x[source_idx, 0].item() if hasattr(graph, 'x') else 0.0
                target_energy = graph.x[target_idx, 0].item() if hasattr(graph, 'x') else 0.0
                energy_cap = get_node_energy_cap()
                energy_mod = (source_energy + target_energy) / (2 * energy_cap) if energy_cap > 0 else 1.0
                energy_factor = max(0.1, min(2.5, energy_mod * 2.5))
                weight = 0.2 * energy_factor

                validation_result = validator.validate_connection(
                    graph, source_id, target_id, 'excitatory', weight
                )

                if validation_result['is_valid']:
                    graph = create_weighted_connection(graph, source_id, target_id, weight, 'excitatory')
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug("[FORMATION] Random connection created: %s -> %s, weight=%s, energy_mod=%.2f, excitatory", source_id, target_id, weight, energy_mod)
                else:
                    for error in validation_result['errors']:
                        logging.warning("Random connection validation failed: %s", error)
    return graph


def intelligent_connection_formation(graph):
    """Create intelligent connections with centralized validation and performance optimizations."""

    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        return graph
    num_nodes = len(graph.node_labels)
    if num_nodes < 2:
        return graph
    if num_nodes > 50000:
        log_step("Skipping intelligent connections for large graph (%s nodes) to maintain performance", num_nodes=num_nodes)
        return graph

    validator = get_connection_validator()
    max_connections = min(50, max(10, num_nodes // 100))
    sensory_sample_size = min(10, max(5, num_nodes // 100))
    dynamic_sample_size = min(20, max(10, num_nodes // 50))
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
    graph, connections_created = _create_sensory_dynamic_connections(graph, sensory_nodes, dynamic_nodes, validator, max_connections)
    graph, connections_created = _create_dynamic_dynamic_connections(graph, dynamic_nodes, validator, max_connections, connections_created)
    graph = _create_behavior_connections(graph, validator)
    graph = _create_random_connections(graph, num_nodes, validator)
    return graph


def update_connection_weights(graph, learning_rate=ConnectionConstants.LEARNING_RATE_DEFAULT):
    """
    Update connection weights based on node activities.

    Args:
        graph: Neural graph object.
        learning_rate (float): Learning rate for weight updates.

    Returns:
        graph: Updated graph.
    """
    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return graph
    for _, edge in enumerate(graph.edge_attributes):
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
    """
    Add dynamic connections to the graph using intelligent formation.

    Args:
        graph: Neural graph object.

    Returns:
        graph: Updated graph with new connections.
    """
    return intelligent_connection_formation(graph)


def add_connections(graph, connection_strategy=None):
    """
    Add connections to the graph using a specified strategy.

    Args:
        graph: Neural graph object.
        connection_strategy (callable): Function to create connections.

    Returns:
        graph: Updated graph with new connections.
    """
    if connection_strategy is None:
        connection_strategy = intelligent_connection_formation
    return connection_strategy(graph)







