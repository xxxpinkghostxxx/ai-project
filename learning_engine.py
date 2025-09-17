
import time
import numpy as np
import torch
from logging_utils import log_step, log_node_state
from config_manager import get_learning_config


class LearningEngine:

    def __init__(self):
        config = get_learning_config()
        self.learning_rate = config.get('plasticity_rate', 0.01)
        self.eligibility_decay = config.get('eligibility_decay', 0.95)
        self.stdp_window = config.get('stdp_window', 20.0)
        self.ltp_rate = config.get('ltp_rate', 0.02)
        self.ltd_rate = config.get('ltd_rate', 0.01)
        self.plasticity_threshold = 0.1
        self.consolidation_threshold = 0.5
        self.learning_stats = {
            'stdp_events': 0,
            'weight_changes': 0,
            'consolidation_events': 0,
            'memory_traces_formed': 0,
            'total_weight_change': 0.0
        }
        self.memory_traces = {}
        self.memory_decay_rate = 0.99
    def apply_timing_learning(self, pre_node, post_node, edge, delta_t):

        if abs(delta_t) <= self.stdp_window / 1000.0:
            if delta_t > 0:
                weight_change = self.ltp_rate * np.exp(-delta_t / 0.01)
                self.learning_stats['stdp_events'] += 1
                log_step("LTP applied",
                        pre_id=pre_node.get('id', '?'),
                        post_id=post_node.get('id', '?'),
                        delta_t=delta_t,
                        weight_change=weight_change)
            else:
                weight_change = -self.ltd_rate * np.exp(delta_t / 0.01)
                self.learning_stats['stdp_events'] += 1
                log_step("LTD applied",
                        pre_id=pre_node.get('id', '?'),
                        post_id=post_node.get('id', '?'),
                        delta_t=delta_t,
                        weight_change=weight_change)
            edge.eligibility_trace += weight_change
            return weight_change
        return 0.0
    def consolidate_connections(self, graph):

        if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
            return graph
        consolidated_count = 0
        total_weight_change = 0.0
        for edge_idx, edge in enumerate(graph.edge_attributes):
            if edge.eligibility_trace > self.consolidation_threshold:
                weight_change = edge.eligibility_trace * self.learning_rate
                new_weight = edge.weight + weight_change
                new_weight = max(0.1, min(5.0, new_weight))
                actual_change = new_weight - edge.weight
                edge.weight = new_weight
                edge.eligibility_trace *= 0.5
                consolidated_count += 1
                total_weight_change += actual_change
                self.learning_stats['consolidation_events'] += 1
                log_step("Connection consolidated",
                        edge_idx=edge_idx,
                        source=edge.source,
                        target=edge.target,
                        weight_change=actual_change,
                        new_weight=new_weight)
        self.learning_stats['weight_changes'] += consolidated_count
        self.learning_stats['total_weight_change'] += total_weight_change
        return graph
    def form_memory_traces(self, graph):

        if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
            return graph
        memory_traces_formed = 0
        for node_idx, node in enumerate(graph.node_labels):
            if node.get('behavior') in ['integrator', 'relay']:
                if self._has_stable_pattern(node, graph):
                    self._create_memory_trace(node_idx, graph)
                    memory_traces_formed += 1
        self.learning_stats['memory_traces_formed'] += memory_traces_formed
        if memory_traces_formed > 0:
            log_step("Memory traces formed", count=memory_traces_formed)
        return graph
    def _has_stable_pattern(self, node, graph):

        last_activation = node.get('last_activation', 0)
        current_time = time.time()
        if (current_time - last_activation < 10.0 and
            node.get('energy', 0.0) > 0.3):
            return True
        return False
    def _create_memory_trace(self, node_idx, graph):

        incoming_edges = []
        if hasattr(graph, 'edge_attributes'):
            for edge in graph.edge_attributes:
                if edge.target == node_idx:
                    incoming_edges.append({
                        'source': edge.source,
                        'weight': edge.weight,
                        'type': edge.type
                    })
        self.memory_traces[node_idx] = {
            'connections': incoming_edges,
            'strength': 1.0,
            'formation_time': time.time(),
            'activation_count': 0
        }
    def apply_memory_influence(self, graph):

        if not self.memory_traces:
            return graph
        current_time = time.time()
        to_remove = []
        for node_idx, memory_trace in self.memory_traces.items():
            age = current_time - memory_trace['formation_time']
            memory_trace['strength'] *= (self.memory_decay_rate ** (age / 60.0))
            if memory_trace['strength'] < 0.1:
                to_remove.append(node_idx)
                continue
            if node_idx < len(graph.node_labels):
                self._reinforce_memory_pattern(node_idx, memory_trace, graph)
        for node_idx in to_remove:
            del self.memory_traces[node_idx]
        return graph
    def _reinforce_memory_pattern(self, node_idx, memory_trace, graph):

        if not hasattr(graph, 'edge_attributes'):
            return
        for edge in graph.edge_attributes:
            if edge.target == node_idx:
                for mem_conn in memory_trace['connections']:
                    if (edge.source == mem_conn['source'] and
                        edge.type == mem_conn['type']):
                        reinforcement = memory_trace['strength'] * 0.1
                        edge.weight = min(edge.weight + reinforcement, 5.0)
                        break
    def get_learning_statistics(self):
        return self.learning_stats.copy()
    def reset_statistics(self):
        self.learning_stats = {
            'stdp_events': 0,
            'weight_changes': 0,
            'consolidation_events': 0,
            'memory_traces_formed': 0,
            'total_weight_change': 0.0
        }
    def get_memory_trace_count(self):
        return len(self.memory_traces)


def calculate_learning_efficiency(graph):

    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return 0.0
    total_weights = sum(edge.weight for edge in graph.edge_attributes)
    avg_weight = total_weights / len(graph.edge_attributes)
    efficiency = min(avg_weight / 2.5, 1.0)
    return efficiency


def detect_learning_patterns(graph):

    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return {'patterns_detected': 0}
    weights = [edge.weight for edge in graph.edge_attributes]
    weight_variance = np.var(weights) if len(weights) > 1 else 0
    edge_types = {}
    for edge in graph.edge_attributes:
        edge_types[edge.type] = edge_types.get(edge.type, 0) + 1
    patterns_detected = 0
    if weight_variance > 0.5:
        patterns_detected += 1
    if len(edge_types) > 1:
        patterns_detected += 1
    return {
        'patterns_detected': patterns_detected,
        'weight_variance': weight_variance,
        'edge_type_distribution': edge_types,
        'total_connections': len(graph.edge_attributes)
    }
