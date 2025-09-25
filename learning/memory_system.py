
import time
import numpy as np
from utils.logging_utils import log_step


DEFAULT_MEMORY_STRENGTH = 1.0
DEFAULT_MEMORY_DECAY_RATE = 0.99
DEFAULT_CONSOLIDATION_THRESHOLD = 0.8
DEFAULT_RECALL_THRESHOLD = 0.3
DEFAULT_MAX_MEMORY_TRACES = 1000


class MemorySystem:

    def __init__(self):
        self.memory_traces = {}
        self.consolidation_threshold = DEFAULT_CONSOLIDATION_THRESHOLD
        self.memory_decay_rate = DEFAULT_MEMORY_DECAY_RATE
        self.recall_threshold = DEFAULT_RECALL_THRESHOLD
        self.max_memory_traces = DEFAULT_MAX_MEMORY_TRACES
        self.memory_stats = {
            'traces_formed': 0,
            'traces_consolidated': 0,
            'traces_decayed': 0,
            'patterns_recalled': 0,
            'total_memory_strength': 0.0
        }
        self.pattern_cache = {}
        self.cache_decay_rate = 0.95
    def form_memory_traces(self, graph):

        if not hasattr(graph, 'node_labels') or not graph.node_labels:
            log_step("Memory trace formation skipped: no node_labels")
            return graph
        memory_traces_formed = 0
        checked_nodes = 0
        stable_nodes = 0
        for node_idx, node in enumerate(graph.node_labels):
            if node.get('behavior') in ['integrator', 'relay']:
                checked_nodes += 1
                if self._has_stable_pattern(node, graph):
                    stable_nodes += 1
                    if len(self.memory_traces) < self.max_memory_traces:
                        self._create_memory_trace(node_idx, graph)
                        memory_traces_formed += 1
                    else:
                        self._replace_weakest_memory(node_idx, graph)
        self.memory_stats['traces_formed'] += memory_traces_formed
        graph.memory_traces = self.memory_traces
        if memory_traces_formed > 0:
            log_step("Memory traces formed", count=memory_traces_formed)
        return graph
    def _has_stable_pattern(self, node, graph):

        last_activation = node.get('last_activation', 0)
        current_time = time.time()
        if not isinstance(last_activation, (int, float)):
            log_step("Stable pattern check failed: invalid last_activation type", node_id=node.get('id'), type=type(last_activation))
            return False
        time_diff = current_time - last_activation
        energy = node.get('energy', 0.0)
        is_recent = time_diff < 10.1
        has_energy = energy > 0.2
        if is_recent and has_energy:
            return True
        return False
    def _create_memory_trace(self, node_idx, graph):

        incoming_edges = []
        if hasattr(graph, 'edge_index') and graph.edge_index is not None and graph.edge_index.numel() > 0:
            edge_index = graph.edge_index.cpu().numpy()
            target_edges = edge_index[1] == node_idx
            for edge_idx in np.where(target_edges)[0]:
                source_idx = edge_index[0, edge_idx]
                source_node = graph.node_labels[source_idx]
                incoming_edges.append({
                    'source': int(source_idx),
                    'weight': 1.0,
                    'type': 'excitatory',
                    'source_behavior': source_node.get('behavior', 'unknown')
                })
        try:
            self.memory_traces[node_idx] = {
                'connections': incoming_edges,
                'strength': DEFAULT_MEMORY_STRENGTH,
                'formation_time': time.time(),
                'activation_count': 0,
                'last_accessed': time.time(),
                'pattern_type': self._classify_pattern(incoming_edges, graph)
            }
        except Exception as e:
            log_step("Failed to create memory trace", error=str(e), node_idx=node_idx)
            return
        log_step("Memory trace created",
                node_id=node_idx,
                pattern_type=self.memory_traces[node_idx]['pattern_type'],
                connection_count=len(incoming_edges))
    def _classify_pattern(self, connections, graph):

        if not connections:
            return 'isolated'
        excitatory_count = sum(1 for conn in connections if conn['type'] == 'excitatory')
        inhibitory_count = sum(1 for conn in connections if conn['type'] == 'inhibitory')
        modulatory_count = sum(1 for conn in connections if conn['type'] == 'modulatory')
        source_behaviors = [conn['source_behavior'] for conn in connections]
        behavior_counts = {}
        for behavior in source_behaviors:
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        if excitatory_count > inhibitory_count + modulatory_count:
            if 'oscillator' in behavior_counts:
                return 'rhythmic_excitatory'
            elif 'integrator' in behavior_counts:
                return 'integrative_excitatory'
            else:
                return 'excitatory_dominant'
        elif inhibitory_count > excitatory_count:
            return 'inhibitory_dominant'
        elif modulatory_count > 0:
            return 'modulatory_influenced'
        else:
            return 'balanced'
    def consolidate_memories(self, graph):

        current_time = time.time()
        consolidated_count = 0
        for node_idx, memory_trace in list(self.memory_traces.items()):
            if node_idx < len(graph.node_labels):
                node = graph.node_labels[node_idx]
                if self._has_stable_pattern(node, graph):
                    memory_trace['strength'] = min(memory_trace['strength'] * 1.1, 2.0)
                    if 'activation_count' not in memory_trace:
                        memory_trace['activation_count'] = 0
                    memory_trace['activation_count'] += 1
                    memory_trace['last_accessed'] = current_time
                    consolidated_count += 1
                    log_step("Memory consolidated",
                            node_id=node_idx,
                            new_strength=memory_trace['strength'],
                            activation_count=memory_trace['activation_count'])
        self.memory_stats['traces_consolidated'] += consolidated_count
        return graph
    def decay_memories(self):

        current_time = time.time()
        to_remove = []
        for node_idx, memory_trace in self.memory_traces.items():
            age = current_time - memory_trace['formation_time']
            base_decay = self.memory_decay_rate ** (age / 60.0)
            strength_factor = memory_trace['strength'] / 2.0
            effective_decay = base_decay * (1.0 - strength_factor * 0.5)
            memory_trace['strength'] *= effective_decay
            if memory_trace['strength'] < 0.1:
                to_remove.append(node_idx)
        for node_idx in to_remove:
            del self.memory_traces[node_idx]
            self.memory_stats['traces_decayed'] += 1
        return len(to_remove)
    def recall_patterns(self, graph, target_node_idx):

        recalled_patterns = []
        for memory_idx, memory_trace in self.memory_traces.items():
            if self._is_pattern_relevant(memory_trace, target_node_idx, graph):
                relevance = self._calculate_pattern_relevance(memory_trace, target_node_idx, graph)
                if relevance > self.recall_threshold:
                    recalled_patterns.append({
                        'memory_id': memory_idx,
                        'pattern': memory_trace,
                        'relevance': relevance
                    })
        recalled_patterns.sort(key=lambda x: x['relevance'], reverse=True)
        if recalled_patterns:
            self.memory_stats['patterns_recalled'] += len(recalled_patterns)
            log_step("Patterns recalled",
                    target_node=target_node_idx,
                    pattern_count=len(recalled_patterns))
        return recalled_patterns
    def _is_pattern_relevant(self, memory_trace, target_node_idx, graph):

        if target_node_idx >= len(graph.node_labels):
            return False
        target_node = graph.node_labels[target_node_idx]
        target_behavior = target_node.get('behavior', 'unknown')
        memory_pattern_type = memory_trace.get('pattern_type', 'unknown')
        if target_behavior in ['integrator', 'relay']:
            if memory_pattern_type in ['integrative_excitatory', 'excitatory_dominant']:
                return True
        return False
    def _calculate_pattern_relevance(self, memory_trace, target_node_idx, graph):

        if target_node_idx >= len(graph.node_labels):
            return 0.0
        target_node = graph.node_labels[target_node_idx]
        base_relevance = memory_trace['strength'] / 2.0
        current_time = time.time()
        last_accessed = memory_trace['last_accessed']
        recency_boost = 1.0 / (1.0 + (current_time - last_accessed) / 60.0)
        activation_boost = min(memory_trace['activation_count'] / 10.0, 0.5)
        relevance = base_relevance + (recency_boost * 0.3) + (activation_boost * 0.2)
        return min(relevance, 1.0)
    def _replace_weakest_memory(self, new_node_idx, graph):

        if not self.memory_traces:
            return
        weakest_idx = min(self.memory_traces.keys(),
                         key=lambda k: self.memory_traces[k]['strength'])
        del self.memory_traces[weakest_idx]
        self._create_memory_trace(new_node_idx, graph)
        log_step("Memory trace replaced",
                removed_node=weakest_idx,
                new_node=new_node_idx)
    def get_memory_statistics(self):
        total_strength = sum(trace['strength'] for trace in self.memory_traces.values())
        self.memory_stats['total_memory_strength'] = total_strength
        return self.memory_stats.copy()
    def reset_statistics(self):
        self.memory_stats = {
            'traces_formed': 0,
            'traces_consolidated': 0,
            'traces_decayed': 0,
            'patterns_recalled': 0,
            'total_memory_strength': 0.0
        }
    def get_memory_trace_count(self):
        return len(self.memory_traces)
    def get_memory_summary(self):
        summary = []
        for node_idx, trace in self.memory_traces.items():
            summary.append({
                'node_id': node_idx,
                'pattern_type': trace['pattern_type'],
                'strength': trace['strength'],
                'activation_count': trace['activation_count'],
                'age_minutes': (time.time() - trace['formation_time']) / 60.0
            })
        return summary
    def get_node_memory_importance(self, node_id):

        if node_id not in self.memory_traces:
            return 0.0
        trace = self.memory_traces[node_id]
        strength_factor = trace['strength'] / 10.0
        activation_factor = min(trace['activation_count'] / 20.0, 1.0)
        age_factor = max(0.0, 1.0 - (time.time() - trace['formation_time']) / 3600.0)
        pattern_importance = {
            'oscillation': 0.8,
            'integration': 0.9,
            'relay': 0.7,
            'highway': 0.6,
            'workspace': 0.95,
            'sensory': 0.5,
            'dynamic': 0.3
        }.get(trace['pattern_type'], 0.5)
        importance = (strength_factor * 0.3 +
                     activation_factor * 0.3 +
                     age_factor * 0.2 +
                     pattern_importance * 0.2)
        return min(importance, 1.0)


def analyze_memory_distribution(memory_system):

    summary = memory_system.get_memory_summary()
    pattern_distribution = {}
    strength_distribution = []
    for memory in summary:
        pattern_type = memory['pattern_type']
        pattern_distribution[pattern_type] = pattern_distribution.get(pattern_type, 0) + 1
        strength_distribution.append(memory['strength'])
    return {
        'pattern_distribution': pattern_distribution,
        'avg_strength': np.mean(strength_distribution) if strength_distribution else 0.0,
        'strength_variance': np.var(strength_distribution) if len(strength_distribution) > 1 else 0.0,
        'total_memories': len(summary)
    }


def calculate_memory_efficiency(memory_system):

    stats = memory_system.get_memory_statistics()
    formation_efficiency = min(stats['traces_formed'] / 10.0, 1.0)
    consolidation_efficiency = min(stats['traces_consolidated'] / 5.0, 1.0)
    recall_efficiency = min(stats['patterns_recalled'] / 3.0, 1.0)
    efficiency = (formation_efficiency * 0.4 +
                 consolidation_efficiency * 0.3 +
                 recall_efficiency * 0.3)
    return efficiency
if __name__ == "__main__":
    memory_system = MemorySystem()
    print("Memory System initialized successfully!")
    print(f"Max memory traces: {memory_system.max_memory_traces}")
    print(f"Memory decay rate: {memory_system.memory_decay_rate}")
    print(f"Consolidation threshold: {memory_system.consolidation_threshold}")
    stats = memory_system.get_memory_statistics()
    print(f"Initial statistics: {stats}")
    print("\nMemory System is ready for integration!")
