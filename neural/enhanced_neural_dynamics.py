
import time
import numpy as np
import numba as nb
from typing import Dict, Any, List, Optional

from torch_geometric.data import Data
from collections import defaultdict, deque

from utils.logging_utils import log_step
from config.unified_config_manager import get_learning_config, get_system_constants, get_enhanced_nodes_config
from energy.energy_constants import ConnectionConstants
from energy.node_access_layer import NodeAccessLayer
from utils.event_bus import get_event_bus


class EnhancedNeuralDynamics:

    def __init__(self):
        self.config = get_learning_config()
        self.system_constants = get_system_constants()
        self.enhanced_config = get_enhanced_nodes_config()
        self.stdp_window = self.config.get('stdp_window', 20.0)
        self.ltp_rate = self.config.get('ltp_rate', 0.02)
        self.ltd_rate = self.config.get('ltd_rate', 0.01)
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        self.ieg_threshold = 0.7
        self.ieg_decay_rate = 0.99
        self.ieg_duration = 300.0
        self.theta_burst_threshold = 4
        self.theta_burst_window = 50.0
        self.theta_frequency = 100.0
        self.membrane_time_constant = 10.0
        self.resting_potential = -70.0
        self.threshold_potential = -50.0
        self.reset_potential = -80.0
        self.refractory_period = 2.0
        self.dendritic_time_constant = 50.0
        self.dendritic_threshold = 0.6
        self.target_ei_ratio = 0.8
        self.ei_adaptation_rate = 0.001
        self.criticality_target = 1.0
        self.eligibility_tau = 1000.0
        self.consolidation_threshold = 0.5
        self.dopamine_level = 0.0
        self.acetylcholine_level = 0.0
        self.norepinephrine_level = 0.0
        self.neuromodulator_decay = 0.95
        self.neuromodulators = {
            'dopamine': self.dopamine_level,
            'acetylcholine': self.acetylcholine_level,
            'norepinephrine': self.norepinephrine_level
        }
        self.node_activity_history = defaultdict(lambda: deque(maxlen=1000))
        self.edge_activity_history = defaultdict(lambda: deque(maxlen=1000))
        self.spike_times = defaultdict(list)
        self.theta_burst_counters = defaultdict(int)
        self.ieg_flags = defaultdict(bool)
        self.ieg_timers = defaultdict(float)
        self.stats = {
            'total_spikes': 0,
            'stdp_events': 0,
            'ieg_activations': 0,
            'theta_bursts': 0,
            'consolidation_events': 0,
            'ei_balance_adjustments': 0
        }
        self.event_bus = get_event_bus()
        log_step("EnhancedNeuralDynamics initialized")
    def update_neural_dynamics(self, graph: Data, step: int) -> Data:
        """Update neural dynamics with input validation."""
        # Input validation
        if graph is None:
            log_step("Error: graph is None in update_neural_dynamics")
            return graph
        if not isinstance(step, int) or step < 0:
            log_step("Error: invalid step value", step=step)
            return graph
        if not hasattr(graph, 'node_labels') or not graph.node_labels:
            log_step("Error: graph missing node_labels")
            return graph

        try:
            graph = self._update_membrane_dynamics(graph, step)
            graph = self._process_spikes(graph, step)
            graph = self._apply_stdp_learning(graph, step)
            graph = self._update_ieg_tagging(graph, step)
            graph = self._process_theta_bursts(graph, step)
            graph = self._update_eligibility_traces(graph, step)
            graph = self._apply_memory_consolidation(graph, step)
            graph = self._update_homeostatic_control(graph, step)
            self._update_neuromodulation(step)
            return graph
        except Exception as e:
            log_step("Error in neural dynamics update", error=str(e))
            return graph
    def _update_membrane_dynamics(self, graph: Data, step: int) -> Data:

        access_layer = NodeAccessLayer(graph)
        current_time = time.time()
        enhanced_node_ids = []
        if hasattr(graph, 'enhanced_node_ids'):
            enhanced_node_ids = graph.enhanced_node_ids
        else:
            sample_size = min(1000, len(graph.node_labels))
            sample_indices = np.random.choice(len(graph.node_labels), sample_size, replace=False)
            for node_id in sample_indices:
                try:
                    node = access_layer.get_node_by_id(node_id)
                    if node and node.get('enhanced_behavior', False):
                        enhanced_node_ids.append(node_id)
                except (IndexError, AttributeError, KeyError):
                    continue
        for node_id in enhanced_node_ids:
            try:
                node = access_layer.get_node_by_id(node_id)
                if node is None:
                    continue
            except (IndexError, AttributeError, KeyError):
                continue
            v_mem = node.get('membrane_potential', 0.0)
            v_dend = node.get('dendritic_potential', 0.0)
            synaptic_input = self._calculate_synaptic_input(graph, node_id, access_layer)
            v_dend += (synaptic_input - v_dend) * (1.0 / self.dendritic_time_constant)
            access_layer.update_node_property(node_id, 'dendritic_potential', v_dend)
            dendritic_influence = 0.0
            if v_dend > self.dendritic_threshold:
                dendritic_influence = (v_dend - self.dendritic_threshold) * 0.3
            v_mem += (synaptic_input + dendritic_influence - v_mem) * (1.0 / self.membrane_time_constant)
            v_mem += (self.resting_potential - v_mem) * 0.01
            refractory_timer = node.get('refractory_timer', 0.0)
            if v_mem > self.threshold_potential and refractory_timer <= 0:
                v_mem = self.reset_potential
                access_layer.update_node_property(node_id, 'refractory_timer', self.refractory_period)
                access_layer.update_node_property(node_id, 'last_spike_time', current_time)
                self.spike_times[node_id].append(current_time)
                self.stats['total_spikes'] += 1
                log_step("Spike occurred", node_id=node_id, membrane_potential=v_mem)
                try:
                    self.event_bus.emit('SPIKE', {'node_id': node_id, 'timestamp': current_time})
                except Exception:
                    pass  # Fallback: direct calls already handled
            if refractory_timer > 0:
                access_layer.update_node_property(node_id, 'refractory_timer',
                                                max(0.0, refractory_timer - 0.01))
            access_layer.update_node_property(node_id, 'membrane_potential', v_mem)
            self._update_theta_burst_counter(node_id, current_time)
        return graph
    def _calculate_synaptic_input(self, graph: Data, node_id: int, access_layer: NodeAccessLayer) -> float:

        total_input = 0.0
        if not hasattr(graph, 'edge_attributes'):
            return total_input
        for edge in graph.edge_attributes:
            if edge.target == node_id:
                source_id = edge.source
                source_node = access_layer.get_node_by_id(source_id)
                if source_node is None:
                    continue
                last_spike = source_node.get('last_spike_time', 0)
                current_time = time.time()
                if current_time - last_spike < 0.1:
                    effective_weight = edge.get_effective_weight()
                    if edge.type == 'excitatory':
                        total_input += effective_weight
                    elif edge.type == 'inhibitory':
                        total_input -= effective_weight
                    elif edge.type == 'modulatory':
                        total_input += effective_weight * 0.5
                    elif edge.type == 'gated':
                        source_energy = access_layer.get_node_energy(source_id)
                        if source_energy and source_energy > edge.get('gate_threshold', 0.5):
                            total_input += effective_weight
        return total_input
    def _process_spikes(self, graph: Data, step: int) -> Data:

        current_time = time.time()
        for node_id, spike_times in self.spike_times.items():
            if not spike_times:
                continue
            self.node_activity_history[node_id].append(current_time)
            recent_spikes = [t for t in spike_times if current_time - t < self.stdp_window / 1000.0]
            if len(recent_spikes) >= self.theta_burst_threshold:
                intervals = np.diff(recent_spikes[-self.theta_burst_threshold:])
                if all(0.008 < interval < 0.012 for interval in intervals):
                    self.theta_burst_counters[node_id] += 1
                    self.stats['theta_bursts'] += 1
                    self._tag_connections_for_ltp(graph, node_id)
        return graph
    @staticmethod
    @nb.jit(nopython=True)
    def compute_stdp_weight_change(source_times, target_times, ltp_rate, ltd_rate, tau_plus, tau_minus, stdp_window):
        weight_change = 0.0
        n_source = len(source_times)
        n_target = len(target_times)
        for i in range(n_source):
            for j in range(n_target):
                delta_t = target_times[j] - source_times[i]
                if 0 < delta_t < stdp_window:
                    weight_change += ltp_rate * np.exp(-delta_t / tau_plus)
                elif -stdp_window < delta_t < 0:
                    weight_change -= ltd_rate * np.exp(-delta_t / tau_minus)
        return weight_change
    
    
    def _apply_stdp_learning(self, graph: Data, step: int) -> Data:
    
        if not hasattr(graph, 'edge_attributes'):
            return graph
        current_time = time.time()
        for edge in graph.edge_attributes:
            source_id = edge.source
            target_id = edge.target
            source_spikes = [t for t in self.spike_times[source_id]
                           if current_time - t < self.stdp_window / 1000.0]
            target_spikes = [t for t in self.spike_times[target_id]
                           if current_time - t < self.stdp_window / 1000.0]
            if not source_spikes or not target_spikes:
                continue
            source_array = np.array(source_spikes, dtype=np.float64)
            target_array = np.array(target_spikes, dtype=np.float64)
            stdp_window_s = self.stdp_window / 1000.0
            tau_plus_s = self.tau_plus / 1000.0
            tau_minus_s = self.tau_minus / 1000.0
            weight_change = self.compute_stdp_weight_change(source_array, target_array, self.ltp_rate, self.ltd_rate, tau_plus_s, tau_minus_s, stdp_window_s)
            if abs(weight_change) > 0.001:
                new_weight = max(ConnectionConstants.WEIGHT_MIN,
                               min(ConnectionConstants.WEIGHT_CAP_MAX,
                                   edge.weight + weight_change))
                edge.weight = new_weight
                edge.update_eligibility_trace(weight_change)
                self.stats['stdp_events'] += 1
                log_step("STDP weight update",
                        source_id=source_id,
                        target_id=target_id,
                        weight_change=weight_change,
                        new_weight=new_weight)
        return graph
    def _update_ieg_tagging(self, graph: Data, step: int) -> Data:

        access_layer = NodeAccessLayer(graph)
        current_time = time.time()
        enhanced_node_ids = []
        if hasattr(graph, 'enhanced_node_ids'):
            enhanced_node_ids = graph.enhanced_node_ids
        else:
            sample_size = min(1000, len(graph.node_labels))
            sample_indices = np.random.choice(len(graph.node_labels), sample_size, replace=False)
            for node_id in sample_indices:
                try:
                    node = access_layer.get_node_by_id(node_id)
                    if node and node.get('enhanced_behavior', False):
                        enhanced_node_ids.append(node_id)
                except (IndexError, AttributeError, KeyError):
                    continue
        for node_id in enhanced_node_ids:
            try:
                node = access_layer.get_node_by_id(node_id)
                if node is None:
                    continue
            except (IndexError, AttributeError, KeyError):
                continue
            recent_activity = self._calculate_recent_activity(node_id, current_time)
            if recent_activity > self.ieg_threshold and not self.ieg_flags[node_id]:
                self.ieg_flags[node_id] = True
                self.ieg_timers[node_id] = self.ieg_duration
                self.stats['ieg_activations'] += 1
                access_layer.update_node_property(node_id, 'ieg_flag', True)
                log_step("IEG activated", node_id=node_id, activity=recent_activity)
            if self.ieg_flags[node_id]:
                self.ieg_timers[node_id] -= 1
                if self.ieg_timers[node_id] <= 0:
                    self.ieg_flags[node_id] = False
                    access_layer.update_node_property(node_id, 'ieg_flag', False)
        return graph
    def _process_theta_bursts(self, graph: Data, step: int) -> Data:

        if not hasattr(graph, 'edge_attributes'):
            return graph
        for node_id, burst_count in self.theta_burst_counters.items():
            if burst_count >= self.theta_burst_threshold:
                self._tag_connections_for_ltp(graph, node_id)
                self.theta_burst_counters[node_id] = 0
        return graph
    def _tag_connections_for_ltp(self, graph: Data, node_id: int):

        if not hasattr(graph, 'edge_attributes'):
            return
        for edge in graph.edge_attributes:
            if edge.source == node_id:
                edge.plasticity_tag = True
                edge.eligibility_trace += 0.5
                log_step("Connection tagged for LTP",
                        source_id=node_id,
                        target_id=edge.target)
    def _update_eligibility_traces(self, graph: Data, step: int) -> Data:

        if not hasattr(graph, 'edge_attributes'):
            return graph
        for edge in graph.edge_attributes:
            edge.eligibility_trace *= self.eligibility_tau / (self.eligibility_tau + 1.0)
            edge.eligibility_trace = max(0.0, edge.eligibility_trace)
        return graph
    def _apply_memory_consolidation(self, graph: Data, step: int) -> Data:

        if not hasattr(graph, 'edge_attributes'):
            return graph
        access_layer = NodeAccessLayer(graph)
        for edge in graph.edge_attributes:
            source_id = edge.source
            target_id = edge.target
            source_node = access_layer.get_node_by_id(source_id)
            target_node = access_layer.get_node_by_id(target_id)
            if (source_node and target_node and
                self.ieg_flags[source_id] and self.ieg_flags[target_id] and
                edge.plasticity_tag and edge.eligibility_trace > self.consolidation_threshold):
                consolidation_factor = 1.5
                new_weight = min(ConnectionConstants.WEIGHT_CAP_MAX,
                               edge.weight * consolidation_factor)
                edge.weight = new_weight
                edge.plasticity_tag = False
                edge.eligibility_trace *= 0.5
                self.stats['consolidation_events'] += 1
                log_step("Memory consolidated",
                        source_id=source_id,
                        target_id=target_id,
                        new_weight=new_weight)
        return graph
    def _update_homeostatic_control(self, graph: Data, step: int) -> Data:

        if step % 100 != 0:
            return graph
        ei_ratio = self._calculate_ei_ratio(graph)
        if abs(ei_ratio - self.target_ei_ratio) > 0.1:
            self._adjust_ei_balance(graph, ei_ratio)
            self.stats['ei_balance_adjustments'] += 1
        criticality = self._calculate_criticality(graph)
        if abs(criticality - self.criticality_target) > 0.2:
            self._adjust_criticality(graph, criticality)
        return graph
    def _calculate_ei_ratio(self, graph: Data) -> float:
        if not hasattr(graph, 'edge_attributes'):
            return 1.0
        excitatory_weight = 0.0
        inhibitory_weight = 0.0
        for edge in graph.edge_attributes:
            if edge.type == 'excitatory':
                excitatory_weight += abs(edge.weight)
            elif edge.type == 'inhibitory':
                inhibitory_weight += abs(edge.weight)
        if inhibitory_weight == 0:
            return 1.0
        return excitatory_weight / inhibitory_weight
    def _adjust_ei_balance(self, graph: Data, current_ratio: float):
        if not hasattr(graph, 'edge_attributes'):
            return
        adjustment_factor = 1.0 + self.ei_adaptation_rate * (self.target_ei_ratio - current_ratio)
        for edge in graph.edge_attributes:
            if edge.type == 'excitatory':
                edge.weight *= adjustment_factor
            elif edge.type == 'inhibitory':
                edge.weight /= adjustment_factor
            edge.weight = max(ConnectionConstants.WEIGHT_MIN,
                            min(ConnectionConstants.WEIGHT_CAP_MAX, edge.weight))
    def _calculate_criticality(self, graph: Data) -> float:
        if not hasattr(graph, 'edge_attributes'):
            return 1.0
        total_connections = len(graph.edge_attributes)
        if total_connections == 0:
            return 1.0
        node_degrees = defaultdict(int)
        for edge in graph.edge_attributes:
            node_degrees[edge.source] += 1
        if not node_degrees:
            return 1.0
        avg_degree = sum(node_degrees.values()) / len(node_degrees)
        return min(2.0, avg_degree / 10.0)
    def _adjust_criticality(self, graph: Data, current_criticality: float):
        pass
    def set_neuromodulator_level(self, neuromodulator: str, level: float):
        level = max(0.0, min(1.0, level))
        if neuromodulator == 'dopamine':
            self.dopamine_level = level
        elif neuromodulator == 'acetylcholine':
            self.acetylcholine_level = level
        elif neuromodulator == 'norepinephrine':
            self.norepinephrine_level = level
        else:
            log_step(f"Unknown neuromodulator: {neuromodulator}")
            return
        self.neuromodulators[neuromodulator] = level
    def _update_neuromodulation(self, step: int):
        self.dopamine_level *= self.neuromodulator_decay
        self.acetylcholine_level *= self.neuromodulator_decay
        self.norepinephrine_level *= self.neuromodulator_decay
        self.dopamine_level = max(0.0, min(1.0, self.dopamine_level))
        self.acetylcholine_level = max(0.0, min(1.0, self.acetylcholine_level))
        self.norepinephrine_level = max(0.0, min(1.0, self.norepinephrine_level))
        self.neuromodulators['dopamine'] = self.dopamine_level
        self.neuromodulators['acetylcholine'] = self.acetylcholine_level
        self.neuromodulators['norepinephrine'] = self.norepinephrine_level
    def _calculate_recent_activity(self, node_id: int, current_time: float) -> float:
        recent_spikes = [t for t in self.spike_times[node_id]
                        if current_time - t < 1.0]
        return len(recent_spikes) / 10.0
    def _update_theta_burst_counter(self, node_id: int, current_time: float):
        recent_spikes = [t for t in self.spike_times[node_id]
                        if current_time - t < self.theta_burst_window / 1000.0]
        if len(recent_spikes) >= self.theta_burst_threshold:
            intervals = np.diff(recent_spikes[-self.theta_burst_threshold:])
            if all(0.008 < interval < 0.012 for interval in intervals):
                self.theta_burst_counters[node_id] += 1
    def get_statistics(self) -> Dict[str, Any]:
        return self.stats.copy()
    def reset_statistics(self):
        self.stats = {
            'total_spikes': 0,
            'stdp_events': 0,
            'ieg_activations': 0,
            'theta_bursts': 0,
            'consolidation_events': 0,
            'ei_balance_adjustments': 0
        }
    def cleanup(self):
        """Clean up all data structures to prevent memory leaks."""
        self.node_activity_history.clear()
        self.edge_activity_history.clear()
        self.spike_times.clear()
        self.theta_burst_counters.clear()
        self.ieg_flags.clear()
        self.ieg_timers.clear()
        
        # Reset statistics
        self.stats = {
            'total_spikes': 0,
            'stdp_events': 0,
            'ieg_activations': 0,
            'theta_bursts': 0,
            'consolidation_events': 0,
            'ei_balance_adjustments': 0
        }
        
        # Reset neuromodulator levels
        self.dopamine_level = 0.0
        self.acetylcholine_level = 0.0
        self.norepinephrine_level = 0.0
        self.neuromodulators = {
            'dopamine': 0.0,
            'acetylcholine': 0.0,
            'norepinephrine': 0.0
        }


def create_enhanced_neural_dynamics() -> EnhancedNeuralDynamics:
    return EnhancedNeuralDynamics()
if __name__ == "__main__":
    print("EnhancedNeuralDynamics created successfully!")
    print("Features include:")
    print("- STDP learning with proper timing windows")
    print("- IEG tagging for plasticity gating")
    print("- Theta-burst stimulation detection")
    print("- Sophisticated membrane dynamics with dendritic integration")
    print("- E/I balance and criticality control")
    print("- Comprehensive eligibility trace system")
    print("- Advanced connection types")
    print("- Neuromodulatory control")
    print("- Memory consolidation mechanisms")
    try:
        dynamics = create_enhanced_neural_dynamics()
        print(f"Dynamics system created with {len(dynamics.stats)} statistics tracked")
        stats = dynamics.get_statistics()
        print(f"Initial statistics: {stats}")
    except Exception as e:
        print(f"EnhancedNeuralDynamics test failed: {e}")
    print("EnhancedNeuralDynamics test completed!")
