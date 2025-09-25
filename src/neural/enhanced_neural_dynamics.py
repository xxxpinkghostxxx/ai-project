
import time
import numpy as np
import numba as nb
import torch
import threading
from typing import Dict, Any, List, Optional

from torch_geometric.data import Data
from collections import defaultdict, deque

from src.utils.logging_utils import log_step
import logging
from config.unified_config_manager import get_learning_config, get_system_constants, get_enhanced_nodes_config
from src.energy.energy_constants import ConnectionConstants
from src.energy.node_access_layer import NodeAccessLayer
from src.utils.event_bus import get_event_bus


class EnhancedNeuralDynamics:

    def __init__(self):
        # Thread safety
        self._lock = threading.RLock()

        try:
            self.config = get_learning_config()
            self.system_constants = get_system_constants()
            self.enhanced_config = get_enhanced_nodes_config()
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            # Use default values
            self.config = {}
            self.system_constants = {}
            self.enhanced_config = {}

        # Validate and set parameters with bounds checking
        self.stdp_window = self._validate_float(self.config.get('stdp_window', 20.0), 1.0, 1000.0, 'stdp_window')
        self.ltp_rate = self._validate_float(self.config.get('ltp_rate', 0.02), 0.0, 1.0, 'ltp_rate')
        self.ltd_rate = self._validate_float(self.config.get('ltd_rate', 0.01), 0.0, 1.0, 'ltd_rate')
        self.tau_plus = self._validate_float(20.0, 1.0, 1000.0, 'tau_plus')
        self.tau_minus = self._validate_float(20.0, 1.0, 1000.0, 'tau_minus')
        self.ieg_threshold = self._validate_float(0.7, 0.0, 1.0, 'ieg_threshold')
        self.ieg_decay_rate = self._validate_float(0.99, 0.0, 1.0, 'ieg_decay_rate')
        self.ieg_duration = self._validate_float(300.0, 1.0, 10000.0, 'ieg_duration')
        self.theta_burst_threshold = max(1, int(4))  # At least 1
        self.theta_burst_window = self._validate_float(50.0, 1.0, 1000.0, 'theta_burst_window')
        self.theta_frequency = self._validate_float(100.0, 1.0, 1000.0, 'theta_frequency')
        self.membrane_time_constant = self._validate_float(10.0, 0.1, 1000.0, 'membrane_time_constant')
        self.resting_potential = self._validate_float(-70.0, -200.0, 0.0, 'resting_potential')
        self.threshold_potential = self._validate_float(-50.0, -100.0, 0.0, 'threshold_potential')
        self.reset_potential = self._validate_float(-80.0, -200.0, 0.0, 'reset_potential')
        self.refractory_period = self._validate_float(2.0, 0.1, 100.0, 'refractory_period')
        self.dendritic_time_constant = self._validate_float(50.0, 0.1, 1000.0, 'dendritic_time_constant')
        self.dendritic_threshold = self._validate_float(0.6, 0.0, 10.0, 'dendritic_threshold')
        self.target_ei_ratio = self._validate_float(0.8, 0.1, 10.0, 'target_ei_ratio')
        self.ei_adaptation_rate = self._validate_float(0.001, 0.0, 1.0, 'ei_adaptation_rate')
        self.criticality_target = self._validate_float(1.0, 0.1, 5.0, 'criticality_target')
        self.eligibility_tau = self._validate_float(1000.0, 1.0, 10000.0, 'eligibility_tau')
        self.consolidation_threshold = self._validate_float(0.5, 0.0, 1.0, 'consolidation_threshold')
        self.dopamine_level = 0.0
        self.acetylcholine_level = 0.0
        self.norepinephrine_level = 0.0
        self.neuromodulator_decay = self._validate_float(0.95, 0.0, 1.0, 'neuromodulator_decay')

        self.neuromodulators = {
            'dopamine': self.dopamine_level,
            'acetylcholine': self.acetylcholine_level,
            'norepinephrine': self.norepinephrine_level
        }

        # Memory-efficient data structures with size limits
        self.node_activity_history = defaultdict(lambda: deque(maxlen=1000))
        self.edge_activity_history = defaultdict(lambda: deque(maxlen=1000))
        self.spike_times = defaultdict(list)
        self.theta_burst_counters = defaultdict(int)
        self.ieg_flags = defaultdict(bool)
        self.ieg_timers = defaultdict(float)

        # Thread-safe statistics
        self.stats = {
            'total_spikes': 0,
            'stdp_events': 0,
            'ieg_activations': 0,
            'theta_bursts': 0,
            'consolidation_events': 0,
            'ei_balance_adjustments': 0
        }

        try:
            self.event_bus = get_event_bus()
        except Exception as e:
            logging.warning(f"Failed to get event bus: {e}")
            self.event_bus = None

        log_step("EnhancedNeuralDynamics initialized with bounds checking and thread safety")

    def _validate_float(self, value: Any, min_val: float, max_val: float, field_name: str) -> float:
        """Validate and clamp float values."""
        try:
            val = float(value)
            if np.isnan(val):
                logging.warning(f"Value {val} for {field_name} is NaN, using default")
                return min_val if min_val > 0 else 0.0
            if not (min_val <= val <= max_val):
                logging.warning(f"Value {val} for {field_name} out of range [{min_val}, {max_val}], clamping")
                val = max(min_val, min(max_val, val))
            return val
        except (ValueError, TypeError, OverflowError):
            logging.warning(f"Invalid value {value} for {field_name}, using default")
            return min_val if min_val > 0 else 0.0

    def update_neural_dynamics(self, graph: Data, step: int) -> Data:
        """Update neural dynamics with comprehensive validation and thread safety."""
        # Input validation
        if graph is None:
            log_step("Error: graph is None in update_neural_dynamics")
            return graph

        if not isinstance(step, int) or step < 0:
            log_step("Error: invalid step value", step=step)
            return graph

        if not hasattr(graph, 'node_labels') or graph.node_labels is None:
            log_step("Error: graph missing node_labels")
            return graph

        if not isinstance(graph.node_labels, (list, tuple)) or len(graph.node_labels) == 0:
            log_step("Error: graph node_labels is empty or invalid")
            return graph

        with self._lock:
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
                logging.error(f"Error in neural dynamics update at step {step}: {e}")
                log_step("Error in neural dynamics update", error=str(e), step=step)
                return graph
    def _update_membrane_dynamics(self, graph: Data, step: int) -> Data:

        access_layer = NodeAccessLayer(graph)
        current_time = time.time()
        enhanced_node_ids = []
        if hasattr(graph, 'enhanced_node_ids'):
            enhanced_node_ids = graph.enhanced_node_ids
        else:
            # Optimized sampling for enhanced nodes
            sample_size = min(1000, len(graph.node_labels))
            if sample_size < len(graph.node_labels):
                sample_indices = np.random.choice(len(graph.node_labels), sample_size, replace=False)
            else:
                sample_indices = range(len(graph.node_labels))

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
        """Calculate synaptic input with bounds checking and error handling."""
        total_input = 0.0

        if not hasattr(graph, 'edge_attributes') or graph.edge_attributes is None:
            return total_input

        if not isinstance(graph.edge_attributes, (list, tuple)):
            logging.warning("edge_attributes is not a list or tuple")
            return total_input

        max_input = 1000.0  # Prevent runaway values
        min_input = -1000.0

        try:
            for edge_idx, edge in enumerate(graph.edge_attributes):
                if edge is None:
                    continue

                # Validate edge structure
                if not hasattr(edge, 'target') or not hasattr(edge, 'source'):
                    continue

                if edge.target != node_id:
                    continue

                source_id = edge.source

                # Validate source_id
                if not isinstance(source_id, int) or source_id < 0:
                    continue

                try:
                    source_node = access_layer.get_node_by_id(source_id)
                    if source_node is None:
                        continue

                    last_spike = source_node.get('last_spike_time', 0)
                    if not isinstance(last_spike, (int, float)):
                        continue

                    current_time = time.time()
                    time_diff = current_time - last_spike

                    # Only consider recent spikes
                    if time_diff < 0.1 and time_diff >= 0:
                        # Get effective weight with bounds checking
                        if hasattr(edge, 'get_effective_weight'):
                            try:
                                effective_weight = edge.get_effective_weight()
                                if not isinstance(effective_weight, (int, float)):
                                    continue
                                effective_weight = max(-100.0, min(100.0, effective_weight))  # Clamp weight
                            except Exception:
                                continue
                        else:
                            # Fallback to weight attribute
                            effective_weight = getattr(edge, 'weight', 0.0)
                            if not isinstance(effective_weight, (int, float)):
                                continue

                        # Process based on edge type
                        edge_type = getattr(edge, 'type', 'unknown')
                        if edge_type == 'excitatory':
                            total_input += effective_weight
                        elif edge_type == 'inhibitory':
                            total_input -= effective_weight
                        elif edge_type == 'modulatory':
                            total_input += effective_weight * 0.5
                        elif edge_type == 'gated':
                            try:
                                source_energy = access_layer.get_node_energy(source_id)
                                gate_threshold = getattr(edge, 'gate_threshold', 0.5)
                                if (source_energy is not None and
                                    isinstance(source_energy, (int, float)) and
                                    source_energy > gate_threshold):
                                    total_input += effective_weight
                            except Exception:
                                pass  # Skip on error

                        # Prevent runaway values
                        total_input = max(min_input, min(max_input, total_input))

                except Exception as e:
                    logging.debug(f"Error processing edge {edge_idx} for node {node_id}: {e}")
                    continue

        except Exception as e:
            logging.warning(f"Error in synaptic input calculation for node {node_id}: {e}")

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
        for edge_idx, edge in enumerate(graph.edge_attributes):
            source_index = edge.source
            target_index = edge.target
            if source_index < len(graph.node_labels) and target_index < len(graph.node_labels):
                source_node = graph.node_labels[source_index]
                target_node = graph.node_labels[target_index]
                source_id = source_node.get('id')
                target_id = target_node.get('id')
                if source_id is not None and target_id is not None:
                    source_spikes = [t for t in self.spike_times[source_id]
                                    if current_time - t < self.stdp_window / 1000.0]
                    target_spikes = [t for t in self.spike_times[target_id]
                                    if current_time - t < self.stdp_window / 1000.0]
                else:
                    continue
            else:
                continue
            if not source_spikes or not target_spikes:
                continue
                
            # Diagnostic: Log STDP attempt
            logging.debug(f"[DYNAMICS] STDP for edge {edge_idx}: source={source_id} ({len(source_spikes)} spikes), target={target_id} ({len(target_spikes)} spikes)")
            source_array = np.array(source_spikes, dtype=np.float64)
            target_array = np.array(target_spikes, dtype=np.float64)
            stdp_window_s = self.stdp_window / 1000.0
            tau_plus_s = self.tau_plus / 1000.0
            tau_minus_s = self.tau_minus / 1000.0
            weight_change = self.compute_stdp_weight_change(source_array, target_array, self.ltp_rate, self.ltd_rate, tau_plus_s, tau_minus_s, stdp_window_s)
            if abs(weight_change) > 0.001:
                # Check if source/target nodes still exist (post-death validation)
                access_layer = NodeAccessLayer(graph)
                source_node = access_layer.get_node_by_id(source_id)
                target_node = access_layer.get_node_by_id(target_id)
                if source_node is None or target_node is None:
                    logging.warning(f"[DYNAMICS] Skipping STDP for invalid edge {edge_idx}: source_id={source_id} (valid={source_node is not None}), target_id={target_id} (valid={target_node is not None})")
                    continue
                
                new_weight = max(ConnectionConstants.WEIGHT_MIN,
                               min(ConnectionConstants.WEIGHT_CAP_MAX,
                                   edge.weight + weight_change))
                edge.weight = new_weight
                edge.update_eligibility_trace(weight_change)
                self.stats['stdp_events'] += 1
                logging.info(f"[DYNAMICS] STDP weight update: source={source_id}, target={target_id}, change={weight_change:.4f}, new_weight={new_weight:.4f}")
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
        """Calculate E/I ratio with division by zero protection."""
        if not hasattr(graph, 'edge_attributes') or graph.edge_attributes is None:
            return 1.0

        if not isinstance(graph.edge_attributes, (list, tuple)):
            return 1.0

        excitatory_weight = 0.0
        inhibitory_weight = 0.0

        try:
            for edge in graph.edge_attributes:
                if edge is None or not hasattr(edge, 'type'):
                    continue

                edge_type = edge.type
                if not hasattr(edge, 'weight'):
                    continue

                weight = edge.weight
                if not isinstance(weight, (int, float)):
                    continue

                weight_abs = abs(weight)

                if edge_type == 'excitatory':
                    excitatory_weight += weight_abs
                elif edge_type == 'inhibitory':
                    inhibitory_weight += weight_abs

            # Prevent division by zero with minimum threshold
            if inhibitory_weight < 0.001:
                return 10.0 if excitatory_weight > 0 else 1.0

            ratio = excitatory_weight / inhibitory_weight

            # Clamp ratio to reasonable bounds
            return max(0.1, min(10.0, ratio))

        except Exception as e:
            logging.warning(f"Error calculating E/I ratio: {e}")
            return 1.0
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
        """Minimal safe implementation for criticality adjustment."""
        if not hasattr(graph, 'edge_attributes'):
            return
        
        # Simple adjustment: add/remove random edges to move toward target
        target_diff = self.criticality_target - current_criticality
        if abs(target_diff) < 0.1:  # Close enough
            return
            
        # Add or remove a few edges based on direction
        if target_diff > 0:  # Need to increase criticality
            # Add a few random edges
            for _ in range(3):
                if len(graph.node_labels) > 1:
                    source = np.random.randint(0, len(graph.node_labels))
                    target = np.random.randint(0, len(graph.node_labels))
                    if source != target:
                        # Simple edge addition (minimal implementation)
                        if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
                            new_edge = torch.tensor([[source], [target]], dtype=torch.long)
                            graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
        else:  # Need to decrease criticality
            # Remove a few random edges
            if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
                num_edges = graph.edge_index.shape[1]
                if num_edges > 0:
                    remove_count = min(3, num_edges)
                    keep_indices = np.random.choice(num_edges, num_edges - remove_count, replace=False)
                    graph.edge_index = graph.edge_index[:, keep_indices]
    def set_neuromodulator_level(self, neuromodulator: str, level: float):
        """Set neuromodulator level with validation."""
        if not isinstance(neuromodulator, str):
            logging.error("Neuromodulator name must be a string")
            return

        if not isinstance(level, (int, float)):
            logging.error("Neuromodulator level must be a number")
            return

        # Validate and clamp level
        level = float(level)
        if np.isnan(level):
            logging.warning(f"Neuromodulator level is NaN, setting to 0.0")
            level = 0.0
        else:
            level = max(0.0, min(1.0, level))

        with self._lock:
            if neuromodulator == 'dopamine':
                self.dopamine_level = level
            elif neuromodulator == 'acetylcholine':
                self.acetylcholine_level = level
            elif neuromodulator == 'norepinephrine':
                self.norepinephrine_level = level
            else:
                logging.warning(f"Unknown neuromodulator: {neuromodulator}")
                return

            self.neuromodulators[neuromodulator] = level
            log_step(f"Neuromodulator {neuromodulator} set to {level}")
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
        """Get statistics with thread safety."""
        with self._lock:
            return self.stats.copy()

    def reset_statistics(self):
        """Reset statistics with thread safety."""
        with self._lock:
            self.stats = {
                'total_spikes': 0,
                'stdp_events': 0,
                'ieg_activations': 0,
                'theta_bursts': 0,
                'consolidation_events': 0,
                'ei_balance_adjustments': 0
            }

    def cleanup(self):
        """Clean up all data structures to prevent memory leaks with thread safety."""
        with self._lock:
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

            log_step("EnhancedNeuralDynamics cleanup completed")


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







