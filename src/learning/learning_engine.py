
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from src.utils.logging_utils import log_step


from config.unified_config_manager import get_learning_config
from src.utils.event_bus import get_event_bus
from typing import Dict, Any, Optional, List, Tuple, Union
from src.core.interfaces.node_access_layer import IAccessLayer
from src.energy.energy_behavior import get_node_energy_cap


class LearningEngine:
    """
    Advanced learning engine that implements energy-modulated synaptic plasticity.

    This engine handles STDP (Spike-Timing Dependent Plasticity), memory consolidation,
    and energy-based learning modulation to create biologically plausible learning dynamics.
    """

    def __init__(self, access_layer: IAccessLayer) -> None:
        """
        Initialize the LearningEngine with configuration parameters.

        Sets up learning rates, thresholds, and event handling for synaptic plasticity.
        """
        config = get_learning_config()

        # Core learning parameters
        self.learning_rate: float = config.get('plasticity_rate', 0.01)
        self.eligibility_decay: float = config.get('eligibility_decay', 0.95)
        self.stdp_window: float = config.get('stdp_window', 20.0)  # milliseconds
        self.ltp_rate: float = config.get('ltp_rate', 0.02)  # Long-term potentiation rate
        self.ltd_rate: float = config.get('ltd_rate', 0.01)  # Long-term depression rate

        # Learning thresholds
        self.plasticity_threshold: float = 0.1
        self.consolidation_threshold: float = 0.5

        # Energy modulation settings
        self.energy_learning_modulation: bool = True

        # Learning statistics tracking
        self.learning_stats: Dict[str, Union[int, float]] = {
            'stdp_events': 0,
            'weight_changes': 0,
            'consolidation_events': 0,
            'memory_traces_formed': 0,
            'total_weight_change': 0.0,
            'energy_modulated_events': 0
        }

        # Memory system
        self.memory_traces: Dict[int, Dict[str, Any]] = {}
        self.memory_decay_rate: float = 0.99

        # Event handling
        self.event_bus = get_event_bus()
        self.access_layer = access_layer
        self.event_bus.subscribe('SPIKE', self._on_spike)
    def _get_node_energy(self, node: Optional[Dict[str, Any]]) -> float:
        """
        Get energy level for a node with optimized access.

        Args:
            node: Node dictionary containing energy information

        Returns:
            float: Energy level of the node, or 0.0 if unavailable
        """
        if node is None:
            return 0.0
        # Prioritize energy field, fallback to membrane_potential
        energy = node.get('energy')
        if energy is not None:
            return energy
        return node.get('membrane_potential', 0.0)

    def _calculate_energy_modulated_rate(self, pre_node: Optional[Dict[str, Any]],
                                        post_node: Optional[Dict[str, Any]],
                                        base_rate: float) -> float:
        """
        Calculate learning rate modulated by node energy levels.

        Higher energy nodes exhibit enhanced synaptic plasticity, creating a biologically
        plausible mechanism where active neurons learn more effectively.

        Args:
            pre_node: Pre-synaptic node dictionary
            post_node: Post-synaptic node dictionary
            base_rate: Base learning rate to modulate

        Returns:
            float: Energy-modulated learning rate
        """
        if not self.energy_learning_modulation:
            return base_rate

        try:
            pre_energy = self._get_node_energy(pre_node)
            post_energy = self._get_node_energy(post_node)

            # Calculate average energy for modulation
            avg_energy = (pre_energy + post_energy) / 2.0

            # Get energy cap for normalization
            energy_cap = get_node_energy_cap()
            if energy_cap <= 0:
                energy_cap = 5.0  # Fallback to new default

            # Normalize energy and apply modulation
            # Range: 1.0x to 1.5x base rate
            normalized_energy = min(avg_energy / energy_cap, 1.0) if energy_cap > 0 else 0.5
            modulated_rate = base_rate * (1.0 + 0.5 * normalized_energy)

            return modulated_rate

        except Exception as e:
            log_step("Error calculating energy-modulated rate", error=str(e))
            return base_rate

    def apply_timing_learning(self, pre_node: Optional[Dict[str, Any]],
                            post_node: Optional[Dict[str, Any]],
                            edge: Optional[Any],
                            delta_t: float) -> float:
        """
        Apply Spike-Timing Dependent Plasticity (STDP) learning.

        Implements energy-modulated STDP where synaptic strength changes based on
        the relative timing of pre- and post-synaptic spikes, with modulation by
        the energy levels of the connected neurons.

        Args:
            pre_node: Pre-synaptic neuron node data
            post_node: Post-synaptic neuron node data
            edge: Synaptic connection edge object
            delta_t: Time difference between spikes (post - pre) in seconds

        Returns:
            float: Weight change applied to the synapse
        """
        # STDP window in milliseconds, delta_t in milliseconds
        stdp_window_ms = self.stdp_window

        if delta_t == 0:
            return 0.0  # No learning for simultaneous spikes

        if abs(delta_t) <= stdp_window_ms:
            if delta_t > 0:
                # Long-Term Potentiation (LTP): Pre before Post
                modulated_ltp_rate = self._calculate_energy_modulated_rate(pre_node, post_node, self.ltp_rate)
                weight_change = modulated_ltp_rate * np.exp(-delta_t / 10.0)

                self.learning_stats['stdp_events'] += 1
                if modulated_ltp_rate != self.ltp_rate:
                    self.learning_stats['energy_modulated_events'] += 1

                # Log LTP event with reduced frequency for performance
                if self.learning_stats['stdp_events'] % 100 == 0:
                    log_step("LTP applied with energy modulation",
                            pre_id=pre_node.get('id', '?') if pre_node else '?',
                            post_id=post_node.get('id', '?') if post_node else '?',
                            delta_t=delta_t,
                            weight_change=weight_change,
                            modulated_rate=modulated_ltp_rate)
            else:
                # Long-Term Depression (LTD): Post before Pre
                modulated_ltd_rate = self._calculate_energy_modulated_rate(pre_node, post_node, self.ltd_rate)
                weight_change = -modulated_ltd_rate * np.exp(delta_t / 10.0)

                self.learning_stats['stdp_events'] += 1
                if modulated_ltd_rate != self.ltd_rate:
                    self.learning_stats['energy_modulated_events'] += 1

                # Log LTD event with reduced frequency for performance
                if self.learning_stats['stdp_events'] % 100 == 0:
                    log_step("LTD applied with energy modulation",
                            pre_id=pre_node.get('id', '?') if pre_node else '?',
                            post_id=post_node.get('id', '?') if post_node else '?',
                            delta_t=delta_t,
                            weight_change=weight_change,
                            modulated_rate=modulated_ltd_rate)

            # Update edge eligibility trace if edge exists
            if edge:
                edge.eligibility_trace += weight_change

            return weight_change

        return 0.0  # No learning if timing difference is too large

    def _on_spike(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle SPIKE event and apply timing learning."""
        try:
            node_id = data['node_id']
            source_id = data.get('source_id', node_id)
            pre_node = self.access_layer.get_node_by_id(source_id)
            post_node = self.access_layer.get_node_by_id(node_id)
            # Dummy edge and delta_t; in full impl, calculate delta_t from timestamps
            edge = None
            delta_t = 0.0  # Would calculate from last spikes
            change = self.apply_timing_learning(pre_node, post_node, edge, delta_t)
            self.emit_learning_update(source_id, node_id, change)
        except Exception as e:
            log_step("Error emitting learning update event", error=str(e), source_id=source_id, target_id=node_id)

    def emit_learning_update(self, source_id: int, target_id: int, weight_change: float) -> None:
        """Emit LEARNING_UPDATE event."""
        try:
            self.event_bus.emit('LEARNING_UPDATE', {
                'source_id': source_id,
                'target_id': target_id,
                'weight_change': weight_change
            })
        except Exception as e:
            log_step("Error handling spike event", error=str(e), source_id=source_id, target_id=target_id)

    def consolidate_connections(self, graph: Any) -> Any:

        if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
            return graph
        consolidated_count = 0
        total_weight_change = 0.0
        for edge_idx, edge in enumerate(graph.edge_attributes):
            if edge.eligibility_trace > self.consolidation_threshold:
                # Energy-modulated consolidation
                base_weight_change = edge.eligibility_trace * self.learning_rate

                # Get energy levels of connected nodes for modulation
                pre_energy = 0.0
                post_energy = 0.0
                if hasattr(graph, 'node_labels') and edge.source < len(graph.node_labels) and edge.target < len(graph.node_labels):
                    pre_node = graph.node_labels[edge.source]
                    post_node = graph.node_labels[edge.target]
                    pre_energy = self._get_node_energy(pre_node)
                    post_energy = self._get_node_energy(post_node)

                # Modulate consolidation by average energy
                energy_cap = get_node_energy_cap()
                if energy_cap <= 0:
                    energy_cap = 5.0  # Fallback to new default

                avg_energy = (pre_energy + post_energy) / 2.0
                energy_factor = min(avg_energy / energy_cap, 1.0) if energy_cap > 0 else 0.5
                weight_change = base_weight_change * (0.3 + 1.2 * energy_factor)  # Enhanced range: 0.3x to 1.5x

                new_weight = edge.weight + weight_change
                new_weight = max(0.1, min(5.0, new_weight))
                actual_change = new_weight - edge.weight
                edge.weight = new_weight
                edge.eligibility_trace *= 0.5
                consolidated_count += 1
                total_weight_change += actual_change
                self.learning_stats['consolidation_events'] += 1
                if energy_factor > 0.5:
                    self.learning_stats['energy_modulated_events'] += 1
                self.emit_learning_update(edge.source, edge.target, actual_change)
                log_step("Connection consolidated with energy modulation",
                        edge_idx=edge_idx,
                        source=edge.source,
                        target=edge.target,
                        weight_change=actual_change,
                        new_weight=new_weight,
                        energy_factor=energy_factor)
        self.learning_stats['weight_changes'] += consolidated_count
        self.learning_stats['total_weight_change'] += total_weight_change
        return graph
    def form_memory_traces(self, graph: Any) -> Any:

        if not hasattr(graph, 'edge_attributes'):
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
    def _has_stable_pattern(self, node: Any, graph: Any) -> bool:

        last_activation = node.get('last_activation', 0)
        current_time = time.time()
        energy = self._get_node_energy(node)

        # Energy-modulated stability criteria
        energy_cap = get_node_energy_cap()
        if energy_cap <= 0:
            energy_cap = 5.0  # Fallback to new default

        # Higher energy nodes have lower stability threshold (more plastic)
        stability_threshold = 0.3 + (0.4 * (1.0 - min(energy / energy_cap, 1.0))) if energy_cap > 0 else 0.3

        time_since_activation = current_time - last_activation
        time_criterion = time_since_activation < 10.0
        energy_criterion = energy > stability_threshold

        return time_criterion and energy_criterion
    def _create_memory_trace(self, node_idx: int, graph: Any) -> None:

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
    def apply_memory_influence(self, graph: Any) -> Any:

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
    def _reinforce_memory_pattern(self, node_idx: int, memory_trace: Dict[str, Any], graph: Any) -> None:

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
    def get_learning_statistics(self) -> Dict[str, Any]:
        return self.learning_stats.copy()
    def reset_statistics(self) -> None:
        self.learning_stats = {
            'stdp_events': 0,
            'weight_changes': 0,
            'consolidation_events': 0,
            'memory_traces_formed': 0,
            'total_weight_change': 0.0,
            'energy_modulated_events': 0
        }
    def get_memory_trace_count(self) -> int:
        return len(self.memory_traces)


def calculate_learning_efficiency(graph: Any) -> float:

    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return 0.0
    total_weights = sum(edge.weight for edge in graph.edge_attributes)
    avg_weight = total_weights / len(graph.edge_attributes)
    efficiency = min(avg_weight / 2.5, 1.0)
    return efficiency


def detect_learning_patterns(graph: Any) -> Dict[str, Any]:

    if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
        return {
            'patterns_detected': 0,
            'weight_variance': 0.0,
            'edge_type_distribution': {},
            'total_connections': 0
        }
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







