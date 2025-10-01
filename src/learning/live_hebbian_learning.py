"""Live Hebbian Learning module for neural network simulation with energy-based modulation."""

import time
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import Data

from src.energy.energy_behavior import get_node_energy_cap
from src.utils.logging_utils import log_step


class LiveHebbianLearning:
    """Class implementing live Hebbian learning with STDP and energy modulation."""

    def __init__(self, simulation_manager=None):

        self.simulation_manager = simulation_manager
        self.learning_active = True
        self.base_learning_rate = 0.01  # Base learning rate
        self.learning_rate = 0.01       # Current learning rate (energy-modulated)
        self.stdp_window = 0.1
        self.eligibility_decay = 0.95
        self.weight_cap = 5.0
        self.weight_min = 0.1
        self.energy_learning_modulation = True  # Enable energy-based learning modulation
        self.learning_stats = {
            'total_weight_changes': 0,
            'stdp_events': 0,
            'connection_strengthened': 0,
            'connection_weakened': 0,
            'learning_rate_updates': 0,
            'energy_modulated_events': 0
        }
        self.node_activity_history = {}
        self.edge_activity_history = {}
        self.last_activity_time = {}
        log_step("LiveHebbianLearning initialized with energy modulation",
                base_learning_rate=self.base_learning_rate,
                stdp_window=self.stdp_window,
                energy_modulation=self.energy_learning_modulation)
    def apply_continuous_learning(self, graph: Data, step: int) -> Data:
        """Apply continuous learning to the neural graph."""
        if not self.learning_active:
            return graph
        try:
            self._update_activity_history(graph, step)
            graph = self._apply_stdp_learning(graph, step)
            graph = self._update_eligibility_traces(graph, step)
            graph = self._consolidate_weights(graph, step)
            self._update_learning_statistics()
            return graph
        except (AttributeError, KeyError, ValueError, IndexError, RuntimeError) as e:
            log_step("Error applying continuous learning", error=str(e))
            return graph
    def _update_activity_history(self, graph: Data, _step: int):

        try:
            current_time = time.time()
            if hasattr(graph, 'x') and graph.x is not None:
                all_energies = graph.x[:, 0]

                # Energy-based activity detection
                if self.energy_learning_modulation:
                    # Use energy levels directly - higher energy = more likely to be active
                    # Sample nodes based on energy probability distribution
                    energy_probs = all_energies / all_energies.sum() if all_energies.sum() > 0 else torch.ones_like(all_energies) / len(all_energies)
                    max_nodes_to_process = min(100, len(all_energies))

                    # Sample nodes proportionally to their energy levels
                    selected_indices = torch.multinomial(energy_probs, max_nodes_to_process, replacement=False)
                else:
                    # Fallback to threshold-based detection
                    default_threshold = 0.5
                    active_mask = all_energies > default_threshold
                    active_indices = torch.where(active_mask)[0]
                    max_nodes_to_process = min(100, len(active_indices))
                    selected_indices = active_indices[:max_nodes_to_process]

                for idx in selected_indices:
                    node_id = graph.node_labels[idx].get('id', idx.item())
                    if node_id not in self.node_activity_history:
                        self.node_activity_history[node_id] = []
                    self.node_activity_history[node_id].append(current_time)
                    self.last_activity_time[node_id] = current_time
                    cutoff_time = current_time - self.stdp_window
                    activity_list = self.node_activity_history[node_id]
                    while activity_list and activity_list[0] <= cutoff_time:
                        activity_list.pop(0)
        except (AttributeError, KeyError, ValueError, IndexError) as e:
            log_step("Error updating activity history", error=str(e))
    def _apply_stdp_learning(self, graph: Data, _step: int) -> Data:

        try:
            if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                return graph
            current_time = time.time()
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
            for edge_idx in range(edge_index.shape[1]):
                source_idx = edge_index[0, edge_idx].item()
                target_idx = edge_index[1, edge_idx].item()
                source_id = graph.node_labels[source_idx].get('id', source_idx)
                target_id = graph.node_labels[target_idx].get('id', target_idx)
                stdp_change = self._calculate_stdp_change(source_id, target_id, current_time)
                if abs(stdp_change) > 0.001:
                    if edge_attr is not None:
                        current_weight = edge_attr[edge_idx, 0].item()
                        new_weight = current_weight + stdp_change
                        new_weight = max(self.weight_min, min(self.weight_cap, new_weight))
                        edge_attr[edge_idx, 0] = new_weight
                        if stdp_change > 0:
                            self.learning_stats['connection_strengthened'] += 1
                        else:
                            self.learning_stats['connection_weakened'] += 1
                        self.learning_stats['stdp_events'] += 1
                        self.learning_stats['total_weight_changes'] += 1
            return graph
        except (AttributeError, KeyError, ValueError, IndexError) as e:
            log_step("Error applying STDP learning", error=str(e))
            return graph
    def _get_node_energy(self, node_id: int) -> float:
        """Get energy level for a node from the simulation manager."""
        energy = 2.0
        try:
            if self.simulation_manager and hasattr(self.simulation_manager, 'get_neural_graph'):
                graph = self.simulation_manager.get_neural_graph()
                found_energy = self._find_node_energy(graph, node_id)
                if found_energy is not None:
                    energy = found_energy
            if energy == 2.0 and isinstance(node_id, int):
                energy_value = 0.5 + 0.4 * ((node_id * 7) % 10) / 10.0
                energy = min(energy_value, 4.5)
        except (AttributeError, KeyError, ValueError, IndexError) as e:
            log_step("Error getting node energy", error=str(e))
            energy = 2.0
        return energy

    def _find_node_energy(self, graph, node_id):
        if hasattr(graph, 'node_labels'):
            for idx, node in enumerate(graph.node_labels):
                if node.get('id') == node_id:
                    if hasattr(graph, 'x') and idx < len(graph.x):
                        return graph.x[idx, 0].item()
                    return node.get('energy', 2.0)
        return None

    def _calculate_energy_modulated_learning_rate(self, source_id: int, target_id: int) -> float:
        """Calculate learning rate modulated by node energy levels."""
        if not self.energy_learning_modulation:
            return self.base_learning_rate

        try:
            source_energy = self._get_node_energy(source_id)
            target_energy = self._get_node_energy(target_id)

            # Energy modulation: higher energy = higher learning rate
            # Base learning rate is modulated by average energy of pre/post synaptic neurons
            avg_energy = (source_energy + target_energy) / 2.0

            # Get energy cap for normalization
            energy_cap = 5.0  # Updated default to match new config
            try:
                energy_cap = get_node_energy_cap()
                if energy_cap <= 0:
                    energy_cap = 5.0  # Fallback to new default
            except ImportError:
                pass

            # Normalize energy and apply modulation
            # Enhanced range: 0.3x to 1.5x base rate for more pronounced effects
            normalized_energy = min(avg_energy / energy_cap, 1.0) if energy_cap > 0 else 0.5
            modulated_rate = self.base_learning_rate * (0.3 + 1.2 * normalized_energy)

            return modulated_rate

        except (AttributeError, KeyError, ValueError, ImportError) as e:
            log_step("Error calculating energy-modulated learning rate", error=str(e))
            return self.base_learning_rate

    def _calculate_stdp_change(self, source_id: int, target_id: int, _current_time: float) -> float:

        try:
            source_activity = self.node_activity_history.get(source_id, [])
            target_activity = self.node_activity_history.get(target_id, [])
            if not source_activity or not target_activity:
                return 0.0

            # Get energy-modulated learning rate
            energy_modulated_rate = self._calculate_energy_modulated_learning_rate(source_id, target_id)

            stdp_change = 0.0
            for source_time in source_activity:
                for target_time in target_activity:
                    time_diff = target_time - source_time
                    if 0 < time_diff < self.stdp_window:
                        ltp_strength = np.exp(-time_diff / (self.stdp_window / 3))
                        stdp_change += energy_modulated_rate * ltp_strength
                    elif -self.stdp_window < time_diff < 0:
                        ltd_strength = np.exp(time_diff / (self.stdp_window / 3))
                        stdp_change -= energy_modulated_rate * ltd_strength * 0.5

            # Track energy-modulated events
            if abs(stdp_change) > 0.001:
                self.learning_stats['energy_modulated_events'] += 1

            return stdp_change
        except (AttributeError, KeyError, ValueError, IndexError) as e:
            log_step("Error calculating STDP change", error=str(e))
            return 0.0
    def _update_eligibility_traces(self, graph: Data, _step: int) -> Data:

        try:
            if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                return graph
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
            if edge_attr is None:
                return graph
            for edge_idx in range(edge_index.shape[1]):
                _source_idx = edge_index[0, edge_idx].item()
                _target_idx = edge_index[1, edge_idx].item()
                if edge_attr.shape[1] > 1:
                    current_trace = edge_attr[edge_idx, 1].item()
                    new_trace = current_trace * self.eligibility_decay
                    edge_attr[edge_idx, 1] = new_trace
            return graph
        except (AttributeError, KeyError, ValueError, IndexError) as e:
            log_step("Error updating eligibility traces", error=str(e))
            return graph
    def _consolidate_weights(self, graph: Data, step: int) -> Data:

        try:
            if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                return graph
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
            if edge_attr is None:
                return graph
            if step % 100 == 0:
                for edge_idx in range(edge_index.shape[1]):
                    current_weight = edge_attr[edge_idx, 0].item()
                    consolidation_factor = 0.999
                    new_weight = current_weight * consolidation_factor
                    new_weight = max(self.weight_min, min(self.weight_cap, new_weight))
                    edge_attr[edge_idx, 0] = new_weight
            return graph
        except (AttributeError, KeyError, ValueError, IndexError) as e:
            log_step("Error consolidating weights", error=str(e))
            return graph
    def _update_learning_statistics(self):
        try:
            total_events = self.learning_stats['stdp_events']
            if total_events > 0:
                efficiency = (self.learning_stats['connection_strengthened'] / total_events) * 100
                self.learning_stats['learning_efficiency'] = efficiency
            if total_events > 1000:
                if efficiency < 30:
                    self.learning_rate = min(0.05, self.learning_rate * 1.1)
                    self.learning_stats['learning_rate_updates'] += 1
                elif efficiency > 70:
                    self.learning_rate = max(0.001, self.learning_rate * 0.9)
                    self.learning_stats['learning_rate_updates'] += 1
        except (AttributeError, KeyError, ValueError, ZeroDivisionError) as e:
            log_step("Error updating learning statistics", error=str(e))
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Return a copy of the learning statistics."""
        return self.learning_stats.copy()
    def reset_learning_statistics(self):
        """Reset all learning statistics to zero."""
        self.learning_stats = {
            'total_weight_changes': 0,
            'stdp_events': 0,
            'connection_strengthened': 0,
            'connection_weakened': 0,
            'learning_rate_updates': 0,
            'learning_efficiency': 0.0,
            'energy_modulated_events': 0
        }
    def set_learning_rate(self, learning_rate: float):
        """Set the learning rate within bounds."""
        self.learning_rate = max(0.001, min(0.1, learning_rate))
        log_step("Learning rate updated", new_rate=self.learning_rate)
    def set_learning_active(self, active: bool):
        """Set whether learning is active."""
        self.learning_active = active
        log_step("Learning active state changed", active=active)
    def get_learning_parameters(self) -> Dict[str, float]:
        """Return a dictionary of current learning parameters."""
        return {
            'learning_rate': self.learning_rate,
            'base_learning_rate': self.base_learning_rate,
            'stdp_window': self.stdp_window,
            'eligibility_decay': self.eligibility_decay,
            'weight_cap': self.weight_cap,
            'weight_min': self.weight_min,
            'learning_active': self.learning_active,
            'energy_learning_modulation': self.energy_learning_modulation
        }
    def cleanup(self):
        """Clean up internal data structures."""
        self.node_activity_history.clear()
        self.edge_activity_history.clear()
        self.last_activity_time.clear()
        log_step("LiveHebbianLearning cleanup completed")


def create_live_hebbian_learning(simulation_manager=None) -> LiveHebbianLearning:
    """Create a new LiveHebbianLearning instance."""
    return LiveHebbianLearning(simulation_manager)
if __name__ == "__main__":
    print("LiveHebbianLearning created successfully!")
    print("Features include:")
    print("- Real-time STDP learning")
    print("- Continuous weight updates")
    print("- Eligibility trace management")
    print("- Learning statistics tracking")
    print("- Adaptive learning rate")
    try:
        learning_system = create_live_hebbian_learning()
        params = learning_system.get_learning_parameters()
        print(f"Learning parameters: {params}")
        stats = learning_system.get_learning_statistics()
        print(f"Learning statistics: {stats}")
    except (AttributeError, KeyError, ValueError) as e:
        print(f"LiveHebbianLearning test failed: {e}")
    print("LiveHebbianLearning test completed!")







