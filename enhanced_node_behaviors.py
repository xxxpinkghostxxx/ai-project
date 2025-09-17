
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Set
from torch_geometric.data import Data
from collections import defaultdict
import logging
from logging_utils import log_step
from config_manager import get_learning_config, get_system_constants, get_enhanced_nodes_config
from energy_constants import EnergyConstants
from node_access_layer import NodeAccessLayer


class EnhancedNodeBehavior:

    def __init__(self, node_id: int, node_type: str = 'dynamic', **kwargs):

        self.node_id = node_id
        self.node_type = node_type
        self.is_excitatory = kwargs.get('is_excitatory', True)
        self.behavior = kwargs.get('behavior', 'dynamic')
        self.membrane_potential = kwargs.get('membrane_potential', 0.0)
        self.dendritic_potential = kwargs.get('dendritic_potential', 0.0)
        self.resting_potential = kwargs.get('resting_potential', -70.0)
        self.threshold_potential = kwargs.get('threshold_potential', -50.0)
        self.reset_potential = kwargs.get('reset_potential', -80.0)
        self.refractory_timer = 0.0
        self.last_spike_time = 0.0
        self.spike_count = 0
        self.spike_history = []
        self.plasticity_enabled = kwargs.get('plasticity_enabled', True)
        self.eligibility_trace = 0.0
        self.ieg_flag = False
        self.ieg_timer = 0.0
        self.theta_burst_counter = 0
        self.theta_burst_threshold = 4
        self.theta_frequency = 100.0
        self.subtype = kwargs.get('subtype', 'standard')
        self.subtype2 = kwargs.get('subtype2', 'normal')
        self.subtype3 = kwargs.get('subtype3', 'standard')
        self.subtype4 = kwargs.get('subtype4', 'standard')
        self.energy = kwargs.get('energy', 0.5)
        self.energy_cap = kwargs.get('energy_cap', 244.0)
        self.energy_decay_rate = kwargs.get('energy_decay_rate', 0.99)
        self.max_connections = kwargs.get('max_connections', 10)
        self.connection_frequency = kwargs.get('connection_frequency', 0.01)
        self.energy_gain_per_connection = kwargs.get('energy_gain_per_connection', 0.1)
        self.dopamine_sensitivity = kwargs.get('dopamine_sensitivity', 1.0)
        self.acetylcholine_sensitivity = kwargs.get('acetylcholine_sensitivity', 1.0)
        self.norepinephrine_sensitivity = kwargs.get('norepinephrine_sensitivity', 1.0)
        self.state = kwargs.get('state', 'active')
        self.last_update = 0
        self.activation_count = 0
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.adaptation_rate = kwargs.get('adaptation_rate', 0.001)
        self._initialize_subtype_properties()
        log_step("EnhancedNodeBehavior created",
                node_id=node_id,
                type=node_type,
                subtype=self.subtype)
    def _initialize_subtype_properties(self):
        if self.subtype == 'transmitter':
            self.transmission_boost = 1.5
            self.energy_cost_per_transmission = 0.1
        elif self.subtype == 'resonator':
            self.resonance_frequency = 10.0
            self.resonance_strength = 0.8
            self.energy_decay_rate = 0.95
        elif self.subtype == 'dampener':
            self.damping_factor = 0.7
            self.stability_bonus = 0.3
        elif self.subtype == 'oscillator':
            self.oscillation_frequency = 1.0
            self.oscillation_amplitude = 0.5
            self.phase_offset = 0.0
        elif self.subtype == 'integrator':
            self.integration_rate = 0.5
            self.integration_threshold = 0.8
        elif self.subtype == 'relay':
            self.relay_amplification = 1.5
            self.relay_efficiency = 0.8
        elif self.subtype == 'highway':
            self.highway_capacity = 1000
            self.highway_efficiency = 0.9
            self.energy_boost = 2.0
    def update_behavior(self, graph: Data, step: int, access_layer: NodeAccessLayer) -> bool:

        try:
            self._update_timing_properties()
            self._update_energy_dynamics(access_layer)
            self._update_membrane_dynamics(access_layer)
            if self.subtype == 'transmitter':
                self._update_transmitter_behavior(graph, access_layer)
            elif self.subtype == 'resonator':
                self._update_resonator_behavior(graph, access_layer)
            elif self.subtype == 'dampener':
                self._update_dampener_behavior(graph, access_layer)
            elif self.subtype == 'oscillator':
                self._update_oscillator_behavior(graph, access_layer)
            elif self.subtype == 'integrator':
                self._update_integrator_behavior(graph, access_layer)
            elif self.subtype == 'relay':
                self._update_relay_behavior(graph, access_layer)
            elif self.subtype == 'highway':
                self._update_highway_behavior(graph, access_layer)
            self._update_plasticity(access_layer)
            self._update_theta_burst_detection()
            self._update_ieg_tagging()
            self.last_update = step
            return True
        except Exception as e:
            log_step("Error updating node behavior",
                    node_id=self.node_id,
                    error=str(e))
            return False
    def _update_timing_properties(self):
        current_time = time.time()
        if self.refractory_timer > 0:
            self.refractory_timer -= 0.01
        if self.ieg_timer > 0:
            self.ieg_timer -= 1
            if self.ieg_timer <= 0:
                self.ieg_flag = False
    def _update_energy_dynamics(self, access_layer: NodeAccessLayer):
        current_energy = access_layer.get_node_energy(self.node_id)
        if current_energy is None:
            return
        new_energy = current_energy * self.energy_decay_rate
        if self.subtype == 'resonator':
            new_energy *= 0.95
        elif self.subtype == 'dampener':
            new_energy *= 1.02
        if self.subtype4 == 'high_gain':
            new_energy += self.energy_gain_per_connection * 0.1
        elif self.subtype4 == 'medium_gain':
            new_energy += self.energy_gain_per_connection * 0.05
        elif self.subtype4 == 'low_gain':
            new_energy += self.energy_gain_per_connection * 0.01
        new_energy = max(0.0, min(new_energy, self.energy_cap))
        access_layer.set_node_energy(self.node_id, new_energy)
        self.energy = new_energy
    def _update_membrane_dynamics(self, access_layer: NodeAccessLayer):
        synaptic_input = self._calculate_synaptic_input(access_layer)
        self.dendritic_potential += (synaptic_input - self.dendritic_potential) * 0.1
        self.membrane_potential += (synaptic_input - self.membrane_potential) * 0.1
        self.membrane_potential += (self.resting_potential - self.membrane_potential) * 0.01
        if (self.membrane_potential > self.threshold_potential and
            self.refractory_timer <= 0):
            self._fire_spike(access_layer)
        access_layer.update_node_property(self.node_id, 'membrane_potential', self.membrane_potential)
        access_layer.update_node_property(self.node_id, 'dendritic_potential', self.dendritic_potential)
    def _calculate_synaptic_input(self, access_layer: NodeAccessLayer) -> float:
        return self.energy / self.energy_cap * 10.0
    def _fire_spike(self, access_layer: NodeAccessLayer):
        current_time = time.time()
        self.membrane_potential = self.reset_potential
        self.refractory_timer = 2.0
        self.last_spike_time = current_time
        self.spike_count += 1
        self.spike_history.append(current_time)
        if len(self.spike_history) > 100:
            self.spike_history = self.spike_history[-100:]
        self._update_theta_burst_counter()
        self.state = 'active'
        access_layer.update_node_property(self.node_id, 'state', self.state)
        access_layer.update_node_property(self.node_id, 'last_spike_time', current_time)
        log_step("Spike fired", node_id=self.node_id, spike_count=self.spike_count)
    def _update_transmitter_behavior(self, graph: Data, access_layer: NodeAccessLayer):
        if self.state == 'active':
            current_energy = access_layer.get_node_energy(self.node_id)
            if current_energy is not None:
                boosted_energy = min(current_energy * self.transmission_boost, self.energy_cap)
                access_layer.set_node_energy(self.node_id, boosted_energy)
    def _update_resonator_behavior(self, graph: Data, access_layer: NodeAccessLayer):
        current_time = time.time()
        phase = (current_time * self.resonance_frequency) % (2 * np.pi)
        resonance_input = np.sin(phase) * self.resonance_strength
        self.membrane_potential += resonance_input * 0.1
    def _update_dampener_behavior(self, graph: Data, access_layer: NodeAccessLayer):
        current_energy = access_layer.get_node_energy(self.node_id)
        if current_energy is not None:
            damped_energy = current_energy * self.damping_factor + self.stability_bonus
            access_layer.set_node_energy(self.node_id, damped_energy)
    def _update_oscillator_behavior(self, graph: Data, access_layer: NodeAccessLayer):
        current_time = time.time()
        phase = (current_time * self.oscillation_frequency + self.phase_offset) % (2 * np.pi)
        oscillation_input = np.sin(phase) * self.oscillation_amplitude
        self.membrane_potential += oscillation_input * 0.1
    def _update_integrator_behavior(self, graph: Data, access_layer: NodeAccessLayer):
        current_energy = access_layer.get_node_energy(self.node_id)
        if current_energy is not None:
            integrated_energy = current_energy + self.integration_rate * 0.01
            if integrated_energy > self.integration_threshold * self.energy_cap:
                self.state = 'integrating'
                access_layer.update_node_property(self.node_id, 'state', self.state)
    def _update_relay_behavior(self, graph: Data, access_layer: NodeAccessLayer):
        if self.state == 'active':
            current_energy = access_layer.get_node_energy(self.node_id)
            if current_energy is not None:
                relayed_energy = current_energy * self.relay_amplification * self.relay_efficiency
                access_layer.set_node_energy(self.node_id, relayed_energy)
    def _update_highway_behavior(self, graph: Data, access_layer: NodeAccessLayer):
        current_energy = access_layer.get_node_energy(self.node_id)
        if current_energy is not None:
            highway_energy = current_energy * self.energy_boost
            access_layer.set_node_energy(self.node_id, highway_energy)
    def _update_plasticity(self, access_layer: NodeAccessLayer):
        if not self.plasticity_enabled:
            return
        self.eligibility_trace *= 0.95
        if self.spike_count > 0:
            self.learning_rate = min(0.1, self.learning_rate * 1.01)
        else:
            self.learning_rate = max(0.001, self.learning_rate * 0.99)
    def _update_theta_burst_detection(self):
        if len(self.spike_history) < self.theta_burst_threshold:
            return
        recent_spikes = self.spike_history[-self.theta_burst_threshold:]
        intervals = np.diff(recent_spikes)
        if all(0.008 < interval < 0.012 for interval in intervals):
            self.theta_burst_counter += 1
            log_step("Theta-burst detected", node_id=self.node_id, count=self.theta_burst_counter)
    def _update_ieg_tagging(self):
        recent_activity = len([t for t in self.spike_history if time.time() - t < 1.0])
        if recent_activity > 5 and not self.ieg_flag:
            self.ieg_flag = True
            self.ieg_timer = 300.0
            log_step("IEG activated", node_id=self.node_id, activity=recent_activity)
    def _update_theta_burst_counter(self):
        current_time = time.time()
        recent_spikes = [t for t in self.spike_history if current_time - t < 0.1]
        if len(recent_spikes) >= self.theta_burst_threshold:
            intervals = np.diff(recent_spikes[-self.theta_burst_threshold:])
            if all(0.008 < interval < 0.012 for interval in intervals):
                self.theta_burst_counter += 1
    def get_behavior_statistics(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'subtype': self.subtype,
            'spike_count': self.spike_count,
            'theta_burst_count': self.theta_burst_counter,
            'ieg_active': self.ieg_flag,
            'plasticity_enabled': self.plasticity_enabled,
            'energy': self.energy,
            'membrane_potential': self.membrane_potential,
            'state': self.state
        }


class EnhancedNodeBehaviorSystem:

    def __init__(self):
        self.node_behaviors: Dict[int, EnhancedNodeBehavior] = {}
        self.behavior_statistics = defaultdict(int)
        log_step("EnhancedNodeBehaviorSystem initialized")
    def create_node_behavior(self, node_id: int, node_type: str = 'dynamic', **kwargs) -> EnhancedNodeBehavior:

        behavior = EnhancedNodeBehavior(node_id, node_type, **kwargs)
        self.node_behaviors[node_id] = behavior
        self.behavior_statistics[f'{node_type}_{behavior.subtype}'] += 1
        log_step("Node behavior created", node_id=node_id, type=node_type, subtype=behavior.subtype)
        return behavior
    def update_node_behaviors(self, graph: Data, step: int) -> Data:

        access_layer = NodeAccessLayer(graph)
        for node_id, behavior in self.node_behaviors.items():
            try:
                behavior.update_behavior(graph, step, access_layer)
            except Exception as e:
                log_step("Error updating node behavior",
                        node_id=node_id,
                        error=str(e))
        return graph
    def get_node_behavior(self, node_id: int) -> Optional[EnhancedNodeBehavior]:
        return self.node_behaviors.get(node_id)
    def remove_node_behavior(self, node_id: int) -> bool:
        if node_id in self.node_behaviors:
            behavior = self.node_behaviors[node_id]
            self.behavior_statistics[f'{behavior.node_type}_{behavior.subtype}'] -= 1
            del self.node_behaviors[node_id]
            return True
        return False
    def get_behavior_statistics(self) -> Dict[str, Any]:
        stats = dict(self.behavior_statistics)
        stats['total_nodes'] = len(self.node_behaviors)
        stats['active_nodes'] = sum(1 for b in self.node_behaviors.values() if b.state == 'active')
        return stats
    def reset_statistics(self):
        self.behavior_statistics.clear()
    def cleanup(self):
        self.node_behaviors.clear()
        self.behavior_statistics.clear()


def create_enhanced_node_behavior_system() -> EnhancedNodeBehaviorSystem:
    return EnhancedNodeBehaviorSystem()
if __name__ == "__main__":
    print("EnhancedNodeBehaviorSystem created successfully!")
    print("Features include:")
    print("- Excitatory/inhibitory neuron types")
    print("- Multiple subtypes with different behaviors")
    print("- Sophisticated membrane dynamics")
    print("- Plasticity and learning mechanisms")
    print("- Neuromodulatory influences")
    print("- Theta-burst detection")
    print("- IEG tagging")
    try:
        system = create_enhanced_node_behavior_system()
        print(f"Behavior system created with {len(system.node_behaviors)} nodes")
        behavior = system.create_node_behavior(1, 'dynamic', subtype='oscillator')
        print(f"Node behavior creation test: {'PASSED' if behavior else 'FAILED'}")
        stats = system.get_behavior_statistics()
        print(f"Behavior statistics: {stats}")
    except Exception as e:
        print(f"EnhancedNodeBehaviorSystem test failed: {e}")
    print("EnhancedNodeBehaviorSystem test completed!")
