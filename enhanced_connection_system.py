
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Set
from torch_geometric.data import Data
from collections import defaultdict
import logging
from logging_utils import log_step
from config_manager import get_learning_config, get_system_constants
from energy_constants import ConnectionConstants
from node_access_layer import NodeAccessLayer


class EnhancedConnection:

    def __init__(self, source_id: int, target_id: int, connection_type: str = 'excitatory',
                 weight: float = 1.0, **kwargs):

        self.source_id = source_id
        self.target_id = target_id
        self.connection_type = connection_type
        self.weight = weight
        self.delay = kwargs.get('delay', 0.0)
        self.transmission_efficiency = kwargs.get('transmission_efficiency', 1.0)
        self.max_weight = kwargs.get('max_weight', ConnectionConstants.WEIGHT_CAP_MAX)
        self.min_weight = kwargs.get('min_weight', ConnectionConstants.WEIGHT_MIN)
        self.plasticity_enabled = kwargs.get('plasticity_enabled', True)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.eligibility_trace = 0.0
        self.last_activity = 0.0
        self.activation_count = 0
        self.gate_threshold = kwargs.get('gate_threshold', 0.5)
        self.gate_state = False
        self.gate_history = []
        self.modulation_strength = kwargs.get('modulation_strength', 0.5)
        self.modulation_target = kwargs.get('modulation_target', 'weight')
        self.weight_history = [weight]
        self.creation_time = time.time()
        self.last_weight_change = 0.0
        self.subtype2 = kwargs.get('subtype2', 'normal')
        self.subtype3 = kwargs.get('subtype3', 'bidirectional')
        self.subtype4 = kwargs.get('subtype4', 'standard')
        self.dopamine_sensitivity = kwargs.get('dopamine_sensitivity', 1.0)
        self.acetylcholine_sensitivity = kwargs.get('acetylcholine_sensitivity', 1.0)
        self.norepinephrine_sensitivity = kwargs.get('norepinephrine_sensitivity', 1.0)
        self.active = True
        self.fatigue_level = 0.0
        self.fatigue_recovery_rate = 0.95
        log_step("EnhancedConnection created",
                source_id=source_id,
                target_id=target_id,
                type=connection_type,
                weight=weight)
    def get_effective_weight(self, neuromodulators: Optional[Dict[str, float]] = None) -> float:

        base_weight = self.weight
        if self.connection_type == 'inhibitory':
            base_weight = -abs(base_weight)
        elif self.connection_type == 'modulatory':
            base_weight *= self.modulation_strength
        elif self.connection_type == 'gated' and not self.gate_state:
            base_weight = 0.0
        if neuromodulators:
            dopamine = neuromodulators.get('dopamine', 0.0)
            acetylcholine = neuromodulators.get('acetylcholine', 0.0)
            norepinephrine = neuromodulators.get('norepinephrine', 0.0)
            if self.connection_type == 'excitatory':
                base_weight *= (1.0 + dopamine * self.dopamine_sensitivity)
            if self.connection_type == 'modulatory':
                base_weight *= (1.0 + acetylcholine * self.acetylcholine_sensitivity)
            base_weight *= (1.0 + norepinephrine * self.norepinephrine_sensitivity)
        base_weight *= (1.0 - self.fatigue_level)
        base_weight *= self.transmission_efficiency
        return base_weight
    def update_weight(self, weight_change: float, learning_rate: Optional[float] = None) -> bool:

        if not self.plasticity_enabled:
            return False
        lr = learning_rate or self.learning_rate
        actual_change = weight_change * lr
        old_weight = self.weight
        self.weight = max(self.min_weight,
                         min(self.max_weight,
                             self.weight + actual_change))
        if abs(self.weight - old_weight) > 0.001:
            self.last_weight_change = actual_change
            self.weight_history.append(self.weight)
            if len(self.weight_history) > 100:
                self.weight_history = self.weight_history[-100:]
            return True
        return False
    def update_eligibility_trace(self, delta_eligibility: float):
        self.eligibility_trace += delta_eligibility
        self.eligibility_trace *= 0.95
        self.eligibility_trace = max(0.0, self.eligibility_trace)
    def record_activation(self, timestamp: float, activity_strength: float = 1.0):
        self.last_activity = timestamp
        self.activation_count += 1
        self.fatigue_level = min(1.0, self.fatigue_level + activity_strength * 0.1)
        if self.connection_type == 'gated':
            self.gate_state = activity_strength > self.gate_threshold
            self.gate_history.append(self.gate_state)
            if len(self.gate_history) > 50:
                self.gate_history = self.gate_history[-50:]
    def update_fatigue(self):
        self.fatigue_level *= self.fatigue_recovery_rate
    def get_connection_strength(self) -> float:
        return abs(self.weight)
    def get_connection_efficiency(self) -> float:
        return self.transmission_efficiency * (1.0 - self.fatigue_level)
    def is_active(self) -> bool:
        return self.active and self.get_connection_efficiency() > 0.1
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'connection_type': self.connection_type,
            'weight': self.weight,
            'delay': self.delay,
            'transmission_efficiency': self.transmission_efficiency,
            'plasticity_enabled': self.plasticity_enabled,
            'learning_rate': self.learning_rate,
            'eligibility_trace': self.eligibility_trace,
            'activation_count': self.activation_count,
            'gate_threshold': self.gate_threshold,
            'modulation_strength': self.modulation_strength,
            'subtype2': self.subtype2,
            'subtype3': self.subtype3,
            'subtype4': self.subtype4,
            'active': self.active,
            'fatigue_level': self.fatigue_level,
            'creation_time': self.creation_time
        }


class EnhancedConnectionSystem:

    def __init__(self):
        self.connections: List[EnhancedConnection] = []
        self.connection_index: Dict[Tuple[int, int], int] = {}
        self.node_connections: Dict[int, List[int]] = defaultdict(list)
        self.stats = {
            'total_connections': 0,
            'connections_by_type': defaultdict(int),
            'weight_changes': 0,
            'connections_pruned': 0,
            'connections_created': 0
        }
        self.neuromodulators = {
            'dopamine': 0.0,
            'acetylcholine': 0.0,
            'norepinephrine': 0.0
        }
        log_step("EnhancedConnectionSystem initialized")
    def create_connection(self, source_id: int, target_id: int,
                         connection_type: str = 'excitatory', **kwargs) -> bool:

        if (source_id, target_id) in self.connection_index:
            return False
        if connection_type not in ['excitatory', 'inhibitory', 'modulatory', 'gated', 'plastic']:
            log_step("Invalid connection type", type=connection_type)
            return False
        connection = EnhancedConnection(source_id, target_id, connection_type, **kwargs)
        connection_idx = len(self.connections)
        self.connections.append(connection)
        self.connection_index[(source_id, target_id)] = connection_idx
        self.node_connections[source_id].append(connection_idx)
        self.node_connections[target_id].append(connection_idx)
        self.stats['total_connections'] += 1
        self.stats['connections_by_type'][connection_type] += 1
        self.stats['connections_created'] += 1
        log_step("Connection created",
                source_id=source_id,
                target_id=target_id,
                type=connection_type)
        return True
    def remove_connection(self, source_id: int, target_id: int) -> bool:

        if (source_id, target_id) not in self.connection_index:
            return False
        connection_idx = self.connection_index[(source_id, target_id)]
        connection = self.connections[connection_idx]
        del self.connection_index[(source_id, target_id)]
        self.node_connections[source_id].remove(connection_idx)
        self.node_connections[target_id].remove(connection_idx)
        connection.active = False
        self.stats['total_connections'] -= 1
        self.stats['connections_by_type'][connection.connection_type] -= 1
        self.stats['connections_pruned'] += 1
        log_step("Connection removed", source_id=source_id, target_id=target_id)
        return True
    def get_connection(self, source_id: int, target_id: int) -> Optional[EnhancedConnection]:
        if (source_id, target_id) not in self.connection_index:
            return None
        connection_idx = self.connection_index[(source_id, target_id)]
        return self.connections[connection_idx]
    def get_connections_for_node(self, node_id: int) -> List[EnhancedConnection]:
        connection_indices = self.node_connections.get(node_id, [])
        return [self.connections[idx] for idx in connection_indices if self.connections[idx].active]
    def update_connections(self, graph: Data, step: int) -> Data:

        for connection in self.connections:
            if not connection.active:
                continue
            connection.update_fatigue()
            connection.update_eligibility_trace(0.0)
        self._update_gated_connections(graph)
        self._update_plastic_connections(graph)
        self._update_modulatory_connections(graph)
        self._prune_weak_connections()
        return graph
    def _update_gated_connections(self, graph: Data):
        access_layer = NodeAccessLayer(graph)
        for connection in self.connections:
            if connection.connection_type != 'gated' or not connection.active:
                continue
            source_node = access_layer.get_node_by_id(connection.source_id)
            if source_node is None:
                continue
            source_energy = access_layer.get_node_energy(connection.source_id)
            if source_energy is None:
                continue
            connection.gate_state = source_energy > connection.gate_threshold
            connection.gate_history.append(connection.gate_state)
            if len(connection.gate_history) > 50:
                connection.gate_history = connection.gate_history[-50:]
    def _update_plastic_connections(self, graph: Data):
        access_layer = NodeAccessLayer(graph)
        for connection in self.connections:
            if connection.connection_type != 'plastic' or not connection.active:
                continue
            source_node = access_layer.get_node_by_id(connection.source_id)
            target_node = access_layer.get_node_by_id(connection.target_id)
            if source_node is None or target_node is None:
                continue
            source_energy = access_layer.get_node_energy(connection.source_id)
            target_energy = access_layer.get_node_energy(connection.target_id)
            if source_energy is None or target_energy is None:
                continue
            activity_correlation = min(source_energy, target_energy) / max(source_energy, target_energy)
            weight_change = (activity_correlation - 0.5) * 0.01
            if abs(weight_change) > 0.001:
                if connection.update_weight(weight_change):
                    self.stats['weight_changes'] += 1
    def _update_modulatory_connections(self, graph: Data):
        for connection in self.connections:
            if connection.connection_type != 'modulatory' or not connection.active:
                continue
            if hasattr(connection, 'target_node_id') and connection.target_node_id is not None:
                modulatory_strength = connection.get_effective_weight()
                if modulatory_strength > 0:
                    connection.modulatory_effect = min(1.0, modulatory_strength * 0.1)
                else:
                    connection.modulatory_effect = max(-1.0, modulatory_strength * 0.1)
    def _prune_weak_connections(self):
        connections_to_remove = []
        for i, connection in enumerate(self.connections):
            if not connection.active:
                continue
            if abs(connection.weight) < ConnectionConstants.WEIGHT_MIN:
                connections_to_remove.append(i)
                continue
            if connection.fatigue_level > 0.9:
                connections_to_remove.append(i)
                continue
        for i in reversed(connections_to_remove):
            connection = self.connections[i]
            self.remove_connection(connection.source_id, connection.target_id)
    def set_neuromodulator_level(self, neuromodulator: str, level: float):
        if neuromodulator in self.neuromodulators:
            self.neuromodulators[neuromodulator] = max(0.0, min(1.0, level))
    def get_effective_weights(self, source_id: int, target_id: int) -> float:
        connection = self.get_connection(source_id, target_id)
        if connection is None:
            return 0.0
        return connection.get_effective_weight(self.neuromodulators)
    def get_connection_statistics(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats['neuromodulators'] = self.neuromodulators.copy()
        stats['active_connections'] = sum(1 for c in self.connections if c.active)
        return stats
    def reset_statistics(self):
        self.stats = {
            'total_connections': 0,
            'connections_by_type': defaultdict(int),
            'weight_changes': 0,
            'connections_pruned': 0,
            'connections_created': 0
        }
    def cleanup(self):
        self.connections.clear()
        self.connection_index.clear()
        self.node_connections.clear()


def create_enhanced_connection_system() -> EnhancedConnectionSystem:
    return EnhancedConnectionSystem()
if __name__ == "__main__":
    print("EnhancedConnectionSystem created successfully!")
    print("Features include:")
    print("- Multiple connection types (excitatory, inhibitory, modulatory, gated, plastic)")
    print("- Sophisticated weight dynamics")
    print("- Delay and transmission properties")
    print("- Plasticity mechanisms")
    print("- Neuromodulatory influences")
    print("- Connection pruning and optimization")
    try:
        system = create_enhanced_connection_system()
        print(f"Connection system created with {len(system.connections)} connections")
        success = system.create_connection(1, 2, 'excitatory', weight=1.5)
        print(f"Connection creation test: {'PASSED' if success else 'FAILED'}")
        stats = system.get_connection_statistics()
        print(f"Connection statistics: {stats}")
    except Exception as e:
        print(f"EnhancedConnectionSystem test failed: {e}")
    print("EnhancedConnectionSystem test completed!")
