
import time
import threading
from typing import Dict, Any, List, Optional, Tuple

from torch_geometric.data import Data
from collections import defaultdict

from utils.logging_utils import log_step
from energy.energy_constants import ConnectionConstants
from energy.node_access_layer import NodeAccessLayer


class EnhancedConnection:

    def __init__(self, source_id: int, target_id: int, connection_type: str = 'excitatory',
                 weight: float = 1.0, **kwargs):

        # Input validation
        if not isinstance(source_id, int) or source_id < 0:
            raise ValueError(f"Invalid source_id: {source_id}")
        if not isinstance(target_id, int) or target_id < 0:
            raise ValueError(f"Invalid target_id: {target_id}")
        if source_id == target_id:
            raise ValueError(f"Cannot create self-connection: {source_id}")
        if not isinstance(connection_type, str):
            raise ValueError(f"Invalid connection_type: {connection_type}")
        if connection_type not in ['excitatory', 'inhibitory', 'modulatory', 'gated', 'plastic']:
            raise ValueError(f"Unsupported connection_type: {connection_type}")

        self.source_id = source_id
        self.target_id = target_id
        self.connection_type = connection_type

        # Validate and clamp weight
        self.weight = self._validate_weight(weight)
        self.delay = self._validate_float(kwargs.get('delay', 0.0), 0.0, 100.0, 'delay')
        self.transmission_efficiency = self._validate_float(kwargs.get('transmission_efficiency', 1.0), 0.0, 1.0, 'transmission_efficiency')
        self.max_weight = self._validate_float(kwargs.get('max_weight', ConnectionConstants.WEIGHT_CAP_MAX), 0.0, 100.0, 'max_weight')
        self.min_weight = self._validate_float(kwargs.get('min_weight', ConnectionConstants.WEIGHT_MIN), -100.0, 0.0, 'min_weight')
        self.plasticity_enabled = bool(kwargs.get('plasticity_enabled', True))
        self.plasticity_tag = False
        self.learning_rate = self._validate_float(kwargs.get('learning_rate', 0.01), 0.0, 1.0, 'learning_rate')
        self.eligibility_trace = 0.0
        self.last_activity = 0.0
        self.activation_count = 0
        self.gate_threshold = self._validate_float(kwargs.get('gate_threshold', 0.5), 0.0, 1.0, 'gate_threshold')
        self.gate_state = False
        self.gate_history = []  # Limited by record_activation
        self.modulation_strength = self._validate_float(kwargs.get('modulation_strength', 0.5), 0.0, 10.0, 'modulation_strength')
        self.modulation_target = str(kwargs.get('modulation_target', 'weight'))
        self.weight_history = [self.weight]  # Limited by update_weight
        self.creation_time = time.time()
        self.last_weight_change = 0.0
        self.subtype2 = str(kwargs.get('subtype2', 'normal'))
        self.subtype3 = str(kwargs.get('subtype3', 'bidirectional'))
        self.subtype4 = str(kwargs.get('subtype4', 'standard'))
        self.dopamine_sensitivity = self._validate_float(kwargs.get('dopamine_sensitivity', 1.0), 0.0, 10.0, 'dopamine_sensitivity')
        self.acetylcholine_sensitivity = self._validate_float(kwargs.get('acetylcholine_sensitivity', 1.0), 0.0, 10.0, 'acetylcholine_sensitivity')
        self.norepinephrine_sensitivity = self._validate_float(kwargs.get('norepinephrine_sensitivity', 1.0), 0.0, 10.0, 'norepinephrine_sensitivity')
        self.active = True
        self.fatigue_level = 0.0
        self.fatigue_recovery_rate = self._validate_float(0.95, 0.0, 1.0, 'fatigue_recovery_rate')

        # Thread safety
        self._lock = threading.RLock()

        log_step("EnhancedConnection created",
                source_id=source_id,
                target_id=target_id,
                type=connection_type,
                weight=self.weight)

    def _validate_float(self, value: Any, min_val: float, max_val: float, field_name: str) -> float:
        """Validate and clamp float values."""
        try:
            val = float(value)
            if not (min_val <= val <= max_val):
                log_step(f"Value {val} for {field_name} out of range [{min_val}, {max_val}], clamping")
                val = max(min_val, min(max_val, val))
            return val
        except (ValueError, TypeError, OverflowError):
            log_step(f"Invalid value {value} for {field_name}, using default")
            return min_val if min_val > 0 else 0.0

    def _validate_weight(self, weight: Any) -> float:
        """Validate and clamp weight values."""
        try:
            val = float(weight)
            # Allow reasonable weight range
            if not (-100.0 <= val <= 100.0):
                log_step(f"Weight {val} out of range, clamping")
                val = max(-100.0, min(100.0, val))
            return val
        except (ValueError, TypeError, OverflowError):
            log_step(f"Invalid weight {weight}, using default 1.0")
            return 1.0
    def get_effective_weight(self, neuromodulators: Optional[Dict[str, float]] = None) -> float:
        """Get effective weight with thread safety."""
        with self._lock:
            base_weight = self.weight
            if self.connection_type == 'inhibitory':
                base_weight = -abs(base_weight)
            elif self.connection_type == 'modulatory':
                base_weight *= self.modulation_strength
            elif self.connection_type == 'gated' and not self.gate_state:
                base_weight = 0.0

            if neuromodulators:
                try:
                    dopamine = self._validate_float(neuromodulators.get('dopamine', 0.0), 0.0, 1.0, 'dopamine')
                    acetylcholine = self._validate_float(neuromodulators.get('acetylcholine', 0.0), 0.0, 1.0, 'acetylcholine')
                    norepinephrine = self._validate_float(neuromodulators.get('norepinephrine', 0.0), 0.0, 1.0, 'norepinephrine')

                    if self.connection_type == 'excitatory':
                        base_weight *= (1.0 + dopamine * self.dopamine_sensitivity)
                    if self.connection_type == 'modulatory':
                        base_weight *= (1.0 + acetylcholine * self.acetylcholine_sensitivity)
                    base_weight *= (1.0 + norepinephrine * self.norepinephrine_sensitivity)
                except Exception as e:
                    log_step(f"Error processing neuromodulators: {e}")

            base_weight *= (1.0 - self.fatigue_level)
            base_weight *= self.transmission_efficiency
            return base_weight
    def update_weight(self, weight_change: float, learning_rate: Optional[float] = None) -> bool:
        """Update weight with thread safety and validation."""
        with self._lock:
            if not self.plasticity_enabled:
                return False

            # Validate inputs
            if not isinstance(weight_change, (int, float)):
                log_step(f"Invalid weight_change: {weight_change}")
                return False

            lr = learning_rate or self.learning_rate
            if not isinstance(lr, (int, float)) or lr < 0:
                log_step(f"Invalid learning_rate: {lr}")
                return False

            actual_change = weight_change * lr
            old_weight = self.weight

            # Update weight with bounds checking
            self.weight = max(self.min_weight,
                              min(self.max_weight,
                                  self.weight + actual_change))


            if abs(self.weight - old_weight) >= 1e-6:
                self.last_weight_change = actual_change
                self.weight_history.append(self.weight)
                # Maintain history size limit
                if len(self.weight_history) > 100:
                    self.weight_history = self.weight_history[-100:]
                return True
            return False
    def update_eligibility_trace(self, delta_eligibility: float):
        """Update eligibility trace with thread safety and validation."""
        with self._lock:
            if not isinstance(delta_eligibility, (int, float)):
                log_step(f"Invalid delta_eligibility: {delta_eligibility}")
                return

            self.eligibility_trace += delta_eligibility
            self.eligibility_trace *= 0.95
            self.eligibility_trace = max(0.0, self.eligibility_trace)

    def record_activation(self, timestamp: float, activity_strength: float = 1.0):
        """Record activation with thread safety and validation."""
        with self._lock:
            # Validate inputs
            if not isinstance(timestamp, (int, float)) or timestamp < 0:
                log_step(f"Invalid timestamp: {timestamp}")
                return

            if not isinstance(activity_strength, (int, float)):
                log_step(f"Invalid activity_strength: {activity_strength}")
                return

            activity_strength = max(0.0, min(10.0, activity_strength))  # Clamp to reasonable range

            self.last_activity = timestamp
            self.activation_count += 1
            self.fatigue_level = min(1.0, self.fatigue_level + activity_strength * 0.1)

            if self.connection_type == 'gated':
                self.gate_state = activity_strength > self.gate_threshold
                self.gate_history.append(self.gate_state)
                # Maintain history size limit
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
        # Thread safety
        self._lock = threading.RLock()

        self.connections: List[EnhancedConnection] = []
        self.connection_index: Dict[Tuple[int, int], int] = {}
        self.node_connections: Dict[int, List[int]] = defaultdict(list)

        # Memory limits
        self.max_connections = 100000  # Prevent excessive memory usage
        self.max_connections_per_node = 1000  # Prevent node overload

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
        log_step("EnhancedConnectionSystem initialized with thread safety")
    def create_connection(self, source_id: int, target_id: int,
                          connection_type: str = 'excitatory', **kwargs) -> bool:
        """Create a connection with comprehensive validation and thread safety."""
        # Input validation
        if not isinstance(source_id, int) or source_id < 0:
            log_step("Error: invalid source_id", source_id=source_id)
            return False
        if not isinstance(target_id, int) or target_id < 0:
            log_step("Error: invalid target_id", target_id=target_id)
            return False
        if source_id == target_id:
            log_step("Error: cannot create self-connection", source_id=source_id)
            return False
        if not isinstance(connection_type, str):
            log_step("Error: connection_type must be string", type=type(connection_type))
            return False
        if connection_type not in ['excitatory', 'inhibitory', 'modulatory', 'gated', 'plastic']:
            log_step("Invalid connection type", type=connection_type)
            return False

        with self._lock:
            # Check memory limits
            if len(self.connections) >= self.max_connections:
                log_step("Maximum connections reached", limit=self.max_connections)
                return False

            # Check connections per node limits
            if len(self.node_connections[source_id]) >= self.max_connections_per_node:
                log_step("Maximum connections per node reached for source", source_id=source_id, limit=self.max_connections_per_node)
                return False
            if len(self.node_connections[target_id]) >= self.max_connections_per_node:
                log_step("Maximum connections per node reached for target", target_id=target_id, limit=self.max_connections_per_node)
                return False

            if (source_id, target_id) in self.connection_index:
                log_step("Connection already exists", source_id=source_id, target_id=target_id)
                return False

            try:
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
            except Exception as e:
                log_step("Error creating connection", error=str(e), source_id=source_id, target_id=target_id)
                return False
    def remove_connection(self, source_id: int, target_id: int) -> bool:
        """Remove connection with thread safety and validation."""
        # Input validation
        if not isinstance(source_id, int) or source_id < 0:
            log_step("Error: invalid source_id for removal", source_id=source_id)
            return False
        if not isinstance(target_id, int) or target_id < 0:
            log_step("Error: invalid target_id for removal", target_id=target_id)
            return False

        with self._lock:
            if (source_id, target_id) not in self.connection_index:
                return False

            try:
                connection_idx = self.connection_index[(source_id, target_id)]
                if connection_idx >= len(self.connections):
                    log_step("Error: invalid connection index", index=connection_idx)
                    return False

                connection = self.connections[connection_idx]
                del self.connection_index[(source_id, target_id)]

                # Safely remove from node connections
                if connection_idx in self.node_connections[source_id]:
                    self.node_connections[source_id].remove(connection_idx)
                if connection_idx in self.node_connections[target_id]:
                    self.node_connections[target_id].remove(connection_idx)

                connection.active = False
                self.stats['total_connections'] -= 1
                self.stats['connections_by_type'][connection.connection_type] -= 1
                self.stats['connections_pruned'] += 1
                log_step("Connection removed", source_id=source_id, target_id=target_id)
                return True
            except Exception as e:
                log_step("Error removing connection", error=str(e), source_id=source_id, target_id=target_id)
                return False

    def get_connection(self, source_id: int, target_id: int) -> Optional[EnhancedConnection]:
        """Get connection with thread safety and validation."""
        # Input validation
        if not isinstance(source_id, int) or source_id < 0:
            return None
        if not isinstance(target_id, int) or target_id < 0:
            return None

        with self._lock:
            if (source_id, target_id) not in self.connection_index:
                return None

            try:
                connection_idx = self.connection_index[(source_id, target_id)]
                if connection_idx >= len(self.connections):
                    log_step("Error: invalid connection index in get_connection", index=connection_idx)
                    return None
                return self.connections[connection_idx]
            except Exception as e:
                log_step("Error getting connection", error=str(e), source_id=source_id, target_id=target_id)
                return None
    def get_connections_for_node(self, node_id: int) -> List[EnhancedConnection]:
        """Get connections for node with thread safety and validation."""
        if not isinstance(node_id, int) or node_id < 0:
            return []

        with self._lock:
            try:
                connection_indices = self.node_connections.get(node_id, [])
                active_connections = []
                for idx in connection_indices:
                    if idx < len(self.connections) and self.connections[idx].active:
                        active_connections.append(self.connections[idx])
                return active_connections
            except Exception as e:
                log_step("Error getting connections for node", error=str(e), node_id=node_id)
                return []

    def update_connections(self, graph: Data, step: int) -> Data:
        """Update connections with thread safety and validation."""
        if graph is None:
            log_step("Error: graph is None in update_connections")
            return graph

        if not isinstance(step, int) or step < 0:
            log_step("Error: invalid step in update_connections", step=step)
            return graph

        with self._lock:
            try:
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
            except Exception as e:
                log_step("Error updating connections", error=str(e), step=step)
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
            source_id = connection.source_id
            target_id = connection.target_id
            self.remove_connection(source_id, target_id)
            
            # Diagnostic log for prune
            log_step(f"[PRUNE] Weak connection pruned: {source_id} -> {target_id}, weight={connection.weight}, reason={'low_weight' if abs(connection.weight) < ConnectionConstants.WEIGHT_MIN else 'high_fatigue'}, fatigue={connection.fatigue_level}")
    def set_neuromodulator_level(self, neuromodulator: str, level: float):
        """Set neuromodulator level with thread safety and validation."""
        if not isinstance(neuromodulator, str):
            log_step(f"Invalid neuromodulator name: {neuromodulator}")
            return

        if not isinstance(level, (int, float)):
            log_step(f"Invalid neuromodulator level: {level}")
            return

        with self._lock:
            if neuromodulator in self.neuromodulators:
                self.neuromodulators[neuromodulator] = max(0.0, min(1.0, float(level)))
                log_step(f"Neuromodulator {neuromodulator} set to {self.neuromodulators[neuromodulator]}")

    def get_effective_weights(self, source_id: int, target_id: int) -> float:
        """Get effective weights with thread safety and validation."""
        # Input validation
        if not isinstance(source_id, int) or source_id < 0:
            return 0.0
        if not isinstance(target_id, int) or target_id < 0:
            return 0.0

        with self._lock:
            try:
                connection = self.get_connection(source_id, target_id)
                if connection is None:
                    return 0.0
                return connection.get_effective_weight(self.neuromodulators)
            except Exception as e:
                log_step("Error getting effective weight", error=str(e), source_id=source_id, target_id=target_id)
                return 0.0

    def get_connection_statistics(self) -> Dict[str, Any]:
        """Get connection statistics with thread safety."""
        with self._lock:
            try:
                stats = self.stats.copy()
                stats['neuromodulators'] = self.neuromodulators.copy()
                stats['active_connections'] = sum(1 for c in self.connections if c.active)
                stats['memory_usage_mb'] = (
                    len(self.connections) * 0.1 +  # Rough estimate per connection
                    len(self.connection_index) * 0.01 +  # Per index entry
                    len(self.node_connections) * 0.05  # Per node entry
                )
                return stats
            except Exception as e:
                log_step("Error getting connection statistics", error=str(e))
                return {}

    def reset_statistics(self):
        """Reset statistics with thread safety."""
        with self._lock:
            self.stats = {
                'total_connections': 0,
                'connections_by_type': defaultdict(int),
                'weight_changes': 0,
                'connections_pruned': 0,
                'connections_created': 0
            }
            log_step("Connection statistics reset")

    def cleanup(self):
        """Clean up all connection data structures to prevent memory leaks with thread safety."""
        with self._lock:
            try:
                # Clear all connections and their data
                for connection in self.connections:
                    if hasattr(connection, 'weight_history'):
                        connection.weight_history.clear()
                    if hasattr(connection, 'gate_history'):
                        connection.gate_history.clear()

                self.connections.clear()
                self.connection_index.clear()
                self.node_connections.clear()

                # Reset statistics
                self.stats = {
                    'total_connections': 0,
                    'connections_by_type': defaultdict(int),
                    'weight_changes': 0,
                    'connections_pruned': 0,
                    'connections_created': 0
                }

                # Reset neuromodulators
                self.neuromodulators = {
                    'dopamine': 0.0,
                    'acetylcholine': 0.0,
                    'norepinephrine': 0.0
                }

                log_step("EnhancedConnectionSystem cleanup completed")
            except Exception as e:
                log_step("Error during cleanup", error=str(e))


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
