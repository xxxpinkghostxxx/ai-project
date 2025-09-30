import time
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import Data

from config.unified_config_manager import (get_enhanced_nodes_config,
                                           get_system_constants)
from src.utils.logging_utils import log_step

try:
    from src.utils.unified_error_handler import get_error_handler
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False
    log_step("Error handler not available, using basic error handling")

try:
    from src.neural.enhanced_neural_integration import \
        create_enhanced_neural_integration
    ENHANCED_SYSTEMS_AVAILABLE = True
except ImportError:
    ENHANCED_SYSTEMS_AVAILABLE = False
    log_step("Enhanced neural systems not available, using basic behaviors")


def get_time_step() -> float:
    constants = get_system_constants()
    return constants.get('time_step', 0.01)


def get_refractory_period() -> float:
    constants = get_system_constants()
    return constants.get('refractory_period', 0.1)
from src.energy.energy_behavior import get_node_energy_cap
from src.energy.node_access_layer import NodeAccessLayer
from src.utils.statistics_utils import create_standard_stats


class BehaviorCache:
    """Cache for behavior-related configuration data to reduce global scope."""
    def __init__(self):
        self._energy_cap_cache = None
        self._energy_cap_cache_time = 0
        self._energy_cap_cache_ttl = 60
        self._enhanced_nodes_cache = None
        self._enhanced_nodes_cache_time = 0
        self._enhanced_nodes_cache_ttl = 60

    def get_energy_cap_255(self) -> float:
        constants = get_system_constants()
        return constants.get('energy_cap_255', 255.0)

    def get_enhanced_nodes_config_cached(self) -> Dict[str, float]:
        current_time_val = time.time()
        if (self._enhanced_nodes_cache is not None and
            current_time_val - self._enhanced_nodes_cache_time < self._enhanced_nodes_cache_ttl):
            return self._enhanced_nodes_cache
        self._enhanced_nodes_cache = get_enhanced_nodes_config()
        self._enhanced_nodes_cache_time = current_time_val
        return self._enhanced_nodes_cache


# Global cache instance - single point of access
_behavior_cache = BehaviorCache()


def get_energy_cap_255() -> float:
    return _behavior_cache.get_energy_cap_255()


def get_enhanced_nodes_config_cached() -> Dict[str, float]:
    return _behavior_cache.get_enhanced_nodes_config_cached()


class BehaviorEngine:
    def __init__(self) -> None:
        self.behavior_handlers: Dict[str, callable] = {
            'sensory': self.update_sensory_node,
            'dynamic': self.update_dynamic_node,
            'oscillator': self.update_oscillator_node,
            'integrator': self.update_integrator_node,
            'relay': self.update_relay_node,
            'highway': self.update_highway_node,
            'workspace': self.update_workspace_node
        }
        self.error_handler = None
        if ERROR_HANDLER_AVAILABLE:
            try:
                self.error_handler = get_error_handler()
                log_step("Error handler initialized in behavior engine")
            except (ImportError, AttributeError, RuntimeError) as e:
                log_step("Failed to initialize error handler", error=str(e))
                self.error_handler = None
        self.enhanced_integration = None
        if ENHANCED_SYSTEMS_AVAILABLE:
            try:
                self.enhanced_integration = create_enhanced_neural_integration()
                log_step("Enhanced neural integration initialized in behavior engine")
            except (ImportError, AttributeError, RuntimeError) as e:
                log_step("Failed to initialize enhanced neural integration", error=str(e))
                self.enhanced_integration = None
        self.behavior_stats = create_standard_stats('behavior_engine')
        # Add behavior-specific stats
        self.behavior_stats.update({
            'workspace_syntheses': 0,
            'enhanced_updates': 0,
            'basic_updates': 0,
            'oscillator_activations': 0,
            'integrator_activations': 0,
            'relay_transfers': 0,
            'highway_regulations': 0
        })
    def update_node_behavior(self, node_id: int, graph: Data, step: int, access_layer=None) -> bool:
        """Update behavior for a specific node with assertions for safety."""
        self._validate_node_behavior_inputs(node_id, graph, step)

        if access_layer is None:
            access_layer = NodeAccessLayer(graph)

        node = access_layer.get_node_by_id(node_id)
        if node is None:
            return False

        # Try enhanced behavior first
        if self._try_enhanced_behavior(node, graph, step, access_layer, node_id):
            return True

        # Fall back to basic behavior
        return self._update_basic_behavior(node, graph, step, access_layer, node_id)
    
    def _validate_node_behavior_inputs(self, node_id: int, graph: Data, step: int):
        """Validate inputs for node behavior update."""
        assert isinstance(node_id, int), "Node ID must be an integer"
        assert node_id >= 0, "Node ID must be non-negative"
        assert graph is not None, "Graph must not be None"
        assert isinstance(step, int), "Step must be an integer"
        assert step >= 0, "Step must be non-negative"
    
    def _try_enhanced_behavior(self, node: Dict, graph: Data, step: int, access_layer, node_id: int) -> bool:
        """Try to update node using enhanced behavior system."""
        has_enhanced_behavior = node.get('enhanced_behavior', False)
        if not has_enhanced_behavior or self.enhanced_integration is None:
            return False

        try:
            if not hasattr(self.enhanced_integration, 'node_behavior_system') or self.enhanced_integration.node_behavior_system is None:
                log_step("Enhanced behavior system not available", node_id=node_id, step=step)
                return False

            enhanced_behavior = self.enhanced_integration.node_behavior_system.get_node_behavior(node_id)
            if enhanced_behavior is None:
                log_step("Enhanced behavior not found for node", node_id=node_id, step=step)
                return False

            success = enhanced_behavior.update_behavior(graph, step, access_layer)
            if success:
                self.behavior_stats['enhanced_updates'] += 1
                return True
            else:
                log_step("Enhanced behavior update failed", node_id=node_id, step=step)
                return False

        except AttributeError as e:
            log_step("Enhanced integration missing required attributes", error=str(e), node_id=node_id, step=step)
        except (KeyError, ValueError, RuntimeError) as e:
            log_step("Error in enhanced behavior update", error=str(e), node_id=node_id, step=step)

        return False

    def _update_basic_behavior(self, node: Dict, graph: Data, step: int, access_layer, node_id: int) -> bool:
        """Update node using basic behavior system."""
        behavior = node.get('behavior', 'dynamic')
        handler = self.behavior_handlers.get(behavior, self.update_dynamic_node)
        log_step(f"Updating {behavior} node behavior", node_id=node_id, step=step)
        
        try:
            handler(node_id, graph, step, access_layer)
            access_layer.update_node_property(node_id, 'last_update', step)
            self.behavior_stats['basic_updates'] += 1
            return True
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            return self._handle_behavior_error(e, behavior, node_id, access_layer, step, "Error updating")
        except (RuntimeError, MemoryError, OSError) as e:
            return self._handle_behavior_error(e, behavior, node_id, access_layer, step, "Unexpected error updating")

    def _handle_behavior_error(self, error: Exception, behavior: str, node_id: int, access_layer, step: int, error_type: str) -> bool:
        """Handle errors during behavior update."""
        log_step(f"{error_type} {behavior} node", error=str(error), node_id=node_id)

        if self.error_handler is not None:
            try:
                recovery_success = self.error_handler.handle_error(
                    error, f"{error_type} {behavior} node",
                    recovery_func=lambda: self._recover_node_behavior(node_id, access_layer),
                    critical=False
                )
                if recovery_success:
                    log_step(f"Recovered from {behavior} node error", node_id=node_id)
                    access_layer.update_node_property(node_id, 'last_update', step)
                    return True
            except (AttributeError, KeyError, RuntimeError) as recovery_error:
                log_step(f"Recovery failed for {behavior} node", error=str(recovery_error), node_id=node_id)

        try:
            access_layer.update_node_property(node_id, 'last_update', step)
        except (AttributeError, KeyError, TypeError):
            pass
        return False
    def _recover_node_behavior(self, node_id: int, access_layer) -> bool:

        try:
            access_layer.update_node_property(node_id, 'state', 'inactive')
            access_layer.update_node_property(node_id, 'membrane_potential', 0.0)
            access_layer.update_node_property(node_id, 'refractory_timer', 0.0)
            access_layer.update_node_property(node_id, 'threshold', 0.5)
            current_energy = access_layer.get_node_energy(node_id)
            if current_energy is not None:
                safe_energy = min(current_energy, get_node_energy_cap() * 0.5)
                access_layer.set_node_energy(node_id, safe_energy)
            log_step(f"Node {node_id} recovered to safe state")
            return True
        except (AttributeError, KeyError, ValueError) as e:
            log_step(f"Recovery failed for node {node_id}", error=str(e))
            return False
    def update_sensory_node(self, node_id: int, graph: Data, _step: int, access_layer=None) -> None:
        """Optimized sensory node update with reduced logging overhead."""
        if access_layer is None:
            access_layer = NodeAccessLayer(graph)

        # Cache time step to avoid repeated function calls
        time_step = get_time_step()

        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        if refractory_timer > 0:
            new_refractory = max(0.0, refractory_timer - time_step)
            access_layer.update_node_property(node_id, 'refractory_timer', new_refractory)
            return

        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            # Cache energy cap to avoid repeated function calls
            energy_cap = get_energy_cap_255()
            membrane = min(energy / energy_cap, 1.0)
            access_layer.update_node_property(node_id, 'membrane_potential', membrane)
            access_layer.update_node_property(node_id, 'state', 'active')
        # Removed excessive debug logging for performance
    def update_dynamic_node(self, node_id: int, graph: Data, _step: int, access_layer=None) -> None:
        """Optimized dynamic node update with cached values and reduced overhead."""
        if access_layer is None:
            access_layer = NodeAccessLayer(graph)

        # Cache frequently used values
        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            # Cache energy cap to avoid repeated function calls
            energy_cap = get_node_energy_cap()
            membrane_potential = min(energy / energy_cap, 1.0)
            access_layer.update_node_property(node_id, 'membrane_potential', membrane_potential)

            threshold = access_layer.get_node_property(node_id, 'threshold', 0.3)
            new_state = 'active' if energy >= threshold else 'inactive'
            access_layer.update_node_property(node_id, 'state', new_state)

            # Handle refractory period
            refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
            if refractory_timer > 0:
                time_step = get_time_step()
                new_refractory = max(0.0, refractory_timer - time_step)
                access_layer.update_node_property(node_id, 'refractory_timer', new_refractory)
    def update_oscillator_node(self, node_id: int, graph: Data, step: int, access_layer=None) -> None:
        """Optimized oscillator node update with cached values and reduced logging."""
        if access_layer is None:
            access_layer = NodeAccessLayer(graph)

        # Cache frequently used values
        config = get_enhanced_nodes_config_cached()
        time_step = get_time_step()
        refractory_period = get_refractory_period()

        oscillation_freq = access_layer.get_node_property(node_id, 'oscillation_freq', config['oscillator_frequency'])
        threshold = access_layer.get_node_property(node_id, 'threshold', 0.8)
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)

        if refractory_timer > 0:
            new_refractory = max(0.0, refractory_timer - time_step)
            access_layer.update_node_property(node_id, 'refractory_timer', new_refractory)
            return

        energy_increment = oscillation_freq * time_step * 0.1
        membrane_potential += energy_increment

        if membrane_potential >= threshold:
            access_layer.update_node_property(node_id, 'last_activation', time.time())
            access_layer.update_node_property(node_id, 'refractory_timer', refractory_period)
            access_layer.update_node_property(node_id, 'membrane_potential', 0.0)
            access_layer.update_node_property(node_id, 'state', 'active')
            self.behavior_stats['oscillator_activations'] += 1
            # Reduced logging frequency for performance
            if step % 100 == 0:
                log_step("Oscillator activated", node_id=node_id, frequency=oscillation_freq, step=step)
        else:
            access_layer.update_node_property(node_id, 'state', 'inactive')

        access_layer.update_node_property(node_id, 'membrane_potential', min(membrane_potential, 1.0))
    def update_integrator_node(self, node_id: int, graph: Data, step: int, access_layer=None) -> None:

        if access_layer is None:
            access_layer = NodeAccessLayer(graph)
        config = get_enhanced_nodes_config_cached()
        integration_rate = access_layer.get_node_property(node_id, 'integration_rate', 0.5)
        threshold = access_layer.get_node_property(node_id, 'threshold', config['integrator_threshold'])
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - get_time_step()))
            return
        energy_increment = integration_rate * get_time_step() * 0.05
        membrane_potential += energy_increment
        if membrane_potential >= threshold:
            access_layer.update_node_property(node_id, 'last_activation', time.time())
            access_layer.update_node_property(node_id, 'refractory_timer', get_refractory_period())
            access_layer.update_node_property(node_id, 'membrane_potential', 0.0)
            access_layer.update_node_property(node_id, 'state', 'consolidating')
            self.behavior_stats['integrator_activations'] += 1
            log_step("Integrator activated",
                    node_id=node_id,
                    rate=integration_rate,
                    step=step)
        else:
            state = 'active' if membrane_potential > threshold * 0.5 else 'inactive'
            access_layer.update_node_property(node_id, 'state', state)
        access_layer.update_node_property(node_id, 'membrane_potential', min(membrane_potential, 1.0))
    def update_relay_node(self, node_id: int, graph: Data, step: int, access_layer=None) -> None:

        if access_layer is None:
            access_layer = NodeAccessLayer(graph)
        config = get_enhanced_nodes_config_cached()
        relay_amplification = access_layer.get_node_property(node_id, 'relay_amplification', config['relay_amplification'])
        threshold = access_layer.get_node_property(node_id, 'threshold', 0.4)
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - get_time_step()))
            return
        energy_increment = 0.02 * get_time_step()
        membrane_potential += energy_increment
        if membrane_potential >= threshold:
            access_layer.update_node_property(node_id, 'last_activation', time.time())
            access_layer.update_node_property(node_id, 'refractory_timer', get_refractory_period() * 0.5)
            access_layer.update_node_property(node_id, 'membrane_potential', 0.0)
            access_layer.update_node_property(node_id, 'state', 'active')
            self.behavior_stats['relay_transfers'] += 1
            log_step("Relay activated",
                    node_id=node_id,
                    amplification=relay_amplification,
                    step=step)
        else:
            state = 'pending' if membrane_potential > threshold * 0.3 else 'inactive'
            access_layer.update_node_property(node_id, 'state', state)
        access_layer.update_node_property(node_id, 'membrane_potential', min(membrane_potential, 1.0))
    def update_highway_node(self, node_id: int, graph: Data, _step: int, access_layer=None) -> None:

        if access_layer is None:
            access_layer = NodeAccessLayer(graph)
        config = get_enhanced_nodes_config_cached()
        highway_energy_boost = access_layer.get_node_property(node_id, 'highway_energy_boost', config['highway_energy_boost'])
        threshold = access_layer.get_node_property(node_id, 'threshold', 0.2)
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - get_time_step()))
            return
        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            access_layer.update_node_property(node_id, 'membrane_potential', min(energy / get_node_energy_cap(), 1.0))
            if membrane_potential >= threshold:
                boosted_energy = min(get_node_energy_cap(), energy * highway_energy_boost)
                access_layer.set_node_energy(node_id, boosted_energy)
                access_layer.update_node_property(node_id, 'refractory_timer', 0.5)
                access_layer.update_node_property(node_id, 'last_activation', time.time())
                access_layer.update_node_property(node_id, 'state', 'regulating')
                self.behavior_stats['highway_regulations'] += 1
            else:
                access_layer.update_node_property(node_id, 'state', 'active')
    def update_workspace_node(self, node_id: int, graph: Data, _step: int, access_layer=None) -> None:

        if access_layer is None:
            access_layer = NodeAccessLayer(graph)
        workspace_capacity = access_layer.get_node_property(node_id, 'workspace_capacity', 5.0)
        workspace_creativity = access_layer.get_node_property(node_id, 'workspace_creativity', 1.5)
        workspace_focus = access_layer.get_node_property(node_id, 'workspace_focus', 3.0)
        threshold = access_layer.get_node_property(node_id, 'threshold', 0.6)
        refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0.0)
        membrane_potential = access_layer.get_node_property(node_id, 'membrane_potential', 0.0)
        if refractory_timer > 0:
            access_layer.update_node_property(node_id, 'refractory_timer', max(0.0, refractory_timer - get_time_step()))
            return
        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            access_layer.update_node_property(node_id, 'membrane_potential', min(energy / get_node_energy_cap(), 1.0))
            if membrane_potential >= threshold:
                if workspace_capacity >= 2.0:
                    synthesis_success = np.random.random() < (workspace_creativity * workspace_focus * 0.1)
                    if synthesis_success:
                        access_layer.update_node_property(node_id, 'state', 'synthesizing')
                        access_layer.update_node_property(node_id, 'refractory_timer', 1.0 / workspace_creativity)
                        access_layer.update_node_property(node_id, 'last_activation', time.time())
                        self.behavior_stats['workspace_syntheses'] += 1
                    else:
                        access_layer.update_node_property(node_id, 'state', 'planning')
                else:
                    access_layer.update_node_property(node_id, 'state', 'imagining')
            elif membrane_potential > threshold * 0.8:
                access_layer.update_node_property(node_id, 'state', 'planning')
            elif membrane_potential > threshold * 0.5:
                access_layer.update_node_property(node_id, 'state', 'imagining')
            else:
                access_layer.update_node_property(node_id, 'state', 'active')
    def set_neuromodulator_level(self, neuromodulator: str, level: float):
        if self.enhanced_integration is not None:
            try:
                if hasattr(self.enhanced_integration, 'set_neuromodulator_level'):
                    self.enhanced_integration.set_neuromodulator_level(neuromodulator, level)
                else:
                    log_step("Enhanced integration missing set_neuromodulator_level method")
            except (AttributeError, KeyError, ValueError) as e:
                log_step("Error setting neuromodulator level", error=str(e))
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        if self.enhanced_integration is not None:
            try:
                if hasattr(self.enhanced_integration, 'get_integration_statistics'):
                    return self.enhanced_integration.get_integration_statistics()
                else:
                    log_step("Enhanced integration missing get_integration_statistics method")
            except (AttributeError, KeyError, RuntimeError) as e:
                log_step("Error getting enhanced statistics", error=str(e))
        return {}
    def get_behavior_statistics(self) -> Dict[str, int]:
        return self.behavior_stats.copy()
    def reset_statistics(self) -> None:
        self.behavior_stats = create_standard_stats('behavior_engine')
        # Add behavior-specific stats
        self.behavior_stats.update({
            'workspace_syntheses': 0,
            'enhanced_updates': 0,
            'basic_updates': 0,
            'oscillator_activations': 0,
            'integrator_activations': 0,
            'relay_transfers': 0,
            'highway_regulations': 0
        })
        if self.enhanced_integration is not None:
            try:
                if hasattr(self.enhanced_integration, 'reset_integration_statistics'):
                    self.enhanced_integration.reset_integration_statistics()
                else:
                    log_step("Enhanced integration missing reset_integration_statistics method")
            except (AttributeError, KeyError, RuntimeError) as e:
                log_step("Error resetting enhanced statistics", error=str(e))



def should_transition_to_learning(node: Dict[str, Any]) -> bool:

    last_activation = node.get('last_activation', 0)
    current_time = time.time()
    if (current_time - last_activation < 5.0 and
        node.get('plasticity_enabled', False) and
        node.get('eligibility_trace', 0.0) > 0.1):
        return True
    return False


def energy_above_threshold(node: Dict[str, Any]) -> bool:

    energy = node.get('energy', 0.0)
    threshold = node.get('threshold', 0.5)
    return energy >= threshold


def has_active_connections(node: Dict[str, Any], graph: Data) -> bool:

    if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
        return False
    node_id = node.get('id', -1)
    if node_id < 0 or node_id >= len(graph.node_labels):
        return False
    has_outgoing = torch.any(graph.edge_index[0] == node_id)
    has_incoming = torch.any(graph.edge_index[1] == node_id)
    energy = node.get('energy', 0.0)
    has_energy = energy > 0.3
    return (has_outgoing or has_incoming) and has_energy







