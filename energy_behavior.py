
import time
import numpy as np
import torch
from logging_utils import log_step, log_node_state
from config_manager import get_learning_config, get_system_constants
from random_seed_manager import random_choice, random_uniform, random_normal, random_randn, random_rand, random_int, random_float, random_bool


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator == 0 or denominator <= 0:
        return fallback
    return numerator / denominator
from energy_constants import (
    EnergyConstants, OscillatorConstants, IntegratorConstants,
    RelayConstants, HighwayConstants
)
_energy_cap_cache = None
_energy_cap_cache_time = 0
_energy_cap_cache_ttl = 300


class EnergyCalculator:
    @staticmethod
    def calculate_energy_cap() -> float:
        return get_node_energy_cap()
    @staticmethod
    def calculate_energy_decay(current_energy: float, decay_rate: float) -> float:
        return current_energy * decay_rate
    @staticmethod
    def calculate_energy_transfer(energy: float, transfer_fraction: float) -> float:
        return energy * transfer_fraction
    @staticmethod
    def calculate_energy_boost(energy: float, boost_amount: float) -> float:
        return min(energy + boost_amount, EnergyCalculator.calculate_energy_cap())
    @staticmethod
    def calculate_membrane_potential(energy: float) -> float:
        energy_cap = EnergyCalculator.calculate_energy_cap()
        if energy_cap <= 0:
            return 0.0
        return min(energy / energy_cap, 1.0)
    @staticmethod
    def apply_energy_bounds(energy: float) -> float:
        return max(0, min(energy, EnergyCalculator.calculate_energy_cap()))


def get_node_energy_cap():
    global _energy_cap_cache, _energy_cap_cache_time
    import time
    current_time = time.time()
    if (_energy_cap_cache is not None and
        current_time - _energy_cap_cache_time < _energy_cap_cache_ttl):
        return _energy_cap_cache
    constants = get_system_constants()
    _energy_cap_cache = constants.get('node_energy_cap', 255.0)
    _energy_cap_cache_time = current_time
    return _energy_cap_cache
_ENERGY_CAP_PRECACHED = None


def _precache_energy_cap():
    global _ENERGY_CAP_PRECACHED
    if _ENERGY_CAP_PRECACHED is None:
        _ENERGY_CAP_PRECACHED = get_node_energy_cap()
    return _ENERGY_CAP_PRECACHED
_precache_energy_cap()


def update_node_energy_with_learning(graph, node_id, delta_energy):

    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    if not access_layer.is_valid_node_id(node_id):
        log_step("Invalid node_id for energy update", node_id=node_id)
        return graph
    current_energy = access_layer.get_node_energy(node_id)
    if current_energy is None:
        log_step("Failed to get energy for node", node_id=node_id)
        return graph
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        log_step("Failed to get node by ID", node_id=node_id)
        return graph
    new_energy = current_energy + delta_energy
    if 'membrane_potential' in node:
        membrane_pot = node['membrane_potential']
        config = get_learning_config()
        threshold = node.get('threshold', EnergyConstants.get_activation_threshold())
        if membrane_pot > threshold:
            node['last_activation'] = time.time()
            node['refractory_timer'] = node.get('refractory_period', EnergyConstants.get_refractory_period())
            if node.get('behavior') == 'oscillator':
                emit_energy_pulse(graph, node_id)
            log_step("Node activated",
                    node_id=node_id,
                    behavior=node.get('behavior', 'unknown'),
                    membrane_potential=membrane_pot,
                    threshold=threshold)
    if node.get('refractory_timer', 0) > 0:
        node['refractory_timer'] -= EnergyConstants.TIME_STEP
        new_energy = current_energy
        log_step("Refractory period active",
                node_id=node_id,
                remaining_time=node['refractory_timer'])
    if not node.get('plasticity_enabled', True):
        log_step("Plasticity disabled",
                node_id=node_id,
                reason="plasticity_gate_closed")
    new_energy = EnergyCalculator.apply_energy_bounds(new_energy)
    if not access_layer.set_node_energy(node_id, new_energy):
        log_step("Failed to update node energy", node_id=node_id, new_energy=new_energy)
        return graph
    if 'membrane_potential' in node:
        new_membrane_potential = EnergyCalculator.calculate_membrane_potential(new_energy)
        access_layer.update_node_property(node_id, 'membrane_potential', new_membrane_potential)
    return graph


def apply_energy_behavior(graph, behavior_params=None, _recursion_depth=0):

    MAX_RECURSION_DEPTH = 10
    if _recursion_depth > MAX_RECURSION_DEPTH:
        log_step("Energy behavior recursion limit reached", depth=_recursion_depth)
        return graph
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x') or graph.x is None:
        return graph
    if hasattr(graph, 'x') and graph.x is not None:
        current_energies = graph.x[:, 0]
        current_energies.mul_(0.99)
        current_energies.clamp_(min=0.0)
        energy_cap = _ENERGY_CAP_PRECACHED
        if energy_cap <= 0:
            energy_cap = 1.0
        membrane_potentials = current_energies / energy_cap
        membrane_potentials.clamp_(max=1.0)
        if len(graph.node_labels) > 10000:
            sample_size = min(10, len(graph.node_labels))
            for i in range(sample_size):
                node_idx = i * (len(graph.node_labels) // sample_size)
                if node_idx < len(graph.node_labels):
                    node = graph.node_labels[node_idx]
                    if 'membrane_potential' in node:
                        node['membrane_potential'] = float(membrane_potentials[node_idx].item())
        else:
            for i in range(len(graph.node_labels)):
                node = graph.node_labels[i]
                if 'membrane_potential' in node:
                    node['membrane_potential'] = float(membrane_potentials[i].item())
    return graph


def apply_oscillator_energy_dynamics(graph, node_id):

    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    current_time = time.time()
    last_activation = node.get('last_activation', 0)
    oscillation_freq = node.get('oscillation_freq', OscillatorConstants.get_oscillation_frequency())
    if oscillation_freq <= 0:
        oscillation_freq = 0.1
    period = 1.0 / oscillation_freq
    if current_time - last_activation >= period:
        current_energy = access_layer.get_node_energy(node_id)
        pulse_energy = EnergyCalculator.calculate_energy_transfer(current_energy, OscillatorConstants.PULSE_ENERGY_FRACTION)
        if hasattr(graph, 'edge_attributes') and hasattr(graph, '_edge_attributes_lock'):
            with graph._edge_attributes_lock:
                for edge in graph.edge_attributes:
                    if edge.source == node_id:
                        target_id = edge.target
                        if access_layer.get_node_by_id(target_id) is not None:
                            target_energy = access_layer.get_node_energy(target_id)
                            if target_energy is not None:
                                new_target_energy = EnergyCalculator.apply_energy_bounds(target_energy + pulse_energy)
                                access_layer.set_node_energy(target_id, new_target_energy)
                                access_layer.update_node_property(target_id, 'membrane_potential', EnergyCalculator.calculate_membrane_potential(new_target_energy))
        access_layer.update_node_property(node_id, 'last_activation', current_time)
        access_layer.update_node_property(node_id, 'refractory_timer', OscillatorConstants.REFRACTORY_PERIOD_SHORT)
        log_step("Oscillator pulse emitted",
                node_id=node_id,
                pulse_energy=pulse_energy,
                frequency=oscillation_freq)
    return graph


def apply_integrator_energy_dynamics(graph, node_id):

    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    integration_rate = node.get('integration_rate', IntegratorConstants.get_integration_rate())
    if hasattr(graph, 'edge_attributes') and hasattr(graph, '_edge_attributes_lock'):
        with graph._edge_attributes_lock:
            accumulated_energy = EnergyConstants.ELIGIBILITY_TRACE_DEFAULT
            for edge in graph.edge_attributes:
                if edge.target == node_id:
                    source_id = edge.source
                    if access_layer.get_node_by_id(source_id) is not None:
                        source_energy = access_layer.get_node_energy(source_id)
                        if source_energy is not None:
                            accumulated_energy += source_energy * edge.weight * integration_rate
            if accumulated_energy > 0:
                current_energy = access_layer.get_node_energy(node_id)
                if current_energy is not None:
                    new_energy = min(current_energy + accumulated_energy, get_node_energy_cap())
                    access_layer.set_node_energy(node_id, new_energy)
                    access_layer.update_node_property(node_id, 'membrane_potential', min(safe_divide(new_energy, get_node_energy_cap()), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
                    log_step("Integrator accumulated energy",
                            node_id=node_id,
                            accumulated=accumulated_energy,
                            new_total=new_energy)
    return graph


def apply_relay_energy_dynamics(graph, node_id):

    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    relay_amplification = node.get('relay_amplification', RelayConstants.get_relay_amplification())
    if hasattr(graph, 'edge_attributes'):
        current_energy = access_layer.get_node_energy(node_id)
        if current_energy is not None:
            transfer_energy = current_energy * RelayConstants.ENERGY_TRANSFER_FRACTION
            for edge in graph.edge_attributes:
                if edge.source == node_id:
                    target_id = edge.target
                    if access_layer.get_node_by_id(target_id) is not None:
                        amplified_transfer = transfer_energy * relay_amplification
                        target_energy = access_layer.get_node_energy(target_id)
                        if target_energy is not None:
                            new_target_energy = min(target_energy + amplified_transfer, get_node_energy_cap())
                            access_layer.set_node_energy(target_id, new_target_energy)
                            access_layer.update_node_property(target_id, 'membrane_potential', min(new_target_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
            new_relay_energy = max(current_energy - amplified_transfer, 0)
            access_layer.set_node_energy(node_id, new_relay_energy)
            access_layer.update_node_property(node_id, 'membrane_potential', min(new_relay_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
            log_step("Relay energy transfer",
                    node_id=node_id,
                    transfer_energy=transfer_energy,
                    amplification=relay_amplification)
    return graph


def apply_highway_energy_dynamics(graph, node_id):

    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    highway_energy_boost = node.get('highway_energy_boost', HighwayConstants.get_highway_energy_boost())
    current_energy = access_layer.get_node_energy(node_id)
    if current_energy is not None:
        if current_energy < HighwayConstants.ENERGY_THRESHOLD_LOW:
            boosted_energy = min(current_energy + HighwayConstants.ENERGY_BOOST_AMOUNT, get_node_energy_cap())
            access_layer.set_node_energy(node_id, boosted_energy)
            access_layer.update_node_property(node_id, 'membrane_potential', min(boosted_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
            log_step("Highway energy boosted",
                    node_id=node_id,
                    old_energy=current_energy,
                    new_energy=boosted_energy)
        if hasattr(graph, 'edge_attributes'):
            distribution_count = 0
            for edge in graph.edge_attributes:
                if edge.source == node_id:
                    target_id = edge.target
                    if access_layer.get_node_by_id(target_id) is not None:
                        distribution_energy = HighwayConstants.DISTRIBUTION_ENERGY_BASE * highway_energy_boost
                        target_energy = access_layer.get_node_energy(target_id)
                        if target_energy is not None:
                            new_target_energy = min(target_energy + distribution_energy, get_node_energy_cap())
                            access_layer.set_node_energy(target_id, new_target_energy)
                            access_layer.update_node_property(target_id, 'membrane_potential', min(new_target_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
                            distribution_count += 1
            if distribution_count > 0:
                log_step("Highway energy distribution",
                        node_id=node_id,
                        targets=distribution_count,
                        energy_per_target=10.0 * highway_energy_boost)
    return graph


def apply_dynamic_energy_dynamics(graph, node_id):

    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    node = access_layer.get_node_by_id(node_id)
    if node is None:
        return graph
    current_energy = access_layer.get_node_energy(node_id)
    if current_energy is None:
        return graph
    config = get_learning_config()
    decay_rate = EnergyConstants.get_decay_rate()
    decayed_energy = current_energy * decay_rate
    new_energy = max(current_energy - decayed_energy, 0)
    access_layer.set_node_energy(node_id, new_energy)
    access_layer.update_node_property(node_id, 'membrane_potential', min(new_energy / get_node_energy_cap(), EnergyConstants.MEMBRANE_POTENTIAL_CAP))
    if new_energy < EnergyConstants.get_plasticity_threshold():
        access_layer.update_node_property(node_id, 'plasticity_enabled', False)
        log_step("Plasticity disabled",
                node_id=node_id,
                reason="low_energy",
                energy=new_energy)
    else:
        access_layer.update_node_property(node_id, 'plasticity_enabled', True)
    return graph


def emit_energy_pulse(graph, source_node_id, _recursion_depth=0):

    MAX_RECURSION_DEPTH = 50
    if _recursion_depth > MAX_RECURSION_DEPTH:
        log_step("Energy pulse recursion limit reached",
                source_node=source_node_id,
                depth=_recursion_depth)
        return graph
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    if not access_layer.is_valid_node_id(source_node_id):
        log_step("Invalid source_node_id for energy pulse", source_node_id=source_node_id)
        return graph
    source_energy = access_layer.get_node_energy(source_node_id)
    if source_energy is None:
        log_step("Failed to get source energy for pulse", source_node_id=source_node_id)
        return graph
    pulse_energy = source_energy * EnergyConstants.PULSE_ENERGY_FRACTION_LARGE
    pulse_targets = []
    if hasattr(graph, 'edge_attributes'):
        for edge in graph.edge_attributes:
            if edge.source == source_node_id:
                target_id = edge.target
                if access_layer.is_valid_node_id(target_id):
                    target_energy = access_layer.get_node_energy(target_id)
                    if target_energy is not None:
                        new_target_energy = min(target_energy + pulse_energy, get_node_energy_cap())
                        if access_layer.set_node_energy(target_id, new_target_energy):
                            new_membrane_potential = min(safe_divide(new_target_energy, get_node_energy_cap()), EnergyConstants.MEMBRANE_POTENTIAL_CAP)
                            access_layer.update_node_property(target_id, 'membrane_potential', new_membrane_potential)
                            pulse_targets.append(target_id)
                        else:
                            log_step("Failed to update target energy", target_id=target_id)
                    else:
                        log_step("Failed to get target energy", target_id=target_id)
                else:
                    log_step("Invalid target_id for energy pulse", target_id=target_id)
    if pulse_targets:
        log_step("Energy pulse emitted",
                source_node=source_node_id,
                pulse_energy=pulse_energy,
                targets=pulse_targets)
    return graph


def update_membrane_potentials(graph):

    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return graph
    if len(graph.node_labels) > 1000:
        return graph
    from node_access_layer import NodeAccessLayer
    access_layer = NodeAccessLayer(graph)
    sample_size = min(10, len(graph.node_labels))
    try:
        all_node_ids = access_layer.get_all_active_ids()
        if not all_node_ids:
            return graph
        sample_step = max(1, len(all_node_ids) // sample_size)
        sampled_ids = all_node_ids[::sample_step][:sample_size]
        for node_id in sampled_ids:
            if access_layer.is_valid_node_id(node_id):
                energy = access_layer.get_node_energy(node_id)
                if energy is not None:
                    energy_cap = _ENERGY_CAP_PRECACHED
                    if energy_cap <= 0:
                        energy_cap = 1.0
                    membrane_potential = min(energy / energy_cap, EnergyConstants.MEMBRANE_POTENTIAL_CAP)
                    access_layer.update_node_property(node_id, 'membrane_potential', membrane_potential)
    except Exception as e:
        log_step("Membrane potential update failed, skipping", error=str(e))
    return graph


def apply_refractory_periods(graph):

    if not hasattr(graph, 'node_labels'):
        return graph
    current_time = time.time()
    time_step = EnergyConstants.TIME_STEP
    sample_size = min(100, len(graph.node_labels))
    refractory_count = 0
    for i in range(sample_size):
        node_idx = i * (len(graph.node_labels) // sample_size)
        if node_idx < len(graph.node_labels):
            node = graph.node_labels[node_idx]
            if 'refractory_timer' in node:
                refractory_timer = node.get('refractory_timer', 0)
                if refractory_timer > 0:
                    new_timer = max(refractory_timer - time_step, 0)
                    node['refractory_timer'] = new_timer
                if 'membrane_potential' in node:
                    node['membrane_potential'] = EnergyConstants.MEMBRANE_POTENTIAL_RESET
                refractory_count += 1
    return graph


def couple_sensory_energy_to_channel(graph):

    return graph


def propagate_sensory_energy(graph):

    return graph
