"""
EnergyManagementService implementation - Central energy integrator service.

This module provides the concrete implementation of IEnergyManager,
handling energy flow, conservation, metabolic costs, and homeostasis
while serving as the central integrator for all neural simulation modules.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch_geometric.data import Data

from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.energy_manager import EnergyFlow, EnergyState, IEnergyManager
from ..interfaces.event_coordinator import IEventCoordinator


class EnergyManagementService(IEnergyManager):
    """
    Concrete implementation of IEnergyManager.

    This service manages all energy-related aspects of the neural simulation,
    serving as the central integrator that coordinates neural activity,
    learning, and system homeostasis.
    """

    def __init__(self,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator):
        """
        Initialize the EnergyManagementService.

        Args:
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
        """
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator

        # Energy management parameters
        self._energy_state = EnergyState()
        self._energy_cap = 5.0  # Updated from 255.0 for better modulation
        self._decay_rate = 0.99
        self._metabolic_cost_per_spike = 0.1
        self._homeostasis_target = 1.0
        self._homeostasis_strength = 0.5

        # Energy tracking
        self._node_energies: Dict[int, float] = {}
        self._energy_flows: List[EnergyFlow] = []
        self._total_energy_history: List[float] = []

    def initialize_energy_state(self, graph: Data) -> bool:
        """
        Initialize energy state for the neural graph.

        Args:
            graph: Neural graph to initialize

        Returns:
            bool: True if initialization successful
        """
        try:
            if graph is None or not hasattr(graph, 'node_labels'):
                logging.error("Graph is None or missing node_labels attribute")
                return False

            if graph.node_labels is None:
                return False

            # Initialize energy for all nodes
            self._node_energies.clear()

            for i, node in enumerate(graph.node_labels):
                node_id = node.get('id', i)
                # Use existing energy if present, otherwise initialize with moderate levels
                # for diversity
                if 'energy' in node:
                    energy_value = node['energy']
                else:
                    energy_value = 0.5 + 0.4 * (i % 5) / 4.0  # Range: 0.5 to 0.9
                energy_value = min(energy_value, self.energy_cap)

                self._node_energies[node_id] = energy_value

                # Update node in graph
                node['energy'] = energy_value
                if hasattr(graph, 'x') and graph.x is not None and i < len(graph.x):
                    graph.x[i, 0] = energy_value

            # Initialize energy state
            self._energy_state.is_initialized = True
            self._energy_state.total_system_energy = self.get_total_system_energy()
            self._energy_state.node_energies = self.node_energies

            return True

        except RuntimeError as e:
            logging.error("Failed to initialize energy state: %s", e)
            return False

    def update_energy_flows(
        self, graph: Data, spike_events: List[Any]
    ) -> Tuple[Data, List[EnergyFlow]]:
        """
        Update energy flows based on neural activity.

        Args:
            graph: Neural graph
            spike_events: List of recent spike events

        Returns:
            Tuple of (updated_graph, energy_flows)
        """
        energy_flows = []

        if graph is None:
            return graph, energy_flows

        try:
            # Apply metabolic decay to all nodes
            for node_id, energy in self.node_energies.items():
                decayed_energy = energy * self.decay_rate
                self.set_node_energy(node_id, decayed_energy)

                # Update graph
                self._update_node_energy_in_graph(graph, node_id, decayed_energy)

                # Record energy flow
                if abs(energy - decayed_energy) > 0.001:
                    energy_flows.append(EnergyFlow(
                        source_id=node_id,
                        target_id=node_id,
                        amount=decayed_energy - energy,
                        flow_type="decay"
                    ))

            # Apply metabolic costs for spiking neurons
            for spike_event in spike_events:
                node_id = spike_event.neuron_id
                if node_id in self.node_energies:
                    current_energy = self.get_node_energy(node_id)
                    metabolic_cost = min(current_energy * 0.1, self.metabolic_cost_per_spike)
                    new_energy = current_energy - metabolic_cost
                    self.set_node_energy(node_id, new_energy)

                    # Update graph
                    self._update_node_energy_in_graph(graph, node_id, new_energy)

                    # Record energy flow
                    energy_flows.append(EnergyFlow(
                        source_id=node_id,
                        target_id=node_id,
                        amount=-metabolic_cost,
                        flow_type="metabolic_cost"
                    ))

            # Update energy state
            self.update_energy_state_totals()
            self._energy_flows.extend(energy_flows)

            return graph, energy_flows

        except RuntimeError as e:
            logging.error("Error updating energy flows: %s", e)
            return graph, energy_flows

    def apply_metabolic_costs(self, graph: Data, time_step: float) -> Data:
        """
        Apply metabolic energy costs for neural computation and maintenance.

        This method implements energy costs for various neural activities:
        spiking, synaptic transmission, membrane maintenance, etc.

        Args:
            graph: Current neural graph
            time_step: Time step for metabolic cost calculation

        Returns:
            Data: Updated graph with applied metabolic costs
        """
        if graph is None:
            return graph

        try:
            # Use time_step to calculate metabolic rate
            metabolic_rate = 0.001 * time_step  # Base metabolic rate scaled by time step

            for node_id, energy in self.node_energies.items():
                metabolic_cost = energy * metabolic_rate
                new_energy = energy - metabolic_cost
                self.set_node_energy(node_id, new_energy)

                self._update_node_energy_in_graph(graph, node_id, new_energy)

            return graph

        except RuntimeError as e:
            logging.error("Error applying metabolic costs: %s", e)
            return graph

    def regulate_energy_homeostasis(self, graph: Data) -> Data:
        """
        Regulate energy homeostasis to maintain system stability.

        Args:
            graph: Neural graph

        Returns:
            Updated neural graph
        """
        if graph is None:
            return graph

        try:
            total_energy = self.get_total_system_energy()
            num_nodes = len(self.node_energies)
            if num_nodes > 0:
                average_energy = total_energy / num_nodes

                for node_id, energy in self.node_energies.items():
                    # Apply homeostasis adjustment towards average
                    adjustment = self.homeostasis_strength * (average_energy - energy)
                    new_energy = energy + adjustment
                    new_energy = max(0.1, min(new_energy, self.energy_cap))

                    self.set_node_energy(node_id, new_energy)
                    self._update_node_energy_in_graph(graph, node_id, new_energy)

            return graph

        except RuntimeError as e:
            logging.error("Error regulating energy homeostasis: %s", e)
            return graph

    def modulate_neural_activity(self, graph: Data) -> Data:
        """
        Modulate neural activity based on energy availability.

        Args:
            graph: Neural graph

        Returns:
            Updated neural graph
        """
        if graph is None or not hasattr(graph, 'node_labels'):
            return graph

        if graph.node_labels is None:
            return graph

        try:
            for i, node in enumerate(graph.node_labels):
                node_id = node.get('id', i)
                energy = self.get_node_energy(node_id)

                # Energy-modulated neural properties
                if energy < 0.3:
                    # Low energy: reduce excitability
                    node['plasticity_enabled'] = False
                    node['threshold'] = node.get('threshold', 0.5) * 1.2  # Higher threshold
                elif energy > 0.7:
                    # High energy: increase excitability
                    node['plasticity_enabled'] = True
                    node['threshold'] = node.get('threshold', 0.5) * 0.8  # Lower threshold
                else:
                    # Moderate energy: normal operation
                    node['plasticity_enabled'] = True
                    node['threshold'] = node.get('threshold', 0.5)

            return graph

        except RuntimeError as e:
            logging.error("Error modulating neural activity: %s", e)
            return graph

    def reset_energy_state(self) -> bool:
        """
        Reset energy state to initial conditions.

        Returns:
            bool: True if reset successful
        """
        try:
            # Clear using public interfaces
            self._node_energies.clear()
            self._energy_flows.clear()
            self._total_energy_history.clear()
            self._energy_state = EnergyState()
            return True
        except RuntimeError as e:
            logging.error("Error resetting energy state: %s", e)
            return False

    def get_energy_state(self) -> EnergyState:
        """Get current energy state."""
        return self._energy_state

    @property
    def energy_cap(self) -> float:
        """Get the energy capacity limit."""
        return self._energy_cap

    @property
    def decay_rate(self) -> float:
        """Get the energy decay rate."""
        return self._decay_rate

    @property
    def metabolic_cost_per_spike(self) -> float:
        """Get the metabolic cost per spike."""
        return self._metabolic_cost_per_spike

    @property
    def homeostasis_target(self) -> float:
        """Get the homeostasis target value."""
        return self._homeostasis_target

    @property
    def homeostasis_strength(self) -> float:
        """Get the homeostasis strength."""
        return self._homeostasis_strength

    @property
    def node_energies(self) -> Dict[int, float]:
        """Get the node energies dictionary."""
        return self._node_energies.copy()

    @property
    def energy_flows(self) -> List[EnergyFlow]:
        """Get the energy flows list."""
        return self._energy_flows.copy()

    @property
    def total_energy_history(self) -> List[float]:
        """Get the total energy history list."""
        return self._total_energy_history.copy()

    def get_node_energy(self, node_id: int) -> float:
        """Get energy for a specific node."""
        return self._node_energies.get(node_id, 0.0)

    def set_node_energy(self, node_id: int, energy: float) -> None:
        """Set energy for a specific node."""
        self._node_energies[node_id] = min(energy, self._energy_cap)

    def add_energy_flow(self, energy_flow: EnergyFlow) -> None:
        """Add an energy flow to the tracking list."""
        self._energy_flows.append(energy_flow)

    def get_total_system_energy(self) -> float:
        """Get the total energy across all nodes."""
        return sum(self._node_energies.values())

    def update_energy_state_totals(self) -> None:
        """Update the energy state totals."""
        self._energy_state.total_system_energy = self.get_total_system_energy()
        self._energy_state.node_energies = self.node_energies

    def get_energy_statistics(self) -> Dict[str, Any]:
        """Get energy management statistics."""
        if not self.node_energies:
            return {}

        energies = list(self.node_energies.values())
        return {
            "total_system_energy": sum(energies),
            "average_energy": np.mean(energies),
            "energy_variance": np.var(energies),
            "min_energy": min(energies),
            "max_energy": max(energies),
            "energy_distribution": np.histogram(energies, bins=10)[0].tolist(),
            "total_energy_flows": len(self.energy_flows)
        }

    def validate_energy_conservation(self, graph: Optional[Data]) -> Dict[str, Any]:
        """
        Validate energy conservation principles.

        Args:
            graph: Neural graph to validate

        Returns:
            Dict with validation results
        """
        issues = []

        if not self.node_energies:
            issues.append("No energy data available")
            return {"valid": False, "issues": issues}

        total_energy = self.get_total_system_energy()

        # Check for energy conservation (should not change dramatically)
        if self.total_energy_history:
            previous_total = self.total_energy_history[-1]
            energy_change_percent = (
                abs(total_energy - previous_total) / max(previous_total, 0.001) * 100
            )

            if energy_change_percent > 20.0:  # More than 20% change
                issues.append(f"Energy change too large: {energy_change_percent:.1f}%")

        self._total_energy_history.append(total_energy)

        # Keep only recent history
        if len(self._total_energy_history) > 100:
            self._total_energy_history.pop(0)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_energy": total_energy,
            "energy_change_percent": (
                energy_change_percent if 'energy_change_percent' in locals() else 0.0
            ),
            "energy_conservation_rate": 1.0 - (energy_change_percent / 100.0) \
                if 'energy_change_percent' in locals() else 1.0
        }

    def _update_node_energy_in_graph(self, graph: Data, node_id: int, energy: float) -> None:
        """
        Update node energy in the graph structure.

        Args:
            graph: Neural graph
            node_id: Node ID to update
            energy: New energy value
        """
        if not hasattr(graph, 'node_labels') or graph.node_labels is None:
            return

        for i, node in enumerate(graph.node_labels):
            if node.get('id') == node_id:
                node['energy'] = energy
                if hasattr(graph, 'x') and graph.x is not None and i < len(graph.x):
                    graph.x[i, 0] = energy
                break

    def calculate_energy_efficiency(self, graph: Data) -> float:
        """
        Calculate the energy efficiency of the neural system.

        Args:
            graph: Current neural graph

        Returns:
            float: Energy efficiency metric
        """
        if not self.node_energies:
            return 0.0

        total_energy = self.get_total_system_energy()
        if total_energy == 0:
            return 0.0

        # Simple efficiency metric based on energy distribution
        energy_variance = np.var(list(self.node_energies.values()))
        efficiency = 1.0 / (1.0 + energy_variance / total_energy)

        return min(1.0, efficiency)

    def configure_energy_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Configure energy management parameters.

        Args:
            parameters: Dictionary of energy parameter updates

        Returns:
            bool: True if parameters updated successfully
        """
        try:
            for key, value in parameters.items():
                if key == 'energy_cap':
                    self._energy_cap = float(value)
                elif key == 'decay_rate':
                    self._decay_rate = float(value)
                elif key == 'metabolic_cost_per_spike':
                    self._metabolic_cost_per_spike = float(value)
                elif key == 'homeostasis_target':
                    self._homeostasis_target = float(value)
                elif key == 'homeostasis_strength':
                    self._homeostasis_strength = float(value)

            return True
        except ValueError as e:
            logging.error("Error configuring energy parameters: %s", e)
            return False

    def get_energy_metrics(self) -> Dict[str, float]:
        """
        Get energy management metrics and statistics.

        Returns:
            Dict[str, float]: Energy metrics
        """
        if not self.node_energies:
            return {}

        energies = list(self.node_energies.values())
        return {
            "total_energy": sum(energies),
            "average_energy": np.mean(energies),
            "energy_variance": np.var(energies),
            "min_energy": min(energies),
            "max_energy": max(energies),
            "energy_efficiency": self.calculate_energy_efficiency(None),  # Simplified
            "energy_flow_count": len(self.energy_flows)
        }

    def apply_energy_boost(self, graph: Data, neuron_ids: List[int], boost_amount: float) -> Data:
        """
        Apply energy boost to specific neurons.

        Args:
            graph: Current neural graph
            neuron_ids: List of neuron IDs to boost
            boost_amount: Amount of energy to add

        Returns:
            Data: Updated graph with energy boosts applied
        """
        if graph is None:
            return graph

        try:
            for node_id in neuron_ids:
                if node_id in self.node_energies:
                    current_energy = self.get_node_energy(node_id)
                    new_energy = min(current_energy + boost_amount, self.energy_cap)
                    self.set_node_energy(node_id, new_energy)

                    self._update_node_energy_in_graph(graph, node_id, new_energy)

                    # Record energy flow
                    self.add_energy_flow(EnergyFlow(
                        source_id=-1,  # External source
                        target_id=node_id,
                        amount=boost_amount,
                        flow_type="boost"
                    ))

            return graph

        except RuntimeError as e:
            logging.error("Error applying energy boost: %s", e)
            return graph

    def detect_energy_anomalies(self, graph: Data) -> List[Dict[str, Any]]:
        """
        Detect energy anomalies in the neural system.

        Args:
            graph: Current neural graph

        Returns:
            List[Dict[str, Any]]: List of detected energy anomalies
        """
        anomalies = []

        if not self.node_energies:
            return anomalies

        try:
            energies = list(self.node_energies.values())
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)

            if std_energy == 0:
                return anomalies

            # Detect anomalies (energies more than 0.5 from mean)
            threshold = 0.5

            for node_id, energy in self.node_energies.items():
                deviation = abs(energy - mean_energy)
                if deviation > threshold:
                    anomaly = {
                        "node_id": node_id,
                        "energy": energy,
                        "mean_energy": mean_energy,
                        "deviation": deviation,
                        "anomaly_type": "energy_outlier",
                        "severity": "high" if deviation > 3 * std_energy else "medium"
                    }
                    anomalies.append(anomaly)

            return anomalies

        except RuntimeError as e:
            logging.error("Error detecting energy anomalies: %s", e)
            return anomalies

    def cleanup(self) -> None:
        """Clean up resources."""
        self._node_energies.clear()
        self._energy_flows.clear()
        self._total_energy_history.clear()
