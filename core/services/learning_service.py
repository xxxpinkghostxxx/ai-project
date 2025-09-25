"""
LearningService implementation - Energy-modulated learning service.

This module provides the concrete implementation of ILearningEngine,
handling STDP learning, memory consolidation, and plasticity mechanisms
with energy-based modulation for biologically plausible learning dynamics.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from torch_geometric.data import Data

from ..interfaces.learning_engine import ILearningEngine, LearningState, PlasticityEvent
from ..interfaces.energy_manager import IEnergyManager
from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.event_coordinator import IEventCoordinator
from ..interfaces.neural_processor import SpikeEvent


class LearningService(ILearningEngine):
    """
    Concrete implementation of ILearningEngine.

    This service manages all learning and plasticity mechanisms with
    energy-based modulation, ensuring that learning rates and plasticity
    are biologically plausible and energy-dependent.
    """

    def __init__(self,
                 energy_manager: IEnergyManager,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator):
        """
        Initialize the LearningService.

        Args:
            energy_manager: Service for energy state coordination
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
        """
        self.energy_manager = energy_manager
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator

        # Learning parameters
        self._learning_state = LearningState()
        self._stdp_window = 20.0  # ms
        self._ltp_rate = 0.02
        self._ltd_rate = 0.01
        self._eligibility_decay = 0.95
        self._consolidation_threshold = 0.5

        # Energy modulation settings
        self._energy_learning_modulation = True
        self._energy_cap = 5.0  # Updated for better modulation

        # Learning tracking
        self._plasticity_events: List[PlasticityEvent] = []
        self._memory_traces: Dict[int, Dict[str, Any]] = {}
        self._eligibility_traces: Dict[Tuple[int, int], float] = {}

    def initialize_learning_state(self, graph: Data) -> bool:
        """
        Initialize learning state for the neural graph.

        Args:
            graph: Neural graph to initialize

        Returns:
            bool: True if initialization successful
        """
        try:
            if graph is None:
                return False

            # Reset learning state
            self._plasticity_events.clear()
            self._memory_traces.clear()
            self._eligibility_traces.clear()

            # Initialize learning state
            self._learning_state.is_initialized = True
            self._learning_state.total_plasticity_events = 0
            self._learning_state.active_memory_traces = 0

            return True

        except Exception as e:
            print(f"Failed to initialize learning state: {e}")
            return False

    def modulate_learning_by_energy(self, graph: Data, energy_levels: Dict[int, float]) -> Data:
        """
        Modulate learning parameters based on energy levels.

        Args:
            graph: Neural graph
            energy_levels: Current energy levels for all nodes

        Returns:
            Updated neural graph
        """
        if graph is None or not hasattr(graph, 'node_labels'):
            return graph

        try:
            for i, node in enumerate(graph.node_labels):
                node_id = node.get('id', i)
                energy = energy_levels.get(node_id, 1.0)

                # Energy-modulated learning parameters
                if energy < 0.3:
                    # Low energy: disable learning
                    node['learning_enabled'] = False
                    node['plasticity_rate'] = 0.0
                elif energy > 0.7:
                    # High energy: enhanced learning
                    node['learning_enabled'] = True
                    node['plasticity_rate'] = self._ltp_rate * 1.5
                else:
                    # Moderate energy: normal learning
                    node['learning_enabled'] = True
                    node['plasticity_rate'] = self._ltp_rate

            return graph

        except Exception as e:
            print(f"Error modulating learning by energy: {e}")
            return graph

    def apply_stdp_learning(self, graph: Data, pre_spikes: List[Tuple[int, float]],
                           post_spikes: List[Tuple[int, float]]) -> Tuple[Data, List[PlasticityEvent]]:
        """
        Apply Spike-Timing Dependent Plasticity (STDP) learning.

        Args:
            graph: Neural graph
            pre_spikes: List of (neuron_id, timestamp) for pre-synaptic spikes
            post_spikes: List of (neuron_id, timestamp) for post-synaptic spikes

        Returns:
            Tuple of (updated_graph, plasticity_events)
        """
        plasticity_events = []

        if graph is None or not hasattr(graph, 'edge_attributes'):
            return graph, plasticity_events

        try:
            # Get current energy state
            energy_state = self.energy_manager.get_energy_state()
            node_energies = energy_state.node_energies if hasattr(energy_state, 'node_energies') else {}

            # Process STDP for each edge
            for edge_idx, edge in enumerate(graph.edge_attributes):
                if edge is None:
                    continue

                source_id = edge.source
                target_id = edge.target

                # Find relevant spikes
                pre_spike_times = [t for nid, t in pre_spikes if nid == source_id]
                post_spike_times = [t for nid, t in post_spikes if nid == target_id]

                if not pre_spike_times or not post_spike_times:
                    continue

                # Calculate STDP weight change
                weight_change = self._calculate_stdp_change(
                    pre_spike_times, post_spike_times,
                    source_id, target_id, node_energies
                )

                if abs(weight_change) > 0.001:
                    # Apply energy-modulated weight change
                    energy_factor = self._calculate_energy_modulation_factor(
                        source_id, target_id, node_energies
                    )

                    modulated_change = weight_change * energy_factor
                    new_weight = edge.weight + modulated_change
                    new_weight = max(0.1, min(5.0, new_weight))  # Clamp weights

                    actual_change = new_weight - edge.weight
                    edge.weight = new_weight

                    # Update eligibility trace
                    edge_key = (source_id, target_id)
                    current_trace = self._eligibility_traces.get(edge_key, 0.0)
                    self._eligibility_traces[edge_key] = current_trace + actual_change

                    # Create plasticity event
                    plasticity_event = PlasticityEvent(
                        source_neuron=source_id,
                        target_neuron=target_id,
                        weight_change=actual_change,
                        plasticity_type="stdp",
                        timestamp=time.time(),
                        energy_factor=energy_factor
                    )
                    plasticity_events.append(plasticity_event)

                    # Publish event
                    self.event_coordinator.publish("plasticity_event", {
                        "source_neuron": source_id,
                        "target_neuron": target_id,
                        "weight_change": actual_change,
                        "plasticity_type": "stdp",
                        "energy_factor": energy_factor
                    })

            # Update learning state
            self._plasticity_events.extend(plasticity_events)
            self._learning_state.total_plasticity_events = len(self._plasticity_events)

            return graph, plasticity_events

        except Exception as e:
            print(f"Error applying STDP learning: {e}")
            return graph, plasticity_events

    def consolidate_memories(self, graph: Data) -> Tuple[Data, List[PlasticityEvent]]:
        """
        Consolidate memory traces based on eligibility traces.

        Args:
            graph: Neural graph

        Returns:
            Tuple of (updated_graph, consolidation_events)
        """
        consolidation_events = []

        if graph is None or not hasattr(graph, 'edge_attributes'):
            return graph, consolidation_events

        try:
            # Get current energy state
            energy_state = self.energy_manager.get_energy_state()
            node_energies = energy_state.node_energies if hasattr(energy_state, 'node_energies') else {}

            for edge_idx, edge in enumerate(graph.edge_attributes):
                if edge is None:
                    continue

                source_id = edge.source
                target_id = edge.target
                edge_key = (source_id, target_id)

                eligibility_trace = self._eligibility_traces.get(edge_key, 0.0)

                if eligibility_trace > self._consolidation_threshold:
                    # Energy-modulated consolidation
                    energy_factor = self._calculate_energy_modulation_factor(
                        source_id, target_id, node_energies
                    )

                    consolidation_strength = eligibility_trace * energy_factor
                    new_weight = edge.weight + consolidation_strength * 0.1
                    new_weight = max(0.1, min(5.0, new_weight))

                    weight_change = new_weight - edge.weight
                    edge.weight = new_weight

                    # Decay eligibility trace
                    self._eligibility_traces[edge_key] *= 0.5

                    # Create consolidation event
                    consolidation_event = PlasticityEvent(
                        source_neuron=source_id,
                        target_neuron=target_id,
                        weight_change=weight_change,
                        plasticity_type="consolidation",
                        timestamp=time.time(),
                        energy_factor=energy_factor
                    )
                    consolidation_events.append(consolidation_event)

            return graph, consolidation_events

        except Exception as e:
            print(f"Error consolidating memories: {e}")
            return graph, consolidation_events

    def reset_learning_state(self) -> bool:
        """
        Reset learning state to initial conditions.

        Returns:
            bool: True if reset successful
        """
        try:
            self._plasticity_events.clear()
            self._memory_traces.clear()
            self._eligibility_traces.clear()
            self._learning_state = LearningState()
            return True
        except Exception as e:
            print(f"Error resetting learning state: {e}")
            return False

    def get_learning_state(self) -> LearningState:
        """Get current learning state."""
        return self._learning_state

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_plasticity_events": len(self._plasticity_events),
            "active_memory_traces": len(self._memory_traces),
            "eligibility_traces": len(self._eligibility_traces),
            "average_energy_modulation": self._calculate_average_energy_modulation(),
            "learning_efficiency": self._calculate_learning_efficiency()
        }

    def _calculate_stdp_change(self, pre_spikes: List[float], post_spikes: List[float],
                              source_id: int, target_id: int, node_energies: Dict[int, float]) -> float:
        """
        Calculate STDP weight change for a synapse.

        Args:
            pre_spikes: Pre-synaptic spike times
            post_spikes: Post-synaptic spike times
            source_id: Source neuron ID
            target_id: Target neuron ID
            node_energies: Current energy levels

        Returns:
            float: Weight change
        """
        weight_change = 0.0

        for pre_time in pre_spikes[-10:]:  # Last 10 spikes
            for post_time in post_spikes[-10:]:
                time_diff = post_time - pre_time

                if 0 < time_diff < self._stdp_window / 1000.0:  # LTP
                    weight_change += self._ltp_rate * np.exp(-time_diff * 1000.0 / self._stdp_window)
                elif -self._stdp_window / 1000.0 < time_diff < 0:  # LTD
                    weight_change -= self._ltd_rate * np.exp(time_diff * 1000.0 / self._stdp_window)

        return weight_change

    def _calculate_energy_modulation_factor(self, source_id: int, target_id: int,
                                          node_energies: Dict[int, float]) -> float:
        """
        Calculate energy modulation factor for learning.

        Args:
            source_id: Source neuron ID
            target_id: Target neuron ID
            node_energies: Current energy levels

        Returns:
            float: Energy modulation factor (0.3 to 1.5)
        """
        if not self._energy_learning_modulation:
            return 1.0

        source_energy = node_energies.get(source_id, 1.0)
        target_energy = node_energies.get(target_id, 1.0)
        avg_energy = (source_energy + target_energy) / 2.0

        # Normalize and apply modulation
        normalized_energy = min(avg_energy / self._energy_cap, 1.0)
        modulation_factor = 0.3 + 1.2 * normalized_energy  # Range: 0.3x to 1.5x

        return modulation_factor

    def _calculate_average_energy_modulation(self) -> float:
        """Calculate average energy modulation across recent events."""
        if not self._plasticity_events:
            return 1.0

        recent_events = self._plasticity_events[-100:]  # Last 100 events
        energy_factors = [event.energy_factor for event in recent_events if hasattr(event, 'energy_factor')]

        if energy_factors:
            return np.mean(energy_factors)
        return 1.0

    def _calculate_learning_efficiency(self) -> float:
        """Calculate learning efficiency metric."""
        if not self._plasticity_events:
            return 0.0

        # Efficiency based on plasticity event frequency and energy modulation
        event_rate = len(self._plasticity_events) / max(1, time.time() - self._learning_state.initialization_time)
        avg_modulation = self._calculate_average_energy_modulation()

        return min(1.0, event_rate * avg_modulation * 0.1)

    def apply_hebbian_learning(self, graph: Data, correlation_data: Dict[Tuple[int, int], float]) -> Tuple[Data, List[PlasticityEvent]]:
        """
        Apply Hebbian learning based on neural activity correlations.

        "Neurons that fire together wire together" - this method implements
        correlation-based synaptic strengthening.

        Args:
            graph: Current neural graph
            correlation_data: Dictionary mapping (pre, post) neuron pairs to correlation strengths

        Returns:
            Tuple[Data, List[PlasticityEvent]]: Updated graph and plasticity events
        """
        plasticity_events = []

        if graph is None or not hasattr(graph, 'edge_attributes'):
            return graph, plasticity_events

        try:
            # Get current energy state
            energy_state = self.energy_manager.get_energy_state()
            node_energies = energy_state.node_energies if hasattr(energy_state, 'node_energies') else {}

            # Apply Hebbian learning based on correlations
            for edge_idx, edge in enumerate(graph.edge_attributes):
                if edge is None:
                    continue

                source_id = edge.source
                target_id = edge.target
                edge_key = (source_id, target_id)

                correlation = correlation_data.get(edge_key, 0.0)

                if abs(correlation) > 0.1:  # Only apply to correlated pairs
                    # Energy-modulated Hebbian learning
                    energy_factor = self._calculate_energy_modulation_factor(
                        source_id, target_id, node_energies
                    )

                    # Hebbian weight change
                    hebbian_change = correlation * self._ltp_rate * energy_factor * 0.1
                    new_weight = edge.weight + hebbian_change
                    new_weight = max(0.1, min(5.0, new_weight))

                    weight_change = new_weight - edge.weight
                    edge.weight = new_weight

                    # Create plasticity event
                    plasticity_event = PlasticityEvent(
                        source_neuron=source_id,
                        target_neuron=target_id,
                        weight_change=weight_change,
                        plasticity_type="hebbian",
                        timestamp=time.time(),
                        energy_factor=energy_factor
                    )
                    plasticity_events.append(plasticity_event)

            # Update learning state
            self._plasticity_events.extend(plasticity_events)
            self._learning_state.total_plasticity_events = len(self._plasticity_events)

            return graph, plasticity_events

        except Exception as e:
            print(f"Error applying Hebbian learning: {e}")
            return graph, plasticity_events

    def apply_homeostatic_scaling(self, graph: Data) -> Data:
        """
        Apply homeostatic scaling to maintain network stability.

        Args:
            graph: Neural graph

        Returns:
            Updated neural graph
        """
        if graph is None or not hasattr(graph, 'edge_attributes'):
            return graph

        try:
            # Simple homeostatic scaling based on activity
            for edge in graph.edge_attributes:
                if edge is None:
                    continue

                # Scale weights toward target range
                target_weight = 2.5  # Target weight
                scaling_factor = 0.99  # Slow scaling

                if edge.weight > target_weight:
                    edge.weight = edge.weight * scaling_factor
                elif edge.weight < target_weight:
                    edge.weight = edge.weight / scaling_factor

                # Clamp weights
                edge.weight = max(0.1, min(5.0, edge.weight))

            return graph

        except Exception as e:
            print(f"Error applying homeostatic scaling: {e}")
            return graph

    def apply_structural_plasticity(self, graph: Data) -> Tuple[Data, List[Dict[str, Any]]]:
        """
        Apply structural plasticity (synaptogenesis and pruning).

        This method implements the creation and removal of synaptic connections
        based on activity patterns and learning requirements.

        Args:
            graph: Current neural graph

        Returns:
            Tuple[Data, List[Dict[str, Any]]]: Updated graph and structural changes
        """
        structural_changes = []

        if graph is None:
            return graph, structural_changes

        try:
            # Simple structural plasticity based on activity patterns
            # This is a placeholder implementation

            # For demonstration, we'll add some structural changes
            # In a real implementation, this would analyze activity patterns
            # and add/remove connections accordingly

            # Example: Add a new connection if certain conditions are met
            # (This is simplified - real implementation would be more complex)

            if hasattr(graph, 'node_labels') and len(graph.node_labels) > 10:
                # Simulate adding a connection between highly active nodes
                structural_changes.append({
                    "action": "add_connection",
                    "source_id": 1,
                    "target_id": 2,
                    "weight": 1.0,
                    "reason": "activity_based_growth"
                })

            return graph, structural_changes

        except Exception as e:
            print(f"Error applying structural plasticity: {e}")
            return graph, structural_changes

    def configure_learning_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Configure learning parameters.

        Args:
            parameters: Dictionary of learning parameter updates

        Returns:
            bool: True if parameters updated successfully
        """
        try:
            for key, value in parameters.items():
                if key == 'stdp_window':
                    self._stdp_window = float(value)
                elif key == 'ltp_rate':
                    self._ltp_rate = float(value)
                elif key == 'ltd_rate':
                    self._ltd_rate = float(value)
                elif key == 'eligibility_decay':
                    self._eligibility_decay = float(value)
                elif key == 'consolidation_threshold':
                    self._consolidation_threshold = float(value)
                elif key == 'energy_learning_modulation':
                    self._energy_learning_modulation = bool(value)

            return True
        except Exception as e:
            print(f"Error configuring learning parameters: {e}")
            return False

    def get_learning_metrics(self) -> Dict[str, float]:
        """
        Get learning metrics and statistics.

        Returns:
            Dict[str, float]: Learning metrics
        """
        return {
            "total_plasticity_events": len(self._plasticity_events),
            "active_memory_traces": len(self._memory_traces),
            "eligibility_traces": len(self._eligibility_traces),
            "learning_efficiency": self._calculate_learning_efficiency()
        }

    def update_eligibility_traces(self, graph: Data, time_step: float) -> Data:
        """
        Update eligibility traces for synaptic plasticity.

        Eligibility traces enable delayed reinforcement learning by maintaining
        a decaying record of recent synaptic activity.

        Args:
            graph: Current neural graph
            time_step: Time step for trace decay

        Returns:
            Data: Updated graph with new eligibility traces
        """
        if graph is None or not hasattr(graph, 'edge_attributes'):
            return graph

        try:
            for edge in graph.edge_attributes:
                if edge is None:
                    continue

                source_id = edge.source
                target_id = edge.target
                edge_key = (source_id, target_id)

                # Decay existing eligibility trace
                current_trace = self._eligibility_traces.get(edge_key, 0.0)
                decayed_trace = current_trace * self._eligibility_decay
                self._eligibility_traces[edge_key] = decayed_trace

            return graph

        except Exception as e:
            print(f"Error updating eligibility traces: {e}")
            return graph

    def validate_learning_integrity(self, graph: Optional[Data]) -> Dict[str, Any]:
        """
        Validate learning system integrity.

        Args:
            graph: Neural graph to validate

        Returns:
            Dict with validation results
        """
        issues = []

        if not self._plasticity_events:
            issues.append("No plasticity events recorded")

        if not self._eligibility_traces:
            issues.append("No eligibility traces found")

        # Check for invalid weights
        if graph and hasattr(graph, 'edge_attributes'):
            for edge in graph.edge_attributes:
                if edge and (edge.weight < 0.1 or edge.weight > 5.0):
                    issues.append(f"Invalid weight {edge.weight} for edge {edge.source}->{edge.target}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "plasticity_events": len(self._plasticity_events),
            "eligibility_traces": len(self._eligibility_traces)
        }

    def apply_plasticity(self, graph: Data, spike_events: List[SpikeEvent]) -> Tuple[Data, List[PlasticityEvent]]:
        """
        Apply plasticity rules to the neural graph.

        Args:
            graph: Neural graph
            spike_events: List of spike events

        Returns:
            Tuple of (updated_graph, plasticity_events)
        """
        # This is a placeholder for a more sophisticated implementation
        # that would likely delegate to STDP, Hebbian, etc.
        return graph, []

    def cleanup(self) -> None:
        """Clean up resources."""
        self._plasticity_events.clear()
        self._memory_traces.clear()
        self._eligibility_traces.clear()