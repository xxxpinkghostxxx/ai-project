"""
Optimized synaptic input calculations for neural simulation.

This module provides accelerated computation of synaptic inputs using Numba JIT
compilation and optimized algorithms. This serves as both a fallback implementation
and a stepping stone toward full Cython/C++ acceleration.
"""

import numpy as np
import numba as nb
from numba import jit, prange
from typing import List, Dict, Any, Optional
import time


@jit(nopython=True, fastmath=True)
def _calculate_synaptic_inputs_numba(
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    edge_weights: np.ndarray,
    edge_types: np.ndarray,
    edge_spike_times: np.ndarray,
    gate_thresholds: np.ndarray,
    node_energies: np.ndarray,
    current_time: float,
    time_window: float,
    num_nodes: int
) -> np.ndarray:
    """
    Numba-optimized synaptic input calculation.

    Args:
        edge_sources: Source node IDs for each edge
        edge_targets: Target node IDs for each edge
        edge_weights: Weight values for each edge
        edge_types: Type codes for each edge (0=excitatory, 1=inhibitory, 2=modulatory, 3=gated)
        edge_spike_times: Last spike time for each edge
        gate_thresholds: Gate threshold for each edge
        node_energies: Energy values for each node
        current_time: Current simulation time
        time_window: Maximum time window for spike consideration
        num_nodes: Total number of nodes

    Returns:
        Array of synaptic inputs for each node
    """
    synaptic_inputs = np.zeros(num_nodes, dtype=np.float64)

    for edge_idx in range(len(edge_sources)):
        source_id = edge_sources[edge_idx]
        target_id = edge_targets[edge_idx]

        if source_id < 0 or target_id < 0 or target_id >= num_nodes:
            continue

        # Check if spike is recent enough
        time_diff = current_time - edge_spike_times[edge_idx]
        if time_diff > time_window or time_diff < 0:
            continue

        weight = edge_weights[edge_idx]
        abs_weight = abs(weight)
        if abs_weight < 1e-10:  # Skip very small weights
            continue

        edge_type = edge_types[edge_idx]
        energy = node_energies[source_id]

        # Apply edge type modulation
        if edge_type == 0:  # excitatory
            modulated_weight = abs_weight
        elif edge_type == 1:  # inhibitory
            modulated_weight = -abs_weight
        elif edge_type == 2:  # modulatory
            modulated_weight = abs_weight * 0.5
        elif edge_type == 3:  # gated
            if energy >= gate_thresholds[edge_idx]:
                modulated_weight = abs_weight
            else:
                modulated_weight = 0.0
        else:
            modulated_weight = abs_weight

        # If no energy, no contribution
        if energy == 0.0:
            modulated_weight = 0.0

        # Clamp to reasonable bounds
        modulated_weight = max(-100.0, min(100.0, modulated_weight))

        # Add to synaptic input
        synaptic_inputs[target_id] += modulated_weight

    return synaptic_inputs


class SynapticCalculator:
    """
    Optimized synaptic input calculator using Numba JIT compilation.

    Provides fast computation of synaptic inputs for neural dynamics,
    with a clean interface compatible with the existing codebase.
    """

    def __init__(self, time_window: float = 0.1):
        """
        Initialize the synaptic calculator.

        Args:
            time_window: Maximum time window for considering spikes (seconds)
        """
        self.time_window = time_window
        self.edge_type_map = {
            'excitatory': 0,
            'inhibitory': 1,
            'modulatory': 2,
            'gated': 3
        }

    def calculate_synaptic_inputs(
        self,
        edge_attributes: List[Any],
        node_energy: Dict[int, float],
        current_time: Optional[float] = None,
        num_nodes: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate synaptic inputs for all nodes.

        Args:
            edge_attributes: List of edge attribute objects
            node_energy: Dictionary mapping node IDs to energy values
            current_time: Current simulation time (uses time.time() if None)
            num_nodes: Total number of nodes (auto-detected if None)

        Returns:
            numpy array of synaptic inputs for each node
        """
        if current_time is None:
            current_time = time.time()

        # Pre-process edge data for Numba
        edge_data = self._preprocess_edges(edge_attributes, node_energy, current_time)

        if not edge_data['sources'].size:
            # No valid edges, return zeros
            if num_nodes is None:
                num_nodes = max(node_energy.keys()) + 1 if node_energy else 0
            return np.zeros(num_nodes, dtype=np.float64)

        if num_nodes is None:
            num_nodes = max(
                max(edge_data['targets']) + 1 if edge_data['targets'].size else 0,
                max(node_energy.keys()) + 1 if node_energy else 0
            )

        # Ensure node_energies array is large enough
        node_energies_array = np.zeros(num_nodes, dtype=np.float64)
        for node_id, energy in node_energy.items():
            if node_id < num_nodes:
                node_energies_array[node_id] = energy

        # Call optimized Numba function
        synaptic_inputs = _calculate_synaptic_inputs_numba(
            edge_data['sources'],
            edge_data['targets'],
            edge_data['weights'],
            edge_data['types'],
            edge_data['edge_spike_times'],
            edge_data['gate_thresholds'],
            node_energies_array,
            current_time,
            self.time_window,
            num_nodes
        )

        return synaptic_inputs

    def _preprocess_edges(self, edge_attributes: List[Any], node_energy: Dict[int, float], current_time: float) -> Dict[str, np.ndarray]:
        """
        Preprocess edge attributes into Numba-compatible arrays.

        Args:
            edge_attributes: List of edge objects
            node_energy: Node energy dictionary

        Returns:
            Dictionary with preprocessed edge data
        """
        if not edge_attributes:
            return {
                'sources': np.array([], dtype=np.int32),
                'targets': np.array([], dtype=np.int32),
                'weights': np.array([], dtype=np.float64),
                'types': np.array([], dtype=np.int32),
                'edge_spike_times': np.array([], dtype=np.float64),
                'gate_thresholds': np.array([], dtype=np.float64)
            }

        sources = []
        targets = []
        weights = []
        types = []
        edge_spike_times = []
        gate_thresholds = []

        for edge in edge_attributes:
            if edge is None:
                continue

            try:
                # Extract edge properties
                source_id = getattr(edge, 'source', -1)
                target_id = getattr(edge, 'target', -1)

                if source_id < 0 or target_id < 0:
                    continue

                # Get weight
                if hasattr(edge, 'get_effective_weight'):
                    weight = float(edge.get_effective_weight())
                else:
                    weight = float(getattr(edge, 'weight', 0.0))

                # Get edge type
                edge_type_str = getattr(edge, 'type', 'excitatory')
                edge_type = self.edge_type_map.get(edge_type_str, 0)

                # Get last spike time
                if hasattr(edge, 'last_spike_time') and isinstance(edge.last_spike_time, (int, float)):
                    last_spike_time = edge.last_spike_time
                else:
                    last_spike_time = current_time

                # Get gate threshold
                gate_threshold = getattr(edge, 'gate_threshold', 0.5)
                if not isinstance(gate_threshold, (int, float)):
                    gate_threshold = 0.5

                sources.append(source_id)
                targets.append(target_id)
                weights.append(weight)
                types.append(edge_type)
                edge_spike_times.append(last_spike_time)
                gate_thresholds.append(gate_threshold)

            except (AttributeError, ValueError, TypeError):
                continue

        return {
            'sources': np.array(sources, dtype=np.int32),
            'targets': np.array(targets, dtype=np.int32),
            'weights': np.array(weights, dtype=np.float64),
            'types': np.array(types, dtype=np.int32),
            'edge_spike_times': np.array(edge_spike_times, dtype=np.float64),
            'gate_thresholds': np.array(gate_thresholds, dtype=np.float64)
        }


def create_synaptic_calculator(time_window: float = 0.1) -> SynapticCalculator:
    """
    Factory function to create a SynapticCalculator instance.

    Args:
        time_window: Time window for spike consideration

    Returns:
        Configured SynapticCalculator instance
    """
    return SynapticCalculator(time_window=time_window)