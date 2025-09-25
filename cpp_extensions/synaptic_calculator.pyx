# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Cython-accelerated synaptic input calculations for neural simulation.

This module provides optimized C implementations of synaptic input computation,
which is a major performance bottleneck in neural dynamics simulation.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, fabs
from cython.parallel import prange


# Type definitions for better performance
ctypedef cnp.float64_t DTYPE_FLOAT
ctypedef cnp.int64_t DTYPE_INT


cdef class SynapticCalculator:
    """
    Cython-optimized synaptic input calculator.

    Provides fast computation of synaptic inputs for neural dynamics,
    handling different edge types and connection strengths.
    """

    cdef:
        public double time_window
        public double excitatory_weight
        public double inhibitory_weight
        public double modulatory_weight
        public double gated_weight

    def __init__(self, time_window=0.1, excitatory_weight=1.0,
                 inhibitory_weight=-1.0, modulatory_weight=0.5, gated_weight=1.0):
        """Initialize the synaptic calculator with parameters."""
        self.time_window = time_window
        self.excitatory_weight = excitatory_weight
        self.inhibitory_weight = inhibitory_weight
        self.modulatory_weight = modulatory_weight
        self.gated_weight = gated_weight

    cpdef cnp.ndarray[DTYPE_FLOAT, ndim=1] calculate_synaptic_inputs(
        self,
        list edge_attributes,
        dict node_energy,
        double current_time,
        int num_nodes
    ):
        """
        Calculate synaptic inputs for all nodes using optimized Cython loops.

        Args:
            edge_attributes: List of edge attribute objects
            node_energy: Dictionary mapping node IDs to energy values
            current_time: Current simulation time
            num_nodes: Total number of nodes

        Returns:
            numpy array of synaptic inputs for each node
        """
        cdef:
            cnp.ndarray[DTYPE_FLOAT, ndim=1] synaptic_inputs = np.zeros(num_nodes, dtype=np.float64)
            int num_edges = len(edge_attributes)
            int edge_idx, source_id, target_id
            double weight, energy, time_diff, input_value
            object edge
            str edge_type

        # Process each edge
        for edge_idx in range(num_edges):
            edge = edge_attributes[edge_idx]
            if edge is None:
                continue

            # Extract edge properties with error checking
            try:
                source_id = getattr(edge, 'source', -1)
                target_id = getattr(edge, 'target', -1)
                if source_id < 0 or target_id < 0 or target_id >= num_nodes:
                    continue

                # Get effective weight
                weight = self._get_effective_weight(edge)
                if fabs(weight) < 1e-10:  # Skip very small weights
                    continue

                # Check if spike is recent enough
                time_diff = self._get_spike_time_diff(edge, current_time)
                if time_diff > self.time_window or time_diff < 0:
                    continue

                # Apply edge type modulation
                edge_type = getattr(edge, 'type', 'unknown')
                input_value = self._apply_edge_modulation(
                    weight, edge_type, source_id, node_energy, edge
                )

                # Add to synaptic input (atomic operation for thread safety if needed)
                synaptic_inputs[target_id] += input_value

            except (AttributeError, TypeError, ValueError):
                # Skip malformed edges
                continue

        return synaptic_inputs

    cdef double _get_effective_weight(self, object edge) nogil:
        """Get the effective weight of an edge."""
        cdef double weight = 0.0

        # Try different weight access methods
        if hasattr(edge, 'get_effective_weight'):
            try:
                weight = (<object>edge).get_effective_weight()
            except:
                weight = 0.0
        else:
            weight = getattr(edge, 'weight', 0.0)

        # Clamp weight to reasonable bounds
        if weight > 100.0:
            weight = 100.0
        elif weight < -100.0:
            weight = -100.0

        return weight

    cdef double _get_spike_time_diff(self, object edge, double current_time) nogil:
        """Calculate time difference since last spike."""
        cdef double last_spike = 0.0

        try:
            # Try to get last spike time from source node
            source_node = getattr(edge, 'source_node', None)
            if source_node is not None:
                last_spike = getattr(source_node, 'last_spike_time', 0.0)
            else:
                # Fallback to edge attribute
                last_spike = getattr(edge, 'last_spike_time', 0.0)

            return current_time - last_spike
        except:
            return 999.0  # Large value to indicate no recent spike

    cdef double _apply_edge_modulation(self, double weight, str edge_type,
                                     int source_id, dict node_energy, object edge):
        """Apply edge-type specific modulation to the weight."""
        cdef double modulated_weight = weight
        cdef double energy = 0.0

        # Get source node energy for gated connections
        try:
            energy = node_energy.get(source_id, 0.0)
        except (KeyError, TypeError):
            energy = 0.0

        # Apply edge type modulation
        if edge_type == 'excitatory':
            modulated_weight *= self.excitatory_weight
        elif edge_type == 'inhibitory':
            modulated_weight *= self.inhibitory_weight
        elif edge_type == 'modulatory':
            modulated_weight *= self.modulatory_weight
        elif edge_type == 'gated':
            # Gated connections only active when source has sufficient energy
            gate_threshold = getattr(edge, 'gate_threshold', 0.5)
            if energy >= gate_threshold:
                modulated_weight *= self.gated_weight
            else:
                modulated_weight = 0.0

        return modulated_weight

    cpdef cnp.ndarray[DTYPE_FLOAT, ndim=1] calculate_batch_synaptic_inputs(
        self,
        list edge_attributes,
        dict node_energy,
        double current_time,
        int num_nodes,
        int batch_size=1000
    ):
        """
        Calculate synaptic inputs in batches for better cache performance.

        Args:
            edge_attributes: List of edge attribute objects
            node_energy: Dictionary mapping node IDs to energy values
            current_time: Current simulation time
            num_nodes: Total number of nodes
            batch_size: Size of processing batches

        Returns:
            numpy array of synaptic inputs for each node
        """
        cdef:
            cnp.ndarray[DTYPE_FLOAT, ndim=1] synaptic_inputs = np.zeros(num_nodes, dtype=np.float64)
            int num_edges = len(edge_attributes)
            int start_idx, end_idx

        # Process edges in batches
        for start_idx in range(0, num_edges, batch_size):
            end_idx = min(start_idx + batch_size, num_edges)
            batch_edges = edge_attributes[start_idx:end_idx]

            # Calculate inputs for this batch
            batch_inputs = self._calculate_batch_inputs(
                batch_edges, node_energy, current_time, num_nodes
            )

            # Accumulate results
            synaptic_inputs += batch_inputs

        return synaptic_inputs

    cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] _calculate_batch_inputs(
        self, list batch_edges, dict node_energy, double current_time, int num_nodes
    ):
        """Calculate synaptic inputs for a batch of edges."""
        cdef:
            cnp.ndarray[DTYPE_FLOAT, ndim=1] batch_inputs = np.zeros(num_nodes, dtype=np.float64)
            int num_batch_edges = len(batch_edges)
            int edge_idx, source_id, target_id
            double weight, input_value
            object edge

        for edge_idx in range(num_batch_edges):
            edge = batch_edges[edge_idx]
            if edge is None:
                continue

            try:
                source_id = getattr(edge, 'source', -1)
                target_id = getattr(edge, 'target', -1)
                if source_id < 0 or target_id < 0 or target_id >= num_nodes:
                    continue

                weight = self._get_effective_weight(edge)
                if fabs(weight) < 1e-10:
                    continue

                time_diff = self._get_spike_time_diff(edge, current_time)
                if time_diff > self.time_window or time_diff < 0:
                    continue

                edge_type = getattr(edge, 'type', 'unknown')
                input_value = self._apply_edge_modulation(
                    weight, edge_type, source_id, node_energy, edge
                )

                batch_inputs[target_id] += input_value

            except (AttributeError, TypeError, ValueError):
                continue

        return batch_inputs


def create_synaptic_calculator(time_window=0.1):
    """Factory function to create a SynapticCalculator instance."""
    return SynapticCalculator(time_window=time_window)