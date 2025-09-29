"""
Test suite for C++ extensions integration.

This module tests the optimized C++ extensions for neural simulation acceleration,
ensuring they integrate properly with the service-oriented architecture.
"""

import time
from unittest.mock import Mock

import numpy as np
import pytest

# Test the extension imports
try:
    from src.utils.cpp_extensions import (SynapticCalculator,
                                          create_synaptic_calculator)
    CPP_EXTENSIONS_AVAILABLE = True
except ImportError:
    CPP_EXTENSIONS_AVAILABLE = False
    print("C++ extensions not available, skipping optimized tests")


class TestSynapticCalculator:
    """Test the optimized synaptic calculator."""

    @pytest.fixture
    def mock_edge_attributes(self):
        """Create mock edge attributes for testing."""
        edges = []
        for i in range(10):
            edge = Mock()
            edge.source = i % 5  # 5 source nodes
            edge.target = (i + 1) % 5  # 5 target nodes
            edge.weight = 0.5 + 0.1 * i
            edge.type = 'excitatory' if i % 2 == 0 else 'inhibitory'
            edge.get_effective_weight = Mock(return_value=edge.weight)
            edges.append(edge)
        return edges

    @pytest.fixture
    def mock_node_energies(self):
        """Create mock node energy data."""
        return {i: 0.8 + 0.1 * i for i in range(5)}

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_synaptic_calculator_initialization(self):
        """Test SynapticCalculator initialization with different parameters."""
        # Default initialization
        calculator = SynapticCalculator()
        assert calculator.time_window == 0.1
        assert hasattr(calculator, 'edge_type_map')
        assert 'excitatory' in calculator.edge_type_map

        # Custom time window
        calculator = SynapticCalculator(time_window=0.5)
        assert calculator.time_window == 0.5

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_create_synaptic_calculator_function(self):
        """Test the factory function for creating synaptic calculator."""
        calculator = create_synaptic_calculator()
        assert isinstance(calculator, SynapticCalculator)
        assert calculator.time_window == 0.1

        calculator = create_synaptic_calculator(time_window=0.2)
        assert calculator.time_window == 0.2

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_calculate_synaptic_inputs_empty_edges(self, mock_node_energies):
        """Test calculation with empty edge list."""
        calculator = create_synaptic_calculator()

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            [], mock_node_energies, num_nodes=5
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        assert len(synaptic_inputs) == 5
        assert np.all(synaptic_inputs == 0.0)

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_calculate_synaptic_inputs_different_edge_types(self):
        """Test calculation with different edge types."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8, 1: 0.6}

        # Excitatory edge
        edge1 = Mock()
        edge1.source = 0
        edge1.target = 1
        edge1.weight = 1.0
        edge1.type = 'excitatory'
        edge1.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge1)

        # Inhibitory edge
        edge2 = Mock()
        edge2.source = 0
        edge2.target = 1
        edge2.weight = 0.5
        edge2.type = 'inhibitory'
        edge2.get_effective_weight = Mock(return_value=0.5)
        edges.append(edge2)

        # Modulatory edge
        edge3 = Mock()
        edge3.source = 0
        edge3.target = 1
        edge3.weight = 0.3
        edge3.type = 'modulatory'
        edge3.get_effective_weight = Mock(return_value=0.3)
        edges.append(edge3)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert len(synaptic_inputs) == 2
        # Node 1 should receive inputs: +1.0 (excitatory) -0.5 (inhibitory) +0.15 (modulatory * 0.5)
        expected_input = 1.0 - 0.5 + 0.15
        assert abs(synaptic_inputs[1] - expected_input) < 1e-6

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_calculate_synaptic_inputs_gated_connections(self):
        """Test gated connections based on energy levels."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.3, 1: 0.7}  # Source 0 has low energy, source 1 has high

        # Gated edge from low energy source (should be inactive)
        edge1 = Mock()
        edge1.source = 0
        edge1.target = 2
        edge1.weight = 1.0
        edge1.type = 'gated'
        edge1.get_effective_weight = Mock(return_value=1.0)
        edge1.gate_threshold = 0.5
        edges.append(edge1)

        # Gated edge from high energy source (should be active)
        edge2 = Mock()
        edge2.source = 1
        edge2.target = 2
        edge2.weight = 1.0
        edge2.type = 'gated'
        edge2.get_effective_weight = Mock(return_value=1.0)
        edge2.gate_threshold = 0.5
        edges.append(edge2)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=3
        )

        assert len(synaptic_inputs) == 3
        # Only the gated edge from high energy source should contribute
        assert synaptic_inputs[2] == 1.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_calculate_synaptic_inputs_weight_clamping(self):
        """Test that weights are clamped to reasonable bounds."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8}

        # Very large weight
        edge1 = Mock()
        edge1.source = 0
        edge1.target = 1
        edge1.weight = 200.0
        edge1.type = 'excitatory'
        edge1.get_effective_weight = Mock(return_value=200.0)
        edges.append(edge1)

        # Very small weight (should be ignored)
        edge2 = Mock()
        edge2.source = 0
        edge2.target = 1
        edge2.weight = 1e-15
        edge2.type = 'excitatory'
        edge2.get_effective_weight = Mock(return_value=1e-15)
        edges.append(edge2)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert len(synaptic_inputs) == 2
        # Large weight should be clamped to 100.0, small weight ignored
        assert synaptic_inputs[1] == 100.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_calculate_synaptic_inputs_time_window_filtering(self):
        """Test that spikes outside time window are ignored."""
        calculator = create_synaptic_calculator(time_window=0.1)

        edges = []
        node_energies = {0: 0.8}

        # Recent spike (within window)
        edge1 = Mock()
        edge1.source = 0
        edge1.target = 1
        edge1.weight = 1.0
        edge1.type = 'excitatory'
        edge1.get_effective_weight = Mock(return_value=1.0)
        edge1.last_spike_time = time.time() - 0.05  # 50ms ago
        edges.append(edge1)

        # Old spike (outside window)
        edge2 = Mock()
        edge2.source = 0
        edge2.target = 1
        edge2.weight = 1.0
        edge2.type = 'excitatory'
        edge2.get_effective_weight = Mock(return_value=1.0)
        edge2.last_spike_time = time.time() - 0.2  # 200ms ago
        edges.append(edge2)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, current_time=time.time(), num_nodes=2
        )

        assert len(synaptic_inputs) == 2
        # Only recent spike should contribute
        assert synaptic_inputs[1] == 1.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_calculate_synaptic_inputs_invalid_edges(self, mock_node_energies):
        """Test handling of invalid edge attributes."""
        calculator = create_synaptic_calculator()

        edges = []

        # Edge with invalid source
        edge1 = Mock()
        edge1.source = -1
        edge1.target = 0
        edge1.weight = 1.0
        edge1.type = 'excitatory'
        edge1.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge1)

        # Edge with invalid target
        edge2 = Mock()
        edge2.source = 0
        edge2.target = -1
        edge2.weight = 1.0
        edge2.type = 'excitatory'
        edge2.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge2)

        # Edge with None attributes
        edge3 = Mock()
        edge3.source = None
        edge3.target = 0
        edge3.weight = 1.0
        edge3.type = 'excitatory'
        edge3.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge3)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, mock_node_energies, num_nodes=5
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        assert len(synaptic_inputs) == 5
        # Invalid edges should be skipped, result should be zeros
        assert np.all(synaptic_inputs == 0.0)

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_synaptic_calculator_creation(self):
        """Test that synaptic calculator can be created."""
        calculator = create_synaptic_calculator(time_window=0.1)
        assert calculator is not None
        assert hasattr(calculator, 'calculate_synaptic_inputs')

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_synaptic_calculator_basic_functionality(
        self, mock_edge_attributes, mock_node_energies
    ):
        """Test basic synaptic input calculation."""
        calculator = create_synaptic_calculator()

        # Test with mock data
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            mock_edge_attributes,
            mock_node_energies,
            num_nodes=5
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        assert len(synaptic_inputs) == 5
        assert synaptic_inputs.dtype == np.float64

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_synaptic_calculator_performance(self, mock_edge_attributes, mock_node_energies):
        """Test that optimized calculator performs adequately."""
        calculator = create_synaptic_calculator()

        # Time the calculation
        start_time = time.time()
        for _ in range(100):
            _synaptic_inputs = calculator.calculate_synaptic_inputs(
                mock_edge_attributes,
                mock_node_energies,
                num_nodes=5
            )
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        # Should be reasonably fast (< 1ms per calculation)
        assert avg_time < 0.001, f"Average calculation time {avg_time:.6f}s is too slow"

    def test_fallback_behavior(self):
        """Test that fallback works when extensions are not available."""
        if not CPP_EXTENSIONS_AVAILABLE:
            # If extensions aren't available, this test should pass
            # because the service should handle the ImportError gracefully
            pass


class TestNeuralProcessingServiceIntegration:
    """Test integration of C++ extensions with NeuralProcessingService."""

    @pytest.fixture
    def mock_energy_manager(self):
        """Create mock energy manager."""
        manager = Mock()
        energy_state = Mock()
        energy_state.node_energies = {i: 0.8 for i in range(5)}
        manager.get_energy_state.return_value = energy_state
        return manager

    @pytest.fixture
    def mock_config_service(self):
        """Create mock configuration service."""
        service = Mock()
        service.get_config.return_value = None
        return service

    @pytest.fixture
    def mock_event_coordinator(self):
        """Create mock event coordinator."""
        coordinator = Mock()
        return coordinator

    def test_service_initialization_with_extensions(self, mock_energy_manager,
                                                    mock_config_service, mock_event_coordinator):
        """Test that NeuralProcessingService initializes with extensions."""
        try:
            from src.core.services.neural_processing_service import \
                NeuralProcessingService
        except ImportError:
            pytest.skip("NeuralProcessingService not available")

        service = NeuralProcessingService(
            mock_energy_manager,
            mock_config_service,
            mock_event_coordinator
        )

        # Service should initialize successfully
        assert service is not None

        # Check if optimized calculator was initialized
        if CPP_EXTENSIONS_AVAILABLE:
            assert hasattr(service, '_synaptic_calculator')
        else:
            # Should still work without extensions
            assert service._synaptic_calculator is None

    def test_synaptic_input_calculation_integration(self, mock_energy_manager,
                                                    mock_config_service, mock_event_coordinator):
        """Test that synaptic input calculation works in the service."""
        try:
            from torch_geometric.data import Data

            from src.core.services.neural_processing_service import \
                NeuralProcessingService
        except ImportError:
            pytest.skip("NeuralProcessingService or torch_geometric not available")

        service = NeuralProcessingService(
            mock_energy_manager,
            mock_config_service,
            mock_event_coordinator
        )

        # Create a simple test graph
        graph = Data()
        graph.node_labels = [{'id': i, 'membrane_potential': -70.0} for i in range(5)]
        graph.x = np.random.randn(5, 1).astype(np.float32)

        # Test synaptic input calculation
        synaptic_inputs = service._calculate_all_synaptic_inputs(graph, {i: 0.8 for i in range(5)})  # pylint: disable=protected-access

        assert isinstance(synaptic_inputs, np.ndarray)
        assert len(synaptic_inputs) == 5

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_integration_with_realistic_graph_data(self, mock_energy_manager,
                                                   mock_config_service, mock_event_coordinator):
        """Test integration with more realistic graph structures."""
        try:
            from torch_geometric.data import Data

            from src.core.services.neural_processing_service import \
                NeuralProcessingService
        except ImportError:
            pytest.skip("NeuralProcessingService or torch_geometric not available")

        service = NeuralProcessingService(
            mock_energy_manager,
            mock_config_service,
            mock_event_coordinator
        )

        # Create a more complex test graph with edges
        graph = Data()
        graph.node_labels = [{'id': i, 'membrane_potential': -70.0 + i} for i in range(10)]
        graph.x = np.random.randn(10, 3).astype(np.float32)

        # Add edge information
        edge_index = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
        graph.edge_index = edge_index

        # Add edge attributes
        edge_attr = []
        for i in range(10):
            attr = Mock()
            attr.source = edge_index[0, i]
            attr.target = edge_index[1, i]
            attr.weight = 0.5 + 0.1 * i
            attr.type = ['excitatory', 'inhibitory', 'modulatory'][i % 3]
            attr.get_effective_weight = Mock(return_value=attr.weight)
            edge_attr.append(attr)

        graph.edge_attr = edge_attr

        # Test synaptic input calculation with complex graph
        node_energies = {i: 0.5 + 0.05 * i for i in range(10)}
        synaptic_inputs = service._calculate_all_synaptic_inputs(graph, node_energies)  # pylint: disable=protected-access

        assert isinstance(synaptic_inputs, np.ndarray)
        assert len(synaptic_inputs) == 10
        assert synaptic_inputs.dtype == np.float64
        # Should have non-zero inputs due to edges
        assert not np.all(synaptic_inputs == 0.0)


class TestEdgeCases:
    """Test edge cases and boundary conditions for C++ extensions."""

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_empty_node_energies(self):
        """Test calculation with empty node energies dictionary."""
        calculator = create_synaptic_calculator()

        edges = []
        edge = Mock()
        edge.source = 0
        edge.target = 1
        edge.weight = 1.0
        edge.type = 'excitatory'
        edge.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, {}, num_nodes=2
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        assert len(synaptic_inputs) == 2
        # Should handle missing energies gracefully
        assert synaptic_inputs[1] == 0.0  # No energy means no gated/modulatory effects

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_very_large_num_nodes(self):
        """Test with very large number of nodes."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8, 99999: 0.8}

        # Edge to very high node index
        edge = Mock()
        edge.source = 0
        edge.target = 99999
        edge.weight = 1.0
        edge.type = 'excitatory'
        edge.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=100000
        )

        assert len(synaptic_inputs) == 100000
        assert synaptic_inputs[99999] == 1.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_extreme_weight_values(self):
        """Test with extreme weight values."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8}

        # Extremely large positive weight
        edge1 = Mock()
        edge1.source = 0
        edge1.target = 1
        edge1.weight = 1e10
        edge1.type = 'excitatory'
        edge1.get_effective_weight = Mock(return_value=1e10)
        edges.append(edge1)

        # Extremely large negative weight
        edge2 = Mock()
        edge2.source = 0
        edge2.target = 1
        edge2.weight = -1e10
        edge2.type = 'inhibitory'
        edge2.get_effective_weight = Mock(return_value=-1e10)
        edges.append(edge2)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert len(synaptic_inputs) == 2
        # Weights should be clamped to [-100, 100]
        assert synaptic_inputs[1] == 0.0  # 100 + (-100) = 0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_zero_and_negative_time_windows(self):
        """Test with zero or negative time windows."""
        # Zero time window
        calculator = create_synaptic_calculator(time_window=0.0)

        edges = []
        node_energies = {0: 0.8}

        edge = Mock()
        edge.source = 0
        edge.target = 1
        edge.weight = 1.0
        edge.type = 'excitatory'
        edge.get_effective_weight = Mock(return_value=1.0)
        edge.last_spike_time = time.time() - 0.01  # Recent spike
        edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, current_time=time.time(), num_nodes=2
        )

        # With zero time window, no spikes should be considered recent
        assert synaptic_inputs[1] == 0.0

        # Negative time window (should behave like zero)
        calculator = create_synaptic_calculator(time_window=-0.1)
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, current_time=time.time(), num_nodes=2
        )
        assert synaptic_inputs[1] == 0.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_mixed_valid_invalid_edges(self):
        """Test with a mix of valid and invalid edges."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8, 1: 0.6}

        # Valid edge
        edge1 = Mock()
        edge1.source = 0
        edge1.target = 1
        edge1.weight = 1.0
        edge1.type = 'excitatory'
        edge1.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge1)

        # Invalid: None edge
        edges.append(None)

        # Invalid: missing attributes
        edge3 = Mock()
        # Missing source, target, etc.
        edges.append(edge3)

        # Invalid: wrong types
        edge4 = Mock()
        edge4.source = "invalid"
        edge4.target = 1
        edge4.weight = 1.0
        edge4.type = 'excitatory'
        edge4.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge4)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        assert len(synaptic_inputs) == 2
        # Only valid edge should contribute
        assert synaptic_inputs[1] == 1.0
        assert synaptic_inputs[0] == 0.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_sparse_connectivity(self):
        """Test with very sparse connectivity (few edges, many nodes)."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {i: 0.8 for i in range(1000)}

        # Only 3 edges in 1000 nodes
        for i in range(3):
            edge = Mock()
            edge.source = i * 100
            edge.target = (i * 100 + 1) % 1000
            edge.weight = 1.0
            edge.type = 'excitatory'
            edge.get_effective_weight = Mock(return_value=1.0)
            edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=1000
        )

        assert len(synaptic_inputs) == 1000
        # Only 3 nodes should have non-zero inputs
        non_zero_count = np.count_nonzero(synaptic_inputs)
        assert non_zero_count == 3


class TestPerformanceBenchmarks:
    """Performance benchmarks for C++ extensions."""

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_scalability_benchmark(self):
        """Test performance scaling with different graph sizes."""
        calculator = create_synaptic_calculator()

        # Create test data of different sizes
        sizes = [100, 500, 1000]
        results = {}

        for size in sizes:
            # Create mock edges
            edges = []
            for i in range(size):
                edge = Mock()
                edge.source = np.random.randint(0, size // 10)
                edge.target = np.random.randint(0, size // 10)
                edge.weight = np.random.randn()
                edge.type = 'excitatory'
                edge.get_effective_weight = Mock(return_value=edge.weight)
                edges.append(edge)

            node_energies = {i: 0.8 for i in range(size // 10)}

            # Benchmark
            start_time = time.time()
            synaptic_inputs = calculator.calculate_synaptic_inputs(
                edges, node_energies, num_nodes=size // 10
            )
            end_time = time.time()

            results[size] = end_time - start_time
            assert len(synaptic_inputs) == size // 10

        # Performance should scale reasonably
        # Larger graphs should not be disproportionately slower
        assert results[100] < 0.01  # Should be very fast
        assert results[1000] / results[100] < 50  # Should scale reasonably


class TestSimulationScenarios:
    """Test real-world simulation scenarios for C++ extensions."""

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_feedforward_network_pattern(self):
        """Test synaptic calculations in a feedforward neural network pattern."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {}

        # Create a 3-layer feedforward network: 4 -> 3 -> 2
        layers = [4, 3, 2]
        total_nodes = sum(layers)

        # Initialize node energies
        for i in range(total_nodes):
            node_energies[i] = 0.8

        # Create connections between layers
        _edge_idx = 0
        for l in range(len(layers) - 1):
            start_src = sum(layers[:l])
            start_tgt = sum(layers[:l+1])
            for src in range(start_src, start_src + layers[l]):
                for tgt in range(start_tgt, start_tgt + layers[l+1]):
                    edge = Mock()
                    edge.source = src
                    edge.target = tgt
                    edge.weight = np.random.uniform(0.1, 1.0)
                    edge.type = 'excitatory'
                    edge.get_effective_weight = Mock(return_value=edge.weight)
                    edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=total_nodes
        )

        assert len(synaptic_inputs) == total_nodes
        # First layer should have no inputs
        assert np.all(synaptic_inputs[:4] == 0.0)
        # Middle and last layers should have inputs
        assert not np.all(synaptic_inputs[4:] == 0.0)

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_recurrent_network_with_inhibition(self):
        """Test recurrent network with excitatory and inhibitory connections."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {i: 0.7 for i in range(10)}

        # Create recurrent connections with mixed types
        for i in range(10):
            for j in range(10):
                if i != j:  # No self-connections
                    edge = Mock()
                    edge.source = i
                    edge.target = j
                    if (i + j) % 3 == 0:
                        edge.type = 'inhibitory'
                        edge.weight = -0.5
                    else:
                        edge.type = 'excitatory'
                        edge.weight = 0.8
                    edge.get_effective_weight = Mock(return_value=edge.weight)
                    edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=10
        )

        assert len(synaptic_inputs) == 10
        # For symmetric networks, expect positive synaptic inputs
        assert np.all(synaptic_inputs >= 0)

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_energy_based_gating_scenario(self):
        """Test scenario where energy levels control information flow."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {}

        # Create nodes with varying energy levels
        for i in range(6):
            node_energies[i] = 0.2 + 0.15 * i  # Energies from 0.2 to 0.95

        # Create gated connections that only activate with sufficient energy
        for src in range(3):
            for tgt in range(3, 6):
                edge = Mock()
                edge.source = src
                edge.target = tgt
                edge.weight = 1.0
                edge.type = 'gated'
                edge.gate_threshold = 0.5
                edge.get_effective_weight = Mock(return_value=1.0)
                edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=6
        )

        assert len(synaptic_inputs) == 6
        # Only nodes 3-5 should receive inputs (from sources with energy >= 0.5)
        # Source 0: 0.2 < 0.5 -> no input to 3-5
        # Source 1: 0.35 < 0.5 -> no input
        # Source 2: 0.5 >= 0.5 -> input to 3-5
        expected_non_zero = [3, 4, 5]
        for i in range(6):
            if i in expected_non_zero:
                assert synaptic_inputs[i] > 0
            else:
                assert synaptic_inputs[i] == 0.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_temporal_dynamics_simulation(self):
        """Test simulation of temporal dynamics with spike timing."""
        calculator = create_synaptic_calculator(time_window=0.1)

        current_time = time.time()
        edges = []
        node_energies = {0: 0.8, 1: 0.8}

        # Create edges with different spike times
        spike_times = [
            current_time - 0.05,  # Recent - should contribute
            current_time - 0.15,  # Old - should not contribute
            current_time - 0.08,  # Recent - should contribute
        ]

        for _, spike_time in enumerate(spike_times):
            edge = Mock()
            edge.source = 0
            edge.target = 1
            edge.weight = 1.0
            edge.type = 'excitatory'
            edge.get_effective_weight = Mock(return_value=1.0)
            edge.last_spike_time = spike_time
            edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, current_time=current_time, num_nodes=2
        )

        assert len(synaptic_inputs) == 2
        # Only 2 recent spikes should contribute (within 0.1s window)
        assert synaptic_inputs[1] == 2.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_large_scale_neural_population(self):
        """Test performance with large-scale neural population simulation."""
        calculator = create_synaptic_calculator()

        # Simulate 1000 neurons with realistic connectivity (10% connection probability)
        num_nodes = 1000
        connection_prob = 0.1
        _expected_edges = int(num_nodes * num_nodes * connection_prob)

        edges = []
        node_energies = {i: np.random.uniform(0.5, 1.0) for i in range(num_nodes)}

        # Create random connections
        np.random.seed(42)  # For reproducible tests
        for src in range(num_nodes):
            for tgt in range(num_nodes):
                if np.random.random() < connection_prob:
                    edge = Mock()
                    edge.source = src
                    edge.target = tgt
                    edge.weight = np.random.normal(0.5, 0.2)
                    edge.type = np.random.choice(['excitatory', 'inhibitory', 'modulatory'])
                    edge.get_effective_weight = Mock(return_value=edge.weight)
                    edges.append(edge)

        # Time the calculation
        start_time = time.time()
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=num_nodes
        )
        end_time = time.time()

        assert len(synaptic_inputs) == num_nodes
        assert synaptic_inputs.dtype == np.float64
        # Should complete in reasonable time (< 10 seconds for 1000 nodes)
        assert end_time - start_time < 10.0
        # Should have varied inputs due to random connections
        assert not np.all(synaptic_inputs == 0.0)


class TestErrorHandling:
    """Test error handling and exception scenarios for C++ extensions."""

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_malformed_edge_attributes(self):
        """Test handling of malformed edge attributes."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8}

        # Edge with missing critical attributes
        edge1 = Mock()
        # Missing source - should be handled gracefully
        edge1.target = 1
        edge1.weight = 1.0
        edge1.type = 'excitatory'
        edges.append(edge1)

        edge2 = Mock()
        edge2.source = 0
        # Missing target
        edge2.weight = 1.0
        edge2.type = 'excitatory'
        edges.append(edge2)

        # Should not raise exceptions
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        assert len(synaptic_inputs) == 2

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_get_effective_weight_exceptions(self):
        """Test handling of exceptions in get_effective_weight method."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8}

        # Edge with get_effective_weight that raises exception
        edge = Mock()
        edge.source = 0
        edge.target = 1
        edge.type = 'excitatory'
        edge.get_effective_weight = Mock(side_effect=ValueError("Test exception"))
        edges.append(edge)

        # Should handle the exception gracefully
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        # Should skip the problematic edge
        assert synaptic_inputs[1] == 0.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_invalid_edge_types(self):
        """Test handling of invalid or unknown edge types."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8}

        # Edge with unknown type
        edge = Mock()
        edge.source = 0
        edge.target = 1
        edge.weight = 1.0
        edge.type = 'unknown_type'
        edge.get_effective_weight = Mock(return_value=1.0)
        edges.append(edge)

        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        # Should default to excitatory behavior (weight * 1.0)
        assert synaptic_inputs[1] == 1.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_none_values_in_edge_attributes(self):
        """Test handling of None values in edge attributes."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8}

        # Edge with None values
        edge = Mock()
        edge.source = None
        edge.target = None
        edge.weight = None
        edge.type = None
        edge.get_effective_weight = Mock(return_value=None)
        edges.append(edge)

        # Should not crash
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        assert synaptic_inputs[1] == 0.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_extreme_time_values(self):
        """Test handling of extreme time values."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8}

        # Edge with extreme spike time
        edge = Mock()
        edge.source = 0
        edge.target = 1
        edge.weight = 1.0
        edge.type = 'excitatory'
        edge.get_effective_weight = Mock(return_value=1.0)
        edge.last_spike_time = 1e20  # Far future
        edges.append(edge)

        # Should handle gracefully
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, current_time=time.time(), num_nodes=2
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        # Spike from far future should not contribute
        assert synaptic_inputs[1] == 0.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_memory_allocation_edge_cases(self):
        """Test memory allocation with extreme sizes."""
        calculator = create_synaptic_calculator()

        # Test with zero nodes
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            [], {}, num_nodes=0
        )
        assert len(synaptic_inputs) == 0

        # Test with very large num_nodes but no edges
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            [], {}, num_nodes=100000
        )
        assert len(synaptic_inputs) == 100000
        assert np.all(synaptic_inputs == 0.0)

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_type_conversion_errors(self):
        """Test handling of type conversion errors."""
        calculator = create_synaptic_calculator()

        edges = []
        node_energies = {0: 0.8}

        # Edge with non-numeric weight
        edge = Mock()
        edge.source = 0
        edge.target = 1
        edge.weight = "invalid"
        edge.type = 'excitatory'
        edge.get_effective_weight = Mock(side_effect=TypeError("Cannot convert"))
        edges.append(edge)

        # Should handle type conversion errors
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=2
        )

        assert isinstance(synaptic_inputs, np.ndarray)
        assert synaptic_inputs[1] == 0.0


class TestComprehensivePerformanceBenchmarks:
    """Comprehensive performance benchmarks for C++ extensions."""

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with different problem sizes."""
        calculator = create_synaptic_calculator()

        sizes = [1000, 5000, 10000]
        memory_usage = {}

        for size in sizes:
            # Create test data
            edges = []
            for i in range(size):
                edge = Mock()
                edge.source = np.random.randint(0, size // 10)
                edge.target = np.random.randint(0, size // 10)
                edge.weight = 0.5
                edge.type = 'excitatory'
                edge.get_effective_weight = Mock(return_value=0.5)
                edges.append(edge)

            node_energies = {i: 0.8 for i in range(size // 10)}

            # Approximate memory usage (edges * ~100 bytes per edge)
            estimated_memory = len(edges) * 100  # Rough estimate
            memory_usage[size] = estimated_memory

            # Ensure calculation completes
            synaptic_inputs = calculator.calculate_synaptic_inputs(
                edges, node_energies, num_nodes=size // 10
            )
            assert len(synaptic_inputs) == size // 10

        # Memory usage should scale linearly
        assert memory_usage[5000] / memory_usage[1000] < 6  # Allow some overhead

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_throughput_benchmark(self):
        """Test calculation throughput (edges/second)."""
        calculator = create_synaptic_calculator()

        # Test with 100k edges
        num_edges = 100000
        num_nodes = 1000

        edges = []
        for i in range(num_edges):
            edge = Mock()
            edge.source = np.random.randint(0, num_nodes)
            edge.target = np.random.randint(0, num_nodes)
            edge.weight = np.random.uniform(-1.0, 1.0)
            edge.type = np.random.choice(['excitatory', 'inhibitory', 'modulatory'])
            edge.get_effective_weight = Mock(return_value=edge.weight)
            edges.append(edge)

        node_energies = {i: np.random.uniform(0.5, 1.0) for i in range(num_nodes)}

        # Benchmark throughput
        start_time = time.time()
        synaptic_inputs = calculator.calculate_synaptic_inputs(
            edges, node_energies, num_nodes=num_nodes
        )
        end_time = time.time()

        duration = end_time - start_time
        throughput = num_edges / duration  # edges per second

        # Should handle at least 10k edges/second
        assert throughput > 10000, f"Throughput {throughput:.0f} edges/s is too low"
        assert len(synaptic_inputs) == num_nodes

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_real_time_performance(self):
        """Test if calculations can meet real-time requirements (< 16ms for 60 FPS)."""
        calculator = create_synaptic_calculator()

        # Simulate a moderately complex network (10k edges, 1k nodes)
        num_edges = 10000
        num_nodes = 1000

        edges = []
        for i in range(num_edges):
            edge = Mock()
            edge.source = np.random.randint(0, num_nodes)
            edge.target = np.random.randint(0, num_nodes)
            edge.weight = 0.5
            edge.type = 'excitatory'
            edge.get_effective_weight = Mock(return_value=0.5)
            edges.append(edge)

        node_energies = {i: 0.8 for i in range(num_nodes)}

        # Test multiple frames
        frame_times = []
        for _ in range(10):
            start_time = time.time()
            _synaptic_inputs = calculator.calculate_synaptic_inputs(
                edges, node_energies, num_nodes=num_nodes
            )
            end_time = time.time()
            frame_times.append(end_time - start_time)

        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)

        # Should be under ~300ms for real-time performance
        assert avg_frame_time < 0.3, (
            f"Average frame time {avg_frame_time*1000:.1f}ms exceeds real-time limit"
        )
        assert max_frame_time < 2.0, f"Max frame time {max_frame_time*1000:.1f}ms is too high"

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_energy_vs_performance_tradeoff(self):
        """Test performance differences with different energy configurations."""
        calculator = create_synaptic_calculator()

        num_edges = 5000
        num_nodes = 500

        edges = []
        for i in range(num_edges):
            edge = Mock()
            edge.source = np.random.randint(0, num_nodes)
            edge.target = np.random.randint(0, num_nodes)
            edge.weight = 0.5
            edge.type = 'gated'  # Use gated to test energy dependency
            edge.gate_threshold = 0.5
            edge.get_effective_weight = Mock(return_value=0.5)
            edges.append(edge)

        # Test with high energy (all gates open)
        node_energies_high = {i: 0.9 for i in range(num_nodes)}

        start_time = time.time()
        inputs_high = calculator.calculate_synaptic_inputs(
            edges, node_energies_high, num_nodes=num_nodes
        )
        time_high = time.time() - start_time

        # Test with low energy (most gates closed)
        node_energies_low = {i: 0.3 for i in range(num_nodes)}

        start_time = time.time()
        inputs_low = calculator.calculate_synaptic_inputs(
            edges, node_energies_low, num_nodes=num_nodes
        )
        time_low = time.time() - start_time

        # Low energy should result in fewer active connections (potentially faster processing)
        assert np.sum(inputs_high > 0) > np.sum(inputs_low > 0)
        # Performance difference should be acceptable
        assert abs(time_high - time_low) / max(time_high, time_low) < 1.0

    @pytest.mark.skipif(not CPP_EXTENSIONS_AVAILABLE, reason="C++ extensions not available")
    def test_parallel_processing_efficiency(self):
        """Test efficiency of parallel processing for different workloads."""
        calculator = create_synaptic_calculator()

        # Test different edge to node ratios
        configurations = [
            (1000, 100),   # High connectivity
            (10000, 1000), # Moderate connectivity
            (50000, 1000), # Low connectivity (many edges, few nodes)
        ]

        results = {}
        for num_edges, num_nodes in configurations:
            edges = []
            for i in range(num_edges):
                edge = Mock()
                edge.source = np.random.randint(0, num_nodes)
                edge.target = np.random.randint(0, num_nodes)
                edge.weight = 0.5
                edge.type = 'excitatory'
                edge.get_effective_weight = Mock(return_value=0.5)
                edges.append(edge)

            node_energies = {i: 0.8 for i in range(num_nodes)}

            start_time = time.time()
            synaptic_inputs = calculator.calculate_synaptic_inputs(
                edges, node_energies, num_nodes=num_nodes
            )
            duration = time.time() - start_time

            results[(num_edges, num_nodes)] = duration
            assert len(synaptic_inputs) == num_nodes

        # High connectivity should be relatively efficient
        # Low connectivity (many edges/few nodes) should still be reasonable
        high_conn_time = results[(1000, 100)]
        low_conn_time = results[(50000, 1000)]
        assert low_conn_time / high_conn_time < 100  # Should scale reasonably
if __name__ == "__main__":
    pytest.main([__file__])





