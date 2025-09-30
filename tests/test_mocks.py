"""
Test mocks and mock classes for the unified testing system.
Contains mock implementations for testing purposes.
"""

import time
from typing import Any, Dict
from unittest.mock import Mock

# Third-party imports
import torch
from torch_geometric.data import Data


class MockSimulationCoordinator:
    """Mock simulation coordinator for testing."""

    def __init__(self):
        self.graph = None
        self.is_running = False
        self.step_count = 0
        self.performance_stats = {
            'fps': 60.0,
            'memory_usage_mb': 100.0,
            'cpu_percent': 50.0
        }

    def initialize_graph(self):
        """Initialize a test graph."""
        self.graph = self._create_test_graph()

    def _create_test_graph(self) -> Data:
        """Create a test graph for testing."""
        num_nodes = 10
        num_edges = 15

        # Create node features
        x = torch.randn(num_nodes, 5)

        # Create edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Create node labels
        node_labels = []
        for i in range(num_nodes):
            node_labels.append({
                'id': i,
                'type': 'dynamic' if i < 5 else 'sensory',
                'energy': float(torch.rand(1).item() * 255),
                'state': 'active'
            })

        return Data(x=x, edge_index=edge_index, node_labels=node_labels)

    def run_single_step(self) -> bool:
        """Simulate a single simulation step."""
        if not self.is_running:
            return False

        self.step_count += 1
        time.sleep(0.01)  # Simulate processing time
        return True

    def start_simulation(self):
        """Start simulation."""
        self.is_running = True

    def stop_simulation(self):
        """Stop simulation."""
        self.is_running = False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


class MockAccessLayer:
    """Mock access layer for testing spike propagation functionality."""

    def __init__(self):
        self.nodes = {1: {'spike_count': 0}, 2: {'threshold': 0.5, 'synaptic_input': 0.0}}

    def get_node_by_id(self, nid):
        """Get node data by node ID."""
        return self.nodes.get(nid, {})

    def update_node_property(self, nid, prop, val):
        """Update a specific property of a node."""
        if nid not in self.nodes:
            self.nodes[nid] = {}
        self.nodes[nid][prop] = val


class MockMemory:
    """Mock memory system for testing memory-influenced node birth parameters."""

    def get_memory_statistics(self):
        """Get memory statistics for testing."""
        return {'traces_formed': 15}


class MockEdge:
    """Mock edge for testing learning consolidation functionality."""

    def __init__(self, trace=0.6, weight=1.0):
        self.eligibility_trace = trace
        self.weight = weight
        self.source = 1
        self.target = 2
        self.type = 'excitatory'


class MockGraph:
    """Mock graph for testing learning consolidation functionality."""

    def __init__(self):
        self.edge_attributes = []


def create_mock_services():
    """Create mocked services for SimulationCoordinator testing."""
    service_registry = Mock()
    neural_processor = Mock()
    energy_manager = Mock()
    learning_engine = Mock()
    sensory_processor = Mock()
    performance_monitor = Mock()
    graph_manager = Mock()
    event_coordinator = Mock()
    configuration_service = Mock()

    return (service_registry, neural_processor, energy_manager,
            learning_engine, sensory_processor, performance_monitor,
            graph_manager, event_coordinator, configuration_service)


def configure_mock_services_for_init(mocks):
    """Configure mocks for initialization."""
    (service_registry, neural_processor, energy_manager,
     learning_engine, sensory_processor, performance_monitor,
     graph_manager, event_coordinator, configuration_service) = mocks

    # Configure mocks for initialization
    graph_manager.initialize_graph.return_value = Data()
    neural_processor.initialize_neural_state.return_value = True
    energy_manager.initialize_energy_state.return_value = True
    learning_engine.initialize_learning_state.return_value = True
    sensory_processor.initialize_sensory_pathways.return_value = True
    performance_monitor.start_monitoring.return_value = True

    return mocks


def configure_mock_services_for_execution(mocks, graph):
    """Configure mocks for step execution."""
    (service_registry, neural_processor, energy_manager,
     learning_engine, sensory_processor, performance_monitor,
     graph_manager, event_coordinator, configuration_service) = mocks

    # Configure for step execution
    neural_processor.process_neural_dynamics.return_value = (graph, [])
    energy_manager.update_energy_flows.return_value = (graph, [])
    learning_engine.apply_plasticity.return_value = (graph, [])
    graph_manager.update_node_lifecycle.return_value = graph
    energy_manager.regulate_energy_homeostasis.return_value = graph

    return mocks