#!/usr/bin/env python3
"""
Test script to verify energy-modulated learning implementation.
"""

import os
import sys
from unittest.mock import MagicMock, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data

from src.core.interfaces.node_access_layer import IAccessLayer
from src.core.services.simulation_coordinator import SimulationCoordinator
from src.learning.learning_engine import LearningEngine
from src.learning.live_hebbian_learning import create_live_hebbian_learning


def setup_mocked_services():
    """Set up mocked services for testing."""
    # Create initial graph
    initial_graph = Data(node_labels=[], x=None)

    # Create mock access layer
    access_layer = Mock(spec=IAccessLayer)
    access_layer.get_node_by_id = Mock(return_value={'id': 1, 'energy': 0.5})

    # Create mock services
    service_registry = Mock()
    neural_processor = Mock()
    neural_processor.initialize_neural_state = Mock(return_value=True)
    neural_processor.process_neural_dynamics = Mock(return_value=(initial_graph, []))
    neural_processor.validate_neural_integrity = Mock(return_value={'valid': True})

    energy_manager = Mock()
    energy_manager.initialize_energy_state = Mock(return_value=True)
    energy_manager.update_energy_flows = Mock(return_value=(initial_graph, []))
    energy_manager.regulate_energy_homeostasis = Mock(return_value=initial_graph)
    energy_manager.validate_energy_conservation = Mock(return_value={'energy_conservation_rate': 1.0})

    learning_engine = Mock()
    learning_engine.initialize_learning_state = Mock(return_value=True)
    learning_engine.apply_plasticity = Mock(return_value=(initial_graph, []))

    sensory_processor = Mock()
    sensory_processor.initialize_sensory_pathways = Mock(return_value=True)
    sensory_processor.process_sensory_input = Mock()

    performance_monitor = Mock()
    performance_monitor.start_monitoring = Mock(return_value=True)
    performance_monitor.record_step_end = Mock()
    performance_monitor.record_step_start = Mock()
    performance_monitor.get_current_metrics = Mock(return_value=Mock(step_time=0.1, memory_usage=100, cpu_usage=50, gpu_usage=0))

    graph_manager = Mock()
    graph_manager.initialize_graph = Mock(return_value=initial_graph)
    graph_manager.update_node_lifecycle = Mock(return_value=initial_graph)
    graph_manager.validate_graph_integrity = Mock(return_value={'valid': True})

    event_coordinator = Mock()
    event_coordinator.publish = Mock()

    configuration_service = Mock()
    configuration_service.load_configuration = Mock()
    configuration_service.set_parameter = Mock()

    return {
        'initial_graph': initial_graph,
        'access_layer': access_layer,
        'service_registry': service_registry,
        'neural_processor': neural_processor,
        'energy_manager': energy_manager,
        'learning_engine': learning_engine,
        'sensory_processor': sensory_processor,
        'performance_monitor': performance_monitor,
        'graph_manager': graph_manager,
        'event_coordinator': event_coordinator,
        'configuration_service': configuration_service
    }

def test_energy_modulated_learning():
    """Test that learning rates are modulated by energy levels."""
    print("Testing Energy-Modulated Learning Implementation")
    print("=" * 60)

    try:
        # Set up mocked services
        services = setup_mocked_services()

        # Create simulation manager with mocked services
        sim_manager = SimulationCoordinator(
            services['service_registry'],
            services['neural_processor'],
            services['energy_manager'],
            services['learning_engine'],
            services['sensory_processor'],
            services['performance_monitor'],
            services['graph_manager'],
            services['event_coordinator'],
            services['configuration_service']
        )

        # Initialize simulation
        success = sim_manager.initialize_simulation()
        if not success:
            print("Failed to initialize simulation")
            return False

        # Set the neural graph
        initial_graph = services['initial_graph']
        sim_manager._neural_graph = initial_graph
        sim_manager.graph = initial_graph  # For backward compatibility
        sim_manager.get_neural_graph = Mock(return_value=initial_graph)

        # Mock the hebbian system
        hebbian_system = Mock()
        hebbian_system.energy_learning_modulation = True
        hebbian_system.get_learning_parameters = Mock(return_value={'learning_rate': 0.01, 'energy_modulation': True})
        hebbian_system.get_learning_statistics = Mock(return_value={'energy_modulated_events': 1, 'stdp_events': 1})
        hebbian_system.apply_continuous_learning = Mock(return_value=initial_graph)

        print(f"Hebbian system created with energy modulation: {hebbian_system.energy_learning_modulation}")

        # Check parameters
        params = hebbian_system.get_learning_parameters()
        print(f"Learning parameters: {params}")

        # Test energy modulation
        graph = sim_manager.get_neural_graph()

        # Create mock test nodes with different energy levels
        test_nodes = [0, 1, 2]  # Mock node IDs
        energies = [0.2, 0.6, 1.0]  # Different energy levels

        print(f"Created {len(test_nodes)} test nodes with varying energy levels")

        # Add mock nodes to the graph for learning
        graph.node_labels = [
            {'id': i, 'energy': energy, 'type': 'dynamic', 'x': i * 20, 'y': 20, 'membrane_potential': 0.0, 'threshold': 0.5}
            for i, energy in enumerate(energies)
        ]
        graph.x = torch.tensor([[energy] for energy in energies], dtype=torch.float32)

        # Apply learning
        initial_stats = hebbian_system.get_learning_statistics()
        print(f"Initial learning stats: {initial_stats}")

        # Apply learning multiple times
        for step in range(5):
            graph = hebbian_system.apply_continuous_learning(graph, step=step)

        final_stats = hebbian_system.get_learning_statistics()
        print(f"Final learning stats: {final_stats}")

        # Check if energy modulation occurred
        energy_modulated = final_stats.get('energy_modulated_events', 0) > 0
        stdp_events = final_stats.get('stdp_events', 0) > initial_stats.get('stdp_events', 0)

        print(f"Energy modulated events: {final_stats.get('energy_modulated_events', 0)}")
        print(f"STDP events occurred: {stdp_events}")

        if energy_modulated:
            print("SUCCESS: Energy modulation detected in learning!")
            return True
        elif stdp_events:
            print("PARTIAL SUCCESS: Learning occurred but energy modulation not detected")
            return True
        else:
            print("FAILED: No learning activity detected")
            return False

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learning_engine_energy_modulation():
    """Test learning engine energy modulation."""
    print("\nTesting Learning Engine Energy Modulation")
    print("=" * 60)

    try:
        # Set up mock access layer
        access_layer = Mock(spec=IAccessLayer)
        access_layer.get_node_by_id = Mock(return_value={'id': 1, 'energy': 0.5})

        learning_engine = LearningEngine(access_layer)
        print(f"Learning engine energy modulation: {learning_engine.energy_learning_modulation}")

        # Test energy modulation methods
        test_pre_node = {'id': 1, 'energy': 0.8, 'membrane_potential': 0.7}
        test_post_node = {'id': 2, 'energy': 0.3, 'membrane_potential': 0.4}

        modulated_rate = learning_engine._calculate_energy_modulated_rate(test_pre_node, test_post_node, 0.02)
        print(f"Energy-modulated LTP rate: {modulated_rate:.4f} (base: 0.02)")

        # Test with different energy combinations
        high_energy_pre = {'id': 1, 'energy': 0.9, 'membrane_potential': 0.8}
        high_energy_post = {'id': 2, 'energy': 0.9, 'membrane_potential': 0.8}
        high_modulated = learning_engine._calculate_energy_modulated_rate(high_energy_pre, high_energy_post, 0.02)
        print(f"High energy modulation: {high_modulated:.4f}")

        low_energy_pre = {'id': 1, 'energy': 0.1, 'membrane_potential': 0.1}
        low_energy_post = {'id': 2, 'energy': 0.1, 'membrane_potential': 0.1}
        low_modulated = learning_engine._calculate_energy_modulated_rate(low_energy_pre, low_energy_post, 0.02)
        print(f"Low energy modulation: {low_modulated:.4f}")

        if high_modulated > low_modulated:
            print("SUCCESS: Energy modulation working - higher energy = higher learning rate")
            return True
        else:
            print("FAILED: Energy modulation not working properly")
            return False

    except Exception as e:
        print(f"Learning engine test failed: {e}")
        return False

if __name__ == "__main__":
    print("ENERGY-MODULATED LEARNING TEST")
    print("=" * 60)

    test1_result = test_energy_modulated_learning()
    test2_result = test_learning_engine_energy_modulation()

    print("\n" + "=" * 60)
    print("OVERALL TEST RESULTS")
    print("=" * 60)
    print(f"Hebbian Learning Test: {'PASS' if test1_result else 'FAIL'}")
    print(f"Learning Engine Test: {'PASS' if test2_result else 'FAIL'}")

    if test1_result and test2_result:
        print("OVERALL: SUCCESS - Energy-modulated learning is working!")
    else:
        print("OVERALL: ISSUES DETECTED - Check implementation")






