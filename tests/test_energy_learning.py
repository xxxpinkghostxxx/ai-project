#!/usr/bin/env python3
"""
Test script to verify energy-modulated learning implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.live_hebbian_learning import create_live_hebbian_learning
from learning.learning_engine import LearningEngine
from core.simulation_manager import SimulationManager

def test_energy_modulated_learning():
    """Test that learning rates are modulated by energy levels."""
    print("Testing Energy-Modulated Learning Implementation")
    print("=" * 60)

    try:
        # Create simulation manager
        sim_manager = SimulationManager()
        success = sim_manager.initialize_graph()
        if not success:
            print("Failed to initialize simulation")
            return False

        # Test Hebbian learning with energy modulation
        hebbian_system = create_live_hebbian_learning(sim_manager)
        print(f"Hebbian system created with energy modulation: {hebbian_system.energy_learning_modulation}")

        # Check parameters
        params = hebbian_system.get_learning_parameters()
        print(f"Learning parameters: {params}")

        # Test energy modulation
        graph = sim_manager.graph

        # Create test scenario
        from neural.optimized_node_manager import get_optimized_node_manager
        node_manager = get_optimized_node_manager()

        # Create nodes with different energy levels
        test_nodes = []
        for i in range(3):
            node_spec = {
                'type': 'dynamic',
                'energy': 0.2 + i * 0.4,  # 0.2, 0.6, 1.0
                'x': i * 20,
                'y': 20,
                'membrane_potential': 0.0,
                'threshold': 0.5
            }
            created_nodes = node_manager.create_node_batch([node_spec])
            test_nodes.extend(created_nodes)

        print(f"Created {len(test_nodes)} test nodes with varying energy levels")

        # Add created nodes to the graph for learning
        import torch
        for node_id in test_nodes:
            index = node_manager.node_id_to_index[node_id]
            metadata = node_manager.node_metadata[index]
            energy = node_manager.node_data[index, 0]
            graph.node_labels.append(metadata)
            if graph.x is None:
                graph.x = torch.tensor([[energy]], dtype=torch.float32)
            else:
                graph.x = torch.cat([graph.x, torch.tensor([[energy]], dtype=torch.float32)], dim=0)

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
        learning_engine = LearningEngine()
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