#!/usr/bin/env python3
"""
Test Energy-Learning Integration
Tests the enhanced energy-modulated learning system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data
from learning.live_hebbian_learning import LiveHebbianLearning
from neural.connection_logic import create_weighted_connection
from energy.energy_system_validator import EnergySystemValidator
from energy.node_access_layer import NodeAccessLayer
from energy.node_id_manager import get_id_manager

def create_test_graph():
    """Create a simple test graph with known energy values."""
    # Reset ID manager to ensure clean state
    from energy.node_id_manager import force_reset_id_manager
    force_reset_id_manager()
    id_manager = get_id_manager()

    # Create node labels with different energy levels
    node_labels = []
    energies = []

    for i in range(6):
        energy = 0.2 + i * 0.15  # Energies: 0.2, 0.35, 0.5, 0.65, 0.8, 0.95
        # Generate unique ID for this node
        node_id = id_manager.generate_unique_id('dynamic', {'energy': energy})

        node_labels.append({
            'id': node_id,  # Use the generated ID
            'type': 'dynamic',
            'energy': energy,
            'x': i * 10,
            'y': 0,
            'membrane_potential': 0.0,
            'threshold': 0.5,
            'behavior': 'dynamic',
            'state': 'active'
        })
        energies.append(energy)

    # Create graph
    graph = Data()
    graph.node_labels = node_labels
    graph.x = torch.tensor(energies, dtype=torch.float32).unsqueeze(1)

    # Initialize edge structures
    graph.edge_index = torch.empty(2, 0, dtype=torch.long)  # Empty edge index
    graph.edge_attr = torch.empty(0, 1, dtype=torch.float32)  # Empty edge attributes

    # Register nodes with ID manager (they're already generated above, just register indices)
    for i, node in enumerate(node_labels):
        id_manager.register_node_index(node['id'], i)

    print(f"Created test graph with {len(node_labels)} nodes")
    print(f"Graph.x shape: {graph.x.shape}")
    print(f"Active IDs in manager: {id_manager.get_all_active_ids()}")

    return graph

def test_energy_modulated_learning():
    """Test energy-modulated learning functionality."""
    print("Testing Energy-Modulated Learning...")
    print("=" * 50)

    # Create test graph
    graph = create_test_graph()
    access_layer = NodeAccessLayer(graph)

    print("Test nodes created:")
    for i in range(6):
        node_id = graph.node_labels[i]['id']  # Get the actual node ID from the graph
        energy = access_layer.get_node_energy(node_id)
        if energy is not None:
            print(f"  Node {node_id}: energy = {energy:.3f}")
        else:
            print(f"  Node {node_id}: energy = None (not found)")

    # Create some initial connections
    print("\nCreating initial connections...")
    # Use the actual node IDs from the graph
    node_ids = [node['id'] for node in graph.node_labels]
    if len(node_ids) >= 2:
        create_weighted_connection(graph, node_ids[0], node_ids[1], 0.5, 'excitatory')  # Low energy -> medium energy
    if len(node_ids) >= 6:
        create_weighted_connection(graph, node_ids[4], node_ids[5], 0.8, 'excitatory')  # High energy -> very high energy

    print(f"Initial connections: {graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0}")

    # Test learning
    print("\nTesting learning...")
    learning_system = LiveHebbianLearning()
    initial_stats = learning_system.get_learning_statistics()

    print(f"Initial learning stats: {initial_stats}")

    # Apply learning multiple times
    for step in range(5):
        graph = learning_system.apply_continuous_learning(graph, step)
        if step % 2 == 0:
            stats = learning_system.get_learning_statistics()
            print(f"Step {step}: stdp_events={stats['stdp_events']}, energy_modulated={stats['energy_modulated_events']}")

    final_stats = learning_system.get_learning_statistics()
    print(f"\nFinal learning stats: {final_stats}")

    # Check for learning effects
    learning_effects = []
    if final_stats['stdp_events'] > initial_stats['stdp_events']:
        learning_effects.append('stdp_events')
    if final_stats['energy_modulated_events'] > 0:
        learning_effects.append('energy_modulated_learning')
    if final_stats['total_weight_changes'] > initial_stats['total_weight_changes']:
        learning_effects.append('weight_changes')

    print(f"\nLearning effects detected: {learning_effects}")

    if learning_effects:
        print("SUCCESS: Energy-modulated learning is working!")
        return True
    else:
        print("FAILURE: No learning effects detected")
        return False

def test_connection_formation():
    """Test energy-modulated connection formation."""
    print("\nTesting Energy-Modulated Connection Formation...")
    print("=" * 50)

    from neural.connection_logic import intelligent_connection_formation

    # Create test graph
    graph = create_test_graph()

    print("Before connection formation:")
    print(f"  Nodes: {len(graph.node_labels)}")
    print(f"  Connections: {graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0}")

    # Apply connection formation
    graph = intelligent_connection_formation(graph)

    print("After connection formation:")
    print(f"  Connections: {graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0}")

    if hasattr(graph, 'edge_index') and graph.edge_index is not None and graph.edge_index.shape[1] > 0:
        print("SUCCESS: Connection formation working!")
        return True
    else:
        print("FAILURE: No connections formed")
        return False

def main():
    """Run all tests."""
    print("ENERGY-LEARNING INTEGRATION TEST")
    print("=" * 60)

    success_count = 0
    total_tests = 2

    # Test 1: Energy-modulated learning
    if test_energy_modulated_learning():
        success_count += 1

    # Test 2: Connection formation
    if test_connection_formation():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("ALL TESTS PASSED! Energy-learning integration is working correctly.")
    else:
        print("Some tests failed. Energy-learning integration needs further work.")

    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)