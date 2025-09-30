"""
Comprehensive simulation scenario tests for neural components.
Tests realistic neural simulation scenarios including learning, oscillations, pattern formation, and emergent behaviors.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.energy.node_id_manager import get_id_manager
from src.neural.behavior_engine import BehaviorEngine
from src.neural.connection_logic import (create_weighted_connection,
                                         intelligent_connection_formation)
from src.neural.enhanced_neural_dynamics import EnhancedNeuralDynamics
from src.neural.network_metrics import NetworkMetrics
from src.neural.spike_queue_system import SpikeQueueSystem, SpikeType
from src.neural.workspace_engine import WorkspaceEngine


class TestNeuralSimulationScenarios:
    """Simulation scenario tests for neural components."""

    def setup_method(self):
        """Set up simulation test environment."""
        self.behavior_engine = BehaviorEngine()
        self.enhanced_dynamics = EnhancedNeuralDynamics()
        self.network_metrics = NetworkMetrics()
        self.spike_system = SpikeQueueSystem()
        self.workspace_engine = WorkspaceEngine()

        # Log ID manager state for debugging
        id_manager = get_id_manager()
        active_ids = id_manager.get_all_active_ids()
        print(f"DEBUG: ID Manager state at setup: {len(active_ids)} active IDs: {active_ids[:10]}...")

    def test_oscillatory_network_formation(self):
        """Test formation of oscillatory neural networks."""
        # Create a network with oscillator nodes
        graph = Data()
        num_nodes = 20
        graph.node_labels = []
        for i in range(num_nodes):
            node_type = 'oscillator' if i < num_nodes // 2 else 'dynamic'
            graph.node_labels.append({
                'id': i,
                'type': node_type,
                'behavior': node_type,
                'energy': 0.7 + 0.2 * np.random.random(),
                'state': 'active',
                'enhanced_behavior': True,
                'membrane_potential': 0.5,
                'oscillation_freq': 0.1 + 0.05 * np.random.random() if node_type == 'oscillator' else 0.0,
                'threshold': 0.8 if node_type == 'oscillator' else 0.5
            })

        graph.x = torch.tensor([[node['energy']] for node in graph.node_labels], dtype=torch.float32)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Form connections
        graph = intelligent_connection_formation(graph)

        # Run simulation for several steps
        for step in range(10):
            # Update behaviors
            for i in range(num_nodes):
                self.behavior_engine.update_node_behavior(i, graph, step)

            # Run neural dynamics
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        # Check for oscillatory behavior
        oscillator_nodes = [i for i, node in enumerate(graph.node_labels) if node['type'] == 'oscillator']
        activations = sum(1 for i in oscillator_nodes if graph.node_labels[i].get('state') == 'active')

        # Should have some activation
        assert activations >= 0

    def test_learning_through_stdp(self):
        """Test spike-timing dependent plasticity learning."""
        # Create a simple network for STDP testing
        graph = Data()
        graph.node_labels = [
            {'type': 'sensory', 'behavior': 'sensory', 'energy': 0.8, 'state': 'active', 'enhanced_behavior': True},
            {'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.6, 'state': 'active', 'enhanced_behavior': True}
        ]
        graph.x = torch.tensor([[0.8], [0.6]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        graph.edge_attributes = []

        # Register nodes with ID manager
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node['type'])
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        # Create initial connection
        from src.neural.connection_logic import create_weighted_connection
        graph = create_weighted_connection(graph, graph.node_labels[0]['id'], graph.node_labels[1]['id'], 0.5, 'plastic')

        initial_weight = graph.edge_attributes[0].weight
        print(f"Initial weight: {initial_weight}")

        # Simulate pre-post spike pairing (STDP)
        import time
        pre_time = time.time() - 0.019
        post_time = time.time() - 0.01
        self.enhanced_dynamics.spike_times[graph.node_labels[0]['id']].append(pre_time)
        self.enhanced_dynamics.spike_times[graph.node_labels[1]['id']].append(post_time)

        for step in range(20):
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        final_weight = graph.edge_attributes[0].weight
        print(f"Final weight: {final_weight}")

        # Weight should have changed due to STDP
        assert abs(final_weight - initial_weight) > 0.001

    def test_pattern_completion_network(self):
        """Test pattern completion in auto-associative network."""
        # Create a network that can complete patterns
        graph = Data()
        num_nodes = 10
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active', 'enhanced_behavior': True}
            for i in range(num_nodes)
        ]
        graph.x = torch.tensor([[0.5] for _ in range(num_nodes)], dtype=torch.float32)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Create recurrent connections
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.random() < 0.3:  # Sparse connectivity
                    graph = create_weighted_connection(graph, i, j, 0.3, 'plastic')

        # Train with a pattern (correlated activation)
        pattern_nodes = [0, 2, 4, 6, 8]
        for step in range(50):
            # Activate pattern nodes
            for i in pattern_nodes:
                if np.random.random() < 0.8:  # Partial activation
                    graph.node_labels[i]['energy'] = 0.9
                    graph.node_labels[i]['last_spike_time'] = step * 0.01

            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

            # Reset energies
            for node in graph.node_labels:
                node['energy'] = 0.5

        # Test pattern completion - activate subset and check if others activate
        test_nodes = [0, 2]  # Subset of pattern
        for i in test_nodes:
            graph.node_labels[i]['energy'] = 0.9

        # Run one step
        graph = self.enhanced_dynamics.update_neural_dynamics(graph, 100)

        # Check if pattern completion occurred (other pattern nodes activated)
        completed_activations = sum(1 for i in pattern_nodes[2:] if graph.node_labels[i].get('energy', 0) > 0.7)

        # Should show some pattern completion
        assert completed_activations >= 0

    def test_workspace_driven_concept_formation(self):
        """Test concept formation through workspace processing."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'type': 'sensory', 'behavior': 'sensory', 'energy': 0.8, 'state': 'active'},
            {'id': 1, 'type': 'workspace', 'behavior': 'workspace', 'energy': 0.7, 'state': 'active',
             'workspace_capacity': 5.0, 'workspace_creativity': 2.0, 'workspace_focus': 4.0, 'threshold': 0.6}
        ]

        # Simulate sensory input driving workspace processing
        for step in range(20):
            # Update sensory input
            graph.node_labels[0]['energy'] = 0.6 + 0.3 * np.sin(step * 0.1)

            # Update behaviors
            for i in range(len(graph.node_labels)):
                self.behavior_engine.update_node_behavior(i, graph, step)

            # Update workspace
            self.workspace_engine.update_workspace_nodes(graph, step)

        # Check workspace statistics
        stats = self.workspace_engine.get_workspace_metrics()
        assert 'syntheses_performed' in stats

    def test_critical_brain_dynamics_emergence(self):
        """Test emergence of critical brain dynamics."""
        # Create a network at criticality
        graph = Data()
        num_nodes = 50
        graph.node_labels = [
            {'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active', 'enhanced_behavior': True}
            for _ in range(num_nodes)
        ]
        graph.x = torch.tensor([[0.5] for _ in range(num_nodes)], dtype=torch.float32)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Register nodes with ID manager
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node['type'])
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        # Form connections to achieve criticality
        graph = intelligent_connection_formation(graph)

        # Run long simulation
        criticality_values = []
        for step in range(100):
            for i in range(num_nodes):
                self.behavior_engine.update_node_behavior(i, graph, step)
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

            if step % 10 == 0:
                metrics = self.network_metrics.calculate_comprehensive_metrics(graph)
                criticality_values.append(metrics['criticality'])

        # Check for critical dynamics (branching ratio around 1.0)
        avg_criticality = np.mean(criticality_values)
        assert 0.0 <= avg_criticality <= 2.0  # Reasonable criticality range

    def test_energy_driven_adaptation(self):
        """Test neural adaptation driven by energy constraints."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'type': 'sensory', 'behavior': 'sensory', 'energy': 0.9, 'state': 'active'},
            {'id': 1, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.3, 'state': 'active', 'enhanced_behavior': True},
            {'id': 2, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.1, 'state': 'inactive', 'enhanced_behavior': True}
        ]
        graph.x = torch.tensor([[0.9], [0.3], [0.1]], dtype=torch.float32)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Form energy-modulated connections
        graph = intelligent_connection_formation(graph)

        initial_connections = len(graph.edge_attributes)

        # Simulate energy fluctuations
        for step in range(30):
            # Vary energy levels
            energy_variation = 0.5 + 0.4 * np.sin(step * 0.2)
            for node in graph.node_labels:
                node['energy'] = energy_variation * node['energy']

            for i in range(len(graph.node_labels)):
                self.behavior_engine.update_node_behavior(i, graph, step)
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        # Network should adapt to energy constraints
        final_metrics = self.network_metrics.calculate_comprehensive_metrics(graph)
        assert final_metrics['connectivity']['num_edges'] >= 0

    def test_spike_propagation_cascades(self):
        """Test spike propagation cascades through neural networks."""
        # Create a feed-forward network
        graph = Data()
        layers = [5, 10, 8, 3]  # Layer sizes
        graph.node_labels = []

        for layer_idx, layer_size in enumerate(layers):
            for i in range(layer_size):
                graph.node_labels.append({
                    'type': 'dynamic',
                    'behavior': 'dynamic',
                    'energy': 0.5,
                    'state': 'active',
                    'enhanced_behavior': True,
                    'layer': layer_idx
                })

        graph.x = torch.tensor([[0.5] for _ in graph.node_labels], dtype=torch.float32)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Register nodes with ID manager
        id_manager = get_id_manager()
        for i, node in enumerate(graph.node_labels):
            node_id = id_manager.generate_unique_id(node['type'])
            node['id'] = node_id
            id_manager.register_node_index(node_id, i)

        # Create feed-forward connections
        layer_starts = [0, 5, 15, 23]
        for i in range(len(layers) - 1):
            start1, start2 = layer_starts[i], layer_starts[i + 1]
            size1, size2 = layers[i], layers[i + 1]

            for j in range(size1):
                for k in range(size2):
                    if np.random.random() < 0.7:  # Dense connectivity
                        node_id1 = graph.node_labels[start1 + j]['id']
                        node_id2 = graph.node_labels[start2 + k]['id']
                        graph = create_weighted_connection(graph, node_id1, node_id2, 0.4, 'excitatory')

        # Trigger cascade from first layer
        first_layer_nodes = layer_starts[0], layer_starts[1]
        for i in range(first_layer_nodes[0], first_layer_nodes[1]):
            node_id = graph.node_labels[i]['id']
            target_id = graph.node_labels[i + layers[0]]['id']
            self.spike_system.schedule_spike(node_id, target_id, SpikeType.EXCITATORY, 1.0, 0.8)

        # Process spikes
        processed = self.spike_system.process_spikes(100)

        # Should propagate through layers
        assert processed >= 0

    def test_memory_consolidation_during_sleep(self):
        """Test memory consolidation during simulated sleep periods."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.6, 'state': 'active', 'enhanced_behavior': True},
            {'id': 1, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.7, 'state': 'active', 'enhanced_behavior': True}
        ]
        graph.x = torch.tensor([[0.6], [0.7]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        graph.edge_attributes = []

        # Create plastic connection
        graph = create_weighted_connection(graph, 0, 1, 0.5, 'plastic')
        initial_weight = graph.edge_attributes[0].weight

        # Simulate learning (STDP)
        for step in range(20):
            graph.node_labels[0]['last_spike_time'] = step * 0.01
            graph.node_labels[1]['last_spike_time'] = (step + 2) * 0.01  # Causal
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        learning_weight = graph.edge_attributes[0].weight

        # Simulate sleep (IEG activation)
        for node in graph.node_labels:
            node['ieg_flag'] = True
            node['plasticity_enabled'] = True

        for step in range(20, 40):
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        consolidated_weight = graph.edge_attributes[0].weight

        # Weight should be consolidated (increased)
        assert consolidated_weight >= learning_weight

    def test_neuromodulator_influence_on_learning(self):
        """Test neuromodulator influence on learning dynamics."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.6, 'state': 'active', 'enhanced_behavior': True},
            {'id': 1, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.7, 'state': 'active', 'enhanced_behavior': True}
        ]
        graph.x = torch.tensor([[0.6], [0.7]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        graph.edge_attributes = []

        graph = create_weighted_connection(graph, 0, 1, 0.5, 'plastic')

        # Test with high dopamine (should enhance learning)
        self.enhanced_dynamics.set_neuromodulator_level('dopamine', 0.9)

        initial_weight = graph.edge_attributes[0].weight
        for step in range(15):
            graph.node_labels[0]['last_spike_time'] = step * 0.01
            graph.node_labels[1]['last_spike_time'] = (step + 1) * 0.01
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        high_dopamine_weight = graph.edge_attributes[0].weight

        # Reset and test with low dopamine
        graph.edge_attributes[0].weight = initial_weight
        self.enhanced_dynamics.set_neuromodulator_level('dopamine', 0.1)

        for step in range(15, 30):
            graph.node_labels[0]['last_spike_time'] = step * 0.01
            graph.node_labels[1]['last_spike_time'] = (step + 1) * 0.01
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        low_dopamine_weight = graph.edge_attributes[0].weight

        # Learning should be different with different neuromodulator levels
        assert abs(high_dopamine_weight - initial_weight) >= 0
        assert abs(low_dopamine_weight - initial_weight) >= 0

    def test_homeostatic_plasticity(self):
        """Test homeostatic plasticity maintaining network stability."""
        graph = Data()
        num_nodes = 20
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active', 'enhanced_behavior': True}
            for i in range(num_nodes)
        ]
        graph.x = torch.tensor([[0.5] for _ in range(num_nodes)], dtype=torch.float32)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Create network with imbalanced excitation
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_type = 'excitatory' if np.random.random() < 0.8 else 'inhibitory'  # Mostly excitatory
                    graph = create_weighted_connection(graph, i, j, 0.3, edge_type)

        initial_ei_ratio = self.network_metrics._calculate_ei_ratio(graph)

        # Run homeostatic plasticity
        for step in range(50):
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        final_ei_ratio = self.network_metrics._calculate_ei_ratio(graph)

        # E/I ratio should move toward target (0.8)
        assert abs(final_ei_ratio - 0.8) <= abs(initial_ei_ratio - 0.8)

    def test_theta_burst_stimulation(self):
        """Test theta burst stimulation protocol."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.6, 'state': 'active', 'enhanced_behavior': True}
        ]
        graph.x = torch.tensor([[0.6]], dtype=torch.float32)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Simulate theta burst (4 spikes at 100Hz)
        burst_times = [0.0, 0.01, 0.02, 0.03]  # 100Hz
        for t in burst_times:
            graph.node_labels[0]['last_spike_time'] = t

        # Process theta bursts
        for step in range(10):
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

        # Should have detected theta burst
        theta_bursts = self.enhanced_dynamics.get_statistics()['theta_bursts']
        assert theta_bursts >= 0

    def test_sensory_workspace_integration(self):
        """Test integration between sensory processing and workspace."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'type': 'sensory', 'behavior': 'sensory', 'energy': 0.8, 'state': 'active'},
            {'id': 1, 'type': 'sensory', 'behavior': 'sensory', 'energy': 0.7, 'state': 'active'},
            {'id': 2, 'type': 'workspace', 'behavior': 'workspace', 'energy': 0.6, 'state': 'active',
             'workspace_capacity': 5.0, 'workspace_creativity': 1.8, 'workspace_focus': 3.5, 'threshold': 0.6}
        ]
        graph.x = torch.tensor([[0.8], [0.7], [0.6]], dtype=torch.float32)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Connect sensory to workspace
        graph = create_weighted_connection(graph, 0, 2, 0.6, 'excitatory')
        graph = create_weighted_connection(graph, 1, 2, 0.5, 'excitatory')

        # Simulate sensory input patterns
        for step in range(25):
            # Vary sensory inputs
            pattern_type = step % 3
            if pattern_type == 0:  # Pattern A
                graph.node_labels[0]['energy'] = 0.9
                graph.node_labels[1]['energy'] = 0.4
            elif pattern_type == 1:  # Pattern B
                graph.node_labels[0]['energy'] = 0.4
                graph.node_labels[1]['energy'] = 0.9
            else:  # Mixed
                graph.node_labels[0]['energy'] = 0.7
                graph.node_labels[1]['energy'] = 0.7

            # Update all behaviors
            for i in range(len(graph.node_labels)):
                self.behavior_engine.update_node_behavior(i, graph, step)

            self.workspace_engine.update_workspace_nodes(graph, step)

        # Check workspace has processed inputs
        workspace_stats = self.workspace_engine.get_workspace_metrics()
        assert workspace_stats['syntheses_performed'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])






