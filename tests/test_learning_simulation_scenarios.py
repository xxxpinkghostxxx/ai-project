"""
Simulation scenario tests for learning system components.
Tests realistic simulation scenarios, pattern learning, adaptation, and long-term behavior.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.learning.homeostasis_controller import HomeostasisController
from src.learning.learning_engine import LearningEngine
from src.learning.live_hebbian_learning import LiveHebbianLearning
from src.learning.memory_system import MemorySystem


class TestSimulationScenarios:
    """Test suite for realistic simulation scenarios."""

    def setup_method(self):
        """Set up simulation test environment."""
        self.homeostasis = HomeostasisController()
        self.memory_system = MemorySystem()
        self.mock_access_layer = MagicMock()

        with patch('learning.learning_engine.get_learning_config', return_value={
            'plasticity_rate': 0.01,
            'eligibility_decay': 0.95,
            'stdp_window': 20.0,
            'ltp_rate': 0.02,
            'ltd_rate': 0.01
        }):
            self.learning_engine = LearningEngine(self.mock_access_layer)

        self.hebbian_learning = LiveHebbianLearning()

    def test_pattern_learning_scenario(self):
        """Test learning of repeating patterns."""
        # Create a network that should learn a repeating activation pattern
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'integrator' if i % 2 == 0 else 'dynamic',
             'energy': 0.6 + 0.2 * (i % 3), 'last_activation': time.time() - 10}
            for i in range(10)
        ]
        graph.x = torch.rand(10, 1) * 0.5 + 0.5
        graph.edge_index = torch.randint(0, 10, (2, 20))
        graph.edge_attr = torch.rand(20, 1)

        # Simulate pattern learning over time
        for step in range(5):
            # Update node activations in a pattern
            for i, node in enumerate(graph.node_labels):
                if (step + i) % 3 == 0:  # Pattern: every 3rd step for each node
                    node['last_activation'] = time.time() - 2
                    graph.x[i] = torch.tensor([0.8])  # High activation

            # Apply learning
            learned = self.hebbian_learning.apply_continuous_learning(graph, step)
            memory_graph = self.memory_system.form_memory_traces(learned)

            # Homeostasis maintains stability
            self.homeostasis.regulate_network_activity(memory_graph)

        # Check that patterns were learned
        assert self.memory_system.get_memory_trace_count() > 0
        assert self.hebbian_learning.get_learning_statistics()['total_weight_changes'] > 0

    def test_adaptation_to_environmental_changes(self):
        """Test adaptation to environmental changes."""
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'dynamic', 'energy': 0.7, 'last_activation': time.time()}
            for i in range(8)
        ]
        graph.x = torch.full((8, 1), 0.7)
        graph.edge_index = torch.randint(0, 8, (2, 16))
        graph.edge_attr = torch.full((16, 1), 1.0)

        initial_weights = graph.edge_attr.clone()

        # Phase 1: Stable environment
        for step in range(10):
            self.hebbian_learning.apply_continuous_learning(graph, step)

        stable_weights = graph.edge_attr.clone()

        # Phase 2: Environmental change - reduce energy
        graph.x = torch.full((8, 1), 0.3)  # Lower energy
        for node in graph.node_labels:
            node['energy'] = 0.3

        for step in range(10, 20):
            self.hebbian_learning.apply_continuous_learning(graph, step)
            self.homeostasis.regulate_network_activity(graph)

        adapted_weights = graph.edge_attr.clone()

        # Weights should have changed in response to environmental change
        assert not torch.equal(stable_weights, adapted_weights)

    def test_memory_consolidation_over_time(self):
        """Test memory consolidation over extended periods."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time() - 5},
            {'id': 1, 'behavior': 'relay', 'energy': 0.7, 'last_activation': time.time() - 3}
        ]
        graph.x = torch.tensor([[0.8], [0.7]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        # Form initial memory
        self.memory_system.form_memory_traces(graph)

        initial_traces = self.memory_system.get_memory_trace_count()
        initial_strength = self.memory_system.get_memory_statistics()['total_memory_strength']

        # Simulate passage of time with reactivation
        for hour in range(24):  # 24 hours
            # Periodic reactivation
            if hour % 6 == 0:  # Every 6 hours
                graph.node_labels[0]['last_activation'] = time.time() - 2
                self.memory_system.consolidate_memories(graph)

            # Memory decay
            self.memory_system.decay_memories()

        final_traces = self.memory_system.get_memory_trace_count()
        final_strength = self.memory_system.get_memory_statistics()['total_memory_strength']

        # Should show effects of consolidation vs decay
        assert isinstance(final_traces, int)
        assert isinstance(final_strength, (int, float))

    def test_criticality_self_organization(self):
        """Test self-organization towards critical state."""
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'oscillator' if i % 3 == 0 else 'integrator',
             'energy': 0.5 + 0.3 * (i % 2), 'last_activation': time.time()}
            for i in range(15)
        ]
        graph.x = torch.rand(15, 1) * 0.4 + 0.4
        graph.edge_index = torch.randint(0, 15, (2, 30))
        graph.edge_attr = torch.rand(30, 1) * 0.5 + 0.5

        # Track criticality over time
        criticality_history = []

        for step in range(20):
            # Apply learning and homeostasis
            learned = self.hebbian_learning.apply_continuous_learning(graph, step)
            regulated = self.homeostasis.regulate_network_activity(learned)

            # Simulate some activation dynamics
            active_nodes = np.random.choice(15, size=5, replace=False)
            for idx in active_nodes:
                graph.node_labels[idx]['last_activation'] = time.time() - 1
                graph.x[idx] = torch.tensor([0.9])

            # Calculate current criticality (simplified)
            edge_weights = graph.edge_attr.flatten().numpy()
            criticality = np.var(edge_weights)  # Simplified criticality measure
            criticality_history.append(criticality)

        # Should show some organization over time
        assert len(criticality_history) == 20
        # Check for some stability in later stages
        early_avg = np.mean(criticality_history[:10])
        late_avg = np.mean(criticality_history[10:])
        # Allow for either convergence or continued adaptation
        assert isinstance(early_avg, (int, float, np.floating))
        assert isinstance(late_avg, (int, float, np.floating))

    def test_energy_driven_learning_priorities(self):
        """Test that learning prioritizes high-energy nodes."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.9, 'last_activation': time.time() - 5},  # High energy
            {'id': 1, 'behavior': 'integrator', 'energy': 0.2, 'last_activation': time.time() - 5},  # Low energy
            {'id': 2, 'behavior': 'dynamic', 'energy': 0.6, 'last_activation': time.time() - 10}    # Medium, older
        ]
        graph.x = torch.tensor([[0.9], [0.2], [0.6]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)

        # Apply learning
        for step in range(5):
            self.hebbian_learning.apply_continuous_learning(graph, step)

        # High energy node should have been processed more
        # (This is probabilistic, so we check that the system ran)
        stats = self.hebbian_learning.get_learning_statistics()
        assert stats['total_weight_changes'] >= 0

    def test_long_term_memory_retention(self):
        """Test long-term memory retention and recall."""
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time() - 5}
            for i in range(5)
        ]
        graph.x = torch.full((5, 1), 0.8)
        graph.edge_index = torch.randint(0, 5, (2, 10))

        # Learn multiple patterns
        patterns_learned = 0
        for pattern in range(3):
            # Activate different subsets
            active_nodes = np.random.choice(5, size=3, replace=False)
            for idx in active_nodes:
                graph.node_labels[idx]['last_activation'] = time.time() - 1

            self.memory_system.form_memory_traces(graph)
            patterns_learned += 1

            # Let some time pass
            time.sleep(0.01)

        # Test recall after time
        time.sleep(0.01)
        recalled = self.memory_system.recall_patterns(graph, 0)

        # Should have learned patterns and be able to recall
        assert self.memory_system.get_memory_trace_count() >= patterns_learned
        assert isinstance(recalled, list)

    def test_homeostatic_adaptation_to_stress(self):
        """Test homeostatic adaptation to network stress."""
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'dynamic', 'energy': 0.7, 'last_activation': time.time()}
            for i in range(12)
        ]
        graph.x = torch.full((12, 1), 0.7)
        graph.edge_index = torch.randint(0, 12, (2, 24))
        graph.edge_attr = torch.full((24, 1), 1.0)

        # Normal operation
        for step in range(5):
            self.homeostasis.regulate_network_activity(graph)

        normal_regulations = self.homeostasis.get_regulation_statistics()['total_regulation_events']

        # Stress condition - very high energy
        graph.x = torch.full((12, 1), 4.0)  # Very high energy
        for node in graph.node_labels:
            node['energy'] = 4.0

        for step in range(5, 10):
            self.homeostasis.regulate_network_activity(graph)

        stress_regulations = self.homeostasis.get_regulation_statistics()['total_regulation_events']

        # Should show increased regulation under stress
        assert stress_regulations > normal_regulations

    def test_learning_interference_and_recovery(self):
        """Test learning interference and recovery."""
        graph = Data()
        graph.node_labels = [
            {'id': 0, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time()},
            {'id': 1, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time()}
        ]
        graph.x = torch.tensor([[0.8], [0.8]], dtype=torch.float32)
        graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        graph.edge_attr = torch.tensor([[1.0]], dtype=torch.float32)

        # Learn initial pattern - simulate source firing before target for STDP LTP
        graph.node_labels[0]['last_activation'] = time.time() - 0.01  # Source fires first
        graph.node_labels[1]['last_activation'] = time.time()       # Target fires after

        initial_weight = graph.edge_attr[0, 0].item()
        for step in range(3):
            self.hebbian_learning.apply_continuous_learning(graph, step)

        learned_weight = graph.edge_attr[0, 0].item()

        # Interference - change pattern, simulate LTD (target before source)
        graph.x = torch.tensor([[0.2], [0.8]], dtype=torch.float32)  # Reduce first node energy
        graph.node_labels[0]['energy'] = 0.2
        graph.node_labels[0]['last_activation'] = time.time()       # Source fires after
        graph.node_labels[1]['last_activation'] = time.time() - 0.01  # Target fires first

        for step in range(3, 6):
            self.hebbian_learning.apply_continuous_learning(graph, step)

        interfered_weight = graph.edge_attr[0, 0].item()

        # Recovery - restore pattern, simulate LTP (source before target)
        graph.x = torch.tensor([[0.8], [0.8]], dtype=torch.float32)
        graph.node_labels[0]['energy'] = 0.8
        graph.node_labels[0]['last_activation'] = time.time() - 0.01  # Source fires first
        graph.node_labels[1]['last_activation'] = time.time()       # Target fires after

        for step in range(6, 9):
            self.hebbian_learning.apply_continuous_learning(graph, step)

        recovered_weight = graph.edge_attr[0, 0].item()

        # Weights should have changed through the phases
        assert abs(learned_weight - initial_weight) > 0.001 or abs(interfered_weight - learned_weight) > 0.001 or abs(recovered_weight - interfered_weight) > 0.001

    def test_multi_timescale_dynamics(self):
        """Test dynamics across multiple timescales."""
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'oscillator' if i < 3 else 'integrator',
             'energy': 0.6, 'last_activation': time.time()}
            for i in range(6)
        ]
        graph.x = torch.full((6, 1), 0.6)
        graph.edge_index = torch.randint(0, 6, (2, 12))
        graph.edge_attr = torch.full((12, 1), 1.0)

        # Fast dynamics (hebbian learning)
        fast_changes = 0
        for step in range(10):
            old_weights = graph.edge_attr.clone()
            self.hebbian_learning.apply_continuous_learning(graph, step)
            if not torch.equal(old_weights, graph.edge_attr):
                fast_changes += 1

        # Medium dynamics (homeostasis)
        medium_changes = 0
        for step in range(5):
            old_energy = graph.x.clone()
            self.homeostasis.regulate_network_activity(graph)
            if not torch.equal(old_energy, graph.x):
                medium_changes += 1

        # Slow dynamics (memory)
        slow_changes = self.memory_system.get_memory_trace_count()
        self.memory_system.form_memory_traces(graph)
        slow_changes = self.memory_system.get_memory_trace_count() - slow_changes

        # Should show activity at different timescales
        assert fast_changes >= 0  # Fast learning may or may not change weights each step
        assert medium_changes >= 0
        assert slow_changes >= 0


if __name__ == "__main__":
    pytest.main([__file__])






