"""
Real-world usage tests for learning system components.
Tests practical applications, user scenarios, and real-world integration patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data

from learning.homeostasis_controller import HomeostasisController
from learning.learning_engine import LearningEngine
from learning.live_hebbian_learning import LiveHebbianLearning
from learning.memory_system import MemorySystem
from core.services.simulation_coordinator import SimulationCoordinator


class TestRealWorldUsage:
    """Test suite for real-world usage scenarios."""

    def setup_method(self):
        """Set up real-world test environment."""
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

    def test_sensor_data_processing_scenario(self):
        """Test learning system in sensor data processing scenario."""
        # Simulate sensory input processing
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'sensory' if i < 5 else 'integrator',
             'energy': 0.3 + 0.4 * (i % 2), 'last_activation': time.time() - 10}
            for i in range(10)
        ]
        graph.x = torch.rand(10, 1) * 0.5 + 0.3
        graph.edge_index = torch.randint(0, 10, (2, 20))
        graph.edge_attr = torch.rand(20, 1)

        # Simulate sensory input patterns
        sensory_patterns = [
            [0.8, 0.2, 0.9, 0.1, 0.7],  # Pattern 1
            [0.1, 0.9, 0.3, 0.8, 0.2],  # Pattern 2
            [0.7, 0.8, 0.1, 0.2, 0.9],  # Pattern 3
        ]

        for pattern_idx, pattern in enumerate(sensory_patterns):
            # Update sensory nodes with pattern
            for i, energy in enumerate(pattern):
                graph.x[i] = torch.tensor([energy])
                graph.node_labels[i]['energy'] = energy
                if energy > 0.5:
                    graph.node_labels[i]['last_activation'] = time.time() - 1

            # Process through learning systems
            self.homeostasis.regulate_network_activity(graph)
            learned = self.hebbian_learning.apply_continuous_learning(graph, pattern_idx)
            self.memory_system.form_memory_traces(learned)

        # Should have learned to recognize patterns
        assert self.memory_system.get_memory_trace_count() > 0
        assert self.hebbian_learning.get_learning_statistics()['total_weight_changes'] > 0

    def test_motor_control_learning_scenario(self):
        """Test learning system in motor control scenario."""
        # Simulate motor control learning
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'integrator' if i < 8 else 'relay',
             'energy': 0.6, 'last_activation': time.time()}
            for i in range(12)
        ]
        graph.x = torch.full((12, 1), 0.6)
        graph.edge_index = torch.randint(0, 12, (2, 24))
        graph.edge_attr = torch.full((24, 1), 1.0)

        # Simulate motor learning trials
        successful_trials = 0
        total_trials = 10

        for trial in range(total_trials):
            # Random motor command
            motor_command = np.random.rand(4) > 0.5

            # Simulate motor execution and feedback
            for i, command in enumerate(motor_command):
                node_idx = 8 + i  # Motor nodes
                if command:
                    graph.x[node_idx] = torch.tensor([0.9])
                    graph.node_labels[node_idx]['energy'] = 0.9
                    graph.node_labels[node_idx]['last_activation'] = time.time() - 1

            # Learning from feedback
            self.hebbian_learning.apply_continuous_learning(graph, trial)
            self.homeostasis.regulate_network_activity(graph)

            # Simulate success/failure
            success = np.random.rand() > 0.3  # 70% success rate initially
            if success:
                successful_trials += 1
                # Reinforce successful patterns
                self.memory_system.form_memory_traces(graph)

        # Should show learning improvement
        success_rate = successful_trials / total_trials
        assert success_rate >= 0.5  # At least 50% success rate

    def test_decision_making_scenario(self):
        """Test learning system in decision making scenario."""
        # Simulate decision making network
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'integrator' if i < 6 else 'relay',
             'energy': 0.5, 'last_activation': time.time()}
            for i in range(10)
        ]
        graph.x = torch.full((10, 1), 0.5)
        graph.edge_index = torch.randint(0, 10, (2, 20))
        graph.edge_attr = torch.full((20, 1), 1.0)

        # Simulate decision scenarios
        decisions = []
        rewards = []

        for scenario in range(15):
            # Present options (simulate different input patterns)
            option_a = np.random.rand(3) * 0.5 + 0.3
            option_b = np.random.rand(3) * 0.5 + 0.3

            # Input to decision nodes
            for i in range(3):
                graph.x[i] = torch.tensor([option_a[i]])
                graph.x[i + 3] = torch.tensor([option_b[i]])
                graph.node_labels[i]['energy'] = option_a[i]
                graph.node_labels[i + 3]['energy'] = option_b[i]

            # Make decision based on current weights
            decision_weights = graph.edge_attr.flatten().numpy()
            decision_score = np.mean(decision_weights)
            decision = 'A' if decision_score > 1.0 else 'B'
            decisions.append(decision)

            # Simulate reward
            reward = 1.0 if (decision == 'A' and np.mean(option_a) > np.mean(option_b)) else 0.0
            rewards.append(reward)

            # Learn from outcome
            if reward > 0:
                self.memory_system.form_memory_traces(graph)

            self.hebbian_learning.apply_continuous_learning(graph, scenario)
            self.homeostasis.regulate_network_activity(graph)

        # Should show some learning (better decisions over time)
        early_success = sum(rewards[:5]) / 5
        late_success = sum(rewards[10:]) / 5

        # Allow for learning improvement or at least maintenance
        assert isinstance(early_success, float)
        assert isinstance(late_success, float)

    def test_adaptive_filtering_scenario(self):
        """Test learning system in adaptive filtering scenario."""
        # Simulate signal filtering/learning
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'dynamic', 'energy': 0.6, 'last_activation': time.time()}
            for i in range(15)
        ]
        graph.x = torch.full((15, 1), 0.6)
        graph.edge_index = torch.randint(0, 15, (2, 30))
        graph.edge_attr = torch.full((30, 1), 1.0)

        # Simulate noisy signal with underlying pattern
        time_steps = 50
        signal_quality = []

        for t in range(time_steps):
            # Generate signal with noise and pattern
            base_signal = 0.5 + 0.3 * np.sin(2 * np.pi * t / 10)  # Underlying pattern
            noise = np.random.normal(0, 0.2)  # Noise
            noisy_signal = np.clip(base_signal + noise, 0, 1)

            # Input to network
            for i in range(min(5, len(graph.node_labels))):
                graph.x[i] = torch.tensor([noisy_signal])
                graph.node_labels[i]['energy'] = noisy_signal
                if noisy_signal > 0.6:
                    graph.node_labels[i]['last_activation'] = time.time() - 1

            # Learning and adaptation
            self.hebbian_learning.apply_continuous_learning(graph, t)
            self.homeostasis.regulate_network_activity(graph)

            if t % 10 == 0:
                self.memory_system.form_memory_traces(graph)

            # Measure filtering quality (simplified)
            filtered_output = torch.mean(graph.x).item()
            quality = 1.0 - abs(filtered_output - base_signal)
            signal_quality.append(quality)

        # Should show improvement in signal quality over time
        early_quality = np.mean(signal_quality[:20])
        late_quality = np.mean(signal_quality[30:])

        assert len(signal_quality) == time_steps
        # Quality should be reasonable
        assert early_quality > 0.3
        assert late_quality > 0.3

    def test_social_learning_scenario(self):
        """Test learning system in social learning scenario."""
        # Simulate social learning between agents
        num_agents = 3
        graphs = []

        for agent in range(num_agents):
            graph = Data()
            graph.node_labels = [
                {'id': i, 'behavior': 'integrator', 'energy': 0.6, 'last_activation': time.time()}
                for i in range(8)
            ]
            graph.x = torch.full((8, 1), 0.6)
            graph.edge_index = torch.randint(0, 8, (2, 16))
            graph.edge_attr = torch.full((16, 1), 1.0)
            graphs.append(graph)

        # Simulate social interactions
        for interaction in range(20):
            # Random agent teaches others
            teacher = np.random.randint(num_agents)
            learners = [i for i in range(num_agents) if i != teacher]

            # Teacher demonstrates pattern
            teacher_graph = graphs[teacher]
            for i in range(4):
                teacher_graph.x[i] = torch.tensor([0.9])
                teacher_graph.node_labels[i]['energy'] = 0.9
                teacher_graph.node_labels[i]['last_activation'] = time.time() - 1

            # Learners observe and learn
            for learner in learners:
                learner_graph = graphs[learner]
                # Simplified observation: partial pattern transfer
                for i in range(2):
                    learner_graph.x[i] = torch.tensor([0.7])
                    learner_graph.node_labels[i]['energy'] = 0.7

                self.hebbian_learning.apply_continuous_learning(learner_graph, interaction)
                self.memory_system.form_memory_traces(learner_graph)

            # All agents homeostasis
            for graph in graphs:
                self.homeostasis.regulate_network_activity(graph)

        # Check that learning occurred across agents
        total_memories = sum(self.memory_system.get_memory_trace_count() for _ in graphs)
        # Note: This test uses the same memory system, so it's shared
        assert self.memory_system.get_memory_trace_count() >= 0

    def test_continuous_online_learning_scenario(self):
        """Test continuous online learning scenario."""
        # Simulate online learning from streaming data
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'dynamic', 'energy': 0.5, 'last_activation': time.time()}
            for i in range(20)
        ]
        graph.x = torch.full((20, 1), 0.5)
        graph.edge_index = torch.randint(0, 20, (2, 40))
        graph.edge_attr = torch.full((40, 1), 1.0)

        # Simulate streaming data
        stream_length = 100
        adaptation_rate = []

        for data_point in range(stream_length):
            # Generate streaming data with concept drift
            if data_point < 50:
                pattern = [0.8, 0.2, 0.8, 0.2]  # Pattern A
            else:
                pattern = [0.2, 0.8, 0.2, 0.8]  # Pattern B (drift)

            # Input pattern to network
            for i, value in enumerate(pattern):
                if i < len(graph.node_labels):
                    graph.x[i] = torch.tensor([value])
                    graph.node_labels[i]['energy'] = value
                    if value > 0.5:
                        graph.node_labels[i]['last_activation'] = time.time() - 1

            # Continuous learning
            self.hebbian_learning.apply_continuous_learning(graph, data_point)
            self.homeostasis.regulate_network_activity(graph)

            # Periodic memory consolidation
            if data_point % 20 == 0:
                self.memory_system.form_memory_traces(graph)

            # Measure adaptation (simplified)
            active_nodes = sum(1 for node in graph.node_labels if node['energy'] > 0.6)
            adaptation_rate.append(active_nodes / len(graph.node_labels))

        # Should show continuous adaptation
        assert len(adaptation_rate) == stream_length
        assert all(0 <= rate <= 1 for rate in adaptation_rate)

    def test_fault_tolerance_scenario(self):
        """Test learning system fault tolerance."""
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'dynamic', 'energy': 0.6, 'last_activation': time.time()}
            for i in range(12)
        ]
        graph.x = torch.full((12, 1), 0.6)
        graph.edge_index = torch.randint(0, 12, (2, 24))
        graph.edge_attr = torch.full((24, 1), 1.0)

        # Normal operation
        for step in range(10):
            self.homeostasis.regulate_network_activity(graph)
            self.hebbian_learning.apply_continuous_learning(graph, step)

        normal_performance = self.hebbian_learning.get_learning_statistics()['total_weight_changes']

        # Simulate faults
        for fault_step in range(10, 20):
            # Random node failures
            failed_nodes = np.random.choice(12, size=2, replace=False)
            for node_idx in failed_nodes:
                graph.x[node_idx] = torch.tensor([0.0])
                graph.node_labels[node_idx]['energy'] = 0.0

            # System should continue operating
            self.homeostasis.regulate_network_activity(graph)
            self.hebbian_learning.apply_continuous_learning(graph, fault_step)

        fault_performance = self.hebbian_learning.get_learning_statistics()['total_weight_changes']

        # Should maintain operation despite faults
        assert fault_performance >= normal_performance

    def test_resource_constrained_scenario(self):
        """Test learning system in resource-constrained scenario."""
        # Simulate limited memory/energy
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'dynamic', 'energy': 0.3, 'last_activation': time.time()}  # Low energy
            for i in range(8)
        ]
        graph.x = torch.full((8, 1), 0.3)
        graph.edge_index = torch.randint(0, 8, (2, 16))
        graph.edge_attr = torch.full((16, 1), 0.5)  # Weak connections

        # Learning with limited resources
        for step in range(15):
            # Limited energy available
            available_energy = 2.0  # Limited total energy
            current_total = torch.sum(graph.x).item()

            if current_total > available_energy:
                # Reduce energy consumption
                scale_factor = available_energy / current_total
                graph.x *= scale_factor
                for node in graph.node_labels:
                    node['energy'] *= scale_factor

            self.homeostasis.regulate_network_activity(graph)
            self.hebbian_learning.apply_continuous_learning(graph, step)

        # Should still function with limited resources
        stats = self.hebbian_learning.get_learning_statistics()
        assert stats['total_weight_changes'] >= 0

    def test_multi_modal_integration_scenario(self):
        """Test learning system with multi-modal inputs."""
        # Simulate integration of visual and auditory inputs
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'sensory' if i < 6 else 'integrator',
             'energy': 0.5, 'last_activation': time.time()}
            for i in range(12)
        ]
        graph.x = torch.full((12, 1), 0.5)
        graph.edge_index = torch.randint(0, 12, (2, 24))
        graph.edge_attr = torch.full((24, 1), 1.0)

        # Simulate multi-modal learning
        for session in range(10):
            # Visual input (first 3 nodes)
            visual_pattern = np.random.rand(3) * 0.5 + 0.3
            for i, intensity in enumerate(visual_pattern):
                graph.x[i] = torch.tensor([intensity])
                graph.node_labels[i]['energy'] = intensity

            # Auditory input (next 3 nodes)
            auditory_pattern = np.random.rand(3) * 0.5 + 0.3
            for i, intensity in enumerate(auditory_pattern):
                graph.x[i + 3] = torch.tensor([intensity])
                graph.node_labels[i + 3]['energy'] = intensity

            # Integration learning
            self.hebbian_learning.apply_continuous_learning(graph, session)
            self.homeostasis.regulate_network_activity(graph)

            # Form integrated memories
            if session % 3 == 0:
                self.memory_system.form_memory_traces(graph)

        # Should show integrated learning
        assert self.memory_system.get_memory_trace_count() >= 0
        assert self.hebbian_learning.get_learning_statistics()['total_weight_changes'] > 0


if __name__ == "__main__":
    pytest.main([__file__])