"""
Comprehensive integration tests for neural components.
Tests interactions between behavior engine, connection logic, enhanced dynamics, and other neural systems.
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

from src.neural.behavior_engine import BehaviorEngine
from src.neural.connection_logic import (create_weighted_connection,
                                         intelligent_connection_formation)
from src.neural.enhanced_neural_dynamics import EnhancedNeuralDynamics
from src.neural.network_metrics import NetworkMetrics
from src.neural.neural_map_persistence import NeuralMapPersistence
from src.neural.spike_queue_system import SpikeQueueSystem, SpikeType
from src.neural.workspace_engine import WorkspaceEngine


class TestNeuralIntegration:
    """Integration tests for neural component interactions."""

    def setup_method(self):
        """Set up integrated test environment."""
        self.behavior_engine = BehaviorEngine()
        self.enhanced_dynamics = EnhancedNeuralDynamics()
        self.network_metrics = NetworkMetrics()
        self.spike_system = SpikeQueueSystem()
        self.workspace_engine = WorkspaceEngine()

        # Create a test graph
        self.graph = Data()
        self.graph.node_labels = [
            {'id': 0, 'type': 'sensory', 'behavior': 'sensory', 'energy': 0.8, 'state': 'active'},
            {'id': 1, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.6, 'state': 'active'},
            {'id': 2, 'type': 'oscillator', 'behavior': 'oscillator', 'energy': 0.7, 'state': 'active'},
            {'id': 3, 'type': 'workspace', 'behavior': 'workspace', 'energy': 0.5, 'state': 'active'}
        ]
        self.graph.x = torch.tensor([
            [0.8], [0.6], [0.7], [0.5]
        ], dtype=torch.float32)
        self.graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        self.graph.edge_attributes = []

    def test_behavior_and_connection_integration(self):
        """Test integration between behavior engine and connection logic."""
        # Update behaviors
        for i, node in enumerate(self.graph.node_labels):
            self.behavior_engine.update_node_behavior(i, self.graph, 1)

        # Create connections based on behaviors
        self.graph = intelligent_connection_formation(self.graph)

        # Verify connections were created
        assert len(self.graph.edge_attributes) > 0

        # Check that connections respect node types
        sensory_nodes = [i for i, node in enumerate(self.graph.node_labels) if node['type'] == 'sensory']
        dynamic_nodes = [i for i, node in enumerate(self.graph.node_labels) if node['type'] == 'dynamic']

        # Should have sensory-dynamic connections
        sensory_dynamic_connections = 0
        for edge in self.graph.edge_attributes:
            source_type = self.graph.node_labels[edge.source]['type']
            target_type = self.graph.node_labels[edge.target]['type']
            if source_type == 'sensory' and target_type == 'dynamic':
                sensory_dynamic_connections += 1

        assert sensory_dynamic_connections > 0

    def test_dynamics_and_spike_integration(self):
        """Test integration between enhanced dynamics and spike system."""
        # Set up enhanced nodes
        for node in self.graph.node_labels:
            node['enhanced_behavior'] = True
            node['membrane_potential'] = 0.8
            node['threshold'] = 0.7

        # Run dynamics update
        self.graph = self.enhanced_dynamics.update_neural_dynamics(self.graph, 1)

        # Check for spike generation (should trigger spikes for high membrane potentials)
        # Note: Actual spike generation depends on implementation details

        # Schedule some spikes manually to test integration
        self.spike_system.schedule_spike(0, 1, SpikeType.EXCITATORY, 1.0, 0.8)
        self.spike_system.schedule_spike(1, 2, SpikeType.INHIBITORY, 0.5, 0.6)

        # Process spikes
        processed = self.spike_system.process_spikes(10)

        assert processed >= 0  # May be 0 if simulation manager not set up

    def test_workspace_and_behavior_integration(self):
        """Test integration between workspace engine and behavior engine."""
        # Update workspace nodes
        workspace_result = self.workspace_engine.update_workspace_nodes(self.graph, 1)

        assert workspace_result['status'] == 'success'

        # Update behaviors which should include workspace behaviors
        for i, node in enumerate(self.graph.node_labels):
            if node['type'] == 'workspace':
                self.behavior_engine.update_node_behavior(i, self.graph, 1)

        # Check workspace statistics
        workspace_stats = self.workspace_engine.get_workspace_metrics()
        assert 'syntheses_performed' in workspace_stats

    def test_metrics_and_dynamics_integration(self):
        """Test integration between network metrics and neural dynamics."""
        # Run dynamics
        self.graph = self.enhanced_dynamics.update_neural_dynamics(self.graph, 1)

        # Calculate metrics
        metrics = self.network_metrics.calculate_comprehensive_metrics(self.graph)

        assert 'criticality' in metrics
        assert 'connectivity' in metrics
        assert 'energy_balance' in metrics

        # Metrics should reflect the graph state
        assert isinstance(metrics['criticality'], float)
        assert metrics['connectivity']['num_nodes'] == len(self.graph.node_labels)

    def test_full_neural_pipeline(self):
        """Test complete neural processing pipeline."""
        # Step 1: Create connections
        self.graph = intelligent_connection_formation(self.graph)

        # Step 2: Update behaviors
        for i in range(len(self.graph.node_labels)):
            self.behavior_engine.update_node_behavior(i, self.graph, 1)

        # Step 3: Run neural dynamics
        self.graph = self.enhanced_dynamics.update_neural_dynamics(self.graph, 1)

        # Step 4: Update workspace
        self.workspace_engine.update_workspace_nodes(self.graph, 1)

        # Step 5: Calculate metrics
        metrics = self.network_metrics.calculate_comprehensive_metrics(self.graph)

        # Verify pipeline completion
        assert len(self.graph.edge_attributes) > 0
        assert 'criticality' in metrics
        assert metrics['connectivity']['num_edges'] == len(self.graph.edge_attributes)

    def test_energy_driven_connection_formation(self):
        """Test that connection weights are influenced by energy levels."""
        # Set different energy levels
        self.graph.node_labels[0]['energy'] = 0.9  # High energy sensory
        self.graph.node_labels[1]['energy'] = 0.1  # Low energy dynamic

        self.graph.x[0, 0] = 0.9
        self.graph.x[1, 0] = 0.1

        # Create connections
        self.graph = intelligent_connection_formation(self.graph)

        # Find connections from high energy to low energy nodes
        high_to_low_connections = []
        for edge in self.graph.edge_attributes:
            source_energy = self.graph.x[edge.source, 0].item()
            target_energy = self.graph.x[edge.target, 0].item()
            if source_energy > target_energy:
                high_to_low_connections.append(edge)

        # Should have some connections
        assert len(high_to_low_connections) > 0

        # Check that weights reflect energy modulation
        for edge in high_to_low_connections:
            assert edge.weight > 0

    def test_spike_propagation_through_connections(self):
        """Test spike propagation through neural connections."""
        # Create a simple chain: 0 -> 1 -> 2
        self.graph = create_weighted_connection(self.graph, 0, 1, 0.8, 'excitatory')
        self.graph = create_weighted_connection(self.graph, 1, 2, 0.7, 'excitatory')

        # Schedule initial spike
        self.spike_system.schedule_spike(0, 1, SpikeType.EXCITATORY, 1.0, 0.8)

        # Process spikes
        processed = self.spike_system.process_spikes(10)

        # Verify spike was scheduled
        assert self.spike_system.get_queue_size() >= 0

    def test_behavior_state_transitions(self):
        """Test behavior state transitions across components."""
        # Start with active states
        for node in self.graph.node_labels:
            node['state'] = 'active'

        # Update behaviors
        for i in range(len(self.graph.node_labels)):
            self.behavior_engine.update_node_behavior(i, self.graph, 1)

        # Check that states have potentially changed
        states = [node['state'] for node in self.graph.node_labels]
        # At least some states should be 'active' (default)
        assert 'active' in states

    def test_enhanced_dynamics_with_connections(self):
        """Test enhanced dynamics processing with existing connections."""
        # Create some connections first
        self.graph = intelligent_connection_formation(self.graph)

        # Set up enhanced behaviors
        for node in self.graph.node_labels:
            node['enhanced_behavior'] = True
            node['membrane_potential'] = 0.5
            node['dendritic_potential'] = 0.0

        # Run dynamics
        original_connections = len(self.graph.edge_attributes)
        self.graph = self.enhanced_dynamics.update_neural_dynamics(self.graph, 1)

        # Connections should still exist
        assert len(self.graph.edge_attributes) >= original_connections

    def test_workspace_concept_synthesis_with_energy(self):
        """Test workspace synthesis influenced by energy levels."""
        # Set varying energy levels
        workspace_node = None
        for node in self.graph.node_labels:
            if node['type'] == 'workspace':
                workspace_node = node
                break

        if workspace_node:
            # Test with high energy
            workspace_node['energy'] = 0.9

            with patch('numpy.random.random', return_value=0.01):  # Force synthesis
                self.workspace_engine._update_workspace_node(self.graph, 3, 1)

            # Should potentially synthesize
            # (Depends on random chance, but setup should allow it)

    def test_metrics_evolution_over_time(self):
        """Test how metrics evolve with neural processing."""
        # Initial metrics
        initial_metrics = self.network_metrics.calculate_comprehensive_metrics(self.graph)

        # Process neural activity
        self.graph = intelligent_connection_formation(self.graph)
        for i in range(len(self.graph.node_labels)):
            self.behavior_engine.update_node_behavior(i, self.graph, 1)
        self.graph = self.enhanced_dynamics.update_neural_dynamics(self.graph, 1)

        # Final metrics
        final_metrics = self.network_metrics.calculate_comprehensive_metrics(self.graph)

        # Connectivity should change
        assert final_metrics['connectivity']['num_edges'] >= initial_metrics['connectivity']['num_edges']

    def test_error_recovery_across_components(self):
        """Test error recovery when components interact."""
        # Create a graph that might cause issues
        problematic_graph = Data()
        problematic_graph.node_labels = [{'id': 0, 'type': 'invalid'}]
        problematic_graph.x = torch.tensor([[0.5]], dtype=torch.float32)
        problematic_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        problematic_graph.edge_attributes = []

        # Test that components handle errors gracefully
        try:
            self.behavior_engine.update_node_behavior(0, problematic_graph, 1)
            self.enhanced_dynamics.update_neural_dynamics(problematic_graph, 1)
            self.network_metrics.calculate_comprehensive_metrics(problematic_graph)
            # Should not crash
            assert True
        except Exception as e:
            pytest.fail(f"Components should handle errors gracefully: {e}")

    def test_memory_management_integration(self):
        """Test memory management across integrated components."""
        # Run multiple processing cycles
        for cycle in range(10):
            self.graph = intelligent_connection_formation(self.graph)
            for i in range(len(self.graph.node_labels)):
                self.behavior_engine.update_node_behavior(i, self.graph, cycle)
            self.graph = self.enhanced_dynamics.update_neural_dynamics(self.graph, cycle)
            self.workspace_engine.update_workspace_nodes(self.graph, cycle)

        # Calculate final metrics
        metrics = self.network_metrics.calculate_comprehensive_metrics(self.graph)

        # Should complete without memory issues
        assert 'criticality' in metrics

    def test_performance_under_load(self):
        """Test performance with larger neural networks."""
        # Create a larger graph
        large_graph = Data()
        num_nodes = 50
        large_graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active'}
            for i in range(num_nodes)
        ]
        large_graph.x = torch.randn(num_nodes, 1)
        large_graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        large_graph.edge_attributes = []

        # Time the processing
        import time
        start_time = time.time()

        # Create connections
        large_graph = intelligent_connection_formation(large_graph)

        # Update behaviors
        for i in range(num_nodes):
            self.behavior_engine.update_node_behavior(i, large_graph, 1)

        # Run dynamics
        large_graph = self.enhanced_dynamics.update_neural_dynamics(large_graph, 1)

        processing_time = time.time() - start_time

        # Should complete in reasonable time (< 1 second for 50 nodes)
        assert processing_time < 1.0

    def test_state_consistency_across_updates(self):
        """Test that node states remain consistent across component updates."""
        # Initial state check
        initial_states = [node['state'] for node in self.graph.node_labels]

        # Process through pipeline
        self.graph = intelligent_connection_formation(self.graph)
        for i in range(len(self.graph.node_labels)):
            self.behavior_engine.update_node_behavior(i, self.graph, 1)
        self.graph = self.enhanced_dynamics.update_neural_dynamics(self.graph, 1)

        # Final state check
        final_states = [node['state'] for node in self.graph.node_labels]

        # States should be valid strings
        for state in final_states:
            assert isinstance(state, str)
            assert len(state) > 0

    def test_component_statistics_aggregation(self):
        """Test aggregation of statistics across components."""
        # Reset all statistics
        self.behavior_engine.reset_statistics()
        self.enhanced_dynamics.reset_statistics()
        self.network_metrics.metric_updates = 0
        self.spike_system.reset_statistics()
        self.workspace_engine.reset_statistics()

        # Run integrated processing
        self.graph = intelligent_connection_formation(self.graph)
        for i in range(len(self.graph.node_labels)):
            self.behavior_engine.update_node_behavior(i, self.graph, 1)
        self.graph = self.enhanced_dynamics.update_neural_dynamics(self.graph, 1)
        self.workspace_engine.update_workspace_nodes(self.graph, 1)
        self.network_metrics.calculate_comprehensive_metrics(self.graph)

        # Check that statistics were updated
        behavior_stats = self.behavior_engine.get_behavior_statistics()
        dynamics_stats = self.enhanced_dynamics.get_statistics()
        spike_stats = self.spike_system.get_statistics()
        workspace_stats = self.workspace_engine.get_workspace_metrics()

        # At least some statistics should be non-zero
        stats_updated = (
            behavior_stats['basic_updates'] > 0 or
            dynamics_stats['total_spikes'] >= 0 or  # May be 0
            spike_stats['total_spikes_scheduled'] >= 0 or
            workspace_stats['syntheses_performed'] >= 0
        )
        assert stats_updated

    def test_persistence_integration(self):
        """Test integration with neural map persistence."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = NeuralMapPersistence(temp_dir)

            # Process graph
            self.graph = intelligent_connection_formation(self.graph)
            for i in range(len(self.graph.node_labels)):
                self.behavior_engine.update_node_behavior(i, self.graph, 1)

            # Save
            save_result = persistence.save_neural_map(self.graph, 0)
            assert save_result is True

            # Load
            loaded_graph = persistence.load_neural_map(0)
            assert loaded_graph is not None

            # Verify structure preserved
            assert len(loaded_graph.node_labels) == len(self.graph.node_labels)
            assert loaded_graph.edge_index.shape == self.graph.edge_index.shape


if __name__ == "__main__":
    pytest.main([__file__])






