"""
Comprehensive integration tests for the energy system.

This module contains integration tests that combine all energy system components
including energy behavior, constants, node access layer, ID manager, flow diagrams,
and system validation to ensure they work together as a cohesive system.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time
import numpy as np
from unittest.mock import Mock, patch
from torch_geometric.data import Data
import torch

from src.energy.energy_behavior import (
    EnergyCalculator, get_node_energy_cap, update_node_energy_with_learning,
    apply_energy_behavior, apply_oscillator_energy_dynamics,
    apply_integrator_energy_dynamics, apply_relay_energy_dynamics,
    apply_highway_energy_dynamics, apply_dynamic_energy_dynamics, emit_energy_pulse
)
from src.energy.energy_constants import EnergyConstants, ConnectionConstants
from src.energy.node_access_layer import NodeAccessLayer
from src.energy.node_id_manager import NodeIDManager, get_id_manager
from src.energy.energy_flow_diagram import EnergyFlowDiagram
from src.energy.energy_system_validator import EnergySystemValidator


class TestEnergySystemIntegration(unittest.TestCase):
    """Integration tests for the complete energy system."""

    def setUp(self):
        """Set up integration test fixtures."""
        print("Starting setUp")
        # Create a comprehensive test graph
        self.graph = Data()
        self.graph.node_labels = []
        self.graph.x = torch.empty((0, 1), dtype=torch.float32)
        self.graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        print("Graph created")

        # Create diverse node types
        node_specs = [
            # Sensory input nodes
            {"type": "sensory", "behavior": "sensory", "energy": 0.2, "threshold": 0.5},
            {"type": "sensory", "behavior": "sensory", "energy": 0.3, "threshold": 0.5},

            # Dynamic processing nodes
            {"type": "dynamic", "behavior": "oscillator", "energy": 1.0, "threshold": 0.5,
             "oscillation_freq": 1.0, "last_activation": 0, "refractory_timer": 0},
            {"type": "dynamic", "behavior": "integrator", "energy": 0.8, "threshold": 0.5,
             "integration_rate": 0.5},
            {"type": "dynamic", "behavior": "relay", "energy": 0.9, "threshold": 0.5,
             "relay_amplification": 1.5},
            {"type": "dynamic", "behavior": "highway", "energy": 0.4, "threshold": 0.5,
             "highway_energy_boost": 2.0},

            # Output nodes
            {"type": "dynamic", "behavior": "dynamic", "energy": 0.6, "threshold": 0.5}
        ]

        # Register nodes and build graph
        print("Getting id_manager")
        self.id_manager = get_id_manager()
        print("Resetting id_manager")
        self.id_manager.reset()
        print("ID manager reset done")

        for i, spec in enumerate(node_specs):
            print(f"Generating ID for node {i}")
            node_id = self.id_manager.generate_unique_id(spec["type"])
            spec["id"] = node_id
            print(f"Registering index for node {i}, id {node_id}")
            self.id_manager.register_node_index(node_id, i)

            self.graph.node_labels.append(spec)
            self.graph.x = torch.cat([self.graph.x, torch.tensor([[spec["energy"]]], dtype=torch.float32)], dim=0)
        print("Nodes registered")

        # Create connections
        connections = [
            (0, 2), (1, 2),  # Sensory to oscillator
            (2, 3), (2, 4),  # Oscillator to integrator/relay
            (3, 5), (4, 5),  # Integrator/relay to highway
            (5, 6)           # Highway to output
        ]

        for source, target in connections:
            self.graph.edge_index = torch.cat([
                self.graph.edge_index,
                torch.tensor([[source], [target]], dtype=torch.long)
            ], dim=1)

        # Create access layer
        self.access_layer = NodeAccessLayer(self.graph, self.id_manager)

    def test_energy_system_initialization(self):
        """Test that all energy system components initialize correctly together."""
        # Test ID manager
        self.assertEqual(self.id_manager.get_all_active_ids(), list(range(1, 8)))

        # Test access layer
        self.assertEqual(self.access_layer.get_node_count(), 7)

        # Test energy constants
        self.assertGreater(EnergyConstants.get_activation_threshold(), 0)
        self.assertGreater(get_node_energy_cap(), 0)

        # Test energy calculator
        cap = EnergyCalculator.calculate_energy_cap()
        self.assertGreater(cap, 0)

    def test_energy_flow_through_system(self):
        """Test energy flow from input through processing to output."""
        # Initial state
        initial_energies = {}
        for node_id in range(7):
            initial_energies[node_id] = self.access_layer.get_node_energy(node_id)

        # Apply sensory input (simulate external stimulation)
        sensory_nodes = self.access_layer.select_nodes_by_type("sensory")
        for node_id in sensory_nodes:
            self.access_layer.set_node_energy(node_id, 1.5)  # High energy input

        # Process through oscillator
        oscillator_nodes = self.access_layer.select_nodes_by_behavior("oscillator")
        for node_id in oscillator_nodes:
            self.graph = apply_oscillator_energy_dynamics(self.graph, node_id)

        # Process through integrator
        integrator_nodes = self.access_layer.select_nodes_by_behavior("integrator")
        for node_id in integrator_nodes:
            self.graph = apply_integrator_energy_dynamics(self.graph, node_id)

        # Process through relay
        relay_nodes = self.access_layer.select_nodes_by_behavior("relay")
        for node_id in relay_nodes:
            self.graph = apply_relay_energy_dynamics(self.graph, node_id)

        # Process through highway
        highway_nodes = self.access_layer.select_nodes_by_behavior("highway")
        for node_id in highway_nodes:
            self.graph = apply_highway_energy_dynamics(self.graph, node_id)

        # Apply general energy behavior
        self.graph = apply_energy_behavior(self.graph)

        # Check that energy flowed through the system
        final_energies = {}
        for node_id in range(7):
            final_energies[node_id] = self.access_layer.get_node_energy(node_id)

        # Sensory nodes should have maintained high energy
        for node_id in sensory_nodes:
            self.assertGreaterEqual(final_energies[node_id], 1.0)

        # Processing nodes should have changed energy
        processing_nodes = oscillator_nodes + integrator_nodes + relay_nodes + highway_nodes
        energy_changed = False
        for node_id in processing_nodes:
            if abs(final_energies[node_id] - initial_energies[node_id]) > 0.01:
                energy_changed = True
                break
        self.assertTrue(energy_changed, "Energy should have flowed through processing nodes")

    def test_energy_learning_integration(self):
        """Test energy integration with learning mechanisms."""
        # Create learning scenario
        pre_node_id = 2  # Oscillator
        post_node_id = 3  # Integrator

        # Set up learning conditions
        self.access_layer.update_node_property(pre_node_id, "membrane_potential", 0.8)
        self.access_layer.update_node_property(post_node_id, "membrane_potential", 0.6)

        # Simulate learning event
        delta_energy = 0.2
        self.graph = update_node_energy_with_learning(self.graph, post_node_id, delta_energy)

        # Check that learning affected energy
        final_energy = self.access_layer.get_node_energy(post_node_id)
        initial_energy = self.graph.x[post_node_id, 0].item()

        # Energy should have changed
        self.assertNotEqual(final_energy, initial_energy)

    def test_energy_conservation_across_system(self):
        """Test energy conservation principles across the system."""
        # Measure total system energy before
        initial_total = 0
        for node_id in range(1, 8):
            energy = self.access_layer.get_node_energy(node_id)
            if energy is not None:
                initial_total += energy

        # Apply various energy operations
        self.graph = apply_energy_behavior(self.graph)
        self.graph = emit_energy_pulse(self.graph, 0)  # Pulse from first node

        # Apply behavior-specific dynamics
        for node_id in range(7):
            behavior = self.access_layer.get_node_property(node_id, "behavior")
            if behavior == "oscillator":
                self.graph = apply_oscillator_energy_dynamics(self.graph, node_id)
            elif behavior == "integrator":
                self.graph = apply_integrator_energy_dynamics(self.graph, node_id)
            elif behavior == "relay":
                self.graph = apply_relay_energy_dynamics(self.graph, node_id)
            elif behavior == "highway":
                self.graph = apply_highway_energy_dynamics(self.graph, node_id)

        # Measure total system energy after
        final_total = 0
        for node_id in range(1, 8):
            energy = self.access_layer.get_node_energy(node_id)
            if energy is not None:
                final_total += energy

        # Energy should be reasonably conserved (not change by more than 200%)
        energy_change_ratio = abs(final_total - initial_total) / max(initial_total, 0.001)
        self.assertLess(energy_change_ratio, 2.0, "Energy should be reasonably conserved")

    def test_energy_adaptation_mechanisms(self):
        """Test energy-based adaptation across the system."""
        # Set up nodes with different energy levels
        low_energy_node = 5  # Highway node with low energy
        high_energy_node = 2  # Oscillator node

        self.access_layer.set_node_energy(low_energy_node, 0.1)  # Very low
        self.access_layer.set_node_energy(high_energy_node, 2.0)  # High

        # Apply adaptation through dynamic behavior
        for node_id in [low_energy_node, high_energy_node]:
            self.graph = apply_dynamic_energy_dynamics(self.graph, node_id)

        # Check adaptation results
        low_plasticity = self.access_layer.get_node_property(low_energy_node, "plasticity_enabled")
        high_plasticity = self.access_layer.get_node_property(high_energy_node, "plasticity_enabled")

        # Low energy node should have plasticity disabled
        self.assertFalse(low_plasticity)

        # High energy node should maintain plasticity
        self.assertTrue(high_plasticity)

    def test_system_visualization_integration(self):
        """Test integration with system visualization."""
        diagram = EnergyFlowDiagram()

        # Create different diagram types
        diagram.create_system_architecture_diagram()
        diagram.create_energy_behavior_diagram()
        diagram.create_learning_integration_diagram()

        # Should have comprehensive system representation
        self.assertGreater(len(diagram.G.nodes()), 10)
        self.assertGreater(len(diagram.G.edges()), 5)

        # Should be able to analyze centrality
        centrality = diagram.create_energy_centrality_analysis()
        self.assertIsInstance(centrality, dict)
        self.assertIn("degree_centrality", centrality)

    def test_system_validation_integration(self):
        """Test integration with system validation."""
        validator = EnergySystemValidator()
        validator.test_graph = self.graph

        # Mock services to avoid full initialization
        with patch.object(validator, '_initialize_services'):
            validator.services = {
                'learning': Mock(),
                'energy': Mock(),
                'neural': Mock()
            }

            # Run validation
            report = validator.validate_energy_as_central_integrator()

            # Should produce validation report
            self.assertIsInstance(report, dict)
            self.assertIn('validation_summary', report)
            self.assertIn('conclusion', report)

    def test_constants_integration(self):
        """Test that constants work correctly across the system."""
        print("Starting test_constants_integration")
        # Test energy cap consistency
        print("Getting calculator_cap")
        calculator_cap = EnergyCalculator.calculate_energy_cap()
        print(f"calculator_cap: {calculator_cap}")
        print("Getting function_cap")
        function_cap = get_node_energy_cap()
        print(f"function_cap: {function_cap}")

        print("Asserting equal")
        self.assertEqual(calculator_cap, function_cap)
        print("Assert passed")

        # Test threshold relationships
        activation_threshold = EnergyConstants.get_activation_threshold()
        self.assertGreater(activation_threshold, 0)
        self.assertLessEqual(activation_threshold, 1)

        # Test refractory periods hierarchy
        short = EnergyConstants.REFRACTORY_PERIOD_SHORT
        medium = EnergyConstants.REFRACTORY_PERIOD_MEDIUM
        long = EnergyConstants.REFRACTORY_PERIOD_LONG

        self.assertLess(short, medium)
        self.assertLess(medium, long)

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test with invalid node ID
        invalid_result = self.access_layer.get_node_energy(999)
        self.assertIsNone(invalid_result)

        # Test with invalid behavior application
        try:
            self.graph = apply_oscillator_energy_dynamics(self.graph, 999)
            # Should not crash
        except:
            self.fail("apply_oscillator_energy_dynamics should handle invalid node IDs gracefully")

        # Test visualization with empty diagram
        diagram = EnergyFlowDiagram()
        centrality = diagram.create_energy_centrality_analysis()
        self.assertIsInstance(centrality, dict)  # Should handle empty gracefully


class TestEnergySystemPerformanceIntegration(unittest.TestCase):
    """Performance integration tests for the energy system."""

    def setUp(self):
        """Set up performance test fixtures."""
        # Create larger test system
        self.large_graph = Data()
        self.large_graph.node_labels = []
        self.large_graph.x = torch.empty((0, 1), dtype=torch.float32)

        num_nodes = 100
        self.id_manager = get_id_manager()
        self.id_manager.reset()

        for i in range(num_nodes):
            behavior = ['oscillator', 'integrator', 'relay', 'highway', 'dynamic'][i % 5]
            node_id = self.id_manager.generate_unique_id("dynamic")
            self.id_manager.register_node_index(node_id, i)

            node = {
                "id": node_id,
                "behavior": behavior,
                "energy": 1.0,
                "threshold": 0.5
            }
            self.large_graph.node_labels.append(node)
            self.large_graph.x = torch.cat([self.large_graph.x, torch.tensor([[1.0]], dtype=torch.float32)], dim=0)

        self.access_layer = NodeAccessLayer(self.large_graph, self.id_manager)

    def test_large_system_energy_processing_performance(self):
        """Test performance of energy processing on large system."""
        start_time = time.time()

        # Apply energy behavior to all nodes
        self.large_graph = apply_energy_behavior(self.large_graph)

        # Apply specific behaviors
        for i in range(100):
            behavior = self.large_graph.node_labels[i]['behavior']
            if behavior == 'oscillator':
                self.large_graph = apply_oscillator_energy_dynamics(self.large_graph, i)
            elif behavior == 'integrator':
                self.large_graph = apply_integrator_energy_dynamics(self.large_graph, i)

        processing_time = time.time() - start_time

        # Should process 100 nodes in reasonable time (< 1 second)
        self.assertLess(processing_time, 1.0)

    def test_large_system_statistics_performance(self):
        """Test performance of statistics calculation on large system."""
        start_time = time.time()
        stats = self.access_layer.get_node_statistics()
        stats_time = time.time() - start_time

        # Should calculate quickly (< 0.1 seconds)
        self.assertLess(stats_time, 0.1)
        self.assertEqual(stats['total_nodes'], 100)

    def test_large_system_validation_performance(self):
        """Test performance of validation on large system."""
        validator = EnergySystemValidator()
        validator.test_graph = self.large_graph

        start_time = time.time()

        with patch.object(validator, '_initialize_services'):
            validator.services = {'learning': Mock(), 'energy': Mock(), 'neural': Mock()}
            report = validator.validate_energy_as_central_integrator()

        validation_time = time.time() - start_time

        # Should validate quickly (< 2 seconds)
        self.assertLess(validation_time, 2.0)
        self.assertIsInstance(report, dict)


class TestEnergySystemRealWorldScenarios(unittest.TestCase):
    """Real-world scenario tests for the integrated energy system."""

    def setUp(self):
        """Set up real-world test fixtures."""
        self.id_manager = get_id_manager()
        self.id_manager.reset()

    def test_sensory_processing_pipeline(self):
        """Test sensory input processing through energy system."""
        # Create sensory → processing → output pipeline
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)
        edges = []

        # Sensory layer (2 nodes)
        for i in range(2):
            node_id = self.id_manager.generate_unique_id("sensory")
            self.id_manager.register_node_index(node_id, i)
            node = {"id": node_id, "type": "sensory", "behavior": "sensory", "energy": 0.1}
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[0.1]], dtype=torch.float32)], dim=0)

        # Processing layer (3 nodes)
        for i in range(2, 5):
            node_id = self.id_manager.generate_unique_id("dynamic")
            self.id_manager.register_node_index(node_id, i)
            behavior = ["oscillator", "integrator", "relay"][i-2]
            node = {"id": node_id, "type": "dynamic", "behavior": behavior, "energy": 0.5}
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[0.5]], dtype=torch.float32)], dim=0)
            # Connect sensory to processing
            edges.extend([[0, i], [1, i]])

        # Output layer (1 node)
        node_id = self.id_manager.generate_unique_id("dynamic")
        self.id_manager.register_node_index(node_id, 5)
        node = {"id": node_id, "type": "dynamic", "behavior": "dynamic", "energy": 0.3}
        graph.node_labels.append(node)
        graph.x = torch.cat([graph.x, torch.tensor([[0.3]], dtype=torch.float32)], dim=0)
        # Connect processing to output
        edges.extend([[2, 5], [3, 5], [4, 5]])

        graph.edge_index = torch.tensor(edges, dtype=torch.long).t()

        access_layer = NodeAccessLayer(graph, self.id_manager)

        # Simulate sensory input
        sensory_ids = access_layer.select_nodes_by_type("sensory")
        for node_id in sensory_ids:
            access_layer.set_node_energy(node_id, 1.2)  # Strong input

        # Process through pipeline
        graph = emit_energy_pulse(graph, sensory_ids[0])  # Pulse from first sensory

        # Apply processing behaviors
        for node_id in [2, 3, 4]:  # Processing nodes
            behavior = access_layer.get_node_property(node_id, "behavior")
            if behavior == "oscillator":
                graph = apply_oscillator_energy_dynamics(graph, node_id)
            elif behavior == "integrator":
                graph = apply_integrator_energy_dynamics(graph, node_id)
            elif behavior == "relay":
                graph = apply_relay_energy_dynamics(graph, node_id)

        # Check output activation
        output_energy = access_layer.get_node_energy(5)
        self.assertIsNotNone(output_energy)

        # Output should have received energy through the pipeline
        initial_output = 0.3
        self.assertNotEqual(output_energy, initial_output)

    def test_learning_adaptation_scenario(self):
        """Test learning and adaptation in energy system."""
        # Create teacher-student learning scenario
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)
        edges = []

        # Teacher node (oscillator)
        teacher_id = self.id_manager.generate_unique_id("dynamic")
        self.id_manager.register_node_index(teacher_id, 0)
        teacher = {"id": teacher_id, "behavior": "oscillator", "energy": 2.0,
                  "oscillation_freq": 2.0, "last_activation": 0, "refractory_timer": 0}
        graph.node_labels.append(teacher)
        graph.x = torch.cat([graph.x, torch.tensor([[2.0]], dtype=torch.float32)], dim=0)

        # Student nodes (integrators)
        student_ids = []
        for i in range(1, 4):
            student_id = self.id_manager.generate_unique_id("dynamic")
            self.id_manager.register_node_index(student_id, i)
            student = {"id": student_id, "behavior": "integrator", "energy": 0.5,
                      "integration_rate": 0.3}
            graph.node_labels.append(student)
            graph.x = torch.cat([graph.x, torch.tensor([[0.5]], dtype=torch.float32)], dim=0)
            edges.extend([[0, i]])  # Connect teacher to students
            student_ids.append(student_id)

        graph.edge_index = torch.tensor(edges, dtype=torch.long).t()

        access_layer = NodeAccessLayer(graph, self.id_manager)

        # Learning session
        for step in range(5):
            # Teacher oscillates
            graph = apply_oscillator_energy_dynamics(graph, teacher_id)

            # Students learn by integration
            for student_id in student_ids:
                graph = apply_integrator_energy_dynamics(graph, student_id)

            # Apply learning adaptation
            for student_id in student_ids:
                graph = update_node_energy_with_learning(graph, student_id, 0.1)

        # Check learning outcomes
        for student_id in student_ids:
            final_energy = access_layer.get_node_energy(student_id)
            self.assertGreater(final_energy, 0.5)  # Should have learned/increased energy

    def test_system_resilience_scenario(self):
        """Test system resilience under stress."""
        # Create system with redundant pathways
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty((0, 1), dtype=torch.float32)
        edges = []

        # Input nodes
        input_ids = []
        for i in range(3):
            node_id = self.id_manager.generate_unique_id("sensory")
            self.id_manager.register_node_index(node_id, i)
            node = {"id": node_id, "behavior": "sensory", "energy": 1.0}
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[1.0]], dtype=torch.float32)], dim=0)
            input_ids.append(node_id)

        # Processing nodes with redundancy
        processing_ids = []
        for i in range(3, 8):
            node_id = self.id_manager.generate_unique_id("dynamic")
            self.id_manager.register_node_index(node_id, i)
            behavior = ["oscillator", "integrator", "relay", "highway", "dynamic"][i-3]
            node = {"id": node_id, "behavior": behavior, "energy": 0.8}
            graph.node_labels.append(node)
            graph.x = torch.cat([graph.x, torch.tensor([[0.8]], dtype=torch.float32)], dim=0)
            processing_ids.append(node_id)

            # Connect inputs to processing with redundancy
            for input_id in input_ids:
                if (node_id - input_id) % 2 == 0:  # Some connections
                    edges.append([input_id, node_id])

        # Output node
        output_id = self.id_manager.generate_unique_id("dynamic")
        self.id_manager.register_node_index(output_id, 8)
        output = {"id": output_id, "behavior": "dynamic", "energy": 0.2}
        graph.node_labels.append(output)
        graph.x = torch.cat([graph.x, torch.tensor([[0.2]], dtype=torch.float32)], dim=0)

        # Connect processing to output
        for proc_id in processing_ids:
            edges.append([proc_id, output_id])

        graph.edge_index = torch.tensor(edges, dtype=torch.long).t()

        access_layer = NodeAccessLayer(graph, self.id_manager)

        # Simulate system stress: disable some processing nodes
        stressed_nodes = processing_ids[:2]  # Disable first 2
        for node_id in stressed_nodes:
            access_layer.set_node_energy(node_id, 0.01)  # Very low energy

        # Process through system
        for input_id in input_ids:
            graph = emit_energy_pulse(graph, input_id)

        # Apply behaviors to remaining nodes
        for node_id in processing_ids[2:]:  # Skip stressed nodes
            behavior = access_layer.get_node_property(node_id, "behavior")
            if behavior == "oscillator":
                graph = apply_oscillator_energy_dynamics(graph, node_id)
            elif behavior == "integrator":
                graph = apply_integrator_energy_dynamics(graph, node_id)
            elif behavior == "relay":
                graph = apply_relay_energy_dynamics(graph, node_id)
            elif behavior == "highway":
                graph = apply_highway_energy_dynamics(graph, node_id)

        # Check system resilience
        output_energy = access_layer.get_node_energy(output_id)
        self.assertGreater(output_energy, 0.2)  # Should still function despite stress

        # Verify stressed nodes remain low energy
        for node_id in stressed_nodes:
            energy = access_layer.get_node_energy(node_id)
            self.assertLess(energy, 0.1)


if __name__ == '__main__':
    unittest.main()






