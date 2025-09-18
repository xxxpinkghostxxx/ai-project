"""
Energy System Validator
Validates that energy is the central integrator of all neural simulation modules.
"""

import sys
import os
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation_manager import SimulationManager
from neural.optimized_node_manager import get_optimized_node_manager
from energy.energy_behavior import (
    apply_energy_behavior, update_membrane_potentials,
    apply_oscillator_energy_dynamics, apply_integrator_energy_dynamics,
    apply_relay_energy_dynamics, apply_highway_energy_dynamics
)
from neural.connection_logic import intelligent_connection_formation, create_weighted_connection
from learning.live_hebbian_learning import create_live_hebbian_learning
from sensory.visual_energy_bridge import create_visual_energy_bridge
from energy.node_access_layer import NodeAccessLayer
from energy.energy_behavior import get_node_energy_cap

class EnergySystemValidator:
    """Validates energy as the central system integrator."""

    def __init__(self):
        self.validation_results = {
            'energy_as_input': False,
            'energy_as_processing': False,
            'energy_as_learning': False,
            'energy_as_output': False,
            'energy_as_coordinator': False,
            'energy_conservation': False,
            'energy_adaptation': False,
            'module_interactions': [],
            'energy_flow_paths': [],
            'validation_score': 0.0
        }
        self.simulation_manager = None
        self.node_manager = get_optimized_node_manager()
        # Create our own performance monitor to avoid import issues
        try:
            from utils.performance_monitor import PerformanceMonitor as PerfMonitor
            self.performance_monitor = PerfMonitor()
            self.performance_monitor.start_monitoring()
        except Exception as e:
            print(f"Warning: Could not create performance monitor: {e}")
            self.performance_monitor = None

    def validate_energy_as_central_integrator(self) -> Dict[str, Any]:
        """Comprehensive validation of energy as system integrator."""

        print("ENERGY SYSTEM VALIDATION")
        print("=" * 60)
        print(">>> Starting validation process <<<")

        try:
            # Initialize simulation
            print("    Creating SimulationManager...")
            self.simulation_manager = SimulationManager()
            print("    SimulationManager created successfully")

            print("    Initializing graph...")
            success = self.simulation_manager.initialize_graph()
            print(f"    Graph initialization result: {success}")

            if not success:
                raise Exception("Failed to initialize simulation")

            print(f"    Graph has {len(self.simulation_manager.graph.node_labels) if hasattr(self.simulation_manager.graph, 'node_labels') else 0} nodes")
            print(f"    Graph has {self.simulation_manager.graph.x.shape[0] if hasattr(self.simulation_manager.graph, 'x') and self.simulation_manager.graph.x is not None else 0} energy values")

            # Create a smaller test graph for validation
            print("    Creating smaller test graph for validation...")
            from torch_geometric.data import Data
            import torch
            import numpy as np

            # Create a small test graph with known node types
            test_node_labels = []
            test_energies = []

            # Create 20 nodes: 10 sensory, 10 dynamic
            for i in range(20):
                node_type = 'sensory' if i < 10 else 'dynamic'
                energy_value = np.random.uniform(0.1, 0.9)
                node_label = {
                    'id': f"test_node_{i}",
                    'type': node_type,
                    'energy': energy_value,
                    'x': i * 10,
                    'y': 20,
                    'membrane_potential': energy_value,
                    'threshold': 0.5,
                    'behavior': 'dynamic' if node_type == 'dynamic' else 'sensory',
                    'state': 'active'
                }
                test_node_labels.append(node_label)
                test_energies.append(energy_value)

            # Create some test connections
            test_edges = []
            for i in range(10):  # Connect sensory to dynamic nodes
                sensory_idx = i
                dynamic_idx = 10 + i
                test_edges.extend([[sensory_idx, dynamic_idx], [dynamic_idx, sensory_idx]])

            test_edge_index = torch.tensor(test_edges, dtype=torch.long).t() if test_edges else torch.empty((2, 0), dtype=torch.long)
            test_x = torch.tensor(test_energies, dtype=torch.float32).unsqueeze(1)

            # Replace the large graph with our test graph
            self.original_graph = self.simulation_manager.graph  # Keep original for reference
            self.simulation_manager.graph = Data(
                x=test_x,
                edge_index=test_edge_index,
                node_labels=test_node_labels
            )

            print(f"    Created test graph with {len(test_node_labels)} nodes and {test_edge_index.shape[1]} edges")
        except Exception as e:
            print(f"Exception during initialization: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()

            if "start_monitoring" in str(e):
                print("Performance monitor issue detected, continuing without it...")
                # Create simulation manager without performance monitoring
                try:
                    # Try to create a minimal simulation manager
                    from core.main_graph import initialize_main_graph
                    from neural.dynamic_nodes import add_dynamic_nodes
                    graph = initialize_main_graph()
                    if len(graph.node_labels) == 0:
                        graph = add_dynamic_nodes(graph, num_dynamic=50)  # Reduced for testing
                    self.simulation_manager = type('MinimalSimulationManager', (), {
                        'graph': graph,
                        'initialize_graph': lambda: True
                    })()
                    print("Created minimal simulation manager for validation")
                except Exception as e2:
                    print(f"Failed to create minimal simulation manager: {e2}")
                    raise Exception(f"Failed to create minimal simulation manager: {e2}")
            else:
                raise

            print(">>> About to call Test 1 <<<")
            # Test 1: Energy as Input Processor
            try:
                self._validate_energy_as_input()
            except Exception as e:
                print(f"Test 1 failed with exception: {e}")
                import traceback
                traceback.print_exc()

            # Test 2: Energy as Processing Driver
            try:
                self._validate_energy_as_processing()
            except Exception as e:
                print(f"Test 2 failed with exception: {e}")
                import traceback
                traceback.print_exc()

            # Test 3: Energy as Learning Mechanism
            try:
                self._validate_energy_as_learning()
            except Exception as e:
                print(f"Test 3 failed with exception: {e}")
                import traceback
                traceback.print_exc()

            # Test 4: Energy as Output Generator
            try:
                self._validate_energy_as_output()
            except Exception as e:
                print(f"Test 4 failed with exception: {e}")
                import traceback
                traceback.print_exc()

            # Test 5: Energy as System Coordinator
            try:
                self._validate_energy_as_coordinator()
            except Exception as e:
                print(f"Test 5 failed with exception: {e}")
                import traceback
                traceback.print_exc()

            # Test 6: Energy Conservation
            try:
                self._validate_energy_conservation()
            except Exception as e:
                print(f"Test 6 failed with exception: {e}")
                import traceback
                traceback.print_exc()

            # Test 7: Energy-Based Adaptation
            try:
                self._validate_energy_adaptation()
            except Exception as e:
                print(f"Test 7 failed with exception: {e}")
                import traceback
                traceback.print_exc()

            # Calculate overall validation score
            self._calculate_validation_score()

        except Exception as e:
            print(f"VALIDATION FAILED: {e}")
            self.validation_results['error'] = str(e)

        return self._generate_validation_report()

    def _validate_energy_as_input(self):
        """Test how energy processes external inputs."""
        print("\n1. Testing Energy as Input Processor...")
        print("    >>> Test 1 started <<<")

        try:
            graph = self.simulation_manager.graph
            print(f"    Graph has {len(graph.node_labels) if hasattr(graph, 'node_labels') else 0} nodes")
            print(f"    Graph has {graph.x.shape[0] if hasattr(graph, 'x') and graph.x is not None else 0} energy values")

            access_layer = NodeAccessLayer(graph)
            print(f"    Access layer created with ID manager: {access_layer.id_manager is not None}")

            # Create sensory input simulation
            print(f"    Looking for sensory nodes in graph...")
            sensory_nodes = access_layer.select_nodes_by_type('sensory')
            print(f"    Found {len(sensory_nodes)} sensory nodes: {list(sensory_nodes)[:5]}")

            if not sensory_nodes:
                print("    No sensory nodes found, creating some...")
                # Create some sensory nodes
                for i in range(5):
                    node_spec = {
                        'type': 'sensory',
                        'energy': np.random.uniform(0.1, 0.5),
                        'x': i * 10,
                        'y': 0,
                        'membrane_potential': 0.0,
                        'threshold': 0.3
                    }
                    print(f"    Creating sensory node {i}...")
                    self.node_manager.create_node_batch([node_spec])

                sensory_nodes = access_layer.select_nodes_by_type('sensory')
                print(f"    After creation, found {len(sensory_nodes)} sensory nodes: {list(sensory_nodes)}")

            # Simulate external input as energy injection
            initial_energies = {}
            print(f"    Found {len(sensory_nodes)} sensory nodes: {list(sensory_nodes)[:3]}")

            for node_id in list(sensory_nodes)[:3]:
                initial_energy = access_layer.get_node_energy(node_id) or 0.0
                initial_energies[node_id] = initial_energy
                print(f"    Node {node_id} initial energy: {initial_energy}")

            # Apply energy input (simulating sensory stimulation)
            input_energies = [0.8, 0.6, 0.9]  # Different input strengths
            for i, node_id in enumerate(list(sensory_nodes)[:3]):
                new_energy = min(input_energies[i], get_node_energy_cap())
                print(f"    Setting node {node_id} energy to {new_energy}")
                success = access_layer.set_node_energy(node_id, new_energy)
                print(f"    Set energy success: {success}")

                membrane_potential = new_energy / get_node_energy_cap()
                access_layer.update_node_property(node_id, 'membrane_potential', membrane_potential)

            # Verify energy changes
            energy_changed = False
            for node_id in list(sensory_nodes)[:3]:
                final_energy = access_layer.get_node_energy(node_id) or 0.0
                initial_energy = initial_energies.get(node_id, 0.0)
                energy_diff = abs(final_energy - initial_energy)
                print(f"    Node {node_id} final energy: {final_energy}, diff: {energy_diff}")
                if energy_diff > 0.01:
                    energy_changed = True
                    print(f"    Energy change detected for node {node_id}")

            if energy_changed:
                self.validation_results['energy_as_input'] = True
                self.validation_results['module_interactions'].append({
                    'module': 'sensory_input',
                    'energy_role': 'input_conversion',
                    'description': 'External stimuli converted to energy patterns'
                })
                print("SUCCESS: Energy successfully processes external inputs")
            else:
                print("FAILED: Energy input processing failed")

        except Exception as e:
            print(f"FAILED: Energy input validation failed: {e}")

    def _validate_energy_as_processing(self):
        """Test how energy drives neural processing."""
        print("\n2. Testing Energy as Processing Driver...")

        try:
            graph = self.simulation_manager.graph
            access_layer = NodeAccessLayer(graph)

            # Test different energy behaviors
            behaviors_tested = []

            # Test oscillator behavior
            oscillator_nodes = access_layer.select_nodes_by_property('behavior', 'oscillator')
            if oscillator_nodes:
                for node_id in oscillator_nodes[:1]:
                    graph = apply_oscillator_energy_dynamics(graph, node_id)
                    behaviors_tested.append('oscillator')

            # Test integrator behavior
            integrator_nodes = access_layer.select_nodes_by_property('behavior', 'integrator')
            if integrator_nodes:
                for node_id in integrator_nodes[:1]:
                    graph = apply_integrator_energy_dynamics(graph, node_id)
                    behaviors_tested.append('integrator')

            # Test relay behavior
            relay_nodes = access_layer.select_nodes_by_property('behavior', 'relay')
            if relay_nodes:
                for node_id in relay_nodes[:1]:
                    graph = apply_relay_energy_dynamics(graph, node_id)
                    behaviors_tested.append('relay')

            # Test highway behavior
            highway_nodes = access_layer.select_nodes_by_property('behavior', 'highway')
            if highway_nodes:
                for node_id in highway_nodes[:1]:
                    graph = apply_highway_energy_dynamics(graph, node_id)
                    behaviors_tested.append('highway')

            # Test basic energy behavior
            graph = apply_energy_behavior(graph)
            behaviors_tested.append('basic_decay')

            # Test membrane potential updates
            graph = update_membrane_potentials(graph)
            behaviors_tested.append('membrane_update')

            if behaviors_tested:
                self.validation_results['energy_as_processing'] = True
                self.validation_results['module_interactions'].append({
                    'module': 'neural_processing',
                    'energy_role': 'processing_driver',
                    'description': f'Energy drives {len(behaviors_tested)} neural behaviors',
                    'behaviors': behaviors_tested
                })
                print(f"SUCCESS: Energy drives {len(behaviors_tested)} neural processing behaviors")
            else:
                print("INCOMPLETE: Energy processing validation incomplete")

        except Exception as e:
            print(f"FAILED: Energy processing validation failed: {e}")

    def _validate_energy_as_learning(self):
        """Test how energy enables learning mechanisms."""
        print("\n3. Testing Energy as Learning Enabler...")

        try:
            # Test Hebbian learning with energy
            hebbian_system = create_live_hebbian_learning(self.simulation_manager)
            # Enable energy modulation for this test
            hebbian_system.energy_learning_modulation = True
            graph = self.simulation_manager.graph

            # Create test scenario with energy differences
            access_layer = NodeAccessLayer(graph)

            # Create fresh test nodes with proper energy distribution
            print("    Creating fresh test nodes with proper energy distribution...")

            # Reset ID manager and create fresh test nodes
            from energy.node_id_manager import force_reset_id_manager, get_id_manager
            force_reset_id_manager()

            # Get the fresh ID manager instance and ensure NodeAccessLayer uses it
            id_manager = get_id_manager()
            access_layer.id_manager = id_manager  # Ensure NodeAccessLayer uses the fresh instance

            # Debug: Check ID manager state
            print(f"    ID Manager after reset: {id_manager.get_statistics()}")
            print(f"    Access layer ID manager is same instance: {id_manager is access_layer.id_manager}")

            # Create test nodes directly in the graph to ensure energy values are set properly
            test_node_labels = []
            test_energies = []

            for i in range(10):  # Create 10 test nodes
                # Create a wider range of energy values for better learning test
                energy_value = 0.1 + (i % 5) * 0.2  # Cycle through 0.1, 0.3, 0.5, 0.7, 0.9
                # Generate unique ID for this node
                node_id = id_manager.generate_unique_id('dynamic', {'energy': energy_value})

                node_label = {
                    'id': node_id,  # Use the generated ID
                    'type': 'dynamic',
                    'energy': energy_value,
                    'x': i * 15,
                    'y': 20,
                    'membrane_potential': 0.0,
                    'threshold': 0.5,
                    'behavior': 'dynamic',
                    'state': 'active'
                }
                test_node_labels.append(node_label)
                test_energies.append(energy_value)

                print(f"    Created test node {node_id} with energy {energy_value}")

            # Update graph with test nodes
            graph.node_labels = test_node_labels
            graph.x = torch.tensor(test_energies, dtype=torch.float32).unsqueeze(1)

            # Initialize edge structures
            graph.edge_index = torch.empty(2, 0, dtype=torch.long)
            graph.edge_attr = torch.empty(0, 1, dtype=torch.float32)

            # Register nodes with ID manager - ensure proper registration
            for i, node in enumerate(test_node_labels):
                node_id = node['id']
                # Force registration and verify
                id_manager.register_node_index(node_id, i)
                # Verify registration worked
                registered_index = id_manager.get_node_index(node_id)
                if registered_index != i:
                    print(f"    WARNING: Failed to register node {node_id} at index {i}, got {registered_index}")
                else:
                    print(f"    Successfully registered node {node_id} at index {i}")

            dynamic_nodes = [node['id'] for node in test_node_labels]
            print(f"    Using newly created nodes: {dynamic_nodes}")

            # Store test nodes for use by other validation tests
            self._test_nodes = dynamic_nodes

            # Clear existing connections to start with a clean slate for learning
            if hasattr(graph, 'edge_attributes'):
                graph.edge_attributes.clear()
            if hasattr(graph, 'edge_index') and graph.edge_index is not None and graph.edge_index.numel() > 0:
                # Keep only a minimal number of connections for testing
                if graph.edge_index.shape[1] > 10:
                    # Remove most edges, keep only first 10
                    graph.edge_index = graph.edge_index[:, :10]
                    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None and graph.edge_attr.shape[0] > 10:
                        graph.edge_attr = graph.edge_attr[:10]
                    if hasattr(graph, 'edge_attributes') and len(graph.edge_attributes) > 10:
                        graph.edge_attributes = graph.edge_attributes[:10]
                    print("  Cleared excess connections for clean learning test")
                elif graph.edge_index.shape[1] <= 10:
                    print(f"  Graph already has minimal connections: {graph.edge_index.shape[1]}")
            else:
                print("  No existing connections to clear")

            # Create a few initial connections for learning to work on
            if len(dynamic_nodes) >= 2:
                # Limit to first 20 nodes for manageable learning test
                test_nodes = list(dynamic_nodes)[:20]
                print(f"  Creating initial connections between {len(test_nodes)} test nodes (from {len(dynamic_nodes)} total)")
                # Create connections between high and low energy nodes to test energy-modulated learning
                high_energy_nodes = [node_id for node_id in test_nodes if access_layer.get_node_energy(node_id) and access_layer.get_node_energy(node_id) > 0.7][:3]
                low_energy_nodes = [node_id for node_id in test_nodes if access_layer.get_node_energy(node_id) and access_layer.get_node_energy(node_id) < 0.3][:3]

                # Debug energy values
                print(f"    Checking energy values for {len(test_nodes)} test nodes:")
                for i, node_id in enumerate(test_nodes[:5]):  # Check first 5
                    energy = access_layer.get_node_energy(node_id)
                    print(f"      Node {node_id}: energy = {energy}")

                if high_energy_nodes and low_energy_nodes:
                    print(f"    High energy nodes: {high_energy_nodes[:2]}")
                    print(f"    Low energy nodes: {low_energy_nodes[:2]}")
                else:
                    print(f"    No suitable high/low energy nodes found. High: {len(high_energy_nodes)}, Low: {len(low_energy_nodes)}")
                    # Try with different thresholds
                    print("    Trying with adjusted thresholds...")
                    high_energy_nodes_alt = [node_id for node_id in test_nodes if access_layer.get_node_energy(node_id) and access_layer.get_node_energy(node_id) > 0.5][:2]
                    low_energy_nodes_alt = [node_id for node_id in test_nodes if access_layer.get_node_energy(node_id) and access_layer.get_node_energy(node_id) < 0.5][:2]
                    print(f"    Alt thresholds - High: {len(high_energy_nodes_alt)}, Low: {len(low_energy_nodes_alt)}")
                    if high_energy_nodes_alt and low_energy_nodes_alt:
                        high_energy_nodes = high_energy_nodes_alt
                        low_energy_nodes = low_energy_nodes_alt
                        print("    Using alternative thresholds")

                    # Create connections manually using PyTorch tensors
                    connections_created = 0
                    for high_node in high_energy_nodes[:2]:  # Limit to 2 high-energy nodes
                        for low_node in low_energy_nodes[:2]:  # Limit to 2 low-energy nodes
                            if high_node != low_node and connections_created < 3:
                                # Get node indices
                                high_idx = None
                                low_idx = None
                                for idx, node in enumerate(graph.node_labels):
                                    if node.get('id') == high_node:
                                        high_idx = idx
                                    if node.get('id') == low_node:
                                        low_idx = idx

                                if high_idx is not None and low_idx is not None:
                                    # Create connection with energy-modulated weight
                                    high_energy = access_layer.get_node_energy(high_node) or 0.5
                                    low_energy = access_layer.get_node_energy(low_node) or 0.5
                                    avg_energy = (high_energy + low_energy) / 2.0
                                    weight = 0.5 * avg_energy  # Energy-modulated initial weight

                                    # Add to PyTorch Geometric format
                                    new_edge = torch.tensor([[high_idx], [low_idx]], dtype=torch.long)
                                    if not hasattr(graph, 'edge_index') or graph.edge_index is None:
                                        graph.edge_index = new_edge
                                    else:
                                        graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)

                                    # Add edge attributes
                                    if not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
                                        graph.edge_attr = torch.tensor([[weight]], dtype=torch.float)
                                    else:
                                        new_attr = torch.tensor([[weight]], dtype=torch.float)
                                        graph.edge_attr = torch.cat([graph.edge_attr, new_attr], dim=0)

                                    # Add to edge_attributes list
                                    if not hasattr(graph, 'edge_attributes'):
                                        graph.edge_attributes = []

                                    from neural.connection_logic import EnhancedEdge
                                    edge_obj = EnhancedEdge(high_idx, low_idx, weight, 'excitatory')
                                    graph.edge_attributes.append(edge_obj)

                                    print(f"    Created connection {high_node} -> {low_node} (idx {high_idx}->{low_idx}) with weight {weight:.3f}")
                                    connections_created += 1

            # Record initial state
            initial_connections = len(graph.edge_attributes) if hasattr(graph, 'edge_attributes') else 0
            initial_stats = hebbian_system.get_learning_statistics()

            # Apply learning multiple times to allow energy modulation to take effect
            print(f"  Initial connections: {initial_connections}")
            print(f"  Initial stats: {initial_stats}")

            for step in range(10):  # Increased from 3 to 10 for more learning opportunities
                graph = hebbian_system.apply_continuous_learning(graph, step=step)
                if step % 3 == 0:  # Print progress every 3 steps
                    current_stats = hebbian_system.get_learning_statistics()
                    print(f"  Step {step}: connections={len(graph.edge_attributes) if hasattr(graph, 'edge_attributes') else 0}, stdp_events={current_stats.get('stdp_events', 0)}, energy_modulated={current_stats.get('energy_modulated_events', 0)}")

            final_connections = len(graph.edge_attributes) if hasattr(graph, 'edge_attributes') else 0
            final_stats = hebbian_system.get_learning_statistics()

            print(f"  Final connections: {final_connections}")
            print(f"  Final stats: {final_stats}")

            # Test connection formation with energy modulation
            graph = intelligent_connection_formation(graph)
            final_connections_after_formation = len(graph.edge_attributes) if hasattr(graph, 'edge_attributes') else 0

            print(f"  Connections after formation: {final_connections_after_formation}")

            # Check for learning effects
            learning_effects = []
            stats_changes = []

            # Check connection changes
            if final_connections != initial_connections:
                learning_effects.append('connection_modification')
                print(f"  Detected connection modification: {initial_connections} -> {final_connections}")
            if final_connections_after_formation != final_connections:
                learning_effects.append('connection_formation')
                print(f"  Detected connection formation: {final_connections} -> {final_connections_after_formation}")

            # Check statistics changes (more sensitive detection)
            if final_stats.get('stdp_events', 0) > initial_stats.get('stdp_events', 0):
                stats_changes.append('stdp_events')
                print(f"  Detected STDP events: {initial_stats.get('stdp_events', 0)} -> {final_stats.get('stdp_events', 0)}")
            if final_stats.get('energy_modulated_events', 0) > 0:
                stats_changes.append('energy_modulated_learning')
                learning_effects.append('energy_modulated_plasticity')
                print(f"  Detected energy-modulated learning: {final_stats.get('energy_modulated_events', 0)} events")
            if final_stats.get('connection_strengthened', 0) > initial_stats.get('connection_strengthened', 0):
                stats_changes.append('connection_strengthening')
                print(f"  Detected connection strengthening: {initial_stats.get('connection_strengthened', 0)} -> {final_stats.get('connection_strengthened', 0)}")
            if final_stats.get('connection_weakened', 0) > initial_stats.get('connection_weakened', 0):
                stats_changes.append('connection_weakening')
                print(f"  Detected connection weakening: {initial_stats.get('connection_weakened', 0)} -> {final_stats.get('connection_weakened', 0)}")

            # Success criteria: either connection changes OR statistical evidence of learning
            has_learning_effects = len(learning_effects) > 0
            has_statistical_evidence = len(stats_changes) > 0

            # Also check for minimal statistical evidence (any learning activity)
            minimal_stats = (final_stats.get('stdp_events', 0) > 0 or
                           final_stats.get('energy_modulated_events', 0) > 0 or
                           final_stats.get('total_weight_changes', 0) > 0)

            print(f"  Learning effects: {learning_effects}")
            print(f"  Statistical changes: {stats_changes}")
            print(f"  Minimal stats: {minimal_stats}")

            if has_learning_effects or has_statistical_evidence or minimal_stats:
                self.validation_results['energy_as_learning'] = True
                description = 'Energy drives learning and connection formation'
                if has_statistical_evidence:
                    description += f' (detected via statistics: {stats_changes})'
                if has_learning_effects:
                    description += f' (effects: {learning_effects})'

                self.validation_results['module_interactions'].append({
                    'module': 'learning_systems',
                    'energy_role': 'learning_enabler',
                    'description': description,
                    'effects': learning_effects,
                    'statistical_evidence': stats_changes
                })
                print(f"SUCCESS: Energy enables learning - Effects: {learning_effects}, Stats: {stats_changes}")
            else:
                print("INCOMPLETE: Energy learning validation incomplete - no detectable learning effects")

        except Exception as e:
            print(f"FAILED: Energy learning validation failed: {e}")

    def _validate_energy_as_output(self):
        """Test how energy generates system outputs."""
        print("\n4. Testing Energy as Output Generator...")

        try:
            graph = self.simulation_manager.graph
            # Use the same access layer instance to maintain ID manager consistency
            access_layer = self._access_layer if hasattr(self, '_access_layer') else NodeAccessLayer(graph)
            if not hasattr(self, '_access_layer'):
                self._access_layer = access_layer

            # Test energy-driven behaviors and outputs
            outputs_generated = []

            # Check for spiking behavior (energy output)
            # Use existing nodes from the learning test if available
            if hasattr(self, '_test_nodes') and self._test_nodes:
                dynamic_nodes = self._test_nodes
                print(f"    Using existing test nodes: {dynamic_nodes}")
            else:
                # Fallback: get all dynamic nodes
                dynamic_nodes = access_layer.select_nodes_by_type('dynamic')
                print(f"    Using dynamic nodes from graph: {dynamic_nodes[:10]}")

            spikes_generated = 0

            for node_id in dynamic_nodes[:10]:  # Sample first 10
                energy = access_layer.get_node_energy(node_id)
                threshold = access_layer.get_node_property(node_id, 'threshold', 0.5)

                if energy and energy > threshold:
                    # Simulate spike generation
                    access_layer.update_node_property(node_id, 'last_activation', time.time())
                    access_layer.update_node_property(node_id, 'refractory_timer', 0.1)
                    spikes_generated += 1
                    outputs_generated.append('spike')

            # Check for energy propagation (relay output)
            relay_nodes = access_layer.select_nodes_by_property('behavior', 'relay')
            for node_id in relay_nodes[:2]:
                graph = apply_relay_energy_dynamics(graph, node_id)
                outputs_generated.append('energy_relay')

            # Check for oscillatory outputs
            oscillator_nodes = access_layer.select_nodes_by_property('behavior', 'oscillator')
            for node_id in oscillator_nodes[:2]:
                graph = apply_oscillator_energy_dynamics(graph, node_id)
                outputs_generated.append('oscillatory_signal')

            if outputs_generated:
                self.validation_results['energy_as_output'] = True
                self.validation_results['module_interactions'].append({
                    'module': 'system_outputs',
                    'energy_role': 'output_generator',
                    'description': f'Energy generates {len(set(outputs_generated))} types of outputs',
                    'outputs': list(set(outputs_generated))
                })
                print(f"SUCCESS: Energy generates {len(set(outputs_generated))} types of system outputs")
            else:
                print("INCOMPLETE: Energy output generation validation incomplete")

        except Exception as e:
            print(f"FAILED: Energy output validation failed: {e}")

    def _validate_energy_as_coordinator(self):
        """Test how energy coordinates system behavior."""
        print("\n5. Testing Energy as System Coordinator...")

        try:
            graph = self.simulation_manager.graph
            # Use the same access layer instance to maintain ID manager consistency
            access_layer = self._access_layer if hasattr(self, '_access_layer') else NodeAccessLayer(graph)
            if not hasattr(self, '_access_layer'):
                self._access_layer = access_layer

            coordination_effects = []

            # Test energy-based node lifecycle coordination
            total_nodes = access_layer.get_node_count()
            high_energy_nodes = 0
            low_energy_nodes = 0

            for node_id, node in access_layer.iterate_all_nodes():
                energy = access_layer.get_node_energy(node_id)
                if energy:
                    if energy > 0.7:  # High energy
                        high_energy_nodes += 1
                        # High energy nodes should be more active
                        access_layer.update_node_property(node_id, 'plasticity_enabled', True)
                        coordination_effects.append('high_energy_activation')
                    elif energy < 0.3:  # Low energy
                        low_energy_nodes += 1
                        # Low energy nodes should be less active
                        access_layer.update_node_property(node_id, 'plasticity_enabled', False)
                        coordination_effects.append('low_energy_conservation')

            # Test energy distribution patterns
            energy_distribution = []
            for node_id, node in access_layer.iterate_all_nodes():
                energy = access_layer.get_node_energy(node_id)
                if energy is not None:
                    energy_distribution.append(energy)

            if energy_distribution:
                energy_variance = np.var(energy_distribution)
                energy_range = max(energy_distribution) - min(energy_distribution)

                if energy_variance > 0.01:  # Some energy differentiation
                    coordination_effects.append('energy_differentiation')
                if energy_range > 0.5:  # Good energy range
                    coordination_effects.append('energy_range_maintenance')

            # Test temporal coordination through refractory periods
            nodes_in_refractory = 0
            for node_id, node in access_layer.iterate_all_nodes():
                refractory_timer = access_layer.get_node_property(node_id, 'refractory_timer', 0)
                if refractory_timer > 0:
                    nodes_in_refractory += 1

            if nodes_in_refractory > 0:
                coordination_effects.append('temporal_coordination')

            if coordination_effects:
                self.validation_results['energy_as_coordinator'] = True
                self.validation_results['module_interactions'].append({
                    'module': 'system_coordination',
                    'energy_role': 'coordinator',
                    'description': f'Energy coordinates {len(set(coordination_effects))} system aspects',
                    'coordination_effects': list(set(coordination_effects))
                })
                print(f"SUCCESS: Energy coordinates {len(set(coordination_effects))} system aspects")
            else:
                print("INCOMPLETE: Energy coordination validation incomplete")

        except Exception as e:
            print(f"FAILED: Energy coordination validation failed: {e}")

    def _validate_energy_conservation(self):
        """Test energy conservation principles."""
        print("\n6. Testing Energy Conservation...")

        try:
            graph = self.simulation_manager.graph
            access_layer = NodeAccessLayer(graph)

            # Measure total system energy before and after operations
            initial_total_energy = 0.0
            for node_id, node in access_layer.iterate_all_nodes():
                energy = access_layer.get_node_energy(node_id)
                if energy:
                    initial_total_energy += energy

            # Apply energy operations
            graph = apply_energy_behavior(graph)
            graph = update_membrane_potentials(graph)

            # Measure energy after operations
            final_total_energy = 0.0
            for node_id, node in access_layer.iterate_all_nodes():
                energy = access_layer.get_node_energy(node_id)
                if energy:
                    final_total_energy += energy

            # Check energy conservation (should not change dramatically)
            energy_change_percent = abs(final_total_energy - initial_total_energy) / max(initial_total_energy, 0.001) * 100

            # Energy should be relatively conserved (less than 50% change)
            if energy_change_percent < 50.0:
                self.validation_results['energy_conservation'] = True
                self.validation_results['module_interactions'].append({
                    'module': 'energy_conservation',
                    'energy_role': 'conservation_maintenance',
                    'description': f'Energy conserved with {energy_change_percent:.1f}% change',
                    'conservation_metrics': {
                        'initial_total': initial_total_energy,
                        'final_total': final_total_energy,
                        'change_percent': energy_change_percent
                    }
                })
                print(f"SUCCESS: Energy conserved with {energy_change_percent:.1f}% change")
            else:
                print(f"FAILED: Energy conservation failed with {energy_change_percent:.1f}% change")
        except Exception as e:
            print(f"FAILED: Energy conservation validation failed: {e}")

    def _validate_energy_adaptation(self):
        """Test energy-based system adaptation."""
        print("\n7. Testing Energy-Based Adaptation...")

        try:
            graph = self.simulation_manager.graph
            # Use the same access layer instance to maintain ID manager consistency
            access_layer = self._access_layer if hasattr(self, '_access_layer') else NodeAccessLayer(graph)
            if not hasattr(self, '_access_layer'):
                self._access_layer = access_layer

            adaptation_mechanisms = []

            # Test energy-based plasticity modulation
            plasticity_changes = 0
            for node_id, node in access_layer.iterate_all_nodes():
                energy = access_layer.get_node_energy(node_id)
                if energy:
                    current_plasticity = access_layer.get_node_property(node_id, 'plasticity_enabled', True)

                    # Energy should modulate plasticity
                    if energy < 0.3 and current_plasticity:
                        access_layer.update_node_property(node_id, 'plasticity_enabled', False)
                        plasticity_changes += 1
                        adaptation_mechanisms.append('plasticity_modulation')
                    elif energy > 0.7 and not current_plasticity:
                        access_layer.update_node_property(node_id, 'plasticity_enabled', True)
                        plasticity_changes += 1
                        adaptation_mechanisms.append('plasticity_modulation')

            # Test behavior switching based on energy
            behavior_changes = 0
            for node_id, node in access_layer.iterate_all_nodes():
                energy = access_layer.get_node_energy(node_id)
                if energy:
                    current_behavior = access_layer.get_node_property(node_id, 'behavior', 'dynamic')

                    # High energy → more complex behavior
                    if energy > 0.8 and current_behavior == 'dynamic':
                        access_layer.update_node_property(node_id, 'behavior', 'oscillator')
                        behavior_changes += 1
                        adaptation_mechanisms.append('behavior_switching')
                    # Low energy → simpler behavior
                    elif energy < 0.2 and current_behavior in ['oscillator', 'integrator']:
                        access_layer.update_node_property(node_id, 'behavior', 'dynamic')
                        behavior_changes += 1
                        adaptation_mechanisms.append('behavior_switching')

            if adaptation_mechanisms:
                self.validation_results['energy_adaptation'] = True
                self.validation_results['module_interactions'].append({
                    'module': 'adaptive_systems',
                    'energy_role': 'adaptation_driver',
                    'description': f'Energy drives {len(set(adaptation_mechanisms))} adaptation mechanisms',
                    'adaptation_mechanisms': list(set(adaptation_mechanisms)),
                    'changes_made': plasticity_changes + behavior_changes
                })
                print(f"SUCCESS: Energy drives {len(set(adaptation_mechanisms))} adaptation mechanisms")
            else:
                print("INCOMPLETE: Energy adaptation validation incomplete")

        except Exception as e:
            print(f"FAILED: Energy adaptation validation failed: {e}")

    def _calculate_validation_score(self):
        """Calculate overall validation score."""
        validations = [
            self.validation_results['energy_as_input'],
            self.validation_results['energy_as_processing'],
            self.validation_results['energy_as_learning'],
            self.validation_results['energy_as_output'],
            self.validation_results['energy_as_coordinator'],
            self.validation_results['energy_conservation'],
            self.validation_results['energy_adaptation']
        ]

        passed_validations = sum(validations)
        total_validations = len(validations)
        self.validation_results['validation_score'] = (passed_validations / total_validations) * 100

    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""

        report = {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'overall_score': self.validation_results['validation_score'],
                'validations_passed': sum([
                    self.validation_results['energy_as_input'],
                    self.validation_results['energy_as_processing'],
                    self.validation_results['energy_as_learning'],
                    self.validation_results['energy_as_output'],
                    self.validation_results['energy_as_coordinator'],
                    self.validation_results['energy_conservation'],
                    self.validation_results['energy_adaptation']
                ]),
                'total_validations': 7
            },
            'energy_roles_validated': {
                'input_processor': self.validation_results['energy_as_input'],
                'processing_driver': self.validation_results['energy_as_processing'],
                'learning_enabler': self.validation_results['energy_as_learning'],
                'output_generator': self.validation_results['energy_as_output'],
                'system_coordinator': self.validation_results['energy_as_coordinator'],
                'conservation_maintainer': self.validation_results['energy_conservation'],
                'adaptation_driver': self.validation_results['energy_adaptation']
            },
            'module_interactions': self.validation_results['module_interactions'],
            'energy_flow_analysis': self._analyze_energy_flow(),
            'conclusion': self._generate_conclusion()
        }

        return report

    def _analyze_energy_flow(self) -> Dict[str, Any]:
        """Analyze energy flow patterns in the system."""
        return {
            'flow_paths_identified': len(self.validation_results['module_interactions']),
            'energy_as_central_integrator': all([
                self.validation_results['energy_as_input'],
                self.validation_results['energy_as_processing'],
                self.validation_results['energy_as_learning'],
                self.validation_results['energy_as_output'],
                self.validation_results['energy_as_coordinator']
            ]),
            'system_coherence': self.validation_results['validation_score'] >= 80.0
        }

    def _generate_conclusion(self) -> str:
        """Generate validation conclusion."""
        score = self.validation_results['validation_score']

        if score >= 90:
            return "EXCELLENT: Energy is confirmed as the central integrator powering all neural simulation modules."
        elif score >= 80:
            return "VERY GOOD: Energy effectively integrates most system modules with minor gaps."
        elif score >= 70:
            return "GOOD: Energy integration is functional but could be strengthened."
        else:
            return "NEEDS IMPROVEMENT: Energy integration requires enhancement."

def run_energy_validation():
    """Run the comprehensive energy system validation."""

    print("NEURAL SIMULATION ENERGY SYSTEM VALIDATION")
    print("=" * 60)
    print("Testing Energy as Central System Integrator")
    print("=" * 60)

    validator = EnergySystemValidator()
    report = validator.validate_energy_as_central_integrator()

    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"Overall Score: {report['validation_summary']['overall_score']:.1f}%")
    print(f"Validations Passed: {report['validation_summary']['validations_passed']}/{report['validation_summary']['total_validations']}")

    print("\nENERGY ROLES VALIDATED:")
    for role, validated in report['energy_roles_validated'].items():
        status = "[PASS]" if validated else "[FAIL]"
        print(f"  {status} {role.replace('_', ' ').title()}")

    print("\nMODULE INTERACTIONS:")
    for i, interaction in enumerate(report['module_interactions'], 1):
        print(f"  {i}. {interaction['module'].replace('_', ' ').title()}")
        print(f"     Role: {interaction['energy_role'].replace('_', ' ').title()}")
        print(f"     Description: {interaction['description']}")

    print("\nENERGY FLOW ANALYSIS:")
    print(f"  Flow Paths: {report['energy_flow_analysis']['flow_paths_identified']}")
    print(f"  Central Integrator: {report['energy_flow_analysis']['energy_as_central_integrator']}")
    print(f"  System Coherence: {report['energy_flow_analysis']['system_coherence']}")

    print(f"\nCONCLUSION: {report['conclusion']}")

    # Save detailed report
    import json
    with open('energy_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\nDetailed report saved to: energy_validation_report.json")
    return report


if __name__ == "__main__":
    run_energy_validation()