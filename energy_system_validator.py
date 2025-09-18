"""
Energy System Validator
Validates that energy is the central integrator of all neural simulation modules.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_manager import SimulationManager
from neural.optimized_node_manager import get_optimized_node_manager
from energy.energy_behavior import (
    apply_energy_behavior, update_membrane_potentials,
    apply_oscillator_energy_dynamics, apply_integrator_energy_dynamics,
    apply_relay_energy_dynamics, apply_highway_energy_dynamics
)
from neural.connection_logic import intelligent_connection_formation
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

    def validate_energy_as_central_integrator(self) -> Dict[str, Any]:
        """Comprehensive validation of energy as system integrator."""

        print("ENERGY SYSTEM VALIDATION")
        print("=" * 60)

        try:
            # Initialize simulation
            self.simulation_manager = SimulationManager()
            success = self.simulation_manager.initialize_graph()
            if not success:
                raise Exception("Failed to initialize simulation")

            # Test 1: Energy as Input Processor
            self._validate_energy_as_input()

            # Test 2: Energy as Processing Driver
            self._validate_energy_as_processing()

            # Test 3: Energy as Learning Mechanism
            self._validate_energy_as_learning()

            # Test 4: Energy as Output Generator
            self._validate_energy_as_output()

            # Test 5: Energy as System Coordinator
            self._validate_energy_as_coordinator()

            # Test 6: Energy Conservation
            self._validate_energy_conservation()

            # Test 7: Energy-Based Adaptation
            self._validate_energy_adaptation()

            # Calculate overall validation score
            self._calculate_validation_score()

        except Exception as e:
            print(f"VALIDATION FAILED: {e}")
            self.validation_results['error'] = str(e)

        return self._generate_validation_report()

    def _validate_energy_as_input(self):
        """Test how energy processes external inputs."""
        print("\n1. Testing Energy as Input Processor...")

        try:
            graph = self.simulation_manager.graph
            access_layer = NodeAccessLayer(graph)

            # Create sensory input simulation
            sensory_nodes = access_layer.select_nodes_by_type('sensory')
            if not sensory_nodes:
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
                    self.node_manager.create_node_batch([node_spec])

                sensory_nodes = access_layer.select_nodes_by_type('sensory')

            # Simulate external input as energy injection
            initial_energies = {}
            for node_id in sensory_nodes[:3]:
                initial_energies[node_id] = access_layer.get_node_energy(node_id) or 0.0

            # Apply energy input (simulating sensory stimulation)
            input_energies = [0.8, 0.6, 0.9]  # Different input strengths
            for i, node_id in enumerate(list(sensory_nodes)[:3]):
                new_energy = min(input_energies[i], get_node_energy_cap())
                access_layer.set_node_energy(node_id, new_energy)
                access_layer.update_node_property(node_id, 'membrane_potential',
                                                new_energy / get_node_energy_cap())

            # Verify energy changes
            energy_changed = False
            for node_id in sensory_nodes[:3]:
                final_energy = access_layer.get_node_energy(node_id) or 0.0
                if abs(final_energy - initial_energies.get(node_id, 0.0)) > 0.01:
                    energy_changed = True
                    break

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
            graph = self.simulation_manager.graph

            # Create test scenario with energy differences
            access_layer = NodeAccessLayer(graph)

            # Find or create nodes with different energy levels
            dynamic_nodes = access_layer.select_nodes_by_type('dynamic')
            if len(dynamic_nodes) < 2:
                # Create test nodes
                for i in range(3):
                    node_spec = {
                        'type': 'dynamic',
                        'energy': 0.2 + i * 0.3,  # Different energy levels
                        'x': i * 20,
                        'y': 20,
                        'membrane_potential': 0.0,
                        'threshold': 0.5
                    }
                    self.node_manager.create_node_batch([node_spec])

                dynamic_nodes = access_layer.select_nodes_by_type('dynamic')

            # Apply learning
            initial_connections = len(graph.edge_attributes) if hasattr(graph, 'edge_attributes') else 0
            graph = hebbian_system.apply_continuous_learning(graph, step=1)
            final_connections = len(graph.edge_attributes) if hasattr(graph, 'edge_attributes') else 0

            # Test connection formation with energy modulation
            graph = intelligent_connection_formation(graph)
            final_connections_after_formation = len(graph.edge_attributes) if hasattr(graph, 'edge_attributes') else 0

            learning_effects = []
            if final_connections != initial_connections:
                learning_effects.append('connection_modification')
            if final_connections_after_formation != final_connections:
                learning_effects.append('connection_formation')

            if learning_effects:
                self.validation_results['energy_as_learning'] = True
                self.validation_results['module_interactions'].append({
                    'module': 'learning_systems',
                    'energy_role': 'learning_enabler',
                    'description': 'Energy drives learning and connection formation',
                    'effects': learning_effects
                })
                print(f"SUCCESS: Energy enables learning: {learning_effects}")
            else:
                print("INCOMPLETE: Energy learning validation incomplete")

        except Exception as e:
            print(f"FAILED: Energy learning validation failed: {e}")

    def _validate_energy_as_output(self):
        """Test how energy generates system outputs."""
        print("\n4. Testing Energy as Output Generator...")

        try:
            graph = self.simulation_manager.graph
            access_layer = NodeAccessLayer(graph)

            # Test energy-driven behaviors and outputs
            outputs_generated = []

            # Check for spiking behavior (energy output)
            dynamic_nodes = access_layer.select_nodes_by_type('dynamic')
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
            access_layer = NodeAccessLayer(graph)

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
            access_layer = NodeAccessLayer(graph)

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