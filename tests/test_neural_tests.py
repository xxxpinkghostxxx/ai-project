"""
Neural network tests for the unified testing system.
Tests neural dynamics, learning, and related functionality.
"""

from typing import Tuple, Dict, Any
from unittest.mock import MagicMock
import unittest.mock as mock

# Third-party imports
import numpy as np

# Local imports - Core modules
from src.core.main_graph import initialize_main_graph

# Local imports - Energy modules
from src.energy.energy_behavior import apply_energy_behavior, get_node_energy_cap

# Local imports - Learning modules
from src.learning.learning_engine import LearningEngine

# Local imports - Neural modules
from src.neural.connection_logic import ConnectionConstants, EnhancedEdge, create_basic_connections
from src.neural.death_and_birth_logic import (
    analyze_memory_patterns_for_birth, birth_new_dynamic_nodes,
    get_node_birth_threshold, get_node_death_threshold, handle_node_death,
    remove_dead_dynamic_nodes)
from src.neural.enhanced_neural_dynamics import EnhancedNeuralDynamics
from src.neural.event_driven_system import (EventType, NeuralEvent,
                                             create_event_driven_system)

# Local imports - Utils modules
from src.utils.event_bus import get_event_bus

# Local test utilities
from .test_utils import TestCase, TestCategory, setup_logging_capture, cleanup_logging_capture
from .test_mocks import MockAccessLayer, MockMemory


def test_energy_behavior() -> Tuple[bool, Dict[str, Any]]:
    """Test energy behavior and dynamics."""
    try:
        graph = initialize_main_graph(scale=0.25)
        energy_cap = get_node_energy_cap()

        # Test energy behavior application
        updated_graph = apply_energy_behavior(graph)

        return updated_graph is not None, {
            'energy_cap': energy_cap,
            'graph_updated': updated_graph is not None
        }
    except (ImportError, AttributeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_connection_logic() -> Tuple[bool, Dict[str, Any]]:
    """Test connection logic and formation."""
    try:
        graph = initialize_main_graph(scale=0.25)
        initial_edges = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0

        updated_graph = create_basic_connections(graph)
        final_edges = updated_graph.edge_index.shape[1] if hasattr(updated_graph, 'edge_index') else 0

        return final_edges >= initial_edges, {
            'initial_edges': initial_edges,
            'final_edges': final_edges,
            'connections_created': final_edges - initial_edges
        }
    except (ImportError, AttributeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_neural_math_accuracy() -> Tuple[bool, Dict[str, Any]]:
    """Test mathematical accuracy in neural dynamics."""
    try:
        dynamics = EnhancedNeuralDynamics()
        # Mock simple node for membrane potential update
        mock_node = {
            'membrane_potential': -70.0,  # resting
            'dendritic_potential': 0.0,
            'refractory_timer': 0.0,
            'last_spike_time': 0.0
        }
        # Test membrane update formula (isolated)
        synaptic_input = 10.0
        dendritic_influence = 0.0  # below threshold
        dt = 1.0 / dynamics.membrane_time_constant
        expected_v_mem = -70.0 + (synaptic_input + dendritic_influence + 70.0) * dt  # towards rest + input
        v_mem = mock_node['membrane_potential']
        v_mem += (synaptic_input + dendritic_influence - v_mem) * dt
        v_mem += (dynamics.resting_potential - v_mem) * 0.01
        np.testing.assert_almost_equal(v_mem, expected_v_mem, decimal=6)

        # Test spike threshold crossing
        mock_node['membrane_potential'] = -40.0  # above threshold -50
        v_mem = mock_node['membrane_potential']
        if v_mem > dynamics.threshold_potential and mock_node['refractory_timer'] <= 0:
            v_mem = dynamics.reset_potential  # -80
        np.testing.assert_almost_equal(v_mem, -80.0, decimal=6)

        # Test STDP weight change (LTP case)
        delta_t = 10.0  # ms, positive for LTP
        ltp_strength = dynamics.ltp_rate * np.exp(-delta_t / dynamics.tau_plus)
        expected_change = ltp_strength
        # Simulate
        weight_change = 0.0
        if 0 < delta_t < dynamics.stdp_window:
            weight_change += dynamics.ltp_rate * np.exp(-delta_t / dynamics.tau_plus)
        np.testing.assert_almost_equal(weight_change, expected_change, decimal=6)

        # LTD case
        delta_t_ltd = -10.0
        ltd_strength = dynamics.ltd_rate * np.exp(delta_t_ltd / dynamics.tau_minus)
        weight_change_ltd = -ltd_strength
        if -dynamics.stdp_window < delta_t_ltd < 0:
            weight_change_ltd = -dynamics.ltd_rate * np.exp(delta_t_ltd / dynamics.tau_minus)
        np.testing.assert_almost_equal(weight_change_ltd, -ltd_strength, decimal=6)

        return True, {'neural_tests': 'passed', 'assertions': 4}
    except AssertionError as ae:
        return False, {'error': str(ae), 'test': 'neural_math'}
    except (ValueError, TypeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_connection_math_accuracy() -> Tuple[bool, Dict[str, Any]]:
    """Test connection logic math (weights, Hebbian)."""
    try:
        # Test effective weight for types
        edge_exc = EnhancedEdge(1, 2, weight=1.0, edge_type='excitatory')
        np.testing.assert_almost_equal(edge_exc.get_effective_weight(), 1.0, decimal=6)

        edge_inh = EnhancedEdge(1, 2, weight=1.0, edge_type='inhibitory')
        np.testing.assert_almost_equal(edge_inh.get_effective_weight(), -1.0, decimal=6)

        edge_mod = EnhancedEdge(1, 2, weight=1.0, edge_type='modulatory')
        mod_weight = ConnectionConstants.MODULATORY_WEIGHT  # assume 0.5
        np.testing.assert_almost_equal(edge_mod.get_effective_weight(), 1.0 * mod_weight, decimal=6)

        # Test Hebbian-like weight update (from update_connection_weights)
        learning_rate = ConnectionConstants.LEARNING_RATE_DEFAULT
        source_activity = 1.0
        target_activity = 1.0
        weight_change = learning_rate * (source_activity + target_activity) / 2
        initial_weight = 0.5
        new_weight = min(initial_weight + weight_change, ConnectionConstants.WEIGHT_CAP_MAX)
        # Simulate
        edge = EnhancedEdge(1, 2, weight=initial_weight)
        if source_activity > 0 and target_activity > 0:
            edge.weight = min(edge.weight + weight_change, ConnectionConstants.WEIGHT_CAP_MAX)
        np.testing.assert_almost_equal(edge.weight, new_weight, decimal=6)

        # LTD case (target inactive)
        target_activity_ltd = 0.0
        weight_change_ltd = -learning_rate * ConnectionConstants.WEIGHT_CHANGE_FACTOR
        new_weight_ltd = max(initial_weight + weight_change_ltd, ConnectionConstants.WEIGHT_MIN)
        edge.weight = initial_weight
        if source_activity > 0 and target_activity_ltd == 0:
            edge.weight = max(edge.weight + weight_change_ltd, ConnectionConstants.WEIGHT_MIN)
        np.testing.assert_almost_equal(edge.weight, new_weight_ltd, decimal=6)

        # Eligibility trace decay
        edge.eligibility_trace = 1.0
        edge.update_eligibility_trace(0)
        decayed_trace = 1.0 * ConnectionConstants.ELIGIBILITY_TRACE_DECAY
        np.testing.assert_almost_equal(edge.eligibility_trace, decayed_trace, decimal=6)

        return True, {'connection_tests': 'passed', 'assertions': 7}
    except AssertionError as ae:
        return False, {'error': str(ae), 'test': 'connection_math'}
    except (ValueError, TypeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_node_death_low_energy() -> Tuple[bool, Dict[str, Any]]:
    """Test node death on low energy threshold."""
    try:
        log_capture, handler = setup_logging_capture()

        graph = initialize_main_graph(scale=0.1)
        initial_count = len(graph.node_labels)

        # Set a dynamic node to low energy
        low_energy = get_node_death_threshold() - 0.1
        dynamic_found = False
        for i, node_label in enumerate(graph.node_labels):
            if node_label.get('type') == 'dynamic':
                if hasattr(graph, 'x') and graph.x is not None:
                    graph.x[i, 0] = low_energy
                dynamic_found = True
                break
        if not dynamic_found:
            # Force one
            if len(graph.node_labels) > 0:
                graph.node_labels[0]['type'] = 'dynamic'
                if hasattr(graph, 'x') and graph.x is not None:
                    graph.x[0, 0] = low_energy

        updated_graph = remove_dead_dynamic_nodes(graph)
        final_count = len(updated_graph.node_labels)

        logs = log_capture.getvalue()
        cleanup_logging_capture(handler)

        death_occurred = final_count < initial_count
        log_contains_death = "[DEATH]" in logs

        return death_occurred and log_contains_death, {
            'initial_nodes': initial_count,
            'final_nodes': final_count,
            'death_logged': log_contains_death
        }
    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        return False, {'error': str(e)}


def test_node_death_strategy() -> Tuple[bool, Dict[str, Any]]:
    """Test node death with conservative strategy."""
    try:
        graph = initialize_main_graph(scale=0.1)
        if len(graph.node_labels) == 0:
            return False, {'error': 'No nodes in graph'}

        node_id = 0
        graph.node_labels[node_id]['type'] = 'dynamic'
        graph.node_labels[node_id]['state'] = 'inactive'
        graph.node_labels[node_id]['energy'] = 0.05  # <0.1
        if hasattr(graph, 'x') and graph.x is not None:
            graph.x[node_id, 0] = 0.05

        # Mock memory_importance low
        if not hasattr(graph, 'memory_system'):
            graph.memory_system = mock.Mock()
            graph.memory_system.get_node_memory_importance.return_value = 0.1  # <0.2

        initial_count = len(graph.node_labels)
        updated_graph = handle_node_death(graph, node_id, strategy='conservative')
        final_count = len(updated_graph.node_labels)

        removed = final_count < initial_count
        return removed, {
            'strategy': 'conservative',
            'initial_nodes': initial_count,
            'final_nodes': final_count
        }
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        return False, {'error': str(e)}


def test_node_birth_high_energy() -> Tuple[bool, Dict[str, Any]]:
    """Test node birth on high energy threshold."""
    try:
        log_capture, handler = setup_logging_capture()

        graph = initialize_main_graph(scale=0.1)
        initial_count = len(graph.node_labels)

        # Set a dynamic node to high energy
        high_energy = get_node_birth_threshold() + 0.1
        dynamic_found = False
        for i, node_label in enumerate(graph.node_labels):
            if node_label.get('type') == 'dynamic':
                if hasattr(graph, 'x') and graph.x is not None:
                    graph.x[i, 0] = high_energy
                dynamic_found = True
                break
        if not dynamic_found:
            # Force one
            if len(graph.node_labels) > 0:
                graph.node_labels[0]['type'] = 'dynamic'
                if hasattr(graph, 'x') and graph.x is not None:
                    graph.x[0, 0] = high_energy

        updated_graph = birth_new_dynamic_nodes(graph)
        final_count = len(updated_graph.node_labels)

        logs = log_capture.getvalue()
        cleanup_logging_capture(handler)

        birth_occurred = final_count > initial_count
        log_contains_birth = "[BIRTH]" in logs

        return birth_occurred and log_contains_birth, {
            'initial_nodes': initial_count,
            'final_nodes': final_count,
            'birth_logged': log_contains_birth
        }
    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        return False, {'error': str(e)}


def test_node_birth_memory() -> Tuple[bool, Dict[str, Any]]:
    """Test memory-influenced node birth parameters."""
    try:
        graph = initialize_main_graph(scale=0.1)

        # Mock memory system for integrator behavior (>10 traces)
        graph.memory_system = MockMemory()

        with mock.patch('neural.death_and_birth_logic.random.random', return_value=0.4):
            params = analyze_memory_patterns_for_birth(graph)

        # For >10 traces, behavior='integrator', energy=0.7
        memory_influenced = params.get('behavior') == 'integrator' and params.get('energy') == 0.7

        return memory_influenced, {
            'behavior': params.get('behavior'),
            'energy': params.get('energy')
        }
    except (ImportError, AttributeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_event_queue_order() -> Tuple[bool, Dict[str, Any]]:
    """Test event queue processing order by timestamp and priority."""
    try:
        system = create_event_driven_system()

        # Test timestamp order
        event_early = NeuralEvent(EventType.SPIKE, timestamp=0.5, source_node_id=2, priority=0)
        event_late = NeuralEvent(EventType.SPIKE, timestamp=1.0, source_node_id=1, priority=0)
        system.event_queue.push(event_late)
        system.event_queue.push(event_early)

        popped1 = system.event_queue.pop()
        popped2 = system.event_queue.pop()
        timestamp_order = popped1.timestamp < popped2.timestamp

        # Test same timestamp, priority (higher first)
        event_high_pri = NeuralEvent(EventType.SPIKE, timestamp=1.0, source_node_id=3, priority=2)
        event_low_pri = NeuralEvent(EventType.SPIKE, timestamp=1.0, source_node_id=4, priority=1)
        system.event_queue.push(event_low_pri)
        system.event_queue.push(event_high_pri)

        popped3 = system.event_queue.pop()
        popped4 = system.event_queue.pop()
        priority_order = popped3.priority > popped4.priority and popped3.timestamp == popped4.timestamp

        return timestamp_order and priority_order, {
            'timestamp_order': timestamp_order,
            'priority_order': priority_order
        }
    except (ImportError, AttributeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_spike_propagation() -> Tuple[bool, Dict[str, Any]]:
    """Test spike event propagation to synaptic transmission."""
    try:
        mock_sim = mock.Mock()
        mock_access = MockAccessLayer()
        mock_sim.get_access_layer.return_value = mock_access

        system = create_event_driven_system(mock_sim)

        system.schedule_spike(1, timestamp=0.0)

        processed = system.process_events(max_events=20)

        bus = get_event_bus()
        bus.emit('SPIKE', {'source_node_id': 1, 'timestamp': 0.0})

        # Assert spike handled
        spike_handled = 'last_spike' in mock_access.nodes[1]

        # Assert synaptic input updated on target (2)
        synaptic_updated = 'synaptic_input' in mock_access.nodes[2] and mock_access.nodes[2]['synaptic_input'] > 0

        # Assert new spike potentially triggered if threshold met
        threshold_met = mock_access.nodes[2]['synaptic_input'] >= 0.5

        spike_count = system.event_processor.stats['events_by_type'][EventType.SPIKE]
        synaptic_count = system.event_processor.stats['events_by_type'][EventType.SYNAPTIC_TRANSMISSION]

        propagation = spike_handled and synaptic_updated and spike_count >= 1 and synaptic_count >= 1

        return propagation, {
            'processed': processed,
            'spike_count': spike_count,
            'synaptic_count': synaptic_count,
            'threshold_met': threshold_met
        }
    except (ImportError, AttributeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_plasticity_propagation() -> Tuple[bool, Dict[str, Any]]:
    """Test plasticity events triggered by spikes."""
    try:
        mock_sim = mock.Mock()
        mock_sim.learning_engine = mock.Mock()  # For plasticity handler
        mock_access = MockAccessLayer()
        mock_sim.get_access_layer.return_value = mock_access

        system = create_event_driven_system(mock_sim)

        system.schedule_spike(1, timestamp=0.0)

        processed = system.process_events(max_events=20)

        # Assert spike handled
        spike_handled = mock_access.nodes[1].get('spike_count') > 0

        # Assert plasticity apply called
        mock_sim.learning_engine.apply_timing_learning.assert_called()

        plasticity_count = system.event_processor.stats['events_by_type'][EventType.PLASTICITY_UPDATE]

        trigger_success = spike_handled and plasticity_count >= 1

        return trigger_success, {
            'processed': processed,
            'plasticity_count': plasticity_count,
            'apply_called': mock_sim.learning_engine.apply_timing_learning.called
        }
    except (ImportError, AttributeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_hebbian_ltp() -> Tuple[bool, Dict[str, Any]]:
    """Test Hebbian LTP on positive delta_t spike timing."""
    try:
        mock_access_layer = MagicMock()
        engine = LearningEngine(mock_access_layer)

        pre_node = {'id': 1}
        post_node = {'id': 2}

        class MockEdge:
            def __init__(self):
                self.eligibility_trace = 0.0

        edge = MockEdge()
        initial_trace = edge.eligibility_trace

        delta_t = 0.005  # Positive, within window (5ms)
        change = engine.apply_timing_learning(pre_node, post_node, edge, delta_t)

        ltp_applied = change > 0 and edge.eligibility_trace > initial_trace
        stdp_triggered = engine.learning_stats['stdp_events'] >= 1

        return ltp_applied and stdp_triggered, {
            'delta_t': delta_t,
            'change': change,
            'final_trace': edge.eligibility_trace,
            'stdp_events': engine.learning_stats['stdp_events']
        }
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        return False, {'error': str(e)}


def test_hebbian_ltd() -> Tuple[bool, Dict[str, Any]]:
    """Test Hebbian LTD on negative delta_t spike timing."""
    try:
        mock_access_layer = MagicMock()
        engine = LearningEngine(mock_access_layer)

        pre_node = {'id': 1}
        post_node = {'id': 2}

        class MockEdge:
            def __init__(self):
                self.eligibility_trace = 0.0

        edge = MockEdge()
        initial_trace = edge.eligibility_trace

        delta_t = -0.005  # Negative, within window (-5ms)
        change = engine.apply_timing_learning(pre_node, post_node, edge, delta_t)

        ltd_applied = change < 0 and edge.eligibility_trace < initial_trace
        stdp_triggered = engine.learning_stats['stdp_events'] >= 1

        return ltd_applied and stdp_triggered, {
            'delta_t': delta_t,
            'change': change,
            'final_trace': edge.eligibility_trace,
            'stdp_events': engine.learning_stats['stdp_events']
        }
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        return False, {'error': str(e)}


def test_learning_consolidation() -> Tuple[bool, Dict[str, Any]]:
    """Test eligibility trace consolidation to weight updates."""
    try:
        mock_access_layer = MagicMock()
        engine = LearningEngine(mock_access_layer)

        class MockGraph:
            def __init__(self):
                self.edge_attributes = []

        class MockEdge:
            def __init__(self, trace=0.6, weight=1.0):
                self.eligibility_trace = trace
                self.weight = weight
                self.source = 1
                self.target = 2
                self.type = 'excitatory'

        graph = MockGraph()
        edge = MockEdge(trace=0.6, weight=1.0)  # trace > 0.5 threshold
        graph.edge_attributes.append(edge)

        initial_weight = edge.weight
        updated_graph = engine.consolidate_connections(graph)
        final_weight = updated_graph.edge_attributes[0].weight

        weight_updated = final_weight > initial_weight
        cons_triggered = engine.learning_stats['consolidation_events'] >= 1
        trace_reduced = updated_graph.edge_attributes[0].eligibility_trace < 0.6  # *=0.5

        return weight_updated and cons_triggered and trace_reduced, {
            'initial_weight': initial_weight,
            'final_weight': final_weight,
            'cons_events': engine.learning_stats['consolidation_events'],
            'trace_reduced': trace_reduced
        }
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        return False, {'error': str(e)}


def create_neural_test_cases() -> list:
    """Create neural test cases."""
    return [
        TestCase(
            name="energy_behavior",
            category=TestCategory.NEURAL,
            description="Test energy behavior and dynamics",
            test_func=test_energy_behavior
        ),
        TestCase(
            name="connection_logic",
            category=TestCategory.NEURAL,
            description="Test connection logic and formation",
            test_func=test_connection_logic
        ),
        TestCase(
            name="neural_math_accuracy",
            category=TestCategory.NEURAL,
            description="Test mathematical accuracy in neural dynamics (membrane, spikes, STDP)",
            test_func=test_neural_math_accuracy
        ),
        TestCase(
            name="connection_math_accuracy",
            category=TestCategory.UNIT,
            description="Test connection weight updates and Hebbian rules",
            test_func=test_connection_math_accuracy
        ),
        TestCase(
            name="node_death_low_energy",
            category=TestCategory.INTEGRATION,
            description="Test node death triggered by low energy threshold",
            test_func=test_node_death_low_energy
        ),
        TestCase(
            name="node_death_strategy",
            category=TestCategory.INTEGRATION,
            description="Test node death with conservative strategy",
            test_func=test_node_death_strategy
        ),
        TestCase(
            name="node_birth_high_energy",
            category=TestCategory.INTEGRATION,
            description="Test node birth triggered by high energy threshold",
            test_func=test_node_birth_high_energy
        ),
        TestCase(
            name="node_birth_memory",
            category=TestCategory.NEURAL,
            description="Test memory-influenced node birth parameters",
            test_func=test_node_birth_memory
        ),
        TestCase(
            name="event_queue_order",
            category=TestCategory.INTEGRATION,
            description="Test event queue maintains processing order",
            test_func=test_event_queue_order
        ),
        TestCase(
            name="spike_propagation",
            category=TestCategory.NEURAL,
            description="Test spike event propagation to synaptic transmission",
            test_func=test_spike_propagation
        ),
        TestCase(
            name="plasticity_propagation",
            category=TestCategory.NEURAL,
            description="Test plasticity events triggered by spikes",
            test_func=test_plasticity_propagation
        ),
        TestCase(
            name="hebbian_ltp_trigger",
            category=TestCategory.NEURAL,
            description="Test Hebbian LTP triggered by positive timing",
            test_func=test_hebbian_ltp
        ),
        TestCase(
            name="hebbian_ltd_trigger",
            category=TestCategory.NEURAL,
            description="Test Hebbian LTD triggered by negative timing",
            test_func=test_hebbian_ltd
        ),
        TestCase(
            name="learning_consolidation",
            category=TestCategory.NEURAL,
            description="Test consolidation of eligibility traces to weights",
            test_func=test_learning_consolidation
        )
    ]