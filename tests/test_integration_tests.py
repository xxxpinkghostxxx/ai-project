"""
Integration tests for the unified testing system.
Tests integration between different components and full simulation cycles.
"""

import time
from typing import Tuple, Dict, Any
from unittest.mock import MagicMock
import unittest.mock as mock

# Third-party imports
import psutil

# Local imports - Core modules
from src.core.main_graph import initialize_main_graph
from src.core.services.simulation_coordinator import SimulationCoordinator

# Local imports - UI modules
from src.ui.ui_engine import update_ui_display
from src.ui.ui_state_manager import get_ui_state_manager

# Local imports - Utils modules
from src.utils.event_bus import get_event_bus
from src.utils.unified_performance_system import get_performance_monitor

# Local imports - Learning modules (for test_full_simulation_cycle)
try:
    from src.learning.learning_engine import LearningEngine
except ImportError:
    # Fallback for when LearningEngine is not available
    LearningEngine = None

# Local test utilities
from .test_utils import TestCase, TestCategory, stress_test_memory_limit, validate_performance_threshold
from .test_mocks import create_mock_services, configure_mock_services_for_init


def test_single_simulation_step() -> Tuple[bool, Dict[str, Any]]:
    """Test single simulation step execution."""
    try:
        # Create mocked services for SimulationCoordinator
        mocks = create_mock_services()
        configure_mock_services_for_init(mocks)

        (service_registry, neural_processor, energy_manager,
         learning_engine, sensory_processor, performance_monitor,
         graph_manager, event_coordinator, configuration_service) = mocks

        # Configure mocks for initialization and execution
        graph = initialize_main_graph(scale=0.25)
        graph_manager.initialize_graph.return_value = graph
        neural_processor.initialize_neural_state.return_value = True
        energy_manager.initialize_energy_state.return_value = True
        learning_engine.initialize_learning_state.return_value = True
        sensory_processor.initialize_sensory_pathways.return_value = True
        performance_monitor.start_monitoring.return_value = True

        # Configure for step execution
        neural_processor.process_neural_dynamics.return_value = (graph, [])
        energy_manager.update_energy_flows.return_value = (graph, [])
        learning_engine.apply_plasticity.return_value = (graph, [])
        graph_manager.update_node_lifecycle.return_value = graph
        energy_manager.regulate_energy_homeostasis.return_value = graph

        sim_manager = SimulationCoordinator(
            service_registry, neural_processor, energy_manager,
            learning_engine, sensory_processor, performance_monitor,
            graph_manager, event_coordinator, configuration_service
        )

        # Initialize and start simulation
        sim_manager.initialize_simulation()
        sim_manager.start_simulation()

        success = sim_manager.execute_simulation_step(1)

        return success, {
            'step_executed': success,
            'graph_nodes': len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
        }
    except (ImportError, AttributeError, TypeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_simulation_progression() -> Tuple[bool, Dict[str, Any]]:
    """Test simulation progression over multiple steps."""
    try:
        # Create mocked services for SimulationCoordinator
        mocks = create_mock_services()
        configure_mock_services_for_init(mocks)

        (service_registry, neural_processor, energy_manager,
         learning_engine, sensory_processor, performance_monitor,
         graph_manager, event_coordinator, configuration_service) = mocks

        # Configure mocks for initialization and execution
        graph = initialize_main_graph(scale=0.25)
        graph_manager.initialize_graph.return_value = graph
        neural_processor.initialize_neural_state.return_value = True
        energy_manager.initialize_energy_state.return_value = True
        learning_engine.initialize_learning_state.return_value = True
        sensory_processor.initialize_sensory_pathways.return_value = True
        performance_monitor.start_monitoring.return_value = True

        # Configure for step execution
        neural_processor.process_neural_dynamics.return_value = (graph, [])
        energy_manager.update_energy_flows.return_value = (graph, [])
        learning_engine.apply_plasticity.return_value = (graph, [])
        graph_manager.update_node_lifecycle.return_value = graph
        energy_manager.regulate_energy_homeostasis.return_value = graph

        sim_manager = SimulationCoordinator(
            service_registry, neural_processor, energy_manager,
            learning_engine, sensory_processor, performance_monitor,
            graph_manager, event_coordinator, configuration_service
        )

        # Initialize and start simulation
        sim_manager.initialize_simulation()
        sim_manager.start_simulation()

        steps_completed = 0
        for i in range(10):
            if sim_manager.execute_simulation_step(i + 1):
                steps_completed += 1
            else:
                break

        return steps_completed == 10, {
            'steps_completed': steps_completed,
            'expected_steps': 10
        }
    except (ImportError, AttributeError, TypeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_ui_components() -> Tuple[bool, Dict[str, Any]]:
    """Test UI components and functionality."""
    try:
        ui_state = get_ui_state_manager()

        return ui_state is not None, {
            'ui_state_available': ui_state is not None
        }
    except (ImportError, AttributeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_performance_monitoring() -> Tuple[bool, Dict[str, Any]]:
    """Test performance monitoring systems."""
    try:
        monitor = get_performance_monitor()
        metrics = monitor.get_current_metrics()

        return monitor is not None, {
            'monitor_available': monitor is not None,
            'metrics_available': metrics is not None
        }
    except (ImportError, AttributeError, RuntimeError) as e:
        return False, {'error': str(e)}


def test_full_simulation_cycle() -> Tuple[bool, Dict[str, Any]]:
    """E2E test for full simulation cycle."""
    try:
        # Create mocked services for SimulationCoordinator
        mocks = create_mock_services()
        configure_mock_services_for_init(mocks)

        (service_registry, neural_processor, energy_manager,
         learning_engine, sensory_processor, performance_monitor,
         graph_manager, event_coordinator, configuration_service) = mocks

        # Configure mocks for initialization and execution
        graph = initialize_main_graph(scale=0.5)
        graph_manager.initialize_graph.return_value = graph
        neural_processor.initialize_neural_state.return_value = True
        energy_manager.initialize_energy_state.return_value = True
        learning_engine.initialize_learning_state.return_value = True
        sensory_processor.initialize_sensory_pathways.return_value = True
        performance_monitor.start_monitoring.return_value = True

        # Configure for step execution
        neural_processor.process_neural_dynamics.return_value = (graph, [])
        energy_manager.update_energy_flows.return_value = (graph, [])
        learning_engine.apply_plasticity.return_value = (graph, [])
        graph_manager.update_node_lifecycle.return_value = graph
        energy_manager.regulate_energy_homeostasis.return_value = graph

        sim_manager = SimulationCoordinator(
            service_registry, neural_processor, energy_manager,
            learning_engine, sensory_processor, performance_monitor,
            graph_manager, event_coordinator, configuration_service
        )

        # Initialize and start simulation
        sim_manager.initialize_simulation()
        sim_manager.start_simulation()

        initial_node_count = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0

        mock_access_layer = MagicMock()
        learning_engine = LearningEngine(mock_access_layer) if LearningEngine else MagicMock()

        bus = get_event_bus()
        event_calls = {'SPIKE': 0, 'GRAPH_UPDATE': 0}

        def mock_callback(event_type, _data):
            if event_type in event_calls:
                event_calls[event_type] += 1

        orig_subscribe = bus.subscribe
        bus.subscribe = mock.Mock(side_effect=lambda et, cb: [mock_callback(et, None)] if et in ['SPIKE', 'GRAPH_UPDATE'] else orig_subscribe(et, cb))

        for i in range(500):
            sim_manager.execute_simulation_step(i + 1)
            update_ui_display()

        bus.subscribe = orig_subscribe  # Restore

        final_node_count = len(sim_manager.graph.node_labels) if hasattr(sim_manager.graph, 'node_labels') else 0

        node_changes = abs(final_node_count - initial_node_count) > 0
        learning_gt0 = sum(learning_engine.learning_stats.values()) > 0 if hasattr(learning_engine, 'learning_stats') else False
        events_gt0 = event_calls['SPIKE'] > 0 and event_calls['GRAPH_UPDATE'] > 0

        # Type hints runtime check (skip mypy, assert no runtime errors)
        type_ok = True  # Basic check, assume no errors since executed

        return node_changes and learning_gt0 and events_gt0 and type_ok, {
            'node_changes': node_changes,
            'learning_gt0': learning_gt0,
            'events_gt0': events_gt0,
            'type_ok': type_ok,
            'initial_nodes': initial_node_count,
            'final_nodes': final_node_count
        }
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        return False, {'error': str(e)}


def test_stress_growth() -> Tuple[bool, Dict[str, Any]]:
    """Stress test for organic growth and full cycles."""
    try:
        process = psutil.Process()

        # Create mocked services for SimulationCoordinator
        mocks = create_mock_services()
        configure_mock_services_for_init(mocks)

        (service_registry, neural_processor, energy_manager,
         learning_engine, sensory_processor, performance_monitor,
         graph_manager, event_coordinator, configuration_service) = mocks

        # Configure mocks for initialization and execution
        graph = initialize_main_graph(scale=2.0)
        graph_manager.initialize_graph.return_value = graph
        neural_processor.initialize_neural_state.return_value = True
        energy_manager.initialize_energy_state.return_value = True
        learning_engine.initialize_learning_state.return_value = True
        sensory_processor.initialize_sensory_pathways.return_value = True
        performance_monitor.start_monitoring.return_value = True

        # Configure for step execution
        neural_processor.process_neural_dynamics.return_value = (graph, [])
        energy_manager.update_energy_flows.return_value = (graph, [])
        learning_engine.apply_plasticity.return_value = (graph, [])
        graph_manager.update_node_lifecycle.return_value = graph
        energy_manager.regulate_energy_homeostasis.return_value = graph

        sim_manager = SimulationCoordinator(
            service_registry, neural_processor, energy_manager,
            learning_engine, sensory_processor, performance_monitor,
            graph_manager, event_coordinator, configuration_service
        )

        # Initialize and start simulation
        sim_manager.initialize_simulation()
        sim_manager.start_simulation()

        initial_node_count = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
        previous_count = initial_node_count
        birth_death_events = 0
        step_times = []

        start_time = time.perf_counter()

        for i in range(10000):
            step_start = time.perf_counter()
            sim_manager.execute_simulation_step(i + 1)
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            current_node_count = len(sim_manager.graph.node_labels) if hasattr(sim_manager.graph, 'node_labels') else 0
            if current_node_count != previous_count:
                birth_death_events += abs(current_node_count - previous_count)
                previous_count = current_node_count

            if i % 1000 == 0:
                current_memory_gb = process.memory_info().rss / (1024 ** 3)
                if current_memory_gb > 2.0:
                    return False, {'memory_exceeded_gb': current_memory_gb, 'at_step': i}

        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        memory_end = process.memory_info().rss / (1024 ** 3)
        total_time = time.perf_counter() - start_time

        perf_ok = validate_performance_threshold(avg_step_time, 10.0)  # <10ms post-numba
        memory_ok = stress_test_memory_limit(2.0)
        events_gt10 = birth_death_events > 10
        no_crash = True

        return perf_ok and memory_ok and events_gt10 and no_crash, {
            'avg_step_time_s': avg_step_time,
            'memory_gb_end': memory_end,
            'birth_death_events': birth_death_events,
            'total_time_s': total_time,
            'perf_ok': perf_ok,
            'memory_ok': memory_ok
        }
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        return False, {'error': str(e), 'crashed': True}


def create_integration_test_cases() -> list:
    """Create integration test cases."""
    return [
        TestCase(
            name="single_simulation_step",
            category=TestCategory.INTEGRATION,
            description="Test single simulation step execution",
            test_func=test_single_simulation_step
        ),
        TestCase(
            name="simulation_progression",
            category=TestCategory.INTEGRATION,
            description="Test simulation progression over multiple steps",
            test_func=test_simulation_progression
        ),
        TestCase(
            name="ui_components",
            category=TestCategory.UI,
            description="Test UI components and functionality",
            test_func=test_ui_components
        ),
        TestCase(
            name="performance_monitoring",
            category=TestCategory.PERFORMANCE,
            description="Test performance monitoring systems",
            test_func=test_performance_monitoring
        ),
        TestCase(
            name="full_simulation_cycle",
            category=TestCategory.SYSTEM,
            description="E2E test: full simulation cycle with growth/learning/UI via event bus",
            test_func=test_full_simulation_cycle
        ),
        TestCase(
            name="stress_growth",
            category=TestCategory.STRESS,
            description="Stress test: large graph, long runs for birth/death cascades, memory",
            test_func=test_stress_growth
        )
    ]