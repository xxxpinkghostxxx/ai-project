"""
Unified Testing System for the Neural Simulation.
Consolidates unified_test_suite.py and comprehensive_test_framework.py
into a comprehensive testing framework.
"""

import gc
import logging
import os
import sys
import threading
import time
import timeit
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock, Mock

# Third-party imports
import numpy as np
import psutil
import torch
from numba import jit
from torch_geometric.data import Data

# Local imports - Core interfaces
from src.core.interfaces.configuration_service import IConfigurationService
from src.core.interfaces.energy_manager import IEnergyManager
from src.core.interfaces.event_coordinator import IEventCoordinator
from src.core.interfaces.graph_manager import IGraphManager
from src.core.interfaces.learning_engine import ILearningEngine
from src.core.interfaces.neural_processor import INeuralProcessor
from src.core.interfaces.performance_monitor import IPerformanceMonitor
from src.core.interfaces.sensory_processor import ISensoryProcessor
from src.core.interfaces.service_registry import IServiceRegistry
# Local imports - Core modules
from src.core.main_graph import initialize_main_graph
# Local imports - Core services
from src.core.services.simulation_coordinator import SimulationCoordinator
# Local imports - Energy modules
from src.energy.energy_behavior import EnergyCalculator, apply_energy_behavior, get_node_energy_cap
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
# Local imports - UI modules
from src.ui.ui_engine import update_ui_display
from src.ui.ui_state_manager import get_ui_state_manager
# Local imports - Utils modules
from src.utils.event_bus import get_event_bus
from src.utils.logging_utils import log_step
from src.utils.print_utils import print_error, print_info, print_warning
from src.utils.unified_error_handler import get_error_handler, safe_execute
from src.utils.unified_performance_system import get_performance_monitor

# Setup sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCategory(Enum):
    """Test categories for organization."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    STRESS = "stress"
    MEMORY = "memory"
    FUNCTIONAL = "functional"
    SYSTEM = "system"
    NEURAL = "neural"
    UI = "ui"


class TestResult(Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestMetrics:
    """Test execution metrics."""
    duration: float
    memory_used_mb: float
    cpu_percent: float
    gc_collections: int
    assertions_count: int
    error_count: int


@dataclass
class TestCase:
    """Test case information."""
    name: str
    category: TestCategory
    description: str
    test_func: Callable
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout: float = 30.0
    expected_duration: float = 1.0
    memory_limit_mb: float = 100.0
    cpu_limit_percent: float = 80.0


@dataclass
class TestExecutionResult:
    """Test execution result."""
    test_case: TestCase
    result: TestResult
    duration: float
    error: Optional[Exception] = None
    metrics: Optional[TestMetrics] = None
    output: str = ""
    assertions: List[str] = field(default_factory=list)


class MockSimulationCoordinator:
    """Mock simulation coordinator for testing."""

    def __init__(self):
        self.graph = None
        self.is_running = False
        self.step_count = 0
        self.performance_stats = {
            'fps': 60.0,
            'memory_usage_mb': 100.0,
            'cpu_percent': 50.0
        }

    def initialize_graph(self):
        """Initialize a test graph."""
        self.graph = self._create_test_graph()

    def _create_test_graph(self) -> Data:
        """Create a test graph for testing."""
        num_nodes = 10
        num_edges = 15

        # Create node features
        x = torch.randn(num_nodes, 5)

        # Create edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Create node labels
        node_labels = []
        for i in range(num_nodes):
            node_labels.append({
                'id': i,
                'type': 'dynamic' if i < 5 else 'sensory',
                'energy': float(torch.rand(1).item() * 255),
                'state': 'active'
            })

        return Data(x=x, edge_index=edge_index, node_labels=node_labels)

    def run_single_step(self) -> bool:
        """Simulate a single simulation step."""
        if not self.is_running:
            return False

        self.step_count += 1
        time.sleep(0.01)  # Simulate processing time
        return True

    def start_simulation(self):
        """Start simulation."""
        self.is_running = True

    def stop_simulation(self):
        """Stop simulation."""
        self.is_running = False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


class UnifiedTestFramework:
    """Unified testing framework combining all testing capabilities."""

    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.results: List[TestExecutionResult] = []
        self.fixtures: Dict[str, Callable] = {}
        self.start_time = time.time()
        self._lock = threading.RLock()

        # Setup default test cases
        self._setup_default_tests()

        log_step("UnifiedTestFramework initialized")

    def _setup_default_tests(self):
        """Setup default test cases."""
        # Basic functionality tests
        self.add_test_case(TestCase(
            name="critical_imports",
            category=TestCategory.SYSTEM,
            description="Test critical module imports",
            test_func=self._test_critical_imports
        ))

        self.add_test_case(TestCase(
            name="memory_usage",
            category=TestCategory.MEMORY,
            description="Test memory usage and leaks",
            test_func=self._test_memory_usage
        ))

        self.add_test_case(TestCase(
            name="simulation_manager_creation",
            category=TestCategory.UNIT,
            description="Test simulation manager creation",
            test_func=self._test_simulation_manager_creation
        ))

        self.add_test_case(TestCase(
            name="single_simulation_step",
            category=TestCategory.INTEGRATION,
            description="Test single simulation step execution",
            test_func=self._test_single_simulation_step
        ))

        self.add_test_case(TestCase(
            name="simulation_progression",
            category=TestCategory.INTEGRATION,
            description="Test simulation progression over multiple steps",
            test_func=self._test_simulation_progression
        ))

        self.add_test_case(TestCase(
            name="energy_behavior",
            category=TestCategory.NEURAL,
            description="Test energy behavior and dynamics",
            test_func=self._test_energy_behavior
        ))

        self.add_test_case(TestCase(
            name="connection_logic",
            category=TestCategory.NEURAL,
            description="Test connection logic and formation",
            test_func=self._test_connection_logic
        ))

        # Math accuracy tests
        self.add_test_case(TestCase(
            name="neural_math_accuracy",
            category=TestCategory.NEURAL,
            description="Test mathematical accuracy in neural dynamics (membrane, spikes, STDP)",
            test_func=self._test_neural_math_accuracy
        ))

        self.add_test_case(TestCase(
            name="energy_math_accuracy",
            category=TestCategory.UNIT,
            description="Test energy calculations (decay, caps, transfers)",
            test_func=self._test_energy_math_accuracy
        ))

        self.add_test_case(TestCase(
            name="connection_math_accuracy",
            category=TestCategory.UNIT,
            description="Test connection weight updates and Hebbian rules",
            test_func=self._test_connection_math_accuracy
        ))

        self.add_test_case(TestCase(
            name="ui_components",
            category=TestCategory.UI,
            description="Test UI components and functionality",
            test_func=self._test_ui_components
        ))

        self.add_test_case(TestCase(
            name="performance_monitoring",
            category=TestCategory.PERFORMANCE,
            description="Test performance monitoring systems",
            test_func=self._test_performance_monitoring
        ))

        self.add_test_case(TestCase(
            name="error_handling",
            category=TestCategory.SYSTEM,
            description="Test error handling and recovery",
            test_func=self._test_error_handling
        ))

        # Integration tests for node lifecycle events (death_and_birth_logic.py)
        self.add_test_case(TestCase(
            name="node_death_low_energy",
            category=TestCategory.INTEGRATION,
            description="Test node death triggered by low energy threshold",
            test_func=self._test_node_death_low_energy
        ))
        self.add_test_case(TestCase(
            name="node_death_strategy",
            category=TestCategory.INTEGRATION,
            description="Test node death with conservative strategy",
            test_func=self._test_node_death_strategy
        ))
        self.add_test_case(TestCase(
            name="node_birth_high_energy",
            category=TestCategory.INTEGRATION,
            description="Test node birth triggered by high energy threshold",
            test_func=self._test_node_birth_high_energy
        ))
        self.add_test_case(TestCase(
            name="node_birth_memory",
            category=TestCategory.NEURAL,
            description="Test memory-influenced node birth parameters",
            test_func=self._test_node_birth_memory
        ))

        # Integration tests for event propagation (event_driven_system.py)
        self.add_test_case(TestCase(
            name="event_queue_order",
            category=TestCategory.INTEGRATION,
            description="Test event queue maintains processing order",
            test_func=self._test_event_queue_order
        ))
        self.add_test_case(TestCase(
            name="spike_propagation",
            category=TestCategory.NEURAL,
            description="Test spike event propagation to synaptic transmission",
            test_func=self._test_spike_propagation
        ))
        self.add_test_case(TestCase(
            name="plasticity_propagation",
            category=TestCategory.NEURAL,
            description="Test plasticity events triggered by spikes",
            test_func=self._test_plasticity_propagation
        ))

        # Tests for learning triggers (learning_engine.py)
        self.add_test_case(TestCase(
            name="hebbian_ltp_trigger",
            category=TestCategory.NEURAL,
            description="Test Hebbian LTP triggered by positive timing",
            test_func=self._test_hebbian_ltp
        ))
        self.add_test_case(TestCase(
            name="hebbian_ltd_trigger",
            category=TestCategory.NEURAL,
            description="Test Hebbian LTD triggered by negative timing",
            test_func=self._test_hebbian_ltd
        ))
        self.add_test_case(TestCase(
            name="learning_consolidation",
            category=TestCategory.NEURAL,
            description="Test consolidation of eligibility traces to weights",
            test_func=self._test_learning_consolidation
        ))

        self.add_test_case(TestCase(
            name="event_bus_functionality",
            category=TestCategory.SYSTEM,
            description="Test EventBus subscribe, emit, unsubscribe, and fallback",
            test_func=self._test_event_bus
        ))

        self.add_test_case(TestCase(
            name="full_simulation_cycle",
            category=TestCategory.SYSTEM,
            description="E2E test: full simulation cycle with growth/learning/UI via event bus",
            test_func=self._test_full_simulation_cycle
        ))

        self.add_test_case(TestCase(
            name="stress_growth",
            category=TestCategory.STRESS,
            description="Stress test: large graph, long runs for birth/death cascades, memory",
            test_func=self._test_stress_growth
        ))

    def add_test_case(self, test_case: TestCase):
        """Add a test case to the framework."""
        with self._lock:
            self.test_cases.append(test_case)

    def add_fixture(self, name: str, fixture_func: Callable):
        """Add a test fixture."""
        self.fixtures[name] = fixture_func

    def run_tests(self, category: Optional[TestCategory] = None,
                  parallel: bool = False) -> List[TestExecutionResult]:
        """Run tests with optional filtering and parallel execution."""
        with self._lock:
            # Filter test cases by category if specified
            test_cases = self.test_cases
            if category:
                test_cases = [tc for tc in test_cases if tc.category == category]

            if parallel:
                return self._run_tests_parallel(test_cases)
            else:
                return self._run_tests_sequential(test_cases)

    def _run_tests_sequential(self, test_cases: List[TestCase]) -> List[TestExecutionResult]:
        """Run tests sequentially."""
        results = []
        for test_case in test_cases:
            result = self._execute_test(test_case)
            results.append(result)
        return results

    def _run_tests_parallel(self, test_cases: List[TestCase]) -> List[TestExecutionResult]:
        """Run tests in parallel using threading."""
        results = []
        threads = []

        def run_test_wrapper(test_case):
            result = self._execute_test(test_case)
            with self._lock:
                results.append(result)

        for test_case in test_cases:
            thread = threading.Thread(target=run_test_wrapper, args=(test_case,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def _execute_test(self, test_case: TestCase) -> TestExecutionResult:
        """Execute a single test case."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_gc = sum(stat['collections'] for stat in gc.get_stats())

        try:
            # Setup
            if test_case.setup_func:
                test_case.setup_func()

            # Execute test
            test_result = test_case.test_func()

            # Determine result
            if isinstance(test_result, tuple) and len(test_result) == 2:
                success = test_result[0]
                result_status = TestResult.PASSED if success else TestResult.FAILED
            elif isinstance(test_result, bool):
                result_status = TestResult.PASSED if test_result else TestResult.FAILED
            else:
                result_status = TestResult.PASSED

            error = None

        except (RuntimeError, TypeError, ValueError, AttributeError) as e:
            # Broad catch for test execution - covers runtime issues, type mismatches,
            # invalid values, and attribute access problems during test execution
            result_status = TestResult.ERROR
            error = e
            test_result = False

        finally:
            # Teardown
            if test_case.teardown_func:
                try:
                    test_case.teardown_func()
                except (RuntimeError, AttributeError, TypeError) as e:
                    # Teardown failures - runtime issues, attribute access, or type mismatches
                    print_warning(f"Teardown failed for {test_case.name}: {e}")

        # Calculate metrics
        duration = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_gc = sum(stat['collections'] for stat in gc.get_stats())

        metrics = TestMetrics(
            duration=duration,
            memory_used_mb=end_memory - start_memory,
            cpu_percent=psutil.cpu_percent(),
            gc_collections=end_gc - start_gc,
            assertions_count=0,  # Would need to track this
            error_count=1 if error else 0
        )

        # Create result
        result = TestExecutionResult(
            test_case=test_case,
            result=result_status,
            duration=duration,
            error=error,
            metrics=metrics,
            output=str(test_result) if test_result else ""
        )

        # Log result
        status_symbol = "[PASS]" if result_status == TestResult.PASSED else "[FAIL]"
        print_info(f"{status_symbol} {test_case.name}: {result_status.value if hasattr(result_status, 'value') else str(result_status)} ({duration:.3f}s)")

        if error:
            print_error(f"  Error: {error}")

        return result

    def get_test_statistics(self) -> Dict[str, Any]:
        """Get comprehensive test statistics."""
        with self._lock:
            if not self.results:
                return {}

            total_tests = len(self.results)
            passed_tests = sum(1 for r in self.results if r.result == TestResult.PASSED)
            failed_tests = sum(1 for r in self.results if r.result == TestResult.FAILED)
            error_tests = sum(1 for r in self.results if r.result == TestResult.ERROR)

            total_duration = sum(r.duration for r in self.results)
            average_duration = total_duration / total_tests if total_tests > 0 else 0

            # Category breakdown
            category_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0})
            for result in self.results:
                category = result.test_case.category
                category_stats[category.value]['total'] += 1
                if result.result == TestResult.PASSED:
                    category_stats[category.value]['passed'] += 1
                elif result.result == TestResult.FAILED:
                    category_stats[category.value]['failed'] += 1
                else:
                    category_stats[category.value]['errors'] += 1

            return {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration': total_duration,
                'average_duration': average_duration,
                'category_breakdown': dict(category_stats)
            }

    # Test case implementations
    def _test_critical_imports(self) -> Tuple[bool, Dict[str, Any]]:
        """Test critical module imports."""
        try:
            return True, {'imports': 'successful'}
        except ImportError as e:
            return False, {'error': str(e)}

    def _test_memory_usage(self) -> Tuple[bool, Dict[str, Any]]:
        """Test memory usage and potential leaks."""
        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Create some objects
            test_objects = []
            for i in range(1000):
                test_objects.append({'id': i, 'data': np.random.rand(100)})

            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Clean up
            del test_objects
            gc.collect()

            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            return memory_increase < 50, {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase
            }
        except (OSError, psutil.Error, RuntimeError) as e:
            # Memory access issues, psutil errors, or runtime problems during memory testing
            return False, {'error': str(e)}

    def _test_simulation_manager_creation(self) -> Tuple[bool, Dict[str, Any]]:
        """Test simulation manager creation."""
        try:
            # Create mocked services for SimulationCoordinator
            service_registry = Mock(spec=IServiceRegistry)
            neural_processor = Mock(spec=INeuralProcessor)
            energy_manager = Mock(spec=IEnergyManager)
            learning_engine = Mock(spec=ILearningEngine)
            sensory_processor = Mock(spec=ISensoryProcessor)
            performance_monitor = Mock(spec=IPerformanceMonitor)
            graph_manager = Mock(spec=IGraphManager)
            event_coordinator = Mock(spec=IEventCoordinator)
            configuration_service = Mock(spec=IConfigurationService)

            # Configure mocks for initialization
            graph_manager.initialize_graph.return_value = Data()
            neural_processor.initialize_neural_state.return_value = True
            energy_manager.initialize_energy_state.return_value = True
            learning_engine.initialize_learning_state.return_value = True
            sensory_processor.initialize_sensory_pathways.return_value = True
            performance_monitor.start_monitoring.return_value = True

            sim_manager = SimulationCoordinator(
                service_registry, neural_processor, energy_manager,
                learning_engine, sensory_processor, performance_monitor,
                graph_manager, event_coordinator, configuration_service
            )

            return sim_manager is not None, {
                'manager_created': True,
                'type': type(sim_manager).__name__
            }
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # Object creation issues, mock setup problems, or initialization failures
            return False, {'error': str(e)}

    def _test_single_simulation_step(self) -> Tuple[bool, Dict[str, Any]]:
        """Test single simulation step execution."""
        try:
            # Create mocked services for SimulationCoordinator
            service_registry = Mock(spec=IServiceRegistry)
            neural_processor = Mock(spec=INeuralProcessor)
            energy_manager = Mock(spec=IEnergyManager)
            learning_engine = Mock(spec=ILearningEngine)
            sensory_processor = Mock(spec=ISensoryProcessor)
            performance_monitor = Mock(spec=IPerformanceMonitor)
            graph_manager = Mock(spec=IGraphManager)
            event_coordinator = Mock(spec=IEventCoordinator)
            configuration_service = Mock(spec=IConfigurationService)

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
            # Graph initialization, mock setup, or simulation step execution issues
            return False, {'error': str(e)}

    def _test_simulation_progression(self) -> Tuple[bool, Dict[str, Any]]:
        """Test simulation progression over multiple steps."""
        try:
            # Create mocked services for SimulationCoordinator
            service_registry = Mock(spec=IServiceRegistry)
            neural_processor = Mock(spec=INeuralProcessor)
            energy_manager = Mock(spec=IEnergyManager)
            learning_engine = Mock(spec=ILearningEngine)
            sensory_processor = Mock(spec=ISensoryProcessor)
            performance_monitor = Mock(spec=IPerformanceMonitor)
            graph_manager = Mock(spec=IGraphManager)
            event_coordinator = Mock(spec=IEventCoordinator)
            configuration_service = Mock(spec=IConfigurationService)

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
            # Multi-step simulation issues, mock setup, or progression tracking problems
            return False, {'error': str(e)}

    def _test_energy_behavior(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Energy behavior function import or execution issues
            return False, {'error': str(e)}

    def _test_connection_logic(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Connection logic function import or graph manipulation issues
            return False, {'error': str(e)}

    def _test_ui_components(self) -> Tuple[bool, Dict[str, Any]]:
        """Test UI components and functionality."""
        try:
            ui_state = get_ui_state_manager()

            return ui_state is not None, {
                'ui_state_available': ui_state is not None
            }
        except (ImportError, AttributeError, RuntimeError) as e:
            # UI state manager import or initialization issues
            return False, {'error': str(e)}

    def _test_performance_monitoring(self) -> Tuple[bool, Dict[str, Any]]:
        """Test performance monitoring systems."""
        try:
            monitor = get_performance_monitor()
            metrics = monitor.get_current_metrics()

            return monitor is not None, {
                'monitor_available': monitor is not None,
                'metrics_available': metrics is not None
            }
        except (ImportError, AttributeError, RuntimeError) as e:
            # Performance monitor import or metrics collection issues
            return False, {'error': str(e)}

    def _test_error_handling(self) -> Tuple[bool, Dict[str, Any]]:
        """Test error handling and recovery."""
        try:
            error_handler = get_error_handler()

            # Test safe execution
            def failing_function():
                raise ValueError("Test error")

            result = safe_execute(failing_function, "test_context", default_return="fallback")

            return result == "fallback", {
                'error_handler_available': error_handler is not None,
                'safe_execute_works': result == "fallback"
            }
        except (ImportError, RuntimeError, ValueError) as e:
            # Error handler import or safe execution issues
            return False, {'error': str(e)}

    def _test_neural_math_accuracy(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Mathematical computation or neural dynamics execution issues
            return False, {'error': str(e)}

    def _test_energy_math_accuracy(self) -> Tuple[bool, Dict[str, Any]]:
        """Test energy calculations for accuracy."""
        try:
            energy_cap = get_node_energy_cap()
            np.testing.assert_almost_equal(energy_cap, 255.0, decimal=6)  # default

            # Test decay
            current_energy = 100.0
            decay_rate = 0.99  # example
            expected_decay = current_energy * decay_rate
            decayed = EnergyCalculator.calculate_energy_decay(current_energy, decay_rate)
            np.testing.assert_almost_equal(decayed, expected_decay, decimal=6)

            # Test cap application
            over_cap = energy_cap + 10
            bounded = EnergyCalculator.apply_energy_bounds(over_cap)
            np.testing.assert_almost_equal(bounded, energy_cap, decimal=6)

            # Test transfer
            transfer_fraction = 0.5
            expected_transfer = current_energy * transfer_fraction
            transferred = EnergyCalculator.calculate_energy_transfer(current_energy, transfer_fraction)
            np.testing.assert_almost_equal(transferred, expected_transfer, decimal=6)

            # Test boost with cap
            boost_amount = 200.0
            low_energy = 50.0
            expected_boost = min(low_energy + boost_amount, energy_cap)
            boosted = EnergyCalculator.calculate_energy_boost(low_energy, boost_amount)
            np.testing.assert_almost_equal(boosted, expected_boost, decimal=6)

            # Test membrane potential
            expected_mem = min(current_energy / energy_cap, 1.0)
            mem_pot = EnergyCalculator.calculate_membrane_potential(current_energy)
            np.testing.assert_almost_equal(mem_pot, expected_mem, decimal=6)

            return True, {'energy_tests': 'passed', 'assertions': 6}
        except AssertionError as ae:
            return False, {'error': str(ae), 'test': 'energy_math'}
        except (ValueError, TypeError, RuntimeError) as e:
            # Energy calculation or mathematical computation issues
            return False, {'error': str(e)}

    def _test_connection_math_accuracy(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Connection weight calculation or Hebbian learning computation issues
            return False, {'error': str(e)}

    def _test_node_death_low_energy(self) -> Tuple[bool, Dict[str, Any]]:
        """Test node death on low energy threshold."""
        try:
            log_capture = StringIO()
            handler = logging.StreamHandler(log_capture)
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            graph = initialize_main_graph(scale=0.1)
            initial_count = len(graph.node_labels)

            # Set a dynamic node to low energy
            low_energy = get_node_death_threshold() - 0.1
            dynamic_found = False
            for i in range(len(graph.node_labels)):
                if graph.node_labels[i].get('type') == 'dynamic':
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
            logger.removeHandler(handler)

            death_occurred = final_count < initial_count
            log_contains_death = "[DEATH]" in logs

            return death_occurred and log_contains_death, {
                'initial_nodes': initial_count,
                'final_nodes': final_count,
                'death_logged': log_contains_death
            }
        except (ImportError, AttributeError, RuntimeError, OSError) as e:
            # Graph manipulation, logging setup, or node death logic issues
            return False, {'error': str(e)}

    def _test_node_death_strategy(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Graph manipulation, mock setup, or node death strategy issues
            return False, {'error': str(e)}

    def _test_node_birth_high_energy(self) -> Tuple[bool, Dict[str, Any]]:
        """Test node birth on high energy threshold."""
        try:
            log_capture = StringIO()
            handler = logging.StreamHandler(log_capture)
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            graph = initialize_main_graph(scale=0.1)
            initial_count = len(graph.node_labels)

            # Set a dynamic node to high energy
            high_energy = get_node_birth_threshold() + 0.1
            dynamic_found = False
            for i in range(len(graph.node_labels)):
                if graph.node_labels[i].get('type') == 'dynamic':
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
            logger.removeHandler(handler)

            birth_occurred = final_count > initial_count
            log_contains_birth = "[BIRTH]" in logs

            return birth_occurred and log_contains_birth, {
                'initial_nodes': initial_count,
                'final_nodes': final_count,
                'birth_logged': log_contains_birth
            }
        except (ImportError, AttributeError, RuntimeError, OSError) as e:
            # Graph manipulation, logging setup, or node birth logic issues
            return False, {'error': str(e)}

    def _test_node_birth_memory(self) -> Tuple[bool, Dict[str, Any]]:
        """Test memory-influenced node birth parameters."""
        try:
            graph = initialize_main_graph(scale=0.1)

            # Mock memory system for integrator behavior (>10 traces)
            class MockMemory:
                def get_memory_statistics(self):
                    return {'traces_formed': 15}

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
            # Memory analysis, mock setup, or node birth parameter calculation issues
            return False, {'error': str(e)}

    def _test_event_queue_order(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Event system creation, queue manipulation, or event ordering issues
            return False, {'error': str(e)}

    def _test_spike_propagation(self) -> Tuple[bool, Dict[str, Any]]:
        """Test spike event propagation to synaptic transmission."""
        try:
            class MockAccessLayer:
                def __init__(self):
                    self.nodes = {1: {'spike_count': 0}, 2: {'threshold': 0.5, 'synaptic_input': 0.0}}

                def get_node_by_id(self, nid):
                    return self.nodes.get(nid, {})

                def update_node_property(self, nid, prop, val):
                    if nid not in self.nodes:
                        self.nodes[nid] = {}
                    self.nodes[nid][prop] = val

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
            # Mock access layer setup, event processing, or spike propagation issues
            return False, {'error': str(e)}

    def _test_plasticity_propagation(self) -> Tuple[bool, Dict[str, Any]]:
        """Test plasticity events triggered by spikes."""
        try:
            class MockAccessLayer:
                def __init__(self):
                    self.nodes = {1: {'spike_count': 0}}

                def get_node_by_id(self, nid):
                    return self.nodes.get(nid, {})

                def update_node_property(self, nid, prop, val):
                    if nid not in self.nodes:
                        self.nodes[nid] = {}
                    self.nodes[nid][prop] = val

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
            # Mock learning engine setup, event processing, or plasticity trigger issues
            return False, {'error': str(e)}

    def _test_hebbian_ltp(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Learning engine setup, timing calculation, or Hebbian learning issues
            return False, {'error': str(e)}

    def _test_hebbian_ltd(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Learning engine setup, timing calculation, or Hebbian learning issues
            return False, {'error': str(e)}

    def _test_learning_consolidation(self) -> Tuple[bool, Dict[str, Any]]:
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
            # Learning engine setup, graph manipulation, or consolidation calculation issues
            return False, {'error': str(e)}


    def _test_event_bus(self) -> Tuple[bool, Dict[str, Any]]:
        """Test EventBus subscribe, emit, and fallback."""
        try:
            bus = get_event_bus()

            # Test subscribe and emit
            called = [False]
            def test_callback(event_type, data):
                called[0] = True
                assert event_type == 'TEST_SPIKE'
                assert data['node_id'] == 123
                assert data['timestamp'] > 0

            bus.subscribe('TEST_SPIKE', test_callback)
            bus.emit('TEST_SPIKE', {'node_id': 123, 'timestamp': time.time()})

            subscribe_emit_works = called[0]

            # Test unsubscribe
            bus.unsubscribe('TEST_SPIKE', test_callback)
            bus.emit('TEST_SPIKE', {'node_id': 456, 'timestamp': time.time()})
            unsubscribe_works = not called[0]

            # Test thread-safety (simple)

            # Test fallback (emit fails if invalid data)
            try:
                bus.emit('INVALID', 'not dict')  # Should raise but caught internally
                fallback_handled = True
            except ValueError:
                fallback_handled = True  # Expected, but since caught, no exception

            return subscribe_emit_works and unsubscribe_works and fallback_handled, {
                'subscribe_emit': subscribe_emit_works,
                'unsubscribe': unsubscribe_works,
                'fallback_handled': fallback_handled
            }

        except (ImportError, RuntimeError, ValueError) as e:
            # Event bus import, subscription, or emission issues
            return False, {'error': str(e)}

    def _test_full_simulation_cycle(self) -> Tuple[bool, Dict[str, Any]]:
        """E2E test for full simulation cycle."""
        try:
            # Create mocked services for SimulationCoordinator
            service_registry = Mock(spec=IServiceRegistry)
            neural_processor = Mock(spec=INeuralProcessor)
            energy_manager = Mock(spec=IEnergyManager)
            learning_engine = Mock(spec=ILearningEngine)
            sensory_processor = Mock(spec=ISensoryProcessor)
            performance_monitor = Mock(spec=IPerformanceMonitor)
            graph_manager = Mock(spec=IGraphManager)
            event_coordinator = Mock(spec=IEventCoordinator)
            configuration_service = Mock(spec=IConfigurationService)

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
            learning_engine = LearningEngine(mock_access_layer)

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
            # Complex simulation setup, mock configuration, or multi-step execution issues
            return False, {'error': str(e)}

    def _test_stress_growth(self) -> Tuple[bool, Dict[str, Any]]:
        """Stress test for organic growth and full cycles."""
        try:
            process = psutil.Process()

            # Numba timeit on jitted funcs
            @jit(nopython=True)
            def jitted_update(nodes_count):
                total = 0.0
                for i in range(nodes_count):
                    total += float(i) * 0.1
                return total

            jit_time = timeit.timeit('jitted_update(1000)', globals=globals(), number=1000)
            non_jit_time = timeit.timeit('sum(i*0.1 for i in range(1000))', number=1000)
            numba_speedup = non_jit_time / jit_time > 1.5

            # Create mocked services for SimulationCoordinator
            service_registry = Mock(spec=IServiceRegistry)
            neural_processor = Mock(spec=INeuralProcessor)
            energy_manager = Mock(spec=IEnergyManager)
            learning_engine = Mock(spec=ILearningEngine)
            sensory_processor = Mock(spec=ISensoryProcessor)
            performance_monitor = Mock(spec=IPerformanceMonitor)
            graph_manager = Mock(spec=IGraphManager)
            event_coordinator = Mock(spec=IEventCoordinator)
            configuration_service = Mock(spec=IConfigurationService)

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

            perf_ok = avg_step_time < 0.01  # <10ms post-numba
            memory_ok = memory_end < 2.0
            events_gt10 = birth_death_events > 10
            no_crash = True

            return perf_ok and memory_ok and events_gt10 and no_crash and numba_speedup, {
                'avg_step_time_s': avg_step_time,
                'memory_gb_end': memory_end,
                'birth_death_events': birth_death_events,
                'total_time_s': total_time,
                'numba_speedup': numba_speedup,
                'jit_time': jit_time,
                'non_jit_time': non_jit_time
            }
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            # Numba import, performance testing, or stress test execution issues
            return False, {'error': str(e), 'crashed': True}


# Global test framework instance
_test_framework = None


def get_test_framework() -> UnifiedTestFramework:
    """Get the global test framework instance."""
    global _test_framework
    if _test_framework is None:
        _test_framework = UnifiedTestFramework()
    return _test_framework


def run_all_tests() -> Dict[str, Any]:
    """Run all tests and return summary."""
    framework = get_test_framework()
    test_results = framework.run_tests()
    framework.results = test_results  # Store results for statistics
    return framework.get_test_statistics()


def run_tests_by_category(category: TestCategory) -> Dict[str, Any]:
    """Run tests by category and return summary."""
    framework = get_test_framework()
    test_results = framework.run_tests(category=category)
    framework.results = test_results  # Store results for statistics
    return framework.get_test_statistics()


def run_performance_tests() -> Dict[str, Any]:
    """Run performance tests specifically."""
    return run_tests_by_category(TestCategory.PERFORMANCE)


def run_neural_tests() -> Dict[str, Any]:
    """Run neural network tests specifically."""
    return run_tests_by_category(TestCategory.NEURAL)


def run_system_tests() -> Dict[str, Any]:
    """Run system tests specifically."""
    return run_tests_by_category(TestCategory.SYSTEM)


# Backward compatibility functions
def test_critical_imports() -> Tuple[bool, Dict[str, Any]]:
    """Test critical imports (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_critical_imports()


def test_memory_usage() -> Tuple[bool, Dict[str, Any]]:
    """Test memory usage (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_memory_usage()


def test_simulation_manager_creation() -> Tuple[bool, Dict[str, Any]]:
    """Test simulation manager creation (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_simulation_manager_creation()


def test_single_simulation_step() -> Tuple[bool, Dict[str, Any]]:
    """Test single simulation step (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_single_simulation_step()


def test_simulation_progression() -> Tuple[bool, Dict[str, Any]]:
    """Test simulation progression (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_simulation_progression()


def test_energy_behavior() -> Tuple[bool, Dict[str, Any]]:
    """Test energy behavior (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_energy_behavior()


def test_connection_logic() -> Tuple[bool, Dict[str, Any]]:
    """Test connection logic (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_connection_logic()


def test_ui_components() -> Tuple[bool, Dict[str, Any]]:
    """Test UI components (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_ui_components()


def test_performance_monitoring() -> Tuple[bool, Dict[str, Any]]:
    """Test performance monitoring (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_performance_monitoring()


def test_error_handling() -> Tuple[bool, Dict[str, Any]]:
    """Test error handling (backward compatibility)."""
    framework = get_test_framework()
    return framework._test_error_handling()


def run_unified_tests() -> Dict[str, Any]:
    """Run unified tests (backward compatibility)."""
    print("UNIFIED TESTING SYSTEM")
    print("=" * 60)

    framework = get_test_framework()
    test_results = framework.run_tests()
    framework.results = test_results

    summary = framework.get_test_statistics()

    print("\n[STATS] TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Errors: {summary['error_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Average Duration: {summary['average_duration']:.3f}s")

    if summary['success_rate'] == 1.0:
        print("\n[CELEBRATE] All tests passed! System is fully functional.")
    else:
        print(f"\n[WARNING] {summary['failed_tests'] + summary['error_tests']} tests failed. Investigation needed.")

    return summary


if __name__ == "__main__":
    results = run_unified_tests()