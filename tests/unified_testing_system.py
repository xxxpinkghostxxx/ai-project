"""
Unified Testing System for the Neural Simulation.
Consolidates unified_test_suite.py and comprehensive_test_framework.py
into a comprehensive testing framework.
"""

import gc
import logging
import os
import psutil
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from src.energy.energy_behavior import apply_energy_behavior, get_node_energy_cap
# Local imports - Learning modules
from src.learning.learning_engine import LearningEngine
# Local imports - Neural modules
from src.neural.connection_logic import create_basic_connections
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

# Local test utilities
from .test_utils import (
    TestCategory, TestResult, TestMetrics, TestCase, TestExecutionResult,
    calculate_test_metrics, setup_logging_capture, cleanup_logging_capture,
    calculate_category_statistics, calculate_overall_statistics, format_test_summary,
    validate_test_case, create_test_fixture, run_with_timeout
)
from .test_mocks import (
    MockSimulationCoordinator, MockAccessLayer, MockMemory, MockEdge, MockGraph,
    create_mock_services, configure_mock_services_for_init, configure_mock_services_for_execution
)
from .test_system_tests import create_system_test_cases
from .test_neural_tests import create_neural_test_cases
from .test_integration_tests import create_integration_test_cases
from .test_performance_tests import create_performance_test_cases

# Setup sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




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
        # Add test cases from separate modules
        for test_case in create_system_test_cases():
            self.add_test_case(test_case)

        for test_case in create_neural_test_cases():
            self.add_test_case(test_case)

        for test_case in create_integration_test_cases():
            self.add_test_case(test_case)

        for test_case in create_performance_test_cases():
            self.add_test_case(test_case)

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
        test_results = []
        for test_case in test_cases:
            result = self._execute_test(test_case)
            test_results.append(result)
        return test_results

    def _run_tests_parallel(self, test_cases: List[TestCase]) -> List[TestExecutionResult]:
        """Run tests in parallel using threading."""
        test_results = []
        threads = []

        def run_test_wrapper(test_case):
            result = self._execute_test(test_case)
            with self._lock:
                test_results.append(result)

        for test_case in test_cases:
            thread = threading.Thread(target=run_test_wrapper, args=(test_case,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return test_results

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



# Global test framework instance
_TEST_FRAMEWORK = None


def get_test_framework() -> UnifiedTestFramework:
    """Get the global test framework instance."""
    global _TEST_FRAMEWORK
    if _TEST_FRAMEWORK is None:
        _TEST_FRAMEWORK = UnifiedTestFramework()
    return _TEST_FRAMEWORK


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
