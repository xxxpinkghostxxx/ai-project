"""
Unified Testing System for the Neural Simulation.
Consolidates unified_test_suite.py and comprehensive_test_framework.py
into a comprehensive testing framework.
"""

import unittest
import time
import threading
import gc
import psutil
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple, Type
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import numpy as np
import torch
from torch_geometric.data import Data
from utils.print_utils import print_info, print_success, print_error, print_warning
from utils.logging_utils import log_step


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
class TestResult:
    """Test execution result."""
    test_case: TestCase
    result: TestResult
    duration: float
    error: Optional[Exception] = None
    metrics: Optional[TestMetrics] = None
    output: str = ""
    assertions: List[str] = field(default_factory=list)


class MockSimulationManager:
    """Mock simulation manager for testing."""
    
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
        self.results: List[TestResult] = []
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
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case to the framework."""
        with self._lock:
            self.test_cases.append(test_case)
    
    def add_fixture(self, name: str, fixture_func: Callable):
        """Add a test fixture."""
        self.fixtures[name] = fixture_func
    
    def run_tests(self, category: Optional[TestCategory] = None, 
                  parallel: bool = False) -> List[TestResult]:
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
    
    def _run_tests_sequential(self, test_cases: List[TestCase]) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        for test_case in test_cases:
            result = self._execute_test(test_case)
            results.append(result)
        return results
    
    def _run_tests_parallel(self, test_cases: List[TestCase]) -> List[TestResult]:
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
    
    def _execute_test(self, test_case: TestCase) -> TestResult:
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
                success, details = test_result
                result_status = TestResult.PASSED if success else TestResult.FAILED
            elif isinstance(test_result, bool):
                result_status = TestResult.PASSED if test_result else TestResult.FAILED
            else:
                result_status = TestResult.PASSED
            
            error = None
            
        except Exception as e:
            result_status = TestResult.ERROR
            error = e
            test_result = False
        
        finally:
            # Teardown
            if test_case.teardown_func:
                try:
                    test_case.teardown_func()
                except Exception as e:
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
        result = TestResult(
            test_case=test_case,
            result=result_status,
            duration=duration,
            error=error,
            metrics=metrics,
            output=str(test_result) if test_result else ""
        )
        
        # Log result
        status_symbol = "✅" if result_status == TestResult.PASSED else "❌"
        print_info(f"{status_symbol} {test_case.name}: {result_status.value} ({duration:.3f}s)")
        
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
            import simulation_manager
            import behavior_engine
            import energy_behavior
            import connection_logic
            import main_graph
            import node_access_layer
            import node_id_manager
            
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
        except Exception as e:
            return False, {'error': str(e)}
    
    def _test_simulation_manager_creation(self) -> Tuple[bool, Dict[str, Any]]:
        """Test simulation manager creation."""
        try:
            from simulation_manager import create_simulation_manager
            
            sim_manager = create_simulation_manager()
            
            return sim_manager is not None, {
                'manager_created': True,
                'type': type(sim_manager).__name__
            }
        except Exception as e:
            return False, {'error': str(e)}
    
    def _test_single_simulation_step(self) -> Tuple[bool, Dict[str, Any]]:
        """Test single simulation step execution."""
        try:
            from simulation_manager import create_simulation_manager
            from main_graph import initialize_main_graph
            
            sim_manager = create_simulation_manager()
            graph = initialize_main_graph(scale=0.25)
            sim_manager.set_graph(graph)
            
            success = sim_manager.run_single_step()
            
            return success, {
                'step_executed': success,
                'graph_nodes': len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
            }
        except Exception as e:
            return False, {'error': str(e)}
    
    def _test_simulation_progression(self) -> Tuple[bool, Dict[str, Any]]:
        """Test simulation progression over multiple steps."""
        try:
            from simulation_manager import create_simulation_manager
            from main_graph import initialize_main_graph
            
            sim_manager = create_simulation_manager()
            graph = initialize_main_graph(scale=0.25)
            sim_manager.set_graph(graph)
            
            steps_completed = 0
            for i in range(10):
                if sim_manager.run_single_step():
                    steps_completed += 1
                else:
                    break
            
            return steps_completed == 10, {
                'steps_completed': steps_completed,
                'expected_steps': 10
            }
        except Exception as e:
            return False, {'error': str(e)}
    
    def _test_energy_behavior(self) -> Tuple[bool, Dict[str, Any]]:
        """Test energy behavior and dynamics."""
        try:
            from energy_behavior import get_node_energy_cap, apply_energy_behavior
            from main_graph import initialize_main_graph
            
            graph = initialize_main_graph(scale=0.25)
            energy_cap = get_node_energy_cap()
            
            # Test energy behavior application
            updated_graph = apply_energy_behavior(graph)
            
            return updated_graph is not None, {
                'energy_cap': energy_cap,
                'graph_updated': updated_graph is not None
            }
        except Exception as e:
            return False, {'error': str(e)}
    
    def _test_connection_logic(self) -> Tuple[bool, Dict[str, Any]]:
        """Test connection logic and formation."""
        try:
            from connection_logic import create_basic_connections, get_edge_attributes
            from main_graph import initialize_main_graph
            
            graph = initialize_main_graph(scale=0.25)
            initial_edges = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0
            
            updated_graph = create_basic_connections(graph)
            final_edges = updated_graph.edge_index.shape[1] if hasattr(updated_graph, 'edge_index') else 0
            
            return final_edges >= initial_edges, {
                'initial_edges': initial_edges,
                'final_edges': final_edges,
                'connections_created': final_edges - initial_edges
            }
        except Exception as e:
            return False, {'error': str(e)}
    
    def _test_ui_components(self) -> Tuple[bool, Dict[str, Any]]:
        """Test UI components and functionality."""
        try:
            from ui_engine import create_main_window, update_ui_display
            from ui_state_manager import get_ui_state_manager
            
            ui_state = get_ui_state_manager()
            
            return ui_state is not None, {
                'ui_state_available': ui_state is not None
            }
        except Exception as e:
            return False, {'error': str(e)}
    
    def _test_performance_monitoring(self) -> Tuple[bool, Dict[str, Any]]:
        """Test performance monitoring systems."""
        try:
            from unified_performance_system import get_performance_monitor
            
            monitor = get_performance_monitor()
            metrics = monitor.get_current_metrics()
            
            return monitor is not None, {
                'monitor_available': monitor is not None,
                'metrics_available': metrics is not None
            }
        except Exception as e:
            return False, {'error': str(e)}
    
    def _test_error_handling(self) -> Tuple[bool, Dict[str, Any]]:
        """Test error handling and recovery."""
        try:
            from unified_error_handler import get_error_handler, safe_execute
            
            error_handler = get_error_handler()
            
            # Test safe execution
            def failing_function():
                raise ValueError("Test error")
            
            result = safe_execute(failing_function, "test_context", default_return="fallback")
            
            return result == "fallback", {
                'error_handler_available': error_handler is not None,
                'safe_execute_works': result == "fallback"
            }
        except Exception as e:
            return False, {'error': str(e)}


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
    results = framework.run_tests()
    framework.results = results  # Store results for statistics
    return framework.get_test_statistics()


def run_tests_by_category(category: TestCategory) -> Dict[str, Any]:
    """Run tests by category and return summary."""
    framework = get_test_framework()
    results = framework.run_tests(category=category)
    framework.results = results  # Store results for statistics
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
    print("🔍 UNIFIED TESTING SYSTEM")
    print("=" * 60)
    
    framework = get_test_framework()
    results = framework.run_tests()
    framework.results = results
    
    summary = framework.get_test_statistics()
    
    print(f"\n[STATS] TEST SUMMARY")
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
