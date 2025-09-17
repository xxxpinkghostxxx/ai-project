"""
Comprehensive Testing Framework for the Neural Simulation System.
Provides unit tests, integration tests, performance tests, and test utilities.
"""

import unittest
import time
import threading
import gc
import psutil
from typing import Dict, List, Any, Optional, Callable, Tuple, Type
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import numpy as np
import torch
from torch_geometric.data import Data

class TestCategory(Enum):
    """Test categories for organization."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    STRESS = "stress"
    MEMORY = "memory"
    FUNCTIONAL = "functional"

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
        self.step_counter = 0
        self.components = {}
        self.callbacks = []
        self.performance_stats = {}
    
    def initialize_graph(self):
        """Initialize a test graph."""
        self.graph = self._create_test_graph()
    
    def _create_test_graph(self) -> Data:
        """Create a test graph for testing."""
        num_nodes = 100
        num_edges = 200
        
        # Create node features
        x = torch.randn(num_nodes, 5)
        
        # Create edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Create node labels
        node_labels = []
        for i in range(num_nodes):
            node_labels.append({
                'id': i,
                'type': 'dynamic',
                'energy': np.random.random(),
                'state': 'inactive',
                'membrane_potential': 0.0,
                'threshold': 0.5,
                'refractory_timer': 0.0,
                'last_activation': 0,
                'plasticity_enabled': True,
                'eligibility_trace': 0.0,
                'last_update': 0
            })
        
        return Data(x=x, edge_index=edge_index, node_labels=node_labels)
    
    def run_single_step(self) -> bool:
        """Simulate a single simulation step."""
        if self.graph is None:
            return False
        
        self.step_counter += 1
        
        # Simulate some processing
        time.sleep(0.01)
        
        # Update node states
        for node in self.graph.node_labels:
            node['energy'] = max(0, node['energy'] - 0.01)
            node['last_update'] = self.step_counter
        
        return True
    
    def start_simulation(self):
        """Start simulation."""
        self.is_running = True
    
    def stop_simulation(self):
        """Stop simulation."""
        self.is_running = False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'step_counter': self.step_counter,
            'node_count': len(self.graph.node_labels) if self.graph else 0,
            'edge_count': self.graph.edge_index.shape[1] if self.graph else 0,
            'is_running': self.is_running
        }

class TestFramework:
    """Comprehensive testing framework."""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.test_results: List[TestResult] = []
        self.test_suites: Dict[str, List[TestCase]] = defaultdict(list)
        self.fixtures: Dict[str, Any] = {}
        self.running = False
        self._lock = threading.RLock()
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case to the framework."""
        with self._lock:
            self.test_cases.append(test_case)
            self.test_suites[test_case.category.value].append(test_case)
    
    def add_fixture(self, name: str, fixture_func: Callable):
        """Add a test fixture."""
        self.fixtures[name] = fixture_func
    
    def run_tests(self, category: Optional[TestCategory] = None, 
                  parallel: bool = False) -> List[TestResult]:
        """Run tests, optionally filtered by category."""
        with self._lock:
            self.running = True
            self.test_results.clear()
            
            # Filter test cases by category
            if category:
                test_cases = [tc for tc in self.test_cases if tc.category == category]
            else:
                test_cases = self.test_cases
            
            if parallel:
                self._run_tests_parallel(test_cases)
            else:
                self._run_tests_sequential(test_cases)
            
            self.running = False
            return self.test_results.copy()
    
    def _run_tests_sequential(self, test_cases: List[TestCase]):
        """Run tests sequentially."""
        for test_case in test_cases:
            result = self._execute_test(test_case)
            self.test_results.append(result)
    
    def _run_tests_parallel(self, test_cases: List[TestCase]):
        """Run tests in parallel."""
        threads = []
        results_lock = threading.Lock()
        
        def run_test_wrapper(test_case):
            result = self._execute_test(test_case)
            with results_lock:
                self.test_results.append(result)
        
        for test_case in test_cases:
            thread = threading.Thread(target=run_test_wrapper, args=(test_case,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    
    def _execute_test(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_gc = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))
        
        result = TestResult(
            test_case=test_case,
            result=TestResult.PASSED,
            duration=0.0,
            metrics=TestMetrics(
                duration=0.0,
                memory_used_mb=0.0,
                cpu_percent=0.0,
                gc_collections=0,
                assertions_count=0,
                error_count=0
            )
        )
        
        try:
            # Setup
            if test_case.setup_func:
                test_case.setup_func()
            
            # Execute test with timeout
            test_thread = threading.Thread(target=test_case.test_func)
            test_thread.start()
            test_thread.join(timeout=test_case.timeout)
            
            if test_thread.is_alive():
                result.result = TestResult.ERROR
                result.error = TimeoutError(f"Test {test_case.name} timed out after {test_case.timeout}s")
            else:
                result.result = TestResult.PASSED
        
        except Exception as e:
            result.result = TestResult.FAILED
            result.error = e
        
        finally:
            # Teardown
            if test_case.teardown_func:
                try:
                    test_case.teardown_func()
                except Exception as e:
                    print(f"Teardown error in {test_case.name}: {e}")
            
            # Calculate metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_gc = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))
            
            result.duration = end_time - start_time
            result.metrics.duration = result.duration
            result.metrics.memory_used_mb = end_memory - start_memory
            result.metrics.gc_collections = end_gc - start_gc
            
            # Check limits
            if result.metrics.memory_used_mb > test_case.memory_limit_mb:
                result.result = TestResult.FAILED
                result.error = MemoryError(f"Memory usage exceeded limit: {result.metrics.memory_used_mb:.2f}MB > {test_case.memory_limit_mb}MB")
        
        return result
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get test execution statistics."""
        with self._lock:
            if not self.test_results:
                return {}
            
            total_tests = len(self.test_results)
            passed = sum(1 for r in self.test_results if r.result == TestResult.PASSED)
            failed = sum(1 for r in self.test_results if r.result == TestResult.FAILED)
            errors = sum(1 for r in self.test_results if r.result == TestResult.ERROR)
            skipped = sum(1 for r in self.test_results if r.result == TestResult.SKIPPED)
            
            total_duration = sum(r.duration for r in self.test_results)
            avg_duration = total_duration / total_tests if total_tests > 0 else 0
            
            # Category breakdown
            category_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0})
            for result in self.test_results:
                category = result.test_case.category.value
                category_stats[category]['total'] += 1
                if result.result == TestResult.PASSED:
                    category_stats[category]['passed'] += 1
                elif result.result == TestResult.FAILED:
                    category_stats[category]['failed'] += 1
                elif result.result == TestResult.ERROR:
                    category_stats[category]['errors'] += 1
            
            return {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'skipped': skipped,
                'success_rate': passed / total_tests if total_tests > 0 else 0,
                'total_duration': total_duration,
                'average_duration': avg_duration,
                'category_stats': dict(category_stats)
            }

# Predefined test cases
def create_basic_tests() -> List[TestCase]:
    """Create basic test cases for the simulation system."""
    tests = []
    
    # Test simulation manager creation
    def test_simulation_manager_creation():
        manager = MockSimulationManager()
        assert manager is not None
        assert not manager.is_running
        assert manager.step_counter == 0
    
    tests.append(TestCase(
        name="test_simulation_manager_creation",
        category=TestCategory.UNIT,
        description="Test simulation manager creation",
        test_func=test_simulation_manager_creation
    ))
    
    # Test graph initialization
    def test_graph_initialization():
        manager = MockSimulationManager()
        manager.initialize_graph()
        assert manager.graph is not None
        assert len(manager.graph.node_labels) > 0
        assert manager.graph.edge_index.shape[1] > 0
    
    tests.append(TestCase(
        name="test_graph_initialization",
        category=TestCategory.UNIT,
        description="Test graph initialization",
        test_func=test_graph_initialization
    ))
    
    # Test single step execution
    def test_single_step_execution():
        manager = MockSimulationManager()
        manager.initialize_graph()
        initial_step = manager.step_counter
        success = manager.run_single_step()
        assert success
        assert manager.step_counter == initial_step + 1
    
    tests.append(TestCase(
        name="test_single_step_execution",
        category=TestCategory.UNIT,
        description="Test single step execution",
        test_func=test_single_step_execution
    ))
    
    # Test performance metrics
    def test_performance_metrics():
        manager = MockSimulationManager()
        manager.initialize_graph()
        stats = manager.get_performance_stats()
        assert 'step_counter' in stats
        assert 'node_count' in stats
        assert 'edge_count' in stats
        assert stats['node_count'] > 0
        assert stats['edge_count'] > 0
    
    tests.append(TestCase(
        name="test_performance_metrics",
        category=TestCategory.UNIT,
        description="Test performance metrics",
        test_func=test_performance_metrics
    ))
    
    return tests

def create_integration_tests() -> List[TestCase]:
    """Create integration test cases."""
    tests = []
    
    # Test simulation loop
    def test_simulation_loop():
        manager = MockSimulationManager()
        manager.initialize_graph()
        manager.start_simulation()
        
        # Run multiple steps
        for _ in range(10):
            success = manager.run_single_step()
            assert success
        
        manager.stop_simulation()
        assert not manager.is_running
        assert manager.step_counter == 10
    
    tests.append(TestCase(
        name="test_simulation_loop",
        category=TestCategory.INTEGRATION,
        description="Test complete simulation loop",
        test_func=test_simulation_loop,
        timeout=10.0
    ))
    
    return tests

def create_performance_tests() -> List[TestCase]:
    """Create performance test cases."""
    tests = []
    
    # Test memory usage
    def test_memory_usage():
        manager = MockSimulationManager()
        manager.initialize_graph()
        
        # Run many steps to test memory usage
        for _ in range(100):
            manager.run_single_step()
        
        # Check that memory usage is reasonable
        stats = manager.get_performance_stats()
        assert stats['step_counter'] == 100
    
    tests.append(TestCase(
        name="test_memory_usage",
        category=TestCategory.PERFORMANCE,
        description="Test memory usage during simulation",
        test_func=test_memory_usage,
        memory_limit_mb=50.0
    ))
    
    # Test execution speed
    def test_execution_speed():
        manager = MockSimulationManager()
        manager.initialize_graph()
        
        start_time = time.time()
        for _ in range(100):
            manager.run_single_step()
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 5.0  # Should complete in less than 5 seconds
    
    tests.append(TestCase(
        name="test_execution_speed",
        category=TestCategory.PERFORMANCE,
        description="Test execution speed",
        test_func=test_execution_speed,
        expected_duration=2.0
    ))
    
    return tests

def create_stress_tests() -> List[TestCase]:
    """Create stress test cases."""
    tests = []
    
    # Test high load
    def test_high_load():
        manager = MockSimulationManager()
        manager.initialize_graph()
        
        # Run many steps under load
        for _ in range(1000):
            manager.run_single_step()
        
        assert manager.step_counter == 1000
    
    tests.append(TestCase(
        name="test_high_load",
        category=TestCategory.STRESS,
        description="Test high load simulation",
        test_func=test_high_load,
        timeout=60.0,
        memory_limit_mb=200.0
    ))
    
    return tests

# Global test framework instance
_test_framework = None

def get_test_framework() -> TestFramework:
    """Get the global test framework instance."""
    global _test_framework
    if _test_framework is None:
        _test_framework = TestFramework()
        
        # Add default test cases
        _test_framework.test_cases.extend(create_basic_tests())
        _test_framework.test_cases.extend(create_integration_tests())
        _test_framework.test_cases.extend(create_performance_tests())
        _test_framework.test_cases.extend(create_stress_tests())
    
    return _test_framework

def run_all_tests() -> Dict[str, Any]:
    """Run all tests and return results."""
    framework = get_test_framework()
    results = framework.run_tests()
    statistics = framework.get_test_statistics()
    
    return {
        'results': results,
        'statistics': statistics
    }

def run_tests_by_category(category: TestCategory) -> Dict[str, Any]:
    """Run tests by category."""
    framework = get_test_framework()
    results = framework.run_tests(category=category)
    statistics = framework.get_test_statistics()
    
    return {
        'results': results,
        'statistics': statistics
    }

if __name__ == "__main__":
    # Run tests when script is executed directly
    print("Running Neural Simulation Tests...")
    results = run_all_tests()
    
    print(f"\nTest Results:")
    print(f"Total Tests: {results['statistics']['total_tests']}")
    print(f"Passed: {results['statistics']['passed']}")
    print(f"Failed: {results['statistics']['failed']}")
    print(f"Errors: {results['statistics']['errors']}")
    print(f"Success Rate: {results['statistics']['success_rate']:.2%}")
    print(f"Total Duration: {results['statistics']['total_duration']:.2f}s")
