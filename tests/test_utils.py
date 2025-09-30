"""
Test utilities and data structures for the unified testing system.
Contains enums, dataclasses, and utility functions.
"""

import gc
import logging
import threading
import time
import timeit
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple
from numba import jit
import psutil

# Third-party imports
import numpy as np


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


def calculate_test_metrics() -> TestMetrics:
    """Calculate current test metrics."""
    duration = time.time()
    memory_used_mb = psutil.Process().memory_info().rss / 1024 / 1024
    cpu_percent = psutil.cpu_percent()
    gc_collections = sum(stat['collections'] for stat in gc.get_stats())

    return TestMetrics(
        duration=duration,
        memory_used_mb=memory_used_mb,
        cpu_percent=cpu_percent,
        gc_collections=gc_collections,
        assertions_count=0,
        error_count=0
    )


def setup_logging_capture() -> Tuple[StringIO, logging.Handler]:
    """Setup logging capture for testing."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return log_capture, handler


def cleanup_logging_capture(handler: logging.Handler):
    """Clean up logging capture."""
    logger = logging.getLogger()
    logger.removeHandler(handler)


def calculate_category_statistics(results: List[TestExecutionResult]) -> Dict[str, Dict[str, int]]:
    """Calculate statistics by test category."""
    category_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0})

    for result in results:
        category = result.test_case.category
        category_stats[category.value]['total'] += 1
        if result.result == TestResult.PASSED:
            category_stats[category.value]['passed'] += 1
        elif result.result == TestResult.FAILED:
            category_stats[category.value]['failed'] += 1
        else:
            category_stats[category.value]['errors'] += 1

    return dict(category_stats)


def calculate_overall_statistics(results: List[TestExecutionResult]) -> Dict[str, Any]:
    """Calculate overall test statistics."""
    if not results:
        return {}

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.result == TestResult.PASSED)
    failed_tests = sum(1 for r in results if r.result == TestResult.FAILED)
    error_tests = sum(1 for r in results if r.result == TestResult.ERROR)

    total_duration = sum(r.duration for r in results)
    average_duration = total_duration / total_tests if total_tests > 0 else 0

    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'error_tests': error_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
        'total_duration': total_duration,
        'average_duration': average_duration,
        'category_breakdown': calculate_category_statistics(results)
    }


def format_test_summary(summary: Dict[str, Any]) -> str:
    """Format test summary for display."""
    lines = []
    lines.append("[STATS] TEST SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Total Tests: {summary['total_tests']}")
    lines.append(f"Passed: {summary['passed_tests']}")
    lines.append(f"Failed: {summary['failed_tests']}")
    lines.append(f"Errors: {summary['error_tests']}")
    lines.append(f"Success Rate: {summary['success_rate']:.1%}")
    lines.append(f"Total Duration: {summary['total_duration']:.2f}s")
    lines.append(f"Average Duration: {summary['average_duration']:.3f}s")
    return "\n".join(lines)


def validate_test_case(test_case: TestCase) -> bool:
    """Validate a test case configuration."""
    if not test_case.name or not test_case.test_func:
        return False

    if test_case.timeout <= 0 or test_case.expected_duration <= 0:
        return False

    if test_case.memory_limit_mb <= 0 or test_case.cpu_limit_percent <= 0:
        return False

    return True


def create_test_fixture(setup_func: Callable, teardown_func: Optional[Callable] = None) -> Dict[str, Callable]:
    """Create a test fixture."""
    return {
        'setup': setup_func,
        'teardown': teardown_func
    }


def run_with_timeout(func: Callable, timeout: float, default_return: Any = None) -> Any:
    """Run a function with a timeout."""
    result = {'value': default_return, 'exception': None, 'completed': False}

    def target():
        try:
            result['value'] = func()
        except (RuntimeError, ValueError, TypeError) as e:
            result['exception'] = e
        finally:
            result['completed'] = True

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if not result['completed']:
        return default_return

    if result['exception'] is not None:
        raise result['exception']  # pylint: disable=raising-bad-type

    if not result['completed']:
        raise RuntimeError(f"Function did not complete within {timeout} seconds")

    return result['value']


def memory_usage_test() -> Tuple[bool, Dict[str, Any]]:
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
        return False, {'error': str(e)}


def performance_benchmark_test() -> Tuple[bool, Dict[str, Any]]:
    """Run performance benchmark tests."""
    try:
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

        return numba_speedup, {
            'jit_time': jit_time,
            'non_jit_time': non_jit_time,
            'speedup_ratio': non_jit_time / jit_time if jit_time > 0 else 0
        }
    except (ImportError, RuntimeError) as e:
        return False, {'error': str(e)}


def stress_test_memory_limit(memory_limit_gb: float = 2.0) -> bool:
    """Check if memory usage exceeds limit during stress testing."""
    try:
        current_memory_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        return current_memory_gb <= memory_limit_gb
    except (OSError, psutil.Error):
        return False


def validate_performance_threshold(step_time: float, threshold_ms: float = 10.0) -> bool:
    """Validate if step time meets performance threshold."""
    return step_time < (threshold_ms / 1000.0)  # Convert ms to seconds


def create_mock_edge_for_testing(eligibility_trace: float = 0.0, weight: float = 1.0) -> Any:
    """Create a mock edge for testing purposes."""
    class MockEdge:
        """Mock edge class for testing purposes."""
        def __init__(self, trace, weight_val):
            self.eligibility_trace = trace
            self.weight = weight_val

        def get_weight(self) -> float:
            """Get the edge weight."""
            return self.weight

        def get_trace(self) -> float:
            """Get the eligibility trace."""
            return self.eligibility_trace

    return MockEdge(trace=eligibility_trace, weight_val=weight)


def create_mock_node_for_testing(node_id: int, node_type: str = 'dynamic') -> Dict[str, Any]:
    """Create a mock node for testing purposes."""
    return {
        'id': node_id,
        'type': node_type,
        'energy': 100.0,
        'state': 'active'
    }
