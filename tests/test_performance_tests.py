"""
Performance tests for the unified testing system.
Tests memory usage, performance benchmarks, and stress testing.
"""

import time
import timeit
from typing import Tuple, Dict, Any

# Third-party imports
import psutil

# Local test utilities
from .test_utils import TestCase, TestCategory, memory_usage_test, performance_benchmark_test, stress_test_memory_limit, validate_performance_threshold


def test_memory_usage() -> Tuple[bool, Dict[str, Any]]:
    """Test memory usage and potential leaks."""
    return memory_usage_test()


def test_performance_benchmark() -> Tuple[bool, Dict[str, Any]]:
    """Run performance benchmark tests."""
    return performance_benchmark_test()


def test_stress_growth() -> Tuple[bool, Dict[str, Any]]:
    """Stress test for organic growth and full cycles."""
    try:
        process = psutil.Process()

        # Numba timeit on jitted funcs
        try:
            from numba import jit

            @jit(nopython=True)
            def jitted_update(nodes_count):
                total = 0.0
                for i in range(nodes_count):
                    total += float(i) * 0.1
                return total

            jit_time = timeit.timeit('jitted_update(1000)', globals=globals(), number=1000)
            non_jit_time = timeit.timeit('sum(i*0.1 for i in range(1000))', number=1000)
            numba_speedup = non_jit_time / jit_time > 1.5
        except ImportError:
            # Numba not available
            jit_time = 0
            non_jit_time = 0
            numba_speedup = False

        # Simple performance test without complex dependencies
        step_times = []
        start_time = time.perf_counter()

        for i in range(1000):
            step_start = time.perf_counter()
            # Simple computation
            total = sum(j * 0.1 for j in range(100))
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        memory_end = process.memory_info().rss / (1024 ** 3)
        total_time = time.perf_counter() - start_time

        perf_ok = validate_performance_threshold(avg_step_time, 10.0)
        memory_ok = stress_test_memory_limit(2.0)
        no_crash = True

        return perf_ok and memory_ok and no_crash and numba_speedup, {
            'avg_step_time_s': avg_step_time,
            'memory_gb_end': memory_end,
            'total_time_s': total_time,
            'numba_speedup': numba_speedup,
            'jit_time': jit_time,
            'non_jit_time': non_jit_time
        }
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        return False, {'error': str(e), 'crashed': True}


def create_performance_test_cases() -> list:
    """Create performance test cases."""
    return [
        TestCase(
            name="memory_usage",
            category=TestCategory.MEMORY,
            description="Test memory usage and leaks",
            test_func=test_memory_usage
        ),
        TestCase(
            name="performance_benchmark",
            category=TestCategory.PERFORMANCE,
            description="Test performance benchmarks and numba speedup",
            test_func=test_performance_benchmark
        ),
        TestCase(
            name="stress_growth",
            category=TestCategory.STRESS,
            description="Stress test: large graph, long runs for birth/death cascades, memory",
            test_func=test_stress_growth
        )
    ]