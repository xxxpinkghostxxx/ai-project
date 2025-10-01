"""
Performance benchmarking system for neural simulation optimization.
Provides comprehensive testing and profiling capabilities.
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List

import psutil

from src.core.services.simulation_coordinator import SimulationCoordinator
from src.neural.optimized_node_manager import get_optimized_node_manager
from src.utils.performance_cache import get_performance_cache_manager

@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    duration: float
    memory_usage_mb: float
    cpu_usage_percent: float
    operations_per_second: float
    timestamp: datetime
    metadata: Dict[str, Any]

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        self._lock = threading.RLock()

    def benchmark_function(self, func: Callable, *args, test_name: str = None,
                          iterations: int = 100, warmup_iterations: int = 10,
                          **kwargs) -> BenchmarkResult:
        """
        Benchmark a function with comprehensive metrics.

        Args:
            func: Function to benchmark
            *args: Positional arguments for the function
            test_name: Name of the test (defaults to function name)
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            **kwargs: Keyword arguments for the function

        Returns:
            BenchmarkResult with comprehensive metrics
        """
        if test_name is None:
            test_name = func.__name__

        # Warmup phase
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Warmup iteration failed: %s", e)

        # Force garbage collection before benchmark
        gc.collect()

        # Benchmark phase
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        _start_cpu = psutil.cpu_percent(interval=None)

        successful_iterations = 0
        for i in range(iterations):
            try:
                func(*args, **kwargs)
                successful_iterations += 1
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Benchmark iteration %d failed: %s", i, e)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        end_cpu = psutil.cpu_percent(interval=None)

        # Calculate metrics
        total_duration = end_time - start_time
        avg_duration = total_duration / successful_iterations if successful_iterations > 0 else 0
        memory_usage = end_memory - start_memory
        operations_per_second = successful_iterations / total_duration if total_duration > 0 else 0

        result = BenchmarkResult(
            test_name=test_name,
            duration=total_duration,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=end_cpu,
            operations_per_second=operations_per_second,
            timestamp=datetime.now(),
            metadata={
                'iterations': successful_iterations,
                'avg_iteration_time': avg_duration,
                'total_iterations': iterations,
                'warmup_iterations': warmup_iterations,
                'success_rate': successful_iterations / iterations if iterations > 0 else 0
            }
        )

        with self._lock:
            self.results.append(result)

        logging.info("Benchmark '%s': %.2f ops/sec, %.2f MB, %.1f%% CPU",
                    test_name, operations_per_second, memory_usage, end_cpu)

        return result

    def benchmark_simulation_startup(self, simulation_manager_class, config: Dict[str, Any] = None) -> BenchmarkResult:
        """Benchmark simulation startup time."""
        def startup_test():
            manager = simulation_manager_class(config)
            # Force cleanup
            if hasattr(manager, 'cleanup'):
                manager.cleanup()

        return self.benchmark_function(
            startup_test,
            test_name="simulation_startup",
            iterations=5,
            warmup_iterations=2
        )

    def benchmark_node_operations(self, node_manager, num_nodes: int = 10000) -> Dict[str, BenchmarkResult]:
        """Benchmark various node operations."""
        results = {}

        # Benchmark node creation
        def create_nodes_test():
            specs = [
                {'type': 'dynamic', 'energy': 1.0, 'x': i % 100, 'y': i // 100}
                for i in range(num_nodes)
            ]
            node_manager.create_node_batch(specs)

        results['node_creation'] = self.benchmark_function(
            create_nodes_test,
            test_name=f"node_creation_{num_nodes}",
            iterations=3,
            warmup_iterations=1
        )

        # Benchmark node updates
        if hasattr(node_manager, 'active_nodes') and node_manager.active_nodes:
            node_ids = list(node_manager.active_nodes)[:min(1000, len(node_manager.active_nodes))]

            def update_nodes_test():
                updates = {'energy': 0.8, 'membrane_potential': 0.5}
                node_manager.update_nodes_batch(node_ids, updates)

            results['node_updates'] = self.benchmark_function(
                update_nodes_test,
                test_name=f"node_updates_{len(node_ids)}",
                iterations=10,
                warmup_iterations=3
            )

        return results

    def benchmark_caching_system(self, cache_manager, _num_operations: int = 10000) -> Dict[str, BenchmarkResult]:
        """Benchmark caching system performance."""
        results = {}

        # Prepare test data
        test_data = {f"node_{i}": {'energy': i / 1000.0, 'type': 'dynamic'} for i in range(1000)}

        # Benchmark cache puts
        def cache_put_test():
            for key, data in test_data.items():
                cache_manager.lru_cache.put(key, data)

        results['cache_put'] = self.benchmark_function(
            cache_put_test,
            test_name=f"cache_put_{len(test_data)}",
            iterations=5,
            warmup_iterations=2
        )

        # Benchmark cache gets
        def cache_get_test():
            for key in test_data.keys():
                cache_manager.lru_cache.get(key)

        results['cache_get'] = self.benchmark_function(
            cache_get_test,
            test_name=f"cache_get_{len(test_data)}",
            iterations=10,
            warmup_iterations=3
        )

        return results

    def compare_with_baseline(self, test_name: str, current_result: BenchmarkResult) -> Dict[str, Any]:
        """Compare current result with baseline."""
        if test_name not in self.baseline_results:
            return {'status': 'no_baseline', 'improvement': 0.0}

        baseline = self.baseline_results[test_name]

        # Calculate improvement (positive = better performance)
        if current_result.operations_per_second > 0 and baseline.operations_per_second > 0:
            ops_improvement = ((current_result.operations_per_second - baseline.operations_per_second) /
                             baseline.operations_per_second) * 100
        else:
            ops_improvement = 0.0

        memory_improvement = baseline.memory_usage_mb - current_result.memory_usage_mb

        return {
            'status': 'compared',
            'ops_improvement_percent': ops_improvement,
            'memory_improvement_mb': memory_improvement,
            'baseline_ops_per_sec': baseline.operations_per_second,
            'current_ops_per_sec': current_result.operations_per_second
        }

    def set_baseline(self, test_name: str, result: BenchmarkResult):
        """Set a baseline result for comparison."""
        with self._lock:
            self.baseline_results[test_name] = result

    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("NEURAL SIMULATION PERFORMANCE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary statistics
        if self.results:
            total_tests = len(self.results)
            _avg_ops_per_sec = sum(r.operations_per_second for r in self.results) / total_tests
            _total_memory = sum(r.memory_usage_mb for r in self.results)
            _avg_cpu = sum(r.cpu_usage_percent for r in self.results) / total_tests

            report_lines.append("OVERALL SUMMARY:")
            report_lines.append(f"  Total Tests: {total_tests}")
            report_lines.append(".2f")
            report_lines.append(".1f")
            report_lines.append(".1f")
            report_lines.append("")

        # Individual test results
        report_lines.append("INDIVIDUAL TEST RESULTS:")
        report_lines.append("-" * 40)

        for result in sorted(self.results, key=lambda x: x.timestamp, reverse=True):
            report_lines.append(f"Test: {result.test_name}")
            report_lines.append(".4f")
            report_lines.append(".2f")
            report_lines.append(".1f")
            report_lines.append(".2f")

            # Compare with baseline if available
            comparison = self.compare_with_baseline(result.test_name, result)
            if comparison['status'] == 'compared':
                report_lines.append(".1f")
                report_lines.append(".1f")

            report_lines.append("")

        # Performance recommendations
        report_lines.append("PERFORMANCE RECOMMENDATIONS:")
        report_lines.append("-" * 30)

        slow_tests = [r for r in self.results if r.operations_per_second < 100]
        if slow_tests:
            report_lines.append("Slow Operations Detected:")
            for test in slow_tests:
                report_lines.append(f"  - {test.test_name}: {test.operations_per_second:.2f} ops/sec")
            report_lines.append("  Consider optimizing these operations.")
            report_lines.append("")

        high_memory_tests = [r for r in self.results if r.memory_usage_mb > 50]
        if high_memory_tests:
            report_lines.append("High Memory Usage Detected:")
            for test in high_memory_tests:
                report_lines.append(f"  - {test.test_name}: {test.memory_usage_mb:.1f} MB")
            report_lines.append("  Consider memory optimization techniques.")
            report_lines.append("")

        return "\n".join(report_lines)

    def save_report(self, filename: str):
        """Save performance report to file."""
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        logging.info("Performance report saved to %s", filename)

def run_comprehensive_benchmark() -> PerformanceBenchmark:
    """Run a comprehensive benchmark suite."""
    bench = PerformanceBenchmark()

    try:

        # Benchmark simulation startup
        startup_result = bench.benchmark_simulation_startup(SimulationCoordinator)
        bench.set_baseline('simulation_startup', startup_result)

        # Benchmark node operations
        node_manager = get_optimized_node_manager()
        node_results = bench.benchmark_node_operations(node_manager, num_nodes=5000)

        for test_name, result in node_results.items():
            bench.set_baseline(test_name, result)

        # Benchmark caching system
        cache_manager = get_performance_cache_manager()
        cache_results = bench.benchmark_caching_system(cache_manager)

        for test_name, result in cache_results.items():
            bench.set_baseline(test_name, result)

        # Generate and save report
        bench.save_report("performance_benchmark_report.txt")

        logging.info("Comprehensive benchmark completed successfully")

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Benchmark failed: %s", e)

    return bench

if __name__ == "__main__":
    print("Running Neural Simulation Performance Benchmark...")
    benchmark = run_comprehensive_benchmark()
    print("Benchmark completed. Check performance_benchmark_report.txt for results.")






