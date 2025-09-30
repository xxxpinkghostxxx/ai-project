"""
Performance benchmark for C++ extensions integration.

This script measures the performance improvements achieved by integrating
C++ extensions into the neural simulation system.
"""

import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import psutil
from torch_geometric.data import Data

from src.neural.enhanced_neural_dynamics import EnhancedNeuralDynamics

# Import components to benchmark
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.utils.cpp_extensions import create_synaptic_calculator
    CPP_EXTENSIONS_AVAILABLE = True
except ImportError:
    CPP_EXTENSIONS_AVAILABLE = False
    print("C++ extensions not available, using fallback")


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for C++ extensions."""

    def __init__(self):
        self.results = []
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total,
            "platform": os.uname().sysname if hasattr(os, 'uname') else "Unknown",
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
        }

    def create_test_graph(self, num_nodes: int, num_edges: int) -> Data:
        """Create a test graph for benchmarking."""
        graph = Data()

        # Create node labels
        graph.node_labels = []
        for i in range(num_nodes):
            graph.node_labels.append({
                'id': i,
                'membrane_potential': -70.0 + np.random.normal(0, 5)
            })

        # Create node features tensor
        graph.x = np.random.randn(num_nodes, 1).astype(np.float32)

        # Create edge attributes
        graph.edge_attributes = []
        for i in range(num_edges):
            edge = MockEdge(
                source=np.random.randint(0, num_nodes),
                target=np.random.randint(0, num_nodes),
                weight=0.5 + 0.5 * np.random.randn(),
                edge_type='excitatory' if np.random.random() > 0.3 else 'inhibitory'
            )
            graph.edge_attributes.append(edge)

        return graph

    def benchmark_synaptic_calculator(self, sizes: List[int]) -> List[Dict[str, Any]]:
        """Benchmark synaptic calculator performance."""
        synaptic_results = []

        if not CPP_EXTENSIONS_AVAILABLE:
            print("C++ extensions not available, skipping synaptic calculator benchmark")
            return synaptic_results

        for num_edges in sizes:
            print(f"Benchmarking synaptic calculator with {num_edges} edges...")

            # Create test data
            num_nodes = max(100, num_edges // 10)
            graph = self.create_test_graph(num_nodes, num_edges)
            node_energies = {i: 0.8 for i in range(num_nodes)}

            # Benchmark optimized calculator
            calculator = create_synaptic_calculator()

            # Warm up
            for _ in range(5):
                _ = calculator.calculate_synaptic_inputs(
                    graph.edge_attributes, node_energies, num_nodes=num_nodes
                )

            # Benchmark
            start_time = time.perf_counter()
            iterations = 100
            for _ in range(iterations):
                _ = calculator.calculate_synaptic_inputs(
                    graph.edge_attributes, node_energies, num_nodes=num_nodes
                )
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / iterations
            throughput = num_edges / avg_time  # edges per second

            synaptic_results.append({
                "test": "synaptic_calculator",
                "size": num_edges,
                "avg_time": avg_time,
                "throughput": throughput,
                "units": "edges/sec"
            })

        return synaptic_results

    def benchmark_neural_dynamics(self, sizes: List[int]) -> List[Dict[str, Any]]:
        """Benchmark neural dynamics performance."""
        neural_results = []

        for num_nodes in sizes:
            print(f"Benchmarking neural dynamics with {num_nodes} nodes...")

            # Create test graph
            graph = self.create_test_graph(num_nodes, num_nodes * 5)

            # Initialize neural dynamics
            dynamics = EnhancedNeuralDynamics()

            # Warm up
            for _ in range(3):
                _ = dynamics.update_neural_dynamics(graph, 0)

            # Benchmark
            start_time = time.perf_counter()
            iterations = 50
            for step in range(iterations):
                graph = dynamics.update_neural_dynamics(graph, step)
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / iterations

            neural_results.append({
                "test": "neural_dynamics",
                "size": num_nodes,
                "avg_time": avg_time,
                "throughput": num_nodes / avg_time,  # nodes per second
                "units": "nodes/sec"
            })

        return neural_results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        print("Running C++ Extensions Performance Benchmark")
        print("=" * 50)

        # Test sizes
        synaptic_sizes = [1000, 5000, 10000, 25000]
        neural_sizes = [1000, 2500, 5000, 10000]

        # Run benchmarks
        synaptic_results = self.benchmark_synaptic_calculator(synaptic_sizes)
        neural_results = self.benchmark_neural_dynamics(neural_sizes)

        all_results = synaptic_results + neural_results
        self.results.extend(all_results)

        # Generate summary
        summary = self._generate_summary(all_results)

        return {
            "system_info": self.system_info,
            "results": all_results,
            "summary": summary,
            "timestamp": time.time()
        }

    def _generate_summary(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {}

        # Group by test type
        test_groups = {}
        for result in benchmark_results:
            test_type = result["test"]
            if test_type not in test_groups:
                test_groups[test_type] = []
            test_groups[test_type].append(result)

        # Calculate statistics for each test type
        for test_type, test_results in test_groups.items():
            sizes = [r["size"] for r in test_results]
            times = [r["avg_time"] for r in test_results]
            throughputs = [r["throughput"] for r in test_results]

            summary[test_type] = {
                "sizes_tested": sizes,
                "avg_times": times,
                "throughputs": throughputs,
                "best_throughput": max(throughputs),
                "worst_throughput": min(throughputs),
                "scalability_ratio": (
                    throughputs[-1] / throughputs[0]
                    if len(throughputs) > 1 else 1.0
                )
            }

        return summary

    def print_report(self, benchmark_results: Dict[str, Any]):
        """Print formatted benchmark report."""
        print("\nC++ Extensions Performance Benchmark Report")
        print("=" * 50)

        print("System Information:")
        for key, value in benchmark_results["system_info"].items():
            print(f"  {key}: {value}")

        print(f"\nBenchmark completed at: {time.ctime(benchmark_results['timestamp'])}")

        print("\nDetailed Results:")
        for result in benchmark_results["results"]:
            print(f"  {result['test']} ({result['size']} items): "
                  f"{result['avg_time']:.6f}s, "
                  f"{result['throughput']:.0f} {result['units']}")

        print("\nSummary:")
        for test_type, stats in benchmark_results["summary"].items():
            print(f"  {test_type}:")
            # Get units from the first result of this test type
            units = next(
                (r['units'] for r in benchmark_results["results"]
                 if r['test'] == test_type),
                'items/sec'
            )
            print(f"    Best throughput: {stats['best_throughput']:.0f} {units.split('/')[0]}/sec")
            print(f"    Scalability ratio: {stats['scalability_ratio']:.2f}x")

        print("\nPerformance Assessment:")
        synaptic_summary = benchmark_results["summary"].get("synaptic_calculator", {})
        if synaptic_summary:
            throughput = synaptic_summary["best_throughput"]
            if throughput > 100000:  # 100k edges/sec
                assessment = "EXCELLENT"
            elif throughput > 50000:  # 50k edges/sec
                assessment = "GOOD"
            elif throughput > 10000:  # 10k edges/sec
                assessment = "FAIR"
            else:
                assessment = "NEEDS_IMPROVEMENT"
            print(f"  Synaptic Calculator: {assessment} ({throughput:.0f} edges/sec)")


class MockEdge:
    """Mock edge object for testing."""
    def __init__(self, source: int, target: int, weight: float, edge_type: str):
        self.source = source
        self.target = target
        self.weight = weight
        self.type = edge_type

    def get_effective_weight(self) -> float:
        """Get the effective weight of the edge.

        Returns:
            float: The weight value of the edge, representing connection strength
                  in the mock testing framework.

        This method serves as a simple getter for the edge weight property,
        providing a consistent interface for accessing edge weight values
        during performance benchmarking and testing scenarios.
        """
        return self.weight

    def get_source(self) -> int:
        """Get the source node ID of the edge.

        Returns:
            int: The source node identifier.
        """
        return self.source

    def get_target(self) -> int:
        """Get the target node ID of the edge.

        Returns:
            int: The target node identifier.
        """
        return self.target

    def get_edge_type(self) -> str:
        """Get the type of the edge.

        Returns:
            str: The edge type ('excitatory' or 'inhibitory').
        """
        return self.type


def run_benchmark():
    """Run the performance benchmark."""
    benchmark = PerformanceBenchmark()
    benchmark_results = benchmark.run_comprehensive_benchmark()
    benchmark.print_report(benchmark_results)
    return benchmark_results
if __name__ == "__main__":
    results = run_benchmark()
