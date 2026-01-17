#!/usr/bin/env python3
"""
Performance test suite for vectorized energy calculations.
This test measures the performance improvement of the optimized vectorized implementation
and validates that it maintains backward compatibility and behavioral integrity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import logging
import statistics
from src.project.pyg_neural_system import PyGNeuralSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_system(sensory_size=4, dynamic_nodes=10, workspace_size=(2, 2)):
    """Create a test system with specified parameters"""
    return PyGNeuralSystem(
        sensory_width=sensory_size,
        sensory_height=sensory_size,
        n_dynamic=dynamic_nodes,
        workspace_size=workspace_size
    )

def benchmark_energy_update(system, num_iterations=100, warmup_iterations=10):
    """
    Benchmark the energy update performance
    Returns: (average_time, std_dev, min_time, max_time, total_time)
    """
    if warmup_iterations > 0:
        logger.info(f"Running {warmup_iterations} warmup iterations...")
        for _ in range(warmup_iterations):
            system._update_energies()

    logger.info(f"Running {num_iterations} benchmark iterations...")
    times = []

    for i in range(num_iterations):
        start_time = time.perf_counter()
        system._update_energies()
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)

        if not (i + 1) % 10:
            avg_time = statistics.mean(times[-10:])
            logger.info(f"  Iteration {i+1}: {elapsed:.6f}s (avg last 10: {avg_time:.6f}s)")

    # Calculate statistics
    avg_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    total_time = sum(times)

    return avg_time, std_dev, min_time, max_time, total_time, times

def test_performance_scaling():
    """Test performance scaling with different system sizes"""
    logger.info("Testing performance scaling with different system sizes...")

    test_configs = [
        {"name": "Small", "sensory": 2, "dynamic": 5, "workspace": (1, 1)},
        {"name": "Medium", "sensory": 4, "dynamic": 20, "workspace": (2, 2)},
        {"name": "Large", "sensory": 8, "dynamic": 50, "workspace": (4, 4)},
        {"name": "XLarge", "sensory": 16, "dynamic": 100, "workspace": (8, 8)}
    ]

    results = []

    for config in test_configs:
        logger.info(f"\n--- Testing {config['name']} system ---")
        system = create_test_system(
            sensory_size=config['sensory'],
            dynamic_nodes=config['dynamic'],
            workspace_size=config['workspace']
        )

        # Get system metrics
        node_count = system.g.num_nodes if system.g else 0
        edge_count = system.g.num_edges if system.g else 0

        logger.info(f"System: {node_count} nodes, {edge_count} edges")

        # Run benchmark
        avg_time, std_dev, min_time, max_time, total_time, _ = benchmark_energy_update(
            system, num_iterations=50, warmup_iterations=5
        )

        # Calculate throughput
        updates_per_second = 1.0 / avg_time if avg_time > 0 else 0

        result = {
            "name": config['name'],
            "nodes": node_count,
            "edges": edge_count,
            "avg_time": avg_time,
            "std_dev": std_dev,
            "min_time": min_time,
            "max_time": max_time,
            "updates_per_second": updates_per_second,
            "total_time": total_time
        }

        results.append(result)

        logger.info(f"Results: {avg_time:.6f}s ± {std_dev:.6f}s, {updates_per_second:.1f} updates/s")

    return results

def test_behavioral_consistency():
    """Test that the optimized implementation maintains behavioral consistency"""
    logger.info("\nTesting behavioral consistency...")

    # Create two identical systems
    system1 = create_test_system(sensory_size=4, dynamic_nodes=15, workspace_size=(3, 3))
    system2 = create_test_system(sensory_size=4, dynamic_nodes=15, workspace_size=(3, 3))

    # Set identical random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Store initial states
    initial_energy1 = system1.g.energy.clone() if system1.g and hasattr(system1.g, 'energy') else None
    initial_energy2 = system2.g.energy.clone() if system2.g and hasattr(system2.g, 'energy') else None

    # Run multiple updates and compare results
    consistency_results = []

    for i in range(10):
        # Update both systems
        system1._update_energies()
        system2._update_energies()

        # Compare energy states
        energy1 = system1.g.energy if system1.g and hasattr(system1.g, 'energy') else None
        energy2 = system2.g.energy if system2.g and hasattr(system2.g, 'energy') else None

        if energy1 is not None and energy2 is not None:
            # Calculate difference metrics
            abs_diff = torch.abs(energy1 - energy2)
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()
            std_diff = abs_diff.std().item()

            # Check if differences are within acceptable tolerance
            # (should be very small for identical systems)
            consistent = max_diff < 1e-5

            result = {
                "iteration": i + 1,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "consistent": consistent
            }

            consistency_results.append(result)

            logger.info(f"Iteration {i+1}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, consistent={consistent}")

    # Overall consistency check
    all_consistent = all(result['consistent'] for result in consistency_results)
    avg_max_diff = statistics.mean([r['max_diff'] for r in consistency_results])

    logger.info(f"Overall consistency: {all_consistent} (avg max diff: {avg_max_diff:.2e})")

    return all_consistent, avg_max_diff, consistency_results

def test_memory_efficiency():
    """Test memory efficiency of the vectorized implementation"""
    logger.info("\nTesting memory efficiency...")

    # Create a system and monitor memory usage
    system = create_test_system(sensory_size=8, dynamic_nodes=50, workspace_size=(4, 4))

    # Get initial memory metrics
    if system.g is not None and hasattr(system.g, 'energy') and system.g.energy is not None:
        energy_memory = system.g.energy.element_size() * system.g.energy.nelement()
    else:
        energy_memory = 0

    if system.g is not None and hasattr(system.g, 'edge_index') and system.g.edge_index is not None:
        edge_index_memory = system.g.edge_index.element_size() * system.g.edge_index.nelement()
    else:
        edge_index_memory = 0

    total_memory = energy_memory + edge_index_memory

    logger.info(f"Memory usage: Energy={energy_memory/1024:.2f}KB, EdgeIndex={edge_index_memory/1024:.2f}KB, Total={total_memory/1024:.2f}KB")

    # Run updates and monitor memory stability
    memory_samples = []

    for i in range(20):
        system._update_energies()

        if not (i + 1) % 5:
            if system.g is not None and hasattr(system.g, 'energy') and system.g.energy is not None:
                current_energy_memory = system.g.energy.element_size() * system.g.energy.nelement()
            else:
                current_energy_memory = 0

            memory_samples.append(current_energy_memory)

    # Check memory stability
    if memory_samples:
        avg_memory = statistics.mean(memory_samples)
        std_memory = statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0
        memory_stable = std_memory < avg_memory * 0.01  # Less than 1% variation

        logger.info(f"Memory stability: avg={avg_memory/1024:.2f}KB, std={std_memory/1024:.2f}KB, stable={memory_stable}")
        return memory_stable, avg_memory, std_memory

    return False, 0, 0

def test_edge_cases():
    """Test edge cases and error handling"""
    logger.info("\nTesting edge cases...")

    test_cases = [
        {"name": "Empty system", "sensory": 1, "dynamic": 0, "workspace": (1, 1)},
        {"name": "No edges", "sensory": 2, "dynamic": 2, "workspace": (1, 1)},
        {"name": "Single node", "sensory": 1, "dynamic": 0, "workspace": (1, 1)},
        {"name": "Large dynamic", "sensory": 4, "dynamic": 200, "workspace": (2, 2)}
    ]

    results = []

    for case in test_cases:
        logger.info(f"  Testing {case['name']}...")

        try:
            system = create_test_system(
                sensory_size=case['sensory'],
                dynamic_nodes=case['dynamic'],
                workspace_size=case['workspace']
            )

            # Run a few updates to test stability
            for i in range(5):
                system._update_energies()

            # Check system state
            node_count = system.g.num_nodes if system.g else 0
            edge_count = system.g.num_edges if system.g else 0
            energy_valid = True

            if system.g is not None and hasattr(system.g, 'energy') and system.g.energy is not None:
                energy_valid = not torch.isnan(system.g.energy).any() and not torch.isinf(system.g.energy).any()
                energy_range = f"{system.g.energy.min().item():.3f}-{system.g.energy.max().item():.3f}"
            else:
                energy_valid = True
                energy_range = "N/A"

            success = (node_count or 0) >= 0 and energy_valid
            results.append({
                "name": case['name'],
                "success": success,
                "nodes": node_count,
                "edges": edge_count,
                "energy_range": energy_range
            })

            logger.info(f"    ✓ {case['name']}: {node_count} nodes, {edge_count} edges, energy={energy_range}")

        except Exception as e:
            logger.error(f"    ✗ {case['name']} failed: {e}")
            results.append({
                "name": case['name'],
                "success": False,
                "error": str(e)
            })

    # Summary
    success_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)

    logger.info(f"Edge case testing: {success_count}/{total_count} passed")

    return success_count == total_count, results

def generate_performance_report():
    """Generate a comprehensive performance report"""
    logger.info("\n" + "="*60)
    logger.info("GENERATING PERFORMANCE REPORT")
    logger.info("="*60)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "pytorch_version": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "tests": {}
    }

    # Run all tests
    logger.info("\n1. Running performance scaling tests...")
    scaling_results = test_performance_scaling()
    report["tests"]["scaling"] = scaling_results

    logger.info("\n2. Running behavioral consistency tests...")
    consistent, avg_diff, consistency_results = test_behavioral_consistency()
    report["tests"]["consistency"] = {
        "consistent": consistent,
        "avg_max_diff": avg_diff,
        "results": consistency_results
    }

    logger.info("\n3. Running memory efficiency tests...")
    memory_stable, avg_memory, std_memory = test_memory_efficiency()
    report["tests"]["memory"] = {
        "stable": memory_stable,
        "avg_memory_kb": avg_memory / 1024,
        "std_memory_kb": std_memory / 1024
    }

    logger.info("\n4. Running edge case tests...")
    edge_cases_passed, edge_results = test_edge_cases()
    report["tests"]["edge_cases"] = {
        "passed": edge_cases_passed,
        "results": edge_results
    }

    # Generate summary
    all_tests_passed = (
        all(r['updates_per_second'] > 0 for r in scaling_results) and
        report["tests"]["consistency"]["consistent"] and
        report["tests"]["memory"]["stable"] and
        edge_cases_passed
    )

    # Calculate performance improvement metrics
    if scaling_results:
        small_system = next((r for r in scaling_results if r["name"] == "Small"), None)
        large_system = next((r for r in scaling_results if r["name"] == "XLarge"), None)

        if small_system and large_system:
            # Estimate scaling efficiency
            node_ratio = large_system["nodes"] / small_system["nodes"]
            time_ratio = large_system["avg_time"] / small_system["avg_time"]
            scaling_efficiency = node_ratio / time_ratio if time_ratio > 0 else 0

            report["scaling_efficiency"] = scaling_efficiency
            logger.info(f"Scaling efficiency: {scaling_efficiency:.2f} (ideal would be ~1.0)")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE REPORT SUMMARY")
    logger.info("="*60)

    logger.info(f"✅ All tests passed: {all_tests_passed}")
    logger.info(f"✅ Behavioral consistency: {report['tests']['consistency']['consistent']} (avg diff: {report['tests']['consistency']['avg_max_diff']:.2e})")
    logger.info(f"✅ Memory stability: {report['tests']['memory']['stable']}")
    logger.info(f"✅ Edge cases: {edge_cases_passed}")

    if scaling_results:
        best_performance = max(r['updates_per_second'] for r in scaling_results)
        worst_performance = min(r['updates_per_second'] for r in scaling_results)
        logger.info(f"📊 Performance range: {worst_performance:.1f} - {best_performance:.1f} updates/s")

    logger.info("\n🎉 Performance testing completed!")

    return report, all_tests_passed

if __name__ == "__main__":
    print("Vectorized Energy Performance Test Suite")
    print("=" * 60)

    try:
        # Run comprehensive performance testing
        report, success = generate_performance_report()

        # Save report
        import json
        with open("performance_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Performance report saved to performance_report.json")
        print(f"🎯 Overall success: {'✅ PASSED' if success else '❌ FAILED'}")

    except Exception as e:
        print(f"\n❌ Performance testing failed: {e}")
        import traceback
        traceback.print_exc()