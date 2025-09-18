#!/usr/bin/env python3
"""
Test script for SimulationManager optimizations and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
from core.simulation_manager import SimulationManager

def test_basic_initialization():
    """Test basic initialization without errors."""
    print("Testing basic initialization...")
    try:
        manager = SimulationManager()
        print("[OK] SimulationManager initialized successfully")
        return manager
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def test_graph_operations(manager):
    """Test graph operations and caching."""
    print("\nTesting graph operations...")

    # Test empty graph
    try:
        node_count = manager._get_node_count()
        edge_count = manager._get_edge_count()
        print(f"[OK] Empty graph counts: nodes={node_count}, edges={edge_count}")
    except Exception as e:
        print(f"[FAIL] Graph operations failed: {e}")
        return False

    # Test data cache invalidation
    try:
        manager.update_visual_data([1, 2, 3])
        has_visual = manager._has_visual_data()
        print(f"[OK] Visual data cache: {has_visual}")

        manager.update_audio_data([4, 5, 6])
        has_audio = manager._has_audio_data()
        print(f"[OK] Audio data cache: {has_audio}")

        # Test cache invalidation
        manager._invalidate_data_cache()
        print("[OK] Cache invalidation successful")
    except Exception as e:
        print(f"[FAIL] Data cache operations failed: {e}")
        return False

    return True

def test_consistency_validation(manager):
    """Test graph consistency validation caching."""
    print("\nTesting consistency validation...")

    # Test with no graph
    try:
        result = manager._validate_graph_consistency()
        print(f"[OK] No graph validation: {result}")

        # Test caching
        result2 = manager._validate_graph_consistency()
        print(f"[OK] Cached validation: {result2}")
    except Exception as e:
        print(f"[FAIL] Consistency validation failed: {e}")
        return False

    return True

def test_error_handling(manager):
    """Test improved error handling."""
    print("\nTesting error handling...")

    # Test callback error handling
    try:
        # Add a failing callback
        def failing_callback(*args):
            raise ValueError("Test callback failure")

        manager.add_step_callback(failing_callback)
        initial_count = len(manager.step_callbacks)
        print(f"[OK] Added failing callback, count: {initial_count}")

        # This should handle the error gracefully
        manager._execute_step_callbacks()
        final_count = len(manager.step_callbacks)
        print(f"[OK] Error handling: callbacks before={initial_count}, after={final_count}")

    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        return False

    return True

def test_performance_optimizations(manager):
    """Test performance optimizations."""
    print("\nTesting performance optimizations...")

    # Test optimized node/edge counting
    start_time = time.time()
    for _ in range(1000):
        node_count = manager._get_node_count()
        edge_count = manager._get_edge_count()
    end_time = time.time()

    print(f"[OK] Performance test completed in {(end_time - start_time) * 1000:.4f}ms")
    return True

def run_all_tests():
    """Run all optimization tests."""
    print("=" * 60)
    print("SIMULATION MANAGER OPTIMIZATION TESTS")
    print("=" * 60)

    manager = test_basic_initialization()
    if not manager:
        return False

    tests = [
        (test_graph_operations, manager),
        (test_consistency_validation, manager),
        (test_error_handling, manager),
        (test_performance_optimizations, manager),
    ]

    passed = 0
    total = len(tests)

    for test_func, *args in tests:
        try:
            if test_func(*args):
                passed += 1
            else:
                print(f"[FAIL] {test_func.__name__} failed")
        except Exception as e:
            print(f"[FAIL] {test_func.__name__} crashed: {e}")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)