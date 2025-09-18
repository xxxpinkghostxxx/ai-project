#!/usr/bin/env python3
"""
Debug script for SimulationManager to identify and fix issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import traceback
from simulation_manager import SimulationManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_simulation_manager_initialization():
    """Test basic SimulationManager initialization."""
    print("=" * 60)
    print("Testing SimulationManager Initialization")
    print("=" * 60)

    try:
        print("Creating SimulationManager...")
        manager = SimulationManager()
        print("[OK] SimulationManager created successfully")

        # Check critical attributes
        required_attrs = [
            'error_count', 'consecutive_errors', 'max_consecutive_errors',
            'use_lazy_loading', 'use_caching', 'use_batch_processing',
            'behavior_engine', 'learning_engine', 'memory_system'
        ]

        print("\nChecking required attributes:")
        for attr in required_attrs:
            if hasattr(manager, attr):
                value = getattr(manager, attr)
                print(f"[OK] {attr}: {value}")
            else:
                print(f"[FAIL] Missing attribute: {attr}")

        # Test graph initialization
        print("\nTesting graph initialization...")
        success = manager.initialize_graph()
        if success:
            print("[OK] Graph initialized successfully")
            if manager.graph:
                node_count = len(manager.graph.node_labels) if hasattr(manager.graph, 'node_labels') else 0
                print(f"   Graph has {node_count} nodes")
        else:
            print("[FAIL] Graph initialization failed")

        return manager

    except Exception as e:
        print(f"[FAIL] SimulationManager initialization failed: {e}")
        traceback.print_exc()
        return None

def test_basic_simulation_step(manager):
    """Test a basic simulation step."""
    print("\n" + "=" * 60)
    print("Testing Basic Simulation Step")
    print("=" * 60)

    if not manager or not manager.graph:
        print("[FAIL] No valid manager or graph to test")
        return False

    try:
        print("Running single simulation step...")
        success = manager.run_single_step()
        if success:
            print("[OK] Simulation step completed successfully")
            return True
        else:
            print("[FAIL] Simulation step failed")
            return False

    except Exception as e:
        print(f"[FAIL] Simulation step error: {e}")
        traceback.print_exc()
        return False

def test_performance_optimizations(manager):
    """Test performance optimization features."""
    print("\n" + "=" * 60)
    print("Testing Performance Optimizations")
    print("=" * 60)

    if not manager:
        print("[FAIL] No valid manager to test")
        return False

    try:
        # Check optimization flags
        optimizations = [
            ('use_lazy_loading', manager.use_lazy_loading),
            ('use_caching', manager.use_caching),
            ('use_batch_processing', manager.use_batch_processing),
        ]

        print("Performance optimization flags:")
        for name, value in optimizations:
            status = "[ENABLED]" if value else "[DISABLED]"
            print(f"   {name}: {status}")

        # Test batch processing
        if hasattr(manager, '_update_node_behaviors_batch'):
            print("\nTesting batch processing...")
            try:
                manager._update_node_behaviors_batch()
                print("[OK] Batch processing works")
            except Exception as e:
                print(f"[FAIL] Batch processing failed: {e}")

        # Test caching
        if hasattr(manager, 'cache_manager'):
            print("[OK] Cache manager available")
        else:
            print("[FAIL] Cache manager not available")

        return True

    except Exception as e:
        print(f"[FAIL] Performance optimization test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("SimulationManager Debug Script")
    print("This script tests the core functionality and identifies issues.")

    # Test initialization
    manager = test_simulation_manager_initialization()

    if manager:
        # Test basic functionality
        test_basic_simulation_step(manager)

        # Test performance features
        test_performance_optimizations(manager)

        # Cleanup
        print("\n" + "=" * 60)
        print("Cleaning up...")
        manager.cleanup()
        print("[OK] Cleanup completed")

    print("\n" + "=" * 60)
    print("Debug script completed")
    print("=" * 60)

if __name__ == "__main__":
    main()