#!/usr/bin/env python3
"""
Test script to validate the tensor shape mismatch fix.
This script simulates the scenario that was causing the original error.
"""

import torch
import numpy as np
import logging
from src.project.pyg_neural_system import PyGNeuralSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tensor_shape_fix():
    """Test that the tensor shape mismatch issue is fixed"""
    print("Starting tensor shape fix validation test...")

    # Create a neural system
    system = PyGNeuralSystem(
        sensory_width=4,
        sensory_height=4,
        n_dynamic=10,
        workspace_size=(2, 2),
        device='cpu'
    )

    # Simulate the problematic scenario:
    # 1. Remove some nodes (which can cause tensor shape mismatches)
    # 2. Try to add new connections (which was failing before)

    print(f"Initial state: {system.g.num_nodes if system.g and hasattr(system.g, 'num_nodes') else 'None'} nodes, {system.g.num_edges if system.g and hasattr(system.g, 'num_edges') else 'None'} edges")

    # Simulate node removal by directly manipulating the graph
    # This creates the same conditions that led to the original error
    if system.g and hasattr(system.g, 'num_nodes') and system.g.num_nodes is not None and system.g.num_nodes > 5:
        # Create a mask to remove some nodes
        node_count = system.g.num_nodes if system.g and hasattr(system.g, 'num_nodes') and system.g.num_nodes is not None else 10
        death_mask = torch.zeros(int(node_count), dtype=torch.bool)
        death_mask[2:5] = True  # Mark nodes 2, 3, 4 for removal

        print(f"Removing nodes with mask: {death_mask.sum().item()} nodes to remove")
        system._remove_nodes(death_mask)

        print(f"After node removal: {system.g.num_nodes if system.g and hasattr(system.g, 'num_nodes') else 'None'} nodes, {system.g.num_edges if system.g and hasattr(system.g, 'num_edges') else 'None'} edges")

        # Validate that all tensors are consistent
        validation_results = system.tensor_manager.validate_tensor_shapes()
        invalid_tensors = [key for key, valid in validation_results.items() if not valid]

        if invalid_tensors:
            print(f"WARNING: Found invalid tensor shapes after node removal: {invalid_tensors}")
            # Try to synchronize
            sync_results = system.tensor_manager.synchronize_all_tensors()
            print(f"Synchronization results: {sync_results}")
        else:
            print("All tensor shapes are valid after node removal")

        # Now try to add connections (this was failing before the fix)
        print("Attempting to add new connections...")

        # Add some connection growth tasks
        system.conn_growth_queue = [
            (0, 1, 0),  # src, dst, subtype3
            (1, 2, 1),
            (2, 3, 2)
        ]

        # Start connection worker if not running
        if not hasattr(system, 'connection_worker') or system.connection_worker is None:
            system.start_connection_worker()

        # Queue a grow task
        if hasattr(system, 'connection_worker'):
            system.connection_worker.task_queue.put({'type': 'grow'})

        # Process the connection worker results (this is where the error was occurring)
        try:
            system.apply_connection_worker_results()
            print("SUCCESS: Connection worker results applied successfully!")

            print(f"Final state: {system.g.num_nodes if system.g and hasattr(system.g, 'num_nodes') else 'None'} nodes, {system.g.num_edges if system.g and hasattr(system.g, 'num_edges') else 'None'} edges")

            # Final validation
            final_validation = system.tensor_manager.validate_tensor_shapes()
            final_invalid = [key for key, valid in final_validation.items() if not valid]

            if final_invalid:
                print(f"WARNING: Still have invalid tensors: {final_invalid}")
                return False
            else:
                print("All tensor shapes are valid after connection addition")
                return True

        except Exception as e:
            print(f"ERROR: Failed to apply connection worker results: {str(e)}")
            return False

    else:
        print("Not enough nodes to test removal scenario")
        return False

def test_edge_case_scenarios():
    """Test various edge case scenarios that could cause tensor mismatches"""
    print("\nTesting edge case scenarios...")

    system = PyGNeuralSystem(
        sensory_width=2,
        sensory_height=2,
        n_dynamic=5,
        workspace_size=(1, 1),
        device='cpu'
    )

    # Test 1: Empty edge tensors
    print("Test 1: Handling empty edge tensors")
    original_edge_count = system.g.num_edges if system.g and hasattr(system.g, 'num_edges') else 0

    # Remove all edges by removing all dynamic nodes
    dynamic_mask = (system.g.node_type == 1) if system.g and hasattr(system.g, 'node_type') else torch.zeros(1, dtype=torch.bool)  # NODE_TYPE_DYNAMIC
    if dynamic_mask.any():
        system._remove_nodes(dynamic_mask)
        print(f"Removed dynamic nodes, now have {system.g.num_nodes if system.g and hasattr(system.g, 'num_nodes') else 'None'} nodes, {system.g.num_edges if system.g and hasattr(system.g, 'num_edges') else 'None'} edges")

    # Test 2: Add nodes back and try connections
    print("Test 2: Adding nodes and connections")
    system._add_nodes(3, 1)  # Add 3 dynamic nodes
    print(f"Added nodes, now have {system.g.num_nodes if system.g and hasattr(system.g, 'num_nodes') else 'None'} nodes")

    # Try to add connections
    system.conn_growth_queue = [(0, 1, 0)]
    if hasattr(system, 'connection_worker') and system.connection_worker:
        system.connection_worker.task_queue.put({'type': 'grow'})

    try:
        system.apply_connection_worker_results()
        print("Edge case test passed!")
        return True
    except Exception as e:
        print(f"Edge case test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Tensor Shape Mismatch Fix Validation")
    print("=" * 50)

    success1 = test_tensor_shape_fix()
    success2 = test_edge_case_scenarios()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ ALL TESTS PASSED: Tensor shape mismatch fix is working correctly!")
    else:
        print("❌ SOME TESTS FAILED: Fix may need additional work")

    print("Test completed.")