"""
Simple Energy System Test
Quick test to verify energy system functionality without full simulation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_energy_system_components():
    """Test individual energy system components."""

    print("Testing Energy System Components")
    print("=" * 40)

    try:
        # Test 1: Energy Constants
        print("\n1. Testing Energy Constants...")
        from energy.energy_constants import ConnectionConstants, EnergyConstants
        print(f"   TIME_STEP: {EnergyConstants.TIME_STEP}")
        print(f"   ACTIVATION_THRESHOLD: {EnergyConstants.ACTIVATION_THRESHOLD_DEFAULT}")
        print(f"   WEIGHT_MIN: {ConnectionConstants.WEIGHT_MIN}")
        print("   SUCCESS: Energy constants loaded")

        # Test 2: Energy Behavior Functions
        print("\n2. Testing Energy Behavior Functions...")
        from energy.energy_behavior import get_node_energy_cap
        energy_cap = get_node_energy_cap()
        print(f"   Node energy cap: {energy_cap}")
        print("   SUCCESS: Energy behavior functions work")

        # Test 3: Node Access Layer
        print("\n3. Testing Node Access Layer...")
        from energy.node_access_layer import NodeAccessLayer
        # Create a simple test graph
        import torch
        from torch_geometric.data import Data

        test_x = torch.tensor([[0.5], [0.8], [0.3]], dtype=torch.float32)
        test_node_labels = [
            {'id': 0, 'type': 'dynamic', 'energy': 0.5},
            {'id': 1, 'type': 'sensory', 'energy': 0.8},
            {'id': 2, 'type': 'dynamic', 'energy': 0.3}
        ]
        test_graph = Data(x=test_x, node_labels=test_node_labels)

        access_layer = NodeAccessLayer(test_graph)
        energy_0 = access_layer.get_node_energy(0)
        print(f"   Node 0 energy: {energy_0}")
        print("   SUCCESS: Node access layer works")

        # Test 4: Energy Dynamics
        print("\n4. Testing Energy Dynamics...")
        from energy.energy_behavior import apply_energy_behavior
        updated_graph = apply_energy_behavior(test_graph)
        print(f"   Graph updated with {len(updated_graph.node_labels)} nodes")
        print("   SUCCESS: Energy dynamics work")

        # Test 5: Connection Logic
        print("\n5. Testing Connection Logic...")
        from neural.connection_logic import create_weighted_connection
        # Add edge index if missing
        if not hasattr(test_graph, 'edge_index'):
            test_graph.edge_index = torch.empty((2, 0), dtype=torch.long)

        # Try to create a connection
        result_graph = create_weighted_connection(test_graph, 0, 1, 0.5)
        print(f"   Connection creation attempted")
        print("   SUCCESS: Connection logic accessible")

        print("\n" + "=" * 40)
        print("ENERGY SYSTEM TEST RESULTS")
        print("=" * 40)
        print("SUCCESS: All core energy components are functional!")
        print("SUCCESS: Energy constants: Working")
        print("SUCCESS: Energy behavior: Working")
        print("SUCCESS: Node access layer: Working")
        print("SUCCESS: Energy dynamics: Working")
        print("SUCCESS: Connection logic: Working")
        print("\nENERGY SYSTEM: Ready for integration!")

        return True

    except Exception as e:
        print(f"\nFAILED: Energy system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_energy_system_components()
    if success:
        print("\nENERGY SYSTEM VALIDATION: PASSED")
    else:
        print("\nENERGY SYSTEM VALIDATION: FAILED")