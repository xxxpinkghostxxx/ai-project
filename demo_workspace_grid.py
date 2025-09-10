"""
Demonstration script for the workspace grid functionality.
This shows how to create, manipulate, and visualize the 16x16 workspace grid.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_graph import create_workspace_grid, merge_graphs
import numpy as np


def demo_workspace_grid_basics():
    """Demonstrate basic workspace grid operations."""
    print("=== Workspace Grid Basics Demo ===\n")
    
    # Create a workspace grid
    print("1. Creating a 16x16 workspace grid...")
    workspace_graph = create_workspace_grid()
    print(f"   ✓ Created grid with {len(workspace_graph.node_labels)} nodes")
    print(f"   ✓ Grid dimensions: {workspace_graph.w}x{workspace_graph.h}")
    
    # Show some node examples
    print("\n2. Sample workspace nodes:")
    for i in [0, 15, 240, 255]:  # Show first, last, and some middle nodes
        node = workspace_graph.node_labels[i]
        print(f"   Node {i}: ({node['x']}, {node['y']}) - Energy: {node['energy']}, State: {node['state']}")
    
    return workspace_graph


def demo_energy_manipulation(workspace_graph):
    """Demonstrate manipulating workspace node energies."""
    print("\n=== Energy Manipulation Demo ===\n")
    
    # Set some specific energies
    print("1. Setting specific node energies...")
    
    # Create a simple pattern
    test_positions = [
        (0, 0, 255.0),    # Top-left: full energy
        (7, 7, 128.0),    # Center: half energy
        (15, 15, 64.0),   # Bottom-right: quarter energy
        (8, 0, 192.0),    # Top-center: high energy
        (0, 8, 96.0),     # Left-center: medium energy
    ]
    
    for x, y, energy in test_positions:
        # Find the node at this position
        for node in workspace_graph.node_labels:
            if node['x'] == x and node['y'] == y:
                node['energy'] = energy
                node['membrane_potential'] = energy / 255.0
                node['state'] = 'active' if energy > 0 else 'inactive'
                print(f"   ✓ Set node ({x}, {y}) to energy {energy}")
                break
    
    # Show the updated nodes
    print("\n2. Updated node states:")
    for x, y, energy in test_positions:
        for node in workspace_graph.node_labels:
            if node['x'] == x and node['y'] == y:
                print(f"   Node ({x}, {y}): Energy={node['energy']:.1f}, "
                      f"Membrane={node['membrane_potential']:.3f}, State={node['state']}")
                break


def demo_grid_visualization(workspace_graph):
    """Demonstrate how the grid would be visualized."""
    print("\n=== Grid Visualization Demo ===\n")
    
    # Create a 16x16 grid array
    grid = np.zeros((16, 16), dtype=np.float32)
    
    # Fill the grid with node energies
    for node in workspace_graph.node_labels:
        x, y = node['x'], node['y']
        grid[y, x] = node['energy']
    
    # Show the grid as a simple text representation
    print("1. Grid energy values (showing non-zero values):")
    for y in range(16):
        row_values = []
        for x in range(16):
            energy = grid[y, x]
            if energy > 0:
                row_values.append(f"{energy:6.1f}")
            else:
                row_values.append("     .")
        if any(val != "     ." for val in row_values):
            print(f"   Row {y:2d}: {' '.join(row_values)}")
    
    # Show statistics
    total_energy = np.sum(grid)
    active_nodes = np.sum(grid > 0)
    avg_energy = total_energy / max(active_nodes, 1)
    
    print(f"\n2. Grid Statistics:")
    print(f"   Total Energy: {total_energy:.1f}")
    print(f"   Active Nodes: {active_nodes}/256")
    print(f"   Average Energy: {avg_energy:.1f}")
    print(f"   Energy Density: {active_nodes/256*100:.1f}%")


def demo_merge_with_main_graph():
    """Demonstrate merging workspace grid with a main graph."""
    print("\n=== Graph Merge Demo ===\n")
    
    # Create a simple main graph (simulating sensory + dynamic nodes)
    import torch
    from torch_geometric.data import Data
    
    print("1. Creating a simple main graph...")
    main_graph = Data(
        x=torch.tensor([[100.0], [150.0], [200.0]], dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_labels=[
            {'type': 'sensory', 'x': 0, 'y': 0, 'energy': 100.0},
            {'type': 'dynamic', 'id': 1, 'energy': 150.0},
            {'type': 'dynamic', 'id': 2, 'energy': 200.0}
        ]
    )
    print(f"   ✓ Main graph has {len(main_graph.node_labels)} nodes")
    
    # Create workspace grid
    print("2. Creating workspace grid...")
    workspace_graph = create_workspace_grid()
    print(f"   ✓ Workspace grid has {len(workspace_graph.node_labels)} nodes")
    
    # Merge them
    print("3. Merging graphs...")
    merged_graph = merge_graphs(main_graph, workspace_graph)
    print(f"   ✓ Merged graph has {len(merged_graph.node_labels)} nodes")
    
    # Show node type distribution
    print("\n4. Node type distribution in merged graph:")
    type_counts = {}
    for node in merged_graph.node_labels:
        node_type = node['type']
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    for node_type, count in type_counts.items():
        print(f"   {node_type.capitalize()}: {count} nodes")
    
    return merged_graph


def main():
    """Run the complete workspace grid demonstration."""
    print("🚀 Workspace Grid Demonstration\n")
    print("This demo shows the 16x16 workspace grid functionality that provides")
    print("a visual workspace where nodes can represent colors/energy patterns.\n")
    
    try:
        # Basic workspace grid creation
        workspace_graph = demo_workspace_grid_basics()
        
        # Energy manipulation
        demo_energy_manipulation(workspace_graph)
        
        # Grid visualization
        demo_grid_visualization(workspace_graph)
        
        # Graph merging
        merged_graph = demo_merge_with_main_graph()
        
        print("\n" + "="*50)
        print("🎉 Workspace Grid Demo Completed Successfully!")
        print("="*50)
        print("\nThe workspace grid provides:")
        print("• 256 nodes arranged in a 16x16 grid")
        print("• Each node can have energy values 0-255 (representing colors)")
        print("• Nodes are protected from automatic birth/death")
        print("• Manual energy manipulation through UI controls")
        print("• Real-time visualization of the workspace state")
        print("\nThis creates an 'imagination space' where the system can")
        print("visually represent concepts, patterns, and creative ideas!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
