"""
Workspace Node System Demonstration

This script demonstrates the workspace node system functionality,
showing how workspace nodes read energy from sensory nodes and
create a 16x16 grid visualization.
"""

import time
import numpy as np
import sys
import os
# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.project.workspace.workspace_system import WorkspaceNodeSystem
from src.project.workspace.config import EnergyReadingConfig
from src.project.workspace.pixel_shading import PixelShadingSystem
from src.project.workspace.renderer import WorkspaceRenderer


class DemoNeuralSystem:
    """Demo neural system for testing workspace functionality."""
    
    def __init__(self, width=256, height=144):
        self.sensory_width = width
        self.sensory_height = height
        self.energies = np.random.rand(width * height) * 244.0
    
    def get_node_energy(self, node_id: int) -> float:
        """Get energy for a specific sensory node."""
        if 0 <= node_id < len(self.energies):
            return float(self.energies[node_id])
        return 0.0
    
    def update_energies(self):
        """Simulate energy updates in the neural system."""
        # Add some dynamic behavior
        noise = np.random.normal(0, 10, self.energies.shape)
        self.energies += noise
        self.energies = np.clip(self.energies, 0, 244.0)


def main():
    """Main demonstration function."""
    print("Workspace Node System Demonstration")
    print("=" * 50)
    
    # Create demo neural system
    print("1. Creating demo neural system...")
    neural_system = DemoNeuralSystem(width=256, height=144)
    
    # Create workspace system configuration
    print("2. Creating workspace system configuration...")
    config = EnergyReadingConfig()
    config.grid_size = (16, 16)
    config.reading_interval_ms = 100  # Update every 100ms
    
    # Initialize workspace system
    print("3. Initializing workspace system...")
    workspace_system = WorkspaceNodeSystem(neural_system, config)
    
    # Create pixel shading system
    print("4. Creating pixel shading system...")
    shading_system = PixelShadingSystem(energy_min=0.0, energy_max=244.0)
    
    # Create renderer
    print("5. Creating workspace renderer...")
    renderer = WorkspaceRenderer(grid_size=(16, 16), pixel_size=20)
    
    print("\nStarting demonstration...")
    print("Press Ctrl+C to stop\n")
    
    try:
        for step in range(50):  # Run for 50 steps
            # Update neural system energies
            neural_system.update_energies()
            
            # Update workspace system
            workspace_system.update()
            
            # Get energy grid
            energy_grid = workspace_system._calculate_energy_grid()
            
            # Convert to pixel values
            pixel_grid = []
            for y in range(16):
                row = []
                for x in range(16):
                    energy = energy_grid[y][x]
                    pixel_value = shading_system.energy_to_pixel_value(energy)
                    row.append(pixel_value)
                pixel_grid.append(row)
            
            # Print energy statistics
            flat_energies = [energy for row in energy_grid for energy in row]
            avg_energy = np.mean(flat_energies)
            max_energy = np.max(flat_energies)
            min_energy = np.min(flat_energies)
            
            print(f"Step {step+1:2d}: Avg={avg_energy:6.1f}, Max={max_energy:6.1f}, Min={min_energy:6.1f}")
            
            # Simulate real-time updates
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user")
    
    print("\nDemonstration completed!")
    print("\nKey Features Demonstrated:")
    print("- 16x16 workspace grid reading from 256x144 sensory grid")
    print("- Energy aggregation from multiple sensory nodes per workspace node")
    print("- Real-time energy visualization with pixel shading")
    print("- Dynamic energy updates and trend detection")


if __name__ == "__main__":
    main()