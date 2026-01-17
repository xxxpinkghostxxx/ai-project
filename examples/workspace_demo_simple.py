"""
Simple Workspace Node System Demonstration

This script provides a straightforward demonstration of the workspace node system
without complex UI dependencies. It focuses on core functionality and real-time
visualization using text-based output.
"""

import time
import sys
import os
import threading
import logging
import numpy as np
from typing import List, Dict, Any

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import workspace system components
from src.project.workspace.workspace_system import WorkspaceNodeSystem
from src.project.workspace.config import EnergyReadingConfig
from src.project.workspace.pixel_shading import PixelShadingSystem

logger = logging.getLogger(__name__)


class SimpleDemoNeuralSystem:
    """Simple demo neural system for testing workspace functionality."""
    
    def __init__(self, width=256, height=144):
        self.sensory_width = width
        self.sensory_height = height
        self.total_nodes = width * height
        self.energies = np.random.rand(self.total_nodes) * 244.0
        self.time_step = 0
    
    def get_node_energy(self, node_id: int) -> float:
        """Get energy for a specific sensory node."""
        if 0 <= node_id < self.total_nodes:
            return float(self.energies[node_id])
        return 0.0
    
    def get_batch_energies(self, node_ids: List[int]) -> List[float]:
        """Get energy for multiple nodes efficiently."""
        energies = []
        for node_id in node_ids:
            if 0 <= node_id < self.total_nodes:
                energies.append(float(self.energies[node_id]))
            else:
                energies.append(0.0)
        return energies
    
    def update_energies(self):
        """Update energies with dynamic patterns."""
        self.time_step += 1
        
        # Generate wave pattern
        x_coords = np.arange(self.sensory_width)
        y_coords = np.arange(self.sensory_height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        frequency = 0.1
        amplitude = 50
        phase = self.time_step * 0.1
        
        wave = amplitude * np.sin(frequency * X + phase) + amplitude * np.cos(frequency * Y + phase)
        
        self.energies = wave.flatten() + 100
        
        # Add noise
        noise = np.random.normal(0, 5, self.energies.shape)
        self.energies += noise
        self.energies = np.clip(self.energies, 0, 244.0)


class SimpleWorkspaceDemo:
    """Simple demonstration of workspace node system."""
    
    def __init__(self):
        self.neural_system = None
        self.workspace_system = None
        self.shading_system = None
        self.demo_running = False
        self.update_count = 0
        
        # Initialize demo
        self._initialize_demo()
    
    def _initialize_demo(self):
        """Initialize demo components."""
        print("Initializing Simple Workspace Node System Demo...")
        
        # Create demo neural system
        self.neural_system = SimpleDemoNeuralSystem(width=256, height=144)
        
        # Create workspace system configuration
        config = EnergyReadingConfig()
        config.grid_size = (16, 16)
        config.reading_interval_ms = 100
        
        # Initialize workspace system
        self.workspace_system = WorkspaceNodeSystem(self.neural_system, config)
        
        # Create pixel shading system
        self.shading_system = PixelShadingSystem(energy_min=0.0, energy_max=244.0)
        
        print("✓ Demo components initialized successfully!")
        print(f"  - Sensory grid: {self.neural_system.sensory_width}x{self.neural_system.sensory_height}")
        print(f"  - Workspace grid: {config.grid_size[0]}x{config.grid_size[1]}")
        print(f"  - Update interval: {config.reading_interval_ms}ms")
    
    def start_demo(self, duration=30):
        """Start the demonstration."""
        print(f"\nStarting demonstration for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        self.demo_running = True
        self.update_count = 0
        start_time = time.time()
        
        try:
            # Start workspace system
            self.workspace_system.start()
            
            while self.demo_running and (time.time() - start_time) < duration:
                self._update_and_display()
                time.sleep(0.1)  # Update every 100ms
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        finally:
            self.stop_demo()
    
    def stop_demo(self):
        """Stop the demonstration."""
        print("\nStopping demonstration...")
        self.demo_running = False
        
        # Stop workspace system
        self.workspace_system.stop()
        
        print("Demo stopped!")
    
    def _update_and_display(self):
        """Update system and display results."""
        self.update_count += 1
        
        # Update neural system
        self.neural_system.update_energies()
        
        # Update workspace system
        self.workspace_system.update()
        
        # Get energy grid
        energy_grid = self.workspace_system._calculate_energy_grid()
        
        # Calculate statistics
        flat_energies = [energy for row in energy_grid for energy in row]
        avg_energy = np.mean(flat_energies) if flat_energies else 0.0
        max_energy = np.max(flat_energies) if flat_energies else 0.0
        min_energy = np.min(flat_energies) if flat_energies else 0.0
        
        # Get system health
        health = self.workspace_system.get_system_health()
        
        # Display statistics
        print(f"Step {self.update_count:3d}: Avg={avg_energy:6.1f}, Max={max_energy:6.1f}, Min={min_energy:6.1f}")
        
        # Display grid visualization (text-based)
        self._display_grid_visualization(energy_grid)
        
        # Display system health periodically
        if self.update_count % 10 == 0:
            print(f"  System Health: Running={health['running']}, "
                  f"Errors={health['error_count']}, "
                  f"Cache={health['cache_size']}")
    
    def _display_grid_visualization(self, energy_grid: List[List[float]]):
        """Display a text-based visualization of the energy grid."""
        print("  Grid Visualization:")
        
        for y in range(16):
            row_str = "    "
            for x in range(16):
                energy = energy_grid[y][x]
                pixel_value = self.shading_system.energy_to_pixel_value(energy)
                
                # Create ASCII representation
                if pixel_value < 64:
                    char = "░"
                elif pixel_value < 128:
                    char = "▒"
                elif pixel_value < 192:
                    char = "▓"
                else:
                    char = "█"
                
                row_str += char
            
            print(row_str)
        
        print()
    
    def run_performance_test(self, iterations=100):
        """Run performance test."""
        print(f"\nRunning performance test with {iterations} iterations...")
        
        update_times = []
        
        # Warm up
        for _ in range(10):
            self.neural_system.update_energies()
            self.workspace_system.update()
        
        # Measure performance
        for _ in range(iterations):
            start_time = time.time()
            self.neural_system.update_energies()
            self.workspace_system.update()
            end_time = time.time()
            update_times.append(end_time - start_time)
        
        avg_time = np.mean(update_times)
        max_time = np.max(update_times)
        min_time = np.min(update_times)
        fps = 1.0 / avg_time
        
        print(f"Performance Results:")
        print(f"  Average update time: {avg_time*1000:.2f}ms")
        print(f"  Maximum update time: {max_time*1000:.2f}ms")
        print(f"  Minimum update time: {min_time*1000:.2f}ms")
        print(f"  Average FPS: {fps:.1f}")
        
        # Check performance requirements
        if avg_time < 0.01:
            print("  ✓ Performance is excellent (< 10ms average)")
        elif avg_time < 0.05:
            print("  ✓ Performance is good (< 50ms average)")
        else:
            print("  ⚠ Performance may be slow (> 50ms average)")
    
    def run_validation_test(self):
        """Run validation tests."""
        print("\nRunning validation tests...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: System initialization
        total_tests += 1
        try:
            assert self.workspace_system is not None
            assert len(self.workspace_system.workspace_nodes) == 256
            assert self.workspace_system.grid_size == (16, 16)
            print("  ✓ System initialization test passed")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ System initialization test failed: {e}")
        
        # Test 2: Energy reading
        total_tests += 1
        try:
            self.neural_system.update_energies()
            self.workspace_system.update()
            
            energy_grid = self.workspace_system._calculate_energy_grid()
            assert len(energy_grid) == 16
            assert all(len(row) == 16 for row in energy_grid)
            
            # Check energy values are reasonable
            for row in energy_grid:
                for energy in row:
                    assert 0.0 <= energy <= 244.0
            
            print("  ✓ Energy reading test passed")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Energy reading test failed: {e}")
        
        # Test 3: Node data access
        total_tests += 1
        try:
            node_data = self.workspace_system.get_node_data(0)
            assert 'node_id' in node_data
            assert 'current_energy' in node_data
            assert 'grid_position' in node_data
            
            print("  ✓ Node data access test passed")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Node data access test failed: {e}")
        
        # Test 4: Error handling
        total_tests += 1
        try:
            # Test invalid node ID
            invalid_energy = self.neural_system.get_node_energy(999999)
            assert invalid_energy == 0.0
            
            # Test batch with invalid IDs
            batch_energies = self.neural_system.get_batch_energies([0, 1, 999999, 2])
            assert len(batch_energies) == 4
            assert batch_energies[2] == 0.0
            
            print("  ✓ Error handling test passed")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Error handling test failed: {e}")
        
        # Test 5: System health
        total_tests += 1
        try:
            health = self.workspace_system.get_system_health()
            assert 'running' in health
            assert 'node_count' in health
            assert 'error_count' in health
            
            print("  ✓ System health test passed")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ System health test failed: {e}")
        
        print(f"\nValidation Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            print("🎉 All validation tests passed!")
        else:
            print(f"⚠️ {total_tests - tests_passed} tests failed")
        
        return tests_passed == total_tests


def main():
    """Main entry point for the simple demo."""
    print("Simple Workspace Node System Demonstration")
    print("=" * 50)
    
    # Create demo
    demo = SimpleWorkspaceDemo()
    
    # Run validation tests first
    validation_passed = demo.run_validation_test()
    
    if not validation_passed:
        print("\nValidation tests failed. Please check the implementation.")
        return
    
    # Run performance test
    demo.run_performance_test()
    
    # Start main demonstration
    demo.start_demo(duration=20)
    
    print("\nDemo completed successfully!")
    print("\nKey features demonstrated:")
    print("  ✓ 16x16 workspace grid reading from 256x144 sensory grid")
    print("  ✓ Energy aggregation from multiple sensory nodes per workspace node")
    print("  ✓ Real-time energy visualization with text-based display")
    print("  ✓ Dynamic energy updates and pattern generation")
    print("  ✓ Performance monitoring and validation")
    print("  ✓ Error handling and system health monitoring")


if __name__ == "__main__":
    main()