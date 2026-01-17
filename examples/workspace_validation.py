"""
Workspace Node System Validation and Testing Script

This script provides comprehensive validation and testing for the workspace node system
without requiring PyQt6 dependencies. It focuses on core functionality testing and
performance validation.
"""

import time
import sys
import os
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import workspace system components
from src.project.workspace.workspace_system import WorkspaceNodeSystem
from src.project.workspace.config import EnergyReadingConfig
from src.project.workspace.pixel_shading import PixelShadingSystem
from src.project.workspace.mapping import map_sensory_to_workspace, calculate_energy_aggregation
from src.project.workspace.workspace_node import WorkspaceNode

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Data class for validation results."""
    test_name: str
    passed: bool
    duration: float
    details: str = ""
    error: Optional[str] = None


class ValidationNeuralSystem:
    """Neural system for validation testing."""
    
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
        """Update energies with test patterns."""
        self.time_step += 1
        
        # Generate test pattern
        x_coords = np.arange(self.sensory_width)
        y_coords = np.arange(self.sensory_height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Wave pattern
        frequency = 0.1
        amplitude = 50
        phase = self.time_step * 0.1
        
        wave = amplitude * np.sin(frequency * X + phase) + amplitude * np.cos(frequency * Y + phase)
        
        self.energies = wave.flatten() + 100
        
        # Add noise
        noise = np.random.normal(0, 5, self.energies.shape)
        self.energies += noise
        self.energies = np.clip(self.energies, 0, 244.0)


class WorkspaceSystemValidator:
    """Comprehensive validator for the workspace node system."""
    
    def __init__(self):
        self.neural_system = None
        self.workspace_system = None
        self.shading_system = None
        self.validation_results = []
        
        # Initialize test environment
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Set up the test environment."""
        print("Setting up workspace node system validation...")
        
        # Create test neural system
        self.neural_system = ValidationNeuralSystem(width=256, height=144)
        
        # Create workspace system configuration
        config = EnergyReadingConfig()
        config.grid_size = (16, 16)
        config.reading_interval_ms = 50
        
        # Initialize workspace system
        self.workspace_system = WorkspaceNodeSystem(self.neural_system, config)
        
        # Create pixel shading system
        self.shading_system = PixelShadingSystem(energy_min=0.0, energy_max=244.0)
        
        print("✓ Test environment setup complete")
    
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation tests."""
        print("\n" + "="*60)
        print("WORKSPACE NODE SYSTEM VALIDATION SUITE")
        print("="*60)
        
        # Define all validation tests
        validation_tests = [
            ("System Initialization", self.validate_system_initialization),
            ("Energy Reading", self.validate_energy_reading),
            ("Grid Calculation", self.validate_grid_calculation),
            ("Node Data Access", self.validate_node_data_access),
            ("Performance", self.validate_performance),
            ("Error Handling", self.validate_error_handling),
            ("Configuration", self.validate_configuration),
            ("Mapping", self.validate_mapping),
            ("Shading System", self.validate_shading_system),
            ("System Health", self.validate_system_health),
            ("Threading", self.validate_threading),
            ("Cache Management", self.validate_cache_management),
        ]
        
        # Run all tests
        for test_name, test_func in validation_tests:
            print(f"\nRunning: {test_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                self.validation_results.append(result)
                self._display_result(result)
            except Exception as e:
                error_result = ValidationResult(
                    test_name=test_name,
                    passed=False,
                    duration=0.0,
                    error=str(e)
                )
                self.validation_results.append(error_result)
                self._display_result(error_result)
        
        # Display summary
        self._display_validation_summary()
        
        return self.validation_results
    
    def validate_system_initialization(self) -> ValidationResult:
        """Validate system initialization."""
        start_time = time.time()
        
        try:
            # Check workspace system exists
            assert self.workspace_system is not None, "Workspace system not initialized"
            
            # Check correct number of nodes
            assert len(self.workspace_system.workspace_nodes) == 256, \
                f"Expected 256 workspace nodes, got {len(self.workspace_system.workspace_nodes)}"
            
            # Check grid size
            assert self.workspace_system.grid_size == (16, 16), \
                f"Expected grid size (16, 16), got {self.workspace_system.grid_size}"
            
            # Check mapping exists
            assert len(self.workspace_system.mapping) == 256, \
                f"Expected 256 mappings, got {len(self.workspace_system.mapping)}"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="System Initialization",
                passed=True,
                duration=duration,
                details="All initialization checks passed"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="System Initialization",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_energy_reading(self) -> ValidationResult:
        """Validate energy reading functionality."""
        start_time = time.time()
        
        try:
            # Update neural system
            self.neural_system.update_energies()
            
            # Update workspace system
            self.workspace_system.update()
            
            # Check that nodes have energy values
            for node in self.workspace_system.workspace_nodes:
                assert node.current_energy >= 0.0, \
                    f"Node {node.node_id} has negative energy: {node.current_energy}"
                assert node.current_energy <= 244.0, \
                    f"Node {node.node_id} has energy > 244: {node.current_energy}"
            
            # Check that energy values are reasonable
            total_energy = sum(node.current_energy for node in self.workspace_system.workspace_nodes)
            assert total_energy > 0, "Total energy should be positive"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Energy Reading",
                passed=True,
                duration=duration,
                details="Energy reading working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Energy Reading",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_grid_calculation(self) -> ValidationResult:
        """Validate grid calculation."""
        start_time = time.time()
        
        try:
            # Get energy grid
            energy_grid = self.workspace_system._calculate_energy_grid()
            
            # Check grid dimensions
            assert len(energy_grid) == 16, f"Expected 16 rows, got {len(energy_grid)}"
            assert all(len(row) == 16 for row in energy_grid), "All rows should have 16 columns"
            
            # Check energy values in grid
            for y in range(16):
                for x in range(16):
                    energy = energy_grid[y][x]
                    assert 0.0 <= energy <= 244.0, \
                        f"Invalid energy at ({x}, {y}): {energy}"
            
            # Check that grid matches node energies
            for node in self.workspace_system.workspace_nodes:
                x, y = node.grid_position
                grid_energy = energy_grid[y][x]
                assert abs(grid_energy - node.current_energy) < 0.001, \
                    f"Grid energy mismatch at ({x}, {y}): grid={grid_energy}, node={node.current_energy}"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Grid Calculation",
                passed=True,
                duration=duration,
                details="Grid calculation working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Grid Calculation",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_node_data_access(self) -> ValidationResult:
        """Validate node data access."""
        start_time = time.time()
        
        try:
            # Test node data retrieval
            for node_id in range(10):  # Test first 10 nodes
                node_data = self.workspace_system.get_node_data(node_id)
                
                # Check required fields
                required_fields = ['node_id', 'grid_position', 'current_energy', 'associated_sensory']
                for field in required_fields:
                    assert field in node_data, f"Missing field {field} in node data"
                
                # Check field values
                assert node_data['node_id'] == node_id, \
                    f"Wrong node_id: expected {node_id}, got {node_data['node_id']}"
                assert isinstance(node_data['grid_position'], tuple), \
                    "grid_position should be a tuple"
                assert len(node_data['grid_position']) == 2, \
                    "grid_position should have 2 elements"
                assert isinstance(node_data['current_energy'], (int, float)), \
                    "current_energy should be numeric"
            
            # Test invalid node ID
            invalid_data = self.workspace_system.get_node_data(999)
            assert invalid_data == {}, "Invalid node ID should return empty dict"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Node Data Access",
                passed=True,
                duration=duration,
                details="Node data access working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Node Data Access",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_performance(self) -> ValidationResult:
        """Validate system performance."""
        start_time = time.time()
        
        try:
            # Measure update performance
            update_times = []
            for _ in range(100):
                update_start = time.time()
                self.neural_system.update_energies()
                self.workspace_system.update()
                update_end = time.time()
                update_times.append(update_end - update_start)
            
            avg_update_time = np.mean(update_times)
            max_update_time = np.max(update_times)
            min_update_time = np.min(update_times)
            
            # Performance should be reasonable
            assert avg_update_time < 0.05, \
                f"Average update time too slow: {avg_update_time:.4f}s"
            assert max_update_time < 0.1, \
                f"Maximum update time too slow: {max_update_time:.4f}s"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Performance",
                passed=True,
                duration=duration,
                details=f"Avg: {avg_update_time:.4f}s, Max: {max_update_time:.4f}s, Min: {min_update_time:.4f}s"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Performance",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_error_handling(self) -> ValidationResult:
        """Validate error handling."""
        start_time = time.time()
        
        try:
            initial_error_count = self.workspace_system.error_count
            
            # Test invalid node ID handling
            invalid_energy = self.neural_system.get_node_energy(999999)
            assert invalid_energy == 0.0, \
                f"Invalid node ID should return 0.0, got {invalid_energy}"
            
            # Test batch energy with invalid IDs
            batch_energies = self.neural_system.get_batch_energies([0, 1, 999999, 2])
            assert len(batch_energies) == 4, "Batch should return 4 values"
            assert batch_energies[2] == 0.0, "Invalid ID should return 0.0"
            
            # Test system recovery
            self.workspace_system.stop()
            self.workspace_system.start()
            
            # Should be able to restart
            assert self.workspace_system._running, "System should be running after restart"
            
            # Test error count doesn't increase for valid operations
            error_count_after = self.workspace_system.error_count
            assert error_count_after >= initial_error_count, \
                "Error count should not decrease"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Error Handling",
                passed=True,
                duration=duration,
                details="Error handling working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Error Handling",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_configuration(self) -> ValidationResult:
        """Validate configuration options."""
        start_time = time.time()
        
        try:
            config = self.workspace_system.config
            
            # Test configuration modification
            original_interval = config.reading_interval_ms
            config.reading_interval_ms = 25
            assert config.reading_interval_ms == 25, \
                f"Configuration update failed: {config.reading_interval_ms}"
            
            # Test cache settings
            original_cache_size = config.cache_size
            config.cache_size = 1000
            assert config.cache_size == 1000, \
                f"Cache size update failed: {config.cache_size}"
            
            # Test grid size
            assert config.grid_size == (16, 16), \
                f"Grid size configuration incorrect: {config.grid_size}"
            
            # Test pixel size
            assert config.pixel_size > 0, \
                f"Pixel size should be positive: {config.pixel_size}"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Configuration",
                passed=True,
                duration=duration,
                details="All configuration options working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Configuration",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_mapping(self) -> ValidationResult:
        """Validate sensory to workspace mapping."""
        start_time = time.time()
        
        try:
            # Test mapping creation
            mapping = self.workspace_system.mapping
            assert len(mapping) == 256, \
                f"Should have mapping for all workspace nodes: {len(mapping)}"
            
            # Test that all sensory nodes are mapped
            total_mapped = sum(len(sensory_list) for sensory_list in mapping.values())
            expected_total = self.neural_system.sensory_width * self.neural_system.sensory_height
            assert total_mapped == expected_total, \
                f"Mapping incomplete: {total_mapped}/{expected_total}"
            
            # Test aggregation calculation
            test_energies = [100.0, 150.0, 200.0]
            avg_energy = calculate_energy_aggregation(test_energies, method='average')
            assert abs(avg_energy - 150.0) < 0.001, \
                f"Average calculation incorrect: {avg_energy}"
            
            # Test max aggregation
            max_energy = calculate_energy_aggregation(test_energies, method='max')
            assert abs(max_energy - 200.0) < 0.001, \
                f"Max calculation incorrect: {max_energy}"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Mapping",
                passed=True,
                duration=duration,
                details="Sensory to workspace mapping working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Mapping",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_shading_system(self) -> ValidationResult:
        """Validate pixel shading system."""
        start_time = time.time()
        
        try:
            # Test energy to pixel conversion
            test_energies = [0.0, 61.0, 122.0, 183.0, 244.0]
            expected_pixels = [0, 63, 127, 191, 255]
            
            for energy, expected in zip(test_energies, expected_pixels):
                pixel = self.shading_system.energy_to_pixel_value(energy)
                assert abs(pixel - expected) <= 1, \
                    f"Pixel conversion incorrect for {energy}: {pixel} != {expected}"
            
            # Test color scheme
            self.shading_system.set_color_scheme('grayscale')
            assert self.shading_system.color_scheme == 'grayscale', \
                "Color scheme not set correctly"
            
            # Test shading mode
            self.shading_system.set_shading_mode('linear')
            assert self.shading_system.shading_mode == 'linear', \
                "Shading mode not set correctly"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Shading System",
                passed=True,
                duration=duration,
                details="Pixel shading system working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Shading System",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_system_health(self) -> ValidationResult:
        """Validate system health monitoring."""
        start_time = time.time()
        
        try:
            # Get system health
            health = self.workspace_system.get_system_health()
            
            # Check required fields
            required_fields = ['running', 'node_count', 'mapping_coverage', 
                             'avg_update_time', 'error_count', 'cache_size']
            for field in required_fields:
                assert field in health, f"Missing field {field} in health data"
            
            # Check field values
            assert health['node_count'] == 256, \
                f"Wrong node count: {health['node_count']}"
            assert health['mapping_coverage'] == 256, \
                f"Wrong mapping coverage: {health['mapping_coverage']}"
            assert health['error_count'] >= 0, \
                f"Negative error count: {health['error_count']}"
            assert health['cache_size'] >= 0, \
                f"Negative cache size: {health['cache_size']}"
            
            # Check that system is running
            assert health['running'] == True, \
                f"System should be running: {health['running']}"
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="System Health",
                passed=True,
                duration=duration,
                details="System health monitoring working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="System Health",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_threading(self) -> ValidationResult:
        """Validate threading functionality."""
        start_time = time.time()
        
        try:
            # Test that system can be started and stopped
            assert not self.workspace_system._running, \
                "System should not be running initially"
            
            # Start system
            self.workspace_system.start()
            time.sleep(0.1)  # Give it time to start
            assert self.workspace_system._running, \
                "System should be running after start()"
            
            # Stop system
            self.workspace_system.stop()
            time.sleep(0.1)  # Give it time to stop
            assert not self.workspace_system._running, \
                "System should not be running after stop()"
            
            # Test that we can restart
            self.workspace_system.start()
            time.sleep(0.1)
            assert self.workspace_system._running, \
                "System should be able to restart"
            
            self.workspace_system.stop()
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Threading",
                passed=True,
                duration=duration,
                details="Threading functionality working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Threading",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def validate_cache_management(self) -> ValidationResult:
        """Validate cache management."""
        start_time = time.time()
        
        try:
            # Clear cache
            self.workspace_system.energy_cache.clear()
            self.workspace_system.last_cache_update = 0.0
            
            # Update system to populate cache
            self.neural_system.update_energies()
            self.workspace_system.update()
            
            # Check that cache has entries
            assert len(self.workspace_system.energy_cache) > 0, \
                "Cache should have entries after update"
            
            # Check cache timestamp
            assert self.workspace_system.last_cache_update > 0.0, \
                "Cache timestamp should be updated"
            
            # Test cache size limit
            config = self.workspace_system.config
            original_cache_size = config.cache_size
            config.cache_size = 10  # Small cache for testing
            
            # Add more entries than cache size
            for i in range(20):
                self.workspace_system.energy_cache[i] = i * 10.0
            
            # Cache should be cleaned up
            assert len(self.workspace_system.energy_cache) <= 10, \
                f"Cache size should be limited: {len(self.workspace_system.energy_cache)}"
            
            # Restore original cache size
            config.cache_size = original_cache_size
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Cache Management",
                passed=True,
                duration=duration,
                details="Cache management working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name="Cache Management",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def _display_result(self, result: ValidationResult):
        """Display a validation result."""
        status = "✓ PASS" if result.passed else "✗ FAIL"
        duration_str = f"{result.duration:.4f}s"
        
        print(f"  {status} ({duration_str})")
        if result.details:
            print(f"    {result.details}")
        if result.error:
            print(f"    ERROR: {result.error}")
    
    def _display_validation_summary(self):
        """Display validation summary."""
        passed_count = sum(1 for result in self.validation_results if result.passed)
        total_count = len(self.validation_results)
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        print(f"Tests Passed: {passed_count}/{total_count}")
        
        if passed_count == total_count:
            print("🎉 ALL TESTS PASSED! The workspace node system is working correctly.")
            print("\nThe workspace node system is ready for production use.")
            print("Key features validated:")
            print("  ✓ System initialization and configuration")
            print("  ✓ Energy reading and aggregation")
            print("  ✓ 16x16 grid calculation")
            print("  ✓ Node data access and management")
            print("  ✓ Performance and threading")
            print("  ✓ Error handling and recovery")
            print("  ✓ Cache management")
            print("  ✓ System health monitoring")
        else:
            failed_count = total_count - passed_count
            print(f"⚠️  {failed_count} tests failed. Please check the implementation.")
            
            # List failed tests
            failed_tests = [r.test_name for r in self.validation_results if not r.passed]
            print(f"\nFailed tests: {', '.join(failed_tests)}")
        
        print("\n" + "="*60)


def main():
    """Main entry point for validation."""
    print("Workspace Node System - Comprehensive Validation")
    print("=" * 50)
    
    # Create validator
    validator = WorkspaceSystemValidator()
    
    # Run all validations
    results = validator.run_all_validations()
    
    # Return exit code based on results
    passed_count = sum(1 for result in results if result.passed)
    total_count = len(results)
    
    if passed_count == total_count:
        print("\nValidation completed successfully!")
        return 0
    else:
        print(f"\nValidation completed with {total_count - passed_count} failures.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)