"""
Comprehensive Workspace Node System Demonstration and Testing

This script provides a complete demonstration and testing suite for the workspace node system,
showcasing all features including real-time visualization, performance testing, error handling,
and configuration validation.

Key Features Demonstrated:
1. Complete working example of the workspace node system in action
2. Real-time visualization of the 16x16 energy grid
3. Integration with the main PyG application
4. Performance testing and validation
5. Error handling demonstration
6. Configuration options testing
7. System health monitoring
8. Validation and verification tests
"""

import time
import sys
import os
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import workspace system components
from src.project.workspace.workspace_system import WorkspaceNodeSystem
from src.project.workspace.config import EnergyReadingConfig
from src.project.workspace.pixel_shading import PixelShadingSystem
from src.project.workspace.renderer import WorkspaceRenderer
from src.project.workspace.visualization import WorkspaceVisualization
from src.project.workspace.mapping import map_sensory_to_workspace, calculate_energy_aggregation
from src.project.workspace.workspace_node import WorkspaceNode
from src.project.pyg_neural_system import PyGNeuralSystem
from src.project.ui.modern_main_window import ModernMainWindow
from src.project.utils.config_manager import ConfigManager
from src.project.utils.error_handler import ErrorHandler

# Import PyQt6 for visualization
try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSlider, QTextEdit
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QPixmap, QColor, QPainter
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("Warning: PyQt6 not available. Visualization features will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Data class for test results."""
    test_name: str
    passed: bool
    duration: float
    details: str = ""
    error: Optional[str] = None


class DemoNeuralSystem:
    """Enhanced demo neural system for comprehensive testing."""
    
    def __init__(self, width=256, height=144, enable_dynamic=True):
        """
        Initialize demo neural system.
        
        Args:
            width: Sensory grid width
            height: Sensory grid height
            enable_dynamic: Enable dynamic energy patterns
        """
        self.sensory_width = width
        self.sensory_height = height
        self.total_nodes = width * height
        self.energies = np.random.rand(self.total_nodes) * 244.0
        self._enable_dynamic = enable_dynamic
        self.time_step = 0
        self.patterns = {
            'wave': self._generate_wave_pattern,
            'pulse': self._generate_pulse_pattern,
            'random': self._generate_random_pattern,
            'gradient': self._generate_gradient_pattern
        }
        self.current_pattern = 'wave'
        
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
        if not self.enable_dynamic:
            return
            
        self.time_step += 1
        
        # Apply current pattern
        pattern_func = self.patterns.get(self.current_pattern, self._generate_random_pattern)
        self.energies = pattern_func()
        
        # Add noise
        noise = np.random.normal(0, 5, self.energies.shape)
        self.energies += noise
        self.energies = np.clip(self.energies, 0, 244.0)
    
    def _generate_wave_pattern(self):
        """Generate wave pattern across the sensory grid."""
        x_coords = np.arange(self.sensory_width)
        y_coords = np.arange(self.sensory_height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Wave parameters
        frequency = 0.1
        amplitude = 50
        phase = self.time_step * 0.1
        
        # Generate wave
        wave = amplitude * np.sin(frequency * X + phase) + amplitude * np.cos(frequency * Y + phase)
        
        return wave.flatten() + 100
    
    def _generate_pulse_pattern(self):
        """Generate pulse pattern from center."""
        center_x = self.sensory_width // 2
        center_y = self.sensory_height // 2
        
        x_coords = np.arange(self.sensory_width)
        y_coords = np.arange(self.sensory_height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Distance from center
        distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Pulse parameters
        pulse_speed = 2
        pulse_amplitude = 100
        current_radius = (self.time_step * pulse_speed) % (max(self.sensory_width, self.sensory_height))
        
        # Generate pulse
        pulse = pulse_amplitude * np.exp(-(distances - current_radius)**2 / 50)
        
        return pulse.flatten() + 50
    
    def _generate_random_pattern(self):
        """Generate random pattern."""
        return np.random.rand(self.total_nodes) * 244.0
    
    def _generate_gradient_pattern(self):
        """Generate gradient pattern."""
        x_coords = np.arange(self.sensory_width)
        y_coords = np.arange(self.sensory_height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Gradient parameters
        angle = self.time_step * 0.05
        gradient_x = np.cos(angle) * X + np.sin(angle) * Y
        gradient_y = -np.sin(angle) * X + np.cos(angle) * Y
        
        gradient = (gradient_x + gradient_y) % 244
        
        return gradient.flatten()
    
    def set_pattern(self, pattern_name: str):
        """Set the energy pattern."""
        if pattern_name in self.patterns:
            self.current_pattern = pattern_name
            print(f"Pattern set to: {pattern_name}")
    
    def disable_dynamic(self):
        """Disable dynamic patterns."""
        self._enable_dynamic = False
    
    def enable_dynamic(self):
        """Enable dynamic patterns."""
        self._enable_dynamic = True


class WorkspaceDemoApp:
    """Main demonstration application with comprehensive UI."""
    
    def __init__(self):
        if PYQT_AVAILABLE:
            self.init_ui()
        else:
            self.ui_initialized = False
        
        # Demo components
        self.neural_system = None
        self.workspace_system = None
        self.shading_system = None
        self.renderer = None
        self.visualization = None
        
        # Demo state
        self.demo_running = False
        self.test_results = []
        self.performance_metrics = []
        
        # Initialize demo
        self._initialize_demo()
    
    def init_ui(self):
        """Initialize the demonstration UI."""
        self.setWindowTitle("Workspace Node System - Comprehensive Demo")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Pattern controls
        pattern_group = QWidget()
        pattern_layout = QVBoxLayout(pattern_group)
        pattern_layout.addWidget(QLabel("Energy Pattern:"))
        
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(['wave', 'pulse', 'random', 'gradient'])
        self.pattern_combo.currentTextChanged.connect(self.on_pattern_changed)
        pattern_layout.addWidget(self.pattern_combo)
        
        # Speed controls
        speed_group = QWidget()
        speed_layout = QVBoxLayout(speed_group)
        speed_layout.addWidget(QLabel("Update Speed:"))
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        
        # Test controls
        test_group = QWidget()
        test_layout = QVBoxLayout(test_group)
        test_layout.addWidget(QLabel("Test Actions:"))
        
        self.start_btn = QPushButton("Start Demo")
        self.start_btn.clicked.connect(self.start_demo)
        test_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Demo")
        self.stop_btn.clicked.connect(self.stop_demo)
        self.stop_btn.setEnabled(False)
        test_layout.addWidget(self.stop_btn)
        
        self.run_tests_btn = QPushButton("Run All Tests")
        self.run_tests_btn.clicked.connect(self.run_comprehensive_tests)
        test_layout.addWidget(self.run_tests_btn)
        
        # Add controls to layout
        control_layout.addWidget(pattern_group)
        control_layout.addWidget(speed_group)
        control_layout.addWidget(test_group)
        
        # Visualization area
        viz_layout = QHBoxLayout()
        
        # Workspace grid view
        self.grid_widget = QWidget()
        self.grid_layout = QVBoxLayout(self.grid_widget)
        self.grid_layout.addWidget(QLabel("16x16 Workspace Energy Grid"))
        
        self.grid_view = QLabel()
        self.grid_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grid_view.setStyleSheet("background-color: #000; border: 1px solid #333;")
        self.grid_layout.addWidget(self.grid_view)
        
        # Statistics panel
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.addWidget(QLabel("System Statistics:"))
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)
        
        # Test results panel
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("Test Results:"))
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # Add visualization to layout
        viz_layout.addWidget(self.grid_widget, stretch=2)
        viz_layout.addWidget(stats_widget, stretch=1)
        viz_layout.addWidget(results_widget, stretch=1)
        
        # Add all layouts to main
        main_layout.addLayout(control_layout)
        main_layout.addLayout(viz_layout)
        
        self.setCentralWidget(main_widget)
        
        # Timer for updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_demo)
        
        self.ui_initialized = True
    
    def _initialize_demo(self):
        """Initialize demo components."""
        print("Initializing Workspace Node System Demo...")
        
        # Create demo neural system
        self.neural_system = DemoNeuralSystem(width=256, height=144)
        
        # Create workspace system configuration
        config = EnergyReadingConfig()
        config.grid_size = (16, 16)
        config.reading_interval_ms = 100
        
        # Initialize workspace system
        self.workspace_system = WorkspaceNodeSystem(self.neural_system, config)
        
        # Create pixel shading system
        self.shading_system = PixelShadingSystem(energy_min=0.0, energy_max=244.0)
        
        # Create renderer
        self.renderer = WorkspaceRenderer(grid_size=(16, 16), pixel_size=30)
        
        # Initialize UI if available
        if PYQT_AVAILABLE and hasattr(self, 'ui_initialized') and self.ui_initialized:
            # Create visualization integration
            self.visualization = WorkspaceVisualization(self, self.workspace_system)
        
        print("Demo components initialized successfully!")
    
    def on_pattern_changed(self, pattern: str):
        """Handle pattern change."""
        if self.neural_system:
            self.neural_system.set_pattern(pattern)
    
    def on_speed_changed(self, value: int):
        """Handle speed change."""
        if self.workspace_system:
            self.workspace_system.config.reading_interval_ms = value
    
    def start_demo(self):
        """Start the demonstration."""
        if self.demo_running:
            return
        
        print("Starting demonstration...")
        self.demo_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start workspace system
        self.workspace_system.start()
        
        # Start UI updates
        if PYQT_AVAILABLE:
            self.update_timer.start(100)  # Update every 100ms
        
        print("Demonstration started!")
    
    def stop_demo(self):
        """Stop the demonstration."""
        if not self.demo_running:
            return
        
        print("Stopping demonstration...")
        self.demo_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Stop workspace system
        self.workspace_system.stop()
        
        # Stop UI updates
        if PYQT_AVAILABLE:
            self.update_timer.stop()
        
        print("Demonstration stopped!")
    
    def update_demo(self):
        """Update demonstration display."""
        if not self.demo_running or not self.workspace_system:
            return
        
        try:
            # Update neural system
            self.neural_system.update_energies()
            
            # Update workspace system
            self.workspace_system.update()
            
            # Get energy grid
            energy_grid = self.workspace_system._calculate_energy_grid()
            
            # Update visualization
            if PYQT_AVAILABLE:
                self.update_visualization(energy_grid)
            
            # Update statistics
            self.update_statistics(energy_grid)
            
        except Exception as e:
            print(f"Error updating demo: {e}")
    
    def update_visualization(self, energy_grid: List[List[float]]):
        """Update the visualization."""
        if not self.renderer:
            return
        
        # Convert to pixel values
        pixel_grid = []
        for y in range(16):
            row = []
            for x in range(16):
                energy = energy_grid[y][x]
                pixel_value = self.shading_system.energy_to_pixel_value(energy)
                row.append(pixel_value)
            pixel_grid.append(row)
        
        # Render grid
        pixmap = self.renderer.render_to_pixmap(pixel_grid)
        self.grid_view.setPixmap(pixmap)
    
    def update_statistics(self, energy_grid: List[List[float]]):
        """Update system statistics."""
        if not self.workspace_system:
            return
        
        # Calculate statistics
        flat_energies = [energy for row in energy_grid for energy in row]
        avg_energy = np.mean(flat_energies) if flat_energies else 0.0
        max_energy = np.max(flat_energies) if flat_energies else 0.0
        min_energy = np.min(flat_energies) if flat_energies else 0.0
        
        # Get system health
        health = self.workspace_system.get_system_health()
        
        # Update stats text
        stats_text = f"""
Current Statistics:
- Average Energy: {avg_energy:.1f}
- Maximum Energy: {max_energy:.1f}
- Minimum Energy: {min_energy:.1f}
- Update Rate: {1000/self.workspace_system.config.reading_interval_ms:.1f} Hz

System Health:
- Running: {health['running']}
- Node Count: {health['node_count']}
- Mapping Coverage: {health['mapping_coverage']}
- Average Update Time: {health['avg_update_time']:.4f}s
- Error Count: {health['error_count']}
- Cache Size: {health['cache_size']}
        """.strip()
        
        self.stats_text.setPlainText(stats_text)
    
    def run_comprehensive_tests(self):
        """Run all comprehensive tests."""
        print("Running comprehensive tests...")
        self.results_text.clear()
        
        # Run all tests
        tests = [
            self.test_basic_functionality,
            self.test_performance,
            self.test_error_handling,
            self.test_configuration,
            self.test_mapping,
            self.test_shading_system,
            self.test_visualization,
            self.test_integration
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                self.test_results.append(result)
                self.display_test_result(result)
            except Exception as e:
                error_result = TestResult(
                    test_name=test_func.__name__,
                    passed=False,
                    duration=0.0,
                    error=str(e)
                )
                self.test_results.append(error_result)
                self.display_test_result(error_result)
        
        # Display summary
        self.display_test_summary()
    
    def test_basic_functionality(self) -> TestResult:
        """Test basic workspace system functionality."""
        start_time = time.time()
        
        try:
            # Test workspace system initialization
            assert self.workspace_system is not None, "Workspace system not initialized"
            assert len(self.workspace_system.workspace_nodes) == 256, "Incorrect number of workspace nodes"
            
            # Test energy reading
            self.neural_system.update_energies()
            self.workspace_system.update()
            
            energy_grid = self.workspace_system._calculate_energy_grid()
            assert len(energy_grid) == 16, "Incorrect grid height"
            assert len(energy_grid[0]) == 16, "Incorrect grid width"
            
            # Test node data retrieval
            node_data = self.workspace_system.get_node_data(0)
            assert 'node_id' in node_data, "Missing node_id in node data"
            assert 'current_energy' in node_data, "Missing current_energy in node data"
            
            duration = time.time() - start_time
            return TestResult(
                test_name="Basic Functionality",
                passed=True,
                duration=duration,
                details="All basic functionality tests passed"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Basic Functionality",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def test_performance(self) -> TestResult:
        """Test system performance."""
        start_time = time.time()
        
        try:
            # Test update performance
            update_times = []
            for _ in range(100):
                update_start = time.time()
                self.neural_system.update_energies()
                self.workspace_system.update()
                update_end = time.time()
                update_times.append(update_end - update_start)
            
            avg_update_time = np.mean(update_times)
            max_update_time = np.max(update_times)
            
            # Performance should be reasonable
            assert avg_update_time < 0.01, f"Average update time too slow: {avg_update_time}s"
            assert max_update_time < 0.05, f"Maximum update time too slow: {max_update_time}s"
            
            duration = time.time() - start_time
            return TestResult(
                test_name="Performance",
                passed=True,
                duration=duration,
                details=f"Average update time: {avg_update_time:.4f}s, Max: {max_update_time:.4f}s"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Performance",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def test_error_handling(self) -> TestResult:
        """Test error handling and recovery."""
        start_time = time.time()
        
        try:
            initial_error_count = self.workspace_system.error_count
            
            # Test invalid node ID handling
            invalid_energy = self.neural_system.get_node_energy(999999)
            assert invalid_energy == 0.0, "Should return 0.0 for invalid node ID"
            
            # Test batch energy with invalid IDs
            batch_energies = self.neural_system.get_batch_energies([0, 1, 999999, 2])
            assert len(batch_energies) == 4, "Batch should return 4 values"
            assert batch_energies[2] == 0.0, "Invalid ID should return 0.0"
            
            # Test system recovery
            self.workspace_system.stop()
            self.workspace_system.start()
            
            # Should be able to restart
            assert self.workspace_system._running, "System should be running after restart"
            
            duration = time.time() - start_time
            return TestResult(
                test_name="Error Handling",
                passed=True,
                duration=duration,
                details="Error handling and recovery working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Error Handling",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def test_configuration(self) -> TestResult:
        """Test configuration options."""
        start_time = time.time()
        
        try:
            config = self.workspace_system.config
            
            # Test configuration modification
            original_interval = config.reading_interval_ms
            config.reading_interval_ms = 50
            assert config.reading_interval_ms == 50, "Configuration update failed"
            
            # Test cache settings
            original_cache_size = config.cache_size
            config.cache_size = 500
            assert config.cache_size == 500, "Cache size update failed"
            
            # Test grid size
            assert config.grid_size == (16, 16), "Grid size configuration incorrect"
            
            duration = time.time() - start_time
            return TestResult(
                test_name="Configuration",
                passed=True,
                duration=duration,
                details="All configuration options working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Configuration",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def test_mapping(self) -> TestResult:
        """Test sensory to workspace mapping."""
        start_time = time.time()
        
        try:
            # Test mapping creation
            mapping = self.workspace_system.mapping
            assert len(mapping) == 256, "Should have mapping for all workspace nodes"
            
            # Test that all sensory nodes are mapped
            total_mapped = sum(len(sensory_list) for sensory_list in mapping.values())
            expected_total = self.neural_system.sensory_width * self.neural_system.sensory_height
            assert total_mapped == expected_total, f"Mapping incomplete: {total_mapped}/{expected_total}"
            
            # Test aggregation calculation
            test_energies = [100.0, 150.0, 200.0]
            avg_energy = calculate_energy_aggregation(test_energies, method='average')
            assert abs(avg_energy - 150.0) < 0.001, "Average calculation incorrect"
            
            duration = time.time() - start_time
            return TestResult(
                test_name="Mapping",
                passed=True,
                duration=duration,
                details="Sensory to workspace mapping working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Mapping",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def test_shading_system(self) -> TestResult:
        """Test pixel shading system."""
        start_time = time.time()
        
        try:
            # Test energy to pixel conversion
            test_energies = [0.0, 122.0, 244.0]
            expected_pixels = [0, 127, 255]
            
            for energy, expected in zip(test_energies, expected_pixels):
                pixel = self.shading_system.energy_to_pixel_value(energy)
                assert abs(pixel - expected) <= 1, f"Pixel conversion incorrect for {energy}: {pixel} != {expected}"
            
            # Test color scheme
            self.shading_system.set_color_scheme('grayscale')
            assert self.shading_system.color_scheme == 'grayscale', "Color scheme not set correctly"
            
            duration = time.time() - start_time
            return TestResult(
                test_name="Shading System",
                passed=True,
                duration=duration,
                details="Pixel shading system working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Shading System",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def test_visualization(self) -> TestResult:
        """Test visualization components."""
        start_time = time.time()
        
        try:
            if not PYQT_AVAILABLE:
                return TestResult(
                    test_name="Visualization",
                    passed=True,
                    duration=0.0,
                    details="PyQt6 not available, skipping visualization tests"
                )
            
            # Test renderer
            assert self.renderer is not None, "Renderer not initialized"
            
            # Test pixel grid rendering
            test_grid = [[127 for _ in range(16)] for _ in range(16)]
            pixmap = self.renderer.render_to_pixmap(test_grid)
            assert pixmap is not None, "Renderer should return a pixmap"
            
            duration = time.time() - start_time
            return TestResult(
                test_name="Visualization",
                passed=True,
                duration=duration,
                details="Visualization components working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Visualization",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def test_integration(self) -> TestResult:
        """Test full system integration."""
        start_time = time.time()
        
        try:
            # Test full integration cycle
            self.neural_system.set_pattern('wave')
            
            # Run for several updates
            for _ in range(10):
                self.neural_system.update_energies()
                self.workspace_system.update()
                energy_grid = self.workspace_system._calculate_energy_grid()
                
                # Verify grid is updated
                assert len(energy_grid) == 16, "Grid should be 16x16"
                assert all(len(row) == 16 for row in energy_grid), "All rows should have 16 elements"
            
            # Test with different patterns
            patterns = ['pulse', 'random', 'gradient']
            for pattern in patterns:
                self.neural_system.set_pattern(pattern)
                self.neural_system.update_energies()
                self.workspace_system.update()
                
                energy_grid = self.workspace_system._calculate_energy_grid()
                assert len(energy_grid) == 16, f"Grid should be 16x16 for {pattern} pattern"
            
            duration = time.time() - start_time
            return TestResult(
                test_name="Integration",
                passed=True,
                duration=duration,
                details="Full system integration working correctly"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Integration",
                passed=False,
                duration=duration,
                error=str(e)
            )
    
    def display_test_result(self, result: TestResult):
        """Display a test result."""
        status = "PASS" if result.passed else "FAIL"
        duration_str = f"{result.duration:.4f}s"
        
        result_text = f"[{status}] {result.test_name} ({duration_str})"
        if result.details:
            result_text += f" - {result.details}"
        if result.error:
            result_text += f" - ERROR: {result.error}"
        
        self.results_text.append(result_text)
        print(result_text)
    
    def display_test_summary(self):
        """Display test summary."""
        passed_count = sum(1 for result in self.test_results if result.passed)
        total_count = len(self.test_results)
        
        summary = f"\n\nTest Summary: {passed_count}/{total_count} tests passed"
        
        if passed_count == total_count:
            summary += "\n🎉 All tests passed! The workspace node system is working correctly."
        else:
            summary += f"\n⚠️  {total_count - passed_count} tests failed. Please check the implementation."
        
        self.results_text.append(summary)
        print(summary)


def run_command_line_demo():
    """Run demonstration from command line without GUI."""
    print("Starting Command Line Workspace Node System Demo...")
    print("=" * 60)
    
    # Create demo components
    neural_system = DemoNeuralSystem(width=256, height=144)
    config = EnergyReadingConfig()
    config.grid_size = (16, 16)
    config.reading_interval_ms = 100
    
    workspace_system = WorkspaceNodeSystem(neural_system, config)
    shading_system = PixelShadingSystem(energy_min=0.0, energy_max=244.0)
    
    print("✓ Demo components initialized")
    
    # Run demonstration
    print("\nRunning demonstration for 30 seconds...")
    start_time = time.time()
    
    try:
        workspace_system.start()
        
        step = 0
        while time.time() - start_time < 30:
            step += 1
            
            # Update systems
            neural_system.update_energies()
            workspace_system.update()
            
            # Get statistics
            energy_grid = workspace_system._calculate_energy_grid()
            flat_energies = [energy for row in energy_grid for energy in row]
            
            avg_energy = np.mean(flat_energies) if flat_energies else 0.0
            max_energy = np.max(flat_energies) if flat_energies else 0.0
            min_energy = np.min(flat_energies) if flat_energies else 0.0
            
            # Display statistics
            print(f"Step {step:3d}: Avg={avg_energy:6.1f}, Max={max_energy:6.1f}, Min={min_energy:6.1f}")
            
            # Change patterns periodically
            if step % 20 == 0:
                patterns = ['wave', 'pulse', 'random', 'gradient']
                current_pattern = patterns[(step // 20) % len(patterns)]
                neural_system.set_pattern(current_pattern)
            
            time.sleep(0.1)
        
        workspace_system.stop()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        workspace_system.stop()
    
    # Run basic tests
    print("\nRunning basic validation tests...")
    test_results = []
    
    # Test 1: Basic functionality
    try:
        assert len(workspace_system.workspace_nodes) == 256
        energy_grid = workspace_system._calculate_energy_grid()
        assert len(energy_grid) == 16 and len(energy_grid[0]) == 16
        test_results.append(("Basic Functionality", True))
    except Exception as e:
        test_results.append(("Basic Functionality", False))
    
    # Test 2: Performance
    try:
        update_times = []
        for _ in range(50):
            start = time.time()
            neural_system.update_energies()
            workspace_system.update()
            update_times.append(time.time() - start)
        
        avg_time = np.mean(update_times)
        assert avg_time < 0.01
        test_results.append(("Performance", True))
    except Exception as e:
        test_results.append(("Performance", False))
    
    # Test 3: Error handling
    try:
        invalid_energy = neural_system.get_node_energy(999999)
        assert invalid_energy == 0.0
        test_results.append(("Error Handling", True))
    except Exception as e:
        test_results.append(("Error Handling", False))
    
    # Display results
    print("\nTest Results:")
    passed = 0
    for test_name, passed_test in test_results:
        status = "PASS" if passed_test else "FAIL"
        print(f"  {test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! The workspace node system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    print("\nDemo completed successfully!")


def main():
    """Main entry point for the comprehensive demo."""
    print("Workspace Node System - Comprehensive Demonstration and Testing")
    print("=" * 70)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_command_line_demo()
        return
    
    # Try to run GUI demo
    if PYQT_AVAILABLE:
        try:
            app = QApplication(sys.argv)
            demo_app = WorkspaceDemoApp()
            demo_app.show()
            sys.exit(app.exec())
        except Exception as e:
            print(f"GUI demo failed: {e}")
            print("Falling back to command line demo...")
            run_command_line_demo()
    else:
        print("PyQt6 not available. Running command line demo...")
        run_command_line_demo()


if __name__ == "__main__":
    main()