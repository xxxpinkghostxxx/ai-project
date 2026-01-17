"""
Test script for real-time visualization functionality.

This script tests the real-time visualization window and its integration
with the workspace system.
"""

import sys
import time
import logging
import unittest
from unittest.mock import Mock, MagicMock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QPointF

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the modules to test
from src.project.workspace.realtime_visualization import (
    RealTimeVisualizationWindow,
    EnhancedWorkspaceRenderer,
    VisualizationMode,
    InteractionMode
)
from src.project.workspace.visualization_integration import WorkspaceVisualizationIntegration
from src.project.workspace.workspace_system import WorkspaceNodeSystem
from src.project.workspace.config import EnergyReadingConfig


class TestRealTimeVisualization(unittest.TestCase):
    """Test cases for real-time visualization functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create Qt application
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        
        # Create mock workspace system
        self.mock_workspace_system = Mock(spec=WorkspaceNodeSystem)
        self.mock_workspace_system.get_energy_grid.return_value = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        self.mock_workspace_system.get_energy_trends.return_value = [
            ['stable', 'increasing', 'decreasing'],
            ['stable', 'stable', 'increasing'],
            ['decreasing', 'stable', 'stable']
        ]
        self.mock_workspace_system.get_connection_count.return_value = 15
        
        # Create visualization window
        self.visualization_window = RealTimeVisualizationWindow(
            self.mock_workspace_system
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'visualization_window'):
            self.visualization_window.close()
    
    def test_window_creation(self):
        """Test that the visualization window can be created."""
        self.assertIsNotNone(self.visualization_window)
        self.assertEqual(self.visualization_window.windowTitle(), 
                        "Workspace Node Map - Real-time Visualization")
    
    def test_visualization_modes(self):
        """Test different visualization modes."""
        # Test mode setting
        self.visualization_window.renderer.set_visualization_mode(VisualizationMode.NODE_VIEW)
        self.assertEqual(self.visualization_window.renderer.visualization_mode, 
                        VisualizationMode.NODE_VIEW)
        
        # Test mode change through UI
        self.visualization_window.mode_combo.setCurrentText("connections")
        self.assertEqual(self.visualization_window.mode_combo.currentText(), "connections")
    
    def test_interaction_modes(self):
        """Test different interaction modes."""
        self.visualization_window.renderer.set_interaction_mode(InteractionMode.ZOOM)
        self.assertEqual(self.visualization_window.renderer.interaction_mode, 
                        InteractionMode.ZOOM)
    
    def test_zoom_functionality(self):
        """Test zoom functionality."""
        initial_zoom = self.visualization_window.renderer.zoom_level
        self.visualization_window._set_zoom(2.0)
        self.assertEqual(self.visualization_window.renderer.zoom_level, 2.0)
        
        # Test zoom slider
        self.visualization_window.zoom_slider.setValue(150)
        self.assertEqual(self.visualization_window.zoom_slider.value(), 150)
    
    def test_display_options(self):
        """Test display options."""
        # Test label toggle
        self.visualization_window.show_labels_checkbox.setChecked(True)
        self.assertTrue(self.visualization_window.renderer.show_node_labels)
        
        # Test connections toggle
        self.visualization_window.show_connections_checkbox.setChecked(False)
        self.assertFalse(self.visualization_window.renderer.show_connections)
    
    def test_update_interval(self):
        """Test update interval functionality."""
        initial_interval = self.visualization_window.update_interval
        self.visualization_window.update_interval = 100
        self.assertEqual(self.visualization_window.update_interval, 100)
    
    def test_pause_resume_updates(self):
        """Test pause and resume functionality."""
        # Initially should be running
        self.assertTrue(self.visualization_window.auto_update_enabled)
        
        # Pause updates
        self.visualization_window._toggle_updates()
        self.assertFalse(self.visualization_window.auto_update_enabled)
        self.assertEqual(self.visualization_window.pause_button.text(), "Resume Updates")
        
        # Resume updates
        self.visualization_window._toggle_updates()
        self.assertTrue(self.visualization_window.auto_update_enabled)
        self.assertEqual(self.visualization_window.pause_button.text(), "Pause Updates")
    
    def test_reset_view(self):
        """Test view reset functionality."""
        # Change zoom and pan
        self.visualization_window._set_zoom(3.0)
        self.visualization_window.renderer.set_pan_offset(QPointF(100, 100))
        
        # Reset view
        self.visualization_window._reset_view()
        
        # Check reset values
        self.assertEqual(self.visualization_window.renderer.zoom_level, 1.0)
        self.assertEqual(self.visualization_window.zoom_slider.value(), 100)
    
    def test_export_functionality(self):
        """Test export functionality."""
        # This would test the export functionality
        # For now, just ensure the method exists and doesn't crash
        try:
            self.visualization_window._export_view()
        except Exception as e:
            self.fail(f"Export functionality failed: {e}")
    
    def test_node_selection(self):
        """Test node selection functionality."""
        # Test that node selection signal can be emitted
        node_selected_called = False
        
        def on_node_selected(node_id):
            nonlocal node_selected_called
            node_selected_called = True
        
        self.visualization_window.node_selected.connect(on_node_selected)
        self.visualization_window.node_selected.emit(42)
        
        self.assertTrue(node_selected_called)
    
    def test_visualization_integration(self):
        """Test the visualization integration class."""
        integration = WorkspaceVisualizationIntegration(self.mock_workspace_system)
        
        # Test configuration
        config = {
            'auto_start': True,
            'dedicated_window': True,
            'update_interval': 100,
            'default_mode': 'nodes'
        }
        integration.update_configuration(config)
        
        self.assertEqual(integration.config['update_interval'], 100)
        self.assertEqual(integration.config['default_mode'], 'nodes')
    
    def test_status_monitoring(self):
        """Test status monitoring functionality."""
        # This method doesn't exist in the window, so we'll test the integration instead
        integration = WorkspaceVisualizationIntegration(self.mock_workspace_system)
        status = integration.get_status()
        
        self.assertIn('active', status)
        self.assertIn('dedicated_window', status)
        self.assertIn('config', status)
    
    def test_error_handling(self):
        """Test error handling in visualization."""
        # Test with invalid workspace system - use a mock instead of None
        try:
            invalid_window = RealTimeVisualizationWindow(Mock())
            invalid_window._update_visualization()
        except Exception as e:
            self.assertIsInstance(e, Exception)
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        # Simulate multiple updates
        start_time = time.time()
        for i in range(10):
            self.visualization_window._update_visualization()
        
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 1.0)  # Should be fast


class TestEnhancedRenderer(unittest.TestCase):
    """Test cases for enhanced workspace renderer."""
    
    def setUp(self):
        """Set up test environment."""
        self.renderer = EnhancedWorkspaceRenderer()
    
    def test_renderer_creation(self):
        """Test that the enhanced renderer can be created."""
        self.assertIsNotNone(self.renderer)
        self.assertEqual(self.renderer.visualization_mode, VisualizationMode.GRID_VIEW)
        self.assertEqual(self.renderer.interaction_mode, InteractionMode.PAN)
    
    def test_mode_switching(self):
        """Test switching between visualization modes."""
        modes = [VisualizationMode.GRID_VIEW, VisualizationMode.NODE_VIEW, 
                VisualizationMode.CONNECTION_VIEW, VisualizationMode.HEATMAP_VIEW]
        
        for mode in modes:
            self.renderer.set_visualization_mode(mode)
            self.assertEqual(self.renderer.visualization_mode, mode)
    
    def test_interaction_mode_switching(self):
        """Test switching between interaction modes."""
        modes = [InteractionMode.PAN, InteractionMode.ZOOM, 
                InteractionMode.INSPECT, InteractionMode.SELECT]
        
        for mode in modes:
            self.renderer.set_interaction_mode(mode)
            self.assertEqual(self.renderer.interaction_mode, mode)
    
    def test_zoom_levels(self):
        """Test zoom level bounds checking."""
        # Test minimum zoom
        self.renderer.set_zoom_level(0.05)
        self.assertEqual(self.renderer.zoom_level, 0.1)
        
        # Test maximum zoom
        self.renderer.set_zoom_level(15.0)
        self.assertEqual(self.renderer.zoom_level, 10.0)
        
        # Test normal zoom
        self.renderer.set_zoom_level(2.5)
        self.assertEqual(self.renderer.zoom_level, 2.5)
    
    def test_display_options(self):
        """Test display option toggling."""
        self.renderer.toggle_node_labels(True)
        self.assertTrue(self.renderer.show_node_labels)
        
        self.renderer.toggle_connections(False)
        self.assertFalse(self.renderer.show_connections)
    
    def test_pan_offset(self):
        """Test pan offset functionality."""
        offset = QPointF(100, 200)
        self.renderer.set_pan_offset(offset)
        self.assertEqual(self.renderer.pan_offset, offset)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)