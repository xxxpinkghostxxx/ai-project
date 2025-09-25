"""
Integration tests for UI components
Tests interactions between UI components, workflows, and real-world scenarios.
"""

import sys
import os
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Dear PyGui and other dependencies
sys.modules['dearpygui'] = Mock()
sys.modules['dearpygui.dearpygui'] = Mock()
sys.modules['numba'] = Mock()
np_mock = Mock()
np_mock.__version__ = '1.24.0'
sys.modules['numpy'] = np_mock
sys.modules['PIL'] = Mock()
sys.modules['PIL.ImageGrab'] = Mock()
sys.modules['cv2'] = Mock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch_geometric'] = Mock()
sys.modules['torch_geometric.data'] = Mock()
sys.modules['mss'] = Mock()

import torch
import dearpygui.dearpygui as dpg
import numpy as np
from torch_geometric.data import Data

from ui.ui_engine import (
    create_main_window, start_simulation_callback, stop_simulation_callback,
    reset_simulation_callback, update_ui_display, update_graph_visualization,
    run_ui, get_coordinator, update_operation_status, clear_operation_status
)
from ui.ui_state_manager import get_ui_state_manager, cleanup_ui_state
from ui.screen_graph import capture_screen, create_pixel_gray_graph
from core.interfaces.service_registry import IServiceRegistry
from core.interfaces.simulation_coordinator import ISimulationCoordinator


class TestUIIntegration(unittest.TestCase):
    """Integration tests for UI component interactions."""

    def setUp(self):
        """Set up integration test environment."""
        self.ui_state = get_ui_state_manager()

        # Mock service registry and coordinator
        self.mock_registry = Mock(spec=IServiceRegistry)
        self.mock_coordinator = Mock(spec=ISimulationCoordinator)
        self.mock_coordinator.start = Mock()
        self.mock_coordinator.stop = Mock()
        self.mock_coordinator.reset = Mock()
        self.mock_coordinator.get_simulation_state = Mock(return_value=Mock(
            step_count=100, total_energy=50.0
        ))
        self.mock_coordinator.get_neural_graph = Mock(return_value=Mock(
            num_nodes=10, num_edges=20
        ))
        self.mock_coordinator.get_performance_metrics = Mock(return_value={
            'health_score': 85.0
        })
        self.mock_coordinator.save_neural_map = Mock(return_value=True)
        self.mock_coordinator.load_neural_map = Mock(return_value=True)
        self.mock_coordinator.update_configuration = Mock()

        self.mock_registry.resolve = Mock(return_value=self.mock_coordinator)

        # Mock visualization service
        self.mock_visualization = Mock()
        self.mock_visualization.initialize_visualization = Mock(return_value=True)
        self.mock_visualization.update_visualization_data = Mock()

        # Setup comprehensive DPG mocks
        self._setup_dpg_mocks()

        # Setup screen capture mocks
        self._setup_screen_mocks()

    def _setup_dpg_mocks(self):
        """Setup comprehensive DPG mocks for integration testing."""
        dpg.set_value = Mock()
        dpg.get_value = Mock(return_value=0.5)
        dpg.configure_item = Mock()
        dpg.add_text = Mock(return_value="mock_text")
        dpg.add_button = Mock(return_value="mock_button")
        dpg.add_checkbox = Mock(return_value="mock_checkbox")
        dpg.add_slider_float = Mock(return_value="mock_slider")
        dpg.add_input_int = Mock(return_value="mock_input")
        dpg.add_color_edit = Mock(return_value="mock_color")
        dpg.add_plot = Mock(return_value="mock_plot")
        dpg.add_plot_axis = Mock(return_value="mock_axis")
        dpg.add_line_series = Mock(return_value="mock_series")
        dpg.add_collapsing_header = Mock(return_value="mock_header")
        dpg.add_group = Mock(return_value="mock_group")
        dpg.add_child_window = Mock(return_value="mock_window")
        dpg.add_tab_bar = Mock(return_value="mock_tab_bar")
        dpg.add_tab = Mock(return_value="mock_tab")
        dpg.add_drawlist = Mock(return_value="mock_drawlist")
        dpg.add_input_text = Mock(return_value="mock_input_text")
        dpg.add_menu_bar = Mock(return_value="mock_menu_bar")
        dpg.add_menu = Mock(return_value="mock_menu")
        dpg.add_menu_item = Mock(return_value="mock_menu_item")
        dpg.add_window = Mock(return_value="mock_window")
        dpg.add_separator = Mock()
        dpg.add_theme = Mock(return_value="mock_theme")
        dpg.add_theme_component = Mock(return_value="mock_component")
        dpg.add_theme_color = Mock()
        dpg.add_theme_style = Mock()
        dpg.bind_theme = Mock()
        dpg.set_global_font_scale = Mock()
        dpg.create_context = Mock()
        dpg.create_viewport = Mock()
        dpg.set_viewport_resizable = Mock()
        dpg.setup_dearpygui = Mock()
        dpg.show_viewport = Mock()
        dpg.is_dearpygui_running = Mock(return_value=False)
        dpg.render_dearpygui_frame = Mock()
        dpg.destroy_context = Mock()
        dpg.clear_draw_list = Mock()
        dpg.draw_circle = Mock()
        dpg.draw_line = Mock()
        dpg.get_item_rect_size = Mock(return_value=[800, 600])
        dpg.set_primary_window = Mock()
        dpg.toggle_viewport_fullscreen = Mock()
        dpg.stop_dearpygui = Mock()
        dpg.add_tooltip = Mock(return_value="mock_tooltip")
        dpg.does_item_exist = Mock(return_value=False)

    def _setup_screen_mocks(self):
        """Setup screen capture mocks."""
        np.array = Mock(return_value=Mock())
        np.dot = Mock(return_value=Mock())
        np.zeros = Mock(return_value=Mock())
        np.uint8 = Mock()
        np.float32 = Mock()

        import mss
        mss.mss = Mock(return_value=Mock())
        mss.mss.return_value.__enter__ = Mock(return_value=Mock())
        mss.mss.return_value.__exit__ = Mock(return_value=None)
        mss.mss.return_value.__enter__.return_value.monitors = [Mock()]
        mss.mss.return_value.__enter__.return_value.grab = Mock(return_value=Mock())

        torch.tensor = Mock(return_value=Mock())
        torch.empty = Mock(return_value=Mock())
        torch.long = Mock()
        torch.float32 = Mock()

    def tearDown(self):
        """Clean up after tests."""
        cleanup_ui_state()

    def test_ui_engine_state_manager_integration(self):
        """Test integration between UI engine and state manager."""
        # Create main window
        create_main_window()

        # Test state changes through callbacks
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            start_simulation_callback()
            self.assertTrue(self.ui_state.get_simulation_running())

            stop_simulation_callback()
            self.assertFalse(self.ui_state.get_simulation_running())

            reset_simulation_callback()
            self.assertFalse(self.ui_state.get_simulation_running())

    def test_ui_display_state_synchronization(self):
        """Test that UI display stays synchronized with state manager."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            # Start simulation
            start_simulation_callback()

            # Update display
            update_ui_display()

            # Verify display calls
            dpg.set_value.assert_any_call("status_text", "Status: Running")
            dpg.set_value.assert_any_call("nodes_text", "Nodes: 10")
            dpg.set_value.assert_any_call("edges_text", "Edges: 20")

            # Stop simulation
            stop_simulation_callback()

            # Update display again
            update_ui_display()

            dpg.set_value.assert_any_call("status_text", "Status: Stopped")

    def test_graph_visualization_state_integration(self):
        """Test graph visualization integration with state manager."""
        # Create mock graph
        mock_graph = Mock()
        mock_graph.node_labels = [
            {'id': 0, 'energy': 0.8, 'state': 'active', 'pos': [100, 100]},
            {'id': 1, 'energy': 0.3, 'state': 'inactive', 'pos': [200, 200]}
        ]
        mock_graph.edge_index = Mock()
        mock_graph.edge_index.shape = [2, 1]
        mock_graph.edge_index.__getitem__ = Mock(return_value=Mock())
        mock_graph.edge_index.__getitem__.return_value.item.return_value = 0

        # Update state with graph
        self.ui_state.update_graph(mock_graph)

        # Update visualization
        update_graph_visualization()

        # Verify visualization used the graph
        dpg.clear_draw_list.assert_called()
        dpg.draw_circle.assert_called()

    def test_screen_capture_ui_integration(self):
        """Test integration of screen capture with UI state."""
        # Capture screen
        screen_data = capture_screen(scale=0.25)
        self.assertIsNotNone(screen_data)

        # Create graph from screen
        graph = create_pixel_gray_graph(screen_data)
        self.assertIsNotNone(graph)

        # Update UI state with graph
        self.ui_state.update_graph(graph)

        # Verify state has the graph
        self.assertEqual(self.ui_state.get_latest_graph(), graph)

    def test_service_registry_ui_integration(self):
        """Test UI integration with service registry."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            # Test coordinator resolution
            coordinator = get_coordinator()
            self.assertEqual(coordinator, self.mock_coordinator)

            # Test UI operations that use coordinator
            start_simulation_callback()
            self.mock_coordinator.start.assert_called_once()

            stop_simulation_callback()
            self.mock_coordinator.stop.assert_called_once()

    def test_ui_workflow_simulation_lifecycle(self):
        """Test complete UI workflow for simulation lifecycle."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            # Initial state
            self.assertFalse(self.ui_state.get_simulation_running())

            # Start simulation via UI
            update_operation_status("Starting simulation", 0.1)
            start_simulation_callback()
            self.assertTrue(self.ui_state.get_simulation_running())

            # Update UI during simulation
            update_ui_display()
            update_graph_visualization()

            # Add some live data
            self.ui_state.add_live_feed_data('energy_history', 0.7)
            self.ui_state.add_live_feed_data('node_activity_history', 50)

            # Stop simulation via UI
            update_operation_status("Stopping simulation", 0.8)
            stop_simulation_callback()
            self.assertFalse(self.ui_state.get_simulation_running())

            # Reset simulation
            reset_simulation_callback()
            clear_operation_status()

            # Verify final state
            self.assertFalse(self.ui_state.get_simulation_running())
            data = self.ui_state.get_live_feed_data()
            self.assertEqual(len(data['energy_history']), 1)

    def test_ui_error_handling_integration(self):
        """Test error handling across UI components."""
        # Test with no service registry
        with patch('ui.ui_engine._service_registry', None):
            start_simulation_callback()
            stop_simulation_callback()
            reset_simulation_callback()

            # Should handle gracefully without crashing
            self.assertFalse(self.ui_state.get_simulation_running())

        # Test with failing coordinator
        self.mock_coordinator.start.side_effect = Exception("Coordinator failed")
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            start_simulation_callback()
            # Should handle exception
            self.assertFalse(self.ui_state.get_simulation_running())

    def test_ui_visualization_service_integration(self):
        """Test UI integration with visualization service."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            with patch('ui.ui_engine._visualization_service', self.mock_visualization):
                # Update display should send data to visualization service
                update_ui_display()

                # Verify visualization service was called
                self.mock_visualization.update_visualization_data.assert_called()

    def test_concurrent_ui_operations(self):
        """Test concurrent UI operations."""
        import threading

        results = []
        errors = []

        def ui_operation_thread(operation_id):
            try:
                # Perform various UI operations
                update_ui_display()
                update_graph_visualization()
                self.ui_state.add_live_feed_data('test', operation_id)
                results.append(operation_id)
            except Exception as e:
                errors.append(f"Thread {operation_id}: {e}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=ui_operation_thread, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All operations should succeed
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)

        # State should be consistent
        data = self.ui_state.get_live_feed_data()
        self.assertEqual(len(data['test']), 5)

    def test_ui_performance_under_load(self):
        """Test UI performance with high-frequency updates."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            start_time = time.time()

            # Simulate high-frequency UI updates
            for i in range(100):
                update_ui_display()
                update_graph_visualization()
                self.ui_state.add_live_feed_data('perf_test', float(i) * 0.01)

            end_time = time.time()
            duration = end_time - start_time

            # Should complete within reasonable time
            self.assertLess(duration, 5.0, "UI updates too slow under load")

            # State should be properly maintained
            data = self.ui_state.get_live_feed_data()
            self.assertEqual(len(data['perf_test']), 100)

    def test_ui_memory_management_integration(self):
        """Test memory management across UI components."""
        # Create multiple graphs
        graphs = []
        for i in range(10):
            mock_graph = Mock()
            mock_graph.node_labels = [{'id': j, 'energy': 0.5} for j in range(100)]
            graphs.append(mock_graph)

        # Update state with graphs multiple times
        for graph in graphs:
            self.ui_state.update_graph(graph)

        # Force cleanup
        cleanup_ui_state()

        # Create new state manager
        new_state = get_ui_state_manager()
        self.assertIsNotNone(new_state)
        self.assertFalse(new_state.get_simulation_running())

    def test_real_world_ui_scenario(self):
        """Test a realistic UI usage scenario."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            # User starts application
            create_main_window()

            # User starts simulation
            start_simulation_callback()
            update_operation_status("Simulation running", 0.5)

            # Simulation runs and updates UI
            for step in range(20):
                # Simulate simulation step
                self.ui_state.add_live_feed_data('energy_history', 0.5 + 0.1 * (step % 10))
                self.ui_state.add_live_feed_data('node_activity_history', step * 2)

                # UI updates
                update_ui_display()
                update_graph_visualization()

                # Simulate some delay
                time.sleep(0.001)

            # User checks metrics
            data = self.ui_state.get_live_feed_data()
            self.assertEqual(len(data['energy_history']), 20)
            self.assertEqual(len(data['node_activity_history']), 20)

            # User stops simulation
            stop_simulation_callback()
            clear_operation_status()

            # User saves state
            self.ui_state.update_system_health({'status': 'completed', 'alerts': []})

            # User exits
            cleanup_ui_state()

    def test_ui_event_driven_updates(self):
        """Test event-driven UI updates."""
        from utils.event_bus import get_event_bus

        event_bus = get_event_bus()

        # Subscribe to events (as done in ui_engine)
        update_calls = []
        def mock_update_ui():
            update_calls.append('ui_updated')

        event_bus.subscribe('UI_REFRESH', lambda *args: mock_update_ui())

        # Trigger event
        event_bus.publish('UI_REFRESH', {})

        # Verify event handling
        self.assertEqual(len(update_calls), 1)

    def test_ui_configuration_workflow(self):
        """Test UI configuration workflow."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            # User adjusts parameters
            dpg.get_value.side_effect = lambda key: {
                'ltp_rate': 0.03,
                'ltd_rate': 0.02,
                'stdp_window': 25.0,
                'node_size': 3.0,
                'edge_thickness': 2.0
            }.get(key, 0.5)

            # Apply configuration
            from ui.ui_engine import apply_config_changes
            apply_config_changes()

            # Verify coordinator was updated
            self.mock_coordinator.update_configuration.assert_called_once()

    def test_ui_fallback_modes(self):
        """Test UI fallback modes when services unavailable."""
        # Test with no registry
        with patch('ui.ui_engine._service_registry', None):
            coordinator = get_coordinator()
            self.assertIsNone(coordinator)

            # UI should still function in limited mode
            update_ui_display()
            # Should set default values
            dpg.set_value.assert_any_call("status_text", "Status: Stopped")

        # Test with failing visualization service
        with patch('ui.ui_engine._visualization_service', self.mock_visualization):
            self.mock_visualization.update_visualization_data.side_effect = Exception("Viz failed")
            update_ui_display()
            # Should continue without crashing

    def test_ui_state_persistence_across_operations(self):
        """Test that UI state persists correctly across operations."""
        # Perform sequence of operations
        self.ui_state.set_simulation_running(True)
        self.ui_state.add_live_feed_data('test_persistence', 1.0)

        mock_graph = Mock()
        self.ui_state.update_graph(mock_graph)

        # Perform UI operations
        update_ui_display()
        update_graph_visualization()

        # Verify state maintained
        self.assertTrue(self.ui_state.get_simulation_running())
        data = self.ui_state.get_live_feed_data()
        self.assertEqual(len(data['test_persistence']), 1)
        self.assertEqual(self.ui_state.get_latest_graph(), mock_graph)

        # Reset and verify
        self.ui_state.set_simulation_running(False)
        self.assertFalse(self.ui_state.get_simulation_running())
        self.assertIsNone(self.ui_state.get_latest_graph())


if __name__ == "__main__":
    unittest.main()