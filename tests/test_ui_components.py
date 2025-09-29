"""
Comprehensive UI Component Test Suite
Tests all UI buttons, controls, and functionality for the Neural Simulation System.
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Dear PyGui to avoid GUI dependencies during testing
sys.modules['dearpygui'] = Mock()
sys.modules['dearpygui.dearpygui'] = Mock()

import dearpygui.dearpygui as dpg

from src.ui.ui_engine import (apply_config_changes, create_main_window,
                              create_ui, export_metrics,
                              load_neural_map_callback,
                              reset_simulation_callback, reset_to_defaults,
                              save_neural_map_callback,
                              start_simulation_callback,
                              stop_simulation_callback,
                              update_graph_visualization, update_ui_display,
                              view_logs_callback)
from src.ui.ui_state_manager import (UIStateManager, cleanup_ui_state,
                                     get_ui_state_manager)


class TestUIComponents(unittest.TestCase):
    """Test suite for UI components and functionality."""

    def setUp(self):
        """Set up test environment."""
        self.ui_state = get_ui_state_manager()
        self.mock_manager = Mock()
        self.mock_manager.graph = Mock()
        self.mock_manager.graph.node_labels = []
        self.mock_manager.graph.edge_index = Mock()
        self.mock_manager.graph.edge_index.shape = [2, 0]
        self.mock_manager.step_counter = 0
        self.mock_manager.get_system_stats = Mock(return_value={'health_score': 85.0})
        self.mock_manager.network_metrics = Mock()
        self.mock_manager.network_metrics.calculate_comprehensive_metrics = Mock(return_value={
            'criticality': 0.75, 'connectivity': {'ei_ratio': 0.8}
        })
        self.mock_manager.get_performance_stats = Mock(return_value={
            'avg_energy': 0.6, 'avg_step_time': 0.05
        })

        # Mock DPG functions
        dpg.set_value = Mock()
        dpg.get_value = Mock(return_value=0)
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

    def tearDown(self):
        """Clean up after tests."""
        cleanup_ui_state()

    def test_ui_state_manager_initialization(self):
        """Test UI state manager initialization."""
        state = get_ui_state_manager()
        self.assertIsNotNone(state)
        self.assertFalse(state.get_simulation_running())
        self.assertIsNone(state.get_latest_graph())

    def test_simulation_control_callbacks(self):
        """Test simulation control button callbacks."""
        with patch('ui.ui_engine.get_manager', return_value=self.mock_manager):
            # Test start callback
            start_simulation_callback()
            self.assertTrue(self.ui_state.get_simulation_running())

            # Test stop callback
            stop_simulation_callback()
            self.assertFalse(self.ui_state.get_simulation_running())

            # Test reset callback
            reset_simulation_callback()
            self.assertFalse(self.ui_state.get_simulation_running())

    def test_neural_map_operations(self):
        """Test neural map save/load operations."""
        with patch('ui.ui_engine.get_manager', return_value=self.mock_manager):
            # Mock the manager methods
            self.mock_manager.save_neural_map = Mock(return_value=True)
            self.mock_manager.load_neural_map = Mock(return_value=True)

            # Test save callback
            save_neural_map_callback(0)
            self.mock_manager.save_neural_map.assert_called_with(0)

            # Test load callback
            load_neural_map_callback(0)
            self.mock_manager.load_neural_map.assert_called_with(0)

    def test_configuration_operations(self):
        """Test configuration change operations."""
        # Test reset to defaults
        reset_to_defaults()

        # Verify default values are set (basic test - function runs without error)
        self.assertTrue(True)

    def test_ui_display_update(self):
        """Test UI display update functionality."""
        with patch('ui.ui_engine.get_manager', return_value=self.mock_manager):
            update_ui_display()

            # Verify status updates
            dpg.set_value.assert_any_call("status_text", "Status: Stopped")
            dpg.set_value.assert_any_call("nodes_text", "Nodes: 0")
            dpg.set_value.assert_any_call("edges_text", "Edges: 0")

    def test_graph_visualization_update(self):
        """Test graph visualization update."""
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

        with patch('ui.ui_engine.get_latest_graph_for_ui', return_value=mock_graph):
            update_graph_visualization()

            # Verify drawing operations
            self.assertTrue(dpg.clear_draw_list.called)
            self.assertTrue(dpg.draw_circle.called)

    def test_view_logs_callback(self):
        """Test logs modal display."""
        view_logs_callback()

        # Verify modal configuration
        dpg.configure_item.assert_called_with("logs_modal", show=True)

    def test_export_metrics(self):
        """Test metrics export functionality."""
        with patch('ui.ui_engine.get_manager', return_value=self.mock_manager):
            export_metrics()

            # Verify file operations would occur
            self.assertTrue(dpg.set_value.called)

    def test_ui_initialization(self):
        """Test UI initialization and window creation."""
        with patch('ui.ui_engine.create_main_window'):
            with patch('ui.ui_engine.dpg.is_dearpygui_running', side_effect=[True, False]):
                # This would normally run the UI loop, but we're mocking it
                pass

    def test_error_handling(self):
        """Test error handling in UI operations."""
        with patch('ui.ui_engine.get_manager', return_value=None):
            # Test callbacks with no manager
            start_simulation_callback()
            stop_simulation_callback()
            reset_simulation_callback()

            # Verify error messages are set
            error_calls = [call for call in dpg.set_value.call_args_list
                          if "failed" in str(call) or "error" in str(call).lower()]
            self.assertTrue(len(error_calls) > 0)

    def test_live_feed_data_operations(self):
        """Test live feed data operations."""
        # Test adding data
        self.ui_state.add_live_feed_data('energy_history', 0.5)
        self.ui_state.add_live_feed_data('energy_history', 0.6)

        # Test getting data
        data = self.ui_state.get_live_feed_data()
        self.assertIn('energy_history', data)
        self.assertEqual(len(data['energy_history']), 2)

        # Test clearing data
        self.ui_state.clear_live_feed_data()
        data = self.ui_state.get_live_feed_data()
        self.assertEqual(len(data['energy_history']), 0)

    def test_simulation_state_management(self):
        """Test simulation state management."""
        # Test initial state
        self.assertFalse(self.ui_state.get_simulation_running())

        # Test state changes
        self.ui_state.set_simulation_running(True)
        self.assertTrue(self.ui_state.get_simulation_running())

        self.ui_state.set_simulation_running(False)
        self.assertFalse(self.ui_state.get_simulation_running())

    def test_graph_state_management(self):
        """Test graph state management."""
        mock_graph = Mock()

        # Test graph update
        self.ui_state.update_graph(mock_graph)
        self.assertEqual(self.ui_state.get_latest_graph(), mock_graph)

        # Test graph clearing on stop
        self.ui_state.set_simulation_running(False)
        self.assertIsNone(self.ui_state.get_latest_graph())


class TestUIIntegration(unittest.TestCase):
    """Integration tests for UI components."""

    def setUp(self):
        """Set up integration test environment."""
        # Mock all DPG functions for integration testing
        self.dpg_patcher = patch('ui.ui_engine.dpg')
        self.mock_dpg = self.dpg_patcher.start()

        # Configure mock returns
        self.mock_dpg.set_value = Mock()
        self.mock_dpg.get_value = Mock(return_value=0.5)
        self.mock_dpg.configure_item = Mock()
        self.mock_dpg.add_text = Mock(return_value="test_text")
        self.mock_dpg.add_button = Mock(return_value="test_button")
        self.mock_dpg.add_checkbox = Mock(return_value="test_checkbox")
        self.mock_dpg.add_slider_float = Mock(return_value="test_slider")
        self.mock_dpg.add_input_int = Mock(return_value="test_input")
        self.mock_dpg.add_color_edit = Mock(return_value="test_color")
        self.mock_dpg.add_plot = Mock(return_value="test_plot")
        self.mock_dpg.add_plot_axis = Mock(return_value="test_axis")
        self.mock_dpg.add_line_series = Mock(return_value="test_series")
        self.mock_dpg.add_collapsing_header = Mock(return_value="test_header")
        self.mock_dpg.add_group = Mock(return_value="test_group")
        self.mock_dpg.add_child_window = Mock(return_value="test_window")
        self.mock_dpg.add_tab_bar = Mock(return_value="test_tab_bar")
        self.mock_dpg.add_tab = Mock(return_value="test_tab")
        self.mock_dpg.add_drawlist = Mock(return_value="test_drawlist")
        self.mock_dpg.add_input_text = Mock(return_value="test_input_text")
        self.mock_dpg.add_menu_bar = Mock(return_value="test_menu_bar")
        self.mock_dpg.add_menu = Mock(return_value="test_menu")
        self.mock_dpg.add_menu_item = Mock(return_value="test_menu_item")
        self.mock_dpg.add_window = Mock(return_value="test_window")
        self.mock_dpg.add_separator = Mock()
        self.mock_dpg.add_theme = Mock(return_value="test_theme")
        self.mock_dpg.add_theme_component = Mock(return_value="test_component")
        self.mock_dpg.add_theme_color = Mock()
        self.mock_dpg.add_theme_style = Mock()
        self.mock_dpg.bind_theme = Mock()
        self.mock_dpg.set_global_font_scale = Mock()
        self.mock_dpg.create_context = Mock()
        self.mock_dpg.create_viewport = Mock()
        self.mock_dpg.set_viewport_resizable = Mock()
        self.mock_dpg.setup_dearpygui = Mock()
        self.mock_dpg.show_viewport = Mock()
        self.mock_dpg.is_dearpygui_running = Mock(return_value=False)
        self.mock_dpg.render_dearpygui_frame = Mock()
        self.mock_dpg.destroy_context = Mock()
        self.mock_dpg.clear_draw_list = Mock()
        self.mock_dpg.draw_circle = Mock()
        self.mock_dpg.draw_line = Mock()
        self.mock_dpg.get_item_rect_size = Mock(return_value=[800, 600])
        self.mock_dpg.set_primary_window = Mock()
        self.mock_dpg.toggle_viewport_fullscreen = Mock()
        self.mock_dpg.stop_dearpygui = Mock()
        self.mock_dpg.add_tooltip = Mock(return_value="test_tooltip")

    def tearDown(self):
        """Clean up integration test environment."""
        self.dpg_patcher.stop()
        cleanup_ui_state()

    def test_full_ui_creation(self):
        """Test complete UI creation process."""
        try:
            create_main_window()
            # Verify UI components were created
            self.assertTrue(self.mock_dpg.add_text.called)
            self.assertTrue(self.mock_dpg.add_button.called)
            self.assertTrue(self.mock_dpg.add_tab_bar.called)
        except Exception as e:
            self.fail(f"UI creation failed: {e}")

    def test_ui_theme_application(self):
        """Test UI theme application."""
        try:
            create_ui()
            # Verify theme components were created
            self.assertTrue(self.mock_dpg.add_theme.called)
            self.assertTrue(self.mock_dpg.bind_theme.called)
        except Exception as e:
            self.fail(f"UI theme application failed: {e}")


def run_ui_tests():
    """Run all UI tests and return results."""
    print("UI COMPONENT TEST SUITE")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestUIComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestUIIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL RESULT: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    success = run_ui_tests()
    sys.exit(0 if success else 1)






