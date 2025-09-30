"""
Comprehensive unit tests for ui_engine.py
Tests all functions, edge cases, error handling, performance, and real-world usage.
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Dear PyGui
sys.modules['dearpygui'] = Mock()
sys.modules['dearpygui.dearpygui'] = Mock()

# Mock torch
sys.modules['torch'] = Mock()
sys.modules['torch.nn'] = Mock()

# Mock torch_geometric
sys.modules['torch_geometric'] = Mock()
sys.modules['torch_geometric.data'] = Mock()
sys.modules['torch_geometric.data'].Data = Mock()

import dearpygui.dearpygui as dpg

from src.ui.ui_engine import (add_live_feed_data, apply_config_changes,
                               auto_start_simulation, clear_live_feed_data,
                               clear_operation_status, create_fallback_ui,
                               create_main_window, create_ui, export_metrics,
                               force_close_application, get_coordinator,
                               get_latest_graph,
                               get_live_feed_data, get_simulation_running,
                               handle_keyboard_shortcut,
                               load_neural_map_callback,
                               reset_simulation_callback, reset_to_defaults,
                               run_ui, save_neural_map_callback,
                               set_simulation_running, show_keyboard_shortcuts,
                               start_simulation_callback,
                               stop_simulation_callback,
                               update_graph, update_graph_visualization,
                               update_operation_status, update_ui_display,
                               view_logs_callback)
from src.ui.ui_state_manager import cleanup_ui_state, get_ui_state_manager


class TestUIEngine(unittest.TestCase):
    """Unit tests for UI engine functions."""

    def setUp(self):
        """Set up test environment."""
        print("setUp start")
        self.ui_state = get_ui_state_manager()

        # Mock service registry and coordinator
        self.mock_registry = Mock()
        self.mock_coordinator = Mock()
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
        self.mock_coordinator.initialize_simulation = Mock(return_value=True)

        self.mock_registry.resolve = Mock(return_value=self.mock_coordinator)

        # Mock DPG functions
        self._setup_dpg_mocks()
        print("setUp end")

    def _setup_dpg_mocks(self):
        """Set up comprehensive DPG mocks."""
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

    def tearDown(self):
        """Clean up after tests."""
        print("tearDown start")
        cleanup_ui_state()
        print("tearDown end")

    def test_create_main_window(self):
        """Test main window creation."""
        create_main_window()
        # Verify UI components were created
        self.assertTrue(dpg.add_window.called)
        self.assertTrue(dpg.add_tab_bar.called)
        self.assertTrue(dpg.add_button.called)

    def test_simulation_callbacks(self):
        """Test simulation control callbacks."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            # Test start
            start_simulation_callback()
            self.mock_coordinator.start.assert_called_once()
            self.assertTrue(self.ui_state.get_simulation_running())

            # Test stop
            stop_simulation_callback()
            self.mock_coordinator.stop.assert_called_once()
            self.assertFalse(self.ui_state.get_simulation_running())

            # Test reset
            reset_simulation_callback()
            self.mock_coordinator.reset.assert_called_once()

    def test_simulation_callbacks_no_coordinator(self):
        """Test simulation callbacks when coordinator is unavailable."""
        with patch('ui.ui_engine._service_registry', None):
            start_simulation_callback()
            stop_simulation_callback()
            reset_simulation_callback()
            # Should not crash, should set error messages
            self.assertTrue(dpg.set_value.called)

    def test_update_ui_display_running_simulation(self):
        """Test UI display update with running simulation."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            self.ui_state.set_simulation_running(True)
            update_ui_display()

            # Verify status updates
            dpg.set_value.assert_any_call("status_text", "Status: Running")
            dpg.set_value.assert_any_call("nodes_text", "Nodes: 10")
            dpg.set_value.assert_any_call("edges_text", "Edges: 20")

    def test_update_ui_display_stopped_simulation(self):
        """Test UI display update with stopped simulation."""
        with patch('ui.ui_engine._service_registry', None):
            self.ui_state.set_simulation_running(False)
            update_ui_display()

            dpg.set_value.assert_any_call("status_text", "Status: Stopped")
            dpg.set_value.assert_any_call("nodes_text", "Nodes: 0")

    def test_update_graph_visualization_with_graph(self):
        """Test graph visualization with valid graph."""
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

            self.assertTrue(dpg.clear_draw_list.called)
            self.assertTrue(dpg.draw_circle.called)

    def test_update_graph_visualization_no_graph(self):
        """Test graph visualization with no graph."""
        with patch('ui.ui_engine.get_latest_graph_for_ui', return_value=None):
            update_graph_visualization()
            # Should not crash
            self.assertTrue(dpg.clear_draw_list.called)

    def test_apply_config_changes(self):
        """Test configuration changes application."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            apply_config_changes()
            self.mock_coordinator.update_configuration.assert_called_once()

    def test_reset_to_defaults(self):
        """Test reset to defaults."""
        reset_to_defaults()
        # Verify default values are set
        expected_calls = [
            unittest.mock.call("ltp_rate", 0.02),
            unittest.mock.call("ltd_rate", 0.01),
            unittest.mock.call("stdp_window", 20.0)
        ]
        for call in expected_calls:
            self.assertIn(call, dpg.set_value.call_args_list)

    def test_force_close_application(self):
        """Test force close application."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            force_close_application()
            self.mock_coordinator.stop.assert_called_once()
            self.assertTrue(dpg.stop_dearpygui.called)

    def test_show_keyboard_shortcuts(self):
        """Test keyboard shortcuts display."""
        show_keyboard_shortcuts()
        # Should create modal if it doesn't exist
        self.assertTrue(dpg.add_window.called or dpg.configure_item.called)

    def test_export_metrics(self):
        """Test metrics export."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            with patch('builtins.open', create=True) as mock_open:
                export_metrics()
                mock_open.assert_called_once()
                self.assertTrue(dpg.set_value.called)

    def test_view_logs_callback(self):
        """Test logs modal display."""
        view_logs_callback()
        dpg.configure_item.assert_called_with("logs_modal", show=True)

    def test_save_neural_map_callback(self):
        """Test neural map save."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            save_neural_map_callback(0)
            self.mock_coordinator.save_neural_map.assert_called_with(0)

    def test_load_neural_map_callback(self):
        """Test neural map load."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            load_neural_map_callback(0)
            self.mock_coordinator.load_neural_map.assert_called_with(0)

    def test_handle_keyboard_shortcut_start(self):
        """Test keyboard shortcut handling for start."""
        with patch('ui.ui_engine.start_simulation_callback') as mock_start:
            handle_keyboard_shortcut('start')
            mock_start.assert_called_once()

    def test_handle_keyboard_shortcut_unknown(self):
        """Test keyboard shortcut handling for unknown action."""
        handle_keyboard_shortcut('unknown')
        # Should not crash

    def test_update_operation_status(self):
        """Test operation status update."""
        update_operation_status("Test Operation", 0.5)
        dpg.set_value.assert_any_call("operation_status_text", "Operation: Test Operation")
        dpg.set_value.assert_any_call("operation_progress", 0.5)

    def test_clear_operation_status(self):
        """Test operation status clear."""
        clear_operation_status()
        dpg.set_value.assert_any_call("operation_status_text", "Operation: Idle")
        dpg.set_value.assert_any_call("operation_progress", 0.0)

    def test_get_coordinator_with_registry(self):
        """Test get_coordinator with service registry."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            coordinator = get_coordinator()
            self.assertEqual(coordinator, self.mock_coordinator)

    def test_get_coordinator_no_registry(self):
        """Test get_coordinator without service registry."""
        with patch('ui.ui_engine._service_registry', None):
            coordinator = get_coordinator()
            self.assertIsNone(coordinator)

    def test_auto_start_simulation_success(self):
        """Test successful auto-start simulation."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            with patch('time.sleep'):  # Speed up test
                auto_start_simulation()
                self.mock_coordinator.initialize_simulation.assert_called_once()
                self.mock_coordinator.start.assert_called_once()

    def test_auto_start_simulation_failure(self):
        """Test failed auto-start simulation."""
        self.mock_coordinator.initialize_simulation.return_value = False
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            auto_start_simulation()
            # Should handle failure gracefully
            self.assertTrue(dpg.set_value.called)

    def test_create_fallback_ui(self):
        """Test fallback UI creation."""
        with patch('time.sleep'):  # Speed up test
            create_fallback_ui()
            self.assertTrue(dpg.create_context.called)
            self.assertTrue(dpg.create_viewport.called)

    def test_create_ui_full(self):
        """Test full UI creation."""
        print("Starting test_create_ui_full")
        with patch('ui.ui_engine.dpg.is_dearpygui_running', side_effect=[True, False]):
            with patch('time.sleep'):  # Speed up test
                with patch('ui.ui_engine.auto_start_simulation'):
                    create_ui()
                    self.assertTrue(dpg.create_context.called)
                    self.assertTrue(dpg.create_viewport.called)
        print("Ending test_create_ui_full")

    def test_run_ui_with_registry(self):
        """Test run_ui with service registry."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            with patch('ui.ui_engine.create_ui'):
                run_ui(self.mock_registry)
                # Should not crash

    # Edge cases and error handling
    def test_update_ui_display_with_exception(self):
        """Test UI display update with exceptions."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            self.mock_coordinator.get_simulation_state.side_effect = Exception("Test error")
            update_ui_display()
            # Should handle exception gracefully

    def test_apply_config_changes_no_coordinator(self):
        """Test config changes without coordinator."""
        with patch('ui.ui_engine._service_registry', None):
            apply_config_changes()
            # Should not crash

    def test_export_metrics_no_coordinator(self):
        """Test metrics export without coordinator."""
        with patch('ui.ui_engine._service_registry', None):
            export_metrics()
            # Should not crash

    def test_save_load_neural_map_no_coordinator(self):
        """Test neural map operations without coordinator."""
        with patch('ui.ui_engine._service_registry', None):
            save_neural_map_callback(0)
            load_neural_map_callback(0)
            # Should not crash

    # Performance tests
    def test_update_graph_visualization_performance(self):
        """Test graph visualization performance with large graph."""
        print("Starting test_update_graph_visualization_performance")
        # Create large mock graph
        num_nodes = 1000
        mock_graph = Mock()
        mock_graph.node_labels = [
            {'id': i, 'energy': 0.5, 'state': 'active', 'pos': [i*10, i*10]}
            for i in range(num_nodes)
        ]
        mock_graph.edge_index = Mock()
        mock_graph.edge_index.shape = [2, num_nodes-1]
        mock_graph.edge_index.__getitem__ = Mock(return_value=Mock())
        mock_graph.edge_index.__getitem__.return_value.item.return_value = 0

        with patch('ui.ui_engine.get_latest_graph_for_ui', return_value=mock_graph):
            start_time = time.time()
            update_graph_visualization()
            end_time = time.time()
            duration = end_time - start_time
            # Should complete within reasonable time (adjust threshold as needed)
            self.assertLess(duration, 1.0, "Graph visualization too slow")
        print("Ending test_update_graph_visualization_performance")

    # Real-world usage scenarios
    def test_simulation_workflow(self):
        """Test complete simulation workflow."""
        with patch('ui.ui_engine._service_registry', self.mock_registry):
            # Start simulation
            start_simulation_callback()
            self.assertTrue(self.ui_state.get_simulation_running())

            # Update display
            update_ui_display()

            # Apply config changes
            apply_config_changes()

            # Stop simulation
            stop_simulation_callback()
            self.assertFalse(self.ui_state.get_simulation_running())

            # Reset
            reset_simulation_callback()

    def test_ui_state_functions(self):
        """Test UI state management functions."""
        # Test state functions
        set_simulation_running(True)
        self.assertTrue(get_simulation_running())

        # Test graph functions
        mock_graph = Mock()
        update_graph(mock_graph)
        self.assertEqual(get_latest_graph(), mock_graph)

        # Test live feed functions
        add_live_feed_data('test', 1.0)
        data = get_live_feed_data()
        self.assertIn('test', data)

        clear_live_feed_data()
        data = get_live_feed_data()
        self.assertEqual(len(data['test']), 0)

    def test_large_graph_handling(self):
        """Test handling of very large graphs."""
        # Create graph with many nodes
        mock_graph = Mock()
        mock_graph.node_labels = [
            {'id': i, 'energy': 0.5, 'state': 'active'}
            for i in range(10000)
        ]
        mock_graph.edge_index = Mock()
        mock_graph.edge_index.shape = [2, 0]  # No edges for simplicity

        with patch('ui.ui_engine.get_latest_graph_for_ui', return_value=mock_graph):
            update_graph_visualization()
            # Should handle large graphs without crashing

    def test_concurrent_ui_updates(self):
        """Test concurrent UI updates (simulated)."""
        print("Starting test_concurrent_ui_updates")
        import threading

        results = []

        def update_ui_thread():
            try:
                update_ui_display()
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        threads = []
        for _ in range(5):
            t = threading.Thread(target=update_ui_thread)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should succeed
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r == "success" for r in results))
        print("Ending test_concurrent_ui_updates")


if __name__ == "__main__":
    unittest.main()






