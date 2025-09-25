"""
Comprehensive unit tests for ui_state_manager.py
Tests all UIStateManager methods, thread safety, edge cases, error handling, performance, and real-world usage.
"""

import sys
import os
import time
import threading
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.ui_state_manager import (
    UIStateManager, get_ui_state_manager, cleanup_ui_state,
    get_simulation_running, set_simulation_running, get_latest_graph,
    get_latest_graph_for_ui, update_graph, add_live_feed_data,
    get_live_feed_data, clear_live_feed_data
)


class TestUIStateManager(unittest.TestCase):
    """Unit tests for UIStateManager class and functions."""

    def setUp(self):
        """Set up test environment."""
        self.state_manager = UIStateManager()

    def tearDown(self):
        """Clean up after tests."""
        cleanup_ui_state()

    def test_initialization(self):
        """Test UIStateManager initialization."""
        self.assertFalse(self.state_manager.simulation_running)
        self.assertIsNone(self.state_manager.latest_graph)
        self.assertIsNone(self.state_manager.latest_graph_for_ui)
        self.assertEqual(len(self.state_manager.live_feed_data), 6)  # energy_history, etc.
        self.assertEqual(self.state_manager._max_history_length, 1000)
        self.assertFalse(self.state_manager.neural_learning_active)

    def test_get_simulation_state(self):
        """Test get_simulation_state method."""
        state = self.state_manager.get_simulation_state()
        expected_keys = ['simulation_running', 'sim_update_counter', 'last_update_time', 'update_for_ui']
        for key in expected_keys:
            self.assertIn(key, state)

    def test_set_simulation_running(self):
        """Test set_simulation_running method."""
        # Test setting to True
        self.state_manager.set_simulation_running(True)
        self.assertTrue(self.state_manager.simulation_running)

        # Test setting to False - should clear graphs
        mock_graph = Mock()
        self.state_manager.latest_graph = mock_graph
        self.state_manager.latest_graph_for_ui = mock_graph
        self.state_manager.set_simulation_running(False)
        self.assertFalse(self.state_manager.simulation_running)
        self.assertIsNone(self.state_manager.latest_graph)
        self.assertIsNone(self.state_manager.latest_graph_for_ui)

    def test_update_graph(self):
        """Test update_graph method."""
        mock_graph = Mock()
        self.state_manager.update_graph(mock_graph)

        self.assertEqual(self.state_manager.latest_graph, mock_graph)
        self.assertEqual(self.state_manager.sim_update_counter, 1)

        # Test UI update every 10 updates
        for i in range(9):
            self.state_manager.update_graph(mock_graph)
        self.assertEqual(self.state_manager.sim_update_counter, 10)
        self.assertEqual(self.state_manager.latest_graph_for_ui, mock_graph)
        self.assertTrue(self.state_manager.update_for_ui)

    def test_get_latest_graph(self):
        """Test get_latest_graph method."""
        mock_graph = Mock()
        self.state_manager.latest_graph = mock_graph
        self.assertEqual(self.state_manager.get_latest_graph(), mock_graph)

    def test_get_latest_graph_for_ui(self):
        """Test get_latest_graph_for_ui method."""
        mock_graph = Mock()
        self.state_manager.latest_graph_for_ui = mock_graph
        self.assertEqual(self.state_manager.get_latest_graph_for_ui(), mock_graph)

    def test_clear_ui_update_flag(self):
        """Test clear_ui_update_flag method."""
        self.state_manager.update_for_ui = True
        self.state_manager.clear_ui_update_flag()
        self.assertFalse(self.state_manager.update_for_ui)

    def test_add_live_feed_data(self):
        """Test add_live_feed_data method."""
        # Test normal addition
        self.state_manager.add_live_feed_data('energy_history', 1.0)
        self.assertEqual(len(self.state_manager.live_feed_data['energy_history']), 1)

        # Test max length enforcement
        for i in range(105):
            self.state_manager.add_live_feed_data('energy_history', float(i))
        # Should be trimmed to max_length (100)
        self.assertEqual(len(self.state_manager.live_feed_data['energy_history']), 100)

    def test_add_live_feed_data_invalid_type(self):
        """Test add_live_feed_data with invalid data type."""
        self.state_manager.add_live_feed_data('invalid_type', 1.0)
        self.assertIn('invalid_type', self.state_manager.live_feed_data)
        self.assertEqual(len(self.state_manager.live_feed_data['invalid_type']), 1)

    def test_get_live_feed_data(self):
        """Test get_live_feed_data method."""
        self.state_manager.add_live_feed_data('energy_history', 1.0)
        data = self.state_manager.get_live_feed_data()
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data['energy_history']), 1)
        # Verify it's a copy
        data['energy_history'].append(2.0)
        self.assertEqual(len(self.state_manager.live_feed_data['energy_history']), 1)

    def test_clear_live_feed_data(self):
        """Test clear_live_feed_data method."""
        self.state_manager.add_live_feed_data('energy_history', 1.0)
        self.state_manager.clear_live_feed_data()
        self.assertEqual(len(self.state_manager.live_feed_data['energy_history']), 0)

    def test_update_system_health(self):
        """Test update_system_health method."""
        health_data = {'status': 'good', 'alerts': ['test']}
        self.state_manager.update_system_health(health_data)
        health = self.state_manager.get_system_health()
        self.assertEqual(health['status'], 'good')
        self.assertEqual(health['alerts'], ['test'])
        self.assertIsInstance(health['last_check'], float)

    def test_get_system_health(self):
        """Test get_system_health method."""
        health = self.state_manager.get_system_health()
        self.assertIsInstance(health, dict)
        self.assertIn('status', health)
        # Verify it's a copy
        health['status'] = 'modified'
        self.assertNotEqual(self.state_manager.system_health['status'], 'modified')

    def test_get_simulation_running(self):
        """Test get_simulation_running method."""
        self.state_manager.simulation_running = True
        self.assertTrue(self.state_manager.get_simulation_running())

    def test_clear_graph_references(self):
        """Test _clear_graph_references method."""
        mock_graph = Mock()
        mock_node = {'test': [1, 2, 3] * 100}  # Large list to test clearing
        mock_graph.node_labels = [mock_node]
        self.state_manager._clear_graph_references(mock_graph)
        # Should not crash

    def test_clear_graph_references_with_tensors(self):
        """Test _clear_graph_references with tensor-like objects."""
        mock_graph = Mock()
        mock_tensor = Mock()
        mock_tensor.cpu = Mock(return_value=Mock())
        mock_node = {'tensor_attr': mock_tensor}
        mock_graph.node_labels = [mock_node]
        self.state_manager._clear_graph_references(mock_graph)
        # Should call cpu() and delete

    def test_cleanup_interface(self):
        """Test _cleanup_interface method."""
        mock_interface = Mock()
        self.state_manager._cleanup_interface(mock_interface)
        mock_interface.cleanup.assert_called_once()

        # Test close method
        mock_interface.reset_mock()
        mock_interface.close = Mock()
        del mock_interface.cleanup
        self.state_manager._cleanup_interface(mock_interface)
        mock_interface.close.assert_called_once()

    def test_add_cleanup_callback(self):
        """Test add_cleanup_callback method."""
        callback = Mock()
        self.state_manager.add_cleanup_callback(callback)
        self.assertIn(callback, self.state_manager._cleanup_callbacks)

    def test_cleanup(self):
        """Test cleanup method."""
        # Set up some state
        mock_graph = Mock()
        self.state_manager.latest_graph = mock_graph
        self.state_manager.latest_graph_for_ui = mock_graph
        self.state_manager.add_live_feed_data('energy_history', 1.0)
        callback = Mock()
        self.state_manager.add_cleanup_callback(callback)

        self.state_manager.cleanup()

        # Verify cleanup
        callback.assert_called_once()
        self.assertIsNone(self.state_manager.latest_graph)
        self.assertIsNone(self.state_manager.latest_graph_for_ui)
        self.assertEqual(len(self.state_manager.live_feed_data['energy_history']), 0)
        self.assertFalse(self.state_manager.simulation_running)

    def test_cleanup_idempotent(self):
        """Test that cleanup is idempotent."""
        self.state_manager.cleanup()
        self.state_manager._cleaned_up = False  # Reset for test
        self.state_manager.cleanup()  # Should not crash

    def test_get_ui_state_manager_singleton(self):
        """Test get_ui_state_manager singleton behavior."""
        manager1 = get_ui_state_manager()
        manager2 = get_ui_state_manager()
        self.assertIs(manager1, manager2)

    def test_cleanup_ui_state(self):
        """Test cleanup_ui_state function."""
        manager = get_ui_state_manager()
        self.assertIsNotNone(manager)
        cleanup_ui_state()
        # Next call should create new instance
        manager2 = get_ui_state_manager()
        self.assertIsNotNone(manager2)

    def test_module_level_functions(self):
        """Test module-level convenience functions."""
        # Test simulation running functions
        set_simulation_running(True)
        self.assertTrue(get_simulation_running())

        # Test graph functions
        mock_graph = Mock()
        update_graph(mock_graph)
        self.assertEqual(get_latest_graph(), mock_graph)
        self.assertEqual(get_latest_graph_for_ui(), mock_graph)

        # Test live feed functions
        add_live_feed_data('test', 1.0)
        data = get_live_feed_data()
        self.assertIn('test', data)
        clear_live_feed_data()
        data = get_live_feed_data()
        self.assertEqual(len(data['test']), 0)

    # Edge cases and error handling

    def test_clear_graph_references_exception(self):
        """Test _clear_graph_references with exception."""
        mock_graph = Mock()
        mock_graph.node_labels = None  # Will cause AttributeError
        self.state_manager._clear_graph_references(mock_graph)
        # Should not crash

    def test_cleanup_interface_exception(self):
        """Test _cleanup_interface with exception."""
        mock_interface = Mock()
        mock_interface.cleanup.side_effect = Exception("Test error")
        self.state_manager._cleanup_interface(mock_interface)
        # Should not crash

    def test_cleanup_callback_exception(self):
        """Test cleanup callback with exception."""
        callback = Mock(side_effect=Exception("Test error"))
        self.state_manager.add_cleanup_callback(callback)
        self.state_manager.cleanup()
        # Should not crash, continue with other callbacks

    # Performance tests
    def test_large_live_feed_data_performance(self):
        """Test performance with large live feed data."""
        start_time = time.time()
        for i in range(1000):
            self.state_manager.add_live_feed_data('energy_history', float(i))
        end_time = time.time()
        duration = end_time - start_time
        self.assertLess(duration, 1.0, "Live feed data addition too slow")
        self.assertEqual(len(self.state_manager.live_feed_data['energy_history']), 100)

    def test_concurrent_access_thread_safety(self):
        """Test thread safety of state manager operations."""
        results = []
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    self.state_manager.add_live_feed_data('energy_history', float(thread_id * 100 + i))
                    state = self.state_manager.get_simulation_state()
                    self.state_manager.set_simulation_running(thread_id % 2 == 0)
                results.append(thread_id)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(results), 5, f"Thread errors: {errors}")
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

    # Real-world usage scenarios
    def test_simulation_lifecycle(self):
        """Test complete simulation lifecycle state management."""
        # Initial state
        self.assertFalse(self.state_manager.get_simulation_running())

        # Start simulation
        self.state_manager.set_simulation_running(True)
        self.assertTrue(self.state_manager.get_simulation_running())

        # Add some live data during simulation
        for i in range(50):
            self.state_manager.add_live_feed_data('energy_history', float(i) * 0.1)
            self.state_manager.add_live_feed_data('node_activity_history', float(i % 10))

        # Update graph multiple times
        for i in range(15):
            mock_graph = Mock()
            mock_graph.num_nodes = i * 10
            self.state_manager.update_graph(mock_graph)

        # Check that UI graph was updated
        self.assertIsNotNone(self.state_manager.get_latest_graph_for_ui())

        # Stop simulation
        self.state_manager.set_simulation_running(False)
        self.assertFalse(self.state_manager.get_simulation_running())
        self.assertIsNone(self.state_manager.get_latest_graph())

        # Check live data persists after stop
        data = self.state_manager.get_live_feed_data()
        self.assertEqual(len(data['energy_history']), 50)

    def test_system_health_monitoring(self):
        """Test system health monitoring scenario."""
        # Simulate health updates over time
        health_updates = [
            {'status': 'initializing', 'energy_flow_rate': 0.0},
            {'status': 'running', 'energy_flow_rate': 1.5, 'alerts': []},
            {'status': 'running', 'energy_flow_rate': 2.1, 'alerts': ['High energy flow']},
            {'status': 'error', 'energy_flow_rate': 0.0, 'alerts': ['System overload']}
        ]

        for update in health_updates:
            self.state_manager.update_system_health(update)
            time.sleep(0.01)  # Small delay to ensure different timestamps

        health = self.state_manager.get_system_health()
        self.assertEqual(health['status'], 'error')
        self.assertEqual(len(health['alerts']), 1)
        self.assertGreater(health['last_check'], 0)

    def test_memory_management_large_graphs(self):
        """Test memory management with large graphs."""
        # Create large graph with many nodes
        large_graph = Mock()
        large_graph.node_labels = [
            {'id': i, 'energy': 0.5, 'state': 'active', 'large_data': [0] * 1000}
            for i in range(1000)
        ]

        # Update graph multiple times
        for i in range(20):
            self.state_manager.update_graph(large_graph)

        # Force cleanup
        self.state_manager.cleanup()

        # Verify cleanup
        self.assertIsNone(self.state_manager.latest_graph)
        self.assertIsNone(self.state_manager.latest_graph_for_ui)

    def test_training_interface_management(self):
        """Test training interface management."""
        mock_interface = Mock()
        self.state_manager.live_training_interface = mock_interface
        self.state_manager.training_active = True

        # Set new interface
        new_interface = Mock()
        self.state_manager.set_training_interface(new_interface)
        self.assertEqual(self.state_manager.live_training_interface, new_interface)

        # Deactivate training
        self.state_manager.set_training_active(False)
        self.assertFalse(self.state_manager.training_active)
        self.assertIsNone(self.state_manager.live_training_interface)

    def test_realistic_data_patterns(self):
        """Test with realistic data patterns."""
        # Simulate realistic energy history
        import math
        for i in range(200):
            energy = 0.5 + 0.3 * math.sin(i * 0.1) + 0.1 * (i / 200.0)
            self.state_manager.add_live_feed_data('energy_history', energy)

            activity = abs(math.sin(i * 0.05)) * 100
            self.state_manager.add_live_feed_data('node_activity_history', activity)

        data = self.state_manager.get_live_feed_data()
        self.assertEqual(len(data['energy_history']), 100)  # Should be trimmed
        self.assertEqual(len(data['node_activity_history']), 100)

        # Verify data ranges are reasonable
        energies = data['energy_history']
        self.assertTrue(all(0 <= e <= 1 for e in energies))


if __name__ == "__main__":
    unittest.main()