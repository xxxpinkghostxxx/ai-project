

"""
UI State Manager

This module provides a thread-safe state manager for UI components in the neural simulation system.
"""

import time
from typing import Dict, Any, Optional, List
from threading import RLock
from src.utils.logging_utils import log_step


class UIStateManager:
    """
    Thread-safe manager for UI state in the neural simulation.

    Handles simulation state, graph updates, live feed data, training interfaces,
    and system health monitoring with proper cleanup mechanisms.
    """

    def __init__(self):
        """Initialize the UIStateManager with default values."""
        self._lock = RLock()
        self._cleanup_callbacks = []
        self.simulation_running = False
        self.latest_graph = None
        self.latest_graph_for_ui = None
        self.update_for_ui = False
        self.sim_update_counter = 0
        self.last_update_time = 0
        self.sensory_texture_tag = "sensory_texture"
        self.sensory_image_tag = "sensory_image"
        self.graph_h = None
        self.graph_w = None
        self.live_feed_data = {
            "energy_history": [],
            "node_activity_history": [],
            "performance_history": [],
            "connection_history": [],
            "birth_rate_history": [],
            "time_history": [],
        }
        self._max_history_length = 1000
        self.neural_learning_active = False
        self.live_feed_config = {
            "max_history_length": 100
        }
        self.system_health = {
            "status": "unknown",
            "last_check": 0,
            "alerts": [],
            "energy_flow_rate": 0.0,
            "connection_activity": 0
        }
        self.live_training_interface = None
        self.training_active = False
        self._cleaned_up = False
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get the current simulation state."""
        with self._lock:
            return {
                'simulation_running': self.simulation_running,
                'sim_update_counter': self.sim_update_counter,
                'last_update_time': self.last_update_time,
                'update_for_ui': self.update_for_ui
            }
    def set_simulation_running(self, running: bool):
        """Set the simulation running state."""
        with self._lock:
            self.simulation_running = running
            if not running:
                self.latest_graph = None
                self.latest_graph_for_ui = None
                self.update_for_ui = False
    def update_graph(self, graph):
        """Update the latest graph and increment counter."""
        with self._lock:
            if self.latest_graph is not None:
                self._clear_graph_references(self.latest_graph)
            self.latest_graph = graph
            self.sim_update_counter += 1
            self.latest_graph_for_ui = graph
            if self.sim_update_counter % 10 == 0:
                self.update_for_ui = True
    def get_latest_graph(self):
        """Get the latest graph."""
        with self._lock:
            return self.latest_graph
    def get_latest_graph_for_ui(self):
        """Get the latest graph for UI display."""
        with self._lock:
            return self.latest_graph_for_ui
    def clear_ui_update_flag(self):
        """Clear the UI update flag."""
        with self._lock:
            self.update_for_ui = False
    def add_live_feed_data(self, data_type: str, value: float):
        """Add data to the live feed history."""
        with self._lock:
            if data_type in self.live_feed_data:
                history = self.live_feed_data[data_type]
                history.append(value)
                max_length = self.live_feed_config["max_history_length"]
                if len(history) > max_length:
                    try:
                        del history[:-max_length]
                    except (IndexError, ValueError):
                        self.live_feed_data[data_type] = history[-max_length:]
                if len(history) > max_length * 2:
                    log_step("Live feed data exceeded safe limits, forcing cleanup",
                            data_type=data_type, size=len(history))
                    self.live_feed_data[data_type] = history[-max_length//2:]
            else:
                self.live_feed_data[data_type] = [value]
    def get_live_feed_data(self) -> Dict[str, List[float]]:
        """Get a copy of the live feed data."""
        with self._lock:
            return {k: v.copy() for k, v in self.live_feed_data.items()}
    def clear_live_feed_data(self):
        """Clear all live feed data."""
        with self._lock:
            for key, value in self.live_feed_data.items():
                value.clear()
    def set_training_interface(self, interface):
        """Set the live training interface."""
        with self._lock:
            if self.live_training_interface is not None:
                self._cleanup_interface(self.live_training_interface)
            self.live_training_interface = interface  # pylint: disable=attribute-defined-outside-init
    def set_training_active(self, active: bool):
        """Set the training active state."""
        with self._lock:
            self.training_active = active  # pylint: disable=attribute-defined-outside-init
            if not active and self.live_training_interface is not None:
                self._cleanup_interface(self.live_training_interface)
                self.live_training_interface = None  # pylint: disable=attribute-defined-outside-init
    def update_system_health(self, health_data: Dict[str, Any]):
        """Update the system health data."""
        with self._lock:
            self.system_health.update(health_data)
            self.system_health["last_check"] = time.time()
    def get_system_health(self) -> Dict[str, Any]:
        """Get the current system health data."""
        with self._lock:
            return self.system_health.copy()
    def get_simulation_running(self) -> bool:
        """Get the simulation running state."""
        with self._lock:
            return self.simulation_running
    def _clear_graph_references(self, graph):
        """Clear references in the graph to free memory."""
        try:
            if hasattr(graph, 'node_labels'):
                for node in graph.node_labels:
                    if isinstance(node, dict):
                        for key, value in node.items():
                            if hasattr(value, 'cpu'):
                                del value
                            elif isinstance(value, (list, tuple)) and len(value) > 100:
                                node[key] = []
        except Exception as e:  # pylint: disable=broad-exception-caught
            log_step("Graph reference clearing error", error=str(e))
    def _cleanup_interface(self, interface):
        """Clean up the training interface."""
        try:
            if hasattr(interface, 'cleanup'):
                interface.cleanup()
            elif hasattr(interface, 'close'):
                interface.close()
        except Exception as e:  # pylint: disable=broad-exception-caught
            log_step("Interface cleanup error", error=str(e))
    def add_cleanup_callback(self, callback):
        """Add a callback to be called during cleanup."""
        with self._lock:
            self._cleanup_callbacks.append(callback)
    def cleanup(self):
        """Clean up the UIStateManager and release resources."""
        if self._cleaned_up:
            return
        with self._lock:
            if hasattr(self, 'latest_graph') and self.latest_graph is not None:
                try:
                    self._clear_graph_references(self.latest_graph)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    log_step("Graph cleanup error", error=str(e))
                finally:
                    self.latest_graph = None
            if hasattr(self, 'latest_graph_for_ui') and self.latest_graph_for_ui is not None:
                try:
                    self._clear_graph_references(self.latest_graph_for_ui)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    log_step("UI graph cleanup error", error=str(e))
                finally:
                    self.latest_graph_for_ui = None
            try:
                if hasattr(self, 'clear_live_feed_data'):
                    self.clear_live_feed_data()
            except Exception as e:  # pylint: disable=broad-exception-caught
                log_step("Live feed cleanup error", error=str(e))
            self.neural_learning_active = False
            callbacks_copy = list(self._cleanup_callbacks)
            for callback in callbacks_copy:
                try:
                    if callable(callback):
                        callback()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    log_step("Cleanup callback error", error=str(e))
            self._cleanup_callbacks.clear()
            self.simulation_running = False
            self.update_for_ui = False
            self.sim_update_counter = 0
            self.neural_learning_active = False
            log_step("UIStateManager cleaned up")
        self._cleaned_up = True  # pylint: disable=attribute-defined-outside-init
_ui_state_manager: Optional[UIStateManager] = None
_state_manager_lock = RLock()


def get_ui_state_manager() -> UIStateManager:
    """Get the singleton UIStateManager instance."""
    global _ui_state_manager  # pylint: disable=global-statement
    if _ui_state_manager is None:
        with _state_manager_lock:
            if _ui_state_manager is None:
                _ui_state_manager = UIStateManager()
    return _ui_state_manager


def cleanup_ui_state():
    """Clean up the global UIStateManager instance."""
    global _ui_state_manager  # pylint: disable=global-statement
    if _ui_state_manager is not None:
        _ui_state_manager.cleanup()
        _ui_state_manager = None


def get_simulation_running():
    """Get the simulation running state."""
    return get_ui_state_manager().get_simulation_state()['simulation_running']


def set_simulation_running(running: bool):
    """Set the simulation running state."""
    get_ui_state_manager().set_simulation_running(running)


def get_latest_graph():
    """Get the latest graph."""
    return get_ui_state_manager().get_latest_graph()


def get_latest_graph_for_ui():
    """Get the latest graph for UI."""
    return get_ui_state_manager().get_latest_graph_for_ui()


def update_graph(graph):
    """Update the graph."""
    get_ui_state_manager().update_graph(graph)


def add_live_feed_data(data_type: str, value: float):
    """Add live feed data."""
    get_ui_state_manager().add_live_feed_data(data_type, value)


def get_live_feed_data():
    """Get live feed data."""
    return get_ui_state_manager().get_live_feed_data()
def clear_live_feed_data():
    """Clear live feed data."""
    get_ui_state_manager().clear_live_feed_data()






