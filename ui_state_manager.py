
import threading
import time
from typing import Dict, Any, Optional, List
from threading import Lock
from logging_utils import log_step


class UIStateManager:

    def __init__(self):
        self._lock = Lock()
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
        log_step("UIStateManager initialized")
    def get_simulation_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'simulation_running': self.simulation_running,
                'sim_update_counter': self.sim_update_counter,
                'last_update_time': self.last_update_time,
                'update_for_ui': self.update_for_ui
            }
    def set_simulation_running(self, running: bool):
        with self._lock:
            self.simulation_running = running
            if not running:
                self.latest_graph = None
                self.latest_graph_for_ui = None
                self.update_for_ui = False
    def update_graph(self, graph):
        with self._lock:
            if self.latest_graph is not None:
                self._clear_graph_references(self.latest_graph)
            self.latest_graph = graph
            self.sim_update_counter += 1
            if self.sim_update_counter % 10 == 0:
                if self.latest_graph_for_ui is not None:
                    self._clear_graph_references(self.latest_graph_for_ui)
                self.latest_graph_for_ui = graph
                self.update_for_ui = True
    def get_latest_graph(self):
        with self._lock:
            return self.latest_graph
    def get_latest_graph_for_ui(self):
        with self._lock:
            return self.latest_graph_for_ui
    def clear_ui_update_flag(self):
        with self._lock:
            self.update_for_ui = False
    def add_live_feed_data(self, data_type: str, value: float):
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
        with self._lock:
            return {k: v.copy() for k, v in self.live_feed_data.items()}
    def clear_live_feed_data(self):
        with self._lock:
            for key in self.live_feed_data:
                self.live_feed_data[key].clear()
    def set_training_interface(self, interface):
        with self._lock:
            if self.live_training_interface is not None:
                self._cleanup_interface(self.live_training_interface)
            self.live_training_interface = interface
    def set_training_active(self, active: bool):
        with self._lock:
            self.training_active = active
            if not active and self.live_training_interface is not None:
                self._cleanup_interface(self.live_training_interface)
                self.live_training_interface = None
    def update_system_health(self, health_data: Dict[str, Any]):
        with self._lock:
            self.system_health.update(health_data)
            self.system_health["last_check"] = time.time()
    def get_system_health(self) -> Dict[str, Any]:
        with self._lock:
            return self.system_health.copy()
    def get_simulation_running(self) -> bool:
        with self._lock:
            return self.simulation_running
    def _clear_graph_references(self, graph):
        try:
            if hasattr(graph, 'node_labels'):
                for node in graph.node_labels:
                    if isinstance(node, dict):
                        for key, value in node.items():
                            if hasattr(value, 'cpu'):
                                del value
                            elif isinstance(value, (list, tuple)) and len(value) > 100:
                                node[key] = []
        except Exception as e:
            log_step("Graph reference clearing error", error=str(e))
    def _cleanup_interface(self, interface):
        try:
            if hasattr(interface, 'cleanup'):
                interface.cleanup()
            elif hasattr(interface, 'close'):
                interface.close()
        except Exception as e:
            log_step("Interface cleanup error", error=str(e))
    def add_cleanup_callback(self, callback):
        with self._lock:
            self._cleanup_callbacks.append(callback)
    def cleanup(self):
        if hasattr(self, '_cleaned_up') and self._cleaned_up:
            return
        with self._lock:
            if hasattr(self, 'latest_graph') and self.latest_graph is not None:
                try:
                    self._clear_graph_references(self.latest_graph)
                except Exception as e:
                    log_step("Graph cleanup error", error=str(e))
                finally:
                    self.latest_graph = None
            if hasattr(self, 'latest_graph_for_ui') and self.latest_graph_for_ui is not None:
                try:
                    self._clear_graph_references(self.latest_graph_for_ui)
                except Exception as e:
                    log_step("UI graph cleanup error", error=str(e))
                finally:
                    self.latest_graph_for_ui = None
            try:
                if hasattr(self, 'clear_live_feed_data'):
                    self.clear_live_feed_data()
            except Exception as e:
                log_step("Live feed cleanup error", error=str(e))
            self.neural_learning_active = False
            callbacks_copy = list(self._cleanup_callbacks)
            for callback in callbacks_copy:
                try:
                    if callable(callback):
                        callback()
                except Exception as e:
                    log_step("Cleanup callback error", error=str(e))
            self._cleanup_callbacks.clear()
            self.simulation_running = False
            self.update_for_ui = False
            self.sim_update_counter = 0
            self.neural_learning_active = False
            log_step("UIStateManager cleaned up")
        self._cleaned_up = True
_ui_state_manager: Optional[UIStateManager] = None
_state_manager_lock = Lock()


def get_ui_state_manager() -> UIStateManager:
    global _ui_state_manager
    if _ui_state_manager is None:
        with _state_manager_lock:
            if _ui_state_manager is None:
                _ui_state_manager = UIStateManager()
    return _ui_state_manager


def cleanup_ui_state():
    global _ui_state_manager
    if _ui_state_manager is not None:
        _ui_state_manager.cleanup()
        _ui_state_manager = None


def get_simulation_running():
    return get_ui_state_manager().get_simulation_state()['simulation_running']


def set_simulation_running(running: bool):
    get_ui_state_manager().set_simulation_running(running)


def get_latest_graph():
    return get_ui_state_manager().get_latest_graph()


def get_latest_graph_for_ui():
    return get_ui_state_manager().get_latest_graph_for_ui()


def update_graph(graph):
    get_ui_state_manager().update_graph(graph)


def add_live_feed_data(data_type: str, value: float):
    get_ui_state_manager().add_live_feed_data(data_type, value)


def get_live_feed_data():
    return get_ui_state_manager().get_live_feed_data()


def clear_live_feed_data():
    get_ui_state_manager().clear_live_feed_data()
