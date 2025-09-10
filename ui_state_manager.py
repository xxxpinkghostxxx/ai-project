"""
ui_state_manager.py

UI state management system to prevent memory leaks from global variables.
Encapsulates UI state and provides proper cleanup mechanisms.
"""

import threading
import time
from typing import Dict, Any, Optional, List
from threading import Lock
from logging_utils import log_step
from memory_leak_detector import MemoryLeakPrevention


class UIStateManager:
    """
    Manages UI state to prevent memory leaks from global variables.
    Provides proper cleanup and reference management.
    """
    
    def __init__(self):
        """Initialize the UI state manager."""
        self._lock = Lock()
        self._cleanup_callbacks = []
        
        # Simulation state
        self.simulation_running = False
        self.latest_graph = None
        self.latest_graph_for_ui = None
        self.update_for_ui = False
        self.sim_update_counter = 0
        self.last_update_time = 0
        
        # UI configuration
        self.sensory_texture_tag = "sensory_texture"
        self.sensory_image_tag = "sensory_image"
        self.graph_h = None
        self.graph_w = None
        
        # Live feed data with size limits
        self.live_feed_data = {
            "energy_history": [],
            "node_activity_history": [],
            "performance_history": [],
            "connection_history": [],
            "birth_rate_history": [],
            "time_history": [],
        }
        
        # Live training system
        self.live_training_interface = None
        self.training_active = False
        
        # Live feed configuration
        self.live_feed_config = {
            "max_history_length": 100
        }
        
        # System health tracking
        self.system_health = {
            "status": "unknown",
            "last_check": 0,
            "alerts": [],
            "energy_flow_rate": 0.0,
            "connection_activity": 0
        }
        
        log_step("UIStateManager initialized")
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        with self._lock:
            return {
                'simulation_running': self.simulation_running,
                'sim_update_counter': self.sim_update_counter,
                'last_update_time': self.last_update_time,
                'update_for_ui': self.update_for_ui
            }
    
    def set_simulation_running(self, running: bool):
        """Set simulation running state."""
        with self._lock:
            self.simulation_running = running
            if not running:
                # Clear graph references when stopping
                self.latest_graph = None
                self.latest_graph_for_ui = None
                self.update_for_ui = False
    
    def update_graph(self, graph):
        """Update the latest graph with proper reference management."""
        with self._lock:
            # Clear old references to prevent memory leaks
            if self.latest_graph is not None:
                self._clear_graph_references(self.latest_graph)
            
            self.latest_graph = graph
            self.sim_update_counter += 1
            
            # Update UI graph every N steps to reduce memory pressure
            if self.sim_update_counter % 10 == 0:  # Every 10 steps
                if self.latest_graph_for_ui is not None:
                    self._clear_graph_references(self.latest_graph_for_ui)
                self.latest_graph_for_ui = graph
                self.update_for_ui = True
    
    def get_latest_graph(self):
        """Get the latest graph."""
        with self._lock:
            return self.latest_graph
    
    def get_latest_graph_for_ui(self):
        """Get the latest graph for UI updates."""
        with self._lock:
            return self.latest_graph_for_ui
    
    def clear_ui_update_flag(self):
        """Clear the UI update flag."""
        with self._lock:
            self.update_for_ui = False
    
    def add_live_feed_data(self, data_type: str, value: float):
        """Add data to live feed with size limits."""
        with self._lock:
            if data_type in self.live_feed_data:
                history = self.live_feed_data[data_type]
                history.append(value)
                
                # Limit history size to prevent memory leaks
                max_length = self.live_feed_config["max_history_length"]
                if len(history) > max_length:
                    self.live_feed_data[data_type] = history[-max_length:]
    
    def get_live_feed_data(self) -> Dict[str, List[float]]:
        """Get live feed data."""
        with self._lock:
            return {k: v.copy() for k, v in self.live_feed_data.items()}
    
    def clear_live_feed_data(self):
        """Clear live feed data to free memory."""
        with self._lock:
            for key in self.live_feed_data:
                self.live_feed_data[key].clear()
    
    def set_training_interface(self, interface):
        """Set the live training interface."""
        with self._lock:
            # Clear old interface to prevent memory leaks
            if self.live_training_interface is not None:
                self._cleanup_interface(self.live_training_interface)
            
            self.live_training_interface = interface
    
    def set_training_active(self, active: bool):
        """Set training active state."""
        with self._lock:
            self.training_active = active
            if not active and self.live_training_interface is not None:
                self._cleanup_interface(self.live_training_interface)
                self.live_training_interface = None
    
    def update_system_health(self, health_data: Dict[str, Any]):
        """Update system health data."""
        with self._lock:
            self.system_health.update(health_data)
            self.system_health["last_check"] = time.time()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health data."""
        with self._lock:
            return self.system_health.copy()
    
    def _clear_graph_references(self, graph):
        """Clear references in a graph to help with garbage collection."""
        try:
            if hasattr(graph, 'node_labels'):
                # Clear large data structures
                for node in graph.node_labels:
                    if isinstance(node, dict):
                        # Clear large arrays and tensors
                        for key, value in node.items():
                            if hasattr(value, 'cpu'):  # PyTorch tensor
                                del value
                            elif isinstance(value, (list, tuple)) and len(value) > 100:
                                node[key] = []
        except Exception as e:
            log_step("Graph reference clearing error", error=str(e))
    
    def _cleanup_interface(self, interface):
        """Clean up training interface resources."""
        try:
            if hasattr(interface, 'cleanup'):
                interface.cleanup()
            elif hasattr(interface, 'close'):
                interface.close()
        except Exception as e:
            log_step("Interface cleanup error", error=str(e))
    
    def add_cleanup_callback(self, callback):
        """Add a cleanup callback."""
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def cleanup(self):
        """Clean up all resources and clear references."""
        with self._lock:
            # Clear graph references
            if self.latest_graph is not None:
                self._clear_graph_references(self.latest_graph)
                self.latest_graph = None
            
            if self.latest_graph_for_ui is not None:
                self._clear_graph_references(self.latest_graph_for_ui)
                self.latest_graph_for_ui = None
            
            # Clear live feed data
            self.clear_live_feed_data()
            
            # Clean up training interface
            if self.live_training_interface is not None:
                self._cleanup_interface(self.live_training_interface)
                self.live_training_interface = None
            
            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    log_step("Cleanup callback error", error=str(e))
            
            # Clear callbacks
            self._cleanup_callbacks.clear()
            
            # Reset state
            self.simulation_running = False
            self.update_for_ui = False
            self.sim_update_counter = 0
            self.training_active = False
            
            log_step("UIStateManager cleaned up")


# Global UI state manager instance
_ui_state_manager: Optional[UIStateManager] = None
_state_manager_lock = Lock()


def get_ui_state_manager() -> UIStateManager:
    """Get the global UI state manager instance."""
    global _ui_state_manager
    if _ui_state_manager is None:
        with _state_manager_lock:
            if _ui_state_manager is None:
                _ui_state_manager = UIStateManager()
    return _ui_state_manager


def cleanup_ui_state():
    """Clean up UI state to prevent memory leaks."""
    global _ui_state_manager
    if _ui_state_manager is not None:
        _ui_state_manager.cleanup()
        _ui_state_manager = None


# Convenience functions for backward compatibility
def get_simulation_running():
    """Get simulation running state."""
    return get_ui_state_manager().get_simulation_state()['simulation_running']


def set_simulation_running(running: bool):
    """Set simulation running state."""
    get_ui_state_manager().set_simulation_running(running)


def get_latest_graph():
    """Get latest graph."""
    return get_ui_state_manager().get_latest_graph()


def get_latest_graph_for_ui():
    """Get latest graph for UI."""
    return get_ui_state_manager().get_latest_graph_for_ui()


def update_graph(graph):
    """Update graph."""
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
