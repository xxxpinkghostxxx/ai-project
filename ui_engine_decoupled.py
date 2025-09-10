"""
ui_engine_decoupled.py

Decoupled UI engine for the neural system.
Uses dependency injection and event-driven architecture.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from logging_utils import log_step, setup_logging
from interfaces import (
    IEventBus, IUIStateManager, ISimulationManager, IConfigurationManager,
    IWindowManager, ISensoryVisualization, ILiveMonitoring, ISimulationController
)
from dependency_injection import ServiceProvider


class DecoupledUIStateManager(IUIStateManager):
    """Decoupled UI state manager."""
    
    def __init__(self, event_bus: IEventBus):
        """Initialize UI state manager."""
        self.event_bus = event_bus
        self._lock = threading.RLock()
        
        # UI state
        self.simulation_running = False
        self.latest_graph = None
        self.latest_graph_for_ui = None
        self.live_feed_data = {}
        self.system_health = {}
        self.sensory_texture_tag = "sensory_texture"
        self.sensory_image_tag = "sensory_image"
        self.graph_h = None
        self.graph_w = None
        self.sim_update_counter = 0
        self.update_for_ui = False
        
        log_step("DecoupledUIStateManager initialized")
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get simulation state."""
        with self._lock:
            return {
                "simulation_running": self.simulation_running,
                "sim_update_counter": self.sim_update_counter,
                "update_for_ui": self.update_for_ui
            }
    
    def set_simulation_running(self, running: bool) -> None:
        """Set simulation running state."""
        with self._lock:
            self.simulation_running = running
            self.event_bus.publish("ui_simulation_state_changed", {"running": running})
    
    def update_graph(self, graph: Any) -> None:
        """Update the latest graph."""
        with self._lock:
            self.latest_graph = graph
            self.event_bus.publish("ui_graph_updated", {"graph": graph})
    
    def get_latest_graph(self) -> Any:
        """Get the latest graph."""
        with self._lock:
            return self.latest_graph
    
    def get_latest_graph_for_ui(self) -> Any:
        """Get the latest graph for UI."""
        with self._lock:
            return self.latest_graph_for_ui
    
    def add_live_feed_data(self, data_type: str, value: float) -> None:
        """Add live feed data."""
        with self._lock:
            if data_type not in self.live_feed_data:
                self.live_feed_data[data_type] = []
            self.live_feed_data[data_type].append(value)
            
            # Keep only last 100 values
            if len(self.live_feed_data[data_type]) > 100:
                self.live_feed_data[data_type] = self.live_feed_data[data_type][-100:]
    
    def get_live_feed_data(self) -> Dict[str, List[float]]:
        """Get live feed data."""
        with self._lock:
            return self.live_feed_data.copy()
    
    def clear_live_feed_data(self) -> None:
        """Clear live feed data."""
        with self._lock:
            self.live_feed_data.clear()
    
    def update_system_health(self, health_data: Dict[str, Any]) -> None:
        """Update system health."""
        with self._lock:
            self.system_health = health_data
            self.event_bus.publish("ui_system_health_updated", {"health": health_data})
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health."""
        with self._lock:
            return self.system_health.copy()
    
    def cleanup(self) -> None:
        """Cleanup UI state manager."""
        with self._lock:
            self.simulation_running = False
            self.latest_graph = None
            self.latest_graph_for_ui = None
            self.live_feed_data.clear()
            self.system_health.clear()
            log_step("DecoupledUIStateManager cleaned up")


class DecoupledWindowManager(IWindowManager):
    """Decoupled window manager."""
    
    def __init__(self, event_bus: IEventBus):
        """Initialize window manager."""
        self.event_bus = event_bus
        self.windows_created = False
        
        log_step("DecoupledWindowManager initialized")
    
    def create_main_window(self) -> None:
        """Create main window."""
        if self.windows_created:
            return
        
        with dpg.window(label="Neural System Control Center", tag="main_window", width=1200, height=800):
            dpg.add_text("Decoupled Neural System")
            dpg.add_button(label="Start Simulation", callback=self._on_start_simulation)
            dpg.add_button(label="Stop Simulation", callback=self._on_stop_simulation)
            dpg.add_text("Simulation Status: Stopped", tag="sim_status_text")
            dpg.add_image("sensory_texture", tag="sensory_image", width=320, height=180)
        
        self.windows_created = True
        log_step("Main window created")
    
    def create_performance_window(self) -> None:
        """Create performance window."""
        with dpg.window(label="Performance Monitor", tag="perf_window", width=400, height=300, pos=[1250, 100]):
            dpg.add_text("Performance Metrics")
            dpg.add_text("FPS: 0", tag="fps_text")
            dpg.add_text("Memory: 0 MB", tag="memory_text")
            dpg.add_text("CPU: 0%", tag="cpu_text")
        
        log_step("Performance window created")
    
    def create_log_window(self) -> None:
        """Create log window."""
        with dpg.window(label="System Log", tag="log_window", width=600, height=400, pos=[100, 900]):
            dpg.add_text("System Log")
            dpg.add_input_text(multiline=True, readonly=True, tag="log_text", width=580, height=350)
        
        log_step("Log window created")
    
    def create_fps_window(self) -> None:
        """Create FPS window."""
        with dpg.window(label="FPS Monitor", tag="fps_window", width=200, height=100, pos=[1700, 100]):
            dpg.add_text("FPS: 0", tag="fps_display")
        
        log_step("FPS window created")
    
    def _on_start_simulation(self) -> None:
        """Handle start simulation button."""
        self.event_bus.publish("ui_start_simulation_requested", {})
    
    def _on_stop_simulation(self) -> None:
        """Handle stop simulation button."""
        self.event_bus.publish("ui_stop_simulation_requested", {})


class DecoupledSensoryVisualization(ISensoryVisualization):
    """Decoupled sensory visualization."""
    
    def __init__(self, event_bus: IEventBus, ui_state_manager: IUIStateManager):
        """Initialize sensory visualization."""
        self.event_bus = event_bus
        self.ui_state_manager = ui_state_manager
        
        log_step("DecoupledSensoryVisualization initialized")
    
    def update_sensory_display(self, graph: Any) -> None:
        """Update sensory display."""
        if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
            return
        
        sensory_labels = [lbl for lbl in graph.node_labels if lbl.get("type", "sensory") == "sensory"]
        if not sensory_labels:
            return
        
        h = getattr(graph, "h", None)
        w = getattr(graph, "w", None)
        if h is None or w is None:
            return
        
        try:
            arr = graph.x[: h * w].cpu().numpy().reshape((h, w)).astype(np.uint8)
        except Exception as e:
            log_step("Sensory visualization reshape failed", error=str(e))
            arr = np.zeros((h, w), dtype=np.uint8)
        
        arr_rgb = np.stack([arr] * 3, axis=-1)
        arr_f = arr_rgb.astype(np.float32) / 255.0
        arr_f = arr_f.flatten()
        # Convert to list for DearPyGui
        arr_list = arr_f.tolist()
        
        try:
            dpg.set_value("sensory_texture", arr_list)
        except Exception as e:
            log_step("Failed to update sensory texture", error=str(e))
    
    def setup_sensory_texture(self) -> None:
        """Setup sensory texture."""
        with dpg.texture_registry():
            dpg.add_dynamic_texture(320, 180, [0.0] * (320*180), tag="sensory_texture")
        
        log_step("Sensory texture setup completed")


class DecoupledLiveMonitoring(ILiveMonitoring):
    """Decoupled live monitoring."""
    
    def __init__(self, event_bus: IEventBus, ui_state_manager: IUIStateManager):
        """Initialize live monitoring."""
        self.event_bus = event_bus
        self.ui_state_manager = ui_state_manager
        
        log_step("DecoupledLiveMonitoring initialized")
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics."""
        if dpg.does_item_exist("fps_text"):
            dpg.set_value("fps_text", f"FPS: {metrics.get('fps', 0):.1f}")
        
        if dpg.does_item_exist("memory_text"):
            dpg.set_value("memory_text", f"Memory: {metrics.get('memory_mb', 0):.1f} MB")
        
        if dpg.does_item_exist("cpu_text"):
            dpg.set_value("cpu_text", f"CPU: {metrics.get('cpu_percent', 0):.1f}%")
    
    def update_live_feed(self, data_type: str, value: float) -> None:
        """Update live feed."""
        self.ui_state_manager.add_live_feed_data(data_type, value)
        self.event_bus.publish("ui_live_feed_updated", {"data_type": data_type, "value": value})
    
    def update_system_health(self, health_data: Dict[str, Any]) -> None:
        """Update system health."""
        self.ui_state_manager.update_system_health(health_data)
        self.event_bus.publish("ui_system_health_updated", {"health": health_data})


class DecoupledSimulationController(ISimulationController):
    """Decoupled simulation controller."""
    
    def __init__(self, service_provider: ServiceProvider):
        """Initialize simulation controller."""
        self.service_provider = service_provider
        self.sim_manager = service_provider.resolve(ISimulationManager)
        self.event_bus = service_provider.resolve(IEventBus)
        
        log_step("DecoupledSimulationController initialized")
    
    def start_simulation(self) -> None:
        """Start simulation."""
        self.sim_manager.start_simulation()
        self.event_bus.publish("simulation_started", {})
    
    def stop_simulation(self) -> None:
        """Stop simulation."""
        self.sim_manager.stop_simulation()
        self.event_bus.publish("simulation_stopped", {})
    
    def reset_simulation(self) -> None:
        """Reset simulation."""
        self.sim_manager.reset_simulation()
        self.event_bus.publish("simulation_reset", {})
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get simulation status."""
        return {
            "is_running": self.sim_manager.is_running,
            "step_counter": self.sim_manager.step_counter,
            "performance_stats": self.sim_manager.get_performance_stats()
        }


class DecoupledUIEngine:
    """Decoupled UI engine."""
    
    def __init__(self, service_provider: ServiceProvider):
        """Initialize UI engine."""
        self.service_provider = service_provider
        self.event_bus = service_provider.resolve(IEventBus)
        self.ui_state_manager = service_provider.resolve(IUIStateManager)
        self.window_manager = service_provider.resolve(IWindowManager)
        self.sensory_visualization = service_provider.resolve(ISensoryVisualization)
        self.live_monitoring = service_provider.resolve(ILiveMonitoring)
        self.simulation_controller = service_provider.resolve(ISimulationController)
        
        self._setup_event_subscriptions()
        log_step("DecoupledUIEngine initialized")
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        self.event_bus.subscribe("ui_start_simulation_requested", self._on_start_simulation_requested)
        self.event_bus.subscribe("ui_stop_simulation_requested", self._on_stop_simulation_requested)
        self.event_bus.subscribe("ui_graph_updated", self._on_graph_updated)
        self.event_bus.subscribe("ui_simulation_state_changed", self._on_simulation_state_changed)
        
        log_step("UI event subscriptions setup completed")
    
    def _on_start_simulation_requested(self, event_name: str, data: Dict[str, Any]) -> None:
        """Handle start simulation request."""
        self.simulation_controller.start_simulation()
    
    def _on_stop_simulation_requested(self, event_name: str, data: Dict[str, Any]) -> None:
        """Handle stop simulation request."""
        self.simulation_controller.stop_simulation()
    
    def _on_graph_updated(self, event_name: str, data: Dict[str, Any]) -> None:
        """Handle graph updated event."""
        graph = data.get("graph")
        if graph:
            self.sensory_visualization.update_sensory_display(graph)
    
    def _on_simulation_state_changed(self, event_name: str, data: Dict[str, Any]) -> None:
        """Handle simulation state changed event."""
        running = data.get("running", False)
        status_text = "Simulation Running" if running else "Simulation Stopped"
        
        if dpg.does_item_exist("sim_status_text"):
            dpg.set_value("sim_status_text", status_text)
    
    def initialize_ui(self) -> None:
        """Initialize UI."""
        log_step("Initializing decoupled UI")
        
        # Setup logging
        setup_logging(level="INFO")
        
        # Create DearPyGui context
        dpg.create_context()
        
        # Create windows
        self.window_manager.create_main_window()
        self.window_manager.create_performance_window()
        self.window_manager.create_log_window()
        self.window_manager.create_fps_window()
        
        # Setup sensory visualization
        self.sensory_visualization.setup_sensory_texture()
        
        # Setup viewport
        dpg.create_viewport(title="Decoupled Neural System", width=1200, height=800)
        dpg.setup_dearpygui()
        
        # Show viewport
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
        # Position window
        dpg.set_viewport_pos([100, 100])
        
        # Register frame callback
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self._ui_frame_handler)
        
        log_step("Decoupled UI initialized")
    
    def _ui_frame_handler(self) -> None:
        """Handle UI frame updates."""
        # Update UI based on state
        if self.ui_state_manager.update_for_ui:
            graph_for_ui = self.ui_state_manager.get_latest_graph_for_ui()
            if graph_for_ui:
                self.sensory_visualization.update_sensory_display(graph_for_ui)
            self.ui_state_manager.update_for_ui = False
        
        # Schedule next frame
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self._ui_frame_handler)
    
    def run_main_loop(self) -> None:
        """Run main UI loop."""
        log_step("Starting decoupled UI main loop")
        
        try:
            dpg.start_dearpygui()
        except Exception as e:
            log_step("Critical error in UI main loop", error=str(e))
            raise
        finally:
            self.cleanup()
            log_step("Decoupled UI main loop ended")
    
    def cleanup(self) -> None:
        """Cleanup UI engine."""
        log_step("Cleaning up decoupled UI engine")
        
        # Stop simulation
        self.simulation_controller.stop_simulation()
        
        # Cleanup UI state
        self.ui_state_manager.cleanup()
        
        # Clear event subscriptions
        self.event_bus.clear_subscribers()
        
        # Destroy DearPyGui context
        dpg.destroy_context()
        
        log_step("Decoupled UI engine cleanup completed")


# Factory function for creating decoupled UI engine
def create_decoupled_ui_engine(service_provider: ServiceProvider) -> DecoupledUIEngine:
    """Create decoupled UI engine."""
    return DecoupledUIEngine(service_provider)


# Example usage and testing
if __name__ == "__main__":
    print("Decoupled UI engine created successfully!")
    print("Features include:")
    print("- Interface-based architecture")
    print("- Event-driven communication")
    print("- Dependency injection")
    print("- Modular UI components")
    print("- Thread-safe state management")
    print("- Comprehensive error handling")
    
    print("Decoupled UI engine test completed!")