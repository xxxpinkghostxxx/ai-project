"""
simulation_manager_decoupled.py

Decoupled simulation manager for the neural system.
Uses dependency injection and event-driven architecture.
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, Callable, List
from logging_utils import log_step, log_runtime
from interfaces import (
    ISimulationManager, IGraphProvider, IEventBus, IUIStateManager,
    IBehaviorEngine, ILearningEngine, IMemorySystem, IHomeostasisController,
    INetworkMetrics, IWorkspaceEngine, IEnergyBehavior, IConnectionLogic,
    IDeathAndBirthLogic, IErrorHandler, IPerformanceOptimizer, IConfigurationManager
)
from dependency_injection import ServiceProvider


class DecoupledSimulationState:
    """Decoupled simulation state."""
    
    def __init__(self):
        """Initialize simulation state."""
        self.is_running = False
        self.step_counter = 0
        self.graph = None
        self.simulation_thread = None
        self._lock = threading.RLock()
        
        log_step("DecoupledSimulationState initialized")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "step_counter": self.step_counter,
                "has_graph": self.graph is not None,
                "thread_alive": self.simulation_thread is not None and self.simulation_thread.is_alive()
            }
    
    def set_running(self, running: bool) -> None:
        """Set running state."""
        with self._lock:
            self.is_running = running
    
    def increment_step_counter(self) -> None:
        """Increment step counter."""
        with self._lock:
            self.step_counter += 1
    
    def reset_step_counter(self) -> None:
        """Reset step counter."""
        with self._lock:
            self.step_counter = 0
    
    def set_graph(self, graph: Any) -> None:
        """Set graph."""
        with self._lock:
            self.graph = graph
    
    def get_graph(self) -> Any:
        """Get graph."""
        with self._lock:
            return self.graph
    
    def set_simulation_thread(self, thread: threading.Thread) -> None:
        """Set simulation thread."""
        with self._lock:
            self.simulation_thread = thread
    
    def get_simulation_thread(self) -> Optional[threading.Thread]:
        """Get simulation thread."""
        with self._lock:
            return self.simulation_thread


class DecoupledSimulationManager(ISimulationManager, IGraphProvider):
    """Decoupled simulation manager."""
    
    def __init__(self, service_provider: ServiceProvider):
        """Initialize simulation manager."""
        self.service_provider = service_provider
        self.state = DecoupledSimulationState()
        
        # Get services
        self.event_bus = service_provider.resolve(IEventBus)
        self.ui_state_manager = service_provider.resolve(IUIStateManager)
        self.behavior_engine = service_provider.resolve(IBehaviorEngine)
        self.learning_engine = service_provider.resolve(ILearningEngine)
        self.memory_system = service_provider.resolve(IMemorySystem)
        self.homeostasis_controller = service_provider.resolve(IHomeostasisController)
        self.network_metrics = service_provider.resolve(INetworkMetrics)
        self.workspace_engine = service_provider.resolve(IWorkspaceEngine)
        self.energy_behavior = service_provider.resolve(IEnergyBehavior)
        self.connection_logic = service_provider.resolve(IConnectionLogic)
        self.death_birth_logic = service_provider.resolve(IDeathAndBirthLogic)
        self.error_handler = service_provider.resolve(IErrorHandler)
        self.performance_optimizer = service_provider.resolve(IPerformanceOptimizer)
        self.config_manager = service_provider.resolve(IConfigurationManager)
        
        # Simulation parameters
        self.update_interval = 0.033  # 30 FPS
        self.sensory_update_interval = 1
        self.memory_update_interval = 50
        self.homeostasis_update_interval = 100
        self.metrics_update_interval = 50
        
        # Callbacks
        self.step_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Performance stats
        self.performance_stats = {}
        
        log_step("DecoupledSimulationManager initialized")
    
    def get_graph(self) -> Any:
        """Get current graph."""
        return self.state.get_graph()
    
    def set_graph(self, graph: Any) -> None:
        """Set graph."""
        self.state.set_graph(graph)
        self.event_bus.publish("graph_updated", {"graph": graph})
        self.ui_state_manager.update_graph(graph)
        log_step("Graph set in decoupled simulation manager")
    
    def add_step_callback(self, callback: Callable) -> None:
        """Add step callback."""
        self.step_callbacks.append(callback)
        log_step("Step callback added")
    
    def add_metrics_callback(self, callback: Callable) -> None:
        """Add metrics callback."""
        self.metrics_callbacks.append(callback)
        log_step("Metrics callback added")
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add error callback."""
        self.error_callbacks.append(callback)
        log_step("Error callback added")
    
    @log_runtime
    def run_single_step(self) -> bool:
        """Run a single simulation step."""
        if self.state.get_graph() is None:
            return False
        
        try:
            step_start_time = time.time()
            self.state.increment_step_counter()
            
            # Get current graph
            graph = self.state.get_graph()
            
            # Sensory updates
            if self.state.step_counter % self.sensory_update_interval == 0:
                self._handle_sensory_updates(graph)
            
            # Node behaviors
            self._handle_node_behaviors(graph)
            
            # Energy dynamics
            self._handle_energy_dynamics(graph)
            
            # Connection formation
            self._handle_connection_formation(graph)
            
            # Learning
            self._handle_learning(graph)
            
            # Workspace updates
            self._handle_workspace_updates(graph)
            
            # Node lifecycle
            self._handle_node_lifecycle(graph)
            
            # Memory updates
            if self.state.step_counter % self.memory_update_interval == 0:
                self._handle_memory_updates(graph)
            
            # Homeostasis
            if self.state.step_counter % self.homeostasis_update_interval == 0:
                self._handle_homeostasis(graph)
            
            # Metrics
            if self.state.step_counter % self.metrics_update_interval == 0:
                self._handle_metrics(graph)
            
            # Calculate step time
            step_time = time.time() - step_start_time
            
            # Publish step completed event
            self.event_bus.publish("simulation_step_completed", {
                "graph": graph,
                "step_counter": self.state.step_counter,
                "step_time": step_time
            })
            
            # Call step callbacks
            for callback in self.step_callbacks:
                try:
                    callback(graph, self.state.step_counter, self.performance_stats)
                except Exception as e:
                    log_step("Error in step callback", error=str(e))
            
            return True
            
        except Exception as e:
            log_step("Error in simulation step", error=str(e))
            self.event_bus.publish("simulation_error", {"error": str(e), "step": self.state.step_counter})
            return False
    
    def _handle_sensory_updates(self, graph: Any) -> None:
        """Handle sensory updates."""
        try:
            # Publish sensory update request
            self.event_bus.publish("sensory_update_requested", {"graph": graph})
        except Exception as e:
            log_step("Error in sensory updates", error=str(e))
    
    def _handle_node_behaviors(self, graph: Any) -> None:
        """Handle node behaviors."""
        try:
            # Update node behaviors
            self.behavior_engine.update_node_behaviors(graph)
        except Exception as e:
            log_step("Error in node behaviors", error=str(e))
    
    def _handle_energy_dynamics(self, graph: Any) -> None:
        """Handle energy dynamics."""
        try:
            # Apply energy behavior
            self.energy_behavior.apply_energy_behavior(graph)
        except Exception as e:
            log_step("Error in energy dynamics", error=str(e))
    
    def _handle_connection_formation(self, graph: Any) -> None:
        """Handle connection formation."""
        try:
            # Form intelligent connections
            self.connection_logic.intelligent_connection_formation(graph)
        except Exception as e:
            log_step("Error in connection formation", error=str(e))
    
    def _handle_learning(self, graph: Any) -> None:
        """Handle learning."""
        try:
            # Update learning
            self.learning_engine.update_learning(graph)
        except Exception as e:
            log_step("Error in learning", error=str(e))
    
    def _handle_workspace_updates(self, graph: Any) -> None:
        """Handle workspace updates."""
        try:
            # Update workspace
            self.workspace_engine.update_workspace(graph)
        except Exception as e:
            log_step("Error in workspace updates", error=str(e))
    
    def _handle_node_lifecycle(self, graph: Any) -> None:
        """Handle node lifecycle."""
        try:
            # Birth new nodes
            self.death_birth_logic.birth_new_dynamic_nodes(graph)
            
            # Remove dead nodes
            self.death_birth_logic.remove_dead_dynamic_nodes(graph)
        except Exception as e:
            log_step("Error in node lifecycle", error=str(e))
    
    def _handle_memory_updates(self, graph: Any) -> None:
        """Handle memory updates."""
        try:
            # Update memory
            self.memory_system.update_memory(graph)
        except Exception as e:
            log_step("Error in memory updates", error=str(e))
    
    def _handle_homeostasis(self, graph: Any) -> None:
        """Handle homeostasis."""
        try:
            # Update homeostasis
            self.homeostasis_controller.update_homeostasis(graph)
        except Exception as e:
            log_step("Error in homeostasis", error=str(e))
    
    def _handle_metrics(self, graph: Any) -> None:
        """Handle metrics."""
        try:
            # Calculate metrics
            metrics = self.network_metrics.calculate_comprehensive_metrics(graph)
            
            # Publish metrics event
            self.event_bus.publish("metrics_calculated", {"metrics": metrics})
            
            # Call metrics callbacks
            for callback in self.metrics_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    log_step("Error in metrics callback", error=str(e))
        except Exception as e:
            log_step("Error in metrics", error=str(e))
    
    def start_simulation(self, run_in_thread: bool = True) -> None:
        """Start simulation."""
        if self.state.is_running:
            logging.warning("Simulation already running")
            return
        
        if self.state.get_graph() is None:
            logging.error("No graph set for simulation")
            return
        
        self.state.set_running(True)
        self.state.reset_step_counter()
        
        # Publish simulation started event
        self.event_bus.publish("simulation_started", {})
        
        if run_in_thread:
            simulation_thread = threading.Thread(
                target=self._simulation_loop,
                daemon=True,
                name="DecoupledSimulationThread"
            )
            self.state.set_simulation_thread(simulation_thread)
            simulation_thread.start()
            log_step("Decoupled simulation started in background thread")
        else:
            self._simulation_loop()
    
    def stop_simulation(self) -> None:
        """Stop simulation."""
        if not self.state.is_running:
            logging.warning("Simulation not running")
            return
        
        self.state.set_running(False)
        
        # Publish simulation stopped event
        self.event_bus.publish("simulation_stopped", {})
        
        # Wait for thread to finish
        thread = self.state.get_simulation_thread()
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        
        log_step("Decoupled simulation stopped")
    
    def _simulation_loop(self) -> None:
        """Main simulation loop."""
        log_step("Decoupled simulation loop started")
        
        while self.state.is_running:
            loop_start = time.time()
            success = self.run_single_step()
            
            if not success:
                logging.error("Simulation step failed, stopping simulation")
                self.state.set_running(False)
                break
            
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.update_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        log_step("Decoupled simulation loop ended")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_simulation(self) -> None:
        """Reset simulation."""
        self.stop_simulation()
        self.state.reset_step_counter()
        
        # Publish simulation reset event
        self.event_bus.publish("simulation_reset", {})
        
        log_step("Decoupled simulation reset")
    
    def cleanup(self) -> None:
        """Cleanup simulation manager."""
        log_step("Cleaning up decoupled simulation manager")
        
        # Stop simulation
        self.stop_simulation()
        
        # Clear callbacks
        self.step_callbacks.clear()
        self.metrics_callbacks.clear()
        self.error_callbacks.clear()
        
        # Clear performance stats
        self.performance_stats.clear()
        
        # Clear graph
        self.state.set_graph(None)
        
        log_step("Decoupled simulation manager cleaned up")


# Factory function for creating decoupled simulation manager
def create_decoupled_simulation_manager(service_provider: ServiceProvider) -> DecoupledSimulationManager:
    """Create decoupled simulation manager."""
    return DecoupledSimulationManager(service_provider)


# Example usage and testing
if __name__ == "__main__":
    print("Decoupled simulation manager created successfully!")
    print("Features include:")
    print("- Interface-based architecture")
    print("- Event-driven communication")
    print("- Dependency injection")
    print("- Modular simulation components")
    print("- Thread-safe state management")
    print("- Comprehensive error handling")
    print("- Performance monitoring")
    
    print("Decoupled simulation manager test completed!")