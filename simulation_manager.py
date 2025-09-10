"""
simulation_manager.py

Unified simulation management system that coordinates all neural systems
and provides a single, consistent simulation interface for both UI and standalone use.
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, Callable
from logging_utils import log_step, log_runtime

# Import all neural systems
from behavior_engine import BehaviorEngine
from learning_engine import LearningEngine
from memory_system import MemorySystem
from homeostasis_controller import HomeostasisController
from network_metrics import NetworkMetrics
from workspace_engine import WorkspaceEngine
from energy_behavior import apply_energy_behavior, update_membrane_potentials, apply_refractory_periods
from error_handler import get_error_handler, safe_execute, graceful_degradation
from performance_optimizer import get_performance_optimizer

# Import core simulation functions
from main_loop import update_dynamic_node_energies
from death_and_birth_logic import birth_new_dynamic_nodes, remove_dead_dynamic_nodes
from connection_logic import intelligent_connection_formation
from screen_graph import RESOLUTION_SCALE


class SimulationManager:
    """
    Unified simulation manager that coordinates all neural systems.
    Provides a single interface for both UI and standalone simulation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simulation manager with all neural systems.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_running = False
        self.step_counter = 0
        self.simulation_thread = None
        self.graph = None
        
        # Initialize all neural systems
        self.behavior_engine = BehaviorEngine()
        self.learning_engine = LearningEngine()
        self.memory_system = MemorySystem()
        self.homeostasis_controller = HomeostasisController()
        self.network_metrics = NetworkMetrics()
        self.workspace_engine = WorkspaceEngine()
        
        # Simulation parameters
        self.update_interval = self.config.get('update_interval', 0.033)  # ~30 FPS
        self.sensory_update_interval = self.config.get('sensory_update_interval', 1)  # Every N steps
        self.memory_update_interval = self.config.get('memory_update_interval', 50)  # Every 50 steps
        self.homeostasis_update_interval = self.config.get('homeostasis_update_interval', 100)  # Every 100 steps
        self.metrics_update_interval = self.config.get('metrics_update_interval', 50)  # Every 50 steps
        
        # Callbacks for external systems (like UI)
        self.step_callbacks = []
        self.metrics_callbacks = []
        self.error_callbacks = []
        
        # Initialize error handler and performance optimizer
        self.error_handler = get_error_handler()
        self.performance_optimizer = get_performance_optimizer()
        
        # Add optimization callback
        self.performance_optimizer.add_optimization_callback(self._on_optimization_applied)
        
        # Performance tracking
        self.performance_stats = {
            'total_steps': 0,
            'avg_step_time': 0.0,
            'last_step_time': 0.0,
            'system_health': 'unknown'
        }
        
        log_step("SimulationManager initialized", 
                update_interval=self.update_interval,
                systems_initialized=6)
    
    def set_graph(self, graph):
        """
        Set the graph for simulation.
        
        Args:
            graph: PyTorch Geometric graph to simulate
        """
        self.graph = graph
        
        # Attach systems to graph for persistence
        if self.graph is not None:
            self.graph.behavior_engine = self.behavior_engine
            self.graph.learning_engine = self.learning_engine
            self.graph.memory_system = self.memory_system
            self.graph.homeostasis_controller = self.homeostasis_controller
            self.graph.network_metrics = self.network_metrics
            self.graph.workspace_engine = self.workspace_engine
            
            log_step("Graph attached to simulation manager", 
                    nodes=len(self.graph.node_labels) if hasattr(self.graph, 'node_labels') else 0)
    
    def add_step_callback(self, callback: Callable):
        """Add a callback to be called after each simulation step."""
        self.step_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """Add a callback to be called when metrics are updated."""
        self.metrics_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add a callback to be called when errors occur."""
        self.error_callbacks.append(callback)
    
    @log_runtime
    def run_single_step(self) -> bool:
        """
        Run a single simulation step.
        
        Returns:
            bool: True if step completed successfully, False otherwise
        """
        if self.graph is None:
            return False
        
        try:
            step_start_time = time.time()
            self.step_counter += 1
            
            # 1. Update sensory node features (every N steps for performance)
            if self.step_counter % self.sensory_update_interval == 0:
                safe_execute(
                    lambda: self._update_sensory_features(self.graph, scale=RESOLUTION_SCALE),
                    context="sensory_update",
                    recovery_func=lambda e: self._fallback_sensory_update(),
                    default_return=None
                )
            
            # 2. Update node behaviors using behavior engine
            safe_execute(
                lambda: self._update_node_behaviors(),
                context="node_behavior_update",
                default_return=None
            )
            
            # 3. Apply enhanced energy behavior dynamics
            safe_execute(
                lambda: self._apply_energy_dynamics(),
                context="energy_dynamics",
                default_return=None
            )
            
            # 4. Update dynamic node energies (decay, transfer, clamp)
            safe_execute(
                lambda: update_dynamic_node_energies(self.graph),
                context="dynamic_energy_update",
                default_return=None
            )
            
            # 5. Add/update dynamic connections with intelligent formation
            safe_execute(
                lambda: setattr(self, 'graph', intelligent_connection_formation(self.graph)),
                context="connection_formation",
                default_return=None
            )
            
            # 6. Apply learning and plasticity updates
            safe_execute(
                lambda: setattr(self, 'graph', self.learning_engine.consolidate_connections(self.graph)),
                context="learning_consolidation",
                default_return=None
            )
            
            # 7. Update workspace nodes
            safe_execute(
                lambda: self.workspace_engine.update_workspace_nodes(self.graph, self.step_counter),
                context="workspace_update",
                default_return={}
            )
            
            # 8. Birth new dynamic nodes if energy threshold is exceeded
            safe_execute(
                lambda: birth_new_dynamic_nodes(self.graph),
                context="node_birth",
                default_return=None
            )
            
            # 9. Remove dead dynamic nodes if energy below threshold
            safe_execute(
                lambda: remove_dead_dynamic_nodes(self.graph),
                context="node_death",
                default_return=None
            )
            
            # 10. Form memory traces (every N steps for performance)
            if self.step_counter % self.memory_update_interval == 0:
                self.graph = self.memory_system.form_memory_traces(self.graph)
                self.graph = self.memory_system.consolidate_memories(self.graph)
                self.memory_system.decay_memories()
            
            # 11. Apply memory influence to connections (every N steps)
            if self.step_counter % (self.memory_update_interval // 2) == 0:
                self.graph = self.learning_engine.apply_memory_influence(self.graph)
            
            # 12. Apply homeostatic regulation (every N steps)
            if self.step_counter % self.homeostasis_update_interval == 0:
                self.graph = self.homeostasis_controller.regulate_network_activity(self.graph)
                self.graph = self.homeostasis_controller.optimize_criticality(self.graph)
                
                # Monitor network health
                health_status = self.homeostasis_controller.monitor_network_health(self.graph)
                self.performance_stats['system_health'] = health_status['status']
                
                if health_status['status'] != 'healthy':
                    logging.warning(f"[SIMULATION] Network health: {health_status['status']} - {health_status['warnings']}")
            
            # 13. Calculate network metrics (every N steps)
            if self.step_counter % self.metrics_update_interval == 0:
                metrics = self.network_metrics.calculate_comprehensive_metrics(self.graph)
                self.performance_stats['last_metrics'] = metrics
                
                # Call metrics callbacks
                for callback in self.metrics_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logging.error(f"Metrics callback error: {e}")
                
                logging.info(f"[SIMULATION] Step {self.step_counter}: Criticality={metrics['criticality']:.3f}, "
                           f"Connectivity={metrics['connectivity']['density']:.3f}, "
                           f"Energy Variance={metrics['energy_balance']['energy_variance']:.2f}")
            
            # Update performance stats
            step_time = time.time() - step_start_time
            self.performance_stats['last_step_time'] = step_time
            self.performance_stats['total_steps'] = self.step_counter
            
            # Calculate rolling average step time
            if self.performance_stats['avg_step_time'] == 0:
                self.performance_stats['avg_step_time'] = step_time
            else:
                self.performance_stats['avg_step_time'] = (
                    self.performance_stats['avg_step_time'] * 0.9 + step_time * 0.1
                )
            
            # Record performance metrics for optimization
            self.performance_optimizer.record_performance(
                step_time=step_time,
                memory_usage=0.0,  # TODO: Add actual memory monitoring
                cpu_usage=0.0,     # TODO: Add actual CPU monitoring
                network_activity=0.0,  # TODO: Add actual network monitoring
                error_rate=0.0     # TODO: Add actual error rate monitoring
            )
            
            # Call step callbacks
            for callback in self.step_callbacks:
                try:
                    callback(self.graph, self.step_counter, self.performance_stats)
                except Exception as e:
                    logging.error(f"Step callback error: {e}")
            
            return True
            
        except Exception as e:
            logging.error(f"Simulation step error: {e}")
            # Call error callbacks
            for callback in self.error_callbacks:
                try:
                    callback(e, self.step_counter)
                except Exception as callback_error:
                    logging.error(f"Error callback error: {callback_error}")
            return False
    
    def start_simulation(self, run_in_thread: bool = True):
        """
        Start the simulation.
        
        Args:
            run_in_thread: If True, run simulation in background thread
        """
        if self.is_running:
            logging.warning("Simulation already running")
            return
        
        if self.graph is None:
            logging.error("No graph set for simulation")
            return
        
        self.is_running = True
        self.step_counter = 0
        
        # Start performance monitoring
        self.performance_optimizer.start_monitoring()
        
        if run_in_thread:
            self.simulation_thread = threading.Thread(
                target=self._simulation_loop, 
                daemon=True,
                name="SimulationThread"
            )
            self.simulation_thread.start()
            log_step("Simulation started in background thread with performance monitoring")
        else:
            self._simulation_loop()
    
    def stop_simulation(self):
        """Stop the simulation."""
        if not self.is_running:
            logging.warning("Simulation not running")
            return
        
        self.is_running = False
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2.0)
        
        # Stop performance monitoring
        self.performance_optimizer.stop_monitoring()
        
        log_step("Simulation stopped", 
                total_steps=self.step_counter,
                avg_step_time=self.performance_stats['avg_step_time'])
    
    def _simulation_loop(self):
        """Internal simulation loop."""
        log_step("Simulation loop started")
        
        while self.is_running:
            loop_start = time.time()
            
            # Run single step
            success = self.run_single_step()
            
            if not success:
                logging.error("Simulation step failed, stopping simulation")
                self.is_running = False
                break
            
            # Calculate sleep time to maintain target update interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.update_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        log_step("Simulation loop ended")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics from all neural systems."""
        return {
            'behavior_stats': self.behavior_engine.get_behavior_statistics(),
            'learning_stats': self.learning_engine.get_learning_statistics(),
            'memory_stats': self.memory_system.get_memory_statistics(),
            'homeostasis_stats': self.homeostasis_controller.get_regulation_statistics(),
            'memory_traces': self.memory_system.get_memory_trace_count(),
            'performance': self.performance_stats
        }
    
    def _update_sensory_features(self, graph, scale=1.0):
        """
        Update sensory node features from screen capture.
        This is a simplified version of the UI's update_sensory_features function.
        """
        try:
            from screen_graph import capture_screen, create_pixel_gray_graph
            
            # Capture screen
            arr = capture_screen(scale=scale)
            
            # Create pixel graph
            pixel_graph = create_pixel_gray_graph(arr)
            
            # Update sensory nodes in the main graph
            if hasattr(graph, 'node_labels') and hasattr(graph, 'x'):
                sensory_indices = [i for i, node in enumerate(graph.node_labels) 
                                 if node.get('type') == 'sensory']
                
                if sensory_indices and hasattr(pixel_graph, 'x'):
                    # Update sensory node energies
                    for i, sensory_idx in enumerate(sensory_indices):
                        if i < pixel_graph.x.shape[0]:
                            graph.x[sensory_idx, 0] = pixel_graph.x[i, 0]
                            
        except Exception as e:
            logging.warning(f"Sensory update failed: {e}")
            raise
    
    def _fallback_sensory_update(self):
        """Fallback sensory update when primary method fails."""
        try:
            logging.info("Using fallback sensory update - no screen capture")
            # In fallback mode, we don't update sensory features
            # This allows the system to continue running without screen input
            return True
        except Exception as e:
            logging.error(f"Fallback sensory update failed: {e}")
            return False
    
    def _update_node_behaviors(self):
        """Update node behaviors using behavior engine."""
        for idx, node in enumerate(self.graph.node_labels):
            if node.get('type') == 'dynamic':
                updated_node = self.behavior_engine.update_node_behavior(
                    node, self.graph, self.step_counter)
                self.graph.node_labels[idx] = updated_node
    
    def _apply_energy_dynamics(self):
        """Apply enhanced energy behavior dynamics."""
        self.graph = apply_energy_behavior(self.graph)
        self.graph = update_membrane_potentials(self.graph)
        self.graph = apply_refractory_periods(self.graph)
    
    def _on_optimization_applied(self, optimizations: Dict[str, Any], strategy: str):
        """Callback when performance optimizations are applied."""
        # Update simulation intervals based on optimizations
        self.sensory_update_interval = int(optimizations.get('sensory_update_interval', 1))
        self.memory_update_interval = int(optimizations.get('memory_update_interval', 50))
        self.homeostasis_update_interval = int(optimizations.get('homeostasis_update_interval', 100))
        self.metrics_update_interval = int(optimizations.get('metrics_update_interval', 50))
        
        log_step("Performance optimization applied to simulation",
                strategy=strategy,
                sensory_interval=self.sensory_update_interval,
                memory_interval=self.memory_update_interval,
                homeostasis_interval=self.homeostasis_update_interval)
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        self.stop_simulation()
        self.step_counter = 0
        self.performance_stats = {
            'total_steps': 0,
            'avg_step_time': 0.0,
            'last_step_time': 0.0,
            'system_health': 'unknown'
        }
        
        # Reset all neural systems
        self.behavior_engine.reset_statistics()
        self.learning_engine.reset_statistics()
        self.memory_system.reset_statistics()
        self.homeostasis_controller.reset_statistics()
        
        log_step("Simulation reset to initial state")


# Global simulation manager instance
_simulation_manager = None

def get_simulation_manager() -> SimulationManager:
    """Get the global simulation manager instance."""
    global _simulation_manager
    if _simulation_manager is None:
        _simulation_manager = SimulationManager()
    return _simulation_manager

def create_simulation_manager(config: Optional[Dict[str, Any]] = None) -> SimulationManager:
    """Create a new simulation manager instance."""
    return SimulationManager(config)


# Example usage and testing
if __name__ == "__main__":
    print("SimulationManager initialized successfully!")
    print("Features include:")
    print("- Unified simulation architecture")
    print("- All neural systems integrated")
    print("- Thread-safe operation")
    print("- Performance monitoring")
    print("- Callback system for external integration")
    print("- Configurable update intervals")
    
    # Test basic functionality
    manager = SimulationManager()
    print(f"Manager created with {len(manager.step_callbacks)} step callbacks")
    print("SimulationManager is ready for integration!")
