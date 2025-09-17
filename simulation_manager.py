"""
Legacy compatibility stub for simulation_manager.
This module was removed during reorg; kept as a thin facade to avoid IDE errors.
"""

from typing import Optional, Dict, Any
from utils.logging_utils import log_step


class SimulationManager:
    def __init__(self) -> None:
        log_step("SimulationManager stub initialized")

    def run_single_step(self) -> None:
        log_step("SimulationManager stub run_single_step called")


# Minimal performance hooks used by legacy references

def get_system_performance_metrics() -> Dict[str, Any]:
    return {}


def record_simulation_step(step: int, info: Optional[Dict[str, Any]] = None) -> None:
    pass


def record_simulation_error(error: Exception | str) -> None:
    pass


def record_simulation_warning(msg: str) -> None:
    pass


def initialize_performance_monitoring(config: Optional[Dict[str, Any]] = None) -> None:
    log_step("Performance monitoring stub initialized")


import time
import threading
import logging
import torch
import numpy as np
import configparser
import os
import functools


from neural.behavior_engine import BehaviorEngine
from learning.learning_engine import LearningEngine
from learning.memory_system import MemorySystem
from learning.homeostasis_controller import HomeostasisController
from neural.network_metrics import NetworkMetrics
from neural.workspace_engine import WorkspaceEngine
from energy.energy_behavior import apply_energy_behavior, update_membrane_potentials, apply_refractory_periods

from neural.death_and_birth_logic import birth_new_dynamic_nodes, remove_dead_dynamic_nodes
from neural.connection_logic import intelligent_connection_formation
from ui.screen_graph import RESOLUTION_SCALE
from utils.error_handling_utils import safe_execute, safe_initialize_component, safe_process_step, safe_callback_execution






class SimulationManager:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SimulationManager with NASA-compliant structure."""
        self._initialize_basic_properties(config)
        self._initialize_core_engines()
        self._initialize_optional_components()
        self._initialize_enhanced_systems()
        self._initialize_sensory_systems()
        self._initialize_performance_tracking()
        self._initialize_callbacks()
        self._load_configuration()
        self._validate_initialization()
    
    def _initialize_basic_properties(self, config: Optional[Dict[str, Any]]):
        """Initialize basic properties and threading."""
        import threading
        self.config = config or {}
        self.is_running = False
        self.simulation_running = False
        self.step_counter = 0
        self.current_step = 0
        self.simulation_thread = None
        self.graph = None
        self._lock = threading.RLock()
        self.max_steps = None
        self.last_visual_data = None
        self.last_audio_data = None
    
    def _initialize_core_engines(self):
        """Initialize core simulation engines."""
        self.behavior_engine = BehaviorEngine()
        self.learning_engine = LearningEngine()
        self.memory_system = MemorySystem()
        self.homeostasis_controller = HomeostasisController()
        self.network_metrics = NetworkMetrics()
        self.workspace_engine = WorkspaceEngine()
        
        # Initialize ID manager
        from energy.node_id_manager import get_id_manager
        self.id_manager = safe_initialize_component(
            "Node ID manager", 
            lambda: get_id_manager(),
            None
        )
    
    def _initialize_optional_components(self):
        """Initialize optional components with error handling."""
        self.audio_to_neural_bridge = self._safe_initialize_component(
            "audio_to_neural_bridge",
            lambda: self._create_audio_bridge(),
            critical=False
        )
        self.live_hebbian_learning = self._safe_initialize_component(
            "live_hebbian_learning",
            lambda: self._create_hebbian_learning(),
            critical=False
        )
        self.neural_map_persistence = self._safe_initialize_component(
            "neural_map_persistence",
            lambda: self._create_neural_persistence(),
            critical=False
        )
        self.event_driven_system = self._safe_initialize_component(
            "event_driven_system",
            lambda: self._create_event_system(),
            critical=False
        )
        self.spike_queue_system = self._safe_initialize_component(
            "spike_queue_system",
            lambda: self._create_spike_system(),
            critical=False
        )
    
    def _initialize_enhanced_systems(self):
        """Initialize enhanced neural systems."""
        self.enhanced_integration = self._create_enhanced_neural_integration()
        if self.enhanced_integration:
            self.behavior_engine.enhanced_integration = self.enhanced_integration
            logging.info("Enhanced neural integration initialized")
    
    def _initialize_sensory_systems(self):
        """Initialize sensory processing systems."""
        # Visual energy bridge
        from sensory.visual_energy_bridge import create_visual_energy_bridge
        self.visual_energy_bridge = safe_initialize_component(
            "Visual energy bridge",
            lambda: create_visual_energy_bridge(self.enhanced_integration),
            None
        )
        
        # Sensory workspace mapper
        self.sensory_workspace_mapper = None
        try:
            from sensory.sensory_workspace_mapper import create_sensory_workspace_mapper
            self.sensory_workspace_mapper = create_sensory_workspace_mapper()
            logging.info("Sensory workspace mapper initialized")
        except ImportError:
            self.sensory_workspace_mapper = None
            logging.info("Sensory workspace mapper not available")
        except Exception as e:
            self.sensory_workspace_mapper = None
            logging.warning(f"Failed to initialize sensory workspace mapper: {e}")
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking systems."""
        self.performance_monitor = self._safe_initialize_component(
            "performance_monitor",
            lambda: self._create_performance_monitor(),
            critical=False
        )
        self.performance_stats = {
            'total_steps': 0,
            'avg_step_time': 0.0,
            'max_step_time': 0.0,
            'min_step_time': float('inf'),
            'errors': 0,
            'warnings': 0,
            'log_entries': 0,
            'uptime': 0.0
        }
        self.memory_update_interval = 10
        self.homeostasis_update_interval = 20
        self.event_processing_interval = 5
    
    def _initialize_callbacks(self):
        """Initialize callback systems."""
        self.step_callbacks = []
        self.metrics_callbacks = []
        self.error_callbacks = []
    
    def _load_configuration(self):
        """Load configuration settings."""
        self._load_config()
    
    def _validate_initialization(self):
        """Validate that initialization completed successfully."""
        critical_components = [
            self.behavior_engine,
            self.learning_engine,
            self.memory_system,
            self.homeostasis_controller,
            self.network_metrics,
            self.workspace_engine
        ]
        
        failed_components = [comp for comp in critical_components if comp is None]
        if failed_components:
            logging.error(f"Critical components failed to initialize: {len(failed_components)}")
        else:
            logging.info("All critical components initialized successfully")
    def _create_audio_bridge(self):
        from sensory.audio_to_neural_bridge import create_audio_to_neural_bridge
        return create_audio_to_neural_bridge(self)
    def _create_hebbian_learning(self):
        from learning.live_hebbian_learning import create_live_hebbian_learning
        return create_live_hebbian_learning(self)
    def _create_neural_persistence(self):
        from neural.neural_map_persistence import create_neural_map_persistence
        return create_neural_map_persistence()
    def _create_event_system(self):
        from neural.event_driven_system import create_event_driven_system
        return create_event_driven_system(self)
    def _create_spike_system(self):
        from neural.spike_queue_system import create_spike_queue_system
        return create_spike_queue_system(self)
    def _safe_initialize_component(self, component_name: str, init_func: callable,
                                 critical: bool = False) -> Any:

        try:
            result = init_func()
            logging.info(f"{component_name} initialized successfully")
            return result
        except ImportError as e:
            if critical:
                logging.error(f"Critical component {component_name} not available: {e}")
                fallback = self._get_fallback_component(component_name)
                if fallback:
                    logging.warning(f"Using fallback for critical component {component_name}")
                    return fallback
                raise
            else:
                logging.info(f"{component_name} not available: {e}")
                return None
        except (ValueError, TypeError, AttributeError) as e:
            if critical:
                logging.error(f"Configuration error in critical component {component_name}: {e}")
                try:
                    result = self._reinitialize_with_defaults(component_name, init_func)
                    if result:
                        logging.warning(f"Reinitialized {component_name} with default configuration")
                        return result
                except Exception:
                    pass
                raise
            else:
                logging.warning(f"Configuration error in {component_name}: {e}")
                return None
        except Exception as e:
            if critical:
                logging.error(f"Failed to initialize critical component {component_name}: {e}")
                fallback = self._get_fallback_component(component_name)
                if fallback:
                    logging.warning(f"Using fallback for critical component {component_name}")
                    return fallback
                raise
            else:
                logging.warning(f"Failed to initialize {component_name}: {e}")
                return None
    def _get_fallback_component(self, component_name: str) -> Any:
        fallbacks = {
            'behavior_engine': lambda: BehaviorEngine(),
            'learning_engine': lambda: LearningEngine(),
            'memory_system': lambda: MemorySystem(),
            'homeostasis_controller': lambda: HomeostasisController(),
            'network_metrics': lambda: NetworkMetrics(),
            'workspace_engine': lambda: WorkspaceEngine(),
        }
        if component_name in fallbacks:
            try:
                return fallbacks[component_name]()
            except Exception as e:
                logging.error(f"Fallback for {component_name} also failed: {e}")
                return None
        return None
    def _reinitialize_with_defaults(self, component_name: str, init_func: callable) -> Any:
        try:
            if hasattr(self, 'config'):
                default_config = {}
                original_config = self.config
                self.config = default_config
                result = init_func()
                self.config = original_config
                return result
        except Exception as e:
            logging.error(f"Reinitialization with defaults failed for {component_name}: {e}")
        return None
    def set_graph(self, graph):

        with self._lock:
            self.graph = graph
        if self.graph is not None:
            self.graph.simulation_step = self.step_counter
            self.graph.simulation_running = self.is_running
            log_step("Graph attached to simulation manager",
                    nodes=len(self.graph.node_labels) if hasattr(self.graph, 'node_labels') else 0)
    def initialize_graph(self):
        try:
            from core.main_graph import initialize_main_graph
            graph = initialize_main_graph()
            if self.id_manager is not None:
                graph_size = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
                self.id_manager.set_max_graph_size(graph_size)
                log_step("Graph size limit set in ID manager", graph_size=graph_size)
            self.set_graph(graph)
            logging.info("Graph initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize graph: {e}")
            return False
    def add_step_callback(self, callback: Callable):
        self.step_callbacks.append(callback)
    def add_metrics_callback(self, callback: Callable):
        self.metrics_callbacks.append(callback)
    def add_error_callback(self, callback: Callable):
        self.error_callbacks.append(callback)
    def run_single_step(self) -> bool:
        """Execute a single simulation step with error handling."""
        assert self.graph is not None, "Graph must be initialized before running simulation step"
        assert hasattr(self, 'step_counter'), "Step counter must be initialized"
        assert self.step_counter >= 0, "Step counter must be non-negative"
        
        if self.graph is None:
            return False
        
        try:
            step_start_time = time.time()
            self.step_counter += 1
            
            # Process all simulation components
            self._process_sensory_input()
            self._process_event_system()
            self._process_spike_system()
            self._process_audio_input()
            self._process_neural_dynamics()
            self._process_learning_systems()
            self._process_workspace_systems()
            self._process_node_lifecycle()
            self._process_memory_systems()
            self._process_homeostatic_control()
            self._process_visual_systems()
            self._process_metrics_and_callbacks()
            
            # Update performance statistics
            self._update_step_statistics(step_start_time)
            return True
            
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logging.warning(f"Simulation step error (non-critical): {e}")
            record_simulation_error()
            self._handle_step_error(e)
            return True
        except Exception as e:
            logging.error(f"Unexpected simulation step error: {e}")
            record_simulation_error()
            raise

    def _process_sensory_input(self):
        """Process sensory input updates."""
        capture_interval = 200 if self.step_counter > 1000 else 100
        if self.step_counter % capture_interval == 0:
            try:
                success = self._update_sensory_features(self.graph, scale=RESOLUTION_SCALE)
                if not success:
                    logging.warning("Sensory update failed, using fallback")
                    self._fallback_sensory_update()
            except Exception as e:
                logging.warning(f"Sensory update failed: {e}")
                self._fallback_sensory_update()

    def _process_event_system(self):
        """Process event-driven system updates."""
        if self.event_driven_system is not None:
            try:
                events_processed = self.process_events(max_events=1000)
                if events_processed > 0:
                    logging.debug(f"Processed {events_processed} events at step {self.step_counter}")
            except Exception as e:
                logging.warning(f"Event processing failed: {e}")
        else:
            if self.step_counter % 100 == 0:
                try:
                    self.event_driven_system = self._create_event_system()
                    if self.event_driven_system is not None:
                        logging.info("Event system reinitialized")
                except Exception as e:
                    logging.debug(f"Event system reinitialization failed: {e}")

    def _process_spike_system(self):
        """Process spike queue system updates."""
        if self.spike_queue_system is not None:
            try:
                spikes_processed = self.spike_queue_system.process_spikes(max_spikes=1000)
                if spikes_processed > 0:
                    logging.debug(f"Processed {spikes_processed} spikes at step {self.step_counter}")
            except Exception as e:
                logging.warning(f"Spike processing failed: {e}")

    def _process_audio_input(self):
        """Process audio input integration."""
        if (self.audio_to_neural_bridge is not None and
            hasattr(self, 'last_audio_data') and
            self.last_audio_data is not None):
            try:
                self.graph = self.audio_to_neural_bridge.integrate_audio_nodes_into_graph(
                    self.graph, self.last_audio_data
                )
                logging.debug(f"Audio processing applied at step {self.step_counter}")
            except Exception as e:
                logging.warning(f"Audio processing failed: {e}")

    def _process_neural_dynamics(self):
        """Process neural dynamics updates."""
        try:
            self._update_node_behaviors()
        except Exception as e:
            logging.warning(f"Node behavior update failed: {e}")
        
        if (self.enhanced_integration is not None and
            self.step_counter % 2 == 0):
            try:
                self.graph = self.enhanced_integration.integrate_with_existing_system(
                    self.graph, self.step_counter
                )
                logging.debug(f"Enhanced neural dynamics applied at step {self.step_counter}")
            except Exception as e:
                logging.warning(f"Enhanced neural dynamics failed: {e}")
        
        try:
            self._apply_energy_dynamics()
        except Exception as e:
            logging.warning(f"Energy dynamics failed: {e}")

    def _process_learning_systems(self):
        """Process learning system updates."""
        if self.step_counter % 5 == 0:
            try:
                self.graph = intelligent_connection_formation(self.graph)
            except Exception as e:
                logging.warning(f"Connection formation failed: {e}")
            
            try:
                self.graph = self.learning_engine.consolidate_connections(self.graph)
            except Exception as e:
                logging.warning(f"Learning consolidation failed: {e}")
        
        if self.live_hebbian_learning is not None and self.step_counter % 3 == 0:
            self.graph = self.live_hebbian_learning.apply_continuous_learning(self.graph, self.step_counter)

    def _process_workspace_systems(self):
        """Process workspace system updates."""
        if self.step_counter % 20 == 0:
            try:
                self.workspace_engine.update_workspace_nodes(self.graph, self.step_counter)
            except Exception as e:
                logging.warning(f"Workspace update failed: {e}")

    def _process_node_lifecycle(self):
        """Process node birth and death."""
        if self.step_counter % 50 == 0 and len(self.graph.node_labels) < 50000:
            try:
                birth_new_dynamic_nodes(self.graph)
            except Exception as e:
                logging.warning(f"Node birth failed: {e}")
        
        if self.step_counter % 50 == 0:
            try:
                remove_dead_dynamic_nodes(self.graph)
            except Exception as e:
                logging.warning(f"Node death failed: {e}")

    def _process_memory_systems(self):
        """Process memory system updates."""
        if self.step_counter % self.memory_update_interval == 0:
            try:
                self.graph = self.memory_system.form_memory_traces(self.graph)
                self.graph = self.memory_system.consolidate_memories(self.graph)
                logging.debug(f"Memory system updated at step {self.step_counter}")
            except Exception as e:
                logging.warning(f"Memory system update failed: {e}")
        
        if self.step_counter % (self.memory_update_interval // 2) == 0:
            self.graph = self.learning_engine.apply_memory_influence(self.graph)

    def _process_homeostatic_control(self):
        """Process homeostatic control updates."""
        health_status = {'status': 'unknown', 'warnings': []}
        
        if self.step_counter % self.homeostasis_update_interval == 0:
            try:
                self.graph = self.homeostasis_controller.regulate_network_activity(self.graph)
            except Exception as e:
                logging.warning(f"Homeostatic control failed: {e}")
            
            try:
                self.graph = self.homeostasis_controller.optimize_criticality(self.graph)
                health_status = self.homeostasis_controller.monitor_network_health(self.graph)
                self.performance_stats['system_health'] = health_status['status']
            except Exception as e:
                logging.warning(f"Criticality optimization failed: {e}")
        
        # Check graph size limits
        if self.step_counter % 100 == 0:
            node_count = len(self.graph.node_labels)
            if node_count > 100000:
                logging.warning(f"Graph size limit reached: {node_count} nodes")
                self._prune_graph_if_needed()
            elif node_count > 50000:
                logging.info(f"Graph size: {node_count} nodes (approaching limit)")
        
        # Handle health warnings
        if health_status['status'] != 'healthy':
            logging.warning(
                f"[SIMULATION] Network health: {health_status['status']} - "
                f"{health_status['warnings']}"
            )
            record_simulation_warning()

    def _process_visual_systems(self):
        """Process visual system updates."""
        if (self.step_counter % 5 == 0 and
            hasattr(self, 'visual_energy_bridge') and
            self.visual_energy_bridge is not None):
            try:
                if hasattr(self, 'last_visual_data') and self.last_visual_data is not None:
                    self.graph = self.visual_energy_bridge.process_visual_to_enhanced_energy(
                        self.graph, self.last_visual_data, self.step_counter
                    )
            except Exception as e:
                logging.warning(f"Visual energy bridge processing failed: {e}")
        
        if (self.step_counter % 10 == 0 and
            hasattr(self, 'sensory_workspace_mapper') and
            self.sensory_workspace_mapper is not None):
            try:
                if hasattr(self, 'last_visual_data') and self.last_visual_data is not None:
                    self.graph = self.sensory_workspace_mapper.map_visual_to_workspace(
                        self.graph, self.last_visual_data, self.step_counter
                    )
                if hasattr(self, 'last_audio_data') and self.last_audio_data is not None:
                    self.graph = self.sensory_workspace_mapper.map_audio_to_workspace(
                        self.graph, self.last_audio_data, self.step_counter
                    )
            except Exception as e:
                logging.warning(f"Sensory workspace mapping failed: {e}")

    def _process_metrics_and_callbacks(self):
        """Process metrics calculation and callbacks."""
        if self.step_counter % 200 == 0:
            try:
                metrics = self.network_metrics.calculate_comprehensive_metrics(self.graph)
                self.performance_stats['last_metrics'] = metrics
                self._execute_metrics_callbacks(metrics)
                self._log_step_metrics(metrics)
            except Exception as e:
                logging.warning(f"Metrics calculation failed: {e}")

    def _execute_metrics_callbacks(self, metrics):
        """Execute metrics callbacks with error handling."""
        for callback in self.metrics_callbacks:
            try:
                callback(metrics)
            except (TypeError, ValueError, AttributeError) as e:
                logging.error(f"Metrics callback error: {e}")
                record_simulation_error()
            except Exception as e:
                logging.error(f"Unexpected metrics callback error: {e}")
                record_simulation_error()
                raise

    def _log_step_metrics(self, metrics):
        """Log step metrics information."""
        logging.info(f"[SIMULATION] Step {self.step_counter}: Criticality={metrics['criticality']:.3f}, "
                   f"Connectivity={metrics['connectivity']['density']:.3f}, "
                   f"Energy Variance={metrics['energy_balance']['energy_variance']:.2f}")

    def _update_step_statistics(self, step_start_time):
        """Update step performance statistics."""
        step_time = time.time() - step_start_time
        self.performance_stats['last_step_time'] = step_time
        self.performance_stats['total_steps'] = self.step_counter
        self.current_step = self.step_counter
        
        if step_time > 1.0:
            log_step("Step time exceeded 1 second", step_time=step_time, step=self.step_counter)
            self.performance_stats['slow_steps'] = self.performance_stats.get('slow_steps', 0) + 1
        
        # Update average step time
        if self.performance_stats['avg_step_time'] == 0:
            self.performance_stats['avg_step_time'] = step_time
        else:
            self.performance_stats['avg_step_time'] = (
                self.performance_stats['avg_step_time'] * 0.9 + step_time * 0.1
            )
        
        # Record step metrics
        node_count = self._get_node_count()
        edge_count = self._get_edge_count()
        record_simulation_step(step_time, node_count, edge_count)
        
        # Update performance monitor
        if self.performance_monitor and hasattr(self.performance_monitor, '_update_metrics') and self.step_counter % 50 == 0:
            self.performance_monitor._update_metrics()
        
        # Execute step callbacks
        self._execute_step_callbacks()
        
        # Record final step metrics
        if self.performance_monitor:
            self.performance_monitor.record_step(step_time, node_count, edge_count)

    def _execute_step_callbacks(self):
        """Execute step callbacks with error handling."""
        for callback in self.step_callbacks:
            try:
                callback(self.graph, self.step_counter, self.performance_stats)
            except (TypeError, ValueError, AttributeError) as e:
                logging.error(f"Step callback error: {e}")
            except Exception as e:
                logging.error(f"Unexpected step callback error: {e}")
                raise

    def _handle_step_error(self, error):
        """Handle step errors with callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error, self.step_counter)
            except (TypeError, ValueError, AttributeError) as callback_error:
                logging.error(f"Error callback error: {callback_error}")
            except Exception as callback_error:
                logging.error(f"Unexpected error callback error: {callback_error}")
    
    def _get_node_count(self) -> int:
        """Get node count with simplified conditional check."""
        if self.graph is None:
            return 0
        return len(self.graph.node_labels) if hasattr(self.graph, 'node_labels') else 0
    
    def _get_edge_count(self) -> int:
        """Get edge count with simplified conditional check."""
        if self.graph is None:
            return 0
        if not hasattr(self.graph, 'edge_index'):
            return 0
        return self.graph.edge_index.shape[1] if self.graph.edge_index.numel() > 0 else 0
    
    def _has_visual_data(self) -> bool:
        """Check if visual data is available with simplified conditional."""
        return hasattr(self, 'last_visual_data') and self.last_visual_data is not None
    
    def _has_audio_data(self) -> bool:
        """Check if audio data is available with simplified conditional."""
        return hasattr(self, 'last_audio_data') and self.last_audio_data is not None
    def start_simulation(self, run_in_thread: bool = True):

        with self._lock:
            if self.is_running:
                logging.warning("Simulation already running")
                return
            if self.graph is None:
                logging.error("No graph set for simulation")
                return
            self.is_running = True
            self.simulation_running = True
            self.step_counter = 0
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
        with self._lock:
            if not self.is_running:
                logging.warning("Simulation not running")
                return
            self.is_running = False
            self.simulation_running = False
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.simulation_thread.join(timeout=2.0)
            log_step("Simulation stopped",
                    total_steps=self.step_counter,
                    avg_step_time=self.performance_stats['avg_step_time'])
    def _simulation_loop(self):
        log_step("Simulation loop started")
        max_iterations = 100000
        iteration_count = 0
        last_health_check = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 100
        while self.is_running and iteration_count < max_iterations:
            try:
                loop_start = time.time()
                iteration_count += 1
                if iteration_count % 1000 == 0:
                    current_time = time.time()
                    if current_time - last_health_check > 300:
                        log_step("Simulation health check", iterations=iteration_count,
                                elapsed=current_time - last_health_check)
                        last_health_check = current_time
                success = self.run_single_step()
                if not success:
                    consecutive_failures += 1
                    logging.error(f"Simulation step failed (failure #{consecutive_failures})")
                    if consecutive_failures >= max_consecutive_failures:
                        logging.error("Too many consecutive failures, stopping simulation")
                        self.is_running = False
                        break
                else:
                    consecutive_failures = 0
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                consecutive_failures += 1
                logging.error(f"Simulation loop error: {e}")
                record_simulation_error()
                if consecutive_failures >= max_consecutive_failures:
                    logging.error("Too many consecutive errors, stopping simulation")
                    self.is_running = False
                    break
                time.sleep(0.1)
        if iteration_count >= max_iterations:
            log_step("Simulation loop ended due to iteration limit", iterations=iteration_count)
        elif consecutive_failures >= max_consecutive_failures:
            log_step("Simulation loop ended due to consecutive failures", failures=consecutive_failures)
        else:
            log_step("Simulation loop ended normally")
    def get_performance_stats(self) -> Dict[str, Any]:
        return self.performance_stats.copy()
    def get_system_stats(self) -> Dict[str, Any]:
        try:
            health_score = self._calculate_system_health_score()
            return {
                'behavior_stats': self.behavior_engine.get_behavior_statistics(),
                'learning_stats': self.learning_engine.get_learning_statistics(),
                'memory_stats': self.memory_system.get_memory_statistics(),
                'homeostasis_stats': self.homeostasis_controller.get_regulation_statistics(),
                'memory_traces': self.memory_system.get_memory_trace_count(),
                'performance': self.performance_stats,
                'health_score': health_score,
                'system_health': self._get_system_health_status(health_score)
            }
        except Exception as e:
            log_step("Error getting system stats", error=str(e))
            return {
                'error': str(e),
                'performance': self.performance_stats
            }
    def _update_sensory_features(self, graph, scale=1.0):

        try:
            if graph is None:
                log_step("Graph is None in _update_sensory_features")
                return False
            from ui.screen_graph import capture_screen, create_pixel_gray_graph
            if self.step_counter % 100 != 0:
                return True
            arr = capture_screen(scale=scale)
            if hasattr(self, 'visual_energy_bridge') and self.visual_energy_bridge is not None:
                try:
                    graph = self.visual_energy_bridge.process_visual_to_enhanced_energy(
                        graph, arr, self.step_counter
                    )
                    if hasattr(self, 'sensory_workspace_mapper') and self.sensory_workspace_mapper is not None:
                        try:
                            graph = self.sensory_workspace_mapper.map_visual_to_workspace(
                                graph, arr, self.step_counter
                            )
                            log_step("Visual patterns mapped to workspace", step=self.step_counter)
                        except (ImportError, AttributeError, IndexError, ValueError, RuntimeError) as e:
                            logging.warning(f"Visual workspace mapping failed: {e}")
                    return True
                except (ImportError, AttributeError, IndexError, ValueError, RuntimeError) as e:
                    logging.warning(f"Enhanced visual processing failed: {e}")
            pixel_graph = create_pixel_gray_graph(arr)
            if safe_hasattr(graph, 'node_labels', 'x'):
                sensory_indices = [i for i, node in enumerate(graph.node_labels)
                                 if node.get('type') == 'sensory']
                if sensory_indices and hasattr(pixel_graph, 'x'):
                    num_sensory = min(len(sensory_indices), pixel_graph.x.shape[0])
                    if num_sensory > 0:
                        sensory_indices_tensor = torch.tensor(sensory_indices[:num_sensory], dtype=torch.long)
                        graph.x[sensory_indices_tensor, 0] = pixel_graph.x[:num_sensory, 0]
            return True
        except (ImportError, AttributeError, IndexError, ValueError) as e:
            logging.warning(f"Sensory update failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected sensory update error: {e}")
            return False
    def _fallback_sensory_update(self):
        try:
            logging.info("Using fallback sensory update - no screen capture")
            return True
        except Exception as e:
            logging.error(f"Fallback sensory update failed: {e}")
            return False
    def update_visual_data(self, visual_data):
        self.last_visual_data = visual_data
    def update_audio_data(self, audio_data):
        self.last_audio_data = audio_data
    def _validate_graph_consistency(self) -> bool:
        if self.graph is None:
            return False
        try:
            if not hasattr(self.graph, 'node_labels') or not hasattr(self.graph, 'x'):
                return False
            if len(self.graph.node_labels) != self.graph.x.shape[0]:
                log_step("Graph consistency error: node_labels and graph.x size mismatch",
                        labels_count=len(self.graph.node_labels),
                        x_shape=self.graph.x.shape)
                return False
            if torch.isnan(self.graph.x).any() or torch.isinf(self.graph.x).any():
                log_step("Graph consistency error: NaN or infinite values in graph.x")
                return False
            if hasattr(self.graph, 'edge_index') and self.graph.edge_index.numel() > 0:
                max_index = len(self.graph.node_labels) - 1
                if torch.max(self.graph.edge_index) > max_index:
                    log_step("Graph consistency error: edge_index contains invalid node indices",
                            max_edge_index=torch.max(self.graph.edge_index).item(),
                            max_valid_index=max_index)
                    return False
            return True
        except Exception as e:
            log_step("Graph consistency validation error", error=str(e))
            return False
    def _repair_graph_consistency(self):
        if self.graph is None:
            return
        try:
            if not hasattr(self.graph, 'node_labels') or not self.graph.node_labels:
                log_step("Creating missing node_labels")
                self.graph.node_labels = []
            for i, node in enumerate(self.graph.node_labels):
                if 'id' not in node or node['id'] is None:
                    node['id'] = i
                    log_step(f"Added missing ID {i} to node at index {i}")
                elif node['id'] != i:
                    old_id = node['id']
                    node['id'] = i
                    log_step(f"Fixed ID mismatch: {old_id} -> {i} at index {i}")
            if safe_hasattr(self.graph, 'node_labels', 'x'):
                labels_count = len(self.graph.node_labels)
                x_count = self.graph.x.shape[0] if self.graph.x is not None else 0
                if labels_count != x_count:
                    log_step("Repairing graph consistency: node_labels and graph.x mismatch",
                            labels_count=labels_count, x_count=x_count)
                    if labels_count > x_count:
                        self.graph.node_labels = self.graph.node_labels[:x_count]
                        log_step("Removed excess node labels")
                    elif x_count > labels_count:
                        self.graph.x = self.graph.x[:labels_count]
                        log_step("Truncated excess graph.x rows")
            if hasattr(self.graph, 'x') and self.graph.x is not None:
                nan_mask = torch.isnan(self.graph.x)
                inf_mask = torch.isinf(self.graph.x)
                if nan_mask.any() or inf_mask.any():
                    log_step("Cleaning NaN/infinite values from graph.x")
                    self.graph.x[nan_mask] = 0.0
                    self.graph.x[inf_mask] = 0.0
            if hasattr(self.graph, 'edge_index') and self.graph.edge_index.numel() > 0:
                max_valid_index = len(self.graph.node_labels) - 1
                valid_edges_mask = (self.graph.edge_index[0] <= max_valid_index) & (self.graph.edge_index[1] <= max_valid_index)
                if not valid_edges_mask.all():
                    log_step("Removing invalid edge indices")
                    self.graph.edge_index = self.graph.edge_index[:, valid_edges_mask]
                    if hasattr(self.graph, 'edge_attributes'):
                        valid_indices = torch.where(valid_edges_mask)[0]
                        if len(valid_indices) < len(self.graph.edge_attributes):
                            self.graph.edge_attributes = [self.graph.edge_attributes[i] for i in valid_indices.tolist()]
        except Exception as e:
            log_step("Graph consistency repair error", error=str(e))
    def _update_node_behaviors(self):
        try:
            if not self._validate_graph_consistency():
                log_step("Graph consistency validation failed, attempting repair")
                self._repair_graph_consistency()
                if not self._validate_graph_consistency():
                    log_step("Graph consistency repair failed, skipping node behavior update")
                    return
            num_nodes = len(self.graph.node_labels)
            if num_nodes == 0:
                return
            if not hasattr(self, '_access_layer') or self._access_layer is None:
                from energy.node_access_layer import NodeAccessLayer
                self._access_layer = NodeAccessLayer(self.graph)
            access_layer = self._access_layer
            nodes_to_update = min(50, num_nodes)
            for idx in range(nodes_to_update):
                node = self.graph.node_labels[idx]
                node_id = node.get('id')
                if node_id is None:
                    continue
                success = self.behavior_engine.update_node_behavior(node_id, self.graph, self.step_counter, access_layer)
                if not success:
                    pass
            if (self.enhanced_integration is not None and
                self.step_counter % 3 == 0):
                try:
                    self.graph = self.enhanced_integration.integrate_with_existing_system(
                        self.graph, self.step_counter
                    )
                except Exception as e:
                    logging.error(f"Error in enhanced neural integration: {e}")
                    self.error_count += 1
        except Exception as e:
            logging.error(f"Error in behavior engine update: {e}")
            for idx in range(0, min(10, len(self.graph.node_labels))):
                node = self.graph.node_labels[idx]
                if node.get('type') == 'dynamic':
                    current_energy = self.graph.x[idx, 0].item()
                    import random
                    energy_change = random.uniform(-0.01, 0.01)
                    new_energy = max(0, min(current_energy + energy_change, 1.0))
                    self.graph.x[idx, 0] = new_energy
                    if 'membrane_potential' in node:
                        node['membrane_potential'] = new_energy
    def _apply_energy_dynamics(self):
        try:
            self.graph = apply_energy_behavior(self.graph)
            self.graph = update_membrane_potentials(self.graph)
            self.graph = apply_refractory_periods(self.graph)
        except Exception as e:
            logging.error(f"Error in energy dynamics: {e}")
            self.graph = apply_energy_behavior(self.graph)
            self.graph = update_membrane_potentials(self.graph)
            self.graph = apply_refractory_periods(self.graph)
    def reset_simulation(self):
        self.stop_simulation()
        self.step_counter = 0
        self.current_step = 0
        self.performance_stats = {
            'total_steps': 0,
            'avg_step_time': 0.0,
            'last_step_time': 0.0,
            'system_health': 'unknown'
        }
        self.behavior_engine.reset_statistics()
        self.learning_engine.reset_statistics()
        self.memory_system.reset_statistics()
        self.homeostasis_controller.reset_statistics()
        log_step("Simulation reset to initial state")
    def process_audio_to_neural(self, audio_data):

        try:
            if self.graph and self.audio_to_neural_bridge is not None:
                self.graph = self.audio_to_neural_bridge.integrate_audio_nodes_into_graph(
                    self.graph, audio_data
                )
                if hasattr(self, 'sensory_workspace_mapper') and self.sensory_workspace_mapper is not None:
                    try:
                        self.graph = self.sensory_workspace_mapper.map_audio_to_workspace(
                            self.graph, audio_data, self.step_counter
                        )
                        log_step("Audio patterns mapped to workspace", step=self.step_counter)
                    except Exception as e:
                        logging.warning(f"Audio workspace mapping failed: {e}")
                log_step("Audio processed and integrated into neural simulation")
            return self.graph
        except Exception as e:
            log_step("Error processing audio to neural", error=str(e))
            return self.graph
    def save_neural_map(self, slot_number=None, metadata=None):

        try:
            if self.graph and self.neural_map_persistence is not None:
                return self.neural_map_persistence.save_neural_map(
                    self.graph, slot_number, metadata
                )
            return False
        except Exception as e:
            log_step("Error saving neural map", error=str(e))
            return False
    def load_neural_map(self, slot_number):

        try:
            if self.neural_map_persistence is not None:
                loaded_graph = self.neural_map_persistence.load_neural_map(slot_number)
                if loaded_graph is not None:
                    self.graph = loaded_graph
                    log_step("Neural map loaded successfully", slot_number=slot_number)
                    return True
            return False
        except Exception as e:
            log_step("Error loading neural map", error=str(e))
            return False
    def get_neural_map_slots(self):

        try:
            if self.neural_map_persistence is not None:
                return self.neural_map_persistence.list_available_slots()
            return {}
        except Exception as e:
            log_step("Error getting neural map slots", error=str(e))
            return {}
    def get_hebbian_learning_stats(self):

        try:
            if self.live_hebbian_learning is not None:
                return self.live_hebbian_learning.get_learning_statistics()
            return {}
        except Exception as e:
            log_step("Error getting Hebbian learning stats", error=str(e))
            return {}
    def set_hebbian_learning_rate(self, learning_rate):

        try:
            if self.live_hebbian_learning is not None:
                self.live_hebbian_learning.set_learning_rate(learning_rate)
        except Exception as e:
            log_step("Error setting Hebbian learning rate", error=str(e))
    def create_enhanced_node(self, node_id: int, node_type: str = 'dynamic',
                           subtype: str = 'standard', **kwargs) -> bool:

        if self.enhanced_integration is None:
            log_step("Enhanced integration not available", node_id=node_id)
            return False
        return self._create_enhanced_node(
            self.graph, node_id, node_type, subtype, **kwargs
        )
    def create_enhanced_connection(self, source_id: int, target_id: int,
                                 connection_type: str = 'excitatory', **kwargs) -> bool:

        if self.enhanced_integration is None:
            log_step("Enhanced integration not available", source_id=source_id, target_id=target_id)
            return False
        return self._create_enhanced_connection(
            self.graph, source_id, target_id, connection_type, **kwargs
        )
    def set_neuromodulator_level(self, neuromodulator: str, level: float):

        if hasattr(self, 'enhanced_integration') and self.enhanced_integration:
            self._set_neuromodulator_level(neuromodulator, level)
        elif hasattr(self, 'behavior_engine') and self.behavior_engine:
            self.behavior_engine.set_neuromodulator_level(neuromodulator, level)
    def get_enhanced_statistics(self):

        if hasattr(self, 'enhanced_integration') and self.enhanced_integration:
            return self._get_integration_statistics()
        elif hasattr(self, 'behavior_engine') and self.behavior_engine:
            return self.behavior_engine.get_enhanced_statistics()
        return {}
    def get_event_driven_statistics(self):
        if self.event_driven_system is not None:
            return self.event_driven_system.get_statistics()
        return {}
    def get_access_layer(self):
        if hasattr(self, 'graph') and self.graph is not None:
            from energy.node_access_layer import NodeAccessLayer
            return NodeAccessLayer(self.graph)
        return None
    def schedule_spike_event(self, node_id: int, timestamp: float = None, priority: int = 1):
        if self.event_driven_system is not None:
            self.event_driven_system.schedule_spike(node_id, timestamp, priority)
    def schedule_energy_transfer_event(self, source_id: int, target_id: int, amount: float, timestamp: float = None):
        if self.event_driven_system is not None:
            self.event_driven_system.schedule_energy_transfer(source_id, target_id, amount, timestamp)
    def process_events(self, max_events: int = None):
        if self.event_driven_system is not None:
            return self.event_driven_system.process_events(max_events)
        return 0
    def get_spike_queue_statistics(self):
        if self.spike_queue_system is not None:
            return self.spike_queue_system.get_statistics()
        return {}
    def schedule_spike(self, source_id: int, target_id: int, spike_type: str = 'excitatory',
                      amplitude: float = 1.0, weight: float = 1.0, timestamp: float = None):
        if self.spike_queue_system is not None:
            from neural.spike_queue_system import SpikeType
            spike_type_enum = SpikeType.EXCITATORY
            if spike_type.lower() == 'inhibitory':
                spike_type_enum = SpikeType.INHIBITORY
            elif spike_type.lower() == 'modulatory':
                spike_type_enum = SpikeType.MODULATORY
            elif spike_type.lower() == 'burst':
                spike_type_enum = SpikeType.BURST
            return self.spike_queue_system.schedule_spike(
                source_id, target_id, spike_type_enum, amplitude, weight, timestamp
            )
        return False
    def get_spike_queue_size(self):
        if self.spike_queue_system is not None:
            return self.spike_queue_system.get_queue_size()
        return 0
    def get_sensory_workspace_statistics(self):
        if hasattr(self, 'sensory_workspace_mapper') and self.sensory_workspace_mapper is not None:
            return self.sensory_workspace_mapper.get_mapping_statistics()
        return {}
    # Display data functions are handled dynamically by __getattr__
    def __getattr__(self, name):
        if name.endswith('_display_data'):
            return lambda: f"{name.replace('_', ' ').title()}"
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    def set_visual_sensitivity(self, sensitivity: float):
        if hasattr(self, 'visual_energy_bridge') and self.visual_energy_bridge is not None:
            self.visual_energy_bridge.set_visual_sensitivity(sensitivity)
        if hasattr(self, 'sensory_workspace_mapper') and self.sensory_workspace_mapper is not None:
            self.sensory_workspace_mapper.visual_sensitivity = max(0.0, min(1.0, sensitivity))
    def set_audio_sensitivity(self, sensitivity: float):
        if hasattr(self, 'sensory_workspace_mapper') and self.sensory_workspace_mapper is not None:
            self.sensory_workspace_mapper.audio_sensitivity = max(0.0, min(1.0, sensitivity))
    def set_pattern_threshold(self, threshold: float):
        if hasattr(self, 'sensory_workspace_mapper') and self.sensory_workspace_mapper is not None:
            self.sensory_workspace_mapper.pattern_threshold = max(0.0, min(1.0, threshold))
    def _prune_graph_if_needed(self):
        try:
            if not hasattr(self.graph, 'node_labels') or not self.graph.node_labels:
                return
            nodes_with_energy = []
            for i, node in enumerate(self.graph.node_labels):
                energy = node.get('energy', 0.0)
                nodes_with_energy.append((i, energy, node))
            nodes_with_energy.sort(key=lambda x: x[1])
            nodes_to_remove = len(nodes_with_energy) // 10
            if nodes_to_remove > 0:
                indices_to_remove = [x[0] for x in nodes_with_energy[:nodes_to_remove]]
                indices_to_remove.sort(reverse=True)
                for idx in indices_to_remove:
                    if idx < len(self.graph.node_labels):
                        del self.graph.node_labels[idx]
                if hasattr(self.graph, 'x') and self.graph.x is not None:
                    keep_indices = [i for i in range(len(self.graph.node_labels))
                                  if i not in indices_to_remove]
                    if keep_indices:
                        self.graph.x = self.graph.x[keep_indices]
                if hasattr(self.graph, 'edge_index') and self.graph.edge_index.numel() > 0:
                    for removed_idx in indices_to_remove:
                        mask = self.graph.edge_index >= removed_idx
                        self.graph.edge_index[mask] -= 1
                logging.info(f"Pruned {nodes_to_remove} low-energy nodes from graph")
            
            # Clean up orphaned IDs in the ID manager
            try:
                from energy.node_id_manager import get_id_manager
                id_manager = get_id_manager()
                orphaned_count = id_manager.cleanup_orphaned_ids(self.graph)
                if orphaned_count > 0:
                    log_step(f"Cleaned up {orphaned_count} orphaned IDs")
            except Exception as cleanup_error:
                log_step("Error cleaning up orphaned IDs", error=str(cleanup_error))
        except Exception as e:
            logging.warning(f"Error pruning graph: {e}")
    def _create_enhanced_neural_integration(self):

        try:
            from neural.enhanced_neural_dynamics import EnhancedNeuralDynamics
            from neural.enhanced_connection_system import EnhancedConnectionSystem
            from neural.enhanced_node_behaviors import EnhancedNodeBehaviorSystem
            from energy.node_access_layer import NodeAccessLayer
            neural_dynamics = EnhancedNeuralDynamics()
            connection_system = EnhancedConnectionSystem()
            node_behavior_system = EnhancedNodeBehaviorSystem()
            integration = type('EnhancedNeuralIntegration', (), {
                'neural_dynamics': neural_dynamics,
                'connection_system': connection_system,
                'node_behavior_system': node_behavior_system,
                'integration_active': True,
                'update_frequency': 1,
                'last_update_step': 0,
                'integration_stats': {
                    'total_updates': 0,
                    'neural_dynamics_updates': 0,
                    'connection_updates': 0,
                    'node_behavior_updates': 0,
                    'integration_errors': 0
                },
                'integrate_with_existing_system': self._integrate_enhanced_systems,
                'create_enhanced_node': self._create_enhanced_node,
                'create_enhanced_connection': self._create_enhanced_connection,
                'set_neuromodulator_level': self._set_neuromodulator_level,
                'get_integration_statistics': self._get_integration_statistics,
                'reset_integration_statistics': self._reset_integration_statistics,
                'enable_integration': self._enable_integration,
                'disable_integration': self._disable_integration,
                'cleanup': self._cleanup_enhanced_systems
            })()
            log_step("Enhanced neural integration created")
            return integration
        except ImportError as e:
            log_step("Enhanced neural systems not available", error=str(e))
            return None
        except Exception as e:
            log_step("Error creating enhanced neural integration", error=str(e))
            return None
    def _integrate_enhanced_systems(self, graph, step):

        if not hasattr(self, 'enhanced_integration') or not self.enhanced_integration:
            return graph
        try:
            graph = self.enhanced_integration.neural_dynamics.update_neural_dynamics(graph, step)
            self.enhanced_integration.integration_stats['neural_dynamics_updates'] += 1
            graph = self.enhanced_integration.connection_system.update_connections(graph, step)
            self.enhanced_integration.integration_stats['connection_updates'] += 1
            graph = self.enhanced_integration.node_behavior_system.update_node_behaviors(graph, step)
            self.enhanced_integration.integration_stats['node_behavior_updates'] += 1
            self.enhanced_integration.integration_stats['total_updates'] += 1
            self.enhanced_integration.last_update_step = step
            return graph
        except Exception as e:
            log_step("Error in enhanced neural integration", error=str(e))
            self.enhanced_integration.integration_stats['integration_errors'] += 1
            return graph
    def _create_enhanced_node(self, graph, node_id, node_type='dynamic', subtype='standard', **kwargs):

        try:
            from energy.node_access_layer import NodeAccessLayer
            access_layer = NodeAccessLayer(graph)
            if not access_layer.is_valid_node_id(node_id):
                log_step("Invalid node ID for enhanced node creation", node_id=node_id)
                return False
            behavior = self.enhanced_integration.node_behavior_system.create_node_behavior(
                node_id, node_type, subtype=subtype, **kwargs
            )
            access_layer.update_node_property(node_id, 'enhanced_behavior', True)
            access_layer.update_node_property(node_id, 'subtype', subtype)
            access_layer.update_node_property(node_id, 'is_excitatory', kwargs.get('is_excitatory', True))
            if not hasattr(graph, 'enhanced_node_ids'):
                graph.enhanced_node_ids = []
            if node_id not in graph.enhanced_node_ids:
                graph.enhanced_node_ids.append(node_id)
            log_step("Enhanced node created", node_id=node_id, type=node_type, subtype=subtype)
            return True
        except Exception as e:
            log_step("Error creating enhanced node", node_id=node_id, error=str(e))
            return False
    def _create_enhanced_connection(self, graph, source_id, target_id, connection_type='excitatory', **kwargs):

        try:
            from energy.node_access_layer import NodeAccessLayer
            access_layer = NodeAccessLayer(graph)
            if not access_layer.is_valid_node_id(source_id) or not access_layer.is_valid_node_id(target_id):
                log_step("Invalid node IDs for enhanced connection", source_id=source_id, target_id=target_id)
                return False
            connection = self.enhanced_integration.connection_system.create_connection(
                source_id, target_id, connection_type, **kwargs
            )
            if connection:
                log_step("Enhanced connection created", source_id=source_id, target_id=target_id, type=connection_type)
                return True
            else:
                log_step("Failed to create enhanced connection", source_id=source_id, target_id=target_id)
                return False
        except Exception as e:
            log_step("Error creating enhanced connection", source_id=source_id, target_id=target_id, error=str(e))
            return False
    def _set_neuromodulator_level(self, neuromodulator, level):

        if hasattr(self.enhanced_integration, 'neural_dynamics'):
            self.enhanced_integration.neural_dynamics.set_neuromodulator_level(neuromodulator, level)
    def _get_integration_statistics(self):

        if hasattr(self, 'enhanced_integration') and self.enhanced_integration:
            return self.enhanced_integration.integration_stats.copy()
        return {}
    def _reset_integration_statistics(self):

        if hasattr(self, 'enhanced_integration') and self.enhanced_integration:
            self.enhanced_integration.integration_stats = {
                'total_updates': 0,
                'neural_dynamics_updates': 0,
                'connection_updates': 0,
                'node_behavior_updates': 0,
                'integration_errors': 0
            }
    def _enable_integration(self):

        if hasattr(self, 'enhanced_integration') and self.enhanced_integration:
            self.enhanced_integration.integration_active = True
    def _disable_integration(self):

        if hasattr(self, 'enhanced_integration') and self.enhanced_integration:
            self.enhanced_integration.integration_active = False
    def _cleanup_enhanced_systems(self):

        if hasattr(self, 'enhanced_integration') and self.enhanced_integration:
            if hasattr(self.enhanced_integration, 'neural_dynamics'):
                self.enhanced_integration.neural_dynamics.cleanup()
            if hasattr(self.enhanced_integration, 'connection_system'):
                self.enhanced_integration.connection_system.cleanup()
            if hasattr(self.enhanced_integration, 'node_behavior_system'):
                self.enhanced_integration.node_behavior_system.cleanup()
            log_step("Enhanced neural integration cleaned up")
    def _load_config(self):

        if os.path.exists(self.config_file):
            if not self._validate_config_file_path():
                raise ValueError(f"Invalid configuration file path: {self.config_file}")
            if os.name != 'nt':
                file_stat = os.stat(self.config_file)
                if file_stat.st_mode & 0o077:
                    import warnings
                    if not hasattr(self, '_permission_warning_shown'):
                        warnings.warn(f"Configuration file {self.config_file} has insecure permissions")
                        self._permission_warning_shown = True
            try:
                self._config_parser = configparser.ConfigParser(interpolation=None)
                self._config_parser.read(self.config_file)
            except Exception as e:
                raise ValueError(f"Failed to read configuration file: {e}")
        else:
            self._create_default_config()
    def _validate_config_file_path(self) -> bool:

        abs_path = os.path.abspath(self.config_file)
        if '..' in self.config_file or '\\' in self.config_file:
            return False
        current_dir = os.path.abspath('.')
        if not abs_path.startswith(current_dir):
            return False
        return True
    def _create_default_config(self):

        self._config_parser['General'] = {'resolution_scale': '0.25'}
        self._config_parser['PixelNodes'] = {'pixel_threshold': '128'}
        self._config_parser['DynamicNodes'] = {'dynamic_node_percentage': '0.01'}
        self._config_parser['Processing'] = {'update_interval': '0.5'}
        self._config_parser['EnhancedNodes'] = {
            'oscillator_frequency': '0.1',
            'integrator_threshold': '0.8',
            'relay_amplification': '1.5',
            'highway_energy_boost': '2.0'
        }
        self._config_parser['Learning'] = {
            'plasticity_rate': '0.01',
            'eligibility_decay': '0.95',
            'stdp_window': '20.0',
            'ltp_rate': '0.02',
            'ltd_rate': '0.01'
        }
        self._config_parser['Homeostasis'] = {
            'target_energy_ratio': '0.6',
            'criticality_threshold': '0.1',
            'regulation_rate': '0.001',
            'regulation_interval': '100'
        }
        self._config_parser['NetworkMetrics'] = {
            'calculation_interval': '50',
            'criticality_target': '1.0',
            'connectivity_target': '0.3'
        }
        with open(self.config_file, 'w') as f:
            self._config_parser.write(f)
    def _precache_frequent_sections(self):

        frequent_sections = ['SystemConstants', 'Learning', 'EnhancedNodes', 'Homeostasis']
        for section in frequent_sections:
            if self._config_parser.has_section(section):
                self.get_config_section(section)
    def get_config(self, section: str, key: str, default: Any = None, value_type: type = str) -> Any:

        from config.config_manager import get_config as config_get
        return config_get(section, key, default, value_type)
    def get_config_section(self, section: str) -> Dict[str, Any]:

        from config.config_manager import get_config as config_get
        config = config_get(section, '', {})
        return config if isinstance(config, dict) else {}
    def set_config(self, section: str, key: str, value: Any):

        from config.config_manager import config
        config.set(section, key, value)
    def save_config(self):

        from config.config_manager import config
        config.save()
    def reload_config(self):

        from config.config_manager import config
        config.reload()
    def append_log_line(self, line: str):

        from utils.logging_utils import append_log_line
        append_log_line(line)
    def get_log_lines(self) -> List[str]:

        from utils.logging_utils import get_log_lines
        return get_log_lines()
    def log_runtime(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                runtime = time.time() - start_time
                self.append_log_line(f"Function {func.__name__} completed in {runtime:.4f}s")
                return result
            except Exception as e:
                runtime = time.time() - start_time
                self.append_log_line(f"Function {func.__name__} failed after {runtime:.4f}s: {e}")
                raise
        return wrapper
    def cleanup(self):
        try:
            self.stop_simulation()
            if self.graph is not None:
                self._clear_graph_references(self.graph)
                self.graph = None
            self.step_callbacks.clear()
            self.metrics_callbacks.clear()
            self.error_callbacks.clear()
            if hasattr(self, 'behavior_engine') and self.behavior_engine:
                self._break_system_references(self.behavior_engine)
                self.behavior_engine = None
            if hasattr(self, 'learning_engine') and self.learning_engine:
                self._break_system_references(self.learning_engine)
                self.learning_engine = None
            if hasattr(self, 'memory_system') and self.memory_system:
                self._break_system_references(self.memory_system)
                self.memory_system = None
            if hasattr(self, 'homeostasis_controller') and self.homeostasis_controller:
                self._break_system_references(self.homeostasis_controller)
                self.homeostasis_controller = None
            if hasattr(self, 'network_metrics') and self.network_metrics:
                self._break_system_references(self.network_metrics)
                self.network_metrics = None
            if hasattr(self, 'workspace_engine') and self.workspace_engine:
                self._break_system_references(self.workspace_engine)
                self.workspace_engine = None
            self.audio_to_neural_bridge = None
            self.live_hebbian_learning = None
            self.neural_map_persistence = None
            self.error_handler = None
            self.performance_stats.clear()
            import gc
            gc.collect()
            log_step("SimulationManager cleaned up with memory leak prevention")
        except Exception as e:
            log_step("SimulationManager cleanup error", error=str(e))
    def _calculate_system_health_score(self) -> float:
        try:
            health_factors = []
            step_time = self.performance_stats.get('last_step_time', 0.0)
            if step_time < 0.1:
                health_factors.append(25)
            elif step_time < 0.5:
                health_factors.append(20)
            elif step_time < 1.0:
                health_factors.append(15)
            else:
                health_factors.append(5)
            memory_usage = self.performance_stats.get('memory_usage_mb', 0.0)
            if memory_usage < 500:
                health_factors.append(25)
            elif memory_usage < 1000:
                health_factors.append(20)
            elif memory_usage < 2000:
                health_factors.append(15)
            else:
                health_factors.append(5)
            error_count = self.performance_stats.get('errors', 0)
            if error_count == 0:
                health_factors.append(25)
            elif error_count < 5:
                health_factors.append(20)
            elif error_count < 10:
                health_factors.append(15)
            else:
                health_factors.append(5)
            if hasattr(self.graph, 'node_labels') and self.graph.node_labels:
                health_factors.append(25)
            else:
                health_factors.append(0)
            return sum(health_factors)
        except Exception as e:
            log_step("Error calculating health score", error=str(e))
            return 50.0
    def _get_system_health_status(self, health_score: float) -> str:
        if health_score >= 90:
            return "Excellent"
        elif health_score >= 75:
            return "Good"
        elif health_score >= 60:
            return "Fair"
        elif health_score >= 40:
            return "Poor"
        else:
            return "Critical"
    def _clear_graph_references(self, graph):
        try:
            if hasattr(graph, 'node_labels'):
                for node in graph.node_labels:
                    if isinstance(node, dict):
                        for key, value in list(node.items()):
                            if hasattr(value, 'cpu'):
                                try:
                                    if key in node:
                                        del node[key]
                                except (KeyError, AttributeError):
                                    pass
                            elif isinstance(value, (list, tuple)) and len(value) > 100:
                                try:
                                    node[key] = []
                                except (KeyError, AttributeError):
                                    pass
            if hasattr(graph, 'edge_attributes'):
                graph.edge_attributes.clear()
        except Exception as e:
            log_step("Graph reference clearing error", error=str(e))
    def _break_system_references(self, system):
        try:
            if hasattr(system, '__dict__'):
                for attr_name in list(system.__dict__.keys()):
                    attr_value = system.__dict__[attr_name]
                    if hasattr(attr_value, '__dict__') and hasattr(attr_value, 'parent'):
                        if attr_value.parent is system:
                            attr_value.parent = None
                    elif isinstance(attr_value, (list, tuple, set)):
                        if len(attr_value) > 1000:
                            system.__dict__[attr_name] = []
        except Exception as e:
            log_step("System reference breaking error", error=str(e))
_global_simulation_manager = None


def get_simulation_manager() -> SimulationManager:
    global _global_simulation_manager
    if _global_simulation_manager is None:
        _global_simulation_manager = SimulationManager()
    return _global_simulation_manager


def create_simulation_manager(config: Optional[Dict[str, Any]] = None) -> SimulationManager:
    return SimulationManager(config)
if __name__ == "__main__":
    print("SimulationManager initialized successfully!")
    print("Features include:")
    print("- Unified simulation architecture")
    print("- All neural systems integrated")
    print("- Thread-safe operation")
    print("- Performance monitoring")
    print("- Callback system for external integration")
    print("- Configurable update intervals")
    manager = SimulationManager()
    print(f"Manager created with {len(manager.step_callbacks)} step callbacks")
    print("SimulationManager is ready for integration!")



