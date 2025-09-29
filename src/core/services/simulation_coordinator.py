"""
SimulationCoordinator implementation - Main orchestration service.

This module provides the concrete implementation of ISimulationCoordinator,
orchestrating all neural simulation services while maintaining clean
separation of concerns and energy-based processing as the central integrator.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from torch_geometric.data import Data

from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.energy_manager import EnergyFlow, IEnergyManager
from ..interfaces.event_coordinator import IEventCoordinator
from ..interfaces.graph_manager import IGraphManager
from ..interfaces.learning_engine import ILearningEngine, PlasticityEvent
from ..interfaces.neural_processor import INeuralProcessor, SpikeEvent
from ..interfaces.performance_monitor import IPerformanceMonitor
from ..interfaces.sensory_processor import ISensoryProcessor
from ..interfaces.service_registry import IServiceRegistry
from ..interfaces.simulation_coordinator import (ISimulationCoordinator,
                                                 SimulationState)


class SimulationCoordinator(ISimulationCoordinator):
    """
    Concrete implementation of ISimulationCoordinator.

    This class orchestrates all neural simulation services, coordinating
    neural processing, energy management, learning, and sensory integration
    while maintaining biological plausibility and performance requirements.
    """

    def __init__(self,
                 service_registry: IServiceRegistry,
                 neural_processor: INeuralProcessor,
                 energy_manager: IEnergyManager,
                 learning_engine: ILearningEngine,
                 sensory_processor: ISensoryProcessor,
                 performance_monitor: IPerformanceMonitor,
                 graph_manager: IGraphManager,
                 event_coordinator: IEventCoordinator,
                 configuration_service: IConfigurationService):
        """
        Initialize the SimulationCoordinator with all required services.

        Args:
            service_registry: Service registry for dependency resolution
            neural_processor: Neural dynamics processing service
            energy_manager: Energy flow and conservation service
            learning_engine: Learning and plasticity service
            sensory_processor: Sensory input processing service
            performance_monitor: Performance monitoring service
            graph_manager: Graph management service
            event_coordinator: Event-driven communication service
            configuration_service: Configuration management service
        """
        self.service_registry = service_registry
        self.neural_processor = neural_processor
        self.energy_manager = energy_manager
        self.learning_engine = learning_engine
        self.sensory_processor = sensory_processor
        self.performance_monitor = performance_monitor
        self.graph_manager = graph_manager
        self.event_coordinator = event_coordinator
        self.configuration_service = configuration_service

        # Simulation state
        self._simulation_state = SimulationState()
        self._neural_graph: Optional[Data] = None
        self._is_initialized = False

        # Performance tracking
        self._step_times: List[float] = []
        self._last_step_start_time = 0.0

    def initialize_simulation(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the neural simulation with comprehensive service coordination.

        This method initializes all services in the correct order, ensuring
        proper dependencies and energy-based integration from the start.

        Args:
            config: Optional configuration dictionary for simulation parameters

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Update configuration if provided
            if config:
                self.configuration_service.load_configuration()
                for key, value in config.items():
                    self.configuration_service.set_parameter(key, value)

            # Initialize services in dependency order
            self._neural_graph = self.graph_manager.initialize_graph()

            # Initialize neural processing with energy integration
            neural_success = self.neural_processor.initialize_neural_state(self._neural_graph)
            if not neural_success:
                raise RuntimeError("Failed to initialize neural processor")

            # Initialize energy management as central integrator
            energy_success = self.energy_manager.initialize_energy_state(self._neural_graph)
            if not energy_success:
                raise RuntimeError("Failed to initialize energy manager")

            # Initialize learning with energy modulation
            learning_success = self.learning_engine.initialize_learning_state(self._neural_graph)
            if not learning_success:
                raise RuntimeError("Failed to initialize learning engine")

            # Initialize sensory processing
            sensory_success = self.sensory_processor.initialize_sensory_pathways(self._neural_graph)
            if not sensory_success:
                raise RuntimeError("Failed to initialize sensory processor")

            # Start performance monitoring
            monitor_success = self.performance_monitor.start_monitoring()
            if not monitor_success:
                print("Warning: Performance monitoring failed to start")

            # Mark as initialized
            self._is_initialized = True
            self._simulation_state.is_running = False

            # Publish initialization event
            self.event_coordinator.publish("simulation_initialized", {
                "timestamp": datetime.now().isoformat(),
                "graph_nodes": len(self._neural_graph.node_labels) if self._neural_graph else 0
            })

            return True

        except Exception as e:
            print(f"Simulation initialization failed: {e}")
            self.event_coordinator.publish("simulation_initialization_failed", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return False

    def start_simulation(self) -> bool:
        """Start the neural simulation execution."""
        if not self._is_initialized:
            print("Error: Simulation not initialized")
            return False

        try:
            self._simulation_state.is_running = True
            self._simulation_state.step_count = 0
            self.event_coordinator.publish("simulation_started", {
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            print(f"Failed to start simulation: {e}")
            return False

    def stop_simulation(self) -> bool:
        """Stop the neural simulation execution."""
        try:
            self._simulation_state.is_running = False
            self.event_coordinator.publish("simulation_stopped", {
                "timestamp": datetime.now().isoformat(),
                "final_step_count": self._simulation_state.step_count
            })
            return True
        except Exception as e:
            print(f"Failed to stop simulation: {e}")
            return False

    def reset_simulation(self) -> bool:
        """Reset the neural simulation to initial state."""
        try:
            # Stop if running
            if self._simulation_state.is_running:
                self.stop_simulation()

            # Reset all services
            self.neural_processor.reset_neural_state()
            self.energy_manager.reset_energy_state()
            self.learning_engine.reset_learning_state()

            # Reset simulation state
            self._simulation_state = SimulationState()
            self._step_times.clear()

            self.event_coordinator.publish("simulation_reset", {
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            print(f"Failed to reset simulation: {e}")
            return False

    def execute_simulation_step(self, step: int) -> bool:
        """
        Execute a single simulation step with comprehensive service coordination.

        This is the core method that orchestrates all neural simulation activities
        for one time step, maintaining energy as the central integrator while
        ensuring biological plausibility and performance requirements.

        Args:
            step: The current simulation step number

        Returns:
            bool: True if step executed successfully, False otherwise
        """
        if not self._is_initialized or not self._simulation_state.is_running or self._neural_graph is None:
            return False

        step_start_time = time.time()

        try:
            # Core simulation logic
            graph = self._neural_graph
            
            # 1. Process sensory input
            self.sensory_processor.process_sensory_input(graph)

            # 2. Event System Processing (handled by publish-subscribe pattern)
            # Events are processed by subscribers when published
            self.event_coordinator.process_events()

            # 3. Neural Dynamics (including energy)
            graph, spike_events = self.neural_processor.process_neural_dynamics(graph)
            
            # 4. Energy Management
            graph, energy_flows = self.energy_manager.update_energy_flows(graph, spike_events)

            # 5. Learning and Plasticity
            graph, plasticity_events = self.learning_engine.apply_plasticity(graph, spike_events)

            # 6. Workspace and Memory Systems
            # (Assuming these will be services called here in a similar fashion)
            # self.workspace_service.update(graph)
            # self.memory_service.update(graph)

            # 7. Node Lifecycle (Birth/Death)
            graph = self.graph_manager.update_node_lifecycle(graph)

            # 8. Homeostatic Control
            graph = self.energy_manager.regulate_energy_homeostasis(graph)

            # Finalize graph state
            self._neural_graph = graph
            
            # Update state and metrics
            step_duration = time.time() - step_start_time
            self._step_times.append(step_duration)
            self._simulation_state.step_count = step
            self._simulation_state.last_step_time = step_duration
            
            self.performance_monitor.record_step_end()
            self.performance_monitor.record_step_start() # for the next step

            self.event_coordinator.publish("simulation_step_completed", {
                "step": step,
                "duration": step_duration,
                "spike_count": len(spike_events),
                "plasticity_event_count": len(plasticity_events)
            })

            return True

        except Exception as e:
            print(f"Simulation step {step} failed: {e}")
            self.event_coordinator.publish("simulation_step_failed", {
                "step": step,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return False

    def get_simulation_state(self) -> SimulationState:
        """Get the current state of the neural simulation."""
        # Update with latest metrics
        if self.performance_monitor:
            metrics = self.performance_monitor.get_current_metrics()
            self._simulation_state.performance_metrics = {
                "step_time": metrics.step_time,
                "memory_usage": metrics.memory_usage,
                "cpu_usage": metrics.cpu_usage
            }
        return self._simulation_state

    def get_neural_graph(self) -> Optional[Data]:
        """Get the current neural graph with all node and connection data."""
        return self._neural_graph

    @property
    def graph(self) -> Optional[Data]:
        """Get the current neural graph (backward compatibility property)."""
        return self._neural_graph

    def get_access_layer(self):
        """Get the node access layer instance for the current neural graph."""
        if self._neural_graph is None:
            return None
        from src.energy.node_access_layer import NodeAccessLayer
        return NodeAccessLayer(self._neural_graph)

    def update_configuration(self, config_updates: Dict[str, Any]) -> bool:
        """Update simulation configuration parameters dynamically."""
        try:
            for key, value in config_updates.items():
                self.configuration_service.set_parameter(key, value)
            return True
        except Exception as e:
            print(f"Configuration update failed: {e}")
            return False

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics for the simulation."""
        if not self.performance_monitor:
            return {}

        metrics = self.performance_monitor.get_current_metrics()
        avg_step_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0

        return {
            "current_step_time": metrics.step_time,
            "average_step_time": avg_step_time,
            "memory_usage": metrics.memory_usage,
            "cpu_usage": metrics.cpu_usage,
            "gpu_usage": metrics.gpu_usage,
            "total_steps": self._simulation_state.step_count,
            "simulation_uptime": time.time() - self._last_step_start_time if self._last_step_start_time else 0
        }

    def validate_simulation_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the neural simulation state."""
        issues = []

        # Validate neural graph
        if self._neural_graph is None:
            issues.append("Neural graph is not initialized")
        else:
            graph_validation = self.graph_manager.validate_graph_integrity(self._neural_graph)
            if not graph_validation.get("valid", True):
                issues.extend(graph_validation.get("issues", []))

        # Validate neural state
        neural_validation = self.neural_processor.validate_neural_integrity(self._neural_graph)
        if not neural_validation.get("valid", True):
            issues.extend(neural_validation.get("issues", []))

        # Validate energy conservation
        energy_validation = self.energy_manager.validate_energy_conservation(self._neural_graph)
        conservation_rate = energy_validation.get("energy_conservation_rate", 1.0)
        if conservation_rate < 0.9:  # Energy loss too high (>10% loss)
            issues.append("Energy conservation violated")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "neural_integrity": neural_validation,
            "energy_conservation": energy_validation,
            "graph_integrity": graph_validation if self._neural_graph else None
        }

    def save_simulation_state(self, filepath: str) -> bool:
        """Save the current simulation state to file."""
        try:
            state_data = {
                "simulation_state": self._simulation_state.to_dict(),
                "neural_state": self.neural_processor.get_neural_state().to_dict(),
                "energy_state": self.energy_manager.get_energy_state().to_dict(),
                "learning_state": self.learning_engine.get_learning_state().to_dict(),
                "configuration": self.configuration_service.get_configuration_schema(),
                "timestamp": datetime.now().isoformat()
            }

            # Save graph separately
            if self._neural_graph:
                self.graph_manager.save_graph(self._neural_graph, filepath + ".graph")

            # Save state data
            import json
            with open(filepath + ".state", 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            return True
        except Exception as e:
            print(f"Failed to save simulation state: {e}")
            return False

    def load_simulation_state(self, filepath: str) -> bool:
        """Load simulation state from file."""
        try:
            # Load state data
            import json
            with open(filepath + ".state", 'r') as f:
                state_data = json.load(f)

            # Load graph
            self._neural_graph = self.graph_manager.load_graph(filepath + ".graph")

            # Restore states
            self._simulation_state = SimulationState()
            # Note: Full state restoration would require more complex implementation

            return True
        except Exception as e:
            print(f"Failed to load simulation state: {e}")
            return False

    # Backward compatibility methods for UI
    def start(self) -> bool:
        """Alias for start_simulation for backward compatibility."""
        return self.start_simulation()

    def stop(self) -> bool:
        """Alias for stop_simulation for backward compatibility."""
        return self.stop_simulation()

    def reset(self) -> bool:
        """Alias for reset_simulation for backward compatibility."""
        return self.reset_simulation()

    def save_neural_map(self, slot: int) -> bool:
        """Save neural map to specified slot for backward compatibility."""
        try:
            filepath = f"data/neural_maps/neural_map_slot_{slot}"
            return self.save_simulation_state(filepath)
        except Exception as e:
            print(f"Failed to save neural map to slot {slot}: {e}")
            return False

    def load_neural_map(self, slot: int) -> bool:
        """Load neural map from specified slot for backward compatibility."""
        try:
            filepath = f"data/neural_maps/neural_map_slot_{slot}"
            return self.load_simulation_state(filepath)
        except Exception as e:
            print(f"Failed to load neural map from slot {slot}: {e}")
            return False

    def run_single_step(self) -> bool:
        """
        Execute a single simulation step, incrementing the step count.

        This method provides a convenient way to run one simulation step,
        automatically managing the step counter.

        Returns:
            bool: True if the step executed successfully, False otherwise
        """
        self._simulation_state.step_count += 1
        return self.execute_simulation_step(self._simulation_state.step_count)

    def cleanup(self):
        """
        Clean up simulation resources and reset state.

        This method stops the simulation if running and performs cleanup operations
        to free resources and reset the coordinator to a clean state.
        """
        try:
            if self._simulation_state.is_running:
                self.stop_simulation()
            # Reset to clean state
            self.reset_simulation()
            # Clear any cached data
            self._neural_graph = None
            self._step_times.clear()
            self._last_step_start_time = 0.0
        except Exception as e:
            print(f"Cleanup failed: {e}")






