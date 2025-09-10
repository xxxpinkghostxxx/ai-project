"""
service_configuration.py

Service configuration for dependency injection container.
Configures all services with their implementations and dependencies.
"""

from typing import Dict, Any, Optional
from logging_utils import log_step
from dependency_injection import ServiceCollection, ServiceLifetime
from interfaces import (
    IBehaviorEngine, ILearningEngine, IMemorySystem, IHomeostasisController,
    INetworkMetrics, IWorkspaceEngine, IEnergyBehavior, IConnectionLogic,
    IDeathAndBirthLogic, IErrorHandler, IPerformanceOptimizer, IConfigurationManager,
    IEventBus, ISimulationState, ICallbackManager, IUIComponent, IWindowManager,
    ISensoryVisualization, ILiveMonitoring, ISimulationController
)
from event_bus import get_event_bus
from config_manager import ConfigManager


class ServiceConfiguration:
    """Configures all services for the dependency injection container."""
    
    def __init__(self):
        """Initialize service configuration."""
        self._services = ServiceCollection()
        log_step("ServiceConfiguration initialized")
    
    def configure_all_services(self) -> ServiceCollection:
        """Configure all services for the application."""
        log_step("Configuring all services")
        
        # Configure core services
        self._configure_core_services()
        
        # Configure neural systems
        self._configure_neural_systems()
        
        # Configure UI services
        self._configure_ui_services()
        
        # Configure simulation services
        self._configure_simulation_services()
        
        log_step("All services configured")
        return self._services
    
    def _configure_core_services(self) -> None:
        """Configure core services."""
        log_step("Configuring core services")
        
        # Configuration manager
        self._services.add_singleton(
            IConfigurationManager,
            instance=ConfigManager()
        )
        
        # Event bus
        self._services.add_singleton(
            IEventBus,
            factory=lambda: get_event_bus("default")
        )
        
        log_step("Core services configured")
    
    def _configure_neural_systems(self) -> None:
        """Configure neural system services."""
        log_step("Configuring neural systems")
        
        # Behavior Engine
        self._services.add_singleton(
            IBehaviorEngine,
            factory=self._create_behavior_engine
        )
        
        # Learning Engine
        self._services.add_singleton(
            ILearningEngine,
            factory=self._create_learning_engine
        )
        
        # Memory System
        self._services.add_singleton(
            IMemorySystem,
            factory=self._create_memory_system
        )
        
        # Homeostasis Controller
        self._services.add_singleton(
            IHomeostasisController,
            factory=self._create_homeostasis_controller
        )
        
        # Network Metrics
        self._services.add_singleton(
            INetworkMetrics,
            factory=self._create_network_metrics
        )
        
        # Workspace Engine
        self._services.add_singleton(
            IWorkspaceEngine,
            factory=self._create_workspace_engine
        )
        
        log_step("Neural systems configured")
    
    def _configure_ui_services(self) -> None:
        """Configure UI services."""
        log_step("Configuring UI services")
        
        # Window Manager
        self._services.add_singleton(
            IWindowManager,
            factory=self._create_window_manager
        )
        
        # Sensory Visualization
        self._services.add_singleton(
            ISensoryVisualization,
            factory=self._create_sensory_visualization
        )
        
        # Live Monitoring
        self._services.add_singleton(
            ILiveMonitoring,
            factory=self._create_live_monitoring
        )
        
        log_step("UI services configured")
    
    def _configure_simulation_services(self) -> None:
        """Configure simulation services."""
        log_step("Configuring simulation services")
        
        # Energy Behavior
        self._services.add_singleton(
            IEnergyBehavior,
            factory=self._create_energy_behavior
        )
        
        # Connection Logic
        self._services.add_singleton(
            IConnectionLogic,
            factory=self._create_connection_logic
        )
        
        # Death and Birth Logic
        self._services.add_singleton(
            IDeathAndBirthLogic,
            factory=self._create_death_birth_logic
        )
        
        # Error Handler
        self._services.add_singleton(
            IErrorHandler,
            factory=self._create_error_handler
        )
        
        # Performance Optimizer
        self._services.add_singleton(
            IPerformanceOptimizer,
            factory=self._create_performance_optimizer
        )
        
        # Simulation Controller
        self._services.add_singleton(
            ISimulationController,
            factory=self._create_simulation_controller
        )
        
        log_step("Simulation services configured")
    
    # Factory methods for creating service instances
    def _create_behavior_engine(self) -> IBehaviorEngine:
        """Create behavior engine instance."""
        from behavior_engine import BehaviorEngine
        return BehaviorEngine()
    
    def _create_learning_engine(self) -> ILearningEngine:
        """Create learning engine instance."""
        from learning_engine import LearningEngine
        return LearningEngine()
    
    def _create_memory_system(self) -> IMemorySystem:
        """Create memory system instance."""
        from memory_system import MemorySystem
        return MemorySystem()
    
    def _create_homeostasis_controller(self) -> IHomeostasisController:
        """Create homeostasis controller instance."""
        from homeostasis_controller import HomeostasisController
        return HomeostasisController()
    
    def _create_network_metrics(self) -> INetworkMetrics:
        """Create network metrics instance."""
        from network_metrics import NetworkMetrics
        return NetworkMetrics()
    
    def _create_workspace_engine(self) -> IWorkspaceEngine:
        """Create workspace engine instance."""
        from workspace_engine import WorkspaceEngine
        return WorkspaceEngine()
    
    def _create_window_manager(self) -> IWindowManager:
        """Create window manager instance."""
        from ui_engine_decoupled import DecoupledWindowManager
        event_bus = self._services.build().resolve(IEventBus)
        return DecoupledWindowManager(event_bus)
    
    def _create_sensory_visualization(self) -> ISensoryVisualization:
        """Create sensory visualization instance."""
        from ui_engine_decoupled import DecoupledSensoryVisualization, DecoupledUIStateManager
        container = self._services.build()
        event_bus = container.resolve(IEventBus)
        ui_state_manager = DecoupledUIStateManager(event_bus)
        return DecoupledSensoryVisualization(event_bus, ui_state_manager)
    
    def _create_live_monitoring(self) -> ILiveMonitoring:
        """Create live monitoring instance."""
        from ui_engine_decoupled import DecoupledLiveMonitoring, DecoupledUIStateManager
        container = self._services.build()
        event_bus = container.resolve(IEventBus)
        ui_state_manager = DecoupledUIStateManager(event_bus)
        return DecoupledLiveMonitoring(event_bus, ui_state_manager)
    
    def _create_energy_behavior(self) -> IEnergyBehavior:
        """Create energy behavior instance."""
        from energy_behavior import EnergyBehaviorAdapter
        return EnergyBehaviorAdapter()
    
    def _create_connection_logic(self) -> IConnectionLogic:
        """Create connection logic instance."""
        from connection_logic import ConnectionLogicAdapter
        return ConnectionLogicAdapter()
    
    def _create_death_birth_logic(self) -> IDeathAndBirthLogic:
        """Create death and birth logic instance."""
        from death_and_birth_logic import DeathAndBirthLogicAdapter
        return DeathAndBirthLogicAdapter()
    
    def _create_error_handler(self) -> IErrorHandler:
        """Create error handler instance."""
        from error_handler import ErrorHandler
        return ErrorHandler()
    
    def _create_performance_optimizer(self) -> IPerformanceOptimizer:
        """Create performance optimizer instance."""
        from performance_optimizer_v2 import PerformanceOptimizer
        return PerformanceOptimizer()
    
    def _create_simulation_controller(self) -> ISimulationController:
        """Create simulation controller instance."""
        from simulation_manager_decoupled import DecoupledSimulationController
        container = self._services.build()
        return DecoupledSimulationController(container)


# Adapter classes to make existing classes implement interfaces
class EnergyBehaviorAdapter(IEnergyBehavior):
    """Adapter for energy behavior to implement interface."""
    
    def __init__(self):
        """Initialize energy behavior adapter."""
        from energy_behavior import apply_energy_behavior, update_membrane_potentials, apply_refractory_periods
        self._apply_energy_behavior = apply_energy_behavior
        self._update_membrane_potentials = update_membrane_potentials
        self._apply_refractory_periods = apply_refractory_periods
    
    def apply_energy_behavior(self, graph, behavior_params=None):
        """Apply energy behavior to graph."""
        return self._apply_energy_behavior(graph, behavior_params)
    
    def update_membrane_potentials(self, graph):
        """Update membrane potentials."""
        return self._update_membrane_potentials(graph)
    
    def apply_refractory_periods(self, graph):
        """Apply refractory periods."""
        return self._apply_refractory_periods(graph)


class ConnectionLogicAdapter(IConnectionLogic):
    """Adapter for connection logic to implement interface."""
    
    def __init__(self):
        """Initialize connection logic adapter."""
        from connection_logic import intelligent_connection_formation, create_weighted_connection
        self._intelligent_connection_formation = intelligent_connection_formation
        self._create_weighted_connection = create_weighted_connection
    
    def intelligent_connection_formation(self, graph):
        """Form intelligent connections."""
        return self._intelligent_connection_formation(graph)
    
    def create_weighted_connection(self, graph, source_id, target_id, weight, edge_type='excitatory'):
        """Create a weighted connection."""
        return self._create_weighted_connection(graph, source_id, target_id, weight, edge_type)


class DeathAndBirthLogicAdapter(IDeathAndBirthLogic):
    """Adapter for death and birth logic to implement interface."""
    
    def __init__(self):
        """Initialize death and birth logic adapter."""
        from death_and_birth_logic import birth_new_dynamic_nodes, remove_dead_dynamic_nodes
        self._birth_new_dynamic_nodes = birth_new_dynamic_nodes
        self._remove_dead_dynamic_nodes = remove_dead_dynamic_nodes
    
    def birth_new_dynamic_nodes(self, graph):
        """Birth new dynamic nodes."""
        return self._birth_new_dynamic_nodes(graph)
    
    def remove_dead_dynamic_nodes(self, graph):
        """Remove dead dynamic nodes."""
        return self._remove_dead_dynamic_nodes(graph)


# Global service configuration
_service_configuration: Optional[ServiceConfiguration] = None


def get_service_configuration() -> ServiceConfiguration:
    """Get the global service configuration."""
    global _service_configuration
    
    if _service_configuration is None:
        _service_configuration = ServiceConfiguration()
    
    return _service_configuration


def configure_services() -> ServiceCollection:
    """Configure all services and return the service collection."""
    config = get_service_configuration()
    return config.configure_all_services()


# Example usage and testing
if __name__ == "__main__":
    print("Service configuration created successfully!")
    print("Features include:")
    print("- Automatic service registration")
    print("- Interface-based dependency injection")
    print("- Adapter pattern for existing classes")
    print("- Factory methods for service creation")
    print("- Comprehensive service configuration")
    
    # Test service configuration
    services = configure_services()
    container = services.build()
    
    print(f"Services configured: {len(container.get_registered_services())}")
    print("Service configuration test completed successfully!")
