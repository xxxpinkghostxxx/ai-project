"""
application_bootstrap.py

Application bootstrap for the decoupled neural system.
Handles service configuration, dependency injection, and application startup.
"""

import sys
import os
import logging
from typing import Optional, Dict, Any
from logging_utils import log_step, setup_logging
from service_configuration import configure_services
from dependency_injection import DependencyInjectionContainer
from interfaces import (
    ISimulationState, IUIStateManager, IEventBus, IConfigurationManager,
    IBehaviorEngine, ILearningEngine, IMemorySystem, IHomeostasisController,
    INetworkMetrics, IWorkspaceEngine, IErrorHandler, IPerformanceOptimizer
)
from main_graph import initialize_main_graph


class ApplicationBootstrap:
    """Bootstrap class for the decoupled neural system application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize application bootstrap."""
        self.config_path = config_path or "config.ini"
        self.service_provider: Optional[DependencyInjectionContainer] = None
        self.is_initialized = False
        
        log_step("ApplicationBootstrap initialized")
    
    def initialize_application(self) -> bool:
        """Initialize the entire application."""
        try:
            log_step("Starting application initialization")
            
            # Step 1: Setup logging
            self._setup_logging()
            
            # Step 2: Configure services
            self._configure_services()
            
            # Step 3: Initialize core systems
            self._initialize_core_systems()
            
            # Step 4: Setup event subscriptions
            self._setup_event_subscriptions()
            
            # Step 5: Initialize graph
            self._initialize_graph()
            
            self.is_initialized = True
            log_step("Application initialization completed successfully")
            return True
            
        except Exception as e:
            log_step("Application initialization failed", error=str(e))
            return False
    
    def _setup_logging(self) -> None:
        """Setup application logging."""
        log_step("Setting up logging")
        setup_logging(level="INFO")
        logging.info("Application logging initialized")
    
    def _configure_services(self) -> None:
        """Configure all services."""
        log_step("Configuring services")
        
        # Configure services
        service_collection = configure_services()
        self.service_provider = service_collection.build()
        
        log_step("Services configured successfully")
    
    def _initialize_core_systems(self) -> None:
        """Initialize core systems."""
        log_step("Initializing core systems")
        
        # Get core services
        config_manager = self.service_provider.resolve(IConfigurationManager)
        event_bus = self.service_provider.resolve(IEventBus)
        
        # Initialize neural systems
        behavior_engine = self.service_provider.resolve(IBehaviorEngine)
        learning_engine = self.service_provider.resolve(ILearningEngine)
        memory_system = self.service_provider.resolve(IMemorySystem)
        homeostasis_controller = self.service_provider.resolve(IHomeostasisController)
        network_metrics = self.service_provider.resolve(INetworkMetrics)
        workspace_engine = self.service_provider.resolve(IWorkspaceEngine)
        
        # Initialize error handling and performance
        error_handler = self.service_provider.resolve(IErrorHandler)
        performance_optimizer = self.service_provider.resolve(IPerformanceOptimizer)
        
        log_step("Core systems initialized")
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        log_step("Setting up event subscriptions")
        
        event_bus = self.service_provider.resolve(IEventBus)
        
        # Subscribe to application events
        event_bus.subscribe("application_started", self._on_application_started)
        event_bus.subscribe("application_stopped", self._on_application_stopped)
        event_bus.subscribe("simulation_started", self._on_simulation_started)
        event_bus.subscribe("simulation_stopped", self._on_simulation_stopped)
        
        log_step("Event subscriptions setup completed")
    
    def _initialize_graph(self) -> None:
        """Initialize the main graph."""
        log_step("Initializing main graph")
        
        # Create initial graph
        graph = initialize_main_graph()
        
        # Get simulation manager and set graph
        sim_manager = self.service_provider.resolve(ISimulationState)
        sim_manager.set_graph(graph)
        
        log_step("Main graph initialized")
    
    def start_application(self) -> bool:
        """Start the application."""
        if not self.is_initialized:
            log_step("Application not initialized, initializing now")
            if not self.initialize_application():
                return False
        
        try:
            log_step("Starting application")
            
            # Publish application started event
            event_bus = self.service_provider.resolve(IEventBus)
            event_bus.publish("application_started", {})
            
            # Start simulation
            sim_manager = self.service_provider.resolve(ISimulationState)
            sim_manager.start_simulation()
            
            log_step("Application started successfully")
            return True
            
        except Exception as e:
            log_step("Failed to start application", error=str(e))
            return False
    
    def stop_application(self) -> None:
        """Stop the application."""
        try:
            log_step("Stopping application")
            
            # Stop simulation
            sim_manager = self.service_provider.resolve(ISimulationState)
            sim_manager.stop_simulation()
            
            # Publish application stopped event
            event_bus = self.service_provider.resolve(IEventBus)
            event_bus.publish("application_stopped", {})
            
            # Cleanup
            self._cleanup()
            
            log_step("Application stopped successfully")
            
        except Exception as e:
            log_step("Error stopping application", error=str(e))
    
    def _cleanup(self) -> None:
        """Cleanup application resources."""
        log_step("Cleaning up application resources")
        
        # Cleanup simulation manager
        sim_manager = self.service_provider.resolve(ISimulationState)
        sim_manager.cleanup()
        
        # Clear event bus
        event_bus = self.service_provider.resolve(IEventBus)
        event_bus.clear_subscribers()
        
        log_step("Application cleanup completed")
    
    # Event handlers
    def _on_application_started(self, event_name: str, data: Dict[str, Any]) -> None:
        """Handle application started event."""
        log_step("Application started event received")
    
    def _on_application_stopped(self, event_name: str, data: Dict[str, Any]) -> None:
        """Handle application stopped event."""
        log_step("Application stopped event received")
    
    def _on_simulation_started(self, event_name: str, data: Dict[str, Any]) -> None:
        """Handle simulation started event."""
        log_step("Simulation started event received")
    
    def _on_simulation_stopped(self, event_name: str, data: Dict[str, Any]) -> None:
        """Handle simulation stopped event."""
        log_step("Simulation stopped event received")
    
    def get_service_provider(self) -> DependencyInjectionContainer:
        """
        Get the service provider.
        
        Returns:
            DependencyInjectionContainer: The service provider instance
        """
        return self.service_provider
    
    def is_application_initialized(self) -> bool:
        """Check if application is initialized."""
        return self.is_initialized


# Global application bootstrap
_application_bootstrap: Optional[ApplicationBootstrap] = None


def get_application_bootstrap() -> ApplicationBootstrap:
    """Get the global application bootstrap."""
    global _application_bootstrap
    
    if _application_bootstrap is None:
        _application_bootstrap = ApplicationBootstrap()
    
    return _application_bootstrap


def initialize_application(config_path: Optional[str] = None) -> bool:
    """Initialize the application."""
    bootstrap = get_application_bootstrap()
    return bootstrap.initialize_application()


def start_application() -> bool:
    """Start the application."""
    bootstrap = get_application_bootstrap()
    return bootstrap.start_application()


def stop_application() -> None:
    """Stop the application."""
    bootstrap = get_application_bootstrap()
    bootstrap.stop_application()


def get_service_provider() -> DependencyInjectionContainer:
    """Get the service provider."""
    bootstrap = get_application_bootstrap()
    return bootstrap.get_service_provider()


# Example usage and testing
if __name__ == "__main__":
    print("Application bootstrap created successfully!")
    print("Features include:")
    print("- Service configuration and dependency injection")
    print("- Event-driven architecture")
    print("- Graceful startup and shutdown")
    print("- Resource cleanup")
    print("- Error handling and logging")
    
    # Test application bootstrap
    try:
        # Initialize application
        if initialize_application():
            print("Application initialized successfully")
            
            # Start application
            if start_application():
                print("Application started successfully")
                
                # Stop application
                stop_application()
                print("Application stopped successfully")
            else:
                print("Failed to start application")
        else:
            print("Failed to initialize application")
            
    except Exception as e:
        print(f"Application bootstrap test failed: {e}")
    
    print("Application bootstrap test completed!")
