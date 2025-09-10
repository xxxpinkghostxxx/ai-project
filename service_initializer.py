"""
service_initializer.py

Service initialization and dependency injection setup.
Centralizes the registration of all system services.
"""

from dependency_injection import get_container, register_service
from error_handler import ErrorHandler
from node_id_manager import NodeIDManager
from simulation_manager import SimulationManager
from config_manager import ConfigManager
from logging_utils import log_step


def initialize_services() -> None:
    """
    Initialize and register all system services in the dependency container.
    This should be called once at application startup.
    """
    container = get_container()
    
    log_step("Initializing system services")
    
    # Register core services as singletons
    try:
        # Configuration manager
        config_manager = ConfigManager()
        register_service(ConfigManager, instance=config_manager)
        log_step("Registered ConfigManager service")
        
        # Error handler
        error_handler = ErrorHandler()
        register_service(ErrorHandler, instance=error_handler)
        log_step("Registered ErrorHandler service")
        
        # Node ID manager
        id_manager = NodeIDManager()
        register_service(NodeIDManager, instance=id_manager)
        log_step("Registered NodeIDManager service")
        
        # Simulation manager (will be created on demand)
        register_service(SimulationManager, implementation_type=SimulationManager)
        log_step("Registered SimulationManager service")
        
        log_step("All system services initialized successfully")
        
    except Exception as e:
        log_step(f"Error initializing services: {e}")
        raise


def initialize_services_with_config(config: dict = None) -> None:
    """
    Initialize services with custom configuration.
    
    Args:
        config: Optional configuration dictionary
    """
    container = get_container()
    
    log_step("Initializing system services with custom config")
    
    try:
        # Configuration manager with custom config
        config_manager = ConfigManager()
        if config:
            # Apply custom configuration
            for section, values in config.items():
                for key, value in values.items():
                    config_manager.set(section, key, value)
        register_service(ConfigManager, instance=config_manager)
        log_step("Registered ConfigManager service with custom config")
        
        # Error handler
        error_handler = ErrorHandler()
        register_service(ErrorHandler, instance=error_handler)
        log_step("Registered ErrorHandler service")
        
        # Node ID manager
        id_manager = NodeIDManager()
        register_service(NodeIDManager, instance=id_manager)
        log_step("Registered NodeIDManager service")
        
        # Simulation manager with custom config
        register_service(SimulationManager, implementation_type=SimulationManager)
        log_step("Registered SimulationManager service with custom config")
        
        log_step("All system services initialized with custom config")
        
    except Exception as e:
        log_step(f"Error initializing services with config: {e}")
        raise


def get_service_status() -> dict:
    """
    Get the status of all registered services.
    
    Returns:
        Dictionary with service registration status
    """
    container = get_container()
    return container.get_registered_services()


def reset_services() -> None:
    """Reset all services and clear the dependency container."""
    container = get_container()
    container.clear()
    log_step("All services reset")


def ensure_services_initialized() -> None:
    """
    Ensure all required services are initialized.
    This is a safety function that can be called anywhere.
    """
    container = get_container()
    
    required_services = [
        'ConfigManager',
        'ErrorHandler', 
        'NodeIDManager'
    ]
    
    missing_services = []
    for service_name in required_services:
        if not any(service_name in services for services in [
            container._singletons.keys(),
            container._factories.keys(),
            container._services.keys()
        ]):
            missing_services.append(service_name)
    
    if missing_services:
        log_step(f"Missing required services: {missing_services}")
        log_step("Auto-initializing missing services")
        initialize_services()
    else:
        log_step("All required services are initialized")


# Auto-initialize services when module is imported
try:
    ensure_services_initialized()
except Exception as e:
    log_step(f"Auto-initialization failed: {e}")
    # Don't raise here to avoid breaking imports
