"""
Core interfaces for the service-oriented neural simulation architecture.

This module defines the abstract interfaces that establish clean contracts
between services, enabling loose coupling and testability while maintaining
biological plausibility and performance requirements.
"""

from .simulation_coordinator import ISimulationCoordinator
from .neural_processor import INeuralProcessor
from .energy_manager import IEnergyManager
from .learning_engine import ILearningEngine
from .sensory_processor import ISensoryProcessor
from .performance_monitor import IPerformanceMonitor
from .graph_manager import IGraphManager
from .event_coordinator import IEventCoordinator
from .configuration_service import IConfigurationService
from .service_registry import IServiceRegistry

__all__ = [
    'ISimulationCoordinator',
    'INeuralProcessor',
    'IEnergyManager',
    'ILearningEngine',
    'ISensoryProcessor',
    'IPerformanceMonitor',
    'IGraphManager',
    'IEventCoordinator',
    'IConfigurationService',
    'IServiceRegistry'
]