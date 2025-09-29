"""
Core interfaces for the service-oriented neural simulation architecture.

This module defines the abstract interfaces that establish clean contracts
between services, enabling loose coupling and testability while maintaining
biological plausibility and performance requirements.
"""

from .configuration_service import IConfigurationService
from .energy_manager import IEnergyManager
from .event_coordinator import IEventCoordinator
from .graph_manager import IGraphManager
from .learning_engine import ILearningEngine
from .neural_processor import INeuralProcessor
from .performance_monitor import IPerformanceMonitor
from .sensory_processor import ISensoryProcessor
from .service_registry import IServiceRegistry
from .simulation_coordinator import ISimulationCoordinator
