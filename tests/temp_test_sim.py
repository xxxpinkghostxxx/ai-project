import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import faulthandler
faulthandler.enable()  # Enable fault handler for segfault tracebacks
from src.core.services.service_registry import ServiceRegistry
from src.core.services.simulation_coordinator import SimulationCoordinator
from src.core.interfaces.simulation_coordinator import ISimulationCoordinator
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')  # More verbose for diagnosis

async def test_sim():
    print('Starting SOA test_sim')

    try:
        # Initialize service registry and services
        service_registry = ServiceRegistry()

        # Register services (minimal set for testing)
        from src.core.services.configuration_service import ConfigurationService
        from src.core.services.performance_monitoring_service import PerformanceMonitoringService
        from src.core.services.graph_management_service import GraphManagementService
        from src.core.services.neural_processing_service import NeuralProcessingService
        from src.core.services.energy_management_service import EnergyManagementService
        from src.core.services.learning_service import LearningService
        from src.core.services.event_coordination_service import EventCoordinationService
        from src.core.services.sensory_processing_service import SensoryProcessingService

        from src.core.interfaces.configuration_service import IConfigurationService
        from src.core.interfaces.performance_monitor import IPerformanceMonitor
        from src.core.interfaces.graph_manager import IGraphManager
        from src.core.interfaces.neural_processor import INeuralProcessor
        from src.core.interfaces.energy_manager import IEnergyManager
        from src.core.interfaces.learning_engine import ILearningEngine
        from src.core.interfaces.event_coordinator import IEventCoordinator
        from src.core.interfaces.sensory_processor import ISensoryProcessor

        service_registry.register(IConfigurationService, ConfigurationService)
        service_registry.register(IPerformanceMonitor, PerformanceMonitoringService)
        service_registry.register(IGraphManager, GraphManagementService)
        service_registry.register(INeuralProcessor, NeuralProcessingService)
        service_registry.register(IEnergyManager, EnergyManagementService)
        service_registry.register(ILearningEngine, LearningService)
        service_registry.register(IEventCoordinator, EventCoordinationService)
        service_registry.register(ISensoryProcessor, SensoryProcessingService)

        # Create coordinator
        coordinator = SimulationCoordinator(
            service_registry,
            service_registry.resolve(INeuralProcessor),
            service_registry.resolve(IEnergyManager),
            service_registry.resolve(ILearningEngine),
            service_registry.resolve(ISensoryProcessor),
            service_registry.resolve(IPerformanceMonitor),
            service_registry.resolve(IGraphManager),
            service_registry.resolve(IEventCoordinator),
            service_registry.resolve(IConfigurationService)
        )

        print('SOA services initialized')

    except Exception as e:
        print(f'Service initialization error: {e}')
        import traceback
        traceback.print_exc()
        return

    try:
        success = coordinator.initialize_simulation()
        if not success:
            print('Coordinator initialization failed')
            return
        print('Simulation initialized')
    except Exception as e:
        print(f'Initialization error: {e}')
        import traceback
        traceback.print_exc()
        return

    try:
        coordinator.start_simulation()
        print('Simulation started')

        for i in range(500):
            success = coordinator.execute_simulation_step(i)
            if not success:
                print(f'Step {i} failed')
                break

            if i % 10 == 0:
                graph = coordinator.get_neural_graph()
                node_count = len(graph.node_labels) if graph and hasattr(graph, 'node_labels') else 'N/A'
                print(f'Step {i}: Node count ~{node_count}')
            if i % 50 == 0:
                print(f'Completed step {i}')
                # Log key metrics
                state = coordinator.get_simulation_state()
                print(f'  Step: {state.step_count}, Total Energy: {state.total_energy:.2f}')

        coordinator.stop_simulation()
        print('Simulation steps complete')

    except Exception as e:
        print(f'Run error at step {i if "i" in locals() else "unknown"}: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test_sim())






