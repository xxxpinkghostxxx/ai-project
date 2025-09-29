#!/usr/bin/env python3
"""
Debug script for SimulationCoordinator to identify and fix issues.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import traceback
from unittest.mock import Mock

from src.core.interfaces.configuration_service import IConfigurationService
from src.core.interfaces.energy_manager import IEnergyManager
from src.core.interfaces.event_coordinator import IEventCoordinator
from src.core.interfaces.graph_manager import IGraphManager
from src.core.interfaces.learning_engine import ILearningEngine
from src.core.interfaces.neural_processor import INeuralProcessor
from src.core.interfaces.performance_monitor import IPerformanceMonitor
from src.core.interfaces.sensory_processor import ISensoryProcessor
from src.core.interfaces.service_registry import IServiceRegistry
from src.core.services.simulation_coordinator import SimulationCoordinator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_simulation_coordinator_initialization():
    """Test basic SimulationCoordinator initialization."""
    print("=" * 60)
    print("Testing SimulationCoordinator Initialization")
    print("=" * 60)

    try:
        print("Creating mock services...")
        # Create all required mocks
        service_registry = Mock()
        neural_processor = Mock()
        energy_manager = Mock()
        learning_engine = Mock()
        sensory_processor = Mock()
        performance_monitor = Mock()
        graph_manager = Mock()
        event_coordinator = Mock()
        configuration_service = Mock()

        # Configure mocks for successful initialization
        from torch_geometric.data import Data
        graph_manager.initialize_graph.return_value = Data(x=[], edge_index=[])
        neural_processor.initialize_neural_state.return_value = True
        energy_manager.initialize_energy_state.return_value = True
        learning_engine.initialize_learning_state.return_value = True
        sensory_processor.initialize_sensory_pathways.return_value = True
        performance_monitor.start_monitoring.return_value = True

        # Configure mocks for simulation step
        neural_processor.process_neural_dynamics.return_value = (Data(), [])
        energy_manager.update_energy_flows.return_value = (Data(), [])
        learning_engine.apply_plasticity.return_value = (Data(), [])
        graph_manager.update_node_lifecycle.return_value = Data()
        energy_manager.regulate_energy_homeostasis.return_value = Data()

        print("Creating SimulationCoordinator...")
        manager = SimulationCoordinator(
            service_registry, neural_processor, energy_manager,
            learning_engine, sensory_processor, performance_monitor,
            graph_manager, event_coordinator, configuration_service
        )
        print("[OK] SimulationCoordinator created successfully")

        # Check critical attributes
        required_attrs = [
            'service_registry', 'neural_processor', 'energy_manager',
            'learning_engine', 'sensory_processor', 'performance_monitor',
            'graph_manager', 'event_coordinator', 'configuration_service'
        ]

        print("\nChecking required attributes:")
        for attr in required_attrs:
            if hasattr(manager, attr):
                value = getattr(manager, attr)
                print(f"[OK] {attr}: {type(value).__name__}")
            else:
                print(f"[FAIL] Missing attribute: {attr}")

        # Test simulation initialization
        print("\nTesting simulation initialization...")
        success = manager.initialize_simulation()
        if success:
            print("[OK] Simulation initialized successfully")
            if manager._neural_graph:
                node_count = len(manager._neural_graph.node_labels) if hasattr(manager._neural_graph, 'node_labels') else 0
                print(f"   Graph has {node_count} nodes")
        else:
            print("[FAIL] Simulation initialization failed")

        return manager

    except Exception as e:
        print(f"[FAIL] SimulationCoordinator initialization failed: {e}")
        traceback.print_exc()
        return None

def test_basic_simulation_step(manager):
    """Test a basic simulation step."""
    print("\n" + "=" * 60)
    print("Testing Basic Simulation Step")
    print("=" * 60)

    if not manager:
        print("[FAIL] No valid manager to test")
        return False
    if not manager._neural_graph:
        print(f"[FAIL] No neural graph: {manager._neural_graph}")
        return False

    try:
        print("Starting simulation...")
        start_success = manager.start_simulation()
        if not start_success:
            print("[FAIL] Failed to start simulation")
            return False

        print("Running single simulation step...")
        success = manager.execute_simulation_step(1)
        if success:
            print("[OK] Simulation step completed successfully")
            return True
        else:
            print("[FAIL] Simulation step failed")
            return False

    except Exception as e:
        print(f"[FAIL] Simulation step error: {e}")
        traceback.print_exc()
        return False

def test_performance_optimizations(manager):
    """Test performance optimization features."""
    print("\n" + "=" * 60)
    print("Testing Performance Optimizations")
    print("=" * 60)

    if not manager:
        print("[FAIL] No valid manager to test")
        return False

    try:
        # Check that services are properly initialized
        services = [
            ('neural_processor', manager.neural_processor),
            ('energy_manager', manager.energy_manager),
            ('learning_engine', manager.learning_engine),
            ('sensory_processor', manager.sensory_processor),
            ('performance_monitor', manager.performance_monitor),
            ('graph_manager', manager.graph_manager),
            ('event_coordinator', manager.event_coordinator),
            ('configuration_service', manager.configuration_service),
        ]

        print("Service availability:")
        for name, service in services:
            status = "[AVAILABLE]" if service else "[MISSING]"
            print(f"   {name}: {status}")

        # Test performance metrics
        print("\nTesting performance metrics...")
        try:
            metrics = manager.get_performance_metrics()
            print(f"[OK] Performance metrics retrieved: {len(metrics)} metrics")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"[FAIL] Performance metrics failed: {e}")

        return True

    except Exception as e:
        print(f"[FAIL] Performance optimization test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("SimulationCoordinator Debug Script")
    print("This script tests the core functionality and identifies issues.")

    # Test initialization
    manager = test_simulation_coordinator_initialization()

    if manager:
        # Test basic functionality
        test_basic_simulation_step(manager)

        # Test performance features
        test_performance_optimizations(manager)

        # Cleanup
        print("\n" + "=" * 60)
        print("Cleaning up...")
        manager.stop_simulation()
        print("[OK] Cleanup completed")

    print("\n" + "=" * 60)
    print("Debug script completed")
    print("=" * 60)

if __name__ == "__main__":
    main()






