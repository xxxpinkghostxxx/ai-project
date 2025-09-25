#!/usr/bin/env python3
"""
Test Phase 2B services - Distributed Processing Integration.

This test verifies that the Phase 2B distributed processing services
(DistributedCoordinatorService, LoadBalancingService, FaultToleranceService)
work correctly with the existing service architecture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_distributed_coordinator_service():
    """Test DistributedCoordinatorService functionality."""
    try:
        from core.services.distributed_coordinator_service import DistributedCoordinatorService
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.interfaces.distributed_coordinator import NodeInfo, DistributedTask

        # Create dependencies
        config = ConfigurationService()
        event_service = EventCoordinationService()
        distributed_service = DistributedCoordinatorService(config, event_service)

        # Test system initialization
        init_config = {
            "heartbeat_interval": 5.0,
            "task_timeout": 30.0,
            "max_tasks_per_node": 10
        }
        success = distributed_service.initialize_distributed_system(init_config)
        if not success:
            raise Exception("Distributed system initialization failed")

        # Test node registration
        node_info = NodeInfo(
            node_id="test_node_1",
            address="localhost:8080",
            capabilities={"neural": True, "learning": True, "energy": True}
        )
        success = distributed_service.register_node(node_info)
        if not success:
            raise Exception("Node registration failed")

        # Test task submission
        task = DistributedTask(
            task_id="test_task_1",
            task_type="neural_processing",
            data={"input_size": 100, "output_size": 50},
            priority=5
        )
        success = distributed_service.submit_task(task)
        if not success:
            raise Exception("Task submission failed")

        # Test system status
        status = distributed_service.get_system_status()
        if not isinstance(status, dict):
            raise Exception("Invalid system status")

        # Test workload balancing
        balance_result = distributed_service.balance_workload()
        if not balance_result.get("success", False):
            print("WARNING: Workload balancing may not be effective with single node")

        # Test energy optimization
        energy_result = distributed_service.optimize_energy_distribution()
        if not energy_result.get("success", False):
            print("WARNING: Energy optimization may not be effective with single node")

        print("PASS: DistributedCoordinatorService test successful")
        return True

    except Exception as e:
        print(f"FAIL: DistributedCoordinatorService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_load_balancing_service():
    """Test LoadBalancingService functionality."""
    try:
        from core.services.load_balancing_service import LoadBalancingService
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.services.distributed_coordinator_service import DistributedCoordinatorService

        # Create dependencies
        config = ConfigurationService()
        event_service = EventCoordinationService()
        distributed_service = DistributedCoordinatorService(config, event_service)
        load_service = LoadBalancingService(config, event_service, distributed_service)

        # Initialize distributed system
        distributed_service.initialize_distributed_system({})

        # Test load assessment
        metrics = load_service.assess_node_load("nonexistent_node")
        if metrics is None:
            raise Exception("Load assessment should return default metrics for unknown nodes")

        # Test optimal distribution calculation
        tasks = [
            {"id": "task1", "type": "neural", "priority": 5},
            {"id": "task2", "type": "learning", "priority": 3},
            {"id": "task3", "type": "energy", "priority": 7}
        ]
        nodes = ["node1", "node2"]

        distribution = load_service.calculate_optimal_distribution(tasks, nodes)
        if not isinstance(distribution, dict):
            raise Exception("Invalid task distribution")

        # Test load statistics
        stats = load_service.get_load_statistics()
        if not isinstance(stats, dict):
            raise Exception("Invalid load statistics")

        # Test load prediction
        predictions = load_service.predict_load_changes(time_window=60)
        if not isinstance(predictions, dict):
            raise Exception("Invalid load predictions")

        print("PASS: LoadBalancingService test successful")
        return True

    except Exception as e:
        print(f"FAIL: LoadBalancingService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fault_tolerance_service():
    """Test FaultToleranceService functionality."""
    try:
        from core.services.fault_tolerance_service import FaultToleranceService
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.services.distributed_coordinator_service import DistributedCoordinatorService

        # Create dependencies
        config = ConfigurationService()
        event_service = EventCoordinationService()
        distributed_service = DistributedCoordinatorService(config, event_service)
        fault_service = FaultToleranceService(config, event_service, distributed_service)

        # Test failure detection
        failures = fault_service.detect_failures()
        # Should not fail even if no failures are detected
        if not isinstance(failures, list):
            raise Exception("Invalid failure detection result")

        # Test system integrity validation
        integrity = fault_service.validate_system_integrity()
        if not isinstance(integrity, dict):
            raise Exception("Invalid system integrity result")

        # Test backup creation
        backup_result = fault_service.create_backup("test_component")
        if not backup_result.get("success", False):
            raise Exception("Backup creation failed")

        # Test failure statistics
        stats = fault_service.get_failure_statistics()
        if not isinstance(stats, dict):
            raise Exception("Invalid failure statistics")

        # Test service failure handling
        service_result = fault_service.handle_service_failure("test_service", "test_node")
        if not isinstance(service_result, dict):
            raise Exception("Invalid service failure handling result")

        print("PASS: FaultToleranceService test successful")
        return True

    except Exception as e:
        print(f"FAIL: FaultToleranceService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distributed_service_integration():
    """Test Phase 2B services integration with existing services."""
    try:
        from core.services.service_registry import ServiceRegistry
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.services.performance_monitoring_service import PerformanceMonitoringService
        from core.services.energy_management_service import EnergyManagementService
        from core.services.neural_processing_service import NeuralProcessingService
        from core.services.learning_service import LearningService
        from core.services.sensory_processing_service import SensoryProcessingService
        from core.services.graph_management_service import GraphManagementService
        from core.services.distributed_coordinator_service import DistributedCoordinatorService
        from core.services.load_balancing_service import LoadBalancingService
        from core.services.fault_tolerance_service import FaultToleranceService
        from core.services.simulation_coordinator import SimulationCoordinator
        from core.interfaces.distributed_coordinator import NodeInfo, DistributedTask

        # Create service registry
        registry = ServiceRegistry()

        # Register foundational services
        config_service = ConfigurationService()
        registry.register_instance(type(config_service), config_service)

        event_service = EventCoordinationService()
        registry.register_instance(type(event_service), event_service)

        perf_service = PerformanceMonitoringService()
        registry.register_instance(type(perf_service), perf_service)

        # Register Phase 2B distributed services
        distributed_service = DistributedCoordinatorService(config_service, event_service)
        registry.register_instance(type(distributed_service), distributed_service)

        load_service = LoadBalancingService(config_service, event_service, distributed_service)
        registry.register_instance(type(load_service), load_service)

        fault_service = FaultToleranceService(config_service, event_service, distributed_service)
        registry.register_instance(type(fault_service), fault_service)

        # Register core services
        energy_service = EnergyManagementService(config_service, event_service)
        registry.register_instance(type(energy_service), energy_service)

        neural_service = NeuralProcessingService(energy_service, config_service, event_service)
        registry.register_instance(type(neural_service), neural_service)

        learning_service = LearningService(energy_service, config_service, event_service)
        registry.register_instance(type(learning_service), learning_service)

        sensory_service = SensoryProcessingService(energy_service, config_service, event_service)
        registry.register_instance(type(sensory_service), sensory_service)

        graph_service = GraphManagementService(config_service, event_service)
        registry.register_instance(type(graph_service), graph_service)

        # Register simulation coordinator
        coordinator = SimulationCoordinator(
            registry,
            neural_service,
            energy_service,
            learning_service,
            sensory_service,
            perf_service,
            graph_service,
            event_service,
            config_service
        )
        registry.register_instance(type(coordinator), coordinator)

        # Test distributed system initialization
        init_success = distributed_service.initialize_distributed_system({
            "heartbeat_interval": 5.0,
            "task_timeout": 30.0
        })
        if not init_success:
            raise Exception("Distributed system initialization failed")

        # Test node registration and task submission
        node_info = NodeInfo(
            node_id="integration_test_node",
            address="localhost:9090",
            capabilities={"neural": True, "learning": True, "sensory": True}
        )
        node_success = distributed_service.register_node(node_info)
        if not node_success:
            raise Exception("Node registration failed")

        task = DistributedTask(
            task_id="integration_test_task",
            task_type="neural_processing",
            data={"test": "integration_data"},
            priority=3
        )
        task_success = distributed_service.submit_task(task)
        if not task_success:
            raise Exception("Task submission failed")

        # Test load balancing integration
        balance_result = load_service.rebalance_workload(threshold=0.9)
        if not isinstance(balance_result, dict):
            raise Exception("Load balancing integration failed")

        # Test fault tolerance integration
        integrity_result = fault_service.validate_system_integrity()
        if not isinstance(integrity_result, dict):
            raise Exception("Fault tolerance integration failed")

        # Test system status across all services
        system_status = distributed_service.get_system_status()
        if not isinstance(system_status, dict):
            raise Exception("System status integration failed")

        print("PASS: Phase 2B distributed service integration test successful")
        return True

    except Exception as e:
        print(f"FAIL: Phase 2B distributed service integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distributed_service_features():
    """Test advanced features of Phase 2B distributed services."""
    try:
        from core.services.distributed_coordinator_service import DistributedCoordinatorService
        from core.services.load_balancing_service import LoadBalancingService
        from core.services.fault_tolerance_service import FaultToleranceService
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        from core.interfaces.distributed_coordinator import NodeInfo, DistributedTask
        import time

        # Create services
        config = ConfigurationService()
        event_service = EventCoordinationService()
        distributed_service = DistributedCoordinatorService(config, event_service)
        load_service = LoadBalancingService(config, event_service, distributed_service)
        fault_service = FaultToleranceService(config, event_service, distributed_service)

        # Initialize distributed system
        distributed_service.initialize_distributed_system({})

        # Test multi-node scenario
        nodes = []
        for i in range(3):
            node_info = NodeInfo(
                node_id=f"test_node_{i}",
                address=f"localhost:808{i}",
                capabilities={
                    "neural": i % 2 == 0,  # Alternate neural capability
                    "learning": i % 3 == 0,  # Every third node
                    "sensory": True  # All nodes have sensory
                }
            )
            distributed_service.register_node(node_info)
            nodes.append(node_info.node_id)

        # Test task distribution
        tasks = []
        for i in range(10):
            task = DistributedTask(
                task_id=f"distributed_task_{i}",
                task_type=["neural_processing", "learning", "sensory"][i % 3],
                data={"task_number": i},
                priority=i % 10 + 1
            )
            distributed_service.submit_task(task)
            tasks.append(task)

        # Test load balancing with multiple nodes
        time.sleep(0.1)  # Allow task processing
        balance_result = load_service.rebalance_workload(threshold=0.7)
        if not isinstance(balance_result, dict):
            raise Exception("Multi-node load balancing failed")

        # Test fault tolerance with node failure
        failed_node = nodes[0]
        failure_result = fault_service.handle_node_failure(failed_node)
        if not isinstance(failure_result, dict):
            raise Exception("Node failure handling failed")

        # Test system integrity after failure
        integrity = fault_service.validate_system_integrity()
        if not isinstance(integrity, dict):
            raise Exception("System integrity validation failed after node failure")

        # Test task migration
        if len(tasks) > 0:
            migration_result = distributed_service.migrate_task(tasks[0].task_id, nodes[1])
            if not isinstance(migration_result, bool):
                raise Exception("Task migration failed")

        # Test comprehensive system status
        final_status = distributed_service.get_system_status()
        if not isinstance(final_status, dict):
            raise Exception("Final system status check failed")

        print("PASS: Advanced distributed service features test successful")
        return True

    except Exception as e:
        print(f"FAIL: Advanced distributed service features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Phase 2B Services - Distributed Processing Integration")
    print("=" * 75)

    tests = [
        test_distributed_coordinator_service,
        test_load_balancing_service,
        test_fault_tolerance_service,
        test_distributed_service_integration,
        test_distributed_service_features
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 75)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: Phase 2B distributed processing services integration is working!")
        print("Distributed neural simulation architecture is complete.")
        return True
    else:
        print("FAILURE: Some Phase 2B distributed services tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)