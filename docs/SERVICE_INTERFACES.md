# Service Interfaces Documentation

## Overview

The AI Neural Simulation System has been fully migrated to a service-oriented architecture (SOA) with 8 specialized services orchestrated by the SimulationCoordinator. The monolithic SimulationManager has been completely removed. All services implement well-defined interfaces for loose coupling, testability, and maintainability. Services communicate through dependency injection via the ServiceRegistry and interface-based contracts.

## Core Service Interfaces

### ISimulationCoordinator

**Location**: `core/interfaces/simulation_coordinator.py`

**Purpose**: Central coordination of the neural simulation lifecycle.

**Key Methods**:
- `initialize_simulation() -> bool`: Initialize the simulation environment
- `start_simulation() -> bool`: Start the simulation loop
- `stop_simulation() -> bool`: Stop the simulation loop
- `run_simulation_step() -> bool`: Execute a single simulation step
- `reset_simulation() -> bool`: Reset simulation to initial state
- `get_simulation_state() -> SimulationState`: Get current simulation state
- `get_neural_graph() -> Data`: Get the current neural graph
- `get_performance_metrics() -> Dict[str, Any]`: Get performance metrics
- `update_configuration(config: Dict[str, Any]) -> bool`: Update configuration
- `save_neural_map(slot: int) -> bool`: Save neural map to slot
- `load_neural_map(slot: int) -> bool`: Load neural map from slot

### IServiceRegistry

**Location**: `core/interfaces/service_registry.py`

**Purpose**: Service registration and dependency injection.

**Key Methods**:
- `register_instance(interface_type: Type, instance: Any) -> None`: Register service instance
- `resolve(interface_type: Type) -> Any`: Resolve service instance
- `has_service(interface_type: Type) -> bool`: Check if service is registered
- `unregister(interface_type: Type) -> None`: Unregister service

### IGraphManager

**Location**: `core/interfaces/graph_manager.py`

**Purpose**: Neural graph management and operations.

**Key Methods**:
- `initialize_graph() -> Data`: Initialize a new neural graph
- `update_graph(graph: Data) -> Data`: Update graph state
- `validate_graph(graph: Data) -> bool`: Validate graph integrity
- `get_graph_statistics(graph: Data) -> Dict[str, Any]`: Get graph statistics

### IPerformanceMonitor

**Location**: `core/interfaces/performance_monitor.py`

**Purpose**: System performance monitoring and metrics collection.

**Key Methods**:
- `start_monitoring() -> bool`: Start performance monitoring
- `stop_monitoring() -> bool`: Stop performance monitoring
- `get_current_metrics() -> PerformanceMetrics`: Get current metrics
- `get_historical_metrics(time_range: int) -> List[PerformanceMetrics]`: Get historical metrics
- `record_step_start() -> None`: Record step start time
- `record_step_end() -> None`: Record step end time

### IEventCoordinator

**Location**: `core/interfaces/event_coordinator.py`

**Purpose**: Event-driven communication between services.

**Key Methods**:
- `publish(event: str, data: Any = None) -> None`: Publish an event
- `subscribe(event: str, callback: Callable) -> None`: Subscribe to an event
- `unsubscribe(event: str, callback: Callable) -> None`: Unsubscribe from an event

### IConfigurationService

**Location**: `core/interfaces/configuration_service.py`

**Purpose**: Centralized configuration management.

**Key Methods**:
- `get_parameter(key: str, default: Any = None) -> Any`: Get configuration parameter
- `set_parameter(key: str, value: Any) -> None`: Set configuration parameter
- `load_configuration(path: str) -> bool`: Load configuration from file
- `save_configuration(path: str) -> bool`: Save configuration to file

### INeuralProcessor

**Location**: `core/interfaces/neural_processor.py`

**Purpose**: Neural processing and dynamics.

**Key Methods**:
- `process_neural_step(graph: Data, step: int) -> Data`: Process neural dynamics
- `update_node_behaviors(graph: Data) -> Data`: Update node behaviors
- `calculate_neural_metrics(graph: Data) -> Dict[str, Any]`: Calculate neural metrics

### IEnergyManager

**Location**: `core/interfaces/energy_manager.py`

**Purpose**: Energy system management.

**Key Methods**:
- `initialize_energy_system(graph: Data) -> bool`: Initialize energy system
- `update_energy_dynamics(graph: Data) -> Data`: Update energy dynamics
- `regulate_energy_homeostasis(graph: Data) -> Data`: Regulate energy homeostasis
- `get_energy_statistics(graph: Data) -> Dict[str, Any]`: Get energy statistics

### ILearningEngine

**Location**: `core/interfaces/learning_engine.py`

**Purpose**: Learning and plasticity mechanisms.

**Key Methods**:
- `apply_learning(graph: Data, step: int) -> Data`: Apply learning rules
- `consolidate_memories(graph: Data) -> Data`: Consolidate memory traces
- `update_plasticity(graph: Data) -> Data`: Update synaptic plasticity

### ISensoryProcessor

**Location**: `core/interfaces/sensory_processor.py`

**Purpose**: Sensory input processing.

**Key Methods**:
- `initialize_sensory_pathways(graph: Data) -> bool`: Initialize sensory pathways
- `process_sensory_input(input_data: Any) -> Dict[str, Any]`: Process sensory input
- `integrate_sensory_data(graph: Data, sensory_data: Dict[str, Any]) -> Data`: Integrate sensory data

### IRealTimeVisualization

**Location**: `core/interfaces/real_time_visualization.py`

**Purpose**: Real-time visualization services.

**Key Methods**:
- `initialize_visualization(config: Dict[str, Any]) -> bool`: Initialize visualization
- `update_visualization_data(layer: str, data: VisualizationData) -> None`: Update visualization data
- `render_frame() -> None`: Render visualization frame
- `cleanup_visualization() -> None`: Clean up visualization resources

## Service Implementation Classes

### SimulationCoordinator

**Location**: `core/services/simulation_coordinator.py`

**Implements**: ISimulationCoordinator

**Dependencies**:
- IServiceRegistry
- IGraphManager
- IPerformanceMonitor
- IEventCoordinator
- INeuralProcessor
- IEnergyManager
- ILearningEngine
- ISensoryProcessor

### ServiceRegistry

**Location**: `core/services/service_registry.py`

**Implements**: IServiceRegistry

**Features**:
- Thread-safe service registration and resolution
- Dependency injection container
- Service lifecycle management (singleton/transient)
- Circular dependency detection
- Interface-based service contracts

### NeuralProcessingService

**Location**: `core/services/neural_processing_service.py`

**Implements**: INeuralProcessor

**Dependencies**:
- IGraphManager
- IEventCoordinator

**Key Responsibilities**:
- Neural dynamics processing and integration
- Node behavior updates and state management
- Spike generation and propagation
- Enhanced neural system coordination
- Neural metrics calculation and reporting

### EnergyManagementService

**Location**: `core/services/energy_management_service.py`

**Implements**: IEnergyManager

**Dependencies**:
- IGraphManager
- IEventCoordinator

**Key Responsibilities**:
- Energy flow and consumption tracking
- Membrane potential dynamics
- Refractory period management
- Homeostatic energy regulation
- Energy conservation logic and validation

### LearningService

**Location**: `core/services/learning_service.py`

**Implements**: ILearningEngine

**Dependencies**:
- INeuralProcessor
- IEnergyManager
- IEventCoordinator

**Key Responsibilities**:
- Hebbian learning implementation
- STDP (Spike-Timing Dependent Plasticity)
- Memory trace formation and consolidation
- Connection plasticity updates
- Energy-modulated learning rates

### SensoryProcessingService

**Location**: `core/services/sensory_processing_service.py`

**Implements**: ISensoryProcessor

**Dependencies**:
- IGraphManager
- IEventCoordinator

**Key Responsibilities**:
- Visual input processing and feature extraction
- Audio input integration and analysis
- Sensory pathway initialization
- Multi-modal sensory data integration
- Sensory-to-neural mapping and translation

### GraphManagementService

**Location**: `core/services/graph_management_service.py`

**Implements**: IGraphManager

**Dependencies**:
- IEventCoordinator

**Key Responsibilities**:
- Neural graph initialization and validation
- Node and edge lifecycle management
- Graph integrity checking and repair
- Versioning and persistence operations
- Graph merging and transformation

### PerformanceMonitoringService

**Location**: `core/services/performance_monitoring_service.py`

**Implements**: IPerformanceMonitor

**Dependencies**:
- IEventCoordinator

**Key Responsibilities**:
- Real-time performance metrics collection
- Memory usage and CPU utilization tracking
- Simulation performance benchmarking
- Historical metrics storage and analysis
- Performance alerting and reporting

### EventCoordinationService

**Location**: `core/services/event_coordination_service.py`

**Implements**: IEventCoordinator

**Dependencies**:
- IConfigurationService

**Key Responsibilities**:
- Event publishing and subscription management
- Asynchronous event processing and queuing
- Service communication coordination
- Event-driven workflow orchestration
- Event filtering and routing

### ConfigurationService

**Location**: `core/services/configuration_service.py`

**Implements**: IConfigurationService

**Dependencies**:
- IServiceRegistry

**Key Responsibilities**:
- Configuration file loading and parsing
- Runtime configuration updates and validation
- Parameter type checking and conversion
- Configuration persistence and backup
- Environment variable integration

## Usage Examples

### Service Registration
```python
from core.services.service_registry import ServiceRegistry
from core.services.simulation_coordinator import SimulationCoordinator

registry = ServiceRegistry()
coordinator = SimulationCoordinator()
registry.register_instance(ISimulationCoordinator, coordinator)
```

### Service Resolution
```python
coordinator = registry.resolve(ISimulationCoordinator)
success = coordinator.start_simulation()
```

### Interface Implementation
```python
from core.interfaces.simulation_coordinator import ISimulationCoordinator

class CustomSimulationCoordinator(ISimulationCoordinator):
    def initialize_simulation(self) -> bool:
        # Implementation
        return True
```

## Benefits of SOA Design

1. **Loose Coupling**: Services communicate through interfaces, not direct dependencies
2. **Testability**: Easy to mock services for unit testing
3. **Maintainability**: Changes to one service don't affect others
4. **Scalability**: Services can be distributed across processes or machines
5. **Flexibility**: Services can be swapped or extended without affecting the system

## Migration from Monolithic Design

The system has been fully migrated from a monolithic SimulationManager to SOA through Phase 1 and Phase 2:

### Phase 1: Service Foundation
- Established ServiceRegistry as dependency injection container
- Created interface contracts for all services
- Set up unified_launcher.py as composition root
- Migrated non-critical SimulationManager responsibilities

### Phase 2: Core Logic Migration
- Migrated all simulation logic from SimulationManager to services
- Implemented SimulationCoordinator as central orchestrator
- Removed all direct dependencies on SimulationManager
- Completed service integration and testing

### Before vs After

| Aspect | Monolithic (SimulationManager) | SOA Architecture |
|--------|-------------------------------|------------------|
| **Structure** | Single 2745+ line god object | 8 specialized services |
| **Dependencies** | Direct instantiation, tight coupling | Dependency injection, loose coupling |
| **Testing** | Difficult, required full system | Easy, services can be mocked |
| **Maintainability** | High risk of breaking changes | Isolated changes, clear boundaries |
| **Scalability** | Monolithic bottleneck | Services can scale independently |
| **Responsibilities** | All in one class | Single responsibility per service |

### Completed Migration Benefits
- **Separation of Concerns**: Each service has a clear, focused responsibility
- **Testability**: Services can be unit tested in isolation with mocked dependencies
- **Maintainability**: Changes to one service don't affect others
- **Scalability**: Services can be distributed across processes or machines
- **Flexibility**: Services can be swapped or extended without affecting the system
- **Reliability**: Fault isolation between services prevents cascading failures