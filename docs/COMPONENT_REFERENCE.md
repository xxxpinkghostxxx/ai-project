# AI Neural Simulation System - SOA Component Reference

## Core SOA Services

### 1. SimulationCoordinator (`core/services/simulation_coordinator.py`)
**Purpose**: Central orchestrator for all neural simulation services
**Interface**: `ISimulationCoordinator`
**Key Responsibilities**:
- Service dependency resolution through `IServiceRegistry`
- Simulation step orchestration across all services
- Graph state management through `IGraphManager`
- Event coordination through `IEventCoordinator`
- Performance monitoring via `IPerformanceMonitor`
- Configuration management through `IConfigurationService`

**Key Methods**:
- `initialize_simulation()`: Initializes simulation with all services
- `start_simulation()`: Starts simulation in thread
- `stop_simulation()`: Stops simulation
- `run_simulation_step()`: Executes one simulation step
- `get_simulation_state()`: Gets current simulation state
- `get_neural_graph()`: Gets neural graph from GraphManager
- `get_performance_metrics()`: Gets metrics from PerformanceMonitor

### 2. NeuralProcessingService (`core/services/neural_processing_service.py`)
**Purpose**: Handles neural dynamics and spiking behavior
**Interface**: `INeuralProcessor`
**Key Responsibilities**:
- Node behavior updates and state management
- Neural dynamics processing and integration
- Spike generation and propagation
- Enhanced neural system coordination
- Neural metrics calculation

**Key Methods**:
- `process_neural_dynamics(graph, step)`: Updates neural dynamics
- `update_node_behaviors(graph, step)`: Updates node behaviors
- `generate_spikes(graph, step)`: Generates neural spikes
- `calculate_neural_metrics(graph)`: Calculates network metrics

### 3. EnergyManagementService (`core/services/energy_management_service.py`)
**Purpose**: Manages energy flow and metabolic processes
**Interface**: `IEnergyManager`
**Key Responsibilities**:
- Energy flow and consumption tracking
- Membrane potential dynamics
- Refractory period management
- Homeostatic energy regulation
- Energy conservation logic

**Key Methods**:
- `update_energy_dynamics(graph, step)`: Updates energy states
- `apply_energy_behavior(graph)`: Applies energy behaviors
- `regulate_homeostasis(graph)`: Maintains energy balance
- `calculate_energy_consumption(graph)`: Calculates metabolic costs

### 4. LearningService (`core/services/learning_service.py`)
**Purpose**: Coordinates plasticity and learning mechanisms
**Interface**: `ILearningEngine`
**Key Responsibilities**:
- Hebbian learning implementation
- STDP (Spike-Timing Dependent Plasticity)
- Memory trace formation
- Connection consolidation
- Plasticity updates based on neural activity

**Key Methods**:
- `apply_learning(graph, step)`: Applies learning rules
- `update_synaptic_weights(graph)`: Updates connection weights
- `consolidate_memories(graph)`: Consolidates memory traces
- `apply_stdp(graph, spikes)`: Applies STDP learning

### 5. SensoryProcessingService (`core/services/sensory_processing_service.py`)
**Purpose**: Handles input processing and sensory data integration
**Interface**: `ISensoryProcessor`
**Key Responsibilities**:
- Visual input processing and feature extraction
- Audio input integration
- Sensory pathway initialization
- Multi-modal sensory data integration
- Sensory-to-neural mapping

**Key Methods**:
- `process_visual_input(visual_data)`: Processes visual sensory data
- `process_audio_input(audio_data)`: Processes audio sensory data
- `integrate_sensory_data(graph)`: Integrates sensory data into graph
- `map_sensory_to_neural(sensory_data)`: Maps sensory to neural activation

### 6. GraphManagementService (`core/services/graph_management_service.py`)
**Purpose**: Manages neural graph structure and operations
**Interface**: `IGraphManager`
**Key Responsibilities**:
- Graph initialization and validation
- Node and edge management
- Graph integrity checking and repair
- Versioning and persistence operations
- Graph merging and transformation

**Key Methods**:
- `initialize_graph()`: Initializes neural graph
- `set_graph(graph)`: Sets current graph
- `get_graph()`: Gets current graph
- `validate_graph(graph)`: Validates graph integrity
- `save_graph(slot)`: Saves graph to slot
- `load_graph(slot)`: Loads graph from slot

### 7. PerformanceMonitoringService (`core/services/performance_monitoring_service.py`)
**Purpose**: System performance monitoring and metrics collection
**Interface**: `IPerformanceMonitor`
**Key Responsibilities**:
- Real-time performance metrics collection
- Memory usage tracking
- CPU utilization monitoring
- Simulation performance benchmarking
- Historical metrics storage and analysis

**Key Methods**:
- `start_monitoring()`: Starts performance monitoring
- `record_step()`: Records simulation step metrics
- `get_current_metrics()`: Gets current performance metrics
- `get_performance_summary()`: Gets performance summary
- `get_cache_performance_stats()`: Gets cache statistics

### 8. EventCoordinationService (`core/services/event_coordination_service.py`)
**Purpose**: Event-driven communication between services
**Interface**: `IEventCoordinator`
**Key Responsibilities**:
- Event publishing and subscription management
- Asynchronous event processing
- Service communication coordination
- Event-driven workflow orchestration

**Key Methods**:
- `publish_event(event)`: Publishes event to subscribers
- `subscribe(event_type, handler)`: Subscribes to event type
- `process_events()`: Processes pending events
- `schedule_event(event, delay)`: Schedules delayed event

### 9. ConfigurationService (`core/services/configuration_service.py`)
**Purpose**: Centralized configuration management
**Interface**: `IConfigurationService`
**Key Responsibilities**:
- Configuration file loading and parsing
- Runtime configuration updates
- Parameter validation and type checking
- Configuration persistence

**Key Methods**:
- `get_config(key)`: Gets configuration value
- `set_config(key, value)`: Sets configuration value
- `load_config(file_path)`: Loads configuration from file
- `validate_config(config)`: Validates configuration

## SOA Infrastructure Components

### ServiceRegistry (`core/services/service_registry.py`)
**Purpose**: Dependency injection container for SOA services
**Interface**: `IServiceRegistry`
**Key Responsibilities**:
- Service registration and resolution by interface
- Singleton and transient service lifecycles
- Dependency validation and circular dependency detection
- Service initialization and cleanup
- Interface-based service contracts

**Key Methods**:
- `register_instance(interface_type, instance)`: Registers service instance
- `resolve(interface_type)`: Resolves service by interface
- `has_service(interface_type)`: Checks if service is registered
- `unregister(interface_type)`: Removes service registration

## Neural System Components

### 11. BehaviorEngine (`neural/behavior_engine.py`)
**Purpose**: Node behavior management and execution
**Key Classes**:
- `BehaviorEngine`: Main behavior coordinator

**Key Methods**:
- `update_node_behavior(node_id, graph, step, access_layer)`: Updates node behavior
- `get_behavior_types()`: Gets available behavior types
- `validate_behavior_config(behavior_config)`: Validates behavior configuration

### 12. ConnectionLogic (`neural/connection_logic.py`)
**Purpose**: Intelligent connection formation and management
**Key Functions**:
- `intelligent_connection_formation(graph)`: Creates intelligent connections
- `connection_pruning(graph, threshold)`: Prunes weak connections
- `connection_reinforcement(graph, activity_data)`: Reinforces active connections

### 13. NetworkMetrics (`neural/network_metrics.py`)
**Purpose**: Network analysis and metrics calculation
**Key Classes**:
- `NetworkMetrics`: Metrics calculator

**Key Methods**:
- `calculate_criticality(graph)`: Calculates network criticality
- `analyze_connectivity(graph)`: Analyzes connectivity patterns
- `calculate_clustering_coefficient(graph)`: Calculates clustering metrics

### 14. EventDrivenSystem (`neural/event_driven_system.py`)
**Purpose**: Event-based processing architecture
**Key Classes**:
- `EventDrivenSystem`: Event system coordinator
- `NeuralEvent`: Neural event data structure
- `EventQueue`: Event queue management

**Key Methods**:
- `process_events()`: Processes neural events
- `schedule_event(event)`: Schedules events
- `schedule_spike(spike)`: Schedules spike events

### 15. SpikeQueueSystem (`neural/spike_queue_system.py`)
**Purpose**: Spike event processing and propagation
**Key Classes**:
- `SpikeQueueSystem`: Spike processing coordinator
- `Spike`: Spike data structure
- `SpikeQueue`: Spike queue management

**Key Methods**:
- `process_spikes()`: Processes spike events
- `schedule_spike(spike)`: Schedules spike events
- `get_queue_size()`: Gets spike queue size

## Learning System Components

### 16. LearningEngine (`learning/learning_engine.py`)
**Purpose**: STDP and pattern learning mechanisms
**Key Classes**:
- `LearningEngine`: Learning coordinator

**Key Methods**:
- `apply_timing_learning(graph, spikes)`: Applies timing-based learning
- `consolidate_connections(graph)`: Consolidates neural connections
- `form_memory_traces(graph)`: Forms memory traces

### 17. LiveHebbianLearning (`learning/live_hebbian_learning.py`)
**Purpose**: Real-time learning with energy modulation
**Key Classes**:
- `LiveHebbianLearning`: Energy-modulated learning

**Key Methods**:
- `apply_continuous_learning(graph, step)`: Applies continuous learning
- `modulate_learning_by_energy(learning_rate, energy_level)`: Energy modulation
- `hebbian_update(weights, pre_activity, post_activity)`: Hebbian weight update

### 18. MemorySystem (`learning/memory_system.py`)
**Purpose**: Memory formation and persistence
**Key Classes**:
- `MemorySystem`: Memory management

**Key Methods**:
- `form_memory_traces(graph)`: Forms memory traces
- `consolidate_memories(graph)`: Consolidates memories
- `recall_pattern(graph, pattern)`: Recalls stored patterns

### 19. HomeostasisController (`learning/homeostasis_controller.py`)
**Purpose**: Energy balance regulation
**Key Classes**:
- `HomeostasisController`: Homeostasis regulator

**Key Methods**:
- `regulate_energy_balance(graph)`: Regulates energy balance
- `adjust_learning_rates(energy_levels)`: Adjusts learning based on energy
- `maintain_criticality(graph)`: Maintains network criticality

## Energy System Components

### 20. EnergyBehavior (`energy/energy_behavior.py`)
**Purpose**: Energy flow and consumption management
**Key Functions**:
- `apply_energy_behavior(graph, behavior_params=None)`: Applies energy behaviors
- `update_energy_with_learning(graph, node_id, delta_energy)`: Updates energy with learning
- `calculate_energy_flow(graph)`: Calculates energy flow dynamics

### 21. EnergyConstants (`energy/energy_constants.py`)
**Purpose**: Centralized energy parameters
**Key Constants**:
- `NODE_ENERGY_CAP`: Maximum node energy
- `ENERGY_DECAY_RATE`: Energy decay rate
- `METABOLIC_COST`: Metabolic energy cost

### 22. NodeAccessLayer (`energy/node_access_layer.py`)
**Purpose**: Safe node access and manipulation
**Key Classes**:
- `NodeAccessLayer`: Node access coordinator

**Key Methods**:
- `get_node_by_id(node_id)`: Gets node by ID
- `select_nodes_by_type(node_type)`: Selects nodes by type
- `update_node_property(node_id, property, value)`: Updates node properties

### 23. NodeIDManager (`energy/node_id_manager.py`)
**Purpose**: Node ID generation and management
**Key Classes**:
- `NodeIDManager`: ID management

**Key Methods**:
- `generate_unique_id()`: Generates unique node IDs
- `register_node_index(node_id, index)`: Registers node indices
- `is_valid_id(node_id)`: Validates node IDs

## Sensory System Components

### 24. VisualEnergyBridge (`sensory/visual_energy_bridge.py`)
**Purpose**: Processes visual input into enhanced neural energy
**Key Classes**:
- `VisualEnergyBridge`: Visual processing

**Key Methods**:
- `process_visual_to_enhanced_energy(graph, screen_data, step)`: Processes visual data
- `extract_visual_features(image_data)`: Extracts visual features
- `detect_visual_patterns(features)`: Detects visual patterns

### 25. AudioToNeuralBridge (`sensory/audio_to_neural_bridge.py`)
**Purpose**: Converts audio input into neural sensory nodes
**Key Classes**:
- `AudioToNeuralBridge`: Audio processing

**Key Methods**:
- `process_audio_to_sensory_nodes(audio_data)`: Converts audio to sensory nodes
- `integrate_audio_nodes_into_graph(graph, sensory_nodes)`: Integrates audio nodes
- `extract_audio_features(audio_data)`: Extracts audio features (MFCC, mel spectrogram)

### 26. SensoryWorkspaceMapper (`sensory/sensory_workspace_mapper.py`)
**Purpose**: Maps sensory input to workspace nodes
**Key Classes**:
- `SensoryWorkspaceMapper`: Sensory mapping

**Key Methods**:
- `map_visual_to_workspace(visual_data)`: Maps visual data to workspace
- `map_audio_to_workspace(audio_data)`: Maps audio data to workspace
- `extract_pattern_correlations(sensory_data)`: Extracts pattern correlations

## System Management Components

### 27. PerformanceMonitor (`utils/performance_monitor.py`)
**Purpose**: Real-time system performance monitoring
**Key Classes**:
- `PerformanceMonitor`: Performance monitoring
- `PerformanceMetrics`: Metrics data structure

**Key Methods**:
- `start_monitoring()`: Starts monitoring
- `record_step()`: Records step metrics
- `get_current_metrics()`: Gets current metrics
- `get_performance_summary()`: Gets summary

### 28. UnifiedErrorHandler (`utils/unified_error_handler.py`)
**Purpose**: Graceful error handling and recovery
**Key Classes**:
- `UnifiedErrorHandler`: Error handling

**Key Methods**:
- `handle_error(error, context)`: Handles errors
- `get_system_health()`: Gets system health
- `add_error_callback(callback)`: Adds error callbacks

### 29. UnifiedConfigManager (`config/unified_config_manager.py`)
**Purpose**: Centralized configuration management
**Key Classes**:
- `UnifiedConfigManager`: Configuration manager

**Key Methods**:
- `get(key)`: Gets configuration values
- `set(key, value)`: Sets configuration values
- `get_system_constants()`: Gets system constants
- `get_learning_config()`: Gets learning configuration

## UI and Visualization Components

### 30. UI Engine (`ui/ui_engine.py`)
**Purpose**: User interface and visualization
**Key Functions**:
- `create_main_window()`: Creates main UI window
- `update_ui_display(graph)`: Updates UI display
- `create_ui()`: Creates UI components

### 31. UI State Manager (`ui/ui_state_manager.py`)
**Purpose**: UI state management
**Key Classes**:
- `UIStateManager`: UI state coordinator

**Key Methods**:
- `update_graph(graph)`: Updates graph for UI
- `add_live_feed_data(data)`: Adds live feed data
- `get_system_health()`: Gets system health for UI

---

*This SOA component reference replaces the following outdated files:*
- *COMPONENT_MAPPING.md*
- *NON_CORE_COMPONENT_MAPPING.md*
