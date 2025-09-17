# Neural Simulation System - Component Mapping

## Overview
This document provides a comprehensive mapping of all useful components in the AI Neural Simulation System. All Python files are located in the main directory, with no subdirectories containing additional Python modules.

## 🧠 Core Simulation Modules

### 1. SimulationManager (`simulation_manager.py`)
**Purpose**: Central coordinator for all neural systems
**Key Classes**:
- `SimulationManager`: Main simulation orchestrator
**Key Methods**:
- `run_single_step()`: Executes one simulation step
- `start_simulation()`: Starts simulation in thread
- `stop_simulation()`: Stops simulation
- `initialize_graph()`: Initializes neural graph
- `create_enhanced_node()`: Creates enhanced neural nodes
- `create_enhanced_connection()`: Creates advanced connections
- `set_neuromodulator_level()`: Controls neuromodulation
**Useful Features**:
- Thread-safe simulation control
- Component lifecycle management
- Error recovery mechanisms
- Enhanced neural integration
- Real-time performance monitoring

### 2. Main Graph (`main_graph.py`)
**Purpose**: Graph creation and management utilities
**Key Functions**:
- `create_workspace_grid()`: Creates workspace grid structure
- `merge_graphs()`: Merges multiple graphs
- `initialize_main_graph()`: Initializes main simulation graph
**Useful Features**:
- Workspace grid creation
- Graph merging capabilities
- Resolution scaling support

## 🔬 Enhanced Neural System Modules

### 3. Enhanced Neural Integration (`enhanced_neural_integration.py`)
**Purpose**: Integrates all enhanced neural systems
**Key Classes**:
- `EnhancedNeuralIntegration`: Main integration coordinator
**Key Methods**:
- `integrate_with_existing_system()`: Integrates with existing graph
- `create_enhanced_node()`: Creates sophisticated nodes
- `create_enhanced_connection()`: Creates advanced connections
- `set_neuromodulator_level()`: Controls neuromodulation
**Useful Features**:
- Unified interface for enhanced systems
- Neural dynamics integration
- Connection system management
- Node behavior coordination

### 4. Enhanced Neural Dynamics (`enhanced_neural_dynamics.py`)
**Purpose**: Implements advanced neural dynamics including STDP and plasticity
**Key Classes**:
- `EnhancedNeuralDynamics`: Advanced neural dynamics engine
**Key Methods**:
- `update_neural_dynamics()`: Updates neural dynamics
- `set_neuromodulator_level()`: Controls neuromodulation
- `get_statistics()`: Returns dynamics statistics
**Useful Features**:
- STDP (Spike-Timing Dependent Plasticity)
- IEG (Immediate Early Gene) tagging
- Theta-burst stimulation
- Membrane dynamics simulation
- E/I balance control
- Memory consolidation
- Neuromodulatory control

### 5. Enhanced Node Behaviors (`enhanced_node_behaviors.py`)
**Purpose**: Sophisticated biological node behaviors
**Key Classes**:
- `EnhancedNodeBehavior`: Individual node behavior
- `EnhancedNodeBehaviorSystem`: Node behavior management
**Node Types**:
- **oscillator**: Rhythmic activity generation
- **integrator**: Information accumulation
- **relay**: Signal amplification
- **highway**: High-capacity energy distribution
- **workspace**: Imagination and flexible thinking
- **transmitter**: Signal transmission
- **resonator**: Frequency-specific responses
- **dampener**: Signal attenuation
**Useful Features**:
- Biologically-inspired behaviors
- Plasticity support
- Theta-burst detection
- IEG tagging
- Energy dynamics

### 6. Enhanced Connection System (`enhanced_connection_system.py`)
**Purpose**: Advanced connection management with plasticity
**Key Classes**:
- `EnhancedConnection`: Individual connection representation
- `EnhancedConnectionSystem`: Connection management system
**Connection Types**:
- **excitatory**: Positive signal transmission
- **inhibitory**: Negative signal transmission
- **modulatory**: Signal modulation
- **plastic**: STDP-enabled connections
- **burst**: Theta-burst connections
- **gated**: Neuromodulator-gated connections
**Useful Features**:
- Advanced connection types
- Weight plasticity
- Eligibility traces
- Neuromodulatory control
- Connection pruning

## ⚡ Energy Management Modules

### 7. Energy Behavior (`energy_behavior.py`)
**Purpose**: Energy flow and consumption management
**Key Classes**:
- `EnergyCalculator`: Static energy calculation utilities
**Key Functions**:
- `get_node_energy_cap()`: Gets energy capacity
- `update_node_energy_with_learning()`: Updates energy with learning
- `apply_energy_behavior()`: Applies energy dynamics
- `update_membrane_potentials()`: Updates membrane potentials
- `apply_refractory_periods()`: Applies refractory periods
**Useful Features**:
- Energy conservation
- Membrane potential dynamics
- Refractory period management
- Learning-based energy updates

### 8. Energy Constants (`energy_constants.py`)
**Purpose**: Energy-related constants and configuration
**Key Classes**:
- `EnergyConstants`: Core energy constants
- `ConnectionConstants`: Connection-related constants
- `OscillatorConstants`: Oscillator-specific constants
- `IntegratorConstants`: Integrator-specific constants
- `RelayConstants`: Relay-specific constants
- `HighwayConstants`: Highway-specific constants
**Useful Features**:
- Centralized constant management
- Configurable parameters
- Type-specific constants

### 9. Node Access Layer (`node_access_layer.py`)
**Purpose**: ID-based node operations
**Key Classes**:
- `NodeAccessLayer`: Node access interface
**Key Methods**:
- `get_node_by_id()`: Gets node by ID
- `get_node_energy()`: Gets node energy
- `set_node_energy()`: Sets node energy
- `update_node_property()`: Updates node properties
- `select_nodes_by_type()`: Selects nodes by type
- `select_nodes_by_behavior()`: Selects nodes by behavior
**Useful Features**:
- ID-based node access
- Property management
- Node selection and filtering
- Cache management

## 🧠 Learning and Memory Modules

### 10. Learning Engine (`learning_engine.py`)
**Purpose**: STDP and pattern learning
**Key Classes**:
- `LearningEngine`: Main learning coordinator
**Key Methods**:
- `apply_timing_learning()`: Applies STDP learning
- `consolidate_connections()`: Consolidates connections
- `form_memory_traces()`: Forms memory traces
- `apply_memory_influence()`: Applies memory influence
**Useful Features**:
- STDP learning implementation
- Pattern recognition
- Memory trace formation
- Connection consolidation

### 11. Memory System (`memory_system.py`)
**Purpose**: Memory formation and persistence
**Key Classes**:
- `MemorySystem`: Memory management system
**Key Methods**:
- `form_memory_traces()`: Forms memory traces
- `consolidate_memories()`: Consolidates memories
- `decay_memories()`: Decays old memories
- `recall_patterns()`: Recalls stored patterns
- `get_node_memory_importance()`: Gets memory importance
**Useful Features**:
- Memory trace formation
- Pattern consolidation
- Memory decay
- Pattern recall
- Importance scoring

### 12. Live Hebbian Learning (`live_hebbian_learning.py`)
**Purpose**: Real-time Hebbian learning system
**Key Classes**:
- `LiveHebbianLearning`: Real-time learning coordinator
**Key Methods**:
- `apply_continuous_learning()`: Applies continuous learning
- `set_learning_rate()`: Sets learning rate
- `set_learning_active()`: Activates/deactivates learning
- `get_learning_statistics()`: Gets learning statistics
**Useful Features**:
- Real-time learning
- Continuous adaptation
- Learning rate control
- Statistics tracking

## 🎯 Sensory Integration Modules

### 13. Visual Energy Bridge (`visual_energy_bridge.py`)
**Purpose**: Processes visual input into neural energy
**Key Classes**:
- `VisualEnergyBridge`: Visual processing system
**Key Methods**:
- `process_visual_to_enhanced_energy()`: Processes visual input
- `set_visual_sensitivity()`: Sets visual sensitivity
- `set_pattern_threshold()`: Sets pattern threshold
- `get_visual_statistics()`: Gets visual statistics
**Useful Features**:
- Visual feature extraction
- Pattern detection
- Energy conversion
- Visual memory formation
- Enhanced neural integration

### 14. Audio to Neural Bridge (`audio_to_neural_bridge.py`)
**Purpose**: Converts audio input to sensory nodes
**Key Classes**:
- `AudioToNeuralBridge`: Audio processing system
**Key Methods**:
- `process_audio_to_sensory_nodes()`: Processes audio input
- `integrate_audio_nodes_into_graph()`: Integrates audio nodes
- `get_audio_feature_statistics()`: Gets audio statistics
**Useful Features**:
- Audio feature extraction (MFCC, mel spectrogram)
- Sensory node creation
- Graph integration
- Fallback support for missing dependencies

### 15. Sensory Workspace Mapper (`sensory_workspace_mapper.py`)
**Purpose**: Maps sensory input to workspace nodes
**Key Classes**:
- `SensoryWorkspaceMapper`: Sensory mapping system
**Key Methods**:
- `map_visual_to_workspace()`: Maps visual to workspace
- `map_audio_to_workspace()`: Maps audio to workspace
- `get_mapping_statistics()`: Gets mapping statistics
**Useful Features**:
- Multi-modal sensory mapping
- Workspace organization
- Pattern-based connections
- Spatial organization

## 🔧 System Management Modules

### 16. Performance Monitor (`performance_monitor.py`)
**Purpose**: Real-time system performance monitoring
**Key Classes**:
- `PerformanceMonitor`: Main performance monitoring
- `PerformanceMetrics`: Performance data structure
- `PerformanceThresholds`: Performance thresholds
**Key Methods**:
- `start_monitoring()`: Starts performance monitoring
- `stop_monitoring()`: Stops monitoring
- `record_step()`: Records simulation step
- `get_current_metrics()`: Gets current metrics
- `get_performance_summary()`: Gets performance summary
**Useful Features**:
- Real-time monitoring
- Memory usage tracking
- CPU usage tracking
- GPU monitoring (optional)
- Alert system
- Performance thresholds

### 17. Error Handler (`error_handler.py`)
**Purpose**: Graceful error handling and recovery
**Key Classes**:
- `ErrorHandler`: Main error handling system
**Key Methods**:
- `handle_error()`: Handles errors
- `get_system_health()`: Gets system health
- `add_error_callback()`: Adds error callbacks
**Useful Functions**:
- `safe_execute()`: Safe function execution
- `error_handler_decorator()`: Error handling decorator
- `graceful_degradation()`: Graceful degradation
- `system_health_check()`: System health check
**Useful Features**:
- Error recovery mechanisms
- System health monitoring
- Callback system
- Graceful degradation
- Error statistics

### 18. Config Manager (`config_manager.py`)
**Purpose**: Centralized configuration management
**Key Classes**:
- `ConfigManager`: Main configuration manager
**Key Methods**:
- `get()`: Gets configuration value
- `set()`: Sets configuration value
- `save()`: Saves configuration
- `reload()`: Reloads configuration
- `get_system_constants()`: Gets system constants
- `get_learning_config()`: Gets learning configuration
**Useful Features**:
- Centralized configuration
- Type conversion
- Caching
- Validation
- Default value support

### 19. Node ID Manager (`node_id_manager.py`)
**Purpose**: Unique ID generation and management
**Key Classes**:
- `NodeIDManager`: ID management system
**Key Methods**:
- `generate_unique_id()`: Generates unique IDs
- `register_node_index()`: Registers node index
- `get_node_index()`: Gets node index
- `is_valid_id()`: Validates ID
- `recycle_node_id()`: Recycles node ID
- `can_expand_graph()`: Checks expansion capacity
- `cleanup_orphaned_ids()`: Cleans up orphaned IDs
**Useful Features**:
- Unique ID generation
- ID recycling
- Graph expansion limits
- Orphaned ID cleanup
- Statistics tracking

### 20. Random Seed Manager (`random_seed_manager.py`)
**Purpose**: Random seed management
**Key Classes**:
- `RandomSeedManager`: Seed management system
**Key Methods**:
- `initialize()`: Initializes seeds
- `get_seed()`: Gets current seed
- `set_seed()`: Sets seed
- `increment_seed()`: Increments seed
- `reset_to_base()`: Resets to base seed
**Useful Functions**:
- `random_choice()`: Random choice
- `random_uniform()`: Random uniform
- `random_normal()`: Random normal
- `random_int()`: Random integer
**Useful Features**:
- Reproducible randomness
- Seed management
- Random number generation
- State management

## 🎨 UI and Interface Modules

### 21. UI Engine (`ui_engine.py`)
**Purpose**: Main UI engine
**Key Functions**:
- `create_main_window()`: Creates main window
- `update_ui_display()`: Updates UI display
- `create_ui()`: Creates UI
- `show_legend_help()`: Shows help
**Useful Features**:
- DearPyGui integration
- Real-time updates
- Status display
- Control buttons

### 22. UI State Manager (`ui_state_manager.py`)
**Purpose**: UI state management
**Key Classes**:
- `UIStateManager`: UI state management
**Key Methods**:
- `get_simulation_state()`: Gets simulation state
- `set_simulation_running()`: Sets simulation state
- `update_graph()`: Updates graph
- `add_live_feed_data()`: Adds live feed data
- `get_system_health()`: Gets system health
**Useful Features**:
- Thread-safe state management
- Live feed data
- System health monitoring
- Graph updates

### 23. Minimal UI (`minimal_ui.py`)
**Purpose**: Minimal UI implementation
**Key Classes**:
- `MinimalUI`: Minimal UI system
**Key Methods**:
- `create_ui()`: Creates minimal UI
- `start_simulation()`: Starts simulation
- `stop_simulation()`: Stops simulation
- `reset_simulation()`: Resets simulation
**Useful Features**:
- Simple interface
- Basic controls
- Live statistics
- Log display

## 🧪 Test and Example Modules

### 24. Unified Test Suite (`unified_test_suite.py`)
**Purpose**: Comprehensive testing system
**Key Classes**:
- `UnifiedTestSuite`: Main test suite
- `TestResult`: Test result data structure
**Key Functions**:
- `test_critical_imports()`: Tests critical imports
- `test_memory_usage()`: Tests memory usage
- `test_simulation_manager_creation()`: Tests simulation manager
- `test_single_simulation_step()`: Tests simulation step
- `test_ui_components()`: Tests UI components
- `test_performance_monitoring()`: Tests performance monitoring
- `test_error_handling()`: Tests error handling
- `run_unified_tests()`: Runs all tests
**Useful Features**:
- Comprehensive testing
- Performance testing
- Error testing
- UI testing
- Statistics tracking

### 25. Enhanced Neural Example (`enhanced_neural_example.py`)
**Purpose**: Enhanced neural system demonstrations
**Key Functions**:
- `create_enhanced_neural_simulation()`: Creates enhanced simulation
- `demonstrate_advanced_features()`: Demonstrates advanced features
- `create_biological_neural_network()`: Creates biological network
**Useful Features**:
- STDP learning demonstration
- IEG tagging demonstration
- Theta-burst detection
- Membrane dynamics
- E/I balance control
- Advanced connection types
- Neuromodulatory control

### 26. Visual Energy Integration Example (`visual_energy_integration_example.py`)
**Purpose**: Visual energy integration demonstrations
**Key Functions**:
- `create_test_graph()`: Creates test graph
- `create_test_visual_data()`: Creates test visual data
- `demonstrate_visual_energy_integration()`: Demonstrates integration
- `demonstrate_visual_pattern_recognition()`: Demonstrates pattern recognition
**Useful Features**:
- Visual feature extraction
- Pattern detection
- Enhanced neural integration
- Real-time processing
- Multi-modal integration

### 27. Sensory Workspace Mapping Example (`sensory_workspace_mapping_example.py`)
**Purpose**: Sensory workspace mapping demonstrations
**Key Functions**:
- `create_test_graph_with_workspace()`: Creates test graph
- `create_test_visual_patterns()`: Creates visual patterns
- `create_test_audio_patterns()`: Creates audio patterns
- `demonstrate_sensory_workspace_mapping()`: Demonstrates mapping
- `demonstrate_workspace_concept_formation()`: Demonstrates concept formation
**Useful Features**:
- Visual pattern mapping
- Audio pattern mapping
- Workspace organization
- Concept formation
- Multi-modal integration

### 28. Test Simulation Progression (`test_simulation_progression.py`, `test_simulation_progression_fixed.py`)
**Purpose**: Simulation progression testing
**Key Functions**:
- `test_simulation_progression()`: Tests simulation progression
- `test_energy_behavior()`: Tests energy behavior
- `test_connection_logic()`: Tests connection logic
**Useful Features**:
- Simulation testing
- Energy behavior testing
- Connection logic testing
- Progression validation

### 29. Test Simple Simulation (`test_simple_simulation.py`)
**Purpose**: Simple simulation testing
**Key Functions**:
- `test_simple_simulation()`: Tests simple simulation
**Useful Features**:
- Basic simulation testing
- Simple validation

## 🔧 Utility Modules

### 30. Connection Logic (`connection_logic.py`)
**Purpose**: Connection creation and management
**Key Classes**:
- `EnhancedEdge`: Enhanced edge representation
**Key Functions**:
- `create_weighted_connection()`: Creates weighted connections
- `get_edge_attributes()`: Gets edge attributes
- `apply_weight_change()`: Applies weight changes
- `create_basic_connections()`: Creates basic connections
- `intelligent_connection_formation()`: Creates intelligent connections
- `update_connection_weights()`: Updates connection weights
**Useful Features**:
- Weighted connections
- Edge attributes
- Weight plasticity
- Intelligent connection formation
- Connection statistics

### 31. Death and Birth Logic (`death_and_birth_logic.py`)
**Purpose**: Node death and birth management
**Key Functions**:
- `handle_node_death()`: Handles node death
- `handle_node_birth()`: Handles node birth
- `remove_node_from_graph()`: Removes node from graph
- `create_new_node()`: Creates new node
- `add_node_to_graph()`: Adds node to graph
- `remove_dead_dynamic_nodes()`: Removes dead nodes
- `birth_new_dynamic_nodes()`: Creates new nodes
**Useful Features**:
- Node lifecycle management
- Graph consistency
- Energy-based birth/death
- Memory pattern analysis

### 32. Dynamic Nodes (`dynamic_nodes.py`)
**Purpose**: Dynamic node management
**Key Functions**:
- `add_dynamic_nodes()`: Adds dynamic nodes
**Useful Features**:
- Dynamic node creation
- Node type management

### 33. Homeostasis Controller (`homeostasis_controller.py`)
**Purpose**: Network homeostasis and regulation
**Key Classes**:
- `HomeostasisController`: Homeostasis management
**Key Methods**:
- `regulate_network_activity()`: Regulates network activity
- `optimize_criticality()`: Optimizes criticality
- `monitor_network_health()`: Monitors network health
- `get_regulation_statistics()`: Gets regulation statistics
**Useful Features**:
- Energy regulation
- Criticality optimization
- Network health monitoring
- Homeostatic balance

### 34. Network Metrics (`network_metrics.py`)
**Purpose**: Network analysis and metrics
**Key Classes**:
- `NetworkMetrics`: Network metrics system
**Key Methods**:
- `calculate_criticality()`: Calculates criticality
- `analyze_connectivity()`: Analyzes connectivity
- `measure_energy_balance()`: Measures energy balance
- `calculate_comprehensive_metrics()`: Calculates comprehensive metrics
- `get_network_health_score()`: Gets health score
**Useful Features**:
- Criticality analysis
- Connectivity analysis
- Energy balance measurement
- Health scoring
- Trend analysis

### 35. Neural Map Persistence (`neural_map_persistence.py`)
**Purpose**: Neural map saving and loading
**Key Classes**:
- `NeuralMapPersistence`: Map persistence system
**Key Methods**:
- `save_neural_map()`: Saves neural map
- `load_neural_map()`: Loads neural map
- `delete_neural_map()`: Deletes neural map
- `list_available_slots()`: Lists available slots
- `get_slot_info()`: Gets slot information
**Useful Features**:
- Map serialization
- Slot management
- Metadata tracking
- Version control

### 36. Workspace Engine (`workspace_engine.py`)
**Purpose**: Workspace node management
**Key Classes**:
- `WorkspaceEngine`: Workspace management
**Key Methods**:
- `update_workspace_nodes()`: Updates workspace nodes
- `create_workspace_node()`: Creates workspace node
- `get_workspace_metrics()`: Gets workspace metrics
**Useful Features**:
- Workspace node management
- Metrics tracking
- Node creation

### 37. Event Driven System (`event_driven_system.py`)
**Purpose**: Event-based processing architecture
**Key Classes**:
- `EventDrivenSystem`: Main event system
- `EventQueue`: Event queue management
- `EventProcessor`: Event processing
- `NeuralEvent`: Event data structure
**Key Methods**:
- `start()`: Starts event system
- `stop()`: Stops event system
- `process_events()`: Processes events
- `schedule_event()`: Schedules event
- `schedule_spike()`: Schedules spike
**Useful Features**:
- Event scheduling
- Priority queues
- Event processing
- Spike scheduling

### 38. Spike Queue System (`spike_queue_system.py`)
**Purpose**: Spike queue management
**Key Classes**:
- `SpikeQueueSystem`: Main spike system
- `SpikeQueue`: Spike queue management
- `SpikePropagator`: Spike propagation
- `Spike`: Spike data structure
**Key Methods**:
- `start()`: Starts spike system
- `stop()`: Stops spike system
- `schedule_spike()`: Schedules spike
- `process_spikes()`: Processes spikes
- `get_queue_size()`: Gets queue size
**Useful Features**:
- Spike scheduling
- Queue management
- Propagation
- Statistics tracking

### 39. Screen Graph (`screen_graph.py`)
**Purpose**: Screen capture and graph creation
**Key Functions**:
- `capture_screen()`: Captures screen
- `create_pixel_gray_graph()`: Creates pixel graph
- `rgb_to_gray()`: Converts RGB to grayscale
**Useful Features**:
- Screen capture
- Graph creation
- Image processing

### 40. Logging Utils (`logging_utils.py`)
**Purpose**: Logging utilities
**Key Functions**:
- `log_step()`: Logs simulation steps
- `log_node_state()`: Logs node state
- `append_log_line()`: Appends log line
- `get_log_lines()`: Gets log lines
- `setup_logging()`: Sets up logging
**Useful Features**:
- Step logging
- Node state logging
- Log management
- UI integration

### 41. Unified Launcher (`unified_launcher.py`)
**Purpose**: Unified system launcher
**Key Classes**:
- `UnifiedLauncher`: Main launcher
**Key Methods**:
- `test_basic_imports()`: Tests imports
- `apply_performance_optimizations()`: Applies optimizations
- `test_system_capacity()`: Tests system capacity
- `launch_with_profile()`: Launches with profile
**Useful Features**:
- Import testing
- Performance optimization
- System capacity testing
- Profile-based launching

## 🎯 Key Features Summary

### Core Capabilities
- **Biologically-inspired neural networks** with energy-based dynamics
- **Real-time visualization** using DearPyGui
- **Advanced learning mechanisms** including STDP, Hebbian learning, and memory formation
- **Event-driven architecture** for efficient processing
- **Multi-modal sensory integration** (visual and audio)
- **Homeostatic regulation** and criticality maintenance

### Advanced Features
- **STDP Learning**: Spike-timing dependent plasticity
- **IEG Tagging**: Immediate early gene expression for plasticity gating
- **Theta-burst Stimulation**: High-frequency stimulation for LTP induction
- **Membrane Dynamics**: Sophisticated somatic and dendritic compartments
- **E/I Balance Control**: Excitatory/inhibitory balance maintenance
- **Neuromodulatory Control**: Dopamine, acetylcholine, norepinephrine
- **Memory Consolidation**: Pattern-based memory formation
- **Enhanced Node Types**: Oscillators, integrators, relays, highways, workspace
- **Advanced Connection Types**: Excitatory, inhibitory, modulatory, plastic, gated
- **Real-time Performance Monitoring**: CPU, memory, GPU monitoring
- **Error Recovery**: Graceful error handling and recovery
- **Configuration Management**: Centralized configuration system
- **ID-based Architecture**: Unique ID generation and management

### Testing and Examples
- **Comprehensive Test Suite**: Full system testing
- **Example Demonstrations**: Visual, audio, and enhanced neural examples
- **Performance Testing**: Memory and performance validation
- **Error Testing**: Error handling validation
- **UI Testing**: Interface component testing

This component mapping provides a complete overview of all useful components in the neural simulation system, organized by functionality and purpose.
