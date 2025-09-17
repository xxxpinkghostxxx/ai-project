# AI Neural Simulation System - Component Reference

## Core Simulation Modules

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

### 2. Main Graph (`main_graph.py`)
**Purpose**: Graph creation and management utilities
**Key Functions**:
- `create_workspace_grid()`: Creates workspace grid structure
- `create_test_graph()`: Creates test neural graph
- `initialize_main_graph()`: Initializes main simulation graph
- `merge_graphs()`: Combines multiple graphs

### 3. Enhanced Neural Integration (`enhanced_neural_integration.py`)
**Purpose**: Advanced neural dynamics and integration
**Key Classes**:
- `EnhancedNeuralIntegration`: Main integration coordinator

**Key Methods**:
- `integrate_with_existing_system()`: Integrates enhanced systems
- `create_enhanced_node()`: Creates advanced neural nodes
- `create_enhanced_connection()`: Creates sophisticated connections

### 4. Neural Dynamics (`enhanced_neural_dynamics.py`)
**Purpose**: Advanced neural dynamics including STDP and plasticity
**Key Classes**:
- `EnhancedNeuralDynamics`: Main dynamics processor

**Key Methods**:
- `update_neural_dynamics()`: Updates neural dynamics
- `set_neuromodulator_level()`: Controls neuromodulation
- `_apply_stdp_learning()`: Applies STDP learning
- `_process_theta_bursts()`: Processes theta-burst stimulation

### 5. Learning Engine (`learning_engine.py`)
**Purpose**: Learning mechanisms and memory formation
**Key Classes**:
- `LearningEngine`: Main learning coordinator

**Key Methods**:
- `apply_timing_learning()`: Applies timing-based learning
- `consolidate_connections()`: Consolidates neural connections
- `form_memory_traces()`: Forms memory traces
- `apply_memory_influence()`: Applies memory influence

## Sensory Processing Modules

### 6. AudioToNeuralBridge (`audio_to_neural_bridge.py`)
**Purpose**: Converts audio input into neural sensory nodes
**Key Classes**:
- `AudioToNeuralBridge`: Main audio processing class

**Key Methods**:
- `process_audio_to_sensory_nodes()`: Converts audio to sensory nodes
- `integrate_audio_nodes_into_graph()`: Integrates audio nodes into neural graph
- `_extract_audio_features()`: Extracts audio features (MFCC, mel spectrogram)

### 7. VisualEnergyBridge (`visual_energy_bridge.py`)
**Purpose**: Processes visual input into enhanced neural energy
**Key Classes**:
- `VisualEnergyBridge`: Main visual processing class

**Key Methods**:
- `process_visual_to_enhanced_energy()`: Processes visual data to neural energy
- `_extract_visual_features()`: Extracts visual features
- `_detect_visual_patterns()`: Detects visual patterns

### 8. SensoryWorkspaceMapper (`sensory_workspace_mapper.py`)
**Purpose**: Maps sensory input to workspace nodes
**Key Classes**:
- `SensoryWorkspaceMapper`: Main mapping coordinator

**Key Methods**:
- `map_visual_to_workspace()`: Maps visual data to workspace
- `map_audio_to_workspace()`: Maps audio data to workspace
- `_extract_visual_patterns()`: Extracts visual patterns
- `_extract_audio_patterns()`: Extracts audio patterns

## System Management Modules

### 9. Performance Monitor (`utils/performance_monitor.py`)
**Purpose**: Real-time system performance monitoring
**Key Classes**:
- `PerformanceMonitor`: Main performance monitoring class
- `PerformanceMetrics`: Performance metrics data structure

**Key Methods**:
- `start_monitoring()`: Starts performance monitoring
- `record_step()`: Records simulation step metrics
- `get_current_metrics()`: Gets current performance metrics
- `get_performance_summary()`: Gets performance summary

### 10. Unified Error Handler (`utils/unified_error_handler.py`)
**Purpose**: Graceful error handling and recovery
**Key Classes**:
- `UnifiedErrorHandler`: Main error handling class

**Key Methods**:
- `handle_error()`: Handles errors with recovery
- `get_system_health()`: Gets system health status
- `add_error_callback()`: Adds error callbacks

### 11. Unified Config Manager (`config/unified_config_manager.py`)
**Purpose**: Centralized configuration management
**Key Classes**:
- `UnifiedConfigManager`: Main configuration manager

**Key Methods**:
- `get()`: Gets configuration values
- `set()`: Sets configuration values
- `get_system_constants()`: Gets system constants
- `get_learning_config()`: Gets learning configuration

## Utility Modules

### 12. Node Access Layer (`node_access_layer.py`)
**Purpose**: Safe node access and manipulation
**Key Classes**:
- `NodeAccessLayer`: Main node access coordinator

**Key Methods**:
- `get_node_by_id()`: Gets node by ID
- `select_nodes_by_type()`: Selects nodes by type
- `update_node_property()`: Updates node properties

### 13. Node ID Manager (`node_id_manager.py`)
**Purpose**: Node ID generation and management
**Key Classes**:
- `NodeIDManager`: Main ID management class

**Key Methods**:
- `generate_unique_id()`: Generates unique node IDs
- `register_node_index()`: Registers node indices
- `is_valid_id()`: Validates node IDs

### 14. Logging Utils (`logging_utils.py`)
**Purpose**: Centralized logging utilities
**Key Functions**:
- `log_step()`: Logs simulation steps
- `log_runtime()`: Logs runtime metrics
- `append_log_line()`: Appends log lines

## Event and Processing Modules

### 15. Event Driven System (`neural/event_driven_system.py`)
**Purpose**: Event-based processing architecture
**Key Classes**:
- `EventDrivenSystem`: Main event system coordinator
- `NeuralEvent`: Neural event data structure
- `EventQueue`: Event queue management

**Key Methods**:
- `process_events()`: Processes neural events
- `schedule_event()`: Schedules events
- `schedule_spike()`: Schedules spike events

### 16. Spike Queue System (`neural/spike_queue_system.py`)
**Purpose**: Spike event processing and propagation
**Key Classes**:
- `SpikeQueueSystem`: Main spike processing coordinator
- `Spike`: Spike data structure
- `SpikeQueue`: Spike queue management

**Key Methods**:
- `process_spikes()`: Processes spike events
- `schedule_spike()`: Schedules spike events
- `get_queue_size()`: Gets spike queue size

## UI and Visualization Modules

### 17. UI Engine (`ui_engine.py`)
**Purpose**: User interface and visualization
**Key Functions**:
- `create_main_window()`: Creates main UI window
- `update_ui_display()`: Updates UI display
- `create_ui()`: Creates UI components

### 18. UI State Manager (`ui_state_manager.py`)
**Purpose**: UI state management
**Key Classes**:
- `UIStateManager`: Main UI state coordinator

**Key Methods**:
- `update_graph()`: Updates graph for UI
- `add_live_feed_data()`: Adds live feed data
- `get_system_health()`: Gets system health for UI

## Testing and Analysis Modules

### 19. Unified Test Suite (`unified_test_suite.py`)
**Purpose**: Comprehensive testing framework
**Key Classes**:
- `UnifiedTestSuite`: Main test coordinator
- `TestResult`: Test result data structure

**Key Methods**:
- `run_test()`: Runs individual tests
- `get_summary()`: Gets test summary
- `test_critical_imports()`: Tests critical imports

### 20. NASA Code Analyzer (`nasa_code_analyzer.py`)
**Purpose**: NASA Power of Ten compliance analysis
**Key Classes**:
- `NASACodeAnalyzer`: Main analysis coordinator

**Key Methods**:
- `analyze_file()`: Analyzes file for compliance
- `analyze_directory()`: Analyzes directory for compliance
- `generate_report()`: Generates compliance report

## Consolidation Utilities

### 21. Print Utils (`print_utils.py`)
**Purpose**: Consolidated print utilities
**Key Functions**:
- `print_error()`: Prints error messages
- `print_warning()`: Prints warning messages
- `print_info()`: Prints info messages
- `print_success()`: Prints success messages

### 22. Exception Utils (`exception_utils.py`)
**Purpose**: Consolidated exception handling
**Key Functions**:
- `safe_execute()`: Safe function execution
- `safe_initialize_component()`: Safe component initialization
- `safe_process_step()`: Safe process step execution

### 23. Statistics Utils (`statistics_utils.py`)
**Purpose**: Consolidated statistics management
**Key Classes**:
- `StatisticsManager`: Main statistics coordinator

**Key Methods**:
- `register_module_stats()`: Registers module statistics
- `update_module_stats()`: Updates module statistics
- `get_module_stats()`: Gets module statistics

---

*This consolidated reference replaces the following files:*
- *COMPONENT_MAPPING.md*
- *NON_CORE_COMPONENT_MAPPING.md*
