# Non-Core Neural Simulation Components - Comprehensive Mapping

## Overview
This document maps all Python files that are NOT part of the main simulation core, identifying useful components, classes, and functions for system management, testing, UI, and utilities.

## 🎵 Audio & Visual Processing Modules

### 1. AudioToNeuralBridge (`audio_to_neural_bridge.py`)
**Purpose**: Converts audio input into neural sensory nodes
**Key Classes**:
- `AudioToNeuralBridge`: Main audio processing class

**Key Methods**:
- `process_audio_to_sensory_nodes(audio_data)`: Converts audio to sensory nodes
- `integrate_audio_nodes_into_graph(graph, audio_data)`: Integrates audio nodes into neural graph
- `_extract_audio_features(audio_data)`: Extracts MFCC, mel spectrogram, spectral features
- `_extract_mfcc(audio_data)`: Extracts Mel-frequency cepstral coefficients
- `_extract_mel_spectrogram(audio_data)`: Extracts mel spectrogram features
- `_extract_spectral_features(audio_data)`: Extracts spectral centroids, rolloff, ZCR
- `_extract_temporal_features(audio_data)`: Extracts RMS, bandwidth features

**Useful Features**:
- Real-time audio processing with librosa integration
- Fallback support for missing dependencies
- Audio feature caching and mapping
- Enhanced neural integration support

### 2. VisualEnergyBridge (`visual_energy_bridge.py`)
**Purpose**: Processes visual input into enhanced neural energy
**Key Classes**:
- `VisualEnergyBridge`: Main visual processing class

**Key Methods**:
- `process_visual_to_enhanced_energy(graph, screen_data, step)`: Main processing pipeline
- `_extract_visual_features(screen_data)`: Extracts visual features from screen data
- `_detect_visual_patterns(visual_features, step)`: Detects patterns in visual data
- `_convert_visual_to_enhanced_energy(graph, visual_features, step)`: Converts to neural energy
- `_process_visual_patterns(graph, visual_patterns, step)`: Processes detected patterns
- `_update_visual_memory(graph, visual_patterns, step)`: Updates visual memory
- `_propagate_visual_energy(graph, step)`: Propagates visual energy through network

**Useful Features**:
- Visual pattern recognition and memory
- Enhanced neural integration
- Configurable sensitivity and thresholds
- Real-time visual processing

### 3. SensoryWorkspaceMapper (`sensory_workspace_mapper.py`)
**Purpose**: Maps sensory input to workspace nodes for concept formation
**Key Classes**:
- `SensoryWorkspaceMapper`: Main sensory mapping class

**Key Methods**:
- `map_visual_to_workspace(graph, visual_data, step)`: Maps visual patterns to workspace
- `map_audio_to_workspace(graph, audio_data, step)`: Maps audio patterns to workspace
- `_extract_visual_patterns(visual_data, step)`: Extracts visual patterns
- `_extract_audio_patterns(audio_data, step)`: Extracts audio patterns
- `_map_patterns_to_workspace(patterns, pattern_type)`: Maps patterns to workspace regions
- `_update_workspace_nodes(graph, workspace_updates, access_layer)`: Updates workspace nodes
- `_create_sensory_workspace_connections(graph, patterns, access_layer)`: Creates connections

**Useful Features**:
- Workspace region mapping (visual_center, audio_low, motion, texture)
- Concept formation and pattern recognition
- Sensory-to-workspace connection creation
- Configurable workspace size and sensitivity

### 4. ScreenGraph (`screen_graph.py`)
**Purpose**: Captures screen data and creates pixel-based graphs
**Key Functions**:
- `capture_screen(scale=1.0)`: Captures screen using mss library
- `create_pixel_gray_graph(arr)`: Creates graph from grayscale pixel data
- `rgb_to_gray(arr)`: Converts RGB to grayscale

**Useful Features**:
- Real-time screen capture
- Grayscale conversion for neural processing
- Configurable scaling for performance

## 🔧 System Management & Utilities

### 5. PerformanceMonitor (`performance_monitor.py`)
**Purpose**: Real-time system performance monitoring and alerting
**Key Classes**:
- `PerformanceMonitor`: Main monitoring class
- `PerformanceMetrics`: Data class for performance metrics
- `PerformanceThresholds`: Data class for alert thresholds

**Key Methods**:
- `start_monitoring()`: Starts background monitoring thread
- `stop_monitoring()`: Stops monitoring
- `record_step(step_time, node_count, edge_count)`: Records simulation step metrics
- `get_current_metrics()`: Gets current performance metrics
- `get_performance_summary()`: Gets comprehensive performance summary
- `_update_memory_metrics()`: Updates memory usage metrics
- `_update_cpu_metrics()`: Updates CPU usage metrics
- `_update_gpu_metrics()`: Updates GPU usage metrics
- `_check_thresholds()`: Checks performance thresholds and triggers alerts

**Useful Features**:
- Multi-threaded monitoring with psutil/GPUtil integration
- Configurable thresholds and alerting
- Memory, CPU, GPU, and network monitoring
- Performance history tracking
- Automatic garbage collection triggers

### 6. ErrorHandler (`error_handler.py`)
**Purpose**: Graceful error handling and recovery system
**Key Classes**:
- `ErrorHandler`: Main error handling class

**Key Methods**:
- `handle_error(error, context, recovery_func, critical)`: Main error handling method
- `_update_system_health(error_type, critical)`: Updates system health status
- `_attempt_recovery(error_key, recovery_func, original_error)`: Attempts error recovery
- `add_error_callback(callback)`: Adds error callback
- `get_system_health()`: Gets current system health status
- `reset_error_counts()`: Resets error counters

**Useful Features**:
- Error categorization and counting
- Recovery function support
- System health tracking (healthy, warning, degraded, critical, failed)
- Callback system for error handling
- Cooldown periods to prevent error spam

### 7. ConfigManager (`config_manager.py`)
**Purpose**: Centralized configuration management with caching
**Key Classes**:
- `ConfigManager`: Main configuration management class

**Key Methods**:
- `get(section, key, default, value_type)`: Gets configuration value with type conversion
- `set(section, key, value)`: Sets configuration value
- `save()`: Saves configuration to file
- `reload()`: Reloads configuration from file
- `get_section(section)`: Gets entire configuration section
- `get_system_constants()`: Gets system constants
- `get_learning_config()`: Gets learning configuration
- `get_homeostasis_config()`: Gets homeostasis configuration
- `_validate_config_file_path()`: Validates config file path security
- `_create_default_config()`: Creates default configuration

**Useful Features**:
- Type-safe configuration access (str, int, float, bool)
- Section-based caching with TTL
- Security validation for config file paths
- Default configuration generation
- Multiple specialized getter methods

### 8. RandomSeedManager (`random_seed_manager.py`)
**Purpose**: Centralized random seed management for reproducible simulations
**Key Classes**:
- `RandomSeedManager`: Main seed management class

**Key Methods**:
- `initialize(seed)`: Initializes random seeds
- `get_seed()`: Gets current seed
- `set_seed(seed)`: Sets new seed
- `increment_seed(increment)`: Increments current seed
- `reset_to_base()`: Resets to base seed
- `get_random_state()`: Gets current random state
- `set_random_state(state)`: Sets random state

**Useful Features**:
- Reproducible random number generation
- State preservation and restoration
- Multiple random number generators (numpy, random)
- Environment variable support for base seed

### 9. NeuralMapPersistence (`neural_map_persistence.py`)
**Purpose**: Save and load neural network states
**Key Classes**:
- `NeuralMapPersistence`: Main persistence class

**Key Methods**:
- `save_neural_map(graph, slot_number, metadata)`: Saves neural map to slot
- `load_neural_map(slot_number)`: Loads neural map from slot
- `delete_neural_map(slot_number)`: Deletes neural map from slot
- `list_available_slots()`: Lists all available slots
- `get_slot_info(slot_number)`: Gets slot metadata
- `set_current_slot(slot_number)`: Sets current active slot
- `_serialize_graph(graph)`: Serializes graph for storage
- `_deserialize_graph(map_data)`: Deserializes graph from storage

**Useful Features**:
- Slot-based neural map storage
- Metadata tracking for each slot
- Graph serialization/deserialization
- Current slot management
- Automatic directory creation

## 🎯 Event & Spike Processing

### 10. EventDrivenSystem (`event_driven_system.py`)
**Purpose**: Event-based processing architecture for neural events
**Key Classes**:
- `EventDrivenSystem`: Main event system class
- `EventQueue`: Priority queue for events
- `EventProcessor`: Event processing engine
- `NeuralEvent`: Event data class

**Key Enums**:
- `EventType`: SPIKE, SYNAPTIC_TRANSMISSION, PLASTICITY_UPDATE, MEMORY_FORMATION, etc.

**Key Methods**:
- `start()`: Starts event processing
- `stop()`: Stops event processing
- `process_events(max_events)`: Processes queued events
- `schedule_event(event)`: Schedules new event
- `schedule_spike(node_id, timestamp, priority)`: Schedules spike event
- `schedule_energy_transfer(source_id, target_id, amount, timestamp)`: Schedules energy transfer

**Useful Features**:
- Priority-based event queue
- Multiple event types with specialized handlers
- Time-based event scheduling
- Event statistics and monitoring
- Thread-safe event processing

### 11. SpikeQueueSystem (`spike_queue_system.py`)
**Purpose**: High-performance spike propagation and queuing system
**Key Classes**:
- `SpikeQueueSystem`: Main spike system class
- `SpikeQueue`: Priority queue for spikes
- `SpikePropagator`: Spike propagation engine
- `Spike`: Spike data class

**Key Enums**:
- `SpikeType`: EXCITATORY, INHIBITORY, MODULATORY, BURST, SINGLE

**Key Methods**:
- `start()`: Starts spike processing
- `stop()`: Stops spike processing
- `schedule_spike(source_id, target_id, spike_type, amplitude, weight, timestamp)`: Schedules spike
- `process_spikes(max_spikes)`: Processes queued spikes
- `get_queue_size()`: Gets current queue size
- `clear_queue()`: Clears spike queue

**Useful Features**:
- High-performance spike queuing
- Multiple spike types with different properties
- Refractory period management
- Cascading spike detection
- Spike propagation statistics

### 12. LoggingUtils (`logging_utils.py`)
**Purpose**: Centralized logging utilities and UI integration
**Key Classes**:
- `UILogHandler`: Custom logging handler for UI integration

**Key Functions**:
- `setup_logging(ui_callback, level)`: Sets up logging system
- `log_step(step_desc, **kwargs)`: Logs simulation steps
- `log_node_state(node_label, prefix)`: Logs node state information
- `append_log_line(line)`: Appends line to log
- `get_log_lines()`: Gets all log lines
- `log_runtime(func)`: Decorator for runtime logging

**Useful Features**:
- UI-integrated logging
- Step-by-step simulation logging
- Node state logging
- Runtime performance logging
- Centralized log management

## 🖥️ User Interface Modules

### 13. UIEngine (`ui_engine.py`)
**Purpose**: Main UI engine with DearPyGui integration
**Key Functions**:
- `create_main_window()`: Creates main UI window
- `create_ui()`: Creates complete UI interface
- `update_ui_display()`: Updates UI display
- `reset_simulation()`: Resets simulation
- `show_legend_help()`: Shows help legend
- `get_simulation_running()`: Gets simulation state
- `set_simulation_running(running)`: Sets simulation state

**Useful Features**:
- DearPyGui-based interface
- Real-time simulation control
- Status display and monitoring
- Help system integration

### 14. UIStateManager (`ui_state_manager.py`)
**Purpose**: Thread-safe UI state management
**Key Classes**:
- `UIStateManager`: Main state management class

**Key Methods**:
- `get_simulation_state()`: Gets current simulation state
- `set_simulation_running(running)`: Sets simulation running state
- `update_graph(graph)`: Updates current graph
- `get_latest_graph()`: Gets latest graph
- `get_latest_graph_for_ui()`: Gets UI-optimized graph
- `add_live_feed_data(data_type, value)`: Adds live data feed
- `get_live_feed_data()`: Gets live data feed
- `update_system_health(health_data)`: Updates system health

**Useful Features**:
- Thread-safe state management
- Live data feed system
- System health monitoring
- Graph state management
- UI update coordination

### 15. MinimalUI (`minimal_ui.py`)
**Purpose**: Lightweight UI for basic simulation control
**Key Classes**:
- `MinimalUI`: Minimal UI implementation

**Key Methods**:
- `create_ui()`: Creates minimal UI interface
- `start_simulation()`: Starts simulation
- `stop_simulation()`: Stops simulation
- `reset_simulation()`: Resets simulation
- `_on_simulation_step(graph, step, perf_stats)`: Handles simulation step updates
- `_update_ui_callback()`: Updates UI display
- `_update_stats_display()`: Updates statistics display

**Useful Features**:
- Lightweight DearPyGui interface
- Basic simulation control
- Real-time statistics display
- Minimal resource usage

## 🧪 Testing & Examples

### 16. UnifiedTestSuite (`unified_test_suite.py`)
**Purpose**: Comprehensive test suite for system validation
**Key Classes**:
- `UnifiedTestSuite`: Main test suite class
- `TestResult`: Test result data class

**Key Methods**:
- `run_test(test_name, test_func, *args, **kwargs)`: Runs individual test
- `add_result(result)`: Adds test result
- `get_summary()`: Gets test summary statistics

**Key Test Functions**:
- `test_critical_imports()`: Tests all critical module imports
- `test_memory_usage()`: Tests memory usage and limits
- `test_simulation_manager_creation()`: Tests simulation manager creation
- `test_single_simulation_step()`: Tests single simulation step
- `test_ui_components()`: Tests UI component functionality
- `test_performance_monitoring()`: Tests performance monitoring
- `test_error_handling()`: Tests error handling system

**Useful Features**:
- Comprehensive system testing
- Import validation
- Performance testing
- Error handling validation
- Detailed test reporting

### 17. UnifiedLauncher (`unified_launcher.py`)
**Purpose**: Unified launcher with multiple profiles and modes
**Key Classes**:
- `UnifiedLauncher`: Main launcher class

**Key Methods**:
- `test_basic_imports()`: Tests basic system imports
- `apply_performance_optimizations()`: Applies performance optimizations
- `test_system_capacity()`: Tests system capacity
- `launch_with_profile(profile)`: Launches with specific profile
- `_launch_test_suite()`: Launches test suite
- `_launch_ui(config)`: Launches UI with configuration
- `show_help()`: Shows help information

**Available Profiles**:
- `minimal`: Minimal UI with basic functionality
- `full`: Full UI with all features
- `optimized`: Optimized UI with performance tuning
- `test`: Run test suite

**Useful Features**:
- Multiple launch profiles
- Performance optimization
- System capacity testing
- Command-line interface
- Help system

### 18. Example Modules
**Purpose**: Demonstration and example implementations

#### EnhancedNeuralExample (`enhanced_neural_example.py`)
- `create_enhanced_neural_simulation()`: Creates enhanced simulation
- `demonstrate_advanced_features()`: Demonstrates advanced features
- `create_biological_neural_network()`: Creates biological network

#### VisualEnergyIntegrationExample (`visual_energy_integration_example.py`)
- `create_test_graph()`: Creates test graph
- `create_test_visual_data()`: Creates test visual data
- `demonstrate_visual_energy_integration()`: Demonstrates visual integration
- `demonstrate_visual_pattern_recognition()`: Demonstrates pattern recognition

#### SensoryWorkspaceMappingExample (`sensory_workspace_mapping_example.py`)
- `create_test_graph_with_workspace()`: Creates test graph with workspace
- `create_test_visual_patterns()`: Creates test visual patterns
- `create_test_audio_patterns()`: Creates test audio patterns
- `demonstrate_sensory_workspace_mapping()`: Demonstrates sensory mapping
- `demonstrate_workspace_concept_formation()`: Demonstrates concept formation

## 🔧 Maintenance & Fix Utilities

### 19. FixSyntaxErrors (`fix_syntax_errors.py`)
**Purpose**: Automated syntax error fixing for Python files
**Key Functions**:
- `fix_docstring_issues(content)`: Fixes docstring formatting issues
- `fix_module_docstring(content)`: Fixes missing module docstrings
- `fix_file(filename)`: Fixes syntax errors in single file
- `main()`: Main function to fix all Python files

**Useful Features**:
- Automated docstring fixing
- Module docstring generation
- Syntax validation
- Batch file processing

### 20. FixDocstrings (`fix_docstrings.py`)
**Purpose**: Specific docstring formatting fixes
**Key Functions**:
- `fix_docstrings(filename)`: Fixes docstring formatting in specific file

**Useful Features**:
- Regex-based docstring fixing
- Function definition pattern matching
- Args section formatting

## 📊 Summary of Non-Core Components

### By Category:
- **Audio/Visual Processing**: 4 modules (AudioToNeuralBridge, VisualEnergyBridge, SensoryWorkspaceMapper, ScreenGraph)
- **System Management**: 4 modules (PerformanceMonitor, ErrorHandler, ConfigManager, RandomSeedManager)
- **Persistence**: 1 module (NeuralMapPersistence)
- **Event/Spike Processing**: 3 modules (EventDrivenSystem, SpikeQueueSystem, LoggingUtils)
- **User Interface**: 3 modules (UIEngine, UIStateManager, MinimalUI)
- **Testing/Examples**: 4 modules (UnifiedTestSuite, UnifiedLauncher, 3 Example modules)
- **Maintenance**: 2 modules (FixSyntaxErrors, FixDocstrings)

### Key Capabilities:
- **Real-time Processing**: Audio, visual, and screen capture
- **System Monitoring**: Performance, errors, and health tracking
- **Configuration Management**: Centralized, type-safe configuration
- **Event Processing**: High-performance event and spike queuing
- **User Interface**: Multiple UI options (full, minimal, optimized)
- **Testing**: Comprehensive test suite and validation
- **Maintenance**: Automated code fixing and formatting

### Integration Points:
- All modules integrate with the core simulation system
- Event-driven architecture for real-time processing
- Thread-safe state management across modules
- Centralized logging and error handling
- Configurable performance and behavior parameters

This comprehensive mapping provides a complete overview of all non-core components, their purposes, key methods, and useful features for system management, testing, UI, and utilities.
