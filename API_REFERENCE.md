# API Reference

## Core Classes and Functions

### SimulationManager

The central coordinator for all neural systems.

#### Constructor
```python
SimulationManager(config: Optional[Dict[str, Any]] = None)
```

#### Key Methods

##### Simulation Control
```python
start_simulation(run_in_thread: bool = True) -> None
stop_simulation() -> None
run_single_step() -> bool
reset_simulation() -> None
```

##### Graph Management
```python
set_graph(graph: Data) -> None
initialize_graph() -> None
get_graph() -> Data
```

##### Enhanced Features
```python
create_enhanced_node(node_id: int, node_type: str = 'dynamic', 
                    subtype: str = 'standard', **kwargs) -> bool
create_enhanced_connection(source_id: int, target_id: int,
                          connection_type: str = 'excitatory', **kwargs) -> bool
set_neuromodulator_level(neuromodulator: str, level: float) -> None
```

##### Statistics and Monitoring
```python
get_performance_stats() -> Dict[str, Any]
get_system_stats() -> Dict[str, Any]
get_enhanced_statistics() -> Dict[str, Any]
```

### EnhancedNeuralIntegration

Integrates all enhanced neural systems.

#### Constructor
```python
EnhancedNeuralIntegration()
```

#### Key Methods
```python
integrate_with_existing_system(graph: Data, step: int) -> Data
create_enhanced_node(graph: Data, node_id: int, node_type: str = 'dynamic', 
                    subtype: str = 'standard', **kwargs) -> bool
create_enhanced_connection(graph: Data, source_id: int, target_id: int,
                          connection_type: str = 'excitatory', **kwargs) -> bool
set_neuromodulator_level(neuromodulator: str, level: float) -> None
get_integration_statistics() -> Dict[str, Any]
```

### EnhancedNeuralDynamics

Implements advanced neural dynamics including STDP and plasticity.

#### Constructor
```python
EnhancedNeuralDynamics()
```

#### Key Methods
```python
update_neural_dynamics(graph: Data, step: int) -> Data
set_neuromodulator_level(neuromodulator: str, level: float) -> None
get_statistics() -> Dict[str, Any]
```

### EnhancedNodeBehavior

Sophisticated biological node behaviors.

#### Constructor
```python
EnhancedNodeBehavior(node_id: int, node_type: str = 'dynamic', **kwargs)
```

#### Node Types
- **oscillator**: Rhythmic activity generation
- **integrator**: Information accumulation
- **relay**: Signal amplification
- **highway**: High-capacity energy distribution
- **workspace**: Imagination and flexible thinking
- **transmitter**: Signal transmission
- **resonator**: Frequency-specific responses
- **dampener**: Signal attenuation

#### Key Methods
```python
update_behavior(graph: Data, step: int, access_layer: NodeAccessLayer) -> bool
get_behavior_statistics() -> Dict[str, Any]
```

### EnhancedConnectionSystem

Advanced connection management with plasticity.

#### Constructor
```python
EnhancedConnectionSystem()
```

#### Connection Types
- **excitatory**: Positive signal transmission
- **inhibitory**: Negative signal transmission
- **modulatory**: Signal modulation
- **plastic**: STDP-enabled connections
- **burst**: Theta-burst connections
- **gated**: Neuromodulator-gated connections

#### Key Methods
```python
create_connection(source_id: int, target_id: int, 
                 connection_type: str = 'excitatory', **kwargs) -> bool
remove_connection(source_id: int, target_id: int) -> bool
update_connections(graph: Data, step: int) -> Data
set_neuromodulator_level(neuromodulator: str, level: float) -> None
get_connection_statistics() -> Dict[str, Any]
```

## Energy Management

### EnergyCalculator

Static utility class for energy calculations.

#### Methods
```python
calculate_energy_cap() -> float
calculate_energy_decay(current_energy: float, decay_rate: float) -> float
calculate_energy_transfer(energy: float, transfer_fraction: float) -> float
calculate_energy_boost(energy: float, boost_amount: float) -> float
calculate_membrane_potential(energy: float) -> float
apply_energy_bounds(energy: float) -> float
```

### Energy Functions
```python
get_node_energy_cap() -> float
update_node_energy_with_learning(graph: Data, node_id: int, delta_energy: float) -> Data
apply_energy_behavior(graph: Data, behavior_params: Optional[Dict] = None) -> Data
update_membrane_potentials(graph: Data) -> Data
apply_refractory_periods(graph: Data) -> Data
```

## Learning and Memory

### LearningEngine

Implements STDP and pattern learning.

#### Constructor
```python
LearningEngine()
```

#### Key Methods
```python
apply_timing_learning(pre_node: Dict, post_node: Dict, edge: Dict, delta_t: float) -> None
consolidate_connections(graph: Data) -> Data
form_memory_traces(graph: Data) -> Data
apply_memory_influence(graph: Data) -> Data
get_learning_statistics() -> Dict[str, Any]
```

### MemorySystem

Memory formation and persistence.

#### Constructor
```python
MemorySystem()
```

#### Key Methods
```python
form_memory_traces(graph: Data) -> Data
consolidate_memories(graph: Data) -> Data
decay_memories() -> None
recall_patterns(graph: Data, target_node_idx: int) -> List[Dict]
get_memory_statistics() -> Dict[str, Any]
get_node_memory_importance(node_id: int) -> float
```

### LiveHebbianLearning

Real-time Hebbian learning system.

#### Constructor
```python
LiveHebbianLearning(simulation_manager: Optional[SimulationManager] = None)
```

#### Key Methods
```python
apply_continuous_learning(graph: Data, step: int) -> Data
set_learning_rate(learning_rate: float) -> None
set_learning_active(active: bool) -> None
get_learning_statistics() -> Dict[str, Any]
```

## Sensory Integration

### VisualEnergyBridge

Processes visual input into neural energy.

#### Constructor
```python
VisualEnergyBridge(enhanced_integration: Optional[EnhancedNeuralIntegration] = None)
```

#### Key Methods
```python
process_visual_to_enhanced_energy(graph: Data, screen_data: np.ndarray, step: int) -> Data
set_visual_sensitivity(sensitivity: float) -> None
set_pattern_threshold(threshold: float) -> None
get_visual_statistics() -> Dict[str, Any]
```

### AudioToNeuralBridge

Converts audio input to sensory nodes.

#### Constructor
```python
AudioToNeuralBridge(neural_simulation: Optional[SimulationManager] = None)
```

#### Key Methods
```python
process_audio_to_sensory_nodes(audio_data: np.ndarray) -> List[Dict[str, Any]]
integrate_audio_nodes_into_graph(graph: Data, audio_data: np.ndarray) -> Data
get_audio_feature_statistics() -> Dict[str, Any]
```

### SensoryWorkspaceMapper

Maps sensory input to workspace nodes.

#### Constructor
```python
SensoryWorkspaceMapper(workspace_size: Tuple[int, int] = (10, 10))
```

#### Key Methods
```python
map_visual_to_workspace(graph: Data, visual_data: np.ndarray, step: int) -> Data
map_audio_to_workspace(graph: Data, audio_data: np.ndarray, step: int) -> Data
get_mapping_statistics() -> Dict[str, Any]
```

## System Management

### PerformanceMonitor

Real-time system performance monitoring.

#### Constructor
```python
PerformanceMonitor(update_interval: float = 1.0, history_size: int = 10, 
                  thresholds: Optional[PerformanceThresholds] = None)
```

#### Key Methods
```python
start_monitoring() -> None
stop_monitoring() -> None
record_step(step_time: float, node_count: int = 0, edge_count: int = 0) -> None
get_current_metrics() -> PerformanceMetrics
get_performance_summary() -> Dict[str, Any]
```

### ErrorHandler

Graceful error handling and recovery.

#### Constructor
```python
ErrorHandler()
```

#### Key Methods
```python
handle_error(error: Exception, context: str = "", 
            recovery_func: Optional[Callable] = None, critical: bool = False) -> bool
get_system_health() -> Dict[str, Any]
add_error_callback(callback: Callable) -> None
```

### ConfigManager

Centralized configuration management.

#### Constructor
```python
ConfigManager(config_file: str = "config.ini")
```

#### Key Methods
```python
get(section: str, key: str, default: Any = None, value_type: type = str) -> Any
set(section: str, key: str, value: Any) -> None
save() -> None
reload() -> None
get_system_constants() -> Dict[str, float]
get_learning_config() -> Dict[str, float]
```

## Event System

### EventDrivenSystem

Event-based processing architecture.

#### Constructor
```python
EventDrivenSystem(simulation_manager: Optional[SimulationManager] = None)
```

#### Key Methods
```python
start() -> None
stop() -> None
process_events(max_events: Optional[int] = None) -> int
schedule_event(event: NeuralEvent) -> None
schedule_spike(node_id: int, timestamp: Optional[float] = None, priority: int = 1) -> None
get_statistics() -> Dict[str, Any]
```

### Event Types
```python
class EventType(Enum):
    SPIKE = "spike"
    SYNAPTIC_TRANSMISSION = "synaptic_transmission"
    PLASTICITY_UPDATE = "plasticity_update"
    MEMORY_FORMATION = "memory_formation"
    HOMEOSTATIC_REGULATION = "homeostatic_regulation"
    NODE_BIRTH = "node_birth"
    NODE_DEATH = "node_death"
    ENERGY_TRANSFER = "energy_transfer"
    THETA_BURST = "theta_burst"
    IEG_TAGGING = "ieg_tagging"
```

## Utility Functions

### Node Access
```python
# NodeAccessLayer
get_node_by_id(node_id: int) -> Optional[Dict[str, Any]]
get_node_energy(node_id: int) -> Optional[float]
set_node_energy(node_id: int, energy: float) -> bool
update_node_property(node_id: int, property_name: str, value: Any) -> bool
select_nodes_by_type(node_type: str) -> List[int]
select_nodes_by_behavior(behavior: str) -> List[int]
```

### ID Management
```python
# NodeIDManager
generate_unique_id(node_type: str = "unknown", metadata: Optional[Dict] = None) -> int
register_node_index(node_id: int, index: int) -> bool
get_node_index(node_id: int) -> Optional[int]
is_valid_id(node_id: int) -> bool
recycle_node_id(node_id: int) -> bool
```

### Logging
```python
# Logging utilities
log_step(step_desc: str, **kwargs) -> None
log_node_state(node_label: Dict, prefix: str = "[NODE_STATE]") -> None
append_log_line(line: str) -> None
get_log_lines() -> List[str]
```

## Configuration Constants

### Energy Constants
```python
class EnergyConstants:
    TIME_STEP = 0.01
    REFRACTORY_PERIOD_SHORT = 0.1
    REFRACTORY_PERIOD_MEDIUM = 0.5
    REFRACTORY_PERIOD_LONG = 1.0
    ACTIVATION_THRESHOLD_DEFAULT = 0.5
    MEMBRANE_POTENTIAL_CAP = 1.0
    ENERGY_TRANSFER_FRACTION = 0.2
    # ... more constants
```

### Connection Constants
```python
class ConnectionConstants:
    EDGE_TYPES = ['excitatory', 'inhibitory', 'modulatory']
    DEFAULT_EDGE_WEIGHT = 1.0
    LEARNING_RATE_DEFAULT = 0.01
    WEIGHT_CAP_MAX = 5.0
    WEIGHT_MIN = 0.1
    # ... more constants
```

## Data Structures

### NeuralEvent
```python
@dataclass
class NeuralEvent:
    event_type: EventType
    timestamp: float
    source_node_id: int
    target_node_id: Optional[int] = None
    data: Dict[str, Any] = None
    priority: int = 0
```

### PerformanceMetrics
```python
@dataclass
class PerformanceMetrics:
    step_time: float = 0.0
    total_runtime: float = 0.0
    fps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    # ... more metrics
```

### Spike
```python
@dataclass
class Spike:
    source_node_id: int
    target_node_id: int
    timestamp: float
    spike_type: SpikeType
    amplitude: float
    delay: float = 0.0
    weight: float = 1.0
    refractory_period: float = 0.0
    propagation_speed: float = 1.0
```

---

This API reference covers the main classes and functions available in the neural simulation system. For more detailed information, refer to the source code and inline documentation.
