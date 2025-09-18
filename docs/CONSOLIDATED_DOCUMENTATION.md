# AI Neural Simulation System - Consolidated Documentation

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [System Architecture](#system-architecture)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Examples](#examples)

## Overview

The AI Neural Simulation System is a comprehensive, biologically-inspired neural network simulation platform that implements advanced neural dynamics, learning mechanisms, and real-time visualization. The system is designed for research, education, and development of biologically-inspired AI systems.

### Key Features

#### 🧠 Biologically-Inspired Neural Dynamics
- **STDP (Spike-Timing Dependent Plasticity)**: Timing-based synaptic plasticity
- **IEG Tagging**: Immediate Early Gene tagging for plasticity gating
- **Theta-Burst Stimulation**: High-frequency stimulation for LTP induction
- **Membrane Dynamics**: Realistic membrane potential modeling
- **Homeostatic Regulation**: Energy balance and criticality maintenance

#### 🔬 Advanced Learning Mechanisms
- **Hebbian Learning**: "Neurons that fire together, wire together"
- **Memory Formation**: Pattern recognition and long-term storage
- **Live Learning**: Real-time adaptation during simulation
- **Memory Consolidation**: Gradual strengthening of important patterns

#### 🎯 Real-Time Visualization
- **Dynamic Graph Visualization**: Real-time neural network display
- **Energy Flow Visualization**: Visual representation of energy dynamics
- **Performance Monitoring**: Real-time system metrics and health monitoring
- **Interactive Controls**: Live parameter adjustment and control

#### 🔧 System Architecture
- **Modular Design**: Clean separation of concerns with pluggable components
- **Event-Driven Processing**: Efficient processing using event queues
- **NASA Power of Ten Compliance**: Safety-critical coding standards
- **Static Memory Allocation**: Predictable performance characteristics

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from simulation_manager import SimulationManager

# Create simulation manager
sim_manager = SimulationManager()

# Initialize with default graph
sim_manager.initialize_graph()

# Start simulation
sim_manager.start_simulation()

# Run single step
success = sim_manager.run_single_step()

# Stop simulation
sim_manager.stop_simulation()
```

### Launch Options
```bash
# Launch with UI
python core/unified_launcher.py --profile ui

# Launch test suite
python core/unified_launcher.py --profile test

# Launch headless
python core/unified_launcher.py --profile headless
```

## System Architecture

### Core Components

#### 1. Simulation Manager (`simulation_manager.py`)
The central coordinator that orchestrates all neural systems and manages the simulation lifecycle.

**Key Responsibilities:**
- Graph management and updates
- Component initialization and coordination
- Event processing and scheduling
- Performance monitoring and statistics
- Error handling and recovery

#### 2. Neural Dynamics (`neural/`)
Implements neural processing and behavior management.

**Key Features:**
- Node behavior management via `behavior_engine.py`
- Intelligent connection formation via `connection_logic.py`
- Network analysis and metrics via `network_metrics.py`
- Event-driven processing via `event_driven_system.py`
- Spike processing via `spike_queue_system.py`

#### 3. Learning Engine (`learning/`)
Handles various learning mechanisms including energy-modulated Hebbian learning.

**Key Features:**
- STDP learning implementation via `learning_engine.py`
- Energy-modulated Hebbian learning via `live_hebbian_learning.py`
- Memory trace formation via `memory_system.py`
- Homeostatic regulation via `homeostasis_controller.py`
- Pattern recognition and connection consolidation

#### 4. Energy System (`energy/`)
Central energy management and integration system.

**Key Features:**
- Energy flow and consumption via `energy_behavior.py`
- Energy constants and parameters via `energy_constants.py`
- Node access layer via `node_access_layer.py`
- Node ID management via `node_id_manager.py`
- Energy integration validation via `energy_system_validator.py`

#### 5. Visualization (`ui/`)
Real-time visualization and user interface components.

**Key Features:**
- Dynamic graph visualization via `ui_engine.py`
- UI state management via `ui_state_manager.py`
- Performance metrics display
- Interactive controls
- Real-time data feeds

### Data Flow

```
Input Data → Sensory Processing → Neural Dynamics → Learning → Memory → Output
     ↓              ↓                ↓            ↓        ↓        ↓
Visual/Audio → Feature Extraction → STDP/IEG → Hebbian → Storage → Visualization
```

### Project Structure Summary

The project features a well-organized folder structure with specialized directories:

- **config/**: Unified configuration management via `unified_config_manager.py`, with `config.ini` for settings.
- **utils/**: Consolidated utilities including `unified_error_handler.py` for error handling, `performance_monitor.py`, `lazy_loader.py`, `performance_cache.py`, `static_allocator.py`, logging, and statistics utilities.
- **core/**: Core simulation files like `simulation_manager.py` and `unified_launcher.py`.
- **neural/**: Neural systems including `behavior_engine.py`, `connection_logic.py`, `network_metrics.py`, `event_driven_system.py`, and `spike_queue_system.py`.
- **energy/**: Energy management modules including `energy_behavior.py`, `energy_constants.py`, `node_access_layer.py`, `node_id_manager.py`, and `energy_system_validator.py`.
- **learning/**: Learning and memory systems including `learning_engine.py`, `live_hebbian_learning.py` (with energy modulation), `memory_system.py`, and `homeostasis_controller.py`.
- **sensory/**: Sensory integration (visual, audio) with `visual_energy_bridge.py`, `audio_to_neural_bridge.py`, and `sensory_workspace_mapper.py`.
- **ui/**: User interface and visualization with `ui_engine.py`, `ui_state_manager.py`, and `screen_graph.py`.
- **docs/**: Comprehensive documentation including README, API references, and specialized guides.
- **tests/**: Testing framework with comprehensive simulation tests, debug utilities, and energy-learning integration tests.
- **analysis/**: Analysis and report files including test reports, energy validation reports, and performance metrics.

This structure eliminates redundant files and centralizes functionality without referencing removed items like old config or error handlers.

## API Reference

### Core Classes

#### SimulationManager
```python
class SimulationManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    def start_simulation(self, run_in_thread: bool = True) -> None
    def stop_simulation(self) -> None
    def run_single_step(self) -> bool
    def reset_simulation(self) -> None
    def set_graph(self, graph: Data) -> None
    def initialize_graph(self) -> None
    def get_graph(self) -> Data
```

#### Enhanced Neural Integration
```python
# Enhanced neural integration is handled through multiple specialized systems:

# Node Behavior Management
from neural.behavior_engine import BehaviorEngine
behavior_engine = BehaviorEngine()
success = behavior_engine.update_node_behavior(node_id, graph, step, access_layer)

# Connection Logic
from neural.connection_logic import intelligent_connection_formation
graph = intelligent_connection_formation(graph)

# Network Metrics
from neural.network_metrics import NetworkMetrics
metrics = NetworkMetrics()
criticality = metrics.calculate_criticality(graph)
```

#### Learning Engine
```python
# Learning Engine - STDP and pattern learning
from learning.learning_engine import LearningEngine
learning_engine = LearningEngine()
graph = learning_engine.consolidate_connections(graph)
graph = learning_engine.apply_memory_influence(graph)

# Live Hebbian Learning with energy modulation
from learning.live_hebbian_learning import create_live_hebbian_learning
learning = create_live_hebbian_learning(simulation_manager)
learning.energy_learning_modulation = True  # Enable energy-based learning
graph = learning.apply_continuous_learning(graph, step)

# Memory System
from learning.memory_system import MemorySystem
memory_system = MemorySystem()
graph = memory_system.form_memory_traces(graph)
graph = memory_system.consolidate_memories(graph)
```

### Utility Functions

#### Node Management
```python
# Node access and manipulation
get_node_by_id(graph, node_id: int) -> Optional[Dict[str, Any]]
select_nodes_by_type(graph, node_type: str) -> List[int]
select_nodes_by_behavior(graph, behavior: str) -> List[int]
```

#### Energy Management
```python
# Energy calculations and dynamics
get_node_energy_cap() -> float
update_node_energy_with_learning(graph, node_id, delta_energy) -> Data
apply_energy_behavior(graph, behavior_params=None) -> Data
```

## Configuration

### Configuration File (`config/config.ini`)
The system uses a comprehensive configuration file for all parameters:

```ini
[System]
time_step = 0.01
max_nodes = 100000
max_edges = 500000

[Learning]
learning_rate = 0.01
stdp_window = 0.1
plasticity_threshold = 0.3

[EnhancedNodes]
oscillator_frequency = 1.0
integrator_threshold = 0.8
relay_amplification = 1.5
highway_energy_boost = 2.0

[Performance]
update_interval = 1.0
history_size = 10
memory_warning_mb = 2000.0
```

### Environment Variables
```bash
# Set random seed
export NEURAL_SIMULATION_SEED=42

# Enable debug mode
export NEURAL_DEBUG=1

# Set log level
export NEURAL_LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues
**Problem**: High memory usage or out of memory errors
**Solution**: 
- Reduce `max_nodes` in configuration
- Enable garbage collection in performance monitor
- Use static memory allocation

#### 2. Performance Issues
**Problem**: Slow simulation or low FPS
**Solution**:
- Check system resources in performance monitor
- Reduce update frequency
- Optimize graph size

#### 3. Import Errors
**Problem**: Module import failures
**Solution**:
- Run `python core/unified_launcher.py --test-imports`
- Check requirements.txt installation
- Verify Python version compatibility

#### 4. Visualization Issues
**Problem**: UI not displaying or updating
**Solution**:
- Check DearPyGui installation
- Verify display settings
- Restart UI engine

### Debug Mode
Enable debug mode for detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Basic Neural Network
```python
from simulation_manager import SimulationManager
from main_graph import create_test_graph

# Create simulation
sim = SimulationManager()

# Create test graph
graph = create_test_graph(num_sensory=100, num_dynamic=50)
sim.set_graph(graph)

# Run simulation
sim.start_simulation()
for i in range(1000):
    sim.run_single_step()
sim.stop_simulation()
```

### Enhanced Neural Dynamics
```python
from neural.behavior_engine import BehaviorEngine
from energy.node_access_layer import NodeAccessLayer

# Create behavior engine and access layer
behavior_engine = BehaviorEngine()
access_layer = NodeAccessLayer(graph)

# Update node behavior
success = behavior_engine.update_node_behavior(node_id, graph, step, access_layer)

# Get node energy and properties
energy = access_layer.get_node_energy(node_id)
behavior = access_layer.get_node_property(node_id, 'behavior')
```

### Energy-Modulated Learning
```python
from learning.live_hebbian_learning import create_live_hebbian_learning

# Create learning system with energy modulation
learning = create_live_hebbian_learning(simulation_manager)
learning.energy_learning_modulation = True

# Learning rates automatically adjust based on energy:
# - High energy nodes (0.8+): 95% of base learning rate
# - Medium energy nodes (0.5): 75% of base learning rate
# - Low energy nodes (0.1): 55% of base learning rate

graph = learning.apply_continuous_learning(graph, step)
```

---

*This consolidated documentation replaces the following files:*
- *API_REFERENCE.md*
- *TECHNICAL_DOCUMENTATION.md* 
- *PROJECT_SUMMARY.md*
- *EXAMPLES_AND_TUTORIALS.md*
- *TROUBLESHOOTING.md*
