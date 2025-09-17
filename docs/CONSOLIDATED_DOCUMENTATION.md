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

#### 2. Neural Dynamics (`enhanced_neural_dynamics.py`)
Implements advanced neural dynamics including STDP, IEG tagging, and theta-burst stimulation.

**Key Features:**
- Membrane potential dynamics
- Synaptic transmission modeling
- Plasticity mechanisms
- Homeostatic control

#### 3. Learning Engine (`learning_engine.py`)
Handles various learning mechanisms including Hebbian learning and memory formation.

**Key Features:**
- STDP learning implementation
- Memory trace formation
- Pattern recognition
- Connection consolidation

#### 4. Event System (`event_driven_system.py`)
Event-driven processing architecture for efficient neural simulation.

**Key Features:**
- Priority-based event scheduling
- Spike event processing
- Synaptic transmission events
- Memory formation events

#### 5. Visualization (`ui_engine.py`)
Real-time visualization and user interface components.

**Key Features:**
- Dynamic graph visualization
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

The project features a cleaned-up folder structure with consolidations in key areas:

- **config/**: Unified configuration management via `unified_config_manager.py`, with `config.ini` for settings.
- **utils/**: Consolidated utilities including `unified_error_handler.py` for error handling, `performance_monitor.py`, logging, and statistics utilities.
- **core/**: Core simulation files like `simulation_manager.py` and `unified_launcher.py`.
- **neural/**: Advanced neural systems (dynamics, connections, behaviors, events).
- **energy/**: Energy management modules.
- **learning/**: Learning and memory systems.
- **sensory/**: Sensory integration (visual, audio).
- **ui/**: User interface and visualization.
- **docs/** and **tests/**: Documentation and testing.

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
class EnhancedNeuralIntegration:
    def integrate_with_existing_system(self, graph: Data, step: int) -> Data
    def create_enhanced_node(self, graph: Data, node_id: int, **kwargs) -> bool
    def create_enhanced_connection(self, graph: Data, source_id: int, target_id: int, **kwargs) -> bool
    def set_neuromodulator_level(self, neuromodulator: str, level: float) -> None
```

#### Learning Engine
```python
class LearningEngine:
    def apply_timing_learning(self, pre_node, post_node, edge, delta_t) -> None
    def consolidate_connections(self, graph: Data) -> Data
    def form_memory_traces(self, graph: Data) -> Data
    def apply_memory_influence(self, graph: Data) -> Data
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
from enhanced_neural_integration import create_enhanced_neural_integration

# Create enhanced integration
enhanced = create_enhanced_neural_integration()

# Set neuromodulator levels
enhanced.set_neuromodulator_level('dopamine', 0.8)
enhanced.set_neuromodulator_level('serotonin', 0.6)

# Integrate with existing system
graph = enhanced.integrate_with_existing_system(graph, step=0)
```

### Custom Node Behaviors
```python
from enhanced_node_behaviors import create_enhanced_node_behavior_system

# Create behavior system
behavior_system = create_enhanced_node_behavior_system()

# Create custom node behavior
node_behavior = behavior_system.create_node_behavior(
    node_id=1,
    node_type='oscillator',
    oscillation_freq=2.0,
    threshold=0.7
)

# Update behavior
success = node_behavior.update_behavior(graph, step=0, access_layer)
```

---

*This consolidated documentation replaces the following files:*
- *API_REFERENCE.md*
- *TECHNICAL_DOCUMENTATION.md* 
- *PROJECT_SUMMARY.md*
- *EXAMPLES_AND_TUTORIALS.md*
- *TROUBLESHOOTING.md*
