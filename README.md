# AI Neural Simulation System

A comprehensive biologically-inspired neural network simulation system with real-time visualization, learning capabilities, and advanced neural dynamics.

## 🧠 Overview

This system implements a sophisticated neural simulation that combines:
- **Biologically-inspired neural networks** with energy-based dynamics
- **Real-time visualization** using DearPyGui
- **Advanced learning mechanisms** including STDP, Hebbian learning, and memory formation
- **Event-driven architecture** for efficient processing
- **Multi-modal sensory integration** (visual and audio)
- **Homeostatic regulation** and criticality maintenance

## 🏗️ Architecture

### Core Components

#### 1. Simulation Manager (`simulation_manager.py`)
- Central coordinator for all neural systems
- Orchestrates simulation loop and component lifecycles
- Manages configuration and error handling
- Integrates enhanced neural processes

#### 2. Enhanced Neural Integration (`enhanced_neural_integration.py`)
- Integrates all enhanced neural systems
- Manages neural dynamics, connections, and behaviors
- Provides unified interface for advanced features

#### 3. Neural Dynamics (`enhanced_neural_dynamics.py`)
- Implements STDP (Spike-Timing Dependent Plasticity)
- IEG (Immediate Early Gene) tagging for plasticity gating
- Theta-burst stimulation for LTP induction
- Membrane dynamics and spike generation

#### 4. Node Behaviors (`enhanced_node_behaviors.py`)
- Sophisticated biological node types:
  - **Oscillators**: Generate rhythmic activity
  - **Integrators**: Accumulate and consolidate information
  - **Relays**: Amplify and transmit signals
  - **Highways**: High-capacity energy distribution
  - **Workspace**: Imagination and flexible thinking

#### 5. Connection System (`enhanced_connection_system.py`)
- Advanced connection types:
  - Excitatory, inhibitory, modulatory
  - Plastic connections with STDP
  - Burst connections for theta rhythms
  - Gated connections with neuromodulation

### Supporting Systems

#### Energy Management
- **Energy Behavior** (`energy_behavior.py`): Energy flow and consumption
- **Energy Constants** (`energy_constants.py`): Centralized energy parameters
- **Node Access Layer** (`node_access_layer.py`): ID-based node operations

#### Learning & Memory
- **Learning Engine** (`learning_engine.py`): STDP and pattern learning
- **Memory System** (`memory_system.py`): Memory formation and persistence
- **Live Hebbian Learning** (`live_hebbian_learning.py`): Real-time learning

#### Sensory Integration
- **Visual Energy Bridge** (`visual_energy_bridge.py`): Visual input processing
- **Audio to Neural Bridge** (`audio_to_neural_bridge.py`): Audio feature extraction
- **Sensory Workspace Mapper** (`sensory_workspace_mapper.py`): Sensory-to-workspace mapping

#### System Management
- **Performance Monitor** (`performance_monitor.py`): Real-time system monitoring
- **Error Handler** (`error_handler.py`): Graceful error handling and recovery
- **Configuration Manager** (`config_manager.py`): Centralized configuration
- **Event-Driven System** (`event_driven_system.py`): Event-based processing

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ai-project
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the simulation**:
```bash
python unified_launcher.py
```

### Basic Usage

```python
from simulation_manager import create_simulation_manager
from main_graph import initialize_main_graph

# Create simulation manager
sim_manager = create_simulation_manager()

# Initialize neural graph
graph = initialize_main_graph(scale=0.25)
sim_manager.set_graph(graph)

# Start simulation
sim_manager.start_simulation()

# Run simulation steps
for step in range(1000):
    sim_manager.run_single_step()
```

## 🔧 Configuration

The system uses `config.ini` for configuration management:

```ini
[SystemConstants]
node_energy_cap = 255.0
time_step = 0.01
refractory_period = 0.1

[Learning]
plasticity_rate = 0.01
stdp_window = 20.0
ltp_rate = 0.02

[EnhancedNodes]
oscillator_frequency = 0.1
integrator_threshold = 0.8
relay_amplification = 1.5
```

## 🧪 Advanced Features

### Enhanced Neural Dynamics

```python
# Create enhanced neural integration
from enhanced_neural_integration import create_enhanced_neural_integration

integration = create_enhanced_neural_integration()

# Create sophisticated nodes
integration.create_enhanced_node(
    graph, node_id=0, node_type='dynamic',
    subtype='oscillator', is_excitatory=True,
    oscillation_frequency=2.0, energy=0.8
)

# Create advanced connections
integration.create_enhanced_connection(
    graph, source_id=0, target_id=1,
    connection_type='excitatory', weight=1.5,
    plasticity_enabled=True, learning_rate=0.02
)
```

### Sensory Integration

```python
# Visual processing
from visual_energy_bridge import create_visual_energy_bridge

visual_bridge = create_visual_energy_bridge(integration)
graph = visual_bridge.process_visual_to_enhanced_energy(
    graph, screen_data, step
)

# Audio processing
from audio_to_neural_bridge import create_audio_to_neural_bridge

audio_bridge = create_audio_to_neural_bridge()
sensory_nodes = audio_bridge.process_audio_to_sensory_nodes(audio_data)
```

### Memory and Learning

```python
# Memory formation
from memory_system import MemorySystem

memory_system = MemorySystem()
memory_system.form_memory_traces(graph)

# Live Hebbian learning
from live_hebbian_learning import create_live_hebbian_learning

learning = create_live_hebbian_learning(sim_manager)
graph = learning.apply_continuous_learning(graph, step)
```

## 📊 Monitoring and Visualization

### Performance Monitoring

```python
from performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
metrics = monitor.get_current_metrics()

print(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")
print(f"CPU usage: {metrics.cpu_percent:.1f}%")
print(f"FPS: {metrics.fps:.1f}")
```

### Network Metrics

```python
from network_metrics import create_network_metrics

metrics = create_network_metrics()
criticality = metrics.calculate_criticality(graph)
connectivity = metrics.analyze_connectivity(graph)
```

## 🧬 Biological Features

### Spike-Timing Dependent Plasticity (STDP)
- Long-term potentiation (LTP) and depression (LTD)
- Timing-dependent weight updates
- Eligibility traces for delayed reinforcement

### Homeostatic Regulation
- Energy balance maintenance
- Criticality optimization
- Network stability preservation

### Neuromodulation
- Dopamine, serotonin, acetylcholine effects
- Global network state modulation
- Learning rate adaptation

### Memory Formation
- Pattern recognition and storage
- Memory consolidation processes
- Recall and pattern matching

## 🔬 Research Applications

This system is designed for:
- **Neuroscience research**: Studying neural dynamics and plasticity
- **AI development**: Biologically-inspired learning algorithms
- **Cognitive modeling**: Understanding brain-like information processing
- **Educational purposes**: Visualizing neural network behavior

## 📁 Project Structure

```
ai-project/
├── simulation_manager.py          # Main simulation coordinator
├── enhanced_neural_*.py          # Advanced neural systems
├── behavior_engine.py            # Node behavior management
├── energy_*.py                   # Energy dynamics
├── learning_*.py                 # Learning mechanisms
├── memory_system.py              # Memory formation
├── sensory_*.py                  # Sensory integration
├── ui_*.py                       # User interface
├── config.ini                    # Configuration
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Geometric for graph neural networks
- DearPyGui for real-time visualization
- The neuroscience community for biological inspiration
- Open source contributors and researchers

---

**Note**: This is a research and educational system. For production use, additional testing and optimization may be required.
