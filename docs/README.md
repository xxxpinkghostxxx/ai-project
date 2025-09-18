# AI Neural Simulation System

A comprehensive biologically-inspired neural network simulation system with real-time visualization, learning capabilities, and advanced neural dynamics.

## 🧠 Overview

This system implements a sophisticated neural simulation that combines:
- **Biologically-inspired neural networks** with energy-based dynamics
- **Real-time visualization** using DearPyGui
- **Advanced learning mechanisms** including STDP, Hebbian learning, and memory formation
- **Energy-modulated learning** where energy levels directly influence synaptic plasticity
- **Event-driven architecture** for efficient processing
- **Multi-modal sensory integration** (visual and audio)
- **Homeostatic regulation** and criticality maintenance

## 🏗️ Architecture

### Core Components

#### 1. Simulation Manager (`core/simulation_manager.py`)
- Central coordinator for all neural systems
- Orchestrates simulation loop and component lifecycles
- Manages configuration and error handling
- Integrates enhanced neural processes with lazy loading

#### 2. Energy System (`energy/`)
- **Energy Behavior** (`energy/energy_behavior.py`): Energy flow and consumption
- **Energy Constants** (`energy/energy_constants.py`): Centralized energy parameters
- **Node Access Layer** (`energy/node_access_layer.py`): ID-based node operations
- **Energy System Validator** (`energy/energy_system_validator.py`): Energy integration validation

#### 3. Neural Dynamics (`neural/`)
- **Behavior Engine** (`neural/behavior_engine.py`): Node behavior management
- **Connection Logic** (`neural/connection_logic.py`): Intelligent connection formation
- **Network Metrics** (`neural/network_metrics.py`): Network analysis and metrics
- **Event-Driven System** (`neural/event_driven_system.py`): Event-based processing

#### 4. Learning Systems (`learning/`)
- **Learning Engine** (`learning/learning_engine.py`): STDP and pattern learning
- **Live Hebbian Learning** (`learning/live_hebbian_learning.py`): Real-time learning with energy modulation
- **Memory System** (`learning/memory_system.py`): Memory formation and persistence
- **Homeostasis Controller** (`learning/homeostasis_controller.py`): Energy balance regulation

#### 5. Sensory Integration (`sensory/`)
- **Visual Energy Bridge** (`sensory/visual_energy_bridge.py`): Visual input processing
- **Audio to Neural Bridge** (`sensory/audio_to_neural_bridge.py`): Audio feature extraction
- **Sensory Workspace Mapper** (`sensory/sensory_workspace_mapper.py`): Sensory-to-workspace mapping

### Supporting Systems

#### Energy Management
- **Energy Behavior** (`energy/energy_behavior.py`): Energy flow and consumption
- **Energy Constants** (`energy/energy_constants.py`): Centralized energy parameters
- **Node Access Layer** (`energy/node_access_layer.py`): ID-based node operations
- **Energy System Validator** (`energy/energy_system_validator.py`): Energy integration validation

#### Learning & Memory
- **Learning Engine** (`learning/learning_engine.py`): STDP and pattern learning
- **Memory System** (`learning/memory_system.py`): Memory formation and persistence
- **Live Hebbian Learning** (`learning/live_hebbian_learning.py`): Real-time learning with energy modulation
- **Homeostasis Controller** (`learning/homeostasis_controller.py`): Energy balance regulation

#### Sensory Integration
- **Visual Energy Bridge** (`sensory/visual_energy_bridge.py`): Visual input processing
- **Audio to Neural Bridge** (`sensory/audio_to_neural_bridge.py`): Audio feature extraction
- **Sensory Workspace Mapper** (`sensory/sensory_workspace_mapper.py`): Sensory-to-workspace mapping

#### System Management
- **Performance Monitor** (`utils/performance_monitor.py`): Real-time system monitoring
- **Unified Error Handler** (`utils/unified_error_handler.py`): Graceful error handling and recovery
- **Unified Config Manager** (`config/unified_config_manager.py`): Centralized configuration
- **Event-Driven System** (`neural/event_driven_system.py`): Event-based processing

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
python core/unified_launcher.py
```

### Basic Usage

```python
from core.simulation_manager import create_simulation_manager
from core.main_graph import initialize_main_graph

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
from neural.enhanced_neural_integration import create_enhanced_neural_integration

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
from sensory.visual_energy_bridge import create_visual_energy_bridge

visual_bridge = create_visual_energy_bridge(integration)
graph = visual_bridge.process_visual_to_enhanced_energy(
    graph, screen_data, step
)

# Audio processing
from sensory.audio_to_neural_bridge import create_audio_to_neural_bridge

audio_bridge = create_audio_to_neural_bridge()
sensory_nodes = audio_bridge.process_audio_to_sensory_nodes(audio_data)
```

### Memory and Learning

```python
# Memory formation
from learning.memory_system import MemorySystem

memory_system = MemorySystem()
memory_system.form_memory_traces(graph)

# Live Hebbian learning with energy modulation
from learning.live_hebbian_learning import create_live_hebbian_learning

learning = create_live_hebbian_learning(sim_manager)
graph = learning.apply_continuous_learning(graph, step)
```

### Energy-Learning Integration

```python
# Energy-modulated learning where energy levels influence synaptic plasticity
from learning.live_hebbian_learning import create_live_hebbian_learning

learning = create_live_hebbian_learning(sim_manager)

# Enable energy modulation for biologically realistic learning
learning.energy_learning_modulation = True

# Learning rates automatically adjust based on node energy levels:
# - High energy nodes (0.8+): 95% of base learning rate
# - Medium energy nodes (0.5): 75% of base learning rate
# - Low energy nodes (0.1): 55% of base learning rate

graph = learning.apply_continuous_learning(graph, step)
```

## 📊 Monitoring and Visualization

### Performance Monitoring

```python
from utils.performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
metrics = monitor.get_current_metrics()

print(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")
print(f"CPU usage: {metrics.cpu_percent:.1f}%")
print(f"FPS: {metrics.fps:.1f}")
```

### Network Metrics

```python
from neural.network_metrics import create_network_metrics

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

The project has been restructured with consolidated directories for better organization. Key consolidations include unified configuration in `config/` and unified utilities (including error handling) in `utils/`.

```
ai-project/
├── core/
│   ├── simulation_manager.py     # Main simulation coordinator
│   ├── unified_launcher.py       # Launch scripts
│   └── main_graph.py             # Graph utilities
├── config/
│   ├── unified_config_manager.py # Unified configuration management
│   └── config.ini                # Configuration file
├── utils/
│   ├── unified_error_handler.py  # Unified error handling
│   ├── performance_monitor.py    # Performance monitoring
│   ├── lazy_loader.py            # Lazy loading system
│   ├── performance_cache.py      # Performance caching
│   ├── static_allocator.py       # Memory allocation
│   └── other_utils.py            # Common utilities (logging, stats, etc.)
├── neural/
│   ├── behavior_engine.py        # Node behavior management
│   ├── connection_logic.py       # Intelligent connection formation
│   ├── network_metrics.py        # Network analysis and metrics
│   ├── event_driven_system.py    # Event-based processing
│   └── spike_queue_system.py     # Spike processing system
├── energy/
│   ├── energy_behavior.py        # Energy flow and consumption
│   ├── energy_constants.py       # Centralized energy parameters
│   ├── node_access_layer.py      # ID-based node operations
│   ├── node_id_manager.py        # Node ID management
│   └── energy_system_validator.py # Energy integration validation
├── learning/
│   ├── learning_engine.py        # STDP and pattern learning
│   ├── live_hebbian_learning.py  # Real-time learning with energy modulation
│   ├── memory_system.py          # Memory formation and persistence
│   └── homeostasis_controller.py # Energy balance regulation
├── sensory/
│   ├── visual_energy_bridge.py   # Visual input processing
│   ├── audio_to_neural_bridge.py # Audio feature extraction
│   └── sensory_workspace_mapper.py # Sensory-to-workspace mapping
├── ui/
│   ├── ui_engine.py              # User interface and visualization
│   ├── ui_state_manager.py       # UI state management
│   └── screen_graph.py           # Screen capture utilities
├── docs/
│   ├── README.md                 # Main documentation
│   ├── CONSOLIDATED_DOCUMENTATION.md # Complete API reference
│   ├── ENERGY_LEARNING_INTEGRATION.md # Energy-learning integration
│   ├── OPTIMIZATION_REPORT.md    # Performance optimizations
│   ├── QUICK_START_GUIDE.md      # Quick start guide
│   └── other_docs.md             # Additional documentation
├── tests/
│   ├── comprehensive_simulation_test.py # Comprehensive testing
│   ├── comprehensive_test_framework.py  # Test framework
│   ├── debug_simulation_manager.py      # Debug utilities
│   ├── simple_energy_test.py            # Energy system tests
│   ├── test_energy_learning.py          # Energy-learning tests
│   └── other_tests.py                    # Additional tests
├── analysis/
│   ├── comprehensive_test_report.json   # Test reports
│   ├── energy_validation_report.json    # Energy validation
│   └── simulation_metrics_*.json        # Performance metrics
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

## 🛠️ Building Standalone Executable

To create a distributable Windows executable (.exe) for the AI Neural Simulation System:

1. **Install PyInstaller**:
   Ensure PyInstaller is installed via `pip install pyinstaller>=6.0` (added to requirements.txt).

2. **Build the Executable**:
   Run the following command from the project root:
   ```
   pyinstaller --onefile --console --name ai-simulation core/unified_launcher.py
   ```
   - `--onefile`: Bundles everything into a single .exe file.
   - `--console`: Enables console output for the simulation (use `--windowed` for GUI-only if needed).
   - Output: The .exe will be created in the `dist/` directory as `ai-simulation.exe`.

3. **Run the Executable**:
   Navigate to `dist/` and execute `ai-simulation.exe`. It supports the same command-line arguments as the Python script (e.g., `ai-simulation.exe full` for full UI mode).

4. **Troubleshooting**:
   - If modules are missing during build, add `--hidden-import <module_name>` (e.g., for torch or dearpygui).
   - The .exe is standalone and does not require a Python environment.
   - For custom icons or advanced config, use a `.spec` file generated by `pyi-makespec`.

**Note**: Building may take several minutes due to large dependencies like Torch and TensorFlow. The resulting .exe is approximately 1-2 GB in size.

## 🙏 Acknowledgments

- PyTorch Geometric for graph neural networks
- DearPyGui for real-time visualization
- The neuroscience community for biological inspiration
- Open source contributors and researchers

---

**Note**: This is a research and educational system. For production use, additional testing and optimization may be required.
