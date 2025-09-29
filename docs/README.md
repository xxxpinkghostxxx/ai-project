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

### Service-Oriented Architecture (SOA)

The system has been fully migrated from a monolithic architecture to a service-oriented architecture with dependency injection. The monolithic `SimulationManager` has been completely removed and replaced with specialized services orchestrated by the `SimulationCoordinator`.

For detailed information about the SOA, see [SOA_ARCHITECTURE.md](SOA_ARCHITECTURE.md).

#### Core Services Summary

The system includes 9 core services plus additional advanced services:

**Core Services:**
- **SimulationCoordinator**: Central orchestrator for all neural simulation services
- **NeuralProcessingService**: Handles neural dynamics and spiking behavior
- **EnergyManagementService**: Manages energy flow and metabolic processes
- **LearningService**: Coordinates plasticity and learning mechanisms
- **SensoryProcessingService**: Processes visual and audio input data
- **GraphManagementService**: Manages neural graph structure and operations
- **PerformanceMonitoringService**: Monitors system performance and resource usage
- **EventCoordinationService**: Manages event-driven communication between services
- **ConfigurationService**: Provides centralized configuration management

**Advanced Services:**
- **RealTimeAnalyticsService**: Real-time performance monitoring and predictive analytics
- **AdaptiveConfigurationService**: Dynamic parameter adjustment based on system performance
- **DistributedCoordinatorService**: Multi-node coordination and load balancing
- **FaultToleranceService**: Failure detection and recovery mechanisms
- **GPUAcceleratorService**: GPU acceleration for compute-intensive operations
- **CloudDeploymentService**: Cloud deployment and scaling management
- **LoadBalancingService**: Load balancing across distributed nodes
- **MLOptimizerService**: ML-based optimization of system parameters

#### Infrastructure Components

##### Service Registry (`core/services/service_registry.py`)
- Dependency injection container for all services
- Manages service registration, resolution, and lifecycle
- Ensures loose coupling through interface-based design

#### Supporting Systems

##### Energy System (`energy/`)
- **Energy Behavior** (`energy/energy_behavior.py`): Energy flow and consumption
- **Energy Constants** (`energy/energy_constants.py`): Centralized energy parameters
- **Node Access Layer** (`energy/node_access_layer.py`): ID-based node operations
- **Energy System Validator** (`energy/energy_system_validator.py`): Energy integration validation

##### Neural Dynamics (`neural/`)
- **Behavior Engine** (`neural/behavior_engine.py`): Node behavior management
- **Connection Logic** (`neural/connection_logic.py`): Intelligent connection formation
- **Network Metrics** (`neural/network_metrics.py`): Network analysis and metrics
- **Event-Driven System** (`neural/event_driven_system.py`): Event-based processing

##### Learning Systems (`learning/`)
- **Learning Engine** (`learning/learning_engine.py`): STDP and pattern learning
- **Live Hebbian Learning** (`learning/live_hebbian_learning.py`): Real-time learning with energy modulation
- **Memory System** (`learning/memory_system.py`): Memory formation and persistence
- **Homeostasis Controller** (`learning/homeostasis_controller.py`): Energy balance regulation

##### Sensory Integration (`sensory/`)
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

### Basic Usage with SOA

```python
from core.services.service_registry import ServiceRegistry
from core.services.simulation_coordinator import SimulationCoordinator
from core.services.neural_processing_service import NeuralProcessingService
from core.services.energy_management_service import EnergyManagementService
from core.services.learning_service import LearningService
from core.services.graph_management_service import GraphManagementService
from core.services.performance_monitoring_service import PerformanceMonitoringService
from core.services.configuration_service import ConfigurationService
from core.services.event_coordination_service import EventCoordinationService
from core.services.sensory_processing_service import SensoryProcessingService
from core.interfaces import *
from main_graph import create_test_graph

# Create service registry
registry = ServiceRegistry()

# Register all services with dependency injection
registry.register_instance(ISimulationCoordinator, SimulationCoordinator(registry))
registry.register_instance(INeuralProcessor, NeuralProcessingService())
registry.register_instance(IEnergyManager, EnergyManagementService())
registry.register_instance(ILearningEngine, LearningService())
registry.register_instance(IGraphManager, GraphManagementService())
registry.register_instance(IPerformanceMonitor, PerformanceMonitoringService())
registry.register_instance(IConfigurationService, ConfigurationService())
registry.register_instance(IEventCoordinator, EventCoordinationService())
registry.register_instance(ISensoryProcessor, SensoryProcessingService())

# Get coordinator and initialize
coordinator = registry.resolve(ISimulationCoordinator)
graph_manager = registry.resolve(IGraphManager)

# Create and set neural graph
graph = create_test_graph(num_sensory=100, num_dynamic=50)
graph_manager.set_graph(graph)

# Initialize and run simulation
coordinator.initialize_simulation()
coordinator.start_simulation()

# Run simulation steps
for step in range(1000):
    coordinator.run_simulation_step()

coordinator.stop_simulation()
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

The project follows a service-oriented architecture with clear separation of concerns. For detailed project structure and organization, see the [Documentation Index](docs/DOCUMENTATION_INDEX.md).

### High-Level Structure

```
ai-project/
├── core/                 # Core SOA infrastructure
│   ├── unified_launcher.py       # Application entry point
│   ├── services/                 # Service implementations
│   └── interfaces/               # Service contracts
├── src/                  # Main source code
│   ├── neural/           # Neural processing systems
│   ├── energy/           # Energy management systems
│   ├── learning/         # Learning and memory systems
│   ├── sensory/          # Sensory integration systems
│   ├── ui/               # User interface systems
│   └── utils/            # Utility systems
├── config/               # Configuration management
├── docs/                 # Documentation
├── tests/                # Test suite
└── requirements.txt      # Dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Documentation

The project includes comprehensive documentation organized for different audiences:

### 📖 For Users
- **[Quick Start Guide](docs/QUICK_START_GUIDE.md)** - Get started quickly with installation and basic usage
- **[README.md](README.md)** - This file with project overview and architecture
- **[Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Complete guide to all documentation

### 🔧 For Developers
- **[Consolidated Documentation](docs/CONSOLIDATED_DOCUMENTATION.md)** - Complete API reference and examples
- **[SOA Architecture](docs/SOA_ARCHITECTURE.md)** - Detailed service-oriented architecture design
- **[Service Interfaces](docs/SERVICE_INTERFACES.md)** - Service interface definitions and contracts
- **[Component Reference](docs/COMPONENT_REFERENCE.md)** - Component implementation details

### 🧬 For Researchers
- **[Energy-Learning Integration](docs/ENERGY_LEARNING_INTEGRATION.md)** - Biological modeling with energy modulation
- **[Energy System Analysis](docs/ENERGY_SYSTEM_ANALYSIS.md)** - Energy dynamics and validation
- **[Optimization Report](docs/OPTIMIZATION_REPORT.md)** - Performance analysis and recommendations

### 📊 Additional Resources
- **[Migration Patterns](docs/MIGRATION_PATTERNS.md)** - Code migration strategies
- **[Optimization Summary](docs/OPTIMIZATION_SUMMARY.md)** - Performance optimization overview

##  License

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
