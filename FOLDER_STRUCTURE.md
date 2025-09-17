# AI Neural Simulation System - Folder Structure

## Project Organization

The project has been reorganized into a clear, purpose-based folder structure:

```
ai-project/
├── core/                           # Core simulation components
│   ├── __init__.py
│   ├── main_graph.py               # Graph creation and management
│   └── unified_launcher.py         # Main application launcher
│
├── neural/                         # Neural network components
│   ├── __init__.py
│   ├── enhanced_neural_integration.py
│   ├── enhanced_neural_dynamics.py
│   ├── enhanced_node_behaviors.py
│   ├── enhanced_connection_system.py
│   ├── behavior_engine.py
│   ├── connection_logic.py
│   ├── dynamic_nodes.py
│   ├── event_driven_system.py
│   ├── spike_queue_system.py
│   ├── workspace_engine.py
│   ├── network_metrics.py
│   ├── neural_map_persistence.py
│   └── death_and_birth_logic.py
│
├── energy/                         # Energy management
│   ├── __init__.py
│   ├── energy_behavior.py
│   ├── energy_constants.py
│   ├── node_access_layer.py
│   └── node_id_manager.py
│
├── learning/                       # Learning and memory
│   ├── __init__.py
│   ├── learning_engine.py
│   ├── live_hebbian_learning.py
│   ├── memory_system.py
│   ├── memory_pool_manager.py
│   └── homeostasis_controller.py
│
├── sensory/                        # Sensory integration
│   ├── __init__.py
│   ├── audio_to_neural_bridge.py
│   ├── visual_energy_bridge.py
│   └── sensory_workspace_mapper.py
│
├── ui/                            # User interface
│   ├── __init__.py
│   ├── ui_engine.py
│   ├── ui_state_manager.py
│   └── screen_graph.py
│
├── utils/                         # Utility components
│   ├── __init__.py
│   ├── common_utils.py
│   ├── pattern_consolidation_utils.py
│   ├── print_utils.py
│   ├── statistics_utils.py
│   ├── logging_utils.py
│   ├── random_seed_manager.py
│   ├── error_handler.py
│   ├── error_handling_utils.py
│   ├── exception_utils.py
│   ├── unified_error_handler.py
│   ├── performance_monitor.py
│   ├── performance_optimizer.py
│   ├── unified_performance_system.py
│   └── static_allocator.py
│
├── config/                        # Configuration
│   ├── __init__.py
│   ├── config_manager.py
│   ├── dynamic_config_manager.py
│   ├── unified_config_manager.py
│   ├── consolidated_constants.py
│   └── config.ini
│
├── tests/                         # Testing
│   ├── __init__.py
│   ├── unified_test_suite.py
│   ├── unified_testing_system.py
│   └── comprehensive_test_framework.py
│
├── analysis/                      # Analysis and optimization
│   ├── __init__.py
│   ├── duplicate_code_detector.py
│   ├── focused_optimizer.py
│   ├── nasa_code_analyzer.py
│   └── verify_nasa_compliance.py
│
├── docs/                          # Documentation
│   ├── README.md
│   ├── COMPONENT_REFERENCE.md
│   ├── CONSOLIDATED_DOCUMENTATION.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── PROJECT_STATUS.md
│   ├── COMPREHENSIVE_IMPROVEMENT_PLAN.md
│   ├── FINAL_IMPROVEMENT_SUMMARY.md
│   ├── UNIFIED_CONSOLIDATION_SUMMARY.md
│   └── CONSOLIDATION_SUMMARY.md
│
├── logs/                          # Log files
│   └── (various log files)
│
├── neural_maps/                   # Neural map data
│   ├── neural_map_slot_0.json
│   └── slot_metadata.json
│
├── venv/                          # Virtual environment
│   └── (Python virtual environment)
│
├── requirements.txt               # Python dependencies
└── FOLDER_STRUCTURE.md           # This file
```

## Folder Purposes

### Core (`core/`)
Contains the main simulation components that orchestrate the entire system:
- **main_graph.py**: Graph creation and management utilities
- **unified_launcher.py**: Main application launcher with different profiles

### Neural (`neural/`)
Contains all neural network simulation components:
- Enhanced neural dynamics, integration, and behaviors
- Connection systems and event-driven processing
- Network metrics and persistence

### Energy (`energy/`)
Contains energy management and node access components:
- Energy behavior and constants
- Node access layer and ID management

### Learning (`learning/`)
Contains learning and memory-related components:
- Learning engines and Hebbian learning
- Memory systems and homeostasis control

### Sensory (`sensory/`)
Contains sensory integration components:
- Audio and visual processing bridges
- Sensory workspace mapping

### UI (`ui/`)
Contains user interface components:
- UI engine and state management
- Screen graph visualization

### Utils (`utils/`)
Contains utility and helper components:
- Common utilities and pattern consolidation
- Print, statistics, and logging utilities
- Error handling and performance systems

### Config (`config/`)
Contains configuration management:
- Configuration managers and constants
- Unified configuration system

### Tests (`tests/`)
Contains testing components:
- Unified test suite and testing system
- Comprehensive test framework

### Analysis (`analysis/`)
Contains analysis and optimization tools:
- Duplicate code detection and optimization
- NASA compliance analysis

### Docs (`docs/`)
Contains all documentation files:
- README, component references, and architecture docs
- Project status and improvement plans

## Benefits of This Structure

1. **Clear Separation of Concerns**: Each folder has a specific purpose
2. **Easy Navigation**: Developers can quickly find relevant components
3. **Modular Design**: Components are logically grouped together
4. **Maintainability**: Easier to maintain and update specific functionality
5. **Scalability**: Easy to add new components to appropriate folders
6. **Python Package Structure**: Proper `__init__.py` files for imports

## Import Examples

With this new structure, imports become more organized:

```python
# Core components
from core import main_graph, unified_launcher

# Neural components
from neural import behavior_engine, connection_logic

# Energy management
from energy import energy_behavior, node_id_manager

# Learning and memory
from learning import learning_engine, memory_system

# Sensory processing
from sensory import audio_to_neural_bridge, visual_energy_bridge

# UI components
from ui import ui_engine, ui_state_manager

# Utilities
from utils import common_utils, print_utils

# Configuration
from config import config_manager, consolidated_constants

# Testing
from tests import unified_test_suite

# Analysis
from analysis import duplicate_code_detector
```
