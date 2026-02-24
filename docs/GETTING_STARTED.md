# Getting Started Guide

Welcome to the PyTorch Geometric Neural System! This guide will help you get started with installation, configuration, and running your first neural system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Understanding the UI](#understanding-the-ui)
6. [Next Steps](#next-steps)

## Prerequisites

Before installing, ensure you have:

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- Basic understanding of Python and neural networks

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/xxxpinkghostxxx/ai-project.git
cd ai-project
```

### Step 2: Set Up Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r config/requirements.txt
```

## Configuration

### Basic Configuration

The system uses a JSON configuration file at `src/project/pyg_config.json`. Key sections:

```json
{
    "sensory": {
        "enabled": true,
        "width": 1920,
        "height": 1080
    },
    "workspace": {
        "width": 128,
        "height": 128,
        "shading_mode": "linear",
        "color_scheme": "grayscale"
    },
    "system": {
        "device": "cuda",
        "update_interval": 16,
        "max_energy": 255.0
    },
    "hybrid": {
        "enabled": true,
        "grid_size": [2560, 1920],
        "tile_mode": true,
        "node_spawn_threshold": 8.0,
        "node_death_threshold": 1.0,
        "excitatory_prob": 0.6
    }
}
```

Static parameters (energy caps, decay rates, node limits) are defined in `src/project/config.py`.

### Configuration Options

- **Sensory**: Screen capture resolution and energy injection settings
- **Workspace**: Workspace grid size, shading, and color scheme
- **System**: Device selection (cuda/cpu), update intervals, energy bounds
- **Hybrid**: Grid dimensions, tile mode, spawn/death thresholds, connection probabilities

## Running the System

### Basic Usage

```bash
python -m project.pyg_main
```

### Command Line Options

```bash
# Run with a specific log level
python -m project.pyg_main --log-level DEBUG

# Available log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

To change the compute device or other settings, edit `src/project/pyg_config.json` (e.g., set `"device": "cpu"` under the `"system"` section).

### First Run Experience

1. The system will initialize with default parameters
2. A main window will appear showing the neural network visualization
3. The system will start capturing screen input and processing it
4. You'll see real-time metrics and network activity

## Understanding the UI

### Main Window Components

- **Network Visualization**: Shows nodes and connections in real-time
- **Metrics Panel**: Displays system performance and statistics
- **Control Panel**: Allows runtime parameter adjustment
- **Status Bar**: Shows current system state and messages

### Common UI Actions

- **Start/Stop Simulation**: Begin or halt neural processing
- **Reset Map**: Clear the current workspace visualization
- **Drain & Suspend**: Drain energy and suspend the system
- **Pulse +10 Energy**: Inject an energy pulse into the system
- **Disable/Enable Sensory Input**: Toggle screen capture input
- **Config Panel**: Open the configuration panel to adjust parameters
- **Test Rules**: Verify node interaction rules

## Next Steps

### Explore the API

Check out the [API Documentation](technical/API_DOCUMENTATION/) to understand the system architecture and available functions.

### Try Advanced Features

- Custom neural system configurations
- Performance profiling and optimization
- Integration with other systems

### Get Involved

- Report issues or suggest features
- Contribute code or documentation
- Join the community discussions

### Additional Resources

- [Troubleshooting Guide](technical/TROUBLESHOOTING.md)
- [Contribution Guidelines](CONTRIBUTING.md)
- [Documentation Standards](development/DOCUMENTATION_GUIDE.md)