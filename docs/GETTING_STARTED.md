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
git clone https://github.com/your-repo/ai-project.git
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

### Step 4: Activate Environment

```bash
# Windows
activate_env.bat

# Linux/Mac
source activate_env.sh
```

## Configuration

### Basic Configuration

The system uses a JSON configuration file located at `project/pyg_config.json`. Here's a basic configuration:

```json
{
    "INITIAL_PROCESSING_NODES": 100,
    "NODE_ENERGY_CAP": 100.0,
    "NODE_DEATH_THRESHOLD": 10.0,
    "NODE_SPAWN_THRESHOLD": 90.0,
    "SENSOR_WIDTH": 64,
    "SENSOR_HEIGHT": 64,
    "CAPTURE_INTERVAL_MS": 100,
    "MOTION_SENSITIVITY": 0.1,
    "PERIODIC_UPDATE_MS": 50,
    "MAX_NODES": 1000,
    "ENERGY_DECAY_RATE": 0.01
}
```

### Configuration Options

- **Neural System**: Control node behavior and energy dynamics
- **Vision System**: Adjust screen capture resolution and sensitivity
- **Performance**: Set update intervals and maximum node limits

## Running the System

### Basic Usage

```bash
python -m project.pyg_main
```

### Command Line Options

```bash
# Run with specific configuration
python -m project.pyg_main --config custom_config.json

# Run in debug mode
python -m project.pyg_main --debug

# Run with CPU only (no CUDA)
python -m project.pyg_main --device cpu
```

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

- **Pause/Resume**: Temporarily stop neural processing
- **Reset**: Clear the current network state
- **Configuration**: Adjust parameters on-the-fly
- **Export**: Save current network state

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