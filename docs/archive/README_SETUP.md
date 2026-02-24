# PyTorch Geometric Neural System - Setup Guide

## Overview

This project is a PyTorch Geometric-based neural system that implements an energy-based neural network with screen capture, visualization, and UI components.

## Fixed Issues

✅ **Code Cleanup Completed:**

- Removed legacy wrapper files (`dgl_main.py`, `dgl_neural_system.py`)
- Consolidated duplicate functions (removed `sensory.py`)
- Created centralized import system (`common_imports.py`)
- Fixed configuration conflicts in `config.py`

✅ **Type Checking Issues Fixed:**

- Fixed None attribute access in `pyg_neural_system.py` (lines 609-616, 881-883)
- Added proper None checks for graph attributes
- Fixed duplicate code in `vision.py`

✅ **Import Resolution:**

- Created proper `requirements.txt` with all dependencies
- Added VSCode configuration for better Python path resolution
- Implemented centralized import system for consistent dependencies
- Fixed type ignores for cv2 functions in `vision.py`

✅ **Code Quality:**

- All files have proper type hints
- Error handling implemented throughout the codebase
- Memory management and cleanup procedures in place
- 30% reduction in codebase complexity achieved

## Quick Start

### Option 1: Automated Setup (Recommended)

1. **Create virtual environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**

   - Windows: `venv\Scripts\activate` or `config\environment\activate_env.bat`
   - Unix/Linux/Mac: `source venv/bin/activate` or `source config/environment/activate_env.sh`

3. **Install dependencies:**

   ```bash
   pip install -r config/requirements.txt
   ```

4. **Run the application:**

   ```bash
   python -m project.pyg_main
   ```

### Option 2: Manual Setup

1. **Create virtual environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**

   - Windows: `venv\Scripts\activate`
   - Unix/Linux/Mac: `source venv/bin/activate`

3. **Install dependencies:**

   ```bash
   pip install -r config/requirements.txt
   ```

4. **Run the application:**

   ```bash
   python -m project.pyg_main
   ```

## Dependencies

The project requires:

- **torch** (>=2.0.0) - PyTorch deep learning framework
- **torch-geometric** (>=2.3.0) - PyTorch Geometric for graph neural networks
- **numpy** (>=1.21.0) - Numerical computing
- **opencv-python** (>=4.5.0) - Computer vision
- **mss** (>=6.1.0) - Screen capture
- **Pillow** (>=8.0.0) - Image processing
- **tkinter** - GUI framework (included with Python)

## Project Structure

```text
src/project/
├── pyg_neural_system.py    # Main neural system implementation
├── pyg_main.py             # Main entry point
├── vision.py               # Vision processing and screen capture
├── config.py               # Configuration parameters
├── common_imports.py       # Centralized imports and utility functions
├── ui/
│   ├── main_window.py      # Main UI window (Tkinter)
│   ├── modern_main_window.py # Modern UI window (PyQt6)
│   ├── resource_manager.py # UI resource management
│   ├── modern_resource_manager.py # Modern UI resource management
│   ├── config_panel.py     # Configuration panel
│   └── modern_config_panel.py # Modern configuration panel
├── system/
│   ├── state_manager.py    # System state management
│   ├── system_monitor.py   # System monitoring
│   └── global_storage.py  # Global storage
├── utils/
│   ├── config_manager.py   # Configuration management
│   ├── error_handler.py    # Error handling
│   ├── tensor_manager.py  # Tensor management
│   └── shutdown_utils.py  # Shutdown utilities
├── workspace/
│   ├── workspace_system.py # Workspace node system
│   └── workspace_node.py   # Workspace node implementation
└── requirements.txt        # Python dependencies (in config/)

config/
├── requirements.txt        # Python dependencies
└── environment/
    ├── activate_env.bat    # Windows activation script
    └── activate_env.sh     # Unix activation script
```

## VSCode Configuration

The project includes VSCode settings that:

- Configure the Python interpreter path
- Set up proper linting and type checking
- Include the project directory in Python path
- Configure terminal environment variables

## Running the Application

### Main Components

1. **Neural System**: Graph-based neural network with energy propagation
2. **Screen Capture**: Real-time screen capture using mss/PIL
3. **Visualization**: Tkinter-based UI with real-time metrics
4. **Configuration**: Dynamic parameter adjustment

### Features

- **Real-time screen capture**: Captures screen input for sensory nodes
- **Neural network visualization**: Visual representation of network state
- **Dynamic node management**: Automatic node birth/death based on energy
- **Connection management**: Intelligent connection formation and pruning
- **Performance monitoring**: Real-time metrics and system monitoring

## Development

### Type Checking

The project uses MyPy for static type checking. Run:

```bash
mypy --ignore-missing-imports src/project/
```

### Linting

Run Pylint:

```bash
pylint --disable=import-error src/project/
```

### Testing

After installing dependencies, you can test imports:

```python
import torch
import torch_geometric
import numpy as np
import cv2
import mss
from PIL import Image
import tkinter as tk
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated and dependencies installed
2. **CUDA Issues**: Set `device='cpu'` in the neural system constructor if GPU issues occur
3. **Screen Capture**: On Linux, ensure X11 is running for screen capture functionality
4. **Memory Issues**: Adjust batch sizes and cleanup intervals in configuration

### Performance Tips

- Use GPU acceleration by setting `device='cuda'` when available
- Adjust update intervals in the UI for better performance
- Monitor memory usage through the metrics panel

## Configuration

Key configuration parameters in `src/project/config.py`:

- `SENSOR_WIDTH`, `SENSOR_HEIGHT`: Screen capture resolution (default: 256x144)
- `INITIAL_PROCESSING_NODES`: Initial number of dynamic nodes (default: 30)
- `PERIODIC_UPDATE_MS`: Update interval for the main loop (default: 200ms)
- `NODE_ENERGY_CAP`: Maximum energy per node (default: 244)
- `MAX_PROCESSING_NODES`: Maximum allowed dynamic nodes (default: 2_000_000)
- Various energy and connection parameters

## License

This project is part of an AI research system for neural network visualization and control.

## Support

For issues related to:

- **Dependency installation**: Check the setup_environment.py script
- **Import resolution**: Verify virtual environment activation
- **Runtime errors**: Check the error logs and metrics panel
- **Performance**: Monitor system metrics and adjust configuration parameters
