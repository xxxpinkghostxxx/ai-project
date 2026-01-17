# Troubleshooting Guide

This guide helps you resolve common issues when working with the PyTorch Geometric Neural System project.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Performance Issues](#performance-issues)
4. [UI Problems](#ui-problems)
5. [Memory and Resource Issues](#memory-and-resource-issues)
6. [System Compatibility](#system-compatibility)
7. [Debugging Tips](#debugging-tips)

---

## Installation Issues

### Python Environment Problems

#### Issue: `python: command not found` or `pip: command not found`

**Solution:**

- Ensure Python 3.8+ is installed
- Use `python3` instead of `python` on some systems
- Check your PATH environment variable

```bash
# Windows
python --version

# Linux/Mac
python3 --version
```

#### Issue: Virtual environment activation fails

**Solution:**

```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# If using provided scripts
activate_env.bat  # Windows
source activate_env.sh  # Linux/Mac
```

### Dependency Installation Issues

#### Issue: `pip install` fails with permission errors

**Solution:**

```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/lib/python*/site-packages
```

#### Issue: PyTorch Geometric installation fails

**Solution:**

```bash
# Install PyTorch first
pip install torch torchvision

# Then install PyTorch Geometric
pip install torch-geometric

# Or use the provided requirements.txt
pip install -r requirements.txt
```

#### Issue: CUDA/GPU related errors during installation

**Solution:**

```bash
# Install CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install -r requirements.txt
```

---

## Runtime Errors

### Import Errors

#### Issue: `ModuleNotFoundError: No module named 'project'`

**Solution:**

1. Ensure you're running from the correct directory

2. Add project to Python path:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

1. Or run with proper module invocation:

```bash
python -m project.pyg_main
```

#### Issue: `ImportError: cannot import name 'X' from 'project.common_imports'`

**Solution:**

- Check the module export list in `common_imports.py`
- Ensure all required modules are properly imported
- Verify the `__all__` list includes the needed imports

### Configuration Errors

#### Issue: `KeyError` when accessing configuration values

**Solution:**

```python
from project.config import config

# Use get() with default values instead of direct access
node_count = config.get('INITIAL_PROCESSING_NODES', 100)
energy_cap = config.get('NODE_ENERGY_CAP', 100.0)

# Or check if key exists first
if 'INITIAL_PROCESSING_NODES' in config:
    node_count = config['INITIAL_PROCESSING_NODES']
```

#### Issue: Configuration file not found

**Solution:**

1. Check if `pyg_config.json` exists in the project directory

2. Verify the config file is valid JSON

3. Check file permissions

---

## Performance Issues

### Slow Processing

#### Issue: Neural system updates are very slow

**Solutions:**

1. **Use GPU acceleration:**

```python
from project.common_imports import get_device
device = get_device()  # Use 'cuda' if available
neural_system = PyGNeuralSystem(config, device=device)
```

1. **Reduce update frequency:**

```python
config['PERIODIC_UPDATE_MS'] = 100  # Increase interval
```

1. **Reduce node count:**

```python
config['INITIAL_PROCESSING_NODES'] = 50  # Decrease initial nodes
config['MAX_NODES'] = 100  # Limit maximum nodes
```

### High CPU Usage

#### Issue: System uses too much CPU

**Solutions:**

1. **Reduce screen capture frequency:**

```python
config['CAPTURE_INTERVAL_MS'] = 200  # Increase capture interval
```

1. **Lower screen resolution:**

```python
config['SENSOR_WIDTH'] = 32
config['SENSOR_HEIGHT'] = 32
```

1. **Enable energy-based throttling:**

```python
config['ENERGY_DECAY_RATE'] = 0.1  # Increase decay rate
```

---

## UI Problems

### Window Not Appearing

#### Issue: Main window doesn't open

**Solutions:**

1. **Check Tkinter installation:**

```python
import tkinter as tk
print("Tkinter version:", tk.TkVersion)
```

1. **Verify display environment (Linux):**

```bash
echo $DISPLAY
# Should show something like :0
```

1. **Run with explicit display:**

```bash
DISPLAY=:0 python project/pyg_main.py
```

### UI Freezing or Crashing

#### Issue: UI becomes unresponsive

**Solutions:**

1. **Check for infinite loops in update logic**

2. **Reduce update rates in configuration**

3. **Add proper exception handling**

```python
try:
    app.update_display()
except Exception as e:
    logger.error(f"UI update failed: {e}")
    # Continue operation
```

---

## Memory and Resource Issues

### High Memory Usage

#### Issue: System runs out of memory

**Solutions:**

1. **Reduce node limits:**

```python
config['MAX_NODES'] = 200  # Lower limit
```

1. **Increase cleanup frequency:**

```python
config['CLEANUP_INTERVAL_MS'] = 1000  # More frequent cleanup
```

1. **Enable garbage collection:**

```python
import gc
gc.collect()  # Manual cleanup
```

### GPU Memory Issues

#### Issue: CUDA out of memory errors

**Solutions:**

1. **Switch to CPU:**

```python
device = 'cpu'
```

1. **Reduce batch sizes and node counts**

1. **Clear GPU cache:**

```python
import torch
torch.cuda.empty_cache()
```

---

## System Compatibility

### Windows-Specific Issues

#### Issue: Screen capture doesn't work on Windows

**Solutions:**

1. **Check screen capture permissions**

2. **Try alternative capture method:**

```python
# In vision.py, try different MSS parameters
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Try different monitor
```

#### Issue: Path separator issues

**Solution:** Use `os.path.join()` for all paths:

```python
import os
config_path = os.path.join("project", "pyg_config.json")
```

### Linux-Specific Issues

#### Issue: X11 display errors

**Solutions:**

1. **Check X11 forwarding:**

```bash
export DISPLAY=:0
```

1. **Install X11 utilities:**

```bash
sudo apt-get install python3-tk
```

#### Issue: Permission denied for screen capture

**Solution:**

```bash
# Add user to necessary groups
sudo usermod -a -G audio,video $USER
```

### Mac-Specific Issues

#### Issue: Screen capture permissions

**Solution:**

1. Go to System Preferences → Security & Privacy → Screen Recording

2. Add Terminal/Python to allowed applications

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set specific logger levels
logging.getLogger('project').setLevel(logging.DEBUG)
```

### Use Profiling Tools

```python
from project.common_imports import profile_section, profile_report

with profile_section("problematic_function"):
    # Your code here
    pass

profile_report()  # Check performance
```

### Check System Information

```python
import platform
import torch

print("OS:", platform.system())
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
```

### Validate Configuration

```python
from project.config import config
from project.common_imports import validate_config

required_keys = [
    'INITIAL_PROCESSING_NODES',
    'NODE_ENERGY_CAP',
    'SENSOR_WIDTH',
    'SENSOR_HEIGHT'
]

if not validate_config(config, required_keys):
    print("Configuration is missing required keys!")
```

---

## Getting Help

### Log Files

Check these log files for detailed error information:

- `pyg_system.log` - Main application log
- Console output - Real-time error messages

### System Status

Monitor system status using the UI metrics panel or programmatically:

```python
from project.pyg_neural_system import PyGNeuralSystem

neural_system = PyGNeuralSystem(config)
status = neural_system.get_network_state()
print("System status:", status)
```

### Error Reporting

When reporting issues, include:

1. **System information** (OS, Python version, PyTorch version)

2. **Error messages** (full traceback)

3. **Configuration** (relevant config values)

4. **Steps to reproduce**

5. **Log file contents**

---

## Common Configuration Values

### For CPU-Only Systems

```python
{
    'device': 'cpu',
    'INITIAL_PROCESSING_NODES': 50,
    'MAX_NODES': 100,
    'PERIODIC_UPDATE_MS': 100,
    'CAPTURE_INTERVAL_MS': 200
}
```

### For High-Performance Systems

```python
{
    'device': 'cuda',
    'INITIAL_PROCESSING_NODES': 200,
    'MAX_NODES': 500,
    'PERIODIC_UPDATE_MS': 50,
    'CAPTURE_INTERVAL_MS': 50
}
```

### For Low-Resource Systems

```python
{
    'device': 'cpu',
    'INITIAL_PROCESSING_NODES': 25,
    'MAX_NODES': 50,
    'PERIODIC_UPDATE_MS': 200,
    'CAPTURE_INTERVAL_MS': 500
}
```

---

## Maintenance

### Regular Cleanup

1. **Clean log files:**

```bash
# Remove old log files
find . -name "*.log" -mtime +7 -delete
```

1. **Check disk space:**

```bash
df -h  # Check available space
```

1. **Monitor memory usage:**

```python
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

### Performance Monitoring

1. **Enable profiling regularly**

2. **Monitor node counts over time**

3. **Check energy distribution**

4. **Review connection statistics**

---

Remember: Most issues can be resolved by checking the configuration, ensuring proper dependencies, and verifying system compatibility. When in doubt, start with the simplest solution and work your way up to more complex troubleshooting steps.
