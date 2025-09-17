# Troubleshooting Guide

## Common Issues and Solutions

### 1. Installation and Setup Issues

#### Issue: Import Errors
**Symptoms**: `ModuleNotFoundError` or `ImportError` when running the simulation

**Solutions**:
```bash
# Install missing dependencies
pip install -r requirements.txt

# For specific missing modules
pip install torch torch-geometric dearpygui numpy

# Check Python version (requires 3.8+)
python --version
```

#### Issue: Configuration File Not Found
**Symptoms**: `FileNotFoundError: config.ini`

**Solutions**:
```python
# The system will create a default config.ini automatically
# If it doesn't, create one manually:
from config_manager import ConfigManager
config = ConfigManager()
config._create_default_config()
```

#### Issue: Permission Errors
**Symptoms**: Permission denied when accessing files

**Solutions**:
```bash
# On Windows, run as administrator
# On Linux/Mac, check file permissions
chmod 644 config.ini
chmod 755 neural_maps/
```

### 2. Runtime Issues

#### Issue: Simulation Stops Unexpectedly
**Symptoms**: Simulation stops without error or with generic error

**Solutions**:
```python
# Enable detailed error logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check simulation state
print(f"Simulation running: {sim_manager.simulation_running}")
print(f"Current step: {sim_manager.current_step}")

# Check for errors in performance stats
stats = sim_manager.get_performance_stats()
print(f"Errors: {stats.get('errors', 0)}")
print(f"Warnings: {stats.get('warnings', 0)}")
```

#### Issue: High Memory Usage
**Symptoms**: System becomes slow or crashes due to memory usage

**Solutions**:
```python
# Monitor memory usage
from performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
metrics = monitor.get_current_metrics()
print(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")

# Force garbage collection
import gc
gc.collect()

# Prune graph if too large
if len(graph.node_labels) > 10000:
    from death_and_birth_logic import remove_dead_dynamic_nodes
    graph = remove_dead_dynamic_nodes(graph)
```

#### Issue: Low Performance/FPS
**Symptoms**: Simulation runs slowly, low FPS

**Solutions**:
```python
# Check performance metrics
metrics = monitor.get_current_metrics()
print(f"FPS: {metrics.fps:.1f}")
print(f"CPU: {metrics.cpu_percent:.1f}%")

# Reduce update frequency
if step % 10 != 0:  # Update every 10th step
    continue

# Reduce graph complexity
if len(graph.node_labels) > 5000:
    # Prune weak connections
    from connection_logic import update_connection_weights
    graph = update_connection_weights(graph, learning_rate=0.001)
```

### 3. Neural Network Issues

#### Issue: No Neural Activity
**Symptoms**: All nodes remain inactive, no spikes generated

**Solutions**:
```python
# Check node energies
for i, node in enumerate(graph.node_labels):
    energy = graph.x[i].item() if hasattr(graph, 'x') else 0
    print(f"Node {i}: energy={energy:.3f}, state={node.get('state')}")

# Increase energy levels
for i in range(len(graph.node_labels)):
    current_energy = graph.x[i].item()
    graph.x[i] = max(current_energy, 0.5)  # Minimum energy threshold

# Check thresholds
for node in graph.node_labels:
    threshold = node.get('threshold', 0.5)
    print(f"Node threshold: {threshold}")
```

#### Issue: Unstable Network Dynamics
**Symptoms**: Network oscillates wildly or becomes chaotic

**Solutions**:
```python
# Check homeostatic regulation
from homeostasis_controller import HomeostasisController
homeostasis = HomeostasisController()
graph = homeostasis.regulate_network_activity(graph)

# Adjust learning rates
from config_manager import get_config
learning_rate = get_config('Learning', 'plasticity_rate', 0.01, float)
# Reduce learning rate if too high
if learning_rate > 0.05:
    learning_rate = 0.01
```

#### Issue: Memory Formation Not Working
**Symptoms**: No memory traces are formed

**Solutions**:
```python
# Check memory system
from memory_system import MemorySystem
memory_system = MemorySystem()

# Ensure stable patterns exist
for node in graph.node_labels:
    if node.get('state') == 'active':
        # Check if node has stable connections
        connections = [edge for edge in graph.edge_attributes 
                     if edge.source == node['id'] or edge.target == node['id']]
        if len(connections) > 2:  # Need stable connections
            memory_system.form_memory_traces(graph)
            break
```

### 4. UI Issues

#### Issue: UI Not Responding
**Symptoms**: DearPyGui interface freezes or doesn't update

**Solutions**:
```python
# Check if UI is running
import dearpygui.dearpygui as dpg
if not dpg.is_dearpygui_running():
    dpg.create_context()
    create_ui()

# Ensure proper rendering loop
while dpg.is_dearpygui_running():
    # Update simulation
    sim_manager.run_single_step()
    
    # Render UI
    dpg.render_dearpygui_frame()
    
    # Small delay to prevent overwhelming
    time.sleep(0.01)
```

#### Issue: Visualization Not Updating
**Symptoms**: UI displays static information

**Solutions**:
```python
# Force UI update
from ui_state_manager import get_ui_state_manager
ui_state = get_ui_state_manager()
ui_state.update_graph(graph)

# Check if graph is being updated
print(f"Graph nodes: {len(graph.node_labels)}")
print(f"Graph edges: {graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0}")
```

### 5. Configuration Issues

#### Issue: Configuration Values Not Applied
**Symptoms**: Changes to config.ini don't take effect

**Solutions**:
```python
# Reload configuration
from config_manager import get_config
config = get_config()
config.reload()

# Check if values are loaded
learning_rate = get_config('Learning', 'plasticity_rate', 0.01, float)
print(f"Learning rate: {learning_rate}")

# Restart simulation to apply changes
sim_manager.stop_simulation()
sim_manager.start_simulation()
```

#### Issue: Invalid Configuration Values
**Symptoms**: Errors when reading configuration

**Solutions**:
```python
# Validate configuration
try:
    config = get_config('Learning', 'plasticity_rate', 0.01, float)
    if not 0.001 <= config <= 0.1:
        print("Warning: Learning rate out of range")
        config = 0.01
except Exception as e:
    print(f"Configuration error: {e}")
    # Use default value
    config = 0.01
```

### 6. Event System Issues

#### Issue: Events Not Processing
**Symptoms**: Events are scheduled but not processed

**Solutions**:
```python
# Check event system state
from event_driven_system import create_event_driven_system
event_system = create_event_driven_system(sim_manager)

if not event_system.running:
    event_system.start()

# Check event queue size
queue_size = event_system.event_queue.size()
print(f"Event queue size: {queue_size}")

# Process events manually
events_processed = event_system.process_events(max_events=100)
print(f"Events processed: {events_processed}")
```

#### Issue: Event Processing Too Slow
**Symptoms**: Events accumulate faster than they're processed

**Solutions**:
```python
# Increase processing batch size
events_processed = event_system.process_events(max_events=1000)

# Process events more frequently
if step % 5 == 0:  # Process every 5th step
    event_system.process_events(max_events=500)

# Check event priorities
for event in event_system.event_queue.get_events_in_timeframe(0, 100):
    print(f"Event: {event.event_type}, priority: {event.priority}")
```

### 7. Performance Issues

#### Issue: Slow Startup
**Symptoms**: System takes long time to initialize

**Solutions**:
```python
# Profile startup time
import time
start_time = time.time()

# Initialize components
sim_manager = create_simulation_manager()
graph = initialize_main_graph(scale=0.25)

end_time = time.time()
print(f"Startup time: {end_time - start_time:.2f} seconds")

# Use smaller initial graph
graph = initialize_main_graph(scale=0.1)  # Smaller scale
```

#### Issue: Memory Leaks
**Symptoms**: Memory usage increases over time

**Solutions**:
```python
# Monitor memory usage
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.1f} MB")

# Force garbage collection
import gc
gc.collect()

# Clean up unused objects
sim_manager.cleanup()
```

### 8. Debugging Techniques

#### Enable Debug Logging
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Check System Health
```python
# Get system health score
health_score = sim_manager._calculate_system_health_score()
print(f"System health: {health_score:.2f}")

# Check error handler
from error_handler import get_error_handler
error_handler = get_error_handler()
health = error_handler.get_system_health()
print(f"Error handler health: {health}")
```

#### Monitor Performance
```python
# Get performance metrics
from performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
metrics = monitor.get_current_metrics()

print(f"Performance metrics:")
print(f"  Memory: {metrics.memory_usage_mb:.1f} MB")
print(f"  CPU: {metrics.cpu_percent:.1f}%")
print(f"  FPS: {metrics.fps:.1f}")
print(f"  Step time: {metrics.step_time:.3f}s")
```

#### Validate Graph Consistency
```python
# Check graph consistency
from node_access_layer import NodeAccessLayer
access_layer = NodeAccessLayer(graph)
consistency = access_layer.validate_consistency()
print(f"Graph consistency: {consistency}")

# Check node counts
print(f"Total nodes: {len(graph.node_labels)}")
print(f"Total edges: {graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0}")
```

### 9. Recovery Procedures

#### Reset Simulation
```python
# Complete reset
sim_manager.stop_simulation()
sim_manager.reset_simulation()
graph = initialize_main_graph(scale=0.25)
sim_manager.set_graph(graph)
sim_manager.start_simulation()
```

#### Recover from Errors
```python
# Use error handler for recovery
from error_handler import get_error_handler
error_handler = get_error_handler()

try:
    # Simulation code
    sim_manager.run_single_step()
except Exception as e:
    # Attempt recovery
    recovery_success = error_handler.handle_error(
        e, "Simulation error", 
        recovery_func=lambda: sim_manager.reset_simulation(),
        critical=False
    )
    
    if recovery_success:
        print("Recovery successful")
    else:
        print("Recovery failed, stopping simulation")
        sim_manager.stop_simulation()
```

### 10. Getting Help

#### Check Logs
```python
# Get log lines
from logging_utils import get_log_lines
log_lines = get_log_lines()
for line in log_lines[-10:]:  # Last 10 lines
    print(line)
```

#### Report Issues
When reporting issues, include:
1. Python version
2. Operating system
3. Error messages and stack traces
4. Configuration file contents
5. Steps to reproduce the issue
6. Performance metrics if applicable

#### Common Error Messages

**"Invalid node ID"**: Node ID doesn't exist in the graph
**"Graph consistency error"**: Graph structure is corrupted
**"Memory allocation failed"**: System is out of memory
**"Configuration error"**: Invalid configuration values
**"Import error"**: Missing dependencies

---

This troubleshooting guide covers the most common issues and their solutions. For additional help, check the logs and performance metrics, and consider reducing the complexity of your simulation if you encounter persistent issues.
