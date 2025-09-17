# Examples and Tutorials

## Getting Started

### Basic Simulation

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

# Run simulation for 1000 steps
for step in range(1000):
    success = sim_manager.run_single_step()
    if not success:
        print(f"Simulation stopped at step {step}")
        break

# Stop simulation
sim_manager.stop_simulation()
```

### Enhanced Neural Network

```python
from enhanced_neural_integration import create_enhanced_neural_integration
from main_graph import initialize_main_graph

# Create enhanced integration
integration = create_enhanced_neural_integration()

# Initialize graph
graph = initialize_main_graph(scale=0.25)

# Create sophisticated nodes
integration.create_enhanced_node(
    graph, node_id=0, node_type='dynamic',
    subtype='oscillator', is_excitatory=True,
    oscillation_frequency=2.0, energy=0.8
)

integration.create_enhanced_node(
    graph, node_id=1, node_type='dynamic',
    subtype='integrator', is_excitatory=True,
    integration_rate=0.8, energy=0.7
)

# Create advanced connections
integration.create_enhanced_connection(
    graph, source_id=0, target_id=1,
    connection_type='excitatory', weight=1.5,
    plasticity_enabled=True, learning_rate=0.02
)

# Run enhanced simulation
for step in range(1000):
    graph = integration.integrate_with_existing_system(graph, step)
```

## Advanced Examples

### 1. Visual Processing Pipeline

```python
import numpy as np
from visual_energy_bridge import create_visual_energy_bridge
from enhanced_neural_integration import create_enhanced_neural_integration
from main_graph import initialize_main_graph

# Create systems
integration = create_enhanced_neural_integration()
visual_bridge = create_visual_energy_bridge(integration)
graph = initialize_main_graph(scale=0.25)

# Simulate visual processing
for step in range(1000):
    # Generate synthetic visual data (replace with real screen capture)
    screen_data = np.random.rand(100, 100, 3) * 255
    
    # Process visual input
    graph = visual_bridge.process_visual_to_enhanced_energy(
        graph, screen_data, step
    )
    
    # Update neural dynamics
    graph = integration.integrate_with_existing_system(graph, step)
    
    # Get visual statistics
    if step % 100 == 0:
        stats = visual_bridge.get_visual_statistics()
        print(f"Step {step}: Visual patterns detected: {stats['patterns_detected']}")
```

### 2. Audio Processing Pipeline

```python
import numpy as np
from audio_to_neural_bridge import create_audio_to_neural_bridge
from simulation_manager import create_simulation_manager

# Create systems
sim_manager = create_simulation_manager()
audio_bridge = create_audio_to_neural_bridge(sim_manager)
graph = sim_manager.graph

# Simulate audio processing
for step in range(1000):
    # Generate synthetic audio data (replace with real audio input)
    audio_data = np.random.randn(44100)  # 1 second of audio
    
    # Process audio input
    sensory_nodes = audio_bridge.process_audio_to_sensory_nodes(audio_data)
    
    # Integrate into graph
    graph = audio_bridge.integrate_audio_nodes_into_graph(graph, audio_data)
    
    # Get audio statistics
    if step % 100 == 0:
        stats = audio_bridge.get_audio_feature_statistics()
        print(f"Step {step}: Audio features extracted: {stats['cached_features']}")
```

### 3. Memory Formation and Recall

```python
from memory_system import MemorySystem
from learning_engine import LearningEngine
from main_graph import initialize_main_graph

# Create systems
memory_system = MemorySystem()
learning_engine = LearningEngine()
graph = initialize_main_graph(scale=0.25)

# Simulate memory formation
for step in range(1000):
    # Form memory traces from stable patterns
    graph = memory_system.form_memory_traces(graph)
    
    # Apply learning
    graph = learning_engine.consolidate_connections(graph)
    
    # Get memory statistics
    if step % 100 == 0:
        stats = memory_system.get_memory_statistics()
        print(f"Step {step}: Memory traces: {stats['memory_traces_formed']}")
        
        # Demonstrate recall
        if stats['memory_traces_formed'] > 0:
            recalled_patterns = memory_system.recall_patterns(graph, target_node_idx=0)
            print(f"Recalled patterns: {len(recalled_patterns)}")
```

### 4. Homeostatic Regulation

```python
from homeostasis_controller import HomeostasisController
from main_graph import initialize_main_graph

# Create systems
homeostasis = HomeostasisController()
graph = initialize_main_graph(scale=0.25)

# Simulate homeostatic regulation
for step in range(1000):
    # Regulate network activity
    graph = homeostasis.regulate_network_activity(graph)
    
    # Optimize criticality
    graph = homeostasis.optimize_criticality(graph)
    
    # Monitor network health
    health_data = homeostasis.monitor_network_health(graph)
    
    # Get regulation statistics
    if step % 100 == 0:
        stats = homeostasis.get_regulation_statistics()
        print(f"Step {step}: Energy regulation: {stats['energy_regulations']}")
        print(f"Criticality optimizations: {stats['criticality_optimizations']}")
```

### 5. Event-Driven Processing

```python
from event_driven_system import create_event_driven_system
from simulation_manager import create_simulation_manager

# Create systems
sim_manager = create_simulation_manager()
event_system = create_event_driven_system(sim_manager)

# Start event system
event_system.start()

# Simulate event-driven processing
for step in range(1000):
    # Schedule some events
    if step % 10 == 0:
        event_system.schedule_spike(node_id=0, timestamp=step * 0.01, priority=1)
    
    # Process events
    events_processed = event_system.process_events(max_events=100)
    
    # Get event statistics
    if step % 100 == 0:
        stats = event_system.get_statistics()
        print(f"Step {step}: Events processed: {events_processed}")
        print(f"Total events: {stats['total_events_processed']}")

# Stop event system
event_system.stop()
```

## Tutorial: Building a Custom Neural Network

### Step 1: Create Basic Network Structure

```python
from main_graph import initialize_main_graph
from enhanced_neural_integration import create_enhanced_neural_integration

# Initialize graph
graph = initialize_main_graph(scale=0.25)
integration = create_enhanced_neural_integration()

print(f"Initial graph has {len(graph.node_labels)} nodes")
```

### Step 2: Add Specialized Nodes

```python
# Create oscillator nodes for rhythm generation
for i in range(5):
    integration.create_enhanced_node(
        graph, node_id=i, node_type='dynamic',
        subtype='oscillator', is_excitatory=True,
        oscillation_frequency=1.0 + i * 0.5, energy=0.8
    )

# Create integrator nodes for information processing
for i in range(5, 10):
    integration.create_enhanced_node(
        graph, node_id=i, node_type='dynamic',
        subtype='integrator', is_excitatory=True,
        integration_rate=0.5 + (i-5) * 0.1, energy=0.6
    )

# Create relay nodes for signal amplification
for i in range(10, 15):
    integration.create_enhanced_node(
        graph, node_id=i, node_type='dynamic',
        subtype='relay', is_excitatory=True,
        relay_amplification=1.0 + (i-10) * 0.2, energy=0.7
    )

print(f"Enhanced graph has {len(graph.node_labels)} nodes")
```

### Step 3: Create Connections

```python
# Connect oscillators to integrators
for i in range(5):
    for j in range(5, 10):
        integration.create_enhanced_connection(
            graph, source_id=i, target_id=j,
            connection_type='excitatory', weight=0.8,
            plasticity_enabled=True, learning_rate=0.01
        )

# Connect integrators to relays
for i in range(5, 10):
    for j in range(10, 15):
        integration.create_enhanced_connection(
            graph, source_id=i, target_id=j,
            connection_type='excitatory', weight=1.2,
            plasticity_enabled=True, learning_rate=0.02
        )

# Create lateral connections between integrators
for i in range(5, 9):
    integration.create_enhanced_connection(
        graph, source_id=i, target_id=i+1,
        connection_type='excitatory', weight=0.5,
        plasticity_enabled=True, learning_rate=0.005
    )

print("Connections created")
```

### Step 4: Run Simulation

```python
# Run simulation
for step in range(1000):
    graph = integration.integrate_with_existing_system(graph, step)
    
    # Monitor network activity
    if step % 100 == 0:
        # Count active nodes
        active_nodes = sum(1 for node in graph.node_labels 
                          if node.get('state') == 'active')
        print(f"Step {step}: {active_nodes} active nodes")
        
        # Get integration statistics
        stats = integration.get_integration_statistics()
        print(f"Total updates: {stats['total_updates']}")
```

## Tutorial: Implementing Custom Node Behavior

### Step 1: Create Custom Node Class

```python
from enhanced_node_behaviors import EnhancedNodeBehavior
from typing import Dict, Any

class CustomNodeBehavior(EnhancedNodeBehavior):
    def __init__(self, node_id: int, **kwargs):
        super().__init__(node_id, 'custom', **kwargs)
        self.custom_parameter = kwargs.get('custom_parameter', 1.0)
        self.custom_state = 'idle'
    
    def update_behavior(self, graph, step, access_layer):
        # Custom behavior logic
        if self.custom_state == 'idle':
            if self.energy > 0.5:
                self.custom_state = 'active'
                self._fire_spike(access_layer)
        elif self.custom_state == 'active':
            if self.energy < 0.3:
                self.custom_state = 'idle'
        
        # Update custom parameter
        self.custom_parameter *= 0.99
        
        return True
```

### Step 2: Integrate Custom Behavior

```python
from enhanced_node_behaviors import EnhancedNodeBehaviorSystem

# Create custom behavior system
behavior_system = EnhancedNodeBehaviorSystem()

# Add custom behavior
custom_behavior = CustomNodeBehavior(
    node_id=0, 
    custom_parameter=2.0,
    energy=0.8
)
behavior_system.node_behaviors[0] = custom_behavior

# Update behaviors
for step in range(1000):
    behavior_system.update_node_behaviors(graph, step)
```

## Tutorial: Performance Optimization

### Step 1: Monitor Performance

```python
from performance_monitor import get_performance_monitor

# Get performance monitor
monitor = get_performance_monitor()

# Start monitoring
monitor.start_monitoring()

# Run simulation with monitoring
for step in range(1000):
    # Your simulation code here
    sim_manager.run_single_step()
    
    # Check performance every 100 steps
    if step % 100 == 0:
        metrics = monitor.get_current_metrics()
        print(f"Step {step}:")
        print(f"  Memory: {metrics.memory_usage_mb:.1f} MB")
        print(f"  CPU: {metrics.cpu_percent:.1f}%")
        print(f"  FPS: {metrics.fps:.1f}")
        
        # Check for performance issues
        if metrics.memory_usage_mb > 2000:
            print("Warning: High memory usage!")
        if metrics.cpu_percent > 80:
            print("Warning: High CPU usage!")
```

### Step 2: Optimize Performance

```python
# Optimize memory usage
import gc

# Force garbage collection periodically
if step % 1000 == 0:
    gc.collect()

# Optimize graph size
if len(graph.node_labels) > 10000:
    # Prune weak connections
    from connection_logic import update_connection_weights
    graph = update_connection_weights(graph, learning_rate=0.001)
    
    # Remove inactive nodes
    from death_and_birth_logic import remove_dead_dynamic_nodes
    graph = remove_dead_dynamic_nodes(graph)

# Optimize update frequency
if step % 10 != 0:  # Skip some updates
    continue
```

## Tutorial: Real-Time Visualization

### Step 1: Set Up UI

```python
from ui_engine import create_ui
import dearpygui.dearpygui as dpg

# Create UI
dpg.create_context()
create_ui()

# Main loop
while dpg.is_dearpygui_running():
    # Update simulation
    sim_manager.run_single_step()
    
    # Update UI
    dpg.render_dearpygui_frame()

dpg.destroy_context()
```

### Step 2: Custom Visualization

```python
def create_custom_visualization():
    with dpg.window(label="Neural Network", width=800, height=600):
        # Add custom visualization elements
        dpg.add_text("Neural Network Visualization")
        
        # Add real-time metrics
        dpg.add_text("Active Nodes: ", tag="active_nodes")
        dpg.add_text("Total Energy: ", tag="total_energy")
        dpg.add_text("Learning Rate: ", tag="learning_rate")
        
        # Add controls
        dpg.add_slider_float("Learning Rate", 0.001, 0.1, 
                            default_value=0.01, tag="lr_slider")
        dpg.add_button("Reset Network", callback=reset_network)

def update_visualization():
    # Update metrics
    active_nodes = sum(1 for node in graph.node_labels 
                      if node.get('state') == 'active')
    dpg.set_value("active_nodes", f"Active Nodes: {active_nodes}")
    
    total_energy = sum(node.get('energy', 0) for node in graph.node_labels)
    dpg.set_value("total_energy", f"Total Energy: {total_energy:.2f}")
    
    learning_rate = sim_manager.learning_engine.learning_rate
    dpg.set_value("learning_rate", f"Learning Rate: {learning_rate:.3f}")
```

## Common Patterns and Best Practices

### 1. Error Handling

```python
try:
    # Simulation code
    sim_manager.run_single_step()
except Exception as e:
    print(f"Error in simulation: {e}")
    # Handle error gracefully
    sim_manager.stop_simulation()
```

### 2. Configuration Management

```python
from config_manager import get_config

# Get configuration values
learning_rate = get_config('Learning', 'plasticity_rate', 0.01, float)
oscillator_freq = get_config('EnhancedNodes', 'oscillator_frequency', 0.1, float)

# Use in simulation
sim_manager.learning_engine.learning_rate = learning_rate
```

### 3. Logging

```python
from logging_utils import log_step

# Log important events
log_step("Simulation started", step=0)
log_step("Node created", node_id=5, node_type="oscillator")
log_step("Connection formed", source=0, target=1, weight=0.8)
```

### 4. Memory Management

```python
# Clean up unused objects
def cleanup_simulation():
    sim_manager.cleanup()
    gc.collect()

# Call periodically
if step % 1000 == 0:
    cleanup_simulation()
```

---

These examples and tutorials provide a comprehensive guide to using the neural simulation system. Start with the basic examples and gradually work your way up to the more advanced features.
