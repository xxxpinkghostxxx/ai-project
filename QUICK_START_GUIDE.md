# Neural Simulation Optimization - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will help you quickly set up and use the optimized neural simulation system.

## Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Dear PyGUI (for UI)
- Basic understanding of neural networks

## 1. Apply Optimizations

```python
from utils.optimization_applier import apply_optimizations, OptimizationConfig

# Quick setup with default optimizations
config = OptimizationConfig()
applier = apply_optimizations(config)

print("Optimizations applied successfully!")
print(applier.get_optimization_summary())
```

## 2. Run Optimized Simulation

```python
from simulation_manager import SimulationManager

# Create optimized simulation manager
sim_manager = SimulationManager()

# Initialize graph (automatically uses optimizations)
success = sim_manager.initialize_graph()
if success:
    print("Graph initialized with optimizations")

# Start simulation
sim_manager.start_simulation(run_in_thread=True)
print("Simulation running with performance optimizations")
```

## 3. Test Performance

```python
from utils.performance_benchmark import run_comprehensive_benchmark

# Run performance tests
benchmark = run_comprehensive_benchmark()
print("Performance test completed")
print(benchmark.generate_report())
```

## 4. Monitor Performance

```python
from utils.performance_monitor import get_performance_monitor

# Get performance metrics
monitor = get_performance_monitor()
metrics = monitor.get_current_metrics()

print(f"FPS: {metrics.fps:.2f}")
print(f"Memory Usage: {metrics.memory_usage_mb:.1f} MB")
print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
```

## 5. Advanced Configuration

```python
# Custom optimization configuration
config = OptimizationConfig(
    use_lazy_loading=True,        # Faster startup
    use_caching=True,            # Better performance
    use_batch_processing=True,   # Handle large datasets
    batch_size=2000,             # Larger batches for high node counts
    cache_size=10000,            # More cache for bigger simulations
    max_nodes=500000,            # Support very large networks
    enable_performance_monitoring=True
)

applier = apply_optimizations(config)
```

## 🎯 Key Features

### Energy Movement & Dynamics
```python
# Energy automatically flows through the network
graph = sim_manager.graph
# Energy values are in graph.x[:, 0]
# Membrane potentials are tracked automatically
```

### Connection Logic & Formation
```python
# Intelligent connections form automatically
from neural.connection_logic import intelligent_connection_formation
graph = intelligent_connection_formation(graph)
```

### Hebbian Learning
```python
# Learning happens continuously during simulation
# "Neurons that fire together wire together"
# Weights adjust based on neural activity patterns
```

### Spike Systems
```python
# Neural spikes propagate through the network
from neural.spike_queue_system import create_spike_queue_system
spike_system = create_spike_queue_system(sim_manager)
```

### Extreme Node Counts
```python
# Handles thousands to millions of nodes efficiently
from neural.optimized_node_manager import get_optimized_node_manager
node_manager = get_optimized_node_manager()

# Create 10,000 nodes efficiently
nodes = node_manager.create_node_batch([
    {'type': 'dynamic', 'energy': 1.0, 'x': i%100, 'y': i//100}
    for i in range(10000)
])
```

### Sensory Input Processing
```python
# Process live PC data as neural input
from sensory.visual_energy_bridge import create_visual_energy_bridge
visual_bridge = create_visual_energy_bridge(None)

# Process screen capture or other sensory data
# Automatically converts visual data to neural activation patterns
```

## 📊 Performance Monitoring

### Real-time Metrics
```python
# Monitor simulation performance
stats = sim_manager.get_performance_stats()
print(f"Steps: {stats['total_steps']}")
print(f"Avg Step Time: {stats['avg_step_time']:.4f}s")
print(f"System Health: {stats['system_health']}")
```

### Cache Performance
```python
from utils.performance_cache import get_performance_cache_manager
cache_manager = get_performance_cache_manager()
cache_stats = cache_manager.get_performance_stats()

print(f"Cache Hit Rate: {cache_stats['cache_hit_rate']:.2f}")
print(f"Cache Size: {cache_stats['lru_cache_size']}")
```

## 🔧 Troubleshooting

### Common Issues

**1. Slow Startup**
```python
# Enable lazy loading
config = OptimizationConfig(use_lazy_loading=True)
```

**2. High Memory Usage**
```python
# Enable memory optimizations
config = OptimizationConfig(use_caching=True)
```

**3. Poor Performance with Large Networks**
```python
# Increase batch size and cache
config = OptimizationConfig(
    batch_size=5000,
    cache_size=50000,
    use_batch_processing=True
)
```

**4. Cache Misses**
```python
# Adjust cache TTL and size
config = OptimizationConfig(
    cache_size=20000,
    cache_ttl=1200  # 20 minutes
)
```

## 📈 Scaling Up

### For Large Networks (100k+ nodes)
```python
config = OptimizationConfig(
    max_nodes=1000000,      # 1M nodes
    batch_size=10000,       # Large batches
    cache_size=100000,      # Big cache
    spatial_index_grid_size=200  # Larger spatial grid
)
```

### For High-Performance Computing
```python
# Enable all optimizations
config = OptimizationConfig(
    use_lazy_loading=True,
    use_caching=True,
    use_batch_processing=True,
    enable_performance_monitoring=True,
    batch_size=10000,
    cache_size=50000,
    max_nodes=1000000
)
```

## 🎯 Best Practices

1. **Always apply optimizations** before starting simulation
2. **Monitor performance metrics** during long runs
3. **Adjust batch sizes** based on your hardware
4. **Use lazy loading** for better startup performance
5. **Enable caching** for repeated operations
6. **Monitor memory usage** to prevent leaks

## 📋 Example Full Setup

```python
#!/usr/bin/env python3
"""
Complete optimized neural simulation setup
"""

from utils.optimization_applier import apply_optimizations, OptimizationConfig
from simulation_manager import SimulationManager
from utils.performance_monitor import get_performance_monitor
import time

def main():
    print("Setting up optimized neural simulation...")

    # 1. Apply optimizations
    config = OptimizationConfig(
        use_lazy_loading=True,
        use_caching=True,
        use_batch_processing=True,
        batch_size=2000,
        cache_size=10000,
        max_nodes=100000,
        enable_performance_monitoring=True
    )

    applier = apply_optimizations(config)
    print("✓ Optimizations applied")

    # 2. Create simulation
    sim_manager = SimulationManager()
    print("✓ Simulation manager created")

    # 3. Initialize graph
    if sim_manager.initialize_graph():
        print("✓ Graph initialized")
    else:
        print("✗ Graph initialization failed")
        return

    # 4. Start simulation
    sim_manager.start_simulation(run_in_thread=True)
    print("✓ Simulation started")

    # 5. Monitor performance
    monitor = get_performance_monitor()
    for i in range(10):
        time.sleep(1)
        metrics = monitor.get_current_metrics()
        print(".2f"
              ".1f"
              ".1f")

    print("Simulation running successfully with optimizations!")

if __name__ == "__main__":
    main()
```

## 🎉 You're Ready!

Your neural simulation system is now optimized and ready to handle:
- ⚡ Fast startup times
- 🧠 Large neural networks (10k+ nodes)
- 🚀 High-performance processing
- 📊 Real-time monitoring
- 🔄 Automatic optimization

**Happy simulating!** 🧠✨