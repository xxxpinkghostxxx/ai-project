# Neural Simulation SOA Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will help you quickly set up and use the optimized neural simulation system with Service-Oriented Architecture (SOA).

## Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Dear PyGUI (for UI)
- Basic understanding of neural networks

## 1. Apply Optimizations

```python
# The system now includes built-in optimizations through SOA services
from core.services.service_registry import ServiceRegistry
from core.services.performance_monitoring_service import PerformanceMonitoringService

# Create service registry
registry = ServiceRegistry()

# Register performance monitoring service with optimizations
registry.register_instance(IPerformanceMonitor, PerformanceMonitoringService())

print("SOA services initialized with built-in optimizations!")
```

## 2. Run Optimized Simulation with SOA

```python
from core.services.service_registry import ServiceRegistry
from core.services.simulation_coordinator import SimulationCoordinator
from core.services.graph_management_service import GraphManagementService
from core.interfaces import ISimulationCoordinator, IGraphManager
from main_graph import create_test_graph

# Create service registry and register services
registry = ServiceRegistry()
registry.register_instance(ISimulationCoordinator, SimulationCoordinator(registry))
registry.register_instance(IGraphManager, GraphManagementService())

# Get coordinator and initialize
coordinator = registry.resolve(ISimulationCoordinator)
graph_manager = registry.resolve(IGraphManager)

# Create and set test graph
graph = create_test_graph(num_sensory=100, num_dynamic=50)
graph_manager.set_graph(graph)

# Initialize and run simulation
success = coordinator.initialize_simulation()
if success:
    coordinator.start_simulation()
    print("SOA-based simulation running with performance optimizations")
```

## 3. Test Performance with SOA Services

```python
from core.services.service_registry import ServiceRegistry
from core.services.performance_monitoring_service import PerformanceMonitoringService
from core.interfaces import IPerformanceMonitor

# Create registry and register performance service
registry = ServiceRegistry()
registry.register_instance(IPerformanceMonitor, PerformanceMonitoringService())

# Get performance monitor
monitor = registry.resolve(IPerformanceMonitor)
metrics = monitor.get_current_metrics()

print("SOA Performance test completed")
print(f"FPS: {metrics.fps:.2f}")
print(f"Memory Usage: {metrics.memory_usage_mb:.1f} MB")
```

## 4. Monitor Performance with SOA

```python
from core.services.service_registry import ServiceRegistry
from core.services.performance_monitoring_service import PerformanceMonitoringService
from core.interfaces import IPerformanceMonitor

# Create registry and register performance service
registry = ServiceRegistry()
registry.register_instance(IPerformanceMonitor, PerformanceMonitoringService())

# Get performance metrics
monitor = registry.resolve(IPerformanceMonitor)
metrics = monitor.get_current_metrics()

print(f"FPS: {metrics.fps:.2f}")
print(f"Memory Usage: {metrics.memory_usage_mb:.1f} MB")
print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
```

## 5. Advanced SOA Configuration

```python
# Custom optimization configuration through SOA services
from core.services.service_registry import ServiceRegistry
from core.services.configuration_service import ConfigurationService
from core.services.performance_monitoring_service import PerformanceMonitoringService
from core.interfaces import IConfigurationService, IPerformanceMonitor

registry = ServiceRegistry()
registry.register_instance(IConfigurationService, ConfigurationService())
registry.register_instance(IPerformanceMonitor, PerformanceMonitoringService())

config_service = registry.resolve(IConfigurationService)

# Configure advanced settings
config_service.set_config('performance.lazy_loading', True)
config_service.set_config('performance.caching', True)
config_service.set_config('performance.batch_processing', True)
config_service.set_config('system.max_nodes', 100000)
config_service.set_config('performance.monitoring_enabled', True)
```

## 🎯 Key SOA Features

### Energy Movement & Dynamics
```python
# Energy automatically flows through the network via EnergyManagementService
from core.services.service_registry import ServiceRegistry
from core.services.energy_management_service import EnergyManagementService
from core.interfaces import IEnergyManager

registry = ServiceRegistry()
registry.register_instance(IEnergyManager, EnergyManagementService())

energy_service = registry.resolve(IEnergyManager)
# Energy values are managed automatically through the service
```

### Connection Logic & Formation
```python
# Intelligent connections form automatically via GraphManagementService
from core.services.service_registry import ServiceRegistry
from core.services.graph_management_service import GraphManagementService
from core.interfaces import IGraphManager
from neural.connection_logic import intelligent_connection_formation

registry = ServiceRegistry()
registry.register_instance(IGraphManager, GraphManagementService())

graph_manager = registry.resolve(IGraphManager)
graph = graph_manager.get_graph()
graph = intelligent_connection_formation(graph)
```

### Hebbian Learning with SOA
```python
# Learning happens continuously during simulation via LearningService
from core.services.service_registry import ServiceRegistry
from core.services.learning_service import LearningService
from core.interfaces import ILearningEngine

registry = ServiceRegistry()
registry.register_instance(ILearningEngine, LearningService(registry))

learning_service = registry.resolve(ILearningEngine)
# "Neurons that fire together wire together" - handled by the service
# Weights adjust based on neural activity patterns automatically
```

### Spike Systems with SOA
```python
# Neural spikes propagate through the network via NeuralProcessingService
from core.services.service_registry import ServiceRegistry
from core.services.neural_processing_service import NeuralProcessingService
from core.interfaces import INeuralProcessor

registry = ServiceRegistry()
registry.register_instance(INeuralProcessor, NeuralProcessingService())

neural_processor = registry.resolve(INeuralProcessor)
# Spike processing handled automatically by the service
```

### Extreme Node Counts with SOA
```python
# Handles thousands to millions of nodes efficiently via SOA services
from core.services.service_registry import ServiceRegistry
from core.services.configuration_service import ConfigurationService
from core.interfaces import IConfigurationService

registry = ServiceRegistry()
registry.register_instance(IConfigurationService, ConfigurationService())

config_service = registry.resolve(IConfigurationService)
config_service.set_config('system.max_nodes', 1000000)  # Support very large networks
```

### Sensory Input Processing with SOA
```python
# Process live PC data as neural input via SensoryProcessingService
from core.services.service_registry import ServiceRegistry
from core.services.sensory_processing_service import SensoryProcessingService
from core.interfaces import ISensoryProcessor

registry = ServiceRegistry()
registry.register_instance(ISensoryProcessor, SensoryProcessingService())

sensory_processor = registry.resolve(ISensoryProcessor)
# Automatically converts visual data to neural activation patterns
```

## 📊 SOA Performance Monitoring

### Real-time Metrics via Services
```python
# Monitor simulation performance through PerformanceMonitoringService
from core.services.service_registry import ServiceRegistry
from core.services.performance_monitoring_service import PerformanceMonitoringService
from core.interfaces import IPerformanceMonitor

registry = ServiceRegistry()
registry.register_instance(IPerformanceMonitor, PerformanceMonitoringService())

monitor = registry.resolve(IPerformanceMonitor)
stats = monitor.get_performance_stats()
print(f"Steps: {stats['total_steps']}")
print(f"Avg Step Time: {stats['avg_step_time']:.4f}s")
print(f"System Health: {stats['system_health']}")
```

### Cache Performance via SOA
```python
# Monitor cache performance through dedicated services
from core.services.service_registry import ServiceRegistry
from core.services.performance_monitoring_service import PerformanceMonitoringService
from core.interfaces import IPerformanceMonitor

registry = ServiceRegistry()
registry.register_instance(IPerformanceMonitor, PerformanceMonitoringService())

monitor = registry.resolve(IPerformanceMonitor)
cache_stats = monitor.get_cache_performance_stats()

print(f"Cache Hit Rate: {cache_stats['cache_hit_rate']:.2f}")
print(f"Cache Size: {cache_stats['lru_cache_size']}")
```

## 🔧 SOA Troubleshooting

### Common Issues

**1. Service Registration Issues**
```python
# Ensure all required services are registered
from core.services.service_registry import ServiceRegistry
from core.services.simulation_coordinator import SimulationCoordinator
from core.interfaces import ISimulationCoordinator

registry = ServiceRegistry()
registry.register_instance(ISimulationCoordinator, SimulationCoordinator(registry))

# Verify service registration
if registry.has_service(ISimulationCoordinator):
    print("Service registered successfully")
```

**2. High Memory Usage**
```python
# Enable memory optimizations through ConfigurationService
from core.services.service_registry import ServiceRegistry
from core.services.configuration_service import ConfigurationService
from core.interfaces import IConfigurationService

registry = ServiceRegistry()
registry.register_instance(IConfigurationService, ConfigurationService())

config_service = registry.resolve(IConfigurationService)
config_service.set_config('performance.caching', True)
```

**3. Poor Performance with Large Networks**
```python
# Increase batch size and cache through ConfigurationService
from core.services.service_registry import ServiceRegistry
from core.services.configuration_service import ConfigurationService
from core.interfaces import IConfigurationService

registry = ServiceRegistry()
registry.register_instance(IConfigurationService, ConfigurationService())

config_service = registry.resolve(IConfigurationService)
config_service.set_config('performance.batch_size', 5000)
config_service.set_config('performance.cache_size', 50000)
config_service.set_config('performance.batch_processing', True)
```

**4. Cache Misses**
```python
# Adjust cache settings through ConfigurationService
from core.services.service_registry import ServiceRegistry
from core.services.configuration_service import ConfigurationService
from core.interfaces import IConfigurationService

registry = ServiceRegistry()
registry.register_instance(IConfigurationService, ConfigurationService())

config_service = registry.resolve(IConfigurationService)
config_service.set_config('performance.cache_size', 20000)
config_service.set_config('performance.cache_ttl', 1200)  # 20 minutes
```

## 📈 Scaling Up with SOA

### For Large Networks (100k+ nodes)
```python
from core.services.service_registry import ServiceRegistry
from core.services.configuration_service import ConfigurationService
from core.interfaces import IConfigurationService

registry = ServiceRegistry()
registry.register_instance(IConfigurationService, ConfigurationService())

config_service = registry.resolve(IConfigurationService)
config_service.set_config('system.max_nodes', 1000000)      # 1M nodes
config_service.set_config('performance.batch_size', 10000)  # Large batches
config_service.set_config('performance.cache_size', 100000) # Big cache
```

### For High-Performance Computing with SOA
```python
# Enable all optimizations through SOA services
from core.services.service_registry import ServiceRegistry
from core.services.configuration_service import ConfigurationService
from core.interfaces import IConfigurationService

registry = ServiceRegistry()
registry.register_instance(IConfigurationService, ConfigurationService())

config_service = registry.resolve(IConfigurationService)
config_service.set_config('performance.lazy_loading', True)
config_service.set_config('performance.caching', True)
config_service.set_config('performance.batch_processing', True)
config_service.set_config('performance.monitoring_enabled', True)
config_service.set_config('performance.batch_size', 10000)
config_service.set_config('performance.cache_size', 50000)
config_service.set_config('system.max_nodes', 1000000)
```

## 🎯 SOA Best Practices

1. **Always register services with the ServiceRegistry** before use
2. **Use interfaces for service dependencies** to maintain loose coupling
3. **Monitor performance metrics** during long runs via PerformanceMonitoringService
4. **Configure settings through ConfigurationService** for consistency
5. **Handle service resolution errors** gracefully
6. **Monitor memory usage** through PerformanceMonitoringService to prevent leaks

## 📋 Complete SOA Setup Example

```python
#!/usr/bin/env python3
"""
Complete optimized SOA neural simulation setup
"""

from core.services.service_registry import ServiceRegistry
from core.services.simulation_coordinator import SimulationCoordinator
from core.services.configuration_service import ConfigurationService
from core.services.performance_monitoring_service import PerformanceMonitoringService
from core.interfaces import *
from main_graph import create_test_graph
import time

def main():
    print("Setting up optimized SOA neural simulation...")

    # 1. Create service registry (dependency injection container)
    registry = ServiceRegistry()

    # 2. Register all SOA services
    registry.register_instance(IConfigurationService, ConfigurationService())
    registry.register_instance(IPerformanceMonitor, PerformanceMonitoringService())
    registry.register_instance(ISimulationCoordinator, SimulationCoordinator(registry))

    # 3. Configure performance optimizations
    config_service = registry.resolve(IConfigurationService)
    config_service.set_config('performance.lazy_loading', True)
    config_service.set_config('performance.caching', True)
    config_service.set_config('performance.batch_processing', True)
    config_service.set_config('performance.batch_size', 2000)
    config_service.set_config('performance.cache_size', 10000)
    config_service.set_config('system.max_nodes', 100000)
    config_service.set_config('performance.monitoring_enabled', True)

    print("✓ SOA services registered and configured")

    # 4. Get coordinator and initialize
    coordinator = registry.resolve(ISimulationCoordinator)

    # 5. Create and set neural graph
    graph = create_test_graph(num_sensory=100, num_dynamic=50)
    graph_manager = registry.resolve(IGraphManager)  # Assuming registered elsewhere
    graph_manager.set_graph(graph)

    # 6. Initialize simulation
    if coordinator.initialize_simulation():
        print("✓ SOA simulation initialized")
    else:
        print("✗ SOA simulation initialization failed")
        return

    # 7. Start simulation
    coordinator.start_simulation()
    print("✓ SOA simulation started")

    # 8. Monitor performance
    monitor = registry.resolve(IPerformanceMonitor)
    for i in range(10):
        time.sleep(1)
        metrics = monitor.get_current_metrics()
        print(f"Step {i+1}: FPS: {metrics.fps:.2f}, Memory: {metrics.memory_usage_mb:.1f} MB, CPU: {metrics.cpu_percent:.1f}%")

    print("SOA simulation running successfully with optimizations!")

    coordinator.stop_simulation()

if __name__ == "__main__":
    main()
```

## 🎉 You're Ready with SOA!

Your neural simulation system is now optimized and ready to handle:
- ⚡ Fast startup times through service initialization
- 🧠 Large neural networks (10k+ nodes) via SOA architecture
- 🚀 High-performance processing through specialized services
- 📊 Real-time monitoring via PerformanceMonitoringService
- 🔄 Automatic optimization through ConfigurationService

**Happy simulating with SOA! 🧠✨**