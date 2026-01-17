# Utility Functions API

Comprehensive utility functions and helper modules for the neural system project.

## Core Utility Modules

### `utils.py` - General Utility Functions

General utility functions for common operations:

#### `clamp(value: int | float, min_value: int | float, max_value: int | float) -> int | float`

Clamp a value between minimum and maximum bounds:

```python
from project.utils import clamp

# Ensure value is within valid range
energy = clamp(energy_value, 0.0, NODE_ENERGY_CAP)
```

#### `lerp(a: int | float, b: int | float, t: int | float) -> int | float`

Linear interpolation between values:

```python
from project.utils import lerp

# Smooth transition between values
position = lerp(start_pos, end_pos, 0.5)
```

#### `random_range(rng: tuple[float, float]) -> float`

Generate random value in range:

```python
from project.utils import random_range

# Random energy variation
energy_variation = random_range((-0.1, 0.1))
```

#### `get_free_ram() -> int | None`

Get available system memory:

```python
from project.utils import get_free_ram

ram_available = get_free_ram()
if ram_available and ram_available < MIN_RAM:
    enable_memory_optimization()
```

#### `has_cupy() -> bool`

Check for CuPy (GPU) support:

```python
from project.utils import has_cupy

if has_cupy():
    enable_gpu_acceleration()
```

#### `get_free_gpu_memory() -> int | None`

Get available GPU memory:

```python
from project.utils import get_free_gpu_memory

gpu_mem = get_free_gpu_memory()
if gpu_mem and gpu_mem < MIN_GPU_MEM:
    reduce_batch_size()
```

#### `get_array_module()`

Get optimal array module (CuPy or NumPy):

```python
from project.utils import get_array_module

xp = get_array_module()
array = xp.array([1, 2, 3, 4, 5])
```

#### `check_cuda_status() -> None`

Diagnose CUDA/GPU status:

```python
from project.utils import check_cuda_status

check_cuda_status()  # Logs CUDA availability and configuration
```

#### `flatten_nodes(nodes: Any) -> list[Any]`

Flatten nested node structures:

```python
from project.utils import flatten_nodes

flat_list = flatten_nodes(nested_nodes)
```

#### `extract_node_attrs(node: Any) -> dict[str, Any]`

Extract node attributes as dictionary:

```python
from project.utils import extract_node_attrs

node_data = extract_node_attrs(node)
```

### Configuration Management Utilities

#### `utils/config_manager.py` - Configuration Management

```python
class ConfigManager:
    def __init__(self, config_file: str = "config.json")
```

**Key Methods:**

- `get_config(section: str, key: str | None = None)`: Get configuration values
- `update_config(section: str, key: str, value: Any) -> bool`: Update configuration
- `save_config() -> bool`: Save configuration to file
- `load_config() -> bool`: Load configuration from file

```python
from project.utils.config_manager import ConfigManager

config = ConfigManager()
update_interval = config.get_config('system', 'update_interval')
```

### Error Handling Utilities

#### `utils/error_handler.py` - Comprehensive Error Handling

```python
class ErrorHandler:
    @staticmethod
    def show_error(title: str, message: str) -> None
    @staticmethod
    def log_error(message: str) -> None
    @staticmethod
    def log_warning(message: str) -> None
    @staticmethod
    def log_info(message: str) -> None
```

**Usage:**

```python
from project.utils.error_handler import ErrorHandler

try:
    risky_operation()
except Exception as e:
    ErrorHandler.show_error("Operation Failed", str(e))
    ErrorHandler.log_error(f"Details: {str(e)}")
```

### Performance Utilities

#### `utils/performance_utils.py` - Performance Monitoring

Performance monitoring and optimization utilities:

- **Timing Functions**: High-resolution timing for performance measurement
- **Profiling Tools**: Code profiling and bottleneck identification
- **Memory Monitoring**: Memory usage tracking and optimization
- **Resource Tracking**: System resource utilization monitoring

```python
from project.utils.performance_utils import time_function

@time_function
def process_data(data):
    # Function execution will be timed automatically
    return data.processed()
```

### Profiling Utilities

#### `utils/profile_section.py` - Code Profiling

Context-based profiling utilities:

```python
from project.utils.profile_section import profile_section

with profile_section("data_processing"):
    # Code execution will be profiled
    result = process_data(input_data)
```

### Security Utilities

#### `utils/security_utils.py` - Security Functions

Security-related utilities:

- **Input Validation**: Safe input handling and validation
- **Resource Protection**: Secure resource access patterns
- **Error Sanitization**: Safe error message handling
- **Configuration Security**: Secure configuration management

```python
from project.utils.security_utils import validate_input

if not validate_input(user_input):
    raise ValueError("Invalid input")
```

### Shutdown Utilities

#### `utils/shutdown_utils.py` - Clean Shutdown Management

```python
class ShutdownDetector:
    @staticmethod
    def safe_cleanup(cleanup_func: Callable, context: str) -> None
```

**Usage:**

```python
from project.utils.shutdown_utils import ShutdownDetector

def cleanup_resources():
    # Resource cleanup logic
    pass

# Safe cleanup on shutdown
ShutdownDetector.safe_cleanup(cleanup_resources, "System shutdown")
```

### Tensor Management Utilities

#### `utils/tensor_manager.py` - Advanced Tensor Operations

```python
class TensorManager:
    def __init__(self, neural_system: PyGNeuralSystem)
```

**Key Features:**

- **Tensor Validation**: Comprehensive tensor shape and content validation
- **Memory Optimization**: Advanced tensor memory management
- **Synchronization**: Tensor synchronization across system components
- **Error Recovery**: Automatic recovery from tensor-related errors

```python
from project.utils.tensor_manager import TensorManager

tensor_manager = TensorManager(neural_system)
validation_results = tensor_manager.validate_tensor_shapes()
```

## Usage Patterns

### Configuration Management Pattern

```python
from project.utils.config_manager import ConfigManager

# Initialize configuration
config = ConfigManager("system_config.json")

# Get configuration values
sensor_width = config.get_config('sensory', 'width')
update_interval = config.get_config('system', 'update_interval')

# Update configuration
config.update_config('system', 'debug_mode', True)

# Save changes
config.save_config()
```

### Error Handling Pattern

```python
from project.utils.error_handler import ErrorHandler

def process_data_safely(data):
    try:
        result = process_data(data)
        ErrorHandler.log_info("Data processing completed")
        return result
    except ValueError as e:
        ErrorHandler.show_error("Invalid Data", str(e))
        ErrorHandler.log_error(f"Data validation failed: {str(e)}")
        return None
    except Exception as e:
        ErrorHandler.show_error("Processing Error", "Unexpected error occurred")
        ErrorHandler.log_error(f"Unexpected error: {str(e)}")
        raise
```

### Performance Monitoring Pattern

```python
from project.utils.performance_utils import PerformanceMonitor

# Create performance monitor
monitor = PerformanceMonitor()

# Monitor function execution
@monitor.time_function
def critical_operation():
    # This function's execution time will be monitored
    pass

# Get performance statistics
stats = monitor.get_statistics()
```

## Best Practices

### Utility Usage

1. **Consistency**: Use utilities consistently across the codebase
2. **Error Handling**: Always implement proper error handling
3. **Performance**: Use profiling utilities to identify bottlenecks
4. **Resource Management**: Ensure proper resource cleanup
5. **Security**: Validate all inputs and handle errors securely

### Configuration Management

1. **Validation**: Validate configuration parameters before use
2. **Documentation**: Document non-standard configuration values
3. **Version Control**: Track configuration changes over time
4. **Environment-Specific**: Use different configurations for different environments
5. **Fallbacks**: Implement sensible defaults and fallback mechanisms

### Performance Optimization

1. **Profiling**: Identify performance bottlenecks before optimizing
2. **Batch Processing**: Use batch operations for better throughput
3. **Memory Management**: Monitor and optimize memory usage
4. **GPU Acceleration**: Use GPU acceleration when available
5. **Caching**: Implement caching for expensive operations

## Troubleshooting

Common utility-related issues and solutions:

- **Configuration Errors**: Validate configuration files and parameters
- **Memory Leaks**: Use resource monitoring to identify leaks
- **Performance Issues**: Profile code to identify bottlenecks
- **Error Handling**: Ensure comprehensive error handling is implemented
- **Compatibility Issues**: Check for required dependencies and versions