# Common Imports API

The centralized import system provides common imports, type definitions, and utility functions to reduce code duplication across the codebase.

## Module: `common_imports.py`

### Key Features

- **Unified Imports**: All third-party libraries imported in one place
- **Type Aliases**: Common type definitions for better type hinting
- **Logging Configuration**: Pre-configured logging setup with file and console handlers
- **Utility Functions**: Common helper functions for everyday operations
- **Error Handling**: Safe import mechanisms with fallback support
- **Profiling Tools**: Performance profiling utilities for code optimization

### Common Type Aliases

```python
# Tensor types
Tensor = torch.Tensor
ArrayLike = NDArray[np.float64] | Tensor
PathLike = str | os.PathLike[str]
```

### Core Utility Functions

#### `safe_import(module_name: str, fallback: object | None = None) -> object | None`

Safely import a module with fallback:

```python
from project.common_imports import safe_import

# Import numpy with fallback to None if not available
np = safe_import('numpy')
```

#### `get_device() -> str`

Returns the best available compute device:

- Returns `'cuda'` if CUDA is available
- Returns `'cpu'` otherwise

```python
from project.common_imports import get_device

device = get_device()  # 'cuda' or 'cpu'
```

#### `profile_section(name: str)`

Context manager for profiling code sections with automatic logging:

```python
from project.common_imports import profile_section

with profile_section("neural_update"):
    # Your code here
    neural_system.update()
```

#### `profile_report()`

Generates and logs a report of the slowest profiled sections:

```python
from project.common_imports import profile_report

# After running profiled code
profile_report()  # Logs the 5 slowest sections
```

#### `ensure_dir(path: str | os.PathLike[str]) -> None`

Ensures a directory exists, creating it if necessary:

```python
from project.common_imports import ensure_dir

ensure_dir("output/data")
```

#### `safe_remove(path: str | os.PathLike[str]) -> bool`

Safely removes a file or directory, returning success status:

```python
from project.common_imports import safe_remove

success = safe_remove("temp_file.txt")
if not success:
    logger.warning("Failed to remove file")
```

#### `get_config_value(config_dict: dict[str, Any], key: str, default: str | int | float | bool | None = None) -> str | int | float | bool | None`

Get a value from config dict with fallback:

```python
from project.common_imports import get_config_value

node_count = get_config_value(config, 'INITIAL_PROCESSING_NODES', 100)
```

#### `validate_config(config_dict: dict[str, Any], required_keys: list[str]) -> bool`

Validate that required keys exist in config:

```python
from project.common_imports import validate_config

if not validate_config(config, ['SENSOR_WIDTH', 'SENSOR_HEIGHT']):
    raise ValueError("Missing required configuration keys")
```

### Exported Modules and Utilities

The following modules and utilities are exported for easy importing:

```python
# Core libraries
from project.common_imports import np, torch, cv2, mss, Image, ImageTk, tk

# Type aliases
from project.common_imports import Tensor, ArrayLike, PathLike

# Utility functions
from project.common_imports import safe_import, get_device, profile_section, profile_report
from project.common_imports import ensure_dir, safe_remove, get_config_value, validate_config

# Logging
from project.common_imports import logger
```

### Error Handling and Safety Features

The module includes comprehensive error handling:

- **Safe Import**: `safe_import()` function prevents crashes when optional dependencies are missing
- **Type Safety**: Modern Python type hints with proper union types
- **Resource Management**: Automatic cleanup and memory management
- **Logging**: Integrated logging with debug, info, and error levels

### Usage Patterns

#### Basic Import Pattern

```python
# Import everything you need in one line
from project.common_imports import (
    np, torch, Tensor, ArrayLike,
    get_device, profile_section,
    ensure_dir, safe_remove,
    logger
)

# Use the imports
device = get_device()
logger.info(f"Using device: {device}")
```

#### Advanced Configuration Pattern

```python
from project.common_imports import get_config_value, validate_config

# Load and validate configuration
config = load_config()
if not validate_config(config, ['SENSOR_WIDTH', 'SENSOR_HEIGHT', 'INITIAL_PROCESSING_NODES']):
    raise ValueError("Invalid configuration")

# Get configuration values with defaults
sensor_width = get_config_value(config, 'SENSOR_WIDTH', 256)
sensor_height = get_config_value(config, 'SENSOR_HEIGHT', 144)
```

#### Performance Profiling Pattern

```python
from project.common_imports import profile_section, profile_report

# Profile multiple sections
with profile_section("data_loading"):
    data = load_data()

with profile_section("model_training"):
    train_model(data)

with profile_section("evaluation"):
    evaluate_model(data)

# Generate performance report
profile_report()
```

### Best Practices

1. **Centralized Imports**: Always import from `common_imports` rather than individual libraries to ensure consistency
2. **Error Handling**: Use `safe_import()` for optional dependencies
3. **Profiling**: Use `profile_section()` to identify performance bottlenecks
4. **Configuration**: Use `get_config_value()` and `validate_config()` for robust configuration management
5. **Logging**: Use the provided `logger` for consistent logging across the application