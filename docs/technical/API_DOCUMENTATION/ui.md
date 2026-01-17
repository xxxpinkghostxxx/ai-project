# UI Components API

Comprehensive user interface system for the neural network with both Tkinter and PyQt6 implementations.

## Module Architecture

The UI system consists of multiple components:

### Core UI Modules

#### `main_window.py` - Traditional Tkinter Interface

```python
class MainWindow:
    def __init__(self, config_manager: ConfigManager, state_manager: StateManager)
```

**Key Features:**

- **Workspace Visualization**: Real-time display of neural network state
- **Sensory Input Display**: Visualization of current sensory input
- **Metrics Panel**: Comprehensive system metrics and statistics
- **Control Panel**: System controls and configuration
- **Resource Management**: Integrated resource monitoring and cleanup

```python
from project.ui.main_window import MainWindow
from project.utils.config_manager import ConfigManager
from project.system.state_manager import StateManager

# Initialize UI components
config_manager = ConfigManager()
state_manager = StateManager()
main_window = MainWindow(config_manager, state_manager)

# Start the system
main_window.run()
```

#### `modern_main_window.py` - Modern PyQt6 Interface

```python
class ModernMainWindow(QMainWindow):
    def __init__(self, config_manager: ConfigManager, state_manager: StateManager)
```

**Advanced Features:**

- **Modern UI Design**: Dark theme with responsive layout
- **Enhanced Visualization**: High-performance graphics with QGraphicsView
- **Advanced Controls**: Styled buttons and interactive elements
- **Configuration Panel**: Integrated configuration management
- **Real-time Updates**: Smooth animations and transitions

```python
from project.ui.modern_main_window import ModernMainWindow

# Create modern interface
modern_window = ModernMainWindow(config_manager, state_manager)
modern_window.run()
```

### Configuration Components

#### `config_panel.py` - Configuration Management

```python
class ConfigPanel:
    def __init__(self, parent: tk.Tk, config_manager: ConfigManager)
```

**Features:**

- **Tabbed Interface**: Organized configuration categories
- **Real-time Validation**: Immediate feedback on parameter changes
- **Profile Management**: Save and load configuration profiles
- **Restart System**: Apply changes with system restart

```python
from project.ui.config_panel import ConfigPanel

# Create and show configuration panel
config_panel = ConfigPanel(main_window.window, config_manager)
config_panel.show()
```

### Resource Management

#### `resource_manager.py` - Tkinter Resource Management

```python
class UIResourceManager:
    def __init__(self, max_images: int = 100, max_windows: int = 10,
                 max_memory_mb: int = 512, enable_monitoring: bool = True)
```

**Key Features:**

- **Image Management**: Prevent garbage collection of UI images
- **Window Tracking**: Monitor and manage UI windows
- **Memory Monitoring**: Track resource usage and prevent leaks
- **Automatic Cleanup**: Background resource optimization

```python
from project.ui.resource_manager import UIResourceManager

# Create resource manager
resource_manager = UIResourceManager(max_images=200, max_memory_mb=1024)

# Register resources
tk_image = resource_manager.create_tk_image(pil_image)
resource_manager.register_window(window)
```

#### `modern_resource_manager.py` - PyQt6 Resource Management

```python
class ModernResourceManager(QObject):
    def __init__(self, max_images: int = 100, max_windows: int = 10,
                 max_memory_mb: int = 512, enable_monitoring: bool = True)
```

**Advanced Features:**

- **Signal-Based Architecture**: Qt signal/slot integration
- **QPixmap Management**: Optimized image handling for Qt
- **Thread-Safe Operations**: Safe resource access from multiple threads
- **Performance Monitoring**: Real-time resource statistics

```python
from project.ui.modern_resource_manager import ModernResourceManager

# Create modern resource manager
modern_resources = ModernResourceManager()

# Connect to resource stats signals
modern_resources.resource_stats_updated.connect(handle_resource_stats)
```

## UI Component Details

### Main Window Components

#### Workspace Visualization

- **Real-time Rendering**: Continuous updates of neural network state
- **Color Mapping**: Energy-based color visualization
- **Zoom and Pan**: Interactive exploration of network structure
- **Performance Optimization**: Efficient rendering for large networks

#### Sensory Input Display

- **Input Preview**: Real-time display of sensory data
- **Processing Pipeline**: Visualization of preprocessing steps
- **Input Statistics**: Metrics on input quality and characteristics
- **Error Detection**: Identification of input processing issues

#### Metrics Panel

- **System Metrics**: Node counts, energy levels, connection statistics
- **Performance Metrics**: FPS, update times, memory usage
- **Historical Data**: Trends and patterns over time
- **Alert System**: Visual indicators for system health

#### Control Panel

- **System Controls**: Start, stop, suspend, resume
- **Parameter Adjustment**: Real-time configuration changes
- **Preset Management**: Save and load system configurations
- **Emergency Controls**: System reset and recovery options

### Advanced UI Features

#### Configuration Management

```python
# Access configuration through UI
current_interval = main_window.interval_slider.get()

# Update configuration
main_window._update_interval_changed("100")

# Open advanced configuration
main_window._open_config_panel()
```

#### System Integration

```python
# Start system components
main_window.start_system(neural_system, screen_capture)

# Handle state changes
def on_state_change(state):
    main_window.on_state_change(state)

state_manager.add_observer(on_state_change)
```

#### Resource Monitoring

```python
# Get resource health report
report = main_window.get_resource_health_report()

# Monitor resource usage
stats = resource_manager.get_resource_statistics()

# Handle memory warnings
if stats['memory_usage'] > stats['memory_limit']:
    perform_cleanup()
```

## Usage Patterns

### Basic UI Setup

```python
from project.ui.main_window import MainWindow
from project.pyg_neural_system import PyGNeuralSystem
from project.vision import ThreadedScreenCapture

# Initialize components
config_manager = ConfigManager()
state_manager = StateManager()
neural_system = PyGNeuralSystem(256, 144, 30)
screen_capture = ThreadedScreenCapture(256, 144)

# Create and start UI
main_window = MainWindow(config_manager, state_manager)
main_window.start_system(neural_system, screen_capture)
main_window.run()
```

### Modern UI with Advanced Features

```python
from project.ui.modern_main_window import ModernMainWindow

# Create modern interface
modern_window = ModernMainWindow(config_manager, state_manager)

# Configure advanced features
modern_window.setWindowTitle("Advanced Neural System")
modern_window.resize(1400, 900)

# Start system
modern_window.start_system(neural_system, screen_capture)
modern_window.run()
```

### Configuration Management Pattern

```python
# Open configuration panel
main_window._open_config_panel()

# Update specific parameter
config_manager.update_config('system', 'update_interval', 150)

# Apply configuration changes
main_window._update_interval_changed(150)
```

## Best Practices

### UI Development

1. **Resource Management**: Always use resource managers to prevent memory leaks
2. **Error Handling**: Implement comprehensive error handling for UI operations
3. **Performance**: Optimize rendering for large or complex visualizations
4. **User Experience**: Provide clear feedback for user actions
5. **Accessibility**: Ensure UI is accessible and usable

### System Integration

1. **Thread Safety**: Use proper synchronization for thread-safe operations
2. **State Management**: Maintain consistent state between UI and system
3. **Configuration**: Validate all configuration changes before applying
4. **Resource Cleanup**: Ensure proper cleanup on application exit
5. **Monitoring**: Implement comprehensive system monitoring and alerts

### Troubleshooting

Common UI issues and solutions:

- **Performance Issues**: Reduce visualization complexity or update frequency
- **Memory Leaks**: Check resource manager configuration and usage
- **Rendering Problems**: Validate input data and visualization parameters
- **Configuration Errors**: Verify configuration values and ranges
- **Threading Issues**: Ensure proper synchronization and thread safety