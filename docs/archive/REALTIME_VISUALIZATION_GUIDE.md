# Real-time Visualization Window Implementation Guide

## Overview

This document describes the implementation of the real-time visualization window for the workspace node map, providing dynamic visualization of node states, connections, and emergent patterns with interactive features.

## Features

### Core Visualization Features

1. **Real-time Updates**: Live visualization of workspace node energy states and trends
2. **Multiple View Modes**: 
   - Grid View: Traditional energy grid visualization
   - Node View: Individual node visualization with energy-based coloring
   - Connection View: Connection network visualization
   - Heatmap View: Heatmap-style energy distribution

3. **Interactive Features**:
   - Zoom and Pan: Smooth zooming and panning with mouse wheel and drag
   - Node Inspection: Click to inspect individual nodes with detailed information
   - Hover Effects: Real-time hover feedback for node information
   - Context Menu: Right-click context menu for quick actions

4. **Display Options**:
   - Node Labels: Toggle node energy labels
   - Connections: Show/hide connection visualization
   - Energy Trends: Display energy trend indicators
   - Color Schemes: Multiple color schemes for different visualization needs

### Window Modes

1. **Dedicated Window**: Separate, resizable window for focused visualization
2. **Embedded Panel**: Integrated panel within the main application window
3. **Toggle Support**: Easy switching between modes

## Architecture

### Main Components

#### 1. RealTimeVisualizationWindow
- **Purpose**: Main window class for the real-time visualization
- **Features**: 
  - Complete UI with controls and visualization area
  - Real-time update timer
  - Signal-based communication
  - Window management (dedicated vs embedded)

#### 2. EnhancedWorkspaceRenderer
- **Purpose**: Enhanced renderer extending the base WorkspaceRenderer
- **Features**:
  - Multiple visualization modes
  - Interactive element management
  - Zoom and pan transformations
  - Performance optimization

#### 3. WorkspaceVisualizationIntegration
- **Purpose**: Integration manager for connecting visualization with workspace system
- **Features**:
  - Configuration management
  - Window lifecycle management
  - Status monitoring
  - Export functionality

#### 4. Visualization Integration Updates
- **Purpose**: Enhanced existing visualization.py for seamless integration
- **Features**:
  - Backward compatibility with existing UI
  - Real-time window button integration
  - Mode synchronization

## Usage

### Basic Usage

```python
from src.project.workspace.realtime_visualization import RealTimeVisualizationWindow
from src.project.workspace.visualization_integration import WorkspaceVisualizationIntegration

# Create visualization window
visualization_window = RealTimeVisualizationWindow(workspace_system)

# Start in dedicated window mode
visualization_window.set_dedicated_window(True)
visualization_window.show()

# Configure visualization
visualization_window.renderer.set_visualization_mode(VisualizationMode.NODE_VIEW)
visualization_window._set_zoom(2.0)
```

### Integration with Existing System

```python
from src.project.workspace.visualization import WorkspaceVisualization

# The existing WorkspaceVisualization class now includes real-time window support
workspace_viz = WorkspaceVisualization(main_window, workspace_system)

# Open real-time window from embedded panel
workspace_viz._open_realtime_view()
```

### Configuration

```python
# Configure visualization integration
config = {
    'auto_start': True,
    'dedicated_window': True,
    'update_interval': 50,  # ms
    'default_mode': 'grid',
    'show_labels': False,
    'show_connections': True
}

integration = WorkspaceVisualizationIntegration(workspace_system)
integration.update_configuration(config)
```

## Technical Details

### Performance Optimization

1. **Frame Throttling**: Automatic frame rate limiting to maintain UI responsiveness
2. **Update Caching**: Smart caching of energy data to reduce computation
3. **Memory Management**: Efficient memory usage with proper cleanup
4. **Thread Safety**: Thread-safe operations for concurrent updates

### Event Handling

1. **Mouse Events**: Custom event filtering for zoom, pan, and inspection
2. **Keyboard Events**: Keyboard shortcuts for common operations
3. **Window Events**: Proper window lifecycle management
4. **Signal/Slot**: Qt signal/slot mechanism for communication

### Data Flow

1. **Workspace System** → **Visualization Window**: Energy grid and trend data
2. **User Input** → **Renderer**: Zoom, pan, and interaction commands
3. **Renderer** → **UI**: Updated visualization elements
4. **Status Updates** → **Integration**: System health and statistics

## API Reference

### RealTimeVisualizationWindow

#### Methods

- `start_system(workspace_system)`: Start the visualization system
- `set_dedicated_window(dedicated)`: Set window mode
- `_update_visualization()`: Update visualization with current data
- `_toggle_updates()`: Pause/resume automatic updates
- `_reset_view()`: Reset zoom and pan to defaults
- `_export_view()`: Export current view to image

#### Properties

- `renderer`: EnhancedWorkspaceRenderer instance
- `auto_update_enabled`: Boolean indicating update status
- `update_interval`: Update interval in milliseconds
- `is_dedicated_window`: Boolean indicating window mode

### EnhancedWorkspaceRenderer

#### Methods

- `set_visualization_mode(mode)`: Set visualization mode
- `set_interaction_mode(mode)`: Set interaction mode
- `set_zoom_level(zoom)`: Set zoom level with bounds checking
- `set_pan_offset(offset)`: Set pan offset
- `toggle_node_labels(show)`: Toggle node labels
- `toggle_connections(show)`: Toggle connections

#### Properties

- `visualization_mode`: Current visualization mode
- `interaction_mode`: Current interaction mode
- `zoom_level`: Current zoom level
- `pan_offset`: Current pan offset
- `show_node_labels`: Boolean for label visibility
- `show_connections`: Boolean for connection visibility

### WorkspaceVisualizationIntegration

#### Methods

- `start_visualization(dedicated_window)`: Start visualization
- `stop_visualization()`: Stop visualization
- `update_configuration(config)`: Update configuration
- `get_status()`: Get system status
- `export_visualization(filename)`: Export visualization
- `toggle_pause()`: Toggle updates
- `reset_view()`: Reset view
- `set_visualization_mode(mode)`: Set mode
- `set_zoom_level(zoom)`: Set zoom
- `show_node_info(node_id)`: Show node information

## Testing

### Test Coverage

The implementation includes comprehensive tests in `tests/test_realtime_visualization.py`:

1. **Window Creation**: Basic window creation and initialization
2. **Visualization Modes**: Testing all visualization modes
3. **Interaction Modes**: Testing all interaction modes
4. **Zoom Functionality**: Zoom level bounds and slider functionality
5. **Display Options**: Label and connection toggling
6. **Update Intervals**: Timer and update functionality
7. **Pause/Resume**: Update control functionality
8. **View Reset**: Zoom and pan reset functionality
9. **Export Functionality**: Image export testing
10. **Node Selection**: Node inspection functionality
11. **Integration**: Integration class functionality
12. **Error Handling**: Error handling and edge cases
13. **Performance**: Performance monitoring and optimization

### Running Tests

```bash
python -m pytest tests/test_realtime_visualization.py -v
```

## Integration with Existing System

### ModernMainWindow Integration

The real-time visualization integrates seamlessly with the existing `ModernMainWindow`:

1. **Embedded Panel**: The existing workspace panel now includes a "Open Real-time View" button
2. **Mode Synchronization**: Visualization modes are synchronized between embedded and dedicated views
3. **Status Updates**: Status bar updates include real-time visualization information
4. **Configuration**: Configuration changes apply to both embedded and dedicated views

### Workspace System Integration

The visualization integrates with the workspace system through:

1. **Observer Pattern**: The visualization observes workspace system updates
2. **Data Flow**: Energy grid and trend data flow from workspace to visualization
3. **Health Monitoring**: System health and performance metrics are displayed
4. **Error Handling**: Errors in workspace system are reflected in visualization

## Future Enhancements

### Planned Features

1. **Advanced Analytics**: Real-time analytics and pattern detection
2. **Export Options**: Multiple export formats (PNG, SVG, video)
3. **Customization**: User-defined color schemes and visualization styles
4. **Multi-monitor**: Support for multi-monitor setups
5. **VR/AR**: Virtual reality and augmented reality support
6. **Collaboration**: Multi-user collaboration features

### Performance Improvements

1. **GPU Acceleration**: GPU-based rendering for large datasets
2. **Level of Detail**: Adaptive detail based on zoom level
3. **Streaming**: Real-time data streaming for large-scale systems
4. **Caching**: Advanced caching strategies for better performance

## Troubleshooting

### Common Issues

1. **High CPU Usage**: Reduce update interval or enable frame throttling
2. **Memory Leaks**: Ensure proper cleanup in window close events
3. **Slow Updates**: Check workspace system performance
4. **Display Issues**: Verify Qt version and graphics drivers

### Debug Information

The visualization provides extensive debug information:

1. **Status Bar**: Real-time status updates
2. **Logging**: Comprehensive logging for troubleshooting
3. **Performance Metrics**: Update times and frame rates
4. **Error Reporting**: Detailed error messages and context

## Conclusion

The real-time visualization window provides a powerful and flexible way to visualize workspace node maps with real-time updates, interactive features, and multiple display modes. The implementation is designed for performance, usability, and integration with the existing system.