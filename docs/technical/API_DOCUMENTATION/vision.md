# Vision Processing API

Handles screen capture and image processing for the neural system with advanced features for real-time visual input.

## Module: `vision.py`

### Main Classes and Functions

#### `ThreadedScreenCapture` Class

Thread-based screen capture system for continuous visual input:

```python
class ThreadedScreenCapture:
    def __init__(self, width: int, height: int, interval: float | None = None)
```

**Parameters:**

- `width`: Capture width in pixels
- `height`: Capture height in pixels
- `interval`: Capture interval in seconds (default: from config)

**Key Features:**

- **Threaded Operation**: Non-blocking screen capture
- **Frame Queue**: Buffered frame storage with size limits
- **Error Handling**: Automatic recovery from capture errors
- **Resource Management**: Memory-efficient frame handling

```python
from project.vision import ThreadedScreenCapture

# Create screen capture with 256x144 resolution
capture = ThreadedScreenCapture(width=256, height=144)

# Start capture thread
capture.start()

# Get latest frame
frame = capture.get_latest()
```

#### `capture_screen(resolution: tuple[int, int] = (SENSOR_WIDTH, SENSOR_HEIGHT)) -> np.ndarray`

Single-frame screen capture with multiple backend support:

```python
from project.vision import capture_screen

# Capture screen at default resolution
screen_data = capture_screen()

# Capture at custom resolution
custom_data = capture_screen((320, 240))
```

**Features:**

- **Multiple Backends**: Supports PIL and MSS backends
- **Fallback Mechanism**: Automatic fallback if primary backend fails
- **Performance Monitoring**: Built-in timing and logging
- **Error Recovery**: Graceful handling of capture failures

#### `preprocess_image(image: np.ndarray) -> np.ndarray`

Image preprocessing for neural system input:

```python
from project.vision import preprocess_image

# Preprocess captured image
processed = preprocess_image(screen_data)
```

**Processing Steps:**

1. Color space conversion (RGB to grayscale)
2. Resizing to target dimensions
3. Normalization of pixel values
4. Format conversion for neural system

### Advanced Features

#### Threaded Capture System

```python
# Context manager usage
with ThreadedScreenCapture(256, 144) as capture:
    while running:
        frame = capture.get_next_frame()
        if frame is not None:
            process_frame(frame)
```

#### Error Handling and Recovery

```python
# Monitor capture health
if capture.error_count > 0:
    logger.warning(f"Capture errors: {capture.error_count}")

# Check if capture is running
if not capture.is_running:
    capture.start()  # Auto-restart if needed
```

### Configuration Parameters

Key configuration parameters from `config.py`:

- `SENSOR_WIDTH = 256`: Default capture width
- `SENSOR_HEIGHT = 144`: Default capture height
- `SCREEN_CAPTURE_QUEUE_SIZE = 100`: Frame queue size
- `PERIODIC_UPDATE_MS = 200`: Default capture interval

### Usage Patterns

#### Real-time Processing

```python
from project.vision import ThreadedScreenCapture
from project.pyg_neural_system import PyGNeuralSystem

# Initialize components
capture = ThreadedScreenCapture(256, 144)
neural_system = PyGNeuralSystem(256, 144, 30)

# Start capture
capture.start()

# Processing loop
while running:
    # Get latest frame
    frame = capture.get_latest()

    # Preprocess and update neural system
    if frame is not None:
        processed = preprocess_image(frame)
        neural_system.update_sensory_nodes(processed)

    # System update
    neural_system.update()
```

#### Performance Monitoring

```python
# Monitor capture performance
start_time = time.time()
frame_count = 0

while frame_count < 100:
    frame = capture.get_next_frame()
    if frame is not None:
        frame_count += 1

end_time = time.time()
fps = frame_count / (end_time - start_time)
logger.info(f"Capture performance: {fps:.1f} FPS")
```

### Best Practices

1. **Thread Management**: Always use context managers or explicit start/stop
2. **Error Handling**: Monitor error counts and restart if needed
3. **Resource Cleanup**: Ensure proper cleanup on application exit
4. **Performance Tuning**: Adjust resolution and interval for target performance
5. **Fallback Handling**: Implement fallback mechanisms for critical operations

### Troubleshooting

Common issues and solutions:

- **Capture Failures**: Check monitor availability and permissions
- **Performance Issues**: Reduce resolution or increase capture interval
- **Memory Problems**: Limit queue size and monitor frame drops
- **Backend Issues**: Ensure required libraries (PIL, MSS) are installed
