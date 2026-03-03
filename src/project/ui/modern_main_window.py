"""
Modern Main Window Module.

This module provides a modern PyQt6-based graphical user interface for the
Energy-Based Neural System, including workspace visualization, sensory input
display, system controls, metrics monitoring, and comprehensive error handling
with detailed error context.
"""

# pylint: disable=no-name-in-module
# PyQt6 classes are dynamically loaded and Pylint cannot detect them
# pylint: disable=import-error
# Import paths are correct but Pylint may not resolve them in all environments

import logging
import time
import threading
from typing import Any
import numpy as np
import numpy.typing as npt
import torch  # For GPU direct capture support
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
                            QLabel, QPushButton, QSlider, QGraphicsView,
                            QGraphicsScene, QStatusBar, QMessageBox, QTextEdit, QDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QPainter

from project.utils.error_handler import (
    ErrorHandler,
    ERROR_SEVERITY_CRITICAL, ERROR_SEVERITY_HIGH, ERROR_SEVERITY_MEDIUM,
    ERROR_SEVERITY_LOW, ERROR_CONTEXT_TIMESTAMP, ERROR_CONTEXT_MODULE,
    ERROR_CONTEXT_FUNCTION, ERROR_CONTEXT_ERROR_TYPE, ERROR_CONTEXT_ERROR_MESSAGE,
    ERROR_CONTEXT_ADDITIONAL_INFO
)
from project.utils.config_manager import ConfigManager
from project.system.state_manager import StateManager
from project.utils.shutdown_utils import ShutdownDetector
from project.vision import ThreadedScreenCapture  # type: ignore[import-not-found]
from project.workspace.workspace_system import WorkspaceNodeSystem  # type: ignore[import-not-found]
from project.workspace.config import EnergyReadingConfig  # type: ignore[import-not-found]
try:
    from project.utils.simulation_validator import SimulationValidator as _SimulationValidator
    _VALIDATOR_AVAILABLE = True
except Exception:  # pylint: disable=broad-exception-caught
    _SimulationValidator = None  # type: ignore[assignment,misc]
    _VALIDATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Note: Broad exception catching (Exception) is intentional in UI code
# to prevent crashes and provide user-friendly error messages.
# pylint: disable=broad-exception-caught,too-many-lines
# This module is a comprehensive UI component that legitimately requires
# many lines for complete functionality

class ModernMainWindow(QMainWindow):
    """
    Modern application window class for the neural system interface using PyQt6.

    This class provides a comprehensive graphical user interface for the
    Energy-Based Neural System, featuring modern design principles, responsive
    controls, and advanced visualization capabilities.

    Key Features:
    - Dark theme UI with modern styling
    - Workspace visualization with real-time updates
    - Sensory input display and monitoring
    - System controls with visual feedback
    - Comprehensive metrics monitoring
    - Error handling with detailed context
    - Responsive design and layout management
    - Configuration management interface

    UI Components:
    - Workspace visualization panel with zoom and pan capabilities
    - Sensory input display with real-time frame updates
    - Metrics panel showing system statistics
    - Control buttons for system management
    - Configuration panel for parameter adjustment
    - Status bar for system notifications

    Thread Safety:
    - Uses Qt's signal/slot mechanism for thread-safe UI updates
    - Implements proper event handling for concurrent operations
    - Ensures UI responsiveness during system operations
    - Provides safe cleanup during application shutdown

    Usage Patterns:
    - Real-time system monitoring and control
    - Interactive workspace visualization
    - Configuration management and tuning
    - System state monitoring and feedback
    - Error reporting and user notifications

    Example:
    ```python
    # Initialize and run the modern main window
    config_manager = ConfigManager()
    state_manager = StateManager()
    main_window = ModernMainWindow(config_manager, state_manager)

    # Start the system and capture components
    # system, capture, workspace = initialize_system(config_manager)
    # main_window.start_system(system, capture)

    # Run the application
    main_window.run()
    ```
    """

    def __init__(self, config_manager: ConfigManager, state_manager: StateManager) -> None:
        """Initialize ModernMainWindow with configuration and state managers."""
        super().__init__()
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.frame_counter = 0
        self.system: Any | None = None
        self.capture: Any | None = None
        self.workspace_system: Any | None = None
        self._workspace_observer_added = False

        # Set up main window
        self.setWindowTitle('Neural Simulation Workspace')
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(self._get_dark_theme_stylesheet())

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create left panel (visualization)
        self.left_panel = QFrame()
        self.left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.left_panel.setStyleSheet("background-color: #181818; border-radius: 8px;")
        main_layout.addWidget(self.left_panel, stretch=3)

        # Create right panel (controls)
        self.right_panel = QFrame()
        self.right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.right_panel.setStyleSheet("background-color: #222222; border-radius: 8px;")
        self.right_panel.setMaximumWidth(350)
        main_layout.addWidget(self.right_panel, stretch=1)

        # Set up left panel layout
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Create workspace visualization
        self.workspace_view = QGraphicsView()
        self.workspace_view.setStyleSheet(
            "background-color: #121212; border: 1px solid #333; border-radius: 6px;"
        )
        self.workspace_scene = QGraphicsScene()
        self.workspace_view.setScene(self.workspace_scene)
        # No smoothing — each pixel must map 1:1 to a grid cell (no bilinear blur)
        self.workspace_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        left_layout.addWidget(self.workspace_view, stretch=2)

        # Create sensory visualization
        self.sensory_view = QGraphicsView()
        self.sensory_view.setStyleSheet(
            "background-color: #121212; border: 1px solid #333; border-radius: 6px;"
        )
        self.sensory_scene = QGraphicsScene()
        self.sensory_view.setScene(self.sensory_scene)
        self.sensory_view.setMaximumHeight(200)
        left_layout.addWidget(self.sensory_view, stretch=1)

        # Create audio spectrum visualization
        self.audio_view = QGraphicsView()
        self.audio_view.setStyleSheet(
            "background-color: #121212; border: 1px solid #333; border-radius: 6px;"
        )
        self.audio_scene = QGraphicsScene()
        self.audio_view.setScene(self.audio_scene)
        self.audio_view.setMaximumHeight(120)
        self.audio_view.setVisible(False)  # Hidden until audio is enabled
        left_layout.addWidget(self.audio_view)

        # Audio state
        self.audio_capture: Any | None = None
        self.audio_output: Any | None = None

        # Create metrics panel
        self.metrics_panel = QFrame()
        self.metrics_panel.setStyleSheet("background-color: #181818; border-radius: 6px;")
        self.metrics_panel.setMaximumHeight(180)
        metrics_layout = QVBoxLayout(self.metrics_panel)
        metrics_layout.setContentsMargins(10, 10, 10, 10)

        self.metrics_label = QLabel()
        self.metrics_label.setStyleSheet(
            "color: #e0e0e0; font-family: 'Consolas'; font-size: 12px;"
        )
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)
        left_layout.addWidget(self.metrics_panel)

        # Set up right panel layout
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Create controls frame
        self.controls_frame = QFrame()
        self.controls_frame.setStyleSheet("background-color: #282828; border-radius: 6px;")
        controls_layout = QVBoxLayout(self.controls_frame)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)
        right_layout.addWidget(self.controls_frame, stretch=2)
        self.controls_layout = controls_layout

        # Create status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "color: #bbbbbb; background-color: #181818; "
            "font-family: 'Consolas'; font-size: 11px;"
        )
        self.setStatusBar(self.status_bar)

        # Create control buttons
        self._create_control_buttons()

        # Create update interval slider
        self._create_interval_slider()

        # Register as state observer
        self.state_manager.add_observer(self)

        # Set up timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        # resource_stats_timer wired but not started — implementation pending
        self.resource_stats_timer = QTimer()
        self.resource_stats_timer.timeout.connect(self._update_resource_stats_display)

        # Initialize UI

        # Performance optimization: Set up frame throttling
        self.last_update_time = 0
        self.last_sensory_canvas_update = 0  # Track sensory canvas updates
        self.last_workspace_canvas_update = 0.0
        self.last_dynamic_canvas_update = 0.0
        self.sensory_canvas_update_interval = 0.5  # Update sensory canvas every 500ms (2 FPS)
        self.dynamic_canvas_update_interval = 0.5  # 2 FPS
        self.canvas_frame_counter = 0  # For skip-frame logic
        self.min_update_interval = 0.001  # Allow up to 1000 FPS (GPU can handle it!)
        self.frame_skip_counter = 0
        self.frame_skip_threshold = 100  # Basically disabled - let GPU run fast!
        
        # UI OPTIMIZATION: Skip frames for expensive node reads (HUGE speedup!)
        self.node_read_skip_counter = 0  # Counter for node reading
        self.node_read_interval = 2  # Read nodes every N frames (2 = 50% less overhead!)

        # Thread safety: Add lock for UI updates
        self._ui_update_lock = threading.Lock()
        self._resource_access_lock = threading.Lock()

    def _get_dark_theme_stylesheet(self) -> str:
        """Get dark theme stylesheet for the application."""
        return """
        QMainWindow {
            background-color: #121212;
        }
        QPushButton {
            background-color: #333333;
            color: #e0e0e0;
            border: 1px solid #444444;
            border-radius: 4px;
            padding: 8px 12px;
            font-family: 'Segoe UI';
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #444444;
        }
        QPushButton:pressed {
            background-color: #222222;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #444444;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
            background: #666666;
        }
        QSlider::handle:horizontal:hover {
            background: #888888;
        }
        QTabWidget::pane {
            border: 1px solid #333333;
            background: #282828;
        }
        QTabBar::tab {
            background: #333333;
            color: #e0e0e0;
            padding: 6px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #444444;
        }
        QLineEdit, QCheckBox {
            color: #e0e0e0;
            background-color: #333333;
            border: 1px solid #444444;
            padding: 4px 8px;
            border-radius: 3px;
        }
        QLabel {
            color: #e0e0e0;
        }
        """

    @staticmethod
    def _button_style(bg: str, hover: str, pressed: str) -> str:
        """Generate a consistent QPushButton stylesheet from three color hex strings."""
        return (
            f"QPushButton {{ background-color: {bg}; color: #e0e0e0; border: none;"
            f" border-radius: 4px; padding: 10px; font-weight: bold; }}"
            f" QPushButton:hover {{ background-color: {hover}; }}"
            f" QPushButton:pressed {{ background-color: {pressed}; }}"
        )

    def _create_control_buttons(self) -> None:
        """Create control buttons with modern styling."""
        # Simple single-state buttons (no hover/pressed variants needed)
        self.start_button = QPushButton("Start Simulation")
        self.start_button.setStyleSheet("background-color: #227722; color: #e0e0e0; font-weight: bold;")
        self.start_button.clicked.connect(self._handle_start)
        self.controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setStyleSheet("background-color: #772222; color: #e0e0e0; font-weight: bold;")
        self.stop_button.clicked.connect(self._handle_stop)
        self.controls_layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("Reset Map")
        self.reset_button.setStyleSheet("background-color: #555577; color: #e0e0e0; font-weight: bold;")
        self.reset_button.clicked.connect(self._handle_reset)
        self.controls_layout.addWidget(self.reset_button)

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.setStyleSheet("background-color: #444444; color: #e0e0e0; font-weight: bold;")
        self.clear_log_button.clicked.connect(self._handle_clear_log)
        self.controls_layout.addWidget(self.clear_log_button)

        self.test_button = QPushButton("Test Rules")
        self.test_button.setStyleSheet("background-color: #7744aa; color: #e0e0e0; font-weight: bold;")
        self.test_button.clicked.connect(self._handle_test_rules)
        self.controls_layout.addWidget(self.test_button)

        # Styled buttons with hover/pressed states
        self.suspend_button = QPushButton("Drain && Suspend")
        self.suspend_button.setStyleSheet(self._button_style("#882222", "#993333", "#661111"))
        self.suspend_button.clicked.connect(self._toggle_suspend)
        self.controls_layout.addWidget(self.suspend_button)

        self.pulse_button = QPushButton("Pulse +10 Energy")
        self.pulse_button.setStyleSheet(self._button_style("#225577", "#3377aa", "#113355"))
        self.pulse_button.clicked.connect(self._pulse_energy)
        self.controls_layout.addWidget(self.pulse_button)

        self.sensory_button = QPushButton("Disable Sensory Input")
        self.sensory_button.setStyleSheet(self._button_style("#228822", "#33aa33", "#115511"))
        self.sensory_button.clicked.connect(self._toggle_sensory)
        self.controls_layout.addWidget(self.sensory_button)

        self.config_button = QPushButton("Config Panel")
        self.config_button.setStyleSheet(self._button_style("#888822", "#aaa933", "#666611"))
        self.config_button.clicked.connect(self._open_config_panel)
        self.controls_layout.addWidget(self.config_button)

        # --- Audio controls ---
        self.audio_toggle_button = QPushButton("Enable Audio")
        self.audio_toggle_button.setStyleSheet(self._button_style("#664488", "#8855aa", "#443366"))
        self.audio_toggle_button.clicked.connect(self._toggle_audio)
        self.controls_layout.addWidget(self.audio_toggle_button)

        self.audio_source_button = QPushButton("Source: Loopback")
        self.audio_source_button.setStyleSheet(self._button_style("#446688", "#5577aa", "#334466"))
        self.audio_source_button.clicked.connect(self._toggle_audio_source)
        self.audio_source_button.setVisible(False)
        self.controls_layout.addWidget(self.audio_source_button)

        # Audio volume slider
        self._audio_volume_frame = QFrame()
        self._audio_volume_frame.setStyleSheet("background-color: #282828; border-radius: 6px;")
        _avl = QVBoxLayout(self._audio_volume_frame)
        _avl.setContentsMargins(8, 8, 8, 8)
        _vol_label = QLabel("Audio Volume:")
        _vol_label.setStyleSheet("color: #e0e0e0; font-family: 'Segoe UI'; font-size: 11px;")
        _avl.addWidget(_vol_label)
        self.audio_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.audio_volume_slider.setRange(0, 100)
        self.audio_volume_slider.setValue(30)
        self.audio_volume_slider.valueChanged.connect(self._audio_volume_changed)
        _avl.addWidget(self.audio_volume_slider)
        self._audio_volume_frame.setVisible(False)
        self.controls_layout.addWidget(self._audio_volume_frame)

    def _create_interval_slider(self) -> None:
        """Create update interval slider with modern styling."""
        interval_frame = QFrame()
        interval_frame.setStyleSheet("background-color: #282828; border-radius: 6px;")
        interval_layout = QVBoxLayout(interval_frame)
        interval_layout.setContentsMargins(8, 8, 8, 8)

        label = QLabel("Capture Interval (ms) [sim runs uncapped]:")
        label.setStyleSheet("color: #e0e0e0; font-family: 'Segoe UI'; font-size: 11px;")
        interval_layout.addWidget(label)

        self.interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.interval_slider.setRange(16, 1000)
        self.interval_slider.setValue(self.config_manager.get_config('system', 'update_interval'))
        self.interval_slider.valueChanged.connect(self._update_interval_changed)
        interval_layout.addWidget(self.interval_slider)

        self.controls_layout.addWidget(interval_frame)

    def _toggle_suspend(self) -> None:
        """Toggle system suspension with enhanced visual feedback."""
        try:
            # Show immediate feedback
            self.status_bar.showMessage("Processing suspension request...")
            self.suspend_button.setEnabled(False)
            self.suspend_button.setText("Processing...")
            self.suspend_button.repaint()

            if self.state_manager.toggle_suspend():
                self.suspend_button.setText("Resume System")
                self.suspend_button.setStyleSheet("""
                    QPushButton {
                        background-color: #225522;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #337733;
                    }
                    QPushButton:pressed {
                        background-color: #113311;
                    }
                """)
                self.status_bar.showMessage("✓ System suspended and drained successfully")
                logger.info("System suspended successfully")
            else:
                self.suspend_button.setText("Drain && Suspend")
                self.suspend_button.setStyleSheet("""
                    QPushButton {
                        background-color: #882222;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #993333;
                    }
                    QPushButton:pressed {
                        background-color: #661111;
                    }
                """)
                self.status_bar.showMessage("✓ System resumed successfully")
                logger.info("System resumed successfully")

        except Exception as e:
            ErrorHandler.show_error("Suspend Error", f"Failed to toggle suspension: {str(e)}")
            self.status_bar.showMessage(f"✗ Error: {str(e)}")
            # Restore button state
            self.suspend_button.setEnabled(True)
            if self.state_manager.get_state().suspended:
                self.suspend_button.setText("Resume System")
            else:
                self.suspend_button.setText("Drain && Suspend")

        finally:
            self.suspend_button.setEnabled(True)

    def _pulse_energy(self) -> None:
        """Pulse energy into the system with enhanced visual feedback."""
        original_text = "Pulse +10 Energy"  # Default text
        try:
            # Show immediate feedback
            if hasattr(self, 'pulse_button') and self.pulse_button:
                original_text = self.pulse_button.text()
                self.pulse_button.setText("Pulsing...")
                self.pulse_button.setEnabled(False)
                self.pulse_button.repaint()

            # Apply energy pulse to the system
            pulse_amt = 0.0
            if self.system:
                pulse_amt = self.system.pulse_energy()
            self.status_bar.showMessage(f"✓ Energy pulse +{pulse_amt:.1f} applied successfully")
            logger.info("Energy pulse applied: +%.2f", pulse_amt)

            # Visual feedback - temporary highlight
            if hasattr(self, 'pulse_button') and self.pulse_button:
                self.pulse_button.setStyleSheet("""
                    QPushButton {
                        background-color: #33aa33;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        padding: 10px;
                        font-weight: bold;
                    }
                """)
                self.pulse_button.repaint()

        except Exception as e:
            ErrorHandler.show_error("Pulse Error", f"Failed to pulse energy: {str(e)}")
            self.status_bar.showMessage(f"✗ Pulse failed: {str(e)}")

        finally:
            # Restore original state
            if hasattr(self, 'pulse_button') and self.pulse_button:
                self.pulse_button.setText(original_text)
                self.pulse_button.setEnabled(True)
                self.pulse_button.setStyleSheet("""
                    QPushButton {
                        background-color: #225577;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #3377aa;
                    }
                    QPushButton:pressed {
                        background-color: #113355;
                    }
                """)

    def _toggle_sensory(self) -> None:
        """Toggle sensory input."""
        if self.state_manager.toggle_sensory():
            self.sensory_button.setText("Disable Sensory Input")
            self.sensory_button.setStyleSheet("""
                QPushButton {
                    background-color: #228822;
                    color: #e0e0e0;
                    border: none;
                    border-radius: 4px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #33aa33;
                }
                QPushButton:pressed {
                    background-color: #115511;
                }
            """)
            self.status_bar.showMessage("Sensory input enabled.")
        else:
            self.sensory_button.setText("Enable Sensory Input")
            self.sensory_button.setStyleSheet("""
                QPushButton {
                    background-color: #882222;
                    color: #e0e0e0;
                    border: none;
                    border-radius: 4px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #993333;
                }
                QPushButton:pressed {
                    background-color: #661111;
                }
            """)
            self.status_bar.showMessage("Sensory input disabled.")

    # ------------------------------------------------------------------
    # Audio controls
    # ------------------------------------------------------------------

    def _toggle_audio(self) -> None:
        """Toggle audio capture and output on/off."""
        if self.audio_capture is None or self.audio_output is None:
            self.status_bar.showMessage("Audio not available (sounddevice missing or audio disabled in config)")
            return

        if self.audio_capture.is_running:
            self.audio_capture.stop()
            self.audio_output.stop()
            self.audio_toggle_button.setText("Enable Audio")
            self.audio_source_button.setVisible(False)
            self._audio_volume_frame.setVisible(False)
            self.audio_view.setVisible(False)
            self.status_bar.showMessage("Audio disabled")
        else:
            self.audio_capture.start()
            self.audio_output.start()
            self.audio_toggle_button.setText("Disable Audio")
            self.audio_source_button.setVisible(True)
            self._audio_volume_frame.setVisible(True)
            self.audio_view.setVisible(True)
            self.status_bar.showMessage("Audio enabled")

    def _toggle_audio_source(self) -> None:
        """Switch audio source between loopback and microphone."""
        if self.audio_capture is None:
            return
        if self.audio_capture.source == "loopback":
            self.audio_capture.set_source("microphone")
            self.audio_source_button.setText("Source: Microphone")
        else:
            self.audio_capture.set_source("loopback")
            self.audio_source_button.setText("Source: Loopback")

    @pyqtSlot(int)
    def _audio_volume_changed(self, value: int) -> None:
        """Handle audio volume slider change."""
        if self.audio_output is not None:
            self.audio_output.set_master_volume(value / 100.0)

    def _update_audio_spectrum_canvas(self, spectrum: 'np.ndarray') -> None:
        """Render a stereo FFT spectrum as a bar chart in the audio view.

        Parameters
        ----------
        spectrum : ndarray
            Shape ``(2, fft_bins)`` — row 0 = left (green), row 1 = right (blue).
        """
        try:
            n_bins = spectrum.shape[1]
            bar_h = 100  # pixel height of the bar area
            # Create RGB image: (bar_h, n_bins, 3)
            img = np.zeros((bar_h, n_bins, 3), dtype=np.uint8)
            for b in range(n_bins):
                h_L = int(min(spectrum[0, b], 1.0) * bar_h)
                h_R = int(min(spectrum[1, b], 1.0) * bar_h)
                if h_L > 0:
                    img[bar_h - h_L:, b, 1] = 180  # green for L
                if h_R > 0:
                    img[bar_h - h_R:, b, 2] = 180  # blue for R

            height, width = img.shape[:2]
            q_image = QImage(
                img.data.tobytes(), width, height, 3 * width,
                QImage.Format.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(q_image)

            if not hasattr(self, '_audio_pixmap_item') or self._audio_pixmap_item is None:
                self.audio_scene.clear()
                self._audio_pixmap_item = self.audio_scene.addPixmap(pixmap)
            else:
                self._audio_pixmap_item.setPixmap(pixmap)

            self.audio_view.fitInView(
                self.audio_scene.itemsBoundingRect(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
            )
        except Exception as e:
            logger.debug("Audio spectrum canvas error: %s", e)

    def _open_config_panel(self) -> None:
        """Open modern configuration panel with enhanced visual feedback."""
        try:
            # Show loading feedback
            self.status_bar.showMessage("Loading configuration panel...")

            # Import and create modern config panel
            # pylint: disable=import-outside-toplevel
            # Lazy import to avoid circular dependencies
            from project.ui.modern_config_panel import ModernConfigPanel
            config_dialog = ModernConfigPanel(self, self.config_manager)

            # Show the dialog
            result = config_dialog.exec()

            if result == 1:  # Accepted
                self.status_bar.showMessage("✓ Configuration changes applied")
                logger.info("Configuration panel closed with changes applied")
            else:
                self.status_bar.showMessage("Configuration panel closed")
                logger.info("Configuration panel closed without changes")

        except Exception as e:
            ErrorHandler.show_error(
                "Config Panel Error",
                f"Failed to open configuration panel: {str(e)}\n\n"
                "Please check:\n"
                "- Configuration files are accessible\n"
                "- Sufficient system resources are available\n"
                "- No permission issues exist",
                severity=ERROR_SEVERITY_MEDIUM
            )
            self.status_bar.showMessage(f"✗ Error loading config: {str(e)}")

    @pyqtSlot(int)
    def _update_interval_changed(self, value: int) -> None:
        """Handle capture interval change. Sim timer stays at 0ms (uncapped)."""
        try:
            self.config_manager.update_config('system', 'update_interval', value)
            fps = max(1, 1000 // value)
            # Push new target FPS to the async capture thread if available
            if self.capture and hasattr(self.capture, 'set_target_fps'):
                self.capture.set_target_fps(fps)
            self.status_bar.showMessage(f"Capture interval: {value}ms (~{fps} fps)")
        except Exception as e:
            ErrorHandler.show_error("Config Error", f"Failed to update capture interval: {str(e)}")

    @pyqtSlot()
    def _update_resource_stats_display(self) -> None:
        """Update resource statistics display. Not yet implemented."""
        pass

    def update_workspace_canvas(
        self, workspace_data: npt.NDArray[np.float64] | None = None
    ) -> None:
        """Update workspace canvas with new data using thread-safe operations (OPTIMIZED!)."""
        try:
            # Thread-safe canvas update
            with self._ui_update_lock:
                if workspace_data is None:
                    # Create empty workspace
                    workspace_config = self.config_manager.get_config('workspace')
                    if workspace_config is None:
                        raise ValueError("Workspace configuration not found")
                    workspace_data = np.zeros((
                        workspace_config['height'], workspace_config['width']
                    ))

                # Soft auto-ranging: tracks observed extremes with slow exponential
                # adaptation (alpha=0.02 ≈ 50-frame window) to avoid the grey-flash
                # that per-frame hard auto-ranging causes, while still using the full
                # pixel scale instead of the ±65000 fixed range that maps actual
                # 0-500 energies to <0.4% of the display scale.
                _arr_min = float(workspace_data.min())
                _arr_max = float(workspace_data.max())
                if not hasattr(self, '_ws_range_min') or self._ws_range_min is None:
                    self._ws_range_min = _arr_min
                    self._ws_range_max = max(_arr_max, _arr_min + 1.0)
                else:
                    _alpha = 0.02
                    self._ws_range_min = self._ws_range_min * (1 - _alpha) + _arr_min * _alpha
                    self._ws_range_max = self._ws_range_max * (1 - _alpha) + _arr_max * _alpha
                    if self._ws_range_max <= self._ws_range_min:
                        self._ws_range_max = self._ws_range_min + 1.0
                # Ensure 2-D before colormap (guard against 1-D legacy paths).
                # Use config dimensions — the square-root assumption breaks non-square grids.
                if workspace_data.ndim == 1:
                    _ws_cfg = self.config_manager.get_config('workspace')
                    _ws_h = int(_ws_cfg['height']) if _ws_cfg else 16
                    _ws_w = int(_ws_cfg['width'])  if _ws_cfg else 16
                    if workspace_data.size == _ws_h * _ws_w:
                        workspace_data = workspace_data.reshape(_ws_h, _ws_w)
                    else:
                        logger.warning(
                            "1-D workspace array size %d does not match config %d×%d; "
                            "displaying as single row.", workspace_data.size, _ws_h, _ws_w
                        )
                        workspace_data = workspace_data.reshape(1, -1)
                arr = np.clip(
                    (workspace_data - self._ws_range_min) / (self._ws_range_max - self._ws_range_min) * 255.0,
                    0, 255
                ).astype(np.uint8)
                arr_rgb = np.repeat(arr[:, :, np.newaxis], 3, axis=2)

                height, width = arr_rgb.shape[:2]
                bytes_per_line = 3 * width

                # Create QImage with proper memory management
                image_data_copy = arr_rgb.copy()
                q_image = QImage(
                    image_data_copy.data.tobytes(), width, height, bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                del image_data_copy  # Free memory

                # Create QPixmap and display
                pixmap = QPixmap.fromImage(q_image)
                if pixmap.isNull():
                    raise ValueError("Failed to create valid QPixmap from workspace data")

                # Update scene without clearing — avoids the brief blank frame
                # that appears between scene.clear() and addPixmap() on each tick.
                if not hasattr(self, '_workspace_pixmap_item') or self._workspace_pixmap_item is None:
                    self.workspace_scene.clear()
                    self._workspace_pixmap_item = self.workspace_scene.addPixmap(pixmap)
                else:
                    self._workspace_pixmap_item.setPixmap(pixmap)

                # Re-fit on every frame so the view stays aligned after window resizes.
                # The pixmap is always 1px-per-node (e.g. 16×16); fitInView scales it up
                # uniformly with KeepAspectRatio. SmoothPixmapTransform is disabled above
                # so each node cell remains a crisp integer-sized block.
                self.workspace_view.fitInView(
                    self.workspace_scene.itemsBoundingRect(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )

        except Exception as e:
            # Enhanced error reporting with detailed context
            error_context: dict[str, Any] = {
                ERROR_CONTEXT_TIMESTAMP: time.time(),
                ERROR_CONTEXT_MODULE: 'modern_main_window',
                ERROR_CONTEXT_FUNCTION: 'update_workspace_canvas',
                ERROR_CONTEXT_ERROR_TYPE: 'CanvasUpdateError',
                ERROR_CONTEXT_ERROR_MESSAGE: str(e),
                ERROR_CONTEXT_ADDITIONAL_INFO: {
                    'workspace_data_shape': (
                        str(workspace_data.shape)
                        if workspace_data is not None else 'None'
                    ),
                    'config_available': True,
                    'thread_safe': True
                }
            }

            severity = ERROR_SEVERITY_MEDIUM
            if 'memory' in str(e).lower() or 'out of' in str(e).lower():
                severity = ERROR_SEVERITY_HIGH

            ErrorHandler.show_error(
                "Canvas Error",
                f"Failed to update workspace: {str(e)}",
                severity=severity,
                context=error_context
            )
            self.status_bar.showMessage(f"✗ Workspace update failed: {str(e)}")

    @pyqtSlot()
    def _process_pending_workspace_update(self) -> None:
        """Process pending workspace update from worker thread (called on main thread via QMetaObject.invokeMethod)."""
        if hasattr(self, '_pending_workspace_grid'):
            energy_grid = self._pending_workspace_grid
            delattr(self, '_pending_workspace_grid')  # Clean up
            self.on_workspace_update(energy_grid)
    
    def on_workspace_update(self, energy_grid) -> None:
        """Handle workspace system updates and render them to the canvas."""
        if not hasattr(self, '_workspace_update_counter'):
            self._workspace_update_counter = 0
        self._workspace_update_counter += 1

        try:
            # Accept both numpy arrays (fast path from cached adapter) and list-of-lists
            if isinstance(energy_grid, np.ndarray):
                workspace_data = energy_grid if energy_grid.dtype == np.float64 else energy_grid.astype(np.float64)
            elif energy_grid:
                workspace_data = np.array(energy_grid, dtype=np.float64)
            else:
                return

            # Validate that the incoming shape matches the configured grid dimensions
            _ws_cfg = self.config_manager.get_config('workspace')
            if _ws_cfg and workspace_data.ndim == 2:
                _exp_h = int(_ws_cfg.get('height', 16))
                _exp_w = int(_ws_cfg.get('width',  16))
                if workspace_data.shape != (_exp_h, _exp_w):
                    logger.warning(
                        "Workspace grid shape %s does not match config %d×%d",
                        workspace_data.shape, _exp_h, _exp_w,
                    )

            if self._workspace_update_counter % 60 == 0:
                logger.info(
                    "on_workspace_update: shape=%s min=%.2f max=%.2f",
                    workspace_data.shape, workspace_data.min(), workspace_data.max(),
                )
            self.update_workspace_canvas(workspace_data)

            # Update status bar using already-computed numpy array (no Python list needed)
            if workspace_data.size > 0:
                self.status_bar.showMessage(
                    f"Workspace: Avg={workspace_data.mean():.1f}, "
                    f"Max={workspace_data.max():.1f}, Min={workspace_data.min():.1f}"
                )
        except Exception as e:
            logger.error(f"Error handling workspace update: {e}")
            self.status_bar.showMessage(f"Workspace visualization error: {str(e)}")

    def update_sensory_canvas(self, sensory_data: npt.NDArray[np.float64]) -> None:
        """Update sensory canvas with new data (OPTIMIZED!)."""
        try:
            # OPTIMIZATION: Downsample for faster rendering (SENSORY-SPECIFIC!)
            ui_config = self.config_manager.get_config('ui')
            scale_factor = 8  # Default 8× downsampling for sensory (performance!)
            if ui_config is not None:
                scale_factor = ui_config.get('sensory_scale_factor', 8)
            if scale_factor > 1 and sensory_data.shape[0] >= scale_factor and sensory_data.shape[1] >= scale_factor:
                # Downsample by taking every Nth pixel
                sensory_data = sensory_data[::scale_factor, ::scale_factor]
            
            # Convert to QImage
            # CRITICAL FIX: Handle corrupted/invalid sensory data
            # Check for NaN, Inf, or invalid values
            if not np.isfinite(sensory_data).all():
                logger.warning("Sensory data contains NaN/Inf values, replacing with zeros")
                sensory_data = np.nan_to_num(sensory_data, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure data is in valid range [0, 1]
            arr = np.clip(sensory_data, 0, 1)
            
            # Check if data is all zeros or all same value (corrupted)
            if arr.max() == arr.min() and arr.max() == 0:
                logger.warning("Sensory data is all zeros - screen capture may be failing")
                # Show a test pattern instead of pure black
                h, w = arr.shape
                arr = np.random.rand(h, w) * 0.1  # Very dim random noise for visibility
            
            arr = (arr * 255).astype(np.uint8)
            arr_rgb = np.repeat(arr[:, :, None], 3, axis=2)

            height, width = arr_rgb.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(
                arr_rgb.data.tobytes(), width, height, bytes_per_line,
                QImage.Format.Format_RGB888
            )

            # Create QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            self.sensory_scene.clear()
            self.sensory_scene.addPixmap(pixmap)
            self.sensory_view.fitInView(
                self.sensory_scene.itemsBoundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio
            )

        except Exception as e:
            # Enhanced error reporting with detailed context
            error_context: dict[str, Any] = {
                ERROR_CONTEXT_TIMESTAMP: time.time(),
                ERROR_CONTEXT_MODULE: 'modern_main_window',
                ERROR_CONTEXT_FUNCTION: 'update_sensory_canvas',
                ERROR_CONTEXT_ERROR_TYPE: 'SensoryUpdateError',
                ERROR_CONTEXT_ERROR_MESSAGE: str(e),
                ERROR_CONTEXT_ADDITIONAL_INFO: {
                    'sensory_data_shape': str(sensory_data.shape),
                    'expected_shape': 'N/A'
                }
            }

            severity = ERROR_SEVERITY_MEDIUM
            if 'memory' in str(e).lower() or 'out of' in str(e).lower():
                severity = ERROR_SEVERITY_HIGH

            ErrorHandler.show_error(
                "Canvas Error",
                f"Failed to update sensory: {str(e)}",
                severity=severity,
                context=error_context
            )

    def update_metrics_panel(self, metrics: dict[str, Any] | None = None) -> None:
        """Update metrics panel with new data."""
        try:
            if metrics is None:
                self.metrics_label.setText("Metrics not available")
                return

            # Format energy values with K suffix for large numbers
            total_e = metrics.get('total_energy', 0)
            total_e_str = f"{total_e/1000:.1f}K" if total_e >= 1000 else f"{total_e:.1f}"
            
            # Node counts
            n_sens = metrics.get('sensory_node_count', 0)
            n_dyn = metrics.get('dynamic_node_count', 0)
            n_ws = metrics.get('workspace_node_count', 0)
            n_total = n_sens + n_dyn + n_ws
            
            # Energy stats
            sens_min = metrics.get('sensory_energy_min', 0)
            sens_max = metrics.get('sensory_energy_max', 0)
            ws_min = metrics.get('workspace_energy_min', 0)
            ws_max = metrics.get('workspace_energy_max', 0)
            ws_avg = metrics.get('workspace_energy_avg', 0)
            dyn_avg = metrics.get('avg_dynamic_energy', 0)
            
            # Connection stats
            conn_count = metrics.get('connection_count', 0)
            conns_per_dyn = metrics.get('conns_per_dynamic', 0)
            
            # Step count
            step_count = metrics.get('step_count', 0)
            
            metrics_text = (
                f"<b>Step:</b> {step_count} | <b>Energy:</b> {total_e_str}<br>"
                f"<b>Nodes:</b> {n_total} (S:{n_sens} D:{n_dyn} W:{n_ws})<br>"
                f"<b>Sensory:</b> [{sens_min:.0f} - {sens_max:.0f}]<br>"
                f"<b>Dynamic:</b> avg {dyn_avg:.1f}<br>"
                f"<b>Workspace:</b> [{ws_min:.0f} - {ws_max:.0f}] avg {ws_avg:.1f}<br>"
                f"<b>Connections:</b> {conn_count} ({conns_per_dyn:.1f}/node)<br>"
                f"<b>Births:</b> N:{metrics.get('total_node_births', 0)} C:{metrics.get('total_conn_births', 0)}<br>"
                f"<b>Deaths:</b> N:{metrics.get('total_node_deaths', 0)} C:{metrics.get('total_conn_deaths', 0)}"
            )
            self.metrics_label.setText(metrics_text)
        except Exception as e:
            # Enhanced error reporting with detailed context
            error_context: dict[str, Any] = {
                ERROR_CONTEXT_TIMESTAMP: time.time(),
                ERROR_CONTEXT_MODULE: 'modern_main_window',
                ERROR_CONTEXT_FUNCTION: 'update_metrics_panel',
                ERROR_CONTEXT_ERROR_TYPE: 'MetricsUpdateError',
                ERROR_CONTEXT_ERROR_MESSAGE: str(e),
                ERROR_CONTEXT_ADDITIONAL_INFO: {
                    'metrics_keys': list(metrics.keys()) if metrics is not None else [],
                    'metrics_available': metrics is not None
                }
            }

            severity = ERROR_SEVERITY_LOW
            if metrics is None:
                severity = ERROR_SEVERITY_MEDIUM

            ErrorHandler.show_error(
                "Metrics Error",
                f"Failed to update metrics: {str(e)}",
                severity=severity,
                context=error_context
            )

    def on_state_change(self, state: Any) -> None:
        """Handle state changes."""
        try:
            # Update UI based on state
            if state.suspended:
                self.suspend_button.setText("Resume System")
                self.suspend_button.setStyleSheet("""
                    QPushButton {
                        background-color: #225522;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #337733;
                    }
                    QPushButton:pressed {
                        background-color: #113311;
                    }
                """)
            else:
                self.suspend_button.setText("Drain && Suspend")
                self.suspend_button.setStyleSheet("""
                    QPushButton {
                        background-color: #882222;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #993333;
                    }
                    QPushButton:pressed {
                        background-color: #661111;
                    }
                """)

            if state.sensory_enabled:
                self.sensory_button.setText("Disable Sensory Input")
                self.sensory_button.setStyleSheet("""
                    QPushButton {
                        background-color: #228822;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #33aa33;
                    }
                    QPushButton:pressed {
                        background-color: #115511;
                    }
                """)
            else:
                self.sensory_button.setText("Enable Sensory Input")
                self.sensory_button.setStyleSheet("""
                    QPushButton {
                        background-color: #882222;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #993333;
                    }
                    QPushButton:pressed {
                        background-color: #661111;
                    }
                """)
        except Exception as e:
            # Enhanced error reporting with detailed context
            error_context: dict[str, Any] = {
                ERROR_CONTEXT_TIMESTAMP: time.time(),
                ERROR_CONTEXT_MODULE: 'modern_main_window',
                ERROR_CONTEXT_FUNCTION: 'on_state_change',
                ERROR_CONTEXT_ERROR_TYPE: 'StateUpdateError',
                ERROR_CONTEXT_ERROR_MESSAGE: str(e),
                ERROR_CONTEXT_ADDITIONAL_INFO: {
                    'state_type': type(state).__name__ if state is not None else 'None',
                    'state_available': state is not None
                }
            }

            severity = ERROR_SEVERITY_MEDIUM
            if 'suspended' in str(e).lower() or 'sensory' in str(e).lower():
                severity = ERROR_SEVERITY_HIGH

            ErrorHandler.show_error(
                "State Error",
                f"Failed to update UI state: {str(e)}",
                severity=severity,
                context=error_context
            )

    def set_components(
        self,
        system: Any,
        capture: Any,
        workspace_system: Any | None = None,
        audio_capture: Any | None = None,
        audio_output: Any | None = None,
    ) -> None:
        """Store components without starting the simulation."""
        self.system = system
        self.capture = capture
        self.workspace_system = workspace_system
        self.audio_capture = audio_capture
        self.audio_output = audio_output
        self._workspace_observer_added = False

        # Sync audio UI state with actual audio state
        if audio_capture is not None and audio_capture.is_running:
            self.audio_toggle_button.setText("Disable Audio")
            self.audio_source_button.setVisible(True)
            self._audio_volume_frame.setVisible(True)
            self.audio_view.setVisible(True)
            if hasattr(audio_capture, 'source'):
                src = audio_capture.source
                self.audio_source_button.setText(f"Source: {src.title()}")

    def start_system(self, system: Any | None = None, capture: Any | None = None, workspace_system: Any | None = None) -> None:
        """
        Start the system, capture, and workspace components.

        This method initializes the main system components and starts the periodic update timer.
        It sets up the neural system, screen capture, and workspace system components for real-time operation.

        Args:
            system: The neural system instance to be controlled by this window
            capture: The screen capture system instance for sensory input
            workspace_system: The workspace system instance for energy visualization

        Example:
        ```python
        # Initialize system components
        # system, capture, workspace = initialize_system(config_manager)
        screen_capture = ScreenCaptureSystem(config_manager)
        workspace_system = WorkspaceNodeSystem(neural_system, config)

        # Start the system in the main window
        main_window.start_system(neural_system, screen_capture, workspace_system)

        # The window will now display real-time updates from all systems
        # and allow user interaction with the neural network and workspace
        ```
        """
        if system is not None:
            self.system = system
        if capture is not None:
            self.capture = capture
        if workspace_system is not None:
            self.workspace_system = workspace_system
        self.frame_counter = 0
        # Allow configurable growth/cull cadence (frames)
        self.growth_interval_frames = int(self.config_manager.get_config('system', 'growth_interval_frames') or 2)
        self.cull_interval_frames = int(self.config_manager.get_config('system', 'cull_interval_frames') or 3)
        if self.growth_interval_frames < 1:
            self.growth_interval_frames = 1
        if self.cull_interval_frames < 1:
            self.cull_interval_frames = 1
        if not self.update_timer.isActive():
            # 0ms interval: fires on every Qt event loop iteration — sim runs uncapped.
            # Screen capture rate is controlled separately via AsyncScreenCapture.set_target_fps().
            self.update_timer.start(0)

        # Start capture thread if available
        if self.capture and hasattr(self.capture, 'start'):
            try:
                self.capture.start()
            except Exception as cap_err:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to start capture: %s", cap_err)

        # Start connection worker if available
        if self.system and hasattr(self.system, 'start_connection_worker'):
            try:
                self.system.start_connection_worker(batch_size=25)
            except Exception as worker_err:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to start connection worker: %s", worker_err)

        # Start workspace system if available
        if self.workspace_system:
            try:
                if hasattr(self.workspace_system, 'add_observer'):
                    if not hasattr(self, '_workspace_observer_added') or not self._workspace_observer_added:
                        self.workspace_system.add_observer(self)
                        self._workspace_observer_added = True
                        logger.info(f"Added self as workspace observer, total observers: {len(self.workspace_system.observers)}")
                if hasattr(self.workspace_system, 'start'):
                    self.workspace_system.start()
                logger.info("Workspace system started successfully")
                # Initialize workspace canvas with placeholder
                self.update_workspace_canvas()
            except Exception as e:
                logger.error(f"Failed to start workspace system: {e}")
                ErrorHandler.show_error("Workspace Error", f"Failed to start workspace system: {str(e)}")

        self.status_bar.showMessage("Simulation started")
        # Prime connection growth to avoid sparse graphs at startup
        if self.system:
            for _ in range(3):
                self.system.queue_connection_growth()

    def stop_system(self) -> None:
        """Stop timers and background components."""
        try:
            if hasattr(self, 'update_timer') and self.update_timer.isActive():
                self.update_timer.stop()
            if self.workspace_system and hasattr(self.workspace_system, 'stop'):
                self.workspace_system.stop()
            if self.capture and hasattr(self.capture, 'stop'):
                self.capture.stop()
            if self.system and hasattr(self.system, 'stop_connection_worker'):
                try:
                    self.system.stop_connection_worker()
                except Exception:  # pylint: disable=broad-exception-caught
                    if hasattr(self.system, 'wait_for_workers_idle'):
                        self.system.wait_for_workers_idle()
            # Stop audio subsystem
            if hasattr(self, 'audio_capture') and self.audio_capture and hasattr(self.audio_capture, 'stop'):
                try:
                    self.audio_capture.stop()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            if hasattr(self, 'audio_output') and self.audio_output and hasattr(self.audio_output, 'stop'):
                try:
                    self.audio_output.stop()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            self.status_bar.showMessage("Simulation stopped")
            logger.info("Simulation stopped")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to stop simulation cleanly: %s", e)

    def _build_fresh_components(self) -> tuple[Any, Any, Any | None] | None:
        """Recreate system, capture, and workspace components from config.
        
        This method now respects hybrid mode configuration and will create
        either a hybrid or traditional system based on the config.
        """
        try:
            # Import the initialize_system function from main
            # This ensures we use the SAME logic that creates systems at startup
            from project.main import initialize_system
            
            logger.info("Rebuilding system components (respects hybrid mode)")
            
            # Use the centralized initialization logic
            system, capture, workspace_system = initialize_system(self.config_manager)
            
            logger.info("System components rebuilt successfully")
            return system, capture, workspace_system
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to build fresh components: %s", e)
            ErrorHandler.show_error("Reset Error", f"Failed to build components: {str(e)}")
            return None

    def _handle_start(self) -> None:
        """User-initiated start."""
        if self.system is None or self.capture is None:
            built = self._build_fresh_components()
            if built is None:
                return
            system, capture, workspace_system = built
            self.set_components(system, capture, workspace_system)
        self.start_system()

    def _handle_stop(self) -> None:
        """User-initiated stop."""
        self.stop_system()

    def _handle_reset(self) -> None:
        """Stop and rebuild a fresh map."""
        try:
            self.stop_system()
            if self.system and hasattr(self.system, 'cleanup'):
                self.system.cleanup()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Cleanup during reset failed: %s", e)
        built = self._build_fresh_components()
        if built is None:
            return
        system, capture, workspace_system = built
        self.set_components(system, capture, workspace_system)
        self.status_bar.showMessage("Reset complete. Press Start to run.")

    def _handle_clear_log(self) -> None:
        """Clear the application log file."""
        try:
            with open('pyg_system.log', 'w', encoding='utf-8'):
                pass
            self.status_bar.showMessage("Log cleared")
            logger.info("Log file cleared by user")
        except Exception as e:  # pylint: disable=broad-exception-caught
            ErrorHandler.show_error("Log Error", f"Failed to clear log: {str(e)}")

    def _handle_test_rules(self) -> None:
        """Run comprehensive rule validation test."""
        try:
            self.test_button.setEnabled(False)
            self.test_button.setText("Testing...")
            self.status_bar.showMessage("Running rule validation test...")

            # Determine device
            device = "cpu"
            try:
                import torch  # type: ignore[import-untyped]
                device_pref = self.config_manager.get_config('system', 'device') or 'auto'
                if isinstance(device_pref, str):
                    device_pref = device_pref.lower()
                if device_pref in ('auto', 'cuda') and torch.cuda.is_available():
                    device = 'cuda'
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            if not _VALIDATOR_AVAILABLE or _SimulationValidator is None:
                results = {
                    "status": "NOT_AVAILABLE",
                    "message": "SimulationValidator not yet migrated to Taichi engine.",
                }
            else:
                validator = _SimulationValidator()
                results = validator.run_full_test(device=device)

            # Display results in a dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Simulation Rule Test Results")
            dialog.setMinimumSize(600, 400)
            layout = QVBoxLayout(dialog)

            status_text = QLabel(f"<h2>Status: {results.get('status', 'UNKNOWN')}</h2>")
            status_text.setStyleSheet("color: #e0e0e0; padding: 10px;")
            layout.addWidget(status_text)

            results_text = QTextEdit()
            results_text.setReadOnly(True)
            results_text.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0; font-family: monospace;")

            report_lines = []
            report_lines.append("=== TEST RESULTS ===\n")
            for key, value in results.items():
                if key not in ('status', 'errors', 'warnings'):
                    report_lines.append(f"{key}: {value}")

            if results.get('errors'):
                report_lines.append("\n=== ERRORS ===")
                for err in results['errors']:
                    report_lines.append(f"✗ {err}")

            if results.get('warnings'):
                report_lines.append("\n=== WARNINGS ===")
                for warn in results['warnings']:
                    report_lines.append(f"⚠ {warn}")

            results_text.setPlainText("\n".join(report_lines))
            layout.addWidget(results_text)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.exec()

            status_msg = f"Test {results.get('status', 'completed')}"
            if results.get('errors'):
                status_msg += f" - {len(results['errors'])} errors"
            self.status_bar.showMessage(status_msg)
            logger.info("Rule validation test completed: %s", results.get('status'))

        except Exception as e:  # pylint: disable=broad-exception-caught
            ErrorHandler.show_error("Test Error", f"Failed to run test: {str(e)}")
            logger.exception("Test execution failed")
        finally:
            self.test_button.setEnabled(True)
            self.test_button.setText("Test Rules")

    def _update_sensory(self, current_time: float) -> tuple[Any, dict[str, float]]:
        """Process screen capture and update sensory nodes.

        Returns:
            (frame_or_None, timing_dict) where timing_dict has keys:
            t_sensory, t_capture, t_canvas, t_nodes, t_convert
        """
        t: dict[str, float] = {
            't_sensory': 0.0, 't_capture': 0.0, 't_canvas': 0.0,
            't_nodes': 0.0, 't_convert': 0.0,
        }
        t_sensory_start = time.time()
        frame = None

        if self.state_manager.get_state().sensory_enabled and self.capture and self.system:
            t_capture_start = time.time()
            frame = self.capture.get_latest()
            t['t_capture'] = (time.time() - t_capture_start) * 1000

        if frame is not None:
            t_convert_start = time.time()
            # OPTIMIZATION: Handle torch.Tensor frames (GPU, no CPU transfer!)
            if isinstance(frame, torch.Tensor):
                # GPU frame - keep on GPU! (NO CPU transfer!)
                sensory_input = frame.float() if frame.dtype == torch.uint8 else frame
            else:
                # CPU frame (numpy) - convert to float32, raw values as energy
                sensory_input = frame.astype(np.float32)
            t['t_convert'] = (time.time() - t_convert_start) * 1000

            # Update sensory CANVAS only every 500ms (not every frame!) - Huge performance boost!
            if (current_time - self.last_sensory_canvas_update) > self.sensory_canvas_update_interval:
                t_canvas_start = time.time()
                # For canvas update, convert to numpy if needed
                if isinstance(sensory_input, torch.Tensor):
                    canvas_input = sensory_input.cpu().numpy()
                    # Normalize to [0, 1] range if it's in [0, 255]
                    if canvas_input.max() > 1.0:
                        canvas_input = canvas_input / 255.0
                else:
                    canvas_input = sensory_input
                    # Normalize to [0, 1] range if needed
                    if canvas_input.max() > 1.0:
                        canvas_input = canvas_input / 255.0
                self.update_sensory_canvas(canvas_input)
                t['t_canvas'] = (time.time() - t_canvas_start) * 1000
                self.last_sensory_canvas_update = current_time

            # Update sensory NODES every frame (OPTIMIZED with GPU direct!)
            t_nodes_start = time.time()
            self.system.update_sensory_nodes(sensory_input)  # Can be torch.Tensor now!
            t['t_nodes'] = (time.time() - t_nodes_start) * 1000
            self.status_bar.showMessage("Sensory input processed")
        else:
            logger.debug("Received null frame from screen capture")
            self.status_bar.showMessage("Waiting for capture frame...")

        t['t_sensory'] = (time.time() - t_sensory_start) * 1000
        return frame, t

    @pyqtSlot()
    def periodic_update(self) -> None:
        """Periodic update function with performance optimization and frame throttling."""
        if not self.state_manager.get_state().suspended:
            try:
                update_start = time.time()
                current_time = time.time()
                time_since_last_update = current_time - self.last_update_time

                # Minimal frame throttling (GPU mode can handle high FPS)
                # Only skip if updates are EXTREMELY fast (< 1ms)
                if time_since_last_update < self.min_update_interval:
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter < self.frame_skip_threshold:
                        # Only skip if we're getting insane FPS
                        return
                    else:
                        self.frame_skip_counter = 0
                else:
                    self.frame_skip_counter = 0

                self.last_update_time = current_time

                # Sensory update with visual feedback + DETAILED PROFILING
                frame, sensory_t = self._update_sensory(current_time)
                t_sensory = sensory_t['t_sensory']

                # System update with progress feedback + DETAILED PROFILING
                t_system_start = time.time()
                t_update = 0.0
                t_update_step = 0.0
                t_engine_step = 0.0
                t_worker = 0.0
                t_metrics = 0.0
                t_pre_sync = 0.0
                t_engine_call = 0.0
                t_result_access = 0.0
                t_gpu_sync = 0.0
                t_adapter = 0.0
                step_result: dict[str, Any] | None = None  # Set by update_step() branch; kept None if update() path taken
                metrics: dict[str, Any] | None = None
                if self.system:
                    t_update_start = time.time()
                    # Break down update() into components
                    if hasattr(self.system, 'update_step'):
                        t_update_step_start = time.time()
                        step_result = self.system.update_step()
                        t_update_step = (time.time() - t_update_step_start) * 1000
                        # Extract detailed timing if available
                        if isinstance(step_result, dict):
                            t_engine_step = step_result.get('total_time', 0) * 1000
                            t_pre_sync = step_result.get('pre_sync_time', 0) * 1000
                            t_engine_call = step_result.get('engine_call_time', 0) * 1000
                            t_result_access = step_result.get('result_access_time', 0) * 1000
                            t_gpu_sync = step_result.get('gpu_sync_time', 0) * 1000
                            t_adapter = step_result.get('adapter_time', 0) * 1000
                        else:
                            t_pre_sync = 0.0
                            t_engine_call = 0.0
                            t_result_access = 0.0
                            t_gpu_sync = 0.0
                            t_adapter = 0.0
                    else:
                        self.system.update()
                    t_update = (time.time() - t_update_start) * 1000
                    
                    t_worker_start = time.time()
                    self.system.apply_connection_worker_results()
                    t_worker = (time.time() - t_worker_start) * 1000
                    
                    self.status_bar.showMessage("System updated")
                    try:
                        t_metrics_start = time.time()
                        metrics = self.system.get_metrics()
                        t_metrics = (time.time() - t_metrics_start) * 1000
                        if metrics is not None:
                            # CRITICAL: Clamp all values to prevent infinity conversion errors
                            total_energy = float(max(-1e6, min(metrics.get('total_energy', 0.0), 1e9)))
                            dyn_count = int(max(0, min(metrics.get('dynamic_node_count', 0), 1e9)))
                            sens_count = int(max(0, min(metrics.get('sensory_node_count', 0), 1e9)))
                            ws_count = int(max(0, min(metrics.get('workspace_node_count', 0), 1e9)))
                            conn_count = int(max(0, min(metrics.get('connection_count', 0), 1e9)))
                            births = int(max(0, min(metrics.get('node_births', 0), 1e9)))
                            deaths = int(max(0, min(metrics.get('node_deaths', 0), 1e9)))
                            logger.debug(
                                "Metrics snapshot | energy=%.2f dyn=%s sens=%s ws=%s edges=%s births=%s deaths=%s",
                                total_energy,
                                dyn_count,
                                sens_count,
                                ws_count,
                                conn_count,
                                births,
                                deaths,
                            )
                    except Exception as metric_error:  # pylint: disable=broad-exception-caught
                        logger.warning("Metrics retrieval failed: %s", metric_error)
                        metrics = None
                t_system = (time.time() - t_system_start) * 1000

                # Workspace system update with visual feedback
                if self.workspace_system:
                    try:
                        # The workspace system runs in its own thread, but we can trigger updates
                        # or check for new data here
                        self.status_bar.showMessage("Workspace system active")
                    except Exception as e:
                        logger.error(f"Error updating workspace system: {e}")
                        self.status_bar.showMessage(f"Workspace error: {str(e)}")

                # Audio processing (inject spectrum + read workspace + update output)
                if (self.audio_capture is not None
                        and self.audio_capture.is_running
                        and self.system is not None):
                    try:
                        spectrum = self.audio_capture.get_latest()  # (2, fft_bins)
                        self.system.process_audio_frame(spectrum)

                        # Update spectrum canvas (~10 FPS to save CPU)
                        if not hasattr(self, '_last_audio_canvas_t'):
                            self._last_audio_canvas_t = 0.0
                        if current_time - self._last_audio_canvas_t > 0.1:
                            self._update_audio_spectrum_canvas(spectrum)
                            self._last_audio_canvas_t = current_time

                        # Feed workspace energy back to audio output
                        if self.audio_output is not None and self.audio_output.is_running:
                            ws_data = self.system.get_audio_workspace_energies()
                            if ws_data is not None:
                                self.audio_output.update_amplitudes(ws_data[0], ws_data[1])
                    except Exception as audio_err:
                        logger.debug("Audio processing error: %s", audio_err)

                # Queue connection tasks with visual feedback
                self.frame_counter += 1

                if self.system:
                    # Throttle growth/cull when edge count is low to avoid starving nodes
                    safe_to_modify_edges = True
                    if metrics is not None:
                        edge_count = metrics.get('connection_count', 0)
                        dyn_count = metrics.get('dynamic_node_count', 0)
                        min_edges = max(100, int(dyn_count * 2.0))
                        safe_to_modify_edges = edge_count >= min_edges
                    if not self.frame_counter % self.growth_interval_frames:
                        self.system.queue_connection_growth()
                        self.status_bar.showMessage("Connection growth queued")
                    if safe_to_modify_edges and (not self.frame_counter % self.cull_interval_frames):
                        self.system.queue_cull()
                        self.status_bar.showMessage("Connection culling queued")
                    elif not safe_to_modify_edges:
                        logger.debug("Skipping cull; edge_count below safe minimum")

                # Update UI with performance monitoring (SKIP FRAMES FOR METRICS!)
                start_ui_update = time.time()
                # Note: Workspace canvas is updated via observer pattern (on_workspace_update)
                # from WorkspaceNodeSystem, not here. Calling update_workspace_canvas() with
                # no args would overwrite the observer data with an empty canvas.
                if self.system:
                    # metrics was already fetched above; pass it directly.
                    self.update_metrics_panel(metrics)
                    if metrics is not None:
                        self.state_manager.update_metrics(
                            metrics.get('total_energy', 0),
                            metrics.get('dynamic_node_count', 0),
                            metrics.get('connection_count', 0)
                        )
                ui_update_time = time.time() - start_ui_update

                # Show performance stats periodically + ULTRA-DETAILED PROFILING
                if self.frame_counter % 90 == 0:  # Log every 90 frames (reduce overhead!)
                    total_update_time = (time.time() - update_start) * 1000
                    fps = 1 / time_since_last_update if time_since_last_update > 0 else 0
                    t_sensory_val = t_sensory
                    t_system_val = t_system
                    t_capture_val = t_capture
                    t_convert_val = t_convert
                    t_nodes_val = t_nodes
                    t_canvas_val = t_canvas
                    t_update_val = t_update
                    t_update_step_val = t_update_step
                    t_engine_step_val = t_engine_step
                    t_engine_call_val = t_engine_call
                    t_result_access_val = t_result_access
                    t_gpu_sync_val = t_gpu_sync
                    t_adapter_val = t_adapter
                    t_worker_val = t_worker
                    t_metrics_val = t_metrics
                    
                    logger.info(f"🔍 ULTRA PROFILING | Total: {total_update_time:.1f}ms | "
                               f"Capture: {t_capture_val:.1f}ms | Convert: {t_convert_val:.1f}ms | "
                               f"Canvas: {t_canvas_val:.1f}ms | Nodes: {t_nodes_val:.1f}ms | "
                               f"Update: {t_update_val:.1f}ms | UpdateStep: {t_update_step_val:.1f}ms | "
                               f"EngineStep: {t_engine_step_val:.1f}ms | Worker: {t_worker_val:.1f}ms | "
                               f"Metrics: {t_metrics_val:.1f}ms | UI: {ui_update_time*1000:.1f}ms | FPS: {fps:.1f}")
                    t_pre_sync_val = t_pre_sync
                    # Extract GPU timing from step_result if available
                    gpu_time_ms = 0.0
                    if isinstance(step_result, dict):
                        gpu_time_ms = step_result.get('gpu_time_ms', 0)
                    
                    logger.info(f"   🔬 GPU SYNC BREAKDOWN | PreSync: {t_pre_sync_val:.2f}ms | "
                               f"EngineCall: {t_engine_call_val:.2f}ms | "
                               f"ResultAccess: {t_result_access_val:.2f}ms | "
                               f"GPUSync: {t_gpu_sync_val:.2f}ms | "
                               f"Adapter: {t_adapter_val:.2f}ms")
                    logger.info(f"   ⚡ CUDA EVENT TIMING | GPUExecution: {gpu_time_ms:.2f}ms | "
                               f"CPUTime: {t_engine_step_val:.2f}ms | "
                               f"Gap: {t_engine_call_val - gpu_time_ms:.2f}ms")
                    self.status_bar.showMessage(
                        f"Performance: {fps:.1f} FPS | "
                        f"UI Update: {ui_update_time*1000:.1f}ms"
                    )

            except Exception as e:
                # Enhanced error reporting with detailed context
                error_context: dict[str, Any] = {
                    ERROR_CONTEXT_TIMESTAMP: time.time(),
                    ERROR_CONTEXT_MODULE: 'modern_main_window',
                    ERROR_CONTEXT_FUNCTION: 'periodic_update',
                    ERROR_CONTEXT_ERROR_TYPE: 'PeriodicUpdateError',
                    ERROR_CONTEXT_ERROR_MESSAGE: str(e),
                    ERROR_CONTEXT_ADDITIONAL_INFO: {
                        'frame_counter': self.frame_counter,
                        'sensory_enabled': (
                            self.state_manager.get_state().sensory_enabled
                            if hasattr(self, 'state_manager') else False
                        ),
                        'system_suspended': (
                            self.state_manager.get_state().suspended
                            if hasattr(self, 'state_manager') else False
                        ),
                        'last_update_time': self.last_update_time,
                        'time_since_last_update': time.time() - self.last_update_time
                    }
                }

                # Periodic update errors are critical for UI responsiveness
                severity = ERROR_SEVERITY_HIGH
                if 'memory' in str(e).lower() or 'out of' in str(e).lower():
                    severity = ERROR_SEVERITY_CRITICAL

                logger.error("Error during update: %s", str(e))
                ErrorHandler.show_error(
                    "Update Error",
                    f"Error during update: {str(e)}",
                    severity=severity,
                    context=error_context
                )
                self.status_bar.showMessage(f"Error: {str(e)} - Check logs for details")

    def closeEvent(self, a0: Any) -> None:  # pylint: disable=invalid-name
        """Handle window closing."""
        # closeEvent is a Qt method name that must match Qt's naming convention
        try:
            ShutdownDetector.safe_cleanup(self.safe_window_cleanup, "MainWindow cleanup")
        except Exception as e:
            ErrorHandler.show_error("Close Error", f"Error during cleanup: {str(e)}")
        a0.accept()

    def safe_window_cleanup(self) -> None:
        """Safe cleanup method that can be called during shutdown."""
        try:
            if hasattr(self, 'update_timer') and self.update_timer.isActive():
                self.update_timer.stop()
            if hasattr(self, 'resource_stats_timer') and self.resource_stats_timer.isActive():
                self.resource_stats_timer.stop()

            # Stop workspace system if it exists
            if hasattr(self, 'workspace_system') and self.workspace_system:
                try:
                    self.workspace_system.stop()
                    logger.info("Workspace system stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping workspace system: {e}")

            # Stop audio subsystem
            for attr in ('audio_capture', 'audio_output'):
                obj = getattr(self, attr, None)
                if obj and hasattr(obj, 'stop'):
                    try:
                        obj.stop()
                    except Exception as e:
                        logger.debug("Error stopping %s: %s", attr, e)
        except Exception as e:
            ErrorHandler.log_warning(f"Error during window cleanup: {str(e)}")

    def run(self) -> None:
        """Start the main window."""
        self.show()
