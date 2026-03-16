# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Constants:
#   logger                          -- Module-level logger
#   _VALIDATOR_AVAILABLE            -- bool, whether SimulationValidator imported successfully
#   _SimulationValidator            -- SimulationValidator class or None
#
# Classes:
#   ModernMainWindow(QMainWindow)
#     __init__(self, config_manager: ConfigManager, state_manager: StateManager) -> None
#       Initialize window with configuration and state managers.
#     _build_header_bar(self) -> QFrame
#       Build the top header bar with FPS/node/energy labels.
#     _build_left_column(self) -> QFrame
#       Build left column with workspace, sensory, audio views and metrics.
#     _build_center_column(self) -> QFrame
#       Build center column with Taichi window toggles and modality legend.
#     _build_right_column(self) -> QFrame
#       Build right column with simulation controls, audio, config, and interval slider.
#     @staticmethod _section_label(layout: QVBoxLayout, text: str) -> None
#       Add a styled section label to a layout.
#     @staticmethod _window_toggle_style(active: bool) -> str
#       Return QPushButton stylesheet for active/inactive window toggle.
#     _create_interval_slider_in(self, layout: QVBoxLayout) -> None
#       Build the update interval slider into the given layout.
#     _update_interval(self, value: int) -> None
#       Handle update interval slider change.
#     _toggle_workspace_window(self) -> None
#       Toggle workspace GGUI window open/closed.
#     _toggle_full_ai_window(self) -> None
#       Toggle full AI structure GGUI window open/closed.
#     _toggle_sensory_window(self) -> None
#       Toggle sensory input GGUI window open/closed.
#     _get_dark_theme_stylesheet(self) -> str
#       Return dark theme stylesheet for the application.
#     @staticmethod _button_style(bg: str, hover: str = '', pressed: str = '') -> str
#       Generate QPushButton stylesheet with consistent structure.
#     _create_control_buttons(self) -> None
#       Create control buttons with modern styling (legacy layout path).
#     _create_interval_slider(self) -> None
#       Create update interval slider with modern styling (legacy layout path).
#     _toggle_suspend(self) -> None
#       Toggle system suspension with visual feedback.
#     _pulse_energy(self) -> None
#       Pulse energy into the system with visual feedback.
#     _toggle_sensory(self) -> None
#       Toggle sensory input on/off.
#     _toggle_audio(self) -> None
#       Toggle audio capture and output on/off.
#     _toggle_audio_source(self) -> None
#       Switch audio source between loopback and microphone.
#     @pyqtSlot(int) _audio_volume_changed(self, value: int) -> None
#       Handle audio volume slider change.
#     _update_audio_spectrum_canvas(self, spectrum: np.ndarray) -> None
#       Render a stereo FFT spectrum as a bar chart in the audio view.
#     _open_config_panel(self) -> None
#       Open modern configuration panel dialog.
#     @pyqtSlot(int) _update_interval_changed(self, value: int) -> None
#       Handle capture interval change (legacy path).
#     @pyqtSlot() _update_resource_stats_display(self) -> None
#       Update resource statistics display (stub).
#     update_workspace_canvas(self, workspace_data: npt.NDArray[np.float64] | None = None) -> None
#       Update workspace canvas with new data using thread-safe operations.
#     @pyqtSlot() _process_pending_workspace_update(self) -> None
#       Process pending workspace update from worker thread on main thread.
#     on_workspace_update(self, energy_grid) -> None
#       Handle workspace system updates and render them to the canvas.
#     update_sensory_canvas(self, sensory_data: npt.NDArray[np.float64]) -> None
#       Update sensory canvas with new data.
#     update_metrics_panel(self, metrics: dict[str, Any] | None = None) -> None
#       Update metrics panel with new data.
#     on_state_change(self, state: Any) -> None
#       Handle state changes and update UI accordingly.
#     set_components(self, system: Any, capture: Any, workspace_system: Any | None = None, audio_capture: Any | None = None, audio_output: Any | None = None) -> None
#       Store components without starting the simulation.
#     start_system(self, system: Any | None = None, capture: Any | None = None, workspace_system: Any | None = None) -> None
#       Start the system, capture, and workspace components.
#     stop_system(self) -> None
#       Stop timers and background components.
#     _build_fresh_components(self) -> tuple[Any, Any, Any | None] | None
#       Recreate system, capture, and workspace components from config.
#     _handle_start(self) -> None
#       User-initiated start.
#     _handle_stop(self) -> None
#       User-initiated stop.
#     _handle_reset(self) -> None
#       Stop and rebuild a fresh map.
#     _handle_clear_log(self) -> None
#       Clear the application log file.
#     _handle_test_rules(self) -> None
#       Run comprehensive rule validation test.
#     _update_sensory(self, current_time: float) -> tuple[Any, dict[str, float]]
#       Process screen capture and update sensory nodes.
#     _update_engine(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, float]]
#       Run one engine step and retrieve metrics.
#     _update_audio(self, current_time: float) -> None
#       Process audio input spectrum and update audio output.
#     _log_frame_profiling(self, update_start: float, time_since_last: float, sensory_t: dict[str, float], engine_t: dict[str, float], step_result: dict[str, Any] | None, ui_time: float) -> None
#       Log detailed profiling info every 90 frames.
#     @pyqtSlot() periodic_update(self) -> None
#       Periodic update function with performance optimization and frame throttling.
#     resizeEvent(self, event: Any) -> None
#       Invalidate cached scene bounds so fitInView re-fires after resize.
#     closeEvent(self, a0: Any) -> None
#       Handle window closing.
#     safe_window_cleanup(self) -> None
#       Safe cleanup method that can be called during shutdown.
#     run(self) -> None
#       Start the main window.
#
# =============================================================================
# TODOS
# =============================================================================
#
# None
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Modern PyQt6 main window for the Energy-Based Neural System interface."""

# pylint: disable=no-name-in-module
# pylint: disable=import-error

import logging
import time
import threading
from typing import Any
import numpy as np
import numpy.typing as npt
import torch
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

# pylint: disable=broad-exception-caught,too-many-lines

class ModernMainWindow(QMainWindow):
    """Modern application window class for the neural system interface using PyQt6."""

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
        self._gui_manager: Any | None = None

        self.setWindowTitle("Neural Engine")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(self._get_dark_theme_stylesheet())

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._header_bar = self._build_header_bar()
        root_layout.addWidget(self._header_bar)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(8)
        root_layout.addWidget(body, stretch=1)

        left_col = self._build_left_column()
        body_layout.addWidget(left_col, stretch=3)

        center_col = self._build_center_column()
        body_layout.addWidget(center_col, stretch=2)

        right_col = self._build_right_column()
        body_layout.addWidget(right_col, stretch=2)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "color: #9999bb; background-color: #0d0d0d; "
            "font-family: 'Consolas'; font-size: 11px; border-top: 1px solid #2a2a4e;"
        )
        self.setStatusBar(self.status_bar)

        self.audio_capture: Any | None = None
        self.audio_output:  Any | None = None

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        self.resource_stats_timer = QTimer()
        self.resource_stats_timer.timeout.connect(self._update_resource_stats_display)

        self.last_update_time = 0
        self.last_sensory_canvas_update = 0
        self.last_workspace_canvas_update = 0.0
        self.last_dynamic_canvas_update = 0.0
        self.sensory_canvas_update_interval = 0.5
        self.dynamic_canvas_update_interval = 0.5
        self.canvas_frame_counter = 0
        self.min_update_interval = 0.001
        self.frame_skip_counter = 0
        self.frame_skip_threshold = 100
        self.node_read_skip_counter = 0
        self.node_read_interval = 2

        self._ui_update_lock = threading.Lock()
        self._resource_access_lock = threading.Lock()

        self.state_manager.add_observer(self)

    def _build_header_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(40)
        bar.setStyleSheet(
            "background-color: #12122a; border-bottom: 1px solid #2a2a4e;"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)

        title = QLabel("\u25cf NEURAL ENGINE")
        title.setStyleSheet("color: #44ff88; font-weight: bold; font-family: 'Consolas';")
        layout.addWidget(title)
        layout.addStretch()

        self._header_fps   = QLabel("FPS: \u2014")
        self._header_nodes = QLabel("Nodes: \u2014")
        self._header_energy = QLabel("Energy: \u2014")
        self._header_status = QLabel("Idle")
        for lbl in (self._header_fps, self._header_nodes, self._header_energy, self._header_status):
            lbl.setStyleSheet("color: #9999cc; font-family: 'Consolas'; font-size: 12px; margin: 0 10px;")
            layout.addWidget(lbl)
        return bar

    def _build_left_column(self) -> QFrame:
        col = QFrame()
        col.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        layout = QVBoxLayout(col)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.workspace_view = QGraphicsView()
        self.workspace_view.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #2a2a4e; border-radius: 4px;"
        )
        self.workspace_scene = QGraphicsScene()
        self.workspace_view.setScene(self.workspace_scene)
        self.workspace_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        layout.addWidget(self.workspace_view, stretch=3)

        self.sensory_view = QGraphicsView()
        self.sensory_view.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #2a2a4e; border-radius: 4px;"
        )
        self.sensory_scene = QGraphicsScene()
        self.sensory_view.setScene(self.sensory_scene)
        self.sensory_view.setMaximumHeight(150)
        self._sensory_pixmap_item = None
        layout.addWidget(self.sensory_view, stretch=1)

        self.audio_view = QGraphicsView()
        self.audio_view.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #2a2a4e; border-radius: 4px;"
        )
        self.audio_scene = QGraphicsScene()
        self.audio_view.setScene(self.audio_scene)
        self.audio_view.setMaximumHeight(90)
        self.audio_view.setVisible(False)
        layout.addWidget(self.audio_view)

        self.metrics_panel = QFrame()
        self.metrics_panel.setStyleSheet(
            "background-color: #12122a; border-radius: 4px; border: 1px solid #2a2a4e;"
        )
        self.metrics_panel.setMaximumHeight(160)
        m_layout = QVBoxLayout(self.metrics_panel)
        m_layout.setContentsMargins(8, 6, 8, 6)
        self.metrics_label = QLabel()
        self.metrics_label.setStyleSheet(
            "color: #c0c0e0; font-family: 'Consolas'; font-size: 11px;"
        )
        self.metrics_label.setWordWrap(True)
        m_layout.addWidget(self.metrics_label)
        layout.addWidget(self.metrics_panel)

        return col

    def _build_center_column(self) -> QFrame:
        col = QFrame()
        col.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        col.setMaximumWidth(280)
        layout = QVBoxLayout(col)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        sec_title = QLabel("TAICHI WINDOWS")
        sec_title.setStyleSheet(
            "color: #666699; font-family: 'Consolas'; font-size: 10px; letter-spacing: 1px;"
        )
        layout.addWidget(sec_title)

        self._btn_workspace_win = QPushButton("\u25b6  Workspace Grid")
        self._btn_workspace_win.setCheckable(True)
        self._btn_workspace_win.setStyleSheet(self._window_toggle_style(False))
        self._btn_workspace_win.clicked.connect(self._toggle_workspace_window)
        layout.addWidget(self._btn_workspace_win)

        self._lbl_workspace_fps = QLabel("  FPS: \u2014")
        self._lbl_workspace_fps.setStyleSheet("color: #666699; font-family: 'Consolas'; font-size: 10px;")
        layout.addWidget(self._lbl_workspace_fps)

        self._btn_full_ai_win = QPushButton("\u25b6  Full AI Structure")
        self._btn_full_ai_win.setCheckable(True)
        self._btn_full_ai_win.setStyleSheet(self._window_toggle_style(False))
        self._btn_full_ai_win.clicked.connect(self._toggle_full_ai_window)
        layout.addWidget(self._btn_full_ai_win)

        self._lbl_full_ai_fps = QLabel("  FPS: \u2014")
        self._lbl_full_ai_fps.setStyleSheet("color: #666699; font-family: 'Consolas'; font-size: 10px;")
        layout.addWidget(self._lbl_full_ai_fps)

        self._btn_sensory_win = QPushButton("\u25b6  Sensory Input")
        self._btn_sensory_win.setCheckable(True)
        self._btn_sensory_win.setStyleSheet(self._window_toggle_style(False))
        self._btn_sensory_win.clicked.connect(self._toggle_sensory_window)
        layout.addWidget(self._btn_sensory_win)

        self._lbl_sensory_fps = QLabel("  FPS: \u2014")
        self._lbl_sensory_fps.setStyleSheet("color: #666699; font-family: 'Consolas'; font-size: 10px;")
        layout.addWidget(self._lbl_sensory_fps)

        layout.addSpacing(8)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #2a2a4e;")
        layout.addWidget(divider)
        layout.addSpacing(4)

        legend_title = QLabel("MODALITY ZONES")
        legend_title.setStyleSheet(
            "color: #666699; font-family: 'Consolas'; font-size: 10px; letter-spacing: 1px;"
        )
        layout.addWidget(legend_title)

        for dot_color, name, attr in [
            ("#44ff88", "Visual",     "_lbl_visual_energy"),
            ("#4488ff", "Audio Left", "_lbl_audio_l_energy"),
            ("#ff4466", "Audio Right","_lbl_audio_r_energy"),
            ("#888888", "Neutral",    "_lbl_neutral_energy"),
        ]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            dot = QLabel("\u25cf")
            dot.setStyleSheet(f"color: {dot_color}; font-size: 14px;")
            row_layout.addWidget(dot)
            label = QLabel(name)
            label.setStyleSheet("color: #c0c0e0; font-family: 'Consolas'; font-size: 11px;")
            row_layout.addWidget(label)
            row_layout.addStretch()
            energy_lbl = QLabel("\u2014")
            energy_lbl.setStyleSheet("color: #666699; font-family: 'Consolas'; font-size: 10px;")
            row_layout.addWidget(energy_lbl)
            setattr(self, attr, energy_lbl)
            layout.addWidget(row)

        layout.addStretch()
        return col

    def _build_right_column(self) -> QFrame:
        col = QFrame()
        col.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        col.setMaximumWidth(260)
        layout = QVBoxLayout(col)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._section_label(layout, "SIMULATION")

        self.start_button = QPushButton("\u25b6  Start")
        self.start_button.setStyleSheet(self._button_style("#1a4d1a", "#22662a", "#0d330d"))
        self.start_button.clicked.connect(self._handle_start)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("\u25a0  Stop")
        self.stop_button.setStyleSheet(self._button_style("#4d1a1a", "#662222", "#330d0d"))
        self.stop_button.clicked.connect(self._handle_stop)
        layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("\u21ba  Reset Map")
        self.reset_button.setStyleSheet(self._button_style("#2a2a55", "#3a3a77", "#1a1a33"))
        self.reset_button.clicked.connect(self._handle_reset)
        layout.addWidget(self.reset_button)

        layout.addSpacing(4)
        self._section_label(layout, "ACTIONS")

        self.suspend_button = QPushButton("Drain && Suspend")
        self.suspend_button.setStyleSheet(self._button_style("#882222", "#993333", "#661111"))
        self.suspend_button.clicked.connect(self._toggle_suspend)
        layout.addWidget(self.suspend_button)

        self.pulse_button = QPushButton("Pulse +10 Energy")
        self.pulse_button.setStyleSheet(self._button_style("#225577", "#3377aa", "#113355"))
        self.pulse_button.clicked.connect(self._pulse_energy)
        layout.addWidget(self.pulse_button)

        self.sensory_button = QPushButton("Disable Sensory")
        self.sensory_button.setStyleSheet(self._button_style("#228822", "#33aa33", "#115511"))
        self.sensory_button.clicked.connect(self._toggle_sensory)
        layout.addWidget(self.sensory_button)

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.setStyleSheet(self._button_style("#444444", "#555555", "#333333"))
        self.clear_log_button.clicked.connect(self._handle_clear_log)
        layout.addWidget(self.clear_log_button)

        self.test_button = QPushButton("Test Rules")
        self.test_button.setStyleSheet(self._button_style("#7744aa", "#9955cc", "#553388"))
        self.test_button.clicked.connect(self._handle_test_rules)
        layout.addWidget(self.test_button)

        layout.addSpacing(4)
        self._section_label(layout, "AUDIO")

        self.audio_toggle_button = QPushButton("Enable Audio")
        self.audio_toggle_button.setStyleSheet(self._button_style("#664488", "#8855aa", "#443366"))
        self.audio_toggle_button.clicked.connect(self._toggle_audio)
        layout.addWidget(self.audio_toggle_button)

        self.audio_source_button = QPushButton("Source: Loopback")
        self.audio_source_button.setStyleSheet(self._button_style("#446688", "#5577aa", "#334466"))
        self.audio_source_button.clicked.connect(self._toggle_audio_source)
        self.audio_source_button.setVisible(False)
        layout.addWidget(self.audio_source_button)

        self._audio_volume_frame = QFrame()
        self._audio_volume_frame.setStyleSheet("background-color: #12122a; border-radius: 4px;")
        _avl = QVBoxLayout(self._audio_volume_frame)
        _avl.setContentsMargins(8, 8, 8, 8)
        _vol_label = QLabel("Audio Volume:")
        _vol_label.setStyleSheet("color: #c0c0e0; font-family: 'Consolas'; font-size: 11px;")
        _avl.addWidget(_vol_label)
        self.audio_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.audio_volume_slider.setRange(0, 100)
        self.audio_volume_slider.setValue(30)
        self.audio_volume_slider.valueChanged.connect(self._audio_volume_changed)
        _avl.addWidget(self.audio_volume_slider)
        self._audio_volume_frame.setVisible(False)
        layout.addWidget(self._audio_volume_frame)

        layout.addSpacing(4)
        self._section_label(layout, "CONFIG")

        self.config_button = QPushButton("Open Config Panel")
        self.config_button.setStyleSheet(self._button_style("#888822", "#aaa933", "#666611"))
        self.config_button.clicked.connect(self._open_config_panel)
        layout.addWidget(self.config_button)

        layout.addSpacing(4)
        self._section_label(layout, "UPDATE INTERVAL")

        self._create_interval_slider_in(layout)

        layout.addStretch()
        return col

    @staticmethod
    def _section_label(layout: QVBoxLayout, text: str) -> None:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #666699; font-family: 'Consolas'; font-size: 10px; "
            "letter-spacing: 1px; margin-top: 2px;"
        )
        layout.addWidget(lbl)

    @staticmethod
    def _window_toggle_style(active: bool) -> str:
        if active:
            return (
                "QPushButton { background-color: #1a3320; color: #44ff88; "
                "border: 1px solid #44ff88; border-radius: 4px; padding: 6px; "
                "font-family: 'Consolas'; font-size: 11px; text-align: left; }"
                "QPushButton:hover { background-color: #224428; }"
            )
        return (
            "QPushButton { background-color: #1a1a2e; color: #9999cc; "
            "border: 1px solid #2a2a4e; border-radius: 4px; padding: 6px; "
            "font-family: 'Consolas'; font-size: 11px; text-align: left; }"
            "QPushButton:hover { background-color: #22223a; }"
        )

    def _create_interval_slider_in(self, layout: QVBoxLayout) -> None:
        """Build the update interval slider into the given layout."""
        self.interval_label = QLabel("Update: 16 ms")
        self.interval_label.setStyleSheet(
            "color: #9999cc; font-family: 'Consolas'; font-size: 11px;"
        )
        layout.addWidget(self.interval_label)

        self.interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.interval_slider.setMinimum(1)
        self.interval_slider.setMaximum(1000)
        self.interval_slider.setValue(16)
        self.interval_slider.valueChanged.connect(self._update_interval)
        layout.addWidget(self.interval_slider)

    def _update_interval(self, value: int) -> None:
        """Handle update interval slider change."""
        self.interval_label.setText(f"Update: {value} ms")
        try:
            self.config_manager.update_config('system', 'update_interval', value)
            fps = max(1, 1000 // value)
            if self.capture and hasattr(self.capture, 'set_target_fps'):
                self.capture.set_target_fps(fps)
            self.status_bar.showMessage(f"Capture interval: {value}ms (~{fps} fps)")
        except Exception as e:
            ErrorHandler.show_error("Config Error", f"Failed to update capture interval: {str(e)}")

    def _toggle_workspace_window(self) -> None:
        if self._gui_manager is None:
            return
        if self._gui_manager.is_open("workspace"):
            self._gui_manager.close_workspace_window()
            self._btn_workspace_win.setText("\u25b6  Workspace Grid")
            self._btn_workspace_win.setStyleSheet(self._window_toggle_style(False))
        else:
            self._gui_manager.open_workspace_window()
            self._btn_workspace_win.setText("\u25a0  Workspace Grid")
            self._btn_workspace_win.setStyleSheet(self._window_toggle_style(True))

    def _toggle_full_ai_window(self) -> None:
        if self._gui_manager is None:
            return
        if self._gui_manager.is_open("full_ai"):
            self._gui_manager.close_full_ai_window()
            self._btn_full_ai_win.setText("\u25b6  Full AI Structure")
            self._btn_full_ai_win.setStyleSheet(self._window_toggle_style(False))
        else:
            self._gui_manager.open_full_ai_window()
            self._btn_full_ai_win.setText("\u25a0  Full AI Structure")
            self._btn_full_ai_win.setStyleSheet(self._window_toggle_style(True))

    def _toggle_sensory_window(self) -> None:
        if self._gui_manager is None:
            return
        if self._gui_manager.is_open("sensory"):
            self._gui_manager.close_sensory_window()
            self._btn_sensory_win.setText("\u25b6  Sensory Input")
            self._btn_sensory_win.setStyleSheet(self._window_toggle_style(False))
        else:
            self._gui_manager.open_sensory_window()
            self._btn_sensory_win.setText("\u25a0  Sensory Input")
            self._btn_sensory_win.setStyleSheet(self._window_toggle_style(True))

    def _get_dark_theme_stylesheet(self) -> str:
        """Get dark theme stylesheet for the application."""
        return """
        QMainWindow, QWidget {
            background-color: #0d0d0d;
        }
        QFrame {
            background-color: #1a1a2e;
        }
        QPushButton {
            background-color: #2a2a4e;
            color: #c0c0e0;
            border: 1px solid #3a3a5e;
            border-radius: 4px;
            padding: 7px 10px;
            font-family: 'Segoe UI';
            font-size: 11px;
        }
        QPushButton:hover { background-color: #3a3a5e; }
        QPushButton:pressed { background-color: #1a1a3e; }
        QPushButton:disabled { color: #555566; }
        QSlider::groove:horizontal {
            height: 5px;
            background: #2a2a4e;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            width: 12px; height: 12px;
            margin: -3px 0;
            border-radius: 6px;
            background: #4488ff;
        }
        QSlider::handle:horizontal:hover { background: #66aaff; }
        QLabel { color: #c0c0e0; }
        QTabWidget::pane {
            border: 1px solid #2a2a4e;
            background: #1a1a2e;
        }
        QTabBar::tab {
            background: #2a2a4e;
            color: #c0c0e0;
            padding: 5px 10px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }
        QTabBar::tab:selected { background: #3a3a5e; }
        QLineEdit, QCheckBox {
            color: #c0c0e0;
            background-color: #2a2a4e;
            border: 1px solid #3a3a5e;
            padding: 3px 6px;
            border-radius: 3px;
        }
        QStatusBar { color: #9999bb; background-color: #0d0d0d; }
        QScrollBar:vertical {
            background: #1a1a2e; width: 8px;
        }
        QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 4px; }
        """

    @staticmethod
    def _button_style(bg: str, hover: str = '', pressed: str = '') -> str:
        """Generate QPushButton stylesheet with consistent structure."""
        hover = hover or bg
        pressed = pressed or bg
        return (
            f"QPushButton {{ background-color: {bg}; color: #e0e0e0; "
            f"border: none; border-radius: 4px; padding: 10px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton:pressed {{ background-color: {pressed}; }}"
        )

    def _create_control_buttons(self) -> None:
        """Create control buttons with modern styling."""
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

        self.audio_toggle_button = QPushButton("Enable Audio")
        self.audio_toggle_button.setStyleSheet(self._button_style("#664488", "#8855aa", "#443366"))
        self.audio_toggle_button.clicked.connect(self._toggle_audio)
        self.controls_layout.addWidget(self.audio_toggle_button)

        self.audio_source_button = QPushButton("Source: Loopback")
        self.audio_source_button.setStyleSheet(self._button_style("#446688", "#5577aa", "#334466"))
        self.audio_source_button.clicked.connect(self._toggle_audio_source)
        self.audio_source_button.setVisible(False)
        self.controls_layout.addWidget(self.audio_source_button)

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
            self.status_bar.showMessage("Processing suspension request...")
            self.suspend_button.setEnabled(False)
            self.suspend_button.setText("Processing...")
            self.suspend_button.repaint()

            if self.state_manager.toggle_suspend():
                self.suspend_button.setText("Resume System")
                self.suspend_button.setStyleSheet(self._button_style("#225522", "#337733", "#113311"))
                self.status_bar.showMessage("\u2713 System suspended and drained successfully")
                logger.info("System suspended successfully")
            else:
                self.suspend_button.setText("Drain && Suspend")
                self.suspend_button.setStyleSheet(self._button_style("#882222", "#993333", "#661111"))
                self.status_bar.showMessage("\u2713 System resumed successfully")
                logger.info("System resumed successfully")

        except Exception as e:
            ErrorHandler.show_error("Suspend Error", f"Failed to toggle suspension: {str(e)}")
            self.status_bar.showMessage(f"\u2717 Error: {str(e)}")
            self.suspend_button.setEnabled(True)
            if self.state_manager.get_state().suspended:
                self.suspend_button.setText("Resume System")
            else:
                self.suspend_button.setText("Drain && Suspend")

        finally:
            self.suspend_button.setEnabled(True)

    def _pulse_energy(self) -> None:
        """Pulse energy into the system with enhanced visual feedback."""
        original_text = "Pulse +10 Energy"
        try:
            if hasattr(self, 'pulse_button') and self.pulse_button:
                original_text = self.pulse_button.text()
                self.pulse_button.setText("Pulsing...")
                self.pulse_button.setEnabled(False)
                self.pulse_button.repaint()

            pulse_amt = 0.0
            if self.system:
                pulse_amt = self.system.pulse_energy()
            self.status_bar.showMessage(f"\u2713 Energy pulse +{pulse_amt:.1f} applied successfully")
            logger.info("Energy pulse applied: +%.2f", pulse_amt)

            if hasattr(self, 'pulse_button') and self.pulse_button:
                self.pulse_button.setStyleSheet(self._button_style("#33aa33"))
                self.pulse_button.repaint()

        except Exception as e:
            ErrorHandler.show_error("Pulse Error", f"Failed to pulse energy: {str(e)}")
            self.status_bar.showMessage(f"\u2717 Pulse failed: {str(e)}")

        finally:
            if hasattr(self, 'pulse_button') and self.pulse_button:
                self.pulse_button.setText(original_text)
                self.pulse_button.setEnabled(True)
                self.pulse_button.setStyleSheet(self._button_style("#225577", "#3377aa", "#113355"))

    def _toggle_sensory(self) -> None:
        """Toggle sensory input."""
        if self.state_manager.toggle_sensory():
            self.sensory_button.setText("Disable Sensory Input")
            self.sensory_button.setStyleSheet(self._button_style("#228822", "#33aa33", "#115511"))
            self.status_bar.showMessage("Sensory input enabled.")
        else:
            self.sensory_button.setText("Enable Sensory Input")
            self.sensory_button.setStyleSheet(self._button_style("#882222", "#993333", "#661111"))
            self.status_bar.showMessage("Sensory input disabled.")

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
            bar_h = 100
            img = np.zeros((bar_h, n_bins, 3), dtype=np.uint8)
            for b in range(n_bins):
                h_L = int(min(spectrum[0, b], 1.0) * bar_h)
                h_R = int(min(spectrum[1, b], 1.0) * bar_h)
                if h_L > 0:
                    img[bar_h - h_L:, b, 1] = 180
                if h_R > 0:
                    img[bar_h - h_R:, b, 2] = 180

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
            self.status_bar.showMessage("Loading configuration panel...")

            # pylint: disable=import-outside-toplevel
            from project.ui.modern_config_panel import ModernConfigPanel
            config_dialog = ModernConfigPanel(self, self.config_manager)

            result = config_dialog.exec()

            if result == 1:
                self.status_bar.showMessage("\u2713 Configuration changes applied")
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
            self.status_bar.showMessage(f"\u2717 Error loading config: {str(e)}")

    @pyqtSlot(int)
    def _update_interval_changed(self, value: int) -> None:
        """Handle capture interval change. Sim timer stays at 0ms (uncapped)."""
        try:
            self.config_manager.update_config('system', 'update_interval', value)
            fps = max(1, 1000 // value)
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
            with self._ui_update_lock:
                if workspace_data is None:
                    workspace_config = self.config_manager.get_config('workspace')
                    if workspace_config is None:
                        raise ValueError("Workspace configuration not found")
                    workspace_data = np.zeros((
                        workspace_config['height'], workspace_config['width']
                    ))

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
                if workspace_data.ndim == 1:
                    _ws_cfg = self.config_manager.get_config('workspace')
                    _ws_h = int(_ws_cfg['height']) if _ws_cfg else 16
                    _ws_w = int(_ws_cfg['width'])  if _ws_cfg else 16
                    if workspace_data.size == _ws_h * _ws_w:
                        workspace_data = workspace_data.reshape(_ws_h, _ws_w)
                    else:
                        logger.warning(
                            "1-D workspace array size %d does not match config %d\u00d7%d; "
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

                image_data_copy = arr_rgb.copy()
                q_image = QImage(
                    image_data_copy.data.tobytes(), width, height, bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                del image_data_copy

                pixmap = QPixmap.fromImage(q_image)
                if pixmap.isNull():
                    raise ValueError("Failed to create valid QPixmap from workspace data")

                if not hasattr(self, '_workspace_pixmap_item') or self._workspace_pixmap_item is None:
                    self.workspace_scene.clear()
                    self._workspace_pixmap_item = self.workspace_scene.addPixmap(pixmap)
                else:
                    self._workspace_pixmap_item.setPixmap(pixmap)

                bounds = self.workspace_scene.itemsBoundingRect()
                if not hasattr(self, '_workspace_last_bounds') or self._workspace_last_bounds != bounds:
                    self._workspace_last_bounds = bounds
                    self.workspace_view.fitInView(
                        bounds,
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )

        except Exception as e:
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
            self.status_bar.showMessage(f"\u2717 Workspace update failed: {str(e)}")

    @pyqtSlot()
    def _process_pending_workspace_update(self) -> None:
        """Process pending workspace update from worker thread (called on main thread via QMetaObject.invokeMethod)."""
        if hasattr(self, '_pending_workspace_grid'):
            energy_grid = self._pending_workspace_grid
            delattr(self, '_pending_workspace_grid')
            self.on_workspace_update(energy_grid)

    def on_workspace_update(self, energy_grid) -> None:
        """Handle workspace system updates and render them to the canvas."""
        now = time.time()
        if hasattr(self, '_workspace_last_render_time'):
            if now - self._workspace_last_render_time < 0.015:
                return
        self._workspace_last_render_time = now

        if not hasattr(self, '_workspace_update_counter'):
            self._workspace_update_counter = 0
        self._workspace_update_counter += 1

        try:
            if isinstance(energy_grid, np.ndarray):
                workspace_data = energy_grid if energy_grid.dtype == np.float64 else energy_grid.astype(np.float64)
            elif energy_grid:
                workspace_data = np.array(energy_grid, dtype=np.float64)
            else:
                return

            _ws_cfg = self.config_manager.get_config('workspace')
            if _ws_cfg and workspace_data.ndim == 2:
                _exp_h = int(_ws_cfg.get('height', 16))
                _exp_w = int(_ws_cfg.get('width',  16))
                if workspace_data.shape != (_exp_h, _exp_w):
                    logger.warning(
                        "Workspace grid shape %s does not match config %d\u00d7%d",
                        workspace_data.shape, _exp_h, _exp_w,
                    )

            if self._workspace_update_counter % 60 == 0:
                logger.info(
                    "on_workspace_update: shape=%s min=%.2f max=%.2f",
                    workspace_data.shape, workspace_data.min(), workspace_data.max(),
                )
            self.update_workspace_canvas(workspace_data)

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
            ui_config = self.config_manager.get_config('ui')
            scale_factor = 8
            if ui_config is not None:
                scale_factor = ui_config.get('sensory_scale_factor', 8)
            if scale_factor > 1 and sensory_data.shape[0] >= scale_factor and sensory_data.shape[1] >= scale_factor:
                sensory_data = sensory_data[::scale_factor, ::scale_factor]

            if not np.isfinite(sensory_data).all():
                logger.warning("Sensory data contains NaN/Inf values, replacing with zeros")
                sensory_data = np.nan_to_num(sensory_data, nan=0.0, posinf=1.0, neginf=0.0)

            arr = np.clip(sensory_data, 0, 1)

            if arr.max() == arr.min() and arr.max() == 0:
                logger.warning("Sensory data is all zeros - screen capture may be failing")
                h, w = arr.shape
                arr = np.random.rand(h, w) * 0.1

            arr = (arr * 255).astype(np.uint8)
            arr_rgb = np.repeat(arr[:, :, None], 3, axis=2)

            height, width = arr_rgb.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(
                arr_rgb.data.tobytes(), width, height, bytes_per_line,
                QImage.Format.Format_RGB888
            )

            pixmap = QPixmap.fromImage(q_image)
            if self._sensory_pixmap_item is None:
                self._sensory_pixmap_item = self.sensory_scene.addPixmap(pixmap)
            else:
                self._sensory_pixmap_item.setPixmap(pixmap)
            self.sensory_view.fitInView(
                self.sensory_scene.itemsBoundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio
            )

        except Exception as e:
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

            total_e = metrics.get('total_energy', 0)
            total_e_str = f"{total_e/1000:.1f}K" if total_e >= 1000 else f"{total_e:.1f}"

            n_sens = metrics.get('sensory_node_count', 0)
            n_dyn = metrics.get('dynamic_node_count', 0)
            n_ws = metrics.get('workspace_node_count', 0)
            n_total = n_sens + n_dyn + n_ws

            sens_min = metrics.get('sensory_energy_min', 0)
            sens_max = metrics.get('sensory_energy_max', 0)
            ws_min = metrics.get('workspace_energy_min', 0)
            ws_max = metrics.get('workspace_energy_max', 0)
            ws_avg = metrics.get('workspace_energy_avg', 0)
            dyn_avg = metrics.get('avg_dynamic_energy', 0)

            conn_count = metrics.get('connection_count', 0)
            conns_per_dyn = metrics.get('conns_per_dynamic', 0)

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
            if state.suspended:
                self.suspend_button.setText("Resume System")
                self.suspend_button.setStyleSheet(self._button_style("#225522", "#337733", "#113311"))
            else:
                self.suspend_button.setText("Drain && Suspend")
                self.suspend_button.setStyleSheet(self._button_style("#882222", "#993333", "#661111"))

            if state.sensory_enabled:
                self.sensory_button.setText("Disable Sensory Input")
                self.sensory_button.setStyleSheet(self._button_style("#228822", "#33aa33", "#115511"))
            else:
                self.sensory_button.setText("Enable Sensory Input")
                self.sensory_button.setStyleSheet(self._button_style("#882222", "#993333", "#661111"))
        except Exception as e:
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

        if audio_capture is not None and audio_capture.is_running:
            self.audio_toggle_button.setText("Disable Audio")
            self.audio_source_button.setVisible(True)
            self._audio_volume_frame.setVisible(True)
            self.audio_view.setVisible(True)
            if hasattr(audio_capture, 'source'):
                src = audio_capture.source
                self.audio_source_button.setText(f"Source: {src.title()}")

    def start_system(self, system: Any | None = None, capture: Any | None = None, workspace_system: Any | None = None) -> None:
        """Start the system, capture, and workspace components."""
        if system is not None:
            self.system = system
        if capture is not None:
            self.capture = capture
        if workspace_system is not None:
            self.workspace_system = workspace_system
        self.frame_counter = 0
        self.growth_interval_frames = int(self.config_manager.get_config('system', 'growth_interval_frames') or 2)
        self.cull_interval_frames = int(self.config_manager.get_config('system', 'cull_interval_frames') or 3)
        if self.growth_interval_frames < 1:
            self.growth_interval_frames = 1
        if self.cull_interval_frames < 1:
            self.cull_interval_frames = 1
        if not self.update_timer.isActive():
            self.update_timer.start(0)

        if self.capture and hasattr(self.capture, 'start'):
            try:
                self.capture.start()
            except Exception as cap_err:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to start capture: %s", cap_err)

        if self.system and hasattr(self.system, 'start_connection_worker'):
            try:
                self.system.start_connection_worker(batch_size=25)
            except Exception as worker_err:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to start connection worker: %s", worker_err)

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
                self.update_workspace_canvas()
            except Exception as e:
                logger.error(f"Failed to start workspace system: {e}")
                ErrorHandler.show_error("Workspace Error", f"Failed to start workspace system: {str(e)}")

        try:
            from project.visualization.taichi_gui_manager import TaichiGUIManager
            sys = self.system
            if hasattr(sys, 'engine'):
                self._gui_manager = TaichiGUIManager(sys.engine)
            elif hasattr(sys, '_engine'):
                self._gui_manager = TaichiGUIManager(sys._engine)
        except Exception as e:
            logger.warning("TaichiGUIManager unavailable: %s", e)

        self.status_bar.showMessage("Simulation started")
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
            from project.main import initialize_system

            logger.info("Rebuilding system components (respects hybrid mode)")

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
                    report_lines.append(f"\u2717 {err}")

            if results.get('warnings'):
                report_lines.append("\n=== WARNINGS ===")
                for warn in results['warnings']:
                    report_lines.append(f"\u26a0 {warn}")

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
            if isinstance(frame, torch.Tensor):
                sensory_input = frame.float() if frame.dtype == torch.uint8 else frame
            else:
                sensory_input = frame.astype(np.float32)
            t['t_convert'] = (time.time() - t_convert_start) * 1000

            if (current_time - self.last_sensory_canvas_update) > self.sensory_canvas_update_interval:
                t_canvas_start = time.time()
                if isinstance(sensory_input, torch.Tensor):
                    canvas_input = sensory_input.cpu().numpy()
                    if canvas_input.max() > 1.0:
                        canvas_input = canvas_input / 255.0
                else:
                    canvas_input = sensory_input
                    if canvas_input.max() > 1.0:
                        canvas_input = canvas_input / 255.0
                self.update_sensory_canvas(canvas_input)
                t['t_canvas'] = (time.time() - t_canvas_start) * 1000
                self.last_sensory_canvas_update = current_time

            t_nodes_start = time.time()
            self.system.update_sensory_nodes(sensory_input)
            t['t_nodes'] = (time.time() - t_nodes_start) * 1000
        else:
            logger.debug("Received null frame from screen capture")

        t['t_sensory'] = (time.time() - t_sensory_start) * 1000
        return frame, t

    def _update_engine(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, float]]:
        """Run one engine step and retrieve metrics.

        Returns:
            (step_result, metrics, timing_dict)
        """
        t: dict[str, float] = {
            't_update': 0.0, 't_update_step': 0.0, 't_engine_step': 0.0,
            't_worker': 0.0, 't_metrics': 0.0, 't_pre_sync': 0.0, 't_engine_call': 0.0,
            't_result_access': 0.0, 't_gpu_sync': 0.0, 't_adapter': 0.0,
        }
        step_result: dict[str, Any] | None = None
        metrics: dict[str, Any] | None = None

        if not self.system:
            return step_result, metrics, t

        t_update_start = time.time()

        if hasattr(self.system, 'update_step'):
            t_update_step_start = time.time()
            step_result = self.system.update_step()
            t['t_update_step'] = (time.time() - t_update_step_start) * 1000
            if isinstance(step_result, dict):
                t['t_engine_step'] = step_result.get('total_time', 0) * 1000
                t['t_pre_sync'] = step_result.get('pre_sync_time', 0) * 1000
                t['t_engine_call'] = step_result.get('engine_call_time', 0) * 1000
                t['t_result_access'] = step_result.get('result_access_time', 0) * 1000
                t['t_gpu_sync'] = step_result.get('gpu_sync_time', 0) * 1000
                t['t_adapter'] = step_result.get('adapter_time', 0) * 1000
            else:
                t['t_pre_sync'] = 0.0
                t['t_engine_call'] = 0.0
                t['t_result_access'] = 0.0
                t['t_gpu_sync'] = 0.0
                t['t_adapter'] = 0.0
        else:
            self.system.update()
        t['t_update'] = (time.time() - t_update_start) * 1000

        t_worker_start = time.time()
        self.system.apply_connection_worker_results()
        t['t_worker'] = (time.time() - t_worker_start) * 1000

        try:
            t_metrics_start = time.time()
            metrics = self.system.get_metrics()
            t['t_metrics'] = (time.time() - t_metrics_start) * 1000
            if metrics is not None:
                logger.debug(
                    "Metrics snapshot: energy=%.1f dyn=%d sens=%d ws=%d conn=%d births=%d deaths=%d",
                    metrics.get('total_energy', 0), metrics.get('dynamic_node_count', 0),
                    metrics.get('sensory_node_count', 0), metrics.get('workspace_node_count', 0),
                    metrics.get('connection_count', 0), metrics.get('node_births', 0),
                    metrics.get('node_deaths', 0),
                )
        except Exception as metric_error:  # pylint: disable=broad-exception-caught
            logger.warning("Metrics retrieval failed: %s", metric_error)
            metrics = None

        return step_result, metrics, t

    def _update_audio(self, current_time: float) -> None:
        """Process audio input spectrum and update audio output."""
        if (self.audio_capture is None
                or not self.audio_capture.is_running
                or self.system is None):
            return
        try:
            spectrum = self.audio_capture.get_latest()
            self.system.process_audio_frame(spectrum)

            if not hasattr(self, '_last_audio_canvas_t'):
                self._last_audio_canvas_t = 0.0
            if current_time - self._last_audio_canvas_t > 0.1:
                self._update_audio_spectrum_canvas(spectrum)
                self._last_audio_canvas_t = current_time

            if self.audio_output is not None and self.audio_output.is_running:
                ws_data = self.system.get_audio_workspace_energies()
                if ws_data is not None:
                    self.audio_output.update_amplitudes(ws_data[0], ws_data[1])
        except Exception as audio_err:
            logger.debug("Audio processing error: %s", audio_err)

    def _log_frame_profiling(self, update_start: float, time_since_last: float,
                              sensory_t: dict[str, float], engine_t: dict[str, float],
                              step_result: 'dict[str, Any] | None', ui_time: float) -> None:
        """Log detailed profiling info every 90 frames."""
        if self.frame_counter % 90 != 0:
            return
        total_ms = (time.time() - update_start) * 1000
        fps = 1 / time_since_last if time_since_last > 0 else 0

        t_engine_step_val = engine_t.get('t_engine_step', 0)
        t_engine_call_val = engine_t.get('t_engine_call', 0)

        logger.info(
            "\U0001f50d ULTRA PROFILING | Total: %.1fms | Capture: %.1fms | Convert: %.1fms | "
            "Canvas: %.1fms | Nodes: %.1fms | Update: %.1fms | "
            "UpdateStep: %.1fms | EngineStep: %.1fms | Worker: %.1fms | "
            "Metrics: %.1fms | UI: %.1fms | FPS: %.1f",
            total_ms,
            sensory_t.get('t_capture', 0), sensory_t.get('t_convert', 0),
            sensory_t.get('t_canvas', 0), sensory_t.get('t_nodes', 0),
            engine_t.get('t_update', 0), engine_t.get('t_update_step', 0),
            t_engine_step_val, engine_t.get('t_worker', 0),
            engine_t.get('t_metrics', 0), ui_time * 1000, fps,
        )

        gpu_time_ms = 0.0
        if isinstance(step_result, dict):
            gpu_time_ms = step_result.get('gpu_time_ms', 0)

        logger.info(
            "   \U0001f52c GPU SYNC BREAKDOWN | PreSync: %.2fms | EngineCall: %.2fms | "
            "ResultAccess: %.2fms | GPUSync: %.2fms | Adapter: %.2fms",
            engine_t.get('t_pre_sync', 0), t_engine_call_val,
            engine_t.get('t_result_access', 0), engine_t.get('t_gpu_sync', 0),
            engine_t.get('t_adapter', 0),
        )
        logger.info(
            "   \u26a1 CUDA EVENT TIMING | GPUExecution: %.2fms | CPUTime: %.2fms | Gap: %.2fms",
            gpu_time_ms, t_engine_step_val, t_engine_call_val - gpu_time_ms,
        )
        self.status_bar.showMessage(
            f"Performance: {fps:.1f} FPS | UI Update: {ui_time * 1000:.1f}ms"
        )

    @pyqtSlot()
    def periodic_update(self) -> None:
        """Periodic update function with performance optimization and frame throttling."""
        if not self.state_manager.get_state().suspended:
            try:
                current_time = time.time()
                update_start = current_time
                time_since_last_update = current_time - self.last_update_time

                if time_since_last_update < self.min_update_interval:
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter < self.frame_skip_threshold:
                        return
                    else:
                        self.frame_skip_counter = 0
                else:
                    self.frame_skip_counter = 0

                self.last_update_time = current_time

                if hasattr(self, '_header_fps') and self.system is not None:
                    try:
                        metrics = self.system.get_metrics() if hasattr(self.system, 'get_metrics') else {}
                        fps_val = getattr(self, '_last_fps', 0)
                        nodes   = metrics.get('num_nodes', 0) if metrics else 0
                        energy  = metrics.get('avg_energy', 0.0) if metrics else 0.0
                        self._header_fps.setText(f"FPS: {fps_val:.0f}")
                        self._header_nodes.setText(f"Nodes: {nodes:,}")
                        self._header_energy.setText(f"Energy: {energy:.1f}")
                        self._header_status.setText("Running")
                    except Exception:
                        pass

                if self._gui_manager is not None:
                    for name, lbl in [
                        ("workspace", self._lbl_workspace_fps),
                        ("full_ai",   self._lbl_full_ai_fps),
                        ("sensory",   self._lbl_sensory_fps),
                    ]:
                        fps = self._gui_manager.get_fps(name)
                        lbl.setText(f"  FPS: {fps:.0f}" if fps > 0 else "  FPS: \u2014")

                frame, sensory_t = self._update_sensory(current_time)

                step_result, metrics, engine_t = self._update_engine()

                if self.workspace_system:
                    try:
                        self.status_bar.showMessage("Workspace system active")
                    except Exception as e:
                        logger.error(f"Error updating workspace system: {e}")
                        self.status_bar.showMessage(f"Workspace error: {str(e)}")

                self._update_audio(current_time)

                self.frame_counter += 1

                if self.system:
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

                start_ui_update = time.time()
                if self.system:
                    self.update_metrics_panel(metrics)
                    if metrics is not None:
                        self.state_manager.update_metrics(
                            metrics.get('total_energy', 0),
                            metrics.get('dynamic_node_count', 0),
                            metrics.get('connection_count', 0)
                        )
                ui_update_time = time.time() - start_ui_update

                self._log_frame_profiling(
                    update_start, time_since_last_update,
                    sensory_t, engine_t, step_result, ui_update_time,
                )

            except Exception as e:
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

    def resizeEvent(self, event: Any) -> None:  # pylint: disable=invalid-name
        """Invalidate cached scene bounds so fitInView re-fires after resize."""
        super().resizeEvent(event)
        self._workspace_last_bounds = None

    def closeEvent(self, a0: Any) -> None:  # pylint: disable=invalid-name
        """Handle window closing."""
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

            if hasattr(self, 'workspace_system') and self.workspace_system:
                try:
                    self.workspace_system.stop()
                    logger.info("Workspace system stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping workspace system: {e}")

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
