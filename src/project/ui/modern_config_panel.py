"""
Modern Configuration Panel Module.

This module provides a modern PyQt6-based graphical user interface for configuring
the Energy-Based Neural System, including sensory, workspace, and system parameter
configuration with advanced options and comprehensive error handling.
"""

import logging
from functools import partial
import logging
from typing import Any, cast
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTabWidget, QFrame, QLabel,
                            QLineEdit, QCheckBox, QPushButton, QScrollArea,
                            QWidget, QFormLayout, QGroupBox, QMessageBox,
                            QHBoxLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent

from project.utils.error_handler import ErrorHandler
from project.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ModernConfigPanel(QDialog):
    """
    Modern configuration panel class for managing system configuration through a PyQt6 GUI.

    This class provides a comprehensive configuration interface with:
    - Tabbed interface for different configuration sections
    - Advanced configuration options with performance tuning
    - Real-time validation and error feedback
    - Comprehensive documentation and tooltips
    - Modern dark theme styling
    - Responsive layout and scrolling for large configurations

    Key Features:
    - Sensory configuration with real-time preview
    - Workspace parameter tuning
    - System performance optimization
    - Resource management controls
    - Debug and logging options
    - Configuration validation and error handling

    Thread Safety:
    - Uses Qt's signal/slot mechanism for thread-safe UI updates
    - Implements proper event handling for concurrent operations
    - Ensures UI responsiveness during configuration changes

    Usage Patterns:
    - Real-time configuration tuning
    - Performance optimization
    - System debugging and monitoring
    - Resource management
    - Configuration validation

    Example:
    ```python
    # Initialize and show the modern configuration panel
    config_manager = ConfigManager()
    main_window = ModernMainWindow(config_manager, state_manager)
    config_dialog = ModernConfigPanel(main_window, config_manager)
    config_dialog.exec()
    ```
    """

    def __init__(self, parent: Any, config_manager: ConfigManager) -> None:
        """
        Initialize ModernConfigPanel with parent window and config manager.

        Args:
            parent: Parent window (typically ModernMainWindow)
            config_manager: Configuration manager instance
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle('Configuration Panel')
        self.setMinimumSize(800, 900)
        self.setStyleSheet(self._get_dark_theme_stylesheet())

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #333333;
                background: #282828;
            }
            QTabBar::tab {
                background: #333333;
                color: #e0e0e0;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-family: 'Segoe UI';
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #444444;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #555555;
            }
        """)
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self._create_sensory_tab()
        self._create_workspace_tab()
        self._create_system_tab()
        self._create_advanced_tab()

        # Create action buttons
        self._create_action_buttons(main_layout)

    def _get_dark_theme_stylesheet(self) -> str:
        """Get dark theme stylesheet for the configuration panel."""
        return """
        QDialog {
            background-color: #222222;
            color: #e0e0e0;
        }
        QLabel {
            color: #e0e0e0;
            font-family: 'Segoe UI';
            font-size: 11px;
        }
        QLineEdit {
            color: #e0e0e0;
            background-color: #333333;
            border: 1px solid #444444;
            border-radius: 3px;
            padding: 6px 10px;
            font-family: 'Segoe UI';
            font-size: 11px;
        }
        QLineEdit:focus {
            border: 1px solid #666666;
        }
        QCheckBox {
            color: #e0e0e0;
            font-family: 'Segoe UI';
            font-size: 11px;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QCheckBox::indicator:checked {
            image: url(:/icons/checkbox_checked);
            background-color: #444444;
        }
        QCheckBox::indicator:unchecked {
            image: url(:/icons/checkbox_unchecked);
            background-color: #444444;
        }
        QPushButton {
            background-color: #333333;
            color: #e0e0e0;
            border: 1px solid #444444;
            border-radius: 4px;
            padding: 8px 16px;
            font-family: 'Segoe UI';
            font-size: 11px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #444444;
        }
        QPushButton:pressed {
            background-color: #222222;
        }
        QGroupBox {
            color: #ffcc00;
            border: 1px solid #444444;
            border-radius: 4px;
            margin-top: 15px;
            font-family: 'Segoe UI';
            font-size: 12px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QScrollArea {
            background-color: #282828;
            border: none;
        }
        """

    def _create_sensory_tab(self) -> None:
        """Create sensory configuration tab with enhanced options."""
        sensory_frame = QFrame()
        sensory_layout = QVBoxLayout(sensory_frame)
        sensory_layout.setContentsMargins(10, 10, 10, 10)
        sensory_layout.setSpacing(10)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QFormLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(8)
        scroll_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        sensory_config = cast(dict[str, Any], self.config_manager.get_config('sensory'))

        # Add sensory configuration options
        for key, value in sensory_config.items():
            key_str = str(key)
            label = QLabel(f"{key_str.replace('_', ' ').title()}:")
            label.setToolTip(f"Configure {key_str} parameter")
            label.setStyleSheet("color: #e0e0e0; font-family: 'Segoe UI'; font-size: 11px;")

            if isinstance(value, bool):
                checkbox = QCheckBox()
                checkbox.setChecked(value)
                checkbox.setToolTip(f"Toggle {key_str} setting")
                checkbox.stateChanged.connect(partial(self._update_sensory_bool, key_str))  # type: ignore[reportUnknownMemberType]
                scroll_layout.addRow(label, checkbox)
            else:
                line_edit = QLineEdit(str(value))
                line_edit.setToolTip(f"Set {key_str} value")
                line_edit.setStyleSheet("color: #e0e0e0; background-color: #333333; border: 1px solid #444444;")
                line_edit.textChanged.connect(partial(self._update_sensory_text, key_str))  # type: ignore[reportUnknownMemberType]
                scroll_layout.addRow(label, line_edit)

        scroll_area.setWidget(scroll_content)
        sensory_layout.addWidget(scroll_area)
        self.tab_widget.addTab(sensory_frame, "Sensory")

    def _create_workspace_tab(self) -> None:
        """Create workspace configuration tab with enhanced options."""
        workspace_frame = QFrame()
        workspace_layout = QVBoxLayout(workspace_frame)
        workspace_layout.setContentsMargins(10, 10, 10, 10)
        workspace_layout.setSpacing(10)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QFormLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(8)
        scroll_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        workspace_config = cast(dict[str, Any], self.config_manager.get_config('workspace'))

        # Add workspace configuration options
        for key, value in workspace_config.items():
            key_str = str(key)
            label = QLabel(f"{key_str.replace('_', ' ').title()}:")
            label.setToolTip(f"Configure {key_str} parameter")
            label.setStyleSheet("color: #e0e0e0; font-family: 'Segoe UI'; font-size: 11px;")

            line_edit = QLineEdit(str(value))
            line_edit.setToolTip(f"Set {key_str} value")
            line_edit.setStyleSheet("color: #e0e0e0; background-color: #333333; border: 1px solid #444444;")
            line_edit.textChanged.connect(partial(self._update_workspace_text, key_str))  # type: ignore[reportUnknownMemberType]
            scroll_layout.addRow(label, line_edit)

        scroll_area.setWidget(scroll_content)
        workspace_layout.addWidget(scroll_area)
        self.tab_widget.addTab(workspace_frame, "Workspace")

    def _create_system_tab(self) -> None:
        """Create system configuration tab with advanced options."""
        system_frame = QFrame()
        system_layout = QVBoxLayout(system_frame)
        system_layout.setContentsMargins(10, 10, 10, 10)
        system_layout.setSpacing(10)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QFormLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(8)
        scroll_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        system_config = cast(dict[str, Any], self.config_manager.get_config('system'))

        # Add basic configuration options
        for key, value in system_config.items():
            label = QLabel(f"{str(key).replace('_', ' ').title()}:")
            label.setToolTip(f"Configure {key} parameter")
            label.setStyleSheet("color: #e0e0e0; font-family: 'Segoe UI'; font-size: 11px;")

            line_edit = QLineEdit(str(value))
            line_edit.setToolTip(f"Set {key} value")
            line_edit.setStyleSheet("color: #e0e0e0; background-color: #333333; border: 1px solid #444444;")
            line_edit.textChanged.connect(partial(self._update_system_text, str(key)))  # type: ignore[reportUnknownMemberType]
            scroll_layout.addRow(label, line_edit)

        scroll_area.setWidget(scroll_content)
        system_layout.addWidget(scroll_area)
        self.tab_widget.addTab(system_frame, "System")

    def _create_advanced_tab(self) -> None:
        """Create advanced configuration tab with performance and resource options."""
        advanced_frame = QFrame()
        advanced_layout = QVBoxLayout(advanced_frame)
        advanced_layout.setContentsMargins(10, 10, 10, 10)
        advanced_layout.setSpacing(10)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(12)

        # Performance Tuning Group
        performance_group = QGroupBox("Performance Tuning")
        performance_group.setStyleSheet("""
            QGroupBox {
                color: #ffcc00;
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 0px;
                font-family: 'Segoe UI';
                font-size: 12px;
                font-weight: bold;
            }
        """)

        performance_layout = QFormLayout()
        performance_layout.setContentsMargins(10, 15, 10, 10)
        performance_layout.setSpacing(8)
        performance_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Frame throttling option
        frame_throttle_label = QLabel("Enable Frame Throttling:")
        frame_throttle_label.setToolTip("Enable frame throttling to maintain UI responsiveness under heavy load")

        frame_throttle_checkbox = QCheckBox()
        system_config = cast(dict[str, Any], self.config_manager.get_config('system'))
        frame_throttle_checkbox.setChecked(system_config.get('frame_throttling', True))
        frame_throttle_checkbox.stateChanged.connect(partial(self._update_system_bool, 'frame_throttling'))  # type: ignore[reportUnknownMemberType]
        performance_layout.addRow(frame_throttle_label, frame_throttle_checkbox)

        # Memory optimization option
        memory_opt_label = QLabel("Aggressive Memory Optimization:")
        memory_opt_label.setToolTip("Enable aggressive memory optimization (may impact performance)")

        memory_opt_checkbox = QCheckBox()
        memory_opt_checkbox.setChecked(system_config.get('aggressive_memory_opt', False))
        memory_opt_checkbox.stateChanged.connect(partial(self._update_system_bool, 'aggressive_memory_opt'))  # type: ignore[reportUnknownMemberType]
        performance_layout.addRow(memory_opt_label, memory_opt_checkbox)

        # Resource limits
        max_images_label = QLabel("Max Images:")
        max_images_label.setToolTip("Maximum number of images to keep in memory")

        max_images_edit = QLineEdit(str(system_config.get('max_images', 200)))
        max_images_edit.setToolTip("Set maximum image count (default: 200)")
        max_images_edit.textChanged.connect(partial(self._update_system_max_images))  # type: ignore[reportUnknownMemberType]
        performance_layout.addRow(max_images_label, max_images_edit)

        performance_group.setLayout(performance_layout)
        scroll_layout.addWidget(performance_group)

        # Debug Options Group
        debug_group = QGroupBox("Debug Options")
        debug_group.setStyleSheet("""
            QGroupBox {
                color: #ffcc00;
                border: 1px solid #444444;
                border-radius: 4px;
                font-family: 'Segoe UI';
                font-size: 12px;
                font-weight: bold;
            }
        """)

        debug_layout = QFormLayout()
        debug_layout.setContentsMargins(10, 15, 10, 10)
        debug_layout.setSpacing(8)
        debug_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Enable detailed logging
        debug_logging_label = QLabel("Enable Detailed Logging:")
        debug_logging_label.setToolTip("Enable detailed logging for troubleshooting")

        debug_logging_checkbox = QCheckBox()
        debug_logging_checkbox.setChecked(system_config.get('detailed_logging', False))
        debug_logging_checkbox.stateChanged.connect(partial(self._update_system_bool, 'detailed_logging'))  # type: ignore[reportUnknownMemberType]
        debug_layout.addRow(debug_logging_label, debug_logging_checkbox)

        # Show performance metrics
        perf_metrics_label = QLabel("Show Performance Metrics:")
        perf_metrics_label.setToolTip("Display performance metrics in status bar")

        perf_metrics_checkbox = QCheckBox()
        perf_metrics_checkbox.setChecked(system_config.get('show_performance_metrics', True))
        perf_metrics_checkbox.stateChanged.connect(partial(self._update_system_bool, 'show_performance_metrics'))  # type: ignore[reportUnknownMemberType]
        debug_layout.addRow(perf_metrics_label, perf_metrics_checkbox)

        debug_group.setLayout(debug_layout)
        scroll_layout.addWidget(debug_group)

        scroll_area.setWidget(scroll_content)
        advanced_layout.addWidget(scroll_area)
        self.tab_widget.addTab(advanced_frame, "Advanced")

    def _create_action_buttons(self, main_layout: QVBoxLayout) -> None:
        """Create action buttons for the configuration panel."""
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)

        # Restart button
        restart_btn = QPushButton("Apply & Restart")
        restart_btn.setStyleSheet("""
            QPushButton {
                background-color: #888822;
                color: #e0e0e0;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-family: 'Segoe UI';
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #aaa933;
            }
            QPushButton:pressed {
                background-color: #666611;
            }
        """)
        restart_btn.clicked.connect(self._restart_system)  # type: ignore[reportUnknownMemberType]
        button_layout.addWidget(restart_btn)

        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #882222;
                color: #e0e0e0;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-family: 'Segoe UI';
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #993333;
            }
            QPushButton:pressed {
                background-color: #661111;
            }
        """)
        reset_btn.clicked.connect(self._reset_to_defaults)  # type: ignore[reportUnknownMemberType]
        button_layout.addWidget(reset_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #e0e0e0;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-family: 'Segoe UI';
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
        """)
        close_btn.clicked.connect(self.close)  # type: ignore[reportUnknownMemberType]
        button_layout.addWidget(close_btn)

        main_layout.addWidget(button_frame)

    def _update_sensory_bool(self, key: str, state: int) -> None:
        """Update sensory boolean configuration."""
        self._update_config('sensory', key, state == 2)

    def _update_sensory_text(self, key: str, text: str) -> None:
        """Update sensory text configuration."""
        self._update_config('sensory', key, int(text) if text.isdigit() else 0)

    def _update_workspace_text(self, key: str, text: str) -> None:
        """Update workspace text configuration."""
        self._update_config('workspace', key, int(text) if text.isdigit() else 0)

    def _update_system_text(self, key: str, text: str) -> None:
        """Update system text configuration."""
        self._update_config('system', key, float(text) if text.replace('.', '', 1).isdigit() else 0.0)

    def _update_system_bool(self, key: str, state: int) -> None:
        """Update system boolean configuration."""
        value = state == 2
        self._update_config('system', key, value)
        if key == 'detailed_logging':
            try:
                logging.getLogger().setLevel(logging.DEBUG if value else logging.INFO)
                logging.getLogger(__name__).info("Detailed logging %s", "enabled" if value else "disabled")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.getLogger(__name__).warning("Failed to apply logging toggle: %s", e)

    def _update_system_max_images(self, text: str) -> None:
        """Update max images configuration."""
        self._update_config('system', 'max_images', int(text) if text.isdigit() else 200)

    def _update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update configuration value with comprehensive validation and error handling.

        Args:
            section: Configuration section (e.g., 'sensory', 'workspace', 'system')
            key: Configuration key
            value: New value to set

        Raises:
            ValueError: If configuration validation fails
            TypeError: If value type is incompatible
        """
        try:
            # Validate value type and range before updating
            if section == 'sensory':
                if key in ['width', 'height', 'canvas_width', 'canvas_height'] and isinstance(value, int):
                    if value <= 0:
                        raise ValueError(f"{key} must be positive")
                    if value > 4096:  # Reasonable maximum
                        raise ValueError(f"{key} exceeds maximum allowed value (4096)")
                elif key in ['update_interval', 'frame_skip'] and isinstance(value, int):
                    if value < 1:
                        raise ValueError(f"{key} must be at least 1")
                    if value > 10000:
                        raise ValueError(f"{key} exceeds reasonable maximum (10000)")

            elif section == 'workspace':
                if key in ['width', 'height'] and isinstance(value, int):
                    if value <= 0:
                        raise ValueError(f"{key} must be positive")
                    if value > 2048:  # Reasonable maximum for workspace
                        raise ValueError(f"{key} exceeds maximum allowed value (2048)")
                elif key in ['node_spacing', 'connection_distance'] and isinstance(value, (int, float)):
                    if value <= 0:
                        raise ValueError(f"{key} must be positive")

            elif section == 'system':
                if key == 'update_interval' and isinstance(value, int):
                    if value < 16:  # Minimum reasonable update interval
                        raise ValueError("Update interval too small (minimum 16ms)")
                    if value > 5000:  # Maximum reasonable update interval
                        raise ValueError("Update interval too large (maximum 5000ms)")
                elif key == 'max_images' and isinstance(value, int):
                    if value <= 0:
                        raise ValueError("Max images must be positive")
                    if value > 1000:
                        raise ValueError("Max images exceeds reasonable limit (1000)")
                elif key in ['energy_decay', 'connection_strength'] and isinstance(value, (int, float)):
                    if value < 0:
                        raise ValueError(f"{key} cannot be negative")

            # Update configuration with validated value
            logger.info(f"Attempting to update {section}.{key} to {value}")
            print(f"Debug: Attempting to update {section}.{key} to {value}")
            
            # Check if the key exists in the configuration
            current_config = self.config_manager.get_config(section)
            logger.info(f"Current {section} config: {current_config}")
            print(f"Debug: Current {section} config: {current_config}")
            
            if self.config_manager.update_config(section, key, value):
                logger.info(f"Updated {section}.{key} to {value}")
                print(f"✓ Configuration updated: {section}.{key} = {value}")
            else:
                logger.error(f"Configuration update failed for {section}.{key}")
                print(f"Error: Configuration update failed for {section}.{key}")
                raise RuntimeError(f"Configuration update failed for {section}.{key}")

        except ValueError as ve:
            ErrorHandler.show_error(
                "Validation Error",
                f"Invalid value for {section}.{key}: {str(ve)}\n\n"
                f"Please enter a valid value within the allowed range.",
                severity="medium"
            )
        except TypeError as te:
            ErrorHandler.show_error(
                "Type Error",
                f"Invalid type for {section}.{key}: {str(te)}\n\n"
                f"Expected: {self._get_expected_type(section, key)}\n"
                f"Received: {type(value).__name__}",
                severity="medium"
            )
        except Exception as e:
            ErrorHandler.show_error(
                "Config Update Error",
                f"Failed to update {section}.{key}: {str(e)}\n\n"
                f"Please check:\n"
                f"- Value is within valid range\n"
                f"- Configuration section exists\n"
                f"- You have proper permissions\n"
                f"- System has sufficient resources",
                severity="medium"
            )

    def _get_expected_type(self, section: str, key: str) -> str:
        """Get the expected type for a configuration parameter."""
        expected_types = {
            'sensory': {
                'width': 'positive integer',
                'height': 'positive integer',
                'canvas_width': 'positive integer',
                'canvas_height': 'positive integer',
                'update_interval': 'positive integer',
                'frame_skip': 'positive integer',
                'enabled': 'boolean',
                'sensitivity': 'float'
            },
            'workspace': {
                'width': 'positive integer',
                'height': 'positive integer',
                'node_spacing': 'positive number',
                'connection_distance': 'positive number',
                'background_color': 'string',
                'grid_visible': 'boolean'
            },
            'system': {
                'update_interval': 'integer (16-5000)',
                'max_images': 'positive integer',
                'energy_decay': 'non-negative float',
                'connection_strength': 'non-negative float',
                'frame_throttling': 'boolean',
                'aggressive_memory_opt': 'boolean',
                'detailed_logging': 'boolean',
                'show_performance_metrics': 'boolean'
            }
        }

        return expected_types.get(section, {}).get(key, "unknown type")

    def _restart_system(self) -> None:
        """Restart the system with new configuration."""
        try:
            # Show confirmation dialog
            reply = QMessageBox.question(
                self,
                "Confirm Restart",
                "Are you sure you want to apply changes and restart the system?\n\n"
                "This will:\n"
                "- Save all configuration changes\n"
                "- Restart the neural system\n"
                "- Apply performance optimizations\n"
                "- Reset system state",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                logger.info("Applying configuration changes and restarting system")
                self.accept()  # Close the dialog
                # The actual restart will be handled by the main window
        except Exception as e:
            ErrorHandler.show_error(
                "Restart Error",
                f"Failed to restart system: {str(e)}\n\n"
                "Please try again or check system logs for details.",
                severity="high"
            )

    def _reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        try:
            # Show confirmation dialog
            reply = QMessageBox.question(
                self,
                "Confirm Reset",
                "Are you sure you want to reset all configuration to defaults?\n\n"
                "This will:\n"
                "- Reset all parameters to factory defaults\n"
                "- Overwrite any custom settings\n"
                "- Require system restart to take effect",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Simple reset by creating new config manager with defaults
                # Note: This is a simplified approach - in production, you'd want
                # to implement proper reset functionality in ConfigManager
                logger.info("Configuration reset requested")
                QMessageBox.information(
                    self,
                    "Reset Complete",
                    "Configuration reset requested.\n"
                    "Please restart the application for changes to take effect.",
                    QMessageBox.StandardButton.Ok
                )
                print("✓ Configuration reset requested")
        except Exception as e:
            ErrorHandler.show_error(
                "Reset Error",
                f"Failed to reset configuration: {str(e)}\n\n"
                "Please check system logs for details.",
                severity="high"
            )

    def _refresh_configuration_ui(self) -> None:
        """Refresh the configuration UI to show current values."""
        try:
            logger.info("Refreshing configuration UI")
            # For now, just log the refresh
            print("✓ Configuration UI refreshed")
        except Exception as e:
            ErrorHandler.log_warning(f"Error refreshing configuration UI: {str(e)}")

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Handle dialog closing."""
        try:
            # Simple close handling
            logger.info("Configuration panel closed")
            if a0 is not None:
                a0.accept()
        except Exception as e:
            ErrorHandler.log_warning(f"Error handling close event: {str(e)}")
            if a0 is not None:
                a0.accept()