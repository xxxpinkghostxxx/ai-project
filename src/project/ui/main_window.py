"""
Main Window Module.

This module provides the main graphical user interface for the Energy-Based Neural System,
including workspace visualization, sensory input display, system controls, and metrics monitoring.
"""

import logging
from typing import Any
from numpy.typing import NDArray
import tkinter as tk
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from datetime import datetime

logger = logging.getLogger(__name__)
from project.utils.error_handler import ErrorHandler
from project.utils.config_manager import ConfigManager
from project.system.state_manager import StateManager
from project.ui.resource_manager import UIResourceManager
from project.utils.shutdown_utils import ShutdownDetector

class MainWindow:
    """Main application window class for the neural system interface."""

    def __init__(self, config_manager: ConfigManager, state_manager: StateManager) -> None:
        """Initialize MainWindow with configuration and state managers."""
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.resource_manager = UIResourceManager(
            max_images=100,
            max_windows=10,
            max_memory_mb=512,
            enable_monitoring=True
        )
        self.frame_counter = 0
        self._resource_stats_timer: str | None = None

        # Create main window
        self.window = tk.Tk()
        self.window.title('PyTorch Geometric AI Workspace Window')
        self.window.configure(bg='#222222')

        # Create main frame
        self.main_frame = tk.Frame(self.window, bg='#222222')
        self.main_frame.pack(fill='both', expand=True)

        # Create left frame (workspace and sensory)
        self.left_frame = tk.Frame(self.main_frame, bg='#222222')
        self.left_frame.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)

        # Create workspace canvas
        self.canvas = tk.Canvas(self.left_frame, bg='#181818', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, pady=(0, 5))
        self.image_id = self.canvas.create_image(0, 0, anchor='nw')  # type: ignore

        # Create metrics panel
        self.metrics_label = tk.Label(
            self.left_frame,
            text="",
            fg='#e0e0e0',
            bg='#222222',
            font=('Consolas', 11, 'bold'),
            justify='left'
        )
        self.metrics_label.pack(fill='x', pady=(0, 5))

        # Create sensory canvas
        sensory_config = self.config_manager.get_config('sensory')
        if sensory_config is None:
            raise ValueError("Sensory configuration not found")
        self.sensory_canvas = tk.Canvas(
            self.left_frame,
            width=sensory_config['canvas_width'],
            height=sensory_config['canvas_height'],
            bg='#222222',
            highlightthickness=0
        )
        self.sensory_canvas.pack(pady=(0, 10))
        self.sensory_image_id = self.sensory_canvas.create_image(0, 0, anchor='nw')  # type: ignore

        # Create right frame (controls)
        self.right_frame = tk.Frame(self.main_frame, bg='#222222')
        self.right_frame.pack(side='right', fill='y', padx=10, pady=10)

        # Create controls frame
        self.controls_frame = tk.Frame(self.right_frame, bg='#222222')
        self.controls_frame.pack(fill='y', pady=(0, 10))

        # Create status bar
        self.status_var = tk.StringVar(value="Running")
        self.status_bar = tk.Label(
            self.right_frame,
            textvariable=self.status_var,
            fg='#bbbbbb',
            bg='#181818',
            anchor='w',
            font=('Consolas', 10)
        )
        self.status_bar.pack(fill='x', side='bottom', pady=(10, 0))

        # Create control buttons
        self._create_control_buttons()

        # Create update interval slider
        self._create_interval_slider()

        # Register window with resource manager
        self.resource_manager.register_window(self.window)

        # Register event handlers
        self.canvas.bind('<Configure>', self._on_resize)
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Register as state observer
        self.state_manager.add_observer(self)

        # Start resource monitoring
        self._start_resource_monitoring()

    def _start_resource_monitoring(self) -> None:
        """Start resource monitoring and statistics display."""
        self._update_resource_stats_display()
        # Update every 30 seconds
        self._resource_stats_timer = self.window.after(30000, self._start_resource_monitoring)

    def _update_resource_stats_display(self) -> None:
        """Update resource statistics display."""
        try:
            if hasattr(self, 'resource_manager'):
                stats = self.resource_manager.get_resource_statistics()

                # Add resource stats to status if we have them
                if 'error' not in stats:
                    memory_info = f" | Images: {stats.get('images_count', 0)}"
                    current_status = self.status_var.get()

                    # Only add if not already there to avoid duplicates
                    if "Images:" not in current_status:
                        new_status = f"{current_status}{memory_info}"
                        self.status_var.set(new_status)

                logger.debug(f"Resource statistics: {stats}")
        except Exception as e:
            logger.warning(f"Error updating resource stats display: {e}")

    def get_resource_health_report(self) -> dict[str, str | dict[str, Any]]:
        """Get comprehensive resource health report."""
        try:
            system_health: dict[str, Any] = {
                'monitoring_active': self._resource_stats_timer is not None,
                'frame_counter': self.frame_counter
            }

            # Add memory info if psutil is available
            try:
                import psutil
                process = psutil.Process()
                system_health.update({
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent()
                })
            except ImportError:
                system_health['memory_info'] = 'psutil not available'

            report: dict[str, str | dict[str, Any]] = {
                'timestamp': datetime.now().isoformat(),
                'resource_manager_stats': self.resource_manager.get_resource_statistics(),
                'system_health': system_health
            }

            return report
        except Exception as e:
            return {'error': f'Failed to generate health report: {str(e)}', 'timestamp': datetime.now().isoformat()}

    def _create_control_buttons(self) -> None:
        """Create control buttons"""
        # Suspend button
        self.suspend_button = tk.Button(
            self.controls_frame,
            text="Drain & Suspend",
            bg='#882222',
            fg='#e0e0e0',
            activebackground='#aa3333',
            activeforeground='#ffffff',
            relief='raised',
            width=22,
            command=self._toggle_suspend
        )
        self.suspend_button.pack(fill='x', padx=4, pady=6)

        # Pulse button
        self.pulse_button = tk.Button(
            self.controls_frame,
            text="Pulse +10 Energy",
            bg='#225577',
            fg='#e0e0e0',
            activebackground='#3377aa',
            activeforeground='#ffffff',
            relief='raised',
            width=22,
            command=self._pulse_energy
        )
        self.pulse_button.pack(fill='x', padx=4, pady=6)

        # Sensory button
        self.sensory_button = tk.Button(
            self.controls_frame,
            text="Disable Sensory Input",
            bg='#228822',
            fg='#e0e0e0',
            activebackground='#33aa33',
            activeforeground='#ffffff',
            relief='raised',
            width=22,
            command=self._toggle_sensory
        )
        self.sensory_button.pack(fill='x', padx=4, pady=6)

        # Config button
        self.config_button = tk.Button(
            self.controls_frame,
            text="Config Panel",
            bg='#888822',
            fg='#e0e0e0',
            activebackground='#aaa933',
            activeforeground='#ffffff',
            relief='raised',
            width=22,
            command=self._open_config_panel
        )
        self.config_button.pack(fill='x', padx=4, pady=6)

    def _create_interval_slider(self) -> None:
        """Create update interval slider"""
        interval_frame = tk.Frame(self.right_frame, bg='#222222')
        interval_frame.pack(fill='x', padx=4, pady=6)

        tk.Label(
            interval_frame,
            text="Update Interval (ms):",
            bg='#222222',
            fg='#e0e0e0'
        ).pack(side='left')

        self.interval_slider = tk.Scale(
            interval_frame,
            from_=16,
            to=1000,
            orient='horizontal',
            command=self._update_interval_changed,
            bg='#444444',
            fg='#e0e0e0',
            troughcolor='#666666',
            highlightthickness=0
        )
        self.interval_slider.set(self.config_manager.get_config('system', 'update_interval'))  # type: ignore
        self.interval_slider.pack(side='left', fill='x', expand=True)

    def _toggle_suspend(self) -> None:
        """Toggle system suspension"""
        if self.state_manager.toggle_suspend():
            self.suspend_button.config(
                text="Resume System",
                bg='#225522',
                command=self._toggle_suspend
            )
            self.status_var.set("System suspended and drained.")
        else:
            self.suspend_button.config(
                text="Drain & Suspend",
                bg='#882222',
                command=self._toggle_suspend
            )
            self.status_var.set("System resumed.")

    def _pulse_energy(self) -> None:
        """Pulse energy into the system"""
        # This will be implemented in the neural system
        self.status_var.set("Last pulse: +10 energy")

    def _toggle_sensory(self) -> None:
        """Toggle sensory input"""
        if self.state_manager.toggle_sensory():
            self.sensory_button.config(
                text="Disable Sensory Input",
                bg='#228822'
            )
            self.status_var.set("Sensory input enabled.")
        else:
            self.sensory_button.config(
                text="Enable Sensory Input",
                bg='#882222'
            )
            self.status_var.set("Sensory input disabled.")

    def _open_config_panel(self) -> None:
        """Open configuration panel"""

    def _update_interval_changed(self, value: str) -> None:
        """Handle update interval change"""
        try:
            interval = int(float(value))
            self.config_manager.update_config('system', 'update_interval', interval)
            self.status_var.set(f"Update interval set to {interval}ms")
        except Exception as e:
            ErrorHandler.show_error("Config Error", f"Failed to update interval: {str(e)}")

    def _on_resize(self, event: tk.Event) -> None:
        """Handle window resize"""
        self.update_workspace_canvas()

    def _on_closing(self) -> None:
        """Handle window closing"""
        try:
            # Use shutdown detector to safely cleanup
            ShutdownDetector.safe_cleanup(self.safe_window_cleanup, "MainWindow cleanup")
        except Exception as e:
            ErrorHandler.show_error("Close Error", f"Error during cleanup: {str(e)}")

    def safe_window_cleanup(self) -> None:
        """Safe cleanup method that can be called during shutdown"""
        try:
            # Check if window is still valid before cleanup
            if hasattr(self.window, 'winfo_exists') and self.window.winfo_exists():
                self.resource_manager.cleanup()
                self.window.destroy()
            else:
                ErrorHandler.log_warning("Window already destroyed during cleanup")
        except Exception as e:
            ErrorHandler.log_warning(f"Error during window cleanup: {str(e)}")
            # Only destroy window if it still exists
            if hasattr(self.window, 'winfo_exists') and self.window.winfo_exists():
                self.window.destroy()

    def update_workspace_canvas(self, workspace_data: NDArray[Any] | None = None) -> None:
        """Update workspace canvas with new data"""
        try:
            if workspace_data is None:
                # Create empty workspace
                workspace_config = self.config_manager.get_config('workspace')
                if workspace_config is None:
                    raise ValueError("Workspace configuration not found")
                workspace_data = np.zeros((workspace_config['height'], workspace_config['width']))

            # Convert to image
            arr = np.clip(workspace_data, 0, 244)
            arr_rgb = np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)

            # Get canvas size
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

            # Create and resize image
            if canvas_w > 0 and canvas_h > 0:
                img = Image.fromarray(arr_rgb, mode='RGB').resize(
                    (canvas_w, canvas_h),
                    resample=Resampling.NEAREST
                )
            else:
                img = Image.fromarray(arr_rgb, mode='RGB')

            # Update canvas
            tk_img = self.resource_manager.create_tk_image(img)
            if tk_img:
                self.canvas.itemconfig(self.image_id, image=tk_img)  # type: ignore
        except Exception as e:
            ErrorHandler.show_error("Canvas Error", f"Failed to update workspace: {str(e)}")

    def update_sensory_canvas(self, sensory_data: NDArray[Any]) -> None:
        """Update sensory canvas with new data"""
        try:
            # Convert to image
            arr = np.clip(sensory_data, 0, 1)
            arr = (arr * 255).astype(np.uint8)
            arr_rgb = np.repeat(arr[:, :, None], 3, axis=2)

            # Get canvas size
            sensory_config = self.config_manager.get_config('sensory')
            if sensory_config is None:
                raise ValueError("Sensory configuration not found")

            # Create and resize image
            img = Image.fromarray(arr_rgb, mode='RGB').resize(
                (sensory_config['canvas_width'], sensory_config['canvas_height']),
                resample=Resampling.NEAREST
            )

            # Update canvas
            tk_img = self.resource_manager.create_tk_image(img)
            if tk_img:
                self.sensory_canvas.itemconfig(self.sensory_image_id, image=tk_img)  # type: ignore
        except Exception as e:
            ErrorHandler.show_error("Canvas Error", f"Failed to update sensory: {str(e)}")

    def update_metrics_panel(self, metrics: dict[str, Any] | None) -> None:
        """Update metrics panel with new data"""
        try:
            if metrics is None:
                self.metrics_label.config(text="Metrics not available")
                return
            metrics_text = (
                f"Total Energy: {metrics.get('total_energy', 0):.2f}\n"
                f"Sensory Nodes: {metrics.get('sensory_node_count', 0)}\n"
                f"Dynamic Nodes: {metrics.get('dynamic_node_count', 0)}\n"
                f"Workspace Nodes: {metrics.get('workspace_node_count', 0)}\n"
                f"Avg Dynamic Energy: {metrics.get('avg_dynamic_energy', 0):.2f}\n"
                f"Node Births: {metrics.get('node_births', 0)} (total {metrics.get('total_node_births', 0)}) | "
                f"Node Deaths: {metrics.get('node_deaths', 0)} (total {metrics.get('total_node_deaths', 0)})\n"
                f"Conn Births: {metrics.get('conn_births', 0)} (total {metrics.get('total_conn_births', 0)}) | "
                f"Conn Deaths: {metrics.get('conn_deaths', 0)} (total {metrics.get('total_conn_deaths', 0)})\n"
                f"Connections: {metrics.get('connection_count', 0)}"
            )
            self.metrics_label.config(text=metrics_text)
        except Exception as e:
            ErrorHandler.show_error("Metrics Error", f"Failed to update metrics: {str(e)}")

    def on_state_change(self, state: Any) -> None:
        """Handle state changes"""
        try:
            # Update UI based on state
            if state.suspended:
                self.suspend_button.config(
                    text="Resume System",
                    bg='#225522'
                )
            else:
                self.suspend_button.config(
                    text="Drain & Suspend",
                    bg='#882222'
                )

            if state.sensory_enabled:
                self.sensory_button.config(
                    text="Disable Sensory Input",
                    bg='#228822'
                )
            else:
                self.sensory_button.config(
                    text="Enable Sensory Input",
                    bg='#882222'
                )
        except Exception as e:
            ErrorHandler.show_error("State Error", f"Failed to update UI state: {str(e)}")

    def start_system(self, system: Any, capture: Any) -> None:
        """Start the system and capture components"""
        self.system = system
        self.capture = capture
        self.frame_counter = 0
        self.window.after(100, self.startup_loop)

    def periodic_update(self) -> None:
        """Periodic update function"""
        if not self.state_manager.get_state().suspended:
            try:
                # Sensory update
                if self.state_manager.get_state().sensory_enabled:
                    frame = self.capture.get_latest()
                    if frame is not None:
                        sensory_input = frame.astype(np.float32) / 255.0
                        self.update_sensory_canvas(sensory_input)
                        self.system.update_sensory_nodes(sensory_input)
                    else:
                        logger.warning("Received null frame from screen capture")

                # System update
                self.system.update()
                self.system.apply_connection_worker_results()

                # Queue connection tasks
                self.frame_counter += 1

                if not self.frame_counter % 2:
                    self.system.queue_connection_growth()
                if not self.frame_counter % 3:
                    self.system.queue_cull()

                # Update UI
                self.update_workspace_canvas()
                metrics = self.system.get_metrics()
                self.update_metrics_panel(metrics)
                if metrics is not None:
                    self.state_manager.update_metrics(
                        metrics.get('total_energy', 0),
                        metrics.get('dynamic_node_count', 0),
                        metrics.get('connection_count', 0)
                    )

            except Exception as e:
                logger.error("Error during update: %s", e)
                ErrorHandler.show_error("Update Error", "Error during update: %s" % str(e))

        # Schedule next update
        update_interval = self.config_manager.get_config('system', 'update_interval')
        self.window.after(update_interval, self.periodic_update)

    def startup_loop(self) -> None:
        """Startup loop function"""
        try:
            # Set batch size for startup
            self.system.startup_batch_size = 500
            self.system.update()

            # Update UI
            self.update_workspace_canvas()
            metrics = self.system.get_metrics()
            self.update_metrics_panel(metrics)

            # Apply connection worker results
            self.system.apply_connection_worker_results()

            # Update state manager with metrics if available
            if metrics is not None:
                self.state_manager.update_metrics(
                    metrics.get('total_energy', 0),
                    metrics.get('dynamic_node_count', 0),
                    metrics.get('connection_count', 0)
                )

            # Queue connection tasks
            self.frame_counter += 1

            if not self.frame_counter % 2:
                self.system.queue_connection_growth()
            if not self.frame_counter % 3:
                self.system.queue_cull()

            # Continue startup or switch to periodic update
            if self.system.startup_phase:
                self.window.after(1, self.startup_loop)
            else:
                update_interval = self.config_manager.get_config('system', 'update_interval')
                self.window.after(update_interval, self.periodic_update)

        except Exception as e:
            logger.error("Error during startup: %s", e)
            ErrorHandler.show_error("Startup Error", "Error during startup: %s" % str(e))

    def run(self) -> None:
        """Start the main window"""
        self.window.mainloop()
