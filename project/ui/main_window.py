import tkinter as tk
import numpy as np
from PIL import Image
from utils.error_handler import ErrorHandler
from utils.config_manager import ConfigManager
from system.state_manager import StateManager
from .resource_manager import UIResourceManager

class MainWindow:
    def __init__(self, config_manager: ConfigManager, state_manager: StateManager):
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.resource_manager = UIResourceManager()
        self.frame_counter = 0

        # Create main window
        self.window = tk.Tk()
        self.window.title('DGL AI Workspace Window')
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
        self.image_id = self.canvas.create_image(0, 0, anchor='nw')

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
        self.sensory_canvas = tk.Canvas(
            self.left_frame,
            width=sensory_config['canvas_width'],
            height=sensory_config['canvas_height'],
            bg='#222222',
            highlightthickness=0
        )
        self.sensory_canvas.pack(pady=(0, 10))
        self.sensory_image_id = self.sensory_canvas.create_image(0, 0, anchor='nw')

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

    def _create_control_buttons(self):
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

    def _create_interval_slider(self):
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
        self.interval_slider.set(self.config_manager.get_config('system', 'update_interval'))
        self.interval_slider.pack(side='left', fill='x', expand=True)

    def _toggle_suspend(self):
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

    def _pulse_energy(self):
        """Pulse energy into the system"""
        # This will be implemented in the neural system
        self.status_var.set("Last pulse: +10 energy")

    def _toggle_sensory(self):
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

    def _open_config_panel(self):
        """Open configuration panel"""
        # This will be implemented in a separate config panel module
        pass

    def _update_interval_changed(self, value):
        """Handle update interval change"""
        try:
            interval = int(float(value))
            self.config_manager.update_config('system', 'update_interval', interval)
            self.status_var.set(f"Update interval set to {interval}ms")
        except Exception as e:
            ErrorHandler.show_error("Config Error", f"Failed to update interval: {str(e)}")

    def _on_resize(self, event):
        """Handle window resize"""
        self.update_workspace_canvas()

    def _on_closing(self):
        """Handle window closing"""
        try:
            self.resource_manager.cleanup()
            self.window.destroy()
        except Exception as e:
            ErrorHandler.show_error("Close Error", f"Error during cleanup: {str(e)}")
            self.window.destroy()

    def update_workspace_canvas(self, workspace_data=None):
        """Update workspace canvas with new data"""
        try:
            if workspace_data is None:
                # Create empty workspace
                workspace_config = self.config_manager.get_config('workspace')
                workspace_data = np.zeros((workspace_config['height'], workspace_config['width']))

            # Convert to image
            arr = np.clip(workspace_data, 0, 244)
            arr_rgb = np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)

            # Get canvas size
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

            # Create and resize image
            if canvas_w > 1 and canvas_h > 1:
                img = Image.fromarray(arr_rgb, mode='RGB').resize(
                    (canvas_w, canvas_h),
                    resample=Image.NEAREST
                )
            else:
                img = Image.fromarray(arr_rgb, mode='RGB')

            # Update canvas
            tk_img = self.resource_manager.create_tk_image(img)
            if tk_img:
                self.canvas.itemconfig(self.image_id, image=tk_img)
        except Exception as e:
            ErrorHandler.show_error("Canvas Error", f"Failed to update workspace: {str(e)}")

    def update_sensory_canvas(self, sensory_data):
        """Update sensory canvas with new data"""
        try:
            # Convert to image
            arr = np.clip(sensory_data, 0, 1)
            arr = (arr * 255).astype(np.uint8)
            arr_rgb = np.repeat(arr[:, :, None], 3, axis=2)

            # Get canvas size
            sensory_config = self.config_manager.get_config('sensory')

            # Create and resize image
            img = Image.fromarray(arr_rgb, mode='RGB').resize(
                (sensory_config['canvas_width'], sensory_config['canvas_height']),
                resample=Image.NEAREST
            )

            # Update canvas
            tk_img = self.resource_manager.create_tk_image(img)
            if tk_img:
                self.sensory_canvas.itemconfig(self.sensory_image_id, image=tk_img)
        except Exception as e:
            ErrorHandler.show_error("Canvas Error", f"Failed to update sensory: {str(e)}")

    def update_metrics_panel(self, metrics):
        """Update metrics panel with new data"""
        try:
            metrics_text = (
                f"Total Energy: {metrics['total_energy']:.2f}\n"
                f"Sensory Nodes: {metrics['sensory_node_count']}\n"
                f"Dynamic Nodes: {metrics['dynamic_node_count']}\n"
                f"Workspace Nodes: {metrics['workspace_node_count']}\n"
                f"Avg Dynamic Energy: {metrics['avg_dynamic_energy']:.2f}\n"
                f"Node Births: {metrics['node_births']} (total {metrics['total_node_births']}) | "
                f"Node Deaths: {metrics['node_deaths']} (total {metrics['total_node_deaths']})\n"
                f"Conn Births: {metrics['conn_births']} (total {metrics['total_conn_births']}) | "
                f"Conn Deaths: {metrics['conn_deaths']} (total {metrics['total_conn_deaths']})\n"
                f"Connections: {metrics['connection_count']}"
            )
            self.metrics_label.config(text=metrics_text)
        except Exception as e:
            ErrorHandler.show_error("Metrics Error", f"Failed to update metrics: {str(e)}")

    def on_state_change(self, state):
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

    def start_system(self, system, capture):
        """Start the system and capture components"""
        self.system = system
        self.capture = capture
        self.frame_counter = 0
        self.window.after(100, self.startup_loop)

    def periodic_update(self):
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

                if self.frame_counter % 2 == 0:
                    self.system.queue_connection_growth()
                if self.frame_counter % 3 == 0:
                    self.system.queue_cull()

                # Update UI
                self.update_workspace_canvas()
                metrics = self.system.get_metrics()
                self.update_metrics_panel(metrics)
                self.state_manager.update_metrics(
                    metrics['total_energy'],
                    metrics['dynamic_node_count'],
                    metrics['connection_count']
                )

            except Exception as e:
                logger.error(f"Error during update: {e}")
                ErrorHandler.show_error("Update Error", f"Error during update: {str(e)}")

        # Schedule next update
        update_interval = self.config_manager.get_config('system', 'update_interval')
        self.window.after(update_interval, self.periodic_update)

    def startup_loop(self):
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

            # Queue connection tasks
            self.frame_counter += 1

            if self.frame_counter % 2 == 0:
                self.system.queue_connection_growth()
            if self.frame_counter % 3 == 0:
                self.system.queue_cull()

            # Continue startup or switch to periodic update
            if self.system.startup_phase:
                self.window.after(1, self.startup_loop)
            else:
                update_interval = self.config_manager.get_config('system', 'update_interval')
                self.window.after(update_interval, self.periodic_update)

        except Exception as e:
            logger.error(f"Error during startup: {e}")
            ErrorHandler.show_error("Startup Error", f"Error during startup: {str(e)}")

    def run(self):
        """Start the main window"""
        self.window.mainloop() 
