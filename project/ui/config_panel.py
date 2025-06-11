import tkinter as tk
from tkinter import ttk
from utils.error_handler import ErrorHandler
from utils.config_manager import ConfigManager

class ConfigPanel:
    def __init__(self, parent, config_manager: ConfigManager):
        self.parent = parent
        self.config_manager = config_manager
        self.window = None

    def show(self):
        """Show the configuration panel"""
        try:
            # Create window
            self.window = tk.Toplevel(self.parent)
            self.window.title('Config Panel')
            self.window.geometry('500x800')

            # Create notebook for tabbed interface
            notebook = ttk.Notebook(self.window)
            notebook.pack(fill='both', expand=True, padx=5, pady=5)

            # Create tabs
            self._create_sensory_tab(notebook)
            self._create_workspace_tab(notebook)
            self._create_system_tab(notebook)

            # Create restart button
            restart_btn = ttk.Button(
                self.window,
                text="Restart System",
                command=self._restart_system
            )
            restart_btn.pack(pady=10)

            # Center window
            self.window.update_idletasks()
            width = self.window.winfo_width()
            height = self.window.winfo_height()
            x = (self.window.winfo_screenwidth() // 2) - (width // 2)
            y = (self.window.winfo_screenheight() // 2) - (height // 2)
            self.window.geometry(f'{width}x{height}+{x}+{y}')

        except Exception as e:
            ErrorHandler.show_error("Config Panel Error", f"Failed to show config panel: {str(e)}")

    def _create_sensory_tab(self, notebook):
        """Create sensory configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Sensory')

        row = 0
        for key in self.config_manager.get_config('sensory'):
            ttk.Label(
                frame,
                text=f"{key.replace('_', ' ').title()}:"
            ).grid(row=row, column=0, sticky='w', padx=5, pady=2)

            if isinstance(self.config_manager.get_config('sensory', key), bool):
                var = tk.BooleanVar(
                    value=self.config_manager.get_config('sensory', key)
                )
                ttk.Checkbutton(
                    frame,
                    variable=var,
                    command=lambda k=key, v=var: self._update_config('sensory', k, v.get())
                ).grid(row=row, column=1, sticky='w')
            else:
                var = tk.StringVar(
                    value=str(self.config_manager.get_config('sensory', key))
                )
                ttk.Entry(frame, textvariable=var).grid(
                    row=row,
                    column=1,
                    sticky='ew',
                    padx=5
                )
                var.trace(
                    'w',
                    lambda *args, k=key, v=var: self._update_config('sensory', k, int(v.get()))
                )
            row += 1

    def _create_workspace_tab(self, notebook):
        """Create workspace configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Workspace')

        row = 0
        for key in self.config_manager.get_config('workspace'):
            ttk.Label(
                frame,
                text=f"{key.replace('_', ' ').title()}:"
            ).grid(row=row, column=0, sticky='w', padx=5, pady=2)

            var = tk.StringVar(
                value=str(self.config_manager.get_config('workspace', key))
            )
            ttk.Entry(frame, textvariable=var).grid(
                row=row,
                column=1,
                sticky='ew',
                padx=5
            )
            var.trace(
                'w',
                lambda *args, k=key, v=var: self._update_config('workspace', k, int(v.get()))
            )
            row += 1

    def _create_system_tab(self, notebook):
        """Create system configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='System')

        row = 0
        for key in self.config_manager.get_config('system'):
            ttk.Label(
                frame,
                text=f"{key.replace('_', ' ').title()}:"
            ).grid(row=row, column=0, sticky='w', padx=5, pady=2)

            var = tk.StringVar(
                value=str(self.config_manager.get_config('system', key))
            )
            ttk.Entry(frame, textvariable=var).grid(
                row=row,
                column=1,
                sticky='ew',
                padx=5
            )
            var.trace(
                'w',
                lambda *args, k=key, v=var: self._update_config('system', k, float(v.get()))
            )
            row += 1

    def _update_config(self, section, key, value):
        """Update configuration value"""
        try:
            if self.config_manager.update_config(section, key, value):
                ErrorHandler.log_info(f"Updated {section}.{key} to {value}")
        except Exception as e:
            ErrorHandler.show_error(
                "Config Update Error",
                f"Failed to update {section}.{key}: {str(e)}"
            )

    def _restart_system(self):
        """Restart the system with new configuration"""
        try:
            if self.window:
                self.window.destroy()
            # The actual restart will be handled by the main window
        except Exception as e:
            ErrorHandler.show_error("Restart Error", f"Failed to restart system: {str(e)}") 