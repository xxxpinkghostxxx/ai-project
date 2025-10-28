from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, Callable
from project.utils.error_handler import ErrorHandler
from project.utils.config_manager import ConfigManager

class ConfigPanel:
    def __init__(self: ConfigPanel, parent: tk.Tk, config_manager: ConfigManager) -> None:
        self.parent = parent
        self.config_manager = config_manager
        self.window: Optional[tk.Toplevel] = None

    def show(self: ConfigPanel) -> None:
        """Show the configuration panel"""
        try:
            # Create window
            self.window = tk.Toplevel(self.parent)
            if self.window is None:
                return
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
            if self.window is not None:
                self.window.update_idletasks()
                width = self.window.winfo_width()
                height = self.window.winfo_height()
                x = (self.window.winfo_screenwidth() // 2) - (width // 2)
                y = (self.window.winfo_screenheight() // 2) - (height // 2)
                self.window.geometry(f'{width}x{height}+{x}+{y}')

        except Exception as e:
            ErrorHandler.show_error("Config Panel Error", f"Failed to show config panel: {str(e)}")

    def _create_sensory_tab(self: ConfigPanel, notebook: ttk.Notebook) -> None:
        """Create sensory configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Sensory')

        sensory_config = self.config_manager.get_config('sensory')
        if not isinstance(sensory_config, dict):
            return

        config_row = 0
        for key in sensory_config:
            ttk.Label(
                frame,
                text=f"{key.replace('_', ' ').title()}:"
            ).grid(row=config_row, column=0, sticky='w', padx=5, pady=2)

            value = sensory_config.get(key)
            if isinstance(value, bool):
                bool_var = tk.BooleanVar(value=value)
                def update_bool() -> None:
                    self._update_config('sensory', key, bool_var.get())

                ttk.Checkbutton(
                    frame,
                    variable=bool_var,
                    command=update_bool
                ).grid(row=config_row, column=1, sticky='w')
            else:
                str_var: tk.StringVar = tk.StringVar(value=str(value))
                ttk.Entry(frame, textvariable=str_var).grid(
                    row=config_row,
                    column=1,
                    sticky='ew',
                    padx=5
                )
                def update_str(*args: Any) -> None:
                    self._update_config('sensory', key, int(str_var.get()) if str_var.get().isdigit() else 0)

                str_var.trace('w', update_str)
            config_row += 1

    def _create_workspace_tab(self: ConfigPanel, notebook: ttk.Notebook) -> None:
        """Create workspace configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Workspace')

        workspace_config = self.config_manager.get_config('workspace')
        if not isinstance(workspace_config, dict):
            return

        config_row = 0
        for key in workspace_config:
            ttk.Label(
                frame,
                text=f"{key.replace('_', ' ').title()}:"
            ).grid(row=config_row, column=0, sticky='w', padx=5, pady=2)

            value = workspace_config.get(key)
            var: tk.StringVar = tk.StringVar(value=str(value))
            ttk.Entry(frame, textvariable=var).grid(
                row=config_row,
                column=1,
                sticky='ew',
                padx=5
            )
            var.trace(
                'w',
                lambda *args, k=key, v=var: self._update_config('workspace', k, int(v.get()))
            )
            config_row += 1

    def _create_system_tab(self: ConfigPanel, notebook: ttk.Notebook) -> None:
        """Create system configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='System')

        system_config = self.config_manager.get_config('system')
        if not isinstance(system_config, dict):
            return

        config_row = 0
        for key in system_config:
            ttk.Label(
                frame,
                text=f"{key.replace('_', ' ').title()}:"
            ).grid(row=config_row, column=0, sticky='w', padx=5, pady=2)

            value = system_config.get(key)
            var: tk.StringVar = tk.StringVar(value=str(value))
            ttk.Entry(frame, textvariable=var).grid(
                row=config_row,
                column=1,
                sticky='ew',
                padx=5
            )
            var.trace(
                'w',
                lambda *args, k=key, v=var: self._update_config('system', k, float(v.get()))
            )
            config_row += 1

    def _update_config(self: ConfigPanel, section: str, key: str, value: Any) -> None:
        """Update configuration value"""
        try:
            if self.config_manager.update_config(section, key, value):
                ErrorHandler.log_info(f"Updated {section}.{key} to {value}")
        except Exception as e:
            ErrorHandler.show_error(
                "Config Update Error",
                f"Failed to update {section}.{key}: {str(e)}"
            )

    def _restart_system(self: ConfigPanel) -> None:
        """Restart the system with new configuration"""
        try:
            if self.window:
                self.window.destroy()
            # The actual restart will be handled by the main window
        except Exception as e:
            ErrorHandler.show_error("Restart Error", f"Failed to restart system: {str(e)}")
