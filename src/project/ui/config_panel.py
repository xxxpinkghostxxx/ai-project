"""
Configuration Panel Module.

This module provides a graphical user interface for configuring the Energy-Based Neural System,
including sensory, workspace, and system parameter configuration.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any
from project.utils.error_handler import ErrorHandler
from project.utils.config_manager import ConfigManager

class ConfigPanel:
    """Configuration panel class for managing system configuration through a GUI."""

    def __init__(self: ConfigPanel, parent: tk.Tk, config_manager: ConfigManager) -> None:
        """Initialize ConfigPanel with parent window and config manager."""
        self.parent = parent
        self.config_manager = config_manager
        self.window: tk.Toplevel | None = None

    def show(self: ConfigPanel) -> None:
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

    def _create_sensory_tab(self: ConfigPanel, notebook: ttk.Notebook) -> None:
        """Create sensory configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Sensory')

        sensory_config = self.config_manager.get_config('sensory')
        if not isinstance(sensory_config, dict):
            return

        config_row = 0
        for key in sensory_config.keys():  # type: ignore
            key_str: str = str(key)  # type: ignore
            ttk.Label(
                frame,
                text=f"{key_str.replace('_', ' ').title()}:"
            ).grid(row=config_row, column=0, sticky='w', padx=5, pady=2)

            value = sensory_config.get(key)  # type: ignore
            if isinstance(value, bool):
                bool_var = tk.BooleanVar(value=value)
                def update_bool() -> None:
                    self._update_config('sensory', key_str, bool_var.get())

                ttk.Checkbutton(
                    frame,
                    variable=bool_var,
                    command=update_bool
                ).grid(row=config_row, column=1, sticky='w')
            else:
                str_var: tk.StringVar = tk.StringVar(value=str(value) if value is not None else "")  # type: ignore
                ttk.Entry(frame, textvariable=str_var).grid(
                    row=config_row,
                    column=1,
                    sticky='ew',
                    padx=5
                )
                def update_str(*_args: Any) -> None:
                    self._update_config('sensory', key_str, int(str_var.get()) if str_var.get().isdigit() else 0)

                str_var.trace_add('write', update_str)
            config_row += 1

    def _create_workspace_tab(self: ConfigPanel, notebook: ttk.Notebook) -> None:
        """Create workspace configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Workspace')

        workspace_config = self.config_manager.get_config('workspace')
        if not isinstance(workspace_config, dict):
            return

        config_row = 0
        for key in workspace_config.keys():  # type: ignore
            key_str: str = str(key)  # type: ignore
            ttk.Label(
                frame,
                text=f"{key_str.replace('_', ' ').title()}:"
            ).grid(row=config_row, column=0, sticky='w', padx=5, pady=2)

            value = workspace_config.get(key)  # type: ignore
            var: tk.StringVar = tk.StringVar(value=str(value) if value is not None else "")  # type: ignore
            ttk.Entry(frame, textvariable=var).grid(
                row=config_row,
                column=1,
                sticky='ew',
                padx=5
            )
            var.trace_add(
                'write',
                lambda *args: self._update_config('workspace', key_str, int(var.get()) if var.get().isdigit() else 0)  # type: ignore
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
        for key in system_config.keys():  # type: ignore
            key_str: str = str(key)  # type: ignore
            ttk.Label(
                frame,
                text=f"{key_str.replace('_', ' ').title()}:"
            ).grid(row=config_row, column=0, sticky='w', padx=5, pady=2)

            value = system_config.get(key)  # type: ignore
            var: tk.StringVar = tk.StringVar(value=str(value) if value is not None else "")  # type: ignore
            ttk.Entry(frame, textvariable=var).grid(
                row=config_row,
                column=1,
                sticky='ew',
                padx=5
            )
            var.trace_add(
                'write',
                lambda *args: self._update_config('system', key_str, float(var.get()) if var.get().replace('.', '', 1).isdigit() else 0.0)  # type: ignore
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
