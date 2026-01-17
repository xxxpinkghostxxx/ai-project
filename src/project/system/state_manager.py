"""
System State Manager Module.

This module provides system state management functionality for the Energy-Based Neural System,
including state tracking, observer pattern implementation, and system metrics management.
"""

from dataclasses import dataclass
from typing import Any

from project.utils.error_handler import ErrorHandler

@dataclass
class SystemState:
    """Data class to hold system state"""
    suspended: bool = False
    sensory_enabled: bool = True
    last_pulse_time: float = 0.0
    last_update_time: float = 0.0
    total_energy: float = 0.0
    node_count: int = 0
    connection_count: int = 0

class StateManager:
    """System state manager class for tracking and managing system state."""

    def __init__(self) -> None:
        """Initialize StateManager with default system state."""
        self.state = SystemState()
        self._observers: list[Any] = []

    def add_observer(self, observer: Any) -> None:
        """Add an observer to be notified of state changes"""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Any) -> None:
        """Remove an observer"""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self) -> None:
        """Notify all observers of state change"""
        for observer in self._observers:
            try:
                observer.on_state_change(self.state)
            except Exception as e:
                ErrorHandler.log_warning(f"Error notifying observer: {str(e)}")

    def toggle_sensory(self) -> bool | None:
        """Toggle sensory input state"""
        try:
            self.state.sensory_enabled = not self.state.sensory_enabled
            self._notify_observers()
            return self.state.sensory_enabled
        except Exception as e:
            ErrorHandler.show_error("State Error", f"Failed to toggle sensory: {str(e)}")
            return None

    def toggle_suspend(self) -> bool | None:
        """Toggle system suspension state"""
        try:
            self.state.suspended = not self.state.suspended
            self._notify_observers()
            return self.state.suspended
        except Exception as e:
            ErrorHandler.show_error("State Error", f"Failed to toggle suspend: {str(e)}")
            return None

    def update_metrics(self, total_energy: float, node_count: int, connection_count: int) -> None:
        """Update system metrics"""
        try:
            self.state.total_energy = total_energy
            self.state.node_count = node_count
            self.state.connection_count = connection_count
            self._notify_observers()
        except Exception as e:
            ErrorHandler.show_error("State Error", f"Failed to update metrics: {str(e)}")

    def get_state(self) -> SystemState:
        """Get current system state"""
        return self.state

    def reset(self) -> None:
        """Reset system state to defaults"""
        try:
            self.state = SystemState()
            self._notify_observers()
        except Exception as e:
            ErrorHandler.show_error("State Error", f"Failed to reset state: {str(e)}")
