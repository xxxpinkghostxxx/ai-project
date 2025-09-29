import threading
from typing import Dict, Callable, Any
from collections import defaultdict


class EventBus:
    """
    Thread-safe in-memory event bus for pub-sub pattern.
    Supports emit(event_type: str, data: dict), publish(event_type: str, data: dict) and subscribe(event_type: str, callback: Callable).
    """
    def __init__(self):
        self._subscribers: Dict[str, list[Callable[[str, Dict[str, Any]], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Subscribe a callback to an event type."""
        if not callable(callback):
            raise ValueError("Callback must be callable")
        with self._lock:
            self._subscribers[event_type].append(callback)

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event with data to all subscribers."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dict")
        with self._lock:
            for callback in self._subscribers.get(event_type, []):
                try:
                    callback(event_type, data)
                except Exception:
                    # Silent fail for robustness; could log in production
                    pass

    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event with data to all subscribers."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dict")
        with self._lock:
            for callback in self._subscribers.get(event_type, []):
                try:
                    callback(event_type, data)
                except Exception:
                    # Silent fail for robustness; could log in production
                    pass

    def unsubscribe(self, event_type: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Unsubscribe a callback from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb != callback
                ]
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]


# Singleton instance
_event_bus_instance: EventBus | None = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the singleton EventBus instance."""
    global _event_bus_instance
    with _event_bus_lock:
        if _event_bus_instance is None:
            _event_bus_instance = EventBus()
    return _event_bus_instance






