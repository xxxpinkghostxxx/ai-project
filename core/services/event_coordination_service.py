"""
EventCoordinationService implementation - Event-driven communication service.

This module provides the concrete implementation of IEventCoordinator,
handling publish-subscribe event communication for loose coupling between
neural simulation services.
"""

import threading
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict, deque
from datetime import datetime

from ..interfaces.event_coordinator import IEventCoordinator, Event


class EventCoordinationService(IEventCoordinator):
    """
    Concrete implementation of IEventCoordinator.

    This service provides publish-subscribe event communication,
    enabling loose coupling between services in the neural simulation system.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._event_history: deque = deque(maxlen=10000)  # Keep last 10k events
        self._subscription_counter = 0

    def publish(self, event_type: str, data: Any, source: Optional[str] = None) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event being published
            data: Event data payload
            source: Source service/component (optional)
        """
        event = Event(event_type, data, source or "unknown")

        with self._lock:
            # Store in history
            self._event_history.append(event)

            # Notify subscribers
            if event_type in self._subscribers:
                for subscriber in self._subscribers[event_type][:]:  # Copy to avoid modification during iteration
                    try:
                        subscriber["handler"](event)
                    except Exception as e:
                        print(f"Error in event handler for {event_type}: {e}")
                        # Remove broken subscribers
                        if subscriber in self._subscribers[event_type]:
                            self._subscribers[event_type].remove(subscriber)

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Event handler function

        Returns:
            str: Subscription ID for unsubscribing
        """
        with self._lock:
            self._subscription_counter += 1
            subscription_id = f"{event_type}_{self._subscription_counter}"

            subscriber = {
                "id": subscription_id,
                "handler": handler,
                "event_type": event_type
            }

            self._subscribers[event_type].append(subscriber)
            return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from an event.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            bool: True if subscription was removed
        """
        with self._lock:
            for event_type, subscribers in self._subscribers.items():
                for subscriber in subscribers:
                    if subscriber["id"] == subscription_id:
                        subscribers.remove(subscriber)
                        return True
            return False

    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """
        Get historical events.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return

        Returns:
            List of historical events
        """
        with self._lock:
            if event_type:
                # Filter by event type
                filtered_events = [event for event in self._event_history if event.event_type == event_type]
                return list(filtered_events)[-limit:]
            else:
                return list(self._event_history)[-limit:]

    def process_events(self) -> None:
        """
        Process any pending events in the event system.

        In the current synchronous implementation, events are processed
        immediately when published, so this method is a no-op.
        """
        # Events are processed synchronously on publish, so no pending events
        pass

    def get_subscription_stats(self) -> Dict[str, Any]:
        """
        Get statistics about event subscriptions.

        Returns:
            Dict with subscription statistics
        """
        with self._lock:
            stats = {}
            total_subscriptions = 0

            for event_type, subscribers in self._subscribers.items():
                stats[event_type] = len(subscribers)
                total_subscriptions += len(subscribers)

            stats["total_subscriptions"] = total_subscriptions
            stats["event_types"] = len(self._subscribers)
            stats["total_events"] = len(self._event_history)

            return stats

    def clear_event_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._event_history.clear()

    def get_active_event_types(self) -> List[str]:
        """
        Get list of event types that have active subscriptions.

        Returns:
            List of active event types
        """
        with self._lock:
            return list(self._subscribers.keys())

    def publish_system_status(self, status_data: Dict[str, Any]) -> None:
        """
        Publish system status update event.

        Args:
            status_data: System status information
        """
        self.publish("system_status", status_data, "system")

    def publish_simulation_step(self, step_data: Dict[str, Any]) -> None:
        """
        Publish simulation step completion event.

        Args:
            step_data: Simulation step information
        """
        self.publish("simulation_step", step_data, "simulation")

    def publish_neural_activity(self, activity_data: Dict[str, Any]) -> None:
        """
        Publish neural activity event.

        Args:
            activity_data: Neural activity information
        """
        self.publish("neural_activity", activity_data, "neural")

    def publish_energy_update(self, energy_data: Dict[str, Any]) -> None:
        """
        Publish energy state update event.

        Args:
            energy_data: Energy state information
        """
        self.publish("energy_update", energy_data, "energy")

    def publish_learning_event(self, learning_data: Dict[str, Any]) -> None:
        """
        Publish learning event.

        Args:
            learning_data: Learning information
        """
        self.publish("learning_event", learning_data, "learning")

    def publish_error_event(self, error_data: Dict[str, Any]) -> None:
        """
        Publish error event.

        Args:
            error_data: Error information
        """
        self.publish("error", error_data, "system")

    def cleanup(self) -> None:
        """Clean up resources."""
        with self._lock:
            self._subscribers.clear()
            self._event_history.clear()