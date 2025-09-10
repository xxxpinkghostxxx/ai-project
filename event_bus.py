"""
event_bus.py

Event bus system for decoupled communication between components.
Implements publish-subscribe pattern to reduce tight coupling.
"""

import time
import threading
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
from logging_utils import log_step
from interfaces import IEventBus, SimulationEvent, EventType


@dataclass
class EventSubscription:
    """Represents an event subscription."""
    callback: Callable
    priority: int = 0  # Higher priority callbacks are called first
    filter_func: Optional[Callable] = None  # Optional filter function
    created_at: float = 0.0


class EventBus(IEventBus):
    """
    Event bus implementation for decoupled communication.
    
    Provides publish-subscribe functionality to reduce tight coupling
    between system components.
    """
    
    def __init__(self, max_subscribers_per_event: int = 100):
        """
        Initialize the event bus.
        
        Args:
            max_subscribers_per_event: Maximum subscribers per event type
        """
        self.subscribers: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.max_subscribers_per_event = max_subscribers_per_event
        self._lock = threading.RLock()
        self.event_history: List[SimulationEvent] = []
        self.max_history_size = 1000
        
        log_step("EventBus initialized", max_subscribers=max_subscribers_per_event)
    
    def subscribe(self, event_type: str, callback: Callable, 
                 priority: int = 0, filter_func: Optional[Callable] = None) -> bool:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is published
            priority: Priority for callback execution (higher = first)
            filter_func: Optional function to filter events
            
        Returns:
            True if subscription successful, False otherwise
        """
        with self._lock:
            if len(self.subscribers[event_type]) >= self.max_subscribers_per_event:
                log_step("Event subscription failed - too many subscribers", 
                        event_type=event_type, 
                        current_count=len(self.subscribers[event_type]))
                return False
            
            subscription = EventSubscription(
                callback=callback,
                priority=priority,
                filter_func=filter_func,
                created_at=time.time()
            )
            
            # Insert subscription in priority order
            subscribers = self.subscribers[event_type]
            inserted = False
            for i, existing_sub in enumerate(subscribers):
                if priority > existing_sub.priority:
                    subscribers.insert(i, subscription)
                    inserted = True
                    break
            
            if not inserted:
                subscribers.append(subscription)
            
            log_step("Event subscription added", 
                    event_type=event_type, 
                    priority=priority,
                    total_subscribers=len(subscribers))
            return True
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        with self._lock:
            subscribers = self.subscribers[event_type]
            for i, subscription in enumerate(subscribers):
                if subscription.callback == callback:
                    subscribers.pop(i)
                    log_step("Event subscription removed", 
                            event_type=event_type,
                            remaining_subscribers=len(subscribers))
                    return True
            
            log_step("Event subscription not found", event_type=event_type)
            return False
    
    def publish(self, event_type: str, data: Any, source: str = "unknown") -> int:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: Type of event to publish
            data: Event data
            source: Source of the event
            
        Returns:
            Number of subscribers that received the event
        """
        with self._lock:
            # Create event object
            event = SimulationEvent(
                event_type=event_type,
                timestamp=time.time(),
                data=data,
                source=source
            )
            
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                self.event_history.pop(0)
            
            # Get subscribers for this event type
            subscribers = self.subscribers[event_type]
            if not subscribers:
                log_step("No subscribers for event", event_type=event_type)
                return 0
            
            # Execute callbacks
            executed_count = 0
            for subscription in subscribers:
                try:
                    # Apply filter if present
                    if subscription.filter_func is not None:
                        if not subscription.filter_func(event):
                            continue
                    
                    # Execute callback
                    subscription.callback(event)
                    executed_count += 1
                    
                except Exception as e:
                    log_step("Event callback error", 
                            event_type=event_type,
                            error=str(e),
                            callback=str(subscription.callback))
            
            log_step("Event published", 
                    event_type=event_type,
                    subscribers_notified=executed_count,
                    total_subscribers=len(subscribers))
            
            return executed_count
    
    def clear_subscribers(self, event_type: str) -> int:
        """
        Clear all subscribers for an event type.
        
        Args:
            event_type: Type of event to clear subscribers for
            
        Returns:
            Number of subscribers removed
        """
        with self._lock:
            count = len(self.subscribers[event_type])
            self.subscribers[event_type].clear()
            log_step("Event subscribers cleared", 
                    event_type=event_type,
                    removed_count=count)
            return count
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type."""
        with self._lock:
            return len(self.subscribers[event_type])
    
    def get_all_event_types(self) -> List[str]:
        """Get all event types with subscribers."""
        with self._lock:
            return list(self.subscribers.keys())
    
    def get_event_history(self, event_type: Optional[str] = None, 
                         limit: int = 100) -> List[SimulationEvent]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        with self._lock:
            if event_type is None:
                events = self.event_history
            else:
                events = [e for e in self.event_history if e.event_type == event_type]
            
            return events[-limit:] if limit > 0 else events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            total_subscribers = sum(len(subs) for subs in self.subscribers.values())
            total_events = len(self.event_history)
            
            return {
                'total_event_types': len(self.subscribers),
                'total_subscribers': total_subscribers,
                'total_events_published': total_events,
                'subscribers_per_event': {
                    event_type: len(subs) 
                    for event_type, subs in self.subscribers.items()
                },
                'recent_event_types': [
                    event.event_type for event in self.event_history[-10:]
                ]
            }
    
    def cleanup(self) -> None:
        """Clean up event bus resources."""
        with self._lock:
            self.subscribers.clear()
            self.event_history.clear()
            log_step("EventBus cleaned up")


class EventFilter:
    """Utility class for creating event filters."""
    
    @staticmethod
    def by_source(source: str) -> Callable:
        """Create filter for events from specific source."""
        def filter_func(event: SimulationEvent) -> bool:
            return event.source == source
        return filter_func
    
    @staticmethod
    def by_data_key(key: str, value: Any) -> Callable:
        """Create filter for events with specific data key-value pair."""
        def filter_func(event: SimulationEvent) -> bool:
            return event.data.get(key) == value
        return filter_func
    
    @staticmethod
    def by_timestamp_range(start_time: float, end_time: float) -> Callable:
        """Create filter for events within timestamp range."""
        def filter_func(event: SimulationEvent) -> bool:
            return start_time <= event.timestamp <= end_time
        return filter_func
    
    @staticmethod
    def by_custom_condition(condition_func: Callable[[SimulationEvent], bool]) -> Callable:
        """Create filter with custom condition function."""
        return condition_func


class EventBusManager:
    """Manages multiple event buses for different domains."""
    
    def __init__(self):
        """Initialize event bus manager."""
        self.buses: Dict[str, EventBus] = {}
        self._lock = threading.RLock()
        log_step("EventBusManager initialized")
    
    def get_bus(self, domain: str = "default") -> EventBus:
        """
        Get event bus for a specific domain.
        
        Args:
            domain: Domain name for the event bus
            
        Returns:
            Event bus for the domain
        """
        with self._lock:
            if domain not in self.buses:
                self.buses[domain] = EventBus()
                log_step("Event bus created for domain", domain=domain)
            
            return self.buses[domain]
    
    def remove_bus(self, domain: str) -> bool:
        """
        Remove event bus for a domain.
        
        Args:
            domain: Domain name to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if domain in self.buses:
                self.buses[domain].cleanup()
                del self.buses[domain]
                log_step("Event bus removed for domain", domain=domain)
                return True
            return False
    
    def get_all_domains(self) -> List[str]:
        """Get all domain names."""
        with self._lock:
            return list(self.buses.keys())
    
    def cleanup_all(self) -> None:
        """Clean up all event buses."""
        with self._lock:
            for bus in self.buses.values():
                bus.cleanup()
            self.buses.clear()
            log_step("All event buses cleaned up")


# Global event bus manager instance
_event_bus_manager: Optional[EventBusManager] = None
_event_bus_manager_lock = threading.Lock()


def get_event_bus(domain: str = "default") -> EventBus:
    """
    Get global event bus instance.
    
    Args:
        domain: Domain name for the event bus
        
    Returns:
        Event bus instance
    """
    global _event_bus_manager
    
    if _event_bus_manager is None:
        with _event_bus_manager_lock:
            if _event_bus_manager is None:
                _event_bus_manager = EventBusManager()
    
    return _event_bus_manager.get_bus(domain)


def cleanup_event_buses() -> None:
    """Clean up all global event buses."""
    global _event_bus_manager
    
    if _event_bus_manager is not None:
        _event_bus_manager.cleanup_all()
        _event_bus_manager = None


# Convenience functions for common event operations
def publish_simulation_event(event_type: str, data: Any, source: str = "simulation") -> int:
    """Publish a simulation event."""
    return get_event_bus("simulation").publish(event_type, data, source)


def publish_ui_event(event_type: str, data: Any, source: str = "ui") -> int:
    """Publish a UI event."""
    return get_event_bus("ui").publish(event_type, data, source)


def publish_system_event(event_type: str, data: Any, source: str = "system") -> int:
    """Publish a system event."""
    return get_event_bus("system").publish(event_type, data, source)


def subscribe_to_simulation_events(event_type: str, callback: Callable, 
                                 priority: int = 0, filter_func: Optional[Callable] = None) -> bool:
    """Subscribe to simulation events."""
    return get_event_bus("simulation").subscribe(event_type, callback, priority, filter_func)


def subscribe_to_ui_events(event_type: str, callback: Callable, 
                         priority: int = 0, filter_func: Optional[Callable] = None) -> bool:
    """Subscribe to UI events."""
    return get_event_bus("ui").subscribe(event_type, callback, priority, filter_func)


def subscribe_to_system_events(event_type: str, callback: Callable, 
                             priority: int = 0, filter_func: Optional[Callable] = None) -> bool:
    """Subscribe to system events."""
    return get_event_bus("system").subscribe(event_type, callback, priority, filter_func)


# Example usage and testing
if __name__ == "__main__":
    # Test event bus functionality
    def test_callback(event: SimulationEvent):
        print(f"Received event: {event.event_type} from {event.source}")
    
    # Get event bus
    bus = get_event_bus("test")
    
    # Subscribe to events
    bus.subscribe("test_event", test_callback, priority=1)
    
    # Publish event
    bus.publish("test_event", {"message": "Hello World"}, "test_source")
    
    # Get statistics
    stats = bus.get_statistics()
    print(f"Event bus statistics: {stats}")
    
    print("Event bus test completed successfully!")
