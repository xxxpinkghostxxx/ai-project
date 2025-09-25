"""
IEventCoordinator interface - Event-driven communication service.

This interface defines the contract for event-driven communication,
providing publish-subscribe pattern for loose coupling between services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime


class Event:
    """Represents an event in the system."""

    def __init__(self, event_type: str, data: Any, source: str):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
        self.id = f"{event_type}_{self.timestamp.timestamp()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp.isoformat()
        }


class IEventCoordinator(ABC):
    """
    Abstract interface for event-driven communication.

    This interface defines the contract for publish-subscribe event communication,
    enabling loose coupling between services in the neural simulation system.
    """

    @abstractmethod
    def publish(self, event_type: str, data: Any, source: Optional[str] = None) -> None:
        """Publish an event to all subscribers."""
        pass

    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> str:
        """Subscribe to an event type."""
        pass

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from an event."""
        pass

    @abstractmethod
    @abstractmethod
    def process_events(self) -> None:
        """Process any pending events in the event system."""
        pass
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get historical events."""
        pass