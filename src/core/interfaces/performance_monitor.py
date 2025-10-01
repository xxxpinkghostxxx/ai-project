"""
IPerformanceMonitor interface - Performance monitoring and metrics service.

This interface defines the contract for monitoring system performance,
collecting metrics, and providing performance insights while maintaining
minimal overhead on the neural simulation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    step_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: Optional[float] = None
    network_traffic: float = 0.0
    active_threads: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'step_time': self.step_time,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'gpu_usage': self.gpu_usage,
            'network_traffic': self.network_traffic,
            'active_threads': self.active_threads,
            'timestamp': self.timestamp.isoformat()
        }


class IPerformanceMonitor(ABC):
    """
    Abstract interface for performance monitoring operations.

    This interface defines the contract for collecting and analyzing
    system performance metrics with minimal impact on simulation performance.
    """

    @abstractmethod
    def start_monitoring(self) -> bool:
        """Start performance monitoring."""
        raise NotImplementedError()

    @abstractmethod
    def stop_monitoring(self) -> bool:
        """Stop performance monitoring."""
        raise NotImplementedError()

    @abstractmethod
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        raise NotImplementedError()

    @abstractmethod
    def get_historical_metrics(self, time_range: int = 60) -> List[PerformanceMetrics]:
        """Get historical performance metrics."""
        raise NotImplementedError()



