"""
IPerformanceMonitor interface - Performance monitoring and metrics service.

This interface defines the contract for monitoring system performance,
collecting metrics, and providing performance insights while maintaining
minimal overhead on the neural simulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self):
        self.step_time: float = 0.0
        self.memory_usage: float = 0.0
        self.cpu_usage: float = 0.0
        self.gpu_usage: Optional[float] = None
        self.network_traffic: float = 0.0
        self.active_threads: int = 0
        self.timestamp: datetime = datetime.now()

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
        pass

    @abstractmethod
    def stop_monitoring(self) -> bool:
        """Stop performance monitoring."""
        pass

    @abstractmethod
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        pass

    @abstractmethod
    def get_historical_metrics(self, time_range: int = 60) -> List[PerformanceMetrics]:
        """Get historical performance metrics."""
        pass