"""
IRealTimeAnalytics interface - Real-time analytics service for neural simulation.

This interface defines the contract for real-time performance monitoring,
predictive analytics, and adaptive optimization of neural simulation systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class AnalyticsMetric:
    """Represents a performance or system metric."""

    def __init__(self, name: str, value: float, timestamp: Optional[float] = None):
        self.name = name
        self.value = value
        self.timestamp = timestamp or datetime.now().timestamp()
        self.unit = "unit"  # e.g., "ms", "MB", "ops/sec"
        self.category = "general"  # e.g., "performance", "memory", "energy"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'unit': self.unit,
            'category': self.category
        }


class PredictionModel:
    """Represents a predictive analytics model."""

    def __init__(self, model_type: str, target_metric: str):
        self.model_type = model_type  # "linear", "exponential", "neural_network"
        self.target_metric = target_metric
        self.accuracy = 0.0
        self.last_trained = 0.0
        self.parameters: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'model_type': self.model_type,
            'target_metric': self.target_metric,
            'accuracy': self.accuracy,
            'last_trained': self.last_trained,
            'parameters': self.parameters.copy()
        }


class IRealTimeAnalytics(ABC):
    """
    Abstract interface for real-time analytics in neural simulation.

    This interface defines the contract for collecting metrics, performing
    predictive analytics, and providing optimization recommendations.
    """

    @abstractmethod
    def collect_system_metrics(self) -> List[AnalyticsMetric]:
        """
        Collect comprehensive system performance metrics.

        Returns:
            List[AnalyticsMetric]: Current system metrics
        """
        pass

    @abstractmethod
    def analyze_performance_trends(self, time_window: int = 300) -> Dict[str, Any]:
        """
        Analyze performance trends over a specified time window.

        Args:
            time_window: Analysis time window in seconds

        Returns:
            Dict[str, Any]: Performance trend analysis
        """
        pass

    @abstractmethod
    def predict_system_behavior(self, prediction_horizon: int = 60) -> Dict[str, Any]:
        """
        Predict future system behavior based on current trends.

        Args:
            prediction_horizon: Prediction time horizon in seconds

        Returns:
            Dict[str, Any]: System behavior predictions
        """
        pass

    @abstractmethod
    def detect_anomalies(self, sensitivity: float = 0.8) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies and system issues.

        Args:
            sensitivity: Anomaly detection sensitivity (0.0-1.0)

        Returns:
            List[Dict[str, Any]]: Detected anomalies
        """
        pass

    @abstractmethod
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on analytics.

        Returns:
            List[Dict[str, Any]]: Optimization recommendations
        """
        pass

    @abstractmethod
    def create_performance_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Create a comprehensive performance report.

        Args:
            report_type: Type of report ("summary", "detailed", "comprehensive")

        Returns:
            Dict[str, Any]: Performance report
        """
        pass

    @abstractmethod
    def monitor_service_health(self) -> Dict[str, Any]:
        """
        Monitor the health of all system services.

        Returns:
            Dict[str, Any]: Service health status
        """
        pass

    @abstractmethod
    def track_energy_efficiency(self) -> Dict[str, Any]:
        """
        Track and analyze energy efficiency metrics.

        Returns:
            Dict[str, Any]: Energy efficiency analysis
        """
        pass






