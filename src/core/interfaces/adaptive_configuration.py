"""
IAdaptiveConfiguration interface - Adaptive configuration service for neural simulation.

This interface defines the contract for self-tuning configuration parameters
based on system performance, workload characteristics, and environmental factors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ConfigurationParameter:
    """Represents a configurable parameter with adaptation capabilities."""

    name: str
    current_value: Any
    value_type: str = "float"  # "int", "float", "bool", "str"
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    default_value: Optional[Any] = None
    adaptation_enabled: bool = True
    last_adapted: float = 0.0
    adaptation_history: List[Tuple[float, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Set default_value if not provided."""
        if self.default_value is None:
            self.default_value = self.current_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter to dictionary."""
        return {
            'name': self.name,
            'current_value': self.current_value,
            'value_type': self.value_type,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'default_value': self.default_value,
            'adaptation_enabled': self.adaptation_enabled,
            'last_adapted': self.last_adapted,
            'adaptation_history': self.adaptation_history.copy()
        }


@dataclass
class AdaptationRule:
    """Represents a rule for parameter adaptation."""

    parameter_name: str
    condition: str  # e.g., "cpu_usage > 0.8"
    action: str  # e.g., "decrease batch_size by 0.2"
    priority: int = 1
    enabled: bool = True
    activation_count: int = 0
    last_activated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            'parameter_name': self.parameter_name,
            'condition': self.condition,
            'action': self.action,
            'priority': self.priority,
            'enabled': self.enabled,
            'activation_count': self.activation_count,
            'last_activated': self.last_activated
        }


class IAdaptiveConfiguration(ABC):
    """
    Abstract interface for adaptive configuration in neural simulation.

    This interface defines the contract for dynamically adjusting system
    parameters based on performance metrics and environmental conditions.
    """

    @abstractmethod
    def register_parameter(self, parameter: ConfigurationParameter) -> bool:
        """
        Register a parameter for adaptive configuration.

        Args:
            parameter: The parameter to register

        Returns:
            bool: True if registration successful
        """

    @abstractmethod
    def add_adaptation_rule(self, rule: AdaptationRule) -> bool:
        """
        Add an adaptation rule for parameter adjustment.

        Args:
            rule: The adaptation rule to add

        Returns:
            bool: True if rule added successfully
        """

    @abstractmethod
    def evaluate_adaptation_rules(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate adaptation rules against current metrics.

        Args:
            metrics: Current system metrics

        Returns:
            List[Dict[str, Any]]: Adaptation actions to take
        """

    @abstractmethod
    def apply_adaptation(self, parameter_name: str, new_value: Any) -> bool:
        """
        Apply an adaptation to a parameter.

        Args:
            parameter_name: Name of the parameter to adapt
            new_value: New value for the parameter

        Returns:
            bool: True if adaptation applied successfully
        """

    @abstractmethod
    def get_optimal_configuration(self, workload_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimal configuration for a given workload profile.

        Args:
            workload_profile: Description of the expected workload

        Returns:
            Dict[str, Any]: Optimal configuration parameters
        """

    @abstractmethod
    def analyze_configuration_impact(self, parameter_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the impact of proposed parameter changes.

        Args:
            parameter_changes: Proposed parameter changes

        Returns:
            Dict[str, Any]: Impact analysis results
        """

    @abstractmethod
    def create_configuration_profile(self, profile_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Create a named configuration profile.

        Args:
            profile_name: Name of the configuration profile
            parameters: Parameter values for the profile

        Returns:
            bool: True if profile created successfully
        """

    @abstractmethod
    def load_configuration_profile(self, profile_name: str) -> bool:
        """
        Load a named configuration profile.

        Args:
            profile_name: Name of the configuration profile to load

        Returns:
            bool: True if profile loaded successfully
        """
