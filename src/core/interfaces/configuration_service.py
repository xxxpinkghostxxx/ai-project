"""
IConfigurationService interface - Configuration management service.

This interface defines the contract for centralized configuration management,
providing validation, migration, and dynamic updates for all system parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum


class ConfigurationScope(Enum):
    """Defines the scope of configuration parameters."""
    GLOBAL = "global"
    SIMULATION = "simulation"
    NEURAL = "neural"
    ENERGY = "energy"
    LEARNING = "learning"
    SENSORY = "sensory"


class IConfigurationService(ABC):
    """
    Abstract interface for configuration management operations.

    This interface defines the contract for managing system configuration,
    including validation, migration, and dynamic parameter updates.
    """

    @abstractmethod
    def load_configuration(self, config_path: Optional[str] = None) -> bool:
        """Load configuration from file or default."""
        pass

    @abstractmethod
    def save_configuration(self, config_path: str) -> bool:
        """Save current configuration to file."""
        pass

    @abstractmethod
    def get_parameter(self, key: str, scope: ConfigurationScope = ConfigurationScope.GLOBAL) -> Any:
        """Get a configuration parameter."""
        pass

    @abstractmethod
    def set_parameter(self, key: str, value: Any, scope: ConfigurationScope = ConfigurationScope.GLOBAL) -> bool:
        """Set a configuration parameter."""
        pass

    @abstractmethod
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration."""
        pass

    @abstractmethod
    def get_configuration_schema(self, scope: Optional[ConfigurationScope] = None) -> Dict[str, Any]:
        """Get configuration schema for validation."""
        pass






