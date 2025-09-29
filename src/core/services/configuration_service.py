"""
ConfigurationService implementation - Centralized configuration management.

This module provides the concrete implementation of IConfigurationService,
handling configuration loading, validation, and dynamic parameter management
for all neural simulation services.
"""

import os
import json
import configparser
from typing import Dict, Any, Optional
from threading import RLock

from ..interfaces.configuration_service import (
    IConfigurationService, ConfigurationScope
)


class ConfigurationService(IConfigurationService):
    """
    Concrete implementation of IConfigurationService.

    This service provides centralized configuration management with
    validation, migration, and dynamic updates for all system parameters.
    """

    def __init__(self):
        self._lock = RLock()
        self._config: Dict[str, Dict[str, Any]] = {}
        self._schema: Dict[str, Dict[str, Any]] = {}
        self._config_path: Optional[str] = None

        # Initialize default configuration
        self._initialize_default_config()

    def _initialize_default_config(self) -> None:
        """Initialize default configuration values."""
        self._config = {
            ConfigurationScope.GLOBAL.value: {
                "simulation_enabled": True,
                "debug_mode": False,
                "log_level": "INFO",
                "max_simulation_steps": 10000,
                "time_step": 0.001,
                "random_seed": 42
            },
            ConfigurationScope.SIMULATION.value: {
                "step_timeout": 1.0,
                "auto_save_interval": 1000,
                "performance_monitoring": True,
                "validation_enabled": True
            },
            ConfigurationScope.NEURAL.value: {
                "membrane_time_constant": 10.0,
                "threshold_potential": -50.0,
                "reset_potential": -80.0,
                "refractory_period": 2.0,
                "resting_potential": -70.0,
                "spike_threshold": 0.5
            },
            ConfigurationScope.ENERGY.value: {
                "energy_cap": 5.0,
                "decay_rate": 0.99,
                "metabolic_cost_per_spike": 0.1,
                "homeostasis_target": 1.0,
                "homeostasis_strength": 0.01,
                "energy_learning_modulation": True
            },
            ConfigurationScope.LEARNING.value: {
                "stdp_window": 20.0,
                "ltp_rate": 0.02,
                "ltd_rate": 0.01,
                "eligibility_decay": 0.95,
                "consolidation_threshold": 0.5,
                "learning_enabled": True
            },
            ConfigurationScope.SENSORY.value: {
                "visual_input_enabled": True,
                "audio_input_enabled": True,
                "sensory_buffer_size": 1000,
                "input_processing_delay": 0.01
            }
        }

        # Initialize configuration schema
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize configuration validation schema."""
        self._schema = {
            ConfigurationScope.GLOBAL.value: {
                "simulation_enabled": {"type": "bool", "required": True},
                "debug_mode": {"type": "bool", "required": False},
                "log_level": {
                    "type": "str",
                    "required": False,
                    "values": ["DEBUG", "INFO", "WARNING", "ERROR"]
                },
                "max_simulation_steps": {"type": "int", "required": False, "min": 1, "max": 100000},
                "time_step": {"type": "float", "required": False, "min": 0.0001, "max": 1.0},
                "random_seed": {"type": "int", "required": False}
            },
            ConfigurationScope.NEURAL.value: {
                "membrane_time_constant": {
                    "type": "float",
                    "required": False,
                    "min": 1.0,
                    "max": 100.0
                },
                "threshold_potential": {
                    "type": "float",
                    "required": False,
                    "min": -100.0,
                    "max": 0.0
                },
                "reset_potential": {"type": "float", "required": False, "min": -100.0, "max": 0.0},
                "refractory_period": {"type": "float", "required": False, "min": 0.1, "max": 10.0},
                "resting_potential": {
                    "type": "float",
                    "required": False,
                    "min": -100.0,
                    "max": 0.0
                },
                "spike_threshold": {"type": "float", "required": False, "min": 0.0, "max": 1.0}
            },
            ConfigurationScope.ENERGY.value: {
                "energy_cap": {"type": "float", "required": False, "min": 1.0, "max": 10.0},
                "decay_rate": {"type": "float", "required": False, "min": 0.9, "max": 1.0},
                "metabolic_cost_per_spike": {
                    "type": "float",
                    "required": False,
                    "min": 0.0,
                    "max": 1.0
                },
                "homeostasis_target": {"type": "float", "required": False, "min": 0.1, "max": 2.0},
                "homeostasis_strength": {
                    "type": "float",
                    "required": False,
                    "min": 0.0,
                    "max": 0.1
                },
                "energy_learning_modulation": {"type": "bool", "required": False}
            },
            ConfigurationScope.LEARNING.value: {
                "stdp_window": {"type": "float", "required": False, "min": 1.0, "max": 100.0},
                "ltp_rate": {"type": "float", "required": False, "min": 0.0, "max": 0.1},
                "ltd_rate": {"type": "float", "required": False, "min": 0.0, "max": 0.1},
                "eligibility_decay": {"type": "float", "required": False, "min": 0.8, "max": 1.0},
                "consolidation_threshold": {
                    "type": "float",
                    "required": False,
                    "min": 0.0,
                    "max": 1.0
                },
                "learning_enabled": {"type": "bool", "required": False}
            }
        }

    def load_configuration(self, config_path: Optional[str] = None) -> bool:
        """
        Load configuration from file or use defaults.

        Args:
            config_path: Path to configuration file (optional)

        Returns:
            bool: True if configuration loaded successfully
        """
        try:
            if config_path and os.path.exists(config_path):
                config = configparser.ConfigParser()
                config.read(config_path)
                loaded_config = {s: dict(config.items(s)) for s in config.sections()}

                # Merge loaded configuration with defaults
                for scope, scope_config in loaded_config.items():
                    try:
                        scope_enum = ConfigurationScope[scope.upper()]
                        if scope_enum.value in self._config:
                            for key, value in scope_config.items():
                                # Attempt to convert to int, float, or bool
                                if value.isdigit():
                                    self._config[scope_enum.value][key] = int(value)
                                elif value.replace('.', '', 1).isdigit():
                                    self._config[scope_enum.value][key] = float(value)
                                elif value.lower() in ['true', 'false']:
                                    self._config[scope_enum.value][key] = value.lower() == 'true'
                                else:
                                    self._config[scope_enum.value][key] = value
                    except KeyError:
                        pass # Ignore unknown scopes

                self._config_path = config_path
                return True
            else:
                # Use defaults
                return True

        except (IOError, OSError, configparser.Error) as e:
            print(f"Error loading configuration: {e}")
            return False

    def save_configuration(self, config_path: str) -> bool:
        """
        Save current configuration to file.

        Args:
            config_path: Path to save configuration

        Returns:
            bool: True if configuration saved successfully
        """
        try:
            with self._lock:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2)

                self._config_path = config_path
                return True

        except (IOError, OSError) as e:
            print(f"Error saving configuration: {e}")
            return False

    def get_parameter(self, key: str, scope: ConfigurationScope = ConfigurationScope.GLOBAL) -> Any:
        """
        Get a configuration parameter.

        Args:
            key: Parameter key
            scope: Configuration scope

        Returns:
            Parameter value or None if not found
        """
        with self._lock:
            if isinstance(scope, str):
                try:
                    scope = ConfigurationScope[scope.upper()]
                except KeyError:
                    return None  # Invalid scope string
            scope_config = self._config.get(scope.value, {})
            return scope_config.get(key)

    def set_parameter(
        self, key: str, value: Any, scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """
        Set a configuration parameter.

        Args:
            key: Parameter key
            value: Parameter value
            scope: Configuration scope

        Returns:
            bool: True if parameter set successfully
        """
        try:
            with self._lock:
                if scope.value not in self._config:
                    self._config[scope.value] = {}

                # Validate parameter if schema exists
                if self._validate_parameter(key, value, scope):
                    self._config[scope.value][key] = value
                    return True
                else:
                    print(f"Parameter validation failed for {key} = {value}")
                    return False

        except ValueError as e:
            print(f"Error setting parameter: {e}")
            return False

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration.

        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []

        with self._lock:
            for scope_name, scope_config in self._config.items():
                scope = ConfigurationScope(scope_name)
                scope_schema = self._schema.get(scope_name, {})

                for key, value in scope_config.items():
                    if key in scope_schema:
                        schema_def = scope_schema[key]
                        validation_result = self._validate_parameter_value(value, schema_def)
                        if not validation_result["valid"]:
                            issues.extend(validation_result["issues"])
                    else:
                        warnings.append(f"Unknown parameter '{key}' in scope '{scope_name}'")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_parameters": sum(len(scope) for scope in self._config.values())
        }

    def get_configuration_schema(
        self, scope: Optional[ConfigurationScope] = None
    ) -> Dict[str, Any]:
        """
        Get configuration schema for validation.

        Args:
            scope: Specific scope to get schema for (optional)

        Returns:
            Configuration schema
        """
        if scope:
            return self._schema.get(scope.value, {})
        else:
            return self._schema.copy()

    def _validate_parameter(self, key: str, value: Any, scope: ConfigurationScope) -> bool:
        """
        Validate a parameter against its schema.

        Args:
            key: Parameter key
            value: Parameter value
            scope: Configuration scope

        Returns:
            bool: True if parameter is valid
        """
        scope_schema = self._schema.get(scope.value, {})
        if key not in scope_schema:
            return True  # Unknown parameters are allowed

        schema_def = scope_schema[key]
        validation = self._validate_parameter_value(value, schema_def)
        return validation["valid"]

    def _validate_parameter_value(self, value: Any, schema_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a parameter value against its schema definition.

        Args:
            value: Parameter value
            schema_def: Schema definition

        Returns:
            Dict with validation results
        """
        issues = []

        # Type validation
        expected_type = schema_def.get("type")
        if expected_type:
            if expected_type == "bool" and not isinstance(value, bool):
                issues.append(f"Expected boolean, got {type(value).__name__}")
            elif expected_type == "int" and not isinstance(value, int):
                issues.append(f"Expected integer, got {type(value).__name__}")
            elif expected_type == "float" and not isinstance(value, (int, float)):
                issues.append(f"Expected number, got {type(value).__name__}")
            elif expected_type == "str" and not isinstance(value, str):
                issues.append(f"Expected string, got {type(value).__name__}")

        # Value range validation
        if "min" in schema_def and isinstance(value, (int, float)):
            if value < schema_def["min"]:
                issues.append(f"Value {value} is below minimum {schema_def['min']}")

        if "max" in schema_def and isinstance(value, (int, float)):
            if value > schema_def["max"]:
                issues.append(f"Value {value} is above maximum {schema_def['max']}")

        # Allowed values validation
        if "values" in schema_def and isinstance(value, str):
            if value not in schema_def["values"]:
                issues.append(f"Value '{value}' not in allowed values: {schema_def['values']}")
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

