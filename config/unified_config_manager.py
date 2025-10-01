"""
Unified Configuration Manager for the Neural Simulation.
Consolidates config_manager.py and dynamic_config_manager.py
into a comprehensive configuration management system.
"""

import configparser
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from src.utils.logging_utils import log_step
from src.utils.print_utils import print_error, print_info, print_warning


def _get_print_utils():
    """Lazy import of print utilities to avoid circular imports."""
    return print_info, print_warning, print_error


def _get_cached_print_utils():
    """Get cached print utilities with lazy loading."""
    if not hasattr(_get_cached_print_utils, 'cache'):
        _get_cached_print_utils.cache = _get_print_utils()
    return _get_cached_print_utils.cache


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    key: str
    config_type: ConfigType
    default_value: Any
    description: str
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True
    validation_func: Optional[Callable[[Any], bool]] = None


@dataclass
class ConfigChange:
    """Configuration change record."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float
    source: str


class ConfigValidator:
    """Configuration value validator."""

    @staticmethod
    def validate_string(value: Any, schema: ConfigSchema) -> bool:
        """Validate string value."""
        if not isinstance(value, str):
            return False
        if schema.allowed_values and value not in schema.allowed_values:
            return False
        return True

    @staticmethod
    def validate_integer(value: Any, schema: ConfigSchema) -> bool:
        """Validate integer value."""
        try:
            int_val = int(value)
            if schema.min_value is not None and int_val < schema.min_value:
                return False
            if schema.max_value is not None and int_val > schema.max_value:
                return False
            if schema.allowed_values and int_val not in schema.allowed_values:
                return False
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_float(value: Any, schema: ConfigSchema) -> bool:
        """Validate float value."""
        try:
            float_val = float(value)
            if schema.min_value is not None and float_val < schema.min_value:
                return False
            if schema.max_value is not None and float_val > schema.max_value:
                return False
            if schema.allowed_values and float_val not in schema.allowed_values:
                return False
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_boolean(value: Any) -> bool:
        """Validate boolean value."""
        return isinstance(value, bool)

    @staticmethod
    def validate_list(value: Any, schema: ConfigSchema) -> bool:
        """Validate list value."""
        if not isinstance(value, list):
            return False
        if schema.allowed_values and not all(item in schema.allowed_values for item in value):
            return False
        return True

    @staticmethod
    def validate_dict(value: Any, schema: ConfigSchema) -> bool:
        """Validate dict value."""
        if not isinstance(value, dict):
            return False
        if schema.allowed_values and not all(key in schema.allowed_values for key in value.keys()):
            return False
        return True

    @classmethod
    def validate(cls, value: Any, schema: ConfigSchema) -> bool:
        """Validate value against schema."""
        if schema.validation_func:
            return schema.validation_func(value)

        validators = {
            ConfigType.STRING: cls.validate_string,
            ConfigType.INTEGER: cls.validate_integer,
            ConfigType.FLOAT: cls.validate_float,
            ConfigType.BOOLEAN: cls.validate_boolean,
            ConfigType.LIST: cls.validate_list,
            ConfigType.DICT: cls.validate_dict,
        }

        if schema.config_type in validators:
            return validators[schema.config_type](value, schema)

        return True


class UnifiedConfigManager:
    """Unified configuration manager with validation and change notifications."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.change_history: List[ConfigChange] = []
        self.watchers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self.validator = ConfigValidator()

        # INI parser for backward compatibility
        self.ini_config = configparser.ConfigParser()

        # Load initial configuration
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        else:
            self._create_default_config()

        # Setup default schemas
        self._setup_default_schemas()

        log_step("UnifiedConfigManager initialized")

    def _flatten_config(self, data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested configuration dictionary to dot notation."""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    self._flatten_config(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)

    def _unflatten_config(self, flat_config: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
        """Unflatten dot notation configuration back to nested dictionary."""
        result: Dict[str, Any] = {}
        for full_key, value in flat_config.items():
            keys = full_key.split(sep)
            d = result
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value
        return result

    def _create_default_config(self):
        """Create default configuration."""
        default_config = {
            'General': {
                'debug_mode': False,
                'log_level': 'INFO',
                'max_threads': 4
            },
            'SystemConstants': {
                'node_energy_cap': 5.0,
                'time_step': 0.01,
                'refractory_period': 0.1
            },
            'Learning': {
                'plasticity_rate': 0.01,
                'stdp_window': 20.0,
                'ltp_rate': 0.02
            },
            'EnhancedNodes': {
                'oscillator_frequency': 1.0,
                'integrator_threshold': 0.8,
                'relay_amplification': 1.5,
                'highway_energy_boost': 2.0
            },
            'Performance': {
                'monitoring_enabled': True,
                'optimization_level': 'medium',
                'memory_pooling': True
            }
        }

        # Convert to flat structure, using schema defaults if available
        for section, values in default_config.items():
            for key, value in values.items():
                full_key = f"{section}.{key}"
                if full_key in self.schemas:
                    self.config[full_key] = self.schemas[full_key].default_value
                else:
                    self.config[full_key] = value

    def _setup_default_schemas(self):
        """Setup default configuration schemas."""
        default_schemas = [
            ConfigSchema('General.debug_mode', ConfigType.BOOLEAN, False, 'Enable debug mode'),
            ConfigSchema('General.log_level', ConfigType.STRING, 'INFO', 'Logging level',
                       allowed_values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
            ConfigSchema('General.max_threads', ConfigType.INTEGER, 4, 'Maximum number of threads',
                       min_value=1, max_value=32),
            ConfigSchema('SystemConstants.node_energy_cap', ConfigType.FLOAT, 5.0,
                        'Maximum node energy capacity', min_value=1.0, max_value=1000.0),
            ConfigSchema('SystemConstants.time_step', ConfigType.FLOAT, 0.01,
                       'Simulation time step', min_value=0.001, max_value=1.0),
            ConfigSchema('SystemConstants.refractory_period', ConfigType.FLOAT, 0.1,
                       'Node refractory period', min_value=0.0, max_value=1.0),
            ConfigSchema('Learning.plasticity_rate', ConfigType.FLOAT, 0.01,
                       'Learning plasticity rate', min_value=0.0, max_value=1.0),
            ConfigSchema('Learning.stdp_window', ConfigType.FLOAT, 20.0,
                       'STDP learning window', min_value=0.0),
            ConfigSchema('Learning.ltp_rate', ConfigType.FLOAT, 0.02,
                       'Long-term potentiation rate', min_value=0.0, max_value=1.0),
            ConfigSchema('EnhancedNodes.oscillator_frequency', ConfigType.FLOAT, 1.0,
                       'Oscillator node frequency', min_value=0.0),
            ConfigSchema('EnhancedNodes.integrator_threshold', ConfigType.FLOAT, 0.8,
                       'Integrator node threshold', min_value=0.0, max_value=1.0),
            ConfigSchema('EnhancedNodes.relay_amplification', ConfigType.FLOAT, 1.5,
                       'Relay node amplification', min_value=0.0),
            ConfigSchema('EnhancedNodes.highway_energy_boost', ConfigType.FLOAT, 2.0,
                       'Highway node energy boost', min_value=0.0),
            ConfigSchema('Performance.monitoring_enabled', ConfigType.BOOLEAN, True,
                       'Enable performance monitoring'),
            ConfigSchema('Performance.optimization_level', ConfigType.STRING, 'medium',
                       'Performance optimization level',
                       allowed_values=['none', 'low', 'medium', 'high', 'maximum']),
            ConfigSchema('Performance.memory_pooling', ConfigType.BOOLEAN, True,
                       'Enable memory pooling')
        ]

        for schema in default_schemas:
            self.schemas[schema.key] = schema

    def register_schema(self, schema: ConfigSchema):
        """Register a configuration schema."""
        with self._lock:
            self.schemas[schema.key] = schema

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        with self._lock:
            return self.config.get(key, default)

    def set(self, key: str, value: Any, source: str = "manual") -> bool:
        """Set a configuration value with validation."""
        with self._lock:
            # Get schema for validation
            schema = self.schemas.get(key)
            if schema:
                if not self.validator.validate(value, schema):
                    raise ValueError(f"Invalid value for {key}: {value}")

            # Store old value for change tracking
            old_value = self.config.get(key)

            # Set new value
            self.config[key] = value

            # Record change
            change = ConfigChange(
                key=key,
                old_value=old_value,
                new_value=value,
                timestamp=time.time(),
                source=source
            )
            self.change_history.append(change)

            # Notify watchers
            self._notify_watchers(key, old_value, value)

            return True

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all configuration values for a section."""
        with self._lock:
            section_config = {}
            for key, value in self.config.items():
                if key.startswith(f"{section}."):
                    section_key = key[len(f"{section}."):]
                    section_config[section_key] = value
            return section_config

    def set_section(self, section: str, values: Dict[str, Any], source: str = "manual"):
        """Set multiple configuration values for a section."""
        with self._lock:
            for key, value in values.items():
                full_key = f"{section}.{key}"
                self.set(full_key, value, source)

    def watch(self, key: str, callback: Callable[[str, Any, Any], None]):
        """Watch for changes to a configuration key."""
        with self._lock:
            self.watchers[key].append(callback)

    def unwatch(self, key: str, callback: Callable[[str, Any, Any], None]):
        """Stop watching for changes to a configuration key."""
        with self._lock:
            if callback in self.watchers[key]:
                self.watchers[key].remove(callback)

    def _notify_watchers(self, key: str, old_value: Any, new_value: Any):
        """Notify watchers of configuration changes."""
        for callback in self.watchers[key]:
            try:
                callback(key, old_value, new_value)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print_error(f"Config watcher callback failed: {e}")

    def load_from_file(self, file_path: str) -> bool:
        """Load configuration from file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print_warning(f"Config file not found: {file_path}")
                return False

            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    flat_data = self._flatten_config(data)
                    self.config.update(flat_data)
            elif file_path.suffix.lower() in ['.yml', '.yaml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    flat_data = self._flatten_config(data)
                    self.config.update(flat_data)
            elif file_path.suffix.lower() == '.ini':
                self.ini_config.read(file_path)
                self._load_ini_config()
            else:
                print_error(f"Unsupported config file format: {file_path.suffix}")
                return False

            # Validate and reset any invalid loaded values
            self._validate_and_reset()

            log_step(f"Configuration loaded from {file_path}")
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            print_error(f"Failed to load config from {file_path}: {e}")
            return False

    def _load_ini_config(self):
        """Load configuration from INI file."""
        for section in self.ini_config.sections():
            for key, value in self.ini_config.items(section):
                full_key = f"{section}.{key}"
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        self.config[full_key] = float(value)
                    else:
                        self.config[full_key] = int(value)
                except (ValueError, TypeError):
                    if value.lower() in ('true', 'false'):
                        self.config[full_key] = value.lower() == 'true'
                    else:
                        self.config[full_key] = value

    def save_to_file(self, file_path: str, fmt: str = "json") -> bool:
        """Save configuration to file."""
        try:
            file_path = Path(file_path)

            if fmt.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self._unflatten_config(self.config), f, indent=2)
            elif fmt.lower() in ['yml', 'yaml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self._unflatten_config(self.config), f, default_flow_style=False)
            elif fmt.lower() == 'ini':
                self._save_ini_config(file_path)
            else:
                print_error(f"Unsupported save format: {fmt}")
                return False

            log_step(f"Configuration saved to {file_path}")
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            print_error(f"Failed to save config to {file_path}: {e}")
            return False

    def _save_ini_config(self, file_path: Path):
        """Save configuration to INI file."""
        sections = defaultdict(dict)
        for key, value in self.config.items():
            if '.' in key:
                section, option = key.split('.', 1)
                sections[section][option] = str(value)
            else:
                sections['General'][key] = str(value)

        with open(file_path, 'w', encoding='utf-8') as f:
            for section, options in sections.items():
                f.write(f"[{section}]\n")
                for option, value in options.items():
                    f.write(f"{option} = {value}\n")
                f.write("\n")

    def get_change_history(self, key: Optional[str] = None, limit: int = 100) -> List[ConfigChange]:
        """Get configuration change history."""
        with self._lock:
            if key:
                changes = [c for c in self.change_history if c.key == key]
            else:
                changes = list(self.change_history)
            return changes[-limit:]

    def reset_to_defaults(self, section: Optional[str] = None):
        """Reset configuration to defaults."""
        with self._lock:
            if section:
                # Reset only specific section
                for schema in self.schemas.values():
                    if schema.key.startswith(f"{section}."):
                        self.config[schema.key] = schema.default_value
            else:
                # Reset all configuration
                for schema in self.schemas.values():
                    self.config[schema.key] = schema.default_value

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration values."""
        with self._lock:
            errors = defaultdict(list)
            for key, value in self.config.items():
                schema = self.schemas.get(key)
                if schema and not self.validator.validate(value, schema):
                    errors[key].append(f"Invalid value: {value}")
            return dict(errors)

    def _validate_and_reset(self):
        """Validate loaded config and reset invalid values to defaults."""
        errors = self.validate_all()
        for key, error_msgs in errors.items():
            if key in self.schemas:
                default_val = self.schemas[key].default_value
                self.config[key] = default_val
                print_warning(f"Reset invalid config '{key}' to default '{default_val}'. Errors: {error_msgs[0] if error_msgs else 'Unknown'}")

    def get_schema(self, key: str) -> Optional[ConfigSchema]:
        """Get schema for a configuration key."""
        return self.schemas.get(key)

    def list_keys(self, section: Optional[str] = None) -> List[str]:
        """List all configuration keys."""
        with self._lock:
            if section:
                return [k for k in self.config if k.startswith(f"{section}.")]
            return list(self.config.keys())

    def export_config(self, fmt: str = "json") -> str:
        """Export configuration as string."""
        if fmt.lower() == 'json':
            return json.dumps(self._unflatten_config(self.config), indent=2)
        if fmt.lower() in ['yml', 'yaml']:
            return yaml.dump(self._unflatten_config(self.config), default_flow_style=False)
        raise ValueError(f"Unsupported export format: {fmt}")

    # Backward compatibility methods
    def get_float(self, section: str, key: str, default: float = 0.0) -> float:
        """Get float value (INI-style)."""
        value = self.get(f"{section}.{key}", default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_int(self, section: str, key: str, default: int = 0) -> int:
        """Get integer value (INI-style)."""
        value = self.get(f"{section}.{key}", default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        """Get boolean value (INI-style)."""
        value = self.get(f"{section}.{key}", default)
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        try:
            return bool(value)
        except (ValueError, TypeError):
            return default

    def get_string(self, section: str, key: str, default: str = "") -> str:
        """Get string value (INI-style)."""
        value = self.get(f"{section}.{key}", default)
        try:
            return str(value)
        except (ValueError, TypeError):
            return default

    def set_value(self, section: str, key: str, value: Any):
        """Set value (INI-style)."""
        self.set(f"{section}.{key}", value)

    def save(self):
        """Save configuration to file."""
        if self.config_file:
            self.save_to_file(self.config_file)

    def reload(self):
        """Reload configuration from file."""
        if self.config_file:
            self.load_from_file(self.config_file)


def get_config_manager() -> UnifiedConfigManager:
    """Get the global configuration manager."""
    if not hasattr(get_config_manager, 'instance'):
        get_config_manager.instance = UnifiedConfigManager()
    return get_config_manager.instance


# Convenience functions for backward compatibility
def get_config(section: str, key: str, default: Any = None, value_type: type = str) -> Any:
    """Get a configuration value (INI-style)."""
    value = get_config_manager().get(f"{section}.{key}", default)
    if value is None:
        return default
    try:
        value_converters = {
            int: int,
            float: float,
            bool: lambda v: v.lower() in ('true', '1', 'yes', 'on') if isinstance(v, str) else bool(v),
            str: str,
        }
        if value_type in value_converters:
            return value_converters[value_type](value)
        return value
    except (ValueError, TypeError):
        return default


def set_config(section: str, key: str, value: Any):
    """Set a configuration value (INI-style)."""
    get_config_manager().set(f"{section}.{key}", value)


def get_system_constants() -> Dict[str, float]:
    """Get system constants."""
    return get_config_manager().get_section("SystemConstants")


def get_enhanced_nodes_config() -> Dict[str, float]:
    """Get enhanced nodes configuration."""
    return get_config_manager().get_section("EnhancedNodes")


def get_learning_config() -> Dict[str, float]:
    """Get learning configuration."""
    return get_config_manager().get_section("Learning")


def get_homeostasis_config() -> Dict[str, float]:
    """Get homeostasis configuration."""
    return get_config_manager().get_section("Homeostasis")


def get_network_metrics_config() -> Dict[str, Union[int, float]]:
    """Get network metrics configuration."""
    return get_config_manager().get_section("NetworkMetrics")


def get_workspace_config() -> Dict[str, float]:
    """Get workspace configuration."""
    return get_config_manager().get_section("Workspace")


def get_processing_config() -> Dict[str, Union[int, float]]:
    """Get processing configuration."""
    return get_config_manager().get_section("Processing")


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    return get_config_manager().get_section("Performance")


def get_neural_config() -> Dict[str, Any]:
    """Get neural network configuration."""
    return get_config_manager().get_section("Neural")


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration."""
    return get_config_manager().get_section("Logging")


# Backward compatibility aliases
ConfigManager = UnifiedConfigManager
DynamicConfigManager = UnifiedConfigManager
config = get_config_manager()







