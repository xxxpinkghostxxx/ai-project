"""
Dynamic Configuration Manager for the Neural Simulation System.
Provides runtime configuration updates, validation, and change notifications.
"""

import json
import yaml
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from collections import defaultdict
import weakref

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
        if not isinstance(value, int):
            return False
        if schema.min_value is not None and value < schema.min_value:
            return False
        if schema.max_value is not None and value > schema.max_value:
            return False
        if schema.allowed_values and value not in schema.allowed_values:
            return False
        return True
    
    @staticmethod
    def validate_float(value: Any, schema: ConfigSchema) -> bool:
        """Validate float value."""
        if not isinstance(value, (int, float)):
            return False
        if schema.min_value is not None and value < schema.min_value:
            return False
        if schema.max_value is not None and value > schema.max_value:
            return False
        if schema.allowed_values and value not in schema.allowed_values:
            return False
        return True
    
    @staticmethod
    def validate_boolean(value: Any, schema: ConfigSchema) -> bool:
        """Validate boolean value."""
        return isinstance(value, bool)
    
    @staticmethod
    def validate_list(value: Any, schema: ConfigSchema) -> bool:
        """Validate list value."""
        if not isinstance(value, list):
            return False
        if schema.allowed_values and value not in schema.allowed_values:
            return False
        return True
    
    @staticmethod
    def validate_dict(value: Any, schema: ConfigSchema) -> bool:
        """Validate dictionary value."""
        if not isinstance(value, dict):
            return False
        if schema.allowed_values and value not in schema.allowed_values:
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
            ConfigType.DICT: cls.validate_dict
        }
        
        validator = validators.get(schema.config_type)
        if validator:
            return validator(value, schema)
        
        return True

class DynamicConfigManager:
    """Dynamic configuration manager with validation and change notifications."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.change_history: List[ConfigChange] = []
        self.watchers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self.validator = ConfigValidator()
        
        # Load initial configuration
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Setup default schemas
        self._setup_default_schemas()
    
    def _setup_default_schemas(self):
        """Setup default configuration schemas."""
        default_schemas = [
            ConfigSchema(
                key="simulation.fps",
                config_type=ConfigType.INTEGER,
                default_value=60,
                description="Target simulation FPS",
                min_value=1,
                max_value=120
            ),
            ConfigSchema(
                key="simulation.max_nodes",
                config_type=ConfigType.INTEGER,
                default_value=10000,
                description="Maximum number of nodes",
                min_value=100,
                max_value=100000
            ),
            ConfigSchema(
                key="simulation.max_edges",
                config_type=ConfigType.INTEGER,
                default_value=50000,
                description="Maximum number of edges",
                min_value=1000,
                max_value=500000
            ),
            ConfigSchema(
                key="performance.memory_limit_mb",
                config_type=ConfigType.INTEGER,
                default_value=1000,
                description="Memory limit in MB",
                min_value=100,
                max_value=10000
            ),
            ConfigSchema(
                key="performance.enable_optimizations",
                config_type=ConfigType.BOOLEAN,
                default_value=True,
                description="Enable performance optimizations"
            ),
            ConfigSchema(
                key="logging.level",
                config_type=ConfigType.STRING,
                default_value="INFO",
                description="Logging level",
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),
            ConfigSchema(
                key="neural.learning_rate",
                config_type=ConfigType.FLOAT,
                default_value=0.01,
                description="Neural learning rate",
                min_value=0.001,
                max_value=1.0
            ),
            ConfigSchema(
                key="neural.energy_cap",
                config_type=ConfigType.FLOAT,
                default_value=255.0,
                description="Maximum energy capacity",
                min_value=1.0,
                max_value=1000.0
            )
        ]
        
        for schema in default_schemas:
            self.register_schema(schema)
    
    def register_schema(self, schema: ConfigSchema):
        """Register a configuration schema."""
        with self._lock:
            self.schemas[schema.key] = schema
            
            # Set default value if not already set
            if schema.key not in self.config:
                self.config[schema.key] = schema.default_value
    
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
            except Exception as e:
                logging.error(f"Error in config watcher for {key}: {e}")
    
    def load_from_file(self, file_path: str) -> bool:
        """Load configuration from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            with open(path, 'r') as f:
                if path.suffix.lower() == '.json':
                    data = json.load(f)
                elif path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                else:
                    return False
            
            with self._lock:
                for key, value in data.items():
                    self.config[key] = value
            
            return True
        except Exception as e:
            logging.error(f"Error loading config from {file_path}: {e}")
            return False
    
    def save_to_file(self, file_path: str, format: str = "json") -> bool:
        """Save configuration to file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._lock:
                config_data = self.config.copy()
            
            with open(path, 'w') as f:
                if format.lower() == 'json':
                    json.dump(config_data, f, indent=2)
                elif format.lower() in ['yml', 'yaml']:
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    return False
            
            return True
        except Exception as e:
            logging.error(f"Error saving config to {file_path}: {e}")
            return False
    
    def get_change_history(self, key: Optional[str] = None, limit: int = 100) -> List[ConfigChange]:
        """Get configuration change history."""
        with self._lock:
            if key:
                changes = [c for c in self.change_history if c.key == key]
            else:
                changes = self.change_history.copy()
            
            return changes[-limit:]
    
    def reset_to_defaults(self, section: Optional[str] = None):
        """Reset configuration to default values."""
        with self._lock:
            if section:
                # Reset only specific section
                for key, schema in self.schemas.items():
                    if key.startswith(f"{section}."):
                        self.config[key] = schema.default_value
            else:
                # Reset all configuration
                for key, schema in self.schemas.items():
                    self.config[key] = schema.default_value
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration values."""
        errors = defaultdict(list)
        
        with self._lock:
            for key, value in self.config.items():
                schema = self.schemas.get(key)
                if schema and not self.validator.validate(value, schema):
                    errors[key].append(f"Invalid value: {value}")
        
        return dict(errors)
    
    def get_schema(self, key: str) -> Optional[ConfigSchema]:
        """Get schema for a configuration key."""
        return self.schemas.get(key)
    
    def list_keys(self, section: Optional[str] = None) -> List[str]:
        """List all configuration keys, optionally filtered by section."""
        with self._lock:
            if section:
                return [key for key in self.config.keys() if key.startswith(f"{section}.")]
            return list(self.config.keys())
    
    def export_config(self, format: str = "json") -> str:
        """Export configuration as string."""
        with self._lock:
            config_data = self.config.copy()
        
        if format.lower() == 'json':
            return json.dumps(config_data, indent=2)
        elif format.lower() in ['yml', 'yaml']:
            return yaml.dump(config_data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> DynamicConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = DynamicConfigManager()
    return _config_manager

def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value."""
    return get_config_manager().get(key, default)

def set_config(key: str, value: Any, source: str = "manual") -> bool:
    """Set a configuration value."""
    return get_config_manager().set(key, value, source)

def watch_config(key: str, callback: Callable[[str, Any, Any], None]):
    """Watch for configuration changes."""
    get_config_manager().watch(key, callback)

def load_config_from_file(file_path: str) -> bool:
    """Load configuration from file."""
    return get_config_manager().load_from_file(file_path)

def save_config_to_file(file_path: str, format: str = "json") -> bool:
    """Save configuration to file."""
    return get_config_manager().save_to_file(file_path, format)

# Convenience functions for common configuration patterns
def get_simulation_config() -> Dict[str, Any]:
    """Get simulation-related configuration."""
    return get_config_manager().get_section("simulation")

def get_performance_config() -> Dict[str, Any]:
    """Get performance-related configuration."""
    return get_config_manager().get_section("performance")

def get_neural_config() -> Dict[str, Any]:
    """Get neural network-related configuration."""
    return get_config_manager().get_section("neural")

def get_logging_config() -> Dict[str, Any]:
    """Get logging-related configuration."""
    return get_config_manager().get_section("logging")

# Configuration decorator for automatic updates
def configurable(keys: List[str]):
    """Decorator to make a class configurable."""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Watch for configuration changes
            for key in keys:
                def make_callback(attr_name):
                    def callback(config_key, old_value, new_value):
                        if hasattr(self, attr_name):
                            setattr(self, attr_name, new_value)
                    return callback
                
                watch_config(key, make_callback(key.split('.')[-1]))
        
        cls.__init__ = new_init
        return cls
    
    return decorator
