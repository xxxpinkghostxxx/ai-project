"""
config_manager.py

Configuration management utility for the AI project.
Provides unified access to configuration parameters from config.ini.
"""

import configparser
import os
from typing import Any, Dict, Union, Optional

class ConfigManager:
    """
    Manages configuration parameters from config.ini file.
    Provides type-safe access to configuration values with defaults.
    """
    
    def __init__(self, config_file: str = "config.ini"):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file with security validation."""
        if os.path.exists(self.config_file):
            # Validate file path to prevent path traversal
            if not self._validate_config_file_path():
                raise ValueError(f"Invalid configuration file path: {self.config_file}")
            
            # Check file permissions (should be readable only by owner)
            file_stat = os.stat(self.config_file)
            if file_stat.st_mode & 0o077:  # Check if others have read/write permissions
                import warnings
                warnings.warn(f"Configuration file {self.config_file} has insecure permissions")
            
            try:
                self.config.read(self.config_file)
            except Exception as e:
                raise ValueError(f"Failed to read configuration file: {e}")
        else:
            # Create default configuration if file doesn't exist
            self._create_default_config()
    
    def _validate_config_file_path(self) -> bool:
        """Validate configuration file path for security."""
        import os.path
        
        # Get absolute path to prevent path traversal
        abs_path = os.path.abspath(self.config_file)
        
        # Check for path traversal patterns
        if '..' in self.config_file or '\\' in self.config_file:
            return False
        
        # Ensure file is within allowed directory
        current_dir = os.path.abspath('.')
        if not abs_path.startswith(current_dir):
            return False
        
        return True
    
    def _create_default_config(self):
        """Create default configuration sections and values."""
        self.config['General'] = {
            'resolution_scale': '0.25'
        }
        
        self.config['PixelNodes'] = {
            'pixel_threshold': '128'
        }
        
        self.config['DynamicNodes'] = {
            'dynamic_node_percentage': '0.01'
        }
        
        self.config['Processing'] = {
            'update_interval': '0.5'
        }
        
        self.config['EnhancedNodes'] = {
            'oscillator_frequency': '0.1',
            'integrator_threshold': '0.8',
            'relay_amplification': '1.5',
            'highway_energy_boost': '2.0'
        }
        
        self.config['Learning'] = {
            'plasticity_rate': '0.01',
            'eligibility_decay': '0.95',
            'stdp_window': '20.0',
            'ltp_rate': '0.02',
            'ltd_rate': '0.01'
        }
        
        self.config['Homeostasis'] = {
            'target_energy_ratio': '0.6',
            'criticality_threshold': '0.1',
            'regulation_rate': '0.001',
            'regulation_interval': '100'
        }
        
        self.config['NetworkMetrics'] = {
            'calculation_interval': '50',
            'criticality_target': '1.0',
            'connectivity_target': '0.3'
        }
        
        # Save the default configuration
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get(self, section: str, key: str, default: Any = None, value_type: type = str) -> Any:
        """
        Get a configuration value with type conversion.
        
        Args:
            section: Configuration section name
            key: Configuration key name
            default: Default value if key not found
            value_type: Type to convert the value to (str, int, float, bool)
            
        Returns:
            Configuration value converted to specified type, or default if not found
        """
        try:
            if not self.config.has_section(section):
                return default
            
            if not self.config.has_option(section, key):
                return default
            
            value = self.config.get(section, key)
            
            # Type conversion
            if value_type == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif value_type == int:
                return int(value)
            elif value_type == float:
                return float(value)
            else:
                return value
                
        except (ValueError, TypeError):
            return default
    
    def get_float(self, section: str, key: str, default: float = 0.0) -> float:
        """Get a float configuration value."""
        return self.get(section, key, default, float)
    
    def get_int(self, section: str, key: str, default: int = 0) -> int:
        """Get an integer configuration value."""
        return self.get(section, key, default, int)
    
    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value."""
        return self.get(section, key, default, bool)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get all configuration values from a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            Dictionary of key-value pairs from the section
        """
        if not self.config.has_section(section):
            return {}
        
        result = {}
        for key in self.config.options(section):
            # Try to infer the type from the value
            value = self.config.get(section, key)
            try:
                if '.' in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                result[key] = value
        
        return result
    
    def set(self, section: str, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            section: Configuration section name
            key: Configuration key name
            value: Value to set
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
    
    def save(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
    
    def get_enhanced_nodes_config(self) -> Dict[str, float]:
        """Get enhanced nodes configuration."""
        return self.get_section('EnhancedNodes')
    
    def get_learning_config(self) -> Dict[str, float]:
        """Get learning configuration."""
        return self.get_section('Learning')
    
    def get_homeostasis_config(self) -> Dict[str, float]:
        """Get homeostasis configuration."""
        return self.get_section('Homeostasis')
    
    def get_network_metrics_config(self) -> Dict[str, Union[int, float]]:
        """Get network metrics configuration."""
        return self.get_section('NetworkMetrics')
    
    def get_workspace_config(self) -> Dict[str, float]:
        """Get workspace configuration."""
        return self.get_section('Workspace')
    
    def get_processing_config(self) -> Dict[str, float]:
        """Get processing configuration."""
        return self.get_section('Processing')
    
    def get_system_constants(self) -> Dict[str, float]:
        """Get system constants configuration."""
        return self.get_section('SystemConstants')

# Global configuration manager instance
config = ConfigManager()

# Convenience functions for backward compatibility
def get_config(section: str, key: str, default: Any = None, value_type: type = str) -> Any:
    """Get configuration value using global config manager."""
    return config.get(section, key, default, value_type)

def get_enhanced_nodes_config() -> Dict[str, float]:
    """Get enhanced nodes configuration."""
    return config.get_enhanced_nodes_config()

def get_learning_config() -> Dict[str, float]:
    """Get learning configuration."""
    return config.get_learning_config()

def get_homeostasis_config() -> Dict[str, float]:
    """Get homeostasis configuration."""
    return config.get_homeostasis_config()

def get_network_metrics_config() -> Dict[str, Union[int, float]]:
    """Get network metrics configuration."""
    return config.get_network_metrics_config()

def get_workspace_config() -> Dict[str, float]:
    """Get workspace configuration."""
    return config.get_workspace_config()

def get_processing_config() -> Dict[str, Union[int, float]]:
    """Get processing configuration."""
    return config.get_processing_config()

def get_system_constants() -> Dict[str, float]:
    """Get system constants configuration."""
    return config.get_system_constants()