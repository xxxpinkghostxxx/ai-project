
import configparser
import os
import time
import logging
from typing import Any, Dict, List, Optional, Union



class ConfigManager:
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._load_config()
        self._section_cache = {}
        self._cache_ttl = 300
        self._last_cache_update = time.time()
        self._precache_frequent_sections()
    def _load_config(self):
        if os.path.exists(self.config_file):
            if not self._validate_config_file_path():
                raise ValueError(f"Invalid configuration file path: {self.config_file}")
            if os.name != 'nt':
                file_stat = os.stat(self.config_file)
                if file_stat.st_mode & 0o077:
                    import warnings
                    if not hasattr(self, '_permission_warning_shown'):
                        warnings.warn(f"Configuration file {self.config_file} has insecure permissions")
                        self._permission_warning_shown = True
            try:
                self.config = configparser.ConfigParser(interpolation=None)
                self.config.read(self.config_file)
            except Exception as e:
                raise ValueError(f"Failed to read configuration file: {e}")
        else:
            self._create_default_config()
    def _validate_config_file_path(self) -> bool:

        abs_path = os.path.abspath(self.config_file)
        if '..' in self.config_file or '\\' in self.config_file:
            return False
        current_dir = os.path.abspath('.')
        if not abs_path.startswith(current_dir):
            return False
        return True
    def _create_default_config(self):
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
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    def _precache_frequent_sections(self):
        frequent_sections = ['SystemConstants', 'Learning', 'EnhancedNodes', 'Homeostasis']
        for section in frequent_sections:
            if self.config.has_section(section):
                self.get_section(section)
    def get(self, section: str, key: str, default: Any = None, value_type: type = str) -> Any:

        cache_key = f"{section}.{key}"
        if cache_key in self._section_cache:
            cached_value = self._section_cache[cache_key]
            if isinstance(cached_value, value_type):
                return cached_value
            try:
                if value_type == bool:
                    return cached_value.lower() in ('true', '1', 'yes', 'on') if isinstance(cached_value, str) else bool(cached_value)
                elif value_type == int:
                    return int(cached_value)
                elif value_type == float:
                    return float(cached_value)
                else:
                    return cached_value
            except (ValueError, TypeError):
                pass
        try:
            if not self.config.has_section(section):
                return default
            if not self.config.has_option(section, key):
                return default
            value = self.config.get(section, key)
            if value_type == bool:
                result = value.lower() in ('true', '1', 'yes', 'on')
            elif value_type == int:
                result = int(value)
            elif value_type == float:
                result = float(value)
            else:
                result = value
            self._section_cache[cache_key] = result
            return result
        except (ValueError, TypeError):
            return default
    def get_float(self, section: str, key: str, default: float = 0.0) -> float:
        return self.get(section, key, default, float)
    def get_int(self, section: str, key: str, default: int = 0) -> int:
        return self.get(section, key, default, int)
    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        return self.get(section, key, default, bool)
    def get_section(self, section: str) -> Dict[str, Any]:

        current_time = time.time()
        if (section in self._section_cache and
            current_time - self._last_cache_update < self._cache_ttl):
            return self._section_cache[section]
        if not self.config.has_section(section):
            return {}
        result = {}
        try:
            for key in self.config.options(section):
                value = self.config.get(section, key)
                try:
                    if '.' in value:
                        result[key] = float(value)
                    else:
                        result[key] = int(value)
                except (ValueError, TypeError):
                    result[key] = value
        except Exception as e:
            logging.warning(f"Error accessing config section {section}: {e}")
            return {}
        self._section_cache[section] = result
        self._last_cache_update = current_time
        return result
    def set(self, section: str, key: str, value: Any):

        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
    def save(self):
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    def reload(self):
        self._load_config()
    def get_enhanced_nodes_config(self) -> Dict[str, float]:
        return self.get_section('EnhancedNodes')
    def get_learning_config(self) -> Dict[str, float]:
        return self.get_section('Learning')
    def get_homeostasis_config(self) -> Dict[str, float]:
        return self.get_section('Homeostasis')
    def get_network_metrics_config(self) -> Dict[str, Union[int, float]]:
        return self.get_section('NetworkMetrics')
    def get_workspace_config(self) -> Dict[str, float]:
        return self.get_section('Workspace')
    def get_processing_config(self) -> Dict[str, float]:
        return self.get_section('Processing')
    def get_system_constants(self) -> Dict[str, float]:
        return self.get_section('SystemConstants')
config = ConfigManager()


def get_config(section: str, key: str, default: Any = None, value_type: type = str) -> Any:
    return config.get(section, key, default, value_type)


def get_enhanced_nodes_config() -> Dict[str, float]:
    return config.get_enhanced_nodes_config()


def get_learning_config() -> Dict[str, float]:
    return config.get_learning_config()


def get_homeostasis_config() -> Dict[str, float]:
    return config.get_homeostasis_config()


def get_network_metrics_config() -> Dict[str, Union[int, float]]:
    return config.get_network_metrics_config()


def get_workspace_config() -> Dict[str, float]:
    return config.get_workspace_config()


def get_processing_config() -> Dict[str, Union[int, float]]:
    return config.get_processing_config()


def get_system_constants() -> Dict[str, float]:
    return config.get_system_constants()
