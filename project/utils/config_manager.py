import os
import json
import shutil
from datetime import datetime
from .error_handler import ErrorHandler

class ConfigManager:
    def __init__(self, config_file='dgl_config.json'):
        self.config_file = config_file
        self.config = {
            'version': '1.0',
            'sensory': {
                'enabled': True,
                'width': 64,
                'height': 64,
                'canvas_width': 192,
                'canvas_height': 108
            },
            'workspace': {
                'width': 16,
                'height': 16,
                'canvas_width': 800,
                'canvas_height': 600
            },
            'system': {
                'update_interval': 100,
                'energy_pulse': 10.0,
                'max_energy': 100.0,
                'min_energy': 0.0
            }
        }
        self.load_config()

    @staticmethod
    def validate_config(config):
        """Validate configuration values"""
        try:
            # Validate version
            if 'version' not in config:
                raise ValueError("Missing version in config")

            # Validate sensory config
            sensory = config.get('sensory', {})
            if not isinstance(sensory.get('enabled'), bool):
                raise ValueError("Sensory enabled must be boolean")
            for key in ['width', 'height', 'canvas_width', 'canvas_height']:
                if not isinstance(sensory.get(key), int) or sensory.get(key) <= 0:
                    raise ValueError(f"Sensory {key} must be positive integer")

            # Validate workspace config
            workspace = config.get('workspace', {})
            for key in ['width', 'height', 'canvas_width', 'canvas_height']:
                if not isinstance(workspace.get(key), int) or workspace.get(key) <= 0:
                    raise ValueError(f"Workspace {key} must be positive integer")

            # Validate system config
            system = config.get('system', {})
            if not isinstance(system.get('update_interval'), int) or system.get('update_interval') < 1:
                raise ValueError("Update interval must be positive integer")
            for key in ['energy_pulse', 'max_energy', 'min_energy']:
                if not isinstance(system.get(key), (int, float)) or system.get(key) < 0:
                    raise ValueError(f"System {key} must be non-negative number")
            if system.get('min_energy') >= system.get('max_energy'):
                raise ValueError("Min energy must be less than max energy")

            return True
        except Exception as e:
            ErrorHandler.show_error("Config Validation Error", str(e))
            return False

    def create_backup(self):
        """Create a backup of the current config file"""
        if os.path.exists(self.config_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{self.config_file}.{timestamp}.bak"
            try:
                shutil.copy2(self.config_file, backup_file)
                ErrorHandler.log_info(f"Created config backup: {backup_file}")
                return True
            except Exception as e:
                ErrorHandler.show_error("Backup Error", f"Failed to create backup: {str(e)}")
                return False
        return True

    def save_config(self):
        """Save configuration to file with backup"""
        try:
            if self.validate_config(self.config):
                if self.create_backup():
                    with open(self.config_file, 'w') as f:
                        json.dump(self.config, f, indent=4)
                    ErrorHandler.log_info("Config saved successfully")
                    return True
            return False
        except Exception as e:
            ErrorHandler.show_error("Config Save Error", f"Failed to save config: {str(e)}")
            return False

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                if self.validate_config(loaded_config):
                    self.config = loaded_config
                    ErrorHandler.log_info("Config loaded successfully")
                    return True
            return False
        except Exception as e:
            ErrorHandler.show_error("Config Load Error", f"Failed to load config: {str(e)}")
            return False

    def update_config(self, section, key, value):
        """Update a configuration value"""
        try:
            if section in self.config and key in self.config[section]:
                old_value = self.config[section][key]
                self.config[section][key] = value
                if not self.validate_config(self.config):
                    self.config[section][key] = old_value
                    return False
                return self.save_config()
            return False
        except Exception as e:
            ErrorHandler.show_error("Config Update Error", f"Failed to update config: {str(e)}")
            return False

    def get_config(self, section=None, key=None):
        """Get configuration value(s)"""
        try:
            if section is None:
                return self.config
            if key is None:
                return self.config.get(section, {})
            return self.config.get(section, {}).get(key)
        except Exception as e:
            ErrorHandler.show_error("Config Get Error", f"Failed to get config value: {str(e)}")
            return None 
