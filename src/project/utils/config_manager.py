"""
Configuration Manager Module.

This module provides configuration management functionality for the Energy-Based Neural System,
including loading, saving, validating, and managing configuration files with
optimized backup strategy to prevent excessive backups during error conditions.
"""

import os
import json
import shutil
import time
import logging
from datetime import datetime
from typing import Any, cast
from collections import deque
from .error_handler import ErrorHandler
from .security_utils import ConfigurationSecurityValidator, SecuritySanitizer, SecureLogger
from .performance_utils import monitor_performance

# Logger for this module
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration Manager class for handling system configuration."""

    def __init__(self: 'ConfigManager', config_file: str | None = None) -> None:
        """Initialize ConfigManager with default configuration.
        
        Always uses src/project/pyg_config.json - never writes to root directory.
        """
        # Always use the config file in src/project/ directory
        if config_file is None:
            # Get the directory where this file is located (src/project/utils/)
            # Then go up one level to src/project/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(current_dir)  # Go up from utils/ to project/
            config_file = os.path.join(project_dir, 'pyg_config.json')
        
        # Normalize the path (resolve to absolute, handle separators correctly)
        config_file = os.path.normpath(os.path.abspath(config_file))
        
        # DO NOT sanitize full paths - sanitize_filename removes path separators!
        # Only sanitize the filename portion if needed
        # For absolute paths, just use them as-is (they're already validated by os.path)
        self.config_file = config_file
        self.config: dict[str, Any] = {
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
                'canvas_height': 600,
                'enabled': True,
                'reading_interval_ms': 50,
                'energy_smoothing': True,
                'smoothing_factor': 0.1,
                'shading_mode': 'linear',
                'color_scheme': 'grayscale',
                'animation_enabled': True,
                'animation_speed': 0.1,
                'batch_updates': True,
                'max_fps': 60,
                'memory_optimization': True,
                'cache_size': 1000,
                'cache_validity_ms': 100,
                'retry_attempts': 3,
                'retry_delay_ms': 10,
                'error_threshold': 0.1
            },
            'system': {
                'update_interval': 100,
                'energy_pulse': 10.0,
                'max_energy': 100.0,
                'min_energy': 0.0,
                'frame_throttling': True,
                'detailed_logging': False
            }
        }

        # Backup rate limiting and error condition detection
        self._last_backup_time: float = 0
        self._backup_rate_limit: float = 300.0  # 5 minutes between backups
        self._error_condition_active: bool = False
        self._error_timestamp: float = 0
        self._error_cooldown: float = 1800.0  # 30 minutes error cooldown period
        self._recent_errors: deque[str] = deque(maxlen=10)  # Track recent errors

        self.load_config()



    def _check_error_condition(self: 'ConfigManager') -> bool:
        """Check if system is in error condition and should suppress backups"""
        current_time = time.time()

        # Check if we're still in error cooldown period
        if self._error_condition_active and (current_time - self._error_timestamp) < self._error_cooldown:
            return True

        # Check if we have recent errors that indicate error condition
        if len(self._recent_errors) >= 3:  # 3 errors in short period indicates error condition
            self._error_condition_active = True
            self._error_timestamp = current_time
            return True

        return False

    def _check_backup_rate_limit(self: 'ConfigManager') -> bool:
        """Check if backup rate limit allows creating a new backup"""
        current_time = time.time()
        if (current_time - self._last_backup_time) < self._backup_rate_limit:
            return False  # Rate limit not reached
        return True  # Rate limit reached, can create backup

    def _update_backup_timestamp(self: 'ConfigManager') -> None:
        """Update the last backup timestamp"""
        self._last_backup_time = time.time()

    def _record_error(self: 'ConfigManager', error_message: str) -> None:
        """Record an error for error condition detection"""
        current_time = time.time()
        self._recent_errors.append(error_message)

        # If we get multiple errors in short time, activate error condition
        if len(self._recent_errors) >= 3 and (current_time - self._error_timestamp) < 60:  # 3 errors in 60 seconds
            self._error_condition_active = True
            self._error_timestamp = current_time

    def create_backup(self: 'ConfigManager') -> bool:
        """Create a backup of the current config file with rate limiting and error condition detection"""
        if not self.config_file or not os.path.exists(self.config_file):
            return True

        # Check if we should suppress backups due to error conditions
        if self._check_error_condition():
            ErrorHandler.log_warning("Backup suppressed due to error condition")
            return True

        # Check rate limiting
        if not self._check_backup_rate_limit():
            ErrorHandler.log_warning("Backup skipped due to rate limiting")
            return True

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{self.config_file}.{timestamp}.bak"
        try:
            shutil.copy2(self.config_file, backup_file)
            ErrorHandler.log_info(f"Created config backup: {backup_file}")
            self._update_backup_timestamp()
            return True
        except Exception as e:
            ErrorHandler.show_error("Backup Error", f"Failed to create backup: {str(e)}")
            self._record_error(f"Backup failed: {str(e)}")
            return False

    def save_config(self: 'ConfigManager') -> bool:
        """Save configuration to file with optimized backup strategy"""
        try:
            if not self.validate_config(self.config):
                return False

            # Create backup with rate limiting and error condition detection
            if not self.create_backup():
                # If backup fails but we're not in error condition, still try to save
                if not self._error_condition_active:
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, indent=4)
                    ErrorHandler.log_info("Config saved successfully (backup skipped)")
                    return True
                return False

            # Normal save with successful backup
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            ErrorHandler.log_info("Config saved successfully")
            return True

        except Exception as e:
            ErrorHandler.show_error("Config Save Error", f"Failed to save config: {str(e)}")
            self._record_error(f"Config save failed: {str(e)}")
            return False

    def load_config(self: 'ConfigManager') -> bool:
        """Load configuration from file"""
        try:
            if self.config_file and os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                if self.validate_config(loaded_config):
                    self.config = loaded_config
                    ErrorHandler.log_info("Config loaded successfully")
                    return True
            return False
        except Exception as e:
            ErrorHandler.show_error("Config Load Error", f"Failed to load config: {str(e)}")
            return False

    @monitor_performance("config_update")
    def update_config(self: 'ConfigManager', section: str, key: str, value: Any) -> bool:
        """Update a configuration value with security validation and performance monitoring"""
        try:
            logger.info(f"ConfigManager: Attempting to update {section}.{key} to {value}")
            print(f"Debug ConfigManager: Attempting to update {section}.{key} to {value}")
            
            # Sanitize inputs
            section = SecuritySanitizer.sanitize_filename(section)
            key = SecuritySanitizer.sanitize_filename(key)
            # Validate configuration key-value pair for security constraint
            full_key = f"{section}_{key}"
            logger.info(f"ConfigManager: Security validation for key: {full_key}")
            print(f"Debug ConfigManager: Security validation for key: {full_key}")
            
            is_valid, error_msg = ConfigurationSecurityValidator.validate_config_value(full_key, value)
            if not is_valid:
                SecureLogger.secure_error(logger, f"Security validation failed: {error_msg}")
                print(f"Debug ConfigManager: Security validation failed: {error_msg}")
                return False
            
            logger.info(f"ConfigManager: Security validation passed for {full_key}")
            print(f"Debug ConfigManager: Security validation passed for {full_key}")
            
            if section in self.config:
                section_dict = self.config[section]
                if isinstance(section_dict, dict) and key in section_dict:
                    old_value: Any = section_dict[key]  # type: ignore[assignment]
                    section_dict[key] = value
                    if not self.validate_config(self.config):
                        section_dict[key] = old_value
                        logger.error(f"ConfigManager: Configuration validation failed for {section}.{key}")
                        print(f"Debug ConfigManager: Configuration validation failed for {section}.{key}")
                        return False
                    return self.save_config()
                else:
                    logger.error(f"ConfigManager: Key {key} not found in section {section}")
                    print(f"Debug ConfigManager: Key {key} not found in section {section}")
                    return False
            else:
                logger.error(f"ConfigManager: Section {section} not found")
                print(f"Debug ConfigManager: Section {section} not found")
                return False
        except Exception as e:
            SecureLogger.secure_error(logger, f"Config update error: {str(e)}")
            print(f"Debug ConfigManager: Exception in update_config: {str(e)}")
            return False
    def validate_config(self: 'ConfigManager', config: dict[str, Any]) -> bool:
        """Enhanced config validation with security constraints and detailed error messages"""
        try:
            # Basic validation
            if not config:  # Check if config is empty or None
                raise ValueError("Configuration cannot be empty or None")
            # Runtime type check for defensive programming (type checker knows it's dict)
            if not isinstance(config, dict):  # type: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f"Configuration must be a dictionary, got {type(config).__name__}")

            # Security validation for entire config section
            security_errors = ConfigurationSecurityValidator.validate_config_section(config)
            if security_errors:
                detailed_errors = "\n".join([f"  - {error}" for error in security_errors])
                logger.warning(f"Security validation errors:\n{detailed_errors}")
                return False

            if 'version' not in config:
                raise ValueError("Missing required 'version' field in configuration")

            # Validate sensory config with detailed error messages
            sensory_raw = config.get('sensory', {})
            if not isinstance(sensory_raw, dict):
                raise ValueError(f"Sensory configuration must be a dictionary, got {type(sensory_raw).__name__}")
            sensory = cast(dict[str, Any], sensory_raw)

            enabled = sensory.get('enabled')
            if not isinstance(enabled, bool):
                raise ValueError(f"Sensory 'enabled' must be boolean, got {type(enabled).__name__}")

            for key in ['width', 'height', 'canvas_width', 'canvas_height']:
                sensory_value_raw = sensory.get(key)
                if sensory_value_raw is None:
                    raise ValueError(f"Sensory '{key}' is missing")
                if not isinstance(sensory_value_raw, int):
                    raise ValueError(f"Sensory '{key}' must be integer, got {type(sensory_value_raw).__name__}")
                sensory_value: int = sensory_value_raw
                if sensory_value <= 0:
                    raise ValueError(f"Sensory '{key}' must be positive integer, got {sensory_value}")

            # Validate workspace config with detailed error messages
            workspace_raw = config.get('workspace', {})
            if not isinstance(workspace_raw, dict):
                raise ValueError(f"Workspace configuration must be a dictionary, got {type(workspace_raw).__name__}")
            workspace = cast(dict[str, Any], workspace_raw)

            for key in ['width', 'height', 'canvas_width', 'canvas_height']:
                workspace_value_raw = workspace.get(key)
                if workspace_value_raw is None:
                    raise ValueError(f"Workspace '{key}' is missing")
                if not isinstance(workspace_value_raw, int):
                    raise ValueError(f"Workspace '{key}' must be integer, got {type(workspace_value_raw).__name__}")
                workspace_value: int = workspace_value_raw
                if workspace_value <= 0:
                    raise ValueError(f"Workspace '{key}' must be positive integer, got {workspace_value}")

            # Validate system config with detailed error messages
            system_raw = config.get('system', {})
            if not isinstance(system_raw, dict):
                raise ValueError(f"System configuration must be a dictionary, got {type(system_raw).__name__}")
            system = cast(dict[str, Any], system_raw)

            update_interval_raw = system.get('update_interval')
            if update_interval_raw is None:
                raise ValueError("System 'update_interval' is missing")
            if not isinstance(update_interval_raw, int):
                raise ValueError(f"System 'update_interval' must be integer, got {type(update_interval_raw).__name__}")
            update_interval: int = update_interval_raw
            if update_interval < 1:
                raise ValueError(f"System 'update_interval' must be positive integer, got {update_interval}")

            for key in ['energy_pulse', 'max_energy', 'min_energy']:
                value_raw = system.get(key)
                if value_raw is None:
                    raise ValueError(f"System '{key}' is missing")
                if not isinstance(value_raw, (int, float)):
                    raise ValueError(f"System '{key}' must be number, got {type(value_raw).__name__}")
                value: int | float = value_raw
                if value < 0:
                    raise ValueError(f"System '{key}' must be non-negative, got {value}")

            min_energy_raw = system.get('min_energy')
            max_energy_raw = system.get('max_energy')
            if min_energy_raw is not None and max_energy_raw is not None:
                min_energy: int | float = min_energy_raw
                max_energy: int | float = max_energy_raw
                if min_energy >= max_energy:
                    raise ValueError(f"System 'min_energy' ({min_energy}) must be less than 'max_energy' ({max_energy})")

            # Validate optional boolean fields
            for key in ['frame_throttling', 'detailed_logging']:
                value_raw = system.get(key)
                if value_raw is not None and not isinstance(value_raw, bool):
                    raise ValueError(f"System '{key}' must be boolean, got {type(value_raw).__name__}")

            logger.info("Configuration validation passed successfully")
            return True
        except Exception as e:
            logger.error(f"Configuration validation error: {str(e)}")
            return False
    def get_config(self: 'ConfigManager', section: str | None = None, key: str | None = None) -> Any:
        """Get configuration value(s)"""
        try:
            if section is None:
                return self.config
            if key is None:
                section_config = self.config.get(section, {})
                return section_config
            section_config = self.config.get(section, {})
            return section_config.get(key) if isinstance(section_config, dict) else None  # type: ignore[union-attr,return-value]
        except Exception as e:
            ErrorHandler.show_error("Config Get Error", f"Failed to get config value: {str(e)}")
            return None

    def clear_error_condition(self: 'ConfigManager') -> None:
        """Clear error condition when system stabilizes"""
        current_time = time.time()
        # Only clear error condition if we haven't had recent errors
        if not self._recent_errors or (current_time - self._error_timestamp) > self._error_cooldown:
            self._error_condition_active = False
            self._recent_errors.clear()
            ErrorHandler.log_info("Error condition cleared, backups will resume normally")

    def get_backup_status(self: 'ConfigManager') -> dict[str, Any]:
        """Get current backup system status for monitoring"""
        return {
            'error_condition_active': self._error_condition_active,
            'last_backup_time': self._last_backup_time,
            'recent_errors': list(self._recent_errors),
            'error_timestamp': self._error_timestamp,
            'backup_rate_limit': self._backup_rate_limit,
            'error_cooldown': self._error_cooldown
        }

    def cleanup_excessive_backups(self: 'ConfigManager', max_backups: int = 5) -> int:
        """Clean up excessive backup files, keeping only the most recent ones"""
        if not self.config_file:
            return 0

        # Find all backup files for this config
        backup_pattern = f"{self.config_file}.*.bak"
        import glob
        backup_files = glob.glob(backup_pattern)

        # Sort by modification time (newest first)
        backup_files.sort(key=os.path.getmtime, reverse=True)

        # Keep only the most recent backups
        files_to_delete = backup_files[max_backups:]

        deleted_count = 0
        for backup_file in files_to_delete:
            try:
                os.remove(backup_file)
                ErrorHandler.log_info(f"Deleted excessive backup: {backup_file}")
                deleted_count += 1
            except Exception as e:
                ErrorHandler.log_warning(f"Failed to delete backup {backup_file}: {str(e)}")

        if deleted_count > 0:
            ErrorHandler.log_info(f"Cleaned up {deleted_count} excessive backup files, kept {len(backup_files) - deleted_count} most recent")

        return deleted_count
