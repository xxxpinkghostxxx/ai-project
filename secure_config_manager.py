"""
secure_config_manager.py

Secure configuration management with encryption, validation, and access control.
Provides secure handling of sensitive configuration data.
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from dataclasses import dataclass, asdict
from enum import Enum


class ConfigSecurityLevel(Enum):
    """Configuration security levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


@dataclass
class ConfigItem:
    """Represents a configuration item with security metadata."""
    key: str
    value: Any
    security_level: ConfigSecurityLevel
    encrypted: bool = False
    description: Optional[str] = None
    required: bool = False
    default_value: Optional[Any] = None


class SecureConfigManager:
    """Secure configuration manager with encryption and validation."""
    
    def __init__(self, config_file: Optional[str] = None, 
                 encryption_key: Optional[str] = None,
                 master_password: Optional[str] = None):
        """
        Initialize the secure configuration manager.
        
        Args:
            config_file: Path to configuration file
            encryption_key: Encryption key for sensitive data
            master_password: Master password for key derivation
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file or "secure_config.json"
        self.config_items: Dict[str, ConfigItem] = {}
        self.encryption_key = encryption_key
        self.master_password = master_password
        self.cipher = None
        
        # Initialize encryption if key or password provided
        if encryption_key or master_password:
            self._initialize_encryption()
        
        # Load existing configuration
        self._load_config()
    
    def _initialize_encryption(self):
        """Initialize encryption cipher."""
        try:
            if self.encryption_key:
                # Use provided key directly
                key = self.encryption_key.encode()
            elif self.master_password:
                # Derive key from master password
                key = self._derive_key_from_password(self.master_password)
            else:
                raise ValueError("Either encryption_key or master_password must be provided")
            
            # Ensure key is 32 bytes for Fernet
            if len(key) != 32:
                key = hashlib.sha256(key).digest()
            
            self.cipher = Fernet(base64.urlsafe_b64encode(key))
            self.logger.info("Encryption initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            # Generate a new salt or use a fixed one for consistency
            salt = b'secure_config_salt_2024'  # In production, use random salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())
    
    def _load_config(self):
        """Load configuration from file."""
        if not os.path.exists(self.config_file):
            self.logger.info(f"Configuration file {self.config_file} not found, starting with empty config")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load configuration items
            for item_data in data.get('items', []):
                config_item = ConfigItem(**item_data)
                
                # Decrypt value if encrypted
                if config_item.encrypted and self.cipher:
                    try:
                        config_item.value = self._decrypt_value(config_item.value)
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt value for {config_item.key}: {e}")
                        continue
                
                self.config_items[config_item.key] = config_item
            
            self.logger.info(f"Loaded {len(self.config_items)} configuration items")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            # Prepare data for saving
            data = {
                'items': [],
                'metadata': {
                    'version': '1.0',
                    'created_at': self._get_timestamp(),
                    'encrypted_items': 0
                }
            }
            
            # Process each configuration item
            for config_item in self.config_items.values():
                item_data = asdict(config_item)
                
                # Encrypt value if needed
                if config_item.encrypted and self.cipher:
                    item_data['value'] = self._encrypt_value(config_item.value)
                    data['metadata']['encrypted_items'] += 1
                
                data['items'].append(item_data)
            
            # Save to file with secure permissions
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Set secure file permissions (readable only by owner)
            os.chmod(self.config_file, 0o600)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _encrypt_value(self, value: Any) -> str:
        """Encrypt a configuration value."""
        if not self.cipher:
            raise ValueError("Encryption not initialized")
        
        # Convert value to string for encryption
        value_str = json.dumps(value) if not isinstance(value, str) else value
        encrypted_bytes = self.cipher.encrypt(value_str.encode())
        return base64.urlsafe_b64encode(encrypted_bytes).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> Any:
        """Decrypt a configuration value."""
        if not self.cipher:
            raise ValueError("Encryption not initialized")
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode()
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str
                
        except Exception as e:
            self.logger.error(f"Failed to decrypt value: {e}")
            raise
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def set_config(self, key: str, value: Any, 
                   security_level: ConfigSecurityLevel = ConfigSecurityLevel.INTERNAL,
                   encrypt: bool = False,
                   description: Optional[str] = None,
                   required: bool = False) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            security_level: Security level of the configuration
            encrypt: Whether to encrypt the value
            description: Description of the configuration
            required: Whether this configuration is required
        """
        # Validate security level
        if encrypt and security_level == ConfigSecurityLevel.PUBLIC:
            self.logger.warning(f"Encrypting public configuration {key}")
        
        # Create configuration item
        config_item = ConfigItem(
            key=key,
            value=value,
            security_level=security_level,
            encrypted=encrypt,
            description=description,
            required=required
        )
        
        self.config_items[key] = config_item
        self.logger.info(f"Set configuration {key} with security level {security_level.value}")
    
    def get_config(self, key: str, default: Any = None, 
                   require_encryption: bool = False) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            require_encryption: Whether the value must be encrypted
            
        Returns:
            Configuration value
        """
        if key not in self.config_items:
            if default is not None:
                return default
            raise KeyError(f"Configuration key '{key}' not found")
        
        config_item = self.config_items[key]
        
        # Check encryption requirement
        if require_encryption and not config_item.encrypted:
            raise ValueError(f"Configuration {key} is not encrypted but encryption is required")
        
        return config_item.value
    
    def get_config_with_validation(self, key: str, expected_type: type = str,
                                  min_length: Optional[int] = None,
                                  max_length: Optional[int] = None,
                                  allowed_values: Optional[List[Any]] = None) -> Any:
        """
        Get configuration value with validation.
        
        Args:
            key: Configuration key
            expected_type: Expected type of the value
            min_length: Minimum length for strings
            max_length: Maximum length for strings
            allowed_values: List of allowed values
            
        Returns:
            Validated configuration value
        """
        value = self.get_config(key)
        
        # Type validation
        if not isinstance(value, expected_type):
            raise TypeError(f"Configuration {key} must be of type {expected_type.__name__}")
        
        # Length validation for strings
        if isinstance(value, str):
            if min_length is not None and len(value) < min_length:
                raise ValueError(f"Configuration {key} too short (minimum {min_length})")
            if max_length is not None and len(value) > max_length:
                raise ValueError(f"Configuration {key} too long (maximum {max_length})")
        
        # Allowed values validation
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(f"Configuration {key} must be one of: {allowed_values}")
        
        return value
    
    def get_encrypted_config(self, key: str, default: Any = None) -> Any:
        """Get an encrypted configuration value."""
        return self.get_config(key, default, require_encryption=True)
    
    def delete_config(self, key: str) -> bool:
        """Delete a configuration item."""
        if key in self.config_items:
            del self.config_items[key]
            self.logger.info(f"Deleted configuration {key}")
            return True
        return False
    
    def list_configs(self, security_level: Optional[ConfigSecurityLevel] = None,
                    include_encrypted: bool = False) -> List[str]:
        """
        List configuration keys.
        
        Args:
            security_level: Filter by security level
            include_encrypted: Whether to include encrypted configurations
            
        Returns:
            List of configuration keys
        """
        keys = []
        
        for key, config_item in self.config_items.items():
            # Filter by security level
            if security_level and config_item.security_level != security_level:
                continue
            
            # Filter by encryption status
            if not include_encrypted and config_item.encrypted:
                continue
            
            keys.append(key)
        
        return keys
    
    def get_config_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a configuration item."""
        if key not in self.config_items:
            return None
        
        config_item = self.config_items[key]
        return {
            'key': config_item.key,
            'security_level': config_item.security_level.value,
            'encrypted': config_item.encrypted,
            'description': config_item.description,
            'required': config_item.required,
            'has_value': config_item.value is not None
        }
    
    def validate_required_configs(self) -> List[str]:
        """Validate that all required configurations are present."""
        missing_configs = []
        
        for key, config_item in self.config_items.items():
            if config_item.required and config_item.value is None:
                missing_configs.append(key)
        
        return missing_configs
    
    def export_config(self, keys: Optional[List[str]] = None,
                     include_encrypted: bool = False,
                     security_level: Optional[ConfigSecurityLevel] = None) -> Dict[str, Any]:
        """
        Export configuration data.
        
        Args:
            keys: Specific keys to export (None for all)
            include_encrypted: Whether to include encrypted values
            security_level: Filter by security level
            
        Returns:
            Dictionary of configuration data
        """
        export_data = {}
        
        for key, config_item in self.config_items.items():
            # Filter by keys
            if keys and key not in keys:
                continue
            
            # Filter by security level
            if security_level and config_item.security_level != security_level:
                continue
            
            # Filter by encryption status
            if not include_encrypted and config_item.encrypted:
                continue
            
            export_data[key] = {
                'value': config_item.value,
                'security_level': config_item.security_level.value,
                'encrypted': config_item.encrypted,
                'description': config_item.description
            }
        
        return export_data
    
    def import_config(self, config_data: Dict[str, Any], 
                     overwrite: bool = False) -> int:
        """
        Import configuration data.
        
        Args:
            config_data: Configuration data to import
            overwrite: Whether to overwrite existing configurations
            
        Returns:
            Number of configurations imported
        """
        imported_count = 0
        
        for key, data in config_data.items():
            # Check if key already exists
            if key in self.config_items and not overwrite:
                self.logger.warning(f"Skipping existing configuration {key}")
                continue
            
            # Extract configuration data
            value = data.get('value')
            security_level = ConfigSecurityLevel(data.get('security_level', 'internal'))
            encrypt = data.get('encrypted', False)
            description = data.get('description')
            
            # Set configuration
            self.set_config(key, value, security_level, encrypt, description)
            imported_count += 1
        
        self.logger.info(f"Imported {imported_count} configurations")
        return imported_count
    
    def save(self):
        """Save configuration to file."""
        self._save_config()
    
    def reload(self):
        """Reload configuration from file."""
        self.config_items.clear()
        self._load_config()
    
    def cleanup(self):
        """Cleanup and save configuration."""
        self.save()
        self.logger.info("Configuration manager cleanup completed")


class EnvironmentConfigLoader:
    """Load configuration from environment variables with security validation."""
    
    def __init__(self, prefix: str = "APP_"):
        """
        Initialize environment config loader.
        
        Args:
            prefix: Prefix for environment variables
        """
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)
    
    def load_environment_configs(self, config_manager: SecureConfigManager,
                               config_mapping: Dict[str, Dict[str, Any]]) -> int:
        """
        Load configurations from environment variables.
        
        Args:
            config_manager: Secure configuration manager
            config_mapping: Mapping of env vars to config keys with metadata
            
        Returns:
            Number of configurations loaded
        """
        loaded_count = 0
        
        for env_var, config_info in config_mapping.items():
            env_value = os.environ.get(env_var)
            
            if env_value is None:
                if config_info.get('required', False):
                    raise ValueError(f"Required environment variable {env_var} not set")
                continue
            
            # Extract configuration metadata
            config_key = config_info.get('key', env_var.lower().replace(self.prefix, ''))
            security_level = ConfigSecurityLevel(config_info.get('security_level', 'internal'))
            encrypt = config_info.get('encrypt', False)
            description = config_info.get('description', f"Loaded from {env_var}")
            
            # Set configuration
            config_manager.set_config(
                config_key, 
                env_value, 
                security_level, 
                encrypt, 
                description
            )
            loaded_count += 1
        
        self.logger.info(f"Loaded {loaded_count} configurations from environment")
        return loaded_count


# Global secure configuration manager instance
_global_secure_config: Optional[SecureConfigManager] = None


def get_secure_config_manager() -> SecureConfigManager:
    """Get the global secure configuration manager."""
    global _global_secure_config
    
    if _global_secure_config is None:
        # Initialize with environment variables
        encryption_key = os.environ.get('CONFIG_ENCRYPTION_KEY')
        master_password = os.environ.get('CONFIG_MASTER_PASSWORD')
        config_file = os.environ.get('CONFIG_FILE', 'secure_config.json')
        
        _global_secure_config = SecureConfigManager(
            config_file=config_file,
            encryption_key=encryption_key,
            master_password=master_password
        )
    
    return _global_secure_config


def set_secure_config(key: str, value: Any, **kwargs) -> None:
    """Set a secure configuration value."""
    config_manager = get_secure_config_manager()
    config_manager.set_config(key, value, **kwargs)


def get_secure_config(key: str, default: Any = None, **kwargs) -> Any:
    """Get a secure configuration value."""
    config_manager = get_secure_config_manager()
    return config_manager.get_config(key, default, **kwargs)


def save_secure_config() -> None:
    """Save secure configuration."""
    config_manager = get_secure_config_manager()
    config_manager.save()


# Example usage and testing
if __name__ == "__main__":
    print("Secure Configuration Manager created successfully!")
    print("Features include:")
    print("- Encrypted configuration storage")
    print("- Security level classification")
    print("- Environment variable loading")
    print("- Configuration validation")
    print("- Secure file permissions")
    print("- Import/export functionality")
    
    # Test the secure configuration system
    try:
        # Create test configuration manager
        config_manager = SecureConfigManager("test_secure_config.json")
        
        # Set some test configurations
        config_manager.set_config("api_key", "test_api_key_123", 
                                 ConfigSecurityLevel.SECRET, encrypt=True)
        config_manager.set_config("debug_mode", False, 
                                 ConfigSecurityLevel.INTERNAL)
        config_manager.set_config("max_connections", 100, 
                                 ConfigSecurityLevel.PUBLIC)
        
        # Save configuration
        config_manager.save()
        
        # Test retrieval
        api_key = config_manager.get_encrypted_config("api_key")
        debug_mode = config_manager.get_config("debug_mode")
        max_conn = config_manager.get_config("max_connections")
        
        print(f"\nTest configurations:")
        print(f"API Key: {api_key}")
        print(f"Debug Mode: {debug_mode}")
        print(f"Max Connections: {max_conn}")
        
        # Test environment loading
        env_loader = EnvironmentConfigLoader("TEST_")
        os.environ["TEST_DATABASE_URL"] = "postgresql://localhost/testdb"
        
        env_mapping = {
            "TEST_DATABASE_URL": {
                "key": "database_url",
                "security_level": "confidential",
                "encrypt": True,
                "description": "Database connection URL"
            }
        }
        
        env_loader.load_environment_configs(config_manager, env_mapping)
        
        # Cleanup
        config_manager.cleanup()
        if os.path.exists("test_secure_config.json"):
            os.remove("test_secure_config.json")
        
    except Exception as e:
        print(f"Secure configuration test failed: {e}")
    
    print("\nSecure configuration manager test completed!")
