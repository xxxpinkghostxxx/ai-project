"""
Global Storage Module.

This module provides centralized storage for system-wide state and configuration,
including device management, global settings, and shared resources.
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)

class GlobalStorage:
    """
    Global system storage for shared state and configuration.

    This class provides a central repository for system-wide settings,
    device management, and shared resources across all system components.
    """

    # Class variable to store singleton instance data
    _storage: dict[str, Any] = {}
    _device: str = 'cpu'
    _initialized: bool = False

    def __init__(self, device: str = 'cpu') -> None:
        """Initialize GlobalStorage with device configuration."""
        # This is only for backwards compatibility - use class methods instead
        logger.warning("GlobalStorage should not be instantiated. Use class methods instead.")

    @classmethod
    def get_device(cls) -> str:
        """Get the current device configuration."""
        return cls._device

    @classmethod
    def set_device(cls, device: str) -> None:
        """Set the device configuration."""
        old_device = cls._device
        cls._device = device
        logger.info(f"Device changed from {old_device} to {cls._device}")

    @classmethod
    def store(cls, key: str, value: Any) -> None:
        """Store a value in global storage."""
        cls._storage[key] = value
        logger.debug(f"Stored {key} in GlobalStorage")

    @classmethod
    def retrieve(cls, key: str, default: Any = None) -> Any:
        """Retrieve a value from global storage."""
        return cls._storage.get(key, default)

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a value from global storage (alias for retrieve)."""
        return cls._storage.get(key, default)

    @classmethod
    def has_key(cls, key: str) -> bool:
        """Check if a key exists in global storage."""
        return key in cls._storage

    @classmethod
    def remove(cls, key: str) -> bool:
        """Remove a key from global storage."""
        if key in cls._storage:
            del cls._storage[key]
            logger.debug(f"Removed {key} from GlobalStorage")
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all stored values."""
        cls._storage.clear()
        logger.debug("GlobalStorage cleared")

    @classmethod
    def initialize(cls, device: str = 'cpu') -> None:
        """Initialize global storage with default values."""
        if not cls._initialized:
            cls._device = device
            cls._storage = {}
            cls._initialized = True
            # Store device in storage for consistency
            cls.store('device', device)
            logger.info(f"GlobalStorage initialization completed with device: {device}")

    @classmethod
    def ensure_device_attribute(cls) -> str:
        """Ensure device attribute exists and return it."""
        if not hasattr(cls, '_device') or not cls._device:
            cls._device = 'cpu'
            logger.warning("Device attribute missing, defaulted to 'cpu'")
        return cls._device

    @classmethod
    def get_storage_info(cls) -> dict[str, Any]:
        """Get information about the global storage state."""
        return {
            'device': cls._device,
            'initialized': cls._initialized,
            'keys': list(cls._storage.keys()),
            'storage_size': len(cls._storage)
        }

    def __str__(self) -> str:
        """String representation of GlobalStorage."""
        return f"GlobalStorage(device={self._device}, initialized={self._initialized}, items={len(self._storage)})"

    def __repr__(self) -> str:
        """Detailed string representation of GlobalStorage."""
        return self.__str__()
