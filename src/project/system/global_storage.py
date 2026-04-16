# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Classes:
#   GlobalStorage:
#     Singleton-pattern class (no instantiation); use class methods directly.
#
#     _storage: dict[str, Any]                     (class variable)
#     _device: str = 'cpu'                         (class variable)
#     _initialized: bool = False                   (class variable)
#
#     __init__(self) -> None
#       Raises TypeError — use class methods directly
#
#     get_device() -> str                          @classmethod
#       Get current device configuration
#
#     set_device(device: str) -> None              @classmethod
#       Set device configuration
#
#     store(key: str, value: Any) -> None          @classmethod
#       Store a value in global storage
#
#     retrieve(key: str, default: Any = None) -> Any    @classmethod
#       Retrieve a value from global storage
#
#     has_key(key: str) -> bool                    @classmethod
#       Check if a key exists
#
#     remove(key: str) -> bool                     @classmethod
#       Remove a key from storage
#
#     clear() -> None                              @classmethod
#       Clear all stored values
#
#     initialize(device: str = 'cpu') -> None      @classmethod
#       Initialize global storage with defaults
#
#     ensure_device_attribute() -> str             @classmethod
#       Ensure device attribute exists and return it
#
#     get_storage_info() -> dict[str, Any]         @classmethod
#       Get storage state information
#
#     __str__() -> str                             @classmethod
#       String representation
#
#     __repr__() -> str                            @classmethod
#       Detailed string representation
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [minor] If needed, store cube parameters (N, device) for shared Taichi init
#   ordering after cube migration.
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Centralized global storage for system-wide state and configuration."""

from typing import Any
import logging

logger = logging.getLogger(__name__)

class GlobalStorage:
    """
    Global system storage for shared state and configuration.

    This class provides a central repository for system-wide settings,
    device management, and shared resources across all system components.
    """

    _storage: dict[str, Any] = {}
    _device: str = 'cpu'
    _initialized: bool = False

    def __init__(self) -> None:
        """Do not instantiate — use class methods directly."""
        raise TypeError(
            "GlobalStorage should not be instantiated. Use class methods directly "
            "(e.g. GlobalStorage.store(), GlobalStorage.retrieve())."
        )

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

    @classmethod
    def __str__(cls) -> str:
        """String representation of GlobalStorage."""
        return f"GlobalStorage(device={cls._device}, initialized={cls._initialized}, items={len(cls._storage)})"

    @classmethod
    def __repr__(cls) -> str:
        """Detailed string representation of GlobalStorage."""
        return cls.__str__()
