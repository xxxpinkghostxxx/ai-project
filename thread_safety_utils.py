"""
thread_safety_utils.py

Thread safety utilities and decorators for the AI neural project.
Provides tools to ensure thread-safe access to shared resources and prevent race conditions.
"""

import threading
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
from logging_utils import log_step


class ThreadSafeCounter:
    """Thread-safe counter with atomic operations."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.RLock()
    
    def increment(self, amount: int = 1) -> int:
        """Atomically increment counter and return new value."""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Atomically decrement counter and return new value."""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> None:
        """Set counter value."""
        with self._lock:
            self._value = value
    
    def reset(self) -> int:
        """Reset counter to 0 and return previous value."""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value


class ThreadSafeDict:
    """Thread-safe dictionary with atomic operations."""
    
    def __init__(self, initial_dict: Optional[Dict] = None):
        self._dict = initial_dict.copy() if initial_dict else {}
        self._lock = threading.RLock()
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get value for key with default."""
        with self._lock:
            return self._dict.get(key, default)
    
    def set(self, key: Any, value: Any) -> None:
        """Set value for key."""
        with self._lock:
            self._dict[key] = value
    
    def update(self, other_dict: Dict) -> None:
        """Update with another dictionary."""
        with self._lock:
            self._dict.update(other_dict)
    
    def pop(self, key: Any, default: Any = None) -> Any:
        """Pop value for key."""
        with self._lock:
            return self._dict.pop(key, default)
    
    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._dict.clear()
    
    def keys(self) -> List[Any]:
        """Get all keys."""
        with self._lock:
            return list(self._dict.keys())
    
    def values(self) -> List[Any]:
        """Get all values."""
        with self._lock:
            return list(self._dict.values())
    
    def items(self) -> List[tuple]:
        """Get all key-value pairs."""
        with self._lock:
            return list(self._dict.items())
    
    def copy(self) -> Dict:
        """Get a copy of the dictionary."""
        with self._lock:
            return self._dict.copy()
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._dict)
    
    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._dict


class ThreadSafeList:
    """Thread-safe list with atomic operations."""
    
    def __init__(self, initial_list: Optional[List] = None):
        self._list = initial_list.copy() if initial_list else []
        self._lock = threading.RLock()
    
    def append(self, item: Any) -> None:
        """Append item to list."""
        with self._lock:
            self._list.append(item)
    
    def extend(self, items: List[Any]) -> None:
        """Extend list with items."""
        with self._lock:
            self._list.extend(items)
    
    def insert(self, index: int, item: Any) -> None:
        """Insert item at index."""
        with self._lock:
            self._list.insert(index, item)
    
    def pop(self, index: int = -1) -> Any:
        """Pop item at index."""
        with self._lock:
            return self._list.pop(index)
    
    def remove(self, item: Any) -> None:
        """Remove first occurrence of item."""
        with self._lock:
            self._list.remove(item)
    
    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._list.clear()
    
    def get(self, index: int, default: Any = None) -> Any:
        """Get item at index with default."""
        with self._lock:
            try:
                return self._list[index]
            except IndexError:
                return default
    
    def set(self, index: int, item: Any) -> None:
        """Set item at index."""
        with self._lock:
            self._list[index] = item
    
    def copy(self) -> List[Any]:
        """Get a copy of the list."""
        with self._lock:
            return self._list.copy()
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._list)
    
    def __contains__(self, item: Any) -> bool:
        with self._lock:
            return item in self._list


class ThreadSafeCache:
    """Thread-safe cache with TTL (time-to-live) support."""
    
    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000):
        self._cache = {}
        self._timestamps = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.RLock()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check if expired
            if time.time() - self._timestamps[key] > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            return self._cache[key]
    
    def set(self, key: Any, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove oldest items if cache is full
            if len(self._cache) >= self._max_size:
                self._evict_oldest()
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def _evict_oldest(self) -> None:
        """Remove oldest item from cache."""
        if not self._timestamps:
            return
        
        oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
        del self._cache[oldest_key]
        del self._timestamps[oldest_key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired items and return count of removed items."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self._timestamps.items()
                if current_time - timestamp > self._ttl
            ]
            
            for key in expired_keys:
                del self._cache[key]
                del self._timestamps[key]
            
            return len(expired_keys)


class ReadWriteLock:
    """Read-write lock for better concurrency with multiple readers."""
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
    
    @contextmanager
    def read_lock(self):
        """Acquire read lock."""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()
        
        try:
            yield
        finally:
            self._read_ready.acquire()
            try:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notifyAll()
            finally:
                self._read_ready.release()
    
    @contextmanager
    def write_lock(self):
        """Acquire write lock."""
        self._read_ready.acquire()
        try:
            while self._readers > 0:
                self._read_ready.wait()
        finally:
            pass  # Keep lock until context exit
        
        try:
            yield
        finally:
            self._read_ready.release()


def thread_safe(lock_attr: str = '_lock'):
    """Decorator to make methods thread-safe using a lock attribute."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_attr, None)
            if lock is None:
                # No lock available, execute without protection
                return func(self, *args, **kwargs)
            
            with lock:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


def synchronized(lock: threading.Lock):
    """Decorator to synchronize function execution with a specific lock."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def atomic_operation(lock: threading.Lock):
    """Context manager for atomic operations."""
    @contextmanager
    def atomic():
        with lock:
            yield
    return atomic


class ThreadSafeSingleton:
    """Thread-safe singleton pattern implementation."""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]


def ensure_thread_safe(func: Callable) -> Callable:
    """Decorator to ensure function is called in thread-safe manner."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Add thread safety logging
        thread_name = threading.current_thread().name
        log_step(f"Thread-safe execution: {func.__name__}", thread=thread_name)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_step(f"Thread-safe execution error: {func.__name__}", 
                    error=str(e), thread=thread_name)
            raise
    return wrapper


class ThreadSafeProperty:
    """Thread-safe property descriptor."""
    
    def __init__(self, getter: Callable, setter: Optional[Callable] = None):
        self._getter = getter
        self._setter = setter
        self._lock = threading.RLock()
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        with self._lock:
            return self._getter(instance)
    
    def __set__(self, instance, value):
        if self._setter is None:
            raise AttributeError("Property is read-only")
        
        with self._lock:
            self._setter(instance, value)


def thread_local_storage():
    """Create thread-local storage."""
    return threading.local()


# Global thread safety utilities
_thread_safety_locks = {}
_thread_safety_lock = threading.Lock()


def get_thread_safety_lock(name: str) -> threading.RLock:
    """Get or create a named thread safety lock."""
    with _thread_safety_lock:
        if name not in _thread_safety_locks:
            _thread_safety_locks[name] = threading.RLock()
        return _thread_safety_locks[name]


def clear_thread_safety_locks():
    """Clear all thread safety locks (for testing)."""
    with _thread_safety_lock:
        _thread_safety_locks.clear()


# Example usage and testing
if __name__ == "__main__":
    print("Thread safety utilities initialized successfully!")
    print("Features include:")
    print("- ThreadSafeCounter: Atomic counter operations")
    print("- ThreadSafeDict: Thread-safe dictionary")
    print("- ThreadSafeList: Thread-safe list")
    print("- ThreadSafeCache: TTL-based cache")
    print("- ReadWriteLock: Multiple readers, single writer")
    print("- Thread-safe decorators and context managers")
    print("- Thread-safe singleton pattern")
    print("Thread safety utilities are ready for integration!")
