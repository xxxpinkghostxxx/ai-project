"""
Reader-Writer Lock Implementation
Provides efficient concurrent access with multiple readers and exclusive writers.
"""

import threading
import time
from contextlib import contextmanager
from typing import Optional


class ReaderWriterLock:
    """Reader-Writer lock allowing multiple readers or single writer."""

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._reader_lock = threading.Condition(threading.Lock())
        self._writer_lock = threading.Condition(threading.Lock())
        self._read_waiting = 0
        self._write_waiting = 0

        # Statistics for monitoring
        self._stats = {
            'read_acquires': 0,
            'write_acquires': 0,
            'read_waits': 0,
            'write_waits': 0,
            'contention_events': 0,
            'created_at': time.time()
        }

    @contextmanager
    def read_lock(self, timeout: Optional[float] = None):
        """Context manager for read access."""
        start_time = time.time()
        acquired = self._acquire_read(timeout)

        if not acquired:
            self._stats['read_waits'] += 1
            raise TimeoutError("Read lock acquisition timed out")

        wait_time = time.time() - start_time
        if wait_time > 0.001:  # Log if wait was significant
            self._stats['contention_events'] += 1

        try:
            self._stats['read_acquires'] += 1
            yield
        finally:
            self._release_read()

    @contextmanager
    def write_lock(self, timeout: Optional[float] = None):
        """Context manager for write access."""
        start_time = time.time()
        acquired = self._acquire_write(timeout)

        if not acquired:
            self._stats['write_waits'] += 1
            raise TimeoutError("Write lock acquisition timed out")

        wait_time = time.time() - start_time
        if wait_time > 0.001:  # Log if wait was significant
            self._stats['contention_events'] += 1

        try:
            self._stats['write_acquires'] += 1
            yield
        finally:
            self._release_write()

    def _acquire_read(self, timeout: Optional[float]) -> bool:
        """Acquire read lock with optional timeout."""
        with self._reader_lock:
            if timeout is None:
                while self._writers > 0 or self._write_waiting > 0:
                    self._reader_lock.wait()
                self._readers += 1
                return True
            else:
                end_time = time.time() + timeout
                while self._writers > 0 or self._write_waiting > 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return False
                    self._reader_lock.wait(remaining)
                self._readers += 1
                return True

    def _release_read(self):
        """Release read lock."""
        with self._reader_lock:
            self._readers -= 1
            if self._readers == 0:
                self._reader_lock.notify_all()

    def _acquire_write(self, timeout: Optional[float]) -> bool:
        """Acquire write lock with optional timeout."""
        with self._writer_lock:
            self._write_waiting += 1

        try:
            with self._reader_lock:
                if timeout is None:
                    while self._readers > 0 or self._writers > 0:
                        self._reader_lock.wait()
                    self._writers += 1
                    return True
                else:
                    end_time = time.time() + timeout
                    while self._readers > 0 or self._writers > 0:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            return False
                        self._reader_lock.wait(remaining)
                    self._writers += 1
                    return True
        finally:
            with self._writer_lock:
                self._write_waiting -= 1

    def _release_write(self):
        """Release write lock."""
        with self._reader_lock:
            self._writers -= 1
            self._reader_lock.notify_all()

    def get_stats(self) -> dict:
        """Get lock statistics."""
        stats = self._stats.copy()
        stats['uptime'] = time.time() - stats['created_at']
        stats['current_readers'] = self._readers
        stats['current_writers'] = self._writers
        stats['waiting_readers'] = self._read_waiting
        stats['waiting_writers'] = self._write_waiting
        return stats

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            'read_acquires': 0,
            'write_acquires': 0,
            'read_waits': 0,
            'write_waits': 0,
            'contention_events': 0,
            'created_at': time.time()
        }


class ReadWriteLockedObject:
    """Wrapper for objects that need reader-writer locking."""

    def __init__(self, obj, lock: Optional[ReaderWriterLock] = None):
        self._obj = obj
        self._lock = lock or ReaderWriterLock()

    def read_operation(self, operation, *args, timeout: Optional[float] = None, **kwargs):
        """Perform a read operation with automatic locking."""
        with self._lock.read_lock(timeout):
            return operation(self._obj, *args, **kwargs)

    def write_operation(self, operation, *args, timeout: Optional[float] = None, **kwargs):
        """Perform a write operation with automatic locking."""
        with self._lock.write_lock(timeout):
            return operation(self._obj, *args, **kwargs)

    @property
    def lock(self) -> ReaderWriterLock:
        """Access to the underlying lock."""
        return self._lock

    @property
    def obj(self):
        """Access to the wrapped object (use with caution)."""
        return self._obj


# Global instances for common use cases
_graph_lock = ReaderWriterLock()
_id_manager_lock = ReaderWriterLock()
_connection_lock = ReaderWriterLock()


def get_graph_lock() -> ReaderWriterLock:
    """Get the global graph reader-writer lock."""
    return _graph_lock


def get_id_manager_lock() -> ReaderWriterLock:
    """Get the global ID manager reader-writer lock."""
    return _id_manager_lock


def get_connection_lock() -> ReaderWriterLock:
    """Get the global connection reader-writer lock."""
    return _connection_lock


def get_lock_stats() -> dict:
    """Get statistics for all global locks."""
    return {
        'graph_lock': _graph_lock.get_stats(),
        'id_manager_lock': _id_manager_lock.get_stats(),
        'connection_lock': _connection_lock.get_stats()
    }






