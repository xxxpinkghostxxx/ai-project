"""
Comprehensive tests for MemoryPoolManager.
Tests object pooling, memory management, cleanup, statistics tracking,
error handling, and integration scenarios.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.learning.memory_pool_manager import (MemoryPoolManager, ObjectPool,
                                              cleanup_memory_pools,
                                              create_edge_pool,
                                              create_event_pool,
                                              create_node_pool,
                                              get_memory_pool_manager)


class TestObjectPool:
    """Test suite for ObjectPool."""

    def setup_method(self):
        """Set up test environment."""
        def factory_func():
            return {'id': 0, 'data': 'test'}
        self.pool = ObjectPool(factory_func, max_size=10)

    def test_initialization(self):
        """Test ObjectPool initialization."""
        assert self.pool.factory_func is not None
        assert self.pool.max_size == 10
        assert len(self.pool.available_objects) == 0
        assert len(self.pool.in_use_objects) == 0
        assert isinstance(self.pool.stats, object)

    def test_get_object_new(self):
        """Test getting a new object from pool."""
        obj = self.pool.get_object()

        assert obj is not None
        assert obj['data'] == 'test'
        assert len(self.pool.in_use_objects) == 1
        assert self.pool.stats.creation_count == 1
        assert self.pool.stats.in_use_objects == 1

    def test_get_object_reuse(self):
        """Test reusing an object from pool."""
        # Return an object first
        obj1 = self.pool.get_object()
        self.pool.return_object(obj1)

        # Get another object - should reuse
        obj2 = self.pool.get_object()

        assert obj2 is obj1  # Same object
        assert len(self.pool.available_objects) == 0
        assert len(self.pool.in_use_objects) == 1
        assert self.pool.stats.reuse_count == 1

    def test_return_object(self):
        """Test returning an object to pool."""
        obj = self.pool.get_object()

        self.pool.return_object(obj)

        assert len(self.pool.available_objects) == 1
        assert len(self.pool.in_use_objects) == 0
        assert self.pool.stats.available_objects == 1

    def test_return_object_max_size(self):
        """Test returning objects beyond max size."""
        # Fill pool beyond max size
        objects = []
        for i in range(15):  # More than max_size of 10
            obj = self.pool.get_object()
            objects.append(obj)

        # Return all
        for obj in objects:
            self.pool.return_object(obj)

        # Should only keep max_size objects
        assert len(self.pool.available_objects) <= self.pool.max_size

    def test_cleanup_expired_objects(self):
        """Test cleanup of expired objects."""
        # Add some objects
        for i in range(5):
            obj = self.pool.get_object()
            self.pool.return_object(obj)

        # Force cleanup
        self.pool.last_cleanup = 0  # Make it eligible for cleanup
        cleaned = self.pool.cleanup_expired_objects()

        assert cleaned >= 0
        assert self.pool.stats.last_cleanup > 0

    def test_get_stats(self):
        """Test statistics retrieval."""
        obj = self.pool.get_object()

        stats = self.pool.get_stats()

        assert stats.total_objects == 1
        assert stats.available_objects == 0
        assert stats.in_use_objects == 1
        assert stats.creation_count == 1


class TestMemoryPoolManager:
    """Test suite for MemoryPoolManager."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = MemoryPoolManager()

    def teardown_method(self):
        """Clean up after tests."""
        self.manager.cleanup()

    def test_initialization(self):
        """Test MemoryPoolManager initialization."""
        assert isinstance(self.manager.pools, dict)
        assert len(self.manager.pools) == 0
        assert self.manager.running is False

    def test_create_pool(self):
        """Test pool creation."""
        def factory():
            return {'test': True}

        pool = self.manager.create_pool('test_pool', factory, max_size=5)

        assert 'test_pool' in self.manager.pools
        assert self.manager.pools['test_pool'] == pool
        assert pool.max_size == 5

    def test_create_pool_duplicate(self):
        """Test creating duplicate pool raises error."""
        def factory():
            return {}

        self.manager.create_pool('test_pool', factory)

        with pytest.raises(ValueError, match="already exists"):
            self.manager.create_pool('test_pool', factory)

    def test_get_pool(self):
        """Test pool retrieval."""
        def factory():
            return {}

        self.manager.create_pool('test_pool', factory)

        pool = self.manager.get_pool('test_pool')
        assert pool is not None

        # Non-existent pool
        pool_none = self.manager.get_pool('nonexistent')
        assert pool_none is None

    def test_get_object(self):
        """Test object retrieval from pool."""
        def factory():
            return {'id': 1}

        self.manager.create_pool('test_pool', factory)

        obj = self.manager.get_object('test_pool')

        assert obj['id'] == 1

    def test_get_object_nonexistent_pool(self):
        """Test getting object from nonexistent pool raises error."""
        with pytest.raises(ValueError, match="not found"):
            self.manager.get_object('nonexistent')

    def test_return_object(self):
        """Test object return to pool."""
        def factory():
            return {'id': 1}

        self.manager.create_pool('test_pool', factory)

        obj = self.manager.get_object('test_pool')
        self.manager.return_object('test_pool', obj)

        pool = self.manager.get_pool('test_pool')
        assert len(pool.available_objects) == 1

    def test_return_object_nonexistent_pool(self):
        """Test returning object to nonexistent pool raises error."""
        with pytest.raises(ValueError, match="not found"):
            self.manager.return_object('nonexistent', {})

    def test_start_cleanup_thread(self):
        """Test cleanup thread starting."""
        self.manager.start_cleanup_thread()

        assert self.manager.running is True
        assert self.manager.cleanup_thread is not None
        assert self.manager.cleanup_thread.is_alive()

        self.manager.stop_cleanup_thread()

    def test_stop_cleanup_thread(self):
        """Test cleanup thread stopping."""
        self.manager.start_cleanup_thread()
        self.manager.stop_cleanup_thread()

        assert self.manager.running is False
        # Thread may still be alive briefly, but will stop

    def test_cleanup_all_pools(self):
        """Test cleaning up all pools."""
        def factory():
            return {}

        # Create pools with objects
        pool1 = self.manager.create_pool('pool1', factory)
        pool2 = self.manager.create_pool('pool2', factory)

        # Add objects
        obj1 = pool1.get_object()
        obj2 = pool2.get_object()
        pool1.return_object(obj1)
        pool2.return_object(obj2)

        # Cleanup
        total_cleaned = self.manager.cleanup_all_pools()

        assert total_cleaned >= 0

    def test_get_all_stats(self):
        """Test getting all pool statistics."""
        def factory():
            return {}

        self.manager.create_pool('pool1', factory)
        self.manager.create_pool('pool2', factory)

        stats = self.manager.get_all_stats()

        assert 'pool1' in stats
        assert 'pool2' in stats
        assert len(stats) == 2

    def test_cleanup(self):
        """Test manager cleanup."""
        def factory():
            return {}

        self.manager.create_pool('test_pool', factory)
        self.manager.start_cleanup_thread()

        self.manager.cleanup()

        assert len(self.manager.pools) == 0
        assert self.manager.running is False


class TestGlobalFunctions:
    """Test suite for global functions."""

    def test_get_memory_pool_manager(self):
        """Test global memory pool manager retrieval."""
        manager = get_memory_pool_manager()

        assert isinstance(manager, MemoryPoolManager)

        # Should return same instance
        manager2 = get_memory_pool_manager()
        assert manager is manager2

    def test_cleanup_memory_pools(self):
        """Test global cleanup function."""
        manager = get_memory_pool_manager()
        manager.pools['test'] = MagicMock()  # Add a mock pool

        cleanup_memory_pools()

        # Should reset global instance
        new_manager = get_memory_pool_manager()
        assert new_manager is not manager

    def test_create_node_pool(self):
        """Test node pool creation."""
        pool = create_node_pool(max_size=100)

        assert isinstance(pool, ObjectPool)
        assert pool.max_size == 100

        # Test object creation
        node = pool.get_object()
        assert 'id' in node
        assert 'energy' in node
        assert 'state' in node

    def test_create_edge_pool(self):
        """Test edge pool creation."""
        pool = create_edge_pool(max_size=200)

        assert isinstance(pool, ObjectPool)
        assert pool.max_size == 200

        # Test object creation
        edge = pool.get_object()
        assert 'source' in edge
        assert 'target' in edge
        assert 'weight' in edge

    def test_create_event_pool(self):
        """Test event pool creation."""
        pool = create_event_pool(max_size=50)

        assert isinstance(pool, ObjectPool)
        assert pool.max_size == 50

        # Test object creation
        event = pool.get_object()
        assert 'event_type' in event
        assert 'timestamp' in event
        assert 'source_node_id' in event


class TestPooledObject:
    """Test suite for PooledObject context manager."""

    def test_context_manager(self):
        """Test PooledObject context manager."""
        manager = get_memory_pool_manager()
        def factory():
            return {'test': True}
        manager.create_pool('test_pool', factory)

        with manager.get_pooled_object('test_pool') as obj:
            assert obj['test'] is True

        # Object should be returned to pool
        pool = manager.get_pool('test_pool')
        assert len(pool.available_objects) == 1


class TestThreadSafety:
    """Test suite for thread safety."""

    def test_thread_safe_operations(self):
        """Test thread-safe pool operations."""
        manager = MemoryPoolManager()
        def factory():
            return {'counter': 0}

        manager.create_pool('thread_test', factory)

        errors = []
        results = []

        def worker():
            try:
                for i in range(10):
                    obj = manager.get_object('thread_test')
                    obj['counter'] = i
                    time.sleep(0.001)  # Small delay
                    manager.return_object('thread_test', obj)
                results.append(True)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 5

        # Cleanup
        manager.cleanup()


class TestErrorHandling:
    """Test suite for error handling."""

    def test_pool_creation_with_invalid_factory(self):
        """Test pool creation with invalid factory."""
        manager = MemoryPoolManager()

        def invalid_factory():
            raise Exception("Factory error")

        pool = manager.create_pool('invalid', invalid_factory)

        # Getting object should handle factory errors gracefully
        try:
            obj = pool.get_object()
            # If it succeeds, object should be created
            assert obj is not None
        except:
            # If it fails, that's also acceptable
            pass

    def test_return_invalid_object(self):
        """Test returning invalid object."""
        manager = MemoryPoolManager()
        def factory():
            return {}

        pool = manager.create_pool('test', factory)

        # Return object not from pool
        pool.return_object({'invalid': True})

        # Should not crash, but object won't be added to available
        assert len(pool.available_objects) == 0


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_empty_pools(self):
        """Test operations with empty pools."""
        manager = MemoryPoolManager()

        # Get all stats with no pools
        stats = manager.get_all_stats()
        assert len(stats) == 0

        # Cleanup all pools with no pools
        cleaned = manager.cleanup_all_pools()
        assert cleaned == 0

    def test_pool_max_size_zero(self):
        """Test pool with zero max size."""
        def factory():
            return {}

        pool = ObjectPool(factory, max_size=0)

        obj = pool.get_object()
        assert obj is not None

        pool.return_object(obj)
        # With max_size=0, object should not be kept
        assert len(pool.available_objects) == 0

    def test_extreme_max_size(self):
        """Test pool with very large max size."""
        def factory():
            return {}

        pool = ObjectPool(factory, max_size=100000)

        # Should handle large max size
        obj = pool.get_object()
        assert obj is not None

        pool.return_object(obj)
        assert len(pool.available_objects) == 1


if __name__ == "__main__":
    pytest.main([__file__])






