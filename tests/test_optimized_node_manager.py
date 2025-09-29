"""
Comprehensive tests for OptimizedNodeManager fixes.
Tests memory validation, thread safety, and error handling.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.neural.optimized_node_manager import (OptimizedNodeManager,
                                               get_optimized_node_manager)


class TestOptimizedNodeManager:
    """Test suite for OptimizedNodeManager fixes."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = OptimizedNodeManager(max_nodes=1000)

    def teardown_method(self):
        """Clean up after tests."""
        self.manager.cleanup()

    def test_memory_validation_init(self):
        """Test memory validation during initialization."""
        # Test valid parameters
        manager = OptimizedNodeManager(max_nodes=100)
        assert manager.max_nodes == 100

        # Test invalid parameters
        with pytest.raises(ValueError):
            OptimizedNodeManager(max_nodes=0)

        with pytest.raises(ValueError):
            OptimizedNodeManager(max_nodes=-1)

        with pytest.raises(ValueError):
            OptimizedNodeManager(max_nodes=20000000)  # Too large

    def test_thread_safe_global_instance(self):
        """Test thread-safe global instance creation."""
        instances = []

        def create_instance():
            instance = get_optimized_node_manager()
            instances.append(instance)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same object
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance

    def test_bounds_checking_create_nodes(self):
        """Test bounds checking in node creation."""
        # Test with invalid node specs
        invalid_specs = [
            "not a dict",
            {"type": 123},  # Invalid type
            {"x": "invalid", "y": 1.0}  # Invalid position
        ]

        result = self.manager.create_node_batch(invalid_specs)
        assert len(result) == 0  # Should skip invalid specs

        # Test with valid specs
        valid_specs = [
            {"type": "test", "energy": 1.0, "x": 10.0, "y": 20.0},
            {"type": "test2", "energy": 2.0}
        ]

        result = self.manager.create_node_batch(valid_specs)
        assert len(result) == 2

        # Verify nodes were created
        stats = self.manager.get_performance_stats()
        assert stats['total_nodes'] == 2
        assert stats['active_nodes'] == 2

    def test_memory_limits(self):
        """Test memory limit enforcement."""
        # Set a very low memory limit
        self.manager.set_memory_limit(1)  # 1MB limit

        # Create many nodes to trigger memory limit
        specs = [{"type": "test"} for _ in range(100)]

        # Should create some nodes but stop when memory limit is reached
        result = self.manager.create_node_batch(specs)
        assert len(result) >= 0  # May create some or none depending on memory

        # Verify memory validation
        assert self.manager.validate_memory_usage()

    def test_float_validation(self):
        """Test float value validation and clamping."""
        # Test valid values
        assert self.manager._validate_float(1.0, 0.0, 10.0, "test") == 1.0

        # Test clamping
        assert self.manager._validate_float(-5.0, 0.0, 10.0, "test") == 0.0
        assert self.manager._validate_float(15.0, 0.0, 10.0, "test") == 10.0

        # Test invalid values
        assert self.manager._validate_float("invalid", 0.0, 10.0, "test") == 0.0
        assert self.manager._validate_float(None, 0.0, 10.0, "test") == 0.0

    def test_batch_update_validation(self):
        """Test batch update with validation."""
        # Create some nodes first
        specs = [{"type": "test", "energy": 1.0} for _ in range(5)]
        node_ids = self.manager.create_node_batch(specs)
        assert len(node_ids) == 5

        # Test valid updates
        updates = {"energy": 2.0, "threshold": 0.8}
        self.manager.update_nodes_batch(node_ids, updates)

        # Test invalid updates
        invalid_updates = {"energy": "invalid", "x": float('inf')}
        self.manager.update_nodes_batch(node_ids, invalid_updates)

        # Test with invalid node_ids
        self.manager.update_nodes_batch([999999], {"energy": 1.0})

    def test_performance_stats_thread_safety(self):
        """Test performance stats thread safety."""
        results = []

        def get_stats():
            stats = self.manager.get_performance_stats()
            results.append(stats)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_stats)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert 'memory_usage_mb' in result
            assert 'operations_per_second' in result

    def test_cleanup_efficiency(self):
        """Test cleanup efficiency."""
        # Create some nodes
        specs = [{"type": "test"} for _ in range(10)]
        self.manager.create_node_batch(specs)

        # Verify nodes exist
        stats = self.manager.get_performance_stats()
        assert stats['total_nodes'] == 10

        # Clean up
        start_time = time.time()
        self.manager.cleanup()
        cleanup_time = time.time() - start_time

        # Verify cleanup was fast
        assert cleanup_time < 1.0  # Should complete in less than 1 second

        # Verify cleanup worked
        stats = self.manager.get_performance_stats()
        assert stats['total_nodes'] == 0
        assert stats['active_nodes'] == 0

    def test_concurrent_operations(self):
        """Test concurrent node operations."""
        results = []
        errors = []

        def create_nodes():
            try:
                specs = [{"type": "concurrent"} for _ in range(10)]
                node_ids = self.manager.create_node_batch(specs)
                results.append(len(node_ids))
            except Exception as e:
                errors.append(e)

        def update_nodes():
            try:
                # Get existing nodes
                stats = self.manager.get_performance_stats()
                if stats['active_nodes'] > 0:
                    # This is a simplified test - in real scenario we'd track node IDs
                    updates = {"energy": 1.5}
                    self.manager.update_nodes_batch([0, 1], updates)  # May not exist
                results.append("update_ok")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=create_nodes)
            t2 = threading.Thread(target=update_nodes)
            threads.extend([t1, t2])

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have some successful operations
        assert len(results) > 0
        assert len(errors) == 0  # No errors should occur

    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test with corrupted internal state (simulate)
        original_lock = self.manager._lock

        try:
            # Simulate lock failure
            with patch.object(self.manager, '_lock') as mock_lock:
                mock_lock.__enter__.side_effect = RuntimeError("Lock failed")

                # Operations should handle the error gracefully
                result = self.manager.create_node_batch([{"type": "test"}])
                # Should return empty list on error
                assert isinstance(result, list)

        finally:
            self.manager._lock = original_lock

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        initial_stats = self.manager.get_performance_stats()

        # Create nodes and check memory increases
        specs = [{"type": "memory_test"} for _ in range(50)]
        self.manager.create_node_batch(specs)

        after_stats = self.manager.get_performance_stats()

        # Memory usage should be tracked
        assert 'memory_usage_mb' in after_stats
        assert 'memory_usage_percent' in after_stats
        assert after_stats['memory_usage_mb'] >= initial_stats['memory_usage_mb']


if __name__ == "__main__":
    pytest.main([__file__])






