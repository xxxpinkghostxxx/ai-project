"""
Comprehensive tests for UnifiedErrorHandler fixes.
Tests thread safety, memory management, and validation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest
import threading
import time
from unittest.mock import MagicMock, patch
from src.utils.unified_error_handler import (
    UnifiedErrorHandler, get_error_handler, safe_execute,
    safe_initialize_component, safe_process_step, safe_callback_execution,
    ErrorSeverity, ErrorCategory
)


class TestUnifiedErrorHandler:
    """Test suite for UnifiedErrorHandler fixes."""

    def setup_method(self):
        """Set up test environment."""
        self.handler = UnifiedErrorHandler()

    def teardown_method(self):
        """Clean up after tests."""
        self.handler.clear_error_history()

    def test_thread_safe_global_instance(self):
        """Test thread-safe global instance creation."""
        instances = []

        def create_instance():
            instance = get_error_handler()
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

    def test_callback_limits(self):
        """Test callback limits to prevent memory leaks."""
        # Add maximum callbacks
        for i in range(self.handler.max_callbacks):
            def callback(severity, context, critical):
                pass
            self.handler.add_error_callback(callback)

        # Try to add one more
        def extra_callback(severity, context, critical):
            pass

        # This should not add the callback due to limit
        self.handler.add_error_callback(extra_callback)
        assert len(self.handler.error_callbacks) == self.handler.max_callbacks

    def test_invalid_callback_validation(self):
        """Test invalid callback validation."""
        with pytest.raises(TypeError):
            self.handler.add_error_callback("not a function")

        with pytest.raises(TypeError):
            self.handler.add_error_callback(123)

    def test_handle_error_input_validation(self):
        """Test input validation in handle_error."""
        # Test with None error
        result = self.handler.handle_error(None, "test")
        assert result is False

        # Test with invalid severity
        result = self.handler.handle_error(ValueError("test"), "test", severity="invalid")
        assert result is True  # Should use default severity

        # Test with invalid category
        result = self.handler.handle_error(ValueError("test"), "test", category="invalid")
        assert result is True  # Should use default category

        # Test with invalid recovery function
        result = self.handler.handle_error(ValueError("test"), "test", recovery_func="invalid")
        assert result is True  # Should ignore invalid recovery function

    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        # Generate many errors to fill up history
        for i in range(50):
            error = ValueError(f"Test error {i}")
            self.handler.handle_error(error, f"context_{i}")

        # Verify errors were recorded
        assert len(self.handler.error_history) > 0
        assert len(self.handler.error_records) > 0

        # Clear history
        self.handler.clear_error_history()

        # Verify cleanup
        assert len(self.handler.error_history) == 0
        assert len(self.handler.error_records) == 0
        assert self.handler.system_health == 'healthy'

    def test_periodic_cleanup(self):
        """Test periodic cleanup of old error records."""
        # Add some error records
        for i in range(10):
            error = ValueError(f"Test error {i}")
            self.handler.handle_error(error, f"context_{i}")

        initial_count = len(self.handler.error_records)

        # Manually trigger cleanup (simulate old records)
        old_time = time.time() - 86401  # More than 24 hours ago
        for record in self.handler.error_records.values():
            record.last_occurrence = old_time

        # Trigger cleanup
        self.handler._periodic_cleanup()

        # Should have cleaned up old records
        assert len(self.handler.error_records) < initial_count

    def test_error_statistics_thread_safety(self):
        """Test error statistics access thread safety."""
        results = []

        def get_stats():
            stats = self.handler.get_error_statistics()
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
            assert 'total_errors' in result
            assert 'system_health' in result

    def test_safe_execute_validation(self):
        """Test safe_execute with input validation."""
        # Test with invalid function
        result = safe_execute("not a function")
        assert result is None

        # Test with valid function
        def test_func():
            return "success"

        result = safe_execute(test_func)
        assert result == "success"

        # Test with function that raises exception
        def failing_func():
            raise ValueError("Test error")

        result = safe_execute(failing_func, default_return="default")
        assert result == "default"

    def test_safe_initialize_component_validation(self):
        """Test safe_initialize_component with input validation."""
        # Test with invalid component name
        result = safe_initialize_component("", lambda: "test")
        assert result is None

        # Test with invalid init function
        result = safe_initialize_component("test", "not callable")
        assert result is None

        # Test with invalid fallback function
        result = safe_initialize_component("test", lambda: "test", "not callable")
        assert result is None

        # Test with valid functions
        def init_func():
            return "initialized"

        result = safe_initialize_component("test_component", init_func)
        assert result == "initialized"

    def test_safe_process_step_validation(self):
        """Test safe_process_step with input validation."""
        # Test with invalid function
        result = safe_process_step("not callable", "test_step")
        assert result is False

        # Test with valid function
        def process_func():
            return True

        result = safe_process_step(process_func, "test_step")
        assert result is True

        # Test with function that raises exception
        def failing_process():
            raise RuntimeError("Process failed")

        result = safe_process_step(failing_process, "failing_step")
        assert result is False

    def test_safe_callback_execution_validation(self):
        """Test safe_callback_execution with input validation."""
        # Test with invalid callback
        result = safe_callback_execution("not callable")
        assert result is None

        # Test with valid callback
        def callback():
            return "callback_result"

        result = safe_callback_execution(callback)
        assert result == "callback_result"

        # Test with callback that raises exception
        def failing_callback():
            raise Exception("Callback failed")

        result = safe_callback_execution(failing_callback)
        assert result is None

    def test_system_health_updates(self):
        """Test system health updates based on error patterns."""
        # Initially healthy
        assert self.handler.system_health == 'healthy'

        # Add a few errors
        for i in range(3):
            error = ValueError(f"Test error {i}")
            self.handler.handle_error(error, f"context_{i}")

        # Should still be healthy or warning
        assert self.handler.system_health in ['healthy', 'warning']

        # Add many errors quickly
        for i in range(15):
            error = RuntimeError(f"Critical error {i}")
            self.handler.handle_error(error, f"critical_{i}")

        # Should be in degraded or critical state
        assert self.handler.system_health in ['warning', 'degraded', 'critical', 'failed']

    def test_error_record_limits(self):
        """Test that error records are limited to prevent memory growth."""
        # Add many errors
        for i in range(self.handler.max_error_records + 100):
            error = ValueError(f"Error {i}")
            self.handler.handle_error(error, f"context_{i}")

        # Should not exceed max_error_records
        assert len(self.handler.error_records) <= self.handler.max_error_records

    def test_recovery_strategy_validation(self):
        """Test recovery strategy validation."""
        from src.utils.unified_error_handler import RecoveryStrategy

        # Test adding invalid strategy
        with pytest.raises(TypeError):
            self.handler.add_recovery_strategy("not a strategy")

        # Test adding valid strategy
        strategy = RecoveryStrategy()
        self.handler.add_recovery_strategy(strategy)
        assert strategy in self.handler.recovery_strategies

    def test_concurrent_error_handling(self):
        """Test concurrent error handling."""
        errors_handled = []

        def handle_errors():
            for i in range(10):
                try:
                    raise ValueError(f"Concurrent error {i}")
                except Exception as e:
                    result = self.handler.handle_error(e, f"concurrent_context_{i}")
                    errors_handled.append(result)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=handle_errors)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have handled some errors
        assert len(errors_handled) > 0
        # All should have been handled (even if recovery failed)
        assert all(isinstance(result, bool) for result in errors_handled)


if __name__ == "__main__":
    pytest.main([__file__])






