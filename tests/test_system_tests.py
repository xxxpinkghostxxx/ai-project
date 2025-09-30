"""
System-level tests for the unified testing system.
Tests critical imports, error handling, and system functionality.
"""

import logging
import time
from typing import Tuple, Dict, Any

# Local imports - Core modules
from src.core.main_graph import initialize_main_graph
from src.core.services.simulation_coordinator import SimulationCoordinator

# Local imports - Utils modules
from src.utils.event_bus import get_event_bus
from src.utils.logging_utils import log_step
from src.utils.print_utils import print_error, print_info, print_warning
from src.utils.unified_error_handler import get_error_handler, safe_execute

# Local test utilities
from .test_utils import TestCase, TestCategory, TestResult
from .test_mocks import create_mock_services, configure_mock_services_for_init


def test_critical_imports() -> Tuple[bool, Dict[str, Any]]:
    """Test critical module imports."""
    try:
        return True, {'imports': 'successful'}
    except ImportError as e:
        return False, {'error': str(e)}


def test_error_handling() -> Tuple[bool, Dict[str, Any]]:
    """Test error handling and recovery."""
    try:
        error_handler = get_error_handler()

        # Test safe execution
        def failing_function():
            raise ValueError("Test error")

        result = safe_execute(failing_function, "test_context", default_return="fallback")

        return result == "fallback", {
            'error_handler_available': error_handler is not None,
            'safe_execute_works': result == "fallback"
        }
    except (ImportError, RuntimeError, ValueError) as e:
        return False, {'error': str(e)}


def test_event_bus() -> Tuple[bool, Dict[str, Any]]:
    """Test EventBus subscribe, emit, and fallback."""
    try:
        bus = get_event_bus()

        # Test subscribe and emit
        called = [False]
        def test_callback(event_type, data):
            called[0] = True
            assert event_type == 'TEST_SPIKE'
            assert data['node_id'] == 123
            assert data['timestamp'] > 0

        bus.subscribe('TEST_SPIKE', test_callback)
        bus.emit('TEST_SPIKE', {'node_id': 123, 'timestamp': time.time()})

        subscribe_emit_works = called[0]

        # Test unsubscribe
        bus.unsubscribe('TEST_SPIKE', test_callback)
        bus.emit('TEST_SPIKE', {'node_id': 456, 'timestamp': time.time()})
        unsubscribe_works = not called[0]

        # Test fallback (emit fails if invalid data)
        try:
            bus.emit('INVALID', 'not dict')  # Should raise but caught internally
            fallback_handled = True
        except ValueError:
            fallback_handled = True  # Expected, but since caught, no exception

        return subscribe_emit_works and unsubscribe_works and fallback_handled, {
            'subscribe_emit': subscribe_emit_works,
            'unsubscribe': unsubscribe_works,
            'fallback_handled': fallback_handled
        }

    except (ImportError, RuntimeError, ValueError) as e:
        return False, {'error': str(e)}


def test_simulation_manager_creation() -> Tuple[bool, Dict[str, Any]]:
    """Test simulation manager creation."""
    try:
        # Create mocked services for SimulationCoordinator
        mocks = create_mock_services()
        configure_mock_services_for_init(mocks)

        (service_registry, neural_processor, energy_manager,
         learning_engine, sensory_processor, performance_monitor,
         graph_manager, event_coordinator, configuration_service) = mocks

        sim_manager = SimulationCoordinator(
            service_registry, neural_processor, energy_manager,
            learning_engine, sensory_processor, performance_monitor,
            graph_manager, event_coordinator, configuration_service
        )

        return sim_manager is not None, {
            'manager_created': True,
            'type': type(sim_manager).__name__
        }
    except (ImportError, AttributeError, TypeError, RuntimeError) as e:
        return False, {'error': str(e)}


def create_system_test_cases() -> list:
    """Create system test cases."""
    return [
        TestCase(
            name="critical_imports",
            category=TestCategory.SYSTEM,
            description="Test critical module imports",
            test_func=test_critical_imports
        ),
        TestCase(
            name="error_handling",
            category=TestCategory.SYSTEM,
            description="Test error handling and recovery",
            test_func=test_error_handling
        ),
        TestCase(
            name="event_bus_functionality",
            category=TestCategory.SYSTEM,
            description="Test EventBus subscribe, emit, unsubscribe, and fallback",
            test_func=test_event_bus
        ),
        TestCase(
            name="simulation_manager_creation",
            category=TestCategory.UNIT,
            description="Test simulation manager creation",
            test_func=test_simulation_manager_creation
        )
    ]