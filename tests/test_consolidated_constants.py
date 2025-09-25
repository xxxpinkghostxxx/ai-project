"""
Tests for consolidated constants.

This module contains tests for the consolidated constants defined in config/consolidated_constants.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest
from config.consolidated_constants import (
    UI_CONSTANTS, ERROR_MESSAGES, LOG_MESSAGES, FILE_PATHS,
    NODE_PROPERTIES, CONNECTION_PROPERTIES, SYSTEM_STATES,
    NODE_STATES, NODE_TYPES, CONNECTION_TYPES, PERFORMANCE_METRICS,
    THRESHOLDS, DEFAULT_VALUES, PRINT_PATTERNS, EXCEPTION_TYPES,
    FUNCTION_NAMES, CLASS_NAMES
)


class TestConsolidatedConstants(unittest.TestCase):
    """Tests for consolidated constants."""

    def test_ui_constants(self):
        """Test UI constants are properly defined."""
        self.assertIsInstance(UI_CONSTANTS, dict)
        self.assertIn('SIMULATION_STATUS_RUNNING', UI_CONSTANTS)
        self.assertIn('MAIN_WINDOW_TAG', UI_CONSTANTS)
        self.assertEqual(UI_CONSTANTS['SIMULATION_STATUS_RUNNING'], 'Running')

    def test_error_messages(self):
        """Test error messages are properly defined."""
        self.assertIsInstance(ERROR_MESSAGES, dict)
        self.assertIn('GRAPH_NONE', ERROR_MESSAGES)
        self.assertIn('INVALID_NODE_ID', ERROR_MESSAGES)
        self.assertEqual(ERROR_MESSAGES['GRAPH_NONE'], 'Graph is None')

    def test_log_messages(self):
        """Test log messages are properly defined."""
        self.assertIsInstance(LOG_MESSAGES, dict)
        self.assertIn('SYSTEM_STARTED', LOG_MESSAGES)
        self.assertIn('SIMULATION_STARTED', LOG_MESSAGES)

    def test_file_paths(self):
        """Test file paths are properly defined."""
        self.assertIsInstance(FILE_PATHS, dict)
        self.assertIn('NEURAL_MAPS_DIR', FILE_PATHS)
        self.assertIn('CONFIG_FILE', FILE_PATHS)

    def test_node_properties(self):
        """Test node properties are properly defined."""
        self.assertIsInstance(NODE_PROPERTIES, dict)
        self.assertIn('ID', NODE_PROPERTIES)
        self.assertIn('ENERGY', NODE_PROPERTIES)
        self.assertIn('TYPE', NODE_PROPERTIES)

    def test_connection_properties(self):
        """Test connection properties are properly defined."""
        self.assertIsInstance(CONNECTION_PROPERTIES, dict)
        self.assertIn('SOURCE', CONNECTION_PROPERTIES)
        self.assertIn('TARGET', CONNECTION_PROPERTIES)
        self.assertIn('WEIGHT', CONNECTION_PROPERTIES)

    def test_system_states(self):
        """Test system states are properly defined."""
        self.assertIsInstance(SYSTEM_STATES, dict)
        self.assertIn('RUNNING', SYSTEM_STATES)
        self.assertIn('STOPPED', SYSTEM_STATES)
        self.assertIn('ERROR', SYSTEM_STATES)

    def test_node_states(self):
        """Test node states are properly defined."""
        self.assertIsInstance(NODE_STATES, dict)
        self.assertIn('ACTIVE', NODE_STATES)
        self.assertIn('INACTIVE', NODE_STATES)
        self.assertIn('PENDING', NODE_STATES)

    def test_node_types(self):
        """Test node types are properly defined."""
        self.assertIsInstance(NODE_TYPES, dict)
        self.assertIn('SENSORY', NODE_TYPES)
        self.assertIn('DYNAMIC', NODE_TYPES)
        self.assertIn('OSCILLATOR', NODE_TYPES)

    def test_connection_types(self):
        """Test connection types are properly defined."""
        self.assertIsInstance(CONNECTION_TYPES, dict)
        self.assertIn('EXCITATORY', CONNECTION_TYPES)
        self.assertIn('INHIBITORY', CONNECTION_TYPES)
        self.assertIn('MODULATORY', CONNECTION_TYPES)

    def test_performance_metrics(self):
        """Test performance metrics are properly defined."""
        self.assertIsInstance(PERFORMANCE_METRICS, dict)
        self.assertIn('STEP_TIME', PERFORMANCE_METRICS)
        self.assertIn('MEMORY_USAGE', PERFORMANCE_METRICS)
        self.assertIn('FPS', PERFORMANCE_METRICS)

    def test_thresholds(self):
        """Test thresholds are properly defined."""
        self.assertIsInstance(THRESHOLDS, dict)
        self.assertIn('MEMORY_WARNING_MB', THRESHOLDS)
        self.assertIn('CPU_WARNING_PERCENT', THRESHOLDS)
        self.assertIsInstance(THRESHOLDS['MEMORY_WARNING_MB'], float)

    def test_default_values(self):
        """Test default values are properly defined."""
        self.assertIsInstance(DEFAULT_VALUES, dict)
        self.assertIn('ENERGY_CAP', DEFAULT_VALUES)
        self.assertIn('THRESHOLD_DEFAULT', DEFAULT_VALUES)
        self.assertIsInstance(DEFAULT_VALUES['ENERGY_CAP'], float)

    def test_print_patterns(self):
        """Test print patterns are properly defined."""
        self.assertIsInstance(PRINT_PATTERNS, dict)
        self.assertIn('ERROR_PREFIX', PRINT_PATTERNS)
        self.assertIn('INFO_PREFIX', PRINT_PATTERNS)

    def test_exception_types(self):
        """Test exception types are properly defined."""
        self.assertIsInstance(EXCEPTION_TYPES, dict)
        self.assertIn('VALUE_ERROR', EXCEPTION_TYPES)
        self.assertEqual(EXCEPTION_TYPES['VALUE_ERROR'], ValueError)

    def test_function_names(self):
        """Test function names are properly defined."""
        self.assertIsInstance(FUNCTION_NAMES, dict)
        self.assertIn('INIT', FUNCTION_NAMES)
        self.assertIn('START', FUNCTION_NAMES)
        self.assertEqual(FUNCTION_NAMES['INIT'], '__init__')

    def test_class_names(self):
        """Test class names are properly defined."""
        self.assertIsInstance(CLASS_NAMES, dict)
        self.assertIn('SIMULATION_COORDINATOR', CLASS_NAMES)
        self.assertIn('BEHAVIOR_ENGINE', CLASS_NAMES)

    def test_no_empty_constants(self):
        """Test that no constant groups are empty."""
        constant_groups = [
            UI_CONSTANTS, ERROR_MESSAGES, LOG_MESSAGES, FILE_PATHS,
            NODE_PROPERTIES, CONNECTION_PROPERTIES, SYSTEM_STATES,
            NODE_STATES, NODE_TYPES, CONNECTION_TYPES, PERFORMANCE_METRICS,
            THRESHOLDS, DEFAULT_VALUES, PRINT_PATTERNS, EXCEPTION_TYPES,
            FUNCTION_NAMES, CLASS_NAMES
        ]

        for group in constant_groups:
            self.assertGreater(len(group), 0, f"Constant group {group} is empty")

    def test_constant_values_types(self):
        """Test that constant values have appropriate types."""
        # String constants
        for key, value in UI_CONSTANTS.items():
            self.assertIsInstance(value, str, f"UI_CONSTANTS[{key}] should be string")

        for key, value in ERROR_MESSAGES.items():
            self.assertIsInstance(value, str, f"ERROR_MESSAGES[{key}] should be string")

        # Numeric constants
        for key, value in THRESHOLDS.items():
            self.assertIsInstance(value, (int, float), f"THRESHOLDS[{key}] should be numeric")

        for key, value in DEFAULT_VALUES.items():
            self.assertIsInstance(value, (int, float, str, bool), f"DEFAULT_VALUES[{key}] should be primitive type")

    def test_constant_key_format(self):
        """Test that constant keys follow naming conventions."""
        # Most constants use UPPER_CASE with underscores
        upper_case_groups = [
            UI_CONSTANTS, ERROR_MESSAGES, LOG_MESSAGES, FILE_PATHS,
            NODE_PROPERTIES, CONNECTION_PROPERTIES, SYSTEM_STATES,
            NODE_STATES, NODE_TYPES, CONNECTION_TYPES, PERFORMANCE_METRICS,
            THRESHOLDS, PRINT_PATTERNS, EXCEPTION_TYPES,
            FUNCTION_NAMES, CLASS_NAMES
        ]

        for group in upper_case_groups:
            for key in group.keys():
                self.assertTrue(key.isupper() or '_' in key or key[0].isupper(),
                              f"Key '{key}' in {group} does not follow naming convention")


if __name__ == '__main__':
    unittest.main()






