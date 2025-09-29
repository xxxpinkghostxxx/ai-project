"""
Comprehensive tests for UnifiedConfigManager.

This module contains unit tests, integration tests, edge cases, error handling,
performance tests, and real-world usage scenarios for the UnifiedConfigManager
and related configuration components.
"""

import json
import os
import tempfile
import threading
import time
import unittest

import yaml

from config.consolidated_constants import (CLASS_NAMES, CONNECTION_PROPERTIES,
                                           CONNECTION_TYPES, DEFAULT_VALUES,
                                           ERROR_MESSAGES, EXCEPTION_TYPES,
                                           FILE_PATHS, FUNCTION_NAMES,
                                           LOG_MESSAGES, NODE_PROPERTIES,
                                           NODE_STATES, NODE_TYPES,
                                           PERFORMANCE_METRICS, PRINT_PATTERNS,
                                           SYSTEM_STATES, THRESHOLDS,
                                           UI_CONSTANTS)
from config.unified_config_manager import (ConfigSchema, ConfigType,
                                           ConfigValidator,
                                           UnifiedConfigManager,
                                           get_config_manager,
                                           get_enhanced_nodes_config,
                                           get_learning_config,
                                           get_system_constants, set_config)


class TestConfigValidator(unittest.TestCase):
    """Unit tests for ConfigValidator."""

    def setUp(self):
        self.validator = ConfigValidator()

    def test_validate_string_success(self):
        """Test successful string validation."""
        schema = ConfigSchema('test', ConfigType.STRING, 'default', 'Test string')
        self.assertTrue(self.validator.validate_string('valid', schema))

    def test_validate_string_with_allowed_values(self):
        """Test string validation with allowed values."""
        schema = ConfigSchema('test', ConfigType.STRING, 'default', 'Test string',
                            allowed_values=['a', 'b', 'c'])
        self.assertTrue(self.validator.validate_string('a', schema))
        self.assertFalse(self.validator.validate_string('d', schema))

    def test_validate_integer_success(self):
        """Test successful integer validation."""
        schema = ConfigSchema('test', ConfigType.INTEGER, 5, 'Test int', min_value=0, max_value=10)
        self.assertTrue(self.validator.validate_integer(7, schema))

    def test_validate_integer_bounds(self):
        """Test integer validation with bounds."""
        schema = ConfigSchema('test', ConfigType.INTEGER, 5, 'Test int', min_value=0, max_value=10)
        self.assertFalse(self.validator.validate_integer(-1, schema))
        self.assertFalse(self.validator.validate_integer(15, schema))

    def test_validate_float_success(self):
        """Test successful float validation."""
        schema = ConfigSchema('test', ConfigType.FLOAT, 1.0, 'Test float', min_value=0.0, max_value=5.0)
        self.assertTrue(self.validator.validate_float(2.5, schema))

    def test_validate_boolean_success(self):
        """Test boolean validation."""
        self.assertTrue(ConfigValidator.validate_boolean(True))
        self.assertTrue(ConfigValidator.validate_boolean(False))
        self.assertFalse(ConfigValidator.validate_boolean('true'))

    def test_validate_list_success(self):
        """Test list validation."""
        schema = ConfigSchema('test', ConfigType.LIST, [], 'Test list', allowed_values=['a', 'b'])
        self.assertTrue(self.validator.validate_list(['a', 'b'], schema))
        self.assertFalse(self.validator.validate_list(['c'], schema))

    def test_validate_dict_success(self):
        """Test dict validation."""
        schema = ConfigSchema('test', ConfigType.DICT, {}, 'Test dict', allowed_values=['key1', 'key2'])
        self.assertTrue(self.validator.validate_dict({'key1': 'val'}, schema))
        self.assertFalse(self.validator.validate_dict({'key3': 'val'}, schema))


class TestUnifiedConfigManager(unittest.TestCase):
    """Unit tests for UnifiedConfigManager."""

    def setUp(self):
        self.manager = UnifiedConfigManager()

    def test_initialization(self):
        """Test manager initialization."""
        self.assertIsInstance(self.manager.config, dict)
        self.assertIsInstance(self.manager.schemas, dict)
        self.assertIsInstance(self.manager.change_history, list)
        self.assertIsInstance(self.manager.watchers, dict)

    def test_get_set_basic(self):
        """Test basic get/set operations."""
        # Set a value
        self.assertTrue(self.manager.set('test.key', 'value'))
        # Get the value
        self.assertEqual(self.manager.get('test.key'), 'value')
        # Get with default
        self.assertEqual(self.manager.get('nonexistent', 'default'), 'default')

    def test_set_with_validation(self):
        """Test set with schema validation."""
        schema = ConfigSchema('test.valid', ConfigType.INTEGER, 5, 'Test', min_value=0, max_value=10)
        self.manager.register_schema(schema)

        # Valid value
        self.assertTrue(self.manager.set('test.valid', 7))
        self.assertEqual(self.manager.get('test.valid'), 7)

        # Invalid value
        with self.assertRaises(ValueError):
            self.manager.set('test.valid', 15)

    def test_change_history(self):
        """Test change history tracking."""
        self.manager.set('test.key', 'value1')
        self.manager.set('test.key', 'value2')

        history = self.manager.get_change_history('test.key')
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].old_value, None)
        self.assertEqual(history[0].new_value, 'value1')
        self.assertEqual(history[1].old_value, 'value1')
        self.assertEqual(history[1].new_value, 'value2')

    def test_watchers(self):
        """Test configuration watchers."""
        callback_called = []
        def callback(key, old_val, new_val):
            callback_called.append((key, old_val, new_val))

        self.manager.watch('test.watch', callback)
        self.manager.set('test.watch', 'new_value')

        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0], ('test.watch', None, 'new_value'))

    def test_sections(self):
        """Test section operations."""
        # Set section values
        section_data = {'key1': 'val1', 'key2': 'val2'}
        self.manager.set_section('TestSection', section_data)

        # Get section
        retrieved = self.manager.get_section('TestSection')
        self.assertEqual(retrieved['key1'], 'val1')
        self.assertEqual(retrieved['key2'], 'val2')

    def test_list_keys(self):
        """Test key listing."""
        self.manager.set('section1.key1', 'val1')
        self.manager.set('section1.key2', 'val2')
        self.manager.set('section2.key1', 'val3')

        all_keys = self.manager.list_keys()
        self.assertIn('section1.key1', all_keys)
        self.assertIn('section2.key1', all_keys)

        section_keys = self.manager.list_keys('section1')
        self.assertEqual(len(section_keys), 2)
        self.assertIn('section1.key1', section_keys)

    def test_reset_to_defaults(self):
        """Test reset to defaults."""
        schema = ConfigSchema('test.reset', ConfigType.STRING, 'default', 'Test')
        self.manager.register_schema(schema)

        self.manager.set('test.reset', 'modified')
        self.assertEqual(self.manager.get('test.reset'), 'modified')

        self.manager.reset_to_defaults()
        self.assertEqual(self.manager.get('test.reset'), 'default')

    def test_validate_all(self):
        """Test full validation."""
        schema = ConfigSchema('test.valid', ConfigType.INTEGER, 5, 'Test', min_value=0, max_value=10)
        self.manager.register_schema(schema)
        out_of_range_schema = ConfigSchema(
            'test.out_of_range', ConfigType.INTEGER, 5, 'Test',
            min_value=0, max_value=10
        )
        self.manager.register_schema(out_of_range_schema)

        self.manager.set('test.valid', 7)  # Valid
        self.manager.set('test.invalid', 'string')  # No schema, should be ok
        self.manager.config['test.out_of_range'] = 15  # Invalid

        errors = self.manager.validate_all()
        self.assertIn('test.out_of_range', errors)


class TestUnifiedConfigManagerFileOperations(unittest.TestCase):
    """File operation tests for UnifiedConfigManager."""

    def setUp(self):
        self.manager = UnifiedConfigManager()

    def test_load_save_json(self):
        """Test JSON file operations."""
        test_data = {'section': {'key': 'value', 'number': 42}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            json_file = f.name

        try:
            # Load
            self.assertTrue(self.manager.load_from_file(json_file))
            self.assertEqual(self.manager.get('section.key'), 'value')
            self.assertEqual(self.manager.get('section.number'), 42)

            # Modify and save
            self.manager.set('section.new_key', 'new_value')
            save_file = json_file.replace('.json', '_saved.json')
            self.assertTrue(self.manager.save_to_file(save_file, 'json'))

            # Verify saved file
            with open(save_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data['section']['new_key'], 'new_value')

        finally:
            os.unlink(json_file)
            if os.path.exists(save_file):
                os.unlink(save_file)

    def test_load_save_yaml(self):
        """Test YAML file operations."""
        test_data = {'section': {'key': 'value', 'number': 42}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_data, f)
            yaml_file = f.name

        try:
            # Load
            self.assertTrue(self.manager.load_from_file(yaml_file))
            self.assertEqual(self.manager.get('section.key'), 'value')

            # Save
            save_file = yaml_file.replace('.yaml', '_saved.yaml')
            self.assertTrue(self.manager.save_to_file(save_file, 'yaml'))

        finally:
            os.unlink(yaml_file)
            if os.path.exists(save_file):
                os.unlink(save_file)

    def test_load_save_ini(self):
        """Test INI file operations."""
        ini_content = """[Section]
key = value
number = 42
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(ini_content)
            ini_file = f.name

        try:
            # Load
            self.assertTrue(self.manager.load_from_file(ini_file))
            self.assertEqual(self.manager.get('Section.key'), 'value')
            self.assertEqual(self.manager.get('Section.number'), 42)

            # Save
            save_file = ini_file.replace('.ini', '_saved.ini')
            self.assertTrue(self.manager.save_to_file(save_file, 'ini'))

        finally:
            os.unlink(ini_file)
            if os.path.exists(save_file):
                os.unlink(save_file)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        self.assertFalse(self.manager.load_from_file('nonexistent.json'))

    def test_save_invalid_format(self):
        """Test saving with invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.invalid')
            self.assertFalse(self.manager.save_to_file(save_path, 'invalid'))


class TestUnifiedConfigManagerIntegration(unittest.TestCase):
    """Integration tests for UnifiedConfigManager."""

    def setUp(self):
        self.manager = UnifiedConfigManager('config/config.ini')

    def test_real_config_loading(self):
        """Test loading real config file."""
        # Should load from config.ini
        self.assertIsNotNone(self.manager.get('General.resolution_scale'))
        self.assertIsNotNone(self.manager.get('SystemConstants.node_energy_cap'))

    def test_backward_compatibility_functions(self):
        """Test backward compatibility functions."""
        # Test get_config
        value = self.manager.get_float('General', 'resolution_scale', 0.5)
        self.assertIsInstance(value, float)

        # Test set_config
        set_config('Test', 'key', 'value')
        self.assertEqual(get_config_manager().get('Test.key'), 'value')

    def test_section_getters(self):
        """Test section getter functions."""
        system_consts = get_system_constants()
        self.assertIsInstance(system_consts, dict)
        self.assertIn('node_energy_cap', system_consts)

        enhanced_nodes = get_enhanced_nodes_config()
        self.assertIsInstance(enhanced_nodes, dict)

        learning = get_learning_config()
        self.assertIsInstance(learning, dict)

    def test_thread_safety(self):
        """Test thread safety of operations."""
        results = []

        def worker():
            for i in range(100):
                self.manager.set(f'test.thread_{i}', f'value_{i}')
                results.append(self.manager.get(f'test.thread_{i}'))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 500)  # 5 threads * 100 operations


class TestUnifiedConfigManagerEdgeCases(unittest.TestCase):
    """Edge case tests for UnifiedConfigManager."""

    def setUp(self):
        self.manager = UnifiedConfigManager()

    def test_empty_config(self):
        """Test with empty configuration."""
        empty_manager = UnifiedConfigManager()
        self.assertEqual(empty_manager.get('nonexistent'), None)

    def test_deeply_nested_keys(self):
        """Test deeply nested configuration keys."""
        self.manager.set('level1.level2.level3.key', 'value')
        self.assertEqual(self.manager.get('level1.level2.level3.key'), 'value')

        section = self.manager.get_section('level1.level2')
        self.assertEqual(section['level3.key'], 'value')

    def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        self.manager.set('test_key.with-dashes.and.dots', 'value')
        self.assertEqual(self.manager.get('test_key.with-dashes.and.dots'), 'value')

    def test_large_config_values(self):
        """Test with large configuration values."""
        large_list = list(range(10000))
        self.manager.set('test.large_list', large_list)
        retrieved = self.manager.get('test.large_list')
        self.assertEqual(len(retrieved), 10000)

    def test_none_values(self):
        """Test None value handling."""
        self.manager.set('test.none', None)
        self.assertIsNone(self.manager.get('test.none'))

    def test_type_coercion(self):
        """Test type coercion in INI loading."""
        # Simulate INI loading behavior
        self.manager.config['test.int'] = '42'
        self.manager.config['test.float'] = '3.14'
        self.manager.config['test.bool'] = 'true'

        # Values should be converted when accessed via backward compatibility
        self.assertEqual(self.manager.get_int('test', 'int', 0), 42)
        self.assertEqual(self.manager.get_float('test', 'float', 0.0), 3.14)
        self.assertEqual(self.manager.get_bool('test', 'bool', False), True)


class TestUnifiedConfigManagerErrorHandling(unittest.TestCase):
    """Error handling tests for UnifiedConfigManager."""

    def setUp(self):
        self.manager = UnifiedConfigManager()

    def test_invalid_schema_registration(self):
        """Test registering invalid schema."""
        # Should not raise exception for basic schema
        schema = ConfigSchema('test', ConfigType.STRING, 'default', 'Test')
        self.manager.register_schema(schema)
        self.assertIn('test', self.manager.schemas)

    def test_validation_errors(self):
        """Test various validation errors."""
        schema = ConfigSchema(
            'test.int', ConfigType.INTEGER, 5, 'Test int',
            min_value=0, max_value=10
        )
        self.manager.register_schema(schema)

        # Test invalid type
        with self.assertRaises(ValueError):
            self.manager.set('test.int', 'not_a_number')

        # Test out of range
        with self.assertRaises(ValueError):
            self.manager.set('test.int', 100)

    def test_watcher_exceptions(self):
        """Test exception handling in watchers."""
        def failing_callback(key, old_val, new_val):
            raise ValueError("Watcher failed")

        self.manager.watch('test.watch', failing_callback)
        # Should not raise exception when watcher fails
        self.manager.set('test.watch', 'value')

    def test_file_operation_errors(self):
        """Test file operation error handling."""
        # Test loading from directory
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(self.manager.load_from_file(tmpdir))

        # Test saving to invalid path
        self.assertFalse(self.manager.save_to_file('C:/Windows/System32/test.json'))

    def test_ini_parsing_errors(self):
        """Test INI parsing error handling."""
        invalid_ini = """[Section
invalid syntax = value
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(invalid_ini)
            ini_file = f.name

        try:
            # Should handle parsing errors gracefully
            result = self.manager.load_from_file(ini_file)
            # May succeed or fail depending on configparser behavior
            self.assertIsInstance(result, bool)
        finally:
            os.unlink(ini_file)


class TestUnifiedConfigManagerPerformance(unittest.TestCase):
    """Performance tests for UnifiedConfigManager."""

    def setUp(self):
        self.manager = UnifiedConfigManager()

    def test_bulk_operations_performance(self):
        """Test performance of bulk operations."""
        start_time = time.time()

        # Bulk set operations
        for i in range(1000):
            self.manager.set(f'perf.test_{i}', f'value_{i}')

        set_time = time.time() - start_time

        # Bulk get operations
        start_time = time.time()
        for i in range(1000):
            self.manager.get(f'perf.test_{i}')

        get_time = time.time() - start_time

        # Should complete in reasonable time
        self.assertLess(set_time, 1.0)  # Less than 1 second for 1000 sets
        self.assertLess(get_time, 1.0)  # Less than 1 second for 1000 gets

    def test_large_config_loading(self):
        """Test loading large configuration."""
        large_config = {}
        for i in range(100):
            section = f'Section{i}'
            large_config[section] = {}
            for j in range(10):
                large_config[section][f'key{j}'] = f'value_{i}_{j}'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_config, f)
            config_file = f.name

        try:
            start_time = time.time()
            self.assertTrue(self.manager.load_from_file(config_file))
            load_time = time.time() - start_time

            self.assertLess(load_time, 5.0)  # Should load in under 5 seconds

        finally:
            os.unlink(config_file)

    def test_concurrent_access_performance(self):
        """Test performance under concurrent access."""
        def concurrent_worker():
            for i in range(100):
                self.manager.set(f'concurrent.key_{threading.current_thread().ident}_{i}', f'value_{i}')
                key = f'concurrent.key_{threading.current_thread().ident}_{i-1}' if i > 0 else 'nonexistent'
                self.manager.get(key)

        threads = [threading.Thread(target=concurrent_worker) for _ in range(10)]

        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_time = time.time() - start_time
        self.assertLess(total_time, 10.0)  # Should complete in under 10 seconds


class TestUnifiedConfigManagerRealWorldUsage(unittest.TestCase):
    """Real-world usage scenario tests for UnifiedConfigManager."""

    def setUp(self):
        self.manager = UnifiedConfigManager()

    def test_simulation_config_scenario(self):
        """Test typical simulation configuration scenario."""
        # Setup simulation config
        sim_config = {
            'simulation': {
                'enabled': True,
                'max_steps': 10000,
                'time_step': 0.01,
                'random_seed': 42
            },
            'neural_network': {
                'num_nodes': 1000,
                'num_connections': 5000,
                'learning_rate': 0.01
            },
            'visualization': {
                'enabled': True,
                'update_interval': 100,
                'resolution': '1920x1080'
            }
        }

        # Flatten and set
        for section, values in sim_config.items():
            self.manager.set_section(section, values)

        # Verify configuration
        self.assertTrue(self.manager.get('simulation.enabled'))
        self.assertEqual(self.manager.get('neural_network.num_nodes'), 1000)
        self.assertEqual(self.manager.get('visualization.resolution'), '1920x1080')

    def test_dynamic_config_updates(self):
        """Test dynamic configuration updates during runtime."""
        update_count = {'count': 0}

        def update_callback(_key, _old_val, _new_val):
            update_count['count'] += 1

        # Watch for changes
        self.manager.watch('dynamic.param', update_callback)

        # Simulate dynamic updates
        for i in range(5):
            self.manager.set('dynamic.param', f'value_{i}')

        self.assertEqual(update_count['count'], 5)

    def test_config_export_import_workflow(self):
        """Test configuration export/import workflow."""
        # Set some config
        self.manager.set('workflow.test1', 'value1')
        self.manager.set('workflow.test2', 42)

        # Export to JSON string
        json_export = self.manager.export_config('json')
        self.assertIn('workflow', json_export)
        self.assertIn('value1', json_export)

        # Create new manager and import
        new_manager = UnifiedConfigManager()
        imported_config = json.loads(json_export)
        for section, values in imported_config.items():
            new_manager.set_section(section, values)

        self.assertEqual(new_manager.get('workflow.test1'), 'value1')
        self.assertEqual(new_manager.get('workflow.test2'), 42)

    def test_multi_environment_config(self):
        """Test configuration for multiple environments."""
        environments = ['development', 'staging', 'production']

        for env in environments:
            self.manager.set(f'{env}.debug_mode', env == 'development')
            self.manager.set(f'{env}.log_level', 'DEBUG' if env == 'development' else 'INFO')
            self.manager.set(f'{env}.max_connections', 10 if env == 'development' else 100)

        # Verify environment-specific configs
        self.assertTrue(self.manager.get('development.debug_mode'))
        self.assertFalse(self.manager.get('production.debug_mode'))
        self.assertEqual(self.manager.get('staging.max_connections'), 100)


class TestConsolidatedConstants(unittest.TestCase):
    """Tests for consolidated constants."""

    def test_constants_defined(self):
        """Test that all constant groups are defined."""
        # Test that constants are dictionaries
        self.assertIsInstance(UI_CONSTANTS, dict)
        self.assertIsInstance(ERROR_MESSAGES, dict)
        self.assertIsInstance(DEFAULT_VALUES, dict)

    def test_key_accessibility(self):
        """Test that keys are accessible."""
        # Test UI constants
        self.assertIn('SIMULATION_STATUS_RUNNING', UI_CONSTANTS)
        self.assertEqual(UI_CONSTANTS['SIMULATION_STATUS_RUNNING'], 'Running')

        # Test default values
        self.assertIn('ENERGY_CAP', DEFAULT_VALUES)
        self.assertIsInstance(DEFAULT_VALUES['ENERGY_CAP'], float)

    def test_no_duplicates(self):
        """Test that there are no duplicate keys across constants."""
        all_constants = [
            UI_CONSTANTS, ERROR_MESSAGES, LOG_MESSAGES, FILE_PATHS,
            NODE_PROPERTIES, CONNECTION_PROPERTIES, SYSTEM_STATES,
            NODE_STATES, NODE_TYPES, CONNECTION_TYPES, PERFORMANCE_METRICS,
            THRESHOLDS, DEFAULT_VALUES, PRINT_PATTERNS, EXCEPTION_TYPES,
            FUNCTION_NAMES, CLASS_NAMES
        ]

        all_keys = []
        for const_dict in all_constants:
            all_keys.extend(const_dict.keys())

        # Check for duplicates
        unique_keys = set(all_keys)
        duplicates = [k for k in all_keys if all_keys.count(k) > 1]
        if duplicates:
            print(f"Duplicate keys: {duplicates}")
        self.assertEqual(len(all_keys), len(unique_keys), "Duplicate keys found in constants")
if __name__ == '__main__':
    unittest.main()

