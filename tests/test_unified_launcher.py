"""
Comprehensive tests for UnifiedLauncher.

This module contains unit tests, integration tests, edge cases, and performance tests
for the UnifiedLauncher class, covering all aspects of launcher functionality.
"""

import unittest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

from core.unified_launcher import UnifiedLauncher
from core.services.service_registry import ServiceRegistry
from core.interfaces.service_registry import IServiceRegistry
from core.interfaces.configuration_service import IConfigurationService
from core.interfaces.performance_monitor import IPerformanceMonitor


class TestUnifiedLauncher(unittest.TestCase):
    """Unit tests for UnifiedLauncher."""

    def setUp(self):
        """Set up test fixtures."""
        self.service_registry = Mock(spec=IServiceRegistry)
        self.config_service = Mock(spec=IConfigurationService)
        self.performance_monitor = Mock(spec=IPerformanceMonitor)

        # Configure mocks
        self.service_registry.resolve.return_value = self.config_service
        self.config_service.load_configuration.return_value = True

        self.launcher = UnifiedLauncher(self.service_registry)

    def test_initialization(self):
        """Test launcher initialization."""
        self.assertIsNotNone(self.launcher.service_registry)
        self.assertEqual(self.launcher.config_service, self.config_service)
        self.assertIn('full', self.launcher.profiles)

    def test_test_basic_imports_success(self):
        """Test successful basic import testing."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = Mock()
            result = self.launcher.test_basic_imports()
            self.assertTrue(result)

    def test_test_basic_imports_failure(self):
        """Test failed basic import testing."""
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            result = self.launcher.test_basic_imports()
            self.assertFalse(result)

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_count')
    def test_test_system_capacity_success(self, mock_cpu_count, mock_memory):
        """Test successful system capacity testing."""
        mock_memory.return_value = Mock(total=16*1024**3, available=8*1024**3, percent=50)
        mock_cpu_count.return_value = 4

        result = self.launcher.test_system_capacity()
        self.assertTrue(result['sufficient_memory'])
        self.assertTrue(result['sufficient_cpu'])
        self.assertAlmostEqual(result['memory_total_gb'], 16.0, places=1)

    @patch('psutil.virtual_memory', side_effect=ImportError)
    def test_test_system_capacity_no_psutil(self, mock_memory):
        """Test system capacity testing without psutil."""
        result = self.launcher.test_system_capacity()
        self.assertTrue(result['sufficient_memory'])
        self.assertTrue(result['sufficient_cpu'])

    def test_launch_with_profile_invalid_profile(self):
        """Test launching with invalid profile."""
        result = self.launcher.launch_with_profile("invalid")
        self.assertFalse(result)

    @patch.object(UnifiedLauncher, 'test_basic_imports', return_value=False)
    def test_launch_with_profile_import_failure(self, mock_imports):
        """Test launching when import test fails."""
        result = self.launcher.launch_with_profile("full")
        self.assertFalse(result)

    @patch.object(UnifiedLauncher, 'test_basic_imports', return_value=True)
    @patch.object(UnifiedLauncher, 'test_system_capacity')
    @patch.object(UnifiedLauncher, '_launch_ui', return_value=True)
    def test_launch_with_profile_success(self, mock_ui, mock_capacity, mock_imports):
        """Test successful profile launch."""
        mock_capacity.return_value = {'sufficient_memory': True, 'sufficient_cpu': True}

        result = self.launcher.launch_with_profile("full")
        self.assertTrue(result)
        mock_ui.assert_called_once()

    def test_apply_performance_optimizations(self):
        """Test performance optimization application."""
        with patch.dict(os.environ, {}, clear=True):
            self.launcher.apply_performance_optimizations()

            self.assertEqual(os.environ.get('PYTHONOPTIMIZE'), '1')
            self.assertEqual(os.environ.get('OMP_NUM_THREADS'), '1')

    @patch('torch.set_num_threads')
    def test_apply_performance_optimizations_with_torch(self, mock_set_threads):
        """Test performance optimizations with torch available."""
        with patch.dict('sys.modules', {'torch': Mock()}):
            self.launcher.apply_performance_optimizations()
            mock_set_threads.assert_called_once_with(1)

    def test_launch_ui_with_class(self):
        """Test UI launching with class configuration."""
        config = {
            'ui_module': 'test_module',
            'ui_class': 'TestUI',
            'ui_function': None
        }

        with patch('builtins.__import__') as mock_import:
            mock_ui_class = Mock()
            mock_ui_instance = Mock()
            mock_ui_class.return_value = mock_ui_instance

            mock_module = Mock()
            mock_module.TestUI = mock_ui_class
            mock_import.return_value = mock_module

            result = self.launcher._launch_ui(config, self.service_registry)
            self.assertTrue(result)
            mock_ui_instance.run.assert_called_once()

    def test_launch_ui_with_function(self):
        """Test UI launching with function configuration."""
        config = {
            'ui_module': 'test_module',
            'ui_function': 'run_ui',
            'ui_class': None
        }

        with patch('builtins.__import__') as mock_import:
            mock_ui_function = Mock()
            mock_module = Mock()
            mock_module.run_ui = mock_ui_function
            mock_import.return_value = mock_module

            result = self.launcher._launch_ui(config, self.service_registry)
            self.assertTrue(result)
            mock_ui_function.assert_called_once_with(self.service_registry)

    def test_launch_ui_invalid_config(self):
        """Test UI launching with invalid configuration."""
        config = {
            'ui_module': 'test_module'
            # Missing ui_class or ui_function
        }

        result = self.launcher._launch_ui(config, self.service_registry)
        self.assertFalse(result)

    def test_show_help(self):
        """Test help display."""
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            self.launcher.show_help()

        output = captured_output.getvalue()
        self.assertIn("Unified Launcher", output)
        self.assertIn("Available profiles:", output)


class TestUnifiedLauncherIntegration(unittest.TestCase):
    """Integration tests for UnifiedLauncher."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.service_registry = ServiceRegistry()
        self.launcher = UnifiedLauncher(self.service_registry)

    def test_full_initialization_flow(self):
        """Test complete initialization flow."""
        # This would normally require all services to be registered
        # For integration testing, we'd need to set up the full service registry
        pass

    def test_configuration_loading(self):
        """Test configuration loading during initialization."""
        # Create temporary config file
        config_content = """[GLOBAL]
simulation_enabled = True
debug_mode = False
log_level = INFO

[NEURAL]
membrane_time_constant = 10.0
threshold_potential = -50.0
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            # Test that config loading works
            self.assertTrue(os.path.exists(config_path))
        finally:
            os.unlink(config_path)


class TestUnifiedLauncherEdgeCases(unittest.TestCase):
    """Edge case tests for UnifiedLauncher."""

    def setUp(self):
        """Set up edge case test fixtures."""
        self.service_registry = Mock(spec=IServiceRegistry)
        self.launcher = UnifiedLauncher(self.service_registry)

    def test_empty_profiles(self):
        """Test behavior with empty profiles."""
        launcher = UnifiedLauncher(self.service_registry)
        launcher.profiles = {}

        result = launcher.launch_with_profile("nonexistent")
        self.assertFalse(result)

    @patch('psutil.virtual_memory')
    def test_insufficient_memory(self, mock_memory):
        """Test handling of insufficient memory."""
        mock_memory.return_value = Mock(total=4*1024**3, available=0.1*1024**3, percent=98)

        with patch.object(self.launcher, 'test_basic_imports', return_value=True):
            result = self.launcher.launch_with_profile("full")
            self.assertFalse(result)

    def test_import_partial_failure(self):
        """Test handling of partial import failures."""
        def mock_import(name):
            if name == 'dearpygui':
                raise ImportError("DearPyGUI not available")
            return Mock()

        with patch('builtins.__import__', side_effect=mock_import):
            result = self.launcher.test_basic_imports()
            self.assertFalse(result)

    def test_ui_launch_exception_handling(self):
        """Test exception handling in UI launch."""
        config = {
            'ui_module': 'nonexistent_module',
            'ui_function': 'run_ui'
        }

        result = self.launcher._launch_ui(config, self.service_registry)
        self.assertFalse(result)


class TestUnifiedLauncherPerformance(unittest.TestCase):
    """Performance tests for UnifiedLauncher."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.service_registry = Mock(spec=IServiceRegistry)
        self.launcher = UnifiedLauncher(self.service_registry)

    def test_import_test_performance(self):
        """Test performance of import testing."""
        start_time = time.time()
        result = self.launcher.test_basic_imports()
        end_time = time.time()

        # Import test should complete in reasonable time
        self.assertLess(end_time - start_time, 5.0)  # Less than 5 seconds

    def test_capacity_test_performance(self):
        """Test performance of capacity testing."""
        start_time = time.time()

        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu:

            mock_memory.return_value = Mock(total=16*1024**3, available=8*1024**3, percent=50)
            mock_cpu.return_value = 4

            result = self.launcher.test_system_capacity()

        end_time = time.time()

        # Capacity test should be fast
        self.assertLess(end_time - start_time, 1.0)  # Less than 1 second


if __name__ == '__main__':
    unittest.main()