"""
Unified Launcher for Neural Simulation System.

This module provides the main entry point for launching the neural simulation system
with different profiles and configurations. It handles service registration, dependency
injection, and system initialization.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import traceback
# Standard library imports
from typing import Any, Dict

# Third-party imports
try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

# Interface imports
from src.core.interfaces.adaptive_configuration import IAdaptiveConfiguration
from src.core.interfaces.cloud_deployment import ICloudDeployment
from src.core.interfaces.configuration_service import IConfigurationService
from src.core.interfaces.distributed_coordinator import IDistributedCoordinator
from src.core.interfaces.energy_manager import IEnergyManager
from src.core.interfaces.event_coordinator import IEventCoordinator
from src.core.interfaces.fault_tolerance import IFaultTolerance
from src.core.interfaces.gpu_accelerator import IGPUAccelerator
from src.core.interfaces.graph_manager import IGraphManager
from src.core.interfaces.learning_engine import ILearningEngine
from src.core.interfaces.load_balancer import ILoadBalancer
from src.core.interfaces.ml_optimizer import IMLOptimizer
from src.core.interfaces.neural_processor import INeuralProcessor
from src.core.interfaces.performance_monitor import IPerformanceMonitor
from src.core.interfaces.real_time_analytics import IRealTimeAnalytics
from src.core.interfaces.real_time_visualization import IRealTimeVisualization
from src.core.interfaces.sensory_processor import ISensoryProcessor
from src.core.interfaces.service_registry import IServiceRegistry
from src.core.interfaces.simulation_coordinator import ISimulationCoordinator
# Service imports
from src.core.services.adaptive_configuration_service import \
    AdaptiveConfigurationService
from src.core.services.cloud_deployment_service import CloudDeploymentService
from src.core.services.configuration_service import ConfigurationService
from src.core.services.distributed_coordinator_service import \
    DistributedCoordinatorService
from src.core.services.energy_management_service import EnergyManagementService
from src.core.services.event_coordination_service import \
    EventCoordinationService
from src.core.services.fault_tolerance_service import FaultToleranceService
from src.core.services.gpu_accelerator_service import GPUAcceleratorService
from src.core.services.graph_management_service import GraphManagementService
from src.core.services.learning_service import LearningService
from src.core.services.load_balancing_service import LoadBalancingService
from src.core.services.ml_optimizer_service import MLOptimizerService
from src.core.services.neural_processing_service import NeuralProcessingService
from src.core.services.performance_monitoring_service import \
    PerformanceMonitoringService
from src.core.services.real_time_analytics_service import \
    RealTimeAnalyticsService
from src.core.services.real_time_visualization_service import \
    RealTimeVisualizationService
from src.core.services.sensory_processing_service import \
    SensoryProcessingService
from src.core.services.service_registry import ServiceRegistry
from src.core.services.simulation_coordinator import SimulationCoordinator
# Local imports
from src.utils.print_utils import (print_error, print_info, print_success,
                                   print_warning)

# Optional imports
try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None



class UnifiedLauncher:
    """
    Main launcher class for the neural simulation system.

    Handles profile-based launching, system capacity testing, and UI initialization.
    """

    def __init__(self, service_registry: IServiceRegistry):
        """
        Initialize the UnifiedLauncher with a service registry.

        Args:
            service_registry: The service registry for dependency injection.
        """
        self.service_registry = service_registry
        self.config_service = service_registry.resolve(IConfigurationService)
        self.config_service.load_configuration('config/config.ini')
        self.profiles = {
            'full': {
                'description': 'Full simulation with UI',
                'ui_module': 'src.ui.ui_engine',
                'ui_function': 'run_ui',
                'logging_level': 'INFO',
                'performance_mode': True
            }
        }

    def test_basic_imports(self) -> bool:
        """
        Test critical module imports to ensure system readiness.

        Returns:
            True if all imports succeed, False otherwise.
        """
        print_info("Testing critical imports...")
        critical_modules = [
            'numpy', 'torch', 'dearpygui', 'src.ui.ui_engine'
        ]
        failed_imports = []
        for module_name in critical_modules:
            try:
                __import__(module_name)
                print(f"  [OK] {module_name}")
            except ImportError as e:
                print(f"  [FAIL] {module_name}: {e}")
                failed_imports.append(module_name)
            except Exception as e:
                logging.exception("Broad exception caught during import test for %s", module_name)
                print(f"  [WARN] {module_name}: {e}")
                failed_imports.append(module_name)
        if failed_imports:
            print(f"\nFailed to import: {', '.join(failed_imports)}")
            return False
        print_success("All critical imports successful!")
        return True
    def apply_performance_optimizations(self):
        """
        Apply performance optimizations for better system performance.
        """
        print_info("Applying performance optimizations...")
        os.environ['PYTHONOPTIMIZE'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        logging.getLogger().setLevel(logging.WARNING)
        if torch is not None:
            torch.set_num_threads(1)
        print_success("Performance optimizations applied!")
    def test_system_capacity(self) -> Dict[str, Any]:
        """
        Test system capacity including memory and CPU resources.

        Returns:
            Dictionary with capacity information.
        """
        print_info("Testing system capacity...")
        if psutil is None:
            print_warning("  psutil not available, skipping capacity test")
            return {'sufficient_memory': True, 'sufficient_cpu': True}
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            capacity_info = {
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'cpu_count': cpu_count,
                'sufficient_memory': memory.available > 0.5 * 1024**3,
                'sufficient_cpu': cpu_count >= 2
            }
            print(f"  Memory: {capacity_info['memory_available_gb']:.1f}GB available")
            print(f"  CPU: {capacity_info['cpu_count']} cores")
            return capacity_info
        except Exception as e:
            logging.exception("Broad exception caught during capacity test")
            print(f"  Capacity test failed: {e}")
            return {'sufficient_memory': True, 'sufficient_cpu': True}
    def launch_with_profile(self, profile: str) -> bool:
        """
        Launch the system with the specified profile.

        Args:
            profile: The profile name to launch.

        Returns:
            True if launch succeeds, False otherwise.
        """
        if profile not in self.profiles:
            print(f"Unknown profile: {profile}")
            print(f"Available profiles: {', '.join(self.profiles.keys())}")
            return False
        config = self.profiles[profile]
        print(f"Launching with profile: {profile} - {config['description']}")
        if not self.test_basic_imports():
            print_error("Import test failed, cannot launch")
            return False
        capacity = self.test_system_capacity()
        if not capacity.get('sufficient_memory', True):
            print_error("Insufficient memory available")
            return False
        if config.get('performance_mode', False):
            self.apply_performance_optimizations()
        log_level = logging.DEBUG
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logging.getLogger().setLevel(log_level)
        try:
            if 'ui_module' in config:
                # Pass services to the UI
                return self._launch_ui(config, self.service_registry)
            else:
                print(f"Invalid configuration for profile: {profile}")
                return False
        except Exception as e:
            logging.exception("Broad exception caught during launch")
            print(f"Launch failed: {e}")
            traceback.print_exc()
            return False
    def _launch_ui(self, config: Dict[str, Any], service_registry: IServiceRegistry) -> bool:
        """
        Launch the UI component based on configuration.

        Args:
            config: Profile configuration dictionary.
            service_registry: The service registry instance.

        Returns:
            True if UI launch succeeds, False otherwise.
        """
        try:
            # Dynamically import the UI module
            parts = config['ui_module'].split('.')
            module_name = ".".join(parts)
            ui_module = __import__(module_name, fromlist=[parts[-1]])

            if 'ui_class' in config:
                ui_class = getattr(ui_module, config['ui_class'])
                # Pass the service registry to the UI class constructor
                ui_instance = ui_class(service_registry)
                ui_instance.run()

            elif 'ui_function' in config:
                ui_function = getattr(ui_module, config['ui_function'])
                # Pass the service registry to the UI function
                ui_function(service_registry)
            else:
                print_error("No UI class or function specified")
                return False
            return True
        except Exception as e:
            logging.exception("Broad exception caught during UI launch")
            print(f"UI launch failed: {e}")
            traceback.print_exc()
            return False
    def show_help(self):
        """
        Display help information for the launcher.
        """
        print_info("Unified Launcher - Neural Simulation System")
        print_info("=" * 50)
        print_info("\nAvailable profiles:")
        for profile, config in self.profiles.items():
            print_info(f"  {profile:12} - {config['description']}")
        print_info("\nUsage:")
        print_info("  python unified_launcher.py")
        print_info("  python unified_launcher.py --help")
        print_info("\nLaunches the full UI by default.")


def main():
    """
    Main entry point for the unified launcher.

    Initializes services, registers them, and launches the system.
    """
    # 1. Composition Root: Initialize Service Registry
    service_registry = ServiceRegistry()

    # 2. Register Core Services

    service_registry.register(IConfigurationService, ConfigurationService)
    service_registry.register(IPerformanceMonitor, PerformanceMonitoringService)
    service_registry.register(IAdaptiveConfiguration, AdaptiveConfigurationService)
    service_registry.register(ICloudDeployment, CloudDeploymentService)
    service_registry.register(IDistributedCoordinator, DistributedCoordinatorService)
    service_registry.register(IEnergyManager, EnergyManagementService)
    service_registry.register(IEventCoordinator, EventCoordinationService)
    service_registry.register(IFaultTolerance, FaultToleranceService)
    service_registry.register(IGPUAccelerator, GPUAcceleratorService)
    service_registry.register(IGraphManager, GraphManagementService)
    service_registry.register(ILearningEngine, LearningService)
    service_registry.register(ILoadBalancer, LoadBalancingService)
    service_registry.register(IMLOptimizer, MLOptimizerService)
    service_registry.register(INeuralProcessor, NeuralProcessingService)
    service_registry.register(IRealTimeAnalytics, RealTimeAnalyticsService)
    service_registry.register(IRealTimeVisualization, RealTimeVisualizationService)
    service_registry.register(ISensoryProcessor, SensoryProcessingService)
    service_registry.register(ISimulationCoordinator, SimulationCoordinator)

    # 3. Initialize and run the launcher
    launcher = UnifiedLauncher(service_registry)

    if '--help' in sys.argv:
        launcher.show_help()
        return 0
    profile = 'full'
    try:
        # Test service resolution
        all_services = [
            IAdaptiveConfiguration, ICloudDeployment, IDistributedCoordinator,
            IEnergyManager, IEventCoordinator, IFaultTolerance, IGPUAccelerator,
            IGraphManager, ILearningEngine, ILoadBalancer, IMLOptimizer,
            INeuralProcessor, IPerformanceMonitor, IRealTimeAnalytics,
            IRealTimeVisualization, ISensoryProcessor, ISimulationCoordinator
        ]
        for service_interface in all_services:
            service_registry.resolve(service_interface)
        print_success("All services resolved successfully!")
    except Exception as e:
        logging.exception("Broad exception caught during service resolution")
        print_error(f"Service resolution failed: {e}")
        return 1
    success = launcher.launch_with_profile(profile)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
