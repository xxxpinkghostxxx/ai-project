
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Dict, Any, List, Optional, Tuple, Callable

import logging
import traceback
import argparse

from utils.print_utils import print_info, print_success, print_error, print_warning
from core.services.service_registry import ServiceRegistry
from core.interfaces.service_registry import IServiceRegistry
from core.services.configuration_service import ConfigurationService
from core.interfaces.configuration_service import IConfigurationService
from core.services.performance_monitoring_service import PerformanceMonitoringService
from core.interfaces.performance_monitor import IPerformanceMonitor



class UnifiedLauncher:
    def __init__(self, service_registry: IServiceRegistry):
        self.service_registry = service_registry
        self.config_service = service_registry.resolve(IConfigurationService)
        self.config_service.load_configuration('config.ini')
        self.profiles = {
            'full': {
                'description': 'Full simulation with UI',
                'ui_module': 'ui.ui_engine',
                'ui_function': 'run_ui',
                'logging_level': 'INFO',
                'performance_mode': True
            }
        }

    def test_basic_imports(self) -> bool:
        print_info("Testing critical imports...")
        critical_modules = [
            'numpy', 'torch', 'dearpygui', 'ui.ui_engine'
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
                print(f"  [WARN] {module_name}: {e}")
                failed_imports.append(module_name)
        if failed_imports:
            print(f"\nFailed to import: {', '.join(failed_imports)}")
            return False
        print_success("All critical imports successful!")
        return True
    def apply_performance_optimizations(self):
        print_info("Applying performance optimizations...")
        os.environ['PYTHONOPTIMIZE'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        logging.getLogger().setLevel(logging.WARNING)
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
        print_success("Performance optimizations applied!")
    def test_system_capacity(self) -> Dict[str, Any]:
        print_info("Testing system capacity...")
        try:
            import psutil
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
        except ImportError:
            print_warning("  psutil not available, skipping capacity test")
            return {'sufficient_memory': True, 'sufficient_cpu': True}
        except Exception as e:
            print(f"  Capacity test failed: {e}")
            return {'sufficient_memory': True, 'sufficient_cpu': True}
    def launch_with_profile(self, profile: str) -> bool:
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
            print(f"Launch failed: {e}")
            traceback.print_exc()
            return False
    def _launch_ui(self, config: Dict[str, Any], service_registry: IServiceRegistry) -> bool:
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
            print(f"UI launch failed: {e}")
            traceback.print_exc()
            return False
    def show_help(self):
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
    # 1. Composition Root: Initialize Service Registry
    service_registry = ServiceRegistry()

    # 2. Register Core Services
    from core.services.adaptive_configuration_service import AdaptiveConfigurationService
    from core.services.cloud_deployment_service import CloudDeploymentService
    from core.services.distributed_coordinator_service import DistributedCoordinatorService
    from core.services.energy_management_service import EnergyManagementService
    from core.services.event_coordination_service import EventCoordinationService
    from core.services.fault_tolerance_service import FaultToleranceService
    from core.services.gpu_accelerator_service import GPUAcceleratorService
    from core.services.graph_management_service import GraphManagementService
    from core.services.learning_service import LearningService
    from core.services.load_balancing_service import LoadBalancingService
    from core.services.ml_optimizer_service import MLOptimizerService
    from core.services.neural_processing_service import NeuralProcessingService
    from core.services.real_time_analytics_service import RealTimeAnalyticsService
    from core.services.real_time_visualization_service import RealTimeVisualizationService
    from core.services.sensory_processing_service import SensoryProcessingService
    from core.services.simulation_coordinator import SimulationCoordinator
    
    from core.interfaces.adaptive_configuration import IAdaptiveConfiguration
    from core.interfaces.cloud_deployment import ICloudDeployment
    from core.interfaces.distributed_coordinator import IDistributedCoordinator
    from core.interfaces.energy_manager import IEnergyManager
    from core.interfaces.event_coordinator import IEventCoordinator
    from core.interfaces.fault_tolerance import IFaultTolerance
    from core.interfaces.gpu_accelerator import IGPUAccelerator
    from core.interfaces.graph_manager import IGraphManager
    from core.interfaces.learning_engine import ILearningEngine
    from core.interfaces.load_balancer import ILoadBalancer
    from core.interfaces.ml_optimizer import IMLOptimizer
    from core.interfaces.neural_processor import INeuralProcessor
    from core.interfaces.performance_monitor import IPerformanceMonitor
    from core.interfaces.real_time_analytics import IRealTimeAnalytics
    from core.interfaces.real_time_visualization import IRealTimeVisualization
    from core.interfaces.sensory_processor import ISensoryProcessor
    from core.interfaces.simulation_coordinator import ISimulationCoordinator

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
        print_error(f"Service resolution failed: {e}")
        return 1
    success = launcher.launch_with_profile(profile)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
