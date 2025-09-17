
import sys
import os
import time
import logging
import traceback
import argparse
from typing import Dict, Any, Optional
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class UnifiedLauncher:
    def __init__(self):
        self.profiles = {
            'full': {
                'description': 'Full UI with all features',
                'ui_module': 'ui_engine',
                'ui_function': 'run_ui',
                'performance_mode': False,
                'logging_level': 'INFO'
            },
            'optimized': {
                'description': 'Optimized UI with performance tuning',
                'ui_module': 'ui_engine',
                'ui_function': 'run_ui',
                'performance_mode': True,
                'logging_level': 'WARNING'
            },
            'test': {
                'description': 'Run test suite',
                'test_module': 'unified_test_suite',
                'test_function': 'run_unified_tests',
                'performance_mode': False,
                'logging_level': 'INFO'
            }
        }
    def test_basic_imports(self) -> bool:
        print("Testing critical imports...")
        critical_modules = [
            'numpy', 'torch', 'dearpygui',
            'simulation_manager', 'ui_engine'
        ]
        failed_imports = []
        for module_name in critical_modules:
            try:
                __import__(module_name)
                print(f"  ✓ {module_name}")
            except ImportError as e:
                print(f"  ✗ {module_name}: {e}")
                failed_imports.append(module_name)
            except Exception as e:
                print(f"  ⚠ {module_name}: {e}")
                failed_imports.append(module_name)
        if failed_imports:
            print(f"\nFailed to import: {', '.join(failed_imports)}")
            return False
        print("All critical imports successful!")
        return True
    def apply_performance_optimizations(self):
        print("Applying performance optimizations...")
        os.environ['PYTHONOPTIMIZE'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        logging.getLogger().setLevel(logging.WARNING)
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
        print("Performance optimizations applied!")
    def test_system_capacity(self) -> Dict[str, Any]:
        print("Testing system capacity...")
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            capacity_info = {
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'cpu_count': cpu_count,
                'sufficient_memory': memory.available > 1024**3,
                'sufficient_cpu': cpu_count >= 2
            }
            print(f"  Memory: {capacity_info['memory_available_gb']:.1f}GB available")
            print(f"  CPU: {capacity_info['cpu_count']} cores")
            return capacity_info
        except ImportError:
            print("  psutil not available, skipping capacity test")
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
            print("Import test failed, cannot launch")
            return False
        capacity = self.test_system_capacity()
        if not capacity.get('sufficient_memory', True):
            print("Insufficient memory available")
            return False
        if config.get('performance_mode', False):
            self.apply_performance_optimizations()
        logging.getLogger().setLevel(getattr(logging, config['logging_level']))
        try:
            if profile == 'test':
                return self._launch_test_suite()
            elif 'ui_module' in config:
                return self._launch_ui(config)
            else:
                print(f"Invalid configuration for profile: {profile}")
                return False
        except Exception as e:
            print(f"Launch failed: {e}")
            traceback.print_exc()
            return False
    def _launch_test_suite(self) -> bool:
        try:
            from unified_test_suite import run_unified_tests
            results = run_unified_tests()
            return results['success_rate'] > 0.8
        except Exception as e:
            print(f"Test suite launch failed: {e}")
            return False
    def _launch_ui(self, config: Dict[str, Any]) -> bool:
        try:
            ui_module = __import__(config['ui_module'])
            if 'ui_class' in config:
                ui_class = getattr(ui_module, config['ui_class'])
                ui_instance = ui_class()
                ui_instance.run()
            elif 'ui_function' in config:
                ui_function = getattr(ui_module, config['ui_function'])
                ui_function()
            else:
                print("No UI class or function specified")
                return False
            return True
        except Exception as e:
            print(f"UI launch failed: {e}")
            traceback.print_exc()
            return False
    def show_help(self):
        print("Unified Launcher - Neural Simulation System")
        print("=" * 50)
        print("\nAvailable profiles:")
        for profile, config in self.profiles.items():
            print(f"  {profile:12} - {config['description']}")
        print("\nUsage:")
        print("  python unified_launcher.py [profile]")
        print("  python unified_launcher.py --help")
        print("  python unified_launcher.py --list")
        print("\nExamples:")
        print("  python unified_launcher.py minimal    # Launch minimal UI")
        print("  python unified_launcher.py full       # Launch full UI")
        print("  python unified_launcher.py optimized  # Launch optimized UI")
        print("  python unified_launcher.py test       # Run test suite")


def main():
    parser = argparse.ArgumentParser(description='Unified Launcher for Neural Simulation System')
    parser.add_argument('profile', nargs='?', default='minimal',
                       help='Launch profile (minimal, full, optimized, test)')
    parser.add_argument('--list', action='store_true', help='List available profiles')
    parser.add_argument('--help-profiles', action='store_true', help='Show detailed profile help')
    args = parser.parse_args()
    launcher = UnifiedLauncher()
    if args.list or args.help_profiles:
        launcher.show_help()
        return 0
    success = launcher.launch_with_profile(args.profile)
    return 0 if success else 1
if __name__ == "__main__":
    sys.exit(main())
