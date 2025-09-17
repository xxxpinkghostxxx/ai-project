

import time
import traceback

import gc
import psutil




import logging
from typing import List, Dict, Any, Tuple, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestResult:

    def __init__(self, name: str, success: bool, duration: float = 0.0,
                 error: str = None, warning: str = None, details: Dict[str, Any] = None):
        self.name = name
        self.success = success
        self.duration = duration
        self.error = error
        self.warning = warning
        self.details = details or {}


class UnifiedTestSuite:

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    def add_result(self, result: TestResult):

        self.results.append(result)
    def run_test(self, test_name: str, test_func: callable, *args, **kwargs) -> TestResult:

        start_time = time.time()
        success = False
        error = None
        warning = None
        details = {}
        try:
            result = test_func(*args, **kwargs)
            if isinstance(result, tuple):
                success, details = result
            elif isinstance(result, bool):
                success = result
            else:
                success = bool(result)
            duration = time.time() - start_time
        except Exception as e:
            success = False
            duration = time.time() - start_time
            error = str(e)
            logger.error(f"Test {test_name} failed: {e}")
            traceback.print_exc()
        result = TestResult(test_name, success, duration, error, warning, details)
        self.add_result(result)
        return result
    def get_summary(self) -> Dict[str, Any]:

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        total_duration = time.time() - self.start_time
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_duration': total_duration,
            'average_duration': sum(r.duration for r in self.results) / total_tests if total_tests > 0 else 0
        }


def test_critical_imports() -> Tuple[bool, Dict[str, Any]]:

    critical_modules = [
        "numpy", "torch", "torch_geometric.data",
        "simulation_manager", "ui.ui_engine", "neural.behavior_engine",
        "energy.node_access_layer", "energy.energy_behavior", "neural.connection_logic",
        "energy.node_id_manager", "config.config_manager", "utils.performance_monitor",
        "neural.spike_queue_system", "neural.event_driven_system", "learning.live_hebbian_learning",
        "neural.neural_map_persistence", "sensory.sensory_workspace_mapper", "sensory.visual_energy_bridge",
        "sensory.audio_to_neural_bridge", "neural.enhanced_neural_integration", "neural.workspace_engine",
        "learning.learning_engine", "learning.homeostasis_controller", "neural.network_metrics", "learning.memory_system"
    ]
    failed_imports = []
    successful_imports = []
    for module_name in critical_modules:
        try:
            if "." in module_name:
                base_module, sub_module = module_name.split(".", 1)
                module = __import__(base_module)
                for part in sub_module.split("."):
                    module = getattr(module, part)
            else:
                module = __import__(module_name)
            successful_imports.append(module_name)
        except ImportError as e:
            failed_imports.append((module_name, str(e)))
        except Exception as e:
            failed_imports.append((module_name, f"Unexpected error: {e}"))
    success = len(failed_imports) == 0
    details = {
        'successful_imports': successful_imports,
        'failed_imports': failed_imports,
        'total_modules': len(critical_modules)
    }
    return success, details


def test_memory_usage() -> Tuple[bool, Dict[str, Any]]:

    try:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        test_data = []
        for i in range(1000):
            test_data.append([i] * 1000)
        peak_memory = process.memory_info().rss / 1024 / 1024
        del test_data
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_leak = final_memory - initial_memory
        success = memory_leak <= 10
        details = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_leak_mb': memory_leak,
            'memory_increase_mb': peak_memory - initial_memory
        }
        return success, details
    except Exception as e:
        return False, {'error': str(e)}


def test_simulation_manager_creation() -> Tuple[bool, Dict[str, Any]]:

    try:
        # Force reset ID manager before test
        from energy.node_id_manager import force_reset_id_manager
        force_reset_id_manager()
        
        from simulation_manager import SimulationManager
        sim_manager = SimulationManager()
        details = {'created': True}
        sim_manager.initialize_graph()
        details['graph_initialized'] = True
        components = {
            'behavior_engine': hasattr(sim_manager, 'behavior_engine') and sim_manager.behavior_engine is not None,
            'learning_engine': hasattr(sim_manager, 'learning_engine') and sim_manager.learning_engine is not None,
            'memory_system': hasattr(sim_manager, 'memory_system') and sim_manager.memory_system is not None,
            'homeostasis_controller': hasattr(sim_manager, 'homeostasis_controller') and sim_manager.homeostasis_controller is not None,
            'network_metrics': hasattr(sim_manager, 'network_metrics') and sim_manager.network_metrics is not None,
            'workspace_engine': hasattr(sim_manager, 'workspace_engine') and sim_manager.workspace_engine is not None
        }
        details['components'] = components
        success = sim_manager.run_single_step()
        details['single_step_success'] = success
        sim_manager.cleanup()
        details['cleanup_completed'] = True
        overall_success = all(components.values()) and success
        return overall_success, details
    except Exception as e:
        return False, {'error': str(e)}


def test_single_simulation_step() -> Tuple[bool, Dict[str, Any]]:

    try:
        # Force reset ID manager before test
        from energy.node_id_manager import force_reset_id_manager
        force_reset_id_manager()
        
        from simulation_manager import SimulationManager
        sim_manager = SimulationManager()
        sim_manager.initialize_graph()
        start_time = time.time()
        success = sim_manager.run_single_step()
        duration = time.time() - start_time
        details = {
            'step_success': success,
            'step_duration': duration,
            'performance_ok': duration < 1.0
        }
        sim_manager.cleanup()
        return success and duration < 1.0, details
    except Exception as e:
        return False, {'error': str(e)}


def test_simulation_progression() -> Tuple[bool, Dict[str, Any]]:

    try:
        # Force reset ID manager before test
        from energy.node_id_manager import force_reset_id_manager
        force_reset_id_manager()
        
        from simulation_manager import SimulationManager
        sim_manager = SimulationManager()
        sim_manager.initialize_graph()
        
        # Disable visual/audio components for faster testing
        sim_manager.visual_energy_bridge = None
        sim_manager.sensory_workspace_mapper = None
        
        print("Running simulation for 150 steps...")
        start_time = time.time()
        successful_steps = 0
        
        for i in range(150):
            success = sim_manager.run_single_step()
            if success:
                successful_steps += 1
            else:
                break
                
            if (i + 1) % 25 == 0:
                print(f"Completed step {i+1} (step_counter: {sim_manager.step_counter}, current_step: {sim_manager.current_step})")
        
        duration = time.time() - start_time
        steps_per_second = successful_steps / duration if duration > 0 else 0
        
        # Check energy behavior
        energy_working = False
        if hasattr(sim_manager.graph, 'x') and sim_manager.graph.x is not None:
            total_energy = float(sim_manager.graph.x[:, 0].sum().item())
            energy_working = total_energy > 0
        
        # Check connection logic
        connections_working = False
        if hasattr(sim_manager.graph, 'edge_index'):
            num_connections = sim_manager.graph.edge_index.shape[1]
            connections_working = num_connections > 0
        
        details = {
            'successful_steps': successful_steps,
            'total_duration': duration,
            'steps_per_second': steps_per_second,
            'final_step_counter': sim_manager.step_counter,
            'final_current_step': sim_manager.current_step,
            'energy_working': energy_working,
            'connections_working': connections_working,
            'progression_past_99': sim_manager.step_counter > 99
        }
        
        sim_manager.cleanup()
        return sim_manager.step_counter > 99 and energy_working and connections_working, details
    except Exception as e:
        return False, {'error': str(e)}


def test_energy_behavior() -> Tuple[bool, Dict[str, Any]]:

    try:
        from energy.node_id_manager import force_reset_id_manager
        force_reset_id_manager()
        
        from simulation_manager import SimulationManager
        sim_manager = SimulationManager()
        sim_manager.initialize_graph()
        
        # Get initial energies
        initial_energies = []
        for node in sim_manager.graph.node_labels[:10]:
            if 'energy' in node:
                initial_energies.append(node['energy'])
        
        # Run a few steps
        for step in range(5):
            sim_manager.run_single_step()
        
        # Get final energies
        final_energies = []
        for node in sim_manager.graph.node_labels[:10]:
            if 'energy' in node:
                final_energies.append(node['energy'])
        
        # Check if energy changed
        energy_changed = any(abs(initial_energies[i] - final_energies[i]) > 0.001
                           for i in range(min(len(initial_energies), len(final_energies))))
        
        details = {
            'initial_energies_sample': initial_energies[:5],
            'final_energies_sample': final_energies[:5],
            'energy_changed': energy_changed,
            'energy_behavior_working': energy_changed
        }
        
        sim_manager.cleanup()
        return energy_changed, details
    except Exception as e:
        return False, {'error': str(e)}


def test_connection_logic() -> Tuple[bool, Dict[str, Any]]:

    try:
        from energy.node_id_manager import force_reset_id_manager
        force_reset_id_manager()
        
        from simulation_manager import SimulationManager
        sim_manager = SimulationManager()
        sim_manager.initialize_graph()
        
        initial_edges = sim_manager.graph.edge_index.numel() if hasattr(sim_manager.graph, 'edge_index') else 0
        
        # Run a few steps
        for step in range(5):
            sim_manager.run_single_step()
        
        final_edges = sim_manager.graph.edge_index.numel() if hasattr(sim_manager.graph, 'edge_index') else 0
        
        details = {
            'initial_edge_count': initial_edges,
            'final_edge_count': final_edges,
            'connections_created': final_edges > 0,
            'connection_logic_working': final_edges > 0
        }
        
        sim_manager.cleanup()
        return final_edges > 0, details
    except Exception as e:
        return False, {'error': str(e)}


def test_ui_components() -> Tuple[bool, Dict[str, Any]]:

    try:
        # Force reset ID manager before test
        from energy.node_id_manager import force_reset_id_manager
        force_reset_id_manager()
        
        import dearpygui.dearpygui as dpg
        dpg.create_context()
        dpg.create_viewport(title="Test Viewport", width=800, height=600)
        with dpg.window(label="Test Window"):
            dpg.add_text("Test Text")
            dpg.add_button(label="Test Button")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        from simulation_manager import SimulationManager
        sim_manager = SimulationManager()
        details = {
            'dearpygui_working': True,
            'simulation_manager_created': True,
            'ui_integration_ok': True
        }
        dpg.stop_dearpygui()
        dpg.destroy_context()
        sim_manager.cleanup()
        return True, details
    except Exception as e:
        return False, {'error': str(e)}


def test_performance_monitoring() -> Tuple[bool, Dict[str, Any]]:

    try:
        from performance_monitor import (
            get_system_performance_metrics,
            record_simulation_step,
            record_simulation_error,
            record_simulation_warning
        )
        metrics = get_system_performance_metrics()
        details = {'metrics_collected': True, 'metrics': metrics}
        record_simulation_step(0.1, 100, 200)
        record_simulation_error()
        record_simulation_warning()
        details['recording_functions_ok'] = True
        return True, details
    except Exception as e:
        return False, {'error': str(e)}


def test_error_handling() -> Tuple[bool, Dict[str, Any]]:

    try:
        # Force reset ID manager before test
        from energy.node_id_manager import force_reset_id_manager
        force_reset_id_manager()
        
        from simulation_manager import SimulationManager
        sim_manager = SimulationManager()
        details = {'error_tests': []}
        test_cases = [
            ('None visual data', lambda: sim_manager._update_sensory_features(None, scale=1.0)),
            ('None audio data', lambda: sim_manager.process_audio_to_neural(None)),
            ('None graph', lambda: setattr(sim_manager, 'graph', None) or sim_manager.run_single_step())
        ]
        for test_name, test_func in test_cases:
            try:
                test_func()
                details['error_tests'].append({'test': test_name, 'handled': True})
            except Exception as e:
                details['error_tests'].append({'test': test_name, 'handled': False, 'error': str(e)})
        sim_manager.cleanup()
        handled_count = sum(1 for test in details['error_tests'] if test['handled'])
        success = handled_count >= len(test_cases) // 2
        return success, details
    except Exception as e:
        return False, {'error': str(e)}


def run_unified_tests() -> Dict[str, Any]:

    print("[SEARCH] UNIFIED TEST SUITE")
    print("=" * 60)
    suite = UnifiedTestSuite()
    tests = [
        ("Critical Imports", test_critical_imports),
        ("Memory Usage", test_memory_usage),
        ("Simulation Manager Creation", test_simulation_manager_creation),
        ("Single Simulation Step", test_single_simulation_step),
        ("Simulation Progression", test_simulation_progression),
        ("Energy Behavior", test_energy_behavior),
        ("Connection Logic", test_connection_logic),
        ("UI Components", test_ui_components),
        ("Performance Monitoring", test_performance_monitoring),
        ("Error Handling", test_error_handling)
    ]
    for test_name, test_func in tests:
        print(f"\n[TEST] Running {test_name}...")
        result = suite.run_test(test_name, test_func)
        status = "[OK] PASS" if result.success else "[ERROR] FAIL"
        print(f"{test_name}: {status} ({result.duration:.3f}s)")
        if result.error:
            print(f"  Error: {result.error}")
        if result.warning:
            print(f"  Warning: {result.warning}")
    summary = suite.get_summary()
    print(f"\n[STATS] TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Average Duration: {summary['average_duration']:.3f}s")
    if summary['success_rate'] == 1.0:
        print("\n[CELEBRATE] All tests passed! System is fully functional.")
    else:
        print(f"\n[WARNING] {summary['failed_tests']} tests failed. Investigation needed.")
    return summary
if __name__ == "__main__":
    results = run_unified_tests()
