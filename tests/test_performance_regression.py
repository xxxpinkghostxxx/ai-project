"""
Performance regression testing for the neural simulation system.
Monitors performance benchmarks and detects regressions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import psutil
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock

from core.services.service_registry import ServiceRegistry
from core.services.simulation_coordinator import SimulationCoordinator
from core.interfaces.graph_manager import IGraphManager
from core.interfaces.performance_monitor import IPerformanceMonitor
from core.interfaces.event_coordinator import IEventCoordinator
from core.interfaces.neural_processor import INeuralProcessor
from core.interfaces.energy_manager import IEnergyManager
from core.interfaces.learning_engine import ILearningEngine
from core.interfaces.sensory_processor import ISensoryProcessor
from core.interfaces.configuration_service import IConfigurationService


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    test_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class RegressionResult:
    """Performance regression analysis result."""
    test_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    regression_percentage: float
    is_regression: bool
    threshold_percentage: float


class PerformanceRegressionTester:
    """Automated performance regression testing."""

    def __init__(self):
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.current_results: List[PerformanceBenchmark] = []
        self.regression_threshold = 10.0  # 10% regression threshold

        # Set performance baselines (these would be loaded from file in real implementation)
        self._set_baselines()

    def _set_baselines(self):
        """Set performance baselines for comparison."""
        # These are example baselines - in real implementation, load from historical data
        self.baselines = {
            "simulation_step": {
                "step_time_ms": 50.0,  # Baseline: 50ms per step
                "memory_mb": 100.0,    # Baseline: 100MB memory usage
                "cpu_percent": 5.0     # Baseline: 5% CPU usage
            },
            "initialization": {
                "init_time_ms": 200.0,  # Baseline: 200ms init time
                "memory_mb": 80.0       # Baseline: 80MB after init
            },
            "service_resolution": {
                "resolve_time_us": 100.0  # Baseline: 100us service resolution
            }
        }

    def run_performance_tests(self) -> List[PerformanceBenchmark]:
        """Run comprehensive performance tests."""
        results = []

        # Test service registry performance
        results.extend(self._test_service_registry_performance())

        # Test simulation coordinator performance
        results.extend(self._test_simulation_performance())

        # Test memory usage
        results.extend(self._test_memory_performance())

        self.current_results = results
        return results

    def _test_service_registry_performance(self) -> List[PerformanceBenchmark]:
        """Test service registry performance."""
        results = []

        registry = ServiceRegistry()

        # Create test types first
        test_types = []
        for i in range(100):
            test_type = type(f"TestService{i}", (), {})
            test_types.append(test_type)

        # Test service registration time
        start_time = time.time()
        for i in range(100):
            mock_service = Mock()
            # Use Mock with spec to satisfy type checking
            mock_service.__class__ = test_types[i]  # Hack to make isinstance check pass
            registry.register_instance(test_types[i], mock_service)
        reg_time = (time.time() - start_time) * 1000  # ms

        results.append(PerformanceBenchmark(
            test_name="service_registry",
            metric_name="registration_time_100_services",
            value=reg_time,
            unit="ms",
            timestamp=time.time(),
            metadata={"service_count": 100}
        ))

        # Test service resolution time
        start_time = time.time()
        for i in range(1000):
            service = registry.resolve(test_types[i % 100])
        resolve_time = (time.time() - start_time) * 1000000 / 1000  # us per resolution

        results.append(PerformanceBenchmark(
            test_name="service_registry",
            metric_name="resolution_time_avg",
            value=resolve_time,
            unit="us",
            timestamp=time.time(),
            metadata={"resolutions": 1000}
        ))

        return results

    def _test_simulation_performance(self) -> List[PerformanceBenchmark]:
        """Test simulation coordinator performance."""
        results = []

        # Setup mock services
        registry = ServiceRegistry()
        mocks = self._setup_mock_services(registry)

        coordinator = SimulationCoordinator(
            registry, mocks['neural_processor'], mocks['energy_manager'],
            mocks['learning_engine'], mocks['sensory_processor'], mocks['perf_monitor'],
            mocks['graph_manager'], mocks['event_coordinator'], mocks['configuration_service']
        )

        # Test initialization time
        start_time = time.time()
        init_success = coordinator.initialize_simulation()
        init_time = (time.time() - start_time) * 1000  # ms

        results.append(PerformanceBenchmark(
            test_name="simulation_coordinator",
            metric_name="initialization_time",
            value=init_time,
            unit="ms",
            timestamp=time.time(),
            metadata={"success": init_success}
        ))

        # Test simulation step time
        if init_success:
            coordinator.start_simulation()

            step_times = []
            for i in range(10):  # Run 10 steps for averaging
                start_time = time.time()
                coordinator.execute_simulation_step(i + 1)
                step_time = (time.time() - start_time) * 1000  # ms
                step_times.append(step_time)

            avg_step_time = statistics.mean(step_times)
            results.append(PerformanceBenchmark(
                test_name="simulation_coordinator",
                metric_name="average_step_time",
                value=avg_step_time,
                unit="ms",
                timestamp=time.time(),
                metadata={
                    "steps_tested": 10,
                    "min_time": min(step_times),
                    "max_time": max(step_times)
                }
            ))

        return results

    def _test_memory_performance(self) -> List[PerformanceBenchmark]:
        """Test memory usage performance."""
        results = []

        process = psutil.Process()

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        results.append(PerformanceBenchmark(
            test_name="memory_usage",
            metric_name="baseline_memory",
            value=baseline_memory,
            unit="MB",
            timestamp=time.time(),
            metadata={}
        ))

        # Test memory after service creation
        registry = ServiceRegistry()
        mocks = self._setup_mock_services(registry)
        coordinator = SimulationCoordinator(
            registry, mocks['neural_processor'], mocks['energy_manager'],
            mocks['learning_engine'], mocks['sensory_processor'], mocks['perf_monitor'],
            mocks['graph_manager'], mocks['event_coordinator'], mocks['configuration_service']
        )

        after_setup_memory = process.memory_info().rss / 1024 / 1024  # MB
        setup_memory_delta = after_setup_memory - baseline_memory

        results.append(PerformanceBenchmark(
            test_name="memory_usage",
            metric_name="setup_memory_delta",
            value=setup_memory_delta,
            unit="MB",
            timestamp=time.time(),
            metadata={"baseline": baseline_memory, "after_setup": after_setup_memory}
        ))

        # Test memory after initialization
        coordinator.initialize_simulation()
        after_init_memory = process.memory_info().rss / 1024 / 1024  # MB
        init_memory_delta = after_init_memory - after_setup_memory

        results.append(PerformanceBenchmark(
            test_name="memory_usage",
            metric_name="initialization_memory_delta",
            value=init_memory_delta,
            unit="MB",
            timestamp=time.time(),
            metadata={"after_setup": after_setup_memory, "after_init": after_init_memory}
        ))

        return results

    def _setup_mock_services(self, registry: ServiceRegistry):
        """Setup mock services for testing."""
        from torch_geometric.data import Data

        # Create configured mocks with spec_set to enforce interface
        graph_manager = Mock(spec_set=IGraphManager)
        graph_manager.initialize_graph = Mock(return_value=Data())
        graph_manager.update_node_lifecycle = Mock(return_value=Data())

        perf_monitor = Mock(spec=IPerformanceMonitor)
        perf_monitor.start_monitoring = Mock(return_value=True)
        perf_monitor.record_step_end = Mock()
        perf_monitor.record_step_start = Mock()
        perf_monitor.get_current_metrics = Mock(return_value=Mock(
            step_time=0.1, memory_usage=512, cpu_usage=75.0, gpu_usage=None
        ))

        event_coordinator = Mock(spec_set=IEventCoordinator)
        event_coordinator.publish = Mock()

        neural_processor = Mock(spec_set=INeuralProcessor)
        neural_processor.initialize_neural_state = Mock(return_value=True)
        neural_processor.process_neural_dynamics = Mock(return_value=(Data(), []))
        neural_processor.reset_neural_state = Mock()
        neural_processor.validate_neural_integrity = Mock(return_value={"valid": True})

        energy_manager = Mock(spec_set=IEnergyManager)
        energy_manager.initialize_energy_state = Mock(return_value=True)
        energy_manager.update_energy_flows = Mock(return_value=(Data(), []))
        energy_manager.reset_energy_state = Mock()
        energy_manager.regulate_energy_homeostasis = Mock(return_value=Data())
        energy_manager.validate_energy_conservation = Mock(return_value={"energy_conservation_rate": 0.95})

        learning_engine = Mock(spec_set=ILearningEngine)
        learning_engine.initialize_learning_state = Mock(return_value=True)
        learning_engine.apply_plasticity = Mock(return_value=(Data(), []))
        learning_engine.reset_learning_state = Mock()

        sensory_processor = Mock(spec_set=ISensoryProcessor)
        sensory_processor.initialize_sensory_pathways = Mock(return_value=True)
        sensory_processor.process_sensory_input = Mock()

        configuration_service = Mock(spec_set=IConfigurationService)
        configuration_service.load_configuration = Mock()
        configuration_service.set_parameter = Mock()
        configuration_service.get_configuration_schema = Mock(return_value={})

        registry.register_instance(IGraphManager, graph_manager)
        registry.register_instance(IPerformanceMonitor, perf_monitor)
        registry.register_instance(IEventCoordinator, event_coordinator)
        registry.register_instance(INeuralProcessor, neural_processor)
        registry.register_instance(IEnergyManager, energy_manager)
        registry.register_instance(ILearningEngine, learning_engine)
        registry.register_instance(ISensoryProcessor, sensory_processor)
        registry.register_instance(IConfigurationService, configuration_service)

        # Return the mocks for coordinator instantiation
        return {
            'graph_manager': graph_manager,
            'perf_monitor': perf_monitor,
            'event_coordinator': event_coordinator,
            'neural_processor': neural_processor,
            'energy_manager': energy_manager,
            'learning_engine': learning_engine,
            'sensory_processor': sensory_processor,
            'configuration_service': configuration_service
        }

    def analyze_regressions(self) -> List[RegressionResult]:
        """Analyze current results against baselines for regressions."""
        regressions = []

        for result in self.current_results:
            if result.test_name in self.baselines and result.metric_name in self.baselines[result.test_name]:
                baseline_value = self.baselines[result.test_name][result.metric_name]
                current_value = result.value

                # Calculate regression percentage
                if baseline_value > 0:
                    regression_pct = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    regression_pct = 0

                # Check if it's a regression (performance degradation)
                is_regression = regression_pct > self.regression_threshold

                regressions.append(RegressionResult(
                    test_name=result.test_name,
                    metric_name=result.metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    regression_percentage=regression_pct,
                    is_regression=is_regression,
                    threshold_percentage=self.regression_threshold
                ))

        return regressions

    def generate_report(self) -> str:
        """Generate performance regression report."""
        results = self.run_performance_tests()
        regressions = self.analyze_regressions()

        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE REGRESSION TEST REPORT")
        report.append("=" * 80)
        report.append("")

        # Current performance results
        report.append("CURRENT PERFORMANCE RESULTS:")
        report.append("-" * 40)
        for result in results:
            report.append(f"{result.test_name}.{result.metric_name}: {result.value:.2f} {result.unit}")
        report.append("")

        # Regression analysis
        report.append("REGRESSION ANALYSIS:")
        report.append("-" * 40)

        if not regressions:
            report.append("No baseline comparisons available.")
        else:
            regression_count = sum(1 for r in regressions if r.is_regression)
            report.append(f"Found {regression_count} performance regressions out of {len(regressions)} metrics tested.")
            report.append("")

            for regression in regressions:
                status = "⚠️  REGRESSION" if regression.is_regression else "✅ OK"
                report.append(f"{status} {regression.test_name}.{regression.metric_name}")
                report.append(f"  Current: {regression.current_value:.2f}")
                report.append(f"  Baseline: {regression.baseline_value:.2f}")
                report.append(f"  Change: {regression.regression_percentage:+.1f}%")
                report.append(f"  Threshold: ±{regression.threshold_percentage:.1f}%")
                report.append("")

        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        if regressions:
            total_regressions = sum(1 for r in regressions if r.is_regression)
            if total_regressions > 0:
                report.append(f"❌ {total_regressions} performance regressions detected!")
                report.append("Consider optimizing the affected components.")
            else:
                report.append("✅ No performance regressions detected.")
        else:
            report.append("No baseline data available for comparison.")

        return "\n".join(report)


def run_performance_regression_test() -> bool:
    """Run performance regression test and return success status."""
    tester = PerformanceRegressionTester()
    report = tester.generate_report()

    print(report)

    # Analyze results
    regressions = tester.analyze_regressions()
    has_regressions = any(r.is_regression for r in regressions)

    return not has_regressions  # Return True if no regressions


if __name__ == "__main__":
    success = run_performance_regression_test()
    exit(0 if success else 1)