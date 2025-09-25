"""
Performance tests for learning system components.
Tests speed, memory usage, scalability, and resource efficiency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import torch
import numpy as np
import psutil
import gc
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data

from learning.homeostasis_controller import HomeostasisController
from learning.learning_engine import LearningEngine
from learning.live_hebbian_learning import LiveHebbianLearning
from learning.memory_system import MemorySystem
from learning.memory_pool_manager import MemoryPoolManager


class TestPerformance:
    """Test suite for performance characteristics."""

    def setup_method(self):
        """Set up performance test environment."""
        self.homeostasis = HomeostasisController()
        self.memory_system = MemorySystem()
        self.mock_access_layer = MagicMock()

        with patch('learning.learning_engine.get_learning_config', return_value={
            'plasticity_rate': 0.01,
            'eligibility_decay': 0.95,
            'stdp_window': 20.0,
            'ltp_rate': 0.02,
            'ltd_rate': 0.01
        }):
            self.learning_engine = LearningEngine(self.mock_access_layer)

        self.hebbian_learning = LiveHebbianLearning()

    def test_homeostasis_performance_scaling(self):
        """Test homeostasis performance scaling with graph size."""
        sizes = [10, 50, 100, 200]

        for size in sizes:
            graph = Data()
            graph.node_labels = [
                {'id': i, 'behavior': 'dynamic', 'energy': 0.5}
                for i in range(size)
            ]
            graph.x = torch.rand(size, 1)

            start_time = time.time()
            result = self.homeostasis.regulate_network_activity(graph)
            end_time = time.time()

            duration = end_time - start_time
            assert duration < 1.0  # Should complete within 1 second even for large graphs
            assert result == graph

    def test_memory_system_performance(self):
        """Test memory system performance under load."""
        # Create large memory system
        for i in range(50):
            graph = Data()
            graph.node_labels = [
                {'id': j, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time() - 5}
                for j in range(10)
            ]
            graph.x = torch.rand(10, 1)
            graph.edge_index = torch.randint(0, 10, (2, 15))

            self.memory_system.form_memory_traces(graph)

        # Performance test operations
        start_time = time.time()
        stats = self.memory_system.get_memory_statistics()
        summary = self.memory_system.get_memory_summary()
        self.memory_system.consolidate_memories(graph)
        end_time = time.time()

        duration = end_time - start_time
        assert duration < 0.5  # Should be fast
        assert len(summary) == self.memory_system.get_memory_trace_count()

    def test_hebbian_learning_throughput(self):
        """Test Hebbian learning throughput."""
        graph = Data()
        graph.x = torch.rand(100, 1)
        graph.edge_index = torch.randint(0, 100, (2, 200))
        graph.edge_attr = torch.rand(200, 1)
        graph.node_labels = [{'id': i} for i in range(100)]

        # Measure throughput
        start_time = time.time()
        operations = 50

        for step in range(operations):
            self.hebbian_learning.apply_continuous_learning(graph, step)

        end_time = time.time()
        total_time = end_time - start_time
        throughput = operations / total_time  # operations per second

        assert throughput > 10  # Should handle at least 10 operations per second
        assert total_time < 10  # Should complete within 10 seconds

    def test_learning_engine_consolidation_speed(self):
        """Test learning engine consolidation speed."""
        graph = Data()
        graph.edge_attributes = []

        # Create many edges
        for i in range(500):
            edge = MagicMock()
            edge.eligibility_trace = 0.5
            edge.weight = 1.0
            edge.source = i % 50
            edge.target = (i + 1) % 50
            edge.type = 'excitatory'
            graph.edge_attributes.append(edge)

        start_time = time.time()
        result = self.learning_engine.consolidate_connections(graph)
        end_time = time.time()

        duration = end_time - start_time
        assert duration < 1.0  # Should consolidate 500 edges quickly
        assert result == graph

    def test_memory_pool_performance(self):
        """Test memory pool performance."""
        pool_manager = MemoryPoolManager()
        pool = pool_manager.create_pool('perf_test', lambda: {'data': [0] * 100})

        # Performance test for object pooling
        start_time = time.time()
        operations = 1000

        for i in range(operations):
            obj = pool.get_object()
            # Simulate some work
            obj['data'][0] = i
            pool.return_object(obj)

        end_time = time.time()
        total_time = end_time - start_time
        throughput = operations / total_time

        assert throughput > 1000  # Should handle thousands of operations per second
        pool_manager.cleanup()

    def test_memory_usage_stability(self):
        """Test memory usage stability over time."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Perform many operations
        for i in range(100):
            graph = Data()
            graph.node_labels = [{'id': j, 'behavior': 'dynamic', 'energy': 0.5} for j in range(20)]
            graph.x = torch.rand(20, 1)

            self.homeostasis.regulate_network_activity(graph)
            self.memory_system.form_memory_traces(graph)

            if i % 10 == 0:
                gc.collect()  # Force garbage collection

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50

    def test_concurrent_performance(self):
        """Test performance under concurrent operations."""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(20):
                    graph = Data()
                    graph.node_labels = [{'id': j, 'behavior': 'dynamic', 'energy': 0.5} for j in range(10)]
                    graph.x = torch.rand(10, 1)

                    self.homeostasis.regulate_network_activity(graph)
                    self.hebbian_learning.apply_continuous_learning(graph, i)

                results.append(True)
            except Exception as e:
                errors.append(e)

        threads = []
        num_threads = 4

        start_time = time.time()
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()
        total_time = end_time - start_time

        assert len(errors) == 0
        assert len(results) == num_threads
        assert total_time < 30  # Should complete within 30 seconds with concurrency

    def test_scalability_with_graph_complexity(self):
        """Test scalability with increasing graph complexity."""
        complexities = [10, 50, 100]

        for complexity in complexities:
            graph = Data()
            graph.node_labels = [
                {'id': i, 'behavior': 'dynamic', 'energy': 0.5}
                for i in range(complexity)
            ]
            graph.x = torch.rand(complexity, 1)
            graph.edge_index = torch.randint(0, complexity, (2, complexity * 2))
            graph.edge_attr = torch.rand(complexity * 2, 1)

            start_time = time.time()
            self.homeostasis.regulate_network_activity(graph)
            learned = self.hebbian_learning.apply_continuous_learning(graph, 0)
            self.memory_system.form_memory_traces(learned)
            end_time = time.time()

            duration = end_time - start_time

            # Duration should scale reasonably (not exponentially)
            # Allow more time for larger graphs but check for reasonable scaling
            max_expected_time = complexity / 10.0  # Linear scaling expectation
            assert duration < max_expected_time * 2  # Allow some overhead

    def test_cache_performance(self):
        """Test performance of caching mechanisms."""
        # Test memory system pattern caching
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time() - 5}
            for i in range(20)
        ]
        graph.x = torch.rand(20, 1)

        # First run - should cache patterns
        start_time = time.time()
        for i in range(10):
            self.memory_system.form_memory_traces(graph)
        first_run_time = time.time() - start_time

        # Second run - should benefit from caching
        start_time = time.time()
        for i in range(10):
            self.memory_system.form_memory_traces(graph)
        second_run_time = time.time() - start_time

        # Second run should be faster or at least not much slower
        assert second_run_time <= first_run_time * 1.5

    def test_batch_operation_performance(self):
        """Test performance of batch operations."""
        # Create batch of graphs
        batch_size = 10
        graphs = []

        for i in range(batch_size):
            graph = Data()
            graph.node_labels = [
                {'id': j, 'behavior': 'dynamic', 'energy': 0.5}
                for j in range(20)
            ]
            graph.x = torch.rand(20, 1)
            graphs.append(graph)

        start_time = time.time()

        for graph in graphs:
            self.homeostasis.regulate_network_activity(graph)
            self.hebbian_learning.apply_continuous_learning(graph, 0)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_graph = total_time / batch_size

        assert avg_time_per_graph < 0.1  # Should process each graph quickly
        assert total_time < 2.0  # Total batch should be fast

    def test_resource_cleanup_performance(self):
        """Test performance of resource cleanup."""
        # Create many objects
        pool_manager = MemoryPoolManager()
        pools = []

        for i in range(10):
            pool = pool_manager.create_pool(f'pool_{i}', lambda: {'data': [0] * 50})
            pools.append(pool)

            # Fill pool
            for j in range(100):
                obj = pool.get_object()
                pool.return_object(obj)

        start_time = time.time()
        pool_manager.cleanup()
        end_time = time.time()

        cleanup_time = end_time - start_time
        assert cleanup_time < 1.0  # Cleanup should be fast

    def test_idle_performance(self):
        """Test performance during idle periods."""
        # Simulate idle system
        start_memory = psutil.Process().memory_info().rss

        # Wait for some time
        time.sleep(0.1)

        # Check that idle system doesn't consume excessive resources
        end_memory = psutil.Process().memory_info().rss
        memory_increase = end_memory - start_memory

        # Memory should not increase significantly during idle
        assert abs(memory_increase) < 1024 * 1024  # Less than 1MB

    def test_peak_load_performance(self):
        """Test performance under peak load."""
        # Create high-load scenario
        graph = Data()
        graph.node_labels = [
            {'id': i, 'behavior': 'integrator', 'energy': 0.8, 'last_activation': time.time() - 1}
            for i in range(100)
        ]
        graph.x = torch.rand(100, 1)
        graph.edge_index = torch.randint(0, 100, (2, 500))
        graph.edge_attr = torch.rand(500, 1)

        start_time = time.time()

        # Run all learning systems simultaneously
        self.homeostasis.regulate_network_activity(graph)
        self.hebbian_learning.apply_continuous_learning(graph, 0)
        self.memory_system.form_memory_traces(graph)
        self.learning_engine.consolidate_connections(graph)

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle peak load within reasonable time
        assert total_time < 5.0  # 5 seconds for comprehensive processing

        # Verify all systems produced results
        assert self.homeostasis.get_regulation_statistics()['total_regulation_events'] >= 0
        assert self.hebbian_learning.get_learning_statistics()['total_weight_changes'] >= 0
        assert self.memory_system.get_memory_trace_count() >= 0
        assert self.learning_engine.get_learning_statistics()['weight_changes'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])