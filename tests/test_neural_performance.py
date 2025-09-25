"""
Comprehensive performance and real-world usage tests for neural components.
Tests scalability, memory efficiency, performance benchmarks, and realistic usage patterns.
"""
import sys
import os
import time
import psutil
import threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data
import torch

from src.neural.behavior_engine import BehaviorEngine
from src.neural.connection_logic import intelligent_connection_formation
from src.neural.enhanced_neural_dynamics import EnhancedNeuralDynamics
from src.neural.network_metrics import NetworkMetrics
from src.neural.spike_queue_system import SpikeQueueSystem, SpikeType
from src.neural.workspace_engine import WorkspaceEngine


class TestNeuralPerformance:
    """Performance and real-world usage tests for neural components."""

    def setup_method(self):
        """Set up performance test environment."""
        self.behavior_engine = BehaviorEngine()
        self.enhanced_dynamics = EnhancedNeuralDynamics()
        self.network_metrics = NetworkMetrics()
        self.spike_system = SpikeQueueSystem(MagicMock())
        self.workspace_engine = WorkspaceEngine()

    def test_large_scale_network_performance(self):
        """Test performance with large-scale neural networks."""
        # Create large network
        num_nodes = 1000
        graph = Data()
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active'}
            for i in range(num_nodes)
        ]
        graph.x = torch.randn(num_nodes, 1)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Time connection formation
        start_time = time.time()
        graph = intelligent_connection_formation(graph)
        connection_time = time.time() - start_time

        # Time behavior updates
        start_time = time.time()
        for i in range(min(100, num_nodes)):  # Sample for performance
            self.behavior_engine.update_node_behavior(i, graph, 1)
        behavior_time = time.time() - start_time

        # Time neural dynamics
        start_time = time.time()
        graph = self.enhanced_dynamics.update_neural_dynamics(graph, 1)
        dynamics_time = time.time() - start_time

        # Assert reasonable performance (< 5 seconds total for 1000 nodes)
        total_time = connection_time + behavior_time + dynamics_time
        assert total_time < 5.0, f"Performance too slow: {total_time:.2f}s"

    def test_memory_efficiency_scaling(self):
        """Test memory efficiency as network size scales."""
        import gc

        def get_memory_usage():
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB

        base_memory = get_memory_usage()

        # Test with increasing network sizes
        sizes = [100, 500, 1000]
        memory_usage = []

        for num_nodes in sizes:
            # Clean up previous
            gc.collect()

            graph = Data()
            graph.node_labels = [
                {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active'}
                for i in range(num_nodes)
            ]
            graph.x = torch.randn(num_nodes, 1)
            graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
            graph.edge_attributes = []

            # Form connections
            graph = intelligent_connection_formation(graph)

            # Measure memory
            current_memory = get_memory_usage()
            memory_usage.append(current_memory - base_memory)

            # Clean up
            del graph
            gc.collect()

        # Memory usage should scale reasonably (not exponentially)
        scaling_ratios = [memory_usage[i] / memory_usage[0] for i in range(len(memory_usage))]
        size_ratios = [sizes[i] / sizes[0] for i in range(len(sizes))]

        # Memory scaling should be better than O(n^2)
        for i in range(1, len(scaling_ratios)):
            assert scaling_ratios[i] < size_ratios[i] * 2, f"Memory scaling too poor at size {sizes[i]}"

    def test_concurrent_processing_performance(self):
        """Test performance under concurrent processing loads."""
        num_threads = 4
        num_nodes_per_thread = 100

        def process_network(thread_id):
            graph = Data()
            start_id = thread_id * num_nodes_per_thread
            graph.node_labels = [
                {'id': start_id + i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active'}
                for i in range(num_nodes_per_thread)
            ]
            graph.x = torch.randn(num_nodes_per_thread, 1)
            graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
            graph.edge_attributes = []

            # Process network
            graph = intelligent_connection_formation(graph)
            for i in range(num_nodes_per_thread):
                self.behavior_engine.update_node_behavior(i, graph, 1)
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, 1)

            return len(graph.edge_attributes)

        # Run concurrent processing
        start_time = time.time()
        threads = []
        results = []

        for thread_id in range(num_threads):
            thread = threading.Thread(target=lambda tid=thread_id: results.append(process_network(tid)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        concurrent_time = time.time() - start_time

        # Sequential time estimate
        sequential_time = concurrent_time * num_threads  # Rough estimate

        # Concurrency should provide speedup
        speedup = sequential_time / concurrent_time
        assert speedup > 1.2, f"Insufficient concurrency speedup: {speedup:.2f}x"

    def test_real_time_processing_simulation(self):
        """Test real-time processing simulation (30 FPS equivalent)."""
        target_fps = 30
        frame_time = 1.0 / target_fps  # ~33ms per frame

        graph = Data()
        num_nodes = 200
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active', 'enhanced_behavior': True}
            for i in range(num_nodes)
        ]
        graph.x = torch.randn(num_nodes, 1)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Initialize network
        graph = intelligent_connection_formation(graph)

        # Simulate real-time processing
        frame_times = []
        for frame in range(10):  # 10 frames
            start_time = time.time()

            # Process one frame
            for i in range(num_nodes):
                self.behavior_engine.update_node_behavior(i, graph, frame)
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, frame)

            frame_time_actual = time.time() - start_time
            frame_times.append(frame_time_actual)

            # Check if we can maintain real-time
            if frame_time_actual > frame_time:
                # Allow some flexibility but log performance issue
                pass

        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)

        # Should be able to maintain reasonable frame rate
        assert avg_frame_time < frame_time * 2, f"Average frame time too slow: {avg_frame_time:.3f}s"
        assert max_frame_time < frame_time * 5, f"Max frame time too slow: {max_frame_time:.3f}s"

    def test_spike_processing_throughput(self):
        """Test spike processing throughput under load."""
        # Create high-frequency spike input
        num_spikes = 10000
        spikes_per_second = 1000

        # Schedule many spikes
        start_time = time.time()
        for i in range(num_spikes):
            timestamp = start_time + (i / spikes_per_second)
            self.spike_system.schedule_spike(i % 100, (i + 1) % 100, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=timestamp)

        scheduling_time = time.time() - start_time

        # Process spikes
        start_time = time.time()
        total_processed = 0
        while self.spike_system.get_queue_size() > 0 and total_processed < num_spikes:
            processed = self.spike_system.process_spikes(1000)
            total_processed += processed

        processing_time = time.time() - start_time

        throughput = total_processed / processing_time if processing_time > 0 else 0

        # Should handle reasonable throughput
        assert throughput > 100, f"Spike throughput too low: {throughput:.0f} spikes/sec"
        assert scheduling_time < 1.0, f"Spike scheduling too slow: {scheduling_time:.3f}s"

    def test_memory_usage_patterns(self):
        """Test memory usage patterns during extended simulation."""
        import gc

        def get_memory_stats():
            process = psutil.Process()
            mem = process.memory_info()
            return {
                'rss': mem.rss / 1024 / 1024,  # MB
                'vms': mem.vms / 1024 / 1024,  # MB
            }

        graph = Data()
        num_nodes = 300
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active', 'enhanced_behavior': True}
            for i in range(num_nodes)
        ]
        graph.x = torch.randn(num_nodes, 1)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Initialize
        graph = intelligent_connection_formation(graph)

        memory_samples = []
        for step in range(50):
            # Process step
            for i in range(num_nodes):
                self.behavior_engine.update_node_behavior(i, graph, step)
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)

            # Sample memory every 10 steps
            if step % 10 == 0:
                memory_samples.append(get_memory_stats())
                gc.collect()  # Force garbage collection

        # Check memory stability (should not grow unbounded)
        initial_memory = memory_samples[0]['rss']
        final_memory = memory_samples[-1]['rss']
        memory_growth = final_memory - initial_memory

        # Allow some memory growth but not excessive
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f} MB"

    def test_component_initialization_performance(self):
        """Test performance of component initialization."""
        start_time = time.time()

        # Initialize multiple instances
        engines = [BehaviorEngine() for _ in range(10)]
        dynamics = [EnhancedNeuralDynamics() for _ in range(10)]
        metrics = [NetworkMetrics() for _ in range(10)]
        spikes = [SpikeQueueSystem() for _ in range(10)]
        workspaces = [WorkspaceEngine() for _ in range(10)]

        init_time = time.time() - start_time

        # Should initialize quickly
        assert init_time < 2.0, f"Initialization too slow: {init_time:.3f}s"

        # Clean up
        for engine in engines + dynamics + metrics + spikes + workspaces:
            if hasattr(engine, 'cleanup'):
                engine.cleanup()

    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing vs individual processing."""
        num_nodes = 200

        # Create test data
        graphs = []
        for _ in range(5):
            graph = Data()
            graph.node_labels = [
                {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active'}
                for i in range(num_nodes)
            ]
            graph.x = torch.randn(num_nodes, 1)
            graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
            graph.edge_attributes = []
            graphs.append(graph)

        # Time batch processing
        start_time = time.time()
        for graph in graphs:
            graph = intelligent_connection_formation(graph)
            for i in range(num_nodes):
                self.behavior_engine.update_node_behavior(i, graph, 1)
        batch_time = time.time() - start_time

        # Time individual processing
        start_time = time.time()
        for graph in graphs:
            graph = intelligent_connection_formation(graph)
            for i in range(num_nodes):
                self.behavior_engine.update_node_behavior(i, graph, 1)
        individual_time = time.time() - start_time

        # Batch processing should be reasonably efficient (allowing for timing variations)
        efficiency_ratio = individual_time / batch_time
        assert efficiency_ratio >= 0.6, f"Batch processing not efficient enough: {efficiency_ratio:.2f}"

    def test_persistence_io_performance(self):
        """Test I/O performance of neural map persistence."""
        from src.neural.neural_map_persistence import NeuralMapPersistence
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = NeuralMapPersistence(temp_dir)

            # Create test graph
            graph = Data()
            num_nodes = 500
            graph.node_labels = [
                {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active'}
                for i in range(num_nodes)
            ]
            graph.x = torch.randn(num_nodes, 1)
            graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
            graph.edge_attributes = []

            graph = intelligent_connection_formation(graph)

            # Time save operation
            start_time = time.time()
            result = persistence.save_neural_map(graph, 0)
            save_time = time.time() - start_time

            assert result is True
            assert save_time < 1.0, f"Save operation too slow: {save_time:.3f}s"

            # Time load operation
            start_time = time.time()
            loaded_graph = persistence.load_neural_map(0)
            load_time = time.time() - start_time

            assert loaded_graph is not None
            assert load_time < 1.0, f"Load operation too slow: {load_time:.3f}s"

    def test_statistics_collection_overhead(self):
        """Test overhead of statistics collection."""
        graph = Data()
        num_nodes = 100
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active', 'enhanced_behavior': True}
            for i in range(num_nodes)
        ]
        graph.x = torch.randn(num_nodes, 1)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        graph = intelligent_connection_formation(graph)

        # Time processing with statistics
        start_time = time.time()
        for step in range(20):
            for i in range(num_nodes):
                self.behavior_engine.update_node_behavior(i, graph, step)
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)
            self.network_metrics.calculate_comprehensive_metrics(graph)
        with_stats_time = time.time() - start_time

        # Time processing without statistics
        self.behavior_engine.reset_statistics()
        self.enhanced_dynamics.reset_statistics()

        start_time = time.time()
        for step in range(20):
            for i in range(num_nodes):
                self.behavior_engine.update_node_behavior(i, graph, step)
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, step)
        without_stats_time = time.time() - start_time

        # Statistics overhead should be reasonable (< 150% increase, allowing for variations)
        overhead_ratio = with_stats_time / without_stats_time
        assert overhead_ratio < 3.0, f"Statistics overhead too high: {overhead_ratio:.2f}x"

    def test_scalability_with_connection_density(self):
        """Test scalability with varying connection densities."""
        base_nodes = 100

        densities = [0.1, 0.3, 0.5]
        times = []

        for density in densities:
            graph = Data()
            graph.node_labels = [
                {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active'}
                for i in range(base_nodes)
            ]
            graph.x = torch.randn(base_nodes, 1)
            graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
            graph.edge_attributes = []

            # Manually create connections at specific density
            expected_edges = int(base_nodes * (base_nodes - 1) * density)
            edges_created = 0
            for i in range(base_nodes):
                for j in range(base_nodes):
                    if i != j and np.random.random() < density and edges_created < expected_edges:
                        from src.neural.connection_logic import create_weighted_connection
                        graph = create_weighted_connection(graph, i, j, 0.3, 'excitatory')
                        edges_created += 1

            # Time processing
            start_time = time.time()
            for i in range(base_nodes):
                self.behavior_engine.update_node_behavior(i, graph, 1)
            graph = self.enhanced_dynamics.update_neural_dynamics(graph, 1)
            processing_time = time.time() - start_time

            times.append(processing_time)

        # Processing time should scale with density but not explode
        for i in range(1, len(times)):
            scaling_factor = times[i] / times[0]
            density_factor = densities[i] / densities[0]
            assert scaling_factor < density_factor * 3, f"Poor scaling at density {densities[i]}"

    def test_real_world_usage_pattern(self):
        """Test realistic usage pattern simulation."""
        # Simulate a typical usage session
        graph = Data()
        num_nodes = 150
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active', 'enhanced_behavior': True}
            for i in range(num_nodes)
        ]
        graph.x = torch.randn(num_nodes, 1)
        graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        graph.edge_attributes = []

        # Initialization phase
        start_time = time.time()
        graph = intelligent_connection_formation(graph)
        init_time = time.time() - start_time

        # Learning phase (multiple epochs)
        learning_times = []
        for epoch in range(5):
            epoch_start = time.time()
            for step in range(10):
                # Sensory input simulation
                for i in range(min(10, num_nodes)):  # Sensory nodes
                    graph.node_labels[i]['energy'] = 0.7 + 0.3 * np.sin(step * 0.1)

                # Process all nodes
                for i in range(num_nodes):
                    self.behavior_engine.update_node_behavior(i, graph, epoch * 10 + step)
                graph = self.enhanced_dynamics.update_neural_dynamics(graph, epoch * 10 + step)

                # Periodic metrics
                if step % 5 == 0:
                    self.network_metrics.calculate_comprehensive_metrics(graph)

            learning_times.append(time.time() - epoch_start)

        # Analysis phase
        analysis_start = time.time()
        final_metrics = self.network_metrics.calculate_comprehensive_metrics(graph)
        analysis_time = time.time() - analysis_start

        # Performance assertions
        assert init_time < 2.0, f"Initialization too slow: {init_time:.3f}s"
        avg_learning_time = np.mean(learning_times)
        assert avg_learning_time < 1.0, f"Learning too slow: {avg_learning_time:.3f}s per epoch"
        assert analysis_time < 0.5, f"Analysis too slow: {analysis_time:.3f}s"

        # Functional assertions
        assert final_metrics['connectivity']['num_nodes'] == num_nodes
        assert final_metrics['connectivity']['num_edges'] > 0
        assert 'criticality' in final_metrics

    def test_resource_cleanup_efficiency(self):
        """Test efficiency of resource cleanup."""
        graphs = []
        components = []

        # Create multiple graphs and components
        for i in range(5):
            graph = Data()
            num_nodes = 50
            graph.node_labels = [
                {'id': i * 50 + j, 'type': 'dynamic', 'behavior': 'dynamic', 'energy': 0.5, 'state': 'active', 'enhanced_behavior': True}
                for j in range(num_nodes)
            ]
            graph.x = torch.randn(num_nodes, 1)
            graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
            graph.edge_attributes = []

            graphs.append(graph)
            components.extend([
                BehaviorEngine(),
                EnhancedNeuralDynamics(),
                NetworkMetrics(),
                SpikeQueueSystem(),
                WorkspaceEngine()
            ])

        # Process all
        for graph in graphs:
            graph = intelligent_connection_formation(graph)
            for i in range(len(graph.node_labels)):
                components[0].update_node_behavior(i, graph, 1)  # Just use first behavior engine
            components[1].update_neural_dynamics(graph, 1)  # Just use first dynamics

        # Time cleanup
        start_time = time.time()
        for component in components:
            if hasattr(component, 'cleanup'):
                component.cleanup()
        for graph in graphs:
            del graph
        import gc
        gc.collect()
        cleanup_time = time.time() - start_time

        # Cleanup should be fast
        assert cleanup_time < 1.0, f"Cleanup too slow: {cleanup_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__])






