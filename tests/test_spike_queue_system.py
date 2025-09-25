"""
Comprehensive tests for SpikeQueueSystem.
Tests spike scheduling, processing, propagation, and queue management.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import threading
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data
import torch

from src.neural.spike_queue_system import SpikeQueueSystem, SpikeQueue, Spike, SpikeType, create_spike_queue_system


class TestSpikeQueueSystem:
    """Test suite for SpikeQueueSystem."""

    def setup_method(self):
        """Set up test environment."""
        self.system = SpikeQueueSystem()

    def teardown_method(self):
        """Clean up after tests."""
        self.system.reset_statistics()

    def test_initialization(self):
        """Test SpikeQueueSystem initialization."""
        assert self.system.spike_propagator is not None
        assert self.system.running is False
        assert self.system.max_spikes_per_step == 1000
        assert self.system.processing_interval == 0.001
        assert isinstance(self.system.stats, dict)

    def test_spike_creation(self):
        """Test Spike dataclass creation."""
        spike = Spike(
            source_node_id=1,
            target_node_id=2,
            timestamp=100.0,
            spike_type=SpikeType.EXCITATORY,
            amplitude=1.0,
            weight=0.8
        )

        assert spike.source_node_id == 1
        assert spike.target_node_id == 2
        assert spike.timestamp == 100.0
        assert spike.spike_type == SpikeType.EXCITATORY
        assert spike.amplitude == 1.0
        assert spike.weight == 0.8

    def test_spike_comparison(self):
        """Test Spike timestamp comparison."""
        spike1 = Spike(1, 2, 100.0, SpikeType.EXCITATORY, 1.0)
        spike2 = Spike(3, 4, 101.0, SpikeType.EXCITATORY, 1.0)

        assert spike1 < spike2
        assert not (spike2 < spike1)

    def test_spike_queue_operations(self):
        """Test SpikeQueue basic operations."""
        queue = SpikeQueue()

        # Test empty queue
        assert queue.size() == 0
        assert queue.pop() is None
        assert queue.peek() is None

        # Add spike
        spike = Spike(1, 2, 100.0, SpikeType.EXCITATORY, 1.0)
        result = queue.push(spike)
        assert result is True
        assert queue.size() == 1

        # Peek and pop
        peeked = queue.peek()
        assert peeked == spike
        assert queue.size() == 1  # Peek doesn't remove

        popped = queue.pop()
        assert popped == spike
        assert queue.size() == 0

    def test_spike_queue_overflow(self):
        """Test SpikeQueue overflow handling."""
        queue = SpikeQueue(max_size=5)

        # Fill queue
        for i in range(5):
            spike = Spike(i, i+1, float(i), SpikeType.EXCITATORY, 1.0)
            result = queue.push(spike)
            assert result is True

        assert queue.size() == 5

        # Try to add one more - should fail
        spike = Spike(5, 6, 5.0, SpikeType.EXCITATORY, 1.0)
        result = queue.push(spike)
        assert result is False
        assert queue.size() == 5

    def test_spike_queue_statistics(self):
        """Test SpikeQueue statistics."""
        queue = SpikeQueue()

        # Add some spikes
        for i in range(3):
            spike = Spike(i, i+1, float(i), SpikeType.EXCITATORY, 1.0)
            queue.push(spike)

        stats = queue.get_statistics()
        assert stats['total_spikes'] == 3
        assert stats['current_queue_size'] == 3
        assert stats['drop_rate'] == 0.0

        # Cause some drops
        queue._dropped_spikes = 2
        stats = queue.get_statistics()
        assert stats['drop_rate'] == (2 / 5) * 100  # 2 dropped out of 5 total attempted

    def test_spike_queue_timeframe_filtering(self):
        """Test filtering spikes by timeframe."""
        queue = SpikeQueue()

        # Add spikes at different times
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, t in enumerate(times):
            spike = Spike(i, i+1, t, SpikeType.EXCITATORY, 1.0)
            queue.push(spike)

        # Filter 2.0 to 4.0
        filtered = queue.get_spikes_in_timeframe(2.0, 4.0)
        assert len(filtered) == 3  # Times 2.0, 3.0, 4.0

        # Verify original queue still has all spikes
        assert queue.size() == 5

    def test_spike_propagator_scheduling(self):
        """Test spike scheduling in SpikePropagator."""
        propagator = self.system.spike_propagator

        result = propagator.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8)
        assert result is True

        assert propagator.spike_queue.size() == 1

        # Check spike properties
        spike = propagator.spike_queue.peek()
        assert spike.source_node_id == 1
        assert spike.target_node_id == 2
        assert spike.spike_type == SpikeType.EXCITATORY
        assert spike.amplitude == 1.0
        assert spike.weight == 0.8

    def test_spike_processing(self):
        """Test spike processing."""
        propagator = self.system.spike_propagator

        # Mock simulation manager
        mock_sim = MagicMock()
        mock_access_layer = MagicMock()
        mock_target_node = {'synaptic_input': 0.0, 'membrane_potential': 0.0, 'spike_count': 0}
        mock_access_layer.get_node_by_id.return_value = mock_target_node
        mock_sim.get_access_layer.return_value = mock_access_layer
        mock_sim.event_bus = MagicMock()

        propagator.simulation_manager = mock_sim

        # Schedule and process spike
        propagator.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=time.time())
        processed = propagator.process_spikes(1)

        assert processed == 1
        mock_access_layer.update_node_property.assert_called()

    def test_spike_propagator_statistics(self):
        """Test SpikePropagator statistics."""
        propagator = self.system.spike_propagator

        # Schedule some spikes
        for i in range(3):
            propagator.schedule_spike(i, i+1, SpikeType.EXCITATORY, 1.0, 0.8)

        # Process them
        processed = propagator.process_spikes(10)

        stats = propagator.get_statistics()
        assert stats['spikes_propagated'] == processed
        assert 'queue_stats' in stats

    def test_system_spike_scheduling(self):
        """Test spike scheduling through SpikeQueueSystem."""
        result = self.system.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8)
        assert result is True

        assert self.system.get_queue_size() == 1
        assert self.system.stats['total_spikes_scheduled'] == 1

    def test_system_spike_processing(self):
        """Test spike processing through SpikeQueueSystem."""
        # Schedule spike
        self.system.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=time.time())

        # Process
        processed = self.system.process_spikes(1)
        assert processed == 1
        assert self.system.stats['total_spikes_processed'] == 1

    def test_system_start_stop(self):
        """Test system start/stop functionality."""
        assert not self.system.running

        self.system.start()
        assert self.system.running

        self.system.stop()
        assert not self.system.running

    def test_system_statistics(self):
        """Test system-level statistics."""
        # Schedule and process some spikes
        for i in range(5):
            self.system.schedule_spike(i, i+1, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=time.time() + i*0.001)

        processed = self.system.process_spikes(10)

        stats = self.system.get_statistics()
        assert stats['total_spikes_scheduled'] == 5
        assert stats['total_spikes_processed'] == processed
        assert 'propagator_stats' in stats

    def test_different_spike_types(self):
        """Test different spike types."""
        spike_types = [SpikeType.EXCITATORY, SpikeType.INHIBITORY, SpikeType.MODULATORY, SpikeType.BURST]

        for spike_type in spike_types:
            result = self.system.schedule_spike(1, 2, spike_type, 1.0, 0.8)
            assert result is True

        assert self.system.get_queue_size() == len(spike_types)

    def test_refractory_period_handling(self):
        """Test refractory period handling in spike processing."""
        propagator = self.system.spike_propagator

        # Mock refractory period check
        propagator._is_in_refractory_period = MagicMock(return_value=True)

        # Schedule spike
        propagator.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=time.time())

        # Process - should fail due to refractory
        processed = propagator.process_spikes(1)
        assert processed == 0

        stats = propagator.get_statistics()
        assert stats['refractory_violations'] == 1

    def test_synaptic_transmission_failure(self):
        """Test handling of synaptic transmission failures."""
        propagator = self.system.spike_propagator

        # Mock transmission failure
        propagator._apply_synaptic_transmission = MagicMock(return_value=False)

        # Schedule and process
        propagator.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=time.time())
        processed = propagator.process_spikes(1)

        assert processed == 1  # Spike was processed but transmission failed

        stats = propagator.get_statistics()
        assert stats['failed_transmissions'] == 1

    def test_cascading_spike_generation(self):
        """Test cascading spike generation."""
        propagator = self.system.spike_propagator

        # Mock conditions for cascading
        mock_sim = MagicMock()
        mock_access_layer = MagicMock()
        mock_target_node = {
            'synaptic_input': 0.0,
            'membrane_potential': 0.9,  # Above threshold
            'spike_count': 0,
            'threshold': 0.5
        }
        mock_access_layer.get_node_by_id.return_value = mock_target_node
        mock_sim.get_access_layer.return_value = mock_access_layer
        mock_sim.event_bus = MagicMock()

        propagator.simulation_manager = mock_sim

        # Schedule spike
        propagator.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=time.time())

        # Process
        processed = propagator.process_spikes(1)
        assert processed == 1

        # Should have attempted cascading
        # (Exact behavior depends on implementation)

    def test_propagation_delay_calculation(self):
        """Test propagation delay calculations."""
        propagator = self.system.spike_propagator

        # Test with numba jit
        delay = propagator.calculate_propagation_delay(1, 100, 0.001, 0.1, 0.005)
        assert isinstance(delay, float)
        assert delay >= 0.001

        # Test instance method
        delay = propagator._calculate_propagation_delay(1, 100)
        assert isinstance(delay, float)

    def test_thread_safety(self):
        """Test thread safety of spike operations."""
        errors = []

        def schedule_spikes():
            try:
                for i in range(10):
                    self.system.schedule_spike(i, i+1, SpikeType.EXCITATORY, 1.0, 0.8)
            except Exception as e:
                errors.append(e)

        def process_spikes():
            try:
                self.system.process_spikes(50)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=schedule_spikes)
            t2 = threading.Thread(target=process_spikes)
            threads.extend([t1, t2])

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0

    def test_queue_clearing(self):
        """Test queue clearing functionality."""
        # Fill queue
        for i in range(10):
            self.system.schedule_spike(i, i+1, SpikeType.EXCITATORY, 1.0, 0.8)

        assert self.system.get_queue_size() == 10

        # Clear
        self.system.clear_queue()
        assert self.system.get_queue_size() == 0

    def test_statistics_reset(self):
        """Test statistics reset."""
        # Generate some stats
        self.system.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8)
        self.system.process_spikes(1)

        stats_before = self.system.get_statistics()
        assert stats_before['total_spikes_scheduled'] > 0

        # Reset
        self.system.reset_statistics()
        stats_after = self.system.get_statistics()
        assert stats_after['total_spikes_scheduled'] == 0

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty processing
        processed = self.system.process_spikes(0)
        assert processed == 0

        # Invalid spike scheduling
        with patch('neural.spike_queue_system.time.time', return_value=float('nan')):
            result = self.system.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8)
            # Should handle NaN gracefully
            assert isinstance(result, bool)

        # Very large timestamps
        result = self.system.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8, timestamp=1e20)
        assert result is True

    def test_memory_efficiency(self):
        """Test memory efficiency with large spike volumes."""
        # Schedule many spikes
        for i in range(1000):
            self.system.schedule_spike(i, (i+1) % 100, SpikeType.EXCITATORY, 1.0, 0.8)

        assert self.system.get_queue_size() == 1000

        # Process in batches
        total_processed = 0
        while self.system.get_queue_size() > 0:
            processed = self.system.process_spikes(100)
            total_processed += processed
            if processed == 0:
                break

        assert total_processed > 0

    def test_spike_type_statistics(self):
        """Test spike type statistics tracking."""
        # Schedule different types
        types_to_test = [
            SpikeType.EXCITATORY,
            SpikeType.INHIBITORY,
            SpikeType.MODULATORY,
            SpikeType.EXCITATORY
        ]

        for spike_type in types_to_test:
            self.system.schedule_spike(1, 2, spike_type, 1.0, 0.8)

        stats = self.system.get_statistics()
        propagator_stats = stats['propagator_stats']
        queue_stats = propagator_stats['queue_stats']

        assert queue_stats['spikes_by_type'][SpikeType.EXCITATORY] == 2
        assert queue_stats['spikes_by_type'][SpikeType.INHIBITORY] == 1
        assert queue_stats['spikes_by_type'][SpikeType.MODULATORY] == 1

    def test_create_spike_queue_system(self):
        """Test factory function."""
        system = create_spike_queue_system()
        assert isinstance(system, SpikeQueueSystem)

    def test_system_with_simulation_manager(self):
        """Test system with simulation manager."""
        mock_sim = MagicMock()
        system = SpikeQueueSystem(mock_sim)

        assert system.simulation_manager == mock_sim
        assert system.spike_propagator.simulation_manager == mock_sim


if __name__ == "__main__":
    pytest.main([__file__])






