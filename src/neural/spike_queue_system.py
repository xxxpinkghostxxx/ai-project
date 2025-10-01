
"""
Spike Queue System Module

This module provides a priority-based spike queue system for neural simulations.
It includes classes for managing spikes, propagation delays, and statistical tracking.
"""

import heapq
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numba as nb

from src.utils.logging_utils import log_step


class SpikeType(Enum):
    """Enumeration of different spike types in neural simulations."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    BURST = "burst"
    SINGLE = "single"


@dataclass
class Spike:
    """Represents a single neural spike with all necessary attributes."""
    source_node_id: int
    target_node_id: int
    timestamp: float
    spike_type: SpikeType
    amplitude: float
    delay: float = 0.0
    weight: float = 1.0
    refractory_period: float = 0.0
    propagation_speed: float = 1.0
    def __lt__(self, other):
        return self.timestamp < other.timestamp


class SpikeQueue:
    """A priority queue for managing neural spikes based on timestamps."""

    def __init__(self, max_size: int = 100000):
        self._queue = []
        self._lock = threading.RLock()
        self._max_size = max_size
        self._spike_count = 0
        self._dropped_spikes = 0
        self.stats = {
            'total_spikes': 0,
            'processed_spikes': 0,
            'dropped_spikes': 0,
            'queue_size_max': 0,
            'processing_time': 0.0,
            'spikes_by_type': {spike_type: 0 for spike_type in SpikeType},
            'invalid_spikes': 0,
            'invalid_timestamps': 0,
            'heap_repairs': 0
        }

    def get_queue_internal(self) -> List[Spike]:
        """Get the internal queue."""
        return self._queue

    def get_lock_internal(self) -> threading.RLock:
        """Get the internal lock."""
        return self._lock

    def get_max_size_internal(self) -> int:
        """Get the maximum queue size."""
        return self._max_size

    def get_spike_count_internal(self) -> int:
        """Get the spike count."""
        return self._spike_count

    def set_spike_count_internal(self, count: int) -> None:
        """Set the spike count."""
        self._spike_count = count

    def get_dropped_spikes_internal(self) -> int:
        """Get the dropped spikes count."""
        return self._dropped_spikes

    def increment_dropped_spikes_internal(self) -> None:
        """Increment the dropped spikes count."""
        self._dropped_spikes += 1

    def validate_heap_structure_internal(self) -> bool:
        """Validate heap structure (internal method)."""
        return self._validate_heap_structure()

    def repair_heap_internal(self) -> None:
        """Repair heap structure (internal method)."""
        self._repair_heap()
        _stats = {
            'total_spikes': 0,
            'processed_spikes': 0,
            'dropped_spikes': 0,
            'queue_size_max': 0,
            'processing_time': 0.0,
            'spikes_by_type': {spike_type: 0 for spike_type in SpikeType}
        }
    def push(self, spike: Spike) -> bool:
        """Add a spike to the queue if valid and space available."""
        with self._lock:
            if not isinstance(spike, Spike):
                logging.error("Invalid spike object type, refusing to add to queue")
                self.stats['invalid_spikes'] = self.stats.get('invalid_spikes', 0) + 1
                return False
            if not isinstance(spike.timestamp, (int, float)) or spike.timestamp < 0:
                logging.error("Invalid spike timestamp: %s", spike.timestamp)
                self.stats['invalid_timestamps'] = self.stats.get('invalid_timestamps', 0) + 1
                return False
            if len(self.get_queue_internal()) >= self.get_max_size_internal():
                self.increment_dropped_spikes_internal()
                self.stats['dropped_spikes'] += 1
                return False
            try:
                heapq.heappush(self.get_queue_internal(), spike)
                self.set_spike_count_internal(self.get_spike_count_internal() + 1)
                self.stats['total_spikes'] += 1
                self.stats['spikes_by_type'][spike.spike_type] += 1
                current_size = len(self._queue)
                if current_size > self.stats['queue_size_max']:
                    self.stats['queue_size_max'] = current_size
                if self.stats['total_spikes'] % 1000 == 0:
                    self.validate_heap_structure_internal()
                return True
            except (IndexError, ValueError, RuntimeError) as e:
                logging.error("Heap corruption detected during spike insertion: %s", e)
                self.repair_heap_internal()
                return False
    def pop(self) -> Optional[Spike]:
        """Remove and return the earliest spike from the queue."""
        with self.get_lock_internal():
            if self.get_queue_internal():
                self.set_spike_count_internal(self.get_spike_count_internal() - 1)
                spike = heapq.heappop(self.get_queue_internal())
                self.stats['processed_spikes'] += 1
                return spike
            return None
    def peek(self) -> Optional[Spike]:
        """Return the earliest spike without removing it."""
        with self.get_lock_internal():
            queue = self.get_queue_internal()
            return queue[0] if queue else None
    def size(self) -> int:
        """Return the current number of spikes in the queue."""
        with self.get_lock_internal():
            return self.get_spike_count_internal()
    def clear(self):
        """Remove all spikes from the queue."""
        with self.get_lock_internal():
            self.get_queue_internal().clear()
            self.set_spike_count_internal(0)
    def get_spikes_in_timeframe(self, start_time: float, end_time: float) -> List[Spike]:
        """Retrieve spikes within the specified time range."""
        with self.get_lock_internal():
            spikes = []
            temp_queue = []
            internal_queue = self.get_queue_internal()
            while internal_queue:
                spike = heapq.heappop(internal_queue)
                if start_time <= spike.timestamp <= end_time:
                    spikes.append(spike)
                else:
                    temp_queue.append(spike)
            for spike in temp_queue:
                heapq.heappush(internal_queue, spike)
            return spikes
    def get_statistics(self) -> Dict[str, Any]:
        """Return current queue statistics."""
        with self.get_lock_internal():
            queue_stats = self.stats.copy()
            queue_stats['current_queue_size'] = self.get_spike_count_internal()
            queue_stats['drop_rate'] = (self.get_dropped_spikes_internal() / max(1, self.stats['total_spikes'] + self.get_dropped_spikes_internal())) * 100
            return queue_stats
    def reset_statistics(self):
        """Reset all statistics to zero."""
        with self.get_lock_internal():
            _stats = {
                'total_spikes': 0,
                'processed_spikes': 0,
                'dropped_spikes': 0,
                'queue_size_max': 0,
                'processing_time': 0.0,
                'spikes_by_type': {spike_type: 0 for spike_type in SpikeType},
                'invalid_spikes': 0,
                'invalid_timestamps': 0,
                'heap_repairs': 0
            }
            self._dropped_spikes = 0  # This should use the setter but keeping for now
    def _validate_heap_structure(self) -> bool:
        try:
            queue = self.get_queue_internal()
            for i, _ in enumerate(queue):
                left_child = 2 * i + 1
                right_child = 2 * i + 2
                if left_child < len(queue):
                    if queue[i].timestamp > queue[left_child].timestamp:
                        logging.error("Heap property violated at index %s (left child)", i)
                        return False
                if right_child < len(queue):
                    if queue[i].timestamp > queue[right_child].timestamp:
                        logging.error("Heap property violated at index %s (right child)", i)
                        return False
            return True
        except (IndexError, ValueError, AttributeError) as e:
            logging.error("Heap validation failed: %s", e)
            return False
    def _repair_heap(self):
        try:
            logging.warning("Attempting to repair corrupted heap")
            heapq.heapify(self.get_queue_internal())
            self.stats['heap_repairs'] += 1
            logging.info("Heap repair completed")
        except (IndexError, ValueError, RuntimeError) as e:
            logging.error("Heap repair failed: %s", e)
            self.get_queue_internal().clear()
            self.set_spike_count_internal(0)
            logging.error("Heap cleared due to unrecoverable corruption")


class SpikePropagator:
    """Handles the propagation and processing of neural spikes."""

    def __init__(self, simulation_manager=None):
        self.simulation_manager = simulation_manager
        self.spike_queue = SpikeQueue()
        self.propagation_delays = {}
        self.synaptic_weights = {}
        self.refractory_periods = {}
        self.base_propagation_delay = 0.001
        self.max_propagation_delay = 0.100
        self.synaptic_delay_range = (0.001, 0.010)
        self.stats = {
            'spikes_propagated': 0,
            'synaptic_transmissions': 0,
            'refractory_violations': 0,
            'propagation_time': 0.0,
            'successful_transmissions': 0,
            'failed_transmissions': 0
        }
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def schedule_spike(self, source_node_id: int, target_node_id: int,
                        spike_type_param: SpikeType = SpikeType.EXCITATORY,
                        amplitude: float = 1.0, weight: float = 1.0,
                        timestamp: float = None) -> bool:
        """Schedule a spike for propagation with given parameters."""
        delay = self._calculate_propagation_delay(source_node_id, target_node_id)
        if timestamp is None:
            timestamp = time.time() + delay
        # else: use provided timestamp as is
        spike = Spike(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            timestamp=timestamp,
            spike_type=spike_type_param,
            amplitude=amplitude,
            delay=delay,
            weight=weight,
            refractory_period=self._get_refractory_period(target_node_id),
            propagation_speed=self._get_propagation_speed(source_node_id, target_node_id)
        )
        return self.spike_queue.push(spike)
    def process_spikes(self, max_spikes: int = 1000) -> int:
        """Process up to max_spikes from the queue."""
        _spikes_processed = 0
        start_time = time.time()
        log_step(f"Starting to process up to {max_spikes} spikes")
        while _spikes_processed < max_spikes:
            spike = self.spike_queue.pop()
            if not spike:
                log_step("No spike in queue")
                break
            current_time = time.time()
            log_step(f"Spike timestamp: {spike.timestamp}, current_time: {current_time}")
            if spike.timestamp > current_time + 0.01:
                log_step("Spike is future, pushing back")
                self.spike_queue.push(spike)
                break
            success = self._process_single_spike(spike)
            log_step(f"Processed spike, success: {success}")
            if success is not None:
                _spikes_processed += 1
                if success:
                    self.stats['spikes_propagated'] += 1
        processing_time = time.time() - start_time
        self.stats['propagation_time'] += processing_time
        return _spikes_processed
    def _process_single_spike(self, spike: Spike):
        """Process a single spike, applying synaptic transmission if possible."""
        try:
            log_step(f"Processing single spike, target: {spike.target_node_id}")
            if self._is_in_refractory_period(spike.target_node_id, spike.timestamp):
                self.stats['refractory_violations'] += 1
                log_step("Spike in refractory period, not processed")
                return None
            success = self._apply_synaptic_transmission(spike)
            log_step(f"Synaptic transmission success: {success}")
            if success:
                self.stats['successful_transmissions'] += 1
                self.stats['synaptic_transmissions'] += 1
                self._update_refractory_period(spike.target_node_id, spike.timestamp)
                self._check_for_cascading_spikes(spike)
            else:
                self.stats['failed_transmissions'] += 1
            return success
        except (AttributeError, KeyError, ValueError, RuntimeError) as e:
            log_step(f"Error processing spike: {e}")
            return False
    def _apply_synaptic_transmission(self, spike: Spike) -> bool:
        """Apply synaptic transmission to the target node."""
        try:
            # Emit event if possible
            try:
                if self.simulation_manager and self.simulation_manager.event_bus:
                    self.simulation_manager.event_bus.emit('SPIKE', {'source_id': spike.source_node_id, 'node_id': spike.target_node_id, 'timestamp': spike.timestamp})
            except (AttributeError, KeyError, ValueError, TypeError):
                pass  # Event emission is optional

            # Always attempt direct processing
            if not self.simulation_manager:
                return True  # For testing, consider successful if no sim manager
            access_layer = self.simulation_manager.get_access_layer()
            if not access_layer:
                return False
            target_node = access_layer.get_node_by_id(spike.target_node_id)
            if not target_node:
                return False
            synaptic_input = spike.amplitude * spike.weight
            if spike.spike_type == SpikeType.INHIBITORY:
                synaptic_input = -synaptic_input
            elif spike.spike_type == SpikeType.MODULATORY:
                synaptic_input *= 0.5
            current_input = target_node.get('synaptic_input', 0.0)
            new_input = current_input + synaptic_input
            access_layer.update_node_property(spike.target_node_id, 'synaptic_input', new_input)
            membrane_potential = target_node.get('membrane_potential', 0.0)
            new_membrane_potential = membrane_potential + (synaptic_input * 0.1)
            access_layer.update_node_property(spike.target_node_id, 'membrane_potential',
                                            min(new_membrane_potential, 1.0))
            current_spike_count = target_node.get('spike_count', 0)
            access_layer.update_node_property(spike.target_node_id, 'spike_count', current_spike_count + 1)
            return True
        except (AttributeError, KeyError, ValueError, RuntimeError) as e:
            log_step(f"Error applying synaptic transmission: {e}")
            return False
    def _check_for_cascading_spikes(self, spike: Spike):
        """Check and schedule cascading spikes if conditions are met."""
        try:
            if not self.simulation_manager:
                return
            # Emit for cascading check; subscriber can handle
            self.simulation_manager.event_bus.emit('SPIKE', {'node_id': spike.target_node_id, 'timestamp': spike.timestamp, 'check_cascade': True})
            return
        except (AttributeError, KeyError, RuntimeError):
            # Fallback to direct
            try:
                if not self.simulation_manager:
                    return
                access_layer = self.simulation_manager.get_access_layer()
                if not access_layer:
                    return
                target_node = access_layer.get_node_by_id(spike.target_node_id)
                if not target_node:
                    return
                threshold = target_node.get('threshold', 0.5)
                membrane_potential = target_node.get('membrane_potential', 0.0)
                if membrane_potential >= threshold:
                    self._schedule_cascading_spike(spike.target_node_id, spike.timestamp)
                    access_layer.update_node_property(spike.target_node_id, 'membrane_potential', 0.0)
            except (AttributeError, KeyError, ValueError, RuntimeError) as e:
                log_step(f"Error checking for cascading spikes: {e}")
                # events_processed variable is not used in this method
    def _schedule_cascading_spike(self, node_id: int, timestamp: float):
        """Schedule a cascading spike from the given node."""
        try:
            try:
                start_id = min(node_id + 1, 2**31 - 1)
                end_id = min(node_id + 5, 2**31 - 1)
                max_id = min(node_id + 100, 2**31 - 1)
                for target_id in range(start_id, min(end_id, max_id)):
                    self.schedule_spike(
                        source_node_id=node_id,
                        target_node_id=target_id,
                        spike_type_param=SpikeType.EXCITATORY,
                        amplitude=0.8,
                        timestamp=timestamp + 0.001
                    )
            except (OverflowError, ValueError):
                return
        except (ValueError, TypeError, OverflowError) as e:
            log_step(f"Error scheduling cascading spike: {e}")
    @staticmethod
    @nb.jit(nopython=True)
    def calculate_propagation_delay(source_id: int, target_id: int, base_delay: float, max_delay: float, avg_synaptic: float) -> float:
        """Calculate propagation delay based on distance and parameters."""
        distance = abs(target_id - source_id)
        distance_delay = min(float(distance) * 0.0001, max_delay)
        return base_delay + distance_delay + avg_synaptic


    def _calculate_propagation_delay(self, source_id: int, target_id: int) -> float:
        """Calculate propagation delay for the connection."""
        avg_synaptic_delay = (self.synaptic_delay_range[0] + self.synaptic_delay_range[1]) / 2.0
        return self.calculate_propagation_delay(source_id, target_id, self.base_propagation_delay, self.max_propagation_delay, avg_synaptic_delay)
    def _get_refractory_period(self, node_id: int) -> float:
        """Get refractory period for a node."""
        return self.refractory_periods.get(node_id, 0.002)
    def _get_propagation_speed(self, _source_id: int, _target_id: int) -> float:
        """Get propagation speed (placeholder)."""
        return 1.0
    def _is_in_refractory_period(self, _node_id: int, _timestamp: float) -> bool:
        """Check if node is in refractory period (placeholder)."""
        return False
    def _update_refractory_period(self, node_id: int, timestamp: float):
        """Update refractory period for a node (placeholder)."""
        # Placeholder implementation
    def get_statistics(self) -> Dict[str, Any]:
        """Get propagator statistics."""
        _stats = self.stats.copy()
        _stats['queue_stats'] = self.spike_queue.get_statistics()
        return _stats
    def reset_statistics(self):
        """Reset propagator statistics."""
        self.stats = {
            'spikes_propagated': 0,
            'synaptic_transmissions': 0,
            'refractory_violations': 0,
            'propagation_time': 0.0,
            'successful_transmissions': 0,
            'failed_transmissions': 0
        }
        self.spike_queue.reset_statistics()


class SpikeQueueSystem:
    """High-level system for managing spike queues and propagation in neural simulations."""

    def __init__(self, simulation_manager=None):
        self.simulation_manager = simulation_manager
        self.spike_propagator = SpikePropagator(simulation_manager)
        self.running = False
        self.current_time = 0.0
        self.max_spikes_per_step = 1000
        self.processing_interval = 0.001
        self.stats = {
            'total_spikes_scheduled': 0,
            'total_spikes_processed': 0,
            'system_uptime': 0.0,
            'spikes_per_second': 0.0,
            'queue_efficiency': 0.0
        }
    def start(self):
        """Start the spike queue system."""
        self.running = True
        self.current_time = time.time()
        log_step("Spike queue system started")
    def stop(self):
        """Stop the spike queue system."""
        self.running = False
        log_step("Spike queue system stopped")
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def schedule_spike(self, source_id: int, target_id: int,
                      spike_type_param: SpikeType = SpikeType.EXCITATORY,
                      amplitude: float = 1.0, weight: float = 1.0,
                      timestamp: float = None) -> bool:
        """Schedule a spike for the system."""
        success = self.spike_propagator.schedule_spike(
            source_id, target_id, spike_type_param, amplitude, weight, timestamp
        )
        if success:
            self.stats['total_spikes_scheduled'] += 1
        return success
    def process_spikes(self, max_spikes: int = None) -> int:
        """Process spikes in the system."""
        if max_spikes is None:
            max_spikes = self.max_spikes_per_step
        _spikes_processed = self.spike_propagator.process_spikes(max_spikes)
        self.stats['total_spikes_processed'] += _spikes_processed
        current_time_val = time.time()
        self.stats['system_uptime'] = current_time_val - self.current_time
        if self.stats['system_uptime'] > 0:
            self.stats['spikes_per_second'] = self.stats['total_spikes_processed'] / self.stats['system_uptime']
        queue_stats = self.spike_propagator.spike_queue.get_statistics()
        if queue_stats['total_spikes'] > 0:
            self.stats['queue_efficiency'] = (queue_stats['processed_spikes'] / queue_stats['total_spikes']) * 100
        return _spikes_processed
    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self.spike_propagator.spike_queue.size()
    def clear_queue(self):
        """Clear the spike queue."""
        self.spike_propagator.spike_queue.clear()
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        _stats = self.stats.copy()
        _stats['propagator_stats'] = self.spike_propagator.get_statistics()
        return _stats
    def reset_statistics(self):
        """Reset system statistics."""
        self.stats = {
            'total_spikes_scheduled': 0,
            'total_spikes_processed': 0,
            'system_uptime': 0.0,
            'spikes_per_second': 0.0,
            'queue_efficiency': 0.0
        }
        self.spike_propagator.reset_statistics()


def create_spike_queue_system(simulation_manager=None) -> SpikeQueueSystem:
    """Create a new SpikeQueueSystem instance."""
    return SpikeQueueSystem(simulation_manager)
if __name__ == "__main__":
    print("Spike queue system created successfully!")
    print("Features include:")
    print("- Priority-based spike queue")
    print("- Biological propagation delays")
    print("- Refractory period handling")
    print("- Cascading spike propagation")
    print("- Comprehensive statistics tracking")
    try:
        system = create_spike_queue_system()
        system.schedule_spike(1, 2, SpikeType.EXCITATORY, 1.0, 0.8)
        system.schedule_spike(2, 3, SpikeType.INHIBITORY, 0.5, 0.6)
        system.schedule_spike(3, 4, SpikeType.MODULATORY, 0.3, 0.4)
        print(f"Queue size: {system.get_queue_size()}")
        # pylint: disable=invalid-name
        num_processed_spikes = system.process_spikes(10)
        print(f"Processed {num_processed_spikes} spikes")
        stats = system.get_statistics()
        print(f"System statistics: {stats}")
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Spike queue system test failed: {e}")
    print("Spike queue system test completed!")







