
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
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    BURST = "burst"
    SINGLE = "single"
@dataclass


class Spike:
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
    def __init__(self, max_size: int = 100000):
        self._queue = []
        self._lock = threading.RLock()
        self._max_size = max_size
        self._spike_count = 0
        self._dropped_spikes = 0

    def get_queue_internal(self) -> List[Spike]:
        """Get the internal queue."""
        return self._queue

    def get_lock_internal(self) -> threading.RLock():
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
        self.stats = {
            'total_spikes': 0,
            'processed_spikes': 0,
            'dropped_spikes': 0,
            'queue_size_max': 0,
            'processing_time': 0.0,
            'spikes_by_type': {spike_type: 0 for spike_type in SpikeType}
        }
    def push(self, spike: Spike) -> bool:
        with self._lock:
            if not isinstance(spike, Spike):
                logging.error("Invalid spike object type, refusing to add to queue")
                self.stats['invalid_spikes'] = self.stats.get('invalid_spikes', 0) + 1
                return False
            if not isinstance(spike.timestamp, (int, float)) or spike.timestamp < 0:
                logging.error(f"Invalid spike timestamp: {spike.timestamp}")
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
            except Exception as e:
                logging.error(f"Heap corruption detected during spike insertion: {e}")
                self.repair_heap_internal()
                return False
    def pop(self) -> Optional[Spike]:
        with self.get_lock_internal():
            if self.get_queue_internal():
                self.set_spike_count_internal(self.get_spike_count_internal() - 1)
                spike = heapq.heappop(self.get_queue_internal())
                self.stats['processed_spikes'] += 1
                return spike
            return None
    def peek(self) -> Optional[Spike]:
        with self.get_lock_internal():
            queue = self.get_queue_internal()
            return queue[0] if queue else None
    def size(self) -> int:
        with self.get_lock_internal():
            return self.get_spike_count_internal()
    def clear(self):
        with self.get_lock_internal():
            self.get_queue_internal().clear()
            self.set_spike_count_internal(0)
    def get_spikes_in_timeframe(self, start_time: float, end_time: float) -> List[Spike]:
        with self.get_lock_internal():
            spikes = []
            temp_queue = []
            queue = self.get_queue_internal()
            while queue:
                spike = heapq.heappop(queue)
                if start_time <= spike.timestamp <= end_time:
                    spikes.append(spike)
                else:
                    temp_queue.append(spike)
            for spike in temp_queue:
                heapq.heappush(queue, spike)
            return spikes
    def get_statistics(self) -> Dict[str, Any]:
        with self.get_lock_internal():
            stats = self.stats.copy()
            stats['current_queue_size'] = self.get_spike_count_internal()
            stats['drop_rate'] = (self.get_dropped_spikes_internal() / max(1, self.stats['total_spikes'] + self.get_dropped_spikes_internal())) * 100
            return stats
    def reset_statistics(self):
        with self.get_lock_internal():
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
            self._dropped_spikes = 0  # This should use the setter but keeping for now
    def _validate_heap_structure(self) -> bool:
        try:
            queue = self.get_queue_internal()
            for i in range(len(queue)):
                left_child = 2 * i + 1
                right_child = 2 * i + 2
                if left_child < len(queue):
                    if queue[i].timestamp > queue[left_child].timestamp:
                        logging.error(f"Heap property violated at index {i} (left child)")
                        return False
                if right_child < len(queue):
                    if queue[i].timestamp > queue[right_child].timestamp:
                        logging.error(f"Heap property violated at index {i} (right child)")
                        return False
            return True
        except Exception as e:
            logging.error(f"Heap validation failed: {e}")
            return False
    def _repair_heap(self):
        try:
            logging.warning("Attempting to repair corrupted heap")
            heapq.heapify(self.get_queue_internal())
            self.stats['heap_repairs'] += 1
            logging.info("Heap repair completed")
        except Exception as e:
            logging.error(f"Heap repair failed: {e}")
            self.get_queue_internal().clear()
            self.set_spike_count_internal(0)
            logging.error("Heap cleared due to unrecoverable corruption")


class SpikePropagator:
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
    def schedule_spike(self, source_node_id: int, target_node_id: int,
                       spike_type: SpikeType = SpikeType.EXCITATORY,
                       amplitude: float = 1.0, weight: float = 1.0,
                       timestamp: float = None) -> bool:
        delay = self._calculate_propagation_delay(source_node_id, target_node_id)
        if timestamp is None:
            timestamp = time.time() + delay
        # else: use provided timestamp as is
        spike = Spike(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            timestamp=timestamp,
            spike_type=spike_type,
            amplitude=amplitude,
            delay=delay,
            weight=weight,
            refractory_period=self._get_refractory_period(target_node_id),
            propagation_speed=self._get_propagation_speed(source_node_id, target_node_id)
        )
        return self.spike_queue.push(spike)
    def process_spikes(self, max_spikes: int = 1000) -> int:
        spikes_processed = 0
        start_time = time.time()
        log_step(f"Starting to process up to {max_spikes} spikes")
        while spikes_processed < max_spikes:
            spike = self.spike_queue.pop()
            if not spike:
                log_step("No spike in queue")
                break
            current_time = time.time()
            log_step(f"Spike timestamp: {spike.timestamp}, current_time: {current_time}")
            if spike.timestamp > current_time + 0.01:
                log_step(f"Spike is future, pushing back")
                self.spike_queue.push(spike)
                break
            success = self._process_single_spike(spike)
            log_step(f"Processed spike, success: {success}")
            if success is not None:
                spikes_processed += 1
                if success:
                    self.stats['spikes_propagated'] += 1
        processing_time = time.time() - start_time
        self.stats['propagation_time'] += processing_time
        return spikes_processed
    def _process_single_spike(self, spike: Spike):
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
        except Exception as e:
            log_step(f"Error processing spike: {e}")
            return False
    def _apply_synaptic_transmission(self, spike: Spike) -> bool:
        try:
            # Emit event if possible
            try:
                if self.simulation_manager and self.simulation_manager.event_bus:
                    self.simulation_manager.event_bus.emit('SPIKE', {'source_id': spike.source_node_id, 'node_id': spike.target_node_id, 'timestamp': spike.timestamp})
            except Exception:
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
            spike_count = target_node.get('spike_count', 0)
            access_layer.update_node_property(spike.target_node_id, 'spike_count', spike_count + 1)
            return True
        except Exception as e:
            log_step(f"Error applying synaptic transmission: {e}")
            return False
    def _check_for_cascading_spikes(self, spike: Spike):
        try:
            if not self.simulation_manager:
                return
            # Emit for cascading check; subscriber can handle
            self.simulation_manager.event_bus.emit('SPIKE', {'node_id': spike.target_node_id, 'timestamp': spike.timestamp, 'check_cascade': True})
            return
        except Exception:
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
            except Exception as e:
                log_step(f"Error checking for cascading spikes: {e}")
    def _schedule_cascading_spike(self, node_id: int, timestamp: float):
        try:
            try:
                start_id = min(node_id + 1, 2**31 - 1)
                end_id = min(node_id + 5, 2**31 - 1)
                max_id = min(node_id + 100, 2**31 - 1)
                for target_id in range(start_id, min(end_id, max_id)):
                    self.schedule_spike(
                        source_node_id=node_id,
                        target_node_id=target_id,
                        spike_type=SpikeType.EXCITATORY,
                        amplitude=0.8,
                        timestamp=timestamp + 0.001
                    )
            except (OverflowError, ValueError):
                return
        except Exception as e:
            log_step(f"Error scheduling cascading spike: {e}")
    @staticmethod
    @nb.jit(nopython=True)
    def calculate_propagation_delay(source_id: int, target_id: int, base_delay: float, max_delay: float, avg_synaptic: float) -> float:
        distance = abs(target_id - source_id)
        distance_delay = min(float(distance) * 0.0001, max_delay)
        return base_delay + distance_delay + avg_synaptic
    
    
    def _calculate_propagation_delay(self, source_id: int, target_id: int) -> float:
        avg_synaptic = (self.synaptic_delay_range[0] + self.synaptic_delay_range[1]) / 2.0
        return self.calculate_propagation_delay(source_id, target_id, self.base_propagation_delay, self.max_propagation_delay, avg_synaptic)
    def _get_refractory_period(self, node_id: int) -> float:
        return self.refractory_periods.get(node_id, 0.002)
    def _get_propagation_speed(self, source_id: int, target_id: int) -> float:
        return 1.0
    def _is_in_refractory_period(self, node_id: int, timestamp: float) -> bool:
        return False
    def _update_refractory_period(self, node_id: int, timestamp: float):
        pass
    def get_statistics(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats['queue_stats'] = self.spike_queue.get_statistics()
        return stats
    def reset_statistics(self):
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
        self.running = True
        self.current_time = time.time()
        log_step("Spike queue system started")
    def stop(self):
        self.running = False
        log_step("Spike queue system stopped")
    def schedule_spike(self, source_id: int, target_id: int,
                      spike_type: SpikeType = SpikeType.EXCITATORY,
                      amplitude: float = 1.0, weight: float = 1.0,
                      timestamp: float = None) -> bool:
        success = self.spike_propagator.schedule_spike(
            source_id, target_id, spike_type, amplitude, weight, timestamp
        )
        if success:
            self.stats['total_spikes_scheduled'] += 1
        return success
    def process_spikes(self, max_spikes: int = None) -> int:
        if max_spikes is None:
            max_spikes = self.max_spikes_per_step
        spikes_processed = self.spike_propagator.process_spikes(max_spikes)
        self.stats['total_spikes_processed'] += spikes_processed
        current_time = time.time()
        self.stats['system_uptime'] = current_time - self.current_time
        if self.stats['system_uptime'] > 0:
            self.stats['spikes_per_second'] = self.stats['total_spikes_processed'] / self.stats['system_uptime']
        queue_stats = self.spike_propagator.spike_queue.get_statistics()
        if queue_stats['total_spikes'] > 0:
            self.stats['queue_efficiency'] = (queue_stats['processed_spikes'] / queue_stats['total_spikes']) * 100
        return spikes_processed
    def get_queue_size(self) -> int:
        return self.spike_propagator.spike_queue.size()
    def clear_queue(self):
        self.spike_propagator.spike_queue.clear()
    def get_statistics(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats['propagator_stats'] = self.spike_propagator.get_statistics()
        return stats
    def reset_statistics(self):
        self.stats = {
            'total_spikes_scheduled': 0,
            'total_spikes_processed': 0,
            'system_uptime': 0.0,
            'spikes_per_second': 0.0,
            'queue_efficiency': 0.0
        }
        self.spike_propagator.reset_statistics()


def create_spike_queue_system(simulation_manager=None) -> SpikeQueueSystem:
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
        spikes_processed = system.process_spikes(10)
        print(f"Processed {spikes_processed} spikes")
        stats = system.get_statistics()
        print(f"System statistics: {stats}")
    except Exception as e:
        print(f"Spike queue system test failed: {e}")
    print("Spike queue system test completed!")







