
import time
import heapq
import threading
from typing import Dict, Any, List, Optional, Callable

from dataclasses import dataclass
from enum import Enum

from src.utils.logging_utils import log_step


class EventType(Enum):
    SPIKE = "spike"
    SYNAPTIC_TRANSMISSION = "synaptic_transmission"
    PLASTICITY_UPDATE = "plasticity_update"
    MEMORY_FORMATION = "memory_formation"
    HOMEOSTATIC_REGULATION = "homeostatic_regulation"
    NODE_BIRTH = "node_birth"
    NODE_DEATH = "node_death"
    ENERGY_TRANSFER = "energy_transfer"
    THETA_BURST = "theta_burst"
    IEG_TAGGING = "ieg_tagging"
@dataclass


class NeuralEvent:
    event_type: EventType
    timestamp: float
    source_node_id: int
    target_node_id: Optional[int] = None
    data: Dict[str, Any] = None
    priority: int = 0
    def __lt__(self, other):
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority > other.priority


class EventQueue:
    def __init__(self):
        self._queue = []
        self._lock = threading.RLock()
        self._event_count = 0
    def push(self, event: NeuralEvent):
        with self._lock:
            heapq.heappush(self._queue, event)
            self._event_count += 1
    def pop(self) -> Optional[NeuralEvent]:
        with self._lock:
            if self._queue:
                self._event_count -= 1
                return heapq.heappop(self._queue)
            return None
    def peek(self) -> Optional[NeuralEvent]:
        with self._lock:
            return self._queue[0] if self._queue else None
    def size(self) -> int:
        with self._lock:
            return self._event_count
    def clear(self):
        with self._lock:
            self._queue.clear()
            self._event_count = 0
    def get_events_in_timeframe(self, start_time: float, end_time: float) -> List[NeuralEvent]:
        with self._lock:
            events = []
            temp_queue = []
            while self._queue:
                event = heapq.heappop(self._queue)
                if start_time <= event.timestamp <= end_time:
                    events.append(event)
                else:
                    temp_queue.append(event)
            for event in temp_queue:
                heapq.heappush(self._queue, event)
            return events


class EventProcessor:
    def __init__(self, simulation_manager=None):
        self.simulation_manager = simulation_manager
        self.event_handlers: Dict[EventType, Callable] = {
            EventType.SPIKE: self._handle_spike_event,
            EventType.SYNAPTIC_TRANSMISSION: self._handle_synaptic_transmission,
            EventType.PLASTICITY_UPDATE: self._handle_plasticity_update,
            EventType.MEMORY_FORMATION: self._handle_memory_formation,
            EventType.HOMEOSTATIC_REGULATION: self._handle_homeostatic_regulation,
            EventType.NODE_BIRTH: self._handle_node_birth,
            EventType.NODE_DEATH: self._handle_node_death,
            EventType.ENERGY_TRANSFER: self._handle_energy_transfer,
            EventType.THETA_BURST: self._handle_theta_burst,
            EventType.IEG_TAGGING: self._handle_ieg_tagging
        }
        self.stats = {
            'events_processed': 0,
            'events_by_type': {event_type: 0 for event_type in EventType},
            'processing_time': 0.0,
            'queue_size_max': 0
        }
    def process_event(self, event: NeuralEvent) -> bool:
        start_time = time.time()
        try:
            handler = self.event_handlers.get(event.event_type)
            if handler:
                success = handler(event)
                self.stats['events_processed'] += 1
                self.stats['events_by_type'][event.event_type] += 1
                processing_time = time.time() - start_time
                self.stats['processing_time'] += processing_time
                return success
            else:
                log_step(f"No handler for event type: {event.event_type}")
                return False
        except Exception as e:
            log_step(f"Error processing event {event.event_type}: {e}")
            return False
    def _handle_spike_event(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager:
                return False
            access_layer = self.simulation_manager.get_access_layer()
            if not access_layer:
                return False
            source_node = access_layer.get_node_by_id(event.source_node_id)
            if not source_node:
                return False
            access_layer.update_node_property(event.source_node_id, 'last_spike', event.timestamp)
            access_layer.update_node_property(event.source_node_id, 'spike_count',
                                            source_node.get('spike_count', 0) + 1)
            self._trigger_synaptic_transmission_events(event.source_node_id, event.timestamp)
            self._trigger_plasticity_events(event.source_node_id, event.timestamp)
            return True
        except Exception as e:
            log_step(f"Error handling spike event: {e}")
            return False
    def _handle_synaptic_transmission(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager or not event.target_node_id:
                return False
            access_layer = self.simulation_manager.get_access_layer()
            if not access_layer:
                return False
            connection_strength = event.data.get('strength', 1.0)
            connection_type = event.data.get('type', 'excitatory')
            target_node = access_layer.get_node_by_id(event.target_node_id)
            if target_node:
                current_input = target_node.get('synaptic_input', 0.0)
                new_input = current_input + (connection_strength if connection_type == 'excitatory' else -connection_strength)
                access_layer.update_node_property(event.target_node_id, 'synaptic_input', new_input)
                threshold = target_node.get('threshold', 0.5)
                if new_input >= threshold:
                    spike_event = NeuralEvent(
                        event_type=EventType.SPIKE,
                        timestamp=event.timestamp + 0.001,
                        source_node_id=event.target_node_id,
                        priority=1
                    )
                    self.simulation_manager.event_queue.push(spike_event)
            return True
        except Exception as e:
            log_step(f"Error handling synaptic transmission: {e}")
            return False
    def _handle_plasticity_update(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager:
                return False
            if hasattr(self.simulation_manager, 'learning_engine'):
                self.simulation_manager.learning_engine.apply_timing_learning(
                    event.source_node_id, event.target_node_id, None, event.data.get('delta_t', 0.0)
                )
            return True
        except Exception as e:
            log_step(f"Error handling plasticity update: {e}")
            return False
    def _handle_memory_formation(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager:
                return False
            if hasattr(self.simulation_manager, 'memory_system'):
                self.simulation_manager.memory_system.form_memory_traces(self.simulation_manager.graph)
            return True
        except Exception as e:
            log_step(f"Error handling memory formation: {e}")
            return False
    def _handle_homeostatic_regulation(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager:
                return False
            if hasattr(self.simulation_manager, 'homeostasis_controller'):
                self.simulation_manager.homeostasis_controller.regulate_network_activity(
                    self.simulation_manager.graph
                )
            return True
        except Exception as e:
            log_step(f"Error handling homeostatic regulation: {e}")
            return False
    def _handle_node_birth(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager:
                return False
            birth_params = event.data or {}
            self.simulation_manager.graph = self.simulation_manager.birth_new_dynamic_nodes(
                self.simulation_manager.graph
            )
            return True
        except Exception as e:
            log_step(f"Error handling node birth: {e}")
            return False
    def _handle_node_death(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager:
                return False
            self.simulation_manager.graph = self.simulation_manager.remove_dead_dynamic_nodes(
                self.simulation_manager.graph
            )
            return True
        except Exception as e:
            log_step(f"Error handling node death: {e}")
            return False
    def _handle_energy_transfer(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager or not event.target_node_id:
                return False
            access_layer = self.simulation_manager.get_access_layer()
            if not access_layer:
                return False
            energy_amount = event.data.get('amount', 0.0)
            source_energy = access_layer.get_node_energy(event.source_node_id) or 0.0
            target_energy = access_layer.get_node_energy(event.target_node_id) or 0.0
            new_source_energy = max(0.0, source_energy - energy_amount)
            new_target_energy = min(255.0, target_energy + energy_amount)
            access_layer.set_node_energy(event.source_node_id, new_source_energy)
            access_layer.set_node_energy(event.target_node_id, new_target_energy)
            return True
        except Exception as e:
            log_step(f"Error handling energy transfer: {e}")
            return False
    def _handle_theta_burst(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager:
                return False
            if hasattr(self.simulation_manager, 'enhanced_integration'):
                pass
            return True
        except Exception as e:
            log_step(f"Error handling theta burst: {e}")
            return False
    def _handle_ieg_tagging(self, event: NeuralEvent) -> bool:
        try:
            if not self.simulation_manager:
                return False
            access_layer = self.simulation_manager.get_access_layer()
            if access_layer:
                access_layer.update_node_property(event.source_node_id, 'IEG_flag', True)
                access_layer.update_node_property(event.source_node_id, 'IEG_timestamp', event.timestamp)
            return True
        except Exception as e:
            log_step(f"Error handling IEG tagging: {e}")
            return False
    def _trigger_synaptic_transmission_events(self, source_node_id: int, timestamp: float):
        try:
            if not self.simulation_manager:
                return
            transmission_event = NeuralEvent(
                event_type=EventType.SYNAPTIC_TRANSMISSION,
                timestamp=timestamp + 0.001,
                source_node_id=source_node_id,
                target_node_id=min(source_node_id + 1, 2**31 - 1),
                data={'strength': 0.5, 'type': 'excitatory'},
                priority=2
            )
            self.simulation_manager.event_queue.push(transmission_event)
        except Exception as e:
            log_step(f"Error triggering synaptic transmission events: {e}")
    def _trigger_plasticity_events(self, source_node_id: int, timestamp: float):
        try:
            if not self.simulation_manager:
                return
            plasticity_event = NeuralEvent(
                event_type=EventType.PLASTICITY_UPDATE,
                timestamp=timestamp + 0.01,
                source_node_id=source_node_id,
                data={'delta_t': 0.01},
                priority=3
            )
            self.simulation_manager.event_queue.push(plasticity_event)
        except Exception as e:
            log_step(f"Error triggering plasticity events: {e}")
    def get_statistics(self) -> Dict[str, Any]:
        return self.stats.copy()
    def reset_statistics(self):
        self.stats = {
            'events_processed': 0,
            'events_by_type': {event_type: 0 for event_type in EventType},
            'processing_time': 0.0,
            'queue_size_max': 0
        }


class EventDrivenSystem:
    def __init__(self, simulation_manager=None):
        self.simulation_manager = simulation_manager
        self.event_queue = EventQueue()
        self.event_processor = EventProcessor(simulation_manager)
        self.running = False
        self.current_time = 0.0
        self.time_step = 0.001
        self.max_events_per_step = 1000
        self.stats = {
            'total_events_processed': 0,
            'simulation_time': 0.0,
            'events_per_second': 0.0,
            'queue_size_history': []
        }
    def start(self):
        self.running = True
        self.current_time = time.time()
        log_step("Event-driven system started")
    def stop(self):
        self.running = False
        log_step("Event-driven system stopped")
    def process_events(self, max_events: int = None) -> int:
        if max_events is None:
            max_events = self.max_events_per_step
        events_processed = 0
        start_time = time.time()
        while events_processed < max_events and self.running:
            event = self.event_queue.pop()
            if not event:
                break
            if self.event_processor.process_event(event):
                events_processed += 1
                self.stats['total_events_processed'] += 1
            self.current_time = max(self.current_time, event.timestamp)
        processing_time = time.time() - start_time
        self.stats['simulation_time'] += processing_time
        if processing_time > 0:
            self.stats['events_per_second'] = events_processed / processing_time
        queue_size = self.event_queue.size()
        self.stats['queue_size_history'].append(queue_size)
        if len(self.stats['queue_size_history']) > 1000:
            self.stats['queue_size_history'] = self.stats['queue_size_history'][-1000:]
        return events_processed
    def schedule_event(self, event: NeuralEvent):
        self.event_queue.push(event)
    def schedule_spike(self, node_id: int, timestamp: float = None, priority: int = 1):
        if timestamp is None:
            timestamp = self.current_time
        spike_event = NeuralEvent(
            event_type=EventType.SPIKE,
            timestamp=timestamp,
            source_node_id=node_id,
            priority=priority
        )
        self.schedule_event(spike_event)
    def schedule_energy_transfer(self, source_id: int, target_id: int, amount: float, timestamp: float = None):
        if timestamp is None:
            timestamp = self.current_time
        transfer_event = NeuralEvent(
            event_type=EventType.ENERGY_TRANSFER,
            timestamp=timestamp,
            source_node_id=source_id,
            target_node_id=target_id,
            data={'amount': amount},
            priority=2
        )
        self.schedule_event(transfer_event)
    def get_statistics(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats['queue_size'] = self.event_queue.size()
        stats['processor_stats'] = self.event_processor.get_statistics()
        return stats
    def reset_statistics(self):
        self.stats = {
            'total_events_processed': 0,
            'simulation_time': 0.0,
            'events_per_second': 0.0,
            'queue_size_history': []
        }
        self.event_processor.reset_statistics()


def create_event_driven_system(simulation_manager=None) -> EventDrivenSystem:
    return EventDrivenSystem(simulation_manager)
if __name__ == "__main__":
    print("Event-driven system created successfully!")
    print("Features include:")
    print("- Priority-based event queue")
    print("- Thread-safe event processing")
    print("- Multiple event types (spikes, synaptic transmission, plasticity)")
    print("- Real-time event scheduling")
    print("- Comprehensive statistics tracking")
    try:
        system = create_event_driven_system()
        system.schedule_spike(1, time.time())
        system.schedule_energy_transfer(1, 2, 10.0, time.time() + 0.001)
        print(f"Event queue size: {system.event_queue.size()}")
        events_processed = system.process_events(10)
        print(f"Processed {events_processed} events")
        stats = system.get_statistics()
        print(f"System statistics: {stats}")
    except Exception as e:
        print(f"Event-driven system test failed: {e}")
    print("Event-driven system test completed!")







