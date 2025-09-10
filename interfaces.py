"""
interfaces.py

Defines interfaces for all major system components to enable loose coupling.
This file contains abstract base classes that define contracts for system components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Protocol
from dataclasses import dataclass
from torch_geometric.data import Data


class INeuralSystem(ABC):
    """Interface for all neural systems."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the neural system."""
        pass
    
    @abstractmethod
    def process(self, graph: Data, step: int) -> bool:
        """Process the neural system for one step."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        pass
    
    @abstractmethod
    def reset_statistics(self) -> None:
        """Reset system statistics."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up system resources."""
        pass


class IBehaviorEngine(INeuralSystem):
    """Interface for behavior engine."""
    
    @abstractmethod
    def update_node_behavior(self, node_id: str, graph: Data, step: int) -> bool:
        """Update a node's behavior."""
        pass
    
    @abstractmethod
    def get_behavior_statistics(self) -> Dict[str, int]:
        """Get behavior statistics."""
        pass


class ILearningEngine(INeuralSystem):
    """Interface for learning engine."""
    
    @abstractmethod
    def apply_timing_learning(self, pre_node: Dict, post_node: Dict, edge: Dict, delta_t: float) -> None:
        """Apply timing-based learning."""
        pass
    
    @abstractmethod
    def consolidate_connections(self, graph: Data) -> None:
        """Consolidate connections based on learning."""
        pass
    
    @abstractmethod
    def form_memory_traces(self, graph: Data) -> None:
        """Form memory traces from patterns."""
        pass


class IMemorySystem(INeuralSystem):
    """Interface for memory system."""
    
    @abstractmethod
    def form_memory_traces(self, graph: Data) -> None:
        """Form memory traces."""
        pass
    
    @abstractmethod
    def consolidate_memories(self, graph: Data) -> None:
        """Consolidate memories."""
        pass
    
    @abstractmethod
    def recall_patterns(self, graph: Data, target_node_idx: int) -> List[Dict]:
        """Recall patterns for a target node."""
        pass
    
    @abstractmethod
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        pass


class IHomeostasisController(INeuralSystem):
    """Interface for homeostasis controller."""
    
    @abstractmethod
    def regulate_network_activity(self, graph: Data) -> None:
        """Regulate network activity."""
        pass
    
    @abstractmethod
    def optimize_criticality(self, graph: Data) -> None:
        """Optimize network criticality."""
        pass
    
    @abstractmethod
    def monitor_network_health(self, graph: Data) -> Dict[str, Any]:
        """Monitor network health."""
        pass


class INetworkMetrics(INeuralSystem):
    """Interface for network metrics."""
    
    @abstractmethod
    def calculate_criticality(self, graph: Data) -> float:
        """Calculate network criticality."""
        pass
    
    @abstractmethod
    def analyze_connectivity(self, graph: Data) -> Dict[str, float]:
        """Analyze network connectivity."""
        pass
    
    @abstractmethod
    def measure_energy_balance(self, graph: Data) -> Dict[str, float]:
        """Measure energy balance."""
        pass
    
    @abstractmethod
    def calculate_comprehensive_metrics(self, graph: Data) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        pass


class IWorkspaceEngine(INeuralSystem):
    """Interface for workspace engine."""
    
    @abstractmethod
    def update_workspace_nodes(self, graph: Data, step: int) -> Dict[str, Any]:
        """Update workspace nodes."""
        pass
    
    @abstractmethod
    def create_workspace_node(self, node_id: int, step: int) -> Dict[str, Any]:
        """Create a workspace node."""
        pass
    
    @abstractmethod
    def get_workspace_metrics(self) -> Dict[str, Any]:
        """Get workspace metrics."""
        pass


class IEnergyBehavior(ABC):
    """Interface for energy behavior system."""
    
    @abstractmethod
    def apply_energy_behavior(self, graph: Data, behavior_params: Optional[Dict] = None) -> None:
        """Apply energy behavior to graph."""
        pass
    
    @abstractmethod
    def update_membrane_potentials(self, graph: Data) -> None:
        """Update membrane potentials."""
        pass
    
    @abstractmethod
    def apply_refractory_periods(self, graph: Data) -> None:
        """Apply refractory periods."""
        pass


class IConnectionLogic(ABC):
    """Interface for connection logic."""
    
    @abstractmethod
    def intelligent_connection_formation(self, graph: Data) -> Data:
        """Form intelligent connections."""
        pass
    
    @abstractmethod
    def create_weighted_connection(self, graph: Data, source_id: str, target_id: str, 
                                 weight: float, edge_type: str = 'excitatory') -> Data:
        """Create a weighted connection."""
        pass


class IDeathAndBirthLogic(ABC):
    """Interface for death and birth logic."""
    
    @abstractmethod
    def birth_new_dynamic_nodes(self, graph: Data) -> Data:
        """Birth new dynamic nodes."""
        pass
    
    @abstractmethod
    def remove_dead_dynamic_nodes(self, graph: Data) -> Data:
        """Remove dead dynamic nodes."""
        pass


class IErrorHandler(ABC):
    """Interface for error handling."""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: str = "", 
                    recovery_func: Optional[Callable] = None, 
                    critical: bool = False) -> bool:
        """Handle an error."""
        pass
    
    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        pass


class IPerformanceOptimizer(ABC):
    """Interface for performance optimization."""
    
    @abstractmethod
    def optimize_graph_processing(self, graph: Data) -> Data:
        """Optimize graph processing."""
        pass
    
    @abstractmethod
    def optimize_connection_formation(self, graph: Data) -> Data:
        """Optimize connection formation."""
        pass
    
    @abstractmethod
    def optimize_birth_death_logic(self, graph: Data) -> Data:
        """Optimize birth/death logic."""
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        pass


class IConfigurationManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def get(self, section: str, key: str, default: Any = None, value_type: type = str) -> Any:
        """Get a configuration value."""
        pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all values from a section."""
        pass
    
    @abstractmethod
    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value."""
        pass
    
    @abstractmethod
    def save(self) -> None:
        """Save configuration to file."""
        pass


class IEventBus(ABC):
    """Interface for event bus system."""
    
    @abstractmethod
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        pass
    
    @abstractmethod
    def publish(self, event_type: str, data: Any) -> None:
        """Publish an event."""
        pass
    
    @abstractmethod
    def clear_subscribers(self, event_type: str) -> None:
        """Clear all subscribers for an event type."""
        pass


class ISimulationState(ABC):
    """Interface for simulation state management."""
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if simulation is running."""
        pass
    
    @abstractmethod
    def set_running(self, running: bool) -> None:
        """Set simulation running state."""
        pass
    
    @abstractmethod
    def get_graph(self) -> Optional[Data]:
        """Get current simulation graph."""
        pass
    
    @abstractmethod
    def set_graph(self, graph: Data) -> None:
        """Set simulation graph."""
        pass
    
    @abstractmethod
    def get_step_counter(self) -> int:
        """Get current step counter."""
        pass
    
    @abstractmethod
    def increment_step_counter(self) -> None:
        """Increment step counter."""
        pass


class ICallbackManager(ABC):
    """Interface for callback management."""
    
    @abstractmethod
    def add_step_callback(self, callback: Callable) -> None:
        """Add a step callback."""
        pass
    
    @abstractmethod
    def add_metrics_callback(self, callback: Callable) -> None:
        """Add a metrics callback."""
        pass
    
    @abstractmethod
    def add_error_callback(self, callback: Callable) -> None:
        """Add an error callback."""
        pass
    
    @abstractmethod
    def execute_step_callbacks(self, graph: Data, step_counter: int, performance_stats: Dict) -> None:
        """Execute all step callbacks."""
        pass
    
    @abstractmethod
    def execute_metrics_callbacks(self, metrics: Dict) -> None:
        """Execute all metrics callbacks."""
        pass
    
    @abstractmethod
    def execute_error_callbacks(self, error: Exception, context: str) -> None:
        """Execute all error callbacks."""
        pass


class IUIComponent(ABC):
    """Interface for UI components."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the UI component."""
        pass
    
    @abstractmethod
    def update(self, data: Any) -> None:
        """Update the UI component with new data."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up UI component resources."""
        pass


class IWindowManager(IUIComponent):
    """Interface for window management."""
    
    @abstractmethod
    def create_window(self, window_type: str, **kwargs) -> bool:
        """Create a window of specified type."""
        pass
    
    @abstractmethod
    def close_window(self, window_id: str) -> bool:
        """Close a window by ID."""
        pass
    
    @abstractmethod
    def get_window(self, window_id: str) -> Optional[Any]:
        """Get a window by ID."""
        pass


class ISensoryVisualization(IUIComponent):
    """Interface for sensory visualization."""
    
    @abstractmethod
    def update_sensory_visualization(self, graph: Data) -> None:
        """Update sensory visualization."""
        pass
    
    @abstractmethod
    def update_sensory_features(self, graph: Data, scale: float = 1.0) -> None:
        """Update sensory features."""
        pass


class ILiveMonitoring(IUIComponent):
    """Interface for live monitoring."""
    
    @abstractmethod
    def update_live_feeds(self, total_energy: float, active_nodes: int, node_types: Dict) -> None:
        """Update live feed displays."""
        pass
    
    @abstractmethod
    def update_live_monitoring_displays(self) -> None:
        """Update all live monitoring displays."""
        pass


class ISimulationController(ABC):
    """Interface for simulation control."""
    
    @abstractmethod
    def start_simulation(self) -> bool:
        """Start the simulation."""
        pass
    
    @abstractmethod
    def stop_simulation(self) -> bool:
        """Stop the simulation."""
        pass
    
    @abstractmethod
    def reset_simulation(self) -> bool:
        """Reset the simulation."""
        pass
    
    @abstractmethod
    def run_single_step(self) -> bool:
        """Run a single simulation step."""
        pass


# Protocol-based interfaces for better type checking
class SimulationStepProcessor(Protocol):
    """Protocol for simulation step processing."""
    
    def run_single_step(self) -> bool:
        """Run a single simulation step."""
        ...


class NodeAccessLayer(Protocol):
    """Protocol for node access layer."""
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID."""
        ...
    
    def get_node_energy(self, node_id: str) -> Optional[float]:
        """Get node energy."""
        ...
    
    def set_node_energy(self, node_id: str, energy: float) -> bool:
        """Set node energy."""
        ...


class IDManager(Protocol):
    """Protocol for ID management."""
    
    def generate_unique_id(self, node_type: str = "unknown", metadata: Optional[Dict] = None) -> str:
        """Generate unique ID."""
        ...
    
    def register_node_index(self, node_id: str, index: int) -> bool:
        """Register node index."""
        ...
    
    def get_node_index(self, node_id: str) -> Optional[int]:
        """Get node index by ID."""
        ...


# Event types for event-driven communication
class EventType:
    """Event type constants."""
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_STOPPED = "simulation_stopped"
    SIMULATION_STEP_COMPLETED = "simulation_step_completed"
    NODE_ACTIVATED = "node_activated"
    NODE_DEACTIVATED = "node_deactivated"
    CONNECTION_FORMED = "connection_formed"
    CONNECTION_REMOVED = "connection_removed"
    ENERGY_TRANSFERRED = "energy_transferred"
    MEMORY_FORMED = "memory_formed"
    MEMORY_CONSOLIDATED = "memory_consolidated"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    UI_UPDATE_REQUIRED = "ui_update_required"
    CONFIGURATION_CHANGED = "configuration_changed"


# Event data structures
@dataclass
class SimulationEvent:
    """Base class for simulation events."""
    event_type: str
    timestamp: float
    data: Dict[str, Any]
    source: str


@dataclass
class NodeEvent(SimulationEvent):
    """Event related to node operations."""
    node_id: str
    node_type: str
    node_state: str


@dataclass
class ConnectionEvent(SimulationEvent):
    """Event related to connection operations."""
    source_id: str
    target_id: str
    connection_type: str
    weight: float


@dataclass
class EnergyEvent(SimulationEvent):
    """Event related to energy operations."""
    source_id: str
    target_id: str
    energy_amount: float
    transfer_type: str


@dataclass
class ErrorEvent(SimulationEvent):
    """Event related to errors."""
    error_type: str
    error_message: str
    context: str
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class PerformanceEvent(SimulationEvent):
    """Event related to performance."""
    metric_name: str
    metric_value: float
    threshold: Optional[float]
    status: str  # 'normal', 'warning', 'critical'


class ISimulationManager(INeuralSystem):
    """Interface for simulation manager."""