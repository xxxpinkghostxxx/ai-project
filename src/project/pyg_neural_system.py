import torch
from torch_geometric.data import Data  # type: ignore[import-untyped,import-not-found]
import numpy as np
import time
import threading
import queue
import logging
from typing import Any, Dict
from numpy.typing import NDArray
from datetime import datetime
from project.utils.config_manager import ConfigManager  # type: ignore[import-not-found]

# Add sklearn import for clustering optimization
try:
    from sklearn.cluster import KMeans  # type: ignore[import-untyped,import-not-found]
except ImportError:
    # Fallback if sklearn is not available
    pass


# Import shutdown utilities for graceful cleanup
from project.utils.shutdown_utils import ShutdownDetector  # type: ignore[import-not-found]
from project.utils.tensor_manager import TensorManager  # type: ignore[import-not-found]
from project.system.vector_engine import (  # type: ignore[import-not-found]
    DensityManager,
    NodeArrayStore,
    NodeClassSpec,
    NodeRuleRegistry,
    SparseNodeStore,
    VectorizedSimulationEngine,
)

NODE_TYPE_SENSORY = 0
NODE_TYPE_DYNAMIC = 1
NODE_TYPE_WORKSPACE = 2
NODE_TYPE_HIGHWAY = 3
NODE_TYPE_NAMES = ['Sensory', 'Dynamic', 'Workspace', 'Highway']

# --- Dynamic Node Subtypes ---
SUBTYPE_TRANSMITTER = 0
SUBTYPE_RESONATOR = 1
SUBTYPE_DAMPENER = 2
SUBTYPE_NAMES = ['Transmitter', 'Resonator', 'Dampener']

# --- Third Subtype Slot (Connection Call Frequency) ---
SUBTYPE3_1P_5 = 0  # 1% every 5 steps
SUBTYPE3_2P_10 = 1 # 2% every 10 steps
SUBTYPE3_3P_20 = 2 # 3% every 20 steps
SUBTYPE3_NAMES = ['1%/5steps', '2%/10steps', '3%/20steps']

# --- Fourth Subtype Slot (Connection-based Energy Gain) ---
SUBTYPE4_GAIN_1 = 0   # Gain 1 per connection per step
SUBTYPE4_GAIN_01 = 1  # Gain 0.1 per connection per step
SUBTYPE4_GAIN_001 = 2 # Gain 0.01 per connection per step
SUBTYPE4_NAMES = ['+1/conn', '+0.1/conn', '+0.01/conn']

# --- Config-like constants (tune as needed) ---
# Note: These should be consolidated with config.py values
NODE_SPAWN_THRESHOLD = 20.0
NODE_DEATH_THRESHOLD = -10.0  # Match config.py value
NODE_ENERGY_SPAWN_COST = 19.52  # From config.py: NODE_ENERGY_CAP * DYNAMIC_SPAWN_COST_PCT
NODE_ENERGY_DECAY = 0.005  # Match config.py DYNAMIC_NODE_ENERGY_DECAY
MAX_NODE_BIRTHS_PER_STEP = 80  # Match config.py value
MAX_CONN_BIRTHS_PER_STEP = 40  # Match config.py value
NODE_ENERGY_CAP = 244.0  # Match config.py value
CONNECTION_MAINTENANCE_COST = 0.122  # From config.py: CONN_MAINTENANCE_COST
TRANSMISSION_LOSS = 0.9  # Fraction of incoming energy actually received (simulate loss)
CONN_ENERGY_TRANSFER_CAPACITY = 0.3  # From config.py: max fraction of energy transferred per step
NODE_ENERGY_NOISE = 0.01  # Small noise added to dynamic nodes each step

# --- Dynamic Node Types (theorycraft) ---
# NODE_TYPE_TRANSMITTER: Boosts outgoing energy transfer (higher weight/capacity)
# NODE_TYPE_RESONATOR: Receives energy more efficiently, but loses more to decay
# NODE_TYPE_DAMPENER: Reduces incoming energy, but is more stable (lower decay)
# These types could be encoded as a new node feature, e.g., 'dynamic_subtype', and used to modulate energy flows.

# --- Connection Types (theorycraft) ---
# TYPE_EXCITATORY: Standard positive-weight connection, transmits energy normally
# TYPE_INHIBITORY: Negative-weight connection, reduces target's energy
# TYPE_GATED: Only transmits energy if source node is above a threshold
# TYPE_PLASTIC: Weight can change over time based on activity (learning)
# These could be encoded as an edge feature (e.g., 'conn_type') and used to modulate transfer, learning, or gating.

# --- Connection Subtypes ---
CONN_TYPE_EXCITATORY = 0
CONN_TYPE_INHIBITORY = 1
CONN_TYPE_GATED = 2
CONN_TYPE_PLASTIC = 3
CONN_TYPE_NAMES = ['Excitatory', 'Inhibitory', 'Gated', 'Plastic']
GATE_THRESHOLD = 0.5  # Example threshold for gated connections

# For plastic connections
PLASTIC_LEARNING_RATE_MIN = 0.001
PLASTIC_LEARNING_RATE_MAX = 0.05

SUBTYPE2_NAMES = ['Fertile', 'Normal', 'Sterile']  # Example names for new subtype

# --- New Connection Subtype3 (Directionality) ---
CONN_SUBTYPE3_ONE_WAY_OUT = 0
CONN_SUBTYPE3_ONE_WAY_IN = 1
CONN_SUBTYPE3_FREE_FLOW = 2
CONN_SUBTYPE3_NAMES = ['OneWayOut', 'OneWayIn', 'FreeFlow']

# --- Highway Node Constants ---
HIGHWAY_CONNECTION_RADIUS = 25  # 25x25 area for highway node connections

logger = logging.getLogger(__name__)

class ConnectionWorker(threading.Thread):
    """
    Connection Worker Thread for handling connection growth and culling operations.

    This worker thread processes connection-related tasks in batches to optimize performance
    and prevent blocking the main thread. It handles both connection growth (adding new edges)
    and connection culling (removing invalid edges) operations.

    Features:
    - Thread-safe operation with locking mechanisms
    - Performance optimization with caching
    - Comprehensive error handling and retry logic
    - Performance metrics tracking
    - Timeout detection and recovery

    Args:
        system: Reference to the PyGNeuralSystem instance
        batch_size: Number of connections to process in each batch (default: 25)
    """

    def __init__(self, system: 'PyGNeuralSystem', batch_size: int = 25) -> None:
        super().__init__()
        self.system = system
        self.batch_size = batch_size
        self.task_queue: queue.Queue[Any] = queue.Queue(maxsize=100)  # Limited queue size
        self.result_queue: queue.Queue[Any] = queue.Queue(maxsize=100)  # Limited queue size
        self.stop_event = threading.Event()
        self.daemon = True
        self._lock = threading.Lock()
        self._processing = False
        self._error_count: int = 0
        self._max_retries: int = 3
        self._retry_delay: float = 1.0
        self._last_activity: float = time.time()
        self._timeout: float = 30.0  # 30 seconds timeout
        self._metrics: dict[str, Any] = {
            'tasks_processed': 0,
            'errors': 0,
            'retries': 0,
            'processing_time': 0.0,
            'batch_processing_time': 0.0,
            'queue_wait_time': 0.0,
            'peak_queue_size': 0,
            'empty_queue_count': 0,
            'queue_utilization': 0.0
        }
        self._batch_processing_start: float = 0.0
        self._queue_wait_start: float = 0.0
        self._performance_cache: dict[str, Any] = {}  # Cache for performance optimization

    def run(self) -> None:
        """Main worker loop with enhanced performance optimization and error handling"""
        while not self.stop_event.is_set():
            try:
                # Check for timeout with performance optimization
                if time.time() - self._last_activity > self._timeout:
                    logger.warning("Connection worker timeout, restarting...")
                    self._error_count += 1
                    if self._error_count >= self._max_retries:
                        logger.error("Max retries exceeded, stopping worker")
                        self.stop_event.set()
                        break
                    time.sleep(self._retry_delay * (2 ** (self._error_count - 1)))
                    continue

                # Get task with timeout and performance monitoring
                self._queue_wait_start = time.time()
                try:
                    task = self.task_queue.get(timeout=0.1)
                    self._metrics['queue_wait_time'] += time.time() - self._queue_wait_start
                except queue.Empty:
                   # Empty queue handling with improved startup behavior
                   # Add a small delay to reduce CPU usage during idle periods
                   time.sleep(0.01)  # 10ms sleep to reduce CPU usage
   
                   # Log queue empty state periodically for debugging
                   self._metrics['empty_queue_count'] = self._metrics.get('empty_queue_count', 0) + 1
                   
                   # During startup phase, be more aggressive about logging empty queue
                   if self._metrics['empty_queue_count'] <= 10:  # Log first 10 empty queue occurrences
                       logger.debug(f"Connection worker: Queue empty during startup (count: {self._metrics['empty_queue_count']})")
                   elif self._metrics['empty_queue_count'] % 100 == 0:  # Then log every 100
                       logger.debug(f"Connection worker: Queue empty (count: {self._metrics['empty_queue_count']})")
                   
                   # If queue has been empty for a while, check if system is still active
                   if self._metrics['empty_queue_count'] > 1000:  # After 1000 empty checks (~10 seconds)
                       if hasattr(self.system, 'g') and self.system.g is not None and self.system.g.num_nodes is not None:
                           logger.info(f"Connection worker: Queue empty but system active with {self.system.g.num_nodes} nodes")
                       self._metrics['empty_queue_count'] = 0  # Reset counter to avoid overflow
                   
                   continue

                # Update peak queue size metric
                current_queue_size = self.task_queue.qsize()
                if current_queue_size > self._metrics['peak_queue_size']:
                    self._metrics['peak_queue_size'] = current_queue_size

                with self._lock:
                    self._processing = True
                    self._batch_processing_start = time.time()

                    try:
                        # Wait for any ongoing graph modifications to complete with timeout optimization
                        if hasattr(self.system, 'graph_modification_lock'):
                            if not self.system.graph_modification_lock.wait(timeout=0.5):  # Reduced timeout
                                logger.warning("Timeout waiting for graph modification to complete")
                                self.result_queue.put({'type': 'error', 'error': 'Timeout waiting for graph modification'})
                                return

                        # CRITICAL: Do NOT cache 'grow' or 'cull' tasks - they must generate NEW connections each time!
                        # Only cache read-only operations like metrics
                        should_use_cache = task['type'] not in ['grow', 'cull']
                        cache_key = f"{task['type']}:{self.batch_size}"
                        if should_use_cache and cache_key in self._performance_cache:
                            # Use cached result for performance optimization (read-only tasks only)
                            cached_result: dict[str, Any] = self._performance_cache[cache_key]
                            self.result_queue.put(cached_result)
                            logger.debug(f"Using cached result for task type: {task['type']}")
                        else:
                            result: dict[str, Any]
                            if task['type'] == 'grow':
                                # Validate graph state before preparing batch with fast validation
                                if not self._fast_validate_graph_state():
                                    logger.warning("Graph state validation failed in grow task")
                                    self.result_queue.put({'type': 'error', 'error': 'Invalid graph state'})
                                    return
                                # Prepare connection growth batch with optimization
                                batch = self.system.prepare_connection_growth_batch(self.batch_size)
                                result = {'type': 'grow', 'batch': batch}
                                # CRITICAL: Do NOT cache grow results - must generate NEW connections each time!
                                self.result_queue.put(result)
                            elif task['type'] == 'cull':
                                # Validate graph state before preparing batch with fast validation
                                if not self._fast_validate_graph_state():
                                    logger.warning("Graph state validation failed in cull task")
                                    self.result_queue.put({'type': 'error', 'error': 'Invalid graph state'})
                                    return
                                # Prepare culling batch with optimization
                                batch = self.system.prepare_cull_batch(self.batch_size)
                                result = {'type': 'cull', 'batch': batch}
                                # CRITICAL: Do NOT cache cull results - must compute fresh each time!
                                self.result_queue.put(result)
                            else:
                                logger.error("Unknown task type: %s", task['type'])
                                self.result_queue.put({'type': 'error', 'error': f"Unknown task type: {task['type']}"})

                        # Update metrics with enhanced performance tracking
                        batch_time = time.time() - self._batch_processing_start
                        self._metrics['tasks_processed'] += 1
                        self._metrics['processing_time'] += batch_time
                        self._metrics['batch_processing_time'] += batch_time
                        self._last_activity = time.time()
                        self._error_count = 0  # Reset error count on success

                        # Log performance metrics periodically
                        if self._metrics['tasks_processed'] % 10:
                            logger.debug(f"Connection worker performance: {batch_time:.4f}s batch time, {self._metrics['queue_wait_time']:.4f}s queue wait")

                    except Exception as e:
                        self._metrics['errors'] += 1
                        logger.error("Error processing task: %s", e)
                        self.result_queue.put({'type': 'error', 'error': str(e)})
                        self._error_count += 1
                        if self._error_count >= self._max_retries:
                            logger.error("Max retries exceeded, stopping worker")
                            self.stop_event.set()
                            break
                    finally:
                        self._processing = False
                        self.task_queue.task_done()

            except Exception as e:
                logger.error("Critical error in connection worker: %s", e)
                self._metrics['errors'] += 1
                self._error_count += 1
                if self._error_count >= self._max_retries:
                    logger.error("Max retries exceeded, stopping worker")
                    self.stop_event.set()
                    break
                time.sleep(float(self._retry_delay * (2 ** (self._error_count - 1))))

    def stop(self) -> None:
        """Safely stop the worker thread"""
        with self._lock:
            if self._processing:
                # Wait for current processing to complete
                while self._processing:
                    time.sleep(0.1)
            self.stop_event.set()
            logger.info("Connection worker stopped")

    def is_processing(self) -> bool:
        """Check if worker is currently processing a task"""
        with self._lock:
            return self._processing

    def clear_queues(self) -> None:
        """Clear all pending tasks and results"""
        with self._lock:
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                    self.task_queue.task_done()
                except queue.Empty:
                    break
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            logger.info("Connection worker queues cleared")

    def get_metrics(self) -> dict[str, Any]:
        """Get worker metrics"""
        with self._lock:
            return self._metrics.copy()

    def queue_task(self, task_type: str, **kwargs: Any) -> bool:
        """Queue a new task with validation"""
        if task_type not in ['grow', 'cull']:
            raise ValueError(f"Invalid task type: {task_type}")

        task: dict[str, Any] = {'type': task_type, **kwargs}
        try:
            self.task_queue.put(task, timeout=1.0)
            return True
        except queue.Full:
            logger.warning("Task queue full, task dropped")
            return False

    def get_result(self, timeout: float = 0.1) -> Any | None:
        """Get a result with timeout"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """Check if worker is alive and healthy"""
        try:
            return (super().is_alive() and
                    not self.stop_event.is_set() and
                    self._error_count < self._max_retries)
        except Exception:
            return False

    def _fast_validate_graph_state(self) -> bool:
        """Fast validation of graph state for performance optimization."""
        try:
            if not hasattr(self.system, 'g') or self.system.g is None:
                return False

            g = self.system.g
            # Basic checks only for performance
            if not hasattr(g, 'num_nodes') or g.num_nodes is None:
                return False

            if not hasattr(g, 'energy') or g.energy is None:
                return False

            # Quick tensor shape validation
            if g.num_nodes != g.energy.shape[0]:
                return False

            return True

        except Exception as e:
            logger.warning(f"Fast graph validation failed: {str(e)}")
            return False

class PyGNeuralSystem:
    """
    PyTorch Geometric Neural System for energy-based neural network simulation.

    This class implements a complex neural system using PyTorch Geometric for graph-based
    neural network operations. It simulates energy flow between nodes, handles dynamic
    node creation/destruction, and manages connection growth and pruning.

    Key Features:
    - Energy-based neural dynamics with multiple node types (sensory, dynamic, workspace, highway)
    - Thread-safe graph operations with synchronization primitives
    - Tensor management and validation
    - Connection worker for background processing
    - Comprehensive error handling and recovery mechanisms
    - Memory management and cleanup
    - Performance optimization and monitoring

    Node Types:
    - NODE_TYPE_SENSORY: Translation nodes that convert external data (pixel values, sound waves, etc.)
      to energy. They always maintain and push out the true value of their assigned pixel's contrast
      level. They push energy out through connections but never receive energy from other nodes.
      Future: Will support multiple types (sound wave arrays, different vision types, VMs, etc.)
    - NODE_TYPE_DYNAMIC: Processing nodes that handle energy transfer and computation
    - NODE_TYPE_WORKSPACE: Output/interface nodes
    - NODE_TYPE_HIGHWAY: High-capacity nodes for efficient energy transfer

    Connection Types:
    - CONN_TYPE_EXCITATORY: Positive energy transfer
    - CONN_TYPE_INHIBITORY: Negative energy transfer
    - CONN_TYPE_GATED: Conditional energy transfer
    - CONN_TYPE_PLASTIC: Adaptive connections with learning capabilities

    Args:
        sensory_width: Width of sensory input grid
        sensory_height: Height of sensory input grid
        n_dynamic: Target number of dynamic nodes
        workspace_size: Tuple of (width, height) for workspace grid
        device: Computing device ('cpu' or 'cuda')
    """

    def __init__(self, sensory_width: int, sensory_height: int, n_dynamic: int, workspace_size: tuple[int, int] = (16, 16), device: str = 'cpu') -> None:
        self.device = device
        self.g: Data | None = None
        self.sensory_width = sensory_width
        self.sensory_height = sensory_height
        self.n_sensory_target = sensory_width * sensory_height
        self.n_dynamic_target = n_dynamic
        self.workspace_size: tuple[int, int] = workspace_size
        self.n_workspace_target: int = workspace_size[0] * workspace_size[1]
        self.n_sensory = 0
        self.n_dynamic = 0
        self.n_workspace = 0
        self.n_total = 0
        self.sensory_warmup_frames = 0
        self._sensory_update_count = 0
        self.config_manager: ConfigManager | None = None  # Store config manager for gain/bias access
        try:
            self.config_manager = ConfigManager()
            warmup_cfg = self.config_manager.get_config('sensory', 'warmup_frames')
            if isinstance(warmup_cfg, int) and warmup_cfg > 0:
                self.sensory_warmup_frames = warmup_cfg
        except Exception:  # pylint: disable=broad-exception-caught
            self.sensory_warmup_frames = 0
        # Thread synchronization primitives
        self._graph_lock = threading.RLock()  # Recursive lock for graph operations
        self.graph_modification_lock = threading.Event()  # Signal when graph is being modified
        self.graph_modification_lock.set()  # Initially unlocked
        self._init_empty_graph()
        # Initialize the graph with nodes
        self._init_graph()

        # Initialize tensor manager
        self.tensor_manager = TensorManager(self)
        # --- Metrics ---
        self.node_births = 0
        self.node_deaths =  0
        self.conn_births = 0
        self.conn_deaths = 0
        self.total_node_births = 0
        self.total_node_deaths = 0
        self.total_conn_births = 0
        self.total_conn_deaths = 0
        self.step_counter = 0
        # --- Startup phase for initial node spawning ---
        self.startup_phase = True
        self.startup_batch_size = 500
        self.startup_sensory_created = 0
        self.startup_dynamic_created = 0
        self.startup_workspace_created = 0
        self.startup_connections_done = False
        self.death_queue: list[int] = []
        self.birth_queue: list[tuple[int, dict[str, Any]]] = []
        self.conn_growth_queue: list[tuple[int, int, int]] = []
        self.conn_candidate_queue: list[int] = []
        # --- Sensory node true values storage ---
        # Sensory nodes are translation nodes that always maintain their assigned pixel/contrast values
        # This stores the true energy values that sensory nodes should always have
        # (scaled from pixel values 0-255 to energy range 0-NODE_ENERGY_CAP)
        self.sensory_true_values: torch.Tensor | None = None  # Stores true energy values for sensory nodes
        # --- Memory management ---
        self._memory_tracker: dict[str, Any] = {
            'peak_nodes': 0,
            'peak_edges': 0,
            'last_cleanup': time.time(),
            'cleanup_interval': 60.0  # Cleanup every 60 seconds
        }
        self._connection_worker = None
        self.vector_engine: VectorizedSimulationEngine | None = None
        self._skip_vector_engine_step = False  # Flag to skip vector engine step (for testing)
        self._use_unified_fast_path = True  # Use vector engine with PyG rules (faster)
        self._init_vector_engine()

    def _init_vector_engine(self) -> None:
        """Create a vectorized engine mirror for high-scale simulation."""
        try:
            capacity = max(int(self.n_total * 2), 1024)
            store = NodeArrayStore(capacity=capacity, device=self.device)

            if self.g is not None and hasattr(self.g, 'energy'):
                energies = self.g.energy.view(-1)
                positions = getattr(self.g, 'pos', torch.zeros((len(energies), 2), device=self.device))
                class_ids = getattr(self.g, 'node_type', torch.zeros_like(energies, dtype=torch.int64))
                scaled_pos = torch.round(positions * 1000).to(torch.int32)
                store.spawn(
                    count=len(energies),
                    class_ids=class_ids.to(torch.int64),
                    energies=energies,
                    positions=scaled_pos
                )

            registry = NodeRuleRegistry()
            registry.register(NodeClassSpec(class_id=NODE_TYPE_SENSORY, name="sensory", decay=0.0, death_threshold=-1e9, max_energy=NODE_ENERGY_CAP))
            registry.register(NodeClassSpec(class_id=NODE_TYPE_DYNAMIC, name="dynamic", decay=NODE_ENERGY_DECAY, death_threshold=NODE_DEATH_THRESHOLD, max_energy=NODE_ENERGY_CAP))
            registry.register(NodeClassSpec(class_id=NODE_TYPE_WORKSPACE, name="workspace", decay=0.0, death_threshold=-1e9, max_energy=NODE_ENERGY_CAP))

            tile_size = max(1, int(max(self.grid_width, self.grid_height) // 16) or 1)
            raw_caps = {
                NODE_TYPE_DYNAMIC: int(self.n_dynamic_target * 2.0) if self.n_dynamic_target else None,
                NODE_TYPE_SENSORY: int(self.n_sensory_target * 2.0) if self.n_sensory_target else None,
                NODE_TYPE_WORKSPACE: int(self.n_workspace_target * 2.0) if self.n_workspace_target else None,
            }
            per_class_caps = {key: val for key, val in raw_caps.items() if val is not None} or None
            density = DensityManager(
                world_size=(max(1, self.grid_width), max(1, self.grid_height)),
                tile_size=tile_size,
                per_class_caps=per_class_caps,
            )
            sparse = None
            try:
                sparse = SparseNodeStore(capacity=capacity * 4)
            except Exception as mmap_err:  # pylint: disable=broad-exception-caught
                logger.warning("Sparse store unavailable (memmap failed): %s", mmap_err)

            self.vector_engine = VectorizedSimulationEngine(store, registry, density, sparse)
            logger.info("Vector engine initialized with capacity %d on %s", capacity, self.device)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Vector engine initialization failed: %s", str(e))
            self.vector_engine = None

    def _sync_vector_store_to_graph(self) -> None:
        """Push vector engine state into the PyG graph for UI/consumers."""
        if self.g is None or self.vector_engine is None:
            return
        try:
            active_nodes = int(self.g.num_nodes) if self.g.num_nodes is not None else 0
            active = min(active_nodes, self.vector_engine.store.capacity)
            if active <= 0:
                return
            energies = self.vector_engine.store.energy[:active]
            if energies.shape[0] == self.g.energy.shape[0]:
                self.g.energy[:active, 0] = energies
        except Exception as sync_error:  # pylint: disable=broad-exception-caught
            logger.debug("Vector engine sync skipped: %s", sync_error)

    def _sync_graph_to_vector_store(self) -> None:
        """Push PyG graph energy state into the vector engine store.
        
        This is the reverse of _sync_vector_store_to_graph() - it copies
        energy values from g.energy TO vector_engine.store.energy.
        Useful when energy values are set directly on g.energy and need
        to be reflected in the vector engine.
        """
        if self.g is None or self.vector_engine is None:
            return
        try:
            active_nodes = int(self.g.num_nodes) if self.g.num_nodes is not None else 0
            active = min(active_nodes, self.vector_engine.store.capacity)
            if active <= 0:
                return
            # Copy FROM g.energy TO vector_engine.store.energy
            if self.g.energy.shape[0] >= active:
                self.vector_engine.store.energy[:active] = self.g.energy[:active, 0]
                # Also sync node types/class_ids if present
                if hasattr(self.g, 'node_type') and self.g.node_type is not None:
                    self.vector_engine.store.class_ids[:active] = self.g.node_type[:active]
                # Mark nodes as active
                self.vector_engine.store.active_mask[:active] = True
                self.vector_engine.store.active_count = active
                logger.debug("Synced %d nodes from graph to vector store", active)
        except Exception as sync_error:  # pylint: disable=broad-exception-caught
            logger.debug("Graph to vector engine sync skipped: %s", sync_error)

    def vector_engine_step(self, dt: float = 1.0) -> None:
        """Step vector engine and sync to the graph if available.
        
        If _skip_vector_engine_step is True, this method does nothing.
        This is useful for testing when you want to validate only the
        PyG-based simulation without the vector engine's decay rules.
        """
        if self._skip_vector_engine_step:
            return
        if self.vector_engine is None:
            return
        self.vector_engine.step(dt)
        density_metrics = self.vector_engine.enforce_density()
        if density_metrics:
            self.vector_engine.metrics.update(density_metrics)
        self._sync_vector_store_to_graph()

    def vector_engine_step_unified(self) -> Dict[str, float]:
        """
        Unified fast path: Vector engine speed with PyG rules.
        
        This method uses the vector engine's vectorized operations but applies
        the same rules as the PyG simulation:
        - Connection-based energy transfer with all types/subtypes
        - Connection maintenance cost (not multiplicative decay)
        - Sensory overwrite from pixel values
        - Workspace adjustment towards target
        
        Returns:
            Metrics dict with transfer and processing stats
        """
        if self.vector_engine is None or self.g is None:
            return {}
        
        metrics: Dict[str, float] = {}
        g = self.g
        
        # Sync current PyG state to vector engine
        self._sync_graph_to_vector_store()
        
        # === STEP 1: Apply connection-based energy transfers ===
        if g.edge_index is not None and g.edge_index.numel() > 0:
            src = g.edge_index[0]
            dst = g.edge_index[1]
            
            # Get connection attributes - squeeze to ensure 1D tensors for vector engine
            weights = g.edge_weight if hasattr(g, 'edge_weight') and g.edge_weight is not None else torch.ones(src.numel(), device=self.device)
            if weights.dim() > 1:
                weights = weights.squeeze(-1)
            conn_types = g.conn_type if hasattr(g, 'conn_type') and g.conn_type is not None else None
            if conn_types is not None and conn_types.dim() > 1:
                conn_types = conn_types.squeeze(-1)
            conn_subtypes = g.conn_subtype3 if hasattr(g, 'conn_subtype3') and g.conn_subtype3 is not None else None
            if conn_subtypes is not None and conn_subtypes.dim() > 1:
                conn_subtypes = conn_subtypes.squeeze(-1)
            
            # Get dynamic subtypes for source/dest nodes - squeeze to ensure 1D
            src_subtypes = None
            dst_subtypes = None
            if hasattr(g, 'dynamic_subtype') and g.dynamic_subtype is not None:
                dyn_subtype = g.dynamic_subtype
                if dyn_subtype.dim() > 1:
                    dyn_subtype = dyn_subtype.squeeze(-1)
                src_subtypes = dyn_subtype[src]
                dst_subtypes = dyn_subtype[dst]
            
            transfer_metrics = self.vector_engine.apply_connection_batch_full(
                src=src,
                dst=dst,
                weights=weights,
                conn_types=conn_types,
                conn_subtypes=conn_subtypes,
                src_subtypes=src_subtypes,
                dst_subtypes=dst_subtypes,
                transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY,
                transmission_loss=TRANSMISSION_LOSS,
                sensory_type_id=NODE_TYPE_SENSORY,
                workspace_type_id=NODE_TYPE_WORKSPACE,
                gate_threshold=GATE_THRESHOLD,
            )
            metrics.update(transfer_metrics)
        
        # === STEP 2: Calculate out-degrees for maintenance cost ===
        out_degrees = torch.zeros(self.vector_engine.store.capacity, device=self.device)
        if g.edge_index is not None and g.edge_index.numel() > 0:
            src = g.edge_index[0]
            ones = torch.ones(src.numel(), device=self.device)
            out_degrees.scatter_add_(0, src, ones)
        
        # === STEP 3: Apply per-node rules (PyG style) ===
        # Flatten sensory_true_values from [N, 1] to [N] for vector engine
        sensory_values_1d = None
        if self.sensory_true_values is not None:
            if self.sensory_true_values.dim() > 1:
                sensory_values_1d = self.sensory_true_values.squeeze(-1)
            else:
                sensory_values_1d = self.sensory_true_values
        
        rule_metrics = self.vector_engine.apply_pyg_rules(
            out_degrees=out_degrees,  # Full capacity - masks in apply_pyg_rules are also full capacity
            maintenance_cost_per_conn=CONNECTION_MAINTENANCE_COST,
            noise_scale=NODE_ENERGY_NOISE,
            energy_cap=NODE_ENERGY_CAP,
            death_threshold=NODE_DEATH_THRESHOLD,
            sensory_type_id=NODE_TYPE_SENSORY,
            workspace_type_id=NODE_TYPE_WORKSPACE,
            dynamic_type_id=NODE_TYPE_DYNAMIC,
            sensory_true_values=sensory_values_1d,
            workspace_target=50.0,
            workspace_adjust_rate=0.1,
        )
        metrics.update(rule_metrics)
        
        # === STEP 4: Sync back to PyG graph ===
        self._sync_vector_store_to_graph()
        
        return metrics

    def validate_graph_state(self) -> bool:
        """Validate that graph state is consistent using the advanced TensorManager"""
        if self.g is None:
            return False

        try:
            # Use the advanced TensorManager for comprehensive validation
            validation_results = self.tensor_manager.validate_tensor_shapes()

            # Check for any validation failures
            invalid_tensors = [key for key, valid in validation_results.items() if not valid]

            if invalid_tensors:
                logger.warning(f"Tensor validation failed for: {invalid_tensors}")
                return False

            # Additional graph-level validation
            if self.g.num_nodes is None:
                logger.warning("Graph num_nodes is None")
                return False

            # Check that tracked counts match actual counts
            if self.g.num_nodes != self.n_total:
                logger.warning(f"Tracked node count {self.n_total} doesn't match graph num_nodes {self.g.num_nodes}")
                # Synchronize the tracked count
                self.n_total = self.g.num_nodes
                logger.info(f"Synchronized tracked node count to {self.n_total}")

            return True

        except Exception as e:
            logger.error(f"Error validating graph state: {e}")
            return False

    def wait_for_workers_idle(self, timeout: float = 5.0) -> bool:
        """Wait for all workers to finish current tasks"""
        if not hasattr(self, '_connection_worker') or self._connection_worker is None:
            return True
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._connection_worker.is_processing():
                return True
            time.sleep(0.01)
        logger.warning("Timeout waiting for workers to become idle")
        return False
    def cleanup(self) -> None:
        """Clean up resources and free memory using advanced TensorManager"""
        try:
            # Check if we're already cleaned up to avoid double cleanup
            if not hasattr(self, 'g') or self.g is None:
                logger.info("Cleanup called but graph already cleaned up")
                return

            logger.info("Cleanup called. Optimizing tensor memory and resources.")

            # Use TensorManager for advanced memory optimization
            optimization_stats = self.tensor_manager.optimize_tensor_memory()
            logger.info(f"Tensor memory optimization: {optimization_stats['memory_freed_mb']:.2f}MB freed, {optimization_stats['tensors_cleaned']} tensors cleaned")

            # Stop connection worker if running
            if self._connection_worker is not None:
                try:
                    self._connection_worker.stop()
                    self._connection_worker.clear_queues()
                except Exception as worker_e:
                    logger.warning(f"Error stopping connection worker: {str(worker_e)}")
                finally:
                    self._connection_worker = None

            # Clear queues
            self.death_queue.clear()
            self.birth_queue.clear()
            self.conn_growth_queue.clear()
            self.conn_candidate_queue.clear()

            # Clear graph data
            self.g = None

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache if using GPU
            if self.device == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception as cuda_e:
                    logger.warning(f"Error clearing CUDA cache: {str(cuda_e)}")

            logger.info("System cleanup completed with advanced tensor optimization")
        except Exception as e:
            logger.error("Error during cleanup: %s", str(e))

    def _check_memory_usage(self) -> None:
        """Check memory usage and trigger cleanup if needed with enhanced fragmentation reduction"""
        current_time = time.time()
        if current_time - self._memory_tracker['last_cleanup'] > self._memory_tracker['cleanup_interval']:
            # Update peak memory usage
            if hasattr(self, 'g') and self.g is not None:
                num_nodes = self.g.num_nodes
                num_edges = self.g.num_edges
                if num_nodes is not None:
                    self._memory_tracker['peak_nodes'] = max(
                        self._memory_tracker['peak_nodes'],
                        num_nodes
                    )

                    self._memory_tracker['peak_edges'] = max(
                        self._memory_tracker['peak_edges'],
                        num_edges
                    )

            # Check if we need cleanup with enhanced fragmentation detection
            needs_cleanup = False
            if (self._memory_tracker['peak_nodes'] > self.n_total * 2) or \
                (self.g is not None and self._memory_tracker['peak_edges'] > self.g.num_edges * 2):
                needs_cleanup = True

            # Additional fragmentation detection
            if hasattr(self, 'g') and self.g is not None:
                # Check for tensor fragmentation
                if self._detect_tensor_fragmentation():
                    needs_cleanup = True

            if needs_cleanup:
                # Perform enhanced cleanup with fragmentation reduction
                self._perform_enhanced_cleanup()
                self._memory_tracker['last_cleanup'] = current_time

    def _detect_tensor_fragmentation(self) -> bool:
        """Detect memory fragmentation in tensors."""
        if not hasattr(self, 'g') or self.g is None:
            return False

        try:
            g = self.g
            fragmented_count = 0
            total_tensors = 0

            # Check node tensors for fragmentation
            node_tensor_keys = ['energy', 'node_type', 'pos', 'dynamic_subtype', 'dynamic_subtype2',
                               'dynamic_subtype3', 'dynamic_subtype4', 'max_connections', 'velocity',
                               'parent', 'phase_offset']

            for key in node_tensor_keys:
                if hasattr(g, key):
                    tensor = getattr(g, key)
                    if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                        total_tensors += 1
                        if not tensor.is_contiguous():
                            fragmented_count += 1

            # Check edge tensors for fragmentation
            edge_tensor_keys = ['weight', 'energy_transfer_capacity', 'conn_type',
                               'plastic_lr', 'gate_threshold', 'conn_subtype2', 'conn_subtype3']

            for key in edge_tensor_keys:
                if hasattr(g, key):
                    tensor = getattr(g, key)
                    if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                        total_tensors += 1
                        if not tensor.is_contiguous():
                            fragmented_count += 1

            # Consider fragmented if more than 20% of tensors are non-contiguous
            fragmentation_percentage = (fragmented_count / total_tensors * 100) if total_tensors > 0 else 0
            logger.debug(f"Tensor fragmentation detected: {fragmented_count}/{total_tensors} tensors ({fragmentation_percentage:.1f}%)")

            return fragmentation_percentage > 20.0

        except Exception as e:
            logger.warning(f"Error detecting tensor fragmentation: {str(e)}")
            return False

    def _perform_enhanced_cleanup(self) -> None:
        """Perform enhanced cleanup with memory defragmentation and optimization."""
        try:
            logger.info("Performing enhanced cleanup with memory defragmentation...")

            # Step 1: Perform regular cleanup
            self.cleanup()

            # Step 2: Perform tensor defragmentation
            defrag_stats = self.tensor_manager.defragment_tensors()
            if defrag_stats['tensors_defragmented'] > 0:
                logger.info(f"Defragmented {defrag_stats['tensors_defragmented']} tensors, improved memory by {defrag_stats['memory_improvement_mb']:.2f}MB")

            # Step 3: Perform advanced memory cleanup
            cleanup_stats = self.tensor_manager.advanced_memory_cleanup()
            if cleanup_stats['tensors_consolidated'] > 0:
                logger.info(f"Advanced memory cleanup: {cleanup_stats['tensors_consolidated']} tensors consolidated, {cleanup_stats['memory_freed_mb']:.2f}MB freed")

            # Step 4: Force garbage collection
            import gc
            collected = gc.collect()
            logger.debug(f"Garbage collection during enhanced cleanup: {collected} objects collected")

            # Step 5: Clear CUDA cache if using GPU
            if self.device == 'cuda':
                try:
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared during enhanced cleanup")
                except Exception as cuda_e:
                    logger.warning(f"Error clearing CUDA cache: {str(cuda_e)}")

            logger.info("Enhanced cleanup completed")

        except Exception as e:
            logger.error(f"Error during enhanced cleanup: {str(e)}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup"""
        ShutdownDetector.safe_cleanup(self.cleanup, "PyGNeuralSystem cleanup")

    def _init_empty_graph(self) -> None:
        logger.info("Initializing empty graph.")
        # Start with an empty graph and all required node features
        self.g = Data()
        self.g.device = self.device  # Add device attribute to graph object
        device = self.device
        self.g.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)  # type: ignore[attr-defined]
        self.g.energy = torch.zeros((0, 1), device=device)  # type: ignore[attr-defined]
        self.g.node_type = torch.zeros((0,), dtype=torch.int64, device=device)  # type: ignore[attr-defined]
        self.g.pos = torch.zeros((0, 2), device=device)  # type: ignore[attr-defined]
        self.g.dynamic_subtype = torch.zeros((0,), dtype=torch.int64, device=device)  # type: ignore[attr-defined]
        self.g.dynamic_subtype2 = torch.zeros((0,), dtype=torch.int64, device=device)  # type: ignore[attr-defined]
        self.g.dynamic_subtype3 = torch.zeros((0,), dtype=torch.int64, device=device)  # type: ignore[attr-defined]
        self.g.dynamic_subtype4 = torch.zeros((0,), dtype=torch.int64, device=device)  # type: ignore[attr-defined]
        self.g.max_connections = torch.zeros((0,), dtype=torch.int64, device=device)  # type: ignore[attr-defined]
        self.g.velocity = torch.zeros((0, 2), device=self.device)  # type: ignore[attr-defined]
        self.g.parent = torch.zeros((0,), dtype=torch.int64, device=device)  # type: ignore[attr-defined]
        self.g.phase_offset = torch.zeros((0,), device=device)  # type: ignore[attr-defined]
        self.grid_width: int = max(self.sensory_width, self.workspace_size[0])
        self.grid_height: int = max(self.sensory_height, self.workspace_size[1])

    def _get_unoccupied_grid_positions(self, n: int, exclude_mask: torch.Tensor | None = None, grid_width: int | None = None, grid_height: int | None = None, min_dist: float = 0.4, fast_unique_grid: bool = False) -> tuple[torch.Tensor, bool]:
        width = int(grid_width if grid_width is not None else self.sensory_width)
        height = int(grid_height if grid_height is not None else self.sensory_height)
        # Fallback to fast_unique_grid if the grid is very large
        if fast_unique_grid or width * height > 10000 or n > 1000:
            # Precompute all possible grid cells, shuffle, and assign
            logger.debug(f"Using fast grid generation for {width}x{height} grid, requesting {n} positions")
            all_cells = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).reshape(-1, 2)
            np.random.shuffle(all_cells)
            if n > len(all_cells):
                n = len(all_cells)
                logger.debug(f"Reduced request from {n} to {len(all_cells)} due to grid size limit")
            selected = all_cells[:n].tolist()
            return torch.tensor(selected, dtype=torch.float32, device=self.device), True
        g = self.g if hasattr(self, 'g') else None
        if g is not None and g.pos is not None:
            pos = g.pos.detach().cpu()
            # Only apply exclude_mask if we have positions to exclude from
            if exclude_mask is not None and exclude_mask.numel() > 0 and len(pos) > 0:
                pos = pos[exclude_mask.cpu()]
            occupied: NDArray[np.float32] = pos.detach().cpu().numpy() if len(pos) > 0 else np.zeros((0, 2), dtype=np.float32)  # type: ignore[assignment]

            # occupied variable already defined above - remove duplicate declaration
        else:
            occupied = np.zeros((0, 2), dtype=np.float32)
        occupied_size = int(occupied.size) if hasattr(occupied, 'size') else len(occupied)
        logger.debug(f"Generating grid cells for {width}x{height} grid with {occupied_size} occupied positions")
        all_cells = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).reshape(-1, 2)
        if occupied.size > 0:  # Use size instead of shape[0] to avoid type issues
            # Vectorized distance calculation using broadcasting
            dists = np.linalg.norm(all_cells[:, None, :] - occupied[None, :, :], axis=2)
            min_dists = dists.min(axis=1)
            mask = min_dists >= min_dist
            free_cells = all_cells[mask]
        else:
            free_cells = all_cells
        np.random.shuffle(free_cells)
        if not free_cells.any():
            width = int(width * 1.1) + 1
            height = int(height * 1.1) + 1
            self.grid_width = width
            self.grid_height = height
            return torch.zeros((0,2), dtype=torch.float32, device=self.device), False
        selected_cells: list[list[int]] = []
        if n == 1:
            selected_cells = [free_cells[0]]
        else:
            # Vectorized node selection using spatial partitioning
            if len(free_cells) <= n:
                selected_cells = free_cells.tolist()
            else:
                # Use k-means clustering for efficient spatial distribution if available
                try:
                    from sklearn.cluster import KMeans  # type: ignore[import-untyped,import-not-found]
                    kmeans: Any = KMeans(n_clusters=min(n, len(free_cells)), random_state=42, n_init=1)  # type: ignore[assignment]
                    kmeans.fit(free_cells)  # type: ignore[attr-defined]
                    # Select cluster centers that are sufficiently far apart
                    cluster_centers: NDArray[np.float64] = kmeans.cluster_centers_  # type: ignore[assignment]

                    selected_cells = []
                    selected_arr: NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
                    for center in cluster_centers:  # type: ignore[assignment]
                        center_arr: NDArray[np.float64] = center  # type: ignore[assignment]
                        if len(selected_cells) >= n:
                            break
                        if not selected_arr.size:  # type: ignore[misc]
                            selected_cells.append(center_arr.tolist())  # type: ignore[arg-type]
                            selected_arr = center_arr.reshape(1, -1)  # type: ignore[assignment]
                        else:
                            center_dists: NDArray[np.float64] = np.linalg.norm(selected_arr - center_arr, axis=1)  # type: ignore[arg-type,no-redef]
                            if (center_dists >= min_dist).all():
                                selected_cells.append(center_arr.tolist())  # type: ignore[arg-type]
                                selected_arr = np.vstack([selected_arr, center_arr])  # type: ignore[arg-type]
                except ImportError:
                    # Fallback to original sequential algorithm if sklearn not available
                    selected_cells = []
                    selected_arr = np.empty((0, 2))
                    for cell in free_cells:
                        if len(selected_cells) >= n:
                            break
                        if not selected_arr.size:
                            selected_cells.append(cell.tolist())
                            selected_arr = cell.reshape(1, -1)
                        else:
                            dists = np.linalg.norm(selected_arr - cell, axis=1)
                            if (dists >= min_dist).all():
                                selected_cells.append(cell.tolist())
                                selected_arr = np.vstack([selected_arr, cell])
        if not selected_cells:
            width = int(width * 1.1) + 1
            height = int(height * 1.1) + 1
            self.grid_width = width
            self.grid_height = height
            return torch.tensor(np.array(selected_cells), dtype=torch.float32, device=self.device), False
        return torch.tensor(np.array(selected_cells), dtype=torch.float32, device=self.device), True

    def _init_graph(self) -> None:
        logger.info("Initializing graph with nodes and edges.")
        device = self.device
        n_sensory = self.n_sensory_target
        n_dynamic = self.n_dynamic_target
        n_workspace = self.n_workspace_target
        n_total = n_sensory + n_dynamic + n_workspace
        # Validate N before using in torch.full
        if n_total <= 0:
            raise ValueError(f"Invalid total node count: {n_total}")
        min_width = max(self.sensory_width, self.workspace_size[0])
        width = int(min_width)
        height = int(np.ceil(n_total / float(width)))
        min_dist = 0.4
        # --- Auto-expand grid until all nodes fit ---
        while True:
            pos = torch.zeros(n_total, 2, device=device)
            node_types = torch.zeros(n_total, dtype=torch.int64, device=device)
            n_sensory_actual = 0
            if n_sensory > 0:
                sensory_pos, ok = self._get_unoccupied_grid_positions(n_sensory, grid_width=width, grid_height=height, min_dist=min_dist)
                pos[:sensory_pos.shape[0], :] = sensory_pos
                n_sensory_actual = sensory_pos.shape[0]
                node_types[:n_sensory_actual] = NODE_TYPE_SENSORY
                if not ok:
                    width = int(width * 1.1) + 1
                    height = int(np.ceil(n_total / float(width)))
                    continue
            n_dynamic_actual = 0
            if n_dynamic > 0:
                # Only use exclude_mask if we have sensory nodes to exclude
                if n_sensory_actual > 0:
                    used_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
                    used_mask[:n_sensory_actual] = True
                    dyn_pos, ok = self._get_unoccupied_grid_positions(n_dynamic, exclude_mask=used_mask, grid_width=width, grid_height=height, min_dist=min_dist)
                else:
                    dyn_pos, ok = self._get_unoccupied_grid_positions(n_dynamic, grid_width=width, grid_height=height, min_dist=min_dist)
                pos[n_sensory_actual:n_sensory_actual+dyn_pos.shape[0], :] = dyn_pos
                n_dynamic_actual = dyn_pos.shape[0]
                node_types[n_sensory_actual:n_sensory_actual+n_dynamic_actual] = NODE_TYPE_DYNAMIC
                if not ok:
                    width = int(width * 1.1) + 1
                    height = int(np.ceil(n_total / float(width)))
                    continue
            n_workspace_actual = 0
            if n_workspace > 0:
                used_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
                used_mask[:n_sensory_actual+n_dynamic_actual] = True
                ws_pos, ok = self._get_unoccupied_grid_positions(n_workspace, exclude_mask=used_mask, grid_width=width, grid_height=height, min_dist=min_dist)
                pos[n_sensory_actual+n_dynamic_actual:n_sensory_actual+n_dynamic_actual+ws_pos.shape[0], :] = ws_pos
                n_workspace_actual = ws_pos.shape[0]
                node_types[n_sensory_actual+n_dynamic_actual:n_sensory_actual+n_dynamic_actual+n_workspace_actual] = NODE_TYPE_WORKSPACE
                if not ok:
                    width = int(width * 1.1) + 1
                    height = int(np.ceil(n_total / float(width)))
                    continue
            break
        self.grid_width = width
        self.grid_height = height
        self.n_sensory = n_sensory_actual
        self.n_dynamic = n_dynamic_actual
        self.n_workspace = n_workspace_actual
        self.n_total = self.n_sensory + self.n_dynamic + self.n_workspace
        N = self.n_total
        # Validate N before using in torch.full
        if N <= 0:
            raise ValueError(f"Invalid node count N: {N}")
        # Initialize sensory_true_values - sensory nodes are translation nodes that maintain true pixel values
        # Initialize to zeros (will be set when update_sensory_nodes is called)
        if self.n_sensory > 0:
            self.sensory_true_values = torch.zeros((self.n_sensory, 1), device=device)
        else:
            self.sensory_true_values = None
        # Initialize parent and phase_offset
        parent = torch.full((N,), -1, dtype=torch.int64, device=device)
        phase_offset = torch.zeros(N, device=device)
        # Node energies
        energies = torch.ones(N, 1, device=device)
        # Randomize initial dynamic energies to avoid uniform startup
        if self.n_dynamic > 0:
            dyn_start = self.n_sensory
            dyn_end = self.n_sensory + self.n_dynamic
            dyn_min = NODE_ENERGY_CAP * 0.05
            dyn_max = NODE_ENERGY_CAP * 0.35
            energies[dyn_start:dyn_end, 0] = torch.rand(self.n_dynamic, device=device) * (dyn_max - dyn_min) + dyn_min
        # Initialize workspace nodes with moderate energy (10-30% of cap) so they're visible on canvas
        if self.n_workspace > 0:
            ws_start = self.n_sensory + self.n_dynamic
            ws_end = ws_start + self.n_workspace
            ws_min = NODE_ENERGY_CAP * 0.1
            ws_max = NODE_ENERGY_CAP * 0.3
            energies[ws_start:ws_end, 0] = torch.rand(self.n_workspace, device=device) * (ws_max - ws_min) + ws_min
        # Dynamic node subtypes
        dynamic_subtypes = torch.full((N,), -1, dtype=torch.int64, device=device)
        dynamic_subtypes2 = torch.full((N,), -1, dtype=torch.int64, device=device)
        dynamic_subtypes3 = torch.full((N,), -1, dtype=torch.int64, device=device)
        dynamic_subtype4 = torch.full((N,), -1, dtype=torch.int64, device=device)
        if self.n_dynamic > 0:
            subtypes = torch.randint(0, 3, (self.n_dynamic,), device=device)
            subtypes2 = torch.randint(0, 3, (self.n_dynamic,), device=device)
            subtypes3 = torch.randint(0, 3, (self.n_dynamic,), device=device)
            subtype4 = torch.randint(0, 3, (self.n_dynamic,), device=device)
            dynamic_subtypes[self.n_sensory:self.n_sensory+self.n_dynamic] = subtypes
            dynamic_subtypes2[self.n_sensory:self.n_sensory+self.n_dynamic] = subtypes2
            dynamic_subtypes3[self.n_sensory:self.n_sensory+self.n_dynamic] = subtypes3
            dynamic_subtype4[self.n_sensory:self.n_sensory+self.n_dynamic] = subtype4
        # --- Initial connections (use runtime connection rules) ---
        edge_src: list[torch.Tensor] = []
        edge_dst: list[torch.Tensor] = []
        edge_types: list[torch.Tensor] = []
        edge_weights: list[torch.Tensor] = []
        edge_caps: list[torch.Tensor] = []
        edge_plastic_lrs: list[torch.Tensor] = []
        edge_gate_thresholds: list[torch.Tensor] = []
        edge_conn_subtype2: list[torch.Tensor] = []
        edge_conn_subtype3: list[torch.Tensor] = []
        # Sensory to dynamic with configurable fanout
        num_sens_dyn = self.n_sensory
        fanout = 3
        try:
            cfg = ConfigManager()
            cfg_fan = cfg.get_config('sensory', 'fanout')
            if isinstance(cfg_fan, int) and cfg_fan > 0:
                fanout = cfg_fan
        except Exception as fan_err:  # pylint: disable=broad-exception-caught
            logger.debug("Sensory fanout config missing or invalid: %s", fan_err)

        if self.n_dynamic > 0 and num_sens_dyn > 0:
            src = torch.arange(self.n_sensory, device=device).repeat_interleave(fanout)
            dst = torch.randint(self.n_sensory, self.n_sensory+self.n_dynamic, (num_sens_dyn * fanout,), device=device)
            conn_types = torch.randint(0, 4, (num_sens_dyn * fanout,), device=device)
            weights = torch.where(conn_types == CONN_TYPE_EXCITATORY, torch.rand(num_sens_dyn * fanout, device=device)*0.2+0.1,
                      torch.where(conn_types == CONN_TYPE_INHIBITORY, -torch.rand(num_sens_dyn * fanout, device=device)*0.2-0.1,
                      torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_sens_dyn * fanout, device=device)*0.2+0.1,
                      torch.rand(num_sens_dyn * fanout, device=device)*0.5-0.25)))
            caps = torch.rand(num_sens_dyn * fanout, device=device)*0.9+0.1
            plastic_lrs = torch.where(conn_types == CONN_TYPE_PLASTIC, torch.rand(num_sens_dyn * fanout, device=device)*(PLASTIC_LEARNING_RATE_MAX-PLASTIC_LEARNING_RATE_MIN)+PLASTIC_LEARNING_RATE_MIN, torch.zeros(num_sens_dyn * fanout, device=device))
            gate_thresholds = torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_sens_dyn * fanout, device=device)*0.9+0.1, torch.zeros(num_sens_dyn * fanout, device=device))
            conn_subtype2 = torch.randint(0, 3, (num_sens_dyn * fanout,), device=device)
            conn_subtype3 = torch.randint(0, 3, (num_sens_dyn * fanout,), device=device)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_types.append(conn_types)
            edge_weights.append(weights)
            edge_caps.append(caps)
            edge_plastic_lrs.append(plastic_lrs)
            edge_gate_thresholds.append(gate_thresholds)
            edge_conn_subtype2.append(conn_subtype2)
            edge_conn_subtype3.append(conn_subtype3)
        # Dynamic to workspace (workspace nodes are sinks, they receive but don't send)
        # Note: Workspace nodes should NOT be sources - they are output/sink nodes
        # Create multiple connections per workspace node (4 connections each for better energy flow)
        conns_per_ws = 4
        num_dyn_ws = self.n_workspace * conns_per_ws
        if self.n_dynamic > 0 and num_dyn_ws > 0:
            src = torch.randint(self.n_sensory, self.n_sensory+self.n_dynamic, (num_dyn_ws,), device=device)
            # Repeat each workspace node index conns_per_ws times
            ws_indices = torch.arange(self.n_sensory+self.n_dynamic, self.n_total, device=device)
            dst = ws_indices.repeat_interleave(conns_per_ws)
            conn_types = torch.randint(0, 4, (num_dyn_ws,), device=device)
            weights = torch.where(conn_types == CONN_TYPE_EXCITATORY, torch.rand(num_dyn_ws, device=device)*0.15+0.05,
                      torch.where(conn_types == CONN_TYPE_INHIBITORY, -torch.rand(num_dyn_ws, device=device)*0.15-0.05,
                      torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_dyn_ws, device=device)*0.15+0.05,
                      torch.rand(num_dyn_ws, device=device)*0.4-0.2)))
            caps = torch.rand(num_dyn_ws, device=device)*0.9+0.1
            plastic_lrs = torch.where(conn_types == CONN_TYPE_PLASTIC, torch.rand(num_dyn_ws, device=device)*(PLASTIC_LEARNING_RATE_MAX-PLASTIC_LEARNING_RATE_MIN)+PLASTIC_LEARNING_RATE_MIN, torch.zeros(num_dyn_ws, device=device))
            gate_thresholds = torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_dyn_ws, device=device)*0.9+0.1, torch.zeros(num_dyn_ws, device=device))
            conn_subtype2 = torch.randint(0, 3, (num_dyn_ws,), device=device)
            # For dynamic→workspace: only use OneWayIn (1) or FreeFlow (2), NOT OneWayOut (0)
            # OneWayOut requires sensory source, but source is dynamic here
            conn_subtype3 = torch.randint(1, 3, (num_dyn_ws,), device=device)  # 1=OneWayIn, 2=FreeFlow
            edge_src.append(src)
            edge_dst.append(dst)
            edge_types.append(conn_types)
            edge_weights.append(weights)
            edge_caps.append(caps)
            edge_plastic_lrs.append(plastic_lrs)
            edge_gate_thresholds.append(gate_thresholds)
            edge_conn_subtype2.append(conn_subtype2)
            edge_conn_subtype3.append(conn_subtype3)
        # Dynamic to dynamic (one per dynamic)
        num_dyn_dyn = self.n_dynamic
        if self.n_dynamic > 0 and num_dyn_dyn > 0:
            src = torch.arange(self.n_sensory, self.n_sensory+self.n_dynamic, device=device)
            dst = torch.randint(self.n_sensory, self.n_sensory+self.n_dynamic, (num_dyn_dyn,), device=device)
            mask = src != dst
            src = src[mask]
            dst = dst[mask]
            num_dyn_dyn = len(src)
            conn_types = torch.randint(0, 4, (num_dyn_dyn,), device=device)
            weights = torch.where(conn_types == CONN_TYPE_EXCITATORY, torch.rand(num_dyn_dyn, device=device)*0.15+0.05,
                      torch.where(conn_types == CONN_TYPE_INHIBITORY, -torch.rand(num_dyn_dyn, device=device)*0.15-0.05,
                      torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_dyn_dyn, device=device)*0.15+0.05,
                      torch.rand(num_dyn_dyn, device=device)*0.4-0.2)))
            caps = torch.rand(num_dyn_dyn, device=device)*0.9+0.1
            plastic_lrs = torch.where(conn_types == CONN_TYPE_PLASTIC, torch.rand(num_dyn_dyn, device=device)*(PLASTIC_LEARNING_RATE_MAX-PLASTIC_LEARNING_RATE_MIN)+PLASTIC_LEARNING_RATE_MIN, torch.zeros(num_dyn_dyn, device=device))
            gate_thresholds = torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_dyn_dyn, device=device)*0.9+0.1, torch.zeros(num_dyn_dyn, device=device))
            conn_subtype2 = torch.randint(0, 3, (num_dyn_dyn,), device=device)
            conn_subtype3 = torch.randint(0, 3, (num_dyn_dyn,), device=device)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_types.append(conn_types)
            edge_weights.append(weights)
            edge_caps.append(caps)
            edge_plastic_lrs.append(plastic_lrs)
            edge_gate_thresholds.append(gate_thresholds)
            edge_conn_subtype2.append(conn_subtype2)
            edge_conn_subtype3.append(conn_subtype3)
        # Build PyG graph
        final_src: torch.Tensor = torch.cat(edge_src) if edge_src else torch.tensor([], dtype=torch.int64, device=device)
        final_dst: torch.Tensor = torch.cat(edge_dst) if edge_dst else torch.tensor([], dtype=torch.int64, device=device)

        self.g = Data(
            edge_index=torch.stack([final_src, final_dst]),
            energy=energies[:self.n_total],
            node_type=node_types[:self.n_total],
            pos=pos[:self.n_total],
            dynamic_subtype=dynamic_subtypes[:self.n_total],
            dynamic_subtype2=dynamic_subtypes2[:self.n_total],
            dynamic_subtype3=dynamic_subtypes3[:self.n_total],
            dynamic_subtype4=dynamic_subtype4[:self.n_total],
            max_connections=torch.randint(5, 26, (self.n_total,), dtype=torch.int64, device=device),
            velocity=torch.zeros((self.n_total, 2), device=device),
            parent=parent[:self.n_total],
            phase_offset=phase_offset[:self.n_total],
            num_nodes=self.n_total  # type: ignore[attr-defined]
        )
        self.g.device = self.device  # Add device attribute to graph object
        if edge_weights:
            self.g.weight = torch.cat(edge_weights).unsqueeze(1)
            self.g.energy_transfer_capacity = torch.cat(edge_caps).unsqueeze(1)
            self.g.conn_type = torch.cat(edge_types).unsqueeze(1)
            self.g.plastic_lr = torch.cat(edge_plastic_lrs).unsqueeze(1)
            self.g.gate_threshold = torch.cat(edge_gate_thresholds).unsqueeze(1)
            self.g.conn_subtype2 = torch.cat(edge_conn_subtype2)
            self.g.conn_subtype3 = torch.cat(edge_conn_subtype3)
        self.last_update_time = time.time()

    def summary(self) -> None:
        if self.g is None:
            print("Graph is None")
            return
        print(f"Nodes: {self.g.num_nodes} (sensory: {self.n_sensory}, dynamic: {self.n_dynamic}, workspace: {self.n_workspace})")
        print(f"Edges: {self.g.num_edges}")
        print(f"Node features: {list(self.g.keys())}")
        print(f"Edge features: {list(self.g.keys())}")

    def update(self) -> None:
        """
        Main system update method that orchestrates the neural network simulation cycle.

        This method performs a complete update cycle including:
        1. Memory usage checking and cleanup
        2. System state validation
        3. Node lifecycle management (birth/death)
        4. Energy transfer and node processing
        5. Connection management
        6. Metrics tracking

        The update method is designed to be robust with comprehensive error handling
        and recovery mechanisms to maintain system stability.

        Key Operations:
        - Processes node death queue (removing nodes with insufficient energy)
        - Processes node birth queue (adding new nodes based on energy thresholds)
        - Updates energy levels across all nodes using vectorized operations
        - Applies connection worker results (adding/removing edges)
        - Tracks system metrics and performance statistics

        Error Handling:
        - Each major operation has its own try-catch block for isolated error handling
        - Critical errors trigger the recovery mechanism
        - All errors are logged with detailed context

        Performance:
        - Uses vectorized PyTorch operations for efficient computation
        - Implements batch processing for node/connection operations
        - Includes memory optimization and cleanup

        Example:
        ```python
        # Create neural system
        system = PyGNeuralSystem(sensory_width=32, sensory_height=32, n_dynamic=100)

        # Run simulation loop
        for step in range(1000):
            system.update()
            metrics = system.get_metrics()
            print(f"Step {step}: Energy={metrics['total_energy']:.2f}, Nodes={metrics['dynamic_node_count']}")
        ```
        """
        try:
            # Reset per-step metrics at the start of update cycle
            # This ensures metrics reflect changes during this update step
            # Metrics are read by get_metrics() after update() completes
            self.node_births = 0
            self.node_deaths = 0
            self.conn_births = 0
            self.conn_deaths = 0

            # Check memory usage
            self._check_memory_usage()

            # Validate system state
            if not hasattr(self, 'g') or self.g is None:
                logger.error("Graph object is None at the beginning of update.")
                raise RuntimeError("Graph not initialized")

            # Validate node counts - use actual tensor size as source of truth
            # Check actual node count from energy tensor (most reliable)
            actual_node_count = self.g.energy.shape[0] if self.g.energy is not None else 0
            
            # Synchronize num_nodes and n_total with actual tensor size
            if self.g.num_nodes != actual_node_count:
                if self.g.num_nodes is not None:
                    logger.debug(f"Synchronizing num_nodes: {self.g.num_nodes} -> {actual_node_count}")
                self.g.num_nodes = actual_node_count
            
            if self.n_total != actual_node_count:
                logger.debug(f"Synchronizing n_total: {self.n_total} -> {actual_node_count}")
                self.n_total = actual_node_count

            # Process death queue
            if self.death_queue:
                try:
                    # Convert death_queue from list[int] to Tensor for _remove_nodes
                    death_tensor = torch.tensor(self.death_queue, dtype=torch.bool, device=self.device)
                    self._remove_nodes(death_tensor)
                    self.death_queue.clear()
                except Exception as e:
                    logger.error("Error processing death queue: %s", str(e))

            # Process birth queue
            if self.birth_queue:
                try:
                    for node_type, args in self.birth_queue:
                        self._add_nodes(1, node_type, **args)
                    self.birth_queue.clear()
                except Exception as e:
                    logger.error("Error processing birth queue: %s", str(e))

            # Update node energies - use unified fast path or traditional PyG path
            try:
                if self._use_unified_fast_path and self.vector_engine is not None and not self._skip_vector_engine_step:
                    # UNIFIED FAST PATH: Vector engine speed with PyG rules
                    # This is faster because it uses vectorized batch operations
                    unified_metrics = self.vector_engine_step_unified()
                    if unified_metrics.get('deaths', 0) > 0:
                        self.node_deaths += int(unified_metrics['deaths'])
                    
                    # CRITICAL: Actually remove dead nodes (unified path was missing this!)
                    # Only dynamic nodes can die (sensory/workspace have death_threshold=-1e9)
                    if self.g is not None and hasattr(self.g, 'node_type'):
                        node_types = self.g.node_type
                        dynamic_mask = (node_types == NODE_TYPE_DYNAMIC)
                        dead_nodes = (self.g.energy <= NODE_DEATH_THRESHOLD).squeeze() & dynamic_mask.squeeze()
                        if dead_nodes.any():
                            dead_count = dead_nodes.sum().item()
                            logger.info(f"Removing {dead_count} dead dynamic nodes")
                            self._remove_nodes(dead_nodes)
                    
                    # Also handle spawning in unified path
                    self._handle_energy_based_spawning()
                else:
                    # TRADITIONAL PATH: Original PyG simulation
                    # Use legacy vector engine step if enabled (different rules)
                    self.vector_engine_step(dt=1.0)
                    self._update_energies()
            except Exception as e:
                logger.error("Error updating energies: %s", str(e))

            # Process connection worker results
            try:
                self.apply_connection_worker_results()
            except Exception as e:
                logger.error("Error applying connection worker results: %s", str(e))

            # Update step counter
            self.step_counter += 1

        except Exception as e:
            logger.error("Critical error in update: %s", str(e))
            # Attempt recovery
            self._attempt_recovery()

    def _attempt_recovery(self) -> None:
        """Attempt to recover from a critical error with enhanced recovery strategies"""
        try:
            logger.info("Starting system recovery process...")

            # Step 1: Stop all workers and clear queues
            if self._connection_worker is not None:
                self._connection_worker.stop()
                self._connection_worker = None

            self.death_queue.clear()
            self.birth_queue.clear()
            self.conn_growth_queue.clear()
            self.conn_candidate_queue.clear()

            # Step 2: Diagnose the current state with more targeted recovery
            recovery_success = self._diagnose_and_recover_graph_state()

            if not recovery_success:
                logger.warning("Graph state recovery failed, attempting targeted recovery")
                # Try more targeted recovery before full reinitialization
                if self._synchronize_all_tensors() and self.validate_graph_state():
                    logger.info("Targeted tensor synchronization successful")
                    recovery_success = True
                else:
                    logger.warning("Targeted recovery failed, attempting full reinitialization")
                    # Reinitialize graph if needed
                    if not hasattr(self, 'g') or self.g is None:
                        self._init_empty_graph()
                    else:
                        # Try to salvage existing graph data
                        self._salvage_graph_data()
                        # Final validation after salvage
                        if self.validate_graph_state():
                            recovery_success = True

            # Step 3: Restart connection worker
            try:
                self.start_connection_worker()
            except Exception as e:
                logger.warning(f"Failed to restart connection worker during recovery: {str(e)}")

            # Step 4: Run streamlined recovery validation
            self._validate_post_recovery_state()

            logger.info("System recovery completed successfully")

        except Exception as e:
            logger.error("Recovery failed: %s", str(e))
            # If recovery fails, we should probably stop the system
            self.cleanup()

    def _validate_post_recovery_state(self) -> bool:
        """
        Validate system state after recovery to ensure integrity and consistency.
        This method performs comprehensive checks to verify that recovery was successful.
        """
        try:
            logger.info("Starting post-recovery validation...")

            # 1. Basic system integrity check
            if self.g is None:
                logger.error("Post-recovery validation failed: graph is None")
                return False

            # 2. Validate tensor shapes
            validation_results = self.tensor_manager.validate_tensor_shapes()
            invalid_tensors = [key for key, valid in validation_results.items() if not valid]

            if invalid_tensors:
                logger.error(f"Post-recovery validation failed: invalid tensor shapes {invalid_tensors}")
                return False

            # 3. Check connection integrity
            if not self.tensor_manager.validate_connection_integrity():
                logger.error("Post-recovery validation failed: invalid connection integrity")
                return False

            # 4. Verify energy conservation (basic check)
            if hasattr(self.g, 'energy') and self.g.energy is not None and self.g.num_nodes is not None:
                total_energy = float(self.g.energy.sum().item())
                expected_max_energy = NODE_ENERGY_CAP * float(self.g.num_nodes)
                if total_energy < 0:
                    logger.warning(f"Post-recovery warning: negative total energy {total_energy}")
                elif total_energy > expected_max_energy:
                    logger.warning(f"Post-recovery warning: excessive total energy {total_energy} (max expected: {expected_max_energy})")

            # 5. Check node counts consistency
            if hasattr(self.g, 'num_nodes') and self.g.num_nodes is not None:
                actual_node_count = self.g.energy.shape[0] if hasattr(self.g, 'energy') else 0
                if self.g.num_nodes != actual_node_count:
                    logger.error(f"Post-recovery validation failed: node count mismatch {self.g.num_nodes} vs {actual_node_count}")
                    return False

            logger.info("Post-recovery validation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Post-recovery validation failed: {str(e)}")
            return False
    def _diagnose_and_recover_graph_state(self) -> bool:
        """Diagnose and attempt to recover from graph state issues"""
        try:
            logger.info("Diagnosing graph state issues...")

            # Check if graph exists
            if not hasattr(self, 'g') or self.g is None:
                logger.warning("Graph is None, cannot recover")
                return False

            # Check for tensor shape mismatches
            if not self.validate_graph_state():
                logger.warning("Graph state validation failed during recovery")
                return False

            # Check for common failure patterns
            if self._check_for_common_failures():
                logger.info("Common failure patterns detected and handled")
                return True

            logger.info("Graph state diagnosis completed successfully")
            return True

        except Exception as e:
            logger.error("Graph state diagnosis failed: %s", str(e))
            return False

    def _check_for_common_failures(self) -> bool:
        """Check for and handle common failure patterns"""
        try:
            # Check for tensor shape mismatches
            if hasattr(self, 'g') and self.g is not None:
                num_nodes = self.g.num_nodes
                if hasattr(self.g, 'plastic_lr') and self.g.plastic_lr.shape[0] != num_nodes:
                    logger.warning(f"Detected plastic_lr shape mismatch: {self.g.plastic_lr.shape[0]} vs {num_nodes}")
                    self._resize_tensor(self.g.plastic_lr, num_nodes)
                    return True

                if hasattr(self.g, 'weight') and self.g.weight.shape[0] != num_nodes:
                    logger.warning(f"Detected weight shape mismatch: {self.g.weight.shape[0]} vs {num_nodes}")
                    self._resize_tensor(self.g.weight, num_nodes)
                    return True

            return False

        except Exception as e:
            logger.error("Common failure check failed: %s", str(e))
            return False

    def _salvage_graph_data(self) -> None:
        """Attempt to salvage important graph data before reinitialization"""
        try:
            logger.info("Attempting to salvage graph data...")

            # Save current graph metrics if available
            if hasattr(self, 'g') and self.g is not None:
                # Save node count and other metrics
                saved_node_count = self.g.num_nodes
                saved_edge_count = self.g.num_edges

                logger.info(f"Salvaged graph metrics: {saved_node_count} nodes, {saved_edge_count} edges")

                # Here you could add more sophisticated salvage operations
                # like saving important tensor data, connection patterns, etc.

        except Exception as e:
            logger.error("Graph data salvage failed: %s", str(e))

    def _resize_tensor(self, tensor: torch.Tensor, target_size: int | None, tensor_name: str = "unknown") -> bool:
        """Resize a tensor to match target size with enhanced logging and error handling"""
        try:
            if target_size is None:
                logger.warning(f"Target size is None for tensor '{tensor_name}', cannot resize")
                return False

            if tensor.shape[0] == target_size:
                return True  # Already correct size

            logger.warning(f"Resizing tensor '{tensor_name}' from shape {tensor.shape} to ({target_size}, *{tensor.shape[1:]})")

            # Create new tensor with correct size
            if len(tensor.shape) == 1:
                new_tensor = torch.zeros(target_size, dtype=tensor.dtype, device=tensor.device)
            else:
                new_tensor = torch.zeros((target_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)

            # Copy existing data (up to minimum size)
            min_size = min(tensor.shape[0], target_size)
            if min_size > 0:
                new_tensor[:min_size] = tensor[:min_size]

            # For remaining elements, initialize with reasonable defaults
            if min_size < target_size:
                if len(tensor.shape) == 1:
                    # Use tensor mean for scalar tensors
                    if tensor.numel() > 0:
                        new_tensor[min_size:] = tensor.mean()
                    else:
                        new_tensor[min_size:] = 0.0  # Default for empty tensors
                else:
                    # For multi-dimensional tensors, repeat the last row or use mean
                    if min_size > 0:
                        new_tensor[min_size:] = tensor[min_size-1:min_size].repeat(target_size - min_size, 1)
                    else:
                        new_tensor[min_size:] = 0.0  # Default for empty tensors

            # Replace the tensor using safe assignment
            # Use copy_() if shapes match, otherwise direct assignment
            if tensor.shape == new_tensor.shape:
                tensor.copy_(new_tensor)
            else:
                # For shape changes, we need to replace the tensor entirely
                tensor.data = new_tensor.data

            logger.info(f"Successfully resized tensor '{tensor_name}' from {tensor.shape[0]} to {target_size}")
            return True

        except Exception as e:
            logger.error(f"Tensor resize failed for '{tensor_name}': {str(e)}")
            return False

    def _synchronize_all_tensors(self) -> dict[str, bool]:
        """Advanced tensor synchronization using TensorManager"""
        if self.g is None:
            return {}

        try:
            # Use the advanced TensorManager for comprehensive synchronization
            sync_results = self.tensor_manager.synchronize_all_tensors()

            # Log synchronization results
            successful_syncs = sum(1 for result in sync_results.values() if result)
            total_syncs = len(sync_results)

            if successful_syncs > 0:
                logger.info(f"Tensor synchronization completed: {successful_syncs}/{total_syncs} tensors synchronized")
            else:
                logger.info("All tensors already synchronized")

            return sync_results

        except Exception as e:
            logger.error(f"Tensor synchronization failed: {str(e)}")
            return {"error": False}

    def _rollback_graph_state(self) -> bool:
        """Rollback graph state to last known good configuration"""
        try:
            logger.info("Attempting graph state rollback")

            # Get backup data from global storage
            from project.system.global_storage import GlobalStorage  # type: ignore[import-not-found]
            backup_data = GlobalStorage.retrieve('graph_backup', {})

            if not backup_data:
                logger.warning("No backup data available for rollback")
                return False

            # Restore from backup
            if hasattr(self, 'g') and self.g is not None:
                # Save current state for potential recovery
                current_state: dict[str, Any] = {
                    'num_nodes': getattr(self.g, 'num_nodes', None),
                    'plastic_lr': getattr(self.g, 'plastic_lr', None),
                    'weight': getattr(self.g, 'weight', None),
                    'timestamp': datetime.now().isoformat()
                }

                # Attempt to restore tensors
                if 'plastic_lr' in backup_data and backup_data['plastic_lr'] is not None:
                    self.g.plastic_lr = backup_data['plastic_lr']
                    logger.info("Restored plastic_lr tensor from backup")

                if 'weight' in backup_data and backup_data['weight'] is not None:
                    self.g.weight = backup_data['weight']
                    logger.info("Restored weight tensor from backup")

                # Validate restored state
                if self.validate_graph_state():
                    logger.info("Graph state rollback successful")
                    return True
                else:
                    logger.warning("Graph state validation failed after rollback")
                    # Attempt to restore from current state if available
                    if current_state:
                        self.g.plastic_lr = current_state.get('plastic_lr')
                        self.g.weight = current_state.get('weight')
                        logger.warning("Restored to pre-rollback state due to validation failure")
                    return False

            return False

        except Exception as e:
            logger.error(f"Graph state rollback failed: {str(e)}")
            return False

    def _log_recovery_event(self, event_type: str, details: dict[str, Any] | None = None) -> bool:
        """Log recovery events for monitoring and debugging with comprehensive tensor tracking"""
        try:
            if details is None:
                details = {}

            # Create comprehensive recovery log entry with tensor dimension tracking
            # Safe access to graph attributes
            node_count = getattr(self.g, 'num_nodes', None) if self.g is not None else None
            edge_count = getattr(self.g, 'num_edges', None) if self.g is not None else None
            total_energy = None
            avg_energy = None

            # Add comprehensive tensor dimension tracking
            tensor_dimensions = {}
            if self.g is not None:
                # Track node tensor dimensions
                node_tensor_keys = ['energy', 'node_type', 'pos', 'dynamic_subtype', 'dynamic_subtype2',
                                  'dynamic_subtype3', 'dynamic_subtype4', 'max_connections', 'velocity',
                                  'parent', 'phase_offset']
                for key in node_tensor_keys:
                    if hasattr(self.g, key):
                        tensor = getattr(self.g, key)
                        if isinstance(tensor, torch.Tensor):
                            tensor_dimensions[f'node_{key}'] = list(tensor.shape)

                # Track edge tensor dimensions
                edge_tensor_keys = ['edge_index', 'weight', 'energy_transfer_capacity', 'conn_type',
                                  'plastic_lr', 'gate_threshold', 'conn_subtype2', 'conn_subtype3']
                for key in edge_tensor_keys:
                    if hasattr(self.g, key):
                        tensor = getattr(self.g, key)
                        if isinstance(tensor, torch.Tensor):
                            tensor_dimensions[f'edge_{key}'] = list(tensor.shape)

            if self.g is not None and hasattr(self.g, 'energy') and self.g.energy is not None and self.g.energy.numel() > 0:
                total_energy = float(self.g.energy.sum().item())
                avg_energy = float(self.g.energy.mean().item())

            log_entry: dict[str, Any] = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'details': details,
                'system_state': {
                    'node_count': node_count,
                    'edge_count': edge_count,
                    'total_energy': total_energy,
                    'avg_energy': avg_energy,
                    'tensor_dimensions': tensor_dimensions
                }
            }

            # Log to file and console with enhanced details
            logger.info(f"Recovery event: {event_type} - Nodes: {node_count}, Edges: {edge_count}")
            logger.debug(f"Recovery event details: {log_entry}")

            # Store in global storage for monitoring
            from project.system.global_storage import GlobalStorage
            recovery_logs = GlobalStorage.retrieve('recovery_logs', [])
            recovery_logs.append(log_entry)
            GlobalStorage.store('recovery_logs', recovery_logs)

            return True

        except Exception as e:
            logger.error(f"Failed to log recovery event: {str(e)}")
            return False

    def update_sensory_nodes(self, sensory_input: np.ndarray[Any, Any] | torch.Tensor) -> None:
        """Update sensory nodes with input validation and proper energy scaling.
        
        Sensory nodes are translation nodes that convert external data (pixel values, sound waves, etc.)
        to energy values. They always maintain and push out the true value of their assigned pixel's
        contrast level. They are not modified by energy transfers from other nodes.
        
        Converts vision pixel values (0-255) to energy values (0-NODE_ENERGY_CAP).
        Vision is a set of color values converted to straight energy for the neural system.
        
        Future: Will support multiple sensory types (sound wave arrays, different vision types,
        virtual machines, or any other value-able information sources).
        """
        try:
            if self.g is None:
                return

            # Validate input
            # Type checking is handled by the function signature and runtime
            # if not isinstance(sensory_input, (np.ndarray, torch.Tensor)):
            #     raise TypeError("sensory_input must be numpy array or torch tensor")

            if sensory_input.shape != (self.sensory_height, self.sensory_width):
                raise ValueError(f"sensory_input shape must be ({self.sensory_height}, {self.sensory_width})")

            # Convert to tensor if needed
            if isinstance(sensory_input, np.ndarray):
                sensory_input = torch.tensor(sensory_input, device=self.device, dtype=torch.float32)

            # Get sensory nodes
            sensory_mask = (self.g.node_type == NODE_TYPE_SENSORY)
            if not sensory_mask.numel() or not sensory_mask.any().item():
                return

            # Convert pixel values to energy values.
            # Accept either:
            # - normalized floats in [0, 1] (common from screen capture pipelines), or
            # - raw 8-bit-like values in [0, 255].
            #
            # Also accept RGB input and reduce to grayscale (black/white intensity).
            if sensory_input.ndim >= 3:
                # Reduce last channel (e.g., RGB) to grayscale intensity.
                sensory_input = sensory_input.mean(dim=-1)

            pixel_values = sensory_input.flatten().unsqueeze(1)  # Shape: (num_sensory_nodes, 1)

            # Normalize to 0-255 if input is 0-1.
            if pixel_values.numel() > 0 and float(pixel_values.max().item()) <= 1.0:
                pixel_values = pixel_values * 255.0

            # Optional gain and offset from config to ensure visible energy even on dark inputs
            sensory_gain = 1.0
            sensory_bias = 0.0
            if hasattr(self, 'config_manager'):
                try:
                    gain_cfg = self.config_manager.get_config('sensory', 'energy_gain')  # type: ignore[attr-defined]
                    bias_cfg = self.config_manager.get_config('sensory', 'energy_bias')  # type: ignore[attr-defined]
                    if isinstance(gain_cfg, (int, float)):
                        sensory_gain = float(gain_cfg)
                    if isinstance(bias_cfg, (int, float)):
                        sensory_bias = float(bias_cfg)
                except Exception as gain_error:  # pylint: disable=broad-exception-caught
                    logger.debug("Sensory gain config missing or invalid: %s", gain_error)

            # Scale from pixel range (0-255) to energy range (0-NODE_ENERGY_CAP)
            # Apply gain and bias to combat low-input scenarios
            energy_values = ((pixel_values / 255.0) * NODE_ENERGY_CAP * sensory_gain) + sensory_bias

            # Warmup ramp to avoid an initial spike
            if self.sensory_warmup_frames > 0:
                self._sensory_update_count += 1
                ramp = min(1.0, self._sensory_update_count / float(self.sensory_warmup_frames))
                energy_values = energy_values * ramp
            
            # Clamp to ensure values are within valid energy range
            energy_values = energy_values.clamp(0.0, NODE_ENERGY_CAP)
            
            # Store true values for sensory nodes - these are the "source of truth"
            # Sensory nodes always maintain these values regardless of energy transfers
            self.sensory_true_values = energy_values.clone()
            
            # Update sensory node energies to their true values
            # These energy values will flow through connections to dynamic nodes
            # Sensory nodes push out energy but always maintain their true pixel values
            self.g.energy[sensory_mask] = energy_values
            
            logger.debug(f"Updated {sensory_mask.sum().item()} sensory nodes with energy range [{energy_values.min().item():.2f}, {energy_values.max().item():.2f}]")

        except Exception as e:
            logger.error("Error updating sensory nodes: %s", str(e))
            # Don't raise the exception, just log it and continue

    @staticmethod
    def _validate_node_type(node_type: int) -> None:
        """Validate node type"""
        valid_types = [NODE_TYPE_SENSORY, NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE]
        if node_type not in valid_types:
            raise ValueError(f"Invalid node type: {node_type}. Must be one of {valid_types}")

    @staticmethod
    def _validate_subtype(subtype: int, subtype_slot: int) -> None:
        """Validate node subtype"""
        valid_subtypes = {
            1: [SUBTYPE_TRANSMITTER, SUBTYPE_RESONATOR, SUBTYPE_DAMPENER],
            2: [0, 1, 2],  # SUBTYPE2 values
            3: [SUBTYPE3_1P_5, SUBTYPE3_2P_10, SUBTYPE3_3P_20],
            4: [SUBTYPE4_GAIN_1, SUBTYPE4_GAIN_01, SUBTYPE4_GAIN_001]
        }
        if subtype_slot not in valid_subtypes:
            raise ValueError(f"Invalid subtype slot: {subtype_slot}")
        if subtype not in valid_subtypes[subtype_slot]:
            raise ValueError(f"Invalid subtype {subtype} for slot {subtype_slot}")

    def _synchronize_energy_tensor(self) -> None:
        """Synchronize energy tensor with current node count"""
        if self.g is None or not hasattr(self.g, 'num_nodes') or self.g.num_nodes is None:
            logger.error("Cannot synchronize energy tensor: graph or num_nodes is None")
            return

        current_node_count = self.g.num_nodes
        if current_node_count <= 0:
            logger.error("Invalid node count for synchronization: %s", current_node_count)
            return

        if not hasattr(self.g, 'energy') or self.g.energy is None:
            logger.error("Energy tensor is None, cannot synchronize")
            return

        current_energy_shape = self.g.energy.shape[0]
        if current_energy_shape == current_node_count:
            logger.debug("Energy tensor already synchronized: %d nodes", current_node_count)
            return

        logger.warning("Synchronizing energy tensor from %d to %d nodes", current_energy_shape, current_node_count)

        try:
            # Create new energy tensor with correct size
            new_energy = torch.zeros((current_node_count, 1), device=self.device)

            # Copy existing energy values (up to minimum size)
            min_size = min(current_energy_shape, current_node_count)
            if min_size > 0:
                new_energy[:min_size, :] = self.g.energy[:min_size, :]

            # For new nodes (if any), initialize with reasonable default energy
            if current_node_count > min_size:
                # Initialize new nodes with average energy of existing nodes
                if min_size > 0:
                    avg_energy = self.g.energy[:min_size, :].mean()
                    new_energy[min_size:, :] = avg_energy
                else:
                    # If no existing nodes, use a reasonable default
                    new_energy[min_size:, :] = NODE_ENERGY_CAP * 0.5

            self.g.energy = new_energy
            logger.info("Successfully synchronized energy tensor to %d nodes", current_node_count)

        except Exception as e:
            logger.error("Failed to synchronize energy tensor: %s", str(e))
            # Fallback: create a fresh energy tensor
            self.g.energy = torch.zeros((current_node_count, 1), device=self.device)

    def _update_energies(self) -> None:
        """
        Update energy levels for all nodes using fully vectorized operations.

        This method implements a sophisticated energy transfer algorithm that:
        1. Validates system state and tensor consistency
        2. Calculates energy changes using vectorized operations
        3. Applies connection-based energy transfer with directionality rules
        4. Handles node type-specific energy processing
        5. Manages node lifecycle (death/spawning)
        6. Includes performance monitoring and optimization

        Algorithm Details:
        - Uses scatter_add_ for efficient out-degree calculation
        - Implements fused operations for energy transfer calculations
        - Applies vectorized masking for different node types
        - Includes noise injection for dynamic behavior
        - Handles connection directionality rules (OneWayOut, OneWayIn, FreeFlow)
        - Implements energy-based node spawning and death

        Energy Transfer Rules:
        - Dynamic nodes: Energy decay + noise + connection-based transfer
        - Sensory/Workspace nodes: Energy reset to maintain stability
        - Connection transfer: Weighted by connection strength and directionality
        - Energy caps: Clamped between NODE_DEATH_THRESHOLD and NODE_ENERGY_CAP

        Performance Optimization:
        - Vectorized operations using PyTorch
        - Memory-efficient in-place operations
        - Batch processing for node operations
        - Periodic memory cleanup to prevent fragmentation

        Args:
            None

        Returns:
            None

        Example:
        ```python
        # This method is called automatically during the update cycle
        # The energy update follows these steps:
        # 1. Validate graph state and tensor shapes
        # 2. Calculate out-degrees for all nodes
        # 3. Compute energy changes: decay + noise + transfer
        # 4. Apply connection-based energy transfer with directionality
        # 5. Handle node type-specific processing
        # 6. Remove dead nodes and spawn new ones
        # 7. Log performance metrics
        ```
        """
        start_time = time.time()
        g = self.g
        if g is None:
            logger.debug("Energy update skipped: graph is None")
            return

        # Performance monitoring - start
        performance_metrics = {
            'start_time': start_time,
            'validation_time': 0.0,
            'edge_consistency_time': 0.0,
            'energy_calculation_time': 0.0,
            'node_processing_time': 0.0,
            'total_time': 0.0
        }

        # Continuous validation before energy update with logging
        validation_start = time.time()
        if not self.validate_graph_state():
            logger.error("Invalid graph state detected before energy update")
            # Attempt to synchronize tensors before giving up
            sync_results = self._synchronize_all_tensors()
            if not any(sync_results.values()):
                performance_metrics['total_time'] = time.time() - start_time
                self._log_performance_metrics(performance_metrics)
                return
        performance_metrics['validation_time'] = time.time() - validation_start

        # Ensure edge tensor consistency before energy update
        edge_consistency_start = time.time()
        self._ensure_edge_tensor_consistency()
        performance_metrics['edge_consistency_time'] = time.time() - edge_consistency_start

        # Add continuous validation for edge tensors with detailed logging
        if hasattr(g, 'num_edges') and g.num_edges > 0:
            edge_tensor_keys = ['weight', 'energy_transfer_capacity', 'conn_type', 'plastic_lr', 'gate_threshold', 'conn_subtype2', 'conn_subtype3']
            for key in edge_tensor_keys:
                if hasattr(g, key):
                    tensor = getattr(g, key)
                    if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                        if tensor.shape[0] != g.num_edges:
                            logger.warning(f"Edge tensor {key} shape mismatch during energy update: {tensor.shape[0]} vs {g.num_edges}")
                            self._resize_tensor(tensor, g.num_edges, key)
                            logger.info(f"Resized edge tensor {key} to match num_edges: {g.num_edges}")

        node_types = g.node_type
        dynamic_mask = (node_types == NODE_TYPE_DYNAMIC)

        # Log energy update start with tensor dimensions and performance metrics
        logger.debug(f"Starting energy update - Nodes: {g.num_nodes if g.num_nodes else 'None'}, Edges: {g.num_edges if g.num_edges else 'None'}")
        logger.debug(f"Energy tensor shape: {g.energy.shape if hasattr(g, 'energy') and g.energy is not None else 'None'}")

        # Fully vectorized energy updates with enhanced optimization
        energy_calculation_start = time.time()

        # CRITICAL: Restore sensory nodes to their true pixel values BEFORE energy transfers
        # This prevents any nodes from altering sensory node energy - sensory nodes are overwritten
        # with their pixel-based energy values at the start of each step
        sensory_mask = (node_types == NODE_TYPE_SENSORY)
        if sensory_mask.any() and self.sensory_true_values is not None:
            # Ensure we have the right number of sensory nodes
            num_sensory = sensory_mask.sum().item()
            if self.sensory_true_values.shape[0] == num_sensory:
                # Overwrite sensory nodes with their true pixel values (before any transfers)
                g.energy[sensory_mask] = self.sensory_true_values
                logger.debug(f"Restored {num_sensory} sensory nodes to their true pixel values (before energy transfers)")
            else:
                # Mismatch - handle by preserving current energy for new nodes
                logger.warning(f"Sensory node count mismatch: stored {self.sensory_true_values.shape[0]}, actual {num_sensory}")
                if num_sensory > 0:
                    if self.sensory_true_values.shape[0] < num_sensory:
                        # Get current energy values for the new sensory nodes
                        sensory_indices = torch.where(sensory_mask)[0]
                        existing_count = self.sensory_true_values.shape[0]
                        new_indices = sensory_indices[existing_count:]
                        # Use current energy values for new nodes (preserve their state)
                        new_values = g.energy[new_indices].clone()
                        self.sensory_true_values = torch.cat([self.sensory_true_values, new_values], dim=0)
                        logger.debug(f"Extended sensory_true_values: {self.sensory_true_values.shape[0]} -> {num_sensory}")
                    elif self.sensory_true_values.shape[0] > num_sensory:
                        # Truncate to match
                        self.sensory_true_values = self.sensory_true_values[:num_sensory]
                    # Now overwrite with matched count
                    g.energy[sensory_mask] = self.sensory_true_values
                    logger.debug(f"Adjusted and overwrote {num_sensory} sensory nodes after count mismatch")

        # Initialize energy changes tensor for batched updates
        energy_changes = torch.zeros_like(g.energy, device=self.device)

        if dynamic_mask.sum() > 0 and g.edge_index is not None:
            # OPTIMIZED: Vectorized out-degree calculation with pre-allocation
            num_nodes = g.num_nodes or 0
            num_edges = g.num_edges or 0
            out_deg = torch.zeros(num_nodes, dtype=torch.float, device=self.device)
            out_deg.scatter_add_(0, g.edge_index[0], torch.ones(num_edges, device=self.device))

            # OPTIMIZED: Combined decay and noise calculation with JIT compilation
            decay = out_deg[dynamic_mask] * CONNECTION_MAINTENANCE_COST
            noise = torch.randn_like(decay, device=self.device) * 1e-6
            energy_changes[dynamic_mask] += (noise - decay).unsqueeze(1)

            # OPTIMIZED: Vectorized energy transfer calculation with fused operations
            src, dst = g.edge_index
            weights = g.weight.squeeze() if hasattr(g, 'weight') else torch.ones(num_edges, device=self.device)
            conn_subtype3 = g.conn_subtype3 if hasattr(g, 'conn_subtype3') else torch.zeros_like(weights, dtype=torch.int64, device=self.device)
            parent = g.parent if hasattr(g, 'parent') else torch.full((num_nodes,), -1, dtype=torch.int64, device=self.device)
            
            # Sensory nodes are translation nodes - they push energy out but don't receive energy from connections.
            # If other nodes try to send to sensory nodes, the attempt "burns" energy; we refund half (net loss 50%).
            sensory_mask = (node_types == NODE_TYPE_SENSORY)
            dst_is_sensory = sensory_mask[dst]
            
            # Get connection types for proper energy transfer (excitatory, inhibitory, gated, plastic)
            conn_types = g.conn_type.squeeze() if hasattr(g, 'conn_type') and g.conn_type is not None else torch.zeros(num_edges, dtype=torch.int64, device=self.device)
            gate_thresholds = g.gate_threshold.squeeze() if hasattr(g, 'gate_threshold') and g.gate_threshold is not None else torch.zeros(num_edges, device=self.device)
            source_energies = g.energy[src, 0]  # Get source node energies for gating

            # Determine transfer capacity once (used for real transfers and sensory-attempt burn/refund)
            try:
                cfg = ConfigManager()
                cfg_cap = cfg.get_config('system', 'conn_transfer_capacity')
                if isinstance(cfg_cap, (int, float)):
                    transfer_capacity = float(cfg_cap)
                else:
                    transfer_capacity = 1.0
            except Exception:  # pylint: disable=broad-exception-caught
                transfer_capacity = 1.0

            # OPTIMIZED: Fused connection directionality calculation
            is_parent = (parent[dst] == src)
            base_allowed = (
                (conn_subtype3 == CONN_SUBTYPE3_FREE_FLOW) |
                ((conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_OUT) & is_parent) |
                ((conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_IN) & (~is_parent))
            )

            # Apply connection type filtering: gated connections only work if source energy exceeds threshold
            gated_mask = (conn_types == CONN_TYPE_GATED)
            gate_passed = ~gated_mask | (source_energies >= gate_thresholds)
            base_allowed = base_allowed & gate_passed

            # Prevent workspace nodes from acting as sources (they are sinks/outputs)
            src_is_workspace = (src >= 0) & (src < num_nodes) & (g.node_type[src] == NODE_TYPE_WORKSPACE)
            base_allowed = base_allowed & (~src_is_workspace)

            # Track attempted transfers TO sensory nodes (would have been allowed except destination is sensory)
            attempted_to_sensory = base_allowed & dst_is_sensory

            # Exclude connections TO sensory nodes for actual transfer (sensory energy is overwritten from pixels)
            allowed = base_allowed & (~dst_is_sensory)

            # Apply "attempt burn with 50% refund" for transfers aimed at sensory nodes.
            # This must run even if there are no other (non-sensory) allowed edges.
            if attempted_to_sensory.any():
                attempted_src = src[attempted_to_sensory]
                attempted_weights = weights[attempted_to_sensory]
                attempted_conn_types = conn_types[attempted_to_sensory]
                attempted_src_energies = g.energy[attempted_src, 0]

                attempted_base_transfer = attempted_src_energies * attempted_weights * transfer_capacity

                # Source subtype scaling (destination is sensory; destination scaling not applicable)
                if hasattr(g, 'dynamic_subtype') and g.dynamic_subtype is not None:
                    attempted_src_subtypes = g.dynamic_subtype[attempted_src]
                    src_scale = torch.ones_like(attempted_base_transfer)
                    src_scale[attempted_src_subtypes == SUBTYPE_TRANSMITTER] = 1.2
                    src_scale[attempted_src_subtypes == SUBTYPE_DAMPENER] = 0.6
                    attempted_base_transfer = attempted_base_transfer * src_scale

                attempted_excitatory = (
                    (attempted_conn_types == CONN_TYPE_EXCITATORY) |
                    (attempted_conn_types == CONN_TYPE_GATED) |
                    (attempted_conn_types == CONN_TYPE_PLASTIC)
                )
                if attempted_excitatory.any():
                    attempted_transfers = attempted_base_transfer[attempted_excitatory] * TRANSMISSION_LOSS
                    net_loss = attempted_transfers * 0.5
                    # Directly apply the loss into energy_changes (sources lose energy)
                    energy_changes[:, 0].sub_(torch.zeros(num_nodes, device=self.device).scatter_add_(0, attempted_src[attempted_excitatory], net_loss.abs()))
                    logger.debug(
                        "Sensory-transfer attempts applied: %d edges, net loss %.2f (50%% refund)",
                        int(attempted_excitatory.sum().item()),
                        float(net_loss.sum().item()),
                    )

            # OPTIMIZED: Calculate energy transfer amounts based on connection types and source energies
            if allowed.any():
                src_allowed = src[allowed]
                dst_allowed = dst[allowed]
                weights_allowed = weights[allowed]
                conn_types_allowed = conn_types[allowed]
                source_energies_allowed = source_energies[allowed]
                
                # Calculate transfer amounts: excitatory transfers energy, inhibitory subtracts it
                # Energy transferred is proportional to source energy and connection weight
                base_transfer = source_energies_allowed * weights_allowed * transfer_capacity

                # Dynamic subtype modulation (source/target)
                subtype_src = g.dynamic_subtype[src_allowed] if hasattr(g, 'dynamic_subtype') else None
                subtype_dst = g.dynamic_subtype[dst_allowed] if hasattr(g, 'dynamic_subtype') else None
                if subtype_src is not None:
                    # Transmitter boosts outgoing, resonator modest, dampener reduces
                    src_scale = torch.ones_like(base_transfer)
                    src_scale[subtype_src == SUBTYPE_TRANSMITTER] = 1.2
                    src_scale[subtype_src == SUBTYPE_RESONATOR] = 1.0
                    src_scale[subtype_src == SUBTYPE_DAMPENER] = 0.6
                    base_transfer = base_transfer * src_scale
                if subtype_dst is not None:
                    # Resonator receives more, dampener absorbs less
                    dst_scale = torch.ones_like(base_transfer)
                    dst_scale[subtype_dst == SUBTYPE_RESONATOR] = 1.2
                    dst_scale[subtype_dst == SUBTYPE_DAMPENER] = 0.5
                    base_transfer = base_transfer * dst_scale
                
                # Excitatory connections: positive transfer
                excitatory_mask = (conn_types_allowed == CONN_TYPE_EXCITATORY)
                # Inhibitory connections: negative transfer (subtract energy)
                inhibitory_mask = (conn_types_allowed == CONN_TYPE_INHIBITORY)
                # Gated connections: positive transfer (already filtered by gate threshold)
                gated_mask_allowed = (conn_types_allowed == CONN_TYPE_GATED)
                # Plastic connections: positive transfer (weight may change over time)
                plastic_mask = (conn_types_allowed == CONN_TYPE_PLASTIC)
                
                # Calculate transfer amounts for each connection type
                transfer_amounts = torch.zeros_like(base_transfer)
                transfer_amounts[excitatory_mask | gated_mask_allowed | plastic_mask] = base_transfer[excitatory_mask | gated_mask_allowed | plastic_mask]
                transfer_amounts[inhibitory_mask] = -base_transfer[inhibitory_mask]  # Negative for inhibitory
                
                # Apply transmission loss
                transfer_amounts.mul_(TRANSMISSION_LOSS)
                
                # Transfer energy FROM source nodes TO destination nodes
                # Source nodes lose energy, destination nodes gain energy
                energy_loss = torch.zeros(num_nodes, device=self.device)
                energy_gain = torch.zeros(num_nodes, device=self.device)
                
                # Source nodes lose energy (only for excitatory, gated, plastic - inhibitory doesn't drain source)
                source_loss_mask = excitatory_mask | gated_mask_allowed | plastic_mask
                if source_loss_mask.any():
                    energy_loss.scatter_add_(0, src_allowed[source_loss_mask], transfer_amounts[source_loss_mask].abs())
                
                # Destination nodes gain/lose energy based on connection type
                energy_gain.scatter_add_(0, dst_allowed, transfer_amounts)
                
                # Apply energy changes: sources lose, destinations gain
                energy_changes[:, 0].sub_(energy_loss)  # Sources lose energy (negative = refund)
                energy_changes[:, 0].add_(energy_gain)  # Destinations gain/lose energy (negative for inhibitory)
                
                logger.debug(f"Applied energy transfer: {energy_gain.sum().item():.2f} gained, {energy_loss.sum().item():.2f} lost")
                
                # Update plastic connection weights based on activity (Hebbian-like learning)
                if plastic_mask.any() and hasattr(g, 'plastic_lr') and g.plastic_lr is not None:
                    # Get indices of plastic connections within the allowed set
                    allowed_indices = torch.where(allowed)[0]
                    plastic_allowed_indices = allowed_indices[plastic_mask]
                    
                    plastic_lrs = g.plastic_lr.squeeze()[plastic_allowed_indices]
                    plastic_src = src_allowed[plastic_mask]
                    plastic_dst = dst_allowed[plastic_mask]
                    
                    # Hebbian learning: strengthen connections between active nodes
                    # Weight change = learning_rate * source_energy * destination_energy_change
                    source_e = g.energy[plastic_src, 0]
                    dest_e_change = energy_gain[plastic_dst]
                    weight_delta = plastic_lrs * source_e * dest_e_change.clamp(-1.0, 1.0)
                    
                    # Update weights in-place (clamped to reasonable range)
                    # plastic_allowed_indices are the actual edge indices in the graph
                    g.weight[plastic_allowed_indices, 0] = (g.weight[plastic_allowed_indices, 0] + weight_delta).clamp(0.01, 1.0)

            # OPTIMIZED: Batch processing for connection-based energy gain
            if hasattr(g, 'dynamic_subtype4') and g.dynamic_subtype4 is not None:
                subtype4 = g.dynamic_subtype4
                connection_gains = torch.zeros_like(energy_changes[:, 0])

                # Vectorized connection gain calculation
                if (subtype4 == SUBTYPE4_GAIN_1).any():
                    connection_gains[subtype4 == SUBTYPE4_GAIN_1] += out_deg[subtype4 == SUBTYPE4_GAIN_1] * 1.0
                if (subtype4 == SUBTYPE4_GAIN_01).any():
                    connection_gains[subtype4 == SUBTYPE4_GAIN_01] += out_deg[subtype4 == SUBTYPE4_GAIN_01] * 0.1
                if (subtype4 == SUBTYPE4_GAIN_001).any():
                    connection_gains[subtype4 == SUBTYPE4_GAIN_001] += out_deg[subtype4 == SUBTYPE4_GAIN_001] * 0.01

                energy_changes[:, 0].add_(connection_gains)

        # Vectorized node type processing with enhanced optimization
        node_processing_start = time.time()

        # NOTE: Sensory nodes are translation nodes that always maintain their true pixel values
        # They push energy out through connections but never receive energy from other nodes
        # (This is handled by filtering connections TO sensory nodes in the energy transfer code above)

        # OPTIMIZED: Vectorized energy_in application with existence check and memory optimization
        if hasattr(g, 'energy_in'):
            # Use in-place addition to avoid creating new tensor
            energy_changes.add_(g.energy_in)
            # Count how many nodes received energy_in (non-zero values)
            energy_in_count = (g.energy_in != 0).sum().item()
            del g.energy_in
            logger.debug(f"Applied energy_in to {energy_in_count} nodes")

        # OPTIMIZED: Fused energy update with clamping and memory-efficient operations
        g.energy.add_(energy_changes).clamp_(NODE_DEATH_THRESHOLD, NODE_ENERGY_CAP)

        logger.debug(f"Applied energy caps: min={NODE_DEATH_THRESHOLD}, max={NODE_ENERGY_CAP}")
        
        # CRITICAL: Overwrite sensory nodes with their true pixel values (post-transfer)
        # Sensory nodes always maintain their assigned pixel/contrast values.
        # Any attempted modifications during the step are overwritten.
        sensory_mask = (node_types == NODE_TYPE_SENSORY)
        if sensory_mask.any() and self.sensory_true_values is not None:
            # Ensure we have the right number of sensory nodes
            num_sensory = sensory_mask.sum().item()
            if self.sensory_true_values.shape[0] == num_sensory:
                # Overwrite sensory nodes to their true values (pixel contrast levels)
                g.energy[sensory_mask] = self.sensory_true_values
                logger.debug(f"Overwrote {num_sensory} sensory nodes to their true pixel values (post-transfer)")
            else:
                # Mismatch - sensory nodes may have been added/removed
                # Try to match by index if possible, otherwise reinitialize
                logger.warning(f"Sensory node count mismatch: stored {self.sensory_true_values.shape[0]}, actual {num_sensory}")
                # If we have fewer stored values than actual nodes, pad with current energy values (not zeros)
                # This preserves energy for newly added sensory nodes until update_sensory_nodes is called
                if num_sensory > 0:
                    if self.sensory_true_values.shape[0] < num_sensory:
                        # Get current energy values for the new sensory nodes
                        sensory_indices = torch.where(sensory_mask)[0]
                        existing_count = self.sensory_true_values.shape[0]
                        new_indices = sensory_indices[existing_count:]
                        # Use current energy values for new nodes (preserve their state)
                        new_values = g.energy[new_indices].clone()
                        self.sensory_true_values = torch.cat([self.sensory_true_values, new_values], dim=0)
                        logger.debug(f"Extended sensory_true_values: {self.sensory_true_values.shape[0]} -> {num_sensory} (preserved energy for new nodes)")
                    elif self.sensory_true_values.shape[0] > num_sensory:
                        # Truncate to match
                        self.sensory_true_values = self.sensory_true_values[:num_sensory]
                        logger.debug(f"Truncated sensory_true_values: {self.sensory_true_values.shape[0]} -> {num_sensory}")
                    # Now overwrite with matched count
                    g.energy[sensory_mask] = self.sensory_true_values
                    logger.debug(f"Adjusted and overwrote {num_sensory} sensory nodes after count mismatch (post-transfer)")

        # OPTIMIZED: Vectorized dead node detection with direct removal and batch processing
        # CRITICAL: Exclude sensory and workspace nodes from death removal
        # They have death_threshold=-1e9 (never die) and should be protected
        # Only dynamic nodes can die (they have death_threshold=NODE_DEATH_THRESHOLD)
        dynamic_mask = (node_types == NODE_TYPE_DYNAMIC)
        dead_nodes = (g.energy <= NODE_DEATH_THRESHOLD).squeeze() & dynamic_mask.squeeze()
        if dead_nodes.any():
            dead_count = dead_nodes.sum().item()
            logger.info(f"Identified {dead_count} dead dynamic nodes for removal")
            self._remove_nodes(dead_nodes)

        # OPTIMIZED: Energy-based node spawning with performance improvements
        self._handle_energy_based_spawning()

        # OPTIMIZED: Add memory cleanup after energy updates to prevent fragmentation
        if self.step_counter % 100 == 0:  # Cleanup every 100 steps
            self._optimize_memory_after_updates()

        # Update performance metrics
        performance_metrics['energy_calculation_time'] = time.time() - energy_calculation_start
        performance_metrics['node_processing_time'] = time.time() - node_processing_start
        performance_metrics['total_time'] = time.time() - start_time

        self._log_performance_metrics(performance_metrics)
        logger.debug("Completed optimized vectorized energy update cycle")

    def _optimize_memory_after_updates(self) -> None:
        """Optimize memory usage after energy updates to prevent fragmentation."""
        try:
            # Use TensorManager for memory optimization
            optimization_stats = self.tensor_manager.optimize_tensor_memory()
            if optimization_stats['memory_freed_mb'] > 0:
                logger.info(f"Memory optimization after updates: {optimization_stats['memory_freed_mb']:.2f}MB freed")

            # Force garbage collection to clean up unused tensors
            import gc
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"Garbage collection after updates: {collected} objects collected")

        except Exception as e:
            logger.warning(f"Error during post-update memory optimization: {str(e)}")

    def _log_performance_metrics(self, metrics: dict[str, Any]) -> None:
        """Log performance metrics for energy update operations"""
        try:
            # Add timestamp and other contextual information
            metrics['timestamp'] = datetime.now().isoformat()
            metrics['node_count'] = self.g.num_nodes if self.g and hasattr(self.g, 'num_nodes') else 0
            metrics['edge_count'] = self.g.num_edges if self.g and hasattr(self.g, 'num_edges') else 0

            # Log to console
            logger.info(f"Energy update performance - Total: {metrics['total_time']:.4f}s, "
                       f"Validation: {metrics['validation_time']:.4f}s, "
                       f"Edge consistency: {metrics['edge_consistency_time']:.4f}s, "
                       f"Energy calc: {metrics['energy_calculation_time']:.4f}s, "
                       f"Node processing: {metrics['node_processing_time']:.4f}s")

            # Store in global storage for monitoring
            from project.system.global_storage import GlobalStorage
            performance_logs = GlobalStorage.retrieve('energy_update_performance_logs', [])
            performance_logs.append(metrics)

            # Keep only the last 1000 entries to prevent memory issues
            if len(performance_logs) > 1000:
                performance_logs = performance_logs[-1000:]

            GlobalStorage.store('energy_update_performance_logs', performance_logs)

        except Exception as e:
            logger.error(f"Failed to log performance metrics: {str(e)}")

    def get_metrics(self) -> dict[str, int | float]:
        if self.g is None:
            return {
                'total_energy': 0.0,
                'avg_dynamic_energy': 0.0,
                'sensory_node_count': 0,
                'dynamic_node_count': 0,
                'workspace_node_count': 0,
                'node_births': self.node_births,
                'node_deaths': self.node_deaths,
                'conn_births': self.conn_births,
                'conn_deaths': self.conn_deaths,
                'total_node_births': self.total_node_births,
                'total_node_deaths': self.total_node_deaths,
                'total_conn_births': self.total_conn_births,
                'total_conn_deaths': self.total_conn_deaths,
                'connection_count': 0,
                'sensory_energy_min': 0.0,
                'sensory_energy_max': 0.0,
                'sensory_energy_avg': 0.0,
                'workspace_energy_min': 0.0,
                'workspace_energy_max': 0.0,
                'workspace_energy_avg': 0.0,
                'conns_per_dynamic': 0.0,
                'step_count': self.step_counter,
            }
        g = self.g
        node_types = g.node_type
        n_sensory = int((node_types == NODE_TYPE_SENSORY).sum().cpu().item())
        n_dynamic = int((node_types == NODE_TYPE_DYNAMIC).sum().cpu().item())
        n_workspace = int((node_types == NODE_TYPE_WORKSPACE).sum().cpu().item())
        
        # Dynamic node energy stats
        dynamic_energies = g.energy[node_types == NODE_TYPE_DYNAMIC].cpu().numpy().flatten()
        avg_dynamic_energy = float(dynamic_energies.mean()) if dynamic_energies.size > 0 else 0.0
        
        # Sensory node energy stats
        sensory_energies = g.energy[node_types == NODE_TYPE_SENSORY].cpu().numpy().flatten()
        sensory_min = float(sensory_energies.min()) if sensory_energies.size > 0 else 0.0
        sensory_max = float(sensory_energies.max()) if sensory_energies.size > 0 else 0.0
        sensory_avg = float(sensory_energies.mean()) if sensory_energies.size > 0 else 0.0
        
        # Workspace node energy stats
        workspace_energies = g.energy[node_types == NODE_TYPE_WORKSPACE].cpu().numpy().flatten()
        workspace_min = float(workspace_energies.min()) if workspace_energies.size > 0 else 0.0
        workspace_max = float(workspace_energies.max()) if workspace_energies.size > 0 else 0.0
        workspace_avg = float(workspace_energies.mean()) if workspace_energies.size > 0 else 0.0
        
        total_energy = float(g.energy.sum().cpu().item())
        
        # Calculate connection count
        if g.edge_index is not None and len(g.edge_index.shape) == 2 and g.edge_index.shape[0] == 2:
            connection_count = g.edge_index.shape[1]
        else:
            connection_count = int(g.num_edges) if hasattr(g, 'num_edges') else 0
        
        # Connections per dynamic node
        conns_per_dynamic = connection_count / n_dynamic if n_dynamic > 0 else 0.0
            
        return {
            'total_energy': total_energy,
            'avg_dynamic_energy': avg_dynamic_energy,
            'sensory_node_count': n_sensory,
            'dynamic_node_count': n_dynamic,
            'workspace_node_count': n_workspace,
            'node_births': self.node_births,
            'node_deaths': self.node_deaths,
            'conn_births': self.conn_births,
            'conn_deaths': self.conn_deaths,
            'total_node_births': self.total_node_births,
            'total_node_deaths': self.total_node_deaths,
            'total_conn_births': self.total_conn_births,
            'total_conn_deaths': self.total_conn_deaths,
            'connection_count': connection_count,
            'sensory_energy_min': sensory_min,
            'sensory_energy_max': sensory_max,
            'sensory_energy_avg': sensory_avg,
            'workspace_energy_min': workspace_min,
            'workspace_energy_max': workspace_max,
            'workspace_energy_avg': workspace_avg,
            'conns_per_dynamic': conns_per_dynamic,
            'step_count': self.step_counter,
        }

    def pulse_energy(self, amount: float | None = None, include_sensory: bool = False) -> float:
        """Inject an energy pulse into the system."""
        pulse_amt = amount
        if pulse_amt is None:
            pulse_amt = 10.0
            try:
                cfg = ConfigManager()
                cfg_amt = cfg.get_config('system', 'energy_pulse')
                if isinstance(cfg_amt, (int, float)):
                    pulse_amt = float(cfg_amt)
            except Exception as cfg_error:  # pylint: disable=broad-exception-caught
                logger.debug("Using default pulse amount (config read failed): %s", cfg_error)

        # Update vector store if present
        if self.vector_engine is not None:
            store = self.vector_engine.store
            mask = store.active_mask.clone()
            if not include_sensory:
                mask = mask & (store.class_ids != NODE_TYPE_SENSORY)
            if mask.any():
                store.energy[mask] = (store.energy[mask] + pulse_amt).clamp(NODE_DEATH_THRESHOLD, NODE_ENERGY_CAP)
                store.active_count = int(store.active_mask.sum().item())
                self._sync_vector_store_to_graph()

        # Update PyG graph energies
        if self.g is not None and hasattr(self.g, 'energy') and self.g.energy is not None:
            node_types = self.g.node_type if hasattr(self.g, 'node_type') else None
            if node_types is not None:
                mask = torch.ones_like(node_types, dtype=torch.bool)
                if not include_sensory:
                    mask = node_types != NODE_TYPE_SENSORY
                self.g.energy[mask] = (self.g.energy[mask] + pulse_amt).clamp(NODE_DEATH_THRESHOLD, NODE_ENERGY_CAP)

        logger.info("Energy pulse applied: %+0.2f (include_sensory=%s)", pulse_amt, include_sensory)
        return float(pulse_amt)

    def _guarantee_minimum_connections(self, node_indices: torch.Tensor | None = None, batch_size: int = 500) -> bool:
        """
        For each node in node_indices (or all nodes if None), ensure it has at least one outgoing connection.
        Adds connections in batches of batch_size.
        Returns True if all nodes are satisfied.
        """
        if self.g is None:
            return False
        g = self.g
        device = self.device
        if node_indices is None:
            if g.num_nodes is not None:
                node_indices = torch.arange(g.num_nodes, device=device)
            else:
                return False
        if g.edge_index is not None:
            assert g.num_nodes is not None and g.num_edges is not None
            out_deg = torch.zeros(g.num_nodes, dtype=torch.long, device=device).scatter_add_(0, g.edge_index[0], torch.ones(g.num_edges, dtype=torch.long, device=device))
            no_conn = node_indices[out_deg[node_indices] <= 0]
            if not no_conn.numel():
                return True  # All satisfied
            # For each, add a connection to a valid target (sensory, workspace, or dynamic, not self)
            targets = torch.arange(g.num_nodes, device=device)
            for i in range(0, len(no_conn), batch_size):
                batch = no_conn[i:i+batch_size]
                src_list: list[int] = []
                dst_list: list[int] = []

                # Vectorized approach to find valid targets for the entire batch
                batch_targets = targets.unsqueeze(0).expand(len(batch), -1)
                valid_mask = batch_targets != batch.unsqueeze(1)

                # Get node types for all valid targets
                all_node_types = g.node_type[targets]
                batch_node_types = all_node_types.unsqueeze(0).expand(len(batch), -1)

                # Create preference masks (workspace > dynamic > sensory)
                ws_pref = (batch_node_types == NODE_TYPE_WORKSPACE) & valid_mask
                dyn_pref = (batch_node_types == NODE_TYPE_DYNAMIC) & valid_mask & (~ws_pref.any(dim=1, keepdim=True))
                sens_pref = (batch_node_types == NODE_TYPE_SENSORY) & valid_mask & (~ws_pref.any(dim=1, keepdim=True)) & (~dyn_pref.any(dim=1, keepdim=True))

                # Find first valid target for each source using preference order
                for src_idx, src in enumerate(batch):
                    logger.debug(f"DEBUG: Processing src_idx={src_idx}, src={src}, type={type(src)}")
                    if ws_pref[src_idx].any():
                        dst_idx = int(torch.nonzero(ws_pref[src_idx], as_tuple=False)[0].item())
                    elif dyn_pref[src_idx].any():
                        dst_idx = int(torch.nonzero(dyn_pref[src_idx], as_tuple=False)[0].item())
                    elif sens_pref[src_idx].any():
                        dst_idx = int(torch.nonzero(sens_pref[src_idx], as_tuple=False)[0].item())
                    else:
                        continue
                    logger.debug(f"DEBUG: Adding connection from {src} to {dst_idx}")
                    src_list.append(int(src))  # type: ignore[arg-type, union-attr]
                    dst_list.append(int(targets[dst_idx].item()))  # type: ignore[arg-type, union-attr]
                if src_list:
                    num_new = len(src_list)
                    conn_types = torch.randint(0, 4, (num_new,), device=device)
                    edge_weights = torch.where(conn_types == CONN_TYPE_EXCITATORY, torch.rand(num_new, device=device)*0.15+0.05,
                                  torch.where(conn_types == 1, -torch.rand(num_new, device=device)*0.15-0.05,
                                  torch.where(conn_types == 2, torch.rand(num_new, device=device)*0.15+0.05,
                                  torch.rand(num_new, device=device)*0.4-0.2)))
                    energy_caps = torch.rand(num_new, device=device)*0.9+0.1
                    plastic_lrs = torch.where(conn_types == 3, torch.rand(num_new, device=device)*(0.05-0.001)+0.001, torch.zeros(num_new, device=device))
                    gate_thresholds = torch.where(conn_types == 2, torch.rand(num_new, device=device)*0.9+0.1, torch.zeros(num_new, device=device))
                    new_edges = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
                    if hasattr(g, 'edge_index') and g.edge_index.numel() > 0:
                        g.edge_index = torch.cat([g.edge_index, new_edges], dim=1)  # type: ignore[arg-type]
                    else:
                        g.edge_index = new_edges
                    g.weight = torch.cat([g.weight, edge_weights.unsqueeze(1)])
                    g.energy_transfer_capacity = torch.cat([g.energy_transfer_capacity, energy_caps.unsqueeze(1)])
                    g.conn_type = torch.cat([g.conn_type, conn_types.unsqueeze(1)])
                    g.plastic_lr = torch.cat([g.plastic_lr, plastic_lrs.unsqueeze(1)])
                    g.gate_threshold = torch.cat([g.gate_threshold, gate_thresholds.unsqueeze(1)])

            # Recheck if all are satisfied
            if hasattr(g, 'edge_index') and g.num_edges and g.num_edges > 0:
                out_deg = torch.zeros(g.num_nodes or 0, dtype=torch.long, device=device).scatter_add_(0, g.edge_index[0], torch.ones(g.num_edges or 0, dtype=torch.long, device=device))  # type: ignore[arg-type]
                return bool((out_deg[node_indices] > 0).all().item())
            else:
                return False
        else:
            return False

    def _cull_invalid_connections(self) -> int:
        """
        Remove invalid connections using advanced TensorManager validation.
        Returns the number of edges removed.
        """
        if self.g is None:
            return 0

        try:
            # Use TensorManager for comprehensive connection validation
            # First validate connection integrity
            if not self.tensor_manager.validate_connection_integrity():
                # Repair invalid connections
                repaired_count = self.tensor_manager.repair_invalid_connections()
                logger.info(f"TensorManager repaired {repaired_count} invalid connections")
                return repaired_count

            return 0

        except Exception as e:
            logger.error(f"Error in connection validation: {str(e)}")
            return 0

    def start_connection_worker(self, batch_size: int = 25) -> None:
        """Start the connection worker thread for background connection processing."""
        if self._connection_worker is not None:
            # Worker already exists, check if it's alive
            if self._connection_worker.is_alive():
                logger.info("Connection worker already running")
                return
            else:
                # Worker exists but is dead, clean it up
                logger.info("Connection worker was dead, cleaning up and restarting")
                self._connection_worker = None
        
        try:
            self._connection_worker = ConnectionWorker(self, batch_size=batch_size)
            # Also set connection_worker (without underscore) for backward compatibility
            self.connection_worker = self._connection_worker
            self._pending_edge_adds: list[Any] = []
            self._pending_edge_removes: list[int] = []
            
            # Initialize task queue with some initial tasks to prevent Empty exceptions
            # Queue initial connection growth and cull tasks to seed the worker
            try:
                self._connection_worker.queue_task('grow')
                self._connection_worker.queue_task('cull')
                logger.info("Connection worker started with initial tasks queued")
            except Exception as e:
                logger.error(f"Failed to queue initial tasks for connection worker: {str(e)}")
                # Start worker anyway, it will handle empty queue gracefully
            
            self._connection_worker.start()
        except Exception as e:
            logger.error(f"Failed to start connection worker: {str(e)}")
            self._connection_worker = None
            self.connection_worker = None
            raise

    def stop_connection_worker(self) -> None:
        """Stop the connection worker thread."""
        worker = getattr(self, 'connection_worker', None) or getattr(self, '_connection_worker', None)
        if worker is not None:
            worker.stop()

    def queue_connection_growth(self) -> None:
        """Queue a connection growth task in the connection worker."""
        worker = getattr(self, 'connection_worker', None) or getattr(self, '_connection_worker', None)
        if worker is None or not worker.is_alive():
            logger.warning("Connection worker not available, cannot queue growth task")
            return
        
        try:
            success = worker.queue_task('grow')
            if not success:
                logger.warning("Failed to queue connection growth task (queue may be full)")
        except Exception as e:
            logger.error(f"Error queueing connection growth task: {str(e)}")

    def queue_cull(self) -> None:
        """Queue a connection culling task in the connection worker."""
        worker = getattr(self, 'connection_worker', None) or getattr(self, '_connection_worker', None)
        if worker is None or not worker.is_alive():
            logger.warning("Connection worker not available, cannot queue cull task")
            return
        
        try:
            success = worker.queue_task('cull')
            if not success:
                logger.warning("Failed to queue connection cull task (queue may be full)")
        except Exception as e:
            logger.error(f"Error queueing connection cull task: {str(e)}")

    def prepare_connection_growth_batch(self, batch_size: int) -> list[Any]:
        # Prepare a batch of connection growth (do not modify graph here)
        # Return a list of (src, dst, subtype3) tuples
        if self.g is None or not self.validate_graph_state():
            logger.warning("Graph state invalid during connection growth batch preparation")
            return []

        # CRITICAL: Populate the growth queue if it's empty
        if not self.conn_growth_queue:
            # Generate new connection candidates that respect node type rules
            g = self.g
            device = self.device
            num_nodes = g.num_nodes or 0
            
            if num_nodes < 2:
                return []
            
            node_type = g.node_type
            n_sensory = int((node_type == NODE_TYPE_SENSORY).sum().item())
            n_dynamic = int((node_type == NODE_TYPE_DYNAMIC).sum().item())
            n_workspace = int((node_type == NODE_TYPE_WORKSPACE).sum().item())
            
            if n_dynamic < 1:
                return []
            
            # Generate candidates that respect connection rules
            candidates_to_generate = min(batch_size * 10, 500)
            
            # CRITICAL: Dynamic-to-dynamic connections have NO restrictions and should dominate (90%)
            # Dynamic can connect to sensory OR workspace (but not both) - 5% each
            # Sensory can connect to dynamic - 5%
            for _ in range(candidates_to_generate):
                rand = torch.rand(1).item()
                if rand < 0.90:  # 90%: dynamic -> dynamic (unrestricted!)
                    src = torch.randint(n_sensory, n_sensory + n_dynamic, (1,), device=device).item()
                    dst = torch.randint(n_sensory, n_sensory + n_dynamic, (1,), device=device).item()
                elif rand < 0.93 and n_sensory > 0:  # 3%: sensory -> dynamic
                    src = torch.randint(0, n_sensory, (1,), device=device).item()
                    dst = torch.randint(n_sensory, n_sensory + n_dynamic, (1,), device=device).item()
                elif rand < 0.96 and n_workspace > 0:  # 3%: dynamic -> workspace
                    src = torch.randint(n_sensory, n_sensory + n_dynamic, (1,), device=device).item()
                    dst = torch.randint(n_sensory + n_dynamic, num_nodes, (1,), device=device).item()
                elif n_sensory > 0:  # 4%: dynamic -> sensory (backprop style)
                    src = torch.randint(n_sensory, n_sensory + n_dynamic, (1,), device=device).item()
                    dst = torch.randint(0, n_sensory, (1,), device=device).item()
                else:  # Fallback: dynamic -> dynamic
                    src = torch.randint(n_sensory, n_sensory + n_dynamic, (1,), device=device).item()
                    dst = torch.randint(n_sensory, n_sensory + n_dynamic, (1,), device=device).item()
                
                if src != dst:  # No self-loops
                    # Use FreeFlow (2) most often, as it has fewest restrictions
                    subtype3 = 2 if torch.rand(1).item() < 0.8 else torch.randint(0, 3, (1,), device=device).item()
                    self.conn_growth_queue.append((int(src), int(dst), int(subtype3)))

        if not self.conn_growth_queue:
            return []

        batch = self.conn_growth_queue[:batch_size]
        self.conn_growth_queue = self.conn_growth_queue[batch_size:]
        return batch

    def prepare_cull_batch(self, batch_size: int) -> list[int]:
        # Prepare a batch of edge indices to remove (do not modify graph here)
        if self.g is None or self.g.edge_index is None:
            return []
        # Validate graph state before preparing cull batch
        if not self.validate_graph_state():
            logger.warning("Graph state invalid during cull batch preparation")
            return []

        # Use caching for expensive cull batch preparation
        cache_key = f"cull_batch:{batch_size}:{self.g.num_edges if self.g.num_edges else 0}"
        from project.utils.performance_utils import get_tensor_cache
        cache = get_tensor_cache()
        cached_result = cache.get(cache_key)

        if cached_result is not None:
            logger.debug(f"Using cached cull batch for size {batch_size}")
            return cached_result

        g = self.g
        assert g.num_nodes is not None
        assert g.num_edges is not None
        assert g.edge_index is not None
        src, dst = g.edge_index
        conn_subtype3 = g.conn_subtype3 if hasattr(g, 'conn_subtype3') else torch.zeros(g.num_edges, dtype=torch.int64, device=g.device)
        node_type = g.node_type
        parent = g.parent if hasattr(g, 'parent') else torch.full((g.num_nodes or 0,), -1, dtype=torch.int64, device=g.device)
        is_parent = (parent[dst] == src)
        valid_direction = (
            (conn_subtype3 == CONN_SUBTYPE3_FREE_FLOW) |
            ((conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_OUT) & is_parent) |
            ((conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_IN) & (~is_parent))
        )
        src_type = node_type[src]
        dst_type = node_type[dst]
        dyn_dyn = (src_type == NODE_TYPE_DYNAMIC) & (dst_type == NODE_TYPE_DYNAMIC)
        not_same_type = (src_type != dst_type) | dyn_dyn
        not_sens_to_ws = ~((src_type == NODE_TYPE_SENSORY) & (dst_type == NODE_TYPE_WORKSPACE))
        to_remove_dyn = torch.zeros_like(src, dtype=torch.bool)
        dyn_src_mask = (src_type == NODE_TYPE_DYNAMIC)
        dyn_src_indices = src[dyn_src_mask]
        dyn_dst_types = dst_type[dyn_src_mask]
        if len(dyn_src_indices) > 0:
            unique_dyn = torch.unique(dyn_src_indices)  # type: ignore[assignment,union-attr]
            for dyn in unique_dyn:  # type: ignore[union-attr,operator]
                mask = (dyn_src_indices == dyn)  # type: ignore[operator,union-attr]
                types = dyn_dst_types[mask]  # type: ignore[index,union-attr]
                has_ws = (types == NODE_TYPE_WORKSPACE).any()  # type: ignore[union-attr,operator]
                has_sens = (types == NODE_TYPE_SENSORY).any()  # type: ignore[union-attr,operator]
                if has_ws.any() and has_sens.any():  # type: ignore[union-attr,operator]
                    remove_mask = mask & (dyn_dst_types == NODE_TYPE_SENSORY)  # type: ignore[operator,union-attr]
                    to_remove_dyn[dyn_src_mask.nonzero(as_tuple=True)[0][remove_mask]] = True  # type: ignore[index,union-attr]
        valid = valid_direction & not_same_type & not_sens_to_ws & (~to_remove_dyn)
        to_remove = (~valid).nonzero(as_tuple=True)[0]
        # Only return a batch
        result = to_remove[:batch_size].cpu().tolist()

        # Cache the result for future use
        cache.set(cache_key, result, ttl=10.0)  # Cache for 10 seconds
        return result

    def _ensure_edge_tensor_consistency(self) -> bool:
        """
        Ensure all edge tensors are consistent with the current edge count.
        This method provides comprehensive synchronization before any edge operations.
        Returns True if all tensors are consistent, False otherwise.
        """
        if self.g is None or not hasattr(self.g, 'num_edges'):
            logger.warning("Cannot ensure edge tensor consistency: graph or num_edges is None")
            return False

        try:
            # Get the current edge count
            current_edge_count = self.g.num_edges

            # List of all edge tensor keys that should match the edge count
            edge_tensor_keys = ['weight', 'energy_transfer_capacity', 'conn_type', 'plastic_lr', 'gate_threshold', 'conn_subtype2', 'conn_subtype3']

            # First, check if edge_index exists and is consistent
            if not hasattr(self.g, 'edge_index') or self.g.edge_index is None:
                logger.warning("Edge index is None, cannot ensure tensor consistency")
                return False

            # Check if edge_index shape matches expected edge count
            if self.g.edge_index.shape[1] != current_edge_count:
                logger.warning(f"Edge index shape mismatch: {self.g.edge_index.shape[1]} vs {current_edge_count}")
                # Try to fix edge_index
                if not current_edge_count:
                    self.g.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                else:
                    # This is a critical error - edge_index should always match num_edges
                    logger.error(f"Critical edge index mismatch: {self.g.edge_index.shape[1]} vs {current_edge_count}")
                    return False

            # Use TensorManager for comprehensive edge tensor validation and synchronization
            # First, validate current tensor shapes
            validation_results = self.tensor_manager.validate_tensor_shapes()
            invalid_tensors = [key for key, valid in validation_results.items() if not valid and key in edge_tensor_keys]

            if invalid_tensors:
                logger.info(f"Detected invalid edge tensor shapes: {invalid_tensors}, attempting synchronization")
                # Use TensorManager's advanced synchronization
                sync_results = self.tensor_manager.synchronize_all_tensors()

                # Check if synchronization was successful
                successful_syncs = sum(1 for key, success in sync_results.items() if key in edge_tensor_keys and success)
                logger.debug(f"Synchronization results: {successful_syncs} successful out of {len(invalid_tensors)} invalid tensors")
                if not successful_syncs:  # Fixed Pylint warning: use implicit booleaness
                    logger.error("Failed to synchronize any invalid edge tensors")
                    return False
                elif successful_syncs < len(invalid_tensors):
                    logger.warning(f"Only synchronized {successful_syncs}/{len(invalid_tensors)} invalid edge tensors")
                else:
                    logger.info(f"Successfully synchronized all {successful_syncs} invalid edge tensors")

            # Additional validation: ensure all edge tensors have the same first dimension
            tensor_shapes: list[int] = []
            for key in edge_tensor_keys:
                if hasattr(self.g, key) and getattr(self.g, key) is not None:
                    tensor = getattr(self.g, key)
                    if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                        tensor_shapes.append(int(tensor.shape[0]))

            if tensor_shapes:
                # All tensors should have the same first dimension (edge count)
                if len(set(tensor_shapes)) > 1:
                    logger.warning(f"Edge tensors have inconsistent shapes: {tensor_shapes}")
                    return False
                elif tensor_shapes[0] != current_edge_count:
                    logger.warning(f"All edge tensors have shape {tensor_shapes[0]} but expected {current_edge_count}")
                    return False

            logger.debug(f"All edge tensors are consistent with {current_edge_count} edges")
            return True

        except Exception as e:
            logger.error(f"Error ensuring edge tensor consistency: {str(e)}")
            return False

    def apply_connection_worker_results(self) -> None:
        """
        Apply connection worker results in a thread-safe manner.
        
        This method processes connection growth and culling batches from the
        connection worker thread and applies them to the graph. It includes
        comprehensive synchronization and error handling to prevent race conditions.
        """
        # Call this in the main thread (main loop) to apply edge adds/removes
        # Use both _connection_worker and connection_worker for compatibility
        worker = getattr(self, 'connection_worker', None) or getattr(self, '_connection_worker', None)
        if worker is None or self.g is None or self.g.num_nodes is None:
            return
        
        # Acquire graph lock for thread-safe operations
        with self._graph_lock:
            # Signal that graph modification is in progress
            self.graph_modification_lock.clear()
            
            try:
                # Validate graph state before processing
                if self.g.num_nodes != self.g.energy.shape[0]:
                    logger.warning("Graph node count mismatch in connection worker: %d vs %d", 
                                 self.g.num_nodes, self.g.energy.shape[0])
                    return

                # Ensure all edge tensors are properly synchronized before processing
                # Use the advanced TensorManager for comprehensive synchronization
                # First, ensure edge tensor consistency using the advanced TensorManager
                edge_consistency_result = self._ensure_edge_tensor_consistency()

                if not edge_consistency_result:
                    logger.warning("Edge tensor consistency check failed before processing connection worker results")
                    # Attempt comprehensive synchronization using TensorManager
                    sync_results = self.tensor_manager.synchronize_all_tensors()
                    successful_syncs = sum(1 for result in sync_results.values() if result)
                    logger.debug(f"Connection worker sync results: {successful_syncs} successful synchronizations")
                    if not successful_syncs:  # Fixed Pylint warning: use implicit booleaness
                        logger.error("Failed to synchronize any edge tensors, aborting connection worker processing")
                        return

                # Calculate current edge count from edge_index (PyG doesn't have num_edges attribute)
                current_edge_count = self.g.edge_index.shape[1] if (hasattr(self.g, 'edge_index') and self.g.edge_index is not None) else 0
                
                # Additional validation: ensure all edge tensors are consistent
                if current_edge_count > 0:
                    edge_tensor_keys = ['weight', 'energy_transfer_capacity', 'conn_type', 'plastic_lr', 'gate_threshold', 'conn_subtype2', 'conn_subtype3']
                    for key in edge_tensor_keys:
                        if hasattr(self.g, key) and getattr(self.g, key) is not None:
                            tensor = getattr(self.g, key)
                            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                                if tensor.shape[0] != current_edge_count:
                                    logger.warning(f"Edge tensor {key} shape mismatch detected before processing: {tensor.shape[0]} vs {current_edge_count}")
                                    # Attempt to synchronize the tensor using the advanced TensorManager
                                    try:
                                        success = self.tensor_manager._intelligent_resize_tensor(tensor, current_edge_count, key, 'edge')  # type: ignore[attr-defined]
                                        if success:
                                            logger.info(f"TensorManager synchronized edge tensor {key} to {current_edge_count} edges")
                                        else:
                                            logger.error(f"TensorManager failed to synchronize edge tensor {key}")
                                            return
                                    except Exception as e:
                                        logger.error(f"Failed to synchronize edge tensor {key}: {str(e)}")
                                        return
                
                result_count = 0
                while not worker.result_queue.empty():
                    result = worker.result_queue.get()
                    result_count += 1
                    if result['type'] == 'grow':
                        # Add edges in main thread, using connection rules
                        g = self.g
                        device = self.device
                        to_add = result['batch']
                        if not to_add:
                            logger.debug(f"Connection worker result {result_count}: empty batch")
                            continue
                        # Validate node indices before processing
                        src_list, dst_list, subtype3_list = zip(*to_add)
                        if max(src_list) >= g.num_nodes or max(dst_list) >= g.num_nodes:
                            logger.warning("Invalid node indices in connection growth batch")
                            continue
                        valid_src: list[int] = []
                        valid_dst: list[int] = []
                        valid_subtype3: list[int] = []
                        node_type = g.node_type
                        parent = g.parent if hasattr(g, 'parent') else torch.full((g.num_nodes or 0,), -1, dtype=torch.int64, device=g.device)
                        for src, dst, subtype3 in zip(src_list, dst_list, subtype3_list):
                            s_type = int(node_type[src].item())
                            d_type = int(node_type[dst].item())
                            # Prevent workspace nodes from being sources (they are sinks/outputs)
                            if s_type == NODE_TYPE_WORKSPACE:
                                continue
                            # No sensory->sensory, workspace->workspace, sensory->workspace
                            if (s_type == NODE_TYPE_SENSORY and d_type == NODE_TYPE_SENSORY) or \
                               (s_type == NODE_TYPE_WORKSPACE and d_type == NODE_TYPE_WORKSPACE) or \
                               (s_type == NODE_TYPE_SENSORY and d_type == NODE_TYPE_WORKSPACE):
                                continue
                            # Dynamic node connection rules:
                            # - Dynamic -> Dynamic: ALWAYS ALLOWED (no restrictions)
                            # - Dynamic -> Workspace: only if node doesn't already connect to Sensory
                            # - Dynamic -> Sensory: only if node doesn't already connect to Workspace
                            if s_type == NODE_TYPE_DYNAMIC and d_type != NODE_TYPE_DYNAMIC:
                                # Only check restrictions for non-dynamic destinations
                                if g.edge_index is not None:  # type: ignore[comparison-overlap]
                                    src_mask = g.edge_index[0] == src
                                    out_edges = g.edge_index[1][src_mask]
                                    out_types = node_type[out_edges] if len(out_edges) > 0 else torch.tensor([], dtype=torch.int64, device=device)
                                    has_ws = bool((out_types == NODE_TYPE_WORKSPACE).any().item())
                                    has_sens = bool((out_types == NODE_TYPE_SENSORY).any().item())
                                    if d_type == NODE_TYPE_WORKSPACE and has_sens:
                                        continue  # Can't connect to workspace if already has sensory
                                    if d_type == NODE_TYPE_SENSORY and has_ws:
                                        continue  # Can't connect to sensory if already has workspace
                            # Directionality rules for conn_subtype3
                            if (
                                subtype3 == CONN_SUBTYPE3_ONE_WAY_OUT
                                and parent[dst] != src
                            ):
                                continue
                            if (
                                subtype3 == CONN_SUBTYPE3_ONE_WAY_IN
                                and parent[dst] == src
                            ):
                                continue
                            valid_src.append(src)
                            valid_dst.append(dst)
                            valid_subtype3.append(subtype3)
                        if not valid_src:
                            continue
                        num_new = len(valid_src)
                        conn_types = torch.randint(0, 4, (num_new,), device=device)
                        edge_weights = torch.where(conn_types == CONN_TYPE_EXCITATORY, torch.rand(num_new, device=device)*0.15+0.05,
                                      torch.where(conn_types == CONN_TYPE_INHIBITORY, -torch.rand(num_new, device=device)*0.15-0.05,
                                      torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_new, device=device)*0.15+0.05,
                                      torch.rand(num_new, device=device)*0.4-0.2)))
                        energy_caps = torch.rand(num_new, device=device)*0.9+0.1
                        plastic_lrs = torch.where(conn_types == CONN_TYPE_PLASTIC, torch.rand(num_new, device=device)*(PLASTIC_LEARNING_RATE_MAX-PLASTIC_LEARNING_RATE_MIN)+PLASTIC_LEARNING_RATE_MIN, torch.zeros(num_new, device=device))
                        gate_thresholds = torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_new, device=device)*0.9+0.1, torch.zeros(num_new, device=device))
                        # Add edges in PyG format
                        new_edges = torch.tensor([valid_src, valid_dst], dtype=torch.long, device=device)

                        # FIX: Remove premature synchronization that causes the tensor shape mismatch
                        # Instead of synchronizing to the OLD edge count, we'll synchronize AFTER adding new edges
                        # This ensures all edge tensors match the NEW edge count

                        # DIAGNOSTIC: Log current edge count before adding new edges
                        current_edge_count = g.edge_index.shape[1] if g.edge_index is not None else 0
                        logger.debug(f"DIAGNOSTIC: Current edge count before adding {num_new} new edges: {current_edge_count}")
                        logger.info(f"FIX APPLIED: Skipping premature edge tensor synchronization to avoid shape mismatch")

                        # Now safely add new edges
                        if g.edge_index is not None:
                            g.edge_index = torch.cat([g.edge_index, new_edges], dim=1)
                        else:
                            g.edge_index = new_edges

                        # DIAGNOSTIC: Log edge count after adding new edges
                        new_edge_count = g.edge_index.shape[1] if hasattr(g, 'edge_index') else 0
                        logger.debug(f"DIAGNOSTIC: Edge count after adding {num_new} new edges: {new_edge_count}")
                        logger.debug(f"DIAGNOSTIC: Expected edge count should be: {current_edge_count + num_new}")

                        # DIAGNOSTIC: Check if this creates a mismatch
                        if new_edge_count != current_edge_count + num_new:
                            logger.warning(f"DIAGNOSTIC: Edge count mismatch after adding edges! Expected {current_edge_count + num_new}, got {new_edge_count}")

                        # Initialize edge attributes if None
                        if not hasattr(g, 'weight') or g.weight is None:
                            g.weight = torch.empty((0, 1), device=device)
                        if not hasattr(g, 'energy_transfer_capacity') or g.energy_transfer_capacity is None:
                            g.energy_transfer_capacity = torch.empty((0, 1), device=device)
                        if not hasattr(g, 'conn_type') or g.conn_type is None:
                            g.conn_type = torch.empty((0, 1), dtype=torch.int64, device=device)
                        if not hasattr(g, 'plastic_lr') or g.plastic_lr is None:
                            g.plastic_lr = torch.empty((0, 1), device=device)
                        if not hasattr(g, 'gate_threshold') or g.gate_threshold is None:
                            g.gate_threshold = torch.empty((0, 1), device=device)
                        if not hasattr(g, 'conn_subtype3') or g.conn_subtype3 is None:
                            g.conn_subtype3 = torch.empty((0,), dtype=torch.int64, device=device)

                        # Add new edge attributes
                        g.weight = torch.cat([g.weight, edge_weights.unsqueeze(1)])
                        g.energy_transfer_capacity = torch.cat([g.energy_transfer_capacity, energy_caps.unsqueeze(1)])
                        g.conn_type = torch.cat([g.conn_type, conn_types.unsqueeze(1)])
                        g.plastic_lr = torch.cat([g.plastic_lr, plastic_lrs.unsqueeze(1)])
                        g.gate_threshold = torch.cat([g.gate_threshold, gate_thresholds.unsqueeze(1)])
                        g.conn_subtype3 = torch.cat([g.conn_subtype3, torch.tensor(valid_subtype3, device=device, dtype=torch.int64)])
                        self.conn_births += num_new
                        self.total_conn_births += num_new

                        # Define edge tensor keys for synchronization
                        edge_tensor_keys = ['weight', 'energy_transfer_capacity', 'conn_type', 'plastic_lr', 'gate_threshold', 'conn_subtype2', 'conn_subtype3']

                        # FIX: Now synchronize edge tensors AFTER adding new edges to ensure they match the NEW edge count
                        final_edge_count = g.edge_index.shape[1] if hasattr(g, 'edge_index') else 0
                        logger.info(f"FIX APPLIED: Synchronizing edge tensors to match new edge count: {final_edge_count}")

                        # Check and synchronize each edge tensor
                        for key in edge_tensor_keys:
                            if hasattr(g, key) and getattr(g, key) is not None:
                                tensor = getattr(g, key)
                                if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                                    tensor_size = tensor.shape[0]
                                    if tensor_size != final_edge_count:
                                        logger.info(f"FIX APPLIED: Resizing edge tensor {key} from {tensor_size} to {final_edge_count} to match new edge count")
                                        # Resize the tensor to match the new edge count
                                        if len(tensor.shape) == 1:
                                            new_tensor = torch.zeros(final_edge_count, dtype=tensor.dtype, device=tensor.device)
                                        else:
                                            new_tensor = torch.zeros((final_edge_count,) + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
                                        min_size = min(tensor.shape[0], final_edge_count)
                                        if min_size > 0:
                                            new_tensor[:min_size] = tensor[:min_size]
                                        setattr(g, key, new_tensor)
                                    else:
                                        logger.debug(f"Edge tensor {key} already has correct size: {tensor_size}")

                        # Ensure node count synchronization after edge additions
                        if g.num_nodes is not None and g.energy is not None:
                            actual_node_count = g.energy.shape[0]
                            if g.num_nodes != actual_node_count:
                                logger.warning(f"Synchronizing node count: {g.num_nodes} -> {actual_node_count}")
                                g.num_nodes = actual_node_count
                                self.n_total = actual_node_count
                    elif result['type'] == 'cull':
                        # Remove edges in main thread
                        g = self.g
                        if g.edge_index is None:
                            continue
                        eids = torch.tensor(result['batch'], dtype=torch.long, device=g.device)
                        # Filter out invalid eids
                        current_edge_count_before = g.edge_index.shape[1]
                        valid_eids = eids[(eids >= 0) & (eids < current_edge_count_before)]
                        invalid_eids = eids[(eids < 0) | (eids >= current_edge_count_before)]
                        if len(invalid_eids) > 0:
                            invalid_ids_list = list(invalid_eids.cpu().tolist()) if hasattr(invalid_eids, 'cpu') else list(invalid_eids)  # type: ignore[arg-type, call-overload]
                            print(f"[WARNING] Tried to remove invalid edge IDs: {invalid_ids_list}")
                        if len(valid_eids) > 0:
                            # Get current edge count from edge_index (most accurate)
                            current_edge_count = g.edge_index.shape[1]
                            
                            # Validate edge indices before removal
                            if valid_eids.max() >= current_edge_count:
                                logger.warning("Invalid edge ID in cull batch: max %d >= %d", valid_eids.max(), current_edge_count)
                                continue
                            
                            # Remove edges in PyG format
                            mask = torch.ones(current_edge_count, dtype=torch.bool, device=g.device)
                            mask[valid_eids] = False
                            g.edge_index = g.edge_index[:, mask]
                            
                            # Update all edge attribute tensors using the mask
                            # Use edge_index.shape[1] as the reference for edge count
                            for key in g.keys():
                                if hasattr(g, key) and g[key] is not None:
                                    attr = g[key]
                                    if isinstance(attr, torch.Tensor) and attr.dim() > 0:
                                        # Check if this is an edge tensor (first dimension matches edge count)
                                        if attr.shape[0] == current_edge_count:
                                            g[key] = attr[mask]
                            
                            # Note: num_edges is a property in PyG Data class, it's automatically computed from edge_index
                            # So we don't need to update it manually
                            new_edge_count = g.edge_index.shape[1]
                            
                            logger.info(f"FIX APPLIED: Successfully removed {len(valid_eids)} edges, new edge count: {new_edge_count}")
                    worker.result_queue.task_done()
            
            finally:
                # Always signal that graph modification is complete
                self.graph_modification_lock.set()

    def _add_nodes(self, n: int, node_type: int, parent_idx: list[int] | None = None) -> None:
        if self.g is None:
            return
        g = self.g
        device = self.device
        if not n:
            return
        if parent_idx is not None:
            parent_arr = torch.tensor(parent_idx, dtype=torch.int64, device=device)
            if len(parent_arr) < n:
                parent_arr = torch.cat([parent_arr, torch.full((n - len(parent_arr),), -1, dtype=torch.int64, device=device)])
        else:
            parent_arr = torch.full((n,), -1, dtype=torch.int64, device=device)

        # For dynamic nodes, give 1% chance to spawn as highway
        if node_type == NODE_TYPE_DYNAMIC:
            # Randomly select 1% of nodes to be highway
            highway_mask = torch.rand(n, device=device) < 0.01
            n_highway = int(highway_mask.sum().item())
            n_dynamic = n - n_highway

            # Add dynamic nodes
            if n_dynamic > 0:
                dynamic_indices = torch.where(~highway_mask)[0]
                subtypes = torch.randint(0, 3, (n_dynamic,), device=device)
                subtypes2 = torch.randint(0, 3, (n_dynamic,), device=device)
                subtypes3 = torch.randint(0, 3, (n_dynamic,), device=device)
                subtype4 = torch.randint(0, 3, (n_dynamic,), device=device)
                pos_new = torch.rand(n_dynamic, 2, device=device) * 100
                max_conns = torch.randint(1, 6, (n_dynamic,), device=device)
                phase_offset = torch.rand(n_dynamic, device=device) * 2 * np.pi
                # Add dynamic nodes in PyG format
                if g.energy is not None and g.node_type is not None and g.pos is not None and g.velocity is not None and g.dynamic_subtype is not None and g.dynamic_subtype2 is not None and g.dynamic_subtype3 is not None and g.dynamic_subtype4 is not None and g.max_connections is not None and g.phase_offset is not None and g.parent is not None:
                    g.energy = torch.cat([g.energy, torch.zeros(n_dynamic, 1, device=device)])
                    g.node_type = torch.cat([g.node_type, torch.full((n_dynamic,), NODE_TYPE_DYNAMIC, dtype=torch.int64, device=device)])
                    g.pos = torch.cat([g.pos, pos_new])
                    g.velocity = torch.cat([g.velocity, torch.zeros(n_dynamic, 2, device=device)])
                    g.dynamic_subtype = torch.cat([g.dynamic_subtype, subtypes])
                    g.dynamic_subtype2 = torch.cat([g.dynamic_subtype2, subtypes2])
                    g.dynamic_subtype3 = torch.cat([g.dynamic_subtype3, subtypes3])
                    g.dynamic_subtype4 = torch.cat([g.dynamic_subtype4, subtype4])
                    g.max_connections = torch.cat([g.max_connections, max_conns])
                    g.phase_offset = torch.cat([g.phase_offset, phase_offset])
                    g.parent = torch.cat([g.parent, parent_arr[dynamic_indices]])

            # Add highway nodes
            if n_highway > 0:
                highway_indices = torch.where(highway_mask)[0]
                pos_new = torch.rand(n_highway, 2, device=device) * 100
                # Add highway nodes in PyG format
                # Highway nodes need to be added to ALL node tensors to maintain consistency
                if g.energy is not None and g.node_type is not None and g.pos is not None and g.velocity is not None and g.max_connections is not None and g.parent is not None:
                    g.energy = torch.cat([g.energy, torch.zeros(n_highway, 1, device=device)])
                    g.node_type = torch.cat([g.node_type, torch.full((n_highway,), NODE_TYPE_HIGHWAY, dtype=torch.int64, device=device)])
                    g.pos = torch.cat([g.pos, pos_new])
                    g.velocity = torch.cat([g.velocity, torch.zeros(n_highway, 2, device=device)])
                    g.max_connections = torch.cat([g.max_connections, torch.full((n_highway,), 1000, dtype=torch.int64, device=device)])
                    g.parent = torch.cat([g.parent, parent_arr[highway_indices]])
                    # Add highway nodes to dynamic subtype tensors with default values (0) to maintain tensor consistency
                    if g.dynamic_subtype is not None:
                        g.dynamic_subtype = torch.cat([g.dynamic_subtype, torch.zeros(n_highway, dtype=torch.int64, device=device)])
                    if g.dynamic_subtype2 is not None:
                        g.dynamic_subtype2 = torch.cat([g.dynamic_subtype2, torch.zeros(n_highway, dtype=torch.int64, device=device)])
                    if g.dynamic_subtype3 is not None:
                        g.dynamic_subtype3 = torch.cat([g.dynamic_subtype3, torch.zeros(n_highway, dtype=torch.int64, device=device)])
                    if g.dynamic_subtype4 is not None:
                        g.dynamic_subtype4 = torch.cat([g.dynamic_subtype4, torch.zeros(n_highway, dtype=torch.int64, device=device)])
                    if g.phase_offset is not None:
                        g.phase_offset = torch.cat([g.phase_offset, torch.zeros(n_highway, device=device)])
        else:
            # Original node type handling
            pos_new = torch.rand(n, 2, device=device) * 100
            if node_type == NODE_TYPE_DYNAMIC:
                subtypes = torch.randint(0, 3, (n,), device=device)
                subtypes2 = torch.randint(0, 3, (n,), device=device)
                subtypes3 = torch.randint(0, 3, (n,), device=device)
                subtype4 = torch.randint(0, 3, (n,), device=device)
                max_conns = torch.randint(1, 6, (n,), device=device)
                phase_offset = torch.rand(n, device=device) * 2 * np.pi
                # Add nodes in PyG format
                if g.energy is not None and g.node_type is not None and g.pos is not None and g.velocity is not None and g.dynamic_subtype is not None and g.dynamic_subtype2 is not None and g.dynamic_subtype3 is not None and g.dynamic_subtype4 is not None and g.max_connections is not None and g.phase_offset is not None and g.parent is not None:
                    g.energy = torch.cat([g.energy, torch.zeros(n, 1, device=device)])
                    g.node_type = torch.cat([g.node_type, torch.full((n,), node_type, dtype=torch.int64, device=device)])
                    g.pos = torch.cat([g.pos, pos_new])
                    g.velocity = torch.cat([g.velocity, torch.zeros(n, 2, device=device)])
                    g.dynamic_subtype = torch.cat([g.dynamic_subtype, subtypes])
                    g.dynamic_subtype2 = torch.cat([g.dynamic_subtype2, subtypes2])
                    g.dynamic_subtype3 = torch.cat([g.dynamic_subtype3, subtypes3])
                    g.dynamic_subtype4 = torch.cat([g.dynamic_subtype4, subtype4])
                    g.max_connections = torch.cat([g.max_connections, max_conns])
                    g.phase_offset = torch.cat([g.phase_offset, phase_offset])
                    g.parent = torch.cat([g.parent, parent_arr])
            else:
                # Add nodes in PyG format (for non-dynamic node types like sensory/workspace)
                # CRITICAL: All node types must have consistent tensor shapes, including subtype tensors
                if g.energy is not None and g.node_type is not None and g.pos is not None and g.velocity is not None and g.parent is not None:
                    g.energy = torch.cat([g.energy, torch.zeros(n, 1, device=device)])
                    g.node_type = torch.cat([g.node_type, torch.full((n,), node_type, dtype=torch.int64, device=device)])
                    g.pos = torch.cat([g.pos, pos_new])
                    g.velocity = torch.cat([g.velocity, torch.zeros(n, 2, device=device)])
                    g.parent = torch.cat([g.parent, parent_arr])
                    # Add subtype tensors with default values (-1) for non-dynamic nodes to maintain tensor consistency
                    if g.dynamic_subtype is not None:
                        g.dynamic_subtype = torch.cat([g.dynamic_subtype, torch.full((n,), -1, dtype=torch.int64, device=device)])
                    if g.dynamic_subtype2 is not None:
                        g.dynamic_subtype2 = torch.cat([g.dynamic_subtype2, torch.full((n,), -1, dtype=torch.int64, device=device)])
                    if g.dynamic_subtype3 is not None:
                        g.dynamic_subtype3 = torch.cat([g.dynamic_subtype3, torch.full((n,), -1, dtype=torch.int64, device=device)])
                    if g.dynamic_subtype4 is not None:
                        g.dynamic_subtype4 = torch.cat([g.dynamic_subtype4, torch.full((n,), -1, dtype=torch.int64, device=device)])
                    if g.max_connections is not None:
                        g.max_connections = torch.cat([g.max_connections, torch.full((n,), 5, dtype=torch.int64, device=device)])
                    if g.phase_offset is not None:
                        g.phase_offset = torch.cat([g.phase_offset, torch.zeros(n, device=device)])

        # CRITICAL FIX: Update num_nodes and n_total to match actual tensor size
        # This prevents tensor synchronization from removing newly spawned nodes
        if g.energy is not None:
            actual_node_count = g.energy.shape[0]
            if g.num_nodes != actual_node_count:
                logger.debug(f"Updating num_nodes from {g.num_nodes} to {actual_node_count} after adding nodes")
                g.num_nodes = actual_node_count
                self.n_total = actual_node_count

    def _event_driven_transfer(self) -> None:
        """Optimized event-driven energy transfer with enhanced debugging and performance improvements"""
        if self.g is None or self.g.edge_index is None:
            logger.debug("Energy transfer skipped: graph or edge_index is None")
            return

        try:
            g = self.g
            assert g.edge_index is not None
            device = self.device
            node_type = g.node_type
            energy = g.energy

            # Debug: Log initial energy statistics
            if logger.isEnabledFor(logging.DEBUG):
                initial_total_energy = energy.sum().item()
                initial_avg_energy = energy.mean().item()
                initial_std_energy = energy.std().item()
                logger.debug("Energy transfer start - Total: %.2f, Avg: %.2f, Std: %.2f", float(initial_total_energy), float(initial_avg_energy), float(initial_std_energy))

            # OPTIMIZED: Vectorized highway node normalization with batch processing
            highway_mask = (node_type == NODE_TYPE_HIGHWAY)
            if highway_mask.sum() > 0:
                highway_energies = energy[highway_mask]
                avg_highway_energy = highway_energies.mean()
                energy_diff = avg_highway_energy - highway_energies
                transfer_amount = energy_diff * 0.1
                energy[highway_mask] += transfer_amount
                logger.debug("Highway energy normalization: adjusted %d nodes", int(highway_mask.sum().item()))

            # OPTIMIZED: Batch process edge transfers with memory-efficient operations
            src, dst = g.edge_index
            src_type = node_type[src]
            dst_type = node_type[dst]

            # OPTIMIZED: Vectorized transfer calculations with fused operations
            src_energy = energy[src]
            dst_energy = energy[dst]
            weights = g.weight if g.weight is not None else torch.ones(g.num_edges, 1, device=device)
            energy_caps = g.energy_transfer_capacity if g.energy_transfer_capacity is not None else torch.ones(g.num_edges, 1, device=device)

            # OPTIMIZED: Fused element-wise operations for better performance
            transfer = torch.mul(torch.mul(src_energy, weights.squeeze()), energy_caps.squeeze())

            # OPTIMIZED: Apply transmission loss factor with in-place operation
            transfer.mul_(TRANSMISSION_LOSS)

            # OPTIMIZED: Vectorized node type masks with pre-computation
            highway_src = (src_type == NODE_TYPE_HIGHWAY)
            highway_dst = (dst_type == NODE_TYPE_HIGHWAY)
            dynamic_dst = (dst_type == NODE_TYPE_DYNAMIC)

            # OPTIMIZED: Vectorized pull calculations with memory-efficient approach
            pull_mask = highway_src & dynamic_dst
            if pull_mask.any():
                dst_energy_needed = NODE_ENERGY_CAP - dst_energy[pull_mask]
                pull_amount = torch.minimum(
                    src_energy[pull_mask] * 0.2,
                    dst_energy_needed
                )
                # Use in-place operations for memory efficiency
                energy[src[pull_mask]].sub_(pull_amount)
                energy[dst[pull_mask]].add_(pull_amount)
                logger.debug("Highway pull transfer: %d connections, avg pull: %.4f", int(pull_mask.sum().item()), float(pull_amount.mean().item()))

            # OPTIMIZED: Vectorized normal transfer with batch processing
            normal_mask = ~(highway_src | highway_dst)
            if normal_mask.any():
                # Use in-place operations for memory efficiency
                energy[src[normal_mask]].sub_(transfer[normal_mask])
                energy[dst[normal_mask]].add_(transfer[normal_mask])
                logger.debug("Normal energy transfer: %d connections", int(normal_mask.sum().item()))

            # OPTIMIZED: Vectorized highway transfer with batch processing
            highway_mask = highway_src & highway_dst
            if highway_mask.any():
                transfer_amount = src_energy[highway_mask]
                # Use in-place operations for memory efficiency
                energy[src[highway_mask]].sub_(transfer_amount)
                energy[dst[highway_mask]].add_(transfer_amount)
                logger.debug("Highway-to-highway transfer: %d connections", int(highway_mask.sum().item()))

            # OPTIMIZED: Apply energy caps with in-place clamping
            energy.clamp_(NODE_DEATH_THRESHOLD, NODE_ENERGY_CAP)

            # OPTIMIZED: Energy-based node type processing with vectorization
            self._process_energy_by_node_types()

            # Debug: Log final energy statistics
            if logger.isEnabledFor(logging.DEBUG):
                final_total_energy = energy.sum().item()
                final_avg_energy = energy.mean().item()
                final_std_energy = energy.std().item()
                logger.debug("Energy transfer complete - Total: %.2f, Avg: %.2f, Std: %.2f", float(final_total_energy), float(final_avg_energy), float(final_std_energy))

            # Emergency shutdown if energy distribution is too extreme
            if energy.std() > NODE_ENERGY_CAP * 0.5:  # If standard deviation exceeds 50% of cap
                logger.warning("Emergency shutdown triggered due to extreme energy distribution")
                self.suspended = True

        except Exception as e:
            logger.error("Error in energy transfer: %s", str(e))
            # Don't raise the exception, just log it and continue

    def _process_energy_by_node_types(self) -> None:
        """Optimized energy processing by node types with vectorized operations."""
        if self.g is None:
            return

        g = self.g
        node_type = g.node_type
        energy = g.energy

        # Vectorized processing for different node types
        sensory_mask = (node_type == NODE_TYPE_SENSORY)
        dynamic_mask = (node_type == NODE_TYPE_DYNAMIC)
        workspace_mask = (node_type == NODE_TYPE_WORKSPACE)
        highway_mask = (node_type == NODE_TYPE_HIGHWAY)

        # Apply type-specific energy processing
        if sensory_mask.any():
            # Sensory nodes: hard overwrite to their true pixel values.
            # This must win over any other processing so sensory nodes cannot be altered by the simulation step.
            if self.sensory_true_values is not None:
                num_sensory = int(sensory_mask.sum().item())
                if self.sensory_true_values.shape[0] == num_sensory:
                    energy[sensory_mask] = self.sensory_true_values
                else:
                    # If mismatch, keep current energy (don't mutate sensory values here).
                    # The main update loop will reconcile sensory_true_values sizing.
                    pass

        if dynamic_mask.any():
            # Dynamic nodes: apply subtype-specific energy modulation
            if hasattr(g, 'dynamic_subtype4') and g.dynamic_subtype4 is not None:
                subtype4 = g.dynamic_subtype4[dynamic_mask]
                dynamic_energy = energy[dynamic_mask]

                # Apply subtype-specific energy gains
                gain_1_mask = (subtype4 == SUBTYPE4_GAIN_1)
                gain_01_mask = (subtype4 == SUBTYPE4_GAIN_01)
                gain_001_mask = (subtype4 == SUBTYPE4_GAIN_001)

                if gain_1_mask.any():
                    dynamic_energy[gain_1_mask].add_(0.1)
                if gain_01_mask.any():
                    dynamic_energy[gain_01_mask].add_(0.01)
                if gain_001_mask.any():
                    dynamic_energy[gain_001_mask].add_(0.001)

        if workspace_mask.any():
            # Workspace nodes: maintain stable energy levels
            # CRITICAL: Workspace nodes are sinks - they receive energy from dynamic nodes
            # Only apply gentle adjustment if energy is above a minimum threshold
            # If energy is very low, don't adjust - let connections fill it first
            workspace_energy = energy[workspace_mask]
            target_energy = NODE_ENERGY_CAP * 0.7  # Target 70% of capacity
            # Only adjust if energy is above a small threshold (allow energy to accumulate from connections)
            # If energy is very low (< 0.1), don't adjust - let connections fill it
            low_energy_mask = workspace_energy.squeeze() < 0.1
            if not low_energy_mask.all():
                # Gentle adjustment towards target (only for nodes with some energy)
                adjustment = (target_energy - workspace_energy) * 0.02
                # Don't apply adjustment to nodes with very low energy (let them accumulate)
                adjustment[low_energy_mask] = 0.0
                energy[workspace_mask].add_(adjustment)
            # If all workspace nodes are low, ensure they don't go negative (clamp to 0.0 minimum)
            # This prevents workspace nodes from going negative while waiting for energy from connections
            if low_energy_mask.all():
                energy[workspace_mask] = torch.clamp(energy[workspace_mask], min=0.0)

        if highway_mask.any():
            # Highway nodes: ensure high energy levels for efficient transfer
            highway_energy = energy[highway_mask]
            target_energy = NODE_ENERGY_CAP * 0.9  # Target 90% of capacity
            # Faster adjustment towards target
            adjustment = (target_energy - highway_energy) * 0.05
            energy[highway_mask].add_(adjustment)

    def _remove_nodes(self, node_mask: torch.Tensor) -> None:
        """Remove nodes based on the given boolean mask with proper synchronization"""
        if self.g is None or not node_mask.numel():
            return

        # Start critical section - acquire locks and signal modification
        with self._graph_lock:
            try:
                # Signal that graph is being modified to other threads
                self.graph_modification_lock.clear()
                # Wait for any ongoing worker operations to complete
                if not self.wait_for_workers_idle(timeout=2.0):
                    logger.warning("Could not wait for workers to become idle before node removal")
                    self._log_recovery_event("worker_timeout", {
                        "action": "node_removal",
                        "worker_status": "busy",
                        "timeout": 2.0
                    })
                g = self.g
                if g.num_nodes is None:
                    logger.error("Graph has None num_nodes during node removal")
                    self._log_recovery_event("critical_error", {
                        "error": "null_num_nodes",
                        "operation": "node_removal",
                        "severity": "high"
                    })
                    return

                # Validate current graph state before modification
                if not self.validate_graph_state():
                    logger.error("Graph state validation failed before node removal")
                    self._log_recovery_event("validation_failure", {
                        "operation": "node_removal",
                        "graph_state": "invalid",
                        "attempted_recovery": False
                    })
                    return

                # Convert mask to indices
                node_indices = torch.where(node_mask)[0]

                # Remove edges connected to these nodes
                if g.edge_index is not None:
                    src, dst = g.edge_index
                    # Validate edge tensor shapes
                    if src.shape[0] != g.num_edges or dst.shape[0] != g.num_edges:
                        logger.warning("Edge tensor shape mismatch during node removal")
                        return

                    # Find edges where either source or destination is in the nodes to remove
                    edges_to_remove = (torch.isin(src, node_indices) | torch.isin(dst, node_indices))
                    if edges_to_remove.any():
                        # Remove edges
                        edge_keep_mask = ~edges_to_remove
                        g.edge_index = g.edge_index[:, edge_keep_mask]
                        # Update edge attributes
                        for key in g.keys():
                            if hasattr(g, key) and g[key] is not None:
                                attr = g[key]
                                # Check if it's a tensor and has the right shape
                                if isinstance(attr, torch.Tensor) and attr.dim() > 0 and attr.shape[0] == g.num_edges:
                                    g[key] = attr[edge_keep_mask]

                # After removing edges, we need to remap the remaining edge indices
                # to account for the removed nodes
                if g.edge_index is not None and g.edge_index.shape[1] > 0:
                    # Create a mapping from old node indices to new node indices
                    old_to_new_mapping = torch.full((g.num_nodes,), -1, dtype=torch.int64, device=self.device)
                    node_keep_mask = ~node_mask
                    new_indices = torch.arange(int(node_keep_mask.sum().item()), dtype=torch.int64, device=self.device)
                    old_to_new_mapping[node_keep_mask] = new_indices

                    # Remap the edge indices
                    src, dst = g.edge_index
                    valid_edges = (src >= 0) & (dst >= 0) & (src < old_to_new_mapping.shape[0]) & (dst < old_to_new_mapping.shape[0])

                    if valid_edges.any():
                        # Remap valid edges
                        src_remapped = old_to_new_mapping[src[valid_edges]]
                        dst_remapped = old_to_new_mapping[dst[valid_edges]]

                        # Filter out any edges that would become invalid after remapping
                        valid_after_remap = (src_remapped >= 0) & (dst_remapped >= 0) & (src_remapped != dst_remapped)
                        if valid_after_remap.any():
                            g.edge_index = torch.stack([src_remapped[valid_after_remap], dst_remapped[valid_after_remap]])

                            # Also remap edge attributes
                            for key in g.keys():
                                if hasattr(g, key) and g[key] is not None:
                                    attr = g[key]
                                    if isinstance(attr, torch.Tensor) and attr.dim() > 0 and attr.shape[0] == g.num_edges:
                                        g[key] = attr[valid_edges][valid_after_remap]
                        else:
                            # No valid edges after remapping
                            g.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                            for key in g.keys():
                                if hasattr(g, key) and g[key] is not None:
                                    attr = g[key]
                                    if isinstance(attr, torch.Tensor) and attr.dim() > 0 and attr.shape[0] == g.num_edges:
                                        g[key] = torch.empty((0,), dtype=attr.dtype, device=self.device)
                    else:
                        # No valid edges at all
                        g.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                        for key in g.keys():
                            if hasattr(g, key) and g[key] is not None:
                                attr = g[key]
                                if isinstance(attr, torch.Tensor) and attr.dim() > 0 and attr.shape[0] == g.num_edges:
                                    g[key] = torch.empty((0,), dtype=attr.dtype, device=self.device)

                # Remove nodes
                keep_mask = ~node_mask
                if keep_mask.any():
                    # Update node attributes
                    for key in g.keys():
                        if hasattr(g, key) and g[key] is not None:
                            attr = g[key]
                            # Check if it's a tensor and has the right shape
                            if isinstance(attr, torch.Tensor) and attr.dim() > 0 and attr.shape[0] == g.num_nodes:
                                g[key] = attr[keep_mask]

                    # Special handling for parent relationships - remap to -1 if parent was removed
                    if hasattr(g, 'parent') and g.parent is not None:
                        # Create mapping from old indices to new indices
                        old_to_new_mapping = torch.full((g.num_nodes,), -1, dtype=torch.int64, device=self.device)
                        node_keep_mask = ~node_mask
                        new_indices = torch.arange(int(node_keep_mask.sum().item()), dtype=torch.int64, device=self.device)
                        old_to_new_mapping[node_keep_mask] = new_indices

                        # Remap parent relationships
                        parent_remapped = old_to_new_mapping[g.parent]
                        # Set invalid parents to -1
                        parent_remapped[parent_remapped < 0] = -1
                        g.parent = parent_remapped
                else:
                    # All nodes removed, reset to empty graph
                    self._init_empty_graph()
                    return

                # Update node counts - ensure num_nodes is synchronized with actual tensor shapes
                if g.energy is not None and g.energy.dim() > 0:
                    actual_node_count = g.energy.shape[0]
                    g.num_nodes = actual_node_count  # Explicitly synchronize num_nodes
                    self.n_total = actual_node_count
                else:
                    self.n_total = 0
                    g.num_nodes = 0  # Ensure consistency

                # Ensure edge tensor consistency after node removal
                self._ensure_edge_tensor_consistency()

                # Synchronize all tensors to ensure consistency after node removal
                self._synchronize_all_tensors()

                if self.n_total > 0:
                    self.n_sensory = int((g.node_type == NODE_TYPE_SENSORY).sum().cpu().item())
                    self.n_dynamic = int((g.node_type == NODE_TYPE_DYNAMIC).sum().cpu().item())
                    self.n_workspace = int((g.node_type == NODE_TYPE_WORKSPACE).sum().cpu().item())
                else:
                    self.n_sensory = 0
                    self.n_dynamic = 0
                    self.n_workspace = 0

                # Update metrics
                num_removed = node_indices.shape[0]
                self.node_deaths += num_removed
                self.total_node_deaths += num_removed

                # Validate state after modification
                if not self.validate_graph_state():
                    logger.error("Graph state validation failed after node removal")
                    self._log_recovery_event("post_removal_validation_failure", {
                        "nodes_removed": int(num_removed),
                        "remaining_nodes": int(self.n_total),
                        "recovery_attempted": False,
                        "severity": "critical"
                    })
                    return

                logger.info("Removed %d nodes. Total nodes: %d", int(num_removed), int(self.n_total))
                self._log_recovery_event("successful_node_removal", {
                    "nodes_removed": int(num_removed),
                    "remaining_nodes": int(self.n_total),
                    "sensory_nodes": self.n_sensory,
                    "dynamic_nodes": self.n_dynamic,
                    "workspace_nodes": self.n_workspace
                })
            except Exception as e:
                logger.error(f"Error during node removal: {e}")
                raise
            finally:
                # Always signal that graph modification is complete
                self.graph_modification_lock.set()

    def _handle_energy_based_spawning(self) -> None:
        """Handle node spawning based on energy thresholds"""
        if self.g is None or self.g.num_nodes is None:
            return

        g = self.g
        energies = g.energy.squeeze()

        # Enforce max node limit based on configuration
        # Default to 100000 for production, test system can override via config
        try:
            max_limit = self.config_manager.get_config('neural', 'max_nodes') or 100000
        except Exception:
            max_limit = 100000  # Sensible default for production
        current_node_count = g.num_nodes or 0
        if current_node_count >= max_limit:
            logger.debug(f"Spawn blocked: at max node limit {max_limit}")
            return  # Don't spawn if at or above limit

        # Check for nodes with sufficient energy to spawn new nodes
        spawn_candidates = energies > NODE_SPAWN_THRESHOLD
        num_candidates = spawn_candidates.sum().item()
        
        if num_candidates == 0:
            return

        # Calculate how many nodes we can spawn based on available energy
        candidate_energies = energies[spawn_candidates]
        spawnable_nodes = torch.floor(candidate_energies / NODE_ENERGY_SPAWN_COST).to(torch.int32)
        total_spawnable = min(int(spawnable_nodes.sum().item()), MAX_NODE_BIRTHS_PER_STEP)

        # Further limit by remaining node capacity
        remaining_capacity = max_limit - current_node_count
        total_spawnable = min(total_spawnable, remaining_capacity)
        
        # Debug log spawn attempt
        if self.step_counter % 10 == 0:
            logger.debug(f"Spawn check: {num_candidates} candidates, {total_spawnable} spawnable, limit={max_limit}, current={current_node_count}")

        if total_spawnable <= 0:
            return

        # Distribute spawning across candidate nodes
        spawn_counts = torch.zeros_like(spawn_candidates, dtype=torch.int32)
        remaining = total_spawnable

        # Simple distribution: give each candidate at least one spawn if possible
        # Create a mapping from global index to candidate index
        candidate_indices = torch.where(spawn_candidates)[0]
        for candidate_idx, global_idx in enumerate(candidate_indices):
            if remaining <= 0:
                break
            spawn_count = min(1, int(spawnable_nodes[candidate_idx].item()))
            spawn_counts[global_idx] = int(spawn_count)
            remaining -= int(spawn_count)

        # Actually spawn the nodes - BATCH ALL SPAWNS for efficiency
        # Collect all parent indices that will spawn
        spawning_parents = []
        for i, count in enumerate(spawn_counts):
            if count > 0:
                spawning_parents.extend([i] * int(count))
        
        if spawning_parents:
            # Batch spawn all nodes at once
            self._add_nodes(len(spawning_parents), NODE_TYPE_DYNAMIC, parent_idx=spawning_parents)
            
            # Deduct energy costs from parents (vectorized)
            unique_parents = torch.unique(torch.tensor(spawning_parents, device=g.energy.device))
            for parent_idx in unique_parents:
                count = spawning_parents.count(parent_idx.item())
                energy_cost = count * NODE_ENERGY_SPAWN_COST
                current_energy = g.energy[parent_idx].item()
                g.energy[parent_idx] = max(current_energy - energy_cost, NODE_DEATH_THRESHOLD)
            
            # Update metrics
            self.node_births += len(spawning_parents)
            self.total_node_births += len(spawning_parents)
            
            logger.info("Spawned %d new nodes based on energy thresholds", len(spawning_parents))

    def get_node_energy(self, node_id: int) -> float:
        """
        Get energy level for a specific node.
        
        This method is used by the workspace system to read energy values
        from sensory nodes for visualization and processing.
        
        Args:
            node_id: The index of the node to query
            
        Returns:
            The energy value of the node, or 0.0 if the node doesn't exist
        """
        if self.vector_engine is not None:
            store = self.vector_engine.store
            if 0 <= node_id < store.capacity and store.active_mask[node_id]:
                try:
                    return float(store.energy[node_id].item())
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

        if self.g is None:
            return 0.0
        
        if not hasattr(self.g, 'energy') or self.g.energy is None:
            return 0.0
        
        if node_id < 0 or node_id >= len(self.g.energy):
            return 0.0
        
        try:
            return float(self.g.energy[node_id].item())
        except Exception as e:
            logger.warning(f"Failed to get energy for node {node_id}: {e}")
            return 0.0

    def get_batch_energies(self, node_ids: list[int]) -> list[float]:
        """
        Get energy levels for multiple nodes efficiently.
        
        This method is used by the workspace system to read energy values
        from multiple sensory nodes in a single call for better performance.
        
        Args:
            node_ids: List of node indices to query
            
        Returns:
            List of energy values corresponding to the node_ids
        """
        if self.g is None:
            return [0.0] * len(node_ids)
        
        if not hasattr(self.g, 'energy') or self.g.energy is None:
            return [0.0] * len(node_ids)
        
        energies = []
        for node_id in node_ids:
            if 0 <= node_id < len(self.g.energy):
                try:
                    energies.append(float(self.g.energy[node_id].item()))
                except Exception:
                    energies.append(0.0)
            else:
                energies.append(0.0)
        
        return energies

    def get_workspace_node_energy(self, workspace_local_id: int) -> float:
        """
        Get energy for a workspace node by its local ID (0 to n_workspace-1).
        
        Workspace nodes are the OUTPUT of the simulation - they receive energy
        from dynamic nodes and display it on the canvas. This is the inverse
        of sensory nodes which INPUT pixel data.
        
        Args:
            workspace_local_id: Local workspace node ID (0 to n_workspace-1)
            
        Returns:
            The energy value of the workspace node, or 0.0 if not found
        """
        if self.g is None or not hasattr(self.g, 'energy') or self.g.energy is None:
            return 0.0
        
        # Workspace nodes are at indices: n_sensory + n_dynamic + workspace_local_id
        pyg_node_id = self.n_sensory + self.n_dynamic + workspace_local_id
        
        if pyg_node_id < 0 or pyg_node_id >= len(self.g.energy):
            return 0.0
        
        try:
            return float(self.g.energy[pyg_node_id].item())
        except Exception as e:
            logger.debug(f"Failed to get workspace energy for local ID {workspace_local_id}: {e}")
            return 0.0

    def get_workspace_energies_grid(self) -> list[list[float]]:
        """
        Get all workspace node energies as a 2D grid for visualization.
        
        Returns:
            2D list of workspace node energies matching the workspace grid layout
        """
        ws_width, ws_height = self.workspace_size
        grid = [[0.0 for _ in range(ws_width)] for _ in range(ws_height)]
        
        if self.g is None or not hasattr(self.g, 'energy') or self.g.energy is None:
            return grid
        
        # Get workspace node indices in PyG graph
        ws_start = self.n_sensory + self.n_dynamic
        ws_end = ws_start + self.n_workspace
        
        if ws_end > len(self.g.energy):
            return grid
        
        try:
            ws_energies = self.g.energy[ws_start:ws_end, 0].cpu().numpy()
            for i, energy in enumerate(ws_energies):
                x = i % ws_width
                y = i // ws_width
                if y < ws_height:
                    grid[y][x] = float(energy)
        except Exception as e:
            logger.debug(f"Failed to get workspace energy grid: {e}")
        
        return grid
