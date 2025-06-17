import sys
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import numpy as np
import time
import threading
import queue
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import traceback
import os

# Configure logging
logger = logging.getLogger(__name__)

# Try to import CUDA kernels, fallback gracefully if not available
try:
    from .cuda_kernels import CUDAModule
except ImportError:
    logger.warning("CUDA kernels not available, falling back to CPU implementation")
    CUDAModule = None

print("Starting PyG Neural System initialization...")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# --- Node Type Constants ---
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
SUBTYPE3_2P_10 = 1  # 2% every 10 steps
SUBTYPE3_3P_20 = 2  # 3% every 20 steps
SUBTYPE3_NAMES = ['1%/5steps', '2%/10steps', '3%/20steps']

# --- Fourth Subtype Slot (Connection-based Energy Gain) ---
SUBTYPE4_GAIN_1 = 0   # Gain 1 per connection per step
SUBTYPE4_GAIN_01 = 1  # Gain 0.1 per connection per step
SUBTYPE4_GAIN_001 = 2  # Gain 0.01 per connection per step
SUBTYPE4_NAMES = ['+1/conn', '+0.1/conn', '+0.01/conn']

# --- Config-like constants (tune as needed) ---
NODE_SPAWN_THRESHOLD = 20.0
NODE_DEATH_THRESHOLD = 0.0
NODE_ENERGY_SPAWN_COST = 5.0
NODE_ENERGY_DECAY = 0.1
MAX_NODE_BIRTHS_PER_STEP = 1000
MAX_CONN_BIRTHS_PER_STEP = 1000
NODE_ENERGY_CAP = 244.0
CONNECTION_MAINTENANCE_COST = 0.02
TRANSMISSION_LOSS = 0.9

# --- Connection Types ---
CONN_TYPE_EXCITATORY = 0
CONN_TYPE_INHIBITORY = 1
CONN_TYPE_GATED = 2
CONN_TYPE_PLASTIC = 3
CONN_TYPE_NAMES = ['Excitatory', 'Inhibitory', 'Gated', 'Plastic']
GATE_THRESHOLD = 0.5

# For plastic connections
PLASTIC_LEARNING_RATE_MIN = 0.001
PLASTIC_LEARNING_RATE_MAX = 0.05

# --- New Connection Subtype3 (Directionality) ---
CONN_SUBTYPE3_ONE_WAY_OUT = 0
CONN_SUBTYPE3_ONE_WAY_IN = 1
CONN_SUBTYPE3_FREE_FLOW = 2
CONN_SUBTYPE3_NAMES = ['OneWayOut', 'OneWayIn', 'FreeFlow']

@dataclass
class NodeData:
    """Data class for node attributes"""
    energy: torch.Tensor
    node_type: torch.Tensor
    pos: torch.Tensor
    velocity: torch.Tensor
    dynamic_subtype: Optional[torch.Tensor] = None
    dynamic_subtype2: Optional[torch.Tensor] = None
    dynamic_subtype3: Optional[torch.Tensor] = None
    dynamic_subtype4: Optional[torch.Tensor] = None
    max_connections: Optional[torch.Tensor] = None
    phase_offset: Optional[torch.Tensor] = None
    parent: Optional[torch.Tensor] = None

@dataclass
class EdgeData:
    """Data class for edge attributes"""
    weight: torch.Tensor
    energy_transfer_capacity: torch.Tensor
    conn_type: torch.Tensor
    plastic_lr: torch.Tensor
    gate_threshold: torch.Tensor
    conn_subtype3: torch.Tensor

class ConnectionWorker(threading.Thread):
    """Worker thread for handling connection operations in parallel"""
    def __init__(self, system, batch_size=25):
        super().__init__()
        self.system = system
        self.batch_size = batch_size
        self.task_queue = queue.Queue(maxsize=1000)  # Limit queue size
        self.result_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.daemon = True
        self._lock = threading.Lock()
        self._processing = False
        self._error_count = 0
        self._max_retries = 3
        self._retry_delay = 1.0
        self._last_activity = time.time()
        self._timeout = 30.0  # 30 seconds timeout
        self._metrics = {
            'tasks_processed': 0,
            'errors': 0,
            'retries': 0,
            'processing_time': 0.0,
            'queue_size': 0,
            'avg_batch_size': 0,
            'success_rate': 1.0
        }
        self._batch_metrics = []
        self._start_time = time.time()

    def run(self):
        """Main worker loop with error handling and recovery"""
        while not self.stop_event.is_set():
            try:
                # Check for timeout
                if time.time() - self._last_activity > self._timeout:
                    logger.warning("Connection worker timeout, restarting...")
                    self._error_count += 1
                    if self._error_count >= self._max_retries:
                        logger.error("Max retries exceeded, stopping worker")
                        self.stop_event.set()
                        break
                    time.sleep(self._retry_delay * (2 ** (self._error_count - 1)))
                    continue

                # Get task with timeout
                try:
                    task = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                with self._lock:
                    self._processing = True
                    start_time = time.time()
                    batch_metrics = {'start_time': start_time, 'size': 0, 'success': False}
                    
                    try:
                        if task['type'] == 'grow':
                            # Prepare connection growth batch
                            batch = self.system._prepare_connection_growth_batch(self.batch_size)
                            batch_metrics['size'] = len(batch)
                            self.result_queue.put({'type': 'grow', 'batch': batch})
                            batch_metrics['success'] = True
                        elif task['type'] == 'cull':
                            # Prepare culling batch
                            batch = self.system._prepare_cull_batch(self.batch_size)
                            batch_metrics['size'] = len(batch)
                            self.result_queue.put({'type': 'cull', 'batch': batch})
                            batch_metrics['success'] = True
                        else:
                            logger.error(f"Unknown task type: {task['type']}")
                            self.result_queue.put({'type': 'error', 'error': f"Unknown task type: {task['type']}"})

                        # Update metrics
                        self._metrics['tasks_processed'] += 1
                        self._metrics['processing_time'] += time.time() - start_time
                        self._metrics['queue_size'] = self.task_queue.qsize()
                        self._last_activity = time.time()
                        self._error_count = 0  # Reset error count on success
                        
                        # Update batch metrics
                        batch_metrics['end_time'] = time.time()
                        self._batch_metrics.append(batch_metrics)
                        if len(self._batch_metrics) > 100:  # Keep last 100 batches
                            self._batch_metrics.pop(0)
                        
                        # Update success rate
                        success_count = sum(1 for m in self._batch_metrics if m['success'])
                        self._metrics['success_rate'] = success_count / len(self._batch_metrics)
                        
                        # Update average batch size
                        self._metrics['avg_batch_size'] = sum(m['size'] for m in self._batch_metrics) / len(self._batch_metrics)

                    except Exception as e:
                        self._metrics['errors'] += 1
                        logger.error(f"Error processing task: {e}")
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
                logger.error(f"Critical error in connection worker: {e}")
                self._metrics['errors'] += 1
                self._error_count += 1
                if self._error_count >= self._max_retries:
                    logger.error("Max retries exceeded, stopping worker")
                    self.stop_event.set()
                    break
                time.sleep(self._retry_delay * (2 ** (self._error_count - 1)))

    def stop(self):
        """Safely stop the worker thread"""
        with self._lock:
            if self._processing:
                # Wait for current processing to complete
                while self._processing:
                    time.sleep(0.1)
            self.stop_event.set()
            logger.info("Connection worker stopped")

    def is_processing(self):
        """Check if worker is currently processing a task"""
        with self._lock:
            return self._processing

    def clear_queues(self):
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

    def get_metrics(self):
        """Get worker metrics"""
        with self._lock:
            metrics = self._metrics.copy()
            metrics['uptime'] = time.time() - self._start_time
            metrics['active'] = super().is_alive()  # Fixed: use super() instead of self
            metrics['processing'] = self._processing
            metrics['error_count'] = self._error_count
            return metrics

    def queue_task(self, task_type, **kwargs):
        """Queue a new task with validation"""
        if task_type not in ['grow', 'cull']:
            raise ValueError(f"Invalid task type: {task_type}")

        task = {'type': task_type, **kwargs}
        try:
            self.task_queue.put(task, timeout=1.0)
            return True
        except queue.Full:
            logger.warning("Task queue full, task dropped")
            return False

    def get_result(self, timeout=0.1):
        """Get a result with timeout"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_alive(self):
        """Check if worker is alive and healthy"""
        return (super().is_alive() and  # Fixed: use super() instead of self
                not self.stop_event.is_set() and 
                self._error_count < self._max_retries)

class PyGNeuralSystem:
    def __init__(self, config: dict):
        """Initialize the neural system with PyTorch Geometric"""
        try:
            # Store configuration
            self.config = config
            
            # Get device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Get dimensions from config
            self.sensory_width = config['sensory']['width']
            self.sensory_height = config['sensory']['height']
            self.workspace_size = (config['workspace']['width'], config['workspace']['height'])
            self.n_dynamic_target = config['system']['max_nodes']
            
            # Validate dimensions
            if self.sensory_width <= 0 or self.sensory_height <= 0:
                raise ValueError("Invalid sensory dimensions")
            if self.workspace_size[0] <= 0 or self.workspace_size[1] <= 0:
                raise ValueError("Invalid workspace dimensions")
            if self.n_dynamic_target <= 0:
                raise ValueError("Invalid dynamic node target")
            
            # Initialize CUDA kernels
            try:
                self.cuda_module = CUDAModule() if CUDAModule is not None else None
                if self.cuda_module:
                    logger.info("CUDA kernels initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA kernels: {e}")
                self.cuda_module = None
            
            # Initialize node counters
            self.n_sensory = 0
            self.n_dynamic = 0
            self.n_workspace = 0
            self.n_total = 0
            
            # Initialize metrics
            self.node_births = 0
            self.node_deaths = 0
            self.conn_births = 0
            self.conn_deaths = 0
            self.total_node_births = 0
            self.total_node_deaths = 0
            self.total_conn_births = 0
            self.total_conn_deaths = 0
            self.step_counter = 0
            
            # Initialize queues
            self.death_queue = []
            self.birth_queue = []
            self.conn_growth_queue = []
            self.conn_candidate_queue = []
            
            # System state
            self.suspended = False
            self.last_update_time = time.time()
            
            # Performance tracking
            self._last_cleanup = time.time()
            self._cleanup_interval = 60.0
            self._last_metrics_update = time.time()
            self._metrics_update_interval = 1.0
            
            # Initialize empty graph
            self._init_empty_graph()
            
            # Initialize connection workers
            self._init_connection_workers()
            
            # Initialize metrics
            self._init_metrics()
            
            logger.info("Neural system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing neural system: {e}")
            logger.error(traceback.format_exc())
            # Cleanup any partially initialized resources
            self.cleanup()
            raise

    def _init_metrics(self):
        """Initialize system metrics"""
        try:
            self.metrics = {
                'update_time': [],
                'memory_usage': [],
                'node_count': [],
                'edge_count': [],
                'energy_distribution': [],
                'connection_worker_metrics': []
            }
            logger.info("Metrics initialized")
        except Exception as e:
            logger.error(f"Error initializing metrics: {e}")
            raise

    def _init_empty_graph(self):
        """Initialize an empty graph with PyTorch Geometric"""
        try:
            # Create empty node features
            node_data = NodeData(
                energy=torch.zeros((0, 1), device=self.device),
                node_type=torch.zeros((0,), dtype=torch.int64, device=self.device),
                pos=torch.zeros((0, 2), device=self.device),
                velocity=torch.zeros((0, 2), device=self.device)
            )
            
            # Create empty edge features
            edge_data = EdgeData(
                weight=torch.zeros((0, 1), device=self.device),
                energy_transfer_capacity=torch.zeros((0, 1), device=self.device),
                conn_type=torch.zeros((0, 1), dtype=torch.int64, device=self.device),
                plastic_lr=torch.zeros((0, 1), device=self.device),
                gate_threshold=torch.zeros((0, 1), device=self.device),
                conn_subtype3=torch.zeros((0,), dtype=torch.int64, device=self.device)
            )
            
            # Create empty edge index
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
            # Create PyG Data object
            self.graph = Data(
                x=node_data.energy,
                edge_index=edge_index,
                edge_attr=edge_data.weight,
                node_type=node_data.node_type,
                pos=node_data.pos,
                velocity=node_data.velocity,
                dynamic_subtype=node_data.dynamic_subtype,
                dynamic_subtype2=node_data.dynamic_subtype2,
                dynamic_subtype3=node_data.dynamic_subtype3,
                dynamic_subtype4=node_data.dynamic_subtype4,
                max_connections=node_data.max_connections,
                phase_offset=node_data.phase_offset,
                parent=node_data.parent,
                edge_weight=edge_data.weight,
                edge_capacity=edge_data.energy_transfer_capacity,
                edge_type=edge_data.conn_type,
                edge_plastic_lr=edge_data.plastic_lr,
                edge_gate_threshold=edge_data.gate_threshold,
                edge_subtype3=edge_data.conn_subtype3
            )
            
            # Set grid dimensions
            self.grid_width = max(self.sensory_width, self.workspace_size[0])
            self.grid_height = max(self.sensory_height, self.workspace_size[1])
            
            # Verify initialization
            if not isinstance(self.graph, Data):
                raise RuntimeError("Failed to create PyG Data object")
            
            # Initialize sensory nodes
            self._init_sensory_nodes()
            
            # Initialize workspace nodes
            self._init_workspace_nodes()
            
            # Initialize dynamic nodes
            self._init_dynamic_nodes()
            
            logger.info(f"Graph initialized with {self.graph.num_nodes} nodes and {self.graph.num_edges} edges")
            
        except Exception as e:
            logger.error(f"Error initializing graph: {e}")
            logger.error(traceback.format_exc())
            raise

    def _init_sensory_nodes(self):
        """Initialize sensory nodes"""
        try:
            n_sensory = self.sensory_width * self.sensory_height
            if n_sensory > 0:
                # Create grid positions
                x = torch.linspace(0, 1, self.sensory_width, device=self.device)
                y = torch.linspace(0, 1, self.sensory_height, device=self.device)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
                
                # Create node data
                node_data = NodeData(
                    energy=torch.zeros((n_sensory, 1), device=self.device),
                    node_type=torch.full((n_sensory,), NODE_TYPE_SENSORY, dtype=torch.int64, device=self.device),
                    pos=pos,
                    velocity=torch.zeros((n_sensory, 2), device=self.device)
                )
                
                # Add nodes to graph
                self._add_nodes_to_graph(node_data)
                self.n_sensory = n_sensory
                logger.info(f"Initialized {n_sensory} sensory nodes")
        except Exception as e:
            logger.error(f"Error initializing sensory nodes: {e}")
            raise

    def _init_workspace_nodes(self):
        """Initialize workspace nodes"""
        try:
            n_workspace = self.workspace_size[0] * self.workspace_size[1]
            if n_workspace > 0:
                # Create grid positions
                x = torch.linspace(0, 1, self.workspace_size[0], device=self.device)
                y = torch.linspace(0, 1, self.workspace_size[1], device=self.device)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
                
                # Create node data
                node_data = NodeData(
                    energy=torch.zeros((n_workspace, 1), device=self.device),
                    node_type=torch.full((n_workspace,), NODE_TYPE_WORKSPACE, dtype=torch.int64, device=self.device),
                    pos=pos,
                    velocity=torch.zeros((n_workspace, 2), device=self.device)
                )
                
                # Add nodes to graph
                self._add_nodes_to_graph(node_data)
                self.n_workspace = n_workspace
                logger.info(f"Initialized {n_workspace} workspace nodes")
        except Exception as e:
            logger.error(f"Error initializing workspace nodes: {e}")
            raise

    def _init_dynamic_nodes(self):
        """Initialize dynamic nodes"""
        try:
            if self.n_dynamic_target > 0:
                # Create random positions
                pos = torch.rand(self.n_dynamic_target, 2, device=self.device)
                
                # Create node data with subtypes
                node_data = NodeData(
                    energy=torch.zeros((self.n_dynamic_target, 1), device=self.device),
                    node_type=torch.full((self.n_dynamic_target,), NODE_TYPE_DYNAMIC, dtype=torch.int64, device=self.device),
                    pos=pos,
                    velocity=torch.zeros((self.n_dynamic_target, 2), device=self.device),
                    dynamic_subtype=torch.randint(0, 3, (self.n_dynamic_target,), device=self.device),
                    dynamic_subtype2=torch.randint(0, 3, (self.n_dynamic_target,), device=self.device),
                    dynamic_subtype3=torch.randint(0, 3, (self.n_dynamic_target,), device=self.device),
                    dynamic_subtype4=torch.randint(0, 3, (self.n_dynamic_target,), device=self.device),
                    max_connections=torch.randint(1, 6, (self.n_dynamic_target,), device=self.device),
                    phase_offset=torch.rand(self.n_dynamic_target, device=self.device) * 2 * np.pi
                )
                
                # Add nodes to graph
                self._add_nodes_to_graph(node_data)
                self.n_dynamic = self.n_dynamic_target
                logger.info(f"Initialized {self.n_dynamic_target} dynamic nodes")
        except Exception as e:
            logger.error(f"Error initializing dynamic nodes: {e}")
            raise

    def _add_nodes_to_graph(self, node_data: NodeData):
        """Helper method to add nodes to the graph"""
        # Get current number of nodes
        n_current = self.graph.num_nodes
        n_new = len(node_data.energy)

        # Create new graph with combined data
        new_graph = Data(
            x=torch.cat([self.graph.x, node_data.energy], dim=0),
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr,
            node_type=torch.cat([self.graph.node_type, node_data.node_type], dim=0),
            pos=torch.cat([self.graph.pos, node_data.pos], dim=0),
            velocity=torch.cat([self.graph.velocity, node_data.velocity], dim=0)
        )

        # Add optional attributes if they exist
        for attr_name in ['dynamic_subtype', 'dynamic_subtype2', 'dynamic_subtype3', 'dynamic_subtype4', 
                         'max_connections', 'phase_offset', 'parent']:
            node_attr = getattr(node_data, attr_name)
            if node_attr is not None:
                if hasattr(self.graph, attr_name) and getattr(self.graph, attr_name) is not None:
                    existing_attr = getattr(self.graph, attr_name)
                else:
                    # Create default values for existing nodes
                    if attr_name in ['max_connections']:
                        existing_attr = torch.full((n_current,), 5, dtype=torch.int64, device=self.device)
                    elif attr_name in ['parent']:
                        existing_attr = torch.full((n_current,), -1, dtype=torch.int64, device=self.device)
                    elif attr_name in ['phase_offset']:
                        existing_attr = torch.zeros(n_current, device=self.device)
                    else:
                        existing_attr = torch.full((n_current,), -1, dtype=torch.int64, device=self.device)
                
                setattr(new_graph, attr_name, torch.cat([existing_attr, node_attr], dim=0))

        # Copy edge attributes
        for edge_attr in ['edge_weight', 'edge_capacity', 'edge_type', 'edge_plastic_lr', 
                         'edge_gate_threshold', 'edge_subtype3']:
            if hasattr(self.graph, edge_attr):
                setattr(new_graph, edge_attr, getattr(self.graph, edge_attr))

        # Update the graph
        self.graph = new_graph
        self.n_total += n_new

    def _init_connection_workers(self):
        """Initialize connection workers"""
        try:
            # Calculate number of workers based on available CPU cores
            n_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 workers
            logger.info(f"Initializing {n_workers} connection workers")
            
            # Create worker pool
            self.worker_pool = ThreadPoolExecutor(max_workers=n_workers)
            
            # Initialize workers
            self.workers = []
            for i in range(n_workers):
                try:
                    worker = ConnectionWorker(self, batch_size=25)
                    self.workers.append(worker)
                    logger.info(f"Initialized worker {i+1}/{n_workers}")
                except Exception as e:
                    logger.error(f"Failed to initialize worker {i+1}: {e}")
                    raise
            
            # Start workers
            for i, worker in enumerate(self.workers):
                try:
                    future = self.worker_pool.submit(worker.run)
                    worker.future = future
                    logger.info(f"Started worker {i+1}/{n_workers}")
                except Exception as e:
                    logger.error(f"Failed to start worker {i+1}: {e}")
                    raise
            
            logger.info("All connection workers initialized and started")
            
        except Exception as e:
            logger.error(f"Error initializing connection workers: {e}")
            logger.error(traceback.format_exc())
            # Cleanup any partially initialized workers
            self._cleanup_workers()
            raise

    def _cleanup_workers(self):
        """Cleanup connection workers"""
        try:
            if hasattr(self, 'workers'):
                for worker in self.workers:
                    try:
                        worker.stop()
                    except Exception as e:
                        logger.error(f"Error stopping worker: {e}")
                self.workers.clear()
            
            if hasattr(self, 'worker_pool'):
                try:
                    self.worker_pool.shutdown(wait=True)
                except Exception as e:
                    logger.error(f"Error shutting down worker pool: {e}")
                self.worker_pool = None
                
            logger.info("Connection workers cleaned up")
            
        except Exception as e:
            logger.error(f"Error during worker cleanup: {e}")
            logger.error(traceback.format_exc())

    def cleanup(self):
        """Clean up resources"""
        try:
            # Shutdown thread pool
            if hasattr(self, 'worker_pool'):
                self.worker_pool.shutdown(wait=True)

            # Cleanup workers
            self._cleanup_workers()

            # Clear queues
            self.death_queue.clear()
            self.birth_queue.clear()
            self.conn_growth_queue.clear()
            self.conn_candidate_queue.clear()

            # Clear graph data
            if hasattr(self, 'graph'):
                self.graph = None

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache if using GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            logger.info("System cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def to(self, device: str):
        """Move the system to a different device"""
        self.device = device
        self.graph = self.graph.to(device)
        return self

    def summary(self):
        """Print a summary of the system state"""
        logger.info(f"Nodes: {self.graph.num_nodes} (sensory: {self.n_sensory}, dynamic: {self.n_dynamic}, workspace: {self.n_workspace})")
        logger.info(f"Edges: {self.graph.num_edges}")
        logger.info(f"Device: {self.device}")

    def get_metrics(self):
        """Get system metrics"""
        node_types = self.graph.node_type
        n_sensory = int((node_types == NODE_TYPE_SENSORY).sum().cpu().item())
        n_dynamic = int((node_types == NODE_TYPE_DYNAMIC).sum().cpu().item())
        n_workspace = int((node_types == NODE_TYPE_WORKSPACE).sum().cpu().item())
        
        dynamic_energies = self.graph.x[node_types == NODE_TYPE_DYNAMIC].cpu().numpy().flatten()
        avg_dynamic_energy = float(dynamic_energies.mean()) if dynamic_energies.size > 0 else 0.0
        total_energy = float(self.graph.x.sum().cpu().item())
        
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
            'connection_count': self.graph.num_edges,
        }

    def _prepare_connection_growth_batch(self, batch_size):
        """Prepare a batch of connection growth candidates"""
        # Placeholder implementation
        return []

    def _prepare_cull_batch(self, batch_size):
        """Prepare a batch of connections to cull"""
        # Placeholder implementation
        return []

    def update(self):
        """Main update method with error handling and validation"""
        try:
            # Check memory usage
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._check_memory_usage()
                self._last_cleanup = time.time()

            # Process death queue
            if self.death_queue:
                self._remove_nodes(self.death_queue)
                self.death_queue.clear()

            # Process birth queue
            if self.birth_queue:
                for node_type, args in self.birth_queue:
                    self._add_nodes(1, node_type, **args)
                self.birth_queue.clear()

            # Update node energies
            start_time = time.time()
            self._update_energies()
            update_time = time.time() - start_time

            # Update metrics
            self.step_counter += 1
            
            # Update performance metrics
            if time.time() - self._last_metrics_update > self._metrics_update_interval:
                self._update_performance_metrics(update_time)
                self._last_metrics_update = time.time()

        except Exception as e:
            logger.error(f"Critical error in update: {str(e)}")
            self._attempt_recovery()

    def _check_memory_usage(self):
        """Monitor and manage memory usage"""
        if self.device == 'cuda':
            # Check CUDA memory
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            
            if allocated > 0.8 * torch.cuda.get_device_properties(0).total_memory / 1024**3:
                # Clear CUDA cache if memory usage is high
                torch.cuda.empty_cache()
                logger.warning("Cleared CUDA cache due to high memory usage")
        else:
            # Check CPU memory
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_usage = memory_info.rss / 1024**3  # GB
                
                if memory_usage > 4.0:  # 4GB threshold
                    self._cleanup_memory()
                    logger.warning("Cleaned up memory due to high usage")
            except ImportError:
                pass  # psutil not available

    def _cleanup_memory(self):
        """Clean up memory and caches"""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        self._last_cleanup = time.time()

    def _update_performance_metrics(self, update_time: float):
        """Update performance metrics"""
        # Update time metrics
        self.metrics['update_time'].append(update_time)
        if len(self.metrics['update_time']) > 100:
            self.metrics['update_time'].pop(0)
        
        # Node and edge counts
        self.metrics['node_count'].append(self.graph.num_nodes)
        self.metrics['edge_count'].append(self.graph.num_edges)
        if len(self.metrics['node_count']) > 100:
            self.metrics['node_count'].pop(0)
            self.metrics['edge_count'].pop(0)
        
        # Energy distribution
        if self.graph.num_nodes > 0:
            energy_std = self.graph.x.std().item()
            self.metrics['energy_distribution'].append(energy_std)
            if len(self.metrics['energy_distribution']) > 100:
                self.metrics['energy_distribution'].pop(0)
        
        # Memory usage
        if self.device == 'cuda':
            memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_usage = memory_info.rss / 1024**3  # GB
            except ImportError:
                memory_usage = 0.0
        
        self.metrics['memory_usage'].append(memory_usage)
        if len(self.metrics['memory_usage']) > 100:
            self.metrics['memory_usage'].pop(0)

    def get_performance_metrics(self):
        """Get performance metrics"""
        metrics = {}
        for key, values in self.metrics.items():
            if values:
                metrics[key] = {
                    'current': values[-1],
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        return metrics

    def _attempt_recovery(self):
        """Attempt to recover from a critical error"""
        try:
            # Clear queues
            self.death_queue.clear()
            self.birth_queue.clear()
            self.conn_growth_queue.clear()
            self.conn_candidate_queue.clear()

            # Reinitialize graph if needed
            if not hasattr(self, 'graph') or self.graph is None:
                self._init_empty_graph()

            logger.info("System recovery completed")
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")
            self.cleanup()

    def _add_nodes(self, n: int, node_type: int, parent_idx: Optional[int] = None):
        """Add nodes to the graph with proper attributes"""
        # Placeholder implementation
        pass

    def _remove_nodes(self, node_indices: List[int]):
        """Remove nodes from the graph"""
        # Placeholder implementation
        pass

    def _update_energies(self):
        """Update energy levels for all nodes using CUDA kernels or CPU fallback"""
        if self.graph.num_nodes == 0:
            return

        # Get node types and energies
        node_types = self.graph.node_type
        energies = self.graph.x

        # Process dynamic nodes
        dynamic_mask = (node_types == NODE_TYPE_DYNAMIC)
        if dynamic_mask.any():
            # Calculate energy decay for dynamic nodes
            try:
                out_deg = torch_geometric.utils.degree(self.graph.edge_index[0], num_nodes=self.graph.num_nodes)
                decay = out_deg[dynamic_mask] * CONNECTION_MAINTENANCE_COST
                energies[dynamic_mask] -= decay.unsqueeze(1)
            except Exception as e:
                logger.warning(f"Error calculating energy decay: {e}")

        # Apply energy caps and death threshold
        energies.clamp_(NODE_DEATH_THRESHOLD, NODE_ENERGY_CAP)

        # Emergency shutdown if energy distribution is too extreme
        if energies.std() > NODE_ENERGY_CAP * 0.5:  # If standard deviation exceeds 50% of cap
            logger.warning("Emergency shutdown triggered due to extreme energy distribution")
            self.suspended = True