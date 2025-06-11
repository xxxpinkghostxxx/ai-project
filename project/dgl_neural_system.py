import sys
print(sys.executable)
import dgl
import torch
import numpy as np
import time
print(sys.executable)
import threading
import queue
import logging

NODE_TYPE_SENSORY = 0
NODE_TYPE_DYNAMIC = 1
NODE_TYPE_WORKSPACE = 2
NODE_TYPE_HIGHWAY = 3  # New highway node type

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
NODE_SPAWN_THRESHOLD = 20.0
NODE_DEATH_THRESHOLD = 0.0
NODE_ENERGY_SPAWN_COST = 5.0
NODE_ENERGY_DECAY = 0.1
MAX_NODE_BIRTHS_PER_STEP = 1000  # Restore cap
MAX_CONN_BIRTHS_PER_STEP = 1000  # Restore cap
NODE_ENERGY_CAP = 244.0
CONNECTION_MAINTENANCE_COST = 0.02  # Energy lost per outgoing connection per node per step
TRANSMISSION_LOSS = 0.9  # Fraction of incoming energy actually received (simulate loss)

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
    def __init__(self, system, batch_size=25):
        super().__init__()
        self.system = system
        self.batch_size = batch_size
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
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
            'processing_time': 0.0
        }

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
                    try:
                        if task['type'] == 'grow':
                            # Prepare connection growth batch
                            batch = self.system._prepare_connection_growth_batch(self.batch_size)
                            self.result_queue.put({'type': 'grow', 'batch': batch})
                        elif task['type'] == 'cull':
                            # Prepare culling batch
                            batch = self.system._prepare_cull_batch(self.batch_size)
                            self.result_queue.put({'type': 'cull', 'batch': batch})
                        else:
                            logger.error(f"Unknown task type: {task['type']}")
                            self.result_queue.put({'type': 'error', 'error': f"Unknown task type: {task['type']}"})

                        # Update metrics
                        self._metrics['tasks_processed'] += 1
                        self._metrics['processing_time'] += time.time() - start_time
                        self._last_activity = time.time()
                        self._error_count = 0  # Reset error count on success

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
            return self._metrics.copy()

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
        return (self.is_alive() and 
                not self.stop_event.is_set() and 
                self._error_count < self._max_retries)

class DGLNeuralSystem:
    def __init__(self, sensory_width, sensory_height, n_dynamic, workspace_size=(16, 16), device='cpu'):
        self.device = device
        self.sensory_width = sensory_width
        self.sensory_height = sensory_height
        self.n_sensory_target = sensory_width * sensory_height
        self.n_dynamic_target = n_dynamic
        self.workspace_size = workspace_size
        self.n_workspace_target = workspace_size[0] * workspace_size[1]
        self.n_sensory = 0
        self.n_dynamic = 0
        self.n_workspace = 0
        self.n_total = 0
        self._init_empty_graph()
        # --- Metrics ---
        self.node_births = 0
        self.node_deaths = 0
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
        self.death_queue = []
        self.birth_queue = []
        self.conn_growth_queue = []
        self.conn_candidate_queue = []
        # --- Memory management ---
        self._memory_tracker = {
            'peak_nodes': 0,
            'peak_edges': 0,
            'last_cleanup': time.time(),
            'cleanup_interval': 60.0  # Cleanup every 60 seconds
        }
        self._connection_worker = None

    def cleanup(self):
        """Clean up resources and free memory"""
        try:
            # Stop connection worker if running
            if self._connection_worker is not None:
                self._connection_worker.stop()
                self._connection_worker.clear_queues()
                self._connection_worker = None

            # Clear queues
            self.death_queue.clear()
            self.birth_queue.clear()
            self.conn_growth_queue.clear()
            self.conn_candidate_queue.clear()

            # Clear graph data
            if hasattr(self, 'g'):
                self.g = None

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache if using GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            print("System cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def _check_memory_usage(self):
        """Check memory usage and trigger cleanup if needed"""
        current_time = time.time()
        if current_time - self._memory_tracker['last_cleanup'] > self._memory_tracker['cleanup_interval']:
            # Update peak memory usage
            if hasattr(self, 'g'):
                self._memory_tracker['peak_nodes'] = max(
                    self._memory_tracker['peak_nodes'],
                    self.g.number_of_nodes()
                )
                self._memory_tracker['peak_edges'] = max(
                    self._memory_tracker['peak_edges'],
                    self.g.number_of_edges()
                )

            # Check if we need cleanup
            if (self._memory_tracker['peak_nodes'] > self.n_total * 2 or
                self._memory_tracker['peak_edges'] > self.g.number_of_edges() * 2):
                self.cleanup()
                self._memory_tracker['last_cleanup'] = current_time

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def _init_empty_graph(self):
        # Start with an empty graph and all required node features
        self.g = dgl.graph(([], []), num_nodes=0, device=self.device)
        device = self.device
        self.g.ndata['energy'] = torch.zeros((0, 1), device=device)
        self.g.ndata['node_type'] = torch.zeros((0,), dtype=torch.int64, device=device)
        self.g.ndata['pos'] = torch.zeros((0, 2), device=device)
        self.g.ndata['dynamic_subtype'] = torch.zeros((0,), dtype=torch.int64, device=device)
        self.g.ndata['dynamic_subtype2'] = torch.zeros((0,), dtype=torch.int64, device=device)
        self.g.ndata['dynamic_subtype3'] = torch.zeros((0,), dtype=torch.int64, device=device)
        self.g.ndata['dynamic_subtype4'] = torch.zeros((0,), dtype=torch.int64, device=device)
        self.g.ndata['max_connections'] = torch.zeros((0,), dtype=torch.int64, device=device)
        self.g.ndata['velocity'] = torch.zeros((0, 2), device=self.device)
        self.grid_width = max(self.sensory_width, self.workspace_size[0])
        self.grid_height = max(self.sensory_height, self.workspace_size[1])

    def _get_unoccupied_grid_positions(self, n, exclude_mask=None, grid_width=None, grid_height=None, min_dist=0.4, fast_unique_grid=False):
        width = grid_width if grid_width is not None else self.sensory_width
        height = grid_height if grid_height is not None else self.sensory_height
        # Fallback to fast_unique_grid if the grid is very large
        if fast_unique_grid or width * height > 10000 or n > 1000:
            # Precompute all possible grid cells, shuffle, and assign
            all_cells = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).reshape(-1,2)
            np.random.shuffle(all_cells)
            if n > len(all_cells):
                n = len(all_cells)
            selected = all_cells[:n]
            return torch.from_numpy(selected).float().to(self.device), True
        g = self.g if hasattr(self, 'g') else None
        if g is not None:
            pos = g.ndata['pos'].detach().cpu()
            if exclude_mask is not None:
                pos = pos[exclude_mask.cpu()]
            occupied = pos.numpy() if len(pos) > 0 else np.zeros((0,2))
        else:
            occupied = np.zeros((0,2))
        all_cells = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).reshape(-1,2)
        if occupied.shape[0] > 0:
            dists = np.linalg.norm(all_cells[None, :, :] - occupied[:, None, :], axis=2)
            min_dists = dists.min(axis=0)
            mask = min_dists >= min_dist
            free_cells = all_cells[mask]
        else:
            free_cells = all_cells
        np.random.shuffle(free_cells)
        if len(free_cells) == 0:
            width = int(width * 1.1) + 1
            height = int(height * 1.1) + 1
            self.grid_width = width
            self.grid_height = height
            return torch.zeros((0,2), dtype=torch.float32, device=self.device), False
        selected = []
        if n == 1:
            selected = [free_cells[0]]
        else:
            selected = [free_cells[0]]
            if n > 1:
                selected_arr = np.array(selected)
                for cell in free_cells[1:]:
                    dists = np.linalg.norm(selected_arr - cell, axis=1)
                    if (dists >= min_dist).all():
                        selected.append(cell)
                        selected_arr = np.vstack([selected_arr, cell])
                    if len(selected) >= n:
                        break
        if len(selected) < n:
            width = int(width * 1.1) + 1
            height = int(height * 1.1) + 1
            self.grid_width = width
            self.grid_height = height
            return torch.from_numpy(np.array(selected)).float().to(self.device), False
        return torch.from_numpy(np.array(selected)).float().to(self.device), True

    def _init_graph(self):
        device = self.device
        n_sensory = self.n_sensory_target
        n_dynamic = self.n_dynamic_target
        n_workspace = self.n_workspace_target
        n_total = n_sensory + n_dynamic + n_workspace
        min_width = max(self.sensory_width, self.workspace_size[0])
        width = min_width
        height = int(np.ceil(n_total / width))
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
                    height = int(np.ceil(n_total / width))
                    continue
            n_dynamic_actual = 0
            if n_dynamic > 0:
                used_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
                used_mask[:n_sensory_actual] = True
                dyn_pos, ok = self._get_unoccupied_grid_positions(n_dynamic, exclude_mask=used_mask, grid_width=width, grid_height=height, min_dist=min_dist)
                pos[n_sensory_actual:n_sensory_actual+dyn_pos.shape[0], :] = dyn_pos
                n_dynamic_actual = dyn_pos.shape[0]
                node_types[n_sensory_actual:n_sensory_actual+n_dynamic_actual] = NODE_TYPE_DYNAMIC
                if not ok:
                    width = int(width * 1.1) + 1
                    height = int(np.ceil(n_total / width))
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
                    height = int(np.ceil(n_total / width))
                    continue
            break
        self.grid_width = width
        self.grid_height = height
        self.n_sensory = n_sensory_actual
        self.n_dynamic = n_dynamic_actual
        self.n_workspace = n_workspace_actual
        self.n_total = self.n_sensory + self.n_dynamic + self.n_workspace
        N = self.n_total
        # Node energies
        energies = torch.ones(N, 1, device=device)
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
        edge_src = []
        edge_dst = []
        edge_types = []
        edge_weights = []
        edge_caps = []
        edge_plastic_lrs = []
        edge_gate_thresholds = []
        edge_conn_subtype2 = []
        edge_conn_subtype3 = []
        # Sensory to dynamic
        num_sens_dyn = self.n_sensory
        if self.n_dynamic > 0 and num_sens_dyn > 0:
            src = torch.arange(self.n_sensory, device=device)
            dst = torch.randint(self.n_sensory, self.n_sensory+self.n_dynamic, (num_sens_dyn,), device=device)
            conn_types = torch.randint(0, 4, (num_sens_dyn,), device=device)
            weights = torch.where(conn_types == CONN_TYPE_EXCITATORY, torch.rand(num_sens_dyn, device=device)*0.15+0.05,
                      torch.where(conn_types == CONN_TYPE_INHIBITORY, -torch.rand(num_sens_dyn, device=device)*0.15-0.05,
                      torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_sens_dyn, device=device)*0.15+0.05,
                      torch.rand(num_sens_dyn, device=device)*0.4-0.2)))
            caps = torch.rand(num_sens_dyn, device=device)*0.9+0.1
            plastic_lrs = torch.where(conn_types == CONN_TYPE_PLASTIC, torch.rand(num_sens_dyn, device=device)*(PLASTIC_LEARNING_RATE_MAX-PLASTIC_LEARNING_RATE_MIN)+PLASTIC_LEARNING_RATE_MIN, torch.zeros(num_sens_dyn, device=device))
            gate_thresholds = torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_sens_dyn, device=device)*0.9+0.1, torch.zeros(num_sens_dyn, device=device))
            conn_subtype2 = torch.randint(0, 3, (num_sens_dyn,), device=device)
            conn_subtype3 = torch.randint(0, 3, (num_sens_dyn,), device=device)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_types.append(conn_types)
            edge_weights.append(weights)
            edge_caps.append(caps)
            edge_plastic_lrs.append(plastic_lrs)
            edge_gate_thresholds.append(gate_thresholds)
            edge_conn_subtype2.append(conn_subtype2)
            edge_conn_subtype3.append(conn_subtype3)
        # Workspace to dynamic
        num_ws_dyn = self.n_workspace
        if self.n_dynamic > 0 and num_ws_dyn > 0:
            src = torch.arange(self.n_sensory+self.n_dynamic, self.n_total, device=device)
            dst = torch.randint(self.n_sensory, self.n_sensory+self.n_dynamic, (num_ws_dyn,), device=device)
            conn_types = torch.randint(0, 4, (num_ws_dyn,), device=device)
            weights = torch.where(conn_types == CONN_TYPE_EXCITATORY, torch.rand(num_ws_dyn, device=device)*0.15+0.05,
                      torch.where(conn_types == CONN_TYPE_INHIBITORY, -torch.rand(num_ws_dyn, device=device)*0.15-0.05,
                      torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_ws_dyn, device=device)*0.15+0.05,
                      torch.rand(num_ws_dyn, device=device)*0.4-0.2)))
            caps = torch.rand(num_ws_dyn, device=device)*0.9+0.1
            plastic_lrs = torch.where(conn_types == CONN_TYPE_PLASTIC, torch.rand(num_ws_dyn, device=device)*(PLASTIC_LEARNING_RATE_MAX-PLASTIC_LEARNING_RATE_MIN)+PLASTIC_LEARNING_RATE_MIN, torch.zeros(num_ws_dyn, device=device))
            gate_thresholds = torch.where(conn_types == CONN_TYPE_GATED, torch.rand(num_ws_dyn, device=device)*0.9+0.1, torch.zeros(num_ws_dyn, device=device))
            conn_subtype2 = torch.randint(0, 3, (num_ws_dyn,), device=device)
            conn_subtype3 = torch.randint(0, 3, (num_ws_dyn,), device=device)
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
        # Build DGL graph
        src = torch.cat(edge_src) if edge_src else torch.tensor([], dtype=torch.int64, device=device)
        dst = torch.cat(edge_dst) if edge_dst else torch.tensor([], dtype=torch.int64, device=device)
        g = dgl.graph((src, dst), num_nodes=self.n_total, device=device)
        g.ndata['energy'] = energies[:self.n_total]
        g.ndata['node_type'] = node_types[:self.n_total]
        g.ndata['pos'] = pos[:self.n_total]
        g.ndata['dynamic_subtype'] = dynamic_subtypes[:self.n_total]
        g.ndata['dynamic_subtype2'] = dynamic_subtypes2[:self.n_total]
        g.ndata['dynamic_subtype3'] = dynamic_subtypes3[:self.n_total]
        g.ndata['dynamic_subtype4'] = dynamic_subtype4[:self.n_total]
        g.ndata['max_connections'] = torch.randint(5, 26, (self.n_total,), dtype=torch.int64, device=device)
        g.ndata['velocity'] = torch.zeros((self.n_total, 2), device=device)
        if edge_weights:
            g.edata['weight'] = torch.cat(edge_weights).unsqueeze(1)
            g.edata['energy_transfer_capacity'] = torch.cat(edge_caps).unsqueeze(1)
            g.edata['conn_type'] = torch.cat(edge_types).unsqueeze(1)
            g.edata['plastic_lr'] = torch.cat(edge_plastic_lrs).unsqueeze(1)
            g.edata['gate_threshold'] = torch.cat(edge_gate_thresholds).unsqueeze(1)
            g.edata['conn_subtype2'] = torch.cat(edge_conn_subtype2)
            g.edata['conn_subtype3'] = torch.cat(edge_conn_subtype3)
        self.g = g
        self.last_update_time = time.time()

    def to(self, device):
        self.device = device
        self.g = self.g.to(device)
        return self

    def summary(self):
        print(f"Nodes: {self.g.num_nodes()} (sensory: {self.n_sensory}, dynamic: {self.n_dynamic}, workspace: {self.n_workspace})")
        print(f"Edges: {self.g.num_edges()}")
        print(f"Node features: {list(self.g.ndata.keys())}")
        print(f"Edge features: {list(self.g.edata.keys())}")

    def update(self):
        """Main update method with error handling and validation"""
        try:
            # Check memory usage
            self._check_memory_usage()

            # Validate system state
            if not hasattr(self, 'g') or self.g is None:
                raise RuntimeError("Graph not initialized")

            # Validate node counts
            if self.g.number_of_nodes() != self.n_total:
                print(f"Warning: Node count mismatch. Expected {self.n_total}, got {self.g.number_of_nodes()}")
                self.n_total = self.g.number_of_nodes()

            # Process death queue
            if self.death_queue:
                try:
                    self._remove_nodes(self.death_queue)
                    self.death_queue.clear()
                except Exception as e:
                    print(f"Error processing death queue: {str(e)}")

            # Process birth queue
            if self.birth_queue:
                try:
                    for node_type, args in self.birth_queue:
                        self._add_nodes(1, node_type, **args)
                    self.birth_queue.clear()
                except Exception as e:
                    print(f"Error processing birth queue: {str(e)}")

            # Update node energies
            try:
                self._update_energies()
            except Exception as e:
                print(f"Error updating energies: {str(e)}")

            # Process connection worker results
            try:
                self.apply_connection_worker_results()
            except Exception as e:
                print(f"Error applying connection worker results: {str(e)}")

            # Update metrics
            self.step_counter += 1

        except Exception as e:
            print(f"Critical error in update: {str(e)}")
            # Attempt recovery
            self._attempt_recovery()

    def _attempt_recovery(self):
        """Attempt to recover from a critical error"""
        try:
            # Stop connection worker
            if self._connection_worker is not None:
                self._connection_worker.stop()
                self._connection_worker = None

            # Clear queues
            self.death_queue.clear()
            self.birth_queue.clear()
            self.conn_growth_queue.clear()
            self.conn_candidate_queue.clear()

            # Reinitialize graph if needed
            if not hasattr(self, 'g') or self.g is None:
                self._init_empty_graph()

            # Restart connection worker
            self.start_connection_worker()

            print("System recovery completed")
        except Exception as e:
            print(f"Recovery failed: {str(e)}")
            # If recovery fails, we should probably stop the system
            self.cleanup()

    def update_sensory_nodes(self, sensory_input):
        """Update sensory nodes with input validation"""
        try:
            # Validate input
            if not isinstance(sensory_input, (np.ndarray, torch.Tensor)):
                raise TypeError("sensory_input must be numpy array or torch tensor")

            if sensory_input.shape != (self.sensory_height, self.sensory_width):
                raise ValueError(f"sensory_input shape must be ({self.sensory_height}, {self.sensory_width})")

            # Convert to tensor if needed
            if isinstance(sensory_input, np.ndarray):
                sensory_input = torch.from_numpy(sensory_input).to(self.device)

            # Get sensory nodes
            sensory_mask = (self.g.ndata['node_type'] == NODE_TYPE_SENSORY)
            if not sensory_mask.any():
                return

            # Update energies
            self.g.ndata['energy'][sensory_mask] = sensory_input.flatten().unsqueeze(1)

        except Exception as e:
            print(f"Error updating sensory nodes: {str(e)}")
            # Don't raise the exception, just log it and continue

    @staticmethod
    def _validate_node_type(node_type):
        """Validate node type"""
        valid_types = [NODE_TYPE_SENSORY, NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE, NODE_TYPE_HIGHWAY]
        if node_type not in valid_types:
            raise ValueError(f"Invalid node type: {node_type}. Must be one of {valid_types}")

    @staticmethod
    def _validate_subtype(subtype, subtype_slot):
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

    def _update_energies(self):
        """Update energy levels for all nodes"""
        g = self.g
        node_types = g.ndata['node_type']
        dynamic_mask = (node_types == NODE_TYPE_DYNAMIC)

        # Batch process energy updates for better performance
        if dynamic_mask.sum() > 0:
            # Calculate energy decay for dynamic nodes (vectorized)
            out_deg = g.out_degrees().float().to(self.device)
            decay = out_deg[dynamic_mask] * CONNECTION_MAINTENANCE_COST
            g.ndata['energy'][dynamic_mask] -= decay.unsqueeze(1)

            # Calculate energy gain from outgoing connections (vectorized)
            src, dst = g.edges()
            weights = g.edata['weight'].squeeze()
            conn_subtype3 = g.edata.get('conn_subtype3', torch.zeros_like(weights, dtype=torch.int64))
            parent = g.ndata.get('parent', torch.full((g.num_nodes(),), -1, dtype=torch.int64, device=g.device))

            # Vectorized allowed connections calculation
            is_parent = (parent[dst] == src)
            allowed = (
                (conn_subtype3 == CONN_SUBTYPE3_FREE_FLOW) |
                ((conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_OUT) & is_parent) |
                ((conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_IN) & (~is_parent))
            )

            # Vectorized energy transfer
            src_allowed = src[allowed]
            weights_allowed = weights[allowed]
            outgoing_energy = torch.zeros(g.num_nodes(), device=g.device)
            outgoing_energy.index_add_(0, src_allowed, weights_allowed)
            g.ndata['energy'][:,0] += 0.1 * outgoing_energy

        # Batch process node type updates
        for node_type, energy_field in [
            (NODE_TYPE_SENSORY, 'sens_energy'),
            (NODE_TYPE_WORKSPACE, 'ws_energy'),
            (NODE_TYPE_HIGHWAY, 'hw_energy')
        ]:
            mask = (node_types == node_type)
            if mask.sum() > 0:
                g.ndata['energy'][mask] = torch.zeros_like(g.ndata['energy'][mask])
                g.update_all(
                    message_func=dgl.function.copy_u(energy_field, 'm'),
                    reduce_func=dgl.function.sum('m', 'energy_in')
                )
                g.ndata['energy'] += g.ndata['energy_in']
                g.ndata.pop(energy_field)
                g.ndata.pop('energy_in')

        # Apply energy caps and death threshold
        g.ndata['energy'].clamp_(NODE_DEATH_THRESHOLD, NODE_ENERGY_CAP)

        # Kill nodes that are below death threshold
        dead_nodes = (g.ndata['energy'] <= NODE_DEATH_THRESHOLD).squeeze()
        if dead_nodes.any():
            self._remove_nodes(dead_nodes)

    def get_metrics(self):
        g = self.g
        node_types = g.ndata['node_type']
        n_sensory = int((node_types == NODE_TYPE_SENSORY).sum().cpu().item())
        n_dynamic = int((node_types == NODE_TYPE_DYNAMIC).sum().cpu().item())
        n_workspace = int((node_types == NODE_TYPE_WORKSPACE).sum().cpu().item())
        dynamic_energies = g.ndata['energy'][node_types == NODE_TYPE_DYNAMIC].cpu().numpy().flatten()
        avg_dynamic_energy = float(dynamic_energies.mean()) if dynamic_energies.size > 0 else 0.0
        total_energy = float(g.ndata['energy'].sum().cpu().item())
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
            'connection_count': g.num_edges(),
        }

    def _guarantee_minimum_connections(self, node_indices=None, batch_size=500):
        """
        For each node in node_indices (or all nodes if None), ensure it has at least one outgoing connection.
        Adds connections in batches of batch_size.
        Returns True if all nodes are satisfied.
        """
        g = self.g
        device = self.device
        if node_indices is None:
            node_indices = torch.arange(g.num_nodes(), device=device)
        out_deg = g.out_degrees(node_indices)
        no_conn = node_indices[out_deg == 0]
        if len(no_conn) == 0:
            return True  # All satisfied
        # For each, add a connection to a valid target (sensory, workspace, or dynamic, not self)
        targets = torch.arange(g.num_nodes(), device=device)
        for i in range(0, len(no_conn), batch_size):
            batch = no_conn[i:i+batch_size]
            src_list = []
            dst_list = []
            for src in batch:
                # Exclude self
                valid_targets = targets[targets != src]
                # Prefer workspace, then dynamic, then sensory
                ws_mask = (g.ndata['node_type'][valid_targets] == NODE_TYPE_WORKSPACE)
                dyn_mask = (g.ndata['node_type'][valid_targets] == NODE_TYPE_DYNAMIC)
                sens_mask = (g.ndata['node_type'][valid_targets] == NODE_TYPE_SENSORY)
                if ws_mask.any():
                    dst = valid_targets[ws_mask][torch.randint(0, ws_mask.sum(), (1,)).item()]
                elif dyn_mask.any():
                    dst = valid_targets[dyn_mask][torch.randint(0, dyn_mask.sum(), (1,)).item()]
                elif sens_mask.any():
                    dst = valid_targets[sens_mask][torch.randint(0, sens_mask.sum(), (1,)).item()]
                else:
                    continue
                src_list.append(src)
                dst_list.append(dst)
            if src_list:
                num_new = len(src_list)
                conn_types = torch.randint(0, 4, (num_new,), device=device)
                edge_weights = torch.where(conn_types == 0, torch.rand(num_new, device=device)*0.15+0.05,
                              torch.where(conn_types == 1, -torch.rand(num_new, device=device)*0.15-0.05,
                              torch.where(conn_types == 2, torch.rand(num_new, device=device)*0.15+0.05,
                              torch.rand(num_new, device=device)*0.4-0.2)))
                energy_caps = torch.rand(num_new, device=device)*0.9+0.1
                plastic_lrs = torch.where(conn_types == 3, torch.rand(num_new, device=device)*(0.05-0.001)+0.001, torch.zeros(num_new, device=device))
                gate_thresholds = torch.where(conn_types == 2, torch.rand(num_new, device=device)*0.9+0.1, torch.zeros(num_new, device=device))
                g.add_edges(
                    torch.tensor(src_list, device=device, dtype=torch.int64),
                    torch.tensor(dst_list, device=device, dtype=torch.int64),
                    data={
                        'weight': edge_weights.unsqueeze(1),
                        'energy_transfer_capacity': energy_caps.unsqueeze(1),
                        'conn_type': conn_types.unsqueeze(1),
                        'plastic_lr': plastic_lrs.unsqueeze(1),
                        'gate_threshold': gate_thresholds.unsqueeze(1)
                    }
                )
        # Recheck if all are satisfied
        out_deg = g.out_degrees(node_indices)
        return (out_deg > 0).all().item() 

    def _cull_invalid_connections(self):
        """
        Remove any edges that do not follow connection rules (directionality, node type constraints, etc).
        Returns the number of edges removed.
        """
        g = self.g
        src, dst = g.edges()
        conn_subtype3 = g.edata['conn_subtype3'] if 'conn_subtype3' in g.edata else torch.zeros(g.num_edges(), dtype=torch.int64, device=g.device)
        node_type = g.ndata['node_type']
        parent = g.ndata['parent'] if 'parent' in g.ndata else torch.full((g.num_nodes(),), -1, dtype=torch.int64, device=g.device)
        is_parent = (parent[dst] == src)
        # Valid if:
        # - FREE_FLOW
        # - ONE_WAY_OUT and src is parent of dst
        # - ONE_WAY_IN and src is NOT parent of dst
        valid_direction = (
            (conn_subtype3 == CONN_SUBTYPE3_FREE_FLOW) |
            ((conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_OUT) & is_parent) |
            ((conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_IN) & (~is_parent))
        )
        src_type = node_type[src]
        dst_type = node_type[dst]
        # Dynamic nodes can always connect to each other
        dyn_dyn = (src_type == NODE_TYPE_DYNAMIC) & (dst_type == NODE_TYPE_DYNAMIC)
        # No sensory->sensory, workspace->workspace
        not_same_type = (src_type != dst_type) | dyn_dyn
        # Sensory nodes cannot connect to workspace nodes
        not_sens_to_ws = ~((src_type == NODE_TYPE_SENSORY) & (dst_type == NODE_TYPE_WORKSPACE))
        # Dynamic nodes can only connect to one type (sensory or workspace, not both, but can always connect to dynamic)
        to_remove_dyn = torch.zeros_like(src, dtype=torch.bool)
        dyn_src_mask = (src_type == NODE_TYPE_DYNAMIC)
        dyn_src_indices = src[dyn_src_mask]
        dyn_dst_types = dst_type[dyn_src_mask]
        if len(dyn_src_indices) > 0:
            unique_dyn, inverse = torch.unique(dyn_src_indices, return_inverse=True)
            for i, dyn in enumerate(unique_dyn):
                mask = (dyn_src_indices == dyn)
                types = dyn_dst_types[mask]
                has_ws = (types == NODE_TYPE_WORKSPACE).any()
                has_sens = (types == NODE_TYPE_SENSORY).any()
                if has_ws and has_sens:
                    # Prefer workspace, remove sensory connections (but keep dynamic)
                    remove_mask = mask & (dyn_dst_types == NODE_TYPE_SENSORY)
                    to_remove_dyn[dyn_src_mask.nonzero(as_tuple=True)[0][remove_mask]] = True
        valid = valid_direction & not_same_type & not_sens_to_ws & (~to_remove_dyn)
        to_remove = (~valid).nonzero(as_tuple=True)[0]
        if len(to_remove) > 0:
            g.remove_edges(to_remove)
        return len(to_remove)

    def start_connection_worker(self, batch_size=25):
        self.connection_worker = ConnectionWorker(self, batch_size=batch_size)
        self.connection_worker.start()
        self._pending_edge_adds = []
        self._pending_edge_removes = []

    def stop_connection_worker(self):
        if hasattr(self, 'connection_worker'):
            self.connection_worker.stop()

    def queue_connection_growth(self):
        if hasattr(self, 'connection_worker'):
            self.connection_worker.task_queue.put({'type': 'grow'})

    def queue_cull(self):
        if hasattr(self, 'connection_worker'):
            self.connection_worker.task_queue.put({'type': 'cull'})

    def _prepare_connection_growth_batch(self, batch_size):
        # Prepare a batch of connection growth (do not modify graph here)
        # Return a list of (src, dst, subtype3) tuples
        if not self.conn_growth_queue:
            return []
        batch = self.conn_growth_queue[:batch_size]
        self.conn_growth_queue = self.conn_growth_queue[batch_size:]
        return batch

    def _prepare_cull_batch(self, batch_size):
        # Prepare a batch of edge indices to remove (do not modify graph here)
        g = self.g
        src, dst = g.edges()
        conn_subtype3 = g.edata['conn_subtype3'] if 'conn_subtype3' in g.edata else torch.zeros(g.num_edges(), dtype=torch.int64, device=g.device)
        node_type = g.ndata['node_type']
        parent = g.ndata['parent'] if 'parent' in g.ndata else torch.full((g.num_nodes(),), -1, dtype=torch.int64, device=g.device)
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
            unique_dyn, inverse = torch.unique(dyn_src_indices, return_inverse=True)
            for i, dyn in enumerate(unique_dyn):
                mask = (dyn_src_indices == dyn)
                types = dyn_dst_types[mask]
                has_ws = (types == NODE_TYPE_WORKSPACE).any()
                has_sens = (types == NODE_TYPE_SENSORY).any()
                if has_ws and has_sens:
                    remove_mask = mask & (dyn_dst_types == NODE_TYPE_SENSORY)
                    to_remove_dyn[dyn_src_mask.nonzero(as_tuple=True)[0][remove_mask]] = True
        valid = valid_direction & not_same_type & not_sens_to_ws & (~to_remove_dyn)
        to_remove = (~valid).nonzero(as_tuple=True)[0]
        # Only return a batch
        return to_remove[:batch_size].cpu().tolist()

    def apply_connection_worker_results(self):
        # Call this in the main thread (main loop) to apply edge adds/removes
        if not hasattr(self, 'connection_worker'):
            return
        while not self.connection_worker.result_queue.empty():
            result = self.connection_worker.result_queue.get()
            if result['type'] == 'grow':
                # Add edges in main thread, using connection rules
                g = self.g
                device = self.device
                to_add = result['batch']
                if not to_add:
                    continue
                src_list, dst_list, subtype3_list = zip(*to_add)
                valid_src = []
                valid_dst = []
                valid_subtype3 = []
                node_type = g.ndata['node_type']
                parent = g.ndata['parent']
                for src, dst, subtype3 in zip(src_list, dst_list, subtype3_list):
                    s_type = int(node_type[src].item())
                    d_type = int(node_type[dst].item())
                    # No sensory->sensory, workspace->workspace, sensory->workspace
                    if (s_type == NODE_TYPE_SENSORY and d_type == NODE_TYPE_SENSORY) or \
                       (s_type == NODE_TYPE_WORKSPACE and d_type == NODE_TYPE_WORKSPACE) or \
                       (s_type == NODE_TYPE_SENSORY and d_type == NODE_TYPE_WORKSPACE):
                        continue
                    # Dynamic can only connect to one of sensory or workspace (not both), but always to dynamic
                    if s_type == NODE_TYPE_DYNAMIC:
                        out_edges = g.out_edges(src)[1]
                        out_types = node_type[out_edges] if len(out_edges) > 0 else torch.tensor([], dtype=torch.int64, device=device)
                        has_ws = (out_types == NODE_TYPE_WORKSPACE).any().item()
                        has_sens = (out_types == NODE_TYPE_SENSORY).any().item()
                        if d_type == NODE_TYPE_WORKSPACE and has_sens:
                            continue
                        if d_type == NODE_TYPE_SENSORY and has_ws:
                            continue
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
                g.add_edges(
                    torch.tensor(valid_src, device=device, dtype=torch.int64),
                    torch.tensor(valid_dst, device=device, dtype=torch.int64),
                    data={
                        'weight': edge_weights.unsqueeze(1),
                        'energy_transfer_capacity': energy_caps.unsqueeze(1),
                        'conn_type': conn_types.unsqueeze(1),
                        'plastic_lr': plastic_lrs.unsqueeze(1),
                        'gate_threshold': gate_thresholds.unsqueeze(1),
                        'conn_subtype3': torch.tensor(valid_subtype3, device=device, dtype=torch.int64)
                    }
                )
                self.conn_births += num_new
                self.total_conn_births += num_new
            elif result['type'] == 'cull':
                # Remove edges in main thread
                g = self.g
                eids = torch.tensor(result['batch'], dtype=torch.long, device=g.device)
                # Filter out invalid eids
                valid_eids = eids[(eids >= 0) & (eids < g.num_edges())]
                invalid_eids = eids[(eids < 0) | (eids >= g.num_edges())]
                if len(invalid_eids) > 0:
                    print(f"[WARNING] Tried to remove invalid edge IDs: {invalid_eids.cpu().numpy().tolist()}")
                if len(valid_eids) > 0:
                    g.remove_edges(valid_eids)
            self.connection_worker.result_queue.task_done()

    def _add_nodes(self, n, node_type, parent_idx=None):
        g = self.g
        device = self.device
        if n == 0:
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
            n_highway = highway_mask.sum().item()
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
                g.add_nodes(
                    n_dynamic,
                    data={
                        'energy': torch.zeros(n_dynamic, 1, device=device),
                        'node_type': torch.full((n_dynamic,), NODE_TYPE_DYNAMIC, dtype=torch.int64, device=device),
                        'pos': pos_new,
                        'velocity': torch.zeros(n_dynamic, 2, device=device),
                        'dynamic_subtype': subtypes,
                        'dynamic_subtype2': subtypes2,
                        'dynamic_subtype3': subtypes3,
                        'dynamic_subtype4': subtype4,
                        'max_connections': max_conns,
                        'phase_offset': phase_offset,
                        'parent': parent_arr[dynamic_indices]
                    }
                )

            # Add highway nodes
            if n_highway > 0:
                highway_indices = torch.where(highway_mask)[0]
                pos_new = torch.rand(n_highway, 2, device=device) * 100
                g.add_nodes(
                    n_highway,
                    data={
                        'energy': torch.zeros(n_highway, 1, device=device),
                        'node_type': torch.full((n_highway,), NODE_TYPE_HIGHWAY, dtype=torch.int64, device=device),
                        'pos': pos_new,
                        'velocity': torch.zeros(n_highway, 2, device=device),
                        'max_connections': torch.full((n_highway,), 1000, dtype=torch.int64, device=device),  # Highway nodes have no connection limit
                        'parent': parent_arr[highway_indices]
                    }
                )
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
                g.add_nodes(
                    n,
                    data={
                        'energy': torch.zeros(n, 1, device=device),
                        'node_type': torch.full((n,), node_type, dtype=torch.int64, device=device),
                        'pos': pos_new,
                        'velocity': torch.zeros(n, 2, device=device),
                        'dynamic_subtype': subtypes,
                        'dynamic_subtype2': subtypes2,
                        'dynamic_subtype3': subtypes3,
                        'dynamic_subtype4': subtype4,
                        'max_connections': max_conns,
                        'phase_offset': phase_offset,
                        'parent': parent_arr
                    }
                )
            else:
                g.add_nodes(
                    n,
                    data={
                        'energy': torch.zeros(n, 1, device=device),
                        'node_type': torch.full((n,), node_type, dtype=torch.int64, device=device),
                        'pos': pos_new,
                        'velocity': torch.zeros(n, 2, device=device),
                        'parent': parent_arr
                    }
                )

    def _event_driven_transfer(self):
        """Optimized event-driven energy transfer"""
        g = self.g
        device = self.device
        node_type = g.ndata['node_type']
        energy = g.ndata['energy']

        # Vectorized highway node normalization
        highway_mask = (node_type == NODE_TYPE_HIGHWAY)
        if highway_mask.sum() > 0:
            highway_energies = energy[highway_mask]
            avg_highway_energy = highway_energies.mean()
            energy_diff = avg_highway_energy - highway_energies
            transfer_amount = energy_diff * 0.1
            energy[highway_mask] += transfer_amount

        # Batch process edge transfers
        src, dst = g.edges()
        src_type = node_type[src]
        dst_type = node_type[dst]
        weights = g.edata['weight']
        energy_caps = g.edata['energy_transfer_capacity']

        # Vectorized transfer calculations
        src_energy = energy[src]
        dst_energy = energy[dst]
        transfer = src_energy * weights * energy_caps

        # Vectorized node type masks
        highway_src = (src_type == NODE_TYPE_HIGHWAY)
        highway_dst = (dst_type == NODE_TYPE_HIGHWAY)
        dynamic_dst = (dst_type == NODE_TYPE_DYNAMIC)

        # Vectorized pull calculations
        pull_mask = highway_src & dynamic_dst
        if pull_mask.any():
            dst_energy_needed = NODE_ENERGY_CAP - dst_energy[pull_mask]
            pull_amount = torch.minimum(
                src_energy[pull_mask] * 0.2,
                dst_energy_needed
            )
            energy[src[pull_mask]] -= pull_amount
            energy[dst[pull_mask]] += pull_amount

        # Vectorized normal transfer
        normal_mask = ~(highway_src | highway_dst)
        if normal_mask.any():
            energy[src[normal_mask]] -= transfer[normal_mask]
            energy[dst[normal_mask]] += transfer[normal_mask]

        # Vectorized highway transfer
        highway_mask = highway_src & highway_dst
        if highway_mask.any():
            transfer_amount = src_energy[highway_mask]
            energy[src[highway_mask]] -= transfer_amount
            energy[dst[highway_mask]] += transfer_amount

        # Apply energy caps and check for emergency shutdown
        energy.clamp_(NODE_DEATH_THRESHOLD, NODE_ENERGY_CAP)

        # Emergency shutdown if energy distribution is too extreme
        if energy.std() > NODE_ENERGY_CAP * 0.5:  # If standard deviation exceeds 50% of cap
            logger.warning("Emergency shutdown triggered due to extreme energy distribution")
            self.suspended = True
