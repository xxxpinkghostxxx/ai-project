from node import Node
from connection import Connection
import random
import numpy as np
import time
import config
from config import NODE_ENERGY_INIT_RANGE, NODE_ENERGY_SPAWN_COST, NODE_ENERGY_CONN_COST, CONN_ENERGY_TRANSFER_CAPACITY, WORKSPACE_SIZE, NODE_ENERGY_CAP
import utils
import concurrent.futures

class NeuralSystem:
    def __init__(self, width, height, initial_nodes=10, logger=None):
        # Initialize metrics FIRST
        self.node_births = 0
        self.node_deaths = 0
        self.conn_births = 0
        self.conn_deaths = 0
        self.total_energy_generated = 0.0
        self.total_energy_consumed = 0.0
        self.last_update_time = time.time()
        self.prune_counter = 0
        self.node_spawn_threshold = 14.0
        self.node_death_threshold = 8.0
        self.conn_maintenance_cost = 0.01
        self.logger = logger

        # Sensory node grid (one node per pixel in capture)
        self.sensory_nodes = np.array([[Node(is_sensory=True, node_type='sensory', pos=(x, y)) for x in range(width)] for y in range(height)], dtype=object)
        # Dynamic processing nodes
        self.processing_nodes = [Node(pos=(random.random(), random.random()), energy=utils.random_range(NODE_ENERGY_INIT_RANGE)) for _ in range(initial_nodes)]
        self.connections = []
        # Each sensory node connects to a random processing node (keep this behavior)
        for row in self.sensory_nodes:
            for s_node in row:
                dyn = random.choice(self.processing_nodes)
                weight = utils.random_range((0.05, 0.2))
                conn = Connection(s_node, dyn, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                s_node.connections.append(conn)
                dyn.incoming_connections.append(conn)
                self.connections.append(conn)
        # Workspace node grid (no workspace-to-workspace connections)
        self.workspace_nodes = np.array([[Node(is_sensory=False, node_type='workspace', pos=(x, y)) for x in range(WORKSPACE_SIZE[0])] for y in range(WORKSPACE_SIZE[1])], dtype=object)
        # For each processing node, connect to a random selection of any node (sensory, workspace, or other processing nodes)
        for proc in self.processing_nodes:
            # Determine if this node will be sensory- or workspace-connected (exclusive)
            has_sensory = any(getattr(c.destination, 'node_type', None) == 'sensory' for c in proc.connections)
            has_workspace = any(getattr(c.destination, 'node_type', None) == 'workspace' for c in proc.connections)
            for _ in range(5):  # e.g., 5 random connections per processing node
                # Enforce exclusivity
                if has_sensory:
                    node_type = random.choices(['sensory', 'processing'], weights=[0.5, 0.5])[0]
                elif has_workspace:
                    node_type = random.choices(['workspace', 'processing'], weights=[0.5, 0.5])[0]
                else:
                    node_type = random.choices(['sensory', 'workspace', 'processing'], weights=[0.3, 0.3, 0.4])[0]
                if node_type == 'sensory':
                    sy = random.randint(0, self.sensory_nodes.shape[0] - 1)
                    sx = random.randint(0, self.sensory_nodes.shape[1] - 1)
                    target = self.sensory_nodes[sy, sx]
                    has_sensory = True
                elif node_type == 'workspace':
                    wy = random.randint(0, self.workspace_nodes.shape[0] - 1)
                    wx = random.randint(0, self.workspace_nodes.shape[1] - 1)
                    target = self.workspace_nodes[wy, wx]
                    has_workspace = True
                else:  # processing
                    target = random.choice(self.processing_nodes)
                    if target == proc:
                        continue  # skip self-connection
                # Prevent sensory <-> workspace connections
                src_type = getattr(proc, 'node_type', None)
                dst_type = getattr(target, 'node_type', None)
                if (src_type == 'sensory' and dst_type == 'workspace') or (src_type == 'workspace' and dst_type == 'sensory'):
                    continue
                weight = utils.random_range((-1.0, 1.0))
                conn = Connection(proc, target, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                proc.connections.append(conn)
                target.incoming_connections.append(conn)
                self.connections.append(conn)
                # --- If connecting to workspace, give a free secondary connection ---
                if getattr(target, 'node_type', None) == 'workspace':
                    # Heavily bias against another workspace node
                    sec_type = random.choices(['sensory', 'workspace', 'processing'], weights=[0.7, 0.05, 0.25])[0]
                    while True:
                        if sec_type == 'sensory':
                            sy = random.randint(0, self.sensory_nodes.shape[0] - 1)
                            sx = random.randint(0, self.sensory_nodes.shape[1] - 1)
                            sec_target = self.sensory_nodes[sy, sx]
                        elif sec_type == 'workspace':
                            wy = random.randint(0, self.workspace_nodes.shape[0] - 1)
                            wx = random.randint(0, self.workspace_nodes.shape[1] - 1)
                            sec_target = self.workspace_nodes[wy, wx]
                        else:
                            sec_target = random.choice(self.processing_nodes)
                            if sec_target == proc:
                                continue
                        if sec_target != target and sec_target != proc:
                            break
                    # Prevent sensory <-> workspace connections
                    sec_src_type = getattr(proc, 'node_type', None)
                    sec_dst_type = getattr(sec_target, 'node_type', None)
                    if (sec_src_type == 'sensory' and sec_dst_type == 'workspace') or (sec_src_type == 'workspace' and sec_dst_type == 'sensory'):
                        continue
                    sec_weight = utils.random_range((-1.0, 1.0))
                    sec_conn = Connection(proc, sec_target, weight=sec_weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                    proc.connections.append(sec_conn)
                    sec_target.incoming_connections.append(sec_conn)
                    self.connections.append(sec_conn)
        self.spawn_random_nodes(100)
        self.node_births = initial_nodes + 100

        # --- Ensure every node has at least one connection (startup) ---
        # Sensory nodes: must connect to a processing node
        for row in self.sensory_nodes:
            for s_node in row:
                if not s_node.connections:
                    dyn = random.choice(self.processing_nodes)
                    weight = utils.random_range((0.05, 0.2))
                    conn = Connection(s_node, dyn, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                    s_node.connections.append(conn)
                    dyn.incoming_connections.append(conn)
                    self.connections.append(conn)
        # Workspace nodes: must connect to a processing node
        for row in self.workspace_nodes:
            for ws_node in row:
                if not ws_node.connections:
                    dyn = random.choice(self.processing_nodes)
                    weight = utils.random_range((0.05, 0.2))
                    conn = Connection(ws_node, dyn, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                    ws_node.connections.append(conn)
                    dyn.incoming_connections.append(conn)
                    self.connections.append(conn)
        # Processing nodes: must have at least one connection, following exclusivity
        for proc in self.processing_nodes:
            if not proc.connections:
                # Determine if this node will be sensory- or workspace-connected (exclusive)
                has_sensory = any(getattr(c.destination, 'node_type', None) == 'sensory' for c in proc.connections)
                has_workspace = any(getattr(c.destination, 'node_type', None) == 'workspace' for c in proc.connections)
                if has_sensory:
                    # Only connect to sensory or processing
                    node_type = random.choices(['sensory', 'processing'], weights=[0.5, 0.5])[0]
                elif has_workspace:
                    node_type = random.choices(['workspace', 'processing'], weights=[0.5, 0.5])[0]
                else:
                    node_type = random.choices(['sensory', 'workspace', 'processing'], weights=[0.3, 0.3, 0.4])[0]
                if node_type == 'sensory':
                    sy = random.randint(0, self.sensory_nodes.shape[0] - 1)
                    sx = random.randint(0, self.sensory_nodes.shape[1] - 1)
                    target = self.sensory_nodes[sy, sx]
                elif node_type == 'workspace':
                    wy = random.randint(0, self.workspace_nodes.shape[0] - 1)
                    wx = random.randint(0, self.workspace_nodes.shape[1] - 1)
                    target = self.workspace_nodes[wy, wx]
                else:
                    target = random.choice(self.processing_nodes)
                    if target == proc:
                        continue  # skip self-connection
                weight = utils.random_range((-1.0, 1.0))
                conn = Connection(proc, target, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                proc.connections.append(conn)
                target.incoming_connections.append(conn)
                self.connections.append(conn)

    def initialize_connections(self):
        # For each processing node, connect to 3 random sensory nodes and 2 other processing nodes
        for proc in self.processing_nodes:
            for _ in range(3):  # 3 sensory connections per processing node
                y = random.randint(0, len(self.workspace_nodes) - 1)
                x = random.randint(0, len(self.workspace_nodes[0]) - 1)
                conn = Connection(self.workspace_nodes[y][x], proc, weight=utils.random_range((0.05, 0.2)), energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                self.workspace_nodes[y][x].connections.append(conn)
                proc.incoming_connections.append(conn)
                self.connections.append(conn)
                self.conn_births += 1
            for _ in range(2):
                target = random.choice(self.processing_nodes)
                if target != proc:
                    conn = Connection(proc, target, weight=utils.random_range((-1.0, 1.0)), energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                    proc.connections.append(conn)
                    target.incoming_connections.append(conn)
                    self.connections.append(conn)
                    self.conn_births += 1

    def update_processing_nodes_gpu(self, batch_size=50000):
        import cupy as cp
        node_count = len(self.processing_nodes)
        energies = cp.array([n.energy for n in self.processing_nodes], dtype=cp.float32)
        n_conns = cp.array([len(getattr(n, 'connections', [])) for n in self.processing_nodes], dtype=cp.int32)
        # --- Base energy generation with connection-based scaling ---
        BASE_GEN = 0.05
        OPTIMAL_CONN = 5
        # Reverse: below optimal is bonus, above optimal is penalty
        scale = -cp.tanh((n_conns - OPTIMAL_CONN) / 2.0)
        # Range: scale ~ +1 (no conns) to -1 (many conns)
        # Shift to [0, 1.5]: 0.5 is neutral, >0.5 is bonus, <0.5 is penalty
        gen = BASE_GEN * (1.0 + 0.5 * scale)
        energies += gen
        # Write back in batches
        for start in range(0, node_count, batch_size):
            end = min(start + batch_size, node_count)
            for i in range(start, end):
                self.processing_nodes[i].energy = float(energies[i].get())

    def update(self):
        if self.logger:
            self.logger.info('[TEST] update() called')
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        timings = {}
        t0 = time.time()
        node_count = len(self.processing_nodes)
        # --- Use config values for all thresholds and costs (live reactivity) ---
        self.node_spawn_threshold = getattr(config, 'NODE_SPAWN_THRESHOLD', 1.0)
        self.node_death_threshold = getattr(config, 'NODE_DEATH_THRESHOLD', -20.0)
        self.conn_maintenance_cost = getattr(config, 'CONN_MAINTENANCE_COST', 0.01)
        self.prune_counter += 1
        if self.prune_counter % 20 == 0:
            print(f"[DEBUG] (start) Node count: {node_count} | Spawn threshold: {self.node_spawn_threshold:.2f} | Death threshold: {self.node_death_threshold:.2f}")
        # --- After sensory node updates, propagate sensory energy ---
        for row in self.sensory_nodes:
            for s_node in row:
                for conn in getattr(s_node, 'connections', []):
                    dst = getattr(conn, 'destination', None)
                    if dst is not None and hasattr(dst, 'energy'):
                        dst.energy = s_node.energy
        # --- Fully vectorized processing node and connection updates (optimized) ---
        t1 = time.time()
        with utils.profile_section('vectorized node+conn updates'):
            try:
                import cupy as cp
                import numba
                xp = cp if utils.has_cupy() and utils.get_array_module().__name__ == 'cupy' else np
                if self.logger:
                    self.logger.info(f'[PERF] Using {"GPU (CuPy)" if xp.__name__ == "cupy" else "CPU (NumPy)"} for full vectorized updates.')
                node_list = self.processing_nodes
                n_nodes = len(node_list)
                # --- Precompute all node energies and connection weights in arrays ---
                energies = xp.array([n.energy for n in node_list], dtype=xp.float32)
                n_conns = xp.array([len(getattr(n, 'connections', [])) for n in node_list], dtype=xp.int32)
                BASE_GEN = getattr(config, 'BASE_GEN', 0.05)
                OPTIMAL_CONN = getattr(config, 'OPTIMAL_CONN', 5)
                CONSUMPTION = 0.01  # or get from config
                scale = -xp.tanh((n_conns - OPTIMAL_CONN) / 2.0)
                gen = BASE_GEN * (1.0 + 0.5 * scale)
                energies += gen * dt
                energies -= CONSUMPTION * dt
                # --- Vectorized connection updates (optimized) ---
                num_conns = len(self.connections)
                if num_conns > 0:
                    node_to_idx = {node: i for i, node in enumerate(node_list)}
                    src_idx = xp.array([node_to_idx.get(c.source, -1) for c in self.connections], dtype=xp.int32)
                    dst_idx = xp.array([node_to_idx.get(c.destination, -1) for c in self.connections], dtype=xp.int32)
                    valid_mask = (src_idx >= 0) & (dst_idx >= 0)
                    src_idx = src_idx[valid_mask]
                    dst_idx = dst_idx[valid_mask]
                    weights = xp.array([c.weight for c in self.connections], dtype=xp.float32)[valid_mask]
                    src_vals = energies[src_idx]
                    dst_vals = energies[dst_idx]
                    activities = xp.abs(src_vals - dst_vals) * xp.abs(weights)
                    active_mask = activities > 0.1
                    transfer = xp.minimum(CONN_ENERGY_TRANSFER_CAPACITY * activities, src_vals)
                    transfer = xp.where(active_mask, transfer, 0)
                    # Batch update energies
                    if xp.__name__ == 'cupy':
                        try:
                            import cupyx
                            cupyx.scatter_add(energies, src_idx, -(transfer + self.conn_maintenance_cost))
                            cupyx.scatter_add(energies, dst_idx, (transfer - self.conn_maintenance_cost))
                        except Exception as e:
                            if self.logger:
                                self.logger.error(f'[PERF] cupyx.scatter_add failed: {e}. Falling back to CPU/NumPy for this step.')
                            energies_cpu = energies.get()
                            src_idx_cpu = src_idx.get()
                            dst_idx_cpu = dst_idx.get()
                            transfer_cpu = transfer.get()
                            np.subtract.at(energies_cpu, src_idx_cpu, transfer_cpu + self.conn_maintenance_cost)
                            np.add.at(energies_cpu, dst_idx_cpu, transfer_cpu - self.conn_maintenance_cost)
                            energies = cp.array(energies_cpu, dtype=cp.float32)
                    else:
                        np.subtract.at(energies, src_idx, transfer + self.conn_maintenance_cost)
                        np.add.at(energies, dst_idx, transfer - self.conn_maintenance_cost)
                ENERGY_CAP = getattr(config, 'NODE_ENERGY_CAP', 10.0)
                energies = xp.clip(energies, 0, ENERGY_CAP)
                # --- Write back to Python objects in one loop, using Numba if on CPU ---
                energies_cpu = energies.get() if xp.__name__ == 'cupy' else energies
                if xp.__name__ == 'cupy':
                    for i, node in enumerate(node_list):
                        node.energy = float(energies_cpu[i])
                else:
                    from numba import njit
                    @njit
                    def write_back(energies_cpu, node_energies):
                        for i, item in enumerate(energies_cpu):
                            node_energies[i] = item
                    node_energies = np.array([n.energy for n in node_list], dtype=np.float32)
                    write_back(energies_cpu, node_energies)
                    for i, node in enumerate(node_list):
                        node.energy = float(node_energies[i])
                # --- Update connection activities in one loop ---
                if num_conns > 0:
                    activities_cpu = activities.get() if xp.__name__ == 'cupy' else activities
                    valid_conns = [c for j, c in enumerate(self.connections) if valid_mask[j]]
                    for i, conn in enumerate(valid_conns):
                        conn.activity = float(activities_cpu[i])
                        conn.last_activity = float(activities_cpu[i])
                        conn.activity_history.append((time.time(), float(activities_cpu[i])))
                        if len(conn.activity_history) > 100:
                            conn.activity_history = conn.activity_history[-100:]
                if xp.__name__ == 'cupy' and self.logger:
                    try:
                        free, total = cp.cuda.runtime.memGetInfo()
                        self.logger.info(f'[PERF] GPU memory: {free/1e6:.2f}MB free / {total/1e6:.2f}MB total')
                    except Exception as e:
                        self.logger.warning(f'[PERF] Could not get GPU memory info: {e}')
            except Exception as e:
                if self.logger:
                    self.logger.error(f'[PERF] Vectorized update failed: {e}')
        t2 = time.time()
        timings['vectorized'] = t2 - t1
        # --- Periodically log nvidia-smi output for real-time GPU utilization ---
        if self.prune_counter % 100 == 0 and self.logger:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'], capture_output=True, text=True)
                self.logger.info(f'[PERF] nvidia-smi: {result.stdout.strip()}')
            except Exception as e:
                self.logger.warning(f'[PERF] Could not run nvidia-smi: {e}')
        if self.prune_counter % 20 == 0:
            energies = [n.energy for n in self.processing_nodes]
            if energies:
                print(f"[DEBUG] (after) Node count: {node_count} | Min energy: {min(energies):.2f} | Max energy: {max(energies):.2f} | Avg energy: {np.mean(energies):.2f}")
                print(f"[DEBUG] (after) Spawn threshold: {self.node_spawn_threshold:.2f} | Death threshold: {self.node_death_threshold:.2f}")
        # --- Parallel workspace node updates ---
        t3 = time.time()
        with utils.profile_section('workspace node updates'), concurrent.futures.ThreadPoolExecutor() as executor:
            for row in self.workspace_nodes:
                list(executor.map(lambda node: node.update(dt), row))
        t4 = time.time()
        timings['workspace'] = t4 - t3
        # --- Parallel sensory node updates ---
        t5 = time.time()
        with utils.profile_section('sensory node updates'), concurrent.futures.ThreadPoolExecutor() as executor:
            for row in self.sensory_nodes:
                executor.map(lambda node: node.update(), row)
        t6 = time.time()
        timings['sensory'] = t6 - t5
        # --- Prune dead nodes (batch with numpy mask) ---
        t7 = time.time()
        with utils.profile_section('prune dead nodes'):
            before = len(self.processing_nodes)
            dying_mask = np.array([getattr(n, '_marked_for_death', False) or n.energy <= self.node_death_threshold for n in self.processing_nodes])
            dying_nodes = [n for n, d in zip(self.processing_nodes, dying_mask) if d]
            for n in dying_nodes:
                try:
                    node_attrs = utils.extract_node_attrs(n)
                    if self.logger:
                        self.logger.info(f"[NODE DEATH] Node id={id(n)} type={getattr(n, 'node_type', '?')} pos={getattr(n, 'pos', '?')} reason=death attrs={node_attrs}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[NODE DEATH LOG ERROR] Could not log node death: {e}")
            dying_set = set(dying_nodes)
            self.connections = [c for c in self.connections if c.source not in dying_set and c.destination not in dying_set]
            for node in self.processing_nodes:
                node.connections = [c for c in node.connections if c.source not in dying_set and c.destination not in dying_set]
                node.incoming_connections = [c for c in node.incoming_connections if c.source not in dying_set and c.destination not in dying_set]
            self.processing_nodes = [n for n in self.processing_nodes if n not in dying_set]
            after = len(self.processing_nodes)
            self.node_deaths += before - after
            before_conn = len(self.connections)
            after_conn = len(self.connections)
            self.conn_deaths += before_conn - after_conn
        t8 = time.time()
        timings['prune'] = t8 - t7
        # --- Adaptive connection pruning (less frequent) ---
        t9 = time.time()
        with utils.profile_section('adaptive connection pruning'):
            if self.prune_counter % 100 == 0 and self.connections:
                sorted_conns = sorted(self.connections, key=lambda c: abs(c.weight))
                n_prune = max(1, len(sorted_conns) // 100)
                to_remove = set(sorted_conns[:n_prune])
                self.connections = [c for c in self.connections if c not in to_remove]
                for node in self.processing_nodes:
                    node.connections = [c for c in node.connections if c not in to_remove]
                    node.incoming_connections = [c for c in node.incoming_connections if c not in to_remove]
        t10 = time.time()
        timings['conn_prune'] = t10 - t9
        # --- Energy redistribution (less frequent) ---
        t11 = time.time()
        with utils.profile_section('energy redistribution'):
            if self.prune_counter % 100 == 0:
                for node in self.processing_nodes:
                    if node.energy > 0.8 * NODE_ENERGY_CAP and node.connections:
                        donation = 0.05 * node.energy
                        node.energy -= donation
                        neighbor = random.choice([c.destination for c in node.connections])
                        neighbor.energy += donation
        t12 = time.time()
        timings['energy_redistribution'] = t12 - t11
        # Node-driven growth
        t13 = time.time()
        with utils.profile_section('node-driven growth'):
            self.node_driven_growth()
        t14 = time.time()
        timings['growth'] = t14 - t13
        # --- Workspace-to-dynamic feedback: workspace nodes send energy to connected dynamic nodes if above threshold ---
        t15 = time.time()
        with utils.profile_section('workspace feedback'):
            for row in self.workspace_nodes:
                for ws_node in row:
                    if ws_node.energy > 0.5 * NODE_ENERGY_CAP:
                        for conn in getattr(ws_node, 'connections', []):
                            dst = getattr(conn, 'destination', None)
                            if getattr(dst, 'node_type', None) == 'dynamic' and dst.energy < NODE_ENERGY_CAP:
                                transfer = min(0.05 * ws_node.energy, NODE_ENERGY_CAP - dst.energy)
                                ws_node.energy -= transfer
                                dst.energy += transfer
        t16 = time.time()
        timings['ws_feedback'] = t16 - t15
        # --- Dynamic node energy pulse: if full, pulse 10% to connected nodes ---
        t17 = time.time()
        with utils.profile_section('dynamic node pulse'):
            for node in self.processing_nodes:
                if getattr(node, 'node_type', None) == 'dynamic' and node.energy >= 0.95 * NODE_ENERGY_CAP and node.connections:
                    pulse_amt = 0.10 * node.energy
                    per_conn = pulse_amt / len(node.connections)
                    for conn in node.connections:
                        dst = getattr(conn, 'destination', None)
                        if hasattr(dst, 'energy') and getattr(dst, 'node_type', None) != 'sensory':
                            transfer = min(per_conn, NODE_ENERGY_CAP - dst.energy, node.energy)
                            if transfer > 0:
                                node.energy -= transfer
                                dst.energy += transfer
        t18 = time.time()
        timings['dynamic_pulse'] = t18 - t17
        # --- Print profile report for this update ---
        utils.profile_report()
        # --- Print detailed timings for this update ---
        print(f"[PROFILE-UPDATE] vectorized: {timings['vectorized']:.4f}s | workspace: {timings['workspace']:.4f}s | sensory: {timings['sensory']:.4f}s | prune: {timings['prune']:.4f}s | conn_prune: {timings['conn_prune']:.4f}s | energy_redistribution: {timings['energy_redistribution']:.4f}s | growth: {timings['growth']:.4f}s | ws_feedback: {timings['ws_feedback']:.4f}s | dynamic_pulse: {timings['dynamic_pulse']:.4f}s | TOTAL: {t18-t0:.4f}s")

    def node_driven_growth(self):
        # --- Batch: Find eligible nodes for growth ---
        node_energies = np.array([n.energy for n in self.processing_nodes], dtype=np.float32)
        eligible_spawn = node_energies > self.node_spawn_threshold
        eligible_conn = node_energies > getattr(config, 'CONN_SPAWN_THRESHOLD', 1.0)
        max_births = getattr(config, 'MAX_NODE_BIRTHS_PER_STEP', 20)
        max_conn_births = getattr(config, 'MAX_CONN_BIRTHS_PER_STEP', 10)
        # --- Node births ---
        spawn_indices = np.where(eligible_spawn)[0]
        if len(spawn_indices) > max_births:
            spawn_indices = np.random.choice(spawn_indices, max_births, replace=False)
        new_nodes = []
        for idx in spawn_indices:
            node = self.processing_nodes[idx]
            if node.pos is not None:
                dx, dy = (np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05))
                new_pos = (min(max(node.pos[0] + dx, 0), 1), min(max(node.pos[1] + dy, 0), 1))
            else:
                new_pos = (np.random.random(), np.random.random())
            initial_energy = NODE_ENERGY_SPAWN_COST + 0.1 * NODE_ENERGY_CAP
            new_node = Node(energy=initial_energy, pos=new_pos)
            new_nodes.append(new_node)
            self.node_births += 1
            node.energy -= NODE_ENERGY_SPAWN_COST
            weight = utils.random_range((-1.0, 1.0))
            conn = Connection(node, new_node, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
            node.connections.append(conn)
            new_node.incoming_connections.append(conn)
            self.connections.append(conn)
            self.conn_births += 1
        # --- Connection births ---
        conn_indices = np.where(eligible_conn)[0]
        if len(conn_indices) > max_conn_births:
            conn_indices = np.random.choice(conn_indices, max_conn_births, replace=False)
        for idx in conn_indices:
            node = self.processing_nodes[idx]
            if len(self.processing_nodes) > 1:
                target = random.choice(self.processing_nodes)
                if target != node and all(c.destination != target for c in node.connections):
                    weight = utils.random_range((-1.0, 1.0))
                    conn = Connection(node, target, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                    node.connections.append(conn)
                    target.incoming_connections.append(conn)
                    self.connections.append(conn)
                    self.conn_births += 1
                    node.energy -= NODE_ENERGY_CONN_COST
            # Workspace connection
            has_sensory = any(getattr(c.destination, 'node_type', None) == 'sensory' for c in node.connections)
            has_workspace = any(getattr(c.destination, 'node_type', None) == 'workspace' for c in node.connections)
            if not has_sensory:
                ws_h = len(self.workspace_nodes)
                ws_w = len(self.workspace_nodes[0])
                wy = random.randint(0, ws_h - 1)
                wx = random.randint(0, ws_w - 1)
                ws_node = self.workspace_nodes[wy][wx]
                if all(c.destination != ws_node for c in node.connections):
                    if getattr(node, 'node_type', None) == 'sensory' or getattr(ws_node, 'node_type', None) == 'sensory':
                        continue
                    weight = utils.random_range((0.1, 1.0))
                    conn = Connection(node, ws_node, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                    node.connections.append(conn)
                    ws_node.incoming_connections.append(conn)
                    self.connections.append(conn)
                    self.conn_births += 1
                    node.energy -= NODE_ENERGY_CONN_COST
            # Sensory connection
            if not has_workspace:
                s_h = len(self.sensory_nodes)
                s_w = len(self.sensory_nodes[0])
                sy = random.randint(0, s_h - 1)
                sx = random.randint(0, s_w - 1)
                s_node = self.sensory_nodes[sy][sx]
                if all(c.destination != s_node for c in node.connections):
                    if getattr(node, 'node_type', None) == 'workspace' or getattr(s_node, 'node_type', None) == 'workspace':
                        continue
                    weight = utils.random_range((0.1, 1.0))
                    conn = Connection(node, s_node, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                    node.connections.append(conn)
                    s_node.incoming_connections.append(conn)
                    self.connections.append(conn)
                    self.conn_births += 1
                    node.energy -= NODE_ENERGY_CONN_COST
        self.processing_nodes.extend(new_nodes)

    def spawn_random_nodes(self, count):
        for _ in range(count):
            energy = utils.random_range(NODE_ENERGY_INIT_RANGE)
            pos = (random.random(), random.random())
            node = Node(energy=energy, pos=pos)
            self.processing_nodes.append(node)
            self.node_births += 1
        for _ in range(count * 2):
            src = random.choice(self.processing_nodes)
            dst = random.choice(self.processing_nodes)
            if src != dst:
                weight = utils.random_range((-1.0, 1.0))
                conn = Connection(src, dst, weight=weight, energy_transfer_capacity=CONN_ENERGY_TRANSFER_CAPACITY)
                src.connections.append(conn)
                self.connections.append(conn)
                self.conn_births += 1

    def get_node_positions_and_energies(self):
        positions = [node.pos for node in self.processing_nodes]
        energies = [node.energy for node in self.processing_nodes]
        return positions, energies

    def get_eye_layer_values(self):
        arr = np.zeros((WORKSPACE_SIZE[1], WORKSPACE_SIZE[0]), dtype=np.float32)
        for y, row in enumerate(self.workspace_nodes):
            for x, node in enumerate(row):
                val = node.channels.get('value', node.energy)
                if isinstance(val, (np.ndarray, list, tuple)):
                    val = float(np.mean(val))
                else:
                    val = float(val)
                arr[y, x] = val
        return arr

    def get_metrics(self):
        return {
            'total_energy': sum([node.energy for node in self.processing_nodes]),
            'node_births': self.node_births,
            'node_deaths': self.node_deaths,
            'conn_births': self.conn_births,
            'conn_deaths': self.conn_deaths,
            'energy_generated': self.total_energy_generated,
            'energy_consumed': self.total_energy_consumed,
            'processing_efficiency': self.total_energy_generated / (self.total_energy_consumed + 1e-6),
            'dynamic_node_count': len(self.processing_nodes),
        }

    def update_sensory_nodes(self, frame):
        import cupy as cp
        print("[DEBUG] update_sensory_nodes called")
        xp = cp if utils.has_cupy() and utils.get_array_module().__name__ == 'cupy' else np
        # Vectorized energy calculation
        energies = 1.0 - (xp.array(frame, dtype=xp.float32) / 255.0)
        # Write back to node objects in a single batch
        flat_nodes = self.sensory_nodes.flat
        flat_energies = energies.flatten()
        energies_cpu = flat_energies.get() if xp.__name__ == 'cupy' else flat_energies
        for node, energy in zip(flat_nodes, energies_cpu):
            node.energy = float(energy)
            # Synchronize connection energy
            for conn in getattr(node, 'connections', []):
                conn.energy_transfer_capacity = node.energy
        # Print a sample of sensory node energies
        sample_energies = [node.energy for node in list(self.sensory_nodes.flat)[:5]]
        print(f"[DEBUG] First 5 sensory node energies: {sample_energies}") 