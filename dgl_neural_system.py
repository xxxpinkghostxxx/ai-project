import sys
print(sys.executable)
import dgl
import torch
import numpy as np
import time
import cv2
import sys
print(sys.executable)

NODE_TYPE_SENSORY = 0
NODE_TYPE_DYNAMIC = 1
NODE_TYPE_WORKSPACE = 2

# --- Dynamic Node Subtypes ---
SUBTYPE_TRANSMITTER = 0
SUBTYPE_RESONATOR = 1
SUBTYPE_DAMPENER = 2
SUBTYPE_NAMES = ['Transmitter', 'Resonator', 'Dampener']

# --- Config-like constants (tune as needed) ---
NODE_SPAWN_THRESHOLD = 20.0
NODE_DEATH_THRESHOLD = 0.0
NODE_ENERGY_SPAWN_COST = 5.0
NODE_ENERGY_DECAY = 0.1
MAX_NODE_BIRTHS_PER_STEP = 10
MAX_CONN_BIRTHS_PER_STEP = 10
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

class DGLNeuralSystem:
    def __init__(self, sensory_width, sensory_height, n_dynamic, workspace_size=(16, 16), device='cpu'):
        self.device = device
        self.sensory_width = sensory_width
        self.sensory_height = sensory_height
        self.n_sensory = sensory_width * sensory_height
        self.n_dynamic = n_dynamic
        self.workspace_size = workspace_size
        self.n_workspace = workspace_size[0] * workspace_size[1]
        self.n_total = self.n_sensory + self.n_dynamic + self.n_workspace
        self._init_graph()
        # --- Metrics ---
        self.node_births = 0
        self.node_deaths = 0
        self.conn_births = 0
        self.conn_deaths = 0
        self.total_node_births = 0
        self.total_node_deaths = 0
        self.total_conn_births = 0
        self.total_conn_deaths = 0

    def _init_graph(self):
        N = self.n_total
        # Node types
        node_types = torch.zeros(N, dtype=torch.int64)
        node_types[:self.n_sensory] = NODE_TYPE_SENSORY
        node_types[self.n_sensory:self.n_sensory+self.n_dynamic] = NODE_TYPE_DYNAMIC
        node_types[self.n_sensory+self.n_dynamic:] = NODE_TYPE_WORKSPACE
        # Node energies
        energies = torch.ones(N, 1)
        # Node positions (optional)
        pos = torch.zeros(N, 2)
        # Dynamic node subtypes (random assignment)
        dynamic_subtypes = torch.full((N,), -1, dtype=torch.int64)
        if self.n_dynamic > 0:
            subtypes = torch.randint(0, 3, (self.n_dynamic,))
            dynamic_subtypes[self.n_sensory:self.n_sensory+self.n_dynamic] = subtypes
        # Workspace and sensory nodes: -1 subtype
        # Sensory node positions
        for i in range(self.n_sensory):
            y, x = divmod(i, self.sensory_width)
            pos[i] = torch.tensor([x, y], dtype=torch.float32)
        # Dynamic node positions (random)
        for i in range(self.n_sensory, self.n_sensory+self.n_dynamic):
            pos[i] = torch.rand(2)
        # Workspace node positions (grid)
        for i in range(self.n_workspace):
            y, x = divmod(i, self.workspace_size[0])
            pos[self.n_sensory+self.n_dynamic+i] = torch.tensor([x, y], dtype=torch.float32)
        # Create connections: sensory->dynamic, dynamic->workspace, dynamic->dynamic
        src = []
        dst = []
        weights = []
        energy_caps = []
        conn_types = []
        plastic_lrs = []
        gate_thresholds = []
        # Sensory to dynamic ONLY
        for i in range(self.n_sensory):
            dyn = np.random.randint(self.n_sensory, self.n_sensory+self.n_dynamic)
            conn_type = np.random.randint(0, 4)
            # Randomize connection settings by subtype
            if conn_type == CONN_TYPE_EXCITATORY:
                w = np.random.uniform(0.05, 0.2)
            elif conn_type == CONN_TYPE_INHIBITORY:
                w = -np.random.uniform(0.05, 0.2)
            elif conn_type == CONN_TYPE_GATED:
                w = np.random.uniform(0.05, 0.2)
            elif conn_type == CONN_TYPE_PLASTIC:
                w = np.random.uniform(-0.2, 0.2)
            weights.append(w)
            energy_caps.append(np.random.uniform(0.1, 1.0))
            conn_types.append(conn_type)
            # Plastic learning rate
            if conn_type == CONN_TYPE_PLASTIC:
                plastic_lrs.append(np.random.uniform(PLASTIC_LEARNING_RATE_MIN, PLASTIC_LEARNING_RATE_MAX))
            else:
                plastic_lrs.append(0.0)
            # Gated threshold
            if conn_type == CONN_TYPE_GATED:
                gate_thresholds.append(np.random.uniform(0.1, 1.0))
            else:
                gate_thresholds.append(0.0)
            src.append(i)
            dst.append(dyn)
        # Dynamic to workspace ONLY
        for i in range(self.n_sensory, self.n_sensory+self.n_dynamic):
            ws = np.random.randint(self.n_sensory+self.n_dynamic, self.n_total)
            conn_type = np.random.randint(0, 4)
            if conn_type == CONN_TYPE_EXCITATORY:
                w = np.random.uniform(0.05, 0.2)
            elif conn_type == CONN_TYPE_INHIBITORY:
                w = -np.random.uniform(0.05, 0.2)
            elif conn_type == CONN_TYPE_GATED:
                w = np.random.uniform(0.05, 0.2)
            elif conn_type == CONN_TYPE_PLASTIC:
                w = np.random.uniform(-0.2, 0.2)
            weights.append(w)
            energy_caps.append(np.random.uniform(0.1, 1.0))
            conn_types.append(conn_type)
            if conn_type == CONN_TYPE_PLASTIC:
                plastic_lrs.append(np.random.uniform(PLASTIC_LEARNING_RATE_MIN, PLASTIC_LEARNING_RATE_MAX))
            else:
                plastic_lrs.append(0.0)
            if conn_type == CONN_TYPE_GATED:
                gate_thresholds.append(np.random.uniform(0.1, 1.0))
            else:
                gate_thresholds.append(0.0)
            src.append(i)
            dst.append(ws)
        # Dynamic to dynamic (random)
        for i in range(self.n_sensory, self.n_sensory+self.n_dynamic):
            j = np.random.randint(self.n_sensory, self.n_sensory+self.n_dynamic)
            if i != j:
                conn_type = np.random.randint(0, 4)
                if conn_type == CONN_TYPE_EXCITATORY:
                    w = np.random.uniform(0.05, 0.2)
                elif conn_type == CONN_TYPE_INHIBITORY:
                    w = -np.random.uniform(0.05, 0.2)
                elif conn_type == CONN_TYPE_GATED:
                    w = np.random.uniform(0.05, 0.2)
                elif conn_type == CONN_TYPE_PLASTIC:
                    w = np.random.uniform(-0.2, 0.2)
                weights.append(w)
                energy_caps.append(np.random.uniform(0.1, 1.0))
                conn_types.append(conn_type)
                if conn_type == CONN_TYPE_PLASTIC:
                    plastic_lrs.append(np.random.uniform(PLASTIC_LEARNING_RATE_MIN, PLASTIC_LEARNING_RATE_MAX))
                else:
                    plastic_lrs.append(0.0)
                if conn_type == CONN_TYPE_GATED:
                    gate_thresholds.append(np.random.uniform(0.1, 1.0))
                else:
                    gate_thresholds.append(0.0)
                src.append(i)
                dst.append(j)
        # Build DGL graph
        g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=self.n_total, device=self.device)
        g.ndata['energy'] = energies.to(self.device)
        g.ndata['node_type'] = node_types.to(self.device)
        g.ndata['pos'] = pos.to(self.device)
        g.ndata['dynamic_subtype'] = dynamic_subtypes.to(self.device)
        g.edata['weight'] = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        g.edata['energy_transfer_capacity'] = torch.tensor(energy_caps, dtype=torch.float32, device=self.device).unsqueeze(1)
        g.edata['conn_type'] = torch.tensor(conn_types, dtype=torch.int64, device=self.device)
        g.edata['plastic_lr'] = torch.tensor(plastic_lrs, dtype=torch.float32, device=self.device).unsqueeze(1)
        g.edata['gate_threshold'] = torch.tensor(gate_thresholds, dtype=torch.float32, device=self.device).unsqueeze(1)
        # --- Ensure every node has at least one connection ---
        in_deg = g.in_degrees()
        out_deg = g.out_degrees()
        for i in range(g.num_nodes()):
            t = int(g.ndata['node_type'][i])
            if out_deg[i] == 0:
                if t == NODE_TYPE_SENSORY:
                    # Sensory: connect to a random dynamic
                    dyn = np.random.randint(self.n_sensory, self.n_sensory+self.n_dynamic)
                    w = np.random.uniform(0.05, 0.2)
                    cap = np.random.uniform(0.1, 1.0)
                    g.add_edges(i, dyn, data={
                        'weight': torch.tensor([[w]], device=self.device),
                        'energy_transfer_capacity': torch.tensor([[cap]], device=self.device)
                    })
                elif t == NODE_TYPE_DYNAMIC:
                    # Dynamic: connect to another dynamic or workspace
                    if np.random.rand() < 0.5:
                        target = np.random.randint(self.n_sensory, self.n_sensory+self.n_dynamic)
                    else:
                        target = np.random.randint(self.n_sensory+self.n_dynamic, self.n_total)
                    w = np.random.uniform(-1.0, 1.0)
                    cap = np.random.uniform(0.1, 1.0)
                    g.add_edges(i, target, data={
                        'weight': torch.tensor([[w]], device=self.device),
                        'energy_transfer_capacity': torch.tensor([[cap]], device=self.device)
                    })
                elif t == NODE_TYPE_WORKSPACE:
                    # Workspace: connect to a random dynamic
                    dyn = np.random.randint(self.n_sensory, self.n_sensory+self.n_dynamic)
                    w = np.random.uniform(0.05, 0.2)
                    cap = np.random.uniform(0.1, 1.0)
                    g.add_edges(i, dyn, data={
                        'weight': torch.tensor([[w]], device=self.device),
                        'energy_transfer_capacity': torch.tensor([[cap]], device=self.device)
                    })
            if in_deg[i] == 0:
                # Ensure at least one incoming connection
                if t == NODE_TYPE_DYNAMIC:
                    # From sensory or dynamic
                    if np.random.rand() < 0.5:
                        src = np.random.randint(0, self.n_sensory)
                    else:
                        src = np.random.randint(self.n_sensory, self.n_sensory+self.n_dynamic)
                    w = np.random.uniform(-1.0, 1.0)
                    cap = np.random.uniform(0.1, 1.0)
                    g.add_edges(src, i, data={
                        'weight': torch.tensor([[w]], device=self.device),
                        'energy_transfer_capacity': torch.tensor([[cap]], device=self.device)
                    })
                elif t == NODE_TYPE_WORKSPACE:
                    # From dynamic only
                    src = np.random.randint(self.n_sensory, self.n_sensory+self.n_dynamic)
                    w = np.random.uniform(0.05, 0.2)
                    cap = np.random.uniform(0.1, 1.0)
                    g.add_edges(src, i, data={
                        'weight': torch.tensor([[w]], device=self.device),
                        'energy_transfer_capacity': torch.tensor([[cap]], device=self.device)
                    })
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
        """
        Step 1: Energy transfer via message passing.
        Step 2: Node energy decay.
        Step 3: Node birth (spawning new dynamic nodes if energy is high).
        Step 4: Node death (removing nodes with low energy).
        Step 5: Growth (add new connections).
        """
        # --- Reset per-step counters ---
        self.node_births = 0
        self.node_deaths = 0
        self.conn_births = 0
        self.conn_deaths = 0
        g = self.g
        prev_num_edges = g.num_edges()
        prev_num_nodes = g.num_nodes()
        # --- Step 1: Energy transfer (as before) ---
        # --- Advanced energy flow by dynamic subtype ---
        # 1. Outgoing transfer: TRANSMITTER boosts outgoing, DAMPENER reduces incoming, RESONATOR normal
        # 2. Decay: RESONATOR decays faster, DAMPENER decays slower
        # 3. Receiving: RESONATOR receives more, DAMPENER less
        # Prepare per-node multipliers
        dynamic_mask = (g.ndata['node_type'] == NODE_TYPE_DYNAMIC)
        subtypes = g.ndata['dynamic_subtype']
        # Outgoing multiplier (for source nodes)
        out_mult = torch.ones(g.num_nodes(), device=self.device)
        out_mult[(dynamic_mask) & (subtypes == SUBTYPE_TRANSMITTER)] = 1.5
        out_mult[(dynamic_mask) & (subtypes == SUBTYPE_DAMPENER)] = 0.7
        # Incoming multiplier (for target nodes)
        in_mult = torch.ones(g.num_nodes(), device=self.device)
        in_mult[(dynamic_mask) & (subtypes == SUBTYPE_RESONATOR)] = 1.3
        in_mult[(dynamic_mask) & (subtypes == SUBTYPE_DAMPENER)] = 0.7
        # Decay multiplier
        decay_mult = torch.ones(g.num_nodes(), device=self.device)
        decay_mult[(dynamic_mask) & (subtypes == SUBTYPE_RESONATOR)] = 1.5
        decay_mult[(dynamic_mask) & (subtypes == SUBTYPE_DAMPENER)] = 0.5
        # --- Message passing with connection subtypes ---
        g.ndata['energy_out'] = g.ndata['energy'] * out_mult.unsqueeze(1)
        # Prepare edge-wise weight and gating
        edge_src, edge_dst = g.edges()
        conn_types = g.edata['conn_type'].flatten()
        weights = g.edata['weight'].flatten()
        # For inhibitory, ensure weight is negative
        weights[conn_types == CONN_TYPE_INHIBITORY] = -weights[conn_types == CONN_TYPE_INHIBITORY].abs()
        # For excitatory, ensure weight is positive
        weights[conn_types == CONN_TYPE_EXCITATORY] = weights[conn_types == CONN_TYPE_EXCITATORY].abs()
        # For gated, only transmit if src energy > per-edge threshold
        gate_thresholds = g.edata['gate_threshold'].flatten()
        gated_mask = (conn_types == CONN_TYPE_GATED)
        if gated_mask.any():
            src_energies = g.ndata['energy_out'][edge_src[gated_mask], 0]
            gates = src_energies > gate_thresholds[gated_mask]
            weights[gated_mask] = weights[gated_mask] * gates.float()
        # For plastic, Hebbian-like learning: if src and dst both have high energy, increase weight; else, decrease
        plastic_mask = (conn_types == CONN_TYPE_PLASTIC)
        if plastic_mask.any():
            src_e = g.ndata['energy_out'][edge_src[plastic_mask], 0]
            dst_e = g.ndata['energy'][edge_dst[plastic_mask], 0]
            lr = g.edata['plastic_lr'][plastic_mask, 0]
            # Hebbian: if both src and dst > 0.5, increase weight; else, decrease
            hebb = ((src_e > 0.5) & (dst_e > 0.5)).float()
            delta = lr * (2 * hebb - 1)  # +lr if both high, -lr otherwise
            weights[plastic_mask] += delta
            # Clamp plastic weights to [-1.0, 1.0]
            weights[plastic_mask] = torch.clamp(weights[plastic_mask], -1.0, 1.0)
        # Re-assign possibly modified weights
        g.edata['weight'][:,0] = weights
        # Standard message passing
        g.update_all(
            dgl.function.u_mul_e('energy_out', 'weight', 'm1'),
            dgl.function.sum('m1', 'energy_in')
        )
        if 'energy_transfer_capacity' in g.edata:
            g.update_all(
                dgl.function.u_mul_e('energy_out', 'energy_transfer_capacity', 'm2'),
                dgl.function.sum('m2', 'energy_in2')
            )
            g.ndata['energy_in'] = g.ndata['energy_in'] + g.ndata['energy_in2']
        # Apply incoming multiplier
        g.ndata['energy_in'] = g.ndata['energy_in'] * in_mult.unsqueeze(1)
        # Apply transmission loss
        g.ndata['energy_in'] = g.ndata['energy_in'] * TRANSMISSION_LOSS
        g.ndata['energy'] = g.ndata['energy'] + g.ndata['energy_in']
        # Clamp energy
        g.ndata['energy'] = torch.clamp(g.ndata['energy'], min=0.0, max=NODE_ENERGY_CAP)
        # --- Step 2: Node energy decay ---
        g.ndata['energy'] = g.ndata['energy'] - (NODE_ENERGY_DECAY * decay_mult.unsqueeze(1))
        # --- Step 2b: Per-connection maintenance cost ---
        out_deg = g.out_degrees().float().to(self.device)
        g.ndata['energy'] = g.ndata['energy'] - (out_deg.unsqueeze(1) * CONNECTION_MAINTENANCE_COST)
        g.ndata['energy'] = torch.clamp(g.ndata['energy'], min=-100.0, max=NODE_ENERGY_CAP)
        # --- Step 3: Node birth (spawning) ---
        dynamic_mask = (g.ndata['node_type'] == NODE_TYPE_DYNAMIC)
        high_energy_mask = (g.ndata['energy'][:,0] > NODE_SPAWN_THRESHOLD) & dynamic_mask
        high_energy_indices = torch.where(high_energy_mask)[0]
        n_births = min(len(high_energy_indices), MAX_NODE_BIRTHS_PER_STEP)
        if n_births > 0:
            self._spawn_dynamic_nodes(n_births, parent_indices=high_energy_indices[:n_births])
            self.node_births += n_births
            self.total_node_births += n_births
        # --- Step 4: Node death (removal) ---
        dynamic_mask = (self.g.ndata['node_type'] == NODE_TYPE_DYNAMIC)
        low_energy_mask = (self.g.ndata['energy'][:,0] < NODE_DEATH_THRESHOLD) & dynamic_mask
        low_energy_indices = torch.where(low_energy_mask)[0]
        n_deaths = len(low_energy_indices)
        if n_deaths > 0:
            self._remove_nodes(low_energy_indices)
            self.node_deaths += n_deaths
            self.total_node_deaths += n_deaths
        # --- Step 5: Growth (add new connections) ---
        prev_num_edges2 = self.g.num_edges()
        self._add_random_connections(MAX_CONN_BIRTHS_PER_STEP)
        new_num_edges = self.g.num_edges()
        births = max(0, new_num_edges - prev_num_edges2)
        self.conn_births += births
        self.total_conn_births += births
        # --- Connection deaths (from node removal) ---
        conn_deaths = max(0, prev_num_edges - self.g.num_edges())
        self.conn_deaths += conn_deaths
        self.total_conn_deaths += conn_deaths

    def _spawn_dynamic_nodes(self, n, parent_indices=None):
        """Add n new dynamic nodes with random connections. Deduct spawn cost from parent nodes. Assign random subtypes."""
        g = self.g
        old_n = g.num_nodes()
        device = self.device
        # Assign random subtypes
        subtypes = torch.randint(0, 3, (n,), device=device)
        g.add_nodes(
            n,
            data={
                'energy': torch.ones(n, 1, device=device),
                'node_type': torch.full((n,), NODE_TYPE_DYNAMIC, dtype=torch.int64, device=device),
                'pos': torch.rand(n, 2, device=device),
                'dynamic_subtype': subtypes
            }
        )
        # Deduct spawn cost from parent nodes
        if parent_indices is not None:
            g.ndata['energy'][parent_indices, 0] -= NODE_ENERGY_SPAWN_COST
        # Add random connections from new node to existing nodes
        births = 0
        for i in range(old_n, old_n + n):
            # Connect to a random workspace node ONLY (not sensory)
            ws_idx = np.random.randint(self.n_sensory + self.n_dynamic, self.n_total)
            weight = np.random.uniform(0.05, 0.2)
            energy_cap = np.random.uniform(0.1, 1.0)
            g.add_edges(
                i, ws_idx,
                data={
                    'weight': torch.tensor([[weight]], device=device),
                    'energy_transfer_capacity': torch.tensor([[energy_cap]], device=device)
                }
            )
            births += 1
        self.conn_births += births
        self.total_conn_births += births

    def _remove_nodes(self, indices):
        """Remove nodes by indices (DGL will remove all their edges)."""
        g = self.g
        # DGL does not support in-place node removal, so we must create a new graph
        keep_mask = torch.ones(g.num_nodes(), dtype=torch.bool, device=self.device)
        keep_mask[indices] = False
        # Map old to new indices
        old2new = torch.full((g.num_nodes(),), -1, dtype=torch.int64, device=self.device)
        old2new[keep_mask] = torch.arange(keep_mask.sum(), device=self.device)
        # Filter edges
        src, dst = g.edges()
        edge_mask = keep_mask[src] & keep_mask[dst]
        new_src = old2new[src[edge_mask]]
        new_dst = old2new[dst[edge_mask]]
        new_g = dgl.graph((new_src, new_dst), num_nodes=keep_mask.sum().item(), device=self.device)
        # Copy features
        for k in g.ndata:
            new_g.ndata[k] = g.ndata[k][keep_mask]
        for k in g.edata:
            new_g.edata[k] = g.edata[k][edge_mask]
        self.g = new_g
        # Recalculate all node counts from the new graph
        node_types = self.g.ndata['node_type']
        self.n_sensory = int((node_types == NODE_TYPE_SENSORY).sum().item())
        self.n_dynamic = int((node_types == NODE_TYPE_DYNAMIC).sum().item())
        self.n_workspace = int((node_types == NODE_TYPE_WORKSPACE).sum().item())
        self.n_total = self.g.num_nodes()
        # NOTE: All masks (dynamic_mask, etc.) must be recomputed after node removal

    def _add_random_connections(self, n):
        """Add n random new connections between dynamic nodes and others. Sensory nodes cannot connect to workspace nodes."""
        g = self.g
        device = self.device
        dynamic_indices = torch.where(g.ndata['node_type'] == NODE_TYPE_DYNAMIC)[0]
        if len(dynamic_indices) == 0:
            return
        births = 0
        for _ in range(n):
            src = dynamic_indices[np.random.randint(0, len(dynamic_indices))].item()
            # Randomly choose a target: dynamic or workspace, but never sensory->workspace
            possible_targets = torch.cat([
                torch.where(g.ndata['node_type'] == NODE_TYPE_DYNAMIC)[0],
                torch.where(g.ndata['node_type'] == NODE_TYPE_WORKSPACE)[0]
            ])
            dst = possible_targets[np.random.randint(0, len(possible_targets))].item()
            # Enforce: dynamic->dynamic or dynamic->workspace only
            if g.ndata['node_type'][src] == NODE_TYPE_DYNAMIC and g.ndata['node_type'][dst] in [NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE]:
                conn_type = np.random.randint(0, 4)
                if conn_type == CONN_TYPE_EXCITATORY:
                    w = np.random.uniform(0.05, 0.2)
                elif conn_type == CONN_TYPE_INHIBITORY:
                    w = -np.random.uniform(0.05, 0.2)
                elif conn_type == CONN_TYPE_GATED:
                    w = np.random.uniform(0.05, 0.2)
                elif conn_type == CONN_TYPE_PLASTIC:
                    w = np.random.uniform(-0.2, 0.2)
                energy_cap = np.random.uniform(0.1, 1.0)
                plastic_lr = np.random.uniform(PLASTIC_LEARNING_RATE_MIN, PLASTIC_LEARNING_RATE_MAX) if conn_type == CONN_TYPE_PLASTIC else 0.0
                gate_threshold = np.random.uniform(0.1, 1.0) if conn_type == CONN_TYPE_GATED else 0.0
                g.add_edges(
                    src, dst,
                    data={
                        'weight': torch.tensor([[w]], device=device, dtype=torch.float32),
                        'energy_transfer_capacity': torch.tensor([[energy_cap]], device=device, dtype=torch.float32),
                        'conn_type': torch.tensor([[conn_type]], device=device, dtype=torch.int64),
                        'plastic_lr': torch.tensor([[plastic_lr]], device=device, dtype=torch.float32),
                        'gate_threshold': torch.tensor([[gate_threshold]], device=device, dtype=torch.float32)
                    }
                )
                births += 1
        self.conn_births += births
        self.total_conn_births += births

    def update_sensory_nodes(self, sensory_input):
        """
        Update the energy of sensory nodes from an external input (e.g., image).
        sensory_input: numpy array or torch tensor of shape (height, width) or (n_sensory,)
        Values should be normalized to [0, 1] or [0, 255].
        """
        if isinstance(sensory_input, np.ndarray):
            arr = torch.from_numpy(sensory_input).float().to(self.device)
        else:
            arr = sensory_input.float().to(self.device)
        # Flatten if 2D
        if arr.dim() == 2:
            arr = arr.flatten()
        # Normalize if needed
        if arr.max() > 1.0:
            arr = arr / 255.0
        # Clamp
        arr = torch.clamp(arr, 0.0, 1.0)
        # Assign to sensory nodes
        self.g.ndata['energy'][:self.n_sensory, 0] = arr 

    def update_from_screen_capture(self, capture):
        """
        Integrate with ThreadedScreenCapture: fetch the latest frame, preprocess, and update sensory nodes.
        capture: ThreadedScreenCapture instance
        """
        frame = capture.get_latest()
        # If frame is color, convert to grayscale
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize if needed
        if frame.shape != (self.sensory_height, self.sensory_width):
            frame = cv2.resize(frame, (self.sensory_width, self.sensory_height))
        self.update_sensory_nodes(frame) 

    def get_metrics(self):
        g = self.g
        node_types = g.ndata['node_type']
        n_dynamic = int((node_types == NODE_TYPE_DYNAMIC).sum().cpu().item())
        n_workspace = int((node_types == NODE_TYPE_WORKSPACE).sum().cpu().item())
        dynamic_energies = g.ndata['energy'][node_types == NODE_TYPE_DYNAMIC].cpu().numpy().flatten()
        avg_dynamic_energy = float(dynamic_energies.mean()) if dynamic_energies.size > 0 else 0.0
        total_energy = float(g.ndata['energy'].sum().cpu().item())
        return {
            'total_energy': total_energy,
            'avg_dynamic_energy': avg_dynamic_energy,
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