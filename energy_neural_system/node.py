import time
from utils import logger

class Node:
    __slots__ = ('node_type', 'ENERGY_CAP', 'energy', 'connections', 'incoming_connections', 'channels', 'is_sensory', 'pos', 'creation_time', 'activity_history', 'energy_generation_rate', 'energy_consumption_rate', 'last_update_time', 'debug_history', '_prev_debug_energy', '_prev_debug_conn', '_zombie_counter', '_marked_for_death')
    def __init__(self, is_sensory=False, energy=None, pos=None, node_type=None):
        # Ensure node_type is always set
        if node_type is None:
            if is_sensory:
                node_type = 'sensory'
            else:
                node_type = 'dynamic'
        self.node_type = node_type
        try:
            import config
            self.ENERGY_CAP = getattr(config, 'NODE_ENERGY_CAP', 10.0)
        except ImportError:
            self.ENERGY_CAP = 10.0
        # --- Workspace node: always start with at least 0.0 energy ---
        if node_type == 'workspace':
            self.energy = 0.0 if energy is None else max(0.0, energy)
        elif energy is not None:
            self.energy = energy
        else:
            self.energy = float('inf') if is_sensory else 1.0
        self.connections = []  # Outgoing connections
        self.incoming_connections = []  # Incoming connections
        self.channels = {}  # Information channels
        self.is_sensory = is_sensory
        self.pos = pos  # (x, y) position for 2D map, default None
        self.creation_time = time.time()
        self.activity_history = []  # List of (timestamp, activity_value)
        self.energy_generation_rate = 0.0
        if node_type == 'workspace':
            # Workspace nodes are energy neutral if left alone
            try:
                import config
                self.energy_generation_rate = getattr(config, 'WORKSPACE_NODE_ENERGY_GEN_RATE', 0.05)
            except ImportError:
                self.energy_generation_rate = 0.05
            self.energy_consumption_rate = self.energy_generation_rate
        else:
            self.energy_consumption_rate = 0.01 if not is_sensory else 0.0  # Idle cost for non-sensory
        self.last_update_time = self.creation_time

    def update_config(self, config):
        self.ENERGY_CAP = getattr(config, 'NODE_ENERGY_CAP', 20.0)

    def update(self, dt=None, config=None, system=None, debug=False):
        """
        Update node energy based on activity, connections, and base generation.
        Args:
            dt: Time delta (float, seconds). If None, computed from last_update_time.
            config: Config module or object with BASE_GEN and OPTIMAL_CONN.
            system: Optional reference to the system (for advanced behaviors).
            debug: If True, print per-node energy changes.
        """
        now = time.time()
        if dt is None:
            dt = now - self.last_update_time
        self.last_update_time = now
        # Baseline consumption
        self.energy -= self.energy_consumption_rate * dt
        # --- Dynamic node base energy generation ---
        if not self.is_sensory and getattr(self, 'node_type', None) == 'dynamic':
            # Configurable parameters
            BASE_GEN = getattr(config, 'BASE_GEN', 0.05) if config else 0.05
            OPTIMAL_CONN = getattr(config, 'OPTIMAL_CONN', 5) if config else 5
            n_conn = len(getattr(self, 'connections', []))
            import math
            scale = math.tanh((n_conn - OPTIMAL_CONN) / 2.0)
            gen = BASE_GEN * (1.0 + 0.5 * scale)
            self.energy += gen * dt
            if debug:
                print(f"[NODE DEBUG] Node {id(self)} base gen: {gen*dt:.4f} (n_conn={n_conn})")
        # --- Workspace node base energy generation ---
        if getattr(self, 'node_type', None) == 'workspace':
            # Log before energy
            before_energy = self.energy
            self.energy += self.energy_generation_rate * dt
            # Clamp before debug print
            self.energy = max(min(self.energy, self.ENERGY_CAP), 0)
            print(f"[DEBUG] Workspace node {id(self)}: dt={dt:.4f}, before={before_energy:.4f}, after={self.energy:.4f}, gen={self.energy_generation_rate * dt:.4f}, cons={self.energy_consumption_rate * dt:.4f}")
            # Log before/after to diagnostic_trace.txt
            try:
                from main import get_log_path, logger
                diag_path = get_log_path('diagnostic_trace.txt')
                with open(diag_path, 'a', encoding='utf-8') as diagf:
                    diagf.write(f"[WS ENERGY] id={id(self)} before={before_energy:.4f} after={self.energy:.4f} gen={self.energy_generation_rate * dt:.4f} cons={self.energy_consumption_rate * dt:.4f}\n")
                if logger:
                    logger.info(f"[WS ENERGY] id={id(self)} before={before_energy:.4f} after={self.energy:.4f} gen={self.energy_generation_rate * dt:.4f} cons={self.energy_consumption_rate * dt:.4f}")
            except Exception as e:
                pass
        # Generate energy from active incoming connections
        generated = 0.0
        for conn in self.incoming_connections:
            if hasattr(conn, 'last_activity') and conn.last_activity > 0:
                generated += conn.last_activity * conn.energy_transfer_capacity * dt
        self.energy += generated
        # Record activity
        activity = self.channels.get('value', 0)
        self.activity_history.append((now, activity))
        # Prune history to last 100 entries
        if len(self.activity_history) > 100:
            self.activity_history = self.activity_history[-100:]
        # Clamp energy and handle zombies
        if not hasattr(self, 'debug_history'):
            self.debug_history = []
        prev_energy = getattr(self, '_prev_debug_energy', self.energy)
        prev_conn = getattr(self, '_prev_debug_conn', len(getattr(self, 'connections', [])))
        if not self.is_sensory:
            # Clamp to [0, ENERGY_CAP]
            self.energy = max(min(self.energy, self.ENERGY_CAP), 0)
            # --- Extra: debug print if workspace node energy is negative ---
            if self.node_type == 'workspace' and self.energy < 0:
                print(f"[ERROR] Workspace node {id(self)} energy is negative after update: {self.energy}")
        n_conn = len(getattr(self, 'connections', []))
        if debug:
            # Only log if energy or connection count changed significantly
            if abs(self.energy - prev_energy) > 0.1 or abs(n_conn - prev_conn) > 0:
                self.debug_history.append((now, round(self.energy,2), n_conn))
                if len(self.debug_history) > 20:
                    self.debug_history = self.debug_history[-20:]
            self._prev_debug_energy = self.energy
            self._prev_debug_conn = n_conn
        if debug:
            print(f"[NODE DEBUG] Node {id(self)} energy after update: {self.energy:.4f}")
        # Sensory/static nodes never die or propagate
        # --- Dynamic node sends energy to connected workspace nodes ---
        if getattr(self, 'node_type', None) == 'dynamic' and self.energy > 1.0:
            for conn in self.connections:
                dst = getattr(conn, 'destination', None)
                if getattr(dst, 'node_type', None) == 'workspace' and dst.energy < dst.ENERGY_CAP:
                    send_amt = min(0.1, self.energy - 1.0, dst.ENERGY_CAP - dst.energy)
                    if send_amt > 0:
                        before_src = self.energy
                        before_dst = dst.energy
                        self.energy -= send_amt
                        dst.energy += send_amt
                        # Log transfer event
                        try:
                            from main import get_log_path, logger
                            diag_path = get_log_path('diagnostic_trace.txt')
                            with open(diag_path, 'a', encoding='utf-8') as diagf:
                                diagf.write(f"[TRANSFER] dynamic_id={id(self)} ws_id={id(dst)} amt={send_amt:.4f} src_before={before_src:.4f} src_after={self.energy:.4f} dst_before={before_dst:.4f} dst_after={dst.energy:.4f}\n")
                            if logger:
                                logger.info(f"[TRANSFER] dynamic_id={id(self)} ws_id={id(dst)} amt={send_amt:.4f} src_before={before_src:.4f} src_after={self.energy:.4f} dst_before={before_dst:.4f} dst_after={dst.energy:.4f}")
                        except Exception as e:
                            pass

    def transfer_energy(self, target_node, amount):
        if self.energy >= amount:
            self.energy -= amount
            target_node.energy += amount

    def __repr__(self):
        return f"Node(energy={self.energy}, pos={self.pos}, is_sensory={self.is_sensory}, node_type={self.node_type})"

    def get_debug_summary(self):
        if not hasattr(self, 'debug_history') or not self.debug_history:
            return ''
        return ' '.join([f"E:{e}|C:{c}" for (_,e,c) in self.debug_history])

class NeutralNode(Node):
    __slots__ = ()
    def __init__(self, energy=0.0):
        super().__init__(is_sensory=False, energy=energy)
    # You can add more logic here if needed 