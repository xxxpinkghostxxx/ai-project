# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Constants:
#   PYQT_AVAILABLE = bool
#     True when PyQt6.QtCore is importable
#
# Classes:
#   WorkspaceNodeSystem:
#     __init__(self, neural_system: Any, config: EnergyReadingConfig)
#
#     _initialize_workspace_nodes(self)
#       Initialize the grid of workspace nodes
#
#     _create_sensory_mapping(self)
#       Inform about unified region topology (ADR-001); no explicit ID mapping
#
#     start(self)
#       Start the workspace system update thread
#
#     stop(self)
#       Stop the workspace system
#
#     _update_loop(self)
#       Main update loop targeting 60 Hz
#
#     update(self)
#       Read workspace node energies and notify observers
#
#     _read_energy_for_node(self, workspace_node: WorkspaceNode)
#       Read energy for a single workspace node from the neural system
#
#     _get_sensory_energy(self, sensory_id: int) -> Optional[float]
#       Get energy for a specific sensory node with caching
#
#     _calculate_energy_grid(self) -> List[List[float]]
#       Calculate 2D energy grid for UI display
#
#     add_observer(self, observer: Any)
#       Add observer for workspace updates
#
#     remove_observer(self, observer: Any)
#       Remove observer from workspace updates
#
#     _notify_observers(self, energy_grid: List[List[float]])
#       Notify all observers thread-safely (Qt queued connection)
#
#     get_node_data(self, node_id: int) -> Dict[str, Any]
#       Get detailed data for a specific workspace node
#
#     get_system_health(self) -> Dict[str, Any]
#       Get system health metrics
#
#     get_energy_grid(self) -> List[List[float]]
#       Get current energy grid for visualization
#
#     get_energy_trends(self) -> List[List[str]]
#       Get energy trends for all nodes
#
#     get_connection_count(self) -> int
#       Get total number of connections (returns 0 under ADR-001)
#
# =============================================================================
# TODOS
# =============================================================================
#
# - [critical] Superseded by cube_panels + cube_vis; remove or replace with thin
#   adapter after cube migration (cube architecture spec).
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Workspace node system managing the grid of workspace nodes and energy readings."""

import time
import threading
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Callable
import numpy as np

try:
    from PyQt6.QtCore import QObject, QMetaObject, Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    class QObject:
        pass
    class QMetaObject:
        @staticmethod
        def invokeMethod(*args, **kwargs):
            pass
    class Qt:
        class ConnectionType:
            QueuedConnection = None

from .workspace_node import WorkspaceNode
from .config import EnergyReadingConfig

logger = logging.getLogger(__name__)


class WorkspaceNodeSystem:
    """Main workspace node system managing 16x16 grid of workspace nodes."""

    def __init__(self, neural_system: Any, config: EnergyReadingConfig):
        """
        Initialize the workspace node system.

        Args:
            neural_system: Reference to the PyG neural system
            config: Configuration parameters
        """
        self.neural_system = neural_system
        self.config = config
        self.grid_size = config.grid_size
        self.workspace_nodes = []
        self.mapping = {}
        self.observers = []
        self.energy_cache = {}
        self.last_cache_update = 0.0

        self.update_times: deque[float] = deque(maxlen=100)
        self.error_count = 0
        self.last_error_time = 0.0

        self._update_lock = threading.Lock()
        self._running = False
        self._update_thread = None
        self._notify_counter = 0

        self._initialize_workspace_nodes()
        self._create_sensory_mapping()

    def _initialize_workspace_nodes(self):
        """Initialize the 16x16 grid of workspace nodes."""
        logger.info(f"Initializing {self.grid_size[0]}x{self.grid_size[1]} workspace grid")

        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                node_id = y * self.grid_size[0] + x
                node = WorkspaceNode(node_id, x, y)
                self.workspace_nodes.append(node)

        logger.info(f"Initialized {len(self.workspace_nodes)} workspace nodes")

    def _create_sensory_mapping(self):
        """Inform about the unified region topology (ADR-001).

        Under the unified dynamic node grid, sensory and workspace are spatial
        *regions* of the same grid — their coupling is implicit in the cluster
        layout and the region registry, not in an explicit node-ID mapping table.

        The old ``map_sensory_to_workspace()`` call is removed here because:
          - It produced an ID-to-ID dict that was never used in the active
            simulation path (energy reads go straight to the GPU field slice).
          - It is dead code from the PyG graph layer that was removed.

        ``self.mapping`` is kept as an empty dict for backward compatibility
        with any callers that inspect it; ``get_connection_count()`` returns 0.
        """
        self.mapping = {}
        logger.info(
            "WorkspaceNodeSystem: sensory→workspace coupling is implicit in the "
            "unified region grid (ADR-001). No explicit ID mapping needed."
        )

    def start(self):
        """Start the workspace system."""
        if self._running:
            logger.warning("Workspace system already running")
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Workspace system started")

    def stop(self):
        """Stop the workspace system."""
        if not self._running:
            return

        self._running = False
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)

        logger.info("Workspace system stopped")

    def _update_loop(self):
        """Main update loop for the workspace system."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(torch.cuda.current_device())
                _ = torch.zeros(1, device='cuda')
        except Exception:
            pass

        while self._running:
            try:
                start_time = time.time()

                self.update()

                update_time = time.time() - start_time
                self.update_times.append(update_time)

                sleep_time = max(0, (1.0 / 60.0) - update_time)
                time.sleep(sleep_time)

            except Exception as e:
                self.error_count += 1
                self.last_error_time = time.time()
                logger.error(f"Error in workspace update loop: {e}")

                time.sleep(0.1)

    def update(self):
        """Main update method for the workspace system.

        Reads workspace node energies from the PyG graph and updates visualization.
        Workspace nodes are OUTPUT nodes - they receive energy from dynamic nodes.
        """
        energy_grid = None
        with self._update_lock:
            try:
                if hasattr(self.neural_system, 'get_workspace_energies_grid'):
                    energy_grid = self.neural_system.get_workspace_energies_grid()
                else:
                    for workspace_node in self.workspace_nodes:
                        self._read_energy_for_node(workspace_node)

                    energy_grid = self._calculate_energy_grid()

                self.last_cache_update = time.time()

            except Exception as e:
                self.error_count += 1
                self.last_error_time = time.time()
                logger.error(f"Error updating workspace system: {e}")
                raise

        if energy_grid is not None:
            self._notify_observers(energy_grid)

    def _read_energy_for_node(self, workspace_node: WorkspaceNode):
        """Read energy from the workspace node itself in the PyG graph.

        Workspace nodes are OUTPUT nodes - they receive energy from dynamic nodes
        and display it on the canvas. This is the inverse of sensory nodes which
        INPUT pixel data.
        """
        try:
            energy = self.neural_system.get_workspace_node_energy(workspace_node.node_id)
            workspace_node.update_energy(energy)
        except Exception as e:
            logger.debug(f"Failed to read energy for workspace node {workspace_node.node_id}: {e}")
            workspace_node.update_energy(0.0)

    def _get_sensory_energy(self, sensory_id: int) -> Optional[float]:
        """Get energy for a specific sensory node with caching."""
        current_time = time.time()

        cache_age = current_time - self.last_cache_update
        if cache_age < (self.config.cache_validity_ms / 1000.0):
            if sensory_id in self.energy_cache:
                return self.energy_cache[sensory_id]

        try:
            energy = self.neural_system.get_node_energy(sensory_id)
            self.energy_cache[sensory_id] = energy

            if len(self.energy_cache) > self.config.cache_size:
                oldest_keys = list(self.energy_cache.keys())[:len(self.energy_cache) // 2]
                for key in oldest_keys:
                    del self.energy_cache[key]

            return energy

        except Exception as e:
            logger.debug(f"Failed to get energy for sensory node {sensory_id}: {e}")
            return None

    def _calculate_energy_grid(self) -> List[List[float]]:
        """Calculate 2D energy grid for UI display."""
        energy_grid = [[0.0 for _ in range(self.grid_size[0])]
                      for _ in range(self.grid_size[1])]

        for workspace_node in self.workspace_nodes:
            x, y = workspace_node.grid_position
            energy_grid[y][x] = workspace_node.current_energy

        return energy_grid

    def add_observer(self, observer: Any):
        """Add observer for workspace updates."""
        if hasattr(observer, 'on_workspace_update'):
            self.observers.append(observer)
        else:
            raise ValueError("Observer must have on_workspace_update method")

    def remove_observer(self, observer: Any):
        """Remove observer from workspace updates."""
        if observer in self.observers:
            self.observers.remove(observer)

    def _notify_observers(self, energy_grid: List[List[float]]):
        """Notify all observers of workspace updates in a thread-safe manner."""
        self._notify_counter += 1
        _count = self._notify_counter

        should_log = (_count % 60 == 0)

        for observer in self.observers:
            try:
                if PYQT_AVAILABLE and hasattr(observer, 'on_workspace_update') and isinstance(observer, QObject):
                    if should_log:
                        _shape = energy_grid.shape if hasattr(energy_grid, 'shape') else (len(energy_grid), '?')
                        logger.info("Notifying Qt observer | grid shape: %s", _shape)
                    observer._pending_workspace_grid = energy_grid
                    QMetaObject.invokeMethod(
                        observer,
                        "_process_pending_workspace_update",
                        Qt.ConnectionType.QueuedConnection,
                    )
                else:
                    if should_log:
                        logger.info("Notifying observer directly")
                    observer.on_workspace_update(energy_grid)
            except Exception as e:
                logger.error("Observer notification failed: %s", e)

    def get_node_data(self, node_id: int) -> Dict[str, Any]:
        """Get detailed data for a specific workspace node."""
        if node_id >= len(self.workspace_nodes):
            return {}

        node = self.workspace_nodes[node_id]
        return {
            'node_id': node.node_id,
            'grid_position': node.grid_position,
            'current_energy': node.current_energy,
            'associated_sensory': node.associated_sensory_nodes,
            'energy_history_length': len(node.energy_history),
            'energy_trend': node.get_energy_trend(),
            'last_update': node.last_update_time
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        avg_update_time = np.mean(self.update_times) if self.update_times else 0.0
        error_rate = self.error_count / max(1, len(self.update_times))

        return {
            'running': self._running,
            'node_count': len(self.workspace_nodes),
            'mapping_coverage': len([n for n in self.workspace_nodes if self.mapping.get(n.node_id, [])]),
            'avg_update_time': avg_update_time,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'cache_size': len(self.energy_cache),
            'last_error_time': self.last_error_time
        }

    def get_energy_grid(self) -> List[List[float]]:
        """Get current energy grid for visualization."""
        return self._calculate_energy_grid()

    def get_energy_trends(self) -> List[List[str]]:
        """Get energy trends for all nodes."""
        trends = []
        for y in range(self.grid_size[1]):
            row_trends = []
            for x in range(self.grid_size[0]):
                node_id = y * self.grid_size[0] + x
                if node_id < len(self.workspace_nodes):
                    trend = self.workspace_nodes[node_id].get_energy_trend()
                    row_trends.append(trend)
                else:
                    row_trends.append('stable')
            trends.append(row_trends)
        return trends

    def get_connection_count(self) -> int:
        """Get total number of connections in the system."""
        return 0
