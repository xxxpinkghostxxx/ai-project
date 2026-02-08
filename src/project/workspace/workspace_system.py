"""
Workspace Node System Implementation

This module implements the main workspace node system that manages
the 16x16 grid of workspace nodes and their energy reading operations.
"""

import time
import threading
import logging
from typing import Any, Dict, List, Optional, Callable
import numpy as np

# Import Qt for thread-safe UI updates
try:
    from PyQt6.QtCore import QTimer, QObject, QMetaObject, Qt, Q_ARG
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    # Define dummy classes if Qt not available
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
from .mapping import map_sensory_to_workspace, calculate_energy_aggregation

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
        
        # Performance monitoring
        self.update_times = []
        self.error_count = 0
        self.last_error_time = 0.0
        
        # Threading
        self._update_lock = threading.Lock()
        self._running = False
        self._update_thread = None
        
        # Initialize system
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
        """Create mapping from workspace nodes to sensory nodes."""
        try:
            sensory_width = self.neural_system.sensory_width
            sensory_height = self.neural_system.sensory_height
            
            logger.info(f"Creating mapping from {sensory_width}x{sensory_height} to {self.grid_size[0]}x{self.grid_size[1]}")
            
            self.mapping = map_sensory_to_workspace(
                sensory_width, sensory_height, self.grid_size
            )
            
            # Validate mapping
            total_mapped_sensory = sum(len(sensory_list) for sensory_list in self.mapping.values())
            total_sensory_nodes = sensory_width * sensory_height
            
            logger.info(f"Mapping created: {total_mapped_sensory}/{total_sensory_nodes} sensory nodes mapped")
            
            if total_mapped_sensory != total_sensory_nodes:
                logger.warning(f"Mapping incomplete: {total_mapped_sensory} mapped vs {total_sensory_nodes} total")
        
        except Exception as e:
            logger.error(f"Failed to create sensory mapping: {e}")
            raise
    
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
        while self._running:
            try:
                start_time = time.time()
                
                # Update workspace nodes
                self.update()
                
                # Calculate update time
                update_time = time.time() - start_time
                self.update_times.append(update_time)
                
                # Keep only last 100 update times
                if len(self.update_times) > 100:
                    self.update_times.pop(0)
                
                # Sleep for configured interval
                sleep_time = max(0, (self.config.reading_interval_ms / 1000.0) - update_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.error_count += 1
                self.last_error_time = time.time()
                logger.error(f"Error in workspace update loop: {e}")
                
                # Prevent rapid error loops
                time.sleep(0.1)
    
    def update(self):
        """Main update method for the workspace system.
        
        Reads workspace node energies from the PyG graph and updates visualization.
        Workspace nodes are OUTPUT nodes - they receive energy from dynamic nodes.
        """
        with self._update_lock:
            try:
                # Try optimized grid read first
                if hasattr(self.neural_system, 'get_workspace_energies_grid'):
                    energy_grid = self.neural_system.get_workspace_energies_grid()
                    
                    # Also update workspace node objects for consistency
                    for workspace_node in self.workspace_nodes:
                        x, y = workspace_node.grid_position
                        if y < len(energy_grid) and x < len(energy_grid[y]):
                            workspace_node.update_energy(energy_grid[y][x])
                else:
                    # Fallback: Read energy for each workspace node individually
                    for workspace_node in self.workspace_nodes:
                        self._read_energy_for_node(workspace_node)
                    
                    # Calculate aggregated energy data
                    energy_grid = self._calculate_energy_grid()
                
                # Notify observers (UI will update canvas)
                self._notify_observers(energy_grid)
                
                # Debug: Log workspace energy stats periodically
                # DISABLED: Too much logging overhead (20x per second × 50ms = 1000ms wasted!)
                # flat = [e for row in energy_grid for e in row]
                # if flat:
                #     avg_e = sum(flat) / len(flat)
                #     max_e = max(flat)
                #     min_e = min(flat)
                #     if max_e > 0.1:
                #         logger.debug(f"Workspace energy: avg={avg_e:.2f}, max={max_e:.2f}, min={min_e:.2f}")
                
                # Update cache timestamp
                self.last_cache_update = time.time()
                
            except Exception as e:
                self.error_count += 1
                self.last_error_time = time.time()
                logger.error(f"Error updating workspace system: {e}")
                raise
    
    def _read_energy_for_node(self, workspace_node: WorkspaceNode):
        """Read energy from the workspace node itself in the PyG graph.
        
        Workspace nodes are OUTPUT nodes - they receive energy from dynamic nodes
        and display it on the canvas. This is the inverse of sensory nodes which
        INPUT pixel data.
        """
        try:
            # Get workspace node energy directly from the PyG graph
            # workspace_node.node_id is the local workspace ID (0 to n_workspace-1)
            energy = self.neural_system.get_workspace_node_energy(workspace_node.node_id)
            workspace_node.update_energy(energy)
        except Exception as e:
            logger.debug(f"Failed to read energy for workspace node {workspace_node.node_id}: {e}")
            workspace_node.update_energy(0.0)
    
    def _get_sensory_energy(self, sensory_id: int) -> Optional[float]:
        """Get energy for a specific sensory node with caching."""
        current_time = time.time()
        
        # Check cache validity
        cache_age = current_time - self.last_cache_update
        if cache_age < (self.config.cache_validity_ms / 1000.0):
            if sensory_id in self.energy_cache:
                return self.energy_cache[sensory_id]
        
        # Fetch from neural system
        try:
            energy = self.neural_system.get_node_energy(sensory_id)
            self.energy_cache[sensory_id] = energy
            
            # Limit cache size
            if len(self.energy_cache) > self.config.cache_size:
                # Remove oldest entries
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
        if not hasattr(self, '_notify_counter'):
            self._notify_counter = 0
        self._notify_counter += 1
        
        # Log every 60 notifications (~2 seconds at 30 FPS)
        should_log = (self._notify_counter % 60 == 0)
        
        for observer in self.observers:
            try:
                # Use Qt's thread-safe mechanism if observer is a Qt object
                if PYQT_AVAILABLE and hasattr(observer, 'on_workspace_update'):
                    # Check if observer is a QObject (Qt widget/window)
                    if isinstance(observer, QObject):
                        # Use QMetaObject.invokeMethod for proper cross-thread communication
                        # This is the CORRECT way to call methods on Qt objects from worker threads
                        if should_log:
                            logger.info(f"📤 NOTIFYING Qt observer via QMetaObject.invokeMethod | grid shape: {len(energy_grid)}x{len(energy_grid[0]) if energy_grid else 0}")
                        
                        # Store energy_grid in observer temporarily for the call
                        # This avoids Q_ARG conversion issues with complex types
                        observer._pending_workspace_grid = energy_grid
                        
                        # Call on_workspace_update on the main thread (Qt.QueuedConnection)
                        QMetaObject.invokeMethod(
                            observer,
                            "_process_pending_workspace_update",
                            Qt.ConnectionType.QueuedConnection
                        )
                    else:
                        # Direct call if not a Qt object
                        if should_log:
                            logger.info(f"📤 NOTIFYING non-Qt observer directly")
                        observer.on_workspace_update(energy_grid)
                else:
                    # Direct call for non-Qt observers
                    if should_log:
                        logger.info(f"📤 NOTIFYING observer without Qt check")
                    observer.on_workspace_update(energy_grid)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")
    
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
        # This would need to be implemented based on the actual connection system
        # For now, return a placeholder
        return 0