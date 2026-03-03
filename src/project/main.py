"""Main entry point for the Taichi Neural System.

This module contains the main entry point and system initialization logic
for the neural simulation application.

Key Features:
- System initialization and configuration management
- Qt application setup and lifecycle management
- Resource management with context-based cleanup
- Comprehensive error handling and logging
- Neural system and UI integration

Usage:
    python project/main.py [--log-level LEVEL]

    Where LEVEL can be: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

import sys
import os
import time
import threading

# Add parent directory to path so we can import 'project' as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
import logging
import contextlib
import argparse
from typing import Any, Tuple
from collections.abc import Generator
import numpy as np
import torch

# Import Qt application components
# pylint: disable=wrong-import-position
from PyQt6.QtWidgets import QApplication  # type: ignore[import-untyped]  # pylint: disable=no-name-in-module

from project.vision import ThreadedScreenCapture  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
try:
    from project.optimized_capture import create_best_capture  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
    OPTIMIZED_CAPTURE_AVAILABLE = True
except ImportError:
    OPTIMIZED_CAPTURE_AVAILABLE = False
    # Note: logger not defined yet, will log later
from project.utils.error_handler import ErrorHandler  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.utils.config_manager import ConfigManager  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.system.state_manager import StateManager  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.ui.modern_main_window import ModernMainWindow  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.workspace.workspace_system import WorkspaceNodeSystem  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.workspace.config import EnergyReadingConfig  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.system.taichi_engine import TaichiNeuralEngine, init_taichi  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error

# Audio modules (optional — graceful degradation if sounddevice missing)
try:
    from project.audio_capture import AudioCapture  # type: ignore[import-untyped,import-not-found]
    from project.audio_output import AudioOutput    # type: ignore[import-untyped,import-not-found]
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pyg_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log optimized capture availability (now that logger is defined)
if OPTIMIZED_CAPTURE_AVAILABLE:
    logger.info("Optimized capture available (will auto-select best method)")
else:
    logger.info("Using standard capture (install mss for faster capture: pip install mss)")


class HybridNeuralSystemAdapter:
    """Adapter to make TaichiNeuralEngine compatible with the existing UI and workspace code."""
    
    def __init__(
        self,
        engine: TaichiNeuralEngine,
        sensory_region: Tuple[int, int, int, int],
        workspace_region: Tuple[int, int, int, int],
        sensory_width: int = 64,
        sensory_height: int = 64,
        audio_sensory_L: Tuple[int, int, int, int] | None = None,
        audio_sensory_R: Tuple[int, int, int, int] | None = None,
        audio_workspace_L: Tuple[int, int, int, int] | None = None,
        audio_workspace_R: Tuple[int, int, int, int] | None = None,
    ):
        """Initialize the adapter.

        Args:
            engine: The Taichi neural engine
            sensory_region: (y_start, y_end, x_start, x_end) for sensory nodes
            workspace_region: (y_start, y_end, x_start, x_end) for workspace nodes
            sensory_width: Width of sensory input
            sensory_height: Height of sensory input
            audio_sensory_L/R: Regions for left/right audio FFT input
            audio_workspace_L/R: Regions for left/right audio oscillator output
        """
        self.engine = engine
        self.sensory_region = sensory_region
        self.workspace_region = workspace_region
        self.sensory_width = sensory_width
        self.sensory_height = sensory_height
        self.device = engine.device

        # Audio regions (None = audio disabled)
        self.audio_sensory_L = audio_sensory_L
        self.audio_sensory_R = audio_sensory_R
        self.audio_workspace_L = audio_workspace_L
        self.audio_workspace_R = audio_workspace_R

        # Track connection worker state (for compatibility)
        self._connection_worker_started = False

        self._step_count = 0  # Number of update_step() calls made
        self._metrics_lock = threading.Lock()
        logger.info("Hybrid adapter initialized")
    
    def process_frame(self, frame_data: Any) -> dict:
        """Process a screen capture frame (compatible with ThreadedScreenCapture).
        
        Optimized for real-time performance with GPU.
        """
        # Convert frame to float32 tensor — raw values become energy directly
        if isinstance(frame_data, torch.Tensor):
            pixels = frame_data.float()
        else:
            pixels = torch.tensor(frame_data, device=self.device).float()

        # Sensory nodes output their data value as energy into attached dynamic nodes.
        # No gain, bias, or normalization — data IS energy.
        self.engine.inject_sensory_data(pixels, self.sensory_region)

        return self.engine.step()

    def update_step(self) -> dict:
        """Single update step (compatible with UI update loop)."""
        t_step_start = time.time()
        self._step_count += 1

        t_engine_call_start = time.time()
        result = self.engine.step()
        t_engine_call_done = time.time()

        # Explicit GPU sync to measure actual GPU work time
        t_gpu_sync_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gpu_sync_done = time.time()

        t_step_done = time.time()

        result['adapter_time']     = t_step_done - t_step_start
        result['pre_sync_time']    = 0.0  # pre-sync removed (was always 0)
        result['engine_call_time'] = t_engine_call_done - t_engine_call_start
        result['gpu_sync_time']    = t_gpu_sync_done - t_gpu_sync_start
        return result
    
    def get_workspace_energies(self) -> torch.Tensor:
        """Get workspace energies for UI display."""
        return self.engine.read_workspace_energies(self.workspace_region)
    
    def get_workspace_energies_grid(self) -> 'np.ndarray':
        """Get workspace energies as 2D (H, W) numpy array for UI display.

        The returned array has shape (workspace_height, workspace_width) and
        dtype float32, ready for direct colormap rendering.

        OPTIMIZED: Result is cached for 8 ms (~120 Hz ceiling) so the GPU→CPU
        transfer runs at most once per display frame even when called more often.
        """
        if not hasattr(self, '_workspace_energies_cache_time'):
            self._workspace_energies_cache_time = 0.0
            self._workspace_energies_cache = None

        now = time.monotonic()
        if now - self._workspace_energies_cache_time >= 0.008 or self._workspace_energies_cache is None:
            energies_tensor = self.get_workspace_energies()
            flat = energies_tensor.cpu().numpy().copy()  # shape: (H*W,) or (H*W, 1)
            flat = flat.ravel()

            # Reshape to 2D (height × width) matching the workspace region
            ws_h = self.workspace_region[1] - self.workspace_region[0]
            ws_w = self.workspace_region[3] - self.workspace_region[2]
            if flat.size == ws_h * ws_w:
                self._workspace_energies_cache = flat.reshape(ws_h, ws_w)
            else:
                logger.warning(
                    "Workspace energy shape mismatch: got %d elements, expected %d×%d=%d; "
                    "returning zeros. Check workspace_region vs engine grid.",
                    flat.size, ws_h, ws_w, ws_h * ws_w,
                )
                self._workspace_energies_cache = np.zeros((ws_h, ws_w), dtype=np.float32)

            self._workspace_energies_cache_time = now

        return self._workspace_energies_cache  # type: ignore[return-value]
    
    def get_workspace_node_energy(self, node_id: int) -> float:
        """Get energy for a specific workspace node (compatibility method).

        DEPRECATED: Triggers a full GPU→CPU transfer for a single float.
        Prefer get_workspace_energies_grid() for batch reads.

        Args:
            node_id: The ID of the workspace node (0-255 for 16x16 grid)

        Returns:
            Energy value for that node
        """
        # Get all workspace energies
        energies = self.get_workspace_energies()
        
        # Convert node_id to grid position
        # Assuming workspace is 16x16, node_id maps to (row, col)
        workspace_height = self.workspace_region[1] - self.workspace_region[0]
        workspace_width = self.workspace_region[3] - self.workspace_region[2]
        
        if node_id >= workspace_height * workspace_width:
            return 0.0
        
        row = node_id // workspace_width
        col = node_id % workspace_width
        
        # Return energy at that position
        if row < energies.shape[0] and col < energies.shape[1]:
            return float(energies[row, col].item())
        return 0.0
    
    def get_node_count(self) -> int:
        """Get current alive node count (high-water mark, no GPU roundtrip)."""
        return self.engine.node_count
    
    def get_energy_stats(self) -> dict:
        """Get energy statistics."""
        return {
            'avg_energy': float(self.engine.energy_field.mean().item()),
            'max_energy': float(self.engine.energy_field.max().item()),
            'min_energy': float(self.engine.energy_field.min().item()),
        }

    def pulse_energy(self) -> float:
        """Add a burst of energy across the whole field. Returns amount added."""
        amount = self.engine.energy_cap * 0.1  # 10% of cap
        self.engine.energy_field.add_(amount)
        self.engine.energy_field.clamp_(self.engine.death_threshold, self.engine.energy_cap)
        return float(amount)

    # ------------------------------------------------------------------
    # Config hot-reload
    # ------------------------------------------------------------------

    def apply_config(self, config_manager: 'ConfigManager') -> None:
        """Push current hybrid config values into the running engine.

        Call this after ConfigManager.update_config() to apply changes
        without restarting the application.
        """
        hybrid = config_manager.get_config('hybrid') or {}
        self.engine.update_parameters(
            transfer_strength=hybrid.get('transfer_strength', self.engine.transfer_strength),
            transfer_dt=hybrid.get('transfer_dt', self.engine.transfer_dt),
            gate_threshold=hybrid.get('gate_threshold', self.engine.gate_threshold),
            spawn_threshold=hybrid.get('node_spawn_threshold', self.engine.spawn_threshold),
            death_threshold=hybrid.get('node_death_threshold', self.engine.death_threshold),
            energy_cap=hybrid.get('node_energy_cap', self.engine.energy_cap),
            child_energy_fraction=hybrid.get('child_energy_fraction', self.engine.child_energy_fraction),
        )

    # ------------------------------------------------------------------
    # Audio methods
    # ------------------------------------------------------------------

    def process_audio_frame(self, spectrum: 'np.ndarray') -> None:
        """Inject stereo FFT spectrum into the audio sensory regions.

        Parameters
        ----------
        spectrum : ndarray
            Shape ``(2, fft_bins)`` — row 0 = left, row 1 = right.
            Raw magnitude values written directly as energy.
        """
        if self.audio_sensory_L is None or self.audio_sensory_R is None:
            return

        y0, y1, x0, x1 = self.audio_sensory_L
        rows = y1 - y0
        cols = x1 - x0

        for ch, region in enumerate([self.audio_sensory_L, self.audio_sensory_R]):
            ch_data = spectrum[ch]  # (fft_bins,)
            # Tile across rows for spatial redundancy
            tiled = np.tile(ch_data, (rows, 1))[:rows, :cols]
            tiled_t = torch.from_numpy(tiled.astype(np.float32))
            self.engine.inject_audio_data(tiled_t, region)

    def get_audio_workspace_energies(self) -> Tuple['np.ndarray', 'np.ndarray'] | None:
        """Read energy from both audio workspace regions.

        Returns ``(left_grid, right_grid)`` as numpy arrays, or ``None``
        if audio regions are not configured. Each array has shape
        ``(rows, cols)`` matching the audio workspace region size.
        """
        if self.audio_workspace_L is None or self.audio_workspace_R is None:
            return None

        if not hasattr(self, '_audio_ws_cache_time'):
            self._audio_ws_cache_time = 0.0
            self._audio_ws_cache: Tuple[np.ndarray, np.ndarray] | None = None

        now = time.monotonic()
        if now - self._audio_ws_cache_time >= 0.008 or self._audio_ws_cache is None:
            left = self.engine.read_audio_workspace_energies(self.audio_workspace_L).numpy()
            right = self.engine.read_audio_workspace_energies(self.audio_workspace_R).numpy()
            self._audio_ws_cache = (left, right)
            self._audio_ws_cache_time = now

        return self._audio_ws_cache

    def stop_connection_worker(self) -> None:
        """No-op: Taichi engine has no connection worker thread."""

    def wait_for_workers_idle(self) -> None:
        """No-op: Taichi engine has no background workers."""

    def get_metrics(self) -> dict:
        """Return performance metrics. Caches expensive GPU reads for 100ms."""
        current_time = time.time()

        with self._metrics_lock:
            # Cache valid for 100ms — reduces GPU round-trips on high-frequency UI polls
            if hasattr(self, '_last_metrics_time'):
                if current_time - self._last_metrics_time < 0.1:
                    return self._cached_metrics.copy()

            # Reuse engine's get_metrics() (single GPU→CPU numpy read for node counts)
            engine_m = self.engine.get_metrics()
            dynamic_count   = engine_m['dynamic_count']
            workspace_count = engine_m['workspace_count']

            # Sensory count = area of sensory field region (not in _node_state)
            sy0_m, sy1_m, sx0_m, sx1_m = self.sensory_region
            sensory_count = (sy1_m - sy0_m) * (sx1_m - sx0_m)
            total_nodes   = engine_m['alive_count'] + sensory_count

            total_energy = engine_m['total_energy']

            # Workspace energy stats — slice directly from GPU field (fast; no node scan)
            wy0, wy1, wx0, wx1 = self.workspace_region
            ws_field = self.engine.energy_field[wy0:wy1, wx0:wx1]
            workspace_energy_avg = float(ws_field.mean().item())
            workspace_energy_min = float(ws_field.min().item())
            workspace_energy_max = float(ws_field.max().item())

            # Sensory region energy stats (cheap slice, no per-node scan)
            sy0, sy1, sx0, sx1 = self.sensory_region
            sens_field = self.engine.energy_field[sy0:sy1, sx0:sx1]
            sensory_energy_min = float(sens_field.min().item())
            sensory_energy_max = float(sens_field.max().item())

            grid_cells = self.engine.grid_size[0] * self.engine.grid_size[1]
            avg_energy = total_energy / grid_cells if total_energy > 0 else 0.0
            metrics = {
                'total_energy':         total_energy,
                'dynamic_node_count':   dynamic_count,
                'sensory_node_count':   sensory_count,
                'workspace_node_count': workspace_count,
                'edge_count':           0,
                'connection_count':     0,
                'conns_per_dynamic':    0.0,  # DNA-based system; no discrete connections
                'total_spawns':         self.engine.total_spawns,
                'total_deaths':         self.engine.total_deaths,
                'node_births':          self.engine.total_spawns,
                'node_deaths':          self.engine.total_deaths,
                'total_node_births':    self.engine.total_spawns,
                'total_node_deaths':    self.engine.total_deaths,
                'total_conn_births':    0,
                'total_conn_deaths':    0,
                'total_nodes':          total_nodes,
                'step_count':           self._step_count,
                'operations_per_second': self.engine.grid_operations_per_step * 60,
                'workspace_energy_avg': workspace_energy_avg,
                'workspace_energy_min': workspace_energy_min,
                'workspace_energy_max': workspace_energy_max,
                'avg_energy':           avg_energy,
                'avg_dynamic_energy':   avg_energy,  # Field-wide proxy; no per-node-type scan
                'sensory_energy_min':   sensory_energy_min,
                'sensory_energy_max':   sensory_energy_max,
            }

            self._cached_metrics = metrics
            self._last_metrics_time = current_time
            return metrics
    
    def queue_cull(self) -> None:
        """Queue connection culling (compatibility method for UI).
        
        Hybrid engine doesn't use explicit connections, so this is a no-op.
        """
        # Hybrid engine uses field-based flow, not explicit connections
        # No culling needed - connections are implicit in the field
        pass
    
    def queue_connection_growth(self) -> None:
        """Queue connection growth (compatibility method for UI).
        
        Hybrid engine doesn't use explicit connections, so this is a no-op.
        """
        # Hybrid engine uses field-based flow, not explicit connections
        # Growth happens automatically via density field evolution
        pass
    
    def start_connection_worker(self, batch_size: int = 25) -> None:
        """Start connection worker (compatibility method - not needed for hybrid)."""
        self._connection_worker_started = True
        logger.info("Hybrid mode: connection worker not needed (using DNA-based transfer)")
    
    def update_sensory_nodes(self, sensory_input: Any) -> None:
        """Update sensory nodes with input (compatibility method).
        
        It converts sensory input (pixels) to energy and injects it into the sensory region.
        
        Args:
            sensory_input: Sensory data (numpy array or torch tensor) with shape (height, width)
                          Values should be in range 0-255 (pixel intensities)
        """
        # Convert to tensor if needed
        if isinstance(sensory_input, np.ndarray):
            sensory_input = torch.tensor(sensory_input, device=self.device, dtype=torch.float32)
        
        # Validate shape
        if sensory_input.shape != (self.sensory_height, self.sensory_width):
            # Reshape or resize if needed
            if sensory_input.numel() == self.sensory_height * self.sensory_width:
                sensory_input = sensory_input.reshape(self.sensory_height, self.sensory_width)
            else:
                # Skip invalid input silently for compatibility
                return
        
        # Inject data using the hybrid engine's method
        self.engine.inject_sensory_data(sensory_input, self.sensory_region)
    
    def update(self) -> None:
        """Run a single simulation step (compatibility method).
        
        This is the main update method called by the UI on each frame.
        It runs the hybrid engine's step function with probabilistic transfer enabled.
        """
        self.update_step()
    
    def apply_connection_worker_results(self) -> None:
        """Apply connection worker results (compatibility method - not needed for hybrid).
        
        The hybrid engine doesn't use a separate connection worker,
        so this is a no-op for compatibility.
        """
        # Hybrid engine doesn't use connection worker
        pass
    
    def shutdown(self) -> None:
        """Shutdown the system."""
        logger.info("Shutting down hybrid engine")
        # Hybrid engine doesn't need special shutdown
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.shutdown()
    
    def stop(self) -> None:
        """Stop the system."""
        self.shutdown()


def resolve_device(pref: str | None) -> str:
    """Resolve device preference string to 'cuda' or 'cpu'."""
    cuda = torch.cuda.is_available()
    if isinstance(pref, str):
        pref = pref.lower()
    if pref in ('auto', None):
        return 'cuda' if cuda else 'cpu'
    if pref == 'cuda' and cuda:
        return 'cuda'
    return 'cpu'


@contextlib.contextmanager
def managed_resources() -> Generator[list[Any], None, None]:
    """Enhanced context manager for system resources with lifecycle management.

    Provides lifecycle management and recovery mechanisms for system resources.
    """
    resources: list[Any] = []
    try:
        yield resources
    finally:
        logger.info("Starting resource cleanup via managed_resources")
        cleanup_errors: list[str] = []

        for resource in reversed(resources):
            try:
                if hasattr(resource, 'shutdown'):
                    resource.shutdown()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'stop'):
                    resource.stop()
                logger.debug("Successfully cleaned up resource: %s", type(resource).__name__)
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_info = f"Error cleaning up resource {type(resource).__name__}: {str(e)}"
                logger.error(error_info)
                cleanup_errors.append(error_info)

        # Report cleanup summary
        if cleanup_errors:
            logger.warning(
                "Resource cleanup completed with %d errors: %s",
                len(cleanup_errors),
                ', '.join(cleanup_errors)
            )
        else:
            logger.info("Resource cleanup completed successfully")

def create_hybrid_neural_system(
    config_manager: ConfigManager,
    sensory_width: int,
    sensory_height: int
) -> HybridNeuralSystemAdapter:
    """Create and initialize the Taichi neural engine and wrap it in the adapter."""
    # Get hybrid configuration
    system_config = config_manager.get_config('system')
    if system_config is None:
        system_config = {}
    
    hybrid_config = config_manager.get_config('hybrid')
    if hybrid_config is None:
        hybrid_config = {}
    
    # Grid size (default: 512x512 for optimal performance)
    grid_size = tuple(hybrid_config.get('grid_size', [512, 512]))
    
    # Device selection
    device = resolve_device(system_config.get('device', 'auto'))
    
    logger.info("="*60)
    logger.info("INITIALIZING HYBRID ENGINE MODE")
    logger.info("="*60)
    logger.info("Grid size: %dx%d = %d cells", grid_size[0], grid_size[1], grid_size[0]*grid_size[1])
    logger.info("Device: %s", device)
    
    # Get hybrid parameters from config
    node_spawn_threshold = hybrid_config.get('node_spawn_threshold', 5.0)
    node_death_threshold = hybrid_config.get('node_death_threshold', -50.0)
    node_energy_cap = hybrid_config.get('node_energy_cap', 500.0)
    spawn_cost = hybrid_config.get('spawn_cost', 2.0)
    transfer_strength = hybrid_config.get('transfer_strength', 0.7)

    logger.info(f"Hybrid params: spawn_threshold={node_spawn_threshold}, "
                f"death_threshold={node_death_threshold}, energy_cap={node_energy_cap}, "
                f"spawn_cost={spawn_cost}, transfer_strength={transfer_strength}")
    
    # Initialize Taichi runtime with config-driven device/memory settings
    device_for_taichi = system_config.get('device', 'auto') or 'auto'
    init_taichi(device=device_for_taichi)

    # Create Taichi engine (single canonical engine, 4M node cap, CUDA)
    logger.info("Creating TaichiNeuralEngine (4M node capacity, CUDA kernels)")
    engine = TaichiNeuralEngine(
        grid_size=grid_size,
        node_spawn_threshold=node_spawn_threshold,
        node_death_threshold=node_death_threshold,
        node_energy_cap=node_energy_cap,
        spawn_cost=spawn_cost,
        transfer_strength=transfer_strength,
        device=device
    )
    
    # Define system regions (scaled to grid size)
    workspace_config = config_manager.get_config('workspace') or {}
    workspace_height = workspace_config.get('height', 16)
    workspace_width = workspace_config.get('width', 16)
    
    # CRITICAL: Clamp sensory region to fit within grid!
    # Sensory input is 1920×1080, grid is now 2560×1920 (accommodates full sensory!)
    # Clamp sensory region to grid boundaries to prevent tensor size mismatch
    sensory_region_y_end = min(sensory_height, grid_size[0])
    sensory_region_x_end = min(sensory_width, grid_size[1])  # Should be 1920 with new grid size
    
    SENSORY_REGION = (0, sensory_region_y_end, 0, sensory_region_x_end)
    WORKSPACE_REGION = (grid_size[0] - workspace_height, grid_size[0], 0, workspace_width)
    
    # ---------------------------------------------------------------
    # Audio regions (placed in unused columns of the bottom strip)
    # ---------------------------------------------------------------
    audio_config = config_manager.get_config('audio') or {}
    audio_enabled = audio_config.get('enabled', False) and AUDIO_AVAILABLE
    fft_bins = audio_config.get('fft_bins', 256)
    audio_rows = workspace_height  # Same row span as visual workspace

    # Bottom strip Y range (same as workspace)
    aw_y0 = grid_size[0] - audio_rows
    aw_y1 = grid_size[0]

    # Column offsets for the four audio regions
    # Validate that audio regions fit within the grid width
    audio_max_x = 1024 + fft_bins  # rightmost edge of AUDIO_WORKSPACE_R
    if audio_enabled and grid_size[1] < audio_max_x:
        logger.warning(
            "Grid width %d is too narrow for audio regions (need %d). "
            "Disabling audio to prevent out-of-bounds writes.",
            grid_size[1], audio_max_x,
        )
        audio_enabled = False

    AUDIO_SENSORY_L_REGION  = (aw_y0, aw_y1, 256, 256 + fft_bins)
    AUDIO_SENSORY_R_REGION  = (aw_y0, aw_y1, 512, 512 + fft_bins)
    AUDIO_WORKSPACE_L_REGION = (aw_y0, aw_y1, 768, 768 + fft_bins)
    AUDIO_WORKSPACE_R_REGION = (aw_y0, aw_y1, 1024, 1024 + fft_bins)

    logger.info("System layout:")
    logger.info("  Sensory region:   [0:%d, 0:%d] = %d nodes (clamped to grid)",
                sensory_region_y_end, sensory_region_x_end, sensory_region_y_end*sensory_region_x_end)
    logger.info("  Dynamic region:   [%d:%d, 0:%d]", sensory_region_y_end, grid_size[0]-workspace_height, grid_size[1])
    logger.info("  Workspace region: [%d:%d, 0:%d] = %d nodes", grid_size[0]-workspace_height, grid_size[0], workspace_width, workspace_height*workspace_width)
    if audio_enabled:
        logger.info("  Audio sensory L:  [%d:%d, 256:%d]", aw_y0, aw_y1, 256 + fft_bins)
        logger.info("  Audio sensory R:  [%d:%d, 512:%d]", aw_y0, aw_y1, 512 + fft_bins)
        logger.info("  Audio workspace L:[%d:%d, 768:%d]", aw_y0, aw_y1, 768 + fft_bins)
        logger.info("  Audio workspace R:[%d:%d, 1024:%d]", aw_y0, aw_y1, 1024 + fft_bins)
    
    # Initialize ALL nodes at once for maximum speed
    logger.info("Initializing all node types...")
    t_start = time.time()
    
    # ARCHITECTURE:
    # - Sensory: Field region only (energy injected, no nodes needed)
    # - Dynamic: Tracked nodes (can spawn/die, type=1)
    # - Workspace: Tracked nodes (immortal/infertile, type=2, can be drained by dynamic)
    
    # Sensory region: [0:sensory_region_y_end, 0:sensory_region_x_end] (clamped to grid!)
    # - Energy injected directly into field each frame
    # - No individual nodes (just a field region)
    # - Input is 1920×1080, but region is clamped to grid size (e.g., 1536×1080)
    sensory_count = sensory_region_y_end * sensory_region_x_end
    logger.info("  Sensory: [0:%d, 0:%d] = %d cells (field region, energy source, clamped to grid)", 
                sensory_region_y_end, sensory_region_x_end, sensory_count)
    
    # Dynamic nodes (can spawn/die, type=1)
    # Seed one dynamic node per sensory column along the bottom edge of the
    # sensory region.  This gives a 1:1 sensory-to-dynamic seeding that will
    # organically balloon outward toward the workspace via ±1 spawning.
    seed_row = sensory_region_y_end  # First row just below sensory
    dynamic_positions = [(seed_row, x) for x in range(sensory_region_x_end)]
    initial_dynamic = len(dynamic_positions)
    dynamic_energies = [100.0] * initial_dynamic
    dynamic_types = [1] * initial_dynamic

    logger.info("  Dynamic: %d initial nodes seeded at sensory border row %d (type=1, will grow via spawning)",
                initial_dynamic, seed_row)
    
    # Workspace nodes (immortal, infertile, type=2)
    # Create a grid of workspace nodes that dynamic nodes can interact with
    workspace_count = workspace_height * workspace_width
    workspace_y_start = grid_size[0] - workspace_height
    # Create workspace nodes on a regular grid
    workspace_positions = [(workspace_y_start + y, x)
                          for y in range(workspace_height)
                          for x in range(workspace_width)]
    workspace_energies = [10.0] * workspace_count  # Start with some energy
    workspace_types = [2] * workspace_count

    # Workspace modality: column-split into thirds (AUDIO_LEFT | VISUAL | AUDIO_RIGHT)
    from project.config import MODALITY_VISUAL, MODALITY_AUDIO_LEFT, MODALITY_AUDIO_RIGHT
    ws_col_third = workspace_width // 3  # ~42 for 128-wide workspace

    def _ws_modality(x: int) -> int:
        if x < ws_col_third:
            return MODALITY_AUDIO_LEFT
        elif x < ws_col_third * 2:
            return MODALITY_VISUAL
        else:
            return MODALITY_AUDIO_RIGHT

    workspace_modalities = [
        _ws_modality(x)
        for y in range(workspace_height)
        for x in range(workspace_width)
    ]
    
    logger.info("  Workspace: %d nodes (type=2, immortal/infertile, at [%d:%d, 0:%d])", 
                workspace_count, workspace_y_start, grid_size[0], workspace_width)
    
    # Audio workspace nodes (immortal, type=2) — placed in L/R audio workspace regions
    audio_ws_positions: list[Tuple[int, int]] = []
    audio_ws_energies: list[float] = []
    audio_ws_types: list[int] = []
    audio_ws_modalities: list[int] = []
    if audio_enabled:
        for region, mod in [
            (AUDIO_WORKSPACE_L_REGION, MODALITY_AUDIO_LEFT),
            (AUDIO_WORKSPACE_R_REGION, MODALITY_AUDIO_RIGHT),
        ]:
            ry0, ry1, rx0, rx1 = region
            for y in range(ry0, ry1):
                for x in range(rx0, rx1):
                    audio_ws_positions.append((y, x))
                    audio_ws_energies.append(5.0)
                    audio_ws_types.append(2)
                    audio_ws_modalities.append(mod)
        logger.info("  Audio workspace: %d nodes (type=2, immortal, L+R)", len(audio_ws_positions))

    # Add ALL node types to the engine
    all_positions = dynamic_positions + workspace_positions + audio_ws_positions
    all_energies = dynamic_energies + workspace_energies + audio_ws_energies
    all_types = dynamic_types + workspace_types + audio_ws_types
    all_modalities = (
        [0] * initial_dynamic           # dynamic: MODALITY_NEUTRAL
        + workspace_modalities
        + audio_ws_modalities
    )

    engine.add_nodes_batch(all_positions, all_energies, all_types, all_modalities)

    t_elapsed = time.time() - t_start
    logger.info("  Initialization complete in %.3fs", t_elapsed)
    total_tracked = initial_dynamic + workspace_count + len(audio_ws_positions)
    logger.info("  Total nodes: %d (dynamic=%d, workspace=%d, audio_ws=%d)",
                total_tracked, initial_dynamic, workspace_count, len(audio_ws_positions))
    
    logger.info("="*60)
    logger.info("HYBRID ENGINE READY")
    logger.info("Architecture: Hybrid field + tracked nodes")
    logger.info("  Sensory: %d cells (field region, raw data = energy)", sensory_count)
    logger.info("  Dynamic: %d seed nodes at sensory border (type=1, will grow via spawning)", initial_dynamic)
    logger.info("  Workspace: %d nodes (type=2, immortal/infertile)", workspace_count)
    logger.info("  Total tracked: %d nodes", total_tracked)
    logger.info("  Grid size: %d×%d = %d cells", grid_size[0], grid_size[1],
                grid_size[0] * grid_size[1])
    logger.info("="*60)

    # Wrap in adapter for compatibility
    return HybridNeuralSystemAdapter(
        engine, SENSORY_REGION, WORKSPACE_REGION, sensory_width, sensory_height,
        audio_sensory_L=AUDIO_SENSORY_L_REGION if audio_enabled else None,
        audio_sensory_R=AUDIO_SENSORY_R_REGION if audio_enabled else None,
        audio_workspace_L=AUDIO_WORKSPACE_L_REGION if audio_enabled else None,
        audio_workspace_R=AUDIO_WORKSPACE_R_REGION if audio_enabled else None,
    )


def initialize_system(
    config_manager: ConfigManager
) -> tuple[HybridNeuralSystemAdapter, ThreadedScreenCapture, WorkspaceNodeSystem | None]:
    """Initialize the neural system, screen capture, and workspace system

    Args:
        config_manager: Configuration manager instance

    Returns:
        Tuple containing initialized neural system, screen capture, and workspace system

    Raises:
        ValueError: If required configurations are not found
        Exception: If system initialization fails
    """
    try:
        # Get configurations with proper type validation
        sensory_config = config_manager.get_config('sensory')
        workspace_config = config_manager.get_config('workspace')
        system_config = config_manager.get_config('system')

        # Check for None configs with more descriptive error messages
        if sensory_config is None:
            raise ValueError("Sensory configuration not found - check pyg_config.json")
        if workspace_config is None:
            raise ValueError("Workspace configuration not found - check pyg_config.json")
        if system_config is None:
            raise ValueError("System configuration not found - check pyg_config.json")

        # Validate configuration types
        if not isinstance(sensory_config, dict):
            raise ValueError(f"Invalid sensory config type: {type(sensory_config)}")
        if not isinstance(workspace_config, dict):
            raise ValueError(f"Invalid workspace config type: {type(workspace_config)}")
        if not isinstance(system_config, dict):
            raise ValueError(f"Invalid system config type: {type(system_config)}")

        # Calculate dynamic nodes with validation
        try:
            width = int(sensory_config['width'])  # type: ignore
            height = int(sensory_config['height'])  # type: ignore
            n_dynamic: int = width * height * 5
            if n_dynamic <= 0:
                raise ValueError("Invalid dynamic node count calculated")
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Missing or invalid sensory configuration: {e}") from e

        # Resolve device preference (cpu/cuda/auto)
        device = resolve_device(system_config.get('device', 'auto'))  # type: ignore

        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:  # pylint: disable=broad-exception-caught
                gpu_name = "CUDA device"
            logger.info("CUDA available, using device=%s (%s)", device, gpu_name)
        else:
            logger.info("CUDA not available, using CPU")

        # Initialize Taichi neural system
        logger.info("="*70)
        logger.info("NEURAL SYSTEM INITIALIZATION (TaichiNeuralEngine)")
        logger.info("="*70)

        try:
            workspace_width  = int(workspace_config['width'])   # type: ignore
            workspace_height = int(workspace_config['height'])  # type: ignore

            system = create_hybrid_neural_system(
                config_manager,
                width,   # type: ignore
                height   # type: ignore
            )
            logger.info("="*70)
            logger.info(">>> TAICHI NEURAL ENGINE INITIALIZED <<<")
            logger.info("="*70)
        except Exception as e:
            logger.error("Failed to initialize neural system: %s", str(e))
            raise RuntimeError(f"Neural system initialization failed: {str(e)}") from e

        # Initialize screen capture with MULTI-TIER fallback system!
        try:
            if OPTIMIZED_CAPTURE_AVAILABLE:
                # Use best available method (auto-detects GPU capabilities!)
                logger.info("Initializing optimized capture (auto-detecting best method)...")
                capture = create_best_capture(
                    region=(0, 0, width, height),
                    device=device  # Use the device variable we set earlier
                )
            else:
                # Fallback to standard capture
                logger.info("Using standard capture (install mss for faster capture: pip install mss)")
                capture = ThreadedScreenCapture(width, height)
        except Exception as e:
            logger.error("Failed to initialize screen capture: %s", str(e))
            raise RuntimeError(f"Screen capture initialization failed: {str(e)}") from e

        # Initialize workspace system with error handling
        try:
            workspace_enabled = workspace_config.get('enabled', True)  # type: ignore
            if workspace_enabled:
                # Create config with values from pyg_config.json
                ws_config = EnergyReadingConfig(
                    grid_size=(workspace_config.get('width', 16), workspace_config.get('height', 16)),
                    reading_interval_ms=workspace_config.get('reading_interval_ms', 50),
                    energy_smoothing=workspace_config.get('energy_smoothing', True),
                    smoothing_factor=workspace_config.get('smoothing_factor', 0.1),
                    shading_mode=workspace_config.get('shading_mode', 'linear'),
                    color_scheme=workspace_config.get('color_scheme', 'grayscale'),
                    animation_enabled=workspace_config.get('animation_enabled', True),
                    animation_speed=workspace_config.get('animation_speed', 0.1),
                    batch_updates=workspace_config.get('batch_updates', True),
                    max_fps=workspace_config.get('max_fps', 60),
                    memory_optimization=workspace_config.get('memory_optimization', True),
                    cache_size=workspace_config.get('cache_size', 1000),
                    cache_validity_ms=workspace_config.get('cache_validity_ms', 100),
                    retry_attempts=workspace_config.get('retry_attempts', 3),
                    retry_delay_ms=workspace_config.get('retry_delay_ms', 10),
                    error_threshold=workspace_config.get('error_threshold', 0.1)
                )
                workspace_system = WorkspaceNodeSystem(system, ws_config)
                logger.info("Workspace system initialized successfully")
            else:
                workspace_system = None
                logger.info("Workspace system disabled in configuration")
        except Exception as e:
            logger.error("Failed to initialize workspace system: %s", str(e))
            raise RuntimeError(f"Workspace system initialization failed: {str(e)}") from e

        logger.info("System initialization completed successfully")
        return system, capture, workspace_system
    except Exception as e:
        logger.error("Failed to initialize system: %s", str(e))
        logger.debug("System initialization error occurred during configuration processing")
        # Enhanced error handling with recovery attempt
        error_msg = f"System initialization failed: {str(e)}"
        logger.error(error_msg)

        # Attempt recovery by validating system state (no recursive retry —
        # if recovery succeeds the caller can retry at a higher level)
        try:
            recovery_success = _attempt_system_recovery()
            if recovery_success:
                logger.info("Recovery cleared state; raising so caller can retry")
            else:
                logger.error("System recovery failed")
        except Exception as recovery_error:  # pylint: disable=broad-exception-caught
            logger.error("Recovery attempt failed: %s", str(recovery_error))

        raise RuntimeError(error_msg) from e

def _attempt_system_recovery() -> bool:
    """
    Attempt to recover the system from initialization failures.

    This function tries to clean up any partially initialized resources
    and reset the system to a clean state for retry.

    Returns:
        True if recovery was successful, False otherwise
    """
    try:
        logger.info("Attempting system recovery...")

        # Clean up global storage
        try:
            from project.system.global_storage import GlobalStorage  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-outside-toplevel,import-error
            GlobalStorage.clear()
            logger.info("Global storage cleared")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Error clearing global storage: %s", str(e))

        # Force garbage collection
        import gc  # pylint: disable=import-outside-toplevel
        gc.collect()
        logger.info("Garbage collection completed")

        # Reset any cached configurations
        try:
            # Reload config from disk to discard any corrupted in-memory state
            ConfigManager.shared().load_config()
            logger.info("Configuration cache reset")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Error resetting configuration cache: %s", str(e))

        logger.info("System recovery completed successfully")
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("System recovery failed: %s", str(e))
        return False

def main() -> None:
    """Main entry point for the PyG Neural System application.

    Handles command line arguments, initializes system components,
    and starts the main application window.

    This function performs the following steps:
    1. Parses command line arguments for log level configuration
    2. Initializes the Qt application (required for UI components)
    3. Sets up configuration and state management
    4. Initializes the neural system and screen capture
    5. Creates and runs the main application window
    6. Manages resource cleanup on exit

    The application follows a structured initialization pattern with
    comprehensive error handling at each stage to ensure robustness.

    Raises:
        RuntimeError: If critical system initialization fails
        Exception: For other unexpected errors during execution

    Note:
        This function should be called as the entry point when running
        the application directly. It handles all Qt application setup
        and teardown automatically.
    """
    parser = argparse.ArgumentParser(
        description='PyG Neural System'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    args = parser.parse_args()

    # Set logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize Qt application first
        try:
            # Check if QApplication already exists
            app: QApplication | None = QApplication.instance()  # type: ignore[assignment]
            if app is None:
                app = QApplication(sys.argv)  # type: ignore[call-overload]
                logger.info("Qt application initialized successfully")
            else:
                logger.info("Using existing Qt application instance")
        except Exception as qt_error:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize Qt application: %s", str(qt_error))
            # Cannot show QMessageBox — QApplication failed to initialize
            raise RuntimeError(f"Qt initialization failed: {str(qt_error)}") from qt_error

        # Initialize managers with error handling
        try:
            config_manager = ConfigManager.shared()
            state_manager = StateManager()
            # Honor config flag for detailed logging at startup
            try:
                detailed_logging = config_manager.get_config('system', 'detailed_logging')
                if isinstance(detailed_logging, bool) and detailed_logging:
                    logging.getLogger().setLevel(logging.DEBUG)
                    logger.info("Detailed logging enabled via config")
            except Exception as log_cfg_error:  # pylint: disable=broad-exception-caught
                logger.warning("Could not apply detailed logging setting: %s", log_cfg_error)
        except Exception as init_error:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize managers: %s", str(init_error))
            init_error_msg = f"Failed to initialize system managers: {str(init_error)}"
            ErrorHandler.show_error("Initialization Error", init_error_msg)  # type: ignore
            raise RuntimeError(f"Manager initialization failed: {str(init_error)}") from init_error

        try:
            with managed_resources() as resources:  # type: ignore[assignment]
                # Initialize system components
                try:
                    system, capture, workspace_system = initialize_system(config_manager)
                    resources.extend([system, capture])
                    if workspace_system:
                        resources.append(workspace_system)
                except Exception as sys_init_error:  # pylint: disable=broad-exception-caught
                    logger.error("System initialization failed: %s", str(sys_init_error))
                    sys_error_msg = f"Failed to initialize system components: {str(sys_init_error)}"
                    ErrorHandler.show_error("System Error", sys_error_msg)  # type: ignore[call-arg]
                    raise

                # Start connection worker with error handling
                try:
                    system.start_connection_worker(batch_size=25)
                    logger.info("Connection worker started successfully")
                except Exception as worker_error:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to start connection worker: %s", str(worker_error))
                    worker_error_msg = f"Failed to start connection worker: {str(worker_error)}"
                    ErrorHandler.show_error(  # type: ignore[union-attr]
                        "Worker Error", worker_error_msg
                    )
                    raise

                # Start screen capture with error handling
                try:
                    capture.start()
                    logger.info("Screen capture started successfully")
                except Exception as capture_error:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to start screen capture: %s", str(capture_error))
                    capture_error_msg = f"Failed to start screen capture: {str(capture_error)}"
                    ErrorHandler.show_error(  # type: ignore[union-attr]
                        "Capture Error", capture_error_msg
                    )
                    raise

                # Initialize audio subsystem if enabled
                audio_capture_obj = None
                audio_output_obj = None
                if AUDIO_AVAILABLE and system.audio_sensory_L is not None:
                    try:
                        audio_cfg = config_manager.get_config('audio') or {}
                        audio_capture_obj = AudioCapture(
                            sample_rate=int(audio_cfg.get('sample_rate', 44100)),
                            fft_size=int(audio_cfg.get('fft_size', 512)),
                            buffer_size=int(audio_cfg.get('buffer_size', 1024)),
                            source=str(audio_cfg.get('source', 'loopback')),
                        )
                        audio_output_obj = AudioOutput(
                            n_bins=int(audio_cfg.get('fft_bins', 256)),
                            sample_rate=int(audio_cfg.get('sample_rate', 44100)),
                            buffer_size=int(audio_cfg.get('buffer_size', 1024)),
                            min_freq=float(audio_cfg.get('min_freq', 80.0)),
                            max_freq=float(audio_cfg.get('max_freq', 8000.0)),
                            master_volume=float(audio_cfg.get('master_volume', 0.3)),
                        )
                        audio_capture_obj.start()
                        audio_output_obj.start()
                        resources.extend([audio_capture_obj, audio_output_obj])
                        logger.info("Audio capture and output started")
                    except Exception as audio_err:
                        logger.warning("Audio init failed (non-fatal): %s", audio_err)
                        audio_capture_obj = None
                        audio_output_obj = None

                # Create main window with error handling
                try:
                    main_window = ModernMainWindow(config_manager, state_manager)
                    resources.append(main_window)
                except Exception as window_error:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to create main window: %s", str(window_error))
                    window_error_msg = f"Failed to create main window: {str(window_error)}"
                    ErrorHandler.show_error(  # type: ignore[union-attr]
                        "UI Error", window_error_msg
                    )
                    raise

                # Provide components to UI but let user explicitly start via button
                try:
                    main_window.set_components(
                        system, capture, workspace_system,
                        audio_capture=audio_capture_obj,
                        audio_output=audio_output_obj,
                    )
                    main_window.show()  # Show the main window
                    logger.info("Entering Qt event loop - application will stay open")
                    app.exec()  # type: ignore[union-attr]
                    logger.info("Qt event loop exited - application closing")
                except Exception as run_error:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to run main window: %s", str(run_error))
                    run_error_msg = f"Failed to run application: {str(run_error)}"
                    ErrorHandler.show_error(  # type: ignore[union-attr]
                        "Runtime Error", run_error_msg
                    )
                    raise

        except Exception as resource_error:  # pylint: disable=broad-exception-caught
            logger.error("Resource management error: %s", str(resource_error))
            resource_error_msg = f"Resource management failed: {str(resource_error)}"
            ErrorHandler.show_error(  # type: ignore[union-attr]
                "Resource Error", resource_error_msg
            )
            raise

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Fatal error (%s): %s", type(e).__name__, str(e))
        fatal_error_msg = f"System failed to start ({type(e).__name__}): {str(e)}"
        ErrorHandler.show_error("Fatal Error", fatal_error_msg)  # type: ignore[union-attr]

        # Clean up Qt application if it was initialized
        try:
            qt_app: QApplication | None = QApplication.instance()  # type: ignore[assignment]
            if qt_app is not None:
                logger.info("Cleaning up Qt application")
                qt_app.quit()  # type: ignore[union-attr]
                qt_app.deleteLater()  # type: ignore[union-attr]
        except Exception as cleanup_error:  # pylint: disable=broad-exception-caught
            logger.warning("Error during Qt cleanup: %s", str(cleanup_error))

        # Add system exit for fatal errors
        sys.exit(1)

if __name__ == '__main__':
    main()
