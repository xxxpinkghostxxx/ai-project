# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Constants:
#   OPTIMIZED_CAPTURE_AVAILABLE = bool
#     True when project.optimized_capture is importable
#   AUDIO_AVAILABLE = bool
#     True when project.audio_capture/audio_output are importable
#
# Module-level Functions:
#   _place_clusters(grid_H: int, grid_W: int, grid_D: int, count: int,
#       nodes_each: int, radius: int, min_separation: int) -> tuple
#     Place count node clusters in a 3D volume with minimum separation
#
#   _build_neutral_dna(n: int, device) -> torch.Tensor
#     Build neutral DNA (MODE=0, PARAM=1111) for all 26 neighbor slots
#
#   resolve_device(pref: str | None) -> str
#     Resolve device preference string to 'cuda' or 'cpu'
#
#   managed_resources() -> Generator[list[Any], None, None]    @contextlib.contextmanager
#     Context manager for system resources with lifecycle management
#
#   create_hybrid_neural_system(config_manager: ConfigManager,
#       sensory_width: int, sensory_height: int) -> HybridNeuralSystemAdapter
#     Create TaichiNeuralEngine, configure regions, seed nodes, wrap in adapter
#
#   initialize_system(config_manager: ConfigManager)
#       -> tuple[HybridNeuralSystemAdapter, ThreadedScreenCapture,
#          WorkspaceNodeSystem | None]
#     Initialize neural system, screen capture, and workspace system
#
#   _attempt_system_recovery() -> bool
#     Clean up partial state after initialization failure
#
#   main() -> None
#     Entry point: parse args, init Qt, create system, run event loop
#
# Classes:
#   HybridNeuralSystemAdapter:
#     Adapter making TaichiNeuralEngine compatible with existing UI/workspace code
#
#     __init__(self, engine: TaichiNeuralEngine,
#         sensory_region: Tuple[int, int, int, int],
#         workspace_region: Tuple[int, int, int, int],
#         sensory_width: int = 64, sensory_height: int = 64,
#         audio_sensory_L: Tuple[...] | None = None,
#         audio_sensory_R: Tuple[...] | None = None,
#         audio_workspace_L: Tuple[...] | None = None,
#         audio_workspace_R: Tuple[...] | None = None,
#         sensory_cluster_centers: list | None = None,
#         sensory_cluster_radius: int = 3)
#
#     process_frame(self, frame_data: Any) -> dict
#       Process screen capture frame; per-cluster or full-region injection -> step()
#
#     update_step(self) -> dict
#       Single update step with GPU sync timing
#
#     get_workspace_energies(self) -> torch.Tensor
#       Read workspace region energies from engine
#
#     get_workspace_energies_grid(self) -> np.ndarray
#       Get workspace energies as 2D numpy array; cached for 8ms
#
#     get_workspace_node_energy(self, node_id: int) -> float
#       Get energy for a single workspace node (deprecated, prefer grid)
#
#     get_node_count(self) -> int
#       Alive node count (high-water mark, no GPU roundtrip)
#
#     get_energy_stats(self) -> dict
#       Energy field min/max/avg statistics
#
#     pulse_energy(self) -> float
#       No-op (ADR-001): external energy injection disabled
#
#     apply_config(self, config_manager: ConfigManager) -> None
#       Push hybrid config values into running engine (hot-reload)
#
#     process_audio_frame(self, spectrum: np.ndarray) -> None
#       Inject stereo FFT spectrum into audio sensory regions
#
#     get_audio_workspace_energies(self)
#         -> Tuple[np.ndarray, np.ndarray] | None
#       Read energy from both audio workspace regions; cached 8ms
#
#     stop_connection_worker(self) -> None
#       No-op compatibility stub
#
#     wait_for_workers_idle(self) -> None
#       No-op compatibility stub
#
#     get_metrics(self) -> dict
#       Return performance metrics; cached for 100ms
#
#     queue_cull(self) -> None
#       No-op compatibility stub
#
#     queue_connection_growth(self) -> None
#       No-op compatibility stub
#
#     start_connection_worker(self, batch_size: int = 25) -> None
#       No-op compatibility stub
#
#     update_sensory_nodes(self, sensory_input: Any) -> None
#       Convert sensory pixels to energy, inject into sensory region
#
#     update(self) -> None
#       Run single simulation step (delegates to update_step)
#
#     apply_connection_worker_results(self) -> None
#       No-op compatibility stub
#
#     shutdown(self) -> None
#       Shutdown the system
#
#     cleanup -> shutdown (alias)
#     stop -> shutdown (alias)
#
# =============================================================================
# TODOS
# =============================================================================
#
# None
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Main entry point and system initialization for the Taichi Neural System."""

import sys
import os
import time
import threading
import random
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
import logging
import contextlib
import argparse
from typing import Any, Tuple
from collections.abc import Generator
import numpy as np
import torch

# pylint: disable=wrong-import-position
from PyQt6.QtWidgets import QApplication  # type: ignore[import-untyped]  # pylint: disable=no-name-in-module

from project.vision import ThreadedScreenCapture  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
try:
    from project.optimized_capture import create_best_capture  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
    OPTIMIZED_CAPTURE_AVAILABLE = True
except ImportError:
    OPTIMIZED_CAPTURE_AVAILABLE = False
from project.utils.error_handler import ErrorHandler  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.utils.config_manager import ConfigManager  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.system.state_manager import StateManager  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.ui.modern_main_window import ModernMainWindow  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.workspace.workspace_system import WorkspaceNodeSystem  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.workspace.config import EnergyReadingConfig  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.system.taichi_engine import TaichiNeuralEngine, init_taichi  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error

try:
    from project.audio_capture import AudioCapture  # type: ignore[import-untyped,import-not-found]
    from project.audio_output import AudioOutput    # type: ignore[import-untyped,import-not-found]
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pyg_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if OPTIMIZED_CAPTURE_AVAILABLE:
    logger.info("Optimized capture available (will auto-select best method)")
else:
    logger.info("Using standard capture (install mss for faster capture: pip install mss)")


def _place_clusters(
    grid_H: int, grid_W: int, grid_D: int,
    count: int, nodes_each: int, radius: int, min_separation: int,
) -> tuple:
    """Place *count* node clusters scattered through a [H, W, D] 3D volume.

    Each cluster has *nodes_each* nodes within a sphere of *radius* cells.
    Centers are chosen at random with a minimum pairwise distance of
    *min_separation* cells.  Returns ``(centers, positions)`` where *centers*
    is a list of ``(cy, cx, cz)`` and *positions* is a flat list of
    ``(y, x, z)`` tuples.
    """
    centers = []
    attempts = 0
    while len(centers) < count and attempts < count * 1000:
        attempts += 1
        cy = random.randint(0, grid_H - 1)
        cx = random.randint(0, grid_W - 1)
        cz = random.randint(0, grid_D - 1)
        too_close = any(
            math.sqrt((cy - oy)**2 + (cx - ox)**2 + (cz - oz)**2) < min_separation
            for oy, ox, oz in centers
        )
        if not too_close:
            centers.append((cy, cx, cz))

    positions = []
    for cy, cx, cz in centers:
        placed = 0
        node_attempts = 0
        while placed < nodes_each and node_attempts < nodes_each * 100:
            node_attempts += 1
            dy = random.randint(-radius, radius)
            dx = random.randint(-radius, radius)
            dz = random.randint(-radius, radius)
            if dy**2 + dx**2 + dz**2 <= radius**2:
                ny = max(0, min(grid_H - 1, cy + dy))
                nx = max(0, min(grid_W - 1, cx + dx))
                nz = max(0, min(grid_D - 1, cz + dz))
                positions.append((ny, nx, nz))
                placed += 1

    return centers, positions


def _build_neutral_dna(n: int, device) -> 'torch.Tensor':
    """Build neutral DNA: MODE=0, PARAM=1111 (15) for all 26 neighbor slots.

    Returns an [n, 3] int64 tensor.  Each 5-bit slot is 0b01111 = 15, meaning
    max lock-and-key compatibility with any other DNA.
    """
    import torch
    from project.system.taichi_engine import DNA_SLOT_WORD, DNA_SLOT_BIT, NUM_NEIGHBORS_3D
    dna = torch.zeros(n, 3, device=device, dtype=torch.int64)
    for s in range(NUM_NEIGHBORS_3D):
        word = DNA_SLOT_WORD[s]
        bit  = DNA_SLOT_BIT[s]
        dna[:, word] |= (15 << bit)
    return dna


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
        sensory_cluster_centers: list | None = None,
        sensory_cluster_radius: int = 3,
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
            sensory_cluster_centers: List of (cy, cx, cz) cluster centers
            sensory_cluster_radius: Radius around each center for injection
        """
        self.engine = engine
        self.sensory_region = sensory_region
        self.workspace_region = workspace_region
        self.sensory_width = sensory_width
        self.sensory_height = sensory_height
        self.device = engine.device

        self.audio_sensory_L = audio_sensory_L
        self.audio_sensory_R = audio_sensory_R
        self.audio_workspace_L = audio_workspace_L
        self.audio_workspace_R = audio_workspace_R

        self._sensory_cluster_centers = sensory_cluster_centers or []
        self._sensory_cluster_radius = sensory_cluster_radius

        self._connection_worker_started = False

        self._step_count = 0
        self._metrics_lock = threading.Lock()
        logger.info("Hybrid adapter initialized")
    
    def process_frame(self, frame_data: Any) -> dict:
        """Process a screen capture frame (compatible with ThreadedScreenCapture).

        Optimized for real-time performance with GPU.
        """
        if isinstance(frame_data, torch.Tensor):
            pixels = frame_data.float()
        else:
            pixels = torch.tensor(frame_data, device=self.device).float()

        if self._sensory_cluster_centers:
            import torch.nn.functional as F
            small = F.interpolate(
                pixels.unsqueeze(0).unsqueeze(0),
                size=(self.engine.H, self.engine.W),
                mode='bilinear', align_corners=False,
            ).squeeze()
            r = self._sensory_cluster_radius
            for cy, cx, cz in self._sensory_cluster_centers:
                y0, y1 = max(0, cy - r), min(self.engine.H, cy + r + 1)
                x0, x1 = max(0, cx - r), min(self.engine.W, cx + r + 1)
                patch = small[y0:y1, x0:x1]
                self.engine.inject_sensory_data(patch, region=(y0, y1, x0, x1), z=cz)
        else:
            self.engine.inject_sensory_data(pixels, self.sensory_region)

        return self.engine.step()

    def update_step(self) -> dict:
        """Single update step (compatible with UI update loop)."""
        t_step_start = time.time()
        self._step_count += 1

        t_engine_call_start = time.time()
        result = self.engine.step()
        t_engine_call_done = time.time()

        t_gpu_sync_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gpu_sync_done = time.time()

        t_step_done = time.time()

        result['adapter_time']     = t_step_done - t_step_start
        result['pre_sync_time']    = 0.0
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
            flat = energies_tensor.cpu().numpy().copy()
            flat = flat.ravel()

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
        energies = self.get_workspace_energies()

        workspace_height = self.workspace_region[1] - self.workspace_region[0]
        workspace_width = self.workspace_region[3] - self.workspace_region[2]
        
        if node_id >= workspace_height * workspace_width:
            return 0.0
        
        row = node_id // workspace_width
        col = node_id % workspace_width
        
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
        """REMOVED (ADR-001): external energy injection violates node economics.

        Global energy pulses bypass the sensory→dynamic→workspace flow and
        create phantom energy that corrupts the system's thermodynamic balance.
        This method is retained as a no-op to avoid breaking callers; remove
        any UI button that triggers it.

        Returns 0.0 without modifying the field.
        """
        logger.warning(
            "pulse_energy() called but is disabled (ADR-001). "
            "Remove the call site — this method will be deleted in a future release."
        )
        return 0.0

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
            ch_data = spectrum[ch]
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
            if hasattr(self, '_last_metrics_time'):
                if current_time - self._last_metrics_time < 0.1:
                    return self._cached_metrics.copy()

            engine_m = self.engine.get_metrics()
            dynamic_count   = engine_m['dynamic_count']
            workspace_count = engine_m['workspace_count']

            sy0_m, sy1_m, sx0_m, sx1_m = self.sensory_region
            sensory_count = (sy1_m - sy0_m) * (sx1_m - sx0_m)
            total_nodes   = engine_m['alive_count'] + sensory_count

            total_energy = engine_m['total_energy']

            wy0, wy1, wx0, wx1 = self.workspace_region
            ws_field = self.engine.energy_field[wy0:wy1, wx0:wx1]
            workspace_energy_avg = float(ws_field.mean().item())
            workspace_energy_min = float(ws_field.min().item())
            workspace_energy_max = float(ws_field.max().item())

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
                'conns_per_dynamic':    0.0,
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
                'avg_dynamic_energy':   avg_energy,
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
        pass

    def queue_connection_growth(self) -> None:
        """Queue connection growth (compatibility method for UI).
        
        Hybrid engine doesn't use explicit connections, so this is a no-op.
        """
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
        if isinstance(sensory_input, np.ndarray):
            sensory_input = torch.tensor(sensory_input, device=self.device, dtype=torch.float32)
        
        if sensory_input.shape != (self.sensory_height, self.sensory_width):
            if sensory_input.numel() == self.sensory_height * self.sensory_width:
                sensory_input = sensory_input.reshape(self.sensory_height, self.sensory_width)
            else:
                return
        
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
        pass

    def shutdown(self) -> None:
        """Shutdown the system (cleanup/stop are aliases)."""
        logger.info("Shutting down hybrid engine")

    cleanup = shutdown
    stop = shutdown


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
    system_config = config_manager.get_config('system')
    if system_config is None:
        system_config = {}
    
    hybrid_config = config_manager.get_config('hybrid')
    if hybrid_config is None:
        hybrid_config = {}
    
    grid_size = tuple(hybrid_config.get('grid_size', [512, 512]))
    
    device = resolve_device(system_config.get('device', 'auto'))
    
    logger.info("="*60)
    logger.info("INITIALIZING HYBRID ENGINE MODE")
    logger.info("="*60)
    logger.info("Grid size: %dx%d = %d cells", grid_size[0], grid_size[1], grid_size[0]*grid_size[1])
    logger.info("Device: %s", device)
    
    node_spawn_threshold = hybrid_config.get('node_spawn_threshold', 5.0)
    node_death_threshold = hybrid_config.get('node_death_threshold', -50.0)
    node_energy_cap = hybrid_config.get('node_energy_cap', 500.0)
    spawn_cost = hybrid_config.get('spawn_cost', 2.0)
    transfer_strength = hybrid_config.get('transfer_strength', 0.7)

    logger.info(f"Hybrid params: spawn_threshold={node_spawn_threshold}, "
                f"death_threshold={node_death_threshold}, energy_cap={node_energy_cap}, "
                f"spawn_cost={spawn_cost}, transfer_strength={transfer_strength}")
    
    device_for_taichi = system_config.get('device', 'auto') or 'auto'
    init_taichi(device=device_for_taichi)

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
    
    H, W = grid_size[0], grid_size[1]
    D = grid_size[2] if len(grid_size) > 2 else 8

    workspace_config = config_manager.get_config('workspace') or {}
    workspace_height = min(workspace_config.get('height', 16), H // 4)
    workspace_width = min(workspace_config.get('width', 16), W)

    sensory_region_y_end = min(sensory_height, H)
    sensory_region_x_end = min(sensory_width, W)

    SENSORY_REGION   = (0, sensory_region_y_end, 0, sensory_region_x_end)
    WORKSPACE_REGION = (H - workspace_height, H, 0, workspace_width)

    engine.register_region(
        y0=0,    y1=H,
        x0=0,    x1=W,
        region_type=1,
        spawn=True,
    )

    logger.info("Region layout: single dynamic region [0,%d)×[0,%d)×%d  (all spawn-enabled)", H, W, D)
    logger.info("  Sensory injection box: [0,%d)×[0,%d)", sensory_region_y_end, sensory_region_x_end)
    logger.info("  Workspace read-back:   [%d,%d)×[0,%d)", H - workspace_height, H, workspace_width)
    logger.info("Region registry: %s", engine.get_region_info())

    audio_config = config_manager.get_config('audio') or {}
    audio_enabled = audio_config.get('enabled', False) and AUDIO_AVAILABLE
    fft_bins = audio_config.get('fft_bins', 256)
    audio_rows = workspace_height

    aw_y0 = H - audio_rows
    aw_y1 = H

    audio_max_x = 1024 + fft_bins
    if audio_enabled and W < audio_max_x:
        logger.warning(
            "Grid width %d is too narrow for audio regions (need %d). "
            "Disabling audio to prevent out-of-bounds writes.",
            W, audio_max_x,
        )
        audio_enabled = False

    AUDIO_SENSORY_L_REGION  = (aw_y0, aw_y1, 256, 256 + fft_bins)
    AUDIO_SENSORY_R_REGION  = (aw_y0, aw_y1, 512, 512 + fft_bins)
    AUDIO_WORKSPACE_L_REGION = (aw_y0, aw_y1, 768, 768 + fft_bins)
    AUDIO_WORKSPACE_R_REGION = (aw_y0, aw_y1, 1024, 1024 + fft_bins)

    import torch
    from project.config import (
        NODE_TYPE_SENSORY, NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE,
        MODALITY_VISUAL, MODALITY_AUDIO_LEFT, MODALITY_AUDIO_RIGHT,
    )

    SEED_COUNT = 1000

    logger.info("Seeding %d neutral-DNA nodes in sensory box [0,%d)×[0,%d) at z=0...",
                SEED_COUNT, sensory_region_y_end, sensory_region_x_end)
    t_start = time.time()

    seed_positions = [
        (random.randint(0, sensory_region_y_end - 1),
         random.randint(0, sensory_region_x_end - 1),
         0)
        for _ in range(SEED_COUNT)
    ]

    neutral_dna = _build_neutral_dna(SEED_COUNT, device)

    engine.add_nodes_batch(
        seed_positions,
        [10.0] * SEED_COUNT,
        [NODE_TYPE_DYNAMIC] * SEED_COUNT,
        [0] * SEED_COUNT,
        dna=neutral_dna,
    )

    t_elapsed = time.time() - t_start

    logger.info("=" * 60)
    logger.info("HYBRID ENGINE READY")
    logger.info("  Grid: %d×%d×%d = %d cells", H, W, D, H * W * D)
    logger.info("  Seed nodes: %d (neutral DNA, sensory box z=0)", SEED_COUNT)
    logger.info("  Init time: %.3fs", t_elapsed)
    logger.info("=" * 60)

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
        sensory_config = config_manager.get_config('sensory')
        workspace_config = config_manager.get_config('workspace')
        system_config = config_manager.get_config('system')

        if sensory_config is None:
            raise ValueError("Sensory configuration not found - check pyg_config.json")
        if workspace_config is None:
            raise ValueError("Workspace configuration not found - check pyg_config.json")
        if system_config is None:
            raise ValueError("System configuration not found - check pyg_config.json")

        if not isinstance(sensory_config, dict):
            raise ValueError(f"Invalid sensory config type: {type(sensory_config)}")
        if not isinstance(workspace_config, dict):
            raise ValueError(f"Invalid workspace config type: {type(workspace_config)}")
        if not isinstance(system_config, dict):
            raise ValueError(f"Invalid system config type: {type(system_config)}")

        try:
            width = int(sensory_config['width'])  # type: ignore
            height = int(sensory_config['height'])  # type: ignore
            n_dynamic: int = width * height * 5
            if n_dynamic <= 0:
                raise ValueError("Invalid dynamic node count calculated")
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Missing or invalid sensory configuration: {e}") from e

        device = resolve_device(system_config.get('device', 'auto'))  # type: ignore

        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:  # pylint: disable=broad-exception-caught
                gpu_name = "CUDA device"
            logger.info("CUDA available, using device=%s (%s)", device, gpu_name)
        else:
            logger.info("CUDA not available, using CPU")

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

        try:
            if OPTIMIZED_CAPTURE_AVAILABLE:
                logger.info("Initializing optimized capture (auto-detecting best method)...")
                capture = create_best_capture(
                    region=(0, 0, width, height),
                    device=device
                )
            else:
                logger.info("Using standard capture (install mss for faster capture: pip install mss)")
                capture = ThreadedScreenCapture(width, height)
        except Exception as e:
            logger.error("Failed to initialize screen capture: %s", str(e))
            raise RuntimeError(f"Screen capture initialization failed: {str(e)}") from e

        try:
            workspace_enabled = workspace_config.get('enabled', True)  # type: ignore
            if workspace_enabled:
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
        error_msg = f"System initialization failed: {str(e)}"
        logger.error(error_msg)

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

        try:
            from project.system.global_storage import GlobalStorage  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-outside-toplevel,import-error
            GlobalStorage.clear()
            logger.info("Global storage cleared")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Error clearing global storage: %s", str(e))

        import gc  # pylint: disable=import-outside-toplevel
        gc.collect()
        logger.info("Garbage collection completed")

        try:
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

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        try:
            app: QApplication | None = QApplication.instance()  # type: ignore[assignment]
            if app is None:
                app = QApplication(sys.argv)  # type: ignore[call-overload]
                logger.info("Qt application initialized successfully")
            else:
                logger.info("Using existing Qt application instance")
        except Exception as qt_error:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize Qt application: %s", str(qt_error))
            raise RuntimeError(f"Qt initialization failed: {str(qt_error)}") from qt_error

        try:
            config_manager = ConfigManager.shared()
            state_manager = StateManager()
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

                try:
                    main_window.set_components(
                        system, capture, workspace_system,
                        audio_capture=audio_capture_obj,
                        audio_output=audio_output_obj,
                    )
                    main_window.show()
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

        try:
            qt_app: QApplication | None = QApplication.instance()  # type: ignore[assignment]
            if qt_app is not None:
                logger.info("Cleaning up Qt application")
                qt_app.quit()  # type: ignore[union-attr]
                qt_app.deleteLater()  # type: ignore[union-attr]
        except Exception as cleanup_error:  # pylint: disable=broad-exception-caught
            logger.warning("Error during Qt cleanup: %s", str(cleanup_error))

        sys.exit(1)

if __name__ == '__main__':
    main()
