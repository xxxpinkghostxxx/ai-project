# Architecture Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all Critical and Important issues from the March 2026 architecture review — thread safety, God method extraction, GPU bottleneck, dead code removal, config singleton, and stylesheet deduplication.

**Architecture:** Extract simulation orchestration from `ModernMainWindow` into a dedicated class. Introduce a `threading.Lock` to synchronize engine access between the main thread and workspace background thread. Replace expensive GPU-to-CPU numpy transfers with a Taichi reduction kernel. Defer Taichi initialization until after config loads. Remove ~1,700 lines of dead code. Make `ConfigManager` a shared singleton.

**Tech Stack:** Python 3.11+, Taichi 1.7.x, PyTorch (CUDA), PyQt6, sounddevice

---

## Phase 1: Safety-Critical Fixes (Thread Safety + GPU Init)

### Task 1: Add engine access lock for thread safety (C-2)

The workspace background thread (60 Hz in `workspace_system.py`) reads `energy_field` concurrently with the main Qt thread calling `engine.step()`. There is no synchronization.

**Files:**
- Modify: `src/project/system/taichi_engine.py:523` (add lock)
- Modify: `src/project/system/taichi_engine.py:589-715` (`step()` method)
- Modify: `src/project/system/taichi_engine.py` (`read_workspace_energies()`)
- Test: `tests/test_engine_thread_safety.py`

**Step 1: Write a failing test**

```python
# tests/test_engine_thread_safety.py
"""Verify TaichiNeuralEngine exposes a step_lock for thread-safe access."""
import threading

def test_engine_has_step_lock():
    """Engine must expose a threading.Lock named step_lock."""
    from project.system.taichi_engine import TaichiNeuralEngine
    engine = TaichiNeuralEngine(H=64, W=64)
    assert hasattr(engine, 'step_lock')
    assert isinstance(engine.step_lock, type(threading.Lock()))
    engine.__del__()
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest tests/test_engine_thread_safety.py -v`
Expected: FAIL — `AttributeError: 'TaichiNeuralEngine' has no attribute 'step_lock'`

**Step 3: Add `step_lock` to TaichiNeuralEngine.__init__**

In `src/project/system/taichi_engine.py`, inside `__init__` (around line 523), add:

```python
self.step_lock = threading.Lock()
```

**Step 4: Wrap `step()` body with the lock**

In `step()` (line 589), wrap the body:

```python
def step(self, **kwargs) -> Dict[str, Any]:
    with self.step_lock:
        # ... existing body unchanged ...
```

**Step 5: Wrap `read_workspace_energies()` with the same lock**

Find `read_workspace_energies` method and wrap its GPU read with:

```python
def read_workspace_energies(self, ...):
    with self.step_lock:
        # ... existing body ...
```

**Step 6: Run test to verify it passes**

Run: `cd src && python -m pytest tests/test_engine_thread_safety.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/project/system/taichi_engine.py tests/test_engine_thread_safety.py
git commit -m "fix: add step_lock to TaichiNeuralEngine for thread-safe workspace reads (C-2)"
```

---

### Task 2: Defer Taichi initialization until after config loads (C-3)

Currently `ti.init(arch=cuda, device_memory_fraction=0.6)` runs at import time (line 64 of `taichi_engine.py`), before any config is loaded.

**Files:**
- Modify: `src/project/system/taichi_engine.py:60-65` (replace module-level `ti.init`)
- Modify: `src/project/main.py` (call init before engine creation)

**Step 1: Replace module-level `ti.init()` with a lazy init function**

In `taichi_engine.py`, replace lines 60-65:

```python
# OLD:
_TAICHI_ARCH = ti.cuda if torch.cuda.is_available() else ti.cpu
ti.init(arch=_TAICHI_ARCH, device_memory_fraction=0.6)
logger.info("Taichi initialized: arch=%s", _TAICHI_ARCH)
```

with:

```python
_taichi_initialized = False

def init_taichi(device: str = 'auto', device_memory_fraction: float = 0.6) -> None:
    """Initialize Taichi runtime. Must be called before creating TaichiNeuralEngine.

    Args:
        device: 'auto', 'cuda', or 'cpu'
        device_memory_fraction: fraction of GPU VRAM to reserve (0.0-1.0)
    """
    global _taichi_initialized
    if _taichi_initialized:
        logger.warning("Taichi already initialized, skipping")
        return
    if device == 'auto':
        arch = ti.cuda if torch.cuda.is_available() else ti.cpu
    elif device == 'cuda':
        arch = ti.cuda
    else:
        arch = ti.cpu
    ti.init(arch=arch, device_memory_fraction=device_memory_fraction)
    _taichi_initialized = True
    logger.info("Taichi initialized: arch=%s, memory_fraction=%.2f", arch, device_memory_fraction)
```

**Step 2: Guard field definitions to run after init**

The module-level Taichi field definitions (lines 87-101) must still be at module level for kernel access. Taichi fields are declared at module level but are only materialized on first kernel call, so they work as-is. However, add a guard in `TaichiNeuralEngine.__init__`:

```python
def __init__(self, H=2560, W=2560, ...):
    if not _taichi_initialized:
        init_taichi()  # fallback: init with defaults if caller forgot
    # ... rest of __init__ ...
```

**Step 3: Call `init_taichi()` from `main.py` after config loads**

In `src/project/main.py`, in the `main()` function, after `ConfigManager` loads config but before `TaichiNeuralEngine` is created, add:

```python
from project.system.taichi_engine import init_taichi
# Read device preference from config
device = config_manager.get_config('system', 'device') or 'auto'
init_taichi(device=device)
```

**Step 4: Run the application to verify startup**

Run: `cd src && python -m project.main --log-level DEBUG`
Expected: Log line "Taichi initialized: arch=..." appears AFTER config load messages.

**Step 5: Commit**

```bash
git add src/project/system/taichi_engine.py src/project/main.py
git commit -m "fix: defer Taichi init until after config loads, support device/memory config (C-3)"
```

---

### Task 3: Replace GPU-to-CPU transfer with Taichi reduction kernel (C-4)

The adaptive spawn reads ~48MB from GPU every 10 frames. Replace with an on-GPU reduction.

**Files:**
- Modify: `src/project/system/taichi_engine.py:646-657` (remove `.to_numpy()` calls)
- Modify: `src/project/system/taichi_engine.py` (add reduction kernel + scalar fields)

**Step 1: Add Taichi scalar fields for the reduction**

Near the existing module-level fields (around line 96), add:

```python
_dyn_energy_sum = ti.field(dtype=ti.f64, shape=())   # sum of dynamic node energies
_dyn_node_count = ti.field(dtype=ti.i32, shape=())   # count of alive dynamic nodes
```

**Step 2: Add a reduction kernel**

After the existing kernels, add:

```python
@ti.kernel
def _reduce_dynamic_energy():
    """Compute sum and count of alive dynamic nodes on-GPU (no CPU transfer)."""
    _dyn_energy_sum[None] = 0.0
    _dyn_node_count[None] = 0
    ti.loop_config(block_dim=256)
    for i in range(_node_count[None]):
        state = _node_state[i]
        if state != 0:
            node_type = (state >> 61) & 3
            if node_type == 1:  # dynamic
                ti.atomic_add(_dyn_energy_sum[None], ti.cast(_node_energy[i], ti.f64))
                ti.atomic_add(_dyn_node_count[None], 1)
```

**Step 3: Replace the numpy transfer in `step()`**

In `step()`, replace lines 646-653:

```python
# OLD:
if self.frame_counter % 10 == 0 or not hasattr(self, '_cached_avg_dyn_e'):
    state_np  = _node_state.to_numpy()[:self._count]
    energy_np = _node_energy.to_numpy()[:self._count]
    dyn_mask  = (state_np != 0) & (((state_np >> 61) & 3) == 1)
    dyn_e     = energy_np[dyn_mask]
    self._cached_avg_dyn_e = (
        float(dyn_e.mean()) if len(dyn_e) > 0 else self.spawn_threshold
    )
```

with:

```python
# NEW: On-GPU reduction — reads 2 scalars instead of ~48MB arrays
if self.frame_counter % 10 == 0 or not hasattr(self, '_cached_avg_dyn_e'):
    _reduce_dynamic_energy()
    dyn_count = int(_dyn_node_count[None])
    if dyn_count > 0:
        self._cached_avg_dyn_e = float(_dyn_energy_sum[None]) / dyn_count
    else:
        self._cached_avg_dyn_e = self.spawn_threshold
```

**Step 4: Run the application and verify spawn behavior**

Run: `cd src && python -m project.main --log-level INFO`
Expected: STEP log lines show spawns/deaths as before; no `.to_numpy()` calls in the hot path.

**Step 5: Commit**

```bash
git add src/project/system/taichi_engine.py
git commit -m "perf: replace 48MB GPU-to-CPU transfer with Taichi reduction kernel (C-4)"
```

---

## Phase 2: God Method Extraction (C-1)

### Task 4: Extract `_update_sensory()` from `periodic_update()`

**Files:**
- Modify: `src/project/ui/modern_main_window.py:1420-1492`

**Step 1: Create `_update_sensory()` method**

Extract lines 1443-1492 into a new method. The method returns the frame (or None) and timing dict:

```python
def _update_sensory(self, current_time: float) -> tuple[Any, dict[str, float]]:
    """Process screen capture and update sensory nodes.

    Returns:
        (frame_or_None, timing_dict) where timing_dict has keys:
        t_sensory, t_capture, t_canvas, t_nodes, t_convert
    """
    t = {'t_sensory': 0.0, 't_capture': 0.0, 't_canvas': 0.0, 't_nodes': 0.0, 't_convert': 0.0}
    t_sensory_start = time.time()
    frame = None

    if self.state_manager.get_state().sensory_enabled and self.capture and self.system:
        t_capture_start = time.time()
        frame = self.capture.get_latest()
        t['t_capture'] = (time.time() - t_capture_start) * 1000

    if frame is not None:
        t_convert_start = time.time()
        if isinstance(frame, torch.Tensor):
            sensory_input = frame.float() if frame.dtype == torch.uint8 else frame
        else:
            sensory_input = frame.astype(np.float32)
        t['t_convert'] = (time.time() - t_convert_start) * 1000

        if (current_time - self.last_sensory_canvas_update) > self.sensory_canvas_update_interval:
            t_canvas_start = time.time()
            if isinstance(sensory_input, torch.Tensor):
                canvas_input = sensory_input.cpu().numpy()
                if canvas_input.max() > 1.0:
                    canvas_input = canvas_input / 255.0
            else:
                canvas_input = sensory_input
                if canvas_input.max() > 1.0:
                    canvas_input = canvas_input / 255.0
            self.update_sensory_canvas(canvas_input)
            t['t_canvas'] = (time.time() - t_canvas_start) * 1000
            self.last_sensory_canvas_update = current_time

        t_nodes_start = time.time()
        self.system.update_sensory_nodes(sensory_input)
        t['t_nodes'] = (time.time() - t_nodes_start) * 1000

    t['t_sensory'] = (time.time() - t_sensory_start) * 1000
    return frame, t
```

**Step 2: Replace the sensory block in `periodic_update()` with a call**

Replace lines 1443-1492 with:

```python
frame, sensory_timing = self._update_sensory(current_time)
```

**Step 3: Run the application to verify**

Run: `cd src && python -m project.main`
Expected: Sensory canvas updates as before, no errors.

**Step 4: Commit**

```bash
git add src/project/ui/modern_main_window.py
git commit -m "refactor: extract _update_sensory() from periodic_update() (C-1 step 1/4)"
```

---

### Task 5: Extract `_update_engine()` from `periodic_update()`

**Files:**
- Modify: `src/project/ui/modern_main_window.py:1494-1564`

**Step 1: Create `_update_engine()` method**

```python
def _update_engine(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, float]]:
    """Run one engine step and retrieve metrics.

    Returns:
        (step_result, metrics, timing_dict)
    """
    t = {'t_system': 0.0, 't_update': 0.0, 't_update_step': 0.0, 't_engine_step': 0.0,
         't_worker': 0.0, 't_metrics': 0.0, 't_pre_sync': 0.0, 't_engine_call': 0.0,
         't_result_access': 0.0, 't_gpu_sync': 0.0, 't_adapter': 0.0}
    step_result = None
    metrics = None

    if not self.system:
        return step_result, metrics, t

    t_system_start = time.time()
    t_update_start = time.time()

    if hasattr(self.system, 'update_step'):
        t_update_step_start = time.time()
        step_result = self.system.update_step()
        t['t_update_step'] = (time.time() - t_update_step_start) * 1000
        if isinstance(step_result, dict):
            t['t_engine_step'] = step_result.get('total_time', 0) * 1000
            t['t_pre_sync'] = step_result.get('pre_sync_time', 0) * 1000
            t['t_engine_call'] = step_result.get('engine_call_time', 0) * 1000
            t['t_result_access'] = step_result.get('result_access_time', 0) * 1000
            t['t_gpu_sync'] = step_result.get('gpu_sync_time', 0) * 1000
            t['t_adapter'] = step_result.get('adapter_time', 0) * 1000
    else:
        self.system.update()
    t['t_update'] = (time.time() - t_update_start) * 1000

    t_worker_start = time.time()
    self.system.apply_connection_worker_results()
    t['t_worker'] = (time.time() - t_worker_start) * 1000

    try:
        t_metrics_start = time.time()
        metrics = self.system.get_metrics()
        t['t_metrics'] = (time.time() - t_metrics_start) * 1000
    except Exception as metric_error:
        logger.warning("Metrics retrieval failed: %s", metric_error)

    t['t_system'] = (time.time() - t_system_start) * 1000
    return step_result, metrics, t
```

**Step 2: Replace the engine block in `periodic_update()` with a call**

Replace lines 1494-1564 with:

```python
step_result, metrics, engine_timing = self._update_engine()
```

Remove the now-unnecessary individual timing variable declarations that were at lines 1496-1507.

**Step 3: Run the application to verify**

Run: `cd src && python -m project.main`
Expected: System continues to run, metrics panel updates correctly.

**Step 4: Commit**

```bash
git add src/project/ui/modern_main_window.py
git commit -m "refactor: extract _update_engine() from periodic_update() (C-1 step 2/4)"
```

---

### Task 6: Extract `_update_audio()` from `periodic_update()`

**Files:**
- Modify: `src/project/ui/modern_main_window.py:1576-1597`

**Step 1: Create `_update_audio()` method**

```python
def _update_audio(self, current_time: float) -> None:
    """Process audio input spectrum and update audio output."""
    if (self.audio_capture is None
            or not self.audio_capture.is_running
            or self.system is None):
        return
    try:
        spectrum = self.audio_capture.get_latest()
        self.system.process_audio_frame(spectrum)

        if not hasattr(self, '_last_audio_canvas_t'):
            self._last_audio_canvas_t = 0.0
        if current_time - self._last_audio_canvas_t > 0.1:
            self._update_audio_spectrum_canvas(spectrum)
            self._last_audio_canvas_t = current_time

        if self.audio_output is not None and self.audio_output.is_running:
            ws_data = self.system.get_audio_workspace_energies()
            if ws_data is not None:
                self.audio_output.update_amplitudes(ws_data[0], ws_data[1])
    except Exception as audio_err:
        logger.debug("Audio processing error: %s", audio_err)
```

**Step 2: Replace the audio block in `periodic_update()` with a call**

Replace lines 1576-1597 with:

```python
self._update_audio(current_time)
```

**Step 3: Run and verify**

Run: `cd src && python -m project.main`
Expected: Audio (if enabled) continues to function.

**Step 4: Commit**

```bash
git add src/project/ui/modern_main_window.py
git commit -m "refactor: extract _update_audio() from periodic_update() (C-1 step 3/4)"
```

---

### Task 7: Extract `_log_frame_profiling()` and clean up `periodic_update()`

**Files:**
- Modify: `src/project/ui/modern_main_window.py:1635-1678`

**Step 1: Create `_log_frame_profiling()` method**

```python
def _log_frame_profiling(self, update_start: float, time_since_last: float,
                          sensory_t: dict, engine_t: dict,
                          step_result: dict | None, ui_time: float) -> None:
    """Log detailed profiling info every 90 frames."""
    if self.frame_counter % 90 != 0:
        return
    total_ms = (time.time() - update_start) * 1000
    fps = 1 / time_since_last if time_since_last > 0 else 0

    logger.info(
        "PROFILING | Total: %.1fms | Capture: %.1fms | Convert: %.1fms | "
        "Canvas: %.1fms | Nodes: %.1fms | Update: %.1fms | "
        "UpdateStep: %.1fms | EngineStep: %.1fms | Worker: %.1fms | "
        "Metrics: %.1fms | UI: %.1fms | FPS: %.1f",
        total_ms, sensory_t.get('t_capture', 0), sensory_t.get('t_convert', 0),
        sensory_t.get('t_canvas', 0), sensory_t.get('t_nodes', 0),
        engine_t.get('t_update', 0), engine_t.get('t_update_step', 0),
        engine_t.get('t_engine_step', 0), engine_t.get('t_worker', 0),
        engine_t.get('t_metrics', 0), ui_time * 1000, fps
    )

    gpu_time_ms = step_result.get('gpu_time_ms', 0) if isinstance(step_result, dict) else 0
    logger.info(
        "   GPU BREAKDOWN | PreSync: %.2fms | EngineCall: %.2fms | "
        "ResultAccess: %.2fms | GPUSync: %.2fms | Adapter: %.2fms",
        engine_t.get('t_pre_sync', 0), engine_t.get('t_engine_call', 0),
        engine_t.get('t_result_access', 0), engine_t.get('t_gpu_sync', 0),
        engine_t.get('t_adapter', 0)
    )

    self.status_bar.showMessage(f"Performance: {fps:.1f} FPS | UI: {ui_time*1000:.1f}ms")
```

**Step 2: Replace the profiling block in `periodic_update()` with a call**

Replace the profiling block (lines 1635-1678) and remove status_bar overwrite storm (I-6). The final `periodic_update()` should look approximately like:

```python
@pyqtSlot()
def periodic_update(self) -> None:
    if not self.state_manager.get_state().suspended:
        try:
            update_start = time.time()
            current_time = time.time()
            time_since_last = current_time - self.last_update_time

            if time_since_last < self.min_update_interval:
                self.frame_skip_counter += 1
                if self.frame_skip_counter < self.frame_skip_threshold:
                    return
                else:
                    self.frame_skip_counter = 0
            else:
                self.frame_skip_counter = 0
            self.last_update_time = current_time

            # Subsystem updates
            frame, sensory_t = self._update_sensory(current_time)
            step_result, metrics, engine_t = self._update_engine()
            self._update_audio(current_time)

            # Connection scheduling
            self.frame_counter += 1
            if self.system and metrics:
                edge_count = metrics.get('connection_count', 0)
                dyn_count = metrics.get('dynamic_node_count', 0)
                min_edges = max(100, int(dyn_count * 2.0))
                safe = edge_count >= min_edges
                if not self.frame_counter % self.growth_interval_frames:
                    self.system.queue_connection_growth()
                if safe and not self.frame_counter % self.cull_interval_frames:
                    self.system.queue_cull()

            # UI update
            start_ui = time.time()
            if self.system:
                self.update_metrics_panel(metrics)
                if metrics is not None:
                    self.state_manager.update_metrics(
                        metrics.get('total_energy', 0),
                        metrics.get('dynamic_node_count', 0),
                        metrics.get('connection_count', 0)
                    )
            ui_time = time.time() - start_ui

            # Profiling (every 90 frames)
            self._log_frame_profiling(update_start, time_since_last,
                                       sensory_t, engine_t, step_result, ui_time)

        except Exception as e:
            logger.error("Error during update: %s", str(e))
            ErrorHandler.show_error("Update Error", f"Error during update: {str(e)}",
                                     severity=ERROR_SEVERITY_HIGH)
            self.status_bar.showMessage(f"Error: {str(e)}")
```

**Step 3: Verify the refactored method works**

Run: `cd src && python -m project.main`
Expected: All subsystems function; profiling logs appear every 90 frames.

**Step 4: Commit**

```bash
git add src/project/ui/modern_main_window.py
git commit -m "refactor: extract _log_frame_profiling(), finalize periodic_update() cleanup (C-1 step 4/4, I-6)"
```

---

## Phase 3: Dead Code Removal

### Task 8: Remove `realtime_visualization.py` (I-4, 827 lines)

**Files:**
- Delete: `src/project/workspace/realtime_visualization.py`

**Step 1: Verify the file is never imported**

Run: `grep -r "realtime_visualization" src/project/`
Expected: No imports found (only the file itself).

**Step 2: Delete the file**

```bash
git rm src/project/workspace/realtime_visualization.py
```

**Step 3: Commit**

```bash
git commit -m "chore: remove dead code realtime_visualization.py (827 lines, never imported) (I-4)"
```

---

### Task 9: Remove `ModernResourceManager` (I-3, 677 lines)

The main window creates pixmaps directly; this class is instantiated but never used by the UI.

**Files:**
- Delete: `src/project/ui/modern_resource_manager.py`
- Modify: `src/project/main.py` (remove any references)
- Modify: `src/project/utils/shutdown_utils.py` (remove GlobalStorage retrieval)

**Step 1: Search for all references**

Run: `grep -r "ModernResourceManager\|modern_resource_manager\|ui_resource_manager" src/project/`
Expected: References in `main.py` (import/instantiation), `shutdown_utils.py` (GlobalStorage retrieval), and the file itself.

**Step 2: Remove references from `main.py`**

Find and remove:
- The import of `ModernResourceManager`
- Any instantiation of `ModernResourceManager`

**Step 3: Remove references from `shutdown_utils.py`**

Find the `GlobalStorage.retrieve('ui_resource_manager')` call and remove it (or replace with a no-op if it's in a cleanup chain).

**Step 4: Delete the file**

```bash
git rm src/project/ui/modern_resource_manager.py
```

**Step 5: Run the application to verify startup**

Run: `cd src && python -m project.main`
Expected: Application starts without errors.

**Step 6: Commit**

```bash
git add -u src/project/
git commit -m "chore: remove unused ModernResourceManager (677 lines) (I-3)"
```

---

### Task 10: Remove ErrorHandler recovery stubs (I-2)

The recovery mechanisms are all unimplemented stubs returning `False`.

**Files:**
- Modify: `src/project/utils/error_handler.py:85-89, 392-496`

**Step 1: Remove the recovery mechanism dict and stub methods**

In `__init__`, remove:
```python
self._recovery_mechanisms = {
    'tensor_synchronization': self._recover_via_tensor_synchronization,
    'graph_validation': self._recover_via_graph_validation,
    'connection_repair': self._recover_via_connection_repair
}
```

Remove the following methods entirely:
- `_attempt_recovery` (lines 392-457)
- `_select_recovery_mechanism` (lines 459-481)
- `_recover_via_tensor_synchronization` (lines 483-486)
- `_recover_via_graph_validation` (lines 488-490)
- `_recover_via_connection_repair` (lines 493-496)

**Step 2: Update the class docstring**

Remove the "Recovery Mechanisms" section from the docstring (lines 53-58) and the "Automatic recovery mechanisms" bullet points. Replace with a brief note:

```python
"""
Enhanced error handling class with standardized patterns, severity classification,
and comprehensive error context.

Provides:
- Standardized error handling patterns across the application
- Enhanced logging with detailed error context and severity classification
- Thread-safe error management with locking mechanisms
- Comprehensive error tracking and statistics
"""
```

**Step 3: Verify no callers depend on the removed methods**

Run: `grep -r "_attempt_recovery\|_select_recovery_mechanism\|_recover_via_" src/project/`
Expected: No matches outside `error_handler.py`.

**Step 4: Run the application to verify**

Run: `cd src && python -m project.main`
Expected: No errors related to missing methods.

**Step 5: Commit**

```bash
git add src/project/utils/error_handler.py
git commit -m "chore: remove unimplemented ErrorHandler recovery stubs (~100 lines) (I-2)"
```

---

### Task 11: Remove unused security utilities

Only `ConfigurationSecurityValidator` is used; `SecurityAudit` and `SecureLogger` are dead weight.

**Files:**
- Modify: `src/project/utils/security_utils.py`

**Step 1: Verify usage**

Run: `grep -r "SecurityAudit\|SecureLogger" src/project/ --include="*.py"`
Expected: `SecureLogger` is used in `config_manager.py`. `SecurityAudit` is not used anywhere.

**Step 2: Remove `SecurityAudit` class**

Delete the entire `SecurityAudit` class from `security_utils.py`.

**Step 3: Keep `SecureLogger` if it's actually used**

If `config_manager.py` imports `SecureLogger`, keep it. Only remove `SecurityAudit`.

**Step 4: Verify no import errors**

Run: `cd src && python -c "from project.utils.security_utils import ConfigurationSecurityValidator, SecuritySanitizer, SecureLogger"`
Expected: No ImportError.

**Step 5: Commit**

```bash
git add src/project/utils/security_utils.py
git commit -m "chore: remove unused SecurityAudit class from security_utils.py (I-3 partial)"
```

---

## Phase 4: Config & UI Polish

### Task 12: Make ConfigManager a shared singleton (I-5)

Multiple `ConfigManager` instances can overwrite each other and have stale caches.

**Files:**
- Modify: `src/project/utils/config_manager.py`

**Step 1: Add a class-level shared instance**

Add at the top of the `ConfigManager` class:

```python
class ConfigManager:
    _instance: 'ConfigManager | None' = None

    @classmethod
    def shared(cls) -> 'ConfigManager':
        """Return the shared ConfigManager instance (created on first call)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

Do NOT make `__init__` private — existing per-use instantiation still works for backward compatibility. The `shared()` classmethod is the new recommended pattern.

**Step 2: Update main.py to use `ConfigManager.shared()`**

In `main.py`, replace `ConfigManager()` with `ConfigManager.shared()`.

**Step 3: Update `modern_main_window.py` to use `ConfigManager.shared()`**

Replace `ConfigManager()` with `ConfigManager.shared()`.

**Step 4: Update `modern_config_panel.py` to use `ConfigManager.shared()`**

Replace `ConfigManager()` with `ConfigManager.shared()`.

**Step 5: Verify all components see the same config**

Run: `cd src && python -c "from project.utils.config_manager import ConfigManager; a = ConfigManager.shared(); b = ConfigManager.shared(); print(a is b)"`
Expected: `True`

**Step 6: Commit**

```bash
git add src/project/utils/config_manager.py src/project/main.py src/project/ui/modern_main_window.py src/project/ui/modern_config_panel.py
git commit -m "fix: add ConfigManager.shared() singleton pattern (I-5)"
```

---

### Task 13: Deduplicate button stylesheets (I-1)

**Files:**
- Modify: `src/project/ui/modern_main_window.py`

**Step 1: Add a `_button_style()` helper method**

Add this method to `ModernMainWindow`:

```python
@staticmethod
def _button_style(bg: str, hover: str = '', pressed: str = '') -> str:
    """Generate QPushButton stylesheet with consistent structure."""
    hover = hover or bg
    pressed = pressed or bg
    return (
        f"QPushButton {{ background-color: {bg}; color: #e0e0e0; "
        f"border: none; border-radius: 4px; padding: 10px; font-weight: bold; }}"
        f"QPushButton:hover {{ background-color: {hover}; }}"
        f"QPushButton:pressed {{ background-color: {pressed}; }}"
    )
```

**Step 2: Replace all inline stylesheet blocks**

Search for all `setStyleSheet("""` blocks on QPushButtons and replace with calls to `_button_style()`. For example:

```python
# OLD (14 lines):
self.suspend_button.setStyleSheet("""
    QPushButton {
        background-color: #225522;
        ...
    }
    ...
""")

# NEW (1 line):
self.suspend_button.setStyleSheet(self._button_style('#225522', '#337733', '#113311'))
```

Repeat for all ~10 occurrences. Common colors:
- Suspend/resume green: `('#225522', '#337733', '#113311')`
- Stop/disable red: `('#882222', '#993333', '#661111')`
- Pulse blue: `('#225577', '#3377aa', '#113355')`
- Success green: `('#33aa33',)` (no hover/pressed)
- Sensory on green: `('#228822', '#33aa33', '#115511')`

**Step 3: Run and verify visual appearance**

Run: `cd src && python -m project.main`
Expected: All buttons look identical to before.

**Step 4: Commit**

```bash
git add src/project/ui/modern_main_window.py
git commit -m "refactor: deduplicate button stylesheets with _button_style() helper (~150 lines saved) (I-1)"
```

---

### Task 14: Fix sensory canvas pixmap item reuse (M-7)

**Files:**
- Modify: `src/project/ui/modern_main_window.py` (the `update_sensory_canvas` method)

**Step 1: Find `update_sensory_canvas` and fix the clear/recreate pattern**

The current code does:
```python
self.sensory_scene.clear()
self.sensory_scene.addPixmap(pixmap)
```

Replace with pixmap-item reuse (matching the workspace canvas pattern):

```python
# In __init__ or _create_sensory_canvas, add:
self._sensory_pixmap_item = None

# In update_sensory_canvas:
if self._sensory_pixmap_item is None:
    self._sensory_pixmap_item = self.sensory_scene.addPixmap(pixmap)
else:
    self._sensory_pixmap_item.setPixmap(pixmap)
```

**Step 2: Remove the `self.sensory_scene.clear()` call**

**Step 3: Commit**

```bash
git add src/project/ui/modern_main_window.py
git commit -m "fix: reuse sensory canvas pixmap item to prevent flicker (M-7)"
```

---

### Task 15: Fix duplicate `time.time()` calls (M-3)

**Files:**
- Modify: `src/project/ui/modern_main_window.py:1425-1426`

**Step 1: Replace the two calls with one**

```python
# OLD:
update_start = time.time()
current_time = time.time()

# NEW:
current_time = time.time()
update_start = current_time
```

Note: This may already be addressed by Task 7's rewrite of `periodic_update()`. If so, skip this task.

**Step 2: Commit**

```bash
git add src/project/ui/modern_main_window.py
git commit -m "fix: remove duplicate time.time() call in periodic_update() (M-3)"
```

---

## Summary

| Task | Issue | Phase | Lines Changed |
|------|-------|-------|---------------|
| 1 | C-2: Thread safety lock | 1 | +15 |
| 2 | C-3: Deferred Taichi init | 1 | +25, -3 |
| 3 | C-4: GPU reduction kernel | 1 | +25, -8 |
| 4 | C-1: Extract `_update_sensory()` | 2 | ~0 net (move) |
| 5 | C-1: Extract `_update_engine()` | 2 | ~0 net (move) |
| 6 | C-1: Extract `_update_audio()` | 2 | ~0 net (move) |
| 7 | C-1+I-6: Final periodic_update cleanup | 2 | -50 |
| 8 | I-4: Remove `realtime_visualization.py` | 3 | -827 |
| 9 | I-3: Remove `ModernResourceManager` | 3 | -677 |
| 10 | I-2: Remove ErrorHandler stubs | 3 | -100 |
| 11 | Dead security utils | 3 | -100~ |
| 12 | I-5: ConfigManager singleton | 4 | +15, ~5 call sites |
| 13 | I-1: Stylesheet deduplication | 4 | -150 |
| 14 | M-7: Sensory pixmap reuse | 4 | +5, -2 |
| 15 | M-3: Duplicate time.time() | 4 | +1, -1 |

**Estimated net reduction: ~1,700 lines of dead code + ~200 lines of duplication.**
