# Performance Guide

## Current Performance

| System Size | Nodes | Avg Time (s) | Updates/s |
|-------------|-------|--------------|-----------|
| Small       | 10    | 0.000854     | 1,170     |
| Medium      | 40    | 0.000842     | 1,187     |
| Large       | 130   | 0.000837     | 1,195     |
| XLarge      | 420   | 0.000872     | 1,146     |

**Peak performance**: 1,195 updates/second (vectorized energy calculations, 41x improvement over baseline).

### Hybrid Engine Performance

| Environment | Grid Size | Ops/second | Speedup |
|-------------|-----------|-----------|---------|
| CPU | 512x512 | 5.45 billion | 5,243x |
| GPU (expected) | 1024x1024 | 20+ billion | 20,000x+ |

---

## Known Bottlenecks

### 1. Blocking GPU Syncs

Multiple `.item()` calls force GPU-to-CPU synchronization, blocking execution every frame.

**Solution**: Cache results and only sync periodically:
```python
if self.frame_counter % 10 == 0:
    energy_sum = self.energy_field.sum().item()
    self._cached_energy_sum = energy_sum
else:
    energy_sum = self._cached_energy_sum
```
**Expected gain**: 5-10ms per frame

### 2. Workspace Seeding Loop

Nested loops with individual tensor operations in `probabilistic_field_engine.py` (workspace seeding neighborhood loop, search for `seed_radius`).

**Solution**: Vectorize the entire operation using `torch.meshgrid` and batch operations.
**Expected gain**: 10-20ms per frame (25x faster for 5x5 neighborhood)

### 3. FFT Called on Sparse Fields

FFT diffusion runs even when energy field is mostly zero.

**Solution**: Check sparsity and skip FFT when field is sparse:
```python
if self._cached_non_zero_ratio > 0.01:
    self.energy_field = self.fft_diffusion(self.energy_field, dt)
```
**Expected gain**: 2-5ms per frame when field is sparse

### 4. Redundant Clamp Operations

Multiple clamp + finite check passes on the same tensors.

**Solution**: Fuse into single operation:
```python
result = torch.where(
    torch.isfinite(result),
    torch.clamp(result, min=self.energy_min, max=self.energy_max),
    torch.zeros_like(result)
)
```
**Expected gain**: 1-2ms per frame

### 5. CPU-GPU Transfers in UI

`get_workspace_energies_grid()` blocks on `.cpu().tolist()`.

**Solution**: Use pinned memory and async transfer with CUDA streams.
**Expected gain**: 2-5ms per frame

---

## Known Bugs

### 1. Float16 Precision Loss (probabilistic_field_engine.py)

Converting `pull_amount` to float16 and back causes precision loss when accumulating small values (search for `BYTE OPTIMIZATION: Ensure float16 dtype is preserved`).

**Fix**: Keep calculations in float32, only convert to float16 for storage:
```python
new_ws_energy = self.workspace_node_energies.float() + pull_amount
self.workspace_node_energies = new_ws_energy.half()
```

### 2. Race Condition in Workspace System (workspace_system.py)

`_notify_observers()` stores `energy_grid` in `observer._pending_workspace_grid` which can be overwritten before the Qt event loop processes it (search for `_notify_observers`).

**Fix**: Deep copy the data or use a queue.

---

## Optimizations Applied

### Vectorized Energy Calculations (41x improvement)

- **Fused operations**: Combined decay + noise into single tensor ops
- **Direct scatter**: Replaced `index_add_` with `scatter_add_` for direct fused operations
- **Combined masks**: Single `non_dynamic_mask` instead of separate sensory/workspace masks
- **Fused energy application**: `g.energy.add_(energy_changes).clamp_(min, max)`

### Byte-Level Compression (30-50% memory reduction)

- Node IDs: int32 -> int16 (50% reduction, supports up to 65,536 nodes)
- Workspace positions: int64 -> int16 (75% reduction)

### Mixed Precision

- Workspace energies stored in float16
- Calculations done in float32 where needed

---

## Optimization Roadmap

### Phase 1: Quick Wins (18-37ms gain per frame)
- [ ] Vectorize workspace seeding loop
- [ ] Eliminate blocking `.item()` calls
- [ ] Fuse clamp operations
- [ ] Add memory pools for temporary buffers

### Phase 2: Medium Effort (4-10ms + 10-20% overall)
- [ ] Implement CUDA streams for overlapping computation/transfer
- [ ] Optimize FFT with sparsity checks
- [ ] Batch workspace operations
- [ ] Async metric collection

### Phase 3: Advanced (2-5x potential)
- [ ] Full mixed precision (FP16) with autocast
- [ ] Sparse field representation for inactive regions
- [ ] Kernel fusion via `torch.jit.script`
- [ ] Adaptive time stepping

### Target Performance

| Phase | ms/frame | FPS |
|-------|----------|-----|
| Current | ~90ms | 11 |
| After Phase 1 | ~50-70ms | 14-20 |
| After Phase 2 | ~40-60ms | 17-25 |
| After Phase 3 | ~20-40ms | 25-50 |
| Goal | <16.67ms | 60 |

---

## Profiling

Use these tools to identify bottlenecks:
- `torch.profiler` for PyTorch operation profiling
- Nsight Compute for CUDA kernel analysis
- CUDA event timing for GPU execution measurement

### Hardware Reference

The development system is an MSI GF63 Thin 10SC:
- CPU: Intel i5-10500H @ 2.50GHz (6 cores / 12 threads)
- GPU: NVIDIA GeForce GTX 1650 Max-Q (4 GB)
- RAM: 32 GB
- Display: 1920x1080
