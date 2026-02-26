# Enhanced DNA System & 8 Connection Types

Date: 2026-02-26
Status: Approved

## Summary

Expand the neural simulation's genetic system from 4 connection types with static DNA probabilities to 8 connection types with DNA-DNA lock-and-key interactions, hereditary mutation, and behavioral micro-instructions.

## Bit Layout (64-bit node state)

```
Bit 63:      ALIVE (1 bit)
Bits 62-61:  NODE_TYPE (2 bits) - 0=sensory, 1=dynamic, 2=workspace
Bits 60-58:  CONN_TYPE (3 bits) - 0-7, eight connection types
Bits 57-18:  DNA (8 directions x 5 bits = 40 bits)
               Each 5-bit slot: [MODE:1][PARAM:4]
                 MODE=0: classic scale (param/15 = probability)
                 MODE=1: [SPECIAL:2][SPARAM:2]
                   00=THRESHOLD  01=INVERT  10=PULSE  11=ABSORB
Bits 17-0:   RESERVED (18 bits)
```

### Changes from current layout

- CONN_TYPE: 2 bits -> 3 bits (shift moves from 59 to 58)
- DNA_BASE_SHIFT: 19 -> 18 (all 8 direction shifts move down by 1)
- DNA slot interpretation: was pure 5-bit probability, now [MODE:1][PARAM:4]
- RESERVED: 19 bits -> 18 bits

## 8 Connection Types

| ID | Name         | Formula                                             | Description                          |
|----|--------------|-----------------------------------------------------|--------------------------------------|
| 0  | Excitatory   | `+delta * compat * 0.7 * dt`                        | Pulls energy toward self             |
| 1  | Inhibitory   | `-delta * compat * 0.7 * dt`                        | Pushes energy away                   |
| 2  | Gated        | `+delta * compat * 0.7 * dt * gate_fire`            | Only fires above energy threshold    |
| 3  | Plastic      | `+compat * 0.7 * dt`                                | Constant baseline flow (ignores delta) |
| 4  | Anti-Plastic | `-compat * 0.7 * dt`                                | Opposes baseline - energy drain      |
| 5  | Damped       | `+delta * compat * 0.7 * dt * exp(-|energy|/50)`    | Friction - weakens at high energy    |
| 6  | Resonant     | `+delta * sin(frame * compat * pi) * 0.7 * dt`      | DNA-frequency oscillation            |
| 7  | Capacitive   | Accumulate |delta|; burst-discharge at gate_threshold | Action potential analog              |

Where `compat` is the lock-and-key compatibility value (see below), replacing the old `dna_prob`.

### Weight table

The old 4x4 weight table is replaced by per-type branching in the kernel. Each type implements its own transfer formula directly. The 0.7 strength constant can be made configurable later.

## DNA-DNA Lock-and-Key Interactions

Transfer between two nodes now depends on BOTH nodes' DNA for the relevant direction.

```
For direction d from node i to neighbor at (ny, nx):
  my_dna       = extract_dna(state_i, d)
  neighbor_id  = _grid_node_id[ny, nx]
  their_dna    = extract_dna(state_neighbor, REVERSE_DIR[d])
  compat       = (my_param & their_param) / 15.0    # bitwise AND of 4-bit params
```

Reverse direction mapping (Moore neighborhood):
- 0 (up) <-> 1 (down)
- 2 (left) <-> 3 (right)
- 4 (up-left) <-> 7 (down-right)
- 5 (up-right) <-> 6 (down-left)

When no neighbor node exists at (ny, nx), `_grid_node_id` is -1 and compat defaults to 0.0 (no transfer to empty cells for node-to-node interactions; energy field transfer still works via the existing atomic operations).

## DNA Micro-Instructions

Each 5-bit DNA slot is interpreted as [MODE:1][PARAM:4]:

### MODE = 0 (Classic Scale)
- `probability = param / 15.0` (16 levels from 0.0 to 1.0)
- Used in lock-and-key: `compat = (my_param & their_param) / 15.0`
- Backward compatible with existing behavior (coarser resolution: 16 vs 32 levels)

### MODE = 1 (Special Behavior)
PARAM is split as [SPECIAL:2][SPARAM:2]:

| Special ID | Name      | Behavior                                                        |
|------------|-----------|-----------------------------------------------------------------|
| 0          | THRESHOLD | Transfer only fires when `|delta| > SPARAM * 16`. Otherwise 0. |
| 1          | INVERT    | Reverses gradient: pushes energy uphill. Strength = SPARAM/3.   |
| 2          | PULSE     | Fires only when `frame % (SPARAM+1) == 0`. Silent otherwise.   |
| 3          | ABSORB    | One-way pull only (positive delta). Ignores push. Strength = SPARAM/3. |

When a special-mode DNA interacts with a classic-mode neighbor via lock-and-key, the special behavior applies with the classic node's param as the compatibility base.

## DNA Mutation & Heredity

Children inherit parent DNA with rare bit flips (replaces fully random DNA generation).

```
For each of 8 DNA directions:
  child_dna[d] = parent_dna[d]
  if random() < 0.1:                  # 10% mutation chance per slot
    bit = random_int(0, 4)            # pick 1 of 5 bits
    child_dna[d] = child_dna[d] XOR (1 << bit)
```

- 90% of slots: exact copy from parent
- 10% of slots: one random bit flipped (can change mode, param, or special type)
- Connection type is still random for children (not inherited)
- Workspace nodes (type=2) retain fixed DNA = all max (unchanged)

### Emergent properties

- Lineages form clusters of nodes with similar DNA
- Lock-and-key ensures compatible lineages cooperate
- Incompatible lineages ignore each other (zero AND overlap)
- Mode bit mutations can create sudden behavioral shifts (classic -> special)
- Natural selection: nodes with energy-efficient DNA survive and reproduce

## New Taichi Fields

| Field            | Shape        | Type    | VRAM   | Purpose                              |
|------------------|-------------|---------|--------|--------------------------------------|
| `_grid_node_id`  | `(H, W)`   | `ti.i32`| ~1 MB  | Maps grid cell -> node index (-1=empty) |
| `_node_charge`   | `MAX_NODES` | `ti.f32`| 16 MB  | Capacitive type charge accumulator   |

Total additional VRAM: ~17 MB (on top of existing ~144 MB).

## Kernel Execution Order (per step)

1. **Clear grid map** - fill `_grid_node_id` with -1
2. **Rebuild grid map** - scatter: `_grid_node_id[pos_y[i], pos_x[i]] = i`
3. **DNA transfer** - enhanced kernel with lock-and-key, 8 conn types, micro-instructions
4. **Sync energies** - field -> node energy (unchanged)
5. **Death** - kill nodes below threshold (unchanged)
6. **Spawn** - hereditary DNA with mutation (replaces random DNA)
7. **Clamp** - energy field bounds (unchanged)

## Files to Modify

- `config.py` - New constants: 8 conn types, updated bit shifts, DNA mode constants
- `system/taichi_engine.py` - New fields, rewritten transfer kernel, updated spawn kernel, grid map kernel
- `main.py` - Updated workspace DNA override for new bit layout
- `workspace/realtime_visualization.py` - New connection type colors for heatmap (4 new types)

## Backward Compatibility

This is a breaking change to the binary state encoding. All existing nodes will be invalid after this change. The system initializes fresh each run (no persistence), so this has no migration impact.
