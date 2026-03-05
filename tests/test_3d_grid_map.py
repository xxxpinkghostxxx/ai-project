# tests/test_3d_grid_map.py
import pytest, torch

def test_grid_map_places_node_at_correct_z():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, MAX_NODES, _node_count

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        ids = engine.add_nodes_batch(
            positions=[(10, 12, 3)],
            energies=[5.0],
            node_types=[1],
        )
        from project.system.taichi_engine import _clear_grid_map, _build_grid_map
        _clear_grid_map(engine.grid_node_id)
        _build_grid_map(engine.grid_node_id)

        val = int(engine.grid_node_id[10, 12, 3])
        assert val == ids[0], f"Expected node {ids[0]} at [10,12,3], got {val}"
        for z in range(4):
            if z != 3:
                val_other = int(engine.grid_node_id[10, 12, z])
                assert val_other == MAX_NODES, f"Expected empty at z={z}, got {val_other}"
    finally:
        TaichiNeuralEngine._instance = None
        _node_count[None] = 0
        del engine
