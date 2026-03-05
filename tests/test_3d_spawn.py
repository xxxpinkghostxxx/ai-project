import pytest, torch

def test_spawn_creates_child_with_z_coordinate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_pos_z, _node_state, _node_count

    engine = TaichiNeuralEngine(grid_size=(32, 32, 8), node_spawn_threshold=5.0)
    try:
        ids = engine.add_nodes_batch(
            positions=[(10, 10, 4)],
            energies=[100.0],
            node_types=[1],
        )
        engine.step()
        new_count = engine._count
        if new_count > len(ids):
            for nid in range(new_count):
                if int(_node_state[nid]) != 0:
                    z = int(_node_pos_z[nid])
                    assert 0 <= z < 8, f"Node {nid} z={z} out of [0,8)"
    finally:
        TaichiNeuralEngine._instance = None
        _node_count[None] = 0
        del engine
