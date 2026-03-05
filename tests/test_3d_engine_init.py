# tests/test_3d_engine_init.py
import pytest, torch

def test_engine_creates_3d_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_count
    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        assert engine.energy_field.shape == (32, 32, 4)
        assert engine.grid_node_id.shape == (32, 32, 4)
        assert engine.D == 4
    finally:
        _node_count[None] = 0
        TaichiNeuralEngine._instance = None
        del engine

def test_engine_3d_has_node_pos_z():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_pos_z, MAX_NODES, _node_count
    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        assert _node_pos_z.shape[0] == MAX_NODES
    finally:
        _node_count[None] = 0
        TaichiNeuralEngine._instance = None
        del engine
