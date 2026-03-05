import pytest, torch

def test_inject_targets_correct_z_layer():
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
        data = torch.ones(32, 32, dtype=torch.float32, device=engine.device) * 42.0
        engine.inject_sensory_data(data, region=(0, 32, 0, 32), z=2)
        val_z2 = float(engine.energy_field[5, 5, 2])
        val_z0 = float(engine.energy_field[5, 5, 0])
        assert abs(val_z2 - 42.0) < 0.01, f"Expected 42.0 at z=2, got {val_z2}"
        assert abs(val_z0) < 0.01, f"Expected 0.0 at z=0, got {val_z0}"
    finally:
        TaichiNeuralEngine._instance = None
        _node_count[None] = 0
        del engine
