# tests/test_cluster_injection.py
import pytest, torch


def test_per_cluster_injection_updates_specific_z():
    """Injecting into cluster centers should update energy at the cluster's Z layer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        data = torch.ones(4, 4, dtype=torch.float32, device=engine.device) * 99.0
        engine.inject_sensory_data(data, region=(14, 18, 14, 18), z=2)
        val = float(engine.energy_field[15, 15, 2])
        assert val > 0.0, f"Expected energy at cluster Z=2, got {val}"
        assert float(engine.energy_field[15, 15, 0]) == 0.0
    finally:
        TaichiNeuralEngine._instance = None
        del engine
