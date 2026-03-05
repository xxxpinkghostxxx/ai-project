import pytest, torch

def test_energy_transfers_across_z_layers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_count

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4), transfer_dt=0.1)
    try:
        ids = engine.add_nodes_batch(
            positions=[(10, 10, 0), (10, 10, 1)],
            energies=[100.0, 0.0],
            node_types=[1, 2],
        )
        engine.step()
        e_recv = float(engine.energy_field[10, 10, 1])
        assert e_recv != 0.0, f"Energy at z=1 should change after transfer, got {e_recv}"
    finally:
        TaichiNeuralEngine._instance = None
        _node_count[None] = 0
        del engine
