# tests/test_3d_end_to_end.py
import pytest, torch


def test_3d_simulation_runs_10_steps():
    """Full simulation: clusters initialized, 10 steps run, no errors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_count
    from project.config import NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        # Seed workspace clusters
        ws_positions = [(28, 5, 1), (28, 15, 2), (28, 25, 3)]
        engine.add_nodes_batch(
            positions=ws_positions,
            energies=[50.0] * 3,
            node_types=[NODE_TYPE_WORKSPACE] * 3,
        )
        # Seed dynamic nodes
        dyn_positions = [(10, x, z) for z in range(4) for x in range(0, 32, 4)]
        engine.add_nodes_batch(
            positions=dyn_positions,
            energies=[30.0] * len(dyn_positions),
            node_types=[NODE_TYPE_DYNAMIC] * len(dyn_positions),
        )
        # Inject sensory data at z=0
        data = torch.rand(32, 32, device=engine.device) * 20
        engine.inject_sensory_data(data, region=(0, 32, 0, 32), z=0)

        for step_i in range(10):
            result = engine.step()
            assert isinstance(result, dict)
            assert result['num_nodes'] > 0
            # Energy field should be finite (no NaN/Inf)
            assert torch.isfinite(engine.energy_field).all(), \
                f"NaN/Inf in energy_field at step {step_i}"

        print(f"Final node count: {engine._count}")
    finally:
        TaichiNeuralEngine._instance = None
        _node_count[None] = 0
        del engine
