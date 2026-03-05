# tests/test_3d_add_nodes.py
import pytest, torch

def test_add_node_stores_z_position():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_pos_z, _node_dna, _node_count

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        ids = engine.add_nodes_batch(
            positions=[(10, 12, 3)],
            energies=[5.0],
            node_types=[1],
        )
        assert _node_pos_z[ids[0]] == 3, f"Expected z=3, got {_node_pos_z[ids[0]]}"
    finally:
        TaichiNeuralEngine._instance = None
        _node_count[None] = 0
        del engine

def test_add_node_stores_dna_in_separate_field():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine, _node_state, _node_dna, _node_count

    engine = TaichiNeuralEngine(grid_size=(32, 32, 4))
    try:
        ids = engine.add_nodes_batch(
            positions=[(5, 5, 1)],
            energies=[10.0],
            node_types=[1],
        )
        nid = ids[0]
        # _node_state should have zero in old DNA bit range (bits 57-18)
        state = int(_node_state[nid])
        dna_bits = (state >> 18) & ((1 << 40) - 1)
        assert dna_bits == 0, f"DNA bits should be 0 in _node_state, got {dna_bits}"
        # _node_dna should be nonzero (random DNA was written)
        dna_w0 = int(_node_dna[nid, 0])
        dna_w1 = int(_node_dna[nid, 1])
        assert (dna_w0 | dna_w1) != 0, "DNA should be nonzero"
    finally:
        TaichiNeuralEngine._instance = None
        _node_count[None] = 0
        del engine
