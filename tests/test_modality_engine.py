"""Tests for modality bit packing and spawn inheritance."""
import pytest
import torch


def test_pack_state_includes_modality():
    """add_nodes_batch must embed modality bits into the packed state."""
    # Skip if no CUDA — engine requires it
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import taichi as ti
    # Guard: don't re-init Taichi if already done in another test
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass

    from project.system.taichi_engine import TaichiNeuralEngine, _node_state, MAX_NODES
    from project.config import MODALITY_VISUAL, MODALITY_AUDIO_LEFT, MODALITY_SHIFT, MODALITY_MASK

    engine = TaichiNeuralEngine(grid_size=(64, 64))
    try:
        # Add 2 workspace nodes: one VISUAL, one AUDIO_LEFT
        ids = engine.add_nodes_batch(
            positions=[(60, 10), (60, 20)],
            energies=[10.0, 10.0],
            node_types=[2, 2],
            modalities=[MODALITY_VISUAL, MODALITY_AUDIO_LEFT],
        )
        # Read back packed states
        state0 = int(_node_state[ids[0]])
        state1 = int(_node_state[ids[1]])
        mod0 = (state0 >> MODALITY_SHIFT) & MODALITY_MASK
        mod1 = (state1 >> MODALITY_SHIFT) & MODALITY_MASK
        assert mod0 == MODALITY_VISUAL
        assert mod1 == MODALITY_AUDIO_LEFT
    finally:
        TaichiNeuralEngine._instance = None
        del engine


def test_modality_neutral_when_omitted():
    """Nodes added without modality arg should have MODALITY_NEUTRAL (0)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass

    from project.system.taichi_engine import TaichiNeuralEngine, _node_state
    from project.config import MODALITY_NEUTRAL, MODALITY_SHIFT, MODALITY_MASK

    engine = TaichiNeuralEngine(grid_size=(64, 64))
    try:
        ids = engine.add_nodes_batch(
            positions=[(30, 30)],
            energies=[50.0],
            node_types=[1],
        )
        state = int(_node_state[ids[0]])
        mod = (state >> MODALITY_SHIFT) & MODALITY_MASK
        assert mod == MODALITY_NEUTRAL
    finally:
        del engine
