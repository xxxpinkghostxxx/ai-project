"""Smoke tests for TaichiGUIManager (no display required — just lifecycle)."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_manager_instantiates():
    """TaichiGUIManager should be constructable given an engine."""
    try:
        from project.system.taichi_engine import init_taichi
        init_taichi('cuda')
    except Exception:
        pass
    from project.system.taichi_engine import TaichiNeuralEngine
    from project.visualization.taichi_gui_manager import TaichiGUIManager

    engine = TaichiNeuralEngine(grid_size=(64, 64))
    try:
        mgr = TaichiGUIManager(engine)
        assert not mgr.is_open("workspace")
        assert not mgr.is_open("full_ai")
        assert not mgr.is_open("sensory")
    finally:
        TaichiNeuralEngine._instance = None
        del engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_display_fields_have_correct_shapes():
    """Module-level display fields must have the expected shapes."""
    from project.visualization.taichi_gui_manager import (
        _display_workspace, _display_full_ai, _display_sensory,
        _WORKSPACE_H, _WORKSPACE_W, _FULL_AI_H, _FULL_AI_W, _SENSORY_H, _SENSORY_W,
    )
    assert _display_workspace.shape == (_WORKSPACE_H, _WORKSPACE_W, 3)
    assert _display_full_ai.shape   == (_FULL_AI_H, _FULL_AI_W, 3)
    assert _display_sensory.shape   == (_SENSORY_H, _SENSORY_W, 3)
