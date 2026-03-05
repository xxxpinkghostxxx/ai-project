# tests/test_projection.py
import torch
from project.system.taichi_engine import project_energy_field_to_2d


def test_max_pool_projection():
    field = torch.zeros(8, 8, 4)
    field[3, 3, 2] = 99.0
    proj = project_energy_field_to_2d(field)
    assert proj.shape == (8, 8)
    assert float(proj[3, 3]) == 99.0
    assert float(proj[0, 0]) == 0.0


def test_2d_passthrough():
    field = torch.ones(8, 8)
    proj = project_energy_field_to_2d(field)
    assert proj.shape == (8, 8)
