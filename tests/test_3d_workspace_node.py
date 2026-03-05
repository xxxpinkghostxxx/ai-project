# tests/test_3d_workspace_node.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from project.workspace.workspace_node import WorkspaceNode


def test_workspace_node_stores_z():
    node = WorkspaceNode(node_id=0, grid_x=5, grid_y=10, grid_z=3)
    assert node.z == 3
    assert node.grid_position == (5, 10)


def test_workspace_node_z_defaults_to_zero():
    node = WorkspaceNode(node_id=1, grid_x=0, grid_y=0)
    assert node.z == 0
