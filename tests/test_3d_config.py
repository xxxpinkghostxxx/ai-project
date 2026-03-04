# tests/test_3d_config.py
import json, os, pytest

def test_grid_size_is_3d():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'project', 'pyg_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg['hybrid']['grid_size'] == [512, 512, 8], "grid_size must be [512, 512, 8]"

def test_clusters_section_exists():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'project', 'pyg_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    cl = cfg.get('clusters', {})
    assert 'sensory_count' in cl
    assert 'workspace_count' in cl
    assert 'min_cluster_separation' in cl
