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


def test_3d_geometry_invariants():
    from project.config import NEIGHBOR_OFFSETS_3D, REVERSE_DIRECTION_3D, DNA_SLOT_BIT

    # 1. 26-neighbor Moore neighbourhood
    assert len(NEIGHBOR_OFFSETS_3D) == 26, (
        f"Expected 26 neighbours, got {len(NEIGHBOR_OFFSETS_3D)}"
    )

    # 2. REVERSE_DIRECTION_3D is an involution: reverse(reverse(i)) == i for all i
    for i in range(len(REVERSE_DIRECTION_3D)):
        assert REVERSE_DIRECTION_3D[REVERSE_DIRECTION_3D[i]] == i, (
            f"REVERSE_DIRECTION_3D involution failed at index {i}"
        )

    # 3. DNA bits fit inside int64 words (max bit offset + 5 bits <= 60 bits per word)
    assert max(DNA_SLOT_BIT) + 5 <= 60, (
        f"DNA bits overflow word boundary: max bit offset {max(DNA_SLOT_BIT)} + 5 > 60"
    )
