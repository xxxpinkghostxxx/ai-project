# tests/test_cluster_init.py
import pytest, torch


def test_clusters_produce_nodes_at_scattered_positions():
    """Sensory and workspace nodes should not all be at Z=0."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from project.main import _place_clusters

    centers, positions = _place_clusters(
        grid_H=32, grid_W=32, grid_D=4,
        count=4, nodes_each=5, radius=2, min_separation=6,
    )
    # Should have 4 centers and ~20 positions
    assert len(centers) > 0
    assert len(positions) > 0
    # All positions in bounds
    for y, x, z in positions:
        assert 0 <= y < 32
        assert 0 <= x < 32
        assert 0 <= z < 4
    # Not all at Z=0 (with 4 clusters and D=4, at least one cluster will be at z>0)
    z_values = {z for _, _, z in positions}
    assert len(z_values) > 1, f"Expected multiple Z values, got {z_values}"
