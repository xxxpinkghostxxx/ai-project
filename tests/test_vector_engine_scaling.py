import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from project.system.vector_engine import (  # type: ignore[import-not-found]
    DensityManager,
    NodeArrayStore,
    NodeClassSpec,
    NodeRuleRegistry,
    SparseNodeStore,
    VectorizedSimulationEngine,
    run_micro_benchmark,
)


def test_spawn_and_rule_application():
    store = NodeArrayStore(capacity=128, device="cpu")
    registry = NodeRuleRegistry()

    def add_energy(node_store: NodeArrayStore, mask: torch.Tensor, dt: float) -> None:
        node_store.energy[mask] += 1.0 * dt

    registry.register(NodeClassSpec(class_id=1, name="dynamic", update_fn=add_energy, decay=0.0, max_energy=256.0))
    density = DensityManager(world_size=(32, 32), tile_size=8)
    engine = VectorizedSimulationEngine(store, registry, density, sparse_store=None)

    classes = torch.ones(10, dtype=torch.int64)
    positions = torch.randint(0, 16, (10, 2))
    energies = torch.zeros(10)
    store.spawn(10, classes, energies=energies, positions=positions)

    metrics = engine.step(dt=1.0)
    assert metrics["active"] == 10
    assert torch.allclose(store.energy[:10], torch.ones(10))


def test_density_manager_cap_enforced():
    density = DensityManager(world_size=(16, 16), tile_size=4, per_class_caps={1: 1})
    positions = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.int32)
    assert density.can_place(positions[:1], class_id=1)
    density.register(positions[:1], class_id=1)
    # Additional nodes in same tile should be blocked
    assert density.can_place(positions[1:2], class_id=1) is False


def test_sparse_store_roundtrip():
    store = NodeArrayStore(capacity=16, device="cpu")
    classes = torch.ones(5, dtype=torch.int64)
    positions = torch.arange(10, 20, dtype=torch.int32).unsqueeze(1).repeat(1, 2)
    energies = torch.arange(5, dtype=torch.float32) + 1.0
    indices = store.spawn(5, classes, energies=energies, positions=positions[:5])

    sparse = SparseNodeStore(capacity=8, directory=".cache/test_vector_engine")
    tokens = sparse.spill(store, indices[:3])
    store.deactivate(indices[:3])

    # Hydrate into free slots
    free_idx = torch.nonzero(~store.active_mask, as_tuple=False).view(-1)
    sparse.materialize(store, tokens, free_idx[: len(tokens)])

    assert store.active_mask.sum().item() == 5
    assert torch.allclose(store.energy[:5].sort().values, energies.sort().values)


def test_connection_batch_updates_energy():
    store = NodeArrayStore(capacity=8, device="cpu")
    registry = NodeRuleRegistry()
    density = DensityManager(world_size=(8, 8), tile_size=4)
    engine = VectorizedSimulationEngine(store, registry, density, sparse_store=None)

    classes = torch.ones(4, dtype=torch.int64)
    positions = torch.zeros((4, 2), dtype=torch.int32)
    energies = torch.tensor([10.0, 10.0, 10.0, 10.0])
    store.spawn(4, classes, energies=energies, positions=positions)

    src = torch.tensor([0, 1], dtype=torch.int64)
    dst = torch.tensor([2, 3], dtype=torch.int64)
    weights = torch.tensor([0.5, 1.0])
    metrics = engine.apply_connection_batch(src, dst, weights, loss=1.0)

    assert metrics["transfers"] == 2.0
    assert store.energy[2] > 10.0
    assert store.energy[0] < 10.0


def test_micro_benchmark_runs():
    results = run_micro_benchmark(device="cpu", n=10_000)
    assert results["active"] == 10_000
    assert "spawn_time_ms" in results and "step_time_ms" in results
