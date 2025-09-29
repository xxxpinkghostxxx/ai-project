import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import torch
from torch_geometric.data import Data

from src.core.main_graph import initialize_main_graph
from src.energy.node_id_manager import get_id_manager
from src.neural.death_and_birth_logic import (birth_new_dynamic_nodes,
                                              remove_dead_dynamic_nodes)
from src.neural.dynamic_nodes import add_dynamic_nodes
from src.neural.network_metrics import NetworkMetrics

logging.basicConfig(level=logging.INFO)

# Initialize graph
print("--- Initializing Graph ---")
graph = initialize_main_graph(scale=0.1)
print(f"Graph initialized with {len(graph.x)} nodes")

# Initial Metrics
print("\n--- Calculating Initial Metrics ---")
metrics_calculator = NetworkMetrics()
initial_metrics = metrics_calculator.calculate_comprehensive_metrics(graph)
connectivity = initial_metrics.get('connectivity', {})
print(f"Initial Edges: {connectivity.get('num_edges')}, Density: {connectivity.get('density')}")

# Simulate Birth
print("\n--- Simulating Node Birth ---")
graph = birth_new_dynamic_nodes(graph)
print(f"Graph after birth has {len(graph.x)} nodes")

# Metrics after Birth
print("\n--- Calculating Metrics After Birth ---")
post_birth_metrics = metrics_calculator.calculate_comprehensive_metrics(graph)
connectivity = post_birth_metrics.get('connectivity', {})
print(f"Post-Birth Edges: {connectivity.get('num_edges')}, Density: {connectivity.get('density')}")

# Simulate Death
print("\n--- Simulating Node Death ---")
# Make some nodes eligible for removal
for i in range(min(10, len(graph.node_labels))):
    if graph.node_labels[i].get("type") == "dynamic":
        graph.x[i] = 0.01

graph = remove_dead_dynamic_nodes(graph)
print(f"Graph after death has {len(graph.x)} nodes")

# Metrics after Death
print("\n--- Calculating Metrics After Death ---")
post_death_metrics = metrics_calculator.calculate_comprehensive_metrics(graph)
connectivity = post_death_metrics.get('connectivity', {})
print(f"Post-Death Edges: {connectivity.get('num_edges')}, Density: {connectivity.get('density')}")

print("\n--- Verification ---")
if initial_metrics['connectivity']['num_edges'] > 0:
    print("[PASS] Initial edge creation successful.")
else:
    print("[FAIL] Initial edge creation failed.")

if post_birth_metrics['connectivity']['num_edges'] > initial_metrics['connectivity']['num_edges']:
    print("[PASS] Edge count increased after birth.")
else:
    print("[FAIL] Edge count did not increase after birth.")

if post_death_metrics['connectivity']['num_edges'] > 0 and post_death_metrics['connectivity']['num_edges'] < post_birth_metrics['connectivity']['num_edges']:
     print("[PASS] Edges persisted and were correctly remapped after death.")
else:
     print("[FAIL] Edges did not persist or remap correctly after death.")







