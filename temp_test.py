import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
print("Sys path set")

import unittest
print("unittest imported")
import numpy as np
print("numpy imported")
from unittest.mock import Mock, patch
print("mock imported")
from torch_geometric.data import Data
print("PyG imported")
import torch
print("torch imported")

print("Starting energy imports")
try:
    from energy.energy_constants import EnergyConstants, ConnectionConstants
    print("energy_constants imported")
except Exception as e:
    print(f"Failed to import energy_constants: {e}")
    exit(1)

try:
    from energy.energy_behavior import (
        EnergyCalculator, get_node_energy_cap, update_node_energy_with_learning,
        apply_energy_behavior, apply_oscillator_energy_dynamics,
        apply_integrator_energy_dynamics, apply_relay_energy_dynamics,
        apply_highway_energy_dynamics, apply_dynamic_energy_dynamics, emit_energy_pulse
    )
    print("energy_behavior imported")
except Exception as e:
    print(f"Failed to import energy_behavior: {e}")
    exit(1)

try:
    from energy.node_id_manager import NodeIDManager, get_id_manager
    print("node_id_manager imported")
except Exception as e:
    print(f"Failed to import node_id_manager: {e}")
    exit(1)

try:
    from energy.node_access_layer import NodeAccessLayer
    print("node_access_layer imported")
except Exception as e:
    print(f"Failed to import node_access_layer: {e}")
    exit(1)

try:
    from energy.energy_flow_diagram import EnergyFlowDiagram
    print("energy_flow_diagram imported")
except Exception as e:
    print(f"Failed to import energy_flow_diagram: {e}")
    exit(1)

try:
    from energy.energy_system_validator import EnergySystemValidator
    print("energy_system_validator imported")
except Exception as e:
    print(f"Failed to import energy_system_validator: {e}")
    exit(1)

print("All imports successful")

# Create a comprehensive test graph
graph = Data()
graph.node_labels = []
graph.x = torch.empty((0, 1), dtype=torch.float32)
graph.edge_index = torch.empty((2, 0), dtype=torch.long)

print("Graph created")

# Create diverse node types
node_specs = [
    # Sensory input nodes
    {"type": "sensory", "behavior": "sensory", "energy": 0.2, "threshold": 0.5},
    {"type": "sensory", "behavior": "sensory", "energy": 0.3, "threshold": 0.5},

    # Dynamic processing nodes
    {"type": "dynamic", "behavior": "oscillator", "energy": 1.0, "threshold": 0.5,
     "oscillation_freq": 1.0, "last_activation": 0, "refractory_timer": 0},
    {"type": "dynamic", "behavior": "integrator", "energy": 0.8, "threshold": 0.5,
     "integration_rate": 0.5},
    {"type": "dynamic", "behavior": "relay", "energy": 0.9, "threshold": 0.5,
     "relay_amplification": 1.5},
    {"type": "dynamic", "behavior": "highway", "energy": 0.4, "threshold": 0.5,
     "highway_energy_boost": 2.0},

    # Output nodes
    {"type": "dynamic", "behavior": "dynamic", "energy": 0.6, "threshold": 0.5}
]

print("Node specs created")

# Register nodes and build graph
id_manager = get_id_manager()
id_manager.reset()

print("ID manager reset")

for i, spec in enumerate(node_specs):
    node_id = id_manager.generate_unique_id(spec["type"])
    spec["id"] = node_id
    id_manager.register_node_index(node_id, i)

    graph.node_labels.append(spec)
    graph.x = torch.cat([graph.x, torch.tensor([[spec["energy"]]], dtype=torch.float32)], dim=0)

print("Nodes registered")

# Create connections
connections = [
    (0, 2), (1, 2),  # Sensory to oscillator
    (2, 3), (2, 4),  # Oscillator to integrator/relay
    (3, 5), (4, 5),  # Integrator/relay to highway
    (5, 6)           # Highway to output
]

for source, target in connections:
    graph.edge_index = torch.cat([
        graph.edge_index,
        torch.tensor([[source], [target]], dtype=torch.long)
    ], dim=1)

print("Connections created")

# Create access layer
access_layer = NodeAccessLayer(graph, id_manager)

print("Access layer created")

print("Setup complete")

# Test constants
print("Testing constants")

calculator_cap = EnergyCalculator.calculate_energy_cap()
function_cap = get_node_energy_cap()

print(f"Calculator cap: {calculator_cap}")
print(f"Function cap: {function_cap}")

assert calculator_cap == function_cap

print("Constants test passed")