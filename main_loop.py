"""
main_loop.py

This module contains the main simulation loop logic that runs after initialization.
It coordinates energy updates, propagation, connection logic, birth/death, and other behaviors.
Designed for modularity and future extension.
"""

from energy_behavior import couple_sensory_energy_to_channel, propagate_sensory_energy
from connection_logic import add_dynamic_connections
from death_and_birth_logic import remove_dead_dynamic_nodes, birth_new_dynamic_nodes

# Add other imports as needed (e.g., time, logging)
import time


def run_main_loop(graph, steps=1000, step_delay=0.01):
    """
    Run the main simulation loop on the initialized graph.
    Args:
        graph: The initialized graph (with sensory and dynamic nodes).
        steps: Number of simulation steps to run (default: 1000).
        step_delay: Delay (in seconds) between steps (default: 0.01).
    Returns:
        The final graph after simulation.
    """
    for step in range(steps):
        # 1. Update/couple sensory node energies to their channel values
        graph = couple_sensory_energy_to_channel(graph)
        # 2. Propagate sensory energy along connections
        graph = propagate_sensory_energy(graph)
        # 3. Dynamic node connection logic (e.g., create new edges)
        graph = add_dynamic_connections(graph)
        # 4. Node birth logic (e.g., high-energy dynamic nodes spawn new nodes)
        graph = birth_new_dynamic_nodes(graph)
        # 5. Node death logic (e.g., low-energy dynamic nodes are removed)
        graph = remove_dead_dynamic_nodes(graph)
        # 6. (Optional) Other behaviors, logging, or hooks
        # print(f"Step {step}: {len(graph.node_labels)} nodes, {graph.edge_index.shape[1]} edges")
        time.sleep(step_delay)
    return graph

# AI/Human: Extend this file with additional simulation logic, hooks, or modular behaviors as needed. 