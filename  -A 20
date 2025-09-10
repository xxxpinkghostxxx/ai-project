"""
main_loop.py

This module contains the main simulation loop logic that runs after initialization.
It coordinates energy updates, propagation, connection logic, birth/death, and other behaviors.
Designed for modularity and future extension.
"""

import numpy as np
import torch
from energy_behavior import couple_sensory_energy_to_channel, propagate_sensory_energy
from connection_logic import add_dynamic_connections
from death_and_birth_logic import remove_dead_dynamic_nodes, birth_new_dynamic_nodes
from logging_utils import log_runtime, log_step, log_node_state
import logging

# Add other imports as needed (e.g., time, logging)
import time
from main_graph import select_nodes_by_type

NODE_ENERGY_CAP = 244.0  # Clamp dynamic node energy to this value
NODE_ENERGY_DECAY = 0.05  # Decay constant per outgoing connection


def update_dynamic_node_energies(graph):
    """
    Update dynamic node energies: subtract decay, add incoming energy, clamp to [0, NODE_ENERGY_CAP].
    Vectorized for performance.
    """
    if (
        not hasattr(graph, "node_labels")
        or not hasattr(graph, "x")
        or not hasattr(graph, "edge_index")
    ):
        return graph
    x = graph.x
    edge_index = graph.edge_index
    node_labels = graph.node_labels
    num_nodes = len(node_labels)
    # Assertion: x and node_labels must match in length
    assert (
        x.shape[0] == num_nodes
    ), f"Mismatch: {x.shape[0]} features vs {num_nodes} labels"
    # Identify dynamic nodes
    dynamic_indices = [
        i for i, lbl in enumerate(node_labels) if lbl.get("type") == "dynamic"
    ]
    # Assertion: dynamic_indices must be within valid range
    assert all(
        0 <= idx < num_nodes for idx in dynamic_indices
    ), "Invalid dynamic node index"
    if not dynamic_indices:
        return graph
    # Out-degree for each node
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    if edge_index.numel() > 0:
        out_degree = np.bincount(
            edge_index[0].cpu().numpy(), minlength=num_nodes
        ).astype(np.float32)
    # Decay: subtract decay_constant * out_degree
    decay = NODE_ENERGY_DECAY * out_degree[dynamic_indices]
    # Incoming energy: sum over all incoming edges (from source to target)
    incoming_energy = np.zeros(len(dynamic_indices), dtype=np.float32)
    if edge_index.numel() > 0:
        src = edge_index[0].cpu().numpy()
        tgt = edge_index[1].cpu().numpy()
        src_energy = x[src].cpu().numpy().flatten()
        dynamic_idx_set = set(dynamic_indices)
        for i, t in enumerate(tgt):
            if t in dynamic_idx_set:
                idx = dynamic_indices.index(t)
                incoming_energy[idx] += src_energy[i]
    for idx in dynamic_indices[:3]:
        logging.info(f"[ENERGY] Before update: node {idx} energy={x[idx].item():.2f}")
    x_np = x.cpu().numpy()
    x_np[dynamic_indices, 0] = x_np[dynamic_indices, 0] - decay + incoming_energy
    x_np[dynamic_indices, 0] = np.clip(x_np[dynamic_indices, 0], 0, NODE_ENERGY_CAP)
    graph.x = torch.tensor(x_np, dtype=x.dtype)
    for idx in dynamic_indices[:3]:
        logging.info(
            f"[ENERGY] After update: node {idx} energy={graph.x[idx].item():.2f}"
        )
    # Assertion: all energies are within [0, NODE_ENERGY_CAP]
    assert np.all(
        (graph.x[dynamic_indices].cpu().numpy() >= 0)
        & (graph.x[dynamic_indices].cpu().numpy() <= NODE_ENERGY_CAP)
    ), "Dynamic node energy out of bounds"
    return graph


def update_dynamic_node_states(graph, step):
    """
    Update the 'state' and 'last_update' fields for dynamic nodes.
    - Set state to 'inactive' if energy is zero, else 'active'.
    - Update 'last_update' to the current step.
    - Apply behavior-specific state transitions for enhanced node types.
    Future: Add more sophisticated behavior/state logic.
    """
    from behavior_engine import should_transition_to_learning, energy_above_threshold, has_active_connections
    
    dynamic_indices = select_nodes_by_type(graph, 'dynamic')
    for idx in dynamic_indices:
        node = graph.node_labels[idx]
        behavior = node.get('behavior', 'dynamic')
        
        # Update last_update field
        node['last_update'] = step
        
        # Apply behavior-specific state transitions
        if behavior == 'oscillator':
            if should_transition_to_learning(node):
                node['state'] = 'learning'
        elif behavior == 'integrator':
            if energy_above_threshold(node):
                node['state'] = 'consolidating'
        elif behavior == 'relay':
            if has_active_connections(node, graph):
                node['state'] = 'active'
            else:
                node['state'] = 'pending'
        elif behavior == 'highway':
            # Highways can be in regulating state when they have high energy
            if node.get('energy', 0.0) > 0.8:
                node['state'] = 'regulating'
        else:
            # Default dynamic node behavior
            energy = node.get('energy', 0.0)
            if energy <= 0:
                node['state'] = 'inactive'
            else:
                node['state'] = 'active'
    
    return graph


@log_runtime
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
    # Import behavior engine for enhanced node behaviors
    from behavior_engine import BehaviorEngine
    
    # Initialize behavior engine
    behavior_engine = BehaviorEngine()
    
    logging.info(
        "[MAIN_LOOP] Starting main simulation loop",
        extra={"steps": steps, "step_delay": step_delay},
    )
    for step in range(steps):
        # Assertion: graph must have node_labels and x
        assert hasattr(graph, "node_labels") and hasattr(
            graph, "x"
        ), "Graph missing node_labels or x"
        log_step("Step start", step=step)
        logging.info(f"[MAIN_LOOP] Step {step} start", extra={"step": step})
        
        # NEW: Update node behaviors using behavior engine
        log_step("Update node behaviors", step=step)
        for idx, node in enumerate(graph.node_labels):
            if node.get('type') == 'dynamic':
                # Update behavior-specific properties
                updated_node = behavior_engine.update_node_behavior(node, graph, step)
                # Update the node label in place
                graph.node_labels[idx] = updated_node
        logging.info(f"[MAIN_LOOP] Updated node behaviors", extra={"step": step})
        
        log_step("Couple sensory energy to channel", step=step)
        graph = couple_sensory_energy_to_channel(graph)
        logging.info(
            f"[MAIN_LOOP] Coupled sensory energy to channel", extra={"step": step}
        )
        log_step("Propagate sensory energy", step=step)
        graph = propagate_sensory_energy(graph)
        logging.info(f"[MAIN_LOOP] Propagated sensory energy", extra={"step": step})
        log_step("Update dynamic node energies", step=step)
        graph = update_dynamic_node_energies(graph)
        logging.info(f"[MAIN_LOOP] Updated dynamic node energies", extra={"step": step})
        log_step("Update dynamic node states", step=step)
        graph = update_dynamic_node_states(graph, step)
        logging.info(f"[MAIN_LOOP] Updated dynamic node states", extra={"step": step})
        
        # Log state of first three dynamic nodes for debugging
        dynamic_indices = select_nodes_by_type(graph, 'dynamic')
        for idx in dynamic_indices[:3]:
            node_label = graph.node_labels[idx]
            # Assertion: required fields present
            for field in ["type", "energy", "behavior", "state", "last_update"]:
                assert field in node_label, f"Node label missing required field: {field}"
            log_node_state(node_label)
        
        log_step("Add dynamic connections", step=step)
        graph = add_dynamic_connections(graph)
        logging.info(
            f"[MAIN_LOOP] Added dynamic connections",
            extra={"step": step, "edge_count": graph.edge_index.shape[1]},
        )
        
        # NEW: Update connection weights based on activity patterns
        log_step("Update connection weights", step=step)
        from connection_logic import update_connection_weights
        graph = update_connection_weights(graph, learning_rate=0.01)
        logging.info(f"[MAIN_LOOP] Updated connection weights", extra={"step": step})
        
        log_step("Birth new dynamic nodes", step=step)
        graph = birth_new_dynamic_nodes(graph)
        logging.info(
            f"[MAIN_LOOP] Birth new dynamic nodes",
            extra={"step": step, "node_count": len(graph.node_labels)},
        )
        log_step("Remove dead dynamic nodes", step=step)
        graph = remove_dead_dynamic_nodes(graph)
        logging.info(
            f"[MAIN_LOOP] Removed dead dynamic nodes",
            extra={"step": step, "node_count": len(graph.node_labels)},
        )
        
        # NEW: Log behavior statistics every 100 steps
        if step % 100 == 0:
            behavior_stats = behavior_engine.get_behavior_statistics()
            logging.info(f"[BEHAVIOR_STATS] Step {step}: {behavior_stats}")
        
        # NEW: Apply learning and plasticity updates
        log_step("Apply learning updates", step=step)
        from learning_engine import LearningEngine
        from memory_system import MemorySystem
        
        # Initialize learning and memory systems if not already done
        if not hasattr(graph, 'learning_engine'):
            graph.learning_engine = LearningEngine()
        if not hasattr(graph, 'memory_system'):
            graph.memory_system = MemorySystem()
        
        # Apply learning updates
        graph = graph.learning_engine.consolidate_connections(graph)
        logging.info(f"[MAIN_LOOP] Applied learning updates", extra={"step": step})
        
        # NEW: Form memory traces every 50 steps
        if step % 50 == 0:
            log_step("Form memory traces", step=step)
            graph = graph.memory_system.form_memory_traces(graph)
            graph = graph.memory_system.consolidate_memories(graph)
            graph = graph.memory_system.decay_memories()
            logging.info(f"[MAIN_LOOP] Memory traces processed", extra={"step": step})
        
        # NEW: Apply memory influence to connections
        if step % 25 == 0:  # Every 25 steps
            log_step("Apply memory influence", step=step)
            graph = graph.learning_engine.apply_memory_influence(graph)
            logging.info(f"[MAIN_LOOP] Memory influence applied", extra={"step": step})
        
        # NEW: Apply enhanced energy behavior dynamics
        log_step("Apply enhanced energy behavior", step=step)
        from energy_behavior import apply_energy_behavior, update_membrane_potentials, apply_refractory_periods
        graph = apply_energy_behavior(graph)
        graph = update_membrane_potentials(graph)
        graph = apply_refractory_periods(graph)
        logging.info(f"[MAIN_LOOP] Enhanced energy behavior applied", extra={"step": step})
        
        # NEW: Apply homeostatic regulation every 100 steps
        if step % 100 == 0:
            log_step("Apply homeostatic regulation", step=step)
            from homeostasis_controller import HomeostasisController
            
            # Initialize homeostasis controller if not already done
            if not hasattr(graph, 'homeostasis_controller'):
                graph.homeostasis_controller = HomeostasisController()
            
            # Regulate network activity and optimize criticality
            graph = graph.homeostasis_controller.regulate_network_activity(graph)
            graph = graph.homeostasis_controller.optimize_criticality(graph)
            
            # Monitor network health
            health_status = graph.homeostasis_controller.monitor_network_health(graph)
            if health_status['status'] != 'healthy':
                logging.warning(f"[HOMEOSTASIS] Network health: {health_status['status']} - {health_status['warnings']}")
            
            logging.info(f"[MAIN_LOOP] Homeostatic regulation applied", extra={"step": step})
        
        log_step(
            "Step end",
            step=step,
            node_count=len(graph.node_labels),
            edge_count=graph.edge_index.shape[1],
        )
        logging.info(
            f"[MAIN_LOOP] Step {step} end",
            extra={
                "step": step,
                "node_count": len(graph.node_labels),
                "edge_count": graph.edge_index.shape[1],
            },
        )
        # Assertion: graph.x and node_labels must remain consistent
        assert graph.x.shape[0] == len(
            graph.node_labels
        ), f"After step {step}: {graph.x.shape[0]} features vs {len(graph.node_labels)} labels"
        time.sleep(step_delay)
    
    # Log final behavior statistics
    final_behavior_stats = behavior_engine.get_behavior_statistics()
    logging.info(f"[MAIN_LOOP] Final behavior statistics: {final_behavior_stats}")
    
    logging.info("[MAIN_LOOP] Main simulation loop complete", extra={"steps": steps})
    return graph


# AI/Human: Extend this file with additional simulation logic, hooks, or modular behaviors as needed.
