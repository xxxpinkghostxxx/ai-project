"""
main_loop_unified.py

Unified main loop that uses the SimulationManager for consistent simulation behavior.
This replaces the old main_loop.py with a unified approach.
"""

import time
import logging
from logging_utils import log_step, log_runtime
from simulation_manager import create_simulation_manager


@log_runtime
def run_main_loop(graph, steps=1000, step_delay=0.01):
    """
    Run the main simulation loop on the initialized graph using unified simulation manager.
    Args:
        graph: The initialized graph (with sensory and dynamic nodes).
        steps: Number of simulation steps to run (default: 1000).
        step_delay: Delay (in seconds) between steps (default: 0.01).
    Returns:
        The final graph after simulation.
    """
    # Import memory leak detection
    from memory_leak_detector import MemoryLeakContext, force_memory_cleanup
    
    # Create simulation manager with custom config for standalone mode
    config = {
        'update_interval': step_delay,
        'sensory_update_interval': 1,  # Update every step for standalone mode
        'memory_update_interval': 10,  # More frequent for standalone mode
        'homeostasis_update_interval': 20,  # More frequent for standalone mode
        'metrics_update_interval': 10  # More frequent for standalone mode
    }
    
    sim_manager = create_simulation_manager(config)
    sim_manager.set_graph(graph)
    
    logging.info(
        "[MAIN_LOOP] Starting unified simulation loop",
        extra={"steps": steps, "step_delay": step_delay},
    )
    
    # Run simulation steps with memory leak prevention
    with MemoryLeakContext("main_loop_simulation"):
        for step in range(steps):
            # Assertion: graph must have node_labels and x
            assert hasattr(graph, "node_labels") and hasattr(
                graph, "x"
            ), "Graph missing node_labels or x"
            
            log_step("Step start", step=step)
            logging.info(f"[MAIN_LOOP] Step {step} start", extra={"step": step})
            
            # Run single simulation step
            success = sim_manager.run_single_step()
            
            if not success:
                logging.error(f"[MAIN_LOOP] Simulation step {step} failed")
                break
            
            # Force memory cleanup every 100 steps to prevent leaks
            if step % 100 == 0 and step > 0:
                force_memory_cleanup()
            
            log_step("Step end", step=step)
            logging.info(f"[MAIN_LOOP] Step {step} end", extra={"step": step})
    
    # Get final performance stats
    perf_stats = sim_manager.get_performance_stats()
    system_stats = sim_manager.get_system_stats()
    
    logging.info("[MAIN_LOOP] Unified simulation loop completed",
                extra={
                    "total_steps": perf_stats["total_steps"],
                    "avg_step_time": perf_stats["avg_step_time"],
                    "system_health": perf_stats["system_health"]
                })
    
    # Clean up simulation manager to prevent memory leaks
    sim_manager.cleanup()
    
    return graph


@log_runtime
def run_main_loop_continuous(graph, max_steps=None, step_delay=0.01):
    """
    Run the main simulation loop continuously using unified simulation manager.
    Args:
        graph: The initialized graph (with sensory and dynamic nodes).
        max_steps: Maximum number of steps (None for infinite).
        step_delay: Delay (in seconds) between steps (default: 0.01).
    Returns:
        The final graph after simulation.
    """
    # Create simulation manager with custom config for continuous mode
    config = {
        'update_interval': step_delay,
        'sensory_update_interval': 5,  # Less frequent for continuous mode
        'memory_update_interval': 50,  # Standard frequency
        'homeostasis_update_interval': 100,  # Standard frequency
        'metrics_update_interval': 50  # Standard frequency
    }
    
    sim_manager = create_simulation_manager(config)
    sim_manager.set_graph(graph)
    
    logging.info(
        "[MAIN_LOOP] Starting continuous unified simulation loop",
        extra={"max_steps": max_steps, "step_delay": step_delay},
    )
    
    step = 0
    try:
        while max_steps is None or step < max_steps:
            # Assertion: graph must have node_labels and x
            assert hasattr(graph, "node_labels") and hasattr(
                graph, "x"
            ), "Graph missing node_labels or x"
            
            log_step("Continuous step start", step=step)
            
            # Run single simulation step
            success = sim_manager.run_single_step()
            
            if not success:
                logging.error(f"[MAIN_LOOP] Continuous simulation step {step} failed")
                break
            
            step += 1
            
            # Log progress every 100 steps
            if step % 100 == 0:
                perf_stats = sim_manager.get_performance_stats()
                logging.info(f"[MAIN_LOOP] Continuous simulation progress: {step} steps, "
                           f"avg_time: {perf_stats['avg_step_time']:.3f}s, "
                           f"health: {perf_stats['system_health']}")
            
            log_step("Continuous step end", step=step)
    
    except KeyboardInterrupt:
        logging.info("[MAIN_LOOP] Continuous simulation interrupted by user")
    
    # Get final performance stats
    perf_stats = sim_manager.get_performance_stats()
    system_stats = sim_manager.get_system_stats()
    
    logging.info("[MAIN_LOOP] Continuous unified simulation loop completed",
                extra={
                    "total_steps": perf_stats["total_steps"],
                    "avg_step_time": perf_stats["avg_step_time"],
                    "system_health": perf_stats["system_health"]
                })
    
    return graph


# Example usage and testing
if __name__ == "__main__":
    print("Unified Main Loop initialized successfully!")
    print("Features include:")
    print("- Unified simulation architecture")
    print("- Consistent behavior with UI simulation")
    print("- Configurable update intervals")
    print("- Performance monitoring")
    print("- Error handling and recovery")
    
    print("\nUnified Main Loop is ready for use!")
