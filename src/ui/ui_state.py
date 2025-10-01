"""
UI state management functions.

This module contains functions for managing the UI state, including
simulation state, graph data, live feed data, and UI display updates.
"""

import sys
import time

from src.ui.ui_constants import ui_state, CONSTANTS, _service_registry, _visualization_service, dpg
from src.core.interfaces.simulation_coordinator import ISimulationCoordinator
from src.core.interfaces.real_time_visualization import VisualizationData
from src.utils.logging_utils import log_step


def get_simulation_running():
    """Get the current simulation running state.

    Returns:
        bool: True if simulation is running, False otherwise.
    """
    return ui_state.get_simulation_state()['simulation_running']


def set_simulation_running(running: bool):
    """Set the simulation running state.

    Args:
        running (bool): True to start simulation, False to stop it.
    """
    ui_state.set_simulation_running(running)


def get_latest_graph():
    """Get the latest neural graph from UI state.

    Returns:
        The latest neural graph object.
    """
    return ui_state.get_latest_graph()


def get_latest_graph_for_ui():
    """Get the latest neural graph formatted for UI display.

    Returns:
        The latest neural graph object formatted for UI rendering.
    """
    return ui_state.get_latest_graph_for_ui()


def update_graph(graph):
    """Update the neural graph in UI state.

    Args:
        graph: The neural graph object to update in UI state.
    """
    ui_state.update_graph(graph)


def add_live_feed_data(data_type: str, value: float):
    """Add data point to live feed for real-time monitoring.

    Args:
        data_type (str): Type of data being added (e.g., 'energy', 'activity').
        value (float): Numeric value of the data point.
    """
    ui_state.add_live_feed_data(data_type, value)


def get_live_feed_data():
    """Get current live feed data for display.

    Returns:
        dict: Dictionary containing live feed data points organized by type.
    """
    return ui_state.get_live_feed_data()


def clear_live_feed_data():
    """Clear all live feed data from UI state."""
    ui_state.clear_live_feed_data()


def update_operation_status(operation: str, progress: float = 0.0):
    """Update operation status and progress indicator."""
    # Skip DPG calls in test environments to avoid crashes
    skip = 'pytest' in sys.modules
    if skip:
        return

    try:
        dpg.set_value("operation_status_text", f"Operation: {operation}")
        dpg.set_value("operation_progress", progress)
    except (RuntimeError, AttributeError, ValueError) as e:
        log_step(f"Failed to update operation status: {e}")


def clear_operation_status():
    """Clear operation status and reset progress."""
    # Skip DPG calls in test environments to avoid crashes
    skip = 'pytest' in sys.modules
    if skip:
        return

    try:
        dpg.set_value("operation_status_text", "Operation: Idle")
        dpg.set_value("operation_progress", 0.0)
    except (RuntimeError, AttributeError, ValueError) as e:
        log_step(f"Failed to clear operation status: {e}")


def get_coordinator() -> ISimulationCoordinator:
    """Get the simulation coordinator from the service registry.

    Returns:
        Optional[ISimulationCoordinator]: The simulation coordinator instance if available,
            None if service registry is not initialized or coordinator not found.
    """
    if _service_registry:
        return _service_registry.resolve(ISimulationCoordinator)
    return None


def _update_simulation_status(is_running):
    """Update the simulation status display."""
    # pylint: disable=no-member
    status = CONSTANTS['SIMULATION_STATUS_RUNNING'] if is_running else CONSTANTS['SIMULATION_STATUS_STOPPED']
    dpg.set_value(CONSTANTS['STATUS_TEXT_TAG'], f"Status: {status}")


def _update_coordinator_data(coordinator, is_running):
    """Update UI with data from the simulation coordinator."""
    # pylint: disable=no-member
    state = coordinator.get_simulation_state()
    dpg.set_value("step_count_text", f"Step Count: {state.step_count}")
    graph = coordinator.get_neural_graph()
    if graph:
        dpg.set_value(CONSTANTS['NODES_TEXT_TAG'], f"Nodes: {graph.num_nodes}")
        dpg.set_value(CONSTANTS['EDGES_TEXT_TAG'], f"Edges: {graph.num_edges}")

    # Update metrics from the coordinator
    metrics = coordinator.get_performance_metrics()
    dpg.set_value("health_text", f"Health: {metrics.get('health_score', 'N/A')}")
    dpg.set_value(CONSTANTS['ENERGY_TEXT_TAG'], f"Energy: {state.total_energy:.2f}")
    dpg.set_value(CONSTANTS['CONNECTIONS_TEXT_TAG'], f"Connections: {graph.num_edges if graph else 0}")

    # Send data to visualization service for real-time rendering
    if _visualization_service and graph:
        try:
            # Create visualization data for neural activity
            neural_data = VisualizationData("neural_activity", time.time())
            neural_data.data = {
                "node_count": graph.num_nodes,
                "edge_count": graph.num_edges,
                "total_energy": state.total_energy,
                "step_count": state.step_count
            }
            neural_data.metadata = {"simulation_running": is_running}
            _visualization_service.update_visualization_data("neural_activity", neural_data)

            # Create visualization data for energy flow
            energy_data = VisualizationData("energy_flow", time.time())
            energy_data.data = {
                "total_energy": state.total_energy,
                "energy_distribution": "active" if state.total_energy > 0 else "inactive"
            }
            _visualization_service.update_visualization_data("energy_flow", energy_data)

        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            log_step(f"Error updating visualization service: {e}")


def _update_fallback_display(is_running):
    """Update UI display when no coordinator is available."""
    # pylint: disable=no-member
    dpg.set_value("step_count_text", "Step Count: 0")
    dpg.set_value(CONSTANTS['NODES_TEXT_TAG'], "Nodes: 0")
    dpg.set_value(CONSTANTS['EDGES_TEXT_TAG'], "Edges: 0")
    dpg.set_value("health_text", "Health: Unknown")
    dpg.set_value("criticality_text", "Criticality: Unknown")
    dpg.set_value("ei_ratio_text", "EI Ratio: Unknown")
    dpg.set_value(CONSTANTS['ENERGY_TEXT_TAG'], "Energy: Unknown")
    dpg.set_value(CONSTANTS['CONNECTIONS_TEXT_TAG'], "Connections: Unknown")
    dpg.set_value("events_log", "No events logged yet. Start simulation for live data." if not is_running else "Simulation active - load backend to see metrics.")


def _update_plots():
    """Update the live feed plots."""
    # pylint: disable=no-member
    energy_history = ui_state.get_live_feed_data().get('energy_history', [])
    if energy_history:
        ts = list(range(len(energy_history)))
        dpg.set_value("energy_series", [ts, energy_history])
    else:
        dpg.set_value("energy_series", [[0], [0.0]])
    activity_history = ui_state.get_live_feed_data().get('node_activity_history', [])
    if activity_history:
        ts = list(range(len(activity_history)))
        dpg.set_value("activity_series", [ts, activity_history])
    else:
        dpg.set_value("activity_series", [[0], [0.0]])
    dpg.set_value("perf_series", [[0], [0.0]])


def update_ui_display():
    """Update UI display with consolidated error handling and safe access."""
    # Skip DPG calls in test environments to avoid crashes
    skip = 'pytest' in sys.modules
    if skip:
        return

    is_running = get_simulation_running()
    coordinator = get_coordinator()

    _update_simulation_status(is_running)

    if coordinator:
        _update_coordinator_data(coordinator, is_running)
    else:
        _update_fallback_display(is_running)
        _update_plots()

