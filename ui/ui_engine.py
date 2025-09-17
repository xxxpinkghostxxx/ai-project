
import dearpygui.dearpygui as dpg
import time
import logging
from .ui_state_manager import get_ui_state_manager
from utils.common_utils import (
    safe_hasattr, safe_get_attr, safe_graph_access,
    create_safe_callback, extract_common_constants, get_common_error_messages
)
from utils.logging_utils import log_step

ui_state = get_ui_state_manager()
CONSTANTS = extract_common_constants()
ERROR_MSGS = get_common_error_messages()


def get_simulation_running():

    return ui_state.get_simulation_state()['simulation_running']


def set_simulation_running(running: bool):

    ui_state.set_simulation_running(running)


def get_latest_graph():

    return ui_state.get_latest_graph()


def get_latest_graph_for_ui():

    return ui_state.get_latest_graph_for_ui()


def update_graph(graph):

    ui_state.update_graph(graph)


def add_live_feed_data(data_type: str, value: float):

    ui_state.add_live_feed_data(data_type, value)


def get_live_feed_data():

    return ui_state.get_live_feed_data()


def clear_live_feed_data():

    ui_state.clear_live_feed_data()


def create_main_window():
    """Create the main UI window with consolidated constants."""
    with dpg.window(label=CONSTANTS['NEURAL_SIMULATION_TITLE'],
                   tag=CONSTANTS['MAIN_WINDOW_TAG'], width=800, height=600):
        dpg.add_text("Neural Simulation System")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start", callback=create_safe_callback(lambda: set_simulation_running(True)))
            dpg.add_button(label="Stop", callback=create_safe_callback(lambda: set_simulation_running(False)))
            dpg.add_button(label="Reset", callback=create_safe_callback(reset_simulation))
        dpg.add_text("Status: ", tag=CONSTANTS['STATUS_TEXT_TAG'])
        dpg.add_text("Nodes: ", tag=CONSTANTS['NODES_TEXT_TAG'])
        dpg.add_text("Edges: ", tag=CONSTANTS['EDGES_TEXT_TAG'])
        with dpg.group():
            dpg.add_text("Live Feed Data:")
            dpg.add_text("Energy: ", tag=CONSTANTS['ENERGY_TEXT_TAG'])
            dpg.add_text("Connections: ", tag=CONSTANTS['CONNECTIONS_TEXT_TAG'])


def reset_simulation():

    set_simulation_running(False)
    clear_live_feed_data()
    logging.info("Simulation reset")


def update_ui_display():
    """Update UI display with consolidated error handling and safe access."""
    try:
        # Update simulation status
        status = CONSTANTS['SIMULATION_STATUS_RUNNING'] if get_simulation_running() else CONSTANTS['SIMULATION_STATUS_STOPPED']
        dpg.set_value(CONSTANTS['STATUS_TEXT_TAG'], f"Status: {status}")

        # Update graph information
        graph = get_latest_graph_for_ui()
        if graph is not None:
            node_count = safe_graph_access(graph, 'get_node_count')
            edge_count = safe_graph_access(graph, 'get_edge_count')

            if node_count is not None:
                dpg.set_value(CONSTANTS['NODES_TEXT_TAG'], f"Nodes: {node_count}")
            if edge_count is not None:
                dpg.set_value(CONSTANTS['EDGES_TEXT_TAG'], f"Edges: {edge_count}")

        # Update live feed data
        live_data = get_live_feed_data()
        if 'energy' in live_data and live_data['energy']:
            dpg.set_value(CONSTANTS['ENERGY_TEXT_TAG'], f"Energy: {live_data['energy'][-1]:.2f}")
        if 'connections' in live_data and live_data['connections']:
            dpg.set_value(CONSTANTS['CONNECTIONS_TEXT_TAG'], f"Connections: {live_data['connections'][-1]:.0f}")
    except Exception as e:
        logging.error(f"{ERROR_MSGS['UI_UPDATE_ERROR']}: {e}")


def create_ui():

    dpg.create_context()
    dpg.create_viewport(title="Neural Simulation System", width=800, height=600)
    create_main_window()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        update_ui_display()
        dpg.render_dearpygui_frame()
        time.sleep(0.01)
    dpg.destroy_context()


def show_legend_help():

    with dpg.window(label="Legend Help", tag="legend_help", width=400, height=300):
        dpg.add_text("Interactive Legend Help:")
        dpg.add_separator()
        dpg.add_text("This legend shows the color coding system used in the neural network visualization.")
        dpg.add_text("")
        dpg.add_text("Sections:")
        dpg.add_text("• Node Types: Different types of neural nodes")
        dpg.add_text("• Node States: Current state of each node")
        dpg.add_text("• Energy Levels: Energy content of nodes")
        dpg.add_text("• Edge Types: Different types of connections")
        dpg.add_text("• Edge Strength: Strength of connections")
        dpg.add_text("")
        dpg.add_text("Controls:")
        dpg.add_text("• Use checkboxes to show/hide sections")
        dpg.add_text("• Descriptions provide detailed explanations")
        dpg.add_text("• Colors match the actual visualization")


def main():

    try:
        create_ui()
    except Exception as e:
        logging.error(f"Error in UI: {e}")
        print(f"UI Error: {e}")
if __name__ == "__main__":
    main()
