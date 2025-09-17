
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from typing import Dict, Any, Optional, List
import time
import logging
from ui_state_manager import get_ui_state_manager
ui_state = get_ui_state_manager()


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

    with dpg.window(label="Neural Simulation", tag="main_window", width=800, height=600):
        dpg.add_text("Neural Simulation System")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start", callback=lambda: set_simulation_running(True))
            dpg.add_button(label="Stop", callback=lambda: set_simulation_running(False))
            dpg.add_button(label="Reset", callback=lambda: reset_simulation())
        dpg.add_text("Status: ", tag="status_text")
        dpg.add_text("Nodes: ", tag="nodes_text")
        dpg.add_text("Edges: ", tag="edges_text")
        with dpg.group():
            dpg.add_text("Live Feed Data:")
            dpg.add_text("Energy: ", tag="energy_text")
            dpg.add_text("Connections: ", tag="connections_text")


def reset_simulation():

    set_simulation_running(False)
    clear_live_feed_data()
    logging.info("Simulation reset")


def update_ui_display():

    try:
        status = "Running" if get_simulation_running() else "Stopped"
        dpg.set_value("status_text", f"Status: {status}")
        graph = get_latest_graph_for_ui()
        if graph is not None:
            if hasattr(graph, 'node_labels'):
                dpg.set_value("nodes_text", f"Nodes: {len(graph.node_labels)}")
            if hasattr(graph, 'edge_index'):
                dpg.set_value("edges_text", f"Edges: {graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0}")
        live_data = get_live_feed_data()
        if 'energy' in live_data and live_data['energy']:
            dpg.set_value("energy_text", f"Energy: {live_data['energy'][-1]:.2f}")
        if 'connections' in live_data and live_data['connections']:
            dpg.set_value("connections_text", f"Connections: {live_data['connections'][-1]:.0f}")
    except Exception as e:
        logging.error(f"Error updating UI display: {e}")


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
