import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dearpygui import dearpygui as dpg
import time
import logging
import json
from datetime import datetime
from ui.ui_state_manager import get_ui_state_manager
from utils.logging_utils import log_step

ui_state = get_ui_state_manager()

# Define constants inline
CONSTANTS = {
    'NEURAL_SIMULATION_TITLE': 'Neural Simulation System',
    'MAIN_WINDOW_TAG': 'main_window',
    'STATUS_TEXT_TAG': 'status_text',
    'NODES_TEXT_TAG': 'nodes_text',
    'EDGES_TEXT_TAG': 'edges_text',
    'ENERGY_TEXT_TAG': 'energy_text',
    'CONNECTIONS_TEXT_TAG': 'connections_text',
    'SIMULATION_STATUS_RUNNING': 'Running',
    'SIMULATION_STATUS_STOPPED': 'Stopped'
}

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
    """Create the main UI window."""
    with dpg.window(label=CONSTANTS['NEURAL_SIMULATION_TITLE'],
                   tag=CONSTANTS['MAIN_WINDOW_TAG'], width=-1, height=-1):
        # Main tabs
        with dpg.tab_bar():
            with dpg.tab(label="Dashboard"):
                dpg.add_text(default_value="Neural Simulation Dashboard")
                dpg.add_separator()
                
                # Control panel
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Start", callback=lambda: set_simulation_running(True), width=100)
                    dpg.add_button(label="Stop", callback=lambda: set_simulation_running(False), width=100)
                    dpg.add_button(label="Reset", callback=reset_simulation, width=100)
                    dpg.add_button(label="Save Neural Map", callback=save_neural_map_callback, width=150)
                    dpg.add_button(label="Load Neural Map", callback=load_neural_map_callback, width=150)
                
                # Status panel
                with dpg.child_window(height=100, width=400, tag="status_panel"):
                    dpg.add_text(default_value="Status: ", tag=CONSTANTS['STATUS_TEXT_TAG'])
                    dpg.add_text(default_value="Nodes: ", tag=CONSTANTS['NODES_TEXT_TAG'])
                    dpg.add_text(default_value="Edges: ", tag=CONSTANTS['EDGES_TEXT_TAG'])
                    dpg.add_text(default_value="Step Count: ", tag="step_count_text")
                    dpg.add_text(default_value="Health: ", tag="health_text")
                
                # Live metrics
                with dpg.child_window(height=200, width=400, tag="metrics_panel"):
                    dpg.add_text(default_value="Live Metrics:")
                    dpg.add_text(default_value="Energy: ", tag=CONSTANTS['ENERGY_TEXT_TAG'])
                    dpg.add_text(default_value="Connections: ", tag=CONSTANTS['CONNECTIONS_TEXT_TAG'])
                    dpg.add_text(default_value="Criticality: ", tag="criticality_text")
                    dpg.add_text(default_value="EI Ratio: ", tag="ei_ratio_text")
                    dpg.add_separator()
                    dpg.add_text(default_value="Recent Events:")
                    dpg.add_input_text(default_value="", tag="events_log", multiline=True, readonly=True, height=80)

            with dpg.tab(label="Graph Visualization"):
                dpg.add_text(default_value="Neural Graph Visualization")
                dpg.add_separator()
                
                # Graph controls
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Show Nodes", tag="show_nodes", default_value=True)
                    dpg.add_checkbox(label="Show Edges", tag="show_edges", default_value=True)
                    dpg.add_checkbox(label="Color by Energy", tag="color_energy", default_value=True)
                    dpg.add_slider_float(label="Node Size", tag="node_size", default_value=2.0, min_value=0.5, max_value=10.0)
                    dpg.add_slider_float(label="Edge Thickness", tag="edge_thickness", default_value=1.0, min_value=0.1, max_value=5.0)
                
                # Visualization area
                dpg.add_drawlist(width=800, height=500, tag="graph_view")
                
                dpg.add_text(default_value="Zoom: Use mouse wheel | Pan: Drag with right mouse button")

            with dpg.tab(label="Metrics & Plots"):
                dpg.add_text(default_value="Real-time Metrics and Historical Plots")
                dpg.add_separator()
                
                # Plots
                with dpg.plot(label="Energy History", height=200, width=500, tag="energy_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="energy_axis")
                    dpg.add_line_series([], [], label="Average Energy", tag="energy_series", parent="energy_axis")
                
                with dpg.plot(label="Node Activity", height=200, width=500, tag="activity_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="activity_axis")
                    dpg.add_line_series([], [], label="Active Nodes", tag="activity_series", parent="activity_axis")
                
                with dpg.plot(label="Performance", height=200, width=500, tag="performance_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="perf_axis")
                    dpg.add_line_series([], [], label="Step Time (ms)", tag="perf_series", parent="perf_axis")

            with dpg.tab(label="Controls & Configuration"):
                dpg.add_text(default_value="Simulation Controls and Parameters")
                dpg.add_separator()
                
                # Parameter sliders
                learning_header = dpg.add_collapsing_header(label="Learning Parameters")
                with dpg.group(parent=learning_header):
                    dpg.add_slider_float(label="LTP Rate", tag="ltp_rate", default_value=0.02, min_value=0.001, max_value=0.1)
                    dpg.add_slider_float(label="LTD Rate", tag="ltd_rate", default_value=0.01, min_value=0.001, max_value=0.1)
                    dpg.add_slider_float(label="STDP Window (ms)", tag="stdp_window", default_value=20.0, min_value=5.0, max_value=50.0)
                
                energy_header = dpg.add_collapsing_header(label="Energy Parameters")
                with dpg.group(parent=energy_header):
                    dpg.add_slider_float(label="Birth Threshold", tag="birth_threshold", default_value=0.8, min_value=0.5, max_value=1.0)
                    dpg.add_slider_float(label="Death Threshold", tag="death_threshold", default_value=0.0, min_value=0.0, max_value=0.5)
                    dpg.add_slider_int(label="Update Interval", tag="update_interval", default_value=50, min_value=10, max_value=200)
                
                viz_header = dpg.add_collapsing_header(label="Visualization")
                with dpg.group(parent=viz_header):
                    dpg.add_color_edit(label="Node Active Color", tag="node_active_color", default_value=[0, 255, 0, 255])
                    dpg.add_color_edit(label="Node Inactive Color", tag="node_inactive_color", default_value=[128, 128, 128, 255])
                    dpg.add_color_edit(label="Edge Color", tag="edge_color", default_value=[255, 255, 255, 255])
                
                dpg.add_separator()
                dpg.add_button(label="Apply Changes", callback=apply_config_changes)
                dpg.add_button(label="Reset to Defaults", callback=reset_to_defaults)

            with dpg.tab(label="Help & Legend"):
                dpg.add_text(default_value="Neural Simulation Legend")
                dpg.add_separator()
                
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=200, height=300):
                        dpg.add_text(default_value="Node Types:")
                        dpg.add_text(default_value="• Sensory: Blue circles")
                        dpg.add_text(default_value="• Dynamic: Green circles")
                        dpg.add_text(default_value="• Oscillator: Pulsing yellow")
                        dpg.add_text(default_value="• Integrator: Purple diamonds")
                        dpg.add_text(default_value="• Relay: Orange squares")
                        
                        dpg.add_separator()
                        dpg.add_text(default_value="Node States:")
                        dpg.add_text(default_value="• Active: Bright color")
                        dpg.add_text(default_value="• Inactive: Dimmed color")
                        dpg.add_text(default_value="• Dying: Red tint")
                        dpg.add_text(default_value="• Newborn: Flashing")
                        
                        dpg.add_separator()
                        dpg.add_text(default_value="Edge Types:")
                        dpg.add_text(default_value="• Excitatory: Solid green")
                        dpg.add_text(default_value="• Inhibitory: Dashed red")
                        dpg.add_text(default_value="• Modulatory: Dotted blue")
                    
                    with dpg.child_window(width=200, height=300):
                        dpg.add_text(default_value="Energy Levels:")
                        dpg.add_text(default_value="• Low (<0.3): Dark")
                        dpg.add_text(default_value="• Medium (0.3-0.7): Medium brightness")
                        dpg.add_text(default_value="• High (>0.7): Bright/glowing")
                        
                        dpg.add_separator()
                        dpg.add_text(default_value="Learning Indicators:")
                        dpg.add_text(default_value="• LTP Active: Green glow on edges")
                        dpg.add_text(default_value="• LTD Active: Red glow on edges")
                        dpg.add_text(default_value="• Memory Trace: Blue outline on nodes")
                        dpg.add_text(default_value="• IEG Tagged: Yellow star")
                        
                        dpg.add_separator()
                        dpg.add_text(default_value="Controls:")
                        dpg.add_text(default_value="• Mouse wheel: Zoom")
                        dpg.add_text(default_value="• Right drag: Pan")
                        dpg.add_text(default_value="• Checkboxes: Toggle layers")
                        dpg.add_text(default_value="• Sliders: Adjust visuals")

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

        # Update step count
        step_count = ui_state.get_simulation_state().get('sim_update_counter', 0)
        dpg.set_value("step_count_text", f"Step Count: {step_count}")

        # Update graph information
        graph = get_latest_graph_for_ui()
        if graph is not None:
            node_count = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
            edge_count = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0

            dpg.set_value(CONSTANTS['NODES_TEXT_TAG'], f"Nodes: {node_count}")
            dpg.set_value(CONSTANTS['EDGES_TEXT_TAG'], f"Edges: {edge_count}")

            # Update health (placeholder)
            dpg.set_value("health_text", "Health: Unknown")

            # Update metrics (placeholder)
            dpg.set_value("criticality_text", "Criticality: 0.000")
            dpg.set_value("ei_ratio_text", "EI Ratio: 1.00")

        # Update live feed data (placeholder)
        dpg.set_value(CONSTANTS['ENERGY_TEXT_TAG'], "Energy: 0.00")
        dpg.set_value(CONSTANTS['CONNECTIONS_TEXT_TAG'], "Connections: 0")

        # Update plots (placeholder)
        dpg.set_value("energy_series", [[0], [0.0]])
        dpg.set_value("activity_series", [[0], [0.0]])
        dpg.set_value("perf_series", [[0], [0.0]])

        # Update events log (placeholder)
        dpg.set_value("events_log", "No events logged yet. Simulation required for live data.")

    except Exception as e:
        logging.error(f"UI update failed: {e}")

def update_graph_visualization():
    """Update the graph visualization drawlist."""
    graph = get_latest_graph_for_ui()
    if not graph:
        return

    show_nodes = dpg.get_value("show_nodes")
    show_edges = dpg.get_value("show_edges")
    color_energy = dpg.get_value("color_energy")
    node_size = dpg.get_value("node_size")
    edge_thickness = dpg.get_value("edge_thickness")

    node_active_color = dpg.get_value("node_active_color")
    node_inactive_color = dpg.get_value("node_inactive_color")
    edge_color = dpg.get_value("edge_color")

    width, height = 800, 500
    center_x, center_y = width / 2, height / 2

    if show_nodes and hasattr(graph, 'node_labels') and graph.node_labels:
        for i, node in enumerate(graph.node_labels[:100]):  # Limit to 100 nodes for performance
            x = (i % 10) * 80 + 50
            y = (i // 10) * 50 + 50
            energy = node.get('energy', 0.0)
            state = node.get('state', 'inactive')

            if color_energy:
                r = int(255 * min(energy / 255.0, 1.0))
                g = 255 - r
                b = 128
                color = [r, g, b, 255]
            else:
                color = node_active_color if state == 'active' else node_inactive_color

            dpg.draw_circle([x, y], node_size, color=color, thickness=2)

    if show_edges and hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
        for j in range(min(graph.edge_index.shape[1], 50)):  # Limit edges
            src = graph.edge_index[0, j].item() % 10 * 80 + 50
            tgt = graph.edge_index[1, j].item() % 10 * 80 + 50
            src_y = (graph.edge_index[0, j].item() // 10) * 50 + 50
            tgt_y = (graph.edge_index[1, j].item() // 10) * 50 + 50
            dpg.draw_line([src, src_y], [tgt, tgt_y], color=edge_color, thickness=edge_thickness)

def save_neural_map_callback():
    """Callback to save neural map."""
    try:
        slot = 0  # Default slot
        from simulation_manager import get_simulation_manager
        manager = get_simulation_manager()
        if manager:
            success = manager.save_neural_map(slot)
            dpg.set_value("events_log", f"Saved neural map to slot {slot}: {'Success' if success else 'Failed'}")
    except Exception as e:
        dpg.set_value("events_log", f"Save failed: {str(e)}")

def load_neural_map_callback():
    """Callback to load neural map."""
    try:
        slot = 0  # Default slot
        from simulation_manager import get_simulation_manager
        manager = get_simulation_manager()
        if manager:
            success = manager.load_neural_map(slot)
            dpg.set_value("events_log", f"Loaded neural map from slot {slot}: {'Success' if success else 'Failed'}")
    except Exception as e:
        dpg.set_value("events_log", f"Load failed: {str(e)}")

def apply_config_changes():
    """Apply configuration changes from UI sliders."""
    try:
        from simulation_manager import get_simulation_manager
        manager = get_simulation_manager()
        if manager:
            # Update learning parameters
            ltp_rate = dpg.get_value("ltp_rate")
            ltd_rate = dpg.get_value("ltd_rate")
            stdp_window = dpg.get_value("stdp_window")
            if hasattr(manager.learning_engine, 'ltp_rate'):
                manager.learning_engine.ltp_rate = ltp_rate
            if hasattr(manager.learning_engine, 'ltd_rate'):
                manager.learning_engine.ltd_rate = ltd_rate
            if hasattr(manager.learning_engine, 'stdp_window'):
                manager.learning_engine.stdp_window = stdp_window / 1000.0  # Convert to seconds
            
            # Update other parameters as needed
            dpg.set_value("events_log", "Configuration changes applied")
    except Exception as e:
        dpg.set_value("events_log", f"Apply failed: {str(e)}")

def reset_to_defaults():
    """Reset all parameters to defaults."""
    dpg.set_value("ltp_rate", 0.02)
    dpg.set_value("ltd_rate", 0.01)
    dpg.set_value("stdp_window", 20.0)
    dpg.set_value("birth_threshold", 0.8)
    dpg.set_value("death_threshold", 0.0)
    dpg.set_value("update_interval", 50)
    dpg.set_value("node_active_color", [0, 255, 0, 255])
    dpg.set_value("node_inactive_color", [128, 128, 128, 255])
    dpg.set_value("edge_color", [255, 255, 255, 255])
    dpg.set_value("events_log", "Reset to default parameters")
 
def update_frame():
    """Update UI and graph every frame."""
    update_ui_display()
    update_graph_visualization()
 
 
def create_ui():
    dpg.create_context()
    dpg.create_viewport(title="Neural Simulation System - Enhanced UI", width=1400, height=900)
    
    # Setup theme for better visuals
    with dpg.theme(tag="dark_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [20, 20, 30, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, [50, 50, 70, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, [70, 70, 90, 255])
            dpg.add_theme_color(dpg.mvThemeCol_Button, [60, 60, 80, 255])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [80, 80, 100, 255])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [100, 100, 120, 255])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [40, 40, 50, 255])
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, [100, 200, 255, 255])
    
    dpg.bind_theme("dark_theme")
    
    create_main_window()
    dpg.set_primary_window(CONSTANTS['MAIN_WINDOW_TAG'])
    
    # Create about dialog window
    with dpg.window(label="About Neural Simulation", modal=True, show=False, tag="about_dialog"):
        dpg.add_text(default_value="Neural Simulation System v2.0")
        dpg.add_text(default_value="Enhanced UI with real-time visualization")
        dpg.add_text(default_value="Features:")
        dpg.add_text(default_value="• Live graph rendering")
        dpg.add_text(default_value="• Real-time metrics plots")
        dpg.add_text(default_value="• Interactive controls")
        dpg.add_text(default_value="• Neural map persistence")
        dpg.add_text(default_value="• Advanced learning visualization")
        dpg.add_separator()
        dpg.add_text(default_value="Built with Dear PyGui and PyTorch Geometric")
        dpg.add_button(label="Close", callback=lambda: dpg.configure_item("about_dialog", show=False))
    
    # Add menu bar
    main_menu = dpg.add_menu_bar(parent=CONSTANTS['MAIN_WINDOW_TAG'], tag="main_menu")
    file_menu = dpg.add_menu(parent=main_menu, label="File", tag="file_menu")
    dpg.add_menu_item(parent=file_menu, label="Save Neural Map", callback=save_neural_map_callback)
    dpg.add_menu_item(parent=file_menu, label="Load Neural Map", callback=load_neural_map_callback)
    dpg.add_menu_item(parent=file_menu, label="Export Metrics", callback=export_metrics)
    dpg.add_menu_item(parent=file_menu, label="Exit", callback=lambda: dpg.stop_dearpygui())
    
    view_menu = dpg.add_menu(parent=main_menu, label="View", tag="view_menu")
    dpg.add_menu_item(parent=view_menu, label="Toggle Fullscreen", callback=lambda: dpg.toggle_viewport_fullscreen())
    
    help_menu = dpg.add_menu(parent=main_menu, label="Help", tag="help_menu")
    dpg.add_menu_item(parent=help_menu, label="About", callback=lambda: dpg.configure_item("about_dialog", show=True))
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    while dpg.is_dearpygui_running():
        update_ui_display()
        dpg.render_dearpygui_frame()
        time.sleep(1/60.0)  # 60 FPS
    
    dpg.destroy_context()


def export_metrics():
    """Export current metrics to file."""
    try:
        live_data = get_live_feed_data()
        health = ui_state.get_system_health()
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'live_data': live_data,
            'health': health,
            'simulation_steps': ui_state.get_simulation_state().get('sim_update_counter', 0)
        }
        filename = f"simulation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        dpg.set_value("events_log", f"Metrics exported to {filename}")
    except Exception as e:
        dpg.set_value("events_log", f"Export failed: {str(e)}")

def main():
    try:
        create_ui()
    except Exception as e:
        logging.error(f"Error in UI: {e}")
        print(f"UI Error: {e}")

if __name__ == "__main__":
    main()
