import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.event_bus import get_event_bus

from dearpygui import dearpygui as dpg
import time
import math
import logging
import json
from datetime import datetime
from ui.ui_state_manager import get_ui_state_manager
from utils.logging_utils import log_step

ui_state = get_ui_state_manager()
event_bus = get_event_bus()
event_bus.subscribe('GRAPH_UPDATE', lambda event_type, data: (ui_state.update_graph(data.get('graph')), update_graph_visualization()))
event_bus.subscribe('UI_REFRESH', lambda event_type, data: update_ui_display())

_manager = None

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
                    dpg.add_button(label="Start", callback=start_simulation_callback, width=100)
                    dpg.add_button(label="Stop", callback=stop_simulation_callback, width=100)
                    dpg.add_button(label="Reset", callback=reset_simulation_callback, width=100)
                    
                    # Neural map slot selector
                    dpg.add_text("Map Slot:")
                    slot_input = dpg.add_input_int(tag="map_slot", default_value=0, width=50)
                    dpg.add_button(label="Save Neural Map", callback=lambda: save_neural_map_callback(dpg.get_value(slot_input)), width=150)
                    dpg.add_button(label="Load Neural Map", callback=lambda: load_neural_map_callback(dpg.get_value(slot_input)), width=150)
                
                # Status panel
                with dpg.child_window(height=120, width=-1, tag="status_panel"):
                    dpg.add_text(default_value="Status: ", tag=CONSTANTS['STATUS_TEXT_TAG'])
                    dpg.add_text(default_value="Nodes: ", tag=CONSTANTS['NODES_TEXT_TAG'])
                    dpg.add_text(default_value="Edges: ", tag=CONSTANTS['EDGES_TEXT_TAG'])
                    dpg.add_text(default_value="Step Count: ", tag="step_count_text")
                    dpg.add_text(default_value="Health: ", tag="health_text")
                
                # Live metrics
                with dpg.child_window(height=250, width=-1, tag="metrics_panel"):
                    dpg.add_text(default_value="Live Metrics:")
                    dpg.add_text(default_value="Energy: ", tag=CONSTANTS['ENERGY_TEXT_TAG'])
                    dpg.add_text(default_value="Connections: ", tag=CONSTANTS['CONNECTIONS_TEXT_TAG'])
                    dpg.add_text(default_value="Criticality: ", tag="criticality_text")
                    dpg.add_text(default_value="EI Ratio: ", tag="ei_ratio_text")
                    dpg.add_separator()
                    dpg.add_text(default_value="Recent Events:")
                    dpg.add_input_text(default_value="", tag="events_log", multiline=True, readonly=True, height=120)

            with dpg.tab(label="Graph Visualization"):
                dpg.add_text(default_value="Neural Graph Visualization")
                dpg.add_separator()
                
                # Graph controls
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Show Nodes", tag="show_nodes", default_value=True, callback=lambda: update_graph_visualization())
                    dpg.add_checkbox(label="Show Edges", tag="show_edges", default_value=True, callback=lambda: update_graph_visualization())
                    dpg.add_checkbox(label="Color by Energy", tag="color_energy", default_value=True, callback=lambda: update_graph_visualization())
                    dpg.add_slider_float(label="Node Size", tag="node_size", default_value=2.0, min_value=0.5, max_value=10.0, callback=lambda: update_graph_visualization())
                    dpg.add_slider_float(label="Edge Thickness", tag="edge_thickness", default_value=1.0, min_value=0.1, max_value=5.0, callback=lambda: update_graph_visualization())
                
                # Visualization area
                dpg.add_drawlist(width=-1, height=600, tag="graph_view")
                
                dpg.add_text(default_value="Zoom: Use mouse wheel | Pan: Drag with right mouse button | Controls update live")

            with dpg.tab(label="Metrics & Plots"):
                dpg.add_text(default_value="Real-time Metrics and Historical Plots")
                dpg.add_separator()
                
                # Plots
                with dpg.plot(label="Energy History", height=300, width=-1, tag="energy_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="energy_axis")
                    dpg.add_line_series([], [], label="Average Energy", tag="energy_series", parent="energy_axis")
                
                with dpg.plot(label="Node Activity", height=300, width=-1, tag="activity_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="activity_axis")
                    dpg.add_line_series([], [], label="Active Nodes", tag="activity_series", parent="activity_axis")
                
                with dpg.plot(label="Performance", height=300, width=-1, tag="performance_plot"):
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

def get_manager():
    global _manager
    if _manager is None:
        try:
            import simulation_manager
            _manager = simulation_manager.get_simulation_manager()
            logging.info("SimulationManager loaded successfully")
        except ImportError as e:
            logging.error(f"Failed to load SimulationManager: {e}")
            _manager = None
    return _manager

def start_simulation_callback():
    global _manager
    try:
        manager = get_manager()
        if manager is None:
            raise ImportError("SimulationManager not available")
        if manager.graph is None:
            manager.initialize_graph()
        manager.start_simulation()
        set_simulation_running(True)
        dpg.set_value("events_log", "Simulation started")
        logging.info("UI Start button: Simulation started")
    except Exception as e:
        dpg.set_value("events_log", f"Start failed: {str(e)}")
        logging.error(f"Start simulation failed: {e}")

def stop_simulation_callback():
    global _manager
    try:
        manager = get_manager()
        if manager is None:
            dpg.set_value("events_log", "Simulation manager not loaded")
            return
        manager.stop_simulation()
        set_simulation_running(False)
        dpg.set_value("events_log", "Simulation stopped")
        logging.info("UI Stop button: Simulation stopped")
    except Exception as e:
        dpg.set_value("events_log", f"Stop failed: {str(e)}")
        logging.error(f"Stop simulation failed: {e}")

def reset_simulation_callback():
    global _manager
    try:
        manager = get_manager()
        if manager is None:
            dpg.set_value("events_log", "Simulation manager not loaded")
            return
        manager.reset_simulation()
        set_simulation_running(False)
        clear_live_feed_data()
        dpg.set_value("events_log", "Simulation reset")
        logging.info("UI Reset button: Simulation reset")
    except Exception as e:
        dpg.set_value("events_log", f"Reset failed: {str(e)}")
        logging.error(f"Reset simulation failed: {e}")

def reset_simulation():
    reset_simulation_callback()

def update_ui_display():
    """Update UI display with consolidated error handling and safe access."""
    global _manager
    is_running = get_simulation_running()
    manager = _manager

    # Update simulation status
    status = CONSTANTS['SIMULATION_STATUS_RUNNING'] if is_running else CONSTANTS['SIMULATION_STATUS_STOPPED']
    dpg.set_value(CONSTANTS['STATUS_TEXT_TAG'], f"Status: {status}")

    # Update step count
    step_count = ui_state.get_simulation_state().get('sim_update_counter', 0)
    if manager and hasattr(manager, 'step_counter'):
        step_count = manager.step_counter
    dpg.set_value("step_count_text", f"Step Count: {step_count}")

    # Update graph information
    graph = get_latest_graph_for_ui()
    node_count = 0
    edge_count = 0
    if graph is not None:
        node_count = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
        edge_count = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0
    elif manager and hasattr(manager, 'graph') and manager.graph is not None:
        graph = manager.graph
        node_count = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
        edge_count = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0

    dpg.set_value(CONSTANTS['NODES_TEXT_TAG'], f"Nodes: {node_count}")
    dpg.set_value(CONSTANTS['EDGES_TEXT_TAG'], f"Edges: {edge_count}")

    # Update health and metrics if manager available
    if manager:
        try:
            system_stats = manager.get_system_stats() if hasattr(manager, 'get_system_stats') else {}
            health = system_stats.get('health_score', 50.0)
            dpg.set_value("health_text", f"Health: {health:.1f}")

            if hasattr(manager, 'network_metrics') and manager.network_metrics and graph:
                metrics = manager.network_metrics.calculate_comprehensive_metrics(graph)
                dpg.set_value("criticality_text", f"Criticality: {metrics.get('criticality', 0.0):.3f}")
                ei_ratio = metrics.get('connectivity', {}).get('ei_ratio', 1.00)
                dpg.set_value("ei_ratio_text", f"EI Ratio: {ei_ratio:.2f}")
            else:
                dpg.set_value("criticality_text", "Criticality: Unknown")
                dpg.set_value("ei_ratio_text", "EI Ratio: Unknown")

            stats = manager.get_performance_stats() if hasattr(manager, 'get_performance_stats') else {}
            avg_energy = stats.get('avg_energy', 0.0)
            dpg.set_value(CONSTANTS['ENERGY_TEXT_TAG'], f"Energy: {avg_energy:.2f}")

            connections = edge_count
            dpg.set_value(CONSTANTS['CONNECTIONS_TEXT_TAG'], f"Connections: {connections}")

            # Update plots with time steps as x-axis
            time_steps = list(range(max(1, len(ui_state.get_live_feed_data().get('energy_history', [])))))
            if ui_state.live_feed_data.get('energy_history'):
                energy_y = ui_state.get_live_feed_data().get('energy_history', [])
                energy_data = [time_steps, energy_y]
                dpg.set_value("energy_series", energy_data)
            else:
                dpg.set_value("energy_series", [[0], [0.0]])

            if ui_state.live_feed_data.get('node_activity_history'):
                activity_y = ui_state.get_live_feed_data().get('node_activity_history', [])
                activity_data = [time_steps, activity_y]
                dpg.set_value("activity_series", activity_data)
            else:
                dpg.set_value("activity_series", [[0], [0.0]])

            perf_y = [stats.get('avg_step_time', 0.0) * 1000] * len(time_steps) if time_steps else [0.0]
            perf_data = [time_steps, perf_y]
            dpg.set_value("perf_series", perf_data)

            dpg.set_value("events_log", "Simulation active - metrics updating.")
        except Exception as e:
            logging.error(f"Manager update error: {e}")
            # Fallback to placeholders
            dpg.set_value("health_text", "Health: Unknown")
            dpg.set_value("criticality_text", "Criticality: Unknown")
            dpg.set_value("ei_ratio_text", "EI Ratio: Unknown")
            dpg.set_value(CONSTANTS['ENERGY_TEXT_TAG'], "Energy: Unknown")
            dpg.set_value(CONSTANTS['CONNECTIONS_TEXT_TAG'], "Connections: Unknown")
            dpg.set_value("events_log", f"Manager error: {str(e)}")
    else:
        # Placeholders when manager not loaded
        dpg.set_value("health_text", "Health: Unknown")
        dpg.set_value("criticality_text", "Criticality: Unknown")
        dpg.set_value("ei_ratio_text", "EI Ratio: Unknown")
        dpg.set_value(CONSTANTS['ENERGY_TEXT_TAG'], "Energy: Unknown")
        dpg.set_value(CONSTANTS['CONNECTIONS_TEXT_TAG'], "Connections: Unknown")
        dpg.set_value("events_log", "No events logged yet. Start simulation for live data." if not is_running else "Simulation active - load backend to see metrics.")
    
        # Update plots when no manager
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

    # Get drawlist size
    width = dpg.get_item_rect_size("graph_view")[0]
    height = dpg.get_item_rect_size("graph_view")[1]
    center_x, center_y = width / 2, height / 2

    # Basic zoom and pan (stored in state)
    zoom = ui_state.get_simulation_state().get('viz_zoom', 1.0)
    pan_x = ui_state.get_simulation_state().get('viz_pan_x', 0.0)
    pan_y = ui_state.get_simulation_state().get('viz_pan_y', 0.0)

    dpg.clear_draw_list()

    if show_nodes and hasattr(graph, 'node_labels') and graph.node_labels:
        # Use node positions if available, else simple layout
        num_nodes = len(graph.node_labels)
        cols = max(1, int(math.sqrt(num_nodes)))
        rows = math.ceil(num_nodes / cols)
        for i, node in enumerate(graph.node_labels[:500]):  # Increased limit
            if 'pos' in node and node['pos']:
                x, y = node['pos']
            else:
                # Improved grid layout
                x = (i % cols) / cols * width * zoom + pan_x
                y = (i // cols) / rows * height * zoom + pan_y
            energy = node.get('energy', 0.0)
            state = node.get('state', 'inactive')

            if color_energy:
                r = int(255 * min(energy / 255.0, 1.0))
                g = 255 - r
                b = 128
                color = [r, g, b, 255]
            else:
                color = node_active_color if state == 'active' else node_inactive_color

            dpg.draw_circle([x, y], node_size * zoom, color=color, thickness=2)

    if show_edges and hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
        num_nodes = len(graph.node_labels)
        cols = max(1, int(math.sqrt(num_nodes)))
        rows = math.ceil(num_nodes / cols)
        for j in range(min(graph.edge_index.shape[1], 200)):  # Increased limit
            src_idx = graph.edge_index[0, j].item()
            tgt_idx = graph.edge_index[1, j].item()
            
            # Get positions similarly
            if src_idx < len(graph.node_labels) and 'pos' in graph.node_labels[src_idx]:
                src_x, src_y = graph.node_labels[src_idx]['pos']
            else:
                src_x = (src_idx % cols) / cols * width * zoom + pan_x
                src_y = (src_idx // cols) / rows * height * zoom + pan_y
            
            if tgt_idx < len(graph.node_labels) and 'pos' in graph.node_labels[tgt_idx]:
                tgt_x, tgt_y = graph.node_labels[tgt_idx]['pos']
            else:
                tgt_x = (tgt_idx % cols) / cols * width * zoom + pan_x
                tgt_y = (tgt_idx // cols) / rows * height * zoom + pan_y
            
            dpg.draw_line([src_x, src_y], [tgt_x, tgt_y], color=edge_color, thickness=edge_thickness * zoom)

    # Note: Full zoom/pan requires mouse handlers; basic scaling here

def save_neural_map_callback(slot):
    global _manager
    """Callback to save neural map."""
    try:
        manager = get_manager()
        if manager is None:
            dpg.set_value("events_log", "Simulation manager not loaded")
            return
        if manager.graph is None:
            manager.initialize_graph()
        success = manager.save_neural_map(slot)
        dpg.set_value("events_log", f"Saved neural map to slot {slot}: {'Success' if success else 'Failed'}")
    except Exception as e:
        dpg.set_value("events_log", f"Save failed: {str(e)}")

def load_neural_map_callback(slot):
    global _manager
    """Callback to load neural map."""
    try:
        manager = get_manager()
        if manager is None:
            dpg.set_value("events_log", "Simulation manager not loaded")
            return
        success = manager.load_neural_map(slot)
        if success:
            ui_state.update_graph(manager.graph)
            update_graph_visualization()
        dpg.set_value("events_log", f"Loaded neural map from slot {slot}: {'Success' if success else 'Failed'}")
    except Exception as e:
        dpg.set_value("events_log", f"Load failed: {str(e)}")

def apply_config_changes():
    global _manager
    """Apply configuration changes from UI sliders."""
    try:
        manager = get_manager()
        if manager is None:
            dpg.set_value("events_log", "Simulation manager not loaded")
            return
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
        
        # Update viz params in state
        ui_state.live_feed_config['node_size'] = dpg.get_value("node_size")
        ui_state.live_feed_config['edge_thickness'] = dpg.get_value("edge_thickness")
        
        # Update manager config if available
        if hasattr(manager, 'set_config'):
            manager.set_config('Learning', 'ltp_rate', ltp_rate)
            manager.set_config('Learning', 'ltd_rate', ltd_rate)
        
        dpg.set_value("events_log", "Configuration changes applied and propagated")
        update_graph_visualization()
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
    dpg.create_viewport(title="Neural Simulation System - Enhanced UI", width=1600, height=1000)
    
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
    dpg.set_primary_window(CONSTANTS['MAIN_WINDOW_TAG'], value=True)
    
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
    dpg.add_menu_item(parent=file_menu, label="Save Neural Map", callback=lambda: save_neural_map_callback(dpg.get_value("map_slot")))
    dpg.add_menu_item(parent=file_menu, label="Load Neural Map", callback=lambda: load_neural_map_callback(dpg.get_value("map_slot")))
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
    global _manager
    """Export current metrics to file."""
    try:
        live_data = get_live_feed_data()
        health = ui_state.get_system_health()
        sim_steps = ui_state.get_simulation_state().get('sim_update_counter', 0)
        if _manager:
            sim_steps = _manager.step_counter if hasattr(_manager, 'step_counter') else sim_steps
            system_stats = _manager.get_system_stats() if hasattr(_manager, 'get_system_stats') else {}
            health = system_stats.get('health', health)
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'live_data': live_data,
            'health': health,
            'simulation_steps': sim_steps
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
