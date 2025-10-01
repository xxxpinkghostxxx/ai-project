"""
Main UI creation and setup functions.

This module contains the main UI window creation, theme setup,
and application entry points.
"""

import logging
import sys
import time
import traceback

import dearpygui as dpg

from src.ui.ui_constants import CONSTANTS, _visualization_service
from src.ui.ui_input import (
    setup_event_subscriptions, start_simulation_callback, stop_simulation_callback,
    reset_simulation_callback, view_logs_callback, auto_start_simulation,
    export_metrics, force_close_application, show_keyboard_shortcuts,
    save_neural_map_callback, load_neural_map_callback, apply_config_changes,
    reset_to_defaults
)
from src.utils.logging_utils import log_step
from src.ui.ui_rendering import update_graph_visualization, update_frame
from src.core.interfaces.real_time_visualization import IRealTimeVisualization
# pylint: disable=invalid-name
_service_registry = None


# pylint: disable=too-many-statements
def create_main_window():
    """Create the main UI window."""
    # Skip DPG calls in test environments to avoid crashes
    skip = 'pytest' in sys.modules
    if skip:
        return

    with dpg.window(label=CONSTANTS['NEURAL_SIMULATION_TITLE'],
                    tag=CONSTANTS['MAIN_WINDOW_TAG'], width=-1, height=-1):
        # Main tabs
        with dpg.tab_bar():
            with dpg.tab(label="Dashboard"):
                dpg.add_text(default_value="Neural Simulation Dashboard", tag="dashboard_title")
                dpg.add_separator()

                with dpg.child_window(height=-1, width=-1, tag="dashboard_scroll", no_scrollbar=False, border=False):

                    # Control panel with improved spacing
                    with dpg.group(horizontal=True):
                        start_btn = dpg.add_button(label="Start", callback=start_simulation_callback, width=100)
                        stop_btn = dpg.add_button(label="Stop", callback=stop_simulation_callback, width=100)
                        reset_btn = dpg.add_button(label="Reset", callback=reset_simulation_callback, width=100)
                        view_logs_btn = dpg.add_button(label="View Logs", callback=view_logs_callback, width=100)
                        force_close_btn = dpg.add_button(label="Force Close", callback=force_close_application, width=100)

                        # Neural map slot selector
                        dpg.add_text("Map Slot:")
                        slot_input = dpg.add_input_int(tag="map_slot", default_value=0, width=50)
                        save_btn = dpg.add_button(label="Save Neural Map", callback=lambda: save_neural_map_callback(dpg.get_value("map_slot")), width=150)
                        load_btn = dpg.add_button(label="Load Neural Map", callback=lambda: load_neural_map_callback(dpg.get_value("map_slot")), width=150)

                    # Tooltips for control panel (moved outside group for container balance)
                    with dpg.tooltip(parent=start_btn):
                        dpg.add_text("Begin the neural simulation (Ctrl+S)")
                    with dpg.tooltip(parent=stop_btn):
                        dpg.add_text("Stop the current simulation (Ctrl+P)")
                    with dpg.tooltip(parent=reset_btn):
                        dpg.add_text("Reset simulation state and clear data (Ctrl+R)")
                    with dpg.tooltip(parent=view_logs_btn):
                        dpg.add_text("Open detailed event logs (Ctrl+L)")
                    with dpg.tooltip(parent=force_close_btn):
                        dpg.add_text("Force close application with cleanup (Ctrl+Q)")
                    with dpg.tooltip(parent=slot_input):
                        dpg.add_text("Select neural map slot (0-9)")
                    with dpg.tooltip(parent=save_btn):
                        dpg.add_text("Save current graph to selected slot")
                    with dpg.tooltip(parent=load_btn):
                        dpg.add_text("Load graph from selected slot")

                    dpg.add_separator()

                    # Status panel - responsive height
                    with dpg.child_window(height=-1, width=-1, tag="status_panel", border=True):
                        dpg.add_text(default_value="Status: ", tag=CONSTANTS['STATUS_TEXT_TAG'])
                        dpg.add_text(default_value="Nodes: ", tag=CONSTANTS['NODES_TEXT_TAG'])
                        dpg.add_text(default_value="Edges: ", tag=CONSTANTS['EDGES_TEXT_TAG'])
                        dpg.add_text(default_value="Step Count: ", tag="step_count_text")
                        dpg.add_text(default_value="Health: ", tag="health_text")
                        dpg.add_text(default_value="Operation: ", tag="operation_status_text")
                        operation_progress = dpg.add_progress_bar(default_value=0.0, tag="operation_progress", width=-1)
                        with dpg.tooltip(parent=operation_progress):
                            dpg.add_text("Progress of current operation")

                    dpg.add_separator()

                    # Live metrics - responsive
                    with dpg.child_window(height=-1, width=-1, tag="metrics_panel", border=True):
                        dpg.add_text(default_value="Live Metrics:")
                        dpg.add_text(default_value="Energy: ", tag=CONSTANTS['ENERGY_TEXT_TAG'])
                        dpg.add_text(default_value="Connections: ", tag=CONSTANTS['CONNECTIONS_TEXT_TAG'])
                        dpg.add_text(default_value="Criticality: ", tag="criticality_text")
                        dpg.add_text(default_value="EI Ratio: ", tag="ei_ratio_text")
                        dpg.add_separator()
                        dpg.add_text(default_value="Recent Events:")
                        events_log = dpg.add_input_text(default_value="", tag="events_log", multiline=True, readonly=True, height=120)
                        with dpg.tooltip(parent=events_log):
                            dpg.add_text("Recent simulation events and logs")

            with dpg.tab(label="Graph Visualization"):
                dpg.add_text(default_value="Neural Graph Visualization")
                dpg.add_separator()

                with dpg.child_window(height=-1, width=-1, tag="graph_scroll", no_scrollbar=False, horizontal_scrollbar=True, border=False):

                    # Graph controls with tooltips
                    with dpg.group(horizontal=True):
                        show_nodes_cb = dpg.add_checkbox(label="Show Nodes", tag="show_nodes", default_value=True, callback=update_graph_visualization)
                        with dpg.tooltip(parent=show_nodes_cb):
                            dpg.add_text("Toggle visibility of neural nodes")
                        show_edges_cb = dpg.add_checkbox(label="Show Edges", tag="show_edges", default_value=True, callback=update_graph_visualization)
                        with dpg.tooltip(parent=show_edges_cb):
                            dpg.add_text("Toggle visibility of connections")
                        color_energy_cb = dpg.add_checkbox(label="Color by Energy", tag="color_energy", default_value=True, callback=update_graph_visualization)
                        with dpg.tooltip(parent=color_energy_cb):
                            dpg.add_text("Color nodes based on energy levels")
                        node_size_slider = dpg.add_slider_float(label="Node Size", tag="node_size", default_value=2.0, min_value=0.5, max_value=10.0, callback=update_graph_visualization)
                        with dpg.tooltip(parent=node_size_slider):
                            dpg.add_text("Adjust size of node circles")
                        edge_thickness_slider = dpg.add_slider_float(label="Edge Thickness", tag="edge_thickness", default_value=1.0, min_value=0.1, max_value=5.0, callback=update_graph_visualization)
                        with dpg.tooltip(parent=edge_thickness_slider):
                            dpg.add_text("Adjust thickness of connection lines")

                    dpg.add_separator()

                    # Visualization area - responsive height
                    with dpg.child_window(height=-1, width=-1, no_scrollbar=False, horizontal_scrollbar=True, border=True, tag="graph_container"):
                        dpg.add_drawlist(width=-1, height=-1, tag="graph_view")

                    dpg.add_text(default_value="Zoom: Use mouse wheel | Pan: Drag with right mouse button | Controls update live", color=[150, 150, 150, 255])

            with dpg.tab(label="Metrics & Plots"):
                dpg.add_text(default_value="Real-time Metrics and Historical Plots")
                dpg.add_separator()

                with dpg.child_window(height=-1, width=-1, tag="metrics_scroll", no_scrollbar=False, horizontal_scrollbar=True, border=False):

                    # Plots - responsive heights
                    with dpg.group(horizontal=False):
                        dpg.add_text("Energy History Plot - Hover for values")
                        energy_plot = dpg.add_plot(label="Energy History", height=200, width=-1, tag="energy_plot")
                        dpg.add_plot_legend(parent=energy_plot)
                        dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="energy_axis", parent=energy_plot)
                        dpg.add_line_series([], [], label="Average Energy", tag="energy_series", parent="energy_axis")

                        dpg.add_text("Node Activity Plot - Hover for values")
                        activity_plot = dpg.add_plot(label="Node Activity", height=200, width=-1, tag="activity_plot")
                        dpg.add_plot_legend(parent=activity_plot)
                        dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="activity_axis", parent=activity_plot)
                        dpg.add_line_series([], [], label="Active Nodes", tag="activity_series", parent="activity_axis")

                        dpg.add_text("Performance Plot - Hover for values")
                        perf_plot = dpg.add_plot(label="Performance", height=200, width=-1, tag="performance_plot")
                        dpg.add_plot_legend(parent=perf_plot)
                        dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="perf_axis", parent=perf_plot)
                        dpg.add_line_series([], [], label="Step Time (ms)", tag="perf_series", parent="perf_axis")

            with dpg.tab(label="Controls & Configuration"):
                dpg.add_text(default_value="Simulation Controls and Parameters")
                dpg.add_separator()

                with dpg.child_window(height=-1, width=-1, tag="controls_scroll", no_scrollbar=False, horizontal_scrollbar=True, border=False):

                    # Parameter sliders with tooltips
                    learning_header = dpg.add_collapsing_header(label="Learning Parameters")
                    with dpg.tooltip(parent=learning_header):
                        dpg.add_text("Adjust synaptic plasticity rates")
                    with dpg.group(parent=learning_header):
                        ltp_slider = dpg.add_slider_float(label="LTP Rate", tag="ltp_rate", default_value=0.02, min_value=0.001, max_value=0.1)
                        with dpg.tooltip(parent=ltp_slider):
                            dpg.add_text("Long-term potentiation learning rate")
                        ltd_slider = dpg.add_slider_float(label="LTD Rate", tag="ltd_rate", default_value=0.01, min_value=0.001, max_value=0.1)
                        with dpg.tooltip(parent=ltd_slider):
                            dpg.add_text("Long-term depression learning rate")
                        stdp_slider = dpg.add_slider_float(label="STDP Window (ms)", tag="stdp_window", default_value=20.0, min_value=5.0, max_value=50.0)
                        with dpg.tooltip(parent=stdp_slider):
                            dpg.add_text("Spike-timing dependent plasticity temporal window")

                    energy_header = dpg.add_collapsing_header(label="Energy Parameters")
                    with dpg.tooltip(parent=energy_header):
                        dpg.add_text("Configure node birth and death thresholds")
                    with dpg.group(parent=energy_header):
                        birth_slider = dpg.add_slider_float(label="Birth Threshold", tag="birth_threshold", default_value=0.8, min_value=0.5, max_value=1.0)
                        with dpg.tooltip(parent=birth_slider):
                            dpg.add_text("Energy level required for new node creation")
                        death_slider = dpg.add_slider_float(label="Death Threshold", tag="death_threshold", default_value=0.0, min_value=0.0, max_value=0.5)
                        with dpg.tooltip(parent=death_slider):
                            dpg.add_text("Energy level below which nodes die")
                        update_slider = dpg.add_slider_int(label="Update Interval", tag="update_interval", default_value=50, min_value=10, max_value=200)
                        with dpg.tooltip(parent=update_slider):
                            dpg.add_text("Simulation update frequency in ms")

                    viz_header = dpg.add_collapsing_header(label="Visualization")
                    with dpg.tooltip(parent=viz_header):
                        dpg.add_text("Customize visual appearance of graph elements")
                    with dpg.group(parent=viz_header):
                        active_color = dpg.add_color_edit(label="Node Active Color", tag="node_active_color", default_value=[0, 255, 0, 255])
                        with dpg.tooltip(parent=active_color):
                            dpg.add_text("Color for active/high-energy nodes")
                        inactive_color = dpg.add_color_edit(label="Node Inactive Color", tag="node_inactive_color", default_value=[128, 128, 128, 255])
                        with dpg.tooltip(parent=inactive_color):
                            dpg.add_text("Color for inactive/low-energy nodes")
                        edge_color_edit = dpg.add_color_edit(label="Edge Color", tag="edge_color", default_value=[255, 255, 255, 255])
                        with dpg.tooltip(parent=edge_color_edit):
                            dpg.add_text("Color for connection edges")

                    dpg.add_separator()
                    apply_btn = dpg.add_button(label="Apply Changes", callback=apply_config_changes)
                    with dpg.tooltip(parent=apply_btn):
                        dpg.add_text("Apply all parameter changes to simulation")
                    reset_btn_config = dpg.add_button(label="Reset to Defaults", callback=reset_to_defaults)
                    with dpg.tooltip(parent=reset_btn_config):
                        dpg.add_text("Restore all parameters to default values")

            with dpg.tab(label="Help & Legend"):
                dpg.add_text(default_value="Neural Simulation Legend")
                dpg.add_separator()

                with dpg.child_window(height=-1, width=-1, tag="help_scroll", no_scrollbar=False, horizontal_scrollbar=True, border=False):

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


# pylint: disable=too-many-statements
def create_ui():
    """Create and initialize the main UI with Dear PyGui.

    This function sets up the complete user interface including:
    - Dear PyGui context and viewport
    - Modern neural-themed styling
    - Main window with tabs (Dashboard, Graph Visualization, Metrics, Controls, Help)
    - Event subscriptions and menu bar
    - Auto-start simulation functionality

    Raises:
        Exception: If UI initialization fails, attempts fallback UI creation.
    """
    try:
        log_step("Creating Dear PyGui context...")
        dpg.create_context()
        log_step("Context created successfully")

        log_step("Creating viewport...")
        dpg.create_viewport(title="Neural Simulation System - Enhanced UI", width=1600, height=1000)
        dpg.set_viewport_resizable(True)
        log_step("Viewport created successfully")
    except Exception as e:
        log_step(f"Failed to initialize Dear PyGui: {e}")
        raise

    # Setup enhanced modern theme
    with dpg.theme(tag="modern_neural_theme"):
        with dpg.theme_component(dpg.mvAll):
            # Background and windows
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [15, 20, 35, 255])  # Deep navy
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [25, 30, 45, 255])
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, [25, 30, 45, 255])

            # Title bars
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, [40, 50, 70, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, [60, 70, 90, 255])

            # Buttons - Neural green accents
            dpg.add_theme_color(dpg.mvThemeCol_Button, [40, 70, 50, 255])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [50, 90, 70, 255])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [60, 110, 90, 255])

            # Frames and inputs
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [30, 40, 55, 255])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, [40, 50, 65, 255])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, [50, 60, 75, 255])

            # Sliders and plots
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, [100, 200, 150, 255])
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, [120, 220, 170, 255])
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, [100, 200, 150, 255])
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, [60, 150, 100, 255])

            # Text and selections
            dpg.add_theme_color(dpg.mvThemeCol_Text, [220, 220, 230, 255])
            dpg.add_theme_color(dpg.mvThemeCol_Header, [50, 70, 90, 255])
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, [60, 80, 100, 255])
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, [70, 90, 110, 255])

            # Tab bar
            dpg.add_theme_color(dpg.mvThemeCol_Tab, [30, 40, 55, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, [40, 50, 65, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, [50, 60, 75, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, [25, 30, 45, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, [35, 40, 55, 255])

            # Rounding for modern look
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)

            # Spacing and padding
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 4, 2)

    dpg.bind_theme("modern_neural_theme")
    dpg.set_global_font_scale(1.2)

    create_main_window()
    setup_event_subscriptions()
    dpg.set_primary_window(CONSTANTS['MAIN_WINDOW_TAG'], True)

    # Create about dialog window
    with dpg.window(label="About Neural Simulation", modal=True, show=False, tag="about_dialog", no_move=True):
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
        close_about = dpg.add_button(label="Close", callback=lambda: dpg.configure_item("about_dialog", show=False))
        with dpg.tooltip(parent=close_about):
            dpg.add_text("Close this dialog")

    # Create logs modal
    with dpg.window(label="Event Logs", modal=True, show=False, tag=CONSTANTS['LOGS_MODAL_TAG'], no_move=True):
        dpg.add_text("Detailed Simulation Logs:")
        dpg.add_separator()
        logs_text = dpg.add_input_text(default_value="No logs available.", tag=CONSTANTS['LOGS_TEXT_TAG'], multiline=True, readonly=True, height=400, width=600)
        with dpg.tooltip(parent=logs_text):
            dpg.add_text("Scrollable log of recent events")
        dpg.add_separator()
        dpg.add_button(label="Close", callback=lambda: dpg.configure_item(CONSTANTS['LOGS_MODAL_TAG'], show=False))

    # Add menu bar
    main_menu = dpg.add_menu_bar(parent=CONSTANTS['MAIN_WINDOW_TAG'], tag="main_menu")
    file_menu = dpg.add_menu(parent=main_menu, label="File", tag="file_menu")
    dpg.add_menu_item(parent=file_menu, label="Save Neural Map", callback=lambda: save_neural_map_callback(dpg.get_value("map_slot")))
    dpg.add_menu_item(parent=file_menu, label="Load Neural Map", callback=lambda: load_neural_map_callback(dpg.get_value("map_slot")))
    dpg.add_menu_item(parent=file_menu, label="Export Metrics", callback=export_metrics)
    dpg.add_menu_item(parent=file_menu, label="Force Close", callback=force_close_application)
    dpg.add_menu_item(parent=file_menu, label="Exit", callback=dpg.stop_dearpygui)

    view_menu = dpg.add_menu(parent=main_menu, label="View", tag="view_menu")
    dpg.add_menu_item(parent=view_menu, label="Toggle Fullscreen", callback=dpg.toggle_viewport_fullscreen)

    help_menu = dpg.add_menu(parent=main_menu, label="Help", tag="help_menu")
    dpg.add_menu_item(parent=help_menu, label="About", callback=lambda: dpg.configure_item("about_dialog", show=True))
    dpg.add_menu_item(parent=help_menu, label="Keyboard Shortcuts", callback=show_keyboard_shortcuts)

    # Keyboard shortcuts disabled for DearPyGui 2.1.0 compatibility
    # The API for keyboard handlers has changed and needs investigation
    log_step("Keyboard shortcuts disabled - requires API compatibility update")

    try:
        log_step("Setting up Dear PyGui...")
        dpg.setup_dearpygui()
        log_step("Dear PyGui setup complete")

        log_step("Showing viewport...")
        dpg.show_viewport()
        log_step("Viewport shown successfully")

        # Auto-start the simulation after UI is set up but before the loop
        log_step("Auto-starting simulation...")
        auto_start_simulation()
        log_step("Simulation auto-started successfully")

        log_step("Starting main UI loop...")
        frame_count = 0
        start_time = time.time()
        max_runtime = 300  # 5 minutes maximum runtime

        while dpg.is_dearpygui_running():
            try:
                # Check for timeout
                if time.time() - start_time > max_runtime:
                    log_step(f"UI timeout reached after {max_runtime} seconds")
                    break

                update_frame()
                dpg.render_dearpygui_frame()
                frame_count += 1
                if frame_count % 300 == 0:  # Log every 5 seconds at 60 FPS
                    log_step(f"UI running - frame {frame_count}")
                time.sleep(1/60.0)  # 60 FPS
            except (RuntimeError, AttributeError, ValueError, TypeError, KeyboardInterrupt) as e:
                log_step(f"Error in UI loop: {e}")

                log_step(f"Traceback: {traceback.format_exc()}")
                break

        log_step("UI loop ended, destroying context...")

        # Stop simulation and DearPyGui properly


        try:
            dpg.destroy_context()
            log_step("DearPyGui context destroyed")
        except (RuntimeError, AttributeError, OSError) as e:  # Specific exceptions for context cleanup
            log_step(f"Error destroying context: {e}")

        log_step("UI shutdown complete")

    except Exception as e:
        log_step(f"Critical UI error: {e}")
        try:
            dpg.destroy_context()
        except (RuntimeError, AttributeError) as cleanup_error:
            log_step(f"Error during context cleanup: {cleanup_error}")
        raise


def create_fallback_ui():
    """Create a minimal fallback UI if the main UI fails."""
    try:
        log_step("Creating fallback UI...")
        dpg.create_context()
        dpg.create_viewport(title="Neural Simulation - Fallback Mode", width=800, height=600)

        with dpg.window(label="Fallback Neural Simulation", tag="fallback_window"):
            dpg.add_text("Neural Simulation System - Fallback Mode")
            dpg.add_separator()
            dpg.add_text("The full UI failed to load. This is a minimal interface.")
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_button(label="Force Close", callback=force_close_application)
                dpg.add_button(label="Exit", callback=dpg.stop_dearpygui)

            dpg.add_separator()
            dpg.add_text("Available features in fallback mode:")
            dpg.add_text("• Force close application")
            dpg.add_text("• Basic exit functionality")
            dpg.add_text("• Error logging")

        dpg.setup_dearpygui()
        dpg.show_viewport()

        log_step("Fallback UI created successfully")

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            time.sleep(1/30.0)  # 30 FPS for fallback

        dpg.destroy_context()

    except Exception as e:  # Broad exception catch needed for UI fallback creation
        log_step(f"Fallback UI creation failed: {e}")
        raise


def run_ui(service_registry):
    """Entry point for launcher to run the UI."""
    global _service_registry, _visualization_service  # pylint: disable=global-statement
    _service_registry = service_registry

    # Initialize visualization service
    try:
        _visualization_service = service_registry.resolve(IRealTimeVisualization)
        viz_config = {
            "target_fps": 30,
            "max_buffer_size": 1000,
            "enable_interpolation": True,
            "default_layers": ["neural_activity", "energy_flow", "connections", "learning_events"]
        }
        if not _visualization_service.initialize_visualization(viz_config):
            log_step("Warning: Real-time visualization service initialization failed")
    except (RuntimeError, AttributeError, ValueError, TypeError, ImportError) as e:
        log_step(f"Visualization service initialization failed: {e}")
        _visualization_service = None

    try:
        log_step("Starting Neural Simulation UI...")
        create_ui()
        log_step("UI initialization completed successfully")

    except (RuntimeError, AttributeError, ValueError, TypeError, ImportError) as e:
        log_step(f"UI initialization failed: {e}")
        log_step("Attempting fallback mode...")

        # Try fallback simple UI
        try:
            create_fallback_ui()
        except (RuntimeError, AttributeError, ValueError, TypeError, ImportError) as e2:
            log_step(f"Fallback UI also failed: {e2}")
            log_step("UI completely failed to initialize")
            raise e from e2


def main():
    """Main entry point for running the UI as a standalone application.

    This function serves as the primary entry point when the module
    is executed directly. It initializes and starts the UI, handling
    any initialization errors that may occur.

    Raises:
        Exception: Re-raises any unhandled exceptions from UI creation.
    """
    try:
        create_ui()
    except (RuntimeError, AttributeError, ImportError, OSError) as e:
        logging.error("Error in UI: %s", e)
        logging.error("UI initialization failed")
