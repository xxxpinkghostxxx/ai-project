import dearpygui.dearpygui as dpg
# Lazy import to avoid hanging during module load
# from main_graph import initialize_main_graph  # <-- Moved to lazy import
import numpy as np
import time
from threading import Thread, Lock
import logging
from logging_utils import (
    setup_logging,
    append_log_line,
    log_runtime,
    log_step,
    get_log_lines,
)
# from error_handling_utils import handle_ui_error, log_ui_error, safe_execute  # Not currently used
# from main_loop import update_dynamic_node_energies  # REMOVED: function no longer exists
from death_and_birth_logic import birth_new_dynamic_nodes, remove_dead_dynamic_nodes

# --- SECTION: Logging Setup ---
# Set up logging to console only (no file output)
setup_logging(level="INFO")

# --- SECTION: Performance Goals ---
TARGET_FPS = 30  # System should ideally run at 30 frames per second
UPDATE_INTERVAL = 1.0 / TARGET_FPS  # seconds between updates
SENSORY_UI_UPDATE_EVERY_N = (
    4  # UI updates its visualization every 4th simulation update
)

# --- DearPyGui UI Engine for Energy-Based Neural System ---
# This file is structured for easy AI/human extension. See section markers.

# --- SECTION: Thread-Safe State Management ---
# Import thread-safe UI state manager
from ui_state_manager import get_ui_state_manager

# Get thread-safe state manager
ui_state = get_ui_state_manager()

# Thread-safe access to global state
def get_simulation_running():
    """
    Get the current simulation running state.
    
    Returns:
        bool: True if simulation is running, False otherwise
    """
    return ui_state.get_simulation_state()['simulation_running']

def set_simulation_running(running: bool):
    """
    Set the simulation running state.
    
    Args:
        running (bool): True to start simulation, False to stop
    """
    ui_state.set_simulation_running(running)

def get_latest_graph():
    """
    Get the latest graph from the simulation.
    
    Returns:
        Any: The latest graph object or None if no graph is available
    """
    return ui_state.get_latest_graph()

def get_latest_graph_for_ui():
    """
    Get the latest graph formatted for UI display.
    
    Returns:
        Any: The latest graph object formatted for UI or None if no graph is available
    """
    return ui_state.get_latest_graph_for_ui()

def update_graph(graph):
    """
    Update the current graph in the UI state.
    
    Args:
        graph: The graph object to update
    """
    ui_state.update_graph(graph)

def add_live_feed_data(data_type: str, value: float):
    """
    Add live feed data to the UI state.
    
    Args:
        data_type (str): Type of data (e.g., 'energy', 'nodes', 'connections')
        value (float): The data value to add
    """
    ui_state.add_live_feed_data(data_type, value)

def get_live_feed_data():
    """
    Get all live feed data from the UI state.
    
    Returns:
        Dict[str, List[float]]: Dictionary mapping data types to their value lists
    """
    return ui_state.get_live_feed_data()

def clear_live_feed_data():
    """
    Clear all live feed data from the UI state.
    """
    ui_state.clear_live_feed_data()

# Backward compatibility - these will be removed in future refactor
simulation_running = property(get_simulation_running, set_simulation_running)
latest_graph = property(get_latest_graph)
latest_graph_for_ui = property(get_latest_graph_for_ui)
live_feed_data = property(get_live_feed_data)

# Additional backward compatibility properties
def get_sensory_texture_tag():
    return ui_state.sensory_texture_tag

def get_sensory_image_tag():
    return ui_state.sensory_image_tag

def get_graph_h():
    return ui_state.graph_h

def get_graph_w():
    return ui_state.graph_w

def get_system_health():
    return ui_state.get_system_health()

def get_graph_lock():
    return ui_state._lock

sensory_texture_tag = property(get_sensory_texture_tag)
sensory_image_tag = property(get_sensory_image_tag)
graph_h = property(get_graph_h)
graph_w = property(get_graph_w)
system_health = property(get_system_health)
graph_lock = property(get_graph_lock)


# --- SECTION: Sensory Feature Update ---
def update_sensory_features(graph, scale=1.0):
    """
    Capture a new screen, convert to grayscale, and update the sensory node features in-place.
    """
    from screen_graph import capture_screen

    arr = capture_screen(scale=scale)
    h, w = arr.shape
    # Only update if shape matches
    if hasattr(graph, "h") and hasattr(graph, "w") and graph.h == h and graph.w == w:
        flat = arr.flatten().astype(np.float32)
        # Convert numpy array to PyTorch tensor for assignment
        import torch
        flat_tensor = torch.tensor(flat, dtype=torch.float32)
        graph.x[: h * w, 0] = flat_tensor
        # Optionally update node_labels' energy field
        for idx in range(h * w):
            graph.node_labels[idx]["energy"] = float(flat[idx])
    else:
        # If resolution changed, signal for full graph rebuild (not implemented here)
        raise RuntimeError("Screen resolution changed; graph rebuild required.")


# --- SECTION: Sensory Visualization Helper ---
@log_runtime
def update_sensory_visualization(graph):
    log_step("update_sensory_visualization start")
    import time

    t0 = time.perf_counter()
    if not hasattr(graph, "node_labels") or not hasattr(graph, "x"):
        log_step("update_sensory_visualization: missing node_labels or x")
        return
    # Extract sensory node info (assume all sensory nodes are first N nodes)
    sensory_labels = [
        lbl for lbl in graph.node_labels if lbl.get("type", "sensory") == "sensory"
    ]
    if not sensory_labels:
        log_step("update_sensory_visualization: no sensory_labels")
        return
    # Get image dimensions
    h = getattr(graph, "h", None)
    w = getattr(graph, "w", None)
    if h is None or w is None:
        log_step("update_sensory_visualization: missing h or w")
        return
    # Vectorized: reconstruct grayscale image from graph.x
    try:
        arr = graph.x[: h * w].cpu().numpy().reshape((h, w)).astype(np.uint8)
        logging.info(f"[UI] Sensory visualization: extracted {h}x{w} image, range {arr.min()}-{arr.max()}")
    except Exception as e:
        logging.warning(f"[UI] Sensory visualization reshape failed: {e}; using zeros")
        arr = np.zeros((h, w), dtype=np.uint8)
    # Convert grayscale to RGB for DPG (repeat channel)
    arr_rgb = np.stack([arr] * 3, axis=-1)
    arr_f = arr_rgb.astype(np.float32) / 255.0
    arr_f = arr_f.flatten()
    # Convert to list for DearPyGui
    arr_list = arr_f.tolist()
    # Update the dynamic texture
    try:
        dpg.set_value("sensory_texture", arr_list)
        logging.info(f"[UI] Sensory texture updated successfully")
    except Exception as e:
        logging.error(f"[UI] Failed to update sensory texture: {e}")
    t1 = time.perf_counter()
    log_step("update_sensory_visualization end", elapsed_ms=(t1 - t0) * 1000)


# --- SECTION: Performance Logging ---
perf_stats = {
    "sim_updates": 0,
    "ui_updates": 0,
    "last_log_time": time.time(),
    "last_sim_update_time": 0.0,
    "last_ui_update_time": 0.0,
    "fps": 0.0,
    "sim_fps": 0.0,
    "last_report": "",
}
PERF_WINDOW_TAG = "perf_window"
PERF_TEXT_TAG = "perf_text"
LOG_WINDOW_TAG = "log_window"
LOG_TEXT_TAG = "log_text"
FPS_WINDOW_TAG = "fps_window"
FPS_TEXT_TAG = "fps_text"

# --- SECTION: Log Routing to UI ---
log_lines = []
MAX_LOG_LINES = 100

# Remove old logging.basicConfig and custom handlers

# --- SECTION: Logging Control State ---
active_logging = False


# --- SECTION: Runtime Log Button Callback ---
def show_runtime_log_callback():
    """Show runtime log in console."""
    log_lines = get_log_lines()
    if log_lines:
        print("\n=== Runtime Log ===")
        for line in log_lines[-50:]:  # Show last 50 lines
            print(line)
        print("=== End Runtime Log ===\n")
    else:
        print("No runtime logs available")


def toggle_active_logging_callback():
    """Toggle active logging on/off."""
    global active_logging
    active_logging = not active_logging
    status = "ON" if active_logging else "OFF"
    print(f"Active logging: {status}")
    logging.info(f"Active logging toggled: {status}")


def active_logging_handler():
    """Handle active logging updates."""
    try:
        if active_logging and latest_graph is not None:
            # Log current system state
            if hasattr(latest_graph, 'node_labels'):
                active_nodes = sum(1 for node in latest_graph.node_labels if node.get('state') == 'active')
                total_nodes = len(latest_graph.node_labels)
                logging.info(f"System state: {active_nodes}/{total_nodes} nodes active")
        
        # Schedule next update
        dpg.set_frame_callback(dpg.get_frame_count() + 30, active_logging_handler)
    except Exception as e:
        logging.error(f"Error in active_logging_handler: {e}")
        # Still schedule next update even if there's an error
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + 30, active_logging_handler)
        except (AttributeError, RuntimeError, OSError):
            pass
        except Exception:
            # Don't let UI scheduling errors crash the system
            pass


def ui_frame_handler():
    """Per-frame UI updates and performance reporting. Reschedules itself."""
    try:
        # Core live updates
        live_update_callback()

        # Comprehensive performance reporting every second
        now = time.time()
        if now - perf_stats.get("last_log_time", now) >= 1.0:
            # Get UI performance metrics
            fps = perf_stats.get("ui_updates", 0) / max(now - perf_stats["last_log_time"], 1e-6)
            sim_fps = perf_stats.get("sim_updates", 0) / max(now - perf_stats["last_log_time"], 1e-6)
            
            # Get comprehensive system performance metrics
            try:
                from performance_monitor import get_system_performance_metrics
                system_metrics = get_system_performance_metrics()
                
                report = (
                    f"UI FPS: {fps:.2f}\n"
                    f"Sim FPS: {sim_fps:.2f}\n"
                    f"Memory: {system_metrics['memory_usage']:.1f} MB\n"
                    f"CPU: {system_metrics['cpu_usage']:.1f}%\n"
                    f"GPU: {system_metrics['gpu_usage']:.1f}%\n"
                    f"Health: {system_metrics['system_health_score']:.1f}%\n"
                    f"Step Time: {system_metrics['step_time']*1000:.1f} ms\n"
                    f"Error Rate: {system_metrics['error_rate']*100:.2f}%"
                )
            except Exception as e:
                logging.error(f"Failed to get system performance metrics: {e}")
                # Fallback to basic reporting
                report = (
                    f"UI FPS: {fps:.2f}\n"
                    f"Sim FPS: {sim_fps:.2f}\n"
                    f"Last Sim Update: {perf_stats['last_sim_update_time']*1000:.1f} ms\n"
                    f"Last UI Update: {perf_stats['last_ui_update_time']*1000:.1f} ms"
                )
            
            perf_stats["fps"] = fps
            perf_stats["sim_fps"] = sim_fps
            perf_stats["last_report"] = report

            try:
                dpg.set_value(PERF_TEXT_TAG, report)
                dpg.set_value(FPS_TEXT_TAG, f"UI FPS: {fps:.2f}\nSim FPS: {sim_fps:.2f}")
                dpg.set_value("ui_fps_text", f"UI FPS: {fps:.2f}")
                dpg.set_value("sim_fps_text", f"Sim FPS: {sim_fps:.2f}")
                dpg.set_value("last_update_text", f"Last Update: {perf_stats['last_ui_update_time']*1000:.1f} ms")
            except Exception as e:
                logging.warning(f"Could not update performance displays: {e}")

            logging.info("[PERF] " + report.replace("\n", ", "))
            perf_stats["sim_updates"] = 0
            perf_stats["ui_updates"] = 0
            perf_stats["last_log_time"] = now
    except Exception as e:
        logging.error(f"Error in ui_frame_handler: {e}")
    finally:
        # Schedule next frame update
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + 1, ui_frame_handler)
        except (AttributeError, RuntimeError, OSError):
            pass
        except Exception:
            # Don't let UI scheduling errors crash the system
            pass

# --- SECTION: Performance Window ---
@log_runtime
def create_performance_window():
    log_step("create_performance_window start")
    logging.info("[PERF] create_performance_window: Creating performance window")
    with dpg.window(
        label="Performance Stats",
        tag=PERF_WINDOW_TAG,
        width=400,
        height=120,
        pos=(420, 420),
    ):
        dpg.add_text("Performance report will appear here...", tag=PERF_TEXT_TAG)
    log_step("create_performance_window end")


# --- SECTION: Log Window ---
@log_runtime
def create_log_window():
    log_step("create_log_window start")
    logging.info("[PERF] create_log_window: Creating log window")
    with dpg.window(
        label="Log Output", tag=LOG_WINDOW_TAG, width=450, height=250, pos=(0, 220)
    ):
        dpg.add_text("", tag=LOG_TEXT_TAG, wrap=440)
    log_step("create_log_window end")


# --- SECTION: FPS Window ---
@log_runtime
def create_fps_window():
    log_step("create_fps_window start")
    logging.info("[PERF] create_fps_window: Creating FPS window")
    with dpg.window(
        label="FPS Only", tag=FPS_WINDOW_TAG, width=200, height=80, pos=(830, 0)
    ):
        dpg.add_text("", tag=FPS_TEXT_TAG)
    log_step("create_fps_window end")


# --- SECTION: Simulation Loop (Background Thread) ---
@log_runtime
def simulation_loop():
    log_step("simulation_loop start")
    import time

    logging.info("[PERF] simulation_loop: start")
    # Initialize the graph once
    from main_graph import initialize_main_graph
    try:
        main_graph = initialize_main_graph()
    except Exception as e:
        logging.error(f"[SIM] Failed to initialize main graph: {e}")
        set_simulation_running(False)
        log_step("simulation_loop end (init failure)")
        return
    
    # Update graph using thread-safe method
    update_graph(main_graph)
    
    # Update graph dimensions in state manager
    ui_state._lock.acquire()
    try:
        ui_state.graph_h = main_graph.h
        ui_state.graph_w = main_graph.w
    finally:
        ui_state._lock.release()
    # Initialize unified simulation manager
    from simulation_manager import get_simulation_manager
    
    # Get the global simulation manager
    sim_manager = get_simulation_manager()
    sim_manager.set_graph(main_graph)
    
    # Add UI update callback with thread-safe access
    def ui_update_callback(graph, step, perf_stats):
        """Callback to update UI with latest graph state."""
        # Use thread-safe state manager
        ui_state.update_graph(graph)
        
        # Update UI-specific state
        ui_state._lock.acquire()
        try:
            ui_state.sim_update_counter += 1
            if ui_state.sim_update_counter % SENSORY_UI_UPDATE_EVERY_N == 0:
                ui_state.latest_graph_for_ui = graph
                ui_state.update_for_ui = True
        finally:
            ui_state._lock.release()
    
    # Add performance callback
    def perf_callback(metrics):
        """Callback to log performance metrics."""
        logging.info(f"[NETWORK_METRICS] Criticality={metrics['criticality']:.3f}, "
                    f"Connectivity={metrics['connectivity']['density']:.3f}, "
                    f"Energy Variance={metrics['energy_balance']['energy_variance']:.2f}")
    
    # Register callbacks
    sim_manager.add_step_callback(ui_update_callback)
    sim_manager.add_metrics_callback(perf_callback)
    
    # Start simulation
    sim_manager.start_simulation(run_in_thread=True)
    
    # Wait for simulation to run
    while simulation_running:
        time.sleep(0.1)  # Check every 100ms
        
        # Update performance stats from simulation manager
        sim_perf_stats = sim_manager.get_performance_stats()
        with graph_lock:
            perf_stats["sim_updates"] = sim_perf_stats["total_steps"]
            perf_stats["last_sim_update_time"] = sim_perf_stats["last_step_time"]
    
    # Stop simulation when done
    sim_manager.stop_simulation()
    logging.info("[PERF] simulation_loop: end")
    log_step("simulation_loop end")


# --- SECTION: Live Update Handler (UI Render Loop) ---
@log_runtime
def live_update_callback():
    log_step("live_update_callback start")
    import time

    global last_update_time, update_for_ui, perf_stats
    t0 = time.perf_counter()
    logging.info("[PERF] live_update_callback: start")
    now = time.time()
    elapsed = now - last_update_time
    if elapsed < UPDATE_INTERVAL:
        log_step(
            "live_update_callback: not enough time elapsed",
            elapsed_ms=(now - last_update_time) * 1000,
        )
        return
    try:
        with graph_lock:
            if update_for_ui and latest_graph_for_ui is not None:
                t1 = time.perf_counter()
                try:
                    update_sensory_visualization(latest_graph_for_ui)
                except Exception as e:
                    logging.warning(f"Error updating sensory visualization: {e}")
                
                try:
                    update_workspace_visualization(latest_graph_for_ui)
                except Exception as e:
                    logging.warning(f"Error updating workspace visualization: {e}")
                
                try:
                    update_system_status()
                except Exception as e:
                    logging.warning(f"Error updating system status: {e}")
                
                t2 = time.perf_counter()
                perf_stats["ui_updates"] += 1
                perf_stats["last_ui_update_time"] = t2 - t1
                update_for_ui = False
    except Exception as e:
        logging.error(f"Error in live_update_callback: {e}")
    
    last_update_time = now
    t3 = time.perf_counter()
    logging.info(f"[PERF] live_update_callback: end, elapsed {(t3-t0)*1000:.2f} ms")
    log_step("live_update_callback end", elapsed_ms=(t3 - t0) * 1000)


# --- SECTION: Simulation Control Callbacks ---
@log_runtime
def start_simulation_callback():
    log_step("start_simulation_callback start")
    import time

    global simulation_running, sim_update_counter, last_update_time
    logging.info("[PERF] start_simulation_callback: Simulation starting")
    if not simulation_running:
        simulation_running = True
        sim_update_counter = 0
        dpg.set_value("sim_status_text", "Simulation Running")
        Thread(target=simulation_loop, daemon=True).start()
    else:
        logging.info("[UI] Simulation already running; start ignored")
    last_update_time = time.time()
    logging.info(f"[UI] Simulation started at {last_update_time}")
    log_step("start_simulation_callback end")


@log_runtime
def stop_simulation_callback():
    log_step("stop_simulation_callback start")
    logging.info("[PERF] stop_simulation_callback: Simulation stopping")
    global simulation_running
    simulation_running = False
    dpg.set_value("sim_status_text", "Simulation Stopped (Hard)")
    logging.info("[UI] Simulation stopped (hard)")
    # AI: EXTEND HERE - Insert simulation stop/cleanup logic
    log_step("stop_simulation_callback end")


@log_runtime
def reset_simulation_callback():
    log_step("reset_simulation_callback start")
    logging.info("[PERF] reset_simulation_callback: Resetting simulation")
    global latest_graph, simulation_running, sim_update_counter, last_update_time
    from main_graph import initialize_main_graph  # Lazy import
    latest_graph = initialize_main_graph() # Re-initialize the graph
    simulation_running = False
    sim_update_counter = 0
    dpg.set_value("sim_status_text", "Simulation Reset")
    last_update_time = time.time() # Reset last update time
    logging.info("[UI] Simulation reset")
    log_step("reset_simulation_callback end")


# --- SECTION: Runtime Log UI Display Callback ---
def show_runtime_log_in_ui():
    logs = get_log_lines()
    if logs:
        log_text = "\n".join(logs[-100:])
        dpg.set_value("runtime_log_text", log_text)
        print(
            "\n--- Current Runtime Log (UI) ---\n"
            + log_text
            + "\n--- End of Log (UI) ---\n"
        )
    else:
        dpg.set_value("runtime_log_text", "No logs recorded.")


# --- SECTION: Main UI Organization ---
@log_runtime
def create_main_window():
    log_step("create_main_window start")
    logging.info("[PERF] create_main_window: Creating main window")
    
    with dpg.window(
        label="Neural System Control Center",
        tag="main_window",
        width=1200,
        height=800,
        pos=(0, 0),
        no_close=True,
        no_collapse=True,
        no_resize=False
    ):
        # Main Menu Bar
        with dpg.menu_bar():
            # File Menu
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Save Configuration", callback=save_configuration)
                dpg.add_menu_item(label="Load Configuration", callback=load_configuration)
                dpg.add_separator()
                dpg.add_menu_item(label="Export Data", callback=export_data)
                dpg.add_menu_item(label="Exit", callback=exit_application)
            
            # View Menu
            with dpg.menu(label="View"):
                dpg.add_menu_item(label="Show All Panels", callback=show_all_panels)
                dpg.add_menu_item(label="Hide All Panels", callback=hide_all_panels)
                dpg.add_separator()
                dpg.add_menu_item(label="Reset Layout", callback=reset_layout)
            
            # Tools Menu
            with dpg.menu(label="Tools"):
                dpg.add_menu_item(label="Performance Monitor", callback=toggle_performance_panel)
                dpg.add_menu_item(label="Log Viewer", callback=toggle_log_panel)
                dpg.add_menu_item(label="Network Analysis", callback=toggle_network_panel)
            
            # Help Menu
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="Documentation", callback=show_documentation)
                dpg.add_menu_item(label="About", callback=show_about)
        
        # Main Content Area with Tabs
        with dpg.tab_bar():
            # Control Tab
            with dpg.tab(label="Control Center"):
                create_control_panel()
            
            # Live Monitoring Tab
            with dpg.tab(label="Live Monitoring"):
                create_live_monitoring_panel()
            
            # Visualization Tab
            with dpg.tab(label="Visualization"):
                create_visualization_panel()
            
            # Analysis Tab
            with dpg.tab(label="Analysis"):
                create_analysis_panel()
            
            # Workspace Tab
            with dpg.tab(label="Workspace"):
                create_workspace_panel()
            
            # Live Training Tab
            with dpg.tab(label="Live Training"):
                create_live_training_panel()
            
            # Settings Tab
            with dpg.tab(label="Settings"):
                create_settings_panel()
    
    log_step("create_main_window end")


def create_control_panel():
    """Create the main control panel with simulation controls."""
    with dpg.group():
        dpg.add_text("Simulation Control", color=(255, 255, 0))
        dpg.add_separator()
        
        # Simulation Controls
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Start Simulation",
                tag="start_sim_button",
                callback=start_simulation_callback,
                width=120
            )
            dpg.add_button(
                label="Stop Simulation",
                tag="stop_sim_button",
                callback=stop_simulation_callback,
                width=120
            )
            dpg.add_button(
                label="Reset Simulation",
                tag="reset_sim_button",
                callback=reset_simulation_callback,
                width=120
            )
        
        dpg.add_separator()
        
        # Real-time Status Dashboard
        with dpg.group():
            dpg.add_text("Live Status Dashboard", color=(0, 255, 255))
            
            # System Health Indicator
            with dpg.group(horizontal=True):
                dpg.add_text("System Health:")
                dpg.add_text("●", tag="health_indicator", color=(255, 0, 0))
                dpg.add_text("Unknown", tag="health_status_text")
            
            # Energy Flow Indicator
            with dpg.group(horizontal=True):
                dpg.add_text("Energy Flow:")
                dpg.add_text("●", tag="energy_flow_indicator", color=(255, 255, 0))
                dpg.add_text("0.0", tag="energy_flow_text")
            
            # Connection Activity
            with dpg.group(horizontal=True):
                dpg.add_text("Connections:")
                dpg.add_text("●", tag="connection_indicator", color=(0, 255, 0))
                dpg.add_text("0", tag="connection_count_text")
        
        dpg.add_separator()
        
        # Status Information
        with dpg.group():
            dpg.add_text("System Status:", color=(0, 255, 0))
            dpg.add_text("Simulation: Stopped", tag="sim_status_text")
            dpg.add_text("Active Nodes: 0", tag="active_nodes_text")
            dpg.add_text("Total Connections: 0", tag="total_connections_text")
            dpg.add_text("Graph Nodes: 0", tag="graph_nodes_text")
            dpg.add_text("Sensory Nodes: 0", tag="sensory_nodes_text")
            dpg.add_text("Dynamic Nodes: 0", tag="dynamic_nodes_text")
            dpg.add_text("Workspace Nodes: 0", tag="workspace_nodes_text")
        
        dpg.add_separator()
        
        # Quick Actions
        with dpg.group():
            dpg.add_text("Quick Actions:", color=(255, 255, 0))
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Show Runtime Log",
                    callback=show_runtime_log_callback,
                    width=120
                )
                dpg.add_button(
                    label="Toggle Logging",
                    callback=toggle_active_logging_callback,
                    width=120
                )
                dpg.add_button(
                    label="Refresh All",
                    callback=refresh_all_displays,
                    width=120
                )
                dpg.add_button(
                    label="Update Status",
                    callback=update_system_status,
                    width=120
                )


def create_live_monitoring_panel():
    """Create the live monitoring panel with real-time data feeds."""
    with dpg.group():
        dpg.add_text("Live System Monitoring", color=(255, 255, 0))
        dpg.add_separator()
        
        # Real-time Metrics Grid
        with dpg.group():
            dpg.add_text("Real-time Metrics", color=(0, 255, 255))
            
            # Top row - Key metrics
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Total Energy", color=(255, 255, 255))
                    dpg.add_text("0.0", tag="live_total_energy", color=(0, 255, 0))
                
                with dpg.group():
                    dpg.add_text("Active Nodes", color=(255, 255, 255))
                    dpg.add_text("0", tag="live_active_nodes", color=(0, 255, 0))
                
                with dpg.group():
                    dpg.add_text("Connections", color=(255, 255, 255))
                    dpg.add_text("0", tag="live_connections", color=(0, 255, 0))
                
                with dpg.group():
                    dpg.add_text("Birth Rate", color=(255, 255, 255))
                    dpg.add_text("0.0/s", tag="live_birth_rate", color=(0, 255, 0))
            
            # Second row - Performance metrics
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("UI FPS", color=(255, 255, 255))
                    dpg.add_text("0.0", tag="live_ui_fps", color=(0, 255, 0))
                
                with dpg.group():
                    dpg.add_text("Sim FPS", color=(255, 255, 255))
                    dpg.add_text("0.0", tag="live_sim_fps", color=(0, 255, 0))
                
                with dpg.group():
                    dpg.add_text("Update Time", color=(255, 255, 255))
                    dpg.add_text("0.0ms", tag="live_update_time", color=(0, 255, 0))
                
                with dpg.group():
                    dpg.add_text("Memory Usage", color=(255, 255, 255))
                    dpg.add_text("0MB", tag="live_memory", color=(0, 255, 0))
        
        dpg.add_separator()
        
        # Live Data Streams
        with dpg.group():
            dpg.add_text("Live Data Streams", color=(0, 255, 255))
            
            # Energy distribution over time
            with dpg.group():
                dpg.add_text("Energy Distribution Over Time")
                with dpg.plot(
                    label="Energy Timeline",
                    height=150,
                    width=600,
                    tag="live_energy_plot"
                ):
                    with dpg.plot_axis(dpg.mvXAxis, label="Time", tag="live_energy_x"):
                        pass  # X-axis setup
                    with dpg.plot_axis(dpg.mvYAxis, label="Energy", tag="live_energy_y"):
                        dpg.add_line_series([], [], label="Total Energy", tag="live_total_energy_series")
                        dpg.add_line_series([], [], label="Avg Energy", tag="live_avg_energy_series")
            
            # Node activity by type
            with dpg.group():
                dpg.add_text("Node Activity by Type")
                with dpg.plot(
                    label="Node Activity",
                    height=150,
                    width=600,
                    tag="live_activity_plot"
                ):
                    with dpg.plot_axis(dpg.mvXAxis, label="Node Type", tag="live_activity_x"):
                        pass  # X-axis setup
                    with dpg.plot_axis(dpg.mvYAxis, label="Count", tag="live_activity_y"):
                        dpg.add_bar_series([], [], label="Active Nodes", tag="live_active_series")
                        dpg.add_bar_series([], [], label="Total Nodes", tag="live_total_series")
        
        dpg.add_separator()
        
        # System Alerts
        with dpg.group():
            dpg.add_text("System Alerts", color=(255, 100, 100))
            dpg.add_text("", tag="system_alerts", wrap=600)


def create_visualization_panel():
    """Create the visualization panel with sensory and workspace displays."""
    with dpg.group():
        dpg.add_text("System Visualization", color=(255, 255, 0))
        dpg.add_separator()
        
        # Sensory Input Display
        with dpg.group():
            dpg.add_text("Sensory Input (Screen Capture)")
            
            # Create a default texture for sensory visualization
            default_w, default_h = 320, 180  # 16:9 aspect ratio, smaller size
            # Create texture data as a list of floats (0.0-1.0 range)
            arr = [0.0] * (default_h * default_w)  # Simple list of floats for DearPyGui
            with dpg.texture_registry():
                dpg.add_dynamic_texture(default_w, default_h, arr, tag="sensory_texture")
            
            dpg.add_image("sensory_texture", tag="sensory_image", width=320, height=180)
            
            # Sensory controls
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Refresh Sensory",
                    callback=refresh_sensory_display,
                    width=100
                )
                dpg.add_button(
                    label="Toggle Capture",
                    callback=toggle_screen_capture,
                    width=100
                )
        
        dpg.add_separator()
        
        # Workspace Grid Display
        with dpg.group():
            dpg.add_text("Workspace Grid (16x16)")
            
            # Create workspace grid texture
            grid_size = 16
            # Create texture data as a list of floats (0.0-1.0 range)
            default_arr = [0.0] * (grid_size * grid_size)  # Simple list of floats for DearPyGui
            with dpg.texture_registry():
                dpg.add_dynamic_texture(grid_size, grid_size, default_arr, tag="workspace_grid_texture")
            
            dpg.add_image("workspace_grid_texture", tag="workspace_grid_image", width=256, height=256)
            
            # Workspace controls
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Refresh Workspace",
                    callback=refresh_workspace_display,
                    width=100
                )
                dpg.add_button(
                    label="Clear All",
                    callback=clear_workspace_grid,
                    width=100
                )
                dpg.add_button(
                    label="Randomize",
                    callback=randomize_workspace_grid,
                    width=100
                )


def create_analysis_panel():
    """Create the analysis panel with network metrics and system analysis."""
    with dpg.group():
        dpg.add_text("System Analysis", color=(255, 255, 0))
        dpg.add_separator()
        
        # Real-time Data Feeds
        with dpg.group():
            dpg.add_text("Live Data Feeds", color=(0, 255, 255))
            
            # Energy Distribution Chart
            with dpg.group():
                dpg.add_text("Energy Distribution")
                with dpg.plot(
                    label="Energy Over Time",
                    height=200,
                    width=400,
                    tag="energy_plot"
                ):
                    with dpg.plot_axis(dpg.mvXAxis, label="Time (s)", tag="energy_x_axis"):
                        pass  # X-axis setup
                    with dpg.plot_axis(dpg.mvYAxis, label="Energy", tag="energy_y_axis"):
                        dpg.add_line_series([], [], label="Total Energy", tag="total_energy_series")
                        dpg.add_line_series([], [], label="Avg Energy", tag="avg_energy_series")
            
            # Node Activity Heatmap
            with dpg.group():
                dpg.add_text("Node Activity Heatmap")
                with dpg.plot(
                    label="Activity Heatmap",
                    height=200,
                    width=400,
                    tag="activity_plot"
                ):
                    with dpg.plot_axis(dpg.mvXAxis, label="Node Type", tag="activity_x_axis"):
                        pass  # X-axis setup
                    with dpg.plot_axis(dpg.mvYAxis, label="Activity Level", tag="activity_y_axis"):
                        dpg.add_bar_series([], [], label="Active Nodes", tag="active_nodes_series")
        
        dpg.add_separator()
        
        # Network Metrics
        with dpg.group():
            dpg.add_text("Network Metrics")
            dpg.add_text("Criticality: --", tag="criticality_text")
            dpg.add_text("Connectivity: --", tag="connectivity_text")
            dpg.add_text("Energy Balance: --", tag="energy_balance_text")
            
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Calculate Metrics",
                    callback=calculate_network_metrics,
                    width=120
                )
                dpg.add_button(
                    label="Show Health Report",
                    callback=show_network_health_report,
                    width=120
                )
        
        dpg.add_separator()
        
        # System Performance
        with dpg.group():
            dpg.add_text("Performance Monitor")
            dpg.add_text("UI FPS: --", tag="ui_fps_text")
            dpg.add_text("Sim FPS: --", tag="sim_fps_text")
            dpg.add_text("Last Update: --", tag="last_update_text")
            
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Show Performance",
                    callback=show_performance_details,
                    width=120
                )
                dpg.add_button(
                    label="Reset Stats",
                    callback=reset_performance_stats,
                    width=120
                )


def create_workspace_panel():
    """Create the workspace panel with detailed workspace controls."""
    with dpg.group():
        dpg.add_text("Workspace Control", color=(255, 255, 0))
        dpg.add_separator()
        
        # Manual Energy Adjustment
        with dpg.group():
            dpg.add_text("Manual Energy Adjustment")
            
            with dpg.group(horizontal=True):
                dpg.add_input_int(label="Grid X", tag="workspace_x_input", default_value=0, min_value=0, max_value=15, width=80)
                dpg.add_input_int(label="Grid Y", tag="workspace_y_input", default_value=0, min_value=0, max_value=15, width=80)
                dpg.add_input_float(label="Energy (0-255)", tag="workspace_energy_input", default_value=128.0, min_value=0.0, max_value=255.0, width=100)
            
            dpg.add_button(
                label="Set Node Energy",
                tag="set_workspace_energy_button",
                callback=set_workspace_node_energy,
                width=120
            )
        
        dpg.add_separator()
        
        # Workspace Statistics
        with dpg.group():
            dpg.add_text("Workspace Statistics")
            dpg.add_text("Active Nodes: 0/256", tag="workspace_stats_text")
            dpg.add_text("Total Energy: 0.0", tag="workspace_total_energy_text")
            dpg.add_text("Average Energy: 0.0", tag="workspace_avg_energy_text")
        
        dpg.add_separator()
        
        # Pattern Controls
        with dpg.group():
            dpg.add_text("Pattern Controls")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Create Cross Pattern",
                    callback=create_cross_pattern,
                    width=120
                )
                dpg.add_button(
                    label="Create Circle Pattern",
                    callback=create_circle_pattern,
                    width=120
                )
                dpg.add_button(
                    label="Create Random Pattern",
                    callback=create_random_pattern,
                    width=120
                )


def create_settings_panel():
    """Create the settings panel with configuration options."""
    with dpg.group():
        dpg.add_text("Configuration Settings", color=(255, 255, 0))
        dpg.add_separator()
        
        # Simulation Settings
        with dpg.group():
            dpg.add_text("Simulation Parameters")
            
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="Target FPS", tag="target_fps_input", default_value=30.0, width=100)
                dpg.add_input_float(label="Update Interval", tag="update_interval_input", default_value=0.033, width=100)
            
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="Node Energy Cap", tag="node_energy_cap_input", default_value=244.0, width=100)
                dpg.add_input_float(label="Birth Threshold", tag="birth_threshold_input", default_value=200.0, width=100)
                dpg.add_input_float(label="Death Threshold", tag="death_threshold_input", default_value=0.0, width=100)
        
        dpg.add_separator()
        
        # Display Settings
        with dpg.group():
            dpg.add_text("Display Options")
            
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Auto-refresh", tag="auto_refresh_checkbox", default_value=True)
                dpg.add_checkbox(label="Show FPS", tag="show_fps_checkbox", default_value=True)
                dpg.add_checkbox(label="Show Performance", tag="show_performance_checkbox", default_value=False)
        
        dpg.add_separator()
        
        # Save/Load Settings
        with dpg.group():
            dpg.add_text("Configuration Management")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Save Settings",
                    callback=save_current_settings,
                    width=120
                )
                dpg.add_button(
                    label="Load Defaults",
                    callback=load_default_settings,
                    width=120
                )
                dpg.add_button(
                    label="Reset to Defaults",
                    callback=reset_to_defaults,
                    width=120
                )


# --- SECTION: Callback Functions ---
def save_configuration():
    """Save current configuration to file."""
    try:
        from config_manager import get_config
        import json
        import os
        
        # Get current UI state
        ui_state = {
            'panel_visibility': {
                'performance': dpg.is_item_visible(PERF_WINDOW_TAG),
                'log': dpg.is_item_visible(LOG_WINDOW_TAG),
                'fps': dpg.is_item_visible(FPS_WINDOW_TAG)
            },
            'window_positions': {
                'main': dpg.get_item_pos("main_window"),
                'performance': dpg.get_item_pos(PERF_WINDOW_TAG),
                'log': dpg.get_item_pos(LOG_WINDOW_TAG),
                'fps': dpg.get_item_pos(FPS_WINDOW_TAG)
            },
            'simulation_settings': {
                'target_fps': dpg.get_value("target_fps_input") if dpg.does_item_exist("target_fps_input") else 30.0,
                'update_interval': dpg.get_value("update_interval_input") if dpg.does_item_exist("update_interval_input") else 0.033,
                'auto_refresh': dpg.get_value("auto_refresh_checkbox") if dpg.does_item_exist("auto_refresh_checkbox") else True
            }
        }
        
        # Save to config file with security validation
        config_path = "ui_config.json"
        
        # Validate configuration data before saving
        from input_validation import validate_dict
        config_result = validate_dict(ui_state, max_keys=50)
        if not config_result.is_valid:
            logging.error(f"Invalid configuration data: {config_result.errors}")
            raise ValueError("Configuration data validation failed")
        
        # Set secure file permissions
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_result.sanitized_value, f, indent=2, ensure_ascii=False)
        
        # Set secure file permissions (readable only by owner)
        os.chmod(config_path, 0o600)
        
        logging.info(f"UI configuration saved to {config_path}")
        print(f"Configuration saved successfully to {config_path}")
        
    except Exception as e:
        logging.error(f"Failed to save configuration: {e}")
        log_step("Configuration save failed", error=str(e))


def load_configuration():
    """Load configuration from file with security validation."""
    try:
        import json
        import os
        from input_validation import validate_file_path, validate_json
        
        config_path = "ui_config.json"
        
        # Validate file path for security
        path_result = validate_file_path(config_path, allowed_extensions=['.json'], max_size=1024*1024)  # 1MB max
        if not path_result.is_valid:
            logging.error(f"Invalid configuration file path: {path_result.errors}")
            print("Configuration file path validation failed. Using defaults.")
            return
        
        if not os.path.exists(config_path):
            print("No configuration file found. Using defaults.")
            return
        
        # Read and validate JSON content
        with open(config_path, 'r', encoding='utf-8') as f:
            json_content = f.read()
        
        # Validate JSON content for security
        json_result = validate_json(json_content, max_depth=5)
        if not json_result.is_valid:
            logging.error(f"Invalid JSON configuration: {json_result.errors}")
            print("Configuration file contains invalid JSON. Using defaults.")
            return
        
        ui_state = json_result.sanitized_value
        
        # Restore panel visibility
        if 'panel_visibility' in ui_state:
            for panel, visible in ui_state['panel_visibility'].items():
                if panel == 'performance' and dpg.does_item_exist(PERF_WINDOW_TAG):
                    dpg.show_item(PERF_WINDOW_TAG) if visible else dpg.hide_item(PERF_WINDOW_TAG)
                elif panel == 'log' and dpg.does_item_exist(LOG_WINDOW_TAG):
                    dpg.show_item(LOG_WINDOW_TAG) if visible else dpg.hide_item(LOG_WINDOW_TAG)
                elif panel == 'fps' and dpg.does_item_exist(FPS_WINDOW_TAG):
                    dpg.show_item(FPS_WINDOW_TAG) if visible else dpg.hide_item(FPS_WINDOW_TAG)
        
        # Restore window positions
        if 'window_positions' in ui_state:
            for window, pos in ui_state['window_positions'].items():
                if dpg.does_item_exist(window):
                    dpg.set_item_pos(window, pos)
        
        # Restore simulation settings
        if 'simulation_settings' in ui_state:
            settings = ui_state['simulation_settings']
            if 'target_fps' in settings and dpg.does_item_exist("target_fps_input"):
                dpg.set_value("target_fps_input", settings['target_fps'])
            if 'update_interval' in settings and dpg.does_item_exist("update_interval_input"):
                dpg.set_value("update_interval_input", settings['update_interval'])
            if 'auto_refresh' in settings and dpg.does_item_exist("auto_refresh_checkbox"):
                dpg.set_value("auto_refresh_checkbox", settings['auto_refresh'])
        
        logging.info("UI configuration loaded successfully")
        print("Configuration loaded successfully")
        
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        log_step("Configuration load failed", error=str(e))


def export_data():
    """Export current system data."""
    try:
        import json
        import time
        from datetime import datetime
        
        # Collect system data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'system_state': {
                'simulation_running': simulation_running,
                'sim_update_counter': sim_update_counter,
                'graph_info': {
                    'num_nodes': len(latest_graph.node_labels) if latest_graph else 0,
                    'num_edges': latest_graph.edge_index.shape[1] if latest_graph and hasattr(latest_graph, 'edge_index') else 0,
                    'graph_h': graph_h,
                    'graph_w': graph_w
                }
            },
            'performance_stats': perf_stats.copy(),
            'node_statistics': get_node_statistics(),
            'network_metrics': get_network_metrics_data()
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_export_{timestamp}.json"
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logging.info(f"System data exported to {filename}")
        print(f"System data exported successfully to {filename}")
        
    except Exception as e:
        logging.error(f"Failed to export data: {e}")
        log_step("Data export failed", error=str(e))


def get_node_statistics():
    """Get current node statistics for export."""
    if not latest_graph or not hasattr(latest_graph, 'node_labels'):
        return {}
    
    stats = {
        'total_nodes': len(latest_graph.node_labels),
        'node_types': {},
        'node_states': {},
        'energy_distribution': {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'total': 0.0
        }
    }
    
    energies = []
    for node in latest_graph.node_labels:
        node_type = node.get('type', 'unknown')
        node_state = node.get('state', 'unknown')
        energy = node.get('energy', 0.0)
        
        stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        stats['node_states'][node_state] = stats['node_states'].get(node_state, 0) + 1
        energies.append(energy)
    
    if energies:
        stats['energy_distribution'] = {
            'min': min(energies),
            'max': max(energies),
            'mean': sum(energies) / len(energies),
            'total': sum(energies)
        }
    
    return stats


def get_network_metrics_data():
    """Get current network metrics for export."""
    if not latest_graph or not hasattr(latest_graph, 'network_metrics'):
        return {}
    
    try:
        metrics = latest_graph.network_metrics.calculate_comprehensive_metrics(latest_graph)
        return {
            'criticality': metrics.get('criticality', 0.0),
            'connectivity': metrics.get('connectivity', {}),
            'energy_balance': metrics.get('energy_balance', {}),
            'performance': metrics.get('performance', {})
        }
    except Exception as e:
        logging.warning(f"Could not get network metrics: {e}")
        return {}


def exit_application():
    """Exit the application."""
    dpg.stop_dearpygui()


def show_all_panels():
    """Show all analysis panels."""
    try:
        panels = [
            (PERF_WINDOW_TAG, "Performance Stats"),
            (LOG_WINDOW_TAG, "Log Output"),
            (FPS_WINDOW_TAG, "FPS Only")
        ]
        
        for panel_tag, panel_name in panels:
            if dpg.does_item_exist(panel_tag):
                dpg.show_item(panel_tag)
                logging.info(f"Showing panel: {panel_name}")
        
        print("All panels shown successfully")
        
    except Exception as e:
        logging.error(f"Error showing panels: {e}")
        log_step("Panel show failed", error=str(e))


def hide_all_panels():
    """Hide all analysis panels."""
    try:
        panels = [
            (PERF_WINDOW_TAG, "Performance Stats"),
            (LOG_WINDOW_TAG, "Log Output"),
            (FPS_WINDOW_TAG, "FPS Only")
        ]
        
        for panel_tag, panel_name in panels:
            if dpg.does_item_exist(panel_tag):
                dpg.hide_item(panel_tag)
                logging.info(f"Hiding panel: {panel_name}")
        
        print("All panels hidden successfully")
        
    except Exception as e:
        logging.error(f"Error hiding panels: {e}")
        log_step("Panel hide failed", error=str(e))


def reset_layout():
    """Reset the UI layout to default."""
    try:
        # Reset window positions to defaults
        default_positions = {
            "main_window": (0, 0),
            PERF_WINDOW_TAG: (420, 420),
            LOG_WINDOW_TAG: (0, 220),
            FPS_WINDOW_TAG: (830, 0)
        }
        
        for window_tag, default_pos in default_positions.items():
            if dpg.does_item_exist(window_tag):
                dpg.set_item_pos(window_tag, default_pos)
        
        # Show all panels
        show_all_panels()
        
        # Reset simulation settings to defaults
        if dpg.does_item_exist("target_fps_input"):
            dpg.set_value("target_fps_input", 30.0)
        if dpg.does_item_exist("update_interval_input"):
            dpg.set_value("update_interval_input", 0.033)
        if dpg.does_item_exist("auto_refresh_checkbox"):
            dpg.set_value("auto_refresh_checkbox", True)
        
        logging.info("UI layout reset to defaults")
        print("Layout reset to defaults successfully")
        
    except Exception as e:
        logging.error(f"Error resetting layout: {e}")
        log_step("Layout reset failed", error=str(e))


def toggle_performance_panel():
    """Toggle performance monitoring panel."""
    try:
        if dpg.does_item_exist(PERF_WINDOW_TAG):
            current_visible = dpg.is_item_visible(PERF_WINDOW_TAG)
            if current_visible:
                dpg.hide_item(PERF_WINDOW_TAG)
                print("Performance panel hidden")
            else:
                dpg.show_item(PERF_WINDOW_TAG)
                print("Performance panel shown")
        else:
            print("Performance panel not found")
            
    except Exception as e:
        logging.error(f"Error toggling performance panel: {e}")
        log_step("Performance panel toggle failed", error=str(e))


def toggle_log_panel():
    """Toggle log viewing panel."""
    try:
        if dpg.does_item_exist(LOG_WINDOW_TAG):
            current_visible = dpg.is_item_visible(LOG_WINDOW_TAG)
            if current_visible:
                dpg.hide_item(LOG_WINDOW_TAG)
                print("Log panel hidden")
            else:
                dpg.show_item(LOG_WINDOW_TAG)
                print("Log panel shown")
        else:
            print("Log panel not found")
            
    except Exception as e:
        logging.error(f"Error toggling log panel: {e}")
        log_step("Log panel toggle failed", error=str(e))


def toggle_network_panel():
    """Toggle network analysis panel."""
    try:
        # For now, toggle the performance panel as it contains network metrics
        # In a full implementation, this would be a dedicated network analysis panel
        if dpg.does_item_exist(PERF_WINDOW_TAG):
            current_visible = dpg.is_item_visible(PERF_WINDOW_TAG)
            if current_visible:
                dpg.hide_item(PERF_WINDOW_TAG)
                print("Network analysis panel hidden")
            else:
                dpg.show_item(PERF_WINDOW_TAG)
                print("Network analysis panel shown")
        else:
            print("Network analysis panel not found")
            
    except Exception as e:
        logging.error(f"Error toggling network panel: {e}")
        log_step("Network panel toggle failed", error=str(e))


def show_documentation():
    """Show system documentation."""
    with dpg.window(label="System Documentation", modal=True, width=800, height=600, tag="doc_window"):
        dpg.add_text("AI Neural System Documentation", bullet=True)
        dpg.add_separator()
        
        with dpg.collapsing_header(label="System Overview", default_open=True):
            dpg.add_text("This is an advanced energy-based neural system with the following features:")
            dpg.add_text("• Sensory input processing from screen capture")
            dpg.add_text("• Dynamic node behaviors (oscillator, integrator, relay, highway)")
            dpg.add_text("• Learning and memory formation")
            dpg.add_text("• Homeostatic regulation and criticality optimization")
            dpg.add_text("• Workspace nodes for imagination and synthesis")
            dpg.add_text("• Real-time network metrics and health monitoring")
        
        with dpg.collapsing_header(label="Node Types"):
            dpg.add_text("Sensory Nodes: Process visual input from screen capture")
            dpg.add_text("Dynamic Nodes: Basic processing units with energy dynamics")
            dpg.add_text("Oscillator Nodes: Emit periodic energy pulses")
            dpg.add_text("Integrator Nodes: Accumulate energy and activate at thresholds")
            dpg.add_text("Relay Nodes: Transfer energy with amplification")
            dpg.add_text("Highway Nodes: High-capacity energy distribution")
            dpg.add_text("Workspace Nodes: Internal workspace for imagination")
        
        with dpg.collapsing_header(label="Controls"):
            dpg.add_text("Start/Stop: Control simulation execution")
            dpg.add_text("Reset: Reset system to initial state")
            dpg.add_text("Export: Save current system state")
            dpg.add_text("Settings: Configure system parameters")
        
        with dpg.collapsing_header(label="Visualization"):
            dpg.add_text("Sensory Display: Real-time visual input processing")
            dpg.add_text("Workspace Grid: 16x16 imagination workspace")
            dpg.add_text("Network Metrics: Criticality, connectivity, energy balance")
            dpg.add_text("Performance: Real-time system performance monitoring")
        
        dpg.add_separator()
        dpg.add_button(label="Close", callback=lambda: dpg.delete_item("doc_window"))


def show_about():
    """Show about information."""
    with dpg.window(label="About AI Neural System", modal=True, width=600, height=400, tag="about_window"):
        dpg.add_text("AI Neural System v1.0", bullet=True)
        dpg.add_separator()
        
        dpg.add_text("Advanced Energy-Based Neural Network")
        dpg.add_text("Built with PyTorch Geometric and DearPyGui")
        dpg.add_separator()
        
        with dpg.collapsing_header(label="Features", default_open=True):
            dpg.add_text("✓ Real-time sensory processing")
            dpg.add_text("✓ Advanced node behaviors")
            dpg.add_text("✓ Learning and memory formation")
            dpg.add_text("✓ Homeostatic regulation")
            dpg.add_text("✓ Network criticality optimization")
            dpg.add_text("✓ Workspace imagination system")
            dpg.add_text("✓ Comprehensive metrics and monitoring")
        
        with dpg.collapsing_header(label="Technical Details"):
            dpg.add_text("Framework: PyTorch Geometric")
            dpg.add_text("UI: DearPyGui")
            dpg.add_text("Architecture: Energy-based neural dynamics")
            dpg.add_text("Learning: STDP-like plasticity")
            dpg.add_text("Memory: Pattern-based consolidation")
            dpg.add_text("Regulation: Homeostatic control")
        
        with dpg.collapsing_header(label="System Status"):
            if latest_graph:
                dpg.add_text(f"Nodes: {len(latest_graph.node_labels)}")
                dpg.add_text(f"Connections: {latest_graph.edge_index.shape[1] if hasattr(latest_graph, 'edge_index') else 0}")
                dpg.add_text(f"Simulation: {'Running' if simulation_running else 'Stopped'}")
            else:
                dpg.add_text("System not initialized")
        
        dpg.add_separator()
        dpg.add_text("© 2024 AI Neural System Project")
        dpg.add_button(label="Close", callback=lambda: dpg.delete_item("about_window"))


def refresh_sensory_display():
    """Refresh the sensory input display."""
    if latest_graph is not None:
        update_sensory_visualization(latest_graph)
        print("Sensory display refreshed")


def toggle_screen_capture():
    """Toggle automatic screen capture."""
    global simulation_running
    
    if simulation_running:
        # Stop simulation
        simulation_running = False
        logging.info("Screen capture and simulation stopped")
        print("Screen capture and simulation stopped")
        
        # Update UI button text
        if dpg.does_item_exist("start_simulation_btn"):
            dpg.set_item_label("start_simulation_btn", "Start Simulation")
    else:
        # Start simulation
        simulation_running = True
        logging.info("Screen capture and simulation started")
        print("Screen capture and simulation started")
        
        # Update UI button text
        if dpg.does_item_exist("start_simulation_btn"):
            dpg.set_item_label("start_simulation_btn", "Stop Simulation")
        
        # Start simulation thread if not already running
        import threading
        if not hasattr(toggle_screen_capture, 'sim_thread') or not toggle_screen_capture.sim_thread.is_alive():
            toggle_screen_capture.sim_thread = threading.Thread(target=simulation_loop, daemon=True)
            toggle_screen_capture.sim_thread.start()


def calculate_network_metrics():
    """Calculate and display network metrics."""
    if latest_graph is not None and hasattr(latest_graph, 'network_metrics'):
        metrics = latest_graph.network_metrics.calculate_comprehensive_metrics()
        
        # Update display
        dpg.set_value("criticality_text", f"Criticality: {metrics['criticality']['branching_ratio']:.3f}")
        dpg.set_value("connectivity_text", f"Connectivity: {metrics['connectivity']['density']:.3f}")
        dpg.set_value("energy_balance_text", f"Energy Balance: {metrics['energy_balance']['total_energy']:.1f}")
        
        print("Network metrics calculated and displayed")
    else:
        print("Network metrics not available")


def show_network_health_report():
    """Show network health report."""
    try:
        if latest_graph is not None and hasattr(latest_graph, 'network_metrics'):
            # Get comprehensive network health data
            health_score = latest_graph.network_metrics.get_network_health_score()
            metrics = latest_graph.network_metrics.calculate_comprehensive_metrics(latest_graph)
            trends = latest_graph.network_metrics.get_metrics_trends()
            
            # Generate detailed health report
            report_lines = [
                "=== NETWORK HEALTH REPORT ===",
                f"Overall Health Score: {health_score['score']:.1f}/100 ({health_score['status']})",
                "",
                "=== CRITICALITY ANALYSIS ===",
                f"Branching Ratio: {health_score['criticality']:.3f}",
                f"Status: {'Optimal' if 0.8 <= health_score['criticality'] <= 1.2 else 'Suboptimal'}",
                "",
                "=== CONNECTIVITY ANALYSIS ===",
                f"Network Density: {health_score['density']:.3f}",
                f"Status: {'Optimal' if 0.1 <= health_score['density'] <= 0.5 else 'Suboptimal'}",
                "",
                "=== ENERGY BALANCE ===",
                f"Energy Variance: {health_score['energy_variance']:.2f}",
                f"Status: {'Stable' if health_score['energy_variance'] < 100 else 'Unstable'}",
                "",
                "=== PERFORMANCE ===",
                f"Calculation Time: {health_score['calculation_time']*1000:.2f} ms",
                f"Status: {'Fast' if health_score['calculation_time'] < 0.001 else 'Slow'}",
                ""
            ]
            
            # Add recommendations
            if health_score['recommendations']:
                report_lines.extend([
                    "=== RECOMMENDATIONS ===",
                    *[f"- {rec}" for rec in health_score['recommendations']],
                    ""
                ])
            
            # Add trend analysis
            if trends:
                report_lines.extend([
                    "=== TREND ANALYSIS ===",
                    f"Criticality Trend: {trends.get('criticality_trend', [0])[-1]:.3f}",
                    f"Density Trend: {trends.get('density_trend', [0])[-1]:.3f}",
                    f"Energy Variance Trend: {trends.get('energy_variance_trend', [0])[-1]:.2f}",
                    ""
                ])
            
            report_lines.append("=== END OF REPORT ===")
            
            # Display report
            report_text = "\n".join(report_lines)
            print(report_text)
            
            # Also log the report
            logging.info("Network Health Report Generated")
            for line in report_lines:
                logging.info(line)
            
        else:
            print("Network metrics not available. Start simulation first.")
            
    except Exception as e:
        logging.error(f"Error generating network health report: {e}")
        log_step("Network health report generation failed", error=str(e))


def show_performance_details():
    """Show detailed performance information."""
    try:
        # Get current performance statistics
        current_perf = perf_stats.copy()
        
        # Get system information
        system_info = {
            'simulation_running': simulation_running,
            'sim_update_counter': sim_update_counter,
            'graph_nodes': len(latest_graph.node_labels) if latest_graph else 0,
            'graph_edges': latest_graph.edge_index.shape[1] if latest_graph and hasattr(latest_graph, 'edge_index') else 0
        }
        
        # Generate detailed performance report
        report_lines = [
            "=== PERFORMANCE DETAILS ===",
            "",
            "=== SIMULATION PERFORMANCE ===",
            f"Simulation Status: {'Running' if simulation_running else 'Stopped'}",
            f"Simulation Updates: {system_info['sim_update_counter']}",
            f"Simulation FPS: {current_perf.get('sim_fps', 0):.2f}",
            f"Last Sim Update Time: {current_perf.get('last_sim_update_time', 0)*1000:.2f} ms",
            "",
            "=== UI PERFORMANCE ===",
            f"UI FPS: {current_perf.get('fps', 0):.2f}",
            f"UI Updates: {current_perf.get('ui_updates', 0)}",
            f"Last UI Update Time: {current_perf.get('last_ui_update_time', 0)*1000:.2f} ms",
            "",
            "=== SYSTEM RESOURCES ===",
            f"Total Nodes: {system_info['graph_nodes']}",
            f"Total Edges: {system_info['graph_edges']}",
            f"Graph Dimensions: {graph_h}x{graph_w}" if graph_h and graph_w else "Graph Dimensions: Unknown",
            "",
            "=== PERFORMANCE TARGETS ===",
            f"Target FPS: {TARGET_FPS}",
            f"Update Interval: {UPDATE_INTERVAL*1000:.1f} ms",
            f"Sensory UI Update Every: {SENSORY_UI_UPDATE_EVERY_N} frames",
            ""
        ]
        
        # Add performance analysis
        ui_fps = current_perf.get('fps', 0)
        sim_fps = current_perf.get('sim_fps', 0)
        
        if ui_fps < TARGET_FPS * 0.8:
            report_lines.append("⚠️  UI Performance: Below target (consider reducing update frequency)")
        elif ui_fps > TARGET_FPS * 1.2:
            report_lines.append("✅ UI Performance: Above target (good)")
        else:
            report_lines.append("✅ UI Performance: On target")
        
        if sim_fps < TARGET_FPS * 0.8:
            report_lines.append("⚠️  Simulation Performance: Below target (consider optimization)")
        elif sim_fps > TARGET_FPS * 1.2:
            report_lines.append("✅ Simulation Performance: Above target (good)")
        else:
            report_lines.append("✅ Simulation Performance: On target")
        
        report_lines.extend([
            "",
            "=== END OF PERFORMANCE REPORT ==="
        ])
        
        # Display report
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Also log the report
        logging.info("Performance Details Report Generated")
        for line in report_lines:
            logging.info(line)
            
    except Exception as e:
        logging.error(f"Error generating performance details: {e}")
        log_step("Performance details generation failed", error=str(e))


def reset_performance_stats():
    """Reset performance statistics."""
    global perf_stats
    perf_stats = {
        "sim_updates": 0,
        "ui_updates": 0,
        "fps": 0.0,
        "sim_fps": 0.0,
        "last_sim_update_time": 0.0,
        "last_ui_update_time": 0.0,
        "last_log_time": time.time(),
        "last_report": ""
    }
    print("Performance statistics reset")


def create_cross_pattern():
    """Create a cross pattern in the workspace."""
    if latest_graph is not None:
        # Create a simple cross pattern
        cross_positions = [
            (7, 0, 255), (7, 1, 255), (7, 2, 255),  # Vertical line
            (0, 7, 255), (1, 7, 255), (2, 7, 255),  # Horizontal line
            (7, 7, 255)  # Center intersection
        ]
        
        for x, y, energy in cross_positions:
            for node_label in latest_graph.node_labels:
                if (node_label.get('type') == 'workspace' and 
                    node_label.get('x') == x and 
                    node_label.get('y') == y):
                    node_label['energy'] = energy
                    node_label['membrane_potential'] = energy / 255.0
                    node_label['state'] = 'active'
                    break
        
        refresh_workspace_display()
        print("Cross pattern created")


def create_circle_pattern():
    """Create a circle pattern in the workspace."""
    if latest_graph is not None:
        import math
        
        # Create a simple circle pattern
        center_x, center_y = 8, 8
        radius = 6
        
        for node_label in latest_graph.node_labels:
            if node_label.get('type') == 'workspace':
                x, y = node_label.get('x'), node_label.get('y')
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                if abs(distance - radius) <= 1.0:  # Circle with some thickness
                    energy = 255.0
                    node_label['energy'] = energy
                    node_label['membrane_potential'] = energy / 255.0
                    node_label['state'] = 'active'
        
        refresh_workspace_display()
        print("Circle pattern created")


def create_random_pattern():
    """Create a random pattern in the workspace."""
    if latest_graph is not None:
        import random
        
        for node_label in latest_graph.node_labels:
            if node_label.get('type') == 'workspace':
                if random.random() < 0.3:  # 30% chance of activation
                    energy = random.uniform(50.0, 255.0)
                    node_label['energy'] = energy
                    node_label['membrane_potential'] = energy / 255.0
                    node_label['state'] = 'active'
                else:
                    node_label['energy'] = 0.0
                    node_label['membrane_potential'] = 0.0
                    node_label['state'] = 'inactive'
        
        refresh_workspace_display()
        print("Random pattern created")


def save_current_settings():
    """Save current UI settings."""
    try:
        # This is the same as save_configuration, but with a different name for clarity
        save_configuration()
        print("Current settings saved successfully")
        
    except Exception as e:
        logging.error(f"Error saving current settings: {e}")
        log_step("Settings save failed", error=str(e))


def load_default_settings():
    """Load default settings."""
    try:
        # Reset all settings to their default values
        default_settings = {
            'target_fps': 30.0,
            'update_interval': 0.033,
            'node_energy_cap': 244.0,
            'birth_threshold': 200.0,
            'death_threshold': 0.0,
            'auto_refresh': True,
            'show_fps': True,
            'show_performance': False
        }
        
        # Apply default settings to UI elements
        for setting, value in default_settings.items():
            if setting == 'target_fps' and dpg.does_item_exist("target_fps_input"):
                dpg.set_value("target_fps_input", value)
            elif setting == 'update_interval' and dpg.does_item_exist("update_interval_input"):
                dpg.set_value("update_interval_input", value)
            elif setting == 'node_energy_cap' and dpg.does_item_exist("node_energy_cap_input"):
                dpg.set_value("node_energy_cap_input", value)
            elif setting == 'birth_threshold' and dpg.does_item_exist("birth_threshold_input"):
                dpg.set_value("birth_threshold_input", value)
            elif setting == 'death_threshold' and dpg.does_item_exist("death_threshold_input"):
                dpg.set_value("death_threshold_input", value)
            elif setting == 'auto_refresh' and dpg.does_item_exist("auto_refresh_checkbox"):
                dpg.set_value("auto_refresh_checkbox", value)
            elif setting == 'show_fps' and dpg.does_item_exist("show_fps_checkbox"):
                dpg.set_value("show_fps_checkbox", value)
            elif setting == 'show_performance' and dpg.does_item_exist("show_performance_checkbox"):
                dpg.set_value("show_performance_checkbox", value)
        
        # Reset layout to defaults
        reset_layout()
        
        logging.info("Default settings loaded successfully")
        print("Default settings loaded successfully")
        
    except Exception as e:
        logging.error(f"Error loading default settings: {e}")
        log_step("Default settings load failed", error=str(e))


def reset_to_defaults():
    """Reset all settings to defaults."""
    try:
        # Load default settings
        load_default_settings()
        
        # Reset performance statistics
        reset_performance_stats()
        
        # Reset simulation state
        global simulation_running, sim_update_counter
        simulation_running = False
        sim_update_counter = 0
        
        # Update UI status
        dpg.set_value("sim_status_text", "Simulation Stopped")
        
        logging.info("All settings reset to defaults")
        print("All settings reset to defaults successfully")
        
    except Exception as e:
        logging.error(f"Error resetting to defaults: {e}")
        log_step("Reset to defaults failed", error=str(e))


def update_system_status():
    """Update the system status display with current graph information."""
    try:
        if latest_graph is not None and hasattr(latest_graph, 'node_labels'):
            # Count node types
            node_types = {}
            active_nodes = 0
            total_energy = 0.0
            
            for node in latest_graph.node_labels:
                node_type = node.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
                
                energy = node.get('energy', 0.0)
                total_energy += energy
                if energy > 0:
                    active_nodes += 1
            
            # Update status displays
            dpg.set_value("graph_nodes_text", f"Graph Nodes: {len(latest_graph.node_labels)}")
            dpg.set_value("sensory_nodes_text", f"Sensory Nodes: {node_types.get('sensory', 0)}")
            dpg.set_value("dynamic_nodes_text", f"Dynamic Nodes: {node_types.get('dynamic', 0)}")
            dpg.set_value("workspace_nodes_text", f"Workspace Nodes: {node_types.get('workspace', 0)}")
            dpg.set_value("active_nodes_text", f"Active Nodes: {active_nodes}")
            dpg.set_value("total_connections_text", f"Total Connections: {latest_graph.edge_index.shape[1] if hasattr(latest_graph, 'edge_index') else 0}")
            
            # Update live feed data
            update_live_feeds(total_energy, active_nodes, node_types)
            
            logging.info(f"[UI] Status updated: {len(latest_graph.node_labels)} nodes, {active_nodes} active, {total_energy:.1f} total energy")
        else:
            dpg.set_value("graph_nodes_text", "Graph Nodes: No graph")
            dpg.set_value("sensory_nodes_text", "Sensory Nodes: No graph")
            dpg.set_value("dynamic_nodes_text", "Dynamic Nodes: No graph")
            dpg.set_value("workspace_nodes_text", "Workspace Nodes: No graph")
            dpg.set_value("active_nodes_text", "Active Nodes: No graph")
            dpg.set_value("total_connections_text", "Total Connections: No graph")
    except Exception as e:
        logging.error(f"Error updating system status: {e}")


def update_live_feeds(total_energy, active_nodes, node_types):
    """Update live feed data and visualizations."""
    try:
        current_time = time.time()
        
        # Update live feed data
        live_feed_data["energy_history"].append(total_energy)
        live_feed_data["node_activity_history"].append(active_nodes)
        live_feed_data["time_history"].append(current_time)
        
        # Calculate average energy
        avg_energy = total_energy / max(len(latest_graph.node_labels), 1) if latest_graph else 0
        live_feed_data["performance_history"].append(avg_energy)
        
        # Update connection count
        connection_count = latest_graph.edge_index.shape[1] if latest_graph and hasattr(latest_graph, 'edge_index') else 0
        live_feed_data["connection_history"].append(connection_count)
        
        # Calculate birth rate (simplified)
        birth_rate = len([n for n in latest_graph.node_labels if n.get('type') == 'dynamic']) / max(current_time - live_feed_data["time_history"][0] if live_feed_data["time_history"] else 1, 1)
        live_feed_data["birth_rate_history"].append(birth_rate)
        
        # Trim history to max length
        for key in live_feed_data:
            if isinstance(live_feed_data[key], list) and len(live_feed_data[key]) > live_feed_data["max_history_length"]:
                live_feed_data[key] = live_feed_data[key][-live_feed_data["max_history_length"]:]
        
        # Update live monitoring displays
        update_live_monitoring_displays()
        
        # Update system health
        update_system_health(total_energy, active_nodes, connection_count)
        
    except Exception as e:
        logging.error(f"Error updating live feeds: {e}")


def update_live_monitoring_displays():
    """Update the live monitoring panel displays."""
    try:
        if not live_feed_data["time_history"]:
            return
        
        # Update live metrics
        current_energy = live_feed_data["energy_history"][-1] if live_feed_data["energy_history"] else 0
        current_active = live_feed_data["node_activity_history"][-1] if live_feed_data["node_activity_history"] else 0
        current_connections = live_feed_data["connection_history"][-1] if live_feed_data["connection_history"] else 0
        current_birth_rate = live_feed_data["birth_rate_history"][-1] if live_feed_data["birth_rate_history"] else 0
        
        # Update live metric displays
        dpg.set_value("live_total_energy", f"{current_energy:.1f}")
        dpg.set_value("live_active_nodes", f"{current_active}")
        dpg.set_value("live_connections", f"{current_connections}")
        dpg.set_value("live_birth_rate", f"{current_birth_rate:.2f}/s")
        
        # Update performance metrics
        dpg.set_value("live_ui_fps", f"{perf_stats.get('fps', 0):.1f}")
        dpg.set_value("live_sim_fps", f"{perf_stats.get('sim_fps', 0):.1f}")
        dpg.set_value("live_update_time", f"{perf_stats.get('last_ui_update_time', 0)*1000:.1f}ms")
        
        # Update plots
        update_live_plots()
        
    except Exception as e:
        logging.error(f"Error updating live monitoring displays: {e}")


def update_live_plots():
    """Update the live monitoring plots."""
    try:
        if len(live_feed_data["time_history"]) < 2:
            return
        
        # Prepare time series data (relative to start)
        start_time = live_feed_data["time_history"][0]
        time_series = [(t - start_time) for t in live_feed_data["time_history"]]
        
        # Update energy plot
        if live_feed_data["energy_history"] and live_feed_data["performance_history"]:
            dpg.set_value("live_total_energy_series", [time_series, live_feed_data["energy_history"]])
            dpg.set_value("live_avg_energy_series", [time_series, live_feed_data["performance_history"]])
        
        # Update activity plot
        if latest_graph and hasattr(latest_graph, 'node_labels'):
            node_types = {}
            active_by_type = {}
            
            for node in latest_graph.node_labels:
                node_type = node.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
                if node.get('energy', 0) > 0:
                    active_by_type[node_type] = active_by_type.get(node_type, 0) + 1
            
            # Prepare bar chart data
            types = list(node_types.keys())
            active_counts = [active_by_type.get(t, 0) for t in types]
            total_counts = [node_types.get(t, 0) for t in types]
            
            dpg.set_value("live_active_series", [types, active_counts])
            dpg.set_value("live_total_series", [types, total_counts])
        
    except Exception as e:
        logging.error(f"Error updating live plots: {e}")


def update_system_health(total_energy, active_nodes, connection_count):
    """Update system health indicators and alerts."""
    try:
        current_time = time.time()
        
        # Calculate energy flow rate
        if len(live_feed_data["energy_history"]) > 1:
            energy_diff = live_feed_data["energy_history"][-1] - live_feed_data["energy_history"][-2]
            time_diff = live_feed_data["time_history"][-1] - live_feed_data["time_history"][-2]
            system_health["energy_flow_rate"] = energy_diff / max(time_diff, 0.001)
        
        # Update connection activity
        system_health["connection_activity"] = connection_count
        
        # Determine system health status
        health_score = 100.0
        alerts = []
        
        # Check energy levels
        if total_energy < 1000:
            health_score -= 20
            alerts.append("Low total energy")
        elif total_energy > 100000:
            health_score -= 15
            alerts.append("High total energy")
        
        # Check active nodes
        if active_nodes < 100:
            health_score -= 25
            alerts.append("Low active node count")
        
        # Check connection count
        if connection_count < 100:
            health_score -= 20
            alerts.append("Low connection count")
        
        # Check performance
        if perf_stats.get('fps', 0) < 20:
            health_score -= 15
            alerts.append("Low UI performance")
        
        # Update health status
        if health_score >= 80:
            system_health["status"] = "healthy"
            health_color = (0, 255, 0)
        elif health_score >= 60:
            system_health["status"] = "warning"
            health_color = (255, 255, 0)
        else:
            system_health["status"] = "critical"
            health_color = (255, 0, 0)
        
        # Update health indicators
        dpg.set_value("health_indicator", "●")
        dpg.configure_item("health_indicator", color=health_color)
        dpg.set_value("health_status_text", system_health["status"].title())
        
        # Update energy flow indicator
        energy_flow = system_health["energy_flow_rate"]
        if abs(energy_flow) > 100:
            flow_color = (0, 255, 0)  # Green for good flow
        elif abs(energy_flow) > 10:
            flow_color = (255, 255, 0)  # Yellow for moderate flow
        else:
            flow_color = (255, 0, 0)  # Red for low flow
        
        dpg.set_value("energy_flow_indicator", "●")
        dpg.configure_item("energy_flow_indicator", color=flow_color)
        dpg.set_value("energy_flow_text", f"{energy_flow:.1f}")
        
        # Update connection indicator
        if connection_count > 1000:
            conn_color = (0, 255, 0)
        elif connection_count > 100:
            conn_color = (255, 255, 0)
        else:
            conn_color = (255, 0, 0)
        
        dpg.set_value("connection_indicator", "●")
        dpg.configure_item("connection_indicator", color=conn_color)
        dpg.set_value("connection_count_text", f"{connection_count}")
        
        # Update system alerts
        system_health["alerts"] = alerts
        alerts_text = "\n".join(alerts) if alerts else "No alerts"
        dpg.set_value("system_alerts", alerts_text)
        
        system_health["last_check"] = current_time
        
    except Exception as e:
        logging.error(f"Error updating system health: {e}")

def refresh_all_displays():
    """Refresh all display elements."""
    if latest_graph is not None:
        update_sensory_visualization(latest_graph)
        update_workspace_visualization(latest_graph)
        update_system_status()
        print("All displays refreshed")


def refresh_workspace_display():
    """Refresh the workspace grid visualization."""
    try:
        if latest_graph is not None and hasattr(latest_graph, 'node_labels'):
            # Create a 16x16 grid from workspace nodes
            grid = np.zeros((16, 16), dtype=np.float32)
            
            workspace_nodes_found = False
            for node_label in latest_graph.node_labels:
                if node_label.get('type') == 'workspace':
                    workspace_nodes_found = True
                    x = node_label.get('x', 0)
                    y = node_label.get('y', 0)
                    if 0 <= x <= 15 and 0 <= y <= 15:
                        energy = node_label.get('energy', 0.0)
                        grid[y, x] = energy
            
            if not workspace_nodes_found:
                print("No workspace nodes found in graph")
                return
            
            # Normalize to 0-1 range for texture
            grid_normalized = grid / 255.0
            grid_flat = grid_normalized.flatten()
            # Convert to list for DearPyGui
            grid_list = grid_flat.tolist()
            
            # Update the texture
            dpg.set_value("workspace_grid_texture", grid_list)
            
            # Update statistics
            total_energy = np.sum(grid)
            active_nodes = np.sum(grid > 0)
            avg_energy = total_energy / max(active_nodes, 1)
            
            stats_text = f"Active Nodes: {active_nodes}/256"
            dpg.set_value("workspace_stats_text", stats_text)
            dpg.set_value("workspace_total_energy_text", f"Total Energy: {total_energy:.1f}")
            dpg.set_value("workspace_avg_energy_text", f"Average Energy: {avg_energy:.1f}")
            
        else:
            print("No graph available")
            
    except Exception as e:
        log_step("Workspace display refresh failed", error=str(e))
        logging.error(f"Error refreshing workspace display: {e}")


def update_workspace_visualization(graph):
    """Update the workspace grid visualization from the graph data."""
    log_step("update_workspace_visualization start")
    
    try:
        if not hasattr(graph, "node_labels"):
            log_step("update_workspace_visualization: missing node_labels")
            return
        
        # Check if workspace nodes exist
        workspace_nodes = [label for label in graph.node_labels if label.get('type') == 'workspace']
        if not workspace_nodes:
            log_step("update_workspace_visualization: no workspace nodes found")
            return
        
        # Create a 16x16 grid from workspace nodes
        grid = np.zeros((16, 16), dtype=np.float32)
        
        for node_label in workspace_nodes:
            x = node_label.get('x', 0)
            y = node_label.get('y', 0)
            if 0 <= x <= 15 and 0 <= y <= 15:
                energy = node_label.get('energy', 0.0)
                grid[y, x] = energy
        
        # Normalize to 0-1 range for texture
        grid_normalized = grid / 255.0
        grid_flat = grid_normalized.flatten()
        # Convert to list for DearPyGui
        grid_list = grid_flat.tolist()
        
        # Update the texture
        dpg.set_value("workspace_grid_texture", grid_list)
        
        # Update statistics
        total_energy = np.sum(grid)
        active_nodes = np.sum(grid > 0)
        avg_energy = total_energy / max(active_nodes, 1)
        
        stats_text = f"Active Nodes: {active_nodes}/256"
        dpg.set_value("workspace_stats_text", stats_text)
        dpg.set_value("workspace_total_energy_text", f"Total Energy: {total_energy:.1f}")
        dpg.set_value("workspace_avg_energy_text", f"Average Energy: {avg_energy:.1f}")
        
        log_step("update_workspace_visualization end")
        
    except Exception as e:
        logging.error(f"Error updating workspace visualization: {e}")
        log_step(f"update_workspace_visualization error: {e}")


def set_workspace_node_energy():
    """Set the energy of a specific workspace node."""
    try:
        x = dpg.get_value("workspace_x_input")
        y = dpg.get_value("workspace_y_input")
        energy = dpg.get_value("workspace_energy_input")
        
        if not (0 <= x <= 15 and 0 <= y <= 15):
            log_step("Invalid grid coordinates", error="Coordinates must be between 0 and 15")
            return
        
        if not (0.0 <= energy <= 255.0):
            log_step("Invalid energy value", error="Energy must be between 0.0 and 255.0")
            return
        
        # Find the workspace node at this grid position
        if latest_graph is not None and hasattr(latest_graph, 'node_labels'):
            for node_label in latest_graph.node_labels:
                if (node_label.get('type') == 'workspace' and 
                    node_label.get('x') == x and 
                    node_label.get('y') == y):
                    # Update the node's energy
                    node_label['energy'] = energy
                    node_label['membrane_potential'] = energy / 255.0
                    node_label['last_update'] = getattr(latest_graph, 'step', 0)
                    print(f"Updated workspace node at ({x}, {y}) with energy {energy}")
                    
                    # Update the visualization
                    refresh_workspace_display()
                    return
            
            print(f"No workspace node found at position ({x}, {y})")
        else:
            print("No graph available")
            
    except Exception as e:
        log_step("Workspace node energy setting failed", error=str(e))


def clear_workspace_grid():
    """Clear all workspace nodes by setting their energy to 0."""
    try:
        if latest_graph is not None and hasattr(latest_graph, 'node_labels'):
            cleared_count = 0
            for node_label in latest_graph.node_labels:
                if node_label.get('type') == 'workspace':
                    node_label['energy'] = 0.0
                    node_label['membrane_potential'] = 0.0
                    node_label['state'] = 'inactive'
                    node_label['last_update'] = getattr(latest_graph, 'step', 0)
                    cleared_count += 1
            
            print(f"Cleared {cleared_count} workspace nodes")
            refresh_workspace_display()
        else:
            print("No graph available")
            
    except Exception as e:
        log_step("Workspace grid clear failed", error=str(e))


def randomize_workspace_grid():
    """Randomize all workspace nodes with random energy values."""
    try:
        import random
        if latest_graph is not None and hasattr(latest_graph, 'node_labels'):
            randomized_count = 0
            for node_label in latest_graph.node_labels:
                if node_label.get('type') == 'workspace':
                    energy = random.uniform(0.0, 255.0)
                    node_label['energy'] = energy
                    node_label['membrane_potential'] = energy / 255.0
                    node_label['state'] = 'active'
                    node_label['last_update'] = getattr(latest_graph, 'step', 0)
                    randomized_count += 1
            
            print(f"Randomized {randomized_count} workspace nodes")
            refresh_workspace_display()
        else:
            print("No graph available")
            
    except Exception as e:
        log_step("Workspace grid randomization failed", error=str(e))


# --- SECTION: DPG App Setup ---
# --- SECTION: Live Training Functions ---

@log_runtime
def initialize_live_training():
    """Initialize the live training system."""
    global live_training_interface, training_active
    
    try:
        # Lazy import to avoid hanging
        from live_training_interface import LiveTrainingInterface
        
        live_training_interface = LiveTrainingInterface(
            model_type="autoencoder", 
            learning_rate=0.001
        )
        
        # Add callbacks
        live_training_interface.add_training_callback(_on_training_update)
        live_training_interface.add_metrics_callback(_on_metrics_update)
        live_training_interface.add_ui_callback(_on_ui_update)
        
        log_step("Live training system initialized")
        return True
        
    except Exception as e:
        log_step("Failed to initialize live training", error=str(e))
        return False

@log_runtime
def start_live_training():
    """Start live training from multiple data sources."""
    global live_training_interface, training_active
    
    try:
        if not live_training_interface:
            if not initialize_live_training():
                return False
        
        # Add default audio stream
        live_training_interface.add_audio_stream(
            name="microphone",
            sample_rate=44100,
            buffer_size=1024,
            channels=1
        )
        
        # Add default visual stream
        live_training_interface.add_visual_stream(
            name="screen_capture",
            width=320,
            height=240,
            fps=30
        )
        
        # Start training
        live_training_interface.start_training()
        training_active = True
        
        log_step("Live training started with audio and visual streams")
        return True
        
    except Exception as e:
        log_step("Failed to start live training", error=str(e))
        return False

@log_runtime
def stop_live_training():
    """Stop live training."""
    global live_training_interface, training_active
    
    try:
        if live_training_interface:
            live_training_interface.stop_training()
        
        training_active = False
        log_step("Live training stopped")
        return True
        
    except Exception as e:
        log_step("Failed to stop live training", error=str(e))
        return False

def _on_training_update(loss: float, samples: int):
    """Handle training updates."""
    try:
        # Update UI with training progress
        if dpg.does_item_exist("training_loss_text"):
            dpg.set_value("training_loss_text", f"Loss: {loss:.4f}")
        
        if dpg.does_item_exist("samples_processed_text"):
            dpg.set_value("samples_processed_text", f"Samples: {samples}")
        
        log_step("Training update", loss=loss, samples=samples)
        
    except Exception as e:
        log_step("Training update callback error", error=str(e))

def _on_metrics_update(metrics):
    """Handle metrics updates."""
    try:
        # Update UI with metrics
        if dpg.does_item_exist("training_speed_text"):
            dpg.set_value("training_speed_text", f"Speed: {metrics.training_speed:.1f} samples/sec")
        
        if dpg.does_item_exist("learning_rate_text"):
            dpg.set_value("learning_rate_text", f"LR: {metrics.learning_rate:.6f}")
        
        log_step("Metrics update", 
                speed=metrics.training_speed,
                lr=metrics.learning_rate)
        
    except Exception as e:
        log_step("Metrics update callback error", error=str(e))

def _on_ui_update(event: str, data: dict):
    """Handle UI updates from training system."""
    try:
        if event == "training_started":
            if dpg.does_item_exist("training_status_text"):
                dpg.set_value("training_status_text", "Training: ACTIVE")
        elif event == "training_stopped":
            if dpg.does_item_exist("training_status_text"):
                dpg.set_value("training_status_text", "Training: INACTIVE")
        
        log_step("UI update", event=event, data=data)
        
    except Exception as e:
        log_step("UI update callback error", error=str(e))

@log_runtime
def create_live_training_panel():
    """Create the live training control panel."""
    with dpg.collapsing_header(label="🎵 Live Training System", default_open=False):
        dpg.add_text("Real-time neural network training from multiple data sources")
        dpg.add_separator()
        
        # Training status
        with dpg.group(horizontal=True):
            dpg.add_text("Status:")
            dpg.add_text("Training: INACTIVE", tag="training_status_text")
        
        # Training controls
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start Live Training", 
                          callback=lambda: start_live_training(),
                          width=150)
            dpg.add_button(label="Stop Live Training", 
                          callback=lambda: stop_live_training(),
                          width=150)
        
        dpg.add_separator()
        
        # Training metrics
        with dpg.group(horizontal=True):
            dpg.add_text("Loss:")
            dpg.add_text("N/A", tag="training_loss_text")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Samples:")
            dpg.add_text("0", tag="samples_processed_text")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Speed:")
            dpg.add_text("0.0 samples/sec", tag="training_speed_text")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Learning Rate:")
            dpg.add_text("0.001", tag="learning_rate_text")
        
        dpg.add_separator()
        
        # Data sources
        dpg.add_text("Data Sources:")
        dpg.add_text("• Audio Stream (Microphone)")
        dpg.add_text("• Visual Stream (Screen Capture)")
        dpg.add_text("• Custom Streams (User-defined)")
        
        dpg.add_separator()
        
        # Performance info
        dpg.add_text("Performance:")
        dpg.add_text("• Real-time audio processing")
        dpg.add_text("• Multi-threaded training")
        dpg.add_text("• Optimized for live data")

@log_runtime
def run_ui():
    log_step("run_ui start")
    import time

    logging.info("[PERF] run_ui: UI setup starting")
    dpg.create_context()
    create_main_window()
    create_performance_window()
    create_log_window()
    create_fps_window()
    dpg.create_viewport(title="Neural System Control Center", width=1200, height=800)
    dpg.setup_dearpygui()
    
    # Ensure window is visible and positioned correctly
    dpg.set_viewport_pos([100, 100])  # Position window at 100,100
    dpg.set_viewport_always_top(False)
    
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    
    # Force window to front
    dpg.set_viewport_always_top(True)
    dpg.set_viewport_always_top(False)
    
    # Start periodic handlers
    dpg.set_frame_callback(dpg.get_frame_count() + 30, active_logging_handler)
    dpg.set_frame_callback(dpg.get_frame_count() + 1, ui_frame_handler)
    # Auto-load configuration after UI is ready
    dpg.set_frame_callback(dpg.get_frame_count() + 5, load_configuration)
    # Auto-start the simulation a few frames after UI shows
    dpg.set_frame_callback(dpg.get_frame_count() + 15, start_simulation_callback)
    
    # Also add a manual start button for testing
    logging.info("[UI] Auto-start callback scheduled for frame " + str(dpg.get_frame_count() + 15))
    
    log_step("run_ui: UI setup complete, entering main loop")
    
    # Add a small delay to ensure UI is fully rendered
    import time
    time.sleep(0.1)
    
    try:
        # Use DearPyGui's built-in event loop to keep the window open
        dpg.start_dearpygui()
    except Exception as e:
        logging.error(f"Critical error in UI main loop: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(2)
    finally:
        dpg.destroy_context()
        logging.info("[PERF] run_ui: UI shutdown complete")
        log_step("run_ui end")


# --- SECTION: Entry Point ---
if __name__ == "__main__":
    run_ui()

# --- END OF FILE ---
# AI: EXTEND HERE - Add more windows, config panels, graph visualizations, etc.
# See DearPyGui docs for advanced usage: https://dearpygui.readthedocs.io/en/latest/tutorials/dpg-structure.html
