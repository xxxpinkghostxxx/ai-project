import dearpygui.dearpygui as dpg
from main_graph import initialize_main_graph  # <-- Import the new graph initializer
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
from main_loop import update_dynamic_node_energies
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

# --- SECTION: Shared State ---
simulation_running = False
latest_graph = None
latest_graph_for_ui = None
update_for_ui = False
sim_update_counter = 0
last_update_time = 0
sensory_texture_tag = "sensory_texture"
sensory_image_tag = "sensory_image"
graph_lock = Lock()
graph_h = None
graph_w = None


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
        graph.x[: h * w, 0] = flat
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
    arr = graph.x[: h * w].cpu().numpy().reshape((h, w)).astype(np.uint8)
    # Convert grayscale to RGB for DPG (repeat channel)
    arr_rgb = np.stack([arr] * 3, axis=-1)
    arr_f = arr_rgb.astype(np.float32) / 255.0
    arr_f = arr_f.flatten()
    # Update the dynamic texture
    dpg.set_value(sensory_texture_tag, arr_f)
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
    log_step("Show Runtime Log button clicked")
    logs = get_log_lines()
    print(
        "\n--- Current Runtime Log ---\n"
        + "\n".join(logs[-100:])
        + "\n--- End of Log ---\n"
    )


# --- SECTION: Toggle Active Logging Callback ---
def toggle_active_logging_callback():
    global active_logging
    active_logging = not active_logging
    log_step("Active logging toggled", enabled=active_logging)
    dpg.set_value(
        "active_logging_status", f"Active Logging: {'ON' if active_logging else 'OFF'}"
    )


# --- SECTION: Active Logging Handler ---
def active_logging_handler():
    if active_logging:
        logs = get_log_lines()
        print(
            "\n--- Active Log Update ---\n" + "\n".join(logs[-10:]) + "\n--- End ---\n"
        )
    # Schedule next check
    dpg.set_frame_callback(
        dpg.get_frame_count() + 30, active_logging_handler
    )  # ~0.5s at 60fps


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

    global latest_graph, simulation_running, sim_update_counter, latest_graph_for_ui, update_for_ui, perf_stats, graph_h, graph_w
    logging.info("[PERF] simulation_loop: start")
    # Initialize the graph once
    from main_graph import initialize_main_graph

    main_graph = initialize_main_graph()
    latest_graph = main_graph
    graph_h = main_graph.h
    graph_w = main_graph.w
    while simulation_running:
        t0 = time.time()
        # 1. Update only the sensory node features in-place
        update_sensory_features(main_graph)
        # 2. Update dynamic node energies (decay, transfer, clamp)
        update_dynamic_node_energies(main_graph)
        # 3. Birth new dynamic nodes if energy threshold is exceeded
        birth_new_dynamic_nodes(main_graph)
        # 4. Remove dead dynamic nodes if energy below threshold
        remove_dead_dynamic_nodes(main_graph)
        # TODO: Add dynamic node connection updates if needed
        t1 = time.time()
        with graph_lock:
            latest_graph = main_graph
            sim_update_counter += 1
            perf_stats["sim_updates"] += 1
            perf_stats["last_sim_update_time"] = t1 - t0
            if sim_update_counter % SENSORY_UI_UPDATE_EVERY_N == 0:
                latest_graph_for_ui = main_graph
                update_for_ui = True
        logging.info(f"[PERF] simulation_loop: graph update took {(t1-t0)*1000:.2f} ms")
        time.sleep(UPDATE_INTERVAL)
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
    with graph_lock:
        if update_for_ui and latest_graph_for_ui is not None:
            t1 = time.perf_counter()
            update_sensory_visualization(latest_graph_for_ui)
            t2 = time.perf_counter()
            perf_stats["ui_updates"] += 1
            perf_stats["last_ui_update_time"] = t2 - t1
            update_for_ui = False
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
    simulation_running = True
    sim_update_counter = 0
    dpg.set_value("sim_status_text", "Simulation Running")
    Thread(target=simulation_loop, daemon=True).start()
    last_update_time = time.time()
    logging.info(f"[UI] Simulation started at {last_update_time}")
    log_step("start_simulation_callback end")


@log_runtime
def hard_stop_simulation_callback():
    log_step("hard_stop_simulation_callback start")
    logging.info("[PERF] hard_stop_simulation_callback: Simulation stopping")
    global simulation_running
    simulation_running = False
    dpg.set_value("sim_status_text", "Simulation Stopped (Hard)")
    logging.info("[UI] Simulation stopped (hard)")
    # AI: EXTEND HERE - Insert simulation stop/cleanup logic
    log_step("hard_stop_simulation_callback end")


# --- SECTION: Runtime Log UI Display Callback ---
def show_runtime_log_in_ui():
    print("DEBUG: Show Runtime Log in UI button pressed")
    logs = get_log_lines()
    print(f"DEBUG: log buffer length = {len(logs)} (UI)")
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
        print("DEBUG: Log buffer is empty (UI).")


# --- SECTION: Main Window Layout ---
@log_runtime
def create_main_window():
    log_step("create_main_window start")
    logging.info("[PERF] create_main_window: Creating main window")
    with dpg.window(label="Main Control", tag="main_window", width=400, height=300):
        dpg.add_text("Energy-Based Neural System UI", tag="main_title")
        dpg.add_separator()
        dpg.add_text("Simulation Status: Stopped", tag="sim_status_text")
        dpg.add_button(
            label="Start Simulation",
            tag="start_button",
            callback=start_simulation_callback,
        )
        dpg.add_button(
            label="Hard Stop",
            tag="hard_stop_button",
            callback=hard_stop_simulation_callback,
        )
        dpg.add_separator()
        dpg.add_button(
            label="Show Runtime Log (Console)",
            tag="show_runtime_log_button",
            callback=show_runtime_log_callback,
        )
        dpg.add_button(
            label="Show Runtime Log in UI",
            tag="show_runtime_log_ui_button",
            callback=show_runtime_log_in_ui,
        )
        dpg.add_text("", tag="runtime_log_text", wrap=380)
        dpg.add_button(
            label="Toggle Active Logging",
            tag="toggle_active_logging_button",
            callback=toggle_active_logging_callback,
        )
        dpg.add_text("Active Logging: OFF", tag="active_logging_status")
    log_step("create_main_window end")


# --- SECTION: Sensory Visualization Window ---
@log_runtime
def create_sensory_visualization_window():
    log_step("create_sensory_visualization_window start")
    logging.info(
        "[PERF] create_sensory_visualization_window: Creating sensory visualization window"
    )
    # Create a placeholder texture (will be updated on simulation start)
    # Use a small default size (e.g., 64x36) for initial allocation
    default_w, default_h = 64, 36
    arr = np.zeros((default_h, default_w), dtype=np.float32).flatten()
    with dpg.texture_registry():
        dpg.add_dynamic_texture(default_w, default_h, arr, tag=sensory_texture_tag)
    with dpg.window(
        label="Sensory Visualization",
        tag="sensory_window",
        width=400,
        height=400,
        pos=(420, 0),
    ):
        dpg.add_text("Sensory Input (RGB, downscaled)")
        dpg.add_image(sensory_texture_tag, tag=sensory_image_tag)
        # AI: EXTEND HERE - Add overlays, controls, etc.
    log_step("create_sensory_visualization_window end")


# --- SECTION: Connection Visualization Window ---
@log_runtime
def create_connection_visualization_window():
    log_step("create_connection_visualization_window start")
    logging.info("[PERF] create_connection_visualization_window: Creating connection visualization window")
    
    with dpg.window(
        label="Connection Visualization",
        tag="connection_window",
        width=400,
        height=300,
        pos=(0, 420),
    ):
        dpg.add_text("Enhanced Connection System")
        dpg.add_separator()
        
        # Connection statistics
        dpg.add_text("Connection Stats:", tag="connection_stats_text")
        dpg.add_text("Edge Types:", tag="edge_types_text")
        dpg.add_text("Weight Distribution:", tag="weight_dist_text")
        
        # Connection controls
        dpg.add_button(
            label="Refresh Connections",
            tag="refresh_connections_button",
            callback=refresh_connection_display,
        )
        dpg.add_button(
            label="Show Connection Matrix",
            tag="show_matrix_button",
            callback=show_connection_matrix,
        )
    
    log_step("create_connection_visualization_window end")


def refresh_connection_display():
    """Refresh the connection visualization display."""
    log_step("refresh_connection_display start")
    
    if latest_graph is None:
        dpg.set_value("connection_stats_text", "No graph available")
        return
    
    try:
        # Get connection statistics
        edge_count = latest_graph.edge_index.shape[1] if latest_graph.edge_index.numel() > 0 else 0
        
        # Count edge types if available
        edge_types = {}
        if hasattr(latest_graph, 'edge_attributes') and latest_graph.edge_attributes:
            for edge in latest_graph.edge_attributes:
                edge_type = edge.type
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Calculate weight statistics
        weights = []
        if hasattr(latest_graph, 'edge_attributes') and latest_graph.edge_attributes:
            weights = [edge.weight for edge in latest_graph.edge_attributes]
        
        # Update display
        stats_text = f"Total Connections: {edge_count}"
        dpg.set_value("connection_stats_text", stats_text)
        
        if edge_types:
            types_text = ", ".join([f"{k}: {v}" for k, v in edge_types.items()])
            dpg.set_value("edge_types_text", f"Edge Types: {types_text}")
        else:
            dpg.set_value("edge_types_text", "Edge Types: Basic connections")
        
        if weights:
            avg_weight = sum(weights) / len(weights)
            max_weight = max(weights)
            min_weight = min(weights)
            weight_text = f"Avg: {avg_weight:.2f}, Max: {max_weight:.2f}, Min: {min_weight:.2f}"
            dpg.set_value("weight_dist_text", f"Weight Distribution: {weight_text}")
        else:
            dpg.set_value("weight_dist_text", "Weight Distribution: No weights available")
            
    except Exception as e:
        logging.error(f"Error refreshing connection display: {e}")
        dpg.set_value("connection_stats_text", f"Error: {str(e)}")
    
    log_step("refresh_connection_display end")


def show_connection_matrix():
    """Show a simple connection matrix visualization."""
    log_step("show_connection_matrix start")
    
    if latest_graph is None:
        print("No graph available for connection matrix")
        return
    
    try:
        # Create a simple connection matrix
        num_nodes = len(latest_graph.node_labels)
        matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
        
        # Fill matrix based on edge_index
        if latest_graph.edge_index.numel() > 0:
            src = latest_graph.edge_index[0].cpu().numpy()
            tgt = latest_graph.edge_index[1].cpu().numpy()
            for i in range(len(src)):
                matrix[src[i], tgt[i]] = True
        
        # Print matrix (for now, just show summary)
        connections = np.sum(matrix)
        density = connections / (num_nodes * num_nodes) if num_nodes > 0 else 0
        
        print(f"\n--- Connection Matrix Summary ---")
        print(f"Nodes: {num_nodes}")
        print(f"Connections: {connections}")
        print(f"Density: {density:.4f}")
        print(f"Matrix shape: {matrix.shape}")
        
        # Show first few connections
        if connections > 0:
            print(f"\nFirst 10 connections:")
            count = 0
            for i in range(min(10, num_nodes)):
                for j in range(min(10, num_nodes)):
                    if matrix[i, j] and count < 10:
                        print(f"  {i} -> {j}")
                        count += 1
        
        print("--- End Connection Matrix ---\n")
        
    except Exception as e:
        logging.error(f"Error showing connection matrix: {e}")
        print(f"Error: {e}")
    
    log_step("show_connection_matrix end")


# --- SECTION: Learning Visualization Window ---
@log_runtime
def create_learning_visualization_window():
    log_step("create_learning_visualization_window start")
    logging.info("[PERF] create_learning_visualization_window: Creating learning visualization window")
    
    with dpg.window(
        label="Learning & Plasticity",
        tag="learning_window",
        width=400,
        height=300,
        pos=(830, 100),
    ):
        dpg.add_text("Learning System Status")
        dpg.add_separator()
        
        # Learning statistics
        dpg.add_text("Learning Stats:", tag="learning_stats_text")
        dpg.add_text("STDP Events:", tag="stdp_events_text")
        dpg.add_text("Weight Changes:", tag="weight_changes_text")
        dpg.add_text("Memory Traces:", tag="memory_traces_text")
        
        # Learning controls
        dpg.add_button(
            label="Refresh Learning Stats",
            tag="refresh_learning_button",
            callback=refresh_learning_display,
        )
        dpg.add_button(
            label="Show Learning Patterns",
            tag="show_patterns_button",
            callback=show_learning_patterns,
        )
    
    log_step("create_learning_visualization_window end")


def refresh_learning_display():
    """Refresh the learning visualization display."""
    log_step("refresh_learning_display start")
    
    if latest_graph is None:
        dpg.set_value("learning_stats_text", "No graph available")
        return
    
    try:
        # Get learning statistics if available
        if hasattr(latest_graph, 'learning_engine'):
            learning_stats = latest_graph.learning_engine.get_learning_statistics()
            
            # Update learning stats display
            dpg.set_value("learning_stats_text", f"Learning Stats: {learning_stats['stdp_events']} STDP, {learning_stats['weight_changes']} changes")
            dpg.set_value("stdp_events_text", f"STDP Events: {learning_stats['stdp_events']}")
            dpg.set_value("weight_changes_text", f"Weight Changes: {learning_stats['weight_changes']}")
            
            # Calculate learning efficiency
            from learning_engine import calculate_learning_efficiency
            efficiency = calculate_learning_efficiency(latest_graph)
            dpg.set_value("memory_traces_text", f"Learning Efficiency: {efficiency:.3f}")
            
        else:
            dpg.set_value("learning_stats_text", "Learning engine not initialized")
            dpg.set_value("stdp_events_text", "STDP Events: N/A")
            dpg.set_value("weight_changes_text", "Weight Changes: N/A")
            dpg.set_value("memory_traces_text", "Learning Efficiency: N/A")
        
        # Get memory statistics if available
        if hasattr(latest_graph, 'memory_system'):
            memory_stats = latest_graph.memory_system.get_memory_statistics()
            memory_count = latest_graph.memory_system.get_memory_trace_count()
            
            # Update memory display
            dpg.set_value("memory_traces_text", f"Memory Traces: {memory_count} active, {memory_stats['total_memory_strength']:.2f} strength")
            
    except Exception as e:
        logging.error(f"Error refreshing learning display: {e}")
        dpg.set_value("learning_stats_text", f"Error: {str(e)}")
    
    log_step("refresh_learning_display end")


def show_learning_patterns():
    """Show learning pattern analysis."""
    log_step("show_learning_patterns start")
    
    if latest_graph is None:
        print("No graph available for learning pattern analysis")
        return
    
    try:
        # Analyze learning patterns
        from learning_engine import detect_learning_patterns
        patterns = detect_learning_patterns(latest_graph)
        
        print(f"\n--- Learning Pattern Analysis ---")
        print(f"Patterns Detected: {patterns['patterns_detected']}")
        print(f"Weight Variance: {patterns['weight_variance']:.3f}")
        print(f"Edge Type Distribution: {patterns['edge_type_distribution']}")
        print(f"Total Connections: {patterns['total_connections']}")
        
        # Analyze memory distribution if available
        if hasattr(latest_graph, 'memory_system'):
            from memory_system import analyze_memory_distribution
            memory_analysis = analyze_memory_distribution(latest_graph.memory_system)
            
            print(f"\n--- Memory System Analysis ---")
            print(f"Total Memories: {memory_analysis['total_memories']}")
            print(f"Average Strength: {memory_analysis['avg_strength']:.3f}")
            print(f"Strength Variance: {memory_analysis['strength_variance']:.3f}")
            print(f"Pattern Distribution: {memory_analysis['pattern_distribution']}")
        
        print("--- End Analysis ---\n")
        
    except Exception as e:
        logging.error(f"Error showing learning patterns: {e}")
        print(f"Error: {e}")
    
    log_step("show_learning_patterns end")


# --- SECTION: Memory Visualization Window ---
@log_runtime
def create_memory_visualization_window():
    log_step("create_memory_visualization_window start")
    logging.info("[PERF] create_memory_visualization_window: Creating memory visualization window")
    
    with dpg.window(
        label="Memory System",
        tag="memory_window",
        width=400,
        height=300,
        pos=(830, 420),
    ):
        dpg.add_text("Memory System Status")
        dpg.add_separator()
        
        # Memory statistics
        dpg.add_text("Memory Stats:", tag="memory_stats_text")
        dpg.add_text("Active Traces:", tag="active_traces_text")
        dpg.add_text("Pattern Types:", tag="pattern_types_text")
        dpg.add_text("Memory Strength:", tag="memory_strength_text")
        
        # Memory controls
        dpg.add_button(
            label="Refresh Memory Stats",
            tag="refresh_memory_button",
            callback=refresh_memory_display,
        )
        dpg.add_button(
            label="Show Memory Summary",
            tag="show_memory_summary_button",
            callback=show_memory_summary,
        )
    
    log_step("create_memory_visualization_window end")


def refresh_memory_display():
    """Refresh the memory visualization display."""
    log_step("refresh_memory_display start")
    
    if latest_graph is None:
        dpg.set_value("memory_stats_text", "No graph available")
        return
    
    try:
        # Get memory statistics if available
        if hasattr(latest_graph, 'memory_system'):
            memory_stats = latest_graph.memory_system.get_memory_statistics()
            memory_summary = latest_graph.memory_system.get_memory_summary()
            
            # Update memory stats display
            dpg.set_value("memory_stats_text", f"Memory Stats: {memory_stats['traces_formed']} formed, {memory_stats['traces_consolidated']} consolidated")
            dpg.set_value("active_traces_text", f"Active Traces: {memory_stats['patterns_recalled']} recalled")
            
            # Analyze pattern types
            if memory_summary:
                pattern_types = {}
                total_strength = 0.0
                for memory in memory_summary:
                    pattern_type = memory['pattern_type']
                    pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
                    total_strength += memory['strength']
                
                pattern_text = ", ".join([f"{k}: {v}" for k, v in pattern_types.items()])
                dpg.set_value("pattern_types_text", f"Pattern Types: {pattern_text}")
                dpg.set_value("memory_strength_text", f"Total Strength: {total_strength:.2f}")
            else:
                dpg.set_value("pattern_types_text", "Pattern Types: None")
                dpg.set_value("memory_strength_text", "Total Strength: 0.0")
        else:
            dpg.set_value("memory_stats_text", "Memory system not initialized")
            dpg.set_value("active_traces_text", "Active Traces: N/A")
            dpg.set_value("pattern_types_text", "Pattern Types: N/A")
            dpg.set_value("memory_strength_text", "Total Strength: N/A")
            
    except Exception as e:
        logging.error(f"Error refreshing memory display: {e}")
        dpg.set_value("memory_stats_text", f"Error: {str(e)}")
    
    log_step("refresh_memory_display end")


def show_memory_summary():
    """Show detailed memory summary."""
    log_step("show_memory_summary start")
    
    if latest_graph is None:
        print("No graph available for memory summary")
        return
    
    try:
        if hasattr(latest_graph, 'memory_system'):
            memory_summary = latest_graph.memory_system.get_memory_summary()
            
            print(f"\n--- Memory System Summary ---")
            print(f"Total Memory Traces: {len(memory_summary)}")
            
            if memory_summary:
                print(f"\nMemory Traces:")
                for i, memory in enumerate(memory_summary[:10]):  # Show first 10
                    print(f"  {i+1}. Node {memory['node_id']}: {memory['pattern_type']}")
                    print(f"     Strength: {memory['strength']:.3f}, Activations: {memory['activation_count']}")
                    print(f"     Age: {memory['age_minutes']:.1f} minutes")
                
                if len(memory_summary) > 10:
                    print(f"  ... and {len(memory_summary) - 10} more")
            else:
                print("  No memory traces formed yet")
            
            print("--- End Memory Summary ---\n")
        else:
            print("Memory system not initialized")
        
    except Exception as e:
        logging.error(f"Error showing memory summary: {e}")
        print(f"Error: {e}")
    
    log_step("show_memory_summary end")


# --- SECTION: Homeostasis Visualization Window ---
@log_runtime
def create_homeostasis_visualization_window():
    log_step("create_homeostasis_visualization_window start")
    logging.info("[PERF] create_homeostasis_visualization_window: Creating homeostasis visualization window")
    
    with dpg.window(
        label="Homeostasis & Network Health",
        tag="homeostasis_window",
        width=400,
        height=300,
        pos=(0, 740),
    ):
        dpg.add_text("Network Regulation Status")
        dpg.add_separator()
        
        # Homeostasis statistics
        dpg.add_text("Regulation Stats:", tag="regulation_stats_text")
        dpg.add_text("Network Health:", tag="network_health_text")
        dpg.add_text("Criticality Status:", tag="criticality_status_text")
        dpg.add_text("Energy Balance:", tag="energy_balance_text")
        
        # Homeostasis controls
        dpg.add_button(
            label="Refresh Homeostasis Stats",
            tag="refresh_homeostasis_button",
            callback=refresh_homeostasis_display,
        )
        dpg.add_button(
            label="Show Network Trends",
            tag="show_trends_button",
            callback=show_network_trends,
        )
    
    log_step("create_homeostasis_visualization_window end")


def refresh_homeostasis_display():
    """Refresh the homeostasis visualization display."""
    log_step("refresh_homeostasis_display start")
    
    if latest_graph is None:
        dpg.set_value("regulation_stats_text", "No graph available")
        return
    
    try:
        # Get homeostasis statistics if available
        if hasattr(latest_graph, 'homeostasis_controller'):
            regulation_stats = latest_graph.homeostasis_controller.get_regulation_statistics()
            health_status = latest_graph.homeostasis_controller.monitor_network_health(latest_graph)
            network_trends = latest_graph.homeostasis_controller.get_network_trends()
            
            # Update regulation stats display
            dpg.set_value("regulation_stats_text", 
                         f"Regulation Stats: {regulation_stats['total_regulation_events']} events, "
                         f"{regulation_stats['energy_regulations']} energy, "
                         f"{regulation_stats['criticality_regulations']} criticality")
            
            # Update network health display
            health_color = "green" if health_status['status'] == 'healthy' else "orange" if health_status['status'] == 'warning' else "red"
            dpg.set_value("network_health_text", 
                         f"Network Health: {health_status['status'].upper()} (Score: {health_status['health_score']:.2f})")
            
            # Update criticality status
            if network_trends:
                branching_trend = network_trends.get('branching_trend', 'unknown')
                dpg.set_value("criticality_status_text", f"Criticality Status: {branching_trend}")
            else:
                dpg.set_value("criticality_status_text", "Criticality Status: Unknown")
            
            # Update energy balance
            if hasattr(latest_graph, 'homeostasis_data'):
                last_regulation = latest_graph.homeostasis_data.get('last_regulation', {})
                if last_regulation:
                    regulation_type = last_regulation.get('type', 'none')
                    dpg.set_value("energy_balance_text", f"Energy Balance: {regulation_type}")
                else:
                    dpg.set_value("energy_balance_text", "Energy Balance: Stable")
            else:
                dpg.set_value("energy_balance_text", "Energy Balance: No data")
                
        else:
            dpg.set_value("regulation_stats_text", "Homeostasis controller not initialized")
            dpg.set_value("network_health_text", "Network Health: Unknown")
            dpg.set_value("criticality_status_text", "Criticality Status: Unknown")
            dpg.set_value("energy_balance_text", "Energy Balance: Unknown")
            
    except Exception as e:
        logging.error(f"Error refreshing homeostasis display: {e}")
        dpg.set_value("regulation_stats_text", f"Error: {str(e)}")
    
    log_step("refresh_homeostasis_display end")


def show_network_trends():
    """Show network trend analysis."""
    log_step("show_network_trends start")
    
    if latest_graph is None:
        print("No graph available for network trend analysis")
        return
    
    try:
        if hasattr(latest_graph, 'homeostasis_controller'):
            trends = latest_graph.homeostasis_controller.get_network_trends()
            health_status = latest_graph.homeostasis_controller.monitor_network_health(latest_graph)
            
            print(f"\n--- Network Trend Analysis ---")
            print(f"Energy Trend: {trends.get('energy_trend', 'unknown')} (slope: {trends.get('energy_slope', 0):.4f})")
            print(f"Branching Trend: {trends.get('branching_trend', 'unknown')} (slope: {trends.get('branching_slope', 0):.4f})")
            print(f"Variance Trend: {trends.get('variance_trend', 'unknown')} (slope: {trends.get('variance_slope', 0):.4f})")
            
            print(f"\n--- Network Health Status ---")
            print(f"Status: {health_status['status'].upper()}")
            print(f"Health Score: {health_status['health_score']:.3f}")
            print(f"Warnings: {len(health_status['warnings'])}")
            
            if health_status['warnings']:
                for warning in health_status['warnings']:
                    print(f"  - {warning}")
            
            print(f"\n--- Network Metrics ---")
            metrics = health_status['metrics']
            print(f"Total Energy: {metrics['total_energy']:.1f}")
            print(f"Node Count: {metrics['num_nodes']}")
            print(f"Average Energy: {metrics['avg_energy']:.1f}")
            print(f"Connection Density: {metrics['connection_density']:.6f}")
            
            print("--- End Analysis ---\n")
        else:
            print("Homeostasis controller not initialized")
        
    except Exception as e:
        logging.error(f"Error showing network trends: {e}")
        print(f"Error: {e}")
    
    log_step("show_network_trends end")


# --- SECTION: Energy Dynamics Visualization Window ---
@log_runtime
def create_energy_dynamics_window():
    log_step("create_energy_dynamics_window start")
    logging.info("[PERF] create_energy_dynamics_window: Creating energy dynamics visualization window")
    
    with dpg.window(
        label="Energy Dynamics",
        tag="energy_dynamics_window",
        width=400,
        height=300,
        pos=(420, 740),
    ):
        dpg.add_text("Energy System Status")
        dpg.add_separator()
        
        # Energy dynamics statistics
        dpg.add_text("Energy Stats:", tag="energy_stats_text")
        dpg.add_text("Membrane Potentials:", tag="membrane_potentials_text")
        dpg.add_text("Refractory Status:", tag="refractory_status_text")
        dpg.add_text("Plasticity Gates:", tag="plasticity_gates_text")
        
        # Energy dynamics controls
        dpg.add_button(
            label="Refresh Energy Stats",
            tag="refresh_energy_button",
            callback=refresh_energy_dynamics_display,
        )
        dpg.add_button(
            label="Show Energy Analysis",
            tag="show_energy_analysis_button",
            callback=show_energy_analysis,
        )
    
    log_step("create_energy_dynamics_window end")


def refresh_energy_dynamics_display():
    """Refresh the energy dynamics visualization display."""
    log_step("refresh_energy_dynamics_display start")
    
    if latest_graph is None:
        dpg.set_value("energy_stats_text", "No graph available")
        return
    
    try:
        # Calculate energy statistics
        energy_values = latest_graph.x[:, 0].cpu().numpy()
        total_energy = float(np.sum(energy_values))
        avg_energy = float(np.mean(energy_values))
        energy_variance = float(np.var(energy_values)) if len(energy_values) > 1 else 0.0
        
        # Update energy stats display
        dpg.set_value("energy_stats_text", 
                     f"Energy Stats: Total={total_energy:.1f}, Avg={avg_energy:.1f}, Var={energy_variance:.1f}")
        
        # Calculate membrane potential statistics
        membrane_potentials = []
        refractory_active = 0
        plasticity_enabled = 0
        
        for node in latest_graph.node_labels:
            if 'membrane_potential' in node:
                membrane_potentials.append(node['membrane_potential'])
            
            if node.get('refractory_timer', 0) > 0:
                refractory_active += 1
            
            if node.get('plasticity_enabled', False):
                plasticity_enabled += 1
        
        if membrane_potentials:
            avg_membrane = np.mean(membrane_potentials)
            dpg.set_value("membrane_potentials_text", f"Membrane Potentials: Avg={avg_membrane:.3f}")
        else:
            dpg.set_value("membrane_potentials_text", "Membrane Potentials: No data")
        
        # Update refractory and plasticity status
        total_nodes = len(latest_graph.node_labels)
        dpg.set_value("refractory_status_text", f"Refractory Status: {refractory_active}/{total_nodes} active")
        dpg.set_value("plasticity_gates_text", f"Plasticity Gates: {plasticity_enabled}/{total_nodes} enabled")
        
    except Exception as e:
        logging.error(f"Error refreshing energy dynamics display: {e}")
        dpg.set_value("energy_stats_text", f"Error: {str(e)}")
    
    log_step("refresh_energy_dynamics_display end")


def show_energy_analysis():
    """Show detailed energy analysis."""
    log_step("show_energy_analysis start")
    
    if latest_graph is None:
        print("No graph available for energy analysis")
        return
    
    try:
        # Analyze energy distribution
        energy_values = latest_graph.x[:, 0].cpu().numpy()
        total_energy = float(np.sum(energy_values))
        avg_energy = float(np.mean(energy_values))
        energy_variance = float(np.var(energy_values)) if len(energy_values) > 1 else 0.0
        
        print(f"\n--- Energy System Analysis ---")
        print(f"Total Energy: {total_energy:.1f}")
        print(f"Average Energy: {avg_energy:.1f}")
        print(f"Energy Variance: {energy_variance:.1f}")
        print(f"Energy Range: {np.min(energy_values):.1f} - {np.max(energy_values):.1f}")
        
        # Analyze behavior-specific energy patterns
        behavior_energy = {}
        for node in latest_graph.node_labels:
            behavior = node.get('behavior', 'unknown')
            if behavior not in behavior_energy:
                behavior_energy[behavior] = []
            
            node_idx = latest_graph.node_labels.index(node)
            if node_idx < len(energy_values):
                behavior_energy[behavior].append(energy_values[node_idx])
        
        print(f"\n--- Behavior Energy Patterns ---")
        for behavior, energies in behavior_energy.items():
            if energies:
                avg_behavior_energy = np.mean(energies)
                print(f"{behavior.capitalize()}: {len(energies)} nodes, avg energy: {avg_behavior_energy:.1f}")
        
        # Analyze membrane potentials
        membrane_potentials = []
        for node in latest_graph.node_labels:
            if 'membrane_potential' in node:
                membrane_potentials.append(node['membrane_potential'])
        
        if membrane_potentials:
            print(f"\n--- Membrane Potential Analysis ---")
            print(f"Average Membrane Potential: {np.mean(membrane_potentials):.3f}")
            print(f"Membrane Potential Range: {np.min(membrane_potentials):.3f} - {np.max(membrane_potentials):.3f}")
        
        # Analyze refractory periods
        refractory_count = sum(1 for node in latest_graph.node_labels if node.get('refractory_timer', 0) > 0)
        print(f"\n--- Refractory Period Status ---")
        print(f"Nodes in Refractory Period: {refractory_count}/{len(latest_graph.node_labels)}")
        
        print("--- End Energy Analysis ---\n")
        
    except Exception as e:
        logging.error(f"Error showing energy analysis: {e}")
        print(f"Error: {e}")
    
    log_step("show_energy_analysis end")


# --- SECTION: DPG App Setup ---
@log_runtime
def run_ui():
    log_step("run_ui start")
    import time

    logging.info("[PERF] run_ui: UI setup starting")
    dpg.create_context()
    create_main_window()
    create_sensory_visualization_window()
    create_performance_window()
    create_log_window()
    create_fps_window()
    create_connection_visualization_window() # Added this line
    create_learning_visualization_window() # Added this line
    create_memory_visualization_window() # Added this line
    create_homeostasis_visualization_window() # Added this line
    create_energy_dynamics_window() # Added this line
    dpg.create_viewport(title="Neural System UI", width=1100, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    # Start active logging handler
    dpg.set_frame_callback(dpg.get_frame_count() + 30, active_logging_handler)
    log_step("run_ui: UI setup complete, entering main loop")
    while dpg.is_dearpygui_running():
        t0 = time.perf_counter()
        live_update_callback()
        dpg.render_dearpygui_frame()
        t1 = time.perf_counter()
        logging.info(f"[PERF] run_ui: frame render elapsed {(t1-t0)*1000:.2f} ms")
        # Performance logging every second (UI only)
        now = time.time()
        if now - perf_stats["last_log_time"] >= 1.0:
            fps = perf_stats["ui_updates"] / (now - perf_stats["last_log_time"])
            sim_fps = perf_stats["sim_updates"] / (now - perf_stats["last_log_time"])
            report = (
                f"UI FPS: {fps:.2f}\n"
                f"Sim FPS: {sim_fps:.2f}\n"
                f"Last Sim Update: {perf_stats['last_sim_update_time']*1000:.1f} ms\n"
                f"Last UI Update: {perf_stats['last_ui_update_time']*1000:.1f} ms"
            )
            perf_stats["fps"] = fps
            perf_stats["sim_fps"] = sim_fps
            perf_stats["last_report"] = report
            dpg.set_value(PERF_TEXT_TAG, report)
            dpg.set_value(FPS_TEXT_TAG, f"UI FPS: {fps:.2f}\nSim FPS: {sim_fps:.2f}")
            logging.info("[PERF] " + report.replace("\n", ", "))
            perf_stats["sim_updates"] = 0
            perf_stats["ui_updates"] = 0
            perf_stats["last_log_time"] = now
    dpg.destroy_context()
    logging.info("[PERF] run_ui: UI shutdown complete")
    log_step("run_ui end")


# --- SECTION: Entry Point ---
if __name__ == "__main__":
    run_ui()

# --- END OF FILE ---
# AI: EXTEND HERE - Add more windows, config panels, graph visualizations, etc.
# See DearPyGui docs for advanced usage: https://dearpygui.readthedocs.io/en/latest/tutorials/dpg-structure.html
