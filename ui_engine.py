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
