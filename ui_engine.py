import dearpygui.dearpygui as dpg
from main_graph import initialize_main_graph  # <-- Import the new graph initializer
import numpy as np
import time
from threading import Thread, Lock
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- SECTION: Performance Goals ---
TARGET_FPS = 30  # System should ideally run at 30 frames per second
UPDATE_INTERVAL = 1.0 / TARGET_FPS  # seconds between updates
SENSORY_UI_UPDATE_EVERY_N = 4  # UI updates its visualization every 4th simulation update

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

# --- SECTION: Sensory Visualization Helper ---
def update_sensory_visualization(graph):
    import time
    t0 = time.perf_counter()
    logging.info("[PERF] update_sensory_visualization: start")
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return
    # Extract sensory node info (assume all sensory nodes are first N nodes)
    sensory_labels = [lbl for lbl in graph.node_labels if lbl.get('type', 'sensory') == 'sensory' or 'channel' in lbl]
    if not sensory_labels:
        return
    # Get image dimensions
    h = getattr(graph, 'h', None)
    w = getattr(graph, 'w', None)
    if h is None or w is None:
        return
    # Reconstruct RGB image from sensory nodes
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, lbl in enumerate(sensory_labels):
        x = lbl['x']
        y = lbl['y']
        channel = lbl['channel']
        c = {'R': 0, 'G': 1, 'B': 2}[channel]
        arr[y, x, c] = int(graph.x[idx].item())
    # Flatten and normalize for DPG (DPG expects float32 in [0,1])
    arr_f = arr.astype(np.float32) / 255.0
    arr_f = arr_f.flatten()
    # Update the dynamic texture
    dpg.set_value(sensory_texture_tag, arr_f)
    t1 = time.perf_counter()
    logging.info(f"[PERF] update_sensory_visualization: end, elapsed {(t1-t0)*1000:.2f} ms")

# --- SECTION: Performance Logging ---
perf_stats = {
    'sim_updates': 0,
    'ui_updates': 0,
    'last_log_time': time.time(),
    'last_sim_update_time': 0.0,
    'last_ui_update_time': 0.0,
    'fps': 0.0,
    'sim_fps': 0.0,
    'last_report': '',
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

def append_log_line(line):
    global log_lines
    log_lines.append(line)
    if len(log_lines) > MAX_LOG_LINES:
        log_lines = log_lines[-MAX_LOG_LINES:]
    if dpg.does_item_exist(LOG_TEXT_TAG):
        dpg.set_value(LOG_TEXT_TAG, "\n".join(log_lines))

# Custom logging handler for UI
import logging
class UILogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        append_log_line(msg)

# Attach handler to root logger
ui_log_handler = UILogHandler()
ui_log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logging.getLogger().addHandler(ui_log_handler)

# --- SECTION: Performance Window ---
def create_performance_window():
    logging.info("[PERF] create_performance_window: Creating performance window")
    with dpg.window(label="Performance Stats", tag=PERF_WINDOW_TAG, width=400, height=120, pos=(420, 420)):
        dpg.add_text("Performance report will appear here...", tag=PERF_TEXT_TAG)

# --- SECTION: Log Window ---
def create_log_window():
    logging.info("[PERF] create_log_window: Creating log window")
    with dpg.window(label="Log Output", tag=LOG_WINDOW_TAG, width=450, height=250, pos=(0, 220)):
        dpg.add_text("", tag=LOG_TEXT_TAG, wrap=440)

# --- SECTION: FPS Window ---
def create_fps_window():
    logging.info("[PERF] create_fps_window: Creating FPS window")
    with dpg.window(label="FPS Only", tag=FPS_WINDOW_TAG, width=200, height=80, pos=(830, 0)):
        dpg.add_text("", tag=FPS_TEXT_TAG)

# --- SECTION: Simulation Loop (Background Thread) ---
def simulation_loop():
    import time
    global latest_graph, simulation_running, sim_update_counter, latest_graph_for_ui, update_for_ui, perf_stats
    logging.info("[PERF] simulation_loop: start")
    while simulation_running:
        t0 = time.time()
        logging.info("[PERF] simulation_loop: calling initialize_main_graph")
        new_graph = initialize_main_graph()  # This captures the screen
        t1 = time.time()
        with graph_lock:
            latest_graph = new_graph
            sim_update_counter += 1
            perf_stats['sim_updates'] += 1
            perf_stats['last_sim_update_time'] = t1 - t0
            if sim_update_counter % SENSORY_UI_UPDATE_EVERY_N == 0:
                latest_graph_for_ui = new_graph
                update_for_ui = True
        logging.info(f"[PERF] simulation_loop: graph update took {(t1-t0)*1000:.2f} ms")
        time.sleep(UPDATE_INTERVAL)
    logging.info("[PERF] simulation_loop: end")

# --- SECTION: Live Update Handler (UI Render Loop) ---
def live_update_callback():
    import time
    global last_update_time, update_for_ui, perf_stats
    t0 = time.perf_counter()
    logging.info("[PERF] live_update_callback: start")
    now = time.time()
    elapsed = now - last_update_time
    if elapsed < UPDATE_INTERVAL:
        return
    with graph_lock:
        if update_for_ui and latest_graph_for_ui is not None:
            t1 = time.perf_counter()
            update_sensory_visualization(latest_graph_for_ui)
            t2 = time.perf_counter()
            perf_stats['ui_updates'] += 1
            perf_stats['last_ui_update_time'] = t2 - t1
            update_for_ui = False
    last_update_time = now
    t3 = time.perf_counter()
    logging.info(f"[PERF] live_update_callback: end, elapsed {(t3-t0)*1000:.2f} ms")

# --- SECTION: Simulation Control Callbacks ---
def start_simulation_callback():
    import time
    global simulation_running, sim_update_counter, last_update_time
    logging.info("[PERF] start_simulation_callback: Simulation starting")
    simulation_running = True
    sim_update_counter = 0
    dpg.set_value("sim_status_text", "Simulation Running")
    Thread(target=simulation_loop, daemon=True).start()
    last_update_time = time.time()
    logging.info(f"[UI] Simulation started at {last_update_time}")


def hard_stop_simulation_callback():
    logging.info("[PERF] hard_stop_simulation_callback: Simulation stopping")
    global simulation_running
    simulation_running = False
    dpg.set_value("sim_status_text", "Simulation Stopped (Hard)")
    logging.info("[UI] Simulation stopped (hard)")
    # AI: EXTEND HERE - Insert simulation stop/cleanup logic

# --- SECTION: Main Window Layout ---
def create_main_window():
    logging.info("[PERF] create_main_window: Creating main window")
    with dpg.window(label="Main Control", tag="main_window", width=400, height=200):
        dpg.add_text("Energy-Based Neural System UI", tag="main_title")
        dpg.add_separator()
        dpg.add_text("Simulation Status: Stopped", tag="sim_status_text")
        dpg.add_button(label="Start Simulation", tag="start_button", callback=start_simulation_callback)
        dpg.add_button(label="Hard Stop", tag="hard_stop_button", callback=hard_stop_simulation_callback)
        # AI: EXTEND HERE - Add more controls, status displays, etc.

# --- SECTION: Sensory Visualization Window ---
def create_sensory_visualization_window():
    logging.info("[PERF] create_sensory_visualization_window: Creating sensory visualization window")
    # Create a placeholder texture (will be updated on simulation start)
    # Use a small default size (e.g., 64x36) for initial allocation
    default_w, default_h = 64, 36
    arr = np.zeros((default_h, default_w, 3), dtype=np.float32).flatten()
    with dpg.texture_registry():
        dpg.add_dynamic_texture(default_w, default_h, arr, tag=sensory_texture_tag)
    with dpg.window(label="Sensory Visualization", tag="sensory_window", width=400, height=400, pos=(420, 0)):
        dpg.add_text("Sensory Input (RGB, downscaled)")
        dpg.add_image(sensory_texture_tag, tag=sensory_image_tag)
        # AI: EXTEND HERE - Add overlays, controls, etc.

# --- SECTION: DPG App Setup ---
def run_ui():
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
    logging.info("[PERF] run_ui: UI setup complete, entering main loop")
    while dpg.is_dearpygui_running():
        t0 = time.perf_counter()
        live_update_callback()
        dpg.render_dearpygui_frame()
        t1 = time.perf_counter()
        logging.info(f"[PERF] run_ui: frame render elapsed {(t1-t0)*1000:.2f} ms")
        # Performance logging every second (UI only)
        now = time.time()
        if now - perf_stats['last_log_time'] >= 1.0:
            fps = perf_stats['ui_updates'] / (now - perf_stats['last_log_time'])
            sim_fps = perf_stats['sim_updates'] / (now - perf_stats['last_log_time'])
            report = (
                f"UI FPS: {fps:.2f}\n"
                f"Sim FPS: {sim_fps:.2f}\n"
                f"Last Sim Update: {perf_stats['last_sim_update_time']*1000:.1f} ms\n"
                f"Last UI Update: {perf_stats['last_ui_update_time']*1000:.1f} ms"
            )
            perf_stats['fps'] = fps
            perf_stats['sim_fps'] = sim_fps
            perf_stats['last_report'] = report
            dpg.set_value(PERF_TEXT_TAG, report)
            dpg.set_value(FPS_TEXT_TAG, f"UI FPS: {fps:.2f}\nSim FPS: {sim_fps:.2f}")
            logging.info("[PERF] " + report.replace("\n", ", "))
            perf_stats['sim_updates'] = 0
            perf_stats['ui_updates'] = 0
            perf_stats['last_log_time'] = now
    dpg.destroy_context()
    logging.info("[PERF] run_ui: UI shutdown complete")

# --- SECTION: Entry Point ---
if __name__ == "__main__":
    run_ui()

# --- END OF FILE ---
# AI: EXTEND HERE - Add more windows, config panels, graph visualizations, etc.
# See DearPyGui docs for advanced usage: https://dearpygui.readthedocs.io/en/latest/tutorials/dpg-structure.html 