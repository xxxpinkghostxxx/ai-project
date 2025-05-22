"""
main.py - Energy-Based Neural System

This script launches the main UI and update loop for an emergent, energy-based neural system.

Project Overview:
- Modular, extensible architecture for self-organizing neural networks.
- Nodes and connections with energy, activity, and adaptive properties.
- Real-time visualization (Tkinter, matplotlib, PyVis) and live control panel.
- Robust logging and auditability (logs/ directory).
- Designed for research, experimentation, and educational use.

Usage:
- Run as a standalone script: python main.py
- Requires dependencies listed in requirements.txt (see README.md).
- All logs and diagnostic files are written to the logs/ directory.

Key Features:
- Modular node/connection classes and NeuralSystem manager.
- Energy-based node birth, death, and adaptation.
- Live dashboard, control panel, and interactive connection graph.
- Configurable via config.py and live control panel.
- All key metrics and events are logged for auditability.

See README.md and project page.txt for full details.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time
from neural_system import NeuralSystem
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from config import SENSOR_WIDTH, SENSOR_HEIGHT, LOG_DIR, DASH_EXPORT_PATH, PERIODIC_UPDATE_MS, DEBUG_ON_ZERO_DYNAMIC_NODES
import threading
from pyvis.network import Network
from datetime import datetime
import sys
import pickle
from vision import ThreadedScreenCapture
import matplotlib
matplotlib.use('TkAgg')
import random
import utils
from utils import logger, LOG_MAX_SIZE, LOG_BACKUPS
import json
from collections import namedtuple

# CUDA/CuPy diagnostic at startup
utils.check_cuda_status()

# Store last sensory input and system for visualization
# last_processed_image: np.ndarray | None = None
last_system = None

# Thread-safe energy history
energy_history = []
energy_history_lock = threading.Lock()
ENERGY_HISTORY_LENGTH = 200

# --- Dashboard functionality ---
sensory_energy_history = []
total_energy_history = []
avg_dynamic_energy_history = []
workspace_avg_energy_history = []
efficiency_history = []
HISTORY_LENGTH = 200

sensory_to_dynamic_conn_history = []
avg_sensory_to_dynamic_weight_history = []
workspace_energy_var_history = []

logger.info('main.py script started')

# --- Log path helper ---
def get_log_path(filename):
    abs_log_dir = os.path.abspath(LOG_DIR)
    os.makedirs(abs_log_dir, exist_ok=True)
    return os.path.join(abs_log_dir, filename)

# --- Struct log rotation helper ---
def rotate_struct_log_if_needed(log_path, max_size=LOG_MAX_SIZE, backups=LOG_BACKUPS):
    if os.path.exists(log_path) and os.path.getsize(log_path) > max_size:
        base, ext = os.path.splitext(log_path)
        for i in range(backups, 0, -1):
            prev = f"{base}_{i}{ext}"
            next = f"{base}_{i+1}{ext}"
            if os.path.exists(prev):
                os.rename(prev, next)
        os.rename(log_path, f"{base}_1{ext}")

# --- Plain text diagnostic log helper ---
class RotatingDiagLogger:
    """
    Rotating plain-text diagnostic logger for workspace and dynamic node state.
    Automatically rotates files when max_size is exceeded, up to max_files.
    """
    def __init__(self, log_dir, base_filename, max_size_mb=1.8, max_files=2):
        self.log_dir = log_dir
        self.base_filename = base_filename
        self.max_size = int(max_size_mb * 1024 * 1024)
        self.max_files = max_files
        self.file_idx = 1
        self.path = get_log_path(f'{base_filename}_{self.file_idx}.txt')
        self.enabled = True
        self.f = open(self.path, 'a', encoding='utf-8')
    def write(self, entry):
        """
        Write a JSON-encoded entry to the current log file. Rotates if needed.
        Args:
            entry (dict): Diagnostic data to log.
        """
        def make_json_safe(obj):
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(x) for x in obj]
            elif hasattr(obj, '__dict__'):
                # Try to use extract_node_attrs if available, else fallback to __dict__
                try:
                    return utils.extract_node_attrs(obj)
                except Exception:
                    return {k: make_json_safe(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        if not self.enabled:
            return
        try:
            safe_entry = make_json_safe(entry)
            line = json.dumps(safe_entry) + '\n'
        except TypeError as e:
            # Fallback: log a string representation and the error
            line = f'{{"error": "JSON serialization failed", "reason": "{str(e)}", "entry_str": "{str(entry)}"}}\n'
        self.f.write(line)
        self.f.flush()
        if os.path.getsize(self.path) > self.max_size:
            self.f.close()
            self.file_idx += 1
            if self.file_idx > self.max_files:
                self.enabled = False
                return
            self.path = get_log_path(f'{self.base_filename}_{self.file_idx}.txt')
            self.f = open(self.path, 'a', encoding='utf-8')
    def close(self):
        """
        Close the current log file if open.
        """
        if self.f:
            self.f.close()

rotating_diag_logger = RotatingDiagLogger(LOG_DIR, 'diagnostic_trace', max_size_mb=1.8, max_files=2)

def copy_log_to_emergency_files(src_path, log_dir, base_name='diagnostic_emergency', max_files=5, max_size=1.8*1024*1024):
    """
    Copy a diagnostic log to a set of emergency files, splitting if needed.
    Args:
        src_path (str): Source log file path.
        log_dir (str): Directory to write emergency logs.
        base_name (str): Base name for output files.
        max_files (int): Max number of emergency files.
        max_size (int): Max size per file in bytes.
    """
    try:
        if not os.path.exists(src_path):
            logger.info(f"[EMERGENCY LOG] Source log {src_path} does not exist.")
            return
        with open(src_path, 'r', encoding='utf-8', errors='ignore') as src:
            file_idx = 1
            out_path = os.path.join(log_dir, f'{base_name}_{file_idx}.txt')
            out_f = open(out_path, 'w', encoding='utf-8')
            current_size = 0
            for line in src:
                encoded = line if line.endswith('\n') else line + '\n'
                size = len(encoded.encode('utf-8'))
                if current_size + size > max_size:
                    out_f.close()
                    file_idx += 1
                    if file_idx > max_files:
                        break
                    out_path = os.path.join(log_dir, f'{base_name}_{file_idx}.txt')
                    out_f = open(out_path, 'w', encoding='utf-8')
                    current_size = 0
                out_f.write(encoded)
                current_size += size
            out_f.close()
        logger.info(f"[EMERGENCY LOG] Copied log to {file_idx} emergency file(s)")
    except Exception as e:
        logger.error(f"[EMERGENCY LOG ERROR] {e}")

# --- Callback helpers for UI controls ---
def log_ai_structure(frames, last_system, logger, LOG_DIR, rotate_struct_log_if_needed, utils):
    """
    Log the AI structure for a number of frames to a struct_log file.
    Args:
        frames (int): Number of frames to log.
        last_system: The current neural system object.
        logger: Logger instance.
        LOG_DIR (str): Directory for logs.
        rotate_struct_log_if_needed: Log rotation helper.
        utils: Utility module for flattening nodes.
    """
    if last_system is None:
        logger.info("[STRUCT LOG] No system to log.")
        return
    for node_type in ['sensory_nodes', 'workspace_nodes', 'processing_nodes']:
        nodes = getattr(last_system, node_type, [])
        flat_nodes = utils.flatten_nodes(nodes)
        logger.info(f"[STRUCT LOG DEBUG] {node_type}: type={type(nodes)}, len={len(nodes)}; flat_len={len(flat_nodes) if hasattr(flat_nodes, '__len__') else 'N/A'}")
        if flat_nodes is not None and len(flat_nodes) > 0:
            logger.info(f"[STRUCT LOG DEBUG] First {node_type} node: type={type(flat_nodes[0])}, attrs={dir(flat_nodes[0])}")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f'struct_log_{ts}.log')
    rotate_struct_log_if_needed(log_path)
    node_type_counts = {}
    zero_energy_count = 0
    none_energy_count = 0
    none_pos_count = 0
    all_nodes_flat = []
    with open(log_path, 'w', encoding='utf-8') as f:
        for i in range(frames):
            f.write(f"--- Frame {i+1} ---\n")
            for node_type in ['sensory_nodes', 'workspace_nodes', 'processing_nodes']:
                nodes = getattr(last_system, node_type, [])
                flat_nodes = utils.flatten_nodes(nodes)
                for n in flat_nodes:
                    all_nodes_flat.append(n)
                    t = getattr(n, 'node_type', '?')
                    node_type_counts[t] = node_type_counts.get(t, 0) + 1
                    pos = getattr(n, 'pos', None)
                    energy = getattr(n, 'energy', None)
                    gen_rate = getattr(n, 'energy_generation_rate', None)
                    cons_rate = getattr(n, 'energy_consumption_rate', None)
                    creation_time = getattr(n, 'creation_time', None)
                    activity_hist = getattr(n, 'activity_history', None)
                    activity_len = len(activity_hist) if activity_hist is not None else 0
                    conns = getattr(n, 'connections', [])
                    in_conns = getattr(n, 'incoming_connections', [])
                    channels = getattr(n, 'channels', None)
                    if t == 'workspace':
                        f.write(f"[WS ENERGY DIAG] id={id(n)} before={energy} gen_rate={gen_rate} cons_rate={cons_rate} conns={len(conns)}\n")
                    if energy is None:
                        none_energy_count += 1
                    elif energy == 0:
                        zero_energy_count += 1
                    if pos is None:
                        none_pos_count += 1
                    energy_str = f'{energy:.4f}' if energy is not None else 'None'
                    gen_rate_str = f'{gen_rate:.4f}' if gen_rate is not None else 'None'
                    cons_rate_str = f'{cons_rate:.4f}' if cons_rate is not None else 'None'
                    f.write(f"Node id={id(n)} type={t} pos={pos} energy={energy_str} gen_rate={gen_rate_str} cons_rate={cons_rate_str} creation_time={creation_time} activity_len={activity_len} conns={len(conns)} in_conns={len(in_conns)} channels={channels}\n")
            for node_type in ['sensory_nodes', 'workspace_nodes', 'processing_nodes']:
                nodes = getattr(last_system, node_type, [])
                flat_nodes = utils.flatten_nodes(nodes)
                for n in flat_nodes:
                    for conn in getattr(n, 'connections', []):
                        dst = getattr(conn, 'destination', None)
                        src_id = id(n)
                        dst_id = id(dst) if dst else 'None'
                        weight = getattr(conn, 'weight', None)
                        cap = getattr(conn, 'energy_transfer_capacity', None)
                        activity_hist = getattr(conn, 'activity_history', None)
                        activity_len = len(activity_hist) if activity_hist is not None else 0
                        conn_id = id(conn)
                        attrs = []
                        for attr in dir(conn):
                            if not attr.startswith('_') and attr not in ('destination', 'weight', 'energy_transfer_capacity', 'activity_history'):
                                val = getattr(conn, attr, None)
                                attrs.append(f"{attr}={val}")
                        weight_str = f'{weight:.4f}' if weight is not None else 'None'
                        cap_str = f'{cap:.4f}' if cap is not None else 'None'
                        f.write(f"Conn id={conn_id} src={src_id} dst={dst_id} weight={weight_str} cap={cap_str} activity_len={activity_len} {' '.join(attrs)}\n")
            f.write("\n")
            if hasattr(last_system, 'update'):
                last_system.update()
        f.write("--- Summary ---\n")
        for t, count in node_type_counts.items():
            f.write(f"Node type {t}: {count}\n")
        f.write(f"Nodes with energy=None: {none_energy_count}\n")
        f.write(f"Nodes with energy==0: {zero_energy_count}\n")
        f.write(f"Nodes with pos=None: {none_pos_count}\n")
    logger.info(f"[STRUCT LOG] AI structure log written to {log_path}")

def pulse_energy(last_system, status_var, logger):
    """
    Pulse +10 energy to all processing nodes and update status bar.
    Args:
        last_system: The current neural system object.
        status_var: Tkinter StringVar for status bar.
        logger: Logger instance.
    """
    if last_system is not None:
        for node in getattr(last_system, 'processing_nodes', []):
            node.energy += 10.0
        status_var.set("Last pulse: +10 energy")
        logger.info("[CONTROL] Pulsed +10 energy to all processing nodes.")

def drain_and_suspend(last_system, status_var, logger, suspend_state, suspend_button):
    """
    Drain all processing node energy and suspend the system.
    Args:
        last_system: The current neural system object.
        status_var: Tkinter StringVar for status bar.
        logger: Logger instance.
        suspend_state (dict): Dict tracking suspension state.
        suspend_button: Tkinter Button to update label/color.
    """
    if last_system is not None:
        for node in getattr(last_system, 'processing_nodes', []):
            node.energy = 0.0
        status_var.set("System suspended and drained.")
        logger.info("[CONTROL] Drained all processing node energy.")
    suspend_state['suspended'] = True
    suspend_button.config(text="Resume System", bg='#225522')

def resume_system(suspend_state, suspend_button, status_var, window, PERIODIC_UPDATE_MS, periodic_update, system, ws_canvas, ws_image_id, metrics_label, capture):
    """
    Resume system from suspended state and restart periodic updates.
    Args:
        suspend_state (dict): Dict tracking suspension state.
        suspend_button: Tkinter Button to update label/color.
        status_var: Tkinter StringVar for status bar.
        window: Tkinter main window.
        PERIODIC_UPDATE_MS (int): Update interval in ms.
        periodic_update: Update function.
        system: Neural system object.
        ws_canvas, ws_image_id, metrics_label: UI elements.
        capture: Screen capture object.
    """
    suspend_state['suspended'] = False
    suspend_button.config(text="Drain & Suspend", bg='#882222')
    status_var.set("System resumed.")
    window.after(PERIODIC_UPDATE_MS, lambda: periodic_update(system, window, ws_canvas, ws_image_id, metrics_label, suspend_state, [0], capture))
    logger.info("[CONTROL] System resumed.")

def toggle_suspend(suspend_state, drain_and_suspend_fn, resume_system_fn):
    """
    Toggle between draining/suspending and resuming the system.
    Args:
        suspend_state (dict): Dict tracking suspension state.
        drain_and_suspend_fn: Function to drain and suspend.
        resume_system_fn: Function to resume system.
    """
    if not suspend_state['suspended']:
        drain_and_suspend_fn()
    else:
        resume_system_fn()

DrawWindow = namedtuple('DrawWindow', ['window', 'ws_canvas', 'ws_image_id', 'metrics_label', 'suspend_state', 'status_bar', 'suspend_button', 'status_var', 'controls_frame'])

def create_draw_window():
    """
    Create and return the main Tkinter draw window and its widgets.
    Returns:
        DrawWindow: Namedtuple with window, canvas, labels, and controls.
    """
    window = tk.Tk()
    window.title('AI Workspace Window')
    window.configure(bg='#222222')
    main_frame = tk.Frame(window, bg='#222222')
    main_frame.pack(fill='both', expand=True)
    left_frame = tk.Frame(main_frame, bg='#222222')
    left_frame.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
    ws_canvas = tk.Canvas(left_frame, bg='#181818', highlightthickness=0)
    ws_canvas.pack(fill='both', expand=True, pady=(0, 5))
    ws_image_id = ws_canvas.create_image(0, 0, anchor='nw')
    metrics_label = tk.Label(left_frame, text="", fg='#e0e0e0', bg='#222222', font=('Consolas', 11, 'bold'), justify='left')
    metrics_label.pack(fill='x', pady=(0, 5))
    right_frame = tk.Frame(main_frame, bg='#222222')
    right_frame.pack(side='right', fill='y', padx=10, pady=10)
    controls_frame = tk.Frame(right_frame, bg='#222222')
    controls_frame.pack(fill='y', pady=(0, 10))
    status_var = tk.StringVar(value="Running")
    status_bar = tk.Label(right_frame, textvariable=status_var, fg='#bbbbbb', bg='#181818', anchor='w', font=('Consolas', 10))
    status_bar.pack(fill='x', side='bottom', pady=(10, 0))
    suspend_state = {'suspended': False}
    suspend_button = tk.Button(controls_frame, text="Drain & Suspend", bg='#882222', fg='#e0e0e0', activebackground='#aa3333', activeforeground='#ffffff', relief='raised', width=22)
    suspend_button.pack(fill='x', padx=4, pady=6)
    # Return controls_frame for button placement
    return DrawWindow(window, ws_canvas, ws_image_id, metrics_label, suspend_state, status_bar, suspend_button, status_var, controls_frame)

def update_draw_window(system, ws_canvas: tk.Canvas, ws_image_id: int, metrics_label: tk.Label) -> None:
    """
    Update the main window with the current workspace grid and diagnostics label.
    Args:
        system: The current neural system object.
        ws_canvas: Tkinter Canvas for workspace.
        ws_image_id: Image ID for canvas.
        metrics_label: Tkinter Label for metrics.
    """
    # --- Workspace grid visualization (grayscale or hue) ---
    if system is not None and hasattr(system, 'workspace_nodes'):
        ws_nodes = system.workspace_nodes
        if ws_nodes is not None and hasattr(ws_nodes, '__len__') and len(ws_nodes) > 0 and hasattr(ws_nodes[0], '__iter__') and len(ws_nodes[0]) > 0:
            ws_h, ws_w = len(ws_nodes), len(ws_nodes[0])
            flat_nodes = np.array(ws_nodes).flatten()
            energies = np.fromiter((float(getattr(n, 'energy', 0.0)) for n in flat_nodes), dtype=np.float32, count=ws_h*ws_w).reshape((ws_h, ws_w))
            energies = np.clip(energies, 0, 244)
            arr_rgb = np.repeat(energies[:, :, None], 3, axis=2).astype(np.uint8)
            # --- Dynamic scaling to fit canvas ---
            canvas_w, canvas_h = ws_canvas.winfo_width(), ws_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                img = Image.fromarray(arr_rgb, mode='RGB').resize((canvas_w, canvas_h), resample=Image.NEAREST)
            else:
                img = Image.fromarray(arr_rgb, mode='RGB')
            tk_img = ImageTk.PhotoImage(img)
            ws_canvas.tk_img = tk_img
            ws_canvas.itemconfig(ws_image_id, image=tk_img)
    # --- Sensory grid visualization (RGB, reverse mapping) ---
    if system is not None and hasattr(system, 'sensory_nodes'):
        sensory_nodes = system.sensory_nodes
        s_h, s_w = len(sensory_nodes), len(sensory_nodes[0])
        flat_nodes = np.array(sensory_nodes).flatten()
        energies = np.fromiter((float(getattr(n, 'energy', 0.0)) for n in flat_nodes), dtype=np.float32, count=s_h*s_w).reshape((s_h, s_w))
        energies = np.clip(energies, 0, 244)
        rev_energies = 244 - energies
        arr_rgb_sens = np.repeat(rev_energies[:, :, None], 3, axis=2).astype(np.uint8)
        # --- Dynamic scaling for sensory grid ---
        if hasattr(ws_canvas, 'sensory_img_id'):
            try:
                canvas_h = ws_canvas.winfo_height()
                canvas_w = ws_canvas.winfo_width()
                if canvas_w > 1 and canvas_h > 1:
                    img_sens = Image.fromarray(arr_rgb_sens, mode='RGB').resize((canvas_w, canvas_h), resample=Image.NEAREST)
                else:
                    img_sens = Image.fromarray(arr_rgb_sens, mode='RGB')
            except Exception:
                img_sens = Image.fromarray(arr_rgb_sens, mode='RGB').resize((128, 128), resample=Image.NEAREST)
        else:
            img_sens = Image.fromarray(arr_rgb_sens, mode='RGB').resize((128, 128), resample=Image.NEAREST)
        tk_img_sens = ImageTk.PhotoImage(img_sens)
        if not hasattr(ws_canvas, 'sensory_img_id'):
            ws_canvas.sensory_img_id = ws_canvas.create_image(0, ws_canvas.winfo_height()+5, anchor='nw', image=tk_img_sens)
        else:
            ws_canvas.itemconfig(ws_canvas.sensory_img_id, image=tk_img_sens)
        ws_canvas.tk_img_sens = tk_img_sens
    # Update diagnostics label with more metrics
    if system is not None:
        metrics = system.get_metrics()
        node_count = len(system.processing_nodes)
        conn_count = len(system.connections)
        avg_energy = 0.0
        if node_count > 0:
            avg_energy = sum(node.energy for node in system.processing_nodes) / node_count
        # Use system.energy_history if available, else 0.0
        total_energy = system.energy_history[-1] if hasattr(system, 'energy_history') and system.energy_history else 0.0
        # Workspace node average energy
        ws_nodes = getattr(system, 'workspace_nodes', [])
        ws_avg_energy = 0.0
        ws_total = 0
        if ws_nodes is not None and hasattr(ws_nodes, 'size') and ws_nodes.size > 0:
            ws_flat = [node.energy for row in ws_nodes for node in row]
            ws_total = len(ws_flat)
            if ws_total > 0:
                ws_avg_energy = sum(ws_flat) / ws_total
        text = (
            f"Energy: {total_energy}\n"
            f"Dynamic Nodes: {metrics['dynamic_node_count']} | Connections: {conn_count}\n"
            f"Avg Node Energy: {avg_energy:.2f}\n"
            f"Node Births: {metrics['node_births']} | Node Deaths: {metrics['node_deaths']}\n"
            f"Conn Births: {metrics['conn_births']} | Conn Deaths: {metrics['conn_deaths']}\n"
            f"Total Energy Generated: {metrics['energy_generated']:.2f} | Consumed: {metrics['energy_consumed']:.2f}\n"
            f"Workspace Avg Energy: {ws_avg_energy:.2f} ({ws_total} nodes)\n"
            f"Eff: {metrics['processing_efficiency']:.3f}"
        )
        metrics_label.config(text=text)

def log_tkinter_exception(type, value, tb) -> None:
    """
    Custom exception handler for Tkinter callbacks that logs to the logger.
    Args:
        type: Exception type.
        value: Exception value.
        tb: Traceback object.
    """
    if logger:
        logger.error("Tkinter callback exception", exc_info=(type, value, tb))
    else:
        if logger:
            logger.error("[TKINTER ERROR]", exc_info=True)

# Patch Tkinter's exception handler
try:
    tk.Tk.report_callback_exception = staticmethod(log_tkinter_exception)
except Exception as e:
    if logger:
        logger.error("Failed to patch Tkinter exception handler", exc_info=True)

def show_connection_graph(background=False, interval=2.0):
    """
    Display a neural network connection graph using PyVis.
    Args:
        background (bool): If True, update graph at regular intervals.
        interval (float): Interval in seconds for background updates.
    """
    def _draw_graph():
        try:
            global last_system
            if last_system is None:
                logger.info('No system state for connection graph.')
                return
            net = Network(height='900px', width='100%', bgcolor='#181818', font_color='white', notebook=False, directed=True)
            type_style = {
                'sensory':  {'color': '#1f77b4', 'shape': 'box'},
                'dynamic':  {'color': '#d62728', 'shape': 'ellipse'},
                'workspace':{'color': '#9467bd', 'shape': 'hexagon'},
                'generic':  {'color': '#7f7f7f', 'shape': 'dot'},
            }
            sensory_nodes = [n for row in getattr(last_system, 'sensory_nodes', []) for n in row]
            dynamic_nodes = last_system.processing_nodes
            workspace_nodes = [n for row in getattr(last_system, 'workspace_nodes', []) for n in row]
            max_sensory = 100
            max_dynamic = 100
            max_workspace = 100
            sample_sensory = sensory_nodes[:max_sensory]
            sample_dynamic = dynamic_nodes[:max_dynamic]
            sample_workspace = workspace_nodes[:max_workspace]
            nodes = sample_sensory + sample_dynamic + sample_workspace
            node_ids = {node: str(id(node)) for node in nodes}
            for node in nodes:
                t = getattr(node, 'node_type', None)
                if t is None:
                    label = '?'
                    style = type_style['generic']
                else:
                    label = {'sensory': 'S', 'dynamic': 'N', 'workspace': 'W'}.get(t, t[0].upper())
                    style = type_style.get(t, type_style['generic'])
                pos = getattr(node, 'pos', None)
                pos_str = f"\nPos: {pos}" if pos is not None else ""
                net.add_node(
                    node_ids[node],
                    label=label,
                    color=style['color'],
                    shape=style['shape'],
                    title=f"Type: {t}\nEnergy: {getattr(node, 'energy', 0):.2f}{pos_str}"
                )
            for node in nodes:
                for conn in getattr(node, 'connections', []):
                    if conn.destination in nodes:
                        src_type = getattr(node, 'node_type', 'generic')
                        dst_type = getattr(conn.destination, 'node_type', 'generic')
                        if src_type == 'sensory' and dst_type == 'dynamic':
                            color = '#1f77b4'
                            dashes = True
                        elif src_type == 'dynamic' and dst_type == 'workspace':
                            color = '#9467bd'
                            dashes = False
                        elif src_type == 'dynamic' and dst_type == 'dynamic':
                            color = '#d62728'
                            dashes = False
                        else:
                            color = '#cccccc'
                            dashes = True
                        net.add_edge(
                            node_ids[node],
                            node_ids[conn.destination],
                            value=abs(getattr(conn, 'weight', 1.0)),
                            color=color,
                            title=f"Weight: {getattr(conn, 'weight', 0):.2f}",
                            dashes=dashes
                        )
            net.write_html('connection_graph.html')
        except Exception as e:
            if logger:
                logger.exception("Exception in show_connection_graph")
            raise
    if background:
        def updater():
            while True:
                _draw_graph()
                time.sleep(interval)
        threading.Thread(target=updater, daemon=True).start()
    else:
        _draw_graph()

def open_control_panel(system) -> None:
    """
    Open the live control panel for adjusting config parameters at runtime.
    Args:
        system: Neural system object.
    """
    import config
    panel = tk.Toplevel()
    panel.title('Live Control Panel')
    panel.geometry('400x400')
    sliders = {}
    param_defs = [
        ('NODE_SPAWN_THRESHOLD', 1.0, 20.0, 0.1),
        ('NODE_DEATH_THRESHOLD', -5.0, 10.0, 0.1),
        ('MAX_PROCESSING_NODES', 100, 2000, 1),
        ('MAX_NODE_BIRTHS_PER_STEP', 1, 20, 1),
        ('CONN_SPAWN_THRESHOLD', 1.0, 20.0, 0.1),
        ('MAX_CONN_BIRTHS_PER_STEP', 1, 40, 1),
        ('NODE_ENERGY_SPAWN_COST', 1.0, 20.0, 0.1),
        ('NODE_ENERGY_CONN_COST', 0.1, 5.0, 0.01),
        ('CONN_MAINTENANCE_COST', 0.01, 0.5, 0.01),
    ]
    row = 0
    for name, minv, maxv, step in param_defs:
        val = getattr(config, name, 1.0)
        tk.Label(panel, text=name).grid(row=row, column=0, sticky='w')
        if isinstance(val, int):
            var = tk.IntVar(value=val)
            slider = tk.Scale(panel, from_=minv, to=maxv, orient='horizontal', variable=var, resolution=step)
        else:
            var = tk.DoubleVar(value=val)
            slider = tk.Scale(panel, from_=minv, to=maxv, orient='horizontal', variable=var, resolution=step)
        slider.grid(row=row, column=1, sticky='ew')
        sliders[name] = (var, slider)
        row += 1
    def apply_changes():
        for name, (var, slider) in sliders.items():
            setattr(config, name, var.get())
        if hasattr(system, 'update_config'):
            system.update_config()
    apply_btn = tk.Button(panel, text='Apply', command=apply_changes)
    apply_btn.grid(row=row, column=0, columnspan=2, pady=10)
    panel.columnconfigure(1, weight=1)

def export_live_dashboard_data(system, frame):
    """
    Export dashboard data (sensory image, node/connection info) to DASH_EXPORT_PATH.
    Args:
        system: Neural system object.
        frame: Latest sensory frame (numpy array).
    """
    logger.debug("[DEBUG] export_live_dashboard_data: start")
    # Sensory image: downscale to 64x36 for dashboard
    if frame is not None:
        img = frame
        logger.debug("[DEBUG] export_live_dashboard_data: before image resize")
        if img.shape[1] > 64 or img.shape[0] > 36:
            img = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)
        sensory_image = img
        logger.debug("[DEBUG] export_live_dashboard_data: after image resize")
    else:
        sensory_image = np.zeros((36, 64), dtype=np.uint8)
    # Node data
    logger.debug("[DEBUG] export_live_dashboard_data: before node data")
    processing_nodes = getattr(system, 'processing_nodes', [])
    node_energies = np.array([n.energy for n in processing_nodes], dtype=np.float32)
    node_positions = np.array([getattr(n, 'pos', (0, 0)) for n in processing_nodes], dtype=np.float32)
    node_energy_hist = node_energies.copy()
    logger.debug("[DEBUG] export_live_dashboard_data: after node data")
    # System energy history (last 100)
    if hasattr(system, 'get_metrics'):
        metrics = system.get_metrics()
    else:
        metrics = {}
    # All nodes (for network graph) - sample for dashboard
    logger.debug("[DEBUG] export_live_dashboard_data: before all_nodes/types/positions")
    all_nodes = []
    all_node_types = []
    all_node_positions = []
    all_node_objs = []
    for node_type in ['sensory_nodes', 'workspace_nodes', 'processing_nodes']:
        nodes = getattr(system, node_type, [])
        flat_nodes = utils.flatten_nodes(nodes)
        for n in flat_nodes:
            all_nodes.append(getattr(n, 'energy', 0.0))
            all_node_types.append(getattr(n, 'node_type', 'unknown'))
            all_node_positions.append(tuple(getattr(n, 'pos', (0, 0))))
            all_node_objs.append(n)
    # Sample nodes for dashboard (max 2000)
    if len(all_node_objs) > 2000:
        sample_indices = random.sample(range(len(all_node_objs)), 2000)
        all_nodes = [all_nodes[i] for i in sample_indices]
        all_node_types = [all_node_types[i] for i in sample_indices]
        all_node_positions = [all_node_positions[i] for i in sample_indices]
        all_node_objs = [all_node_objs[i] for i in sample_indices]
    logger.debug("[DEBUG] export_live_dashboard_data: after all_nodes/types/positions")
    # All connections (for network graph) - sample for dashboard
    logger.debug("[DEBUG] export_live_dashboard_data: before all_connections")
    pos_to_idx = {pos: idx for idx, pos in enumerate(all_node_positions)}
    all_connections = []
    for i, node in enumerate(all_node_objs):
        if hasattr(node, 'connections'):
            for conn in getattr(node, 'connections', []):
                dst = getattr(conn, 'destination', None)
                if dst and hasattr(dst, 'pos'):
                    dst_pos = tuple(getattr(dst, 'pos', (0, 0)))
                    dst_idx = pos_to_idx.get(dst_pos, None)
                    if dst_idx is not None:
                        all_connections.append((i, dst_idx, getattr(conn, 'weight', 1.0)))
    # Sample connections for dashboard (max 5000)
    if len(all_connections) > 5000:
        all_connections = random.sample(all_connections, 5000)
    logger.debug("[DEBUG] export_live_dashboard_data: after all_connections")
    # Compose data dict
    data = {
        'sensory_image': sensory_image,
        'node_energies': node_energies,
        'energy_history': getattr(system, 'energy_history', [])[-100:],
        'node_positions': node_positions,
        'node_energy_hist': node_energy_hist,
        'metrics': metrics,
        'all_nodes': all_nodes,
        'all_node_types': all_node_types,
        'all_node_positions': all_node_positions,
        'all_connections': all_connections,
    }
    logger.debug("[DEBUG] export_live_dashboard_data: before pickle dump")
    try:
        with open(DASH_EXPORT_PATH, 'wb') as f:
            pickle.dump(data, f)
        logger.debug("[DEBUG] export_live_dashboard_data: after pickle dump")
    except Exception as e:
        logger.error(f"[DASHBOARD EXPORT ERROR] {e}")

# --- Helper functions for periodic_update ---
def get_next_frame(capture):
    """
    Get the next frame from the capture device, or None if unavailable.
    Args:
        capture: ThreadedScreenCapture object.
    Returns:
        frame (numpy array or None): Captured frame.
    """
    return capture.get_next_frame(timeout=0.01)

def update_sensory(system, frame, logger):
    system.update_sensory_nodes(frame)

def update_system(system, logger):
    """
    Update the neural system state.
    Args:
        system: Neural system object.
        logger: Logger instance.
    """
    system.update()

def update_ui(system, ws_canvas, ws_image_id, metrics_label, logger):
    """
    Update the UI with the latest system state.
    Args:
        system: Neural system object.
        ws_canvas: Tkinter Canvas.
        ws_image_id: Image ID for canvas.
        metrics_label: Tkinter Label for metrics.
        logger: Logger instance.
    """
    update_draw_window(system, ws_canvas, ws_image_id, metrics_label)

def export_dashboard(system, frame, frame_counter, logger):
    """
    Export dashboard data every 10 frames.
    Args:
        system: Neural system object.
        frame: Latest sensory frame.
        frame_counter (list): Mutable int counter.
        logger: Logger instance.
    """
    if frame_counter[0] % 10 == 0:
        export_live_dashboard_data(system, frame)

def log_diagnostics(system, rotating_diag_logger, logger):
    """
    Log diagnostic information about workspace and dynamic nodes.
    Args:
        system: Neural system object.
        rotating_diag_logger: RotatingDiagLogger instance.
        logger: Logger instance.
    """
    ws_nodes = getattr(system, 'workspace_nodes', None)
    dyn_nodes = getattr(system, 'processing_nodes', None)
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'logger': 'set' if logger else 'None',
        'ws_nodes': 'not None' if ws_nodes is not None else 'None',
        'dyn_nodes': 'not None' if dyn_nodes is not None else 'None',
        'node_births': getattr(system, 'node_births', None),
        'node_deaths': getattr(system, 'node_deaths', None),
    }
    # Patch: Only log serializable node and connection attributes
    def safe_conn_attrs(conn):
        # Extract only serializable attributes from a connection
        attrs = {}
        attrs['id'] = id(conn)
        attrs['weight'] = getattr(conn, 'weight', None)
        attrs['energy_transfer_capacity'] = getattr(conn, 'energy_transfer_capacity', None)
        attrs['source_id'] = id(getattr(conn, 'source', None))
        attrs['destination_id'] = id(getattr(conn, 'destination', None))
        attrs['activity'] = getattr(conn, 'activity', None)
        attrs['last_activity'] = getattr(conn, 'last_activity', None)
        return attrs
    if ws_nodes is not None:
        ws_flat = [node for row in ws_nodes for node in row] if hasattr(ws_nodes, '__iter__') and len(ws_nodes) > 0 and hasattr(ws_nodes[0], '__iter__') else list(ws_nodes) if ws_nodes else []
        if ws_flat:
            ws_sample = random.choice(ws_flat)
            ws_attrs = utils.extract_node_attrs(ws_sample)
            # Add a sample of connection attributes if present
            conns = getattr(ws_sample, 'connections', [])
            if conns:
                ws_attrs['connections_sample'] = [safe_conn_attrs(conns[0])] if conns else []
            log_entry['ws_sample'] = ws_attrs
        else:
            log_entry['ws_sample'] = 'No workspace nodes present'
    else:
        log_entry['ws_sample'] = 'ws_nodes is None'
    if dyn_nodes is not None:
        if dyn_nodes:
            dyn_sample = random.choice(dyn_nodes)
            dyn_attrs = utils.extract_node_attrs(dyn_sample)
            # Add a sample of connection attributes if present
            conns = getattr(dyn_sample, 'connections', [])
            if conns:
                dyn_attrs['connections_sample'] = [safe_conn_attrs(conns[0])] if conns else []
            log_entry['dyn_sample'] = dyn_attrs
        else:
            log_entry['dyn_sample'] = 'No dynamic nodes present'
    else:
        log_entry['dyn_sample'] = 'dyn_nodes is None'
    try:
        rotating_diag_logger.write(log_entry)
    except Exception as file_exc:
        if logger:
            logger.error(f"[DIAG ERROR] Could not write to rotating diagnostic log: {file_exc}")

def handle_emergency(system, rotating_diag_logger, logger, LOG_DIR):
    """
    Handle emergency logging if dynamic nodes reach zero.
    Args:
        system: Neural system object.
        rotating_diag_logger: RotatingDiagLogger instance.
        logger: Logger instance.
        LOG_DIR (str): Directory for logs.
    """
    dyn_nodes = getattr(system, 'processing_nodes', None)
    if dyn_nodes is not None and len(dyn_nodes) == 0:
        if DEBUG_ON_ZERO_DYNAMIC_NODES:
            logger.warning('[DEBUG_ON_ZERO_DYNAMIC_NODES] Dynamic node count is zero! Extra diagnostics enabled.')
        diag_path = rotating_diag_logger.path
        copy_log_to_emergency_files(diag_path, LOG_DIR)

# --- Sensory Grid Window ---
def create_sensory_window():
    import config
    global sensory_window, sensory_canvas, sensory_image_id
    width = getattr(config, 'SENSOR_WIDTH', 64)
    height = getattr(config, 'SENSOR_HEIGHT', 64)
    window = tk.Toplevel()
    window.title('Sensory Grid')
    window.configure(bg='#222222')
    canvas = tk.Canvas(window, width=width, height=height, bg='#181818', highlightthickness=0)
    canvas.pack(padx=10, pady=10, fill='both', expand=True)
    image_id = canvas.create_image(0, 0, anchor='nw')
    window.geometry(f"{width+20}x{height+40}")
    window.resizable(True, True)

    def on_close():
        global sensory_window, sensory_canvas, sensory_image_id
        window.destroy()
        sensory_window = None
        sensory_canvas = None
        sensory_image_id = None

    window.protocol("WM_DELETE_WINDOW", on_close)

    def on_resize(event):
        update_sensory_window(canvas, image_id)
    window.bind('<Configure>', on_resize)
    return window, canvas, image_id

def update_sensory_window(canvas, image_id):
    global last_system
    if last_system is not None and hasattr(last_system, 'sensory_nodes'):
        sensory_nodes = last_system.sensory_nodes
        # Print a sample of sensory node energies
        sample_energies = [sensory_nodes.flat[i].energy for i in range(min(5, sensory_nodes.size))]
        print(f"[DEBUG] update_sensory_window: First 5 sensory node energies: {sample_energies}")
        s_h = len(sensory_nodes)
        s_w = len(sensory_nodes[0])
        aspect = s_w / s_h if s_h else 1.0
        arr_rgb = np.zeros((s_h, s_w, 3), dtype=np.float32)
        sample_vals = [sensory_nodes[0][0].energy, sensory_nodes[s_h//2][s_w//2].energy, sensory_nodes[-1][-1].energy]
        for y in range(s_h):
            for x in range(s_w):
                node = sensory_nodes[y][x]
                val = node.energy
                val = float(val) if val is not None else 0.0
                val = min(1.0, max(0.0, val))
                rev_val = 1.0 - val
                arr_rgb[y, x, :] = rev_val
        arr_rgb = (arr_rgb * 255).astype(np.uint8)
        # --- Always scale image to fit current canvas size ---
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            img = Image.fromarray(arr_rgb, mode='RGB').resize((canvas_w, canvas_h), resample=Image.NEAREST)
        else:
            img = Image.fromarray(arr_rgb, mode='RGB')
        tk_img = ImageTk.PhotoImage(img)
        canvas.tk_img = tk_img
        canvas.itemconfig(image_id, image=tk_img)
        canvas.update_idletasks()

# Function to reopen the sensory window

def reopen_sensory_window():
    global sensory_window, sensory_canvas, sensory_image_id
    if sensory_window is None:
        sensory_window, sensory_canvas, sensory_image_id = create_sensory_window()

# TEMPLATE for other windows:
# def create_dashboard_window(): ...
# def reopen_dashboard_window(): ...

# --- Profiling: rolling averages and file output ---
profile_history = {
    'Capture': [], 'Sensory': [], 'System': [], 'UI': [], 'SensWin': [], 'Dashboard': [], 'Diagnostics': [], 'Total': []
}
PROFILE_AVG_WINDOW = 100  # Number of frames to average over
PROFILE_TXT_PATH = 'pipeline_profile.txt'

def update_profile_averages(timings):
    for k, v in timings.items():
        profile_history[k].append(v)
        if len(profile_history[k]) > PROFILE_AVG_WINDOW:
            profile_history[k].pop(0)
    # Compute averages
    avgs = {k: (sum(profile_history[k]) / len(profile_history[k]) if profile_history[k] else 0.0) for k in profile_history}
    # Write to txt file
    with open(PROFILE_TXT_PATH, 'w') as f:
        for k in ['Capture', 'Sensory', 'System', 'UI', 'SensWin', 'Dashboard', 'Diagnostics', 'Total']:
            f.write(f"{k}: {avgs[k]:.6f}s\n")

# --- Shared state for UI diagnostics ---
class SharedSystemState:
    def __init__(self):
        self.lock = threading.Lock()
        self.system = None
        self.frame = None
        self.metrics = None
        self.energy_history = []
        self.last_update_time = 0
        self.running = True
        self.exception = None

shared_state = SharedSystemState()

# --- Background computation thread ---
def background_system_loop(system, capture, shared_state, rotating_diag_logger, logger, LOG_DIR, PERIODIC_UPDATE_MS):
    frame_counter = 0
    try:
        while shared_state.running:
            t_start = time.time()
            frame = get_next_frame(capture)
            update_sensory(system, frame, logger)
            update_system(system, logger)
            # Save state for UI
            with shared_state.lock:
                shared_state.system = system
                shared_state.frame = frame
                shared_state.metrics = system.get_metrics() if hasattr(system, 'get_metrics') else None
                shared_state.energy_history = getattr(system, 'energy_history', [])[-200:]
                shared_state.last_update_time = time.time()
            export_dashboard(system, frame, [frame_counter], logger)
            log_diagnostics(system, rotating_diag_logger, logger)
            handle_emergency(system, rotating_diag_logger, logger, LOG_DIR)
            frame_counter += 1
            # Cap update rate
            elapsed = time.time() - t_start
            min_frame_time = PERIODIC_UPDATE_MS / 1000.0
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)
    except Exception as e:
        with shared_state.lock:
            shared_state.exception = e
        if logger:
            logger.exception("Exception in background_system_loop")

# --- Lightweight UI polling update ---
def periodic_update_ui(ws_canvas, ws_image_id, metrics_label, shared_state, window, suspend_state):
    try:
        with shared_state.lock:
            system = shared_state.system
            if system is not None:
                update_draw_window(system, ws_canvas, ws_image_id, metrics_label)
    except Exception as e:
        pass
    if not suspend_state['suspended']:
        window.after(50, lambda: periodic_update_ui(ws_canvas, ws_image_id, metrics_label, shared_state, window, suspend_state))

def live_console_config_updater():
    import config
    print("[LiveConfig] Type 'set PARAM VALUE' to update config, or 'list' to show current values.")
    while True:
        try:
            cmd = sys.stdin.readline()
            if not cmd:
                continue
            cmd = cmd.strip()
            if cmd.lower() == 'list':
                for k in dir(config):
                    if k.isupper():
                        print(f"{k} = {getattr(config, k)}")
                continue
            if cmd.lower().startswith('set '):
                parts = cmd.split()
                if len(parts) < 3:
                    print("[LiveConfig] Usage: set PARAM VALUE")
                    continue
                param = parts[1]
                value = ' '.join(parts[2:])
                # Try to infer type from current value
                if hasattr(config, param):
                    old_val = getattr(config, param)
                    try:
                        if isinstance(old_val, bool):
                            new_val = value.lower() in ('1', 'true', 'yes', 'on')
                        elif isinstance(old_val, int):
                            new_val = int(value)
                        elif isinstance(old_val, float):
                            new_val = float(value)
                        else:
                            new_val = value
                        setattr(config, param, new_val)
                        print(f"[LiveConfig] {param} updated: {old_val} -> {new_val}")
                    except Exception as e:
                        print(f"[LiveConfig] Error: {e}")
                else:
                    print(f"[LiveConfig] Unknown config param: {param}")
            else:
                print("[LiveConfig] Unknown command. Use 'set PARAM VALUE' or 'list'.")
        except Exception as e:
            print(f"[LiveConfig] Exception: {e}")

# --- Refactored main() ---
def main() -> None:
    """
    Main entry point for the neural system UI and periodic update loop.
    Sets up the system, UI, logging, and starts the main event loop.
    """
    if logger:
        logger.info('main() function entered')
    global last_system, energy_history, sensory_window, sensory_canvas, sensory_image_id
    start = time.time()
    sensory_count = SENSOR_WIDTH * SENSOR_HEIGHT
    initial_dynamic_nodes = int(sensory_count * 5)  # 5x sensory node count
    logger.info(f"[INFO] Setting initial dynamic node count to {initial_dynamic_nodes} (5x sensory count {sensory_count})")
    system = NeuralSystem(SENSOR_WIDTH, SENSOR_HEIGHT, initial_nodes=initial_dynamic_nodes, logger=logger)
    logger.info(f"[TIMING] NeuralSystem init: {time.time() - start:.2f}s")
    start = time.time()
    drawwin = create_draw_window()
    logger.info(f"[TIMING] Draw window init: {time.time() - start:.2f}s")
    drawwin.window.update()
    # Start threaded screen capture
    start = time.time()
    capture = ThreadedScreenCapture(SENSOR_WIDTH, SENSOR_HEIGHT, interval=PERIODIC_UPDATE_MS / 1000.0)
    capture.start()
    logger.info(f"[TIMING] Screen capture init: {time.time() - start:.2f}s")
    # Start live-updating connection graph in background
    start = time.time()
    show_connection_graph(background=True, interval=2.0)
    logger.info(f"[TIMING] Connection graph thread start: {time.time() - start:.2f}s")
    # Wire up UI buttons to modular callbacks
    import functools
    struct_log_btn = tk.Button(drawwin.controls_frame, text="Log AI Structure (5 frames)",
        command=functools.partial(log_ai_structure, 5, system, logger, LOG_DIR, rotate_struct_log_if_needed, utils),
        bg='#444444', fg='#e0e0e0', activebackground='#555555', activeforeground='#ffffff', relief='raised', width=22)
    struct_log_btn.pack(fill='x', padx=4, pady=4)
    pulse_btn = tk.Button(drawwin.controls_frame, text="Pulse +10 Energy",
        command=functools.partial(pulse_energy, system, drawwin.status_var, logger),
        bg='#225577', fg='#e0e0e0', activebackground='#3377aa', activeforeground='#ffffff', relief='raised', width=22)
    pulse_btn.pack(fill='x', padx=4, pady=4)
    conn_map_btn = tk.Button(drawwin.controls_frame, text="Show Connectivity Map",
        command=lambda: show_connection_graph(background=False),
        bg='#228822', fg='#e0e0e0', activebackground='#33aa33', activeforeground='#ffffff', relief='raised', width=22)
    conn_map_btn.pack(fill='x', padx=4, pady=4)
    # Node testing buttons (unchanged)
    # ... existing code ...
    # Bind canvas resize to redraw
    drawwin.ws_canvas.bind('<Configure>', lambda e: update_draw_window(drawwin.ws_canvas, drawwin.ws_image_id, drawwin.metrics_label))
    # Window close handler
    def on_closing():
        if logger:
            logger.info('Tkinter window closed by user. Shutting down.')
        capture.stop()
        shared_state.running = False
        drawwin.window.destroy()
    drawwin.window.protocol("WM_DELETE_WINDOW", on_closing)
    sensory_window, sensory_canvas, sensory_image_id = create_sensory_window()
    # Add window management menu (unchanged)
    menubar = tk.Menu(drawwin.window)
    window_menu = tk.Menu(menubar, tearoff=0)
    window_menu.add_command(label="Open Sensory Window", command=reopen_sensory_window)
    system_ref = [system]
    capture_ref = [capture]
    window_menu.add_command(label="Change Sensory Resolution", command=lambda: change_sensory_resolution(drawwin, system_ref, capture_ref))
    menubar.add_cascade(label="Windows", menu=window_menu)
    drawwin.window.config(menu=menubar)
    # Start background computation thread
    bg_thread = threading.Thread(target=background_system_loop, args=(system, capture, shared_state, rotating_diag_logger, logger, LOG_DIR, PERIODIC_UPDATE_MS), daemon=True)
    bg_thread.start()
    # Start lightweight UI polling
    drawwin.window.after(50, lambda: periodic_update_ui(drawwin.ws_canvas, drawwin.ws_image_id, drawwin.metrics_label, shared_state, drawwin.window, drawwin.suspend_state))
    # Start live console config updater in background
    threading.Thread(target=live_console_config_updater, daemon=True).start()
    drawwin.window.mainloop()

# Utility functions for writing transcript and summary logs

def write_transcript_log(transcript_text):
    """
    Write the raw conversation transcript to a timestamped log file in logs/.
    Args:
        transcript_text (str): The transcript to write.
    """
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = get_log_path(f'transcript_{ts}.log')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        logger.info(f'[LOGGING] Raw transcript log written to {filename}')
    except Exception as e:
        logger.error(f'[LOGGING ERROR] Failed to write transcript log: {e}')

def write_summary_log(summary_text):
    """
    Write the project summary to a timestamped log file in logs/.
    Args:
        summary_text (str): The summary to write.
    """
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = get_log_path(f'summary_{ts}.log')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.info(f'[LOGGING] Summary log written to {filename}')
    except Exception as e:
        logger.error(f'[LOGGING ERROR] Failed to write summary log: {e}')

if __name__ == "__main__":
    main()