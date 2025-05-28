import os
import time
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import numpy as np
from dgl_neural_system import DGLNeuralSystem
from vision import ThreadedScreenCapture
from datetime import datetime
import pickle
from pyvis.network import Network
from tkinter import messagebox

# --- Config ---
SENSORY_WIDTH = 32
SENSORY_HEIGHT = 32
N_DYNAMIC = SENSORY_WIDTH * SENSORY_HEIGHT * 5
WORKSPACE_SIZE = (16, 16)
PERIODIC_UPDATE_MS = 100
LOG_DIR = 'logs'
LOG_PATH = os.path.join(LOG_DIR, 'dgl_system_metrics.csv')
CONN_GRAPH_PATH = os.path.join(LOG_DIR, 'dgl_connection_graph.pkl')

# --- DGL System Initialization ---
system = DGLNeuralSystem(SENSORY_WIDTH, SENSORY_HEIGHT, N_DYNAMIC, workspace_size=WORKSPACE_SIZE)

# --- Screen Capture Initialization ---
capture = ThreadedScreenCapture(SENSORY_WIDTH, SENSORY_HEIGHT)
capture.start()

# --- Tkinter UI Setup ---
window = tk.Tk()
window.title('DGL AI Workspace Window')
window.configure(bg='#222222')

main_frame = tk.Frame(window, bg='#222222')
main_frame.pack(fill='both', expand=True)

# Left: Workspace canvas
left_frame = tk.Frame(main_frame, bg='#222222')
left_frame.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
canvas = tk.Canvas(left_frame, bg='#181818', highlightthickness=0)
canvas.pack(fill='both', expand=True, pady=(0, 5))
image_id = canvas.create_image(0, 0, anchor='nw')

# Metrics panel (below workspace, above sensory)
metrics_label = tk.Label(left_frame, text="", fg='#e0e0e0', bg='#222222', font=('Consolas', 11, 'bold'), justify='left')
metrics_label.pack(fill='x', pady=(0, 5))

# Sensory input canvas (below metrics)
sensory_canvas = tk.Canvas(left_frame, width=SENSORY_WIDTH*4, height=SENSORY_HEIGHT*4, bg='#222222', highlightthickness=0)
sensory_canvas.pack(pady=(0, 10))
sensory_image_id = sensory_canvas.create_image(0, 0, anchor='nw')

# Right: Controls
right_frame = tk.Frame(main_frame, bg='#222222')
right_frame.pack(side='right', fill='y', padx=10, pady=10)
controls_frame = tk.Frame(right_frame, bg='#222222')
controls_frame.pack(fill='y', pady=(0, 10))

status_var = tk.StringVar(value="Running")
status_bar = tk.Label(right_frame, textvariable=status_var, fg='#bbbbbb', bg='#181818', anchor='w', font=('Consolas', 10))
status_bar.pack(fill='x', side='bottom', pady=(10, 0))

suspend_state = {'suspended': False}
sensory_enabled = {'enabled': True}

# --- Workspace Visualization ---
def update_workspace_canvas():
    ws_nodes = system.g.ndata['node_type'] == 2  # NODE_TYPE_WORKSPACE
    energies = system.g.ndata['energy'][ws_nodes].cpu().numpy().flatten()
    ws_h, ws_w = WORKSPACE_SIZE
    if energies.size != ws_h * ws_w:
        arr = np.zeros((ws_h, ws_w), dtype=np.float32)
    else:
        arr = energies.reshape((ws_h, ws_w))
    arr = np.clip(arr, 0, 244)
    arr_rgb = np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
    if canvas_w > 1 and canvas_h > 1:
        img = Image.fromarray(arr_rgb, mode='RGB').resize((canvas_w, canvas_h), resample=Image.NEAREST)
    else:
        img = Image.fromarray(arr_rgb, mode='RGB')
    tk_img = ImageTk.PhotoImage(img)
    canvas.tk_img = tk_img
    canvas.itemconfig(image_id, image=tk_img)

# --- Sensory Visualization ---
def update_sensory_canvas(sensory_input):
    arr = np.clip(sensory_input, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr_rgb = np.repeat(arr[:, :, None], 3, axis=2)
    img = Image.fromarray(arr_rgb, mode='RGB').resize((SENSORY_WIDTH*4, SENSORY_HEIGHT*4), resample=Image.NEAREST)
    tk_img = ImageTk.PhotoImage(img)
    sensory_canvas.tk_img = tk_img
    sensory_canvas.itemconfig(sensory_image_id, image=tk_img)

# --- Controls ---
def pulse_energy():
    dynamic_mask = (system.g.ndata['node_type'] == 1)  # NODE_TYPE_DYNAMIC
    system.g.ndata['energy'][dynamic_mask] += 10.0
    status_var.set("Last pulse: +10 energy")

def drain_and_suspend():
    dynamic_mask = (system.g.ndata['node_type'] == 1)
    system.g.ndata['energy'][dynamic_mask] = 0.0
    suspend_state['suspended'] = True
    status_var.set("System suspended and drained.")
    suspend_button.config(text="Resume System", bg='#225522', command=resume_system)

def resume_system():
    suspend_state['suspended'] = False
    status_var.set("System resumed.")
    suspend_button.config(text="Drain & Suspend", bg='#882222', command=drain_and_suspend)
    window.after(PERIODIC_UPDATE_MS, periodic_update)

def toggle_sensory():
    sensory_enabled['enabled'] = not sensory_enabled['enabled']
    if sensory_enabled['enabled']:
        sensory_btn.config(text="Disable Sensory Input", bg='#228822')
        status_var.set("Sensory input enabled.")
    else:
        sensory_btn.config(text="Enable Sensory Input", bg='#882222')
        status_var.set("Sensory input disabled.")

suspend_button = tk.Button(controls_frame, text="Drain & Suspend", bg='#882222', fg='#e0e0e0',
    activebackground='#aa3333', activeforeground='#ffffff', relief='raised', width=22,
    command=drain_and_suspend)
suspend_button.pack(fill='x', padx=4, pady=6)

pulse_btn = tk.Button(controls_frame, text="Pulse +10 Energy", bg='#225577', fg='#e0e0e0',
    activebackground='#3377aa', activeforeground='#ffffff', relief='raised', width=22,
    command=pulse_energy)
pulse_btn.pack(fill='x', padx=4, pady=6)

sensory_btn = tk.Button(controls_frame, text="Disable Sensory Input", bg='#228822', fg='#e0e0e0',
    activebackground='#33aa33', activeforeground='#ffffff', relief='raised', width=22,
    command=toggle_sensory)
sensory_btn.pack(fill='x', padx=4, pady=6)

# Export Screenshot Button

def export_screenshot():
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ws_path = os.path.join(LOG_DIR, f'dgl_workspace_{ts}.png')
        sens_path = os.path.join(LOG_DIR, f'dgl_sensory_{ts}.png')
        html_path = os.path.join(os.path.dirname(__file__), 'connection_graph.html')
        # Workspace canvas
        window.update_idletasks()
        x = window.winfo_rootx() + canvas.winfo_rootx()
        y = window.winfo_rooty() + canvas.winfo_rooty()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        img.save(ws_path)
        # Sensory canvas
        x2 = window.winfo_rootx() + sensory_canvas.winfo_rootx()
        y2 = window.winfo_rooty() + sensory_canvas.winfo_rooty()
        w2 = sensory_canvas.winfo_width()
        h2 = sensory_canvas.winfo_height()
        img2 = ImageGrab.grab(bbox=(x2, y2, x2 + w2, y2 + h2))
        img2.save(sens_path)
        # PyVis node map
        g = system.g
        net = Network(height='900px', width='100%', bgcolor='#181818', font_color='white', notebook=False, directed=True)
        node_types = g.ndata['node_type'].cpu().numpy()
        node_energies = g.ndata['energy'].cpu().numpy().flatten()
        node_pos = g.ndata['pos'].cpu().numpy()
        # Dynamic node subtypes
        if 'dynamic_subtype' in g.ndata:
            from dgl_neural_system import SUBTYPE_NAMES
            node_subtypes = g.ndata['dynamic_subtype'].cpu().numpy()
        else:
            node_subtypes = None
        type_style = {
            0: {'color': '#1f77b4', 'shape': 'box', 'label': 'S'},
            1: {'color': '#d62728', 'shape': 'ellipse', 'label': 'N'},
            2: {'color': '#9467bd', 'shape': 'hexagon', 'label': 'W'},
        }
        for i in range(g.num_nodes()):
            t = int(node_types[i])
            style = type_style.get(t, {'color': '#7f7f7f', 'shape': 'dot', 'label': '?'})
            pos = tuple(float(x) for x in node_pos[i])
            # Add dynamic subtype info to tooltip
            if node_subtypes is not None and t == 1:
                subtype = int(node_subtypes[i])
                subtype_str = SUBTYPE_NAMES[subtype]
                title = f"Type: {t} (Dynamic/{subtype_str})\nEnergy: {float(node_energies[i]):.2f}\nPos: {pos}"
            else:
                title = f"Type: {t}\nEnergy: {float(node_energies[i]):.2f}\nPos: {pos}"
            net.add_node(
                int(i),
                label=style['label'],
                color=style['color'],
                shape=style['shape'],
                title=title
            )
        src, dst = g.edges()
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()
        weights = g.edata['weight'].cpu().numpy().flatten()
        conn_types = g.edata['conn_type'].cpu().numpy().flatten() if 'conn_type' in g.edata else None
        edge_colors = ['#00cc44', '#cc2222', '#ffaa00', '#00bfff']  # Excitatory, Inhibitory, Gated, Plastic
        gate_thresholds = g.edata['gate_threshold'].cpu().numpy().flatten() if 'gate_threshold' in g.edata else None
        plastic_lrs = g.edata['plastic_lr'].cpu().numpy().flatten() if 'plastic_lr' in g.edata else None
        from dgl_neural_system import CONN_TYPE_NAMES
        for idx, (s, d, w) in enumerate(zip(src, dst, weights)):
            if conn_types is not None:
                ctype = int(conn_types[idx])
                color = edge_colors[ctype]
                ctype_str = CONN_TYPE_NAMES[ctype]
                title = f"Type: {ctype_str}\nWeight: {float(w):.3f}"
                if ctype == 2 and gate_thresholds is not None:
                    title += f"\nGate Thresh: {float(gate_thresholds[idx]):.2f}"
                if ctype == 3 and plastic_lrs is not None:
                    title += f"\nPlastic LR: {float(plastic_lrs[idx]):.4f}"
            else:
                color = '#cccccc'
                title = f"Weight: {float(w):.2f}"
            net.add_edge(int(s), int(d), value=abs(float(w)), title=title, color=color)
        net.write_html(html_path)
        status_var.set(f"Exported screenshots and node map to {ws_path}, {sens_path}, {html_path}")
    except Exception as e:
        status_var.set(f"Screenshot export failed: {str(e)}")

screenshot_btn = tk.Button(controls_frame, text="Export Screenshot", bg='#444444', fg='#e0e0e0',
    activebackground='#555555', activeforeground='#ffffff', relief='raised', width=22,
    command=export_screenshot)
screenshot_btn.pack(fill='x', padx=4, pady=6)

# Open Config Panel Button

def open_config_panel():
    panel = tk.Toplevel()
    panel.title('Config Panel')
    panel.geometry('500x800')
    sliders = {}
    import dgl_neural_system as dglns
    global SENSORY_WIDTH, SENSORY_HEIGHT, system, capture
    # --- Parameter definitions grouped by category ---
    param_groups = [
        ("Sensory Grid", [
            ('SENSORY_WIDTH', 8, 256, 1),
            ('SENSORY_HEIGHT', 8, 256, 1),
        ]),
        ("Energy Economy", [
            ('NODE_ENERGY_CAP', 10.0, 500.0, 1.0),
            ('NODE_ENERGY_DECAY', 0.001, 1.0, 0.001),
            ('NODE_ENERGY_SPAWN_COST', 0.1, 50.0, 0.1),
        ]),
        ("Growth/Pruning", [
            ('NODE_SPAWN_THRESHOLD', 0.0, 100.0, 0.1),
            ('NODE_DEATH_THRESHOLD', -100.0, 50.0, 0.1),
            ('MAX_NODE_BIRTHS_PER_STEP', 1, 100, 1),
            ('MAX_CONN_BIRTHS_PER_STEP', 1, 100, 1),
        ]),
    ]
    row = 0
    for group_name, params in param_groups:
        tk.Label(panel, text=group_name, font=('Consolas', 11, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,2))
        row += 1
        for name, minv, maxv, step in params:
            val = globals().get(name, getattr(dglns, name, 1.0))
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
    # --- Preset resolution buttons ---
    def set_resolution(w, h):
        sliders['SENSORY_WIDTH'][0].set(w)
        sliders['SENSORY_HEIGHT'][0].set(h)
        restart_system()
        status_var.set(f"Set sensory resolution to {w}x{h} and restarted system.")
    preset_frame = tk.Frame(panel)
    preset_frame.grid(row=row, column=0, columnspan=2, pady=10)
    tk.Label(preset_frame, text="Presets:").pack(side='left')
    tk.Button(preset_frame, text="16x16", command=lambda: set_resolution(16,16)).pack(side='left', padx=2)
    tk.Button(preset_frame, text="32x32", command=lambda: set_resolution(32,32)).pack(side='left', padx=2)
    tk.Button(preset_frame, text="144p", command=lambda: set_resolution(256,144)).pack(side='left', padx=2)
    tk.Button(preset_frame, text="360p", command=lambda: set_resolution(480,360)).pack(side='left', padx=2)
    tk.Button(preset_frame, text="480p", command=lambda: set_resolution(640,480)).pack(side='left', padx=2)
    row += 1
    # --- Save/Load/Reset Buttons ---
    def save_config():
        import json
        config_dict = {name: var.get() for name, (var, slider) in sliders.items()}
        with open('dgl_config_saved.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        status_var.set("Config saved to dgl_config_saved.json")
    def load_config():
        import json
        try:
            with open('dgl_config_saved.json', 'r') as f:
                config_dict = json.load(f)
            for name, value in config_dict.items():
                if name in sliders:
                    sliders[name][0].set(value)
            apply_changes()
            status_var.set("Config loaded from dgl_config_saved.json")
        except Exception as e:
            status_var.set(f"Load failed: {e}")
    def reset_defaults():
        defaults = {
            'SENSORY_WIDTH': 32,
            'SENSORY_HEIGHT': 32,
            'NODE_ENERGY_CAP': 244.0,
            'NODE_ENERGY_DECAY': 0.1,
            'NODE_ENERGY_SPAWN_COST': 5.0,
            'NODE_SPAWN_THRESHOLD': 20.0,
            'NODE_DEATH_THRESHOLD': 0.0,
            'MAX_NODE_BIRTHS_PER_STEP': 10,
            'MAX_CONN_BIRTHS_PER_STEP': 10,
        }
        for name, (var, slider) in sliders.items():
            var.set(defaults.get(name, var.get()))
        apply_changes()
        status_var.set("Config reset to defaults.")
    def apply_changes():
        for name, (var, slider) in sliders.items():
            if name in globals():
                globals()[name] = var.get()
            if hasattr(dglns, name):
                setattr(dglns, name, var.get())
            if hasattr(system, name):
                setattr(system, name, var.get())
        status_var.set("Config updated.")
        messagebox.showinfo("Config Panel", "Parameters updated.")
    def restart_system():
        global system, capture, SENSORY_WIDTH, SENSORY_HEIGHT
        # Stop old capture
        try:
            capture.stop()
        except Exception:
            pass
        # Recreate system and capture
        width = sliders['SENSORY_WIDTH'][0].get()
        height = sliders['SENSORY_HEIGHT'][0].get()
        SENSORY_WIDTH = width
        SENSORY_HEIGHT = height
        n_dynamic = SENSORY_WIDTH * SENSORY_HEIGHT * 5
        system = dglns.DGLNeuralSystem(SENSORY_WIDTH, SENSORY_HEIGHT, n_dynamic, workspace_size=system.workspace_size)
        capture = ThreadedScreenCapture(SENSORY_WIDTH, SENSORY_HEIGHT)
        capture.start()
        # Update UI canvases
        canvas.config(width=system.workspace_size[0]*4, height=system.workspace_size[1]*4)
        sensory_canvas.config(width=SENSORY_WIDTH*4, height=SENSORY_HEIGHT*4)
        status_var.set(f"System restarted with sensory size {SENSORY_WIDTH}x{SENSORY_HEIGHT}.")
        update_workspace_canvas()
        update_sensory_canvas(np.zeros((SENSORY_HEIGHT, SENSORY_WIDTH), dtype=np.float32))
    apply_btn = tk.Button(panel, text='Apply', command=apply_changes)
    apply_btn.grid(row=row, column=0, pady=10)
    save_btn = tk.Button(panel, text='Save Config', command=save_config)
    save_btn.grid(row=row, column=1, pady=10)
    row += 1
    load_btn = tk.Button(panel, text='Load Config', command=load_config)
    load_btn.grid(row=row, column=0, pady=2)
    reset_btn = tk.Button(panel, text='Reset to Defaults', command=reset_defaults)
    reset_btn.grid(row=row, column=1, pady=2)
    row += 1
    restart_btn = tk.Button(panel, text='Restart System', command=restart_system, bg='#225577', fg='#fff')
    restart_btn.grid(row=row, column=0, columnspan=2, pady=10)
    panel.columnconfigure(1, weight=1)

config_btn = tk.Button(controls_frame, text="Config Panel", bg='#888822', fg='#e0e0e0',
    activebackground='#aaa933', activeforeground='#ffffff', relief='raised', width=22,
    command=open_config_panel)
config_btn.pack(fill='x', padx=4, pady=6)

# --- Logging Setup ---
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_PATH):
    try:
        with open(LOG_PATH, 'w') as f:
            f.write('timestamp,avg_dynamic_energy,total_energy,n_dynamic,n_workspace\n')
    except Exception:
        pass

log_counter = {'count': 0}

def log_system_metrics():
    try:
        g = system.g
        dynamic_mask = (g.ndata['node_type'] == 1)
        workspace_mask = (g.ndata['node_type'] == 2)
        dynamic_energies = g.ndata['energy'][dynamic_mask].cpu().numpy().flatten()
        workspace_energies = g.ndata['energy'][workspace_mask].cpu().numpy().flatten()
        avg_dynamic_energy = float(dynamic_energies.mean()) if dynamic_energies.size > 0 else 0.0
        total_energy = float(g.ndata['energy'].sum().cpu().item())
        n_dynamic = int(dynamic_mask.sum().cpu().item())
        n_workspace = int(workspace_mask.sum().cpu().item())
        ts = datetime.now().isoformat()
        with open(LOG_PATH, 'a') as f:
            f.write(f'{ts},{avg_dynamic_energy},{total_energy},{n_dynamic},{n_workspace}\n')
    except Exception:
        pass

def export_connection_graph():
    try:
        g = system.g
        node_positions = g.ndata['pos'].cpu().numpy()
        node_types = g.ndata['node_type'].cpu().numpy()
        node_energies = g.ndata['energy'].cpu().numpy().flatten()
        src, dst = g.edges()
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()
        weights = g.edata['weight'].cpu().numpy().flatten()
        edges = list(zip(src.tolist(), dst.tolist(), weights.tolist()))
        ts = datetime.now().isoformat()
        data = {
            'timestamp': ts,
            'node_positions': node_positions,
            'node_types': node_types,
            'node_energies': node_energies,
            'edges': edges
        }
        with open(CONN_GRAPH_PATH, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass

def update_metrics_panel():
    g = system.g
    try:
        metrics = system.get_metrics()
        metrics_text = (
            f"Total Energy: {metrics['total_energy']:.2f}\n"
            f"Dynamic Nodes: {metrics['dynamic_node_count']}\n"
            f"Workspace Nodes: {metrics['workspace_node_count']}\n"
            f"Avg Dynamic Energy: {metrics['avg_dynamic_energy']:.2f}\n"
            f"Node Births: {metrics['node_births']} (total {metrics['total_node_births']}) | Node Deaths: {metrics['node_deaths']} (total {metrics['total_node_deaths']})\n"
            f"Conn Births: {metrics['conn_births']} (total {metrics['total_conn_births']}) | Conn Deaths: {metrics['conn_deaths']} (total {metrics['total_conn_deaths']})\n"
            f"Connections: {metrics['connection_count']}"
        )
    except Exception:
        metrics_text = "Metrics unavailable"
    metrics_label.config(text=metrics_text)

def print_metrics_to_terminal():
    try:
        metrics = system.get_metrics()
        print(f"[DGL SYSTEM METRICS] Total Energy: {metrics['total_energy']:.2f} | Dynamic Nodes: {metrics['dynamic_node_count']} | Workspace Nodes: {metrics['workspace_node_count']} | Avg Dynamic Energy: {metrics['avg_dynamic_energy']:.2f} | Node Births: {metrics['node_births']} (total {metrics['total_node_births']}) | Node Deaths: {metrics['node_deaths']} (total {metrics['total_node_deaths']}) | Conn Births: {metrics['conn_births']} (total {metrics['total_conn_births']}) | Conn Deaths: {metrics['conn_deaths']} (total {metrics['total_conn_deaths']}) | Connections: {metrics['connection_count']}")
    except Exception:
        print("[DGL SYSTEM METRICS] Metrics unavailable")

# --- Periodic Update Loop ---
def periodic_update():
    if not hasattr(periodic_update, 'profile_history'):
        periodic_update.profile_history = {
            'sensory': [], 'system': [], 'workspace': [], 'sensory_draw': [], 'metrics': [], 'total': []
        }
    t0 = time.perf_counter()
    if not suspend_state['suspended']:
        # Sensory update
        t1 = time.perf_counter()
        if sensory_enabled['enabled']:
            try:
                frame = capture.get_latest()
                sensory_input = frame.astype(np.float32) / 255.0
                t2 = time.perf_counter()
                update_sensory_canvas(sensory_input)
                t3 = time.perf_counter()
                system.update_sensory_nodes(sensory_input)
            except Exception as e:
                sensory_input = np.random.rand(SENSORY_HEIGHT, SENSORY_WIDTH).astype(np.float32)
                t2 = time.perf_counter()
                update_sensory_canvas(sensory_input)
                t3 = time.perf_counter()
                system.update_sensory_nodes(sensory_input)
        else:
            t2 = t3 = time.perf_counter()
        t4 = time.perf_counter()
        # System update
        system.update()
        t5 = time.perf_counter()
        # Workspace redraw
        update_workspace_canvas()
        t6 = time.perf_counter()
        # Metrics update
        update_metrics_panel()
        t7 = time.perf_counter()
        # Profiling
        profile = periodic_update.profile_history
        profile['sensory'].append(t2-t1)
        profile['sensory_draw'].append(t3-t2)
        profile['system'].append(t5-t4)
        profile['workspace'].append(t6-t5)
        profile['metrics'].append(t7-t6)
        profile['total'].append(t7-t0)
        # Keep last 100
        for k in profile:
            if len(profile[k]) > 100:
                profile[k].pop(0)
        log_counter['count'] += 1
        if log_counter['count'] % 10 == 0:
            log_system_metrics()
            export_connection_graph()
            print_metrics_to_terminal()
            # Print profiling summary
            print("[PROFILE] avg sensory: {:.4f}s | sensory_draw: {:.4f}s | system: {:.4f}s | workspace: {:.4f}s | metrics: {:.4f}s | total: {:.4f}s".format(
                sum(profile['sensory'])/len(profile['sensory']) if profile['sensory'] else 0.0,
                sum(profile['sensory_draw'])/len(profile['sensory_draw']) if profile['sensory_draw'] else 0.0,
                sum(profile['system'])/len(profile['system']) if profile['system'] else 0.0,
                sum(profile['workspace'])/len(profile['workspace']) if profile['workspace'] else 0.0,
                sum(profile['metrics'])/len(profile['metrics']) if profile['metrics'] else 0.0,
                sum(profile['total'])/len(profile['total']) if profile['total'] else 0.0
            ))
        window.after(PERIODIC_UPDATE_MS, periodic_update)

# --- Window Resize Redraw ---
def on_resize(event):
    update_workspace_canvas()
canvas.bind('<Configure>', on_resize)

# --- Window Close Handler ---
def on_closing():
    capture.stop()
    window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)

# --- Start Main Loop ---
window.after(100, periodic_update)
window.mainloop()

# TODO: Add more advanced dashboard features (live web dashboard, etc.)
# TODO: Add config panel and runtime parameter adjustment 
