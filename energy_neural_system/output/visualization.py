import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from config import DRAW_GRID_SIZE

# Tkinter draw window

def create_draw_window():
    window = tk.Tk()
    window.title('AI Draw Window')
    canvas = tk.Canvas(window, width=320, height=320)
    canvas.pack()
    arr = np.zeros((320, 320), dtype=np.uint8)
    img = Image.fromarray(arr, mode='L')
    tk_img = ImageTk.PhotoImage(img)
    image_id = canvas.create_image(0, 0, anchor='nw', image=tk_img)
    canvas.tk_img = tk_img
    metrics_label = tk.Label(window, text="")
    metrics_label.pack()
    return window, canvas, image_id, metrics_label

def update_draw_window(canvas, image_id, grid):
    arr = np.array(grid, dtype=np.float32)
    arr = (arr / 10.0 * 255).astype(np.uint8)
    arr = np.repeat(np.repeat(arr, 20, axis=0), 20, axis=1)
    img = Image.fromarray(arr, mode='L')
    tk_img = ImageTk.PhotoImage(img)
    canvas.tk_img = tk_img
    canvas.itemconfig(image_id, image=tk_img)

# Matplotlib live visualizations
live_fig = None
live_axes = None
live_initialized = [False]
live_colorbar = [None]

def is_figure_closed(fig):
    try:
        return not plt.fignum_exists(fig.number)
    except Exception:
        return True

def show_live_visualizations(last_processed_image, last_system, energy_history):
    global live_fig, live_axes, live_initialized, live_colorbar
    if last_processed_image is None or last_system is None:
        return
    if live_fig is not None and is_figure_closed(live_fig):
        live_fig = None
        live_axes = None
        live_initialized[0] = False
        live_colorbar[0] = None
    if not live_initialized[0]:
        plt.ion()
        live_fig, live_axes = plt.subplots(2, 3, figsize=(18, 8))
        live_fig.canvas.manager.set_window_title('AI Visualizations (Live)')
        live_initialized[0] = True
        plt.show()
    ax = live_axes
    for row in ax:
        for a in row:
            a.clear()
    # Sensory Input
    ax[0, 0].set_title('Sensory Input')
    if last_processed_image is not None:
        ax[0, 0].imshow(last_processed_image, cmap='gray')
    else:
        ax[0, 0].text(0.5, 0.5, 'No Data', ha='center', va='center')
    ax[0, 0].axis('off')
    # Processing Node Energies
    energies = [node.energy for node in last_system.processing_nodes]
    ax[0, 1].set_title('Processing Node Energies')
    if len(energies) > 0:
        ax[0, 1].bar(range(len(energies)), energies, color='#1f77b4', label='Node Energy')
    else:
        ax[0, 1].text(0.5, 0.5, 'No Data', ha='center', va='center')
    ax[0, 1].set_xlabel('Node Index')
    ax[0, 1].set_ylabel('Energy')
    ax[0, 1].grid(True, linestyle='--', alpha=0.5)
    ax[0, 1].legend()
    # System Total Energy (History)
    ax[0, 2].set_title('System Total Energy (History)')
    if len(energy_history) > 0:
        ax[0, 2].plot(energy_history, color='blue', label='Total Energy')
        if len(energy_history) >= 20:
            ma = np.convolve(energy_history, np.ones(20)/20, mode='valid')
            ax[0, 2].plot(range(19, 19+len(ma)), ma, color='orange', linestyle='--', label='Moving Avg (20)')
    else:
        ax[0, 2].text(0.5, 0.5, 'No Data', ha='center', va='center')
    ax[0, 2].set_xlabel('Time')
    ax[0, 2].set_ylabel('Total Energy')
    ax[0, 2].grid(True, linestyle='--', alpha=0.5)
    ax[0, 2].legend()
    # 2D Node Map
    ax[1, 0].set_title('2D Node Map')
    positions, node_energies = last_system.get_node_positions_and_energies()
    filtered = [(p, e) for p, e in zip(positions, node_energies) if p is not None and isinstance(p, tuple) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)]
    if live_colorbar[0] is not None:
        live_colorbar[0].remove()
        live_colorbar[0] = None
    if filtered:
        xs, ys = zip(*[p for p, _ in filtered])
        filtered_energies = [e for _, e in filtered]
        sc = ax[1, 0].scatter(xs, ys, c=filtered_energies, cmap='viridis', s=[max(10, e*10) for e in filtered_energies], alpha=0.8, edgecolors='k', marker='o', label='Node')
        for conn in last_system.connections:
            if conn.source.pos and conn.destination.pos and \
               isinstance(conn.source.pos, tuple) and isinstance(conn.destination.pos, tuple) and \
               len(conn.source.pos) == 2 and len(conn.destination.pos) == 2:
                x0, y0 = conn.source.pos
                x1, y1 = conn.destination.pos
                ax[1, 0].plot([x0, x1], [y0, y1], 'gray', alpha=0.2, linewidth=0.5, zorder=0)
        live_colorbar[0] = live_fig.colorbar(sc, ax=ax[1, 0], label='Node Energy')
    else:
        ax[1, 0].text(0.5, 0.5, 'No Data', ha='center', va='center')
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Y')
    ax[1, 0].grid(True, linestyle='--', alpha=0.5)
    ax[1, 0].legend()
    # Eye Layer (Full Screen)
    ax[1, 1].set_title('Eye Layer (Full Screen)')
    eye_img = last_system.get_eye_layer_values()
    if eye_img is not None:
        ax[1, 1].imshow(eye_img, cmap='gray')
    else:
        ax[1, 1].text(0.5, 0.5, 'No Data', ha='center', va='center')
    ax[1, 1].axis('off')
    # Histogram of Node Energies
    ax[1, 2].set_title('Node Energy Histogram')
    if len(energies) > 0:
        ax[1, 2].hist(energies, bins=20, color='#ff7f0e', alpha=0.7, edgecolor='black')
        ax[1, 2].set_xlabel('Energy')
        ax[1, 2].set_ylabel('Count')
        ax[1, 2].grid(True, linestyle='--', alpha=0.5)
    else:
        ax[1, 2].text(0.5, 0.5, 'No Data', ha='center', va='center')
    live_fig.tight_layout()
    live_fig.canvas.draw()
    live_fig.canvas.flush_events()
    try:
        live_fig.canvas.manager.window.attributes('-topmost', 1)
        live_fig.canvas.manager.window.attributes('-topmost', 0)
    except Exception:
        pass 