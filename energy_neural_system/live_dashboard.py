import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import base64
from PIL import Image
import io
import os
import pickle
from datetime import datetime
import sys

# Path to the data file written by the main system
DATA_PATH = os.path.join(os.path.dirname(__file__), 'live_data.pkl')

def create_desktop_shortcut():
    """Create a desktop shortcut to launch the dashboard with Python."""
    try:
        import winshell
        from win32com.client import Dispatch
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, 'AI Neural Dashboard.lnk')
        target = 'pythonw.exe'
        script = os.path.abspath(__file__)
        icon = script  # Use script as icon, or specify a .ico file if available
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = target
        shortcut.Arguments = f'"{script}"'
        shortcut.WorkingDirectory = os.path.dirname(script)
        shortcut.IconLocation = icon
        shortcut.save()
        print(f"[DASHBOARD] Desktop shortcut created at {shortcut_path}")
    except ImportError:
        print("[DASHBOARD] winshell and pywin32 are required to create a desktop shortcut. Install with 'pip install winshell pywin32'.")
    except Exception as e:
        print(f"[DASHBOARD] Failed to create desktop shortcut: {e}")

# Configurable update interval (ms)
def get_update_interval():
    for arg in sys.argv:
        if arg.startswith('--interval='):
            try:
                return int(arg.split('=')[1])
            except Exception:
                pass
    return 1000

UPDATE_INTERVAL = get_update_interval()

# Helper to load data (replace with your real data structure)
def load_live_data():
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, 'rb') as f:
                data = pickle.load(f)
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = os.path.getmtime(DATA_PATH)
            return data
        except Exception as e:
            print(f"[DASHBOARD] Failed to load live_data.pkl: {e}")
    # Dummy data for initial layout
    return {
        'sensory_image': np.zeros((36, 64), dtype=np.uint8),
        'node_energies': np.random.rand(100),
        'energy_history': np.random.rand(100).tolist(),
        'node_positions': np.random.rand(100, 2),
        'node_energy_hist': np.random.rand(100),
        'metrics': {
            'System Energy': 0.0,
            'Nodes': 0,
            'Connections': 0,
            'Efficiency': 0.0,
            'Node Births': 0,
            'Node Deaths': 0,
            'Conn Births': 0,
            'Conn Deaths': 0,
            'Energy Gen': 0.0,
            'Energy Consumed': 0.0,
            'Avg Node Energy': 0.0,
        },
        'timestamp': None
    }

def np_img_to_base64(img_arr):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('AI Neural System Live Dashboard'),
    html.Div(id='data-age-warning', style={'color': 'red', 'fontWeight': 'bold'}),
    html.Div([
        html.Div([
            html.H3('Sensory Input'),
            html.Img(id='sensory-image', style={'width': '320px', 'height': '180px', 'border': '1px solid #ccc'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '40px'}),
        html.Div([
            html.H3('Metrics Summary'),
            html.Pre(id='metrics-summary', style={'fontSize': '16px', 'color': '#222', 'background': '#f7f7f7', 'padding': '10px', 'borderRadius': '8px'}),
            html.Div(id='data-age', style={'fontSize': '12px', 'color': '#888'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top'})
    ]),
    html.Div([
        dcc.Graph(id='node-energies'),
        dcc.Graph(id='system-energy-history'),
    ], style={'display': 'flex'}),
    html.Div([
        dcc.Graph(id='node-map'),
        dcc.Graph(id='node-energy-hist'),
    ], style={'display': 'flex'}),
    html.Div([
        dcc.Graph(id='all-nodes-graph'),
        dcc.Graph(id='network-graph'),
    ], style={'display': 'flex'}),
    dcc.Interval(id='interval', interval=UPDATE_INTERVAL, n_intervals=0)
])

@app.callback([
    Output('sensory-image', 'src'),
    Output('metrics-summary', 'children'),
    Output('node-energies', 'figure'),
    Output('system-energy-history', 'figure'),
    Output('node-map', 'figure'),
    Output('node-energy-hist', 'figure'),
    Output('all-nodes-graph', 'figure'),
    Output('network-graph', 'figure'),
    Output('data-age', 'children'),
    Output('data-age-warning', 'children'),
], [Input('interval', 'n_intervals')])
def update_dashboard(n):
    print(f"[DEBUG] Dashboard callback triggered at {datetime.now()} | DATA_PATH={DATA_PATH}")
    data = load_live_data()
    # Sensory image
    img_src = np_img_to_base64(data['sensory_image'])
    # Metrics summary
    metrics = data['metrics']
    metrics_text = '\n'.join(f"{k}: {v}" for k, v in metrics.items())
    # Node energies bar
    node_energies = data['node_energies']
    node_energies_fig = go.Figure(data=[go.Bar(y=node_energies)])
    node_energies_fig.update_layout(title='Node Energies', xaxis_title='Node', yaxis_title='Energy')
    # System energy history
    energy_history = data['energy_history']
    system_energy_fig = go.Figure(data=[go.Scatter(y=energy_history, mode='lines')])
    system_energy_fig.update_layout(title='System Total Energy (History)', xaxis_title='Time', yaxis_title='Total Energy')
    # 2D node map
    node_positions = data['node_positions']
    if not isinstance(node_positions, np.ndarray):
        node_positions = np.array(node_positions)
    if node_positions.size == 0 or node_positions.ndim != 2 or node_positions.shape[1] < 2:
        node_map_fig = go.Figure()
        node_map_fig.update_layout(title='2D Node Map (No Data)', xaxis_title='X', yaxis_title='Y')
    else:
        # Ensure node_energies is the right length
        if len(node_energies) != node_positions.shape[0]:
            node_energies = np.resize(node_energies, node_positions.shape[0])
        node_map_fig = go.Figure(data=[go.Scatter(x=node_positions[:,0], y=node_positions[:,1], mode='markers', marker=dict(size=8, color=node_energies, colorscale='Viridis', colorbar=dict(title='Energy')))])
        node_map_fig.update_layout(title='2D Node Map', xaxis_title='X', yaxis_title='Y')
    # Node energy histogram
    node_energy_hist = data['node_energy_hist']
    node_energy_hist_fig = go.Figure(data=[go.Histogram(x=node_energy_hist, nbinsx=20)])
    node_energy_hist_fig.update_layout(title='Node Energy Histogram', xaxis_title='Energy', yaxis_title='Count')
    # --- NEW: All nodes graph (all types) ---
    all_nodes = data.get('all_nodes', [])
    all_node_types = data.get('all_node_types', [])
    all_node_positions = data.get('all_node_positions', [])
    type_colors = {'sensory': 'blue', 'draw': 'green', 'dynamic': 'red'}
    colors = [type_colors.get(t, 'gray') for t in all_node_types]
    all_nodes_fig = go.Figure(data=[
        go.Scatter(
            x=[p[0] for p in all_node_positions],
            y=[p[1] for p in all_node_positions],
            mode='markers',
            marker=dict(size=8, color=colors),
            text=[f"{t}: {e:.2f}" for t, e in zip(all_node_types, all_nodes)]
        )
    ])
    all_nodes_fig.update_layout(title='All Nodes (Sensory, Draw, Dynamic)', xaxis_title='X', yaxis_title='Y')
    # --- NEW: Network graph ---
    all_connections = data.get('all_connections', [])
    edge_x = []
    edge_y = []
    for src_idx, dst_idx, weight in all_connections:
        if src_idx < len(all_node_positions) and dst_idx < len(all_node_positions):
            x0, y0 = all_node_positions[src_idx]
            x1, y1 = all_node_positions[dst_idx]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
    network_fig = go.Figure()
    network_fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='gray'), hoverinfo='none'))
    network_fig.add_trace(go.Scatter(
        x=[p[0] for p in all_node_positions],
        y=[p[1] for p in all_node_positions],
        mode='markers',
        marker=dict(size=8, color=colors),
        text=[f"{t}: {e:.2f}" for t, e in zip(all_node_types, all_nodes)]
    ))
    network_fig.update_layout(title='Network Structure (All Nodes & Connections)', xaxis_title='X', yaxis_title='Y', showlegend=False)
    # Data age and warning
    now = datetime.now().timestamp()
    data_time = data.get('timestamp', None)
    if data_time:
        age_sec = now - data_time
        age_str = f"Data age: {age_sec:.1f} seconds ago"
        warning = "" if age_sec < 5 else "Warning: Data is stale!"
    else:
        age_str = "Data age: unknown"
        warning = "Warning: No recent data!"
    return img_src, metrics_text, node_energies_fig, system_energy_fig, node_map_fig, node_energy_hist_fig, all_nodes_fig, network_fig, age_str, warning

if __name__ == '__main__':
    if '--pin' in sys.argv:
        create_desktop_shortcut()
        sys.exit(0)
    app.run(debug=True) 