import subprocess
import sys
import os
import webbrowser
import time

# Paths to your scripts
MAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "dgl_main.py")
DASHBOARD_SCRIPT = os.path.join(os.path.dirname(__file__), "live_dashboard.py")
CONN_MAP_HTML = os.path.join(os.path.dirname(__file__), "connection_graph.html")

# Start the main system
main_proc = subprocess.Popen([sys.executable, MAIN_SCRIPT])

# Start the dashboard
# Use pythonw.exe for dashboard to avoid extra console window, if available
pythonw = sys.executable.replace('python.exe', 'pythonw.exe') if sys.executable.endswith('python.exe') else sys.executable
if os.path.exists(pythonw):
    dashboard_proc = subprocess.Popen([pythonw, DASHBOARD_SCRIPT])
else:
    dashboard_proc = subprocess.Popen([sys.executable, DASHBOARD_SCRIPT])

# Wait a moment for the dashboard to start, then open in browser
print("Waiting for dashboard to start...")
time.sleep(2)
webbrowser.open("http://127.0.0.1:8050")

# Open the connection map HTML in the default browser
if os.path.exists(CONN_MAP_HTML):
    webbrowser.open(f"file://{CONN_MAP_HTML}")
else:
    print(f"Connection map HTML not found at {CONN_MAP_HTML}")

print("AI System and Dashboard started. Press Ctrl+C to exit.")

try:
    main_proc.wait()
    dashboard_proc.wait()
except KeyboardInterrupt:
    print("Shutting down...")
    main_proc.terminate()
    dashboard_proc.terminate() 
