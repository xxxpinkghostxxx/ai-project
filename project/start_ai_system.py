import subprocess
import sys
import os
import webbrowser
import time

# Paths to your scripts
MAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "main.py")
DASHBOARD_SCRIPT = os.path.join(os.path.dirname(__file__), "live_dashboard.py")

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

print("AI System and Dashboard started. Press Ctrl+C to exit.")

try:
    main_proc.wait()
    dashboard_proc.wait()
except KeyboardInterrupt:
    print("Shutting down...")
    main_proc.terminate()
    dashboard_proc.terminate() 
