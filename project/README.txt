Energy Neural System - Shareable Build
=====================================

This folder contains everything you need to run and modify the Energy Neural System project.

---

## How to Run (No Python Needed)

1. Double-click `start_ai_system.exe` (or `energy_neural_system.exe`) to launch the system.
2. The main interface and dashboard will open automatically in your browser.
3. The connection map will also open in your browser.

---

## How to Run/Modify the Python Source

1. Install Python 3.10+ (recommended: from https://www.python.org/).
2. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the main system:
   ```
   python start_ai_system.py
   ```
5. You can edit any `.py` file to change or extend the system.

---

## Folder Contents

- `start_ai_system.exe` - Windows executable launcher
- `*.py` - Python source code
- `lib/` - Required JavaScript/CSS libraries for visualization
- `connection_graph.html` - The interactive connection map
- `requirements.txt` - Python dependencies

---

## Notes
- For best results, run on a machine with a modern CPU and (optionally) a CUDA-capable GPU.
- If you encounter issues, try running from source (see above).
- The system will generate logs and screenshots in the `logs/` folder.

---

Enjoy exploring and modifying the Energy Neural System! 