# Wrapper to redirect DGL imports to PyG implementation
import sys
sys.path.append('.')

from project.pyg_neural_system import PyGNeuralSystem

# Expose the PyG class as the old DGL class name for compatibility
DGLNeuralSystem = PyGNeuralSystem